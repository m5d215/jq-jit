//! Recursive descent parser for jq filter expressions.
//!
//! Parses jq filter strings directly into our IR (`Expr`). This gives us
//! full control over execution and lets the eval / JIT layers see a
//! higher-level form than jq's stack-based bytecode.

use std::rc::Rc;
use anyhow::{Result, bail};

use crate::ir::*;

/// Variable scope for tracking user-defined variables and functions.
struct Scope {
    /// Variable name → var_index mapping.
    vars: Vec<(String, u16)>,
    /// Function name → (func_id, nargs) mapping.
    funcs: Vec<(String, usize, usize)>,
    /// Next available var_index.
    next_var: u16,
    /// Compiled function bodies.
    compiled_funcs: Vec<CompiledFunc>,
    /// Captured outer filter-parameter slots per func_id. A nested `def` that
    /// references an enclosing def's filter parameter is lambda-lifted: the
    /// captured slot becomes a hidden trailing parameter, and every call site
    /// forwards `LoadVar{captured_slot}`. Parse-time only — `eval` just sees
    /// the extra args/param_vars. See #714.
    func_captures: std::collections::HashMap<usize, Vec<u16>>,
    /// Next available memoize slot id. Each lexical occurrence of `memoize(...)`
    /// gets a unique slot; the Env allocates one cache map per slot.
    next_memo_slot: u32,
    /// Monotonic binding counter shared by `def`s and filter parameters, so the
    /// lexically innermost binding of a name can be identified when a 0-arg call
    /// could resolve to either (#766). Higher = bound later = more local.
    next_bind_seq: u32,
    /// `var_index` → binding sequence, for filter-parameter vars.
    var_bind_seq: std::collections::HashMap<u16, u32>,
    /// `func_id` → binding sequence, for user-defined functions.
    func_bind_seq: std::collections::HashMap<usize, u32>,
}

impl Scope {
    fn new() -> Self {
        Scope {
            vars: Vec::new(),
            funcs: Vec::new(),
            next_var: 0,
            compiled_funcs: Vec::new(),
            func_captures: std::collections::HashMap::new(),
            next_memo_slot: 0,
            next_bind_seq: 0,
            var_bind_seq: std::collections::HashMap::new(),
            func_bind_seq: std::collections::HashMap::new(),
        }
    }

    /// Allocate the next shared binding sequence number (#766).
    fn next_bind_seq(&mut self) -> u32 {
        let s = self.next_bind_seq;
        self.next_bind_seq += 1;
        s
    }

    /// Whether the def `func_id` is lexically more local (bound later) than the
    /// filter-parameter var `var_idx` of the same name. Innermost wins (#766).
    fn func_shadows_param(&self, func_id: usize, var_idx: u16) -> bool {
        let fs = self.func_bind_seq.get(&func_id).copied().unwrap_or(0);
        let vs = self.var_bind_seq.get(&var_idx).copied().unwrap_or(0);
        fs > vs
    }

    fn alloc_memo_slot(&mut self) -> u32 {
        let id = self.next_memo_slot;
        self.next_memo_slot += 1;
        id
    }

    fn alloc_var(&mut self, name: &str) -> u16 {
        let idx = self.next_var;
        self.next_var += 1;
        self.vars.push((name.to_string(), idx));
        let seq = self.next_bind_seq();
        self.var_bind_seq.insert(idx, seq);
        idx
    }

    fn lookup_var(&self, name: &str) -> Option<u16> {
        self.vars.iter().rev()
            .find(|(n, _)| n == name)
            .map(|(_, idx)| *idx)
    }

    fn define_func(&mut self, name: &str, nargs: usize, body: Expr, param_vars: Vec<u16>) -> usize {
        let func_id = self.compiled_funcs.len();
        self.funcs.push((name.to_string(), func_id, nargs));
        let seq = self.next_bind_seq();
        self.func_bind_seq.insert(func_id, seq);
        self.compiled_funcs.push(CompiledFunc {
            name: Some(name.to_string()),
            nargs,
            body,
            param_vars,
        });
        func_id
    }

    /// Push a function body to `compiled_funcs` only, without registering it
    /// in the name-lookup table. Used by module loading for nested defs that
    /// must keep a slot for runtime FuncCall dispatch but should not be
    /// callable by name from outside their parent.
    fn define_anon_func(&mut self, nargs: usize, body: Expr, param_vars: Vec<u16>) -> usize {
        let func_id = self.compiled_funcs.len();
        self.compiled_funcs.push(CompiledFunc {
            name: None,
            nargs,
            body,
            param_vars,
        });
        func_id
    }

    fn update_func_body(&mut self, func_id: usize, body: Expr, param_vars: Vec<u16>) {
        if let Some(f) = self.compiled_funcs.get_mut(func_id) {
            f.body = body;
            f.param_vars = param_vars;
        }
    }

    fn lookup_func(&self, name: &str, nargs: usize) -> Option<usize> {
        self.funcs.iter().rev()
            .find(|(n, _, na)| n == name && *na == nargs)
            .map(|(_, id, _)| *id)
    }

    /// Record the captured outer filter-param slots for a lambda-lifted func.
    fn set_func_captures(&mut self, func_id: usize, captures: Vec<u16>) {
        if !captures.is_empty() {
            self.func_captures.insert(func_id, captures);
        }
    }

    /// Trailing capture args (`LoadVar{slot}` per captured slot) a call to
    /// `func_id` must forward, in declared order. Empty for ordinary funcs.
    fn func_capture_args(&self, func_id: usize) -> Vec<Expr> {
        self.func_captures.get(&func_id)
            .map(|caps| caps.iter().map(|&v| Expr::LoadVar { var_index: v }).collect())
            .unwrap_or_default()
    }

    /// Build a `FuncCall`, appending any lambda-lifted capture args after the
    /// caller-supplied `args` (#714). Use at every call-emission site so
    /// captures are forwarded uniformly.
    fn make_funccall(&self, func_id: usize, mut args: Vec<Expr>) -> Expr {
        args.extend(self.func_capture_args(func_id));
        Expr::FuncCall { func_id, args }
    }

    fn save_func_scope(&self) -> usize {
        self.funcs.len()
    }

    fn restore_func_scope(&mut self, saved: usize) {
        self.funcs.truncate(saved);
    }
}

/// Token types for the lexer.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Literals
    Num(f64, Option<Rc<str>>),
    Str(String),          // already unescaped
    Ident(String),
    Variable(String),     // $name (without the $)
    Format(String),       // @name

    // Punctuation
    Dot,                  // .
    Pipe,                 // |
    Comma,                // ,
    Colon,                // :
    Semicolon,            // ;
    Question,             // ?
    LParen,               // (
    RParen,               // )
    LBracket,             // [
    RBracket,             // ]
    LBrace,               // {
    RBrace,               // }

    // Operators
    Plus,                 // +
    Minus,                // -
    Star,                 // *
    Slash,                // /
    Percent,              // %
    Eq,                   // ==
    Ne,                   // !=
    Lt,                   // <
    Gt,                   // >
    Le,                   // <=
    Ge,                   // >=
    Assign,               // =
    UpdateAssign,         // |=
    AddAssign,            // +=
    SubAssign,            // -=
    MulAssign,            // *=
    DivAssign,            // /=
    ModAssign,            // %=
    AltAssign,            // //=
    Alt,                  // //
    AltDestructure,       // ?//

    // Keywords
    If, Then, Elif, Else, End,
    Try, Catch,
    Reduce, Foreach, As,
    Def,
    And, Or, Not,
    Label, Break,
    Import, Include, Module,
    Null, True, False,
    Empty, Error,
    Recurse,              // ..

    Eof,
}

/// Lexer for jq filter strings.
struct Lexer {
    chars: Vec<char>,
    pos: usize,
    tokens: Vec<Token>,
    /// Source char offset where each token in `tokens` begins. Parallel to
    /// `tokens`; backfilled per tokenize iteration so multi-token emitters
    /// (string interpolation) share the start of their originating token.
    /// Converted to 1-based line numbers in `token_lines` after tokenize so
    /// `$__loc__` can report the actual source line (#778).
    token_starts: Vec<usize>,
    /// 1-based source line for each token in `tokens`, computed from
    /// `token_starts` at the end of `tokenize`.
    token_lines: Vec<usize>,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Lexer {
            chars: input.chars().collect(),
            pos: 0,
            tokens: Vec::new(),
            token_starts: Vec::new(),
            token_lines: Vec::new(),
        }
    }

    fn tokenize(&mut self) -> Result<Vec<Token>> {
        while self.pos < self.chars.len() {
            self.skip_whitespace_and_comments();
            if self.pos >= self.chars.len() { break; }

            let tok_start = self.pos;
            let ch = self.chars[self.pos];
            match ch {
                '|' => {
                    self.pos += 1;
                    if self.peek() == Some('=') {
                        self.pos += 1;
                        self.tokens.push(Token::UpdateAssign);
                    } else {
                        self.tokens.push(Token::Pipe);
                    }
                }
                ',' => { self.pos += 1; self.tokens.push(Token::Comma); }
                ':' => { self.pos += 1; self.tokens.push(Token::Colon); }
                ';' => { self.pos += 1; self.tokens.push(Token::Semicolon); }
                '(' => { self.pos += 1; self.tokens.push(Token::LParen); }
                ')' => { self.pos += 1; self.tokens.push(Token::RParen); }
                '[' => { self.pos += 1; self.tokens.push(Token::LBracket); }
                ']' => { self.pos += 1; self.tokens.push(Token::RBracket); }
                '{' => { self.pos += 1; self.tokens.push(Token::LBrace); }
                '}' => { self.pos += 1; self.tokens.push(Token::RBrace); }
                '+' => {
                    self.pos += 1;
                    if self.peek() == Some('=') {
                        self.pos += 1;
                        self.tokens.push(Token::AddAssign);
                    } else {
                        self.tokens.push(Token::Plus);
                    }
                }
                '-' => {
                    self.pos += 1;
                    if self.peek() == Some('=') {
                        self.pos += 1;
                        self.tokens.push(Token::SubAssign);
                    } else {
                        self.tokens.push(Token::Minus);
                    }
                }
                '*' => {
                    self.pos += 1;
                    if self.peek() == Some('=') {
                        self.pos += 1;
                        self.tokens.push(Token::MulAssign);
                    } else {
                        self.tokens.push(Token::Star);
                    }
                }
                '/' => {
                    self.pos += 1;
                    if self.peek() == Some('/') {
                        self.pos += 1;
                        if self.peek() == Some('=') {
                            self.pos += 1;
                            self.tokens.push(Token::AltAssign);
                        } else {
                            self.tokens.push(Token::Alt);
                        }
                    } else if self.peek() == Some('=') {
                        self.pos += 1;
                        self.tokens.push(Token::DivAssign);
                    } else {
                        self.tokens.push(Token::Slash);
                    }
                }
                '%' => {
                    self.pos += 1;
                    if self.peek() == Some('=') {
                        self.pos += 1;
                        self.tokens.push(Token::ModAssign);
                    } else {
                        self.tokens.push(Token::Percent);
                    }
                }
                '=' => {
                    self.pos += 1;
                    if self.peek() == Some('=') {
                        self.pos += 1;
                        self.tokens.push(Token::Eq);
                    } else {
                        self.tokens.push(Token::Assign);
                    }
                }
                '!' => {
                    self.pos += 1;
                    if self.peek() == Some('=') {
                        self.pos += 1;
                        self.tokens.push(Token::Ne);
                    } else {
                        bail!("unexpected character '!' at position {}", self.pos - 1);
                    }
                }
                '<' => {
                    self.pos += 1;
                    if self.peek() == Some('=') {
                        self.pos += 1;
                        self.tokens.push(Token::Le);
                    } else {
                        self.tokens.push(Token::Lt);
                    }
                }
                '>' => {
                    self.pos += 1;
                    if self.peek() == Some('=') {
                        self.pos += 1;
                        self.tokens.push(Token::Ge);
                    } else {
                        self.tokens.push(Token::Gt);
                    }
                }
                '?' => {
                    self.pos += 1;
                    if self.peek() == Some('/') && self.peek_at(1) == Some('/') {
                        self.pos += 2;
                        self.tokens.push(Token::AltDestructure);
                    } else {
                        self.tokens.push(Token::Question);
                    }
                }
                '.' => {
                    self.pos += 1;
                    if self.peek() == Some('.') {
                        self.pos += 1;
                        self.tokens.push(Token::Recurse);
                    } else if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                        // .123 is a number, back up
                        self.pos -= 1;
                        self.read_number()?;
                    } else if self.peek().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') {
                        // .field — immediately followed by identifier, always treat as field name
                        self.tokens.push(Token::Dot);
                        let ident = self.read_ident_str();
                        self.tokens.push(Token::Ident(ident));
                    } else {
                        self.tokens.push(Token::Dot);
                    }
                }
                '"' => {
                    self.read_string()?;
                }
                '@' => {
                    self.pos += 1;
                    let name = self.read_ident_str();
                    if name.is_empty() {
                        bail!("expected format name after @");
                    }
                    self.tokens.push(Token::Format(name));
                }
                '$' => {
                    self.pos += 1;
                    let name = self.read_ident_str();
                    if name.is_empty() {
                        bail!("expected variable name after $");
                    }
                    self.tokens.push(Token::Variable(name));
                }
                c if c.is_ascii_digit() => {
                    self.read_number()?;
                }
                c if c.is_ascii_alphabetic() || c == '_' => {
                    let ident = self.read_ident_str();
                    let tok = match ident.as_str() {
                        "if" => Token::If,
                        "then" => Token::Then,
                        "elif" => Token::Elif,
                        "else" => Token::Else,
                        "end" => Token::End,
                        "try" => Token::Try,
                        "catch" => Token::Catch,
                        "reduce" => Token::Reduce,
                        "foreach" => Token::Foreach,
                        "as" => Token::As,
                        "def" => Token::Def,
                        "and" => Token::And,
                        "or" => Token::Or,
                        "not" => Token::Not,
                        "label" => Token::Label,
                        "break" => Token::Break,
                        "import" => Token::Import,
                        "include" => Token::Include,
                        "module" => Token::Module,
                        "null" => Token::Null,
                        "true" => Token::True,
                        "false" => Token::False,
                        "empty" => Token::Empty,
                        "error" => Token::Error,
                        _ => Token::Ident(ident),
                    };
                    self.tokens.push(tok);
                }
                c if c.is_whitespace() => {
                    self.pos += 1;
                }
                _ => {
                    bail!("unexpected character '{}' at position {}", ch, self.pos);
                }
            }
            // Attribute every token emitted this iteration (one, or several for
            // string interpolation) to the char offset where the iteration began.
            while self.token_starts.len() < self.tokens.len() {
                self.token_starts.push(tok_start);
            }
        }
        self.tokens.push(Token::Eof);
        while self.token_starts.len() < self.tokens.len() {
            self.token_starts.push(self.chars.len());
        }
        // Convert char offsets to 1-based line numbers in a single forward pass
        // (token_starts is non-decreasing across the main token stream).
        self.token_lines.clear();
        self.token_lines.reserve(self.token_starts.len());
        let mut cursor = 0usize;
        let mut line = 1usize;
        for &start in &self.token_starts {
            while cursor < start {
                if self.chars[cursor] == '\n' {
                    line += 1;
                }
                cursor += 1;
            }
            self.token_lines.push(line);
        }
        Ok(self.tokens.clone())
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.chars.len() {
            let ch = self.chars[self.pos];
            if ch.is_whitespace() {
                self.pos += 1;
            } else if ch == '#' {
                // Line comment
                while self.pos < self.chars.len() && self.chars[self.pos] != '\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn read_ident_str(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.chars.len() {
            let ch = self.chars[self.pos];
            if ch.is_ascii_alphanumeric() || ch == '_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.chars[start..self.pos].iter().collect()
    }

    fn read_number(&mut self) -> Result<()> {
        let start = self.pos;
        // Optional leading dot for .123
        if self.pos < self.chars.len() && self.chars[self.pos] == '.' {
            self.pos += 1;
        }
        while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos < self.chars.len() && self.chars[self.pos] == '.' && self.pos > start {
            self.pos += 1;
            while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        // Exponent
        if self.pos < self.chars.len() && (self.chars[self.pos] == 'e' || self.chars[self.pos] == 'E') {
            self.pos += 1;
            if self.pos < self.chars.len() && (self.chars[self.pos] == '+' || self.chars[self.pos] == '-') {
                self.pos += 1;
            }
            while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        let num_str: String = self.chars[start..self.pos].iter().collect();
        let n: f64 = num_str.parse().map_err(|e| anyhow::anyhow!("invalid number '{}': {}", num_str, e))?;
        // Preserve original string when the canonical literal form (jq's
        // decnum-style uppercase `E+`, decimal-expanded when |te| is small)
        // differs from the f64-default form. Without this, scientific
        // literals like `1e-100` would lose their repr and re-render as
        // lowercase `e-` from the no-repr path, dropping jq's literal
        // preservation. Drop forms that aren't valid JSON (e.g. `.5`) so
        // downstream emitters can rely on repr being JSON-safe.
        let canonical = crate::value::normalize_jq_repr(&num_str).unwrap_or_else(|| num_str.clone());
        let f64_repr = crate::value::format_jq_number(n);
        let repr = if canonical != f64_repr {
            let norm = normalize_num_repr(&num_str);
            if crate::value::is_valid_json_number(&norm) {
                Some(Rc::from(norm))
            } else {
                None
            }
        } else {
            None
        };
        self.tokens.push(Token::Num(n, repr));
        Ok(())
    }

}

/// Normalize a number literal to match jq's output format.
fn normalize_num_repr(s: &str) -> String {
    let s = s.trim();
    if let Some(e_pos) = s.find(['e', 'E']) {
        let mantissa = &s[..e_pos];
        let exp_str = &s[e_pos + 1..];
        let exp: i64 = exp_str.parse().unwrap_or(0);

        let (sign, mantissa_abs) = if let Some(rest) = mantissa.strip_prefix('-') {
            ("-", rest)
        } else {
            ("", mantissa)
        };

        let (int_part, frac_part) = if let Some(dot) = mantissa_abs.find('.') {
            (&mantissa_abs[..dot], &mantissa_abs[dot + 1..])
        } else {
            (mantissa_abs, "")
        };

        // Find position of first significant digit in combined digits
        let all_digits: String = format!("{}{}", int_part, frac_part);
        let digits: Vec<char> = all_digits.chars().collect();
        let first_sig = digits.iter().position(|c| *c != '0').unwrap_or(digits.len());

        if first_sig >= digits.len() {
            // All-zero mantissa. jq's decnum collapses the fractional digits
            // and the exponent into one effective exponent before deciding
            // the print form: `0e1` → `0E+1`, `0.00e1` → `0.0`, `0e-7` →
            // `0E-7`, `0.0e-7` → `0E-8` (#452 / #611). Mirror
            // `normalize_jq_repr`'s pure-zero branch.
            let effective_exp = (exp as i64) - (frac_part.len() as i64);
            if effective_exp >= 1 {
                return format!("{}0E+{}", sign, effective_exp);
            }
            if effective_exp == 0 {
                return format!("{}0", sign);
            }
            if effective_exp < -6 {
                return format!("{}0E{}", sign, effective_exp);
            }
            let zeros = (-effective_exp) as usize;
            let mut out = String::with_capacity(2 + zeros);
            out.push_str(sign);
            out.push_str("0.");
            for _ in 0..zeros {
                out.push('0');
            }
            return out;
        }

        // Compute normalized exponent
        let new_exp = exp + (int_part.len() as i64) - 1 - (first_sig as i64);

        // Build normalized mantissa from significant digits.
        // jq's decnum keeps the literal mantissa's trailing zeros so
        // `1.0e0` → `1.0`, `1.0e-5` → `0.000010`. Don't trim. See #457.
        let sig_digits: String = digits[first_sig..].iter().collect();
        let sig_digits = sig_digits.as_str();

        let exp_sign = if new_exp >= 0 { "+" } else { "" };
        if sig_digits.len() <= 1 {
            format!("{}{}E{}{}", sign, &sig_digits[..1], exp_sign, new_exp)
        } else {
            format!("{}{}.{}E{}{}", sign, &sig_digits[..1], &sig_digits[1..], exp_sign, new_exp)
        }
    } else {
        s.to_string()
    }
}

impl Lexer {
    fn read_string(&mut self) -> Result<()> {
        self.pos += 1; // skip opening quote
        // We need to handle string interpolation: "...\(expr)..."
        // We'll collect segments and emit either a plain Str token or
        // an interpolated string token sequence.
        let mut segments: Vec<StringSegment> = Vec::new();
        let mut current = String::new();

        while self.pos < self.chars.len() {
            let ch = self.chars[self.pos];
            match ch {
                '"' => {
                    self.pos += 1;
                    if segments.is_empty() {
                        self.tokens.push(Token::Str(current));
                    } else {
                        // Has interpolation
                        segments.push(StringSegment::Literal(current));
                        self.emit_interpolated_string(segments);
                    }
                    return Ok(());
                }
                '\\' => {
                    self.pos += 1;
                    if self.pos >= self.chars.len() {
                        bail!("unterminated string escape");
                    }
                    let esc = self.chars[self.pos];
                    match esc {
                        '(' => {
                            // String interpolation \(expr)
                            self.pos += 1;
                            segments.push(StringSegment::Literal(std::mem::take(&mut current)));
                            // We need to collect tokens for the expression inside
                            // Save current token position, tokenize the inner expression
                            let expr_start = self.pos;
                            let mut depth = 1;
                            while self.pos < self.chars.len() && depth > 0 {
                                match self.chars[self.pos] {
                                    '(' => depth += 1,
                                    ')' => depth -= 1,
                                    '"' => {
                                        // Skip string literals inside interpolation
                                        self.pos += 1;
                                        while self.pos < self.chars.len() && self.chars[self.pos] != '"' {
                                            if self.chars[self.pos] == '\\' { self.pos += 1; }
                                            self.pos += 1;
                                        }
                                    }
                                    _ => {}
                                }
                                if depth > 0 { self.pos += 1; }
                            }
                            let expr_str: String = self.chars[expr_start..self.pos].iter().collect();
                            self.pos += 1; // skip closing )
                            segments.push(StringSegment::Expr(expr_str));
                        }
                        'n' => { self.pos += 1; current.push('\n'); }
                        't' => { self.pos += 1; current.push('\t'); }
                        'r' => { self.pos += 1; current.push('\r'); }
                        '\\' => { self.pos += 1; current.push('\\'); }
                        '"' => { self.pos += 1; current.push('"'); }
                        '/' => { self.pos += 1; current.push('/'); }
                        'b' => { self.pos += 1; current.push('\u{08}'); }
                        'f' => { self.pos += 1; current.push('\u{0c}'); }
                        'u' => {
                            self.pos += 1;
                            let hex: String = self.chars[self.pos..self.pos.min(self.chars.len()).max(self.pos)+4]
                                .iter().collect();
                            if hex.len() < 4 {
                                bail!("incomplete unicode escape");
                            }
                            self.pos += 4;
                            let cp = u32::from_str_radix(&hex, 16)
                                .map_err(|_| anyhow::anyhow!("invalid unicode escape: \\u{}", hex))?;

                            // Handle surrogate pairs
                            if (0xD800..=0xDBFF).contains(&cp) {
                                // High surrogate. jq requires the next escape
                                // to be a valid low surrogate `\uDC00..=\uDFFF`;
                                // anything else (EOF, plain char, another high
                                // surrogate, low-surrogate-with-junk) is a parse
                                // error. Silently dropping it would diverge from
                                // jq, which rejects the literal at parse time.
                                let invalid_pair = || anyhow::anyhow!(
                                    "Invalid \\uXXXX\\uXXXX surrogate pair escape"
                                );
                                if self.pos + 5 >= self.chars.len()
                                    || self.chars[self.pos] != '\\'
                                    || self.chars[self.pos + 1] != 'u'
                                {
                                    return Err(invalid_pair());
                                }
                                self.pos += 2;
                                let hex2: String = self.chars[self.pos..self.pos+4].iter().collect();
                                self.pos += 4;
                                let cp2 = u32::from_str_radix(&hex2, 16)
                                    .map_err(|_| anyhow::anyhow!("invalid unicode escape"))?;
                                if !(0xDC00..=0xDFFF).contains(&cp2) {
                                    return Err(invalid_pair());
                                }
                                let combined = ((cp - 0xD800) << 10) + (cp2 - 0xDC00) + 0x10000;
                                if let Some(c) = char::from_u32(combined) {
                                    current.push(c);
                                }
                            } else if let Some(c) = char::from_u32(cp) {
                                current.push(c);
                            } else {
                                // Standalone low surrogate (DC00-DFFF) — jq
                                // emits U+FFFD (replacement character) for these
                                // rather than dropping or erroring.
                                current.push('\u{FFFD}');
                            }
                        }
                        _ => {
                            self.pos += 1;
                            current.push('\\');
                            current.push(esc);
                        }
                    }
                }
                _ => {
                    self.pos += 1;
                    current.push(ch);
                }
            }
        }
        bail!("unterminated string")
    }

    fn emit_interpolated_string(&mut self, segments: Vec<StringSegment>) {
        // Emit as: __INTERP_START, segments..., __INTERP_END
        // We'll use a special representation in the token stream
        // Actually, let's just pre-process this into a special token
        self.tokens.push(Token::Ident("__string_interp__".to_string()));
        self.tokens.push(Token::LParen);
        for (i, seg) in segments.iter().enumerate() {
            if i > 0 {
                self.tokens.push(Token::Semicolon);
            }
            match seg {
                StringSegment::Literal(s) => {
                    self.tokens.push(Token::Str(s.clone()));
                }
                StringSegment::Expr(expr_str) => {
                    // Tokenize the inner expression and wrap in parens
                    self.tokens.push(Token::Ident("__expr__".to_string()));
                    self.tokens.push(Token::LParen);
                    // We need to recursively tokenize the expression
                    let mut inner_lexer = Lexer::new(expr_str);
                    if let Ok(inner_tokens) = inner_lexer.tokenize() {
                        for t in &inner_tokens {
                            if *t == Token::Eof { break; }
                            self.tokens.push(t.clone());
                        }
                    }
                    self.tokens.push(Token::RParen);
                }
            }
        }
        self.tokens.push(Token::RParen);
    }
}

enum StringSegment {
    Literal(String),
    Expr(String),
}

/// Parser state.
pub struct Parser {
    tokens: Vec<Token>,
    /// 1-based source line for each token in `tokens` (parallel vector).
    /// Used to give `$__loc__` the actual source line (#778). May be empty
    /// for parsers built without line info, in which case lookups fall back
    /// to line 1.
    token_lines: Vec<usize>,
    pos: usize,
    scope: Scope,
    /// `var_index` of the reserved top-level `$ENV` binding. A `$ENV` reference
    /// resolves to the built-in environment (`Expr::Env`) only while this is the
    /// innermost binding of the name; a user `. as $ENV` (or `{$ENV}` pattern)
    /// allocates a deeper binding that shadows it, matching jq. See #886.
    env_var_idx: u16,
    lib_dirs: Vec<String>,
    /// The `compiled_funcs` id of the function body currently being parsed, or
    /// `None` at the top level. Used to attribute a deferred unbound-variable
    /// reference to its enclosing def. #765
    current_func: Option<usize>,
    /// Unbound `$var` references parked instead of erroring eagerly, as
    /// `(enclosing_func_id, name)`. After the whole program is parsed, only
    /// those reachable from the top-level expression (top-level refs, plus refs
    /// inside transitively-called defs) are errors — jq never compiles the body
    /// of an uncalled def. #765
    deferred_unbound: Vec<(Option<usize>, String)>,
    /// Unknown function/builtin references parked instead of erroring eagerly,
    /// as `(enclosing_func_id, name, nargs)`. Same reachability rule as
    /// `deferred_unbound`: jq resolves function names lazily, so an undefined
    /// name in the body of an uncalled def is not an error. #807
    deferred_unknown_func: Vec<(Option<usize>, String, usize)>,
}

/// Result of parsing: expression + compiled functions.
pub struct ParseResult {
    pub expr: Expr,
    pub funcs: Vec<CompiledFunc>,
    /// Total number of `memoize(...)` slots required by the program. The
    /// runtime `Env` allocates one cache map per slot.
    pub memo_slots: u32,
}

impl Parser {
    pub fn parse(input: &str) -> Result<ParseResult> {
        Self::parse_with_libs(input, &[])
    }

    pub fn parse_with_libs(input: &str, lib_dirs: &[String]) -> Result<ParseResult> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let token_lines = std::mem::take(&mut lexer.token_lines);
        let mut parser = Parser {
            tokens,
            token_lines,
            pos: 0,
            scope: Scope::new(),
            env_var_idx: 0,
            lib_dirs: lib_dirs.to_vec(),
            current_func: None,
            deferred_unbound: Vec::new(),
            deferred_unknown_func: Vec::new(),
        };

        // Pre-register $ENV
        parser.env_var_idx = parser.scope.alloc_var("ENV");

        let expr = parser.parse_program()?;
        if !parser.at_eof() {
            bail!("unexpected token {:?} at position {}", parser.current(), parser.pos);
        }
        parser.check_unbound_reachability(&expr)?;
        Ok(ParseResult {
            expr,
            funcs: parser.scope.compiled_funcs,
            memo_slots: parser.scope.next_memo_slot,
        })
    }

    /// Record an unbound `$name` reference (attributed to the def currently
    /// being parsed) and return a placeholder that errors with jq's message if
    /// it is ever evaluated. The eager error is deferred so a never-called def
    /// with a free variable does not abort compilation; the real decision is
    /// made by `check_unbound_reachability`. #765
    fn defer_unbound_var(&mut self, name: &str) -> Expr {
        self.deferred_unbound.push((self.current_func, name.to_string()));
        Expr::Error {
            msg: Some(Box::new(Expr::Literal(Literal::Str(format!("${} is not defined", name))))),
        }
    }

    /// Record an unknown function/builtin reference (attributed to the def
    /// currently being parsed) and return a placeholder that errors with jq's
    /// message if it is ever evaluated. Deferred so a never-called def whose
    /// body names an undefined function does not abort compilation — jq
    /// resolves function names lazily. The real decision is made by
    /// `check_unbound_reachability`. #807
    fn defer_unknown_func(&mut self, name: &str, nargs: usize) -> Expr {
        self.deferred_unknown_func
            .push((self.current_func, name.to_string(), nargs));
        Expr::Error {
            msg: Some(Box::new(Expr::Literal(Literal::Str(format!(
                "{}/{} is not defined",
                name, nargs
            ))))),
        }
    }

    /// Resolve `$name`, deferring the "not defined" error for an unbound
    /// reference (see `defer_unbound_var`). #765
    fn load_or_defer_var(&mut self, name: &str) -> Expr {
        match self.scope.lookup_var(name) {
            Some(idx) => Expr::LoadVar { var_index: idx },
            None => self.defer_unbound_var(name),
        }
    }

    /// Resolve a `$ENV` reference: the built-in environment object unless a
    /// user binding (`. as $ENV`, `{$ENV}` pattern, etc.) shadows the reserved
    /// top-level binding, in which case the lexically innermost binding wins.
    /// jq lets `$ENV` be shadowed like any other variable. See #886.
    fn resolve_env_ref(&self) -> Expr {
        match self.scope.lookup_var("ENV") {
            Some(idx) if idx != self.env_var_idx => Expr::LoadVar { var_index: idx },
            _ => Expr::Env,
        }
    }

    /// Turn parked unbound `$var` references and unknown function references
    /// into errors, but only those that are actually reachable: a top-level
    /// reference, or one inside a def that the top-level program calls
    /// (transitively). jq never compiles the body of an uncalled def, so a
    /// free variable (#765) or undefined function name (#807) there is not an
    /// error.
    fn check_unbound_reachability(&self, program: &Expr) -> Result<()> {
        if self.deferred_unbound.is_empty() && self.deferred_unknown_func.is_empty() {
            return Ok(());
        }
        let mut reachable: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut stack: Vec<usize> = Vec::new();
        crate::eval::collect_func_calls(program, &mut stack);
        while let Some(fid) = stack.pop() {
            if reachable.insert(fid) {
                if let Some(f) = self.scope.compiled_funcs.get(fid) {
                    crate::eval::collect_func_calls(&f.body, &mut stack);
                }
            }
        }
        let is_reachable = |fid: &Option<usize>| match fid {
            None => true, // top-level reference
            Some(id) => reachable.contains(id),
        };
        for (fid, name) in &self.deferred_unbound {
            if is_reachable(fid) {
                bail!("${} is not defined", name);
            }
        }
        for (fid, name, nargs) in &self.deferred_unknown_func {
            if is_reachable(fid) {
                bail!("{}/{} is not defined", name, nargs);
            }
        }
        Ok(())
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    /// 1-based source line of the token at the current cursor, for `$__loc__`
    /// (#778). Falls back to line 1 when line info is unavailable.
    fn current_line(&self) -> usize {
        self.token_lines.get(self.pos).copied().unwrap_or(1)
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos + 1).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &Token) -> Result<()> {
        let tok = self.advance();
        if &tok != expected {
            bail!("expected {:?}, got {:?}", expected, tok);
        }
        Ok(())
    }

    fn at_eof(&self) -> bool {
        matches!(self.current(), Token::Eof)
    }

    fn at(&self, tok: &Token) -> bool {
        self.current() == tok
    }

    fn eat(&mut self, tok: &Token) -> bool {
        if self.at(tok) {
            self.advance();
            true
        } else {
            false
        }
    }

    // -----------------------------------------------------------------------
    // Grammar rules
    // -----------------------------------------------------------------------

    fn parse_program(&mut self) -> Result<Expr> {
        // Handle module statement (skip it)
        if matches!(self.current(), Token::Module) {
            self.advance();
            // Skip metadata
            while !self.at(&Token::Semicolon) && !self.at_eof() {
                self.advance();
            }
            if self.at(&Token::Semicolon) { self.advance(); }
        }

        // Collect all top-level imports/includes and defs
        let mut import_bindings: Vec<(u16, Expr)> = Vec::new();
        loop {
            if self.at(&Token::Def) {
                self.parse_funcdef()?;
            } else if matches!(self.current(), Token::Import) {
                let binding = self.parse_import()?;
                if let Some(b) = binding {
                    import_bindings.push(b);
                }
            } else if matches!(self.current(), Token::Include) {
                self.parse_include()?;
            } else {
                break;
            }
        }
        let body = self.parse_pipe()?;
        // Wrap body in LetBindings for data imports
        let mut result = body;
        for (var_idx, value_expr) in import_bindings.into_iter().rev() {
            result = Expr::LetBinding {
                var_index: var_idx,
                value: Box::new(value_expr),
                body: Box::new(result),
            };
        }
        Ok(result)
    }

    fn parse_funcdef(&mut self) -> Result<()> {
        self.expect(&Token::Def)?;
        let name = match self.advance() {
            Token::Ident(s) => s,
            t => bail!("expected function name, got {:?}", t),
        };

        // Parse parameters: both `x` (filter param) and `$x` (value param) syntax
        let mut params: Vec<(String, bool)> = Vec::new(); // (name, is_value_param)
        if self.eat(&Token::LParen) {
            loop {
                match self.advance() {
                    Token::Ident(p) => params.push((p, false)),
                    Token::Variable(p) => params.push((p, true)),
                    Token::RParen => break,
                    Token::Semicolon => continue,
                    t => bail!("expected parameter name, got {:?}", t),
                }
            }
        }

        self.expect(&Token::Colon)?;

        // Save var scope to restore after parsing body
        let saved_vars = self.scope.vars.len();

        // Allocate variables for parameters
        // Filter params use a special prefix to avoid collision with $vars of the same name
        let mut param_vars = Vec::new();
        let mut value_param_bindings: Vec<(u16, u16)> = Vec::new(); // (filter_var, value_var)
        for (p, is_value) in &params {
            let fparam_name = format!("\x00fparam:{}", p);
            let filter_idx = self.scope.alloc_var(&fparam_name);
            param_vars.push(filter_idx);
            if *is_value {
                // For $x params: filter_idx is for substitution, value_idx for $x in body
                let value_idx = self.scope.alloc_var(p);
                value_param_bindings.push((filter_idx, value_idx));
            }
        }

        // Pre-register the function for recursive calls (placeholder body)
        let func_id = self.scope.define_func(&name, params.len(), Expr::Empty, param_vars.clone());

        let saved_funcs = self.scope.save_func_scope();
        // Attribute any unbound `$var` parked while parsing this body to this
        // def, so the reachability check can ignore it if the def is never
        // called (#765). Nested defs save/restore around their own bodies.
        let saved_current_func = self.current_func;
        self.current_func = Some(func_id);
        let mut body = self.parse_pipe()?;
        self.current_func = saved_current_func;
        self.scope.restore_func_scope(saved_funcs);
        self.expect(&Token::Semicolon)?;

        // Restore var scope (remove param vars)
        self.scope.vars.truncate(saved_vars);

        // For $x params, wrap body with LetBinding to eagerly evaluate and bind
        for (filter_var, value_var) in value_param_bindings.into_iter().rev() {
            body = Expr::LetBinding {
                var_index: value_var,
                value: Box::new(Expr::LoadVar { var_index: filter_var }),
                body: Box::new(body),
            };
        }

        // Lambda-lift captured enclosing filter parameters (#714). A filter
        // parameter is passed by beta-substitution into the def's body, but a
        // *nested* def's body lives in a separate func slot the outer
        // substitution never reaches — so a nested reference to an enclosing
        // filter param would resolve to an unbound slot (`null`). Promote each
        // captured enclosing filter param to a hidden trailing parameter:
        // rewrite the body to read the hidden slot, forward it through the
        // def's own recursive self-calls, and record it so every call site
        // (handled in the call-emission paths) passes the captured slot along.
        // Value params and let-bindings already work via the env, so only
        // `\x00fparam:`-prefixed (filter) params from the enclosing scope need
        // this.
        let mut captured: Vec<u16> = Vec::new();
        for (vname, vidx) in self.scope.vars[..saved_vars].iter() {
            if vname.starts_with('\x00')
                && vname.starts_with("\x00fparam:")
                && !param_vars.contains(vidx)
                && !captured.contains(vidx)
                && crate::eval::expr_uses_var(&body, *vidx)
            {
                captured.push(*vidx);
            }
        }
        let mut param_vars = param_vars;
        if !captured.is_empty() {
            let hidden: Vec<u16> = captured.iter()
                .map(|c| self.scope.alloc_var(&format!("\x00fparam:cap:{}", c)))
                .collect();
            let hidden_loadvars: Vec<Expr> = hidden.iter()
                .map(|&h| Expr::LoadVar { var_index: h })
                .collect();
            // References to the captured params become the hidden params.
            body = crate::eval::substitute_params(&body, &captured, &hidden_loadvars);
            // Forward the hidden params through the def's recursive self-calls
            // (emitted as plain calls before the captures were known).
            body = crate::eval::append_call_args(&body, func_id, &hidden_loadvars);
            param_vars.extend(hidden);
            self.scope.set_func_captures(func_id, captured);
        }

        // Update the function body (replacing placeholder)
        self.scope.update_func_body(func_id, body, param_vars);
        Ok(())
    }

    /// Parse `import "path" as alias;` or `import "path" as $var;`
    /// Returns Some((var_index, value_expr)) for data imports, None for code imports.
    fn parse_import(&mut self) -> Result<Option<(u16, Expr)>> {
        self.advance(); // import

        // Get module path
        let path = match self.advance() {
            Token::Str(s) => s,
            t => bail!("expected string after import, got {:?}", t),
        };

        // Parse optional metadata {search:"./"}
        let mut search_path = None;
        if self.at(&Token::LBrace) {
            // Skip metadata but extract search if present
            // This is simplified - just look for search:"..."
            let _start = self.pos;
            self.advance(); // {
            while !self.at(&Token::RBrace) && !self.at_eof() {
                if matches!(self.current(), Token::Ident(s) if s == "search") {
                    self.advance(); // search
                    if self.eat(&Token::Colon) {
                        if let Token::Str(s) = self.advance() {
                            search_path = Some(s);
                        }
                    }
                } else {
                    self.advance();
                }
            }
            if self.at(&Token::RBrace) { self.advance(); }
        }

        self.expect(&Token::As)?;

        // Check if it's a data import ($var) or code import (alias)
        match self.current().clone() {
            Token::Variable(var_name) => {
                self.advance();
                // Parse optional metadata after alias (may contain search path)
                if self.at(&Token::LBrace) {
                    self.advance(); // {
                    while !self.at(&Token::RBrace) && !self.at_eof() {
                        if matches!(self.current(), Token::Ident(s) if s == "search") {
                            self.advance(); // search
                            if self.eat(&Token::Colon) {
                                if let Token::Str(s) = self.advance() {
                                    search_path = Some(s);
                                }
                            }
                        } else {
                            self.advance();
                        }
                    }
                    if self.at(&Token::RBrace) { self.advance(); }
                }
                self.expect(&Token::Semicolon)?;

                // Data import: load JSON file and wrap in array
                let json_path = self.resolve_data_module(&path, search_path.as_deref())?;
                let json_content = std::fs::read_to_string(&json_path)
                    .map_err(|e| anyhow::anyhow!("Cannot load data module '{}': {}", path, e))?;
                // Data modules are wrapped in an array per jq convention
                let array_json = format!("[{}]", json_content.trim());
                let value_expr = Expr::Literal(Literal::Str(array_json));
                // Parse at runtime via fromjson
                let fromjson_expr = Expr::CallBuiltin {
                    name: "fromjson".to_string(),
                    args: vec![],
                };
                let pipe_expr = Expr::Pipe {
                    left: Box::new(value_expr),
                    right: Box::new(fromjson_expr),
                };
                let var_idx = self.scope.alloc_var(&var_name);
                Ok(Some((var_idx, pipe_expr)))
            }
            Token::Ident(alias) => {
                self.advance();
                // Parse optional metadata after alias (may contain search path)
                if self.at(&Token::LBrace) {
                    self.advance(); // {
                    while !self.at(&Token::RBrace) && !self.at_eof() {
                        if matches!(self.current(), Token::Ident(s) if s == "search") {
                            self.advance(); // search
                            if self.eat(&Token::Colon) {
                                if let Token::Str(s) = self.advance() {
                                    search_path = Some(s);
                                }
                            }
                        } else {
                            self.advance();
                        }
                    }
                    if self.at(&Token::RBrace) { self.advance(); }
                }
                self.expect(&Token::Semicolon)?;

                // Code import: load and parse module
                let mod_path = self.resolve_code_module(&path, search_path.as_deref())?;
                self.load_code_module(&mod_path, &alias)?;
                Ok(None)
            }
            t => bail!("expected variable or identifier after 'as', got {:?}", t),
        }
    }

    /// Parse `include "path";`
    fn parse_include(&mut self) -> Result<()> {
        self.advance(); // include
        let path = match self.advance() {
            Token::Str(s) => s,
            t => bail!("expected string after include, got {:?}", t),
        };
        // Skip optional metadata
        if self.at(&Token::LBrace) {
            while !self.at(&Token::RBrace) && !self.at_eof() { self.advance(); }
            if self.at(&Token::RBrace) { self.advance(); }
        }
        self.expect(&Token::Semicolon)?;

        // Load and parse the module without namespace prefix
        let mod_path = self.resolve_code_module(&path, None)?;
        self.load_code_module(&mod_path, "")?;
        Ok(())
    }

    /// Resolve a data module ("name" → path/name.json)
    fn resolve_data_module(&self, name: &str, search: Option<&str>) -> Result<String> {
        for dir in self.search_dirs(search) {
            let json_path = format!("{}/{}.json", dir, name);
            if std::path::Path::new(&json_path).exists() {
                return Ok(json_path);
            }
        }
        bail!("Cannot find data module '{}'", name)
    }

    /// Resolve a code module ("name" → path/name.jq or path/name/name.jq)
    fn resolve_code_module(&self, name: &str, search: Option<&str>) -> Result<String> {
        for dir in self.search_dirs(search) {
            // Try name.jq
            let jq_path = format!("{}/{}.jq", dir, name);
            if std::path::Path::new(&jq_path).exists() {
                return Ok(jq_path);
            }
            // Try name/name.jq
            let jq_path2 = format!("{}/{}/{}.jq", dir, name, name);
            if std::path::Path::new(&jq_path2).exists() {
                return Ok(jq_path2);
            }
        }
        bail!("Cannot find module '{}'", name)
    }

    /// Get search directories for module resolution
    fn search_dirs(&self, search: Option<&str>) -> Vec<String> {
        let mut dirs: Vec<String> = Vec::new();
        if let Some(s) = search {
            // Relative search paths - resolve relative to each lib_dir
            for lib_dir in &self.lib_dirs {
                let resolved = std::path::Path::new(lib_dir).join(s);
                dirs.push(resolved.to_string_lossy().into_owned());
            }
        }
        dirs.extend(self.lib_dirs.iter().cloned());
        dirs
    }

    /// Load and parse a code module, registering its functions with namespace prefix
    fn load_code_module(&mut self, file_path: &str, namespace: &str) -> Result<()> {
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| anyhow::anyhow!("Cannot load module '{}': {}", file_path, e))?;

        let mut lexer = Lexer::new(&content);
        let tokens = lexer.tokenize()?;
        let mod_token_lines = std::mem::take(&mut lexer.token_lines);

        // Add the module's directory to lib_dirs for resolving relative imports
        let mut mod_lib_dirs = self.lib_dirs.clone();
        if let Some(parent) = std::path::Path::new(file_path).parent() {
            let parent_str = parent.to_string_lossy().into_owned();
            if !mod_lib_dirs.contains(&parent_str) {
                mod_lib_dirs.insert(0, parent_str);
            }
        }

        // Parse the module tokens to extract function definitions.
        // Start module var indices after the main scope's to prevent collisions
        // when closure expressions reference main-scope variables.
        let mut mod_scope = Scope::new();
        mod_scope.next_var = self.scope.next_var;
        let mut mod_parser = Parser {
            tokens,
            token_lines: mod_token_lines,
            pos: 0,
            scope: mod_scope,
            env_var_idx: self.env_var_idx,
            lib_dirs: mod_lib_dirs,
            current_func: None,
            deferred_unbound: Vec::new(),
            deferred_unknown_func: Vec::new(),
        };

        // Skip module statement
        if matches!(mod_parser.current(), Token::Module) {
            mod_parser.advance();
            while !mod_parser.at(&Token::Semicolon) && !mod_parser.at_eof() {
                mod_parser.advance();
            }
            if mod_parser.at(&Token::Semicolon) { mod_parser.advance(); }
        }

        // Parse imports and defs in the module, collecting data import bindings
        let mut data_bindings: Vec<(u16, Expr)> = Vec::new();
        loop {
            if mod_parser.at(&Token::Def) {
                mod_parser.parse_funcdef()?;
            } else if matches!(mod_parser.current(), Token::Import) {
                let binding = mod_parser.parse_import()?;
                if let Some(b) = binding {
                    data_bindings.push(b);
                }
            } else if matches!(mod_parser.current(), Token::Include) {
                mod_parser.parse_include()?;
            } else {
                break;
            }
        }

        // Register module's functions into our scope with namespace prefix.
        // Build a func_id mapping (module-internal → main scope) so that
        // intra-module FuncCall references get remapped correctly.
        //
        // Top-level defs go through `define_func` so the namespaced name is
        // looked up by callers in the main script. Inner defs (nested inside
        // an outer def) are listed in `compiled_funcs` but not in
        // `scope.funcs` — they still need a slot in the main scope's
        // `compiled_funcs` so their `FuncCall` references resolve at runtime,
        // but they must not be reachable by name from outside their parent
        // (#638). Without this, a nested recursive `def g` inside a module's
        // exported `def f` triggered "undefined function id" at the recursive
        // self-call site.
        let mut func_id_map: Vec<(usize, usize)> = Vec::new(); // (old, new)
        for (mod_func_id, mod_func) in mod_parser.scope.compiled_funcs.iter().enumerate() {
            let top_level_name = mod_parser.scope.funcs.iter()
                .find(|(_, fid, _)| *fid == mod_func_id)
                .map(|(name, _, _)| name.clone());
            let new_id = if let Some(name) = top_level_name {
                let new_name = if namespace.is_empty() { name } else { format!("{}::{}", namespace, name) };
                self.scope.define_func(&new_name, mod_func.nargs, Expr::Empty, mod_func.param_vars.clone())
            } else {
                self.scope.define_anon_func(mod_func.nargs, Expr::Empty, mod_func.param_vars.clone())
            };
            func_id_map.push((mod_func_id, new_id));
        }

        // Second pass: remap func_ids in bodies and install them
        for (mod_func_id, new_func_id) in &func_id_map {
            let func = mod_parser.scope.compiled_funcs[*mod_func_id].clone();
            let mut body = remap_func_ids(func.body, &func_id_map);
            // Wrap function body with data import bindings (in reverse order)
            for (var_idx, value_expr) in data_bindings.iter().rev() {
                body = Expr::LetBinding {
                    var_index: *var_idx,
                    value: Box::new(value_expr.clone()),
                    body: Box::new(body),
                };
            }
            self.scope.update_func_body(*new_func_id, body, func.param_vars.clone());
        }

        // Advance main scope's var counter past module's allocations
        if mod_parser.scope.next_var > self.scope.next_var {
            self.scope.next_var = mod_parser.scope.next_var;
        }

        Ok(())
    }

    fn parse_pipe(&mut self) -> Result<Expr> {
        // pipe = assign_expr ('|' pipe)?
        let mut expr = self.parse_comma()?;

        // Check for 'as' binding: expr as $var | body
        // or 'as' pattern: expr as {a: $a} | body
        if self.at(&Token::As) {
            self.advance();
            return self.parse_as_binding(expr);
        }

        if self.eat(&Token::Pipe) {
            let right = self.parse_pipe()?;
            expr = optimize_pipe(expr, right);
        }
        Ok(expr)
    }

    /// Like parse_pipe but does not consume comma at the top level.
    /// Used for object values where comma separates key-value pairs.
    fn parse_pipe_nocomma(&mut self) -> Result<Expr> {
        let mut expr = self.parse_alt_top()?;

        // Check for 'as' binding
        if self.at(&Token::As) {
            self.advance();
            return self.parse_as_binding(expr);
        }

        if self.eat(&Token::Pipe) {
            let right = self.parse_pipe_nocomma()?;
            expr = optimize_pipe(expr, right);
        }
        Ok(expr)
    }

    fn parse_as_binding(&mut self, value_expr: Expr) -> Result<Expr> {
        // 'as' $var '|' body
        // 'as' pattern ('?//' pattern)* '|' body (destructuring with alternatives)
        let first_pattern = self.parse_pattern()?;

        // Check for ?// alternative patterns
        let mut alt_patterns: Vec<Pattern> = vec![first_pattern];
        while self.eat(&Token::AltDestructure) {
            alt_patterns.push(self.parse_pattern()?);
        }

        if alt_patterns.len() == 1 {
            // No ?// alternatives - simple binding. Snapshot the scope
            // before allocating the pattern's vars so a same-name shadow
            // (`5 as $x | (10 as $x | …), $x`) doesn't leak into outer
            // lookups after the body is parsed. See #499.
            let pattern = alt_patterns.into_iter().next().unwrap();
            let saved_vars = self.scope.vars.len();
            let allocs = self.alloc_pattern_vars(&pattern);
            self.expect(&Token::Pipe)?;
            let body = self.parse_pipe()?;
            self.scope.vars.truncate(saved_vars);
            return self.build_binding(value_expr, pattern, allocs, body);
        }

        // For ?// alternatives, all patterns must bind to the same variable names.
        // Collect unique variable names from all patterns and allocate each once.
        let mut var_names: Vec<String> = Vec::new();
        for pat in &alt_patterns {
            self.collect_pattern_var_names(pat, &mut var_names);
        }
        // Deduplicate while preserving order
        let mut seen = std::collections::HashSet::new();
        let unique_vars: Vec<String> = var_names.into_iter()
            .filter(|n| seen.insert(n.clone()))
            .collect();

        // Snapshot scope before allocating the alt-destructure vars so
        // they don't leak into outer lookups (#499).
        let saved_vars = self.scope.vars.len();
        // Allocate once for shared variables
        let mut var_map: std::collections::HashMap<String, u16> = std::collections::HashMap::new();
        for name in &unique_vars {
            let idx = self.scope.alloc_var(name);
            var_map.insert(name.clone(), idx);
        }

        self.expect(&Token::Pipe)?;
        let body = self.parse_pipe()?;
        self.scope.vars.truncate(saved_vars);

        // Build the `try (bind P1 | body) catch (bind P2 | body) …` chain through
        // the shared helper, which captures the original `.` up front and
        // restores it at the head of every alternative's body. Without that, a
        // fallback alternative would run `body` with `.` set to the caught
        // destructuring *error* string instead of the original input (#736).
        // The helper clones `value_expr` into each alternative rather than
        // sharing it through a tmp slot; only one alternative ultimately runs
        // (first success wins), matching jq.
        self.build_alt_destructure(&value_expr, &alt_patterns, &var_map, &body)
    }

    /// Allocate the shared variables for a `?//` alternative-destructuring
    /// chain. All alternatives bind the same set of names, so each unique name
    /// gets one slot (jq requires the alternatives to agree on variables). The
    /// caller is responsible for snapshotting/truncating `scope.vars`.
    fn alloc_alt_destructure_vars(&mut self, alt_patterns: &[Pattern]) -> std::collections::HashMap<String, u16> {
        let mut var_names: Vec<String> = Vec::new();
        for pat in alt_patterns {
            self.collect_pattern_var_names(pat, &mut var_names);
        }
        let mut seen = std::collections::HashSet::new();
        let mut var_map = std::collections::HashMap::new();
        for name in var_names.into_iter().filter(|n| seen.insert(n.clone())) {
            let idx = self.scope.alloc_var(&name);
            var_map.insert(name, idx);
        }
        var_map
    }

    /// Build the `?//` try-catch chain that binds `value_expr` through each
    /// alternative pattern in turn — the first whose destructuring (and body)
    /// succeeds wins; the last alternative runs without a catch so its error
    /// propagates. Mirrors the chain in `parse_as_binding`, but without the
    /// outer `LetBinding`, so callers like reduce/foreach can supply the bound
    /// value through their own element slot. See #712.
    fn build_alt_destructure(
        &mut self,
        value_expr: &Expr,
        alt_patterns: &[Pattern],
        var_map: &std::collections::HashMap<String, u16>,
        body: &Expr,
    ) -> Result<Expr> {
        // jq applies `?//` *per source value*: for each value of `value_expr`
        // it tries P1, and only the values that fail P1 fall through to P2,
        // etc. The source generator must therefore be bound ONCE, outside the
        // try/catch — binding it per alternative (re-running it inside every
        // `catch`) re-applies the fallback to values that already matched an
        // earlier pattern, so `({a:1},2) as {a:$x} ?// $x | $x` leaked the
        // whole `{a:1}` through the fallback alongside its real match (#819).
        //
        // Capture the original input in `$__altdot__` and the per-element
        // source value in `$__altsrc__`. Each alternative is run as
        // `$__altdot__ | (BIND-Pi | BODY)`, so the catch arm's `.` (set to the
        // caught error by `try … catch`) is restored to the original input for
        // both the pattern's computed keys and the body (#736/#803). The
        // source binding sits outside the try/catch, evaluated against the
        // original `.`, so reduce/foreach element sources (#712) and
        // `.`-reading sources (#736) still see the right input.
        let dot_var = self.scope.alloc_var("__altdot__");
        let src_var = self.scope.alloc_var("__altsrc__");
        let mut result: Option<Expr> = None;
        for pattern in alt_patterns.iter().rev() {
            let binding = self.build_binding_with_varmap(
                Expr::LoadVar { var_index: src_var },
                pattern,
                var_map,
                body.clone(),
            )?;
            let restored = Expr::Pipe {
                left: Box::new(Expr::LoadVar { var_index: dot_var }),
                right: Box::new(binding),
            };
            result = Some(match result {
                None => restored,
                Some(prev) => Expr::TryCatch {
                    try_expr: Box::new(restored),
                    catch_expr: Box::new(prev),
                    // `?//` keeps `.` = original input across fallbacks in a
                    // path context, so the body stays path-transparent (#840).
                    restore_dot: true,
                },
            });
        }
        let chain = result.expect("alt_patterns is non-empty");
        // `value_expr as $__altsrc__ | chain`, iterated per source value.
        let src_binding = Expr::LetBinding {
            var_index: src_var,
            value: Box::new(value_expr.clone()),
            body: Box::new(chain),
        };
        Ok(Expr::LetBinding {
            var_index: dot_var,
            value: Box::new(Expr::Input),
            body: Box::new(src_binding),
        })
    }

    fn collect_pattern_var_names(&self, pattern: &Pattern, names: &mut Vec<String>) {
        match pattern {
            Pattern::Var(name) => names.push(name.clone()),
            Pattern::Array(pats) => {
                for p in pats { self.collect_pattern_var_names(p, names); }
            }
            Pattern::Object(pats) => {
                for (_, p) in pats { self.collect_pattern_var_names(p, names); }
            }
            Pattern::VarAndSub(name, sub) => {
                names.push(name.clone());
                self.collect_pattern_var_names(sub, names);
            }
        }
    }

    fn build_binding_with_varmap(&mut self, value_expr: Expr, pattern: &Pattern, var_map: &std::collections::HashMap<String, u16>, body: Expr) -> Result<Expr> {
        match pattern {
            Pattern::Var(name) => {
                let var_idx = var_map[name];
                Ok(Expr::LetBinding {
                    var_index: var_idx,
                    value: Box::new(value_expr),
                    body: Box::new(body),
                })
            }
            Pattern::Array(pats) => {
                let tmp_idx = self.scope.alloc_var("__destruct_tmp__");
                let tmp_ref = Expr::LoadVar { var_index: tmp_idx };
                let inner = self.build_array_destructure_varmap(tmp_ref, pats, var_map, body)?;
                Ok(Expr::LetBinding {
                    var_index: tmp_idx,
                    value: Box::new(value_expr),
                    body: Box::new(inner),
                })
            }
            Pattern::Object(pats) => {
                let tmp_idx = self.scope.alloc_var("__destruct_tmp__");
                let tmp_ref = Expr::LoadVar { var_index: tmp_idx };
                let inner = self.build_object_destructure_varmap(tmp_ref, pats, var_map, body)?;
                Ok(Expr::LetBinding {
                    var_index: tmp_idx,
                    value: Box::new(value_expr),
                    body: Box::new(inner),
                })
            }
            Pattern::VarAndSub(name, sub) => {
                let inner = self.build_binding_with_varmap(value_expr.clone(), sub, var_map, body)?;
                let var_idx = var_map[name];
                Ok(Expr::LetBinding {
                    var_index: var_idx,
                    value: Box::new(value_expr),
                    body: Box::new(inner),
                })
            }
        }
    }

    fn build_array_destructure_varmap(&mut self, value: Expr, pats: &[Pattern], var_map: &std::collections::HashMap<String, u16>, body: Expr) -> Result<Expr> {
        let mut result = body;
        for (i, pat) in pats.iter().enumerate().rev() {
            let elem_expr = Expr::Index {
                expr: Box::new(value.clone()),
                key: Box::new(Expr::Literal(Literal::Num(i as f64, None))),
            };
            result = self.build_pattern_binding_varmap(pat, elem_expr, var_map, result)?;
        }
        Ok(result)
    }

    fn build_object_destructure_varmap(&mut self, value: Expr, pats: &[(Expr, Pattern)], var_map: &std::collections::HashMap<String, u16>, body: Expr) -> Result<Expr> {
        let mut result = body;
        for (key, pat) in pats.iter().rev() {
            let field_expr = obj_pat_field_expr(value.clone(), key.clone());
            result = self.build_pattern_binding_varmap(pat, field_expr, var_map, result)?;
        }
        Ok(result)
    }

    fn build_pattern_binding_varmap(&mut self, pat: &Pattern, value: Expr, var_map: &std::collections::HashMap<String, u16>, body: Expr) -> Result<Expr> {
        match pat {
            Pattern::Var(name) => {
                let var_idx = var_map[name];
                Ok(Expr::LetBinding {
                    var_index: var_idx,
                    value: Box::new(value),
                    body: Box::new(body),
                })
            }
            Pattern::Array(sub_pats) => {
                let tmp_idx = self.scope.alloc_var("__destruct_tmp__");
                let tmp_ref = Expr::LoadVar { var_index: tmp_idx };
                let inner = self.build_array_destructure_varmap(tmp_ref, sub_pats, var_map, body)?;
                Ok(Expr::LetBinding {
                    var_index: tmp_idx,
                    value: Box::new(value),
                    body: Box::new(inner),
                })
            }
            Pattern::Object(sub_pats) => {
                let tmp_idx = self.scope.alloc_var("__destruct_tmp__");
                let tmp_ref = Expr::LoadVar { var_index: tmp_idx };
                let inner = self.build_object_destructure_varmap(tmp_ref, sub_pats, var_map, body)?;
                Ok(Expr::LetBinding {
                    var_index: tmp_idx,
                    value: Box::new(value),
                    body: Box::new(inner),
                })
            }
            Pattern::VarAndSub(name, sub) => {
                let inner = self.build_pattern_binding_varmap(sub, value.clone(), var_map, body)?;
                let var_idx = var_map[name];
                Ok(Expr::LetBinding {
                    var_index: var_idx,
                    value: Box::new(value),
                    body: Box::new(inner),
                })
            }
        }
    }

    fn build_binding(&mut self, value_expr: Expr, pattern: Pattern, allocs: Vec<u16>, body: Expr) -> Result<Expr> {
        match pattern {
            Pattern::Var(_name) => {
                let var_idx = allocs[0];
                Ok(Expr::LetBinding {
                    var_index: var_idx,
                    value: Box::new(value_expr),
                    body: Box::new(body),
                })
            }
            Pattern::Array(pats) => {
                let tmp_idx = self.scope.alloc_var("__destruct_tmp__");
                let tmp_ref = Expr::LoadVar { var_index: tmp_idx };
                let inner = self.build_array_destructure(tmp_ref, &pats, &allocs, body)?;
                Ok(Expr::LetBinding {
                    var_index: tmp_idx,
                    value: Box::new(value_expr),
                    body: Box::new(inner),
                })
            }
            Pattern::Object(pats) => {
                let tmp_idx = self.scope.alloc_var("__destruct_tmp__");
                let tmp_ref = Expr::LoadVar { var_index: tmp_idx };
                let inner = self.build_object_destructure(tmp_ref, &pats, &allocs, body)?;
                Ok(Expr::LetBinding {
                    var_index: tmp_idx,
                    value: Box::new(value_expr),
                    body: Box::new(inner),
                })
            }
            Pattern::VarAndSub(_, sub) => {
                let inner = self.build_binding(value_expr.clone(), *sub, allocs[1..].to_vec(), body)?;
                Ok(Expr::LetBinding {
                    var_index: allocs[0],
                    value: Box::new(value_expr),
                    body: Box::new(inner),
                })
            }
        }
    }

    /// Pre-allocate all variables from a pattern before parsing the body.
    fn alloc_pattern_vars(&mut self, pattern: &Pattern) -> Vec<u16> {
        // For repeated names within a single pattern, share the same slot —
        // necessary so object-pattern first-wins (#206) and array-pattern
        // last-wins land on the right binding without false aliasing across
        // separate names.
        let mut seen: std::collections::HashMap<String, u16> = std::collections::HashMap::new();
        self.alloc_pattern_vars_inner(pattern, &mut seen)
    }

    fn alloc_pattern_vars_inner(&mut self, pattern: &Pattern, seen: &mut std::collections::HashMap<String, u16>) -> Vec<u16> {
        match pattern {
            Pattern::Var(name) => {
                let idx = if let Some(&existing) = seen.get(name) {
                    existing
                } else {
                    let idx = self.scope.alloc_var(name);
                    seen.insert(name.clone(), idx);
                    idx
                };
                vec![idx]
            }
            Pattern::Array(pats) => {
                pats.iter().flat_map(|p| self.alloc_pattern_vars_inner(p, seen)).collect()
            }
            Pattern::Object(pats) => {
                pats.iter().flat_map(|(_, p)| self.alloc_pattern_vars_inner(p, seen)).collect()
            }
            Pattern::VarAndSub(name, sub) => {
                let head_idx = if let Some(&existing) = seen.get(name) {
                    existing
                } else {
                    let idx = self.scope.alloc_var(name);
                    seen.insert(name.clone(), idx);
                    idx
                };
                let mut vars = vec![head_idx];
                vars.extend(self.alloc_pattern_vars_inner(sub, seen));
                vars
            }
        }
    }

    fn parse_pattern(&mut self) -> Result<Pattern> {
        match self.current().clone() {
            Token::Variable(name) => {
                // jq rejects the reserved `$__loc__` as a binding target with a
                // compile-time syntax error (it is a loc literal, not a BINDING).
                // `$ENV` is allowed and shadows the built-in. See #886.
                if name == "__loc__" {
                    bail!("syntax error, unexpected $__loc__, expecting BINDING or '[' or '{{'");
                }
                self.advance();
                Ok(Pattern::Var(name))
            }
            Token::LBracket => {
                self.advance();
                let mut pats = Vec::new();
                while !self.at(&Token::RBracket) && !self.at_eof() {
                    pats.push(self.parse_pattern()?);
                    if !self.eat(&Token::Comma) { break; }
                }
                self.expect(&Token::RBracket)?;
                Ok(Pattern::Array(pats))
            }
            Token::LBrace => {
                self.advance();
                let mut pats = Vec::new();
                while !self.at(&Token::RBrace) && !self.at_eof() {
                    // {key: $var} or {$var} (shorthand)
                    let (key, pat) = self.parse_obj_pattern_pair()?;
                    pats.push((key, pat));
                    if !self.eat(&Token::Comma) { break; }
                }
                self.expect(&Token::RBrace)?;
                Ok(Pattern::Object(pats))
            }
            _ => bail!("expected pattern (variable, array, or object), got {:?}", self.current()),
        }
    }

    fn parse_obj_pattern_pair(&mut self) -> Result<(Expr, Pattern)> {
        match self.current().clone() {
            Token::Variable(name) => {
                // jq rejects `$__loc__` as a binding target (see parse_pattern).
                if name == "__loc__" {
                    bail!("syntax error, unexpected $__loc__, expecting BINDING or '[' or '{{'");
                }
                self.advance();
                if self.eat(&Token::Colon) {
                    // $var: pattern — key is variable name, bind $var AND destructure
                    let sub_pat = self.parse_pattern()?;
                    Ok((Expr::Literal(Literal::Str(name.clone())), Pattern::VarAndSub(name, Box::new(sub_pat))))
                } else {
                    // $var shorthand — key is variable name, bind to $var
                    Ok((Expr::Literal(Literal::Str(name.clone())), Pattern::Var(name)))
                }
            }
            Token::Ident(key) | Token::Str(key) => {
                self.advance();
                self.expect(&Token::Colon)?;
                let pat = self.parse_pattern()?;
                Ok((Expr::Literal(Literal::Str(key)), pat))
            }
            Token::LParen => {
                // Computed key pattern: (expr): $var — defer key evaluation
                // to runtime so non-literal expressions like `(.x)` work.
                self.advance();
                let key_expr = self.parse_pipe()?;
                self.expect(&Token::RParen)?;
                // jq rejects a constant non-string literal key at compile time
                // ("Cannot use number (0) as object key"); jq-jit otherwise
                // deferred to runtime and treated `(0)` as the index `.[0]`,
                // returning a value for array input. Only literal constants are
                // rejected — runtime expressions like `(.x)` / `(0|tostring)`
                // and a negated `(-0)` (a Negate node, not a literal) defer as
                // before. See #888.
                if let Expr::Literal(lit) = &key_expr {
                    let bad = match lit {
                        Literal::Num(n, _) => Some(("number", crate::value::format_jq_number(*n))),
                        Literal::Null => Some(("null", "null".to_string())),
                        Literal::True => Some(("boolean", "true".to_string())),
                        Literal::False => Some(("boolean", "false".to_string())),
                        Literal::Str(_) => None,
                    };
                    if let Some((ty, val)) = bad {
                        bail!("Cannot use {} ({}) as object key", ty, val);
                    }
                }
                self.expect(&Token::Colon)?;
                let pat = self.parse_pattern()?;
                Ok((key_expr, pat))
            }
            ref tok if Self::keyword_as_string(tok).is_some() => {
                let key = Self::keyword_as_string(tok).unwrap().to_string();
                self.advance();
                self.expect(&Token::Colon)?;
                let pat = self.parse_pattern()?;
                Ok((Expr::Literal(Literal::Str(key)), pat))
            }
            _ => bail!("expected object pattern key, got {:?}", self.current()),
        }
    }

    fn build_array_destructure(&mut self, value: Expr, pats: &[Pattern], allocs: &[u16], body: Expr) -> Result<Expr> {
        let mut result = body;
        let mut alloc_idx = allocs.len();
        for (i, pat) in pats.iter().enumerate().rev() {
            let count = self.count_pattern_vars(pat);
            alloc_idx -= count;
            let elem_expr = Expr::Index {
                expr: Box::new(value.clone()),
                key: Box::new(Expr::Literal(Literal::Num(i as f64, None))),
            };
            result = self.build_pattern_binding(pat, elem_expr, &allocs[alloc_idx..alloc_idx+count], result)?;
        }
        Ok(result)
    }

    fn build_object_destructure(&mut self, value: Expr, pats: &[(Expr, Pattern)], allocs: &[u16], body: Expr) -> Result<Expr> {
        // jq's object-pattern destructure is first-wins for repeated variable
        // names in the same pattern (#206). With slot dedup in
        // `alloc_pattern_vars`, all references to `$a` map to one idx — so a
        // later pair that only binds already-seen idx would shadow the first.
        // Pre-scan source-order to mark pairs whose entire binding set is
        // covered by an earlier pair, then drop them when emitting.
        let mut pair_offsets: Vec<usize> = Vec::with_capacity(pats.len());
        let mut acc = 0;
        for (_, pat) in pats {
            pair_offsets.push(acc);
            acc += self.count_pattern_vars(pat);
        }
        let mut keep: Vec<bool> = vec![false; pats.len()];
        let mut bound: std::collections::HashSet<u16> = std::collections::HashSet::new();
        for (i, (_, pat)) in pats.iter().enumerate() {
            let count = self.count_pattern_vars(pat);
            let pair_allocs = &allocs[pair_offsets[i]..pair_offsets[i]+count];
            if pair_allocs.is_empty() {
                keep[i] = true;
                continue;
            }
            let has_new = pair_allocs.iter().any(|idx| !bound.contains(idx));
            if has_new {
                keep[i] = true;
                for &idx in pair_allocs { bound.insert(idx); }
            }
        }

        let mut result = body;
        for (i, (key, pat)) in pats.iter().enumerate().rev() {
            if !keep[i] { continue; }
            let count = self.count_pattern_vars(pat);
            let pair_allocs = &allocs[pair_offsets[i]..pair_offsets[i]+count];
            let field_expr = obj_pat_field_expr(value.clone(), key.clone());
            result = self.build_pattern_binding(pat, field_expr, pair_allocs, result)?;
        }
        Ok(result)
    }

    /// Recursively build bindings for a pattern. Handles nested arrays and objects.
    fn build_pattern_binding(&mut self, pat: &Pattern, value: Expr, allocs: &[u16], body: Expr) -> Result<Expr> {
        match pat {
            Pattern::Var(_) => {
                Ok(Expr::LetBinding {
                    var_index: allocs[0],
                    value: Box::new(value),
                    body: Box::new(body),
                })
            }
            Pattern::Array(sub_pats) => {
                let tmp_idx = self.scope.alloc_var("__destruct_tmp__");
                let tmp_ref = Expr::LoadVar { var_index: tmp_idx };
                let inner = self.build_array_destructure(tmp_ref, sub_pats, allocs, body)?;
                Ok(Expr::LetBinding {
                    var_index: tmp_idx,
                    value: Box::new(value),
                    body: Box::new(inner),
                })
            }
            Pattern::Object(sub_pats) => {
                let tmp_idx = self.scope.alloc_var("__destruct_tmp__");
                let tmp_ref = Expr::LoadVar { var_index: tmp_idx };
                let inner = self.build_object_destructure(tmp_ref, sub_pats, allocs, body)?;
                Ok(Expr::LetBinding {
                    var_index: tmp_idx,
                    value: Box::new(value),
                    body: Box::new(inner),
                })
            }
            Pattern::VarAndSub(_, sub) => {
                // Bind $var to the whole value, then destructure via sub-pattern
                let inner = self.build_pattern_binding(sub, value.clone(), &allocs[1..], body)?;
                Ok(Expr::LetBinding {
                    var_index: allocs[0],
                    value: Box::new(value),
                    body: Box::new(inner),
                })
            }
        }
    }

    fn count_pattern_vars(&self, pattern: &Pattern) -> usize {
        match pattern {
            Pattern::Var(_) => 1,
            Pattern::Array(pats) => pats.iter().map(|p| self.count_pattern_vars(p)).sum(),
            Pattern::Object(pats) => pats.iter().map(|(_, p)| self.count_pattern_vars(p)).sum(),
            Pattern::VarAndSub(_, sub) => 1 + self.count_pattern_vars(sub),
        }
    }

    fn parse_comma(&mut self) -> Result<Expr> {
        let mut expr = self.parse_alt_top()?;
        while self.eat(&Token::Comma) {
            let right = self.parse_alt_top()?;
            expr = Expr::Comma {
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    /// `//` sits between `,` and `=`/`|=` in jq's precedence table — lower
    /// than the assignment operators, so `.a |= . // 0` parses as
    /// `(.a |= .) // 0`, not `.a |= (. // 0)` (issue #155). Right-associative.
    fn parse_alt_top(&mut self) -> Result<Expr> {
        let expr = self.parse_assign()?;
        if self.eat(&Token::Alt) {
            let right = self.parse_alt_top()?;
            return Ok(Expr::Alternative {
                primary: Box::new(expr),
                fallback: Box::new(right),
            });
        }
        Ok(expr)
    }

    fn parse_assign(&mut self) -> Result<Expr> {
        let expr = self.parse_or()?;

        match self.current() {
            Token::Assign => {
                self.advance();
                let value = self.parse_or()?;
                Ok(Expr::Assign {
                    path_expr: Box::new(expr),
                    value_expr: Box::new(value),
                })
            }
            Token::UpdateAssign => {
                self.advance();
                let update = self.parse_or()?;
                // `path |= empty` deletes the path and yields the modified
                // container exactly once. The generic update path emits zero
                // outputs when the update generator never fires, so rewrite
                // to `del(path)` at parse time — both eval and JIT then see
                // the deletion form (issue #155).
                if matches!(&update, Expr::Empty) {
                    return Ok(Expr::CallBuiltin {
                        name: "del".to_string(),
                        args: vec![expr],
                    });
                }
                Ok(Expr::Update {
                    path_expr: Box::new(expr),
                    update_expr: Box::new(update),
                })
            }
            Token::AddAssign => {
                self.advance();
                let rhs = self.parse_or()?;
                let rhs_var = self.scope.alloc_var("__opassign_rhs__");
                Ok(Expr::LetBinding {
                    var_index: rhs_var,
                    value: Box::new(rhs),
                    body: Box::new(Expr::Update {
                        path_expr: Box::new(expr),
                        update_expr: Box::new(Expr::BinOp {
                            op: BinOp::Add,
                            lhs: Box::new(Expr::Input),
                            rhs: Box::new(Expr::LoadVar { var_index: rhs_var }),
                        }),
                    }),
                })
            }
            Token::SubAssign => {
                self.advance();
                let rhs = self.parse_or()?;
                let rhs_var = self.scope.alloc_var("__opassign_rhs__");
                Ok(Expr::LetBinding {
                    var_index: rhs_var,
                    value: Box::new(rhs),
                    body: Box::new(Expr::Update {
                        path_expr: Box::new(expr),
                        update_expr: Box::new(Expr::BinOp {
                            op: BinOp::Sub,
                            lhs: Box::new(Expr::Input),
                            rhs: Box::new(Expr::LoadVar { var_index: rhs_var }),
                        }),
                    }),
                })
            }
            Token::MulAssign => {
                self.advance();
                let rhs = self.parse_or()?;
                let rhs_var = self.scope.alloc_var("__opassign_rhs__");
                Ok(Expr::LetBinding {
                    var_index: rhs_var,
                    value: Box::new(rhs),
                    body: Box::new(Expr::Update {
                        path_expr: Box::new(expr),
                        update_expr: Box::new(Expr::BinOp {
                            op: BinOp::Mul,
                            lhs: Box::new(Expr::Input),
                            rhs: Box::new(Expr::LoadVar { var_index: rhs_var }),
                        }),
                    }),
                })
            }
            Token::DivAssign => {
                self.advance();
                let rhs = self.parse_or()?;
                let rhs_var = self.scope.alloc_var("__opassign_rhs__");
                Ok(Expr::LetBinding {
                    var_index: rhs_var,
                    value: Box::new(rhs),
                    body: Box::new(Expr::Update {
                        path_expr: Box::new(expr),
                        update_expr: Box::new(Expr::BinOp {
                            op: BinOp::Div,
                            lhs: Box::new(Expr::Input),
                            rhs: Box::new(Expr::LoadVar { var_index: rhs_var }),
                        }),
                    }),
                })
            }
            Token::ModAssign => {
                self.advance();
                let rhs = self.parse_or()?;
                let rhs_var = self.scope.alloc_var("__opassign_rhs__");
                Ok(Expr::LetBinding {
                    var_index: rhs_var,
                    value: Box::new(rhs),
                    body: Box::new(Expr::Update {
                        path_expr: Box::new(expr),
                        update_expr: Box::new(Expr::BinOp {
                            op: BinOp::Mod,
                            lhs: Box::new(Expr::Input),
                            rhs: Box::new(Expr::LoadVar { var_index: rhs_var }),
                        }),
                    }),
                })
            }
            Token::AltAssign => {
                self.advance();
                let rhs = self.parse_or()?;
                let rhs_var = self.scope.alloc_var("__opassign_rhs__");
                Ok(Expr::LetBinding {
                    var_index: rhs_var,
                    value: Box::new(rhs),
                    body: Box::new(Expr::Update {
                        path_expr: Box::new(expr),
                        update_expr: Box::new(Expr::Alternative {
                            primary: Box::new(Expr::Input),
                            fallback: Box::new(Expr::LoadVar { var_index: rhs_var }),
                        }),
                    }),
                })
            }
            _ => Ok(expr),
        }
    }

    fn parse_or(&mut self) -> Result<Expr> {
        let mut expr = self.parse_and()?;
        while self.eat(&Token::Or) {
            let right = self.parse_and()?;
            expr = Expr::BinOp {
                op: BinOp::Or,
                lhs: Box::new(expr),
                rhs: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_and(&mut self) -> Result<Expr> {
        let mut expr = self.parse_not()?;
        while self.eat(&Token::And) {
            let right = self.parse_not()?;
            expr = Expr::BinOp {
                op: BinOp::And,
                lhs: Box::new(expr),
                rhs: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_not(&mut self) -> Result<Expr> {
        let expr = self.parse_compare()?;
        // 'not' is postfix in jq
        // It's handled as a builtin function call
        Ok(expr)
    }

    fn parse_compare(&mut self) -> Result<Expr> {
        // `==`/`!=`/`<`/`>` etc. bind tighter than `//`, so the operands
        // stay below the `//` rung — read parse_add directly. The `//`
        // operator is handled at parse_alt_top above parse_assign.
        let mut expr = self.parse_add()?;
        // Check for ?// (alternative destructuring)
        while self.at(&Token::AltDestructure) {
            self.advance();
            let right = self.parse_add()?;
            expr = Expr::AlternativeDestructure {
                alternatives: vec![expr, right],
            };
        }
        loop {
            let op = match self.current() {
                Token::Eq => BinOp::Eq,
                Token::Ne => BinOp::Ne,
                Token::Lt => BinOp::Lt,
                Token::Gt => BinOp::Gt,
                Token::Le => BinOp::Le,
                Token::Ge => BinOp::Ge,
                _ => break,
            };
            self.advance();
            let right = self.parse_add()?;
            expr = Expr::BinOp {
                op,
                lhs: Box::new(expr),
                rhs: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_add(&mut self) -> Result<Expr> {
        let mut expr = self.parse_mul()?;
        loop {
            match self.current() {
                Token::Plus => {
                    self.advance();
                    let right = self.parse_mul()?;
                    expr = Expr::BinOp {
                        op: BinOp::Add,
                        lhs: Box::new(expr),
                        rhs: Box::new(right),
                    };
                }
                Token::Minus => {
                    self.advance();
                    let right = self.parse_mul()?;
                    expr = Expr::BinOp {
                        op: BinOp::Sub,
                        lhs: Box::new(expr),
                        rhs: Box::new(right),
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_mul(&mut self) -> Result<Expr> {
        let mut expr = self.parse_unary()?;
        loop {
            match self.current() {
                Token::Star => {
                    self.advance();
                    let right = self.parse_unary()?;
                    expr = Expr::BinOp {
                        op: BinOp::Mul,
                        lhs: Box::new(expr),
                        rhs: Box::new(right),
                    };
                }
                Token::Slash => {
                    self.advance();
                    let right = self.parse_unary()?;
                    expr = Expr::BinOp {
                        op: BinOp::Div,
                        lhs: Box::new(expr),
                        rhs: Box::new(right),
                    };
                }
                Token::Percent => {
                    self.advance();
                    let right = self.parse_unary()?;
                    expr = Expr::BinOp {
                        op: BinOp::Mod,
                        lhs: Box::new(expr),
                        rhs: Box::new(right),
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        if self.eat(&Token::Minus) {
            // Allow chained unary minus: `- -1`, `- - -1`.
            let operand = self.parse_unary()?;
            Ok(Expr::Negate { operand: Box::new(operand) })
        } else {
            self.parse_postfix()
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.current() {
                Token::Dot => {
                    // .field or .["field"]
                    self.advance();
                    match self.current().clone() {
                        Token::Ident(field) => {
                            self.advance();
                            let key = Expr::Literal(Literal::Str(field));
                            let optional = self.eat(&Token::Question);
                            if optional {
                                expr = Expr::IndexOpt {
                                    expr: Box::new(expr),
                                    key: Box::new(key),
                                };
                            } else {
                                expr = Expr::Index {
                                    expr: Box::new(expr),
                                    key: Box::new(key),
                                };
                            }
                        }
                        Token::Str(field) => {
                            self.advance();
                            let key = Expr::Literal(Literal::Str(field));
                            let optional = self.eat(&Token::Question);
                            if optional {
                                expr = Expr::IndexOpt {
                                    expr: Box::new(expr),
                                    key: Box::new(key),
                                };
                            } else {
                                expr = Expr::Index {
                                    expr: Box::new(expr),
                                    key: Box::new(key),
                                };
                            }
                        }
                        Token::LBracket => {
                            // .expr.[] or .expr.[key] — handle like LBracket postfix
                            // Don't advance — let the LBracket case handle it
                            continue;
                        }
                        _ => {
                            // Just a trailing dot - this shouldn't normally happen after a postfix
                            // Put back the dot context
                            self.pos -= 1;
                            break;
                        }
                    }
                }
                Token::LBracket => {
                    self.advance();
                    if self.eat(&Token::RBracket) {
                        // .[]
                        let optional = self.eat(&Token::Question);
                        if optional {
                            expr = Expr::EachOpt { input_expr: Box::new(expr) };
                        } else {
                            expr = Expr::Each { input_expr: Box::new(expr) };
                        }
                    } else {
                        // Check for slice: .[from:to]
                        let first = if self.at(&Token::Colon) {
                            None
                        } else {
                            Some(self.parse_pipe()?)
                        };
                        if self.eat(&Token::Colon) {
                            // Slice
                            let second = if self.at(&Token::RBracket) {
                                None
                            } else {
                                Some(self.parse_pipe()?)
                            };
                            // jq rejects `.[:]` (both bounds absent) at parse
                            // time; the explicit `.[null:null]` form is fine.
                            // See #438.
                            if first.is_none() && second.is_none() {
                                anyhow::bail!("syntax error, unexpected ']'");
                            }
                            self.expect(&Token::RBracket)?;
                            let _optional = self.eat(&Token::Question);
                            expr = Expr::Slice {
                                expr: Box::new(expr),
                                from: first.map(Box::new),
                                to: second.map(Box::new),
                            };
                        } else {
                            // Regular index
                            let key = first.unwrap();
                            self.expect(&Token::RBracket)?;
                            let optional = self.eat(&Token::Question);
                            if optional {
                                expr = Expr::IndexOpt {
                                    expr: Box::new(expr),
                                    key: Box::new(key),
                                };
                            } else {
                                expr = Expr::Index {
                                    expr: Box::new(expr),
                                    key: Box::new(key),
                                };
                            }
                        }
                    }
                }
                Token::Question => {
                    self.advance();
                    // Try-catch with empty catch
                    expr = Expr::TryCatch {
                        try_expr: Box::new(expr),
                        catch_expr: Box::new(Expr::Empty),
                        restore_dot: false,
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        let tok = self.current().clone();
        match tok {
            Token::Dot => {
                self.advance();
                // Check for .field
                match self.current().clone() {
                    Token::Ident(field) => {
                        self.advance();
                        let optional = self.eat(&Token::Question);
                        if optional {
                            Ok(Expr::IndexOpt {
                                expr: Box::new(Expr::Input),
                                key: Box::new(Expr::Literal(Literal::Str(field))),
                            })
                        } else {
                            Ok(Expr::Index {
                                expr: Box::new(Expr::Input),
                                key: Box::new(Expr::Literal(Literal::Str(field))),
                            })
                        }
                    }
                    Token::Str(field) => {
                        self.advance();
                        let optional = self.eat(&Token::Question);
                        if optional {
                            Ok(Expr::IndexOpt {
                                expr: Box::new(Expr::Input),
                                key: Box::new(Expr::Literal(Literal::Str(field))),
                            })
                        } else {
                            Ok(Expr::Index {
                                expr: Box::new(Expr::Input),
                                key: Box::new(Expr::Literal(Literal::Str(field))),
                            })
                        }
                    }
                    Token::LBracket => {
                        self.advance();
                        if self.eat(&Token::RBracket) {
                            // .[]
                            let optional = self.eat(&Token::Question);
                            if optional {
                                Ok(Expr::EachOpt { input_expr: Box::new(Expr::Input) })
                            } else {
                                Ok(Expr::Each { input_expr: Box::new(Expr::Input) })
                            }
                        } else {
                            // .[expr] or .[from:to]
                            let first = if self.at(&Token::Colon) {
                                None
                            } else {
                                Some(self.parse_pipe()?)
                            };
                            if self.eat(&Token::Colon) {
                                let second = if self.at(&Token::RBracket) {
                                    None
                                } else {
                                    Some(self.parse_pipe()?)
                                };
                                if first.is_none() && second.is_none() {
                                    anyhow::bail!("syntax error, unexpected ']'");
                                }
                                self.expect(&Token::RBracket)?;
                                Ok(Expr::Slice {
                                    expr: Box::new(Expr::Input),
                                    from: first.map(Box::new),
                                    to: second.map(Box::new),
                                })
                            } else {
                                let key = first.unwrap();
                                self.expect(&Token::RBracket)?;
                                let optional = self.eat(&Token::Question);
                                if optional {
                                    Ok(Expr::IndexOpt {
                                        expr: Box::new(Expr::Input),
                                        key: Box::new(key),
                                    })
                                } else {
                                    Ok(Expr::Index {
                                        expr: Box::new(Expr::Input),
                                        key: Box::new(key),
                                    })
                                }
                            }
                        }
                    }
                    _ => Ok(Expr::Input), // bare '.'
                }
            }

            Token::Null => { self.advance(); Ok(Expr::Literal(Literal::Null)) }
            Token::True => { self.advance(); Ok(Expr::Literal(Literal::True)) }
            Token::False => { self.advance(); Ok(Expr::Literal(Literal::False)) }
            Token::Num(n, repr) => { self.advance(); Ok(Expr::Literal(Literal::Num(n, repr))) }
            Token::Str(s) => { self.advance(); Ok(Expr::Literal(Literal::Str(s))) }

            Token::Recurse => {
                self.advance();
                // `..` is `recurse`, which jq defines as `recurse(.[]?)`.
                // Use `EachOpt(Input)` as the AST step so eval/JIT can
                // distinguish the descent-only 0-arg form from the
                // user-written `recurse(.)` (which loops forever). See #497.
                Ok(Expr::Recurse {
                    input_expr: Box::new(Expr::EachOpt { input_expr: Box::new(Expr::Input) }),
                })
            }

            Token::LParen => {
                self.advance();
                let saved = self.scope.save_func_scope();
                let expr = self.parse_pipe()?;
                self.scope.restore_func_scope(saved);
                self.expect(&Token::RParen)?;
                Ok(expr)
            }

            Token::LBracket => {
                self.advance();
                if self.eat(&Token::RBracket) {
                    Ok(Expr::Collect { generator: Box::new(Expr::Empty) })
                } else {
                    let saved = self.scope.save_func_scope();
                    let inner = self.parse_pipe()?;
                    self.scope.restore_func_scope(saved);
                    self.expect(&Token::RBracket)?;
                    Ok(Expr::Collect { generator: Box::new(inner) })
                }
            }

            Token::LBrace => {
                self.advance();
                let saved = self.scope.save_func_scope();
                let mut pairs = Vec::new();
                if !self.at(&Token::RBrace) {
                    loop {
                        let (key_expr, val_expr) = self.parse_object_pair()?;
                        pairs.push((key_expr, val_expr));
                        if !self.eat(&Token::Comma) { break; }
                        // Allow a trailing comma (`{a:1,}`, `{a,}`, etc.)
                        // to match jq 1.8.1's parser.
                        if self.at(&Token::RBrace) { break; }
                    }
                }
                self.scope.restore_func_scope(saved);
                self.expect(&Token::RBrace)?;
                Ok(Expr::ObjectConstruct { pairs })
            }

            Token::If => {
                self.advance();
                self.parse_if_then_else()
            }

            Token::Try => {
                self.advance();
                let try_expr = self.parse_unary()?;
                let catch_expr = if self.eat(&Token::Catch) {
                    self.parse_unary()?
                } else {
                    Expr::Empty
                };
                Ok(Expr::TryCatch {
                    try_expr: Box::new(try_expr),
                    catch_expr: Box::new(catch_expr),
                    restore_dot: false,
                })
            }

            Token::Reduce => {
                self.advance();
                self.parse_reduce()
            }

            Token::Foreach => {
                self.advance();
                self.parse_foreach()
            }

            Token::Label => {
                self.advance();
                match self.advance() {
                    Token::Variable(name) => {
                        // Register label-typed bindings under a sentinel prefix so they
                        // can only be referenced via `break $name`, never as a bare $name.
                        // The binding is lexically scoped to the body: pop it afterwards
                        // so a sibling `break $name` following an inner `label $name`
                        // resolves to the still-live *outer* label rather than the
                        // already-finished inner one (which `lookup_var` would otherwise
                        // pick as the innermost match, dropping the whole output). #776
                        let saved_vars = self.scope.vars.len();
                        let var_idx = self.scope.alloc_var(&format!("\x00label:{}", name));
                        self.expect(&Token::Pipe)?;
                        let body = self.parse_pipe()?;
                        self.scope.vars.truncate(saved_vars);
                        Ok(Expr::Label {
                            var_index: var_idx,
                            body: Box::new(body),
                        })
                    }
                    t => bail!("expected $variable after label, got {:?}", t),
                }
            }

            Token::Break => {
                self.advance();
                match self.advance() {
                    Token::Variable(name) => {
                        let var_idx = self.scope.lookup_var(&format!("\x00label:{}", name))
                            .ok_or_else(|| anyhow::anyhow!("${} is not defined", name))?;
                        Ok(Expr::Break {
                            var_index: var_idx,
                            value: Box::new(Expr::Input),
                        })
                    }
                    t => bail!("expected $variable after break, got {:?}", t),
                }
            }

            Token::Empty => {
                self.advance();
                Ok(Expr::Empty)
            }

            Token::Error => {
                self.advance();
                if self.eat(&Token::LParen) {
                    let msg = self.parse_pipe()?;
                    self.expect(&Token::RParen)?;
                    Ok(Expr::Error { msg: Some(Box::new(msg)) })
                } else {
                    Ok(Expr::Error { msg: None })
                }
            }

            Token::Not => {
                self.advance();
                Ok(Expr::Not)
            }

            Token::Variable(name) => {
                let loc_line = self.current_line();
                self.advance();
                // Check for $var::name (namespace access for data imports)
                if self.at(&Token::Colon) && matches!(self.tokens.get(self.pos + 1), Some(Token::Colon)) {
                    self.advance(); // first :
                    self.advance(); // second :
                    match self.advance() {
                        Token::Ident(_member) => {
                            // $var::name is equivalent to $var for data imports
                        }
                        t => bail!("expected identifier after '::', got {:?}", t),
                    }
                }
                if name == "__loc__" {
                    Ok(Expr::Loc { file: "<top-level>".to_string(), line: loc_line as i64 })
                } else if name == "ENV" {
                    Ok(self.resolve_env_ref())
                } else {
                    Ok(self.load_or_defer_var(&name))
                }
            }

            Token::Format(name) => {
                self.advance();
                // @base64, @uri, etc.
                // May be followed by a string for @base64 "str" or interpolated string
                if matches!(self.current(), Token::Str(_)) {
                    let s = match self.advance() {
                        Token::Str(s) => s,
                        _ => unreachable!(),
                    };
                    Ok(Expr::Format {
                        name,
                        expr: Box::new(Expr::Literal(Literal::Str(s))),
                    })
                } else if matches!(self.current(), Token::Ident(n) if n == "__string_interp__") {
                    // Interpolated string after format: @html "<b>\(.)</b>"
                    // Apply format to each interpolated expr, not literals
                    self.advance();
                    let interp = self.parse_string_interpolation()?;
                    if let Expr::StringInterpolation { parts } = interp {
                        let new_parts = parts.into_iter().map(|p| match p {
                            StringPart::Literal(s) => StringPart::Literal(s),
                            StringPart::Expr(e) => StringPart::Expr(Expr::Format {
                                name: name.clone(),
                                expr: Box::new(e),
                            }),
                        }).collect();
                        Ok(Expr::StringInterpolation { parts: new_parts })
                    } else {
                        // Shouldn't happen but fallback
                        Ok(Expr::Format {
                            name,
                            expr: Box::new(interp),
                        })
                    }
                } else {
                    Ok(Expr::Format {
                        name,
                        expr: Box::new(Expr::Input),
                    })
                }
            }

            Token::Ident(ref name) if name == "__string_interp__" => {
                self.advance();
                self.parse_string_interpolation()
            }

            Token::Ident(name) => {
                self.advance();
                // Check for namespace:: prefix (e.g., foo::a)
                let full_name = if self.at(&Token::Colon) && matches!(self.tokens.get(self.pos + 1), Some(Token::Colon)) {
                    self.advance(); // first :
                    self.advance(); // second :
                    match self.advance() {
                        Token::Ident(member) => format!("{}::{}", name, member),
                        t => bail!("expected identifier after '::', got {:?}", t),
                    }
                } else {
                    name
                };
                self.parse_funcall_or_builtin(&full_name)
            }

            Token::Def => {
                // Local function definition
                self.parse_funcdef()?;
                self.parse_pipe()
            }

            _ => {
                bail!("unexpected token {:?}", tok);
            }
        }
    }

    fn parse_object_pair(&mut self) -> Result<(Expr, Expr)> {
        match self.current().clone() {
            Token::Ident(key) if !matches!(self.peek(), Token::LParen) => {
                self.advance();
                if self.eat(&Token::Colon) {
                    let val = self.parse_pipe_nocomma()?;
                    Ok((Expr::Literal(Literal::Str(key)), val))
                } else {
                    // Shorthand: {foo} = {foo: .foo}
                    Ok((
                        Expr::Literal(Literal::Str(key.clone())),
                        Expr::Index {
                            expr: Box::new(Expr::Input),
                            key: Box::new(Expr::Literal(Literal::Str(key))),
                        },
                    ))
                }
            }
            Token::Variable(name) => {
                let loc_line = self.current_line();
                self.advance();
                if self.eat(&Token::Colon) {
                    let val = self.parse_pipe_nocomma()?;
                    // $var: value — key is the variable's value converted to string
                    let key_expr = self.load_or_defer_var(&name);
                    Ok((key_expr, val))
                } else {
                    // Shorthand: {$x} = {"x": $x}
                    let val_expr = if name == "__loc__" {
                        Expr::Loc { file: "<top-level>".to_string(), line: loc_line as i64 }
                    } else if name == "ENV" {
                        self.resolve_env_ref()
                    } else {
                        self.load_or_defer_var(&name)
                    };
                    Ok((
                        Expr::Literal(Literal::Str(name)),
                        val_expr,
                    ))
                }
            }
            Token::Str(key) => {
                self.advance();
                if self.eat(&Token::Colon) {
                    let val = self.parse_pipe_nocomma()?;
                    Ok((Expr::Literal(Literal::Str(key)), val))
                } else {
                    // Shorthand: {"foo"} = {"foo": .foo}
                    Ok((
                        Expr::Literal(Literal::Str(key.clone())),
                        Expr::Index {
                            expr: Box::new(Expr::Input),
                            key: Box::new(Expr::Literal(Literal::Str(key))),
                        },
                    ))
                }
            }
            Token::LParen => {
                // Computed key: {(expr): value}
                self.advance();
                let key_expr = self.parse_pipe()?;
                self.expect(&Token::RParen)?;
                // A constant non-string key is a compile-time error in jq, so it
                // cannot be caught by `try`/`?` (#726). Fold the key the way jq's
                // compiler does and reject a non-string result here, before the
                // program runs. Runtime-computed keys (`{(.k):2}`) fold to None
                // and keep their runtime error.
                if let Some(v) = crate::runtime::const_fold(&key_expr) {
                    if !matches!(v, crate::value::Value::Str(_)) {
                        bail!(
                            "Cannot use {} as object key",
                            crate::runtime::errdesc_pub(&v)
                        );
                    }
                }
                self.expect(&Token::Colon)?;
                let val = self.parse_pipe_nocomma()?;
                Ok((key_expr, val))
            }
            Token::Format(ref name) => {
                let name = name.clone();
                self.advance();
                let key_expr = Expr::Format {
                    name,
                    expr: Box::new(Expr::Input),
                };
                if self.eat(&Token::Colon) {
                    let val = self.parse_pipe_nocomma()?;
                    Ok((key_expr, val))
                } else {
                    Ok((key_expr, Expr::Input))
                }
            }
            Token::Ident(ref name) if name == "__string_interp__" => {
                self.advance();
                let key_expr = self.parse_string_interpolation()?;
                if self.eat(&Token::Colon) {
                    let val = self.parse_pipe_nocomma()?;
                    Ok((key_expr, val))
                } else {
                    // Shorthand: {"foo\(x)"} = {("foo\(x)"): .["foo\(x)"]}
                    let val_expr = Expr::Index {
                        expr: Box::new(Expr::Input),
                        key: Box::new(key_expr.clone()),
                    };
                    Ok((key_expr, val_expr))
                }
            }
            // Keywords as object keys: {if:0, and:1, ...}
            ref tok if Self::keyword_as_string(tok).is_some() => {
                let key = Self::keyword_as_string(tok).unwrap().to_string();
                self.advance();
                if self.eat(&Token::Colon) {
                    let val = self.parse_pipe_nocomma()?;
                    Ok((Expr::Literal(Literal::Str(key)), val))
                } else {
                    // Shorthand: {as} = {as: .as}
                    Ok((
                        Expr::Literal(Literal::Str(key.clone())),
                        Expr::Index {
                            expr: Box::new(Expr::Input),
                            key: Box::new(Expr::Literal(Literal::Str(key))),
                        },
                    ))
                }
            }
            _ => bail!("expected object key, got {:?}", self.current()),
        }
    }

    /// Convert a keyword token to its string representation (for use as field names/object keys).
    fn keyword_as_string(tok: &Token) -> Option<&'static str> {
        match tok {
            Token::If => Some("if"),
            Token::Then => Some("then"),
            Token::Elif => Some("elif"),
            Token::Else => Some("else"),
            Token::End => Some("end"),
            Token::Try => Some("try"),
            Token::Catch => Some("catch"),
            Token::Reduce => Some("reduce"),
            Token::Foreach => Some("foreach"),
            Token::As => Some("as"),
            Token::Def => Some("def"),
            Token::And => Some("and"),
            Token::Or => Some("or"),
            Token::Not => Some("not"),
            Token::Label => Some("label"),
            Token::Break => Some("break"),
            Token::Import => Some("import"),
            Token::Include => Some("include"),
            Token::Module => Some("module"),
            Token::Null => Some("null"),
            Token::True => Some("true"),
            Token::False => Some("false"),
            Token::Empty => Some("empty"),
            Token::Error => Some("error"),
            _ => None,
        }
    }

    fn parse_if_then_else(&mut self) -> Result<Expr> {
        let cond = self.parse_pipe()?;
        self.expect(&Token::Then)?;
        let then_branch = self.parse_pipe()?;

        if self.eat(&Token::Elif) {
            let else_branch = self.parse_if_then_else()?;
            Ok(Expr::IfThenElse {
                cond: Box::new(cond),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            })
        } else if self.eat(&Token::Else) {
            let else_branch = self.parse_pipe()?;
            self.expect(&Token::End)?;
            Ok(Expr::IfThenElse {
                cond: Box::new(cond),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            })
        } else {
            self.expect(&Token::End)?;
            Ok(Expr::IfThenElse {
                cond: Box::new(cond),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(Expr::Input), // no else = identity
            })
        }
    }

    fn parse_reduce(&mut self) -> Result<Expr> {
        // reduce SOURCE as PATTERN ('?//' PATTERN)* (INIT; UPDATE)
        let source = self.parse_or()?;
        self.expect(&Token::As)?;

        // Parse the pattern chain syntactically (no var alloc yet). A bare
        // `$x` parses as Pattern::Var, so the simple-variable case is just a
        // single-element chain. `?//` alternatives extend it. See #712.
        let first_pattern = self.parse_pattern()?;
        let mut alt_patterns: Vec<Pattern> = vec![first_pattern];
        while self.eat(&Token::AltDestructure) {
            alt_patterns.push(self.parse_pattern()?);
        }

        // Parse INIT before binding any pattern var: jq scopes the binding to
        // UPDATE only, so a stray reference inside INIT is a compile error
        // (#202).
        self.expect(&Token::LParen)?;
        let init = self.parse_pipe()?;
        self.expect(&Token::Semicolon)?;
        // Snapshot scope so a same-name `$x` shadow doesn't leak after this
        // reduce expression closes (#499).
        let saved_vars = self.scope.vars.len();

        if alt_patterns.len() == 1 {
            if let Pattern::Var(var_name) = &alt_patterns[0] {
                // Simple single-variable binding (no destructure tmp).
                let var_idx = self.scope.alloc_var(var_name);
                let acc_idx = self.scope.alloc_var("__acc__");
                let update = self.parse_pipe()?;
                self.expect(&Token::RParen)?;
                self.scope.vars.truncate(saved_vars);
                return Ok(Expr::Reduce {
                    source: Box::new(source),
                    init: Box::new(init),
                    var_index: var_idx,
                    acc_index: acc_idx,
                    update: Box::new(update),
                });
            }
            // Single destructuring pattern.
            let pattern = alt_patterns.into_iter().next().unwrap();
            let allocs = self.alloc_pattern_vars(&pattern);
            let tmp_var = self.scope.alloc_var("__reduce_item__");
            let acc_idx = self.scope.alloc_var("__acc__");
            let update_raw = self.parse_pipe()?;
            self.expect(&Token::RParen)?;
            self.scope.vars.truncate(saved_vars);
            let update = self.build_binding(
                Expr::LoadVar { var_index: tmp_var },
                pattern,
                allocs,
                update_raw,
            )?;
            return Ok(Expr::Reduce {
                source: Box::new(source),
                init: Box::new(init),
                var_index: tmp_var,
                acc_index: acc_idx,
                update: Box::new(update),
            });
        }

        // `?//` alternative destructuring: allocate shared vars once, then
        // wrap UPDATE in the try-catch chain that binds the element value
        // (held in the reduce's own slot) through each alternative.
        let var_map = self.alloc_alt_destructure_vars(&alt_patterns);
        let tmp_var = self.scope.alloc_var("__reduce_item__");
        let acc_idx = self.scope.alloc_var("__acc__");
        let update_raw = self.parse_pipe()?;
        self.expect(&Token::RParen)?;
        let update = self.build_alt_destructure(
            &Expr::LoadVar { var_index: tmp_var },
            &alt_patterns,
            &var_map,
            &update_raw,
        )?;
        self.scope.vars.truncate(saved_vars);

        Ok(Expr::Reduce {
            source: Box::new(source),
            init: Box::new(init),
            var_index: tmp_var,
            acc_index: acc_idx,
            update: Box::new(update),
        })
    }

    fn parse_foreach(&mut self) -> Result<Expr> {
        // foreach SOURCE as PATTERN ('?//' PATTERN)* (INIT; UPDATE [; EXTRACT])
        let source = self.parse_or()?;
        self.expect(&Token::As)?;

        // Parse the pattern chain syntactically (a bare `$x` is Pattern::Var).
        // See #712.
        let first_pattern = self.parse_pattern()?;
        let mut alt_patterns: Vec<Pattern> = vec![first_pattern];
        while self.eat(&Token::AltDestructure) {
            alt_patterns.push(self.parse_pattern()?);
        }

        // Parse INIT before binding any pattern var (#202).
        self.expect(&Token::LParen)?;
        let init = self.parse_pipe()?;
        self.expect(&Token::Semicolon)?;
        // Snapshot scope so a same-name `$x` shadow doesn't leak after this
        // foreach expression closes (#499).
        let saved_vars = self.scope.vars.len();

        if alt_patterns.len() == 1 {
            if let Pattern::Var(var_name) = &alt_patterns[0] {
                // Simple single-variable binding (no destructure tmp).
                let var_idx = self.scope.alloc_var(var_name);
                let acc_idx = self.scope.alloc_var("__acc__");
                let update = self.parse_pipe()?;
                let extract = if self.eat(&Token::Semicolon) {
                    Some(Box::new(self.parse_pipe()?))
                } else {
                    None
                };
                self.expect(&Token::RParen)?;
                self.scope.vars.truncate(saved_vars);
                return Ok(Expr::Foreach {
                    source: Box::new(source),
                    init: Box::new(init),
                    var_index: var_idx,
                    acc_index: acc_idx,
                    update: Box::new(update),
                    extract,
                });
            }
            // Single destructuring pattern.
            let pattern = alt_patterns.into_iter().next().unwrap();
            let allocs = self.alloc_pattern_vars(&pattern);
            let tmp_var = self.scope.alloc_var("__foreach_item__");
            let acc_idx = self.scope.alloc_var("__acc__");
            let update_raw = self.parse_pipe()?;
            let extract_raw = if self.eat(&Token::Semicolon) {
                Some(self.parse_pipe()?)
            } else {
                None
            };
            self.expect(&Token::RParen)?;
            self.scope.vars.truncate(saved_vars);

            // Wrap update in pattern binding
            let update = self.build_binding(
                Expr::LoadVar { var_index: tmp_var },
                pattern.clone(),
                allocs.clone(),
                update_raw,
            )?;
            let extract = if let Some(extract_expr) = extract_raw {
                Some(Box::new(self.build_binding(
                    Expr::LoadVar { var_index: tmp_var },
                    pattern,
                    allocs,
                    extract_expr,
                )?))
            } else {
                None
            };

            return Ok(Expr::Foreach {
                source: Box::new(source),
                init: Box::new(init),
                var_index: tmp_var,
                acc_index: acc_idx,
                update: Box::new(update),
                extract,
            });
        }

        // `?//` alternative destructuring: allocate shared vars once, then
        // wrap UPDATE and EXTRACT in the try-catch alternative chain binding
        // the element value (held in the foreach's own slot).
        let var_map = self.alloc_alt_destructure_vars(&alt_patterns);
        let tmp_var = self.scope.alloc_var("__foreach_item__");
        let acc_idx = self.scope.alloc_var("__acc__");
        let update_raw = self.parse_pipe()?;
        let extract_raw = if self.eat(&Token::Semicolon) {
            Some(self.parse_pipe()?)
        } else {
            None
        };
        self.expect(&Token::RParen)?;

        let item_ref = Expr::LoadVar { var_index: tmp_var };
        let update = self.build_alt_destructure(&item_ref, &alt_patterns, &var_map, &update_raw)?;
        let extract = if let Some(extract_expr) = extract_raw {
            Some(Box::new(self.build_alt_destructure(&item_ref, &alt_patterns, &var_map, &extract_expr)?))
        } else {
            None
        };
        self.scope.vars.truncate(saved_vars);

        Ok(Expr::Foreach {
            source: Box::new(source),
            init: Box::new(init),
            var_index: tmp_var,
            acc_index: acc_idx,
            update: Box::new(update),
            extract,
        })
    }

    fn parse_funcall_or_builtin(&mut self, name: &str) -> Result<Expr> {
        // Check for well-known builtins and functions
        // Some take arguments in parens with ; as separator

        // Check for builtins with special parsing
        match name {
            // 0-arg builtins (no parens)
            "length" | "utf8bytelength" | "type" | "infinite" | "nan"
            | "isinfinite" | "isnan" | "isnormal" | "isfinite"
            | "tostring" | "tonumber" | "tojson" | "fromjson"
            | "explode" | "implode"
            | "ascii_downcase" | "ascii_upcase" | "ltrim" | "rtrim" | "trim"
            | "floor" | "ceil" | "round" | "fabs" | "sqrt"
            | "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
            | "sinh" | "cosh" | "tanh" | "asinh" | "acosh" | "atanh"
            | "exp" | "exp2" | "exp10" | "log" | "log2" | "log10"
            | "expm1" | "log1p" | "erf" | "erfc"
            | "cbrt" | "significand" | "exponent" | "logb"
            | "nearbyint" | "trunc" | "rint" | "j0" | "j1" | "y0" | "y1"
            | "gamma" | "tgamma" | "lgamma" | "lgamma_r" | "frexp"
            | "have_literal_numbers"
            | "keys" | "keys_unsorted" | "values" | "sort" | "reverse"
            | "unique" | "flatten" | "min" | "max" | "add" | "any" | "all"
            | "transpose" | "to_entries" | "from_entries"
            | "gmtime" | "localtime" | "mktime" | "now" | "abs"
            | "not" | "env" | "builtins" | "input" | "inputs"
            | "debug" | "stderr" | "modulemeta" | "path"
            | "with_entries" | "recurse"
            | "has" | "in" | "contains" | "inside"
            | "getpath" | "setpath" | "delpaths"
            | "to_number" | "to_string" | "type_error"
            | "objects" | "arrays" | "strings" | "numbers" | "booleans" | "nulls"
            | "iterables" | "scalars" | "normals" | "finites" | "infinite_values"
            | "nan_values" | "isempty" | "have_decnum"
            | "halt" | "halt_error" | "ascii_downcase_" | "ascii_upcase_"
            | "indices" | "index" | "rindex" | "paths" | "getpath_" | "map_values"
            | "first" | "last" | "nth" | "range" | "limit" | "until" | "while" | "repeat"
            | "select" | "map"
            | "toboolean" | "walk" | "pick" | "bsearch" | "skip" | "del"
            | "IN" | "INDEX" | "JOIN" | "strflocaltime"
            | "fromcsv" | "fromtsv" | "fromcsvh" | "fromtsvh"
            | "fromdateiso8601" | "todateiso8601" | "fromisodate" | "toisodate"
            | "todate" | "fromdate" | "date"
            | "input_line_number" | "input_filename"
            | "get_jq_origin" | "get_prog_origin" | "get_search_list"
            | "tostream"
            | "combinations" | "modf"
            if !matches!(self.current(), Token::LParen) => {
                self.compile_builtin_noargs(name)
            }

            _ => {
                if self.eat(&Token::LParen) {
                    // Function call with arguments
                    let mut args = Vec::new();
                    if !self.at(&Token::RParen) {
                        loop {
                            args.push(self.parse_pipe()?);
                            if !self.eat(&Token::Semicolon) { break; }
                        }
                    }
                    self.expect(&Token::RParen)?;
                    self.compile_funcall(name, args)
                } else {
                    // No-arg function call or builtin
                    self.compile_builtin_noargs(name)
                }
            }
        }
    }

    fn compile_builtin_noargs(&mut self, name: &str) -> Result<Expr> {
        // A bare 0-arg name can resolve to a user `def` or a filter parameter
        // (including the implicit `x` filter introduced by a `$x` value param).
        // Both shadow same-named builtins, and between the two the lexically
        // innermost binding wins: a parameter shadows an enclosing `def`, while
        // a `def` nested inside the parameter's body shadows the parameter (#766).
        let func = self.scope.lookup_func(name, 0);
        let fparam = self.scope.lookup_var(&format!("\x00fparam:{}", name));
        match (func, fparam) {
            (Some(func_id), Some(var_idx)) => {
                if self.scope.func_shadows_param(func_id, var_idx) {
                    return Ok(self.scope.make_funccall(func_id, vec![]));
                }
                return Ok(Expr::LoadVar { var_index: var_idx });
            }
            (Some(func_id), None) => return Ok(self.scope.make_funccall(func_id, vec![])),
            (None, Some(var_idx)) => return Ok(Expr::LoadVar { var_index: var_idx }),
            (None, None) => {}
        }
        match name {
            "not" => Ok(Expr::Not),
            "empty" => Ok(Expr::Empty),
            "env" => Ok(Expr::Env),
            "builtins" => Ok(Expr::Builtins),
            "input" => Ok(Expr::ReadInput),
            "inputs" => Ok(Expr::ReadInputs),
            "debug" => Ok(Expr::Debug { expr: Box::new(Expr::Input) }),
            "stderr" => Ok(Expr::Stderr { expr: Box::new(Expr::Input) }),
            "modulemeta" => Ok(Expr::ModuleMeta),
            "infinite" => Ok(Expr::Literal(Literal::Num(f64::INFINITY, None))),
            "nan" => Ok(Expr::Literal(Literal::Num(f64::NAN, None))),
            "null" => Ok(Expr::Literal(Literal::Null)),
            "true" => Ok(Expr::Literal(Literal::True)),
            "false" => Ok(Expr::Literal(Literal::False)),
            "path" => Ok(Expr::PathExpr { expr: Box::new(Expr::Input) }),
            "first" => Ok(Expr::Index {
                expr: Box::new(Expr::Input),
                key: Box::new(Expr::Literal(Literal::Num(0.0, None))),
            }),
            "last" => Ok(Expr::Index {
                expr: Box::new(Expr::Input),
                key: Box::new(Expr::Literal(Literal::Num(-1.0, None))),
            }),
            "paths" => {
                // paths = [path(..[])] but without the empty root path.
                // Use the `EachOpt(Input)` sentinel so the `Recurse` shape
                // matches the descent semantics, not `recurse(.)`. See #497.
                Ok(Expr::Pipe {
                    left: Box::new(Expr::PathExpr {
                        expr: Box::new(Expr::Recurse {
                            input_expr: Box::new(Expr::EachOpt { input_expr: Box::new(Expr::Input) }),
                        }),
                    }),
                    right: Box::new(Expr::IfThenElse {
                        cond: Box::new(Expr::BinOp {
                            op: BinOp::Gt,
                            lhs: Box::new(Expr::UnaryOp { op: UnaryOp::Length, operand: Box::new(Expr::Input) }),
                            rhs: Box::new(Expr::Literal(Literal::Num(0.0, None))),
                        }),
                        then_branch: Box::new(Expr::Input),
                        else_branch: Box::new(Expr::Empty),
                    }),
                })
            }
            "recurse" => {
                // jq: `def recurse: recurse(.[]?);`
                // `EachOpt(Input)` represents `.[]?` and lets eval/JIT take
                // the descent fast path; `recurse(.)` (which is infinite)
                // takes the slow custom-step path instead. See #497.
                // Note: `recurse_down/0` (a deprecated alias removed from jq
                // before 1.8) is intentionally NOT handled here — jq 1.8.1
                // compile-errors on it, and it is not a documented jqx
                // extension, so it falls through to the undefined-function
                // path ("recurse_down/0 is not defined") (#821).
                Ok(Expr::Recurse {
                    input_expr: Box::new(Expr::EachOpt { input_expr: Box::new(Expr::Input) }),
                })
            }
            "gamma" | "tgamma" | "lgamma" | "lgamma_r" | "frexp"
            | "expm1" | "log1p" | "erf" | "erfc" | "y0" | "y1"
            | "have_literal_numbers" => {
                // libm-backed math builtins not represented as UnaryOp.
                // Run through CallBuiltin so the runtime dispatches them.
                // `have_literal_numbers` is a feature-flag builtin; jq 1.8.1
                // returns true (decnum is enabled in mainline). See #473.
                Ok(Expr::CallBuiltin { name: name.to_string(), args: vec![] })
            }
            "values" => {
                // values = select(. != null) - type filter
                Ok(Expr::IfThenElse {
                    cond: Box::new(Expr::BinOp {
                        op: BinOp::Ne,
                        lhs: Box::new(Expr::UnaryOp { op: UnaryOp::Type, operand: Box::new(Expr::Input) }),
                        rhs: Box::new(Expr::Literal(Literal::Str("null".to_string()))),
                    }),
                    then_branch: Box::new(Expr::Input),
                    else_branch: Box::new(Expr::Empty),
                })
            }
            "objects" => Ok(Expr::Pipe {
                left: Box::new(Expr::Input),
                right: Box::new(Expr::IfThenElse {
                    cond: Box::new(Expr::BinOp {
                        op: BinOp::Eq,
                        lhs: Box::new(Expr::UnaryOp { op: UnaryOp::Type, operand: Box::new(Expr::Input) }),
                        rhs: Box::new(Expr::Literal(Literal::Str("object".to_string()))),
                    }),
                    then_branch: Box::new(Expr::Input),
                    else_branch: Box::new(Expr::Empty),
                }),
            }),
            "arrays" => Ok(make_type_select("array")),
            "strings" => Ok(make_type_select("string")),
            "numbers" => Ok(make_type_select("number")),
            "booleans" => Ok(make_type_select("boolean")),
            "nulls" => Ok(make_type_select("null")),
            "iterables" => {
                // select(type == "array" or type == "object")
                Ok(Expr::IfThenElse {
                    cond: Box::new(Expr::BinOp {
                        op: BinOp::Or,
                        lhs: Box::new(Expr::BinOp {
                            op: BinOp::Eq,
                            lhs: Box::new(Expr::UnaryOp { op: UnaryOp::Type, operand: Box::new(Expr::Input) }),
                            rhs: Box::new(Expr::Literal(Literal::Str("array".to_string()))),
                        }),
                        rhs: Box::new(Expr::BinOp {
                            op: BinOp::Eq,
                            lhs: Box::new(Expr::UnaryOp { op: UnaryOp::Type, operand: Box::new(Expr::Input) }),
                            rhs: Box::new(Expr::Literal(Literal::Str("object".to_string()))),
                        }),
                    }),
                    then_branch: Box::new(Expr::Input),
                    else_branch: Box::new(Expr::Empty),
                })
            }
            "scalars" => {
                Ok(Expr::IfThenElse {
                    cond: Box::new(Expr::BinOp {
                        op: BinOp::Or,
                        lhs: Box::new(Expr::BinOp {
                            op: BinOp::Eq,
                            lhs: Box::new(Expr::UnaryOp { op: UnaryOp::Type, operand: Box::new(Expr::Input) }),
                            rhs: Box::new(Expr::Literal(Literal::Str("array".to_string()))),
                        }),
                        rhs: Box::new(Expr::BinOp {
                            op: BinOp::Eq,
                            lhs: Box::new(Expr::UnaryOp { op: UnaryOp::Type, operand: Box::new(Expr::Input) }),
                            rhs: Box::new(Expr::Literal(Literal::Str("object".to_string()))),
                        }),
                    }),
                    then_branch: Box::new(Expr::Empty),
                    else_branch: Box::new(Expr::Input),
                })
            }
            "finites" => {
                // finites = select(type == "number" and (isinfinite | not))
                // Equivalent pipe form: numbers | select(isinfinite | not)
                Ok(Expr::Pipe {
                    left: Box::new(make_type_select("number")),
                    right: Box::new(Expr::IfThenElse {
                        cond: Box::new(Expr::UnaryOp {
                            op: UnaryOp::Not,
                            operand: Box::new(Expr::UnaryOp {
                                op: UnaryOp::IsInfinite,
                                operand: Box::new(Expr::Input),
                            }),
                        }),
                        then_branch: Box::new(Expr::Input),
                        else_branch: Box::new(Expr::Empty),
                    }),
                })
            }
            "normals" => {
                // normals = select(type == "number" and isnormal)
                // Equivalent pipe form: numbers | select(isnormal)
                Ok(Expr::Pipe {
                    left: Box::new(make_type_select("number")),
                    right: Box::new(Expr::IfThenElse {
                        cond: Box::new(Expr::UnaryOp {
                            op: UnaryOp::IsNormal,
                            operand: Box::new(Expr::Input),
                        }),
                        then_branch: Box::new(Expr::Input),
                        else_branch: Box::new(Expr::Empty),
                    }),
                })
            }
            "isempty" => {
                // isempty = first(empty) // true; first = limit(1; .)
                // Actually: def isempty(f): first((f | false), true);
                // For no-arg: isempty just returns... it's actually isempty(f) normally
                // But as 0-arg it would be identity
                Ok(Expr::Input)
            }
            "have_decnum" | "have_decnum_" => {
                // We don't have decimal number support
                Ok(Expr::Literal(Literal::False))
            }
            "toboolean" => {
                Ok(Expr::CallBuiltin { name: "toboolean".to_string(), args: vec![] })
            }
            "halt" => {
                Ok(Expr::CallBuiltin { name: "halt".to_string(), args: vec![] })
            }
            "halt_error" => {
                Ok(Expr::CallBuiltin { name: "halt_error".to_string(), args: vec![] })
            }
            "input_line_number" => {
                // Resolved at eval/JIT time from the current input's line counter
                // (set by the CLI before each input is processed).
                Ok(Expr::CallBuiltin { name: "input_line_number".to_string(), args: vec![] })
            }
            "fromcsv" | "fromtsv" | "fromcsvh" | "fromtsvh" => {
                Ok(Expr::CallBuiltin { name: name.to_string(), args: vec![] })
            }
            "fromdateiso8601" | "todateiso8601" | "fromisodate" | "toisodate" => {
                Ok(Expr::CallBuiltin { name: name.to_string(), args: vec![] })
            }
            "todate" | "fromdate" | "date" => {
                Ok(Expr::CallBuiltin { name: name.to_string(), args: vec![] })
            }
            "combinations" | "modf" => {
                Ok(Expr::CallBuiltin { name: name.to_string(), args: vec![] })
            }
            "get_jq_origin" | "get_prog_origin" | "get_search_list" => {
                Ok(Expr::CallBuiltin { name: name.to_string(), args: vec![] })
            }
            "tostream" => {
                Ok(Expr::CallBuiltin { name: "tostream".to_string(), args: vec![] })
            }
            "input_filename" => {
                // jq emits "<stdin>" for the default stdin input source
                // (and the file path when one is specified). jq-jit reads
                // exclusively from stdin and does not plumb file paths
                // through the pipeline, so always return "<stdin>" — this
                // matches jq for the common stdin pipeline; in `-n` mode
                // jq would return null, which is a known divergence.
                Ok(Expr::Literal(Literal::Str("<stdin>".to_string())))
            }
            _ => {
                // User defs and filter parameters were already resolved at the
                // top of this function (#766), so anything reaching here is a
                // 0-arg builtin handled via runtime — or an undefined name.
                // jq resolves names lazily, so an undefined 0-arg name is only
                // an error when reachable: defer it instead of erroring eagerly
                // (#807). The deferred placeholder errors at runtime if it is
                // reached, and `check_unbound_reachability` turns it into a
                // compile error if its enclosing def is actually called.
                match name_to_unary_op(name) {
                    Ok(op) => Ok(Expr::UnaryOp {
                        op,
                        operand: Box::new(Expr::Input),
                    }),
                    Err(_) => Ok(self.defer_unknown_func(name, 0)),
                }
            }
        }
    }

    /// Desugar `INDEX(stream; f)` to jq's real definition
    /// `reduce stream as $x ({}; .[$x|f|tostring] = $x)`. The assignment LHS is
    /// a path expression whose key (`$x|f|tostring`) may be a generator; each
    /// produced key is then set to `$x` (last-write-wins across the fan-out).
    /// The earlier `. + {($x|f|tostring): $x}` object-merge collapsed a
    /// multi-output index expression to only its last key. See #883.
    fn index_reduce(stream: Expr, f: Expr, x_var: u16, acc_var: u16) -> Expr {
        let key = Expr::Pipe {
            left: Box::new(Expr::LoadVar { var_index: x_var }),
            right: Box::new(Expr::Pipe {
                left: Box::new(f),
                right: Box::new(Expr::UnaryOp { op: UnaryOp::ToString, operand: Box::new(Expr::Input) }),
            }),
        };
        Expr::Reduce {
            source: Box::new(stream),
            init: Box::new(Expr::ObjectConstruct { pairs: vec![] }),
            var_index: x_var,
            acc_index: acc_var,
            update: Box::new(Expr::Assign {
                path_expr: Box::new(Expr::Index {
                    expr: Box::new(Expr::Input),
                    key: Box::new(key),
                }),
                value_expr: Box::new(Expr::LoadVar { var_index: x_var }),
            }),
        }
    }

    fn compile_funcall(&mut self, name: &str, args: Vec<Expr>) -> Result<Expr> {
        // User-defined functions shadow builtins (matches jq semantics).
        if let Some(func_id) = self.scope.lookup_func(name, args.len()) {
            // Append any lambda-lifted capture params after the user args (#714).
            return Ok(self.scope.make_funccall(func_id, args));
        }
        match (name, args.len()) {
            // Standard library functions
            ("select", 1) => {
                let cond = args.into_iter().next().unwrap();
                Ok(Expr::IfThenElse {
                    cond: Box::new(cond),
                    then_branch: Box::new(Expr::Input),
                    else_branch: Box::new(Expr::Empty),
                })
            }
            ("map", 1) => {
                let f = args.into_iter().next().unwrap();
                Ok(Expr::Collect {
                    generator: Box::new(Expr::Pipe {
                        left: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                        right: Box::new(f),
                    }),
                })
            }
            ("map_values", 1) => {
                let f = args.into_iter().next().unwrap();
                Ok(Expr::Update {
                    path_expr: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                    update_expr: Box::new(f),
                })
            }
            ("with_entries", 1) => {
                let f = args.into_iter().next().unwrap();
                // to_entries | map(f) | from_entries
                Ok(Expr::Pipe {
                    left: Box::new(Expr::UnaryOp {
                        op: UnaryOp::ToEntries,
                        operand: Box::new(Expr::Input),
                    }),
                    right: Box::new(Expr::Pipe {
                        left: Box::new(Expr::Collect {
                            generator: Box::new(Expr::Pipe {
                                left: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                                right: Box::new(f),
                            }),
                        }),
                        right: Box::new(Expr::UnaryOp {
                            op: UnaryOp::FromEntries,
                            operand: Box::new(Expr::Input),
                        }),
                    }),
                })
            }
            ("sort_by", 1) => Ok(Expr::ClosureOp { op: ClosureOpKind::SortBy, input_expr: Box::new(Expr::Input), key_expr: Box::new(args.into_iter().next().unwrap()) }),
            ("group_by", 1) => Ok(Expr::ClosureOp { op: ClosureOpKind::GroupBy, input_expr: Box::new(Expr::Input), key_expr: Box::new(args.into_iter().next().unwrap()) }),
            ("unique_by", 1) => Ok(Expr::ClosureOp { op: ClosureOpKind::UniqueBy, input_expr: Box::new(Expr::Input), key_expr: Box::new(args.into_iter().next().unwrap()) }),
            ("min_by", 1) => Ok(Expr::ClosureOp { op: ClosureOpKind::MinBy, input_expr: Box::new(Expr::Input), key_expr: Box::new(args.into_iter().next().unwrap()) }),
            ("max_by", 1) => Ok(Expr::ClosureOp { op: ClosureOpKind::MaxBy, input_expr: Box::new(Expr::Input), key_expr: Box::new(args.into_iter().next().unwrap()) }),
            ("any", 1) => {
                let f = args.into_iter().next().unwrap();
                Ok(Expr::AnyShort {
                    generator: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                    predicate: Box::new(f),
                })
            }
            ("all", 1) => {
                let f = args.into_iter().next().unwrap();
                Ok(Expr::AllShort {
                    generator: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                    predicate: Box::new(f),
                })
            }
            ("any", 2) => {
                let mut args = args.into_iter();
                let generator = args.next().unwrap();
                let cond = args.next().unwrap();
                Ok(Expr::AnyShort {
                    generator: Box::new(generator),
                    predicate: Box::new(cond),
                })
            }
            ("all", 2) => {
                let mut args = args.into_iter();
                let generator = args.next().unwrap();
                let cond = args.next().unwrap();
                Ok(Expr::AllShort {
                    generator: Box::new(generator),
                    predicate: Box::new(cond),
                })
            }
            ("range", 1) => {
                let to = args.into_iter().next().unwrap();
                Ok(Expr::Range {
                    from: Box::new(Expr::Literal(Literal::Num(0.0, None))),
                    to: Box::new(to),
                    step: None,
                })
            }
            ("range", 2) => {
                let mut args = args.into_iter();
                let from = args.next().unwrap();
                let to = args.next().unwrap();
                Ok(Expr::Range { from: Box::new(from), to: Box::new(to), step: None })
            }
            ("range", 3) => {
                let mut args = args.into_iter();
                let from = args.next().unwrap();
                let to = args.next().unwrap();
                let step = args.next().unwrap();
                Ok(Expr::Range { from: Box::new(from), to: Box::new(to), step: Some(Box::new(step)) })
            }
            ("limit", 2) => {
                let mut args = args.into_iter();
                let count = args.next().unwrap();
                let generator = args.next().unwrap();
                Ok(Expr::Limit { count: Box::new(count), generator: Box::new(generator) })
            }
            ("first", 0) => {
                // first = .[0]
                Ok(Expr::Index {
                    expr: Box::new(Expr::Input),
                    key: Box::new(Expr::Literal(Literal::Num(0.0, None))),
                })
            }
            ("first", 1) => {
                let generator = args.into_iter().next().unwrap();
                Ok(Expr::Limit {
                    count: Box::new(Expr::Literal(Literal::Num(1.0, None))),
                    generator: Box::new(generator),
                })
            }
            ("last", 0) => {
                // last = .[-1]
                Ok(Expr::Index {
                    expr: Box::new(Expr::Input),
                    key: Box::new(Expr::Literal(Literal::Num(-1.0, None))),
                })
            }
            ("last", 1) => {
                // last(g) = reduce g as $x ([]; [$x]) | if length > 0 then .[0] else empty end
                let generator = args.into_iter().next().unwrap();

                // Optimize last(range(n)) → if n > 0 then n - 1 else empty end
                if let Expr::Range { ref from, ref to, ref step } = generator {
                    let is_from_zero = matches!(from.as_ref(),
                        Expr::Literal(Literal::Num(n, _)) if *n == 0.0);
                    if is_from_zero && step.is_none() {
                        return Ok(Expr::IfThenElse {
                            cond: Box::new(Expr::BinOp {
                                op: BinOp::Gt,
                                lhs: to.clone(),
                                rhs: Box::new(Expr::Literal(Literal::Num(0.0, None))),
                            }),
                            then_branch: Box::new(Expr::BinOp {
                                op: BinOp::Sub,
                                lhs: to.clone(),
                                rhs: Box::new(Expr::Literal(Literal::Num(1.0, None))),
                            }),
                            else_branch: Box::new(Expr::Empty),
                        });
                    }
                }
                let var_idx = self.scope.alloc_var("__last__");
                let acc_idx = self.scope.alloc_var("__last_acc__");
                Ok(Expr::Pipe {
                    left: Box::new(Expr::Reduce {
                        source: Box::new(generator),
                        init: Box::new(Expr::Collect { generator: Box::new(Expr::Empty) }), // []
                        var_index: var_idx,
                        acc_index: acc_idx,
                        update: Box::new(Expr::Collect {
                            generator: Box::new(Expr::LoadVar { var_index: var_idx }),
                        }), // [$x]
                    }),
                    right: Box::new(Expr::IfThenElse {
                        cond: Box::new(Expr::BinOp {
                            op: BinOp::Gt,
                            lhs: Box::new(Expr::UnaryOp { op: UnaryOp::Length, operand: Box::new(Expr::Input) }),
                            rhs: Box::new(Expr::Literal(Literal::Num(0.0, None))),
                        }),
                        then_branch: Box::new(Expr::Index {
                            expr: Box::new(Expr::Input),
                            key: Box::new(Expr::Literal(Literal::Num(0.0, None))),
                        }),
                        else_branch: Box::new(Expr::Empty),
                    }),
                })
            }
            ("while", 2) => {
                let mut args = args.into_iter();
                let cond = args.next().unwrap();
                let update = args.next().unwrap();
                Ok(Expr::While { cond: Box::new(cond), update: Box::new(update) })
            }
            ("until", 2) => {
                let mut args = args.into_iter();
                let cond = args.next().unwrap();
                let update = args.next().unwrap();
                Ok(Expr::Until { cond: Box::new(cond), update: Box::new(update) })
            }
            ("repeat", 1) => {
                let update = args.into_iter().next().unwrap();
                Ok(Expr::Repeat { update: Box::new(update) })
            }
            ("isempty", 1) => {
                let f = args.into_iter().next().unwrap();
                // isempty(f) = first((f | false), true)
                Ok(Expr::Limit {
                    count: Box::new(Expr::Literal(Literal::Num(1.0, None))),
                    generator: Box::new(Expr::Comma {
                        left: Box::new(Expr::Pipe {
                            left: Box::new(f),
                            right: Box::new(Expr::Literal(Literal::False)),
                        }),
                        right: Box::new(Expr::Literal(Literal::True)),
                    }),
                })
            }
            ("recurse", 1) => {
                let f = args.into_iter().next().unwrap();
                // recurse(f) = def r: ., (f | r); r
                // This is a recursive pattern - use Recurse node
                Ok(Expr::Recurse { input_expr: Box::new(f) })
            }
            ("mutate", 1) => {
                let body = args.into_iter().next().unwrap();
                // jqx extension. Body must be a top-level path-update operator
                // (`=`, `|=`, `+=`, `-=`, `*=`, `/=`, `%=`, `//=`); composite
                // forms must distribute the marker inward (wrap each leaf
                // separately). `+=` and friends desugar to `LetBinding(body:
                // Update)` upstream, so peel intermediate `LetBinding`s before
                // checking the leaf. See issue #666.
                wrap_mutate(body)
            }
            ("memoize", 1) => {
                let body = args.into_iter().next().unwrap();
                let slot_id = self.scope.alloc_memo_slot();
                Ok(Expr::Memoize { slot_id, key: None, body: Box::new(body) })
            }
            ("memoize", 2) => {
                let mut it = args.into_iter();
                let body = it.next().unwrap();
                let key = it.next().unwrap();
                let slot_id = self.scope.alloc_memo_slot();
                Ok(Expr::Memoize { slot_id, key: Some(Box::new(key)), body: Box::new(body) })
            }
            ("recurse", 2) => {
                let mut args = args.into_iter();
                let f = args.next().unwrap();
                let cond = args.next().unwrap();
                // jq: def recurse(f; cond): def r: ., (f | select(cond) | r); r;
                // `cond` filters the values produced by `f`, so it must run
                // AFTER `f`, not before. Swapping the order was issue #49.
                Ok(Expr::Recurse {
                    input_expr: Box::new(Expr::Pipe {
                        left: Box::new(f),
                        right: Box::new(Expr::IfThenElse {
                            cond: Box::new(cond),
                            then_branch: Box::new(Expr::Input),
                            else_branch: Box::new(Expr::Empty),
                        }),
                    }),
                })
            }
            ("path", 1) => {
                let f = args.into_iter().next().unwrap();
                Ok(Expr::PathExpr { expr: Box::new(f) })
            }
            ("paths", 1) => {
                let f = args.into_iter().next().unwrap();
                // paths(f) = paths | select(getpath(.) | f) in jq's builtin.jq.
                // We approximate as path(recurse | select(f)), but must also drop the
                // root (empty) path that recurse yields — jq's `paths` filters `length > 0`.
                Ok(Expr::Pipe {
                    left: Box::new(Expr::PathExpr {
                        expr: Box::new(Expr::Pipe {
                            // Use the EachOpt(Input) sentinel for descent
                            // semantics (#497).
                            left: Box::new(Expr::Recurse {
                                input_expr: Box::new(Expr::EachOpt { input_expr: Box::new(Expr::Input) }),
                            }),
                            right: Box::new(Expr::IfThenElse {
                                cond: Box::new(f),
                                then_branch: Box::new(Expr::Input),
                                else_branch: Box::new(Expr::Empty),
                            }),
                        }),
                    }),
                    right: Box::new(Expr::IfThenElse {
                        cond: Box::new(Expr::BinOp {
                            op: BinOp::Gt,
                            lhs: Box::new(Expr::UnaryOp { op: UnaryOp::Length, operand: Box::new(Expr::Input) }),
                            rhs: Box::new(Expr::Literal(Literal::Num(0.0, None))),
                        }),
                        then_branch: Box::new(Expr::Input),
                        else_branch: Box::new(Expr::Empty),
                    }),
                })
            }
            ("getpath", 1) => {
                let path = args.into_iter().next().unwrap();
                // Don't rewrite `getpath([...keys])` → `.key.key.key`: the
                // chained-Index path inherits a pre-existing divergence where
                // `.a` on a non-indexable (number/string/boolean) yields null
                // instead of erroring, which hides the type error `getpath`
                // must raise (issue #77). `rt_getpath` surfaces the error.
                Ok(Expr::GetPath { path: Box::new(path) })
            }
            ("setpath", 2) => {
                let mut args = args.into_iter();
                let path = args.next().unwrap();
                let value = args.next().unwrap();
                Ok(Expr::SetPath { path: Box::new(path), value: Box::new(value) })
            }
            ("delpaths", 1) => {
                let paths = args.into_iter().next().unwrap();
                Ok(Expr::DelPaths { paths: Box::new(paths) })
            }
            ("has", 1) => {
                let key = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "has".to_string(), args: vec![key] })
            }
            ("in", 1) => {
                let container = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "in".to_string(), args: vec![container] })
            }
            ("contains", 1) => {
                let other = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "contains".to_string(), args: vec![other] })
            }
            ("inside", 1) => {
                let other = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "inside".to_string(), args: vec![other] })
            }
            ("test", 1) | ("test", 2) => {
                let mut args = args.into_iter();
                let re = args.next().unwrap();
                let flags = args.next().unwrap_or(Expr::Literal(Literal::Null));
                Ok(Expr::RegexTest {
                    input_expr: Box::new(Expr::Input),
                    re: Box::new(re),
                    flags: Box::new(flags),
                })
            }
            ("match", 1) | ("match", 2) => {
                let mut args = args.into_iter();
                let re = args.next().unwrap();
                let flags = args.next().unwrap_or(Expr::Literal(Literal::Null));
                Ok(Expr::RegexMatch {
                    input_expr: Box::new(Expr::Input),
                    re: Box::new(re),
                    flags: Box::new(flags),
                })
            }
            ("capture", 1) | ("capture", 2) => {
                let mut args = args.into_iter();
                let re = args.next().unwrap();
                let flags = args.next().unwrap_or(Expr::Literal(Literal::Null));
                Ok(Expr::RegexCapture {
                    input_expr: Box::new(Expr::Input),
                    re: Box::new(re),
                    flags: Box::new(flags),
                })
            }
            ("scan", 1) | ("scan", 2) => {
                let mut args = args.into_iter();
                let re = args.next().unwrap();
                let flags = args.next().unwrap_or(Expr::Literal(Literal::Null));
                Ok(Expr::RegexScan {
                    input_expr: Box::new(Expr::Input),
                    re: Box::new(re),
                    flags: Box::new(flags),
                })
            }
            ("sub", 2) | ("sub", 3) => {
                let mut args = args.into_iter();
                let re = args.next().unwrap();
                let tostr = args.next().unwrap();
                let flags = args.next().unwrap_or(Expr::Literal(Literal::Null));
                Ok(Expr::RegexSub {
                    input_expr: Box::new(Expr::Input),
                    re: Box::new(re),
                    tostr: Box::new(tostr),
                    flags: Box::new(flags),
                })
            }
            ("gsub", 2) | ("gsub", 3) => {
                let mut args = args.into_iter();
                let re = args.next().unwrap();
                let tostr = args.next().unwrap();
                let flags = args.next().unwrap_or(Expr::Literal(Literal::Null));
                Ok(Expr::RegexGsub {
                    input_expr: Box::new(Expr::Input),
                    re: Box::new(re),
                    tostr: Box::new(tostr),
                    flags: Box::new(flags),
                })
            }
            ("flatten", 1) => {
                let depth = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "flatten".to_string(), args: vec![depth] })
            }
            ("splits", 1) | ("splits", 2) => {
                // splits(re) = split(re; null) | .[] — regex split streamed
                // splits(re; flags) = split(re; flags) | .[]
                let mut args_iter = args.into_iter();
                let re = args_iter.next().unwrap();
                let flags = args_iter.next().unwrap_or(Expr::Literal(Literal::Null));
                Ok(Expr::Pipe {
                    left: Box::new(Expr::CallBuiltin { name: "split".to_string(), args: vec![re, flags] }),
                    right: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                })
            }
            ("split", 1) | ("split", 2) => {
                let n = args.len();
                let mut args = args.into_iter();
                let sep = args.next().unwrap();
                if n == 2 {
                    let flags = args.next().unwrap();
                    Ok(Expr::CallBuiltin { name: "split".to_string(), args: vec![sep, flags] })
                } else {
                    Ok(Expr::CallBuiltin { name: "split".to_string(), args: vec![sep] })
                }
            }
            ("join", 1) => {
                let sep = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "join".to_string(), args: vec![sep] })
            }
            ("ascii_downcase", 0) => Ok(Expr::UnaryOp { op: UnaryOp::AsciiDowncase, operand: Box::new(Expr::Input) }),
            ("ascii_upcase", 0) => Ok(Expr::UnaryOp { op: UnaryOp::AsciiUpcase, operand: Box::new(Expr::Input) }),
            ("ltrimstr", 1) => {
                let s = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "ltrimstr".to_string(), args: vec![s] })
            }
            ("rtrimstr", 1) => {
                let s = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "rtrimstr".to_string(), args: vec![s] })
            }
            ("startswith", 1) => {
                let s = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "startswith".to_string(), args: vec![s] })
            }
            ("endswith", 1) => {
                let s = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "endswith".to_string(), args: vec![s] })
            }
            ("indices", 1) | ("index", 1) | ("rindex", 1) => {
                let s = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: name.to_string(), args: vec![s] })
            }
            ("error", 1) => {
                let msg = args.into_iter().next().unwrap();
                Ok(Expr::Error { msg: Some(Box::new(msg)) })
            }
            ("debug", 1) => {
                let msg = args.into_iter().next().unwrap();
                Ok(Expr::Debug { expr: Box::new(msg) })
            }
            ("halt_error", 1) => {
                // halt_error(exit_code): evaluate the argument to an exit
                // code (default 5 only on failure to evaluate a number),
                // print the *input* to stderr, then terminate. Runtime
                // handles message encoding — see eval_call_builtin.
                let code = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "halt_error".to_string(), args: vec![code] })
            }
            ("pow", 2) | ("atan2", 2) | ("fma", 3)
            | ("remainder", 2) | ("drem", 2) | ("hypot", 2)
            | ("ldexp", 2) | ("scalb", 2) | ("scalbln", 2)
            | ("copysign", 2) | ("fdim", 2) | ("fmax", 2) | ("fmin", 2)
            | ("fmod", 2) | ("nextafter", 2) | ("nexttoward", 2)
            | ("jn", 2) | ("yn", 2) => {
                Ok(Expr::CallBuiltin { name: name.to_string(), args })
            }
            ("nth", 1) => {
                // jq 1.8.1: def nth($n): .[$n];
                // — supports negative indices and string indices, unlike nth/2.
                let n_expr = args.into_iter().next().unwrap();
                let n_var = self.scope.alloc_var("__nth_n__");
                Ok(Expr::LetBinding {
                    var_index: n_var,
                    value: Box::new(n_expr),
                    body: Box::new(Expr::Index {
                        expr: Box::new(Expr::Input),
                        key: Box::new(Expr::LoadVar { var_index: n_var }),
                    }),
                })
            }
            ("nth", 2) => {
                let mut args = args.into_iter();
                let n_expr = args.next().unwrap();
                let generator = args.next().unwrap();
                // nth(n; g) = n as $n | if $n < 0 then error
                //   else foreach g as $x (-1; .+1; if . == $n then $x else empty end) end
                let n_var = self.scope.alloc_var("__nth_n__");
                let x_var = self.scope.alloc_var("__nth_x__");
                let cnt_var = self.scope.alloc_var("__nth_cnt__");
                let foreach_expr = Expr::Foreach {
                    source: Box::new(generator),
                    init: Box::new(Expr::Literal(Literal::Num(-1.0, None))),
                    var_index: x_var,
                    acc_index: cnt_var,
                    update: Box::new(Expr::BinOp {
                        op: BinOp::Add,
                        lhs: Box::new(Expr::Input),
                        rhs: Box::new(Expr::Literal(Literal::Num(1.0, None))),
                    }),
                    extract: Some(Box::new(Expr::IfThenElse {
                        cond: Box::new(Expr::BinOp {
                            op: BinOp::Eq,
                            lhs: Box::new(Expr::Input),
                            // Floor the index per-item rather than eagerly at the
                            // binding: jq evaluates the index lazily, so a
                            // non-numeric index errors only once the generator
                            // yields. An empty generator therefore produces no
                            // output instead of an eager floor error (#806).
                            rhs: Box::new(Expr::UnaryOp {
                                op: UnaryOp::Floor,
                                operand: Box::new(Expr::LoadVar { var_index: n_var }),
                            }),
                        }),
                        then_branch: Box::new(Expr::LoadVar { var_index: x_var }),
                        else_branch: Box::new(Expr::Empty),
                    })),
                };
                // first(foreach ...) to get only the first match
                let first_match = Expr::Limit {
                    count: Box::new(Expr::Literal(Literal::Num(1.0, None))),
                    generator: Box::new(foreach_expr),
                };
                let body = Expr::IfThenElse {
                    cond: Box::new(Expr::BinOp {
                        op: BinOp::Lt,
                        lhs: Box::new(Expr::LoadVar { var_index: n_var }),
                        rhs: Box::new(Expr::Literal(Literal::Num(0.0, None))),
                    }),
                    then_branch: Box::new(Expr::Error {
                        msg: Some(Box::new(Expr::Literal(Literal::Str("nth doesn't support negative indices".to_string())))),
                    }),
                    else_branch: Box::new(first_match),
                };
                // jq's nth(n; g) floors the index: nth(0.5) == nth(0), and a
                // non-numeric index raises (floor errors on non-numbers). The
                // foreach counter is integer-valued, so `counter == $n` never
                // matched for fractional n (silently empty) and a string index
                // fell through to empty instead of erroring (#719). The floor
                // now lives in the per-item comparison above so it stays lazy:
                // the negative-index guard sees the raw value (`"a" < 0` is
                // false, so a string index proceeds to the deferred error) and
                // an empty generator never forces the floor at all (#806).
                Ok(Expr::LetBinding {
                    var_index: n_var,
                    value: Box::new(n_expr),
                    body: Box::new(body),
                })
            }
            ("label", 1) => {
                let body = args.into_iter().next().unwrap();
                let var_idx = self.scope.alloc_var("__label__");
                Ok(Expr::Label {
                    var_index: var_idx,
                    body: Box::new(body),
                })
            }
            ("tojson", 0) => Ok(Expr::UnaryOp { op: UnaryOp::ToJson, operand: Box::new(Expr::Input) }),
            ("fromjson", 0) => Ok(Expr::UnaryOp { op: UnaryOp::FromJson, operand: Box::new(Expr::Input) }),
            ("strftime", 1) | ("strptime", 1)
            | ("todate", 0) | ("fromdate", 0) | ("date", 0) => {
                Ok(Expr::CallBuiltin { name: name.to_string(), args })
            }
            ("combinations", 1) => {
                Ok(Expr::CallBuiltin { name: "combinations".to_string(), args })
            }
            ("input", 0) => Ok(Expr::ReadInput),
            ("inputs", 0) => Ok(Expr::ReadInputs),
            ("genlabel", 0) => Ok(Expr::GenLabel),
            ("format", 1) => {
                // `format(f)` is the dynamic form of `@<fmt>`: evaluate `f`
                // at runtime to get the format name (one of csv, tsv, json,
                // text, html, sh, uri, base64, base64d), then apply that
                // format to the current input. Delegate to the runtime so
                // the directive name can vary per input.
                let fmt = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "format".to_string(), args: vec![fmt] })
            }
            ("length", 0) => Ok(Expr::UnaryOp { op: UnaryOp::Length, operand: Box::new(Expr::Input) }),
            ("type", 0) => Ok(Expr::UnaryOp { op: UnaryOp::Type, operand: Box::new(Expr::Input) }),
            ("trimstr", 1) => {
                let s = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "trimstr".to_string(), args: vec![s] })
            }
            // toboolean/0: convert to boolean
            ("toboolean", 0) => {
                Ok(Expr::CallBuiltin { name: "toboolean".to_string(), args: vec![] })
            }
            // add/1: add(f) = reduce .[] as $x (null; . + ($x | f))
            // But it's simpler to delegate to CallBuiltin
            ("add", 1) => {
                let f = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "add".to_string(), args: vec![f] })
            }
            // skip/2: skip(n; gen) = limit(n; gen) | empty, limit(n; gen) outputs n items then the rest
            // Actually: skip(n; exp) = def _skip(n; exp): if n > 0 then (exp | _skip(n-1; exp)) else ., exp end;
            // Simpler: skip(n; exp) is like foreach range(n) as $_ (.; exp) | exp... no.
            // jq def: def skip($n; exp): def _skip: if $n > 0 then (exp | . as $x | $n - 1 | _skip) else ., exp end; _skip;
            // Let's use CallBuiltin and handle in eval
            ("skip", 2) => {
                let mut args = args.into_iter();
                let n = args.next().unwrap();
                let generator = args.next().unwrap();
                Ok(Expr::CallBuiltin { name: "skip".to_string(), args: vec![n, generator] })
            }
            // pick/1: pick(f) extracts paths from . that f generates
            ("pick", 1) => {
                let f = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "pick".to_string(), args: vec![f] })
            }
            // bsearch/1: binary search
            ("bsearch", 1) => {
                let target = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "bsearch".to_string(), args: vec![target] })
            }
            // strflocaltime/1
            ("strflocaltime", 1) => {
                let fmt = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "strflocaltime".to_string(), args: vec![fmt] })
            }
            // walk/1: walk(f) recursively applies f to all values
            ("walk", 1) => {
                let f = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "walk".to_string(), args: vec![f] })
            }
            // fromstream/1, truncate_stream/1: stream reassembly + slicing (#89)
            ("fromstream", 1) => {
                let f = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "fromstream".to_string(), args: vec![f] })
            }
            ("truncate_stream", 1) => {
                let f = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "truncate_stream".to_string(), args: vec![f] })
            }
            // exec/1: execute shell command and return stdout
            ("exec", 1) => {
                let cmd = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "exec".to_string(), args: vec![cmd] })
            }
            // execv/1: execute shell command, return {exitcode, stdout, stderr}
            ("execv", 1) => {
                let cmd = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "execv".to_string(), args: vec![cmd] })
            }
            // exec/2: pipe generator output to shell command, yield stdout lines
            ("exec", 2) => {
                let mut args = args.into_iter();
                let gen = args.next().unwrap();
                let cmd = args.next().unwrap();
                Ok(Expr::CallBuiltin { name: "exec".to_string(), args: vec![gen, cmd] })
            }
            // fromcsv/0, fromtsv/0: parse CSV/TSV string, yield arrays per row
            ("fromcsv", 0) => {
                Ok(Expr::CallBuiltin { name: "fromcsv".to_string(), args: vec![] })
            }
            ("fromtsv", 0) => {
                Ok(Expr::CallBuiltin { name: "fromtsv".to_string(), args: vec![] })
            }
            // fromcsvh/0, fromcsvh/1: parse CSV with headers, yield objects per row
            ("fromcsvh", 0) => {
                Ok(Expr::CallBuiltin { name: "fromcsvh".to_string(), args: vec![] })
            }
            ("fromcsvh", 1) => {
                let headers = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "fromcsvh".to_string(), args: vec![headers] })
            }
            // fromtsvh/0, fromtsvh/1: parse TSV with headers, yield objects per row
            ("fromtsvh", 0) => {
                Ok(Expr::CallBuiltin { name: "fromtsvh".to_string(), args: vec![] })
            }
            ("fromtsvh", 1) => {
                let headers = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "fromtsvh".to_string(), args: vec![headers] })
            }
            // fromdateiso8601/0, todateiso8601/0 (canonical jq names)
            // plus fromisodate/0, toisodate/0 aliases kept for
            // backward compatibility: all four route to the same
            // runtime impl.
            ("fromdateiso8601", 0) | ("fromisodate", 0) => {
                Ok(Expr::CallBuiltin { name: "fromisodate".to_string(), args: vec![] })
            }
            ("todateiso8601", 0) | ("toisodate", 0) => {
                Ok(Expr::CallBuiltin { name: "toisodate".to_string(), args: vec![] })
            }
            // IN/1: IN(s) = any(. == s; .)... actually IN(s) = . as $x | first(s | if . == $x then true else empty end) // false
            ("IN", 1) => {
                let s = args.into_iter().next().unwrap();
                let x_var = self.scope.alloc_var("__in_x__");
                Ok(Expr::LetBinding {
                    var_index: x_var,
                    value: Box::new(Expr::Input),
                    body: Box::new(Expr::Alternative {
                        primary: Box::new(Expr::Limit {
                            count: Box::new(Expr::Literal(Literal::Num(1.0, None))),
                            generator: Box::new(Expr::Pipe {
                                left: Box::new(s),
                                right: Box::new(Expr::IfThenElse {
                                    cond: Box::new(Expr::BinOp {
                                        op: BinOp::Eq,
                                        lhs: Box::new(Expr::Input),
                                        rhs: Box::new(Expr::LoadVar { var_index: x_var }),
                                    }),
                                    then_branch: Box::new(Expr::Literal(Literal::True)),
                                    else_branch: Box::new(Expr::Empty),
                                }),
                            }),
                        }),
                        fallback: Box::new(Expr::Literal(Literal::False)),
                    }),
                })
            }
            // IN/2: jq's `IN(src; s)` is `any(src == s; .)` — BOTH `src` and the
            // candidate set `s` are evaluated against the original input. The old
            // desugar fed the candidate set the reduce accumulator (the update's
            // `.`), so `IN(1; .[])` iterated `false` and `IN(false; .)` compared
            // against the accumulator. Bind the original input as `$dot` and pipe
            // the candidate set from it: #846
            //   . as $dot
            //   | reduce src as $x (false;
            //       . or (first(($dot | s) | if . == $x then true else empty) // false))
            ("IN", 2) => {
                let mut args = args.into_iter();
                let src = args.next().unwrap();
                let s = args.next().unwrap();
                let dot_var = self.scope.alloc_var("__in2_dot__");
                let x_var = self.scope.alloc_var("__in2_x__");
                let acc_var = self.scope.alloc_var("__in2_acc__");
                let in_check = Expr::Alternative {
                    primary: Box::new(Expr::Limit {
                        count: Box::new(Expr::Literal(Literal::Num(1.0, None))),
                        generator: Box::new(Expr::Pipe {
                            left: Box::new(Expr::Pipe {
                                left: Box::new(Expr::LoadVar { var_index: dot_var }),
                                right: Box::new(s),
                            }),
                            right: Box::new(Expr::IfThenElse {
                                cond: Box::new(Expr::BinOp {
                                    op: BinOp::Eq,
                                    lhs: Box::new(Expr::Input),
                                    rhs: Box::new(Expr::LoadVar { var_index: x_var }),
                                }),
                                then_branch: Box::new(Expr::Literal(Literal::True)),
                                else_branch: Box::new(Expr::Empty),
                            }),
                        }),
                    }),
                    fallback: Box::new(Expr::Literal(Literal::False)),
                };
                Ok(Expr::LetBinding {
                    var_index: dot_var,
                    value: Box::new(Expr::Input),
                    body: Box::new(Expr::Reduce {
                        source: Box::new(src),
                        init: Box::new(Expr::Literal(Literal::False)),
                        var_index: x_var,
                        acc_index: acc_var,
                        update: Box::new(Expr::BinOp {
                            op: BinOp::Or,
                            lhs: Box::new(Expr::Input),
                            rhs: Box::new(in_check),
                        }),
                    }),
                })
            }
            // INDEX/1: INDEX(idx_expr) = INDEX(.[]; idx_expr).
            // jq advertised both /1 and /2 in `builtins`, but /1 had no
            // dispatch path so callers got an "unknown function" error
            // (#476). Desugar straight to the same Reduce shape as /2.
            ("INDEX", 1) => {
                let f = args.into_iter().next().unwrap();
                let stream = Expr::Each { input_expr: Box::new(Expr::Input) };
                let x_var = self.scope.alloc_var("__idx_x__");
                let acc_var = self.scope.alloc_var("__idx_acc__");
                Ok(Self::index_reduce(stream, f, x_var, acc_var))
            }
            // INDEX/2: INDEX(stream; f) = reduce stream as $x ({}; .[$x|f|tostring] = $x)
            ("INDEX", 2) => {
                let mut args = args.into_iter();
                let stream = args.next().unwrap();
                let f = args.next().unwrap();
                let x_var = self.scope.alloc_var("__idx_x__");
                let acc_var = self.scope.alloc_var("__idx_acc__");
                Ok(Self::index_reduce(stream, f, x_var, acc_var))
            }
            // JOIN/2: JOIN($idx; f) = [.[] | [., $idx[f|tostring]]]
            ("JOIN", 2) => {
                let mut args = args.into_iter();
                let idx_expr = args.next().unwrap();
                let f = args.next().unwrap();
                let idx_var = self.scope.alloc_var("__join_idx__");
                // Bind idx to a variable, then [.[] | [., $idx[f|tostring]]]
                Ok(Expr::LetBinding {
                    var_index: idx_var,
                    value: Box::new(idx_expr),
                    body: Box::new(Expr::Collect {
                        generator: Box::new(Expr::Pipe {
                            left: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                            right: Box::new(Expr::Collect {
                                generator: Box::new(Expr::Comma {
                                    left: Box::new(Expr::Input),
                                    right: Box::new(Expr::Index {
                                        expr: Box::new(Expr::LoadVar { var_index: idx_var }),
                                        key: Box::new(Expr::Pipe {
                                            left: Box::new(f),
                                            right: Box::new(Expr::UnaryOp { op: UnaryOp::ToString, operand: Box::new(Expr::Input) }),
                                        }),
                                    }),
                                }),
                            }),
                        }),
                    }),
                })
            }
            // JOIN/3: JOIN($idx; stream; idx_expr) = stream | [., $idx[idx_expr]]
            ("JOIN", 3) => {
                let mut args = args.into_iter();
                let idx = args.next().unwrap();
                let stream = args.next().unwrap();
                let idx_expr = args.next().unwrap();
                let idx_var = self.scope.alloc_var("__join_idx__");
                Ok(Expr::LetBinding {
                    var_index: idx_var,
                    value: Box::new(idx),
                    body: Box::new(Expr::Pipe {
                        left: Box::new(stream),
                        right: Box::new(Expr::Collect {
                            generator: Box::new(Expr::Comma {
                                left: Box::new(Expr::Input),
                                right: Box::new(Expr::Index {
                                    expr: Box::new(Expr::LoadVar { var_index: idx_var }),
                                    key: Box::new(idx_expr),
                                }),
                            }),
                        }),
                    }),
                })
            }
            // JOIN/4: JOIN($idx; stream; idx_expr; join_expr)
            //         = stream | [., $idx[idx_expr]] | join_expr
            ("JOIN", 4) => {
                let mut args = args.into_iter();
                let idx = args.next().unwrap();
                let stream = args.next().unwrap();
                let idx_expr = args.next().unwrap();
                let join_expr = args.next().unwrap();
                let idx_var = self.scope.alloc_var("__join_idx__");
                Ok(Expr::LetBinding {
                    var_index: idx_var,
                    value: Box::new(idx),
                    body: Box::new(Expr::Pipe {
                        left: Box::new(stream),
                        right: Box::new(Expr::Pipe {
                            left: Box::new(Expr::Collect {
                                generator: Box::new(Expr::Comma {
                                    left: Box::new(Expr::Input),
                                    right: Box::new(Expr::Index {
                                        expr: Box::new(Expr::LoadVar { var_index: idx_var }),
                                        key: Box::new(idx_expr),
                                    }),
                                }),
                            }),
                            right: Box::new(join_expr),
                        }),
                    }),
                })
            }
            // del/1: del(f) — delegate to eval for proper slice handling
            ("del", 1) => {
                let f = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "del".to_string(), args: vec![f] })
            }
            _ => {
                // Check user-defined functions
                if let Some(func_id) = self.scope.lookup_func(name, args.len()) {
                    Ok(self.scope.make_funccall(func_id, args))
                } else {
                    // Undefined function name. jq resolves names lazily, so this
                    // is only an error when reachable: defer it (#807).
                    Ok(self.defer_unknown_func(name, args.len()))
                }
            }
        }
    }

    fn parse_string_interpolation(&mut self) -> Result<Expr> {
        self.expect(&Token::LParen)?;
        let mut parts = Vec::new();

        while !self.at(&Token::RParen) && !self.at_eof() {
            match self.current().clone() {
                Token::Str(s) => {
                    self.advance();
                    parts.push(StringPart::Literal(s));
                }
                Token::Ident(ref name) if name == "__expr__" => {
                    self.advance();
                    self.expect(&Token::LParen)?;
                    let expr = self.parse_pipe()?;
                    self.expect(&Token::RParen)?;
                    parts.push(StringPart::Expr(expr));
                }
                Token::Semicolon => {
                    self.advance();
                }
                _ => {
                    bail!("unexpected token in string interpolation: {:?}", self.current());
                }
            }
        }
        self.expect(&Token::RParen)?;

        if parts.len() == 1 {
            if let StringPart::Literal(s) = &parts[0] {
                return Ok(Expr::Literal(Literal::Str(s.clone())));
            }
        }

        Ok(Expr::StringInterpolation { parts })
    }
}

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Pattern {
    Var(String),
    Array(Vec<Pattern>),
    /// Object-pattern destructure. The key is an Expr so computed-key
    /// patterns like `. as {(.x): $v}` can defer key evaluation to
    /// runtime; literal keys are stored as `Expr::Literal(Literal::Str)`.
    Object(Vec<(Expr, Pattern)>),
    /// $var: sub_pattern — binds $var to whole value AND destructures via sub_pattern
    VarAndSub(String, Box<Pattern>),
}

/// Build the field-lookup expression for an object-pattern pair.
///
/// jq evaluates the computed key against the destructured value (not the
/// outer pipeline input), so we wrap the index in `value | .[key]` for
/// non-literal keys. Literal-string keys are emitted as a plain index to
/// keep the compact case fast.
fn obj_pat_field_expr(value: Expr, key: Expr) -> Expr {
    if matches!(key, Expr::Literal(Literal::Str(_))) {
        Expr::Index {
            expr: Box::new(value),
            key: Box::new(key),
        }
    } else {
        Expr::Pipe {
            left: Box::new(value),
            right: Box::new(Expr::Index {
                expr: Box::new(Expr::Input),
                key: Box::new(key),
            }),
        }
    }
}

/// Recursively wrap the leaf path-update of a `mutate(...)` body in
/// `Expr::Mutate`. Peels intermediate `LetBinding` wrappers — `+=`/`-=`/
/// etc. desugar to `LetBinding { body: Update }` at parse time, and any
/// hand-written `let ... in path-update` inside `mutate(...)` is treated
/// the same way. Any leaf that is not `Expr::Assign` or `Expr::Update` is
/// a parse error: composite forms like `if/then/else`, `reduce`, `,`, and
/// `|` must distribute the marker inward by wrapping each leaf instead.
fn wrap_mutate(body: Expr) -> Result<Expr> {
    match body {
        Expr::Assign { path_expr, value_expr } => Ok(Expr::Mutate {
            path_expr, value_expr, kind: MutateKind::Assign,
        }),
        Expr::Update { path_expr, update_expr } => Ok(Expr::Mutate {
            path_expr, value_expr: update_expr, kind: MutateKind::Update,
        }),
        Expr::LetBinding { var_index, value, body } => Ok(Expr::LetBinding {
            var_index, value, body: Box::new(wrap_mutate(*body)?),
        }),
        _ => bail!(
            "mutate(...) body must be a top-level path-update operator \
             (=, |=, +=, -=, *=, /=, %=, //=); wrap composite forms by \
             distributing mutate inward across each leaf"
        ),
    }
}

/// Remap FuncCall func_ids in an expression tree using a mapping table.
fn remap_func_ids(expr: Expr, map: &[(usize, usize)]) -> Expr {
    match expr {
        Expr::FuncCall { func_id, args } => {
            let new_id = map.iter().find(|(old, _)| *old == func_id).map(|(_, new)| *new).unwrap_or(func_id);
            Expr::FuncCall { func_id: new_id, args: args.into_iter().map(|a| remap_func_ids(a, map)).collect() }
        }
        Expr::Pipe { left, right } => Expr::Pipe {
            left: Box::new(remap_func_ids(*left, map)),
            right: Box::new(remap_func_ids(*right, map)),
        },
        Expr::Comma { left, right } => Expr::Comma {
            left: Box::new(remap_func_ids(*left, map)),
            right: Box::new(remap_func_ids(*right, map)),
        },
        Expr::BinOp { op, lhs, rhs } => Expr::BinOp {
            op, lhs: Box::new(remap_func_ids(*lhs, map)), rhs: Box::new(remap_func_ids(*rhs, map)),
        },
        Expr::UnaryOp { op, operand } => Expr::UnaryOp {
            op, operand: Box::new(remap_func_ids(*operand, map)),
        },
        Expr::Index { expr, key } => Expr::Index {
            expr: Box::new(remap_func_ids(*expr, map)), key: Box::new(remap_func_ids(*key, map)),
        },
        Expr::IndexOpt { expr, key } => Expr::IndexOpt {
            expr: Box::new(remap_func_ids(*expr, map)), key: Box::new(remap_func_ids(*key, map)),
        },
        Expr::IfThenElse { cond, then_branch, else_branch } => Expr::IfThenElse {
            cond: Box::new(remap_func_ids(*cond, map)),
            then_branch: Box::new(remap_func_ids(*then_branch, map)),
            else_branch: Box::new(remap_func_ids(*else_branch, map)),
        },
        Expr::TryCatch { try_expr, catch_expr, restore_dot } => Expr::TryCatch {
            restore_dot,
            try_expr: Box::new(remap_func_ids(*try_expr, map)),
            catch_expr: Box::new(remap_func_ids(*catch_expr, map)),
        },
        Expr::Each { input_expr } => Expr::Each { input_expr: Box::new(remap_func_ids(*input_expr, map)) },
        Expr::EachOpt { input_expr } => Expr::EachOpt { input_expr: Box::new(remap_func_ids(*input_expr, map)) },
        Expr::LetBinding { var_index, value, body } => Expr::LetBinding {
            var_index, value: Box::new(remap_func_ids(*value, map)), body: Box::new(remap_func_ids(*body, map)),
        },
        Expr::Collect { generator } => Expr::Collect { generator: Box::new(remap_func_ids(*generator, map)) },
        Expr::ObjectConstruct { pairs } => Expr::ObjectConstruct {
            pairs: pairs.into_iter().map(|(k, v)| (remap_func_ids(k, map), remap_func_ids(v, map))).collect(),
        },
        Expr::Alternative { primary, fallback } => Expr::Alternative {
            primary: Box::new(remap_func_ids(*primary, map)),
            fallback: Box::new(remap_func_ids(*fallback, map)),
        },
        Expr::Negate { operand } => Expr::Negate { operand: Box::new(remap_func_ids(*operand, map)) },
        Expr::Recurse { input_expr } => Expr::Recurse { input_expr: Box::new(remap_func_ids(*input_expr, map)) },
        Expr::Reduce { source, init, var_index, acc_index, update } => Expr::Reduce {
            source: Box::new(remap_func_ids(*source, map)),
            init: Box::new(remap_func_ids(*init, map)),
            var_index, acc_index,
            update: Box::new(remap_func_ids(*update, map)),
        },
        Expr::Foreach { source, init, var_index, acc_index, update, extract } => Expr::Foreach {
            source: Box::new(remap_func_ids(*source, map)),
            init: Box::new(remap_func_ids(*init, map)),
            var_index, acc_index,
            update: Box::new(remap_func_ids(*update, map)),
            extract: extract.map(|e| Box::new(remap_func_ids(*e, map))),
        },
        Expr::Range { from, to, step } => Expr::Range {
            from: Box::new(remap_func_ids(*from, map)),
            to: Box::new(remap_func_ids(*to, map)),
            step: step.map(|s| Box::new(remap_func_ids(*s, map))),
        },
        Expr::Label { var_index, body } => Expr::Label {
            var_index, body: Box::new(remap_func_ids(*body, map)),
        },
        Expr::Break { var_index, value } => Expr::Break {
            var_index, value: Box::new(remap_func_ids(*value, map)),
        },
        Expr::Update { path_expr, update_expr } => Expr::Update {
            path_expr: Box::new(remap_func_ids(*path_expr, map)),
            update_expr: Box::new(remap_func_ids(*update_expr, map)),
        },
        Expr::Assign { path_expr, value_expr } => Expr::Assign {
            path_expr: Box::new(remap_func_ids(*path_expr, map)),
            value_expr: Box::new(remap_func_ids(*value_expr, map)),
        },
        Expr::Mutate { path_expr, value_expr, kind } => Expr::Mutate {
            path_expr: Box::new(remap_func_ids(*path_expr, map)),
            value_expr: Box::new(remap_func_ids(*value_expr, map)),
            kind,
        },
        Expr::PathExpr { expr } => Expr::PathExpr { expr: Box::new(remap_func_ids(*expr, map)) },
        Expr::SetPath { path, value } => Expr::SetPath {
            path: Box::new(remap_func_ids(*path, map)), value: Box::new(remap_func_ids(*value, map)),
        },
        Expr::GetPath { path } => Expr::GetPath { path: Box::new(remap_func_ids(*path, map)) },
        Expr::DelPaths { paths } => Expr::DelPaths { paths: Box::new(remap_func_ids(*paths, map)) },
        Expr::Limit { count, generator } => Expr::Limit {
            count: Box::new(remap_func_ids(*count, map)),
            generator: Box::new(remap_func_ids(*generator, map)),
        },
        Expr::While { cond, update } => Expr::While {
            cond: Box::new(remap_func_ids(*cond, map)),
            update: Box::new(remap_func_ids(*update, map)),
        },
        Expr::Until { cond, update } => Expr::Until {
            cond: Box::new(remap_func_ids(*cond, map)),
            update: Box::new(remap_func_ids(*update, map)),
        },
        Expr::Repeat { update } => Expr::Repeat { update: Box::new(remap_func_ids(*update, map)) },
        Expr::AllShort { generator, predicate } => Expr::AllShort {
            generator: Box::new(remap_func_ids(*generator, map)),
            predicate: Box::new(remap_func_ids(*predicate, map)),
        },
        Expr::AnyShort { generator, predicate } => Expr::AnyShort {
            generator: Box::new(remap_func_ids(*generator, map)),
            predicate: Box::new(remap_func_ids(*predicate, map)),
        },
        Expr::Error { msg } => Expr::Error { msg: msg.map(|m| Box::new(remap_func_ids(*m, map))) },
        Expr::Format { name, expr } => Expr::Format { name, expr: Box::new(remap_func_ids(*expr, map)) },
        Expr::ClosureOp { op, input_expr, key_expr } => Expr::ClosureOp {
            op, input_expr: Box::new(remap_func_ids(*input_expr, map)),
            key_expr: Box::new(remap_func_ids(*key_expr, map)),
        },
        Expr::StringInterpolation { parts } => Expr::StringInterpolation {
            parts: parts.into_iter().map(|p| match p {
                StringPart::Expr(e) => StringPart::Expr(remap_func_ids(e, map)),
                lit => lit,
            }).collect(),
        },
        Expr::Slice { expr, from, to } => Expr::Slice {
            expr: Box::new(remap_func_ids(*expr, map)),
            from: from.map(|f| Box::new(remap_func_ids(*f, map))),
            to: to.map(|t| Box::new(remap_func_ids(*t, map))),
        },
        Expr::Debug { expr } => Expr::Debug { expr: Box::new(remap_func_ids(*expr, map)) },
        Expr::Stderr { expr } => Expr::Stderr { expr: Box::new(remap_func_ids(*expr, map)) },
        Expr::RegexTest { input_expr, re, flags } => Expr::RegexTest {
            input_expr: Box::new(remap_func_ids(*input_expr, map)),
            re: Box::new(remap_func_ids(*re, map)),
            flags: Box::new(remap_func_ids(*flags, map)),
        },
        Expr::RegexMatch { input_expr, re, flags } => Expr::RegexMatch {
            input_expr: Box::new(remap_func_ids(*input_expr, map)),
            re: Box::new(remap_func_ids(*re, map)),
            flags: Box::new(remap_func_ids(*flags, map)),
        },
        Expr::RegexCapture { input_expr, re, flags } => Expr::RegexCapture {
            input_expr: Box::new(remap_func_ids(*input_expr, map)),
            re: Box::new(remap_func_ids(*re, map)),
            flags: Box::new(remap_func_ids(*flags, map)),
        },
        Expr::RegexScan { input_expr, re, flags } => Expr::RegexScan {
            input_expr: Box::new(remap_func_ids(*input_expr, map)),
            re: Box::new(remap_func_ids(*re, map)),
            flags: Box::new(remap_func_ids(*flags, map)),
        },
        Expr::RegexSub { input_expr, re, tostr, flags } => Expr::RegexSub {
            input_expr: Box::new(remap_func_ids(*input_expr, map)),
            re: Box::new(remap_func_ids(*re, map)),
            tostr: Box::new(remap_func_ids(*tostr, map)),
            flags: Box::new(remap_func_ids(*flags, map)),
        },
        Expr::RegexGsub { input_expr, re, tostr, flags } => Expr::RegexGsub {
            input_expr: Box::new(remap_func_ids(*input_expr, map)),
            re: Box::new(remap_func_ids(*re, map)),
            tostr: Box::new(remap_func_ids(*tostr, map)),
            flags: Box::new(remap_func_ids(*flags, map)),
        },
        Expr::AlternativeDestructure { alternatives } => Expr::AlternativeDestructure {
            alternatives: alternatives.into_iter().map(|a| remap_func_ids(a, map)).collect(),
        },
        Expr::CallBuiltin { name, args } => Expr::CallBuiltin {
            name, args: args.into_iter().map(|a| remap_func_ids(a, map)).collect(),
        },
        Expr::Memoize { slot_id, key, body } => Expr::Memoize {
            slot_id,
            key: key.map(|k| Box::new(remap_func_ids(*k, map))),
            body: Box::new(remap_func_ids(*body, map)),
        },
        // Leaf nodes: no remapping needed
        Expr::Input | Expr::Literal(_) | Expr::LoadVar { .. } | Expr::Empty
        | Expr::Not | Expr::Loc { .. } | Expr::Env | Expr::Builtins
        | Expr::ReadInput | Expr::ReadInputs | Expr::ModuleMeta | Expr::GenLabel => expr,
    }
}

fn make_type_select(type_name: &str) -> Expr {
    Expr::IfThenElse {
        cond: Box::new(Expr::BinOp {
            op: BinOp::Eq,
            lhs: Box::new(Expr::UnaryOp { op: UnaryOp::Type, operand: Box::new(Expr::Input) }),
            rhs: Box::new(Expr::Literal(Literal::Str(type_name.to_string()))),
        }),
        then_branch: Box::new(Expr::Input),
        else_branch: Box::new(Expr::Empty),
    }
}

fn name_to_unary_op(name: &str) -> Result<UnaryOp> {
    match name {
        "length" => Ok(UnaryOp::Length),
        "utf8bytelength" => Ok(UnaryOp::Utf8ByteLength),
        "type" => Ok(UnaryOp::Type),
        "tostring" => Ok(UnaryOp::ToString),
        "tonumber" => Ok(UnaryOp::ToNumber),
        "tojson" => Ok(UnaryOp::ToJson),
        "fromjson" => Ok(UnaryOp::FromJson),
        "explode" => Ok(UnaryOp::Explode),
        "implode" => Ok(UnaryOp::Implode),
        "ascii_downcase" => Ok(UnaryOp::AsciiDowncase),
        "ascii_upcase" => Ok(UnaryOp::AsciiUpcase),
        "ltrim" => Ok(UnaryOp::Ltrim),
        "rtrim" => Ok(UnaryOp::Rtrim),
        "trim" => Ok(UnaryOp::Trim),
        "floor" => Ok(UnaryOp::Floor),
        "ceil" => Ok(UnaryOp::Ceil),
        "round" => Ok(UnaryOp::Round),
        "fabs" => Ok(UnaryOp::Fabs),
        "sqrt" => Ok(UnaryOp::Sqrt),
        "sin" => Ok(UnaryOp::Sin),
        "cos" => Ok(UnaryOp::Cos),
        "tan" => Ok(UnaryOp::Tan),
        "asin" => Ok(UnaryOp::Asin),
        "acos" => Ok(UnaryOp::Acos),
        "atan" => Ok(UnaryOp::Atan),
        "sinh" => Ok(UnaryOp::Sinh),
        "cosh" => Ok(UnaryOp::Cosh),
        "tanh" => Ok(UnaryOp::Tanh),
        "asinh" => Ok(UnaryOp::Asinh),
        "acosh" => Ok(UnaryOp::Acosh),
        "atanh" => Ok(UnaryOp::Atanh),
        "exp" => Ok(UnaryOp::Exp),
        "exp2" => Ok(UnaryOp::Exp2),
        "exp10" => Ok(UnaryOp::Exp10),
        "log" => Ok(UnaryOp::Log),
        "log2" => Ok(UnaryOp::Log2),
        "log10" => Ok(UnaryOp::Log10),
        "cbrt" => Ok(UnaryOp::Cbrt),
        "significand" => Ok(UnaryOp::Significand),
        "exponent" => Ok(UnaryOp::Exponent),
        "logb" => Ok(UnaryOp::Logb),
        "nearbyint" => Ok(UnaryOp::NearbyInt),
        "trunc" => Ok(UnaryOp::Trunc),
        "rint" => Ok(UnaryOp::Rint),
        "j0" => Ok(UnaryOp::J0),
        "j1" => Ok(UnaryOp::J1),
        "keys" => Ok(UnaryOp::Keys),
        "keys_unsorted" => Ok(UnaryOp::KeysUnsorted),
        "values" => Ok(UnaryOp::Values),
        "sort" => Ok(UnaryOp::Sort),
        "reverse" => Ok(UnaryOp::Reverse),
        "unique" => Ok(UnaryOp::Unique),
        "flatten" => Ok(UnaryOp::Flatten),
        "min" => Ok(UnaryOp::Min),
        "max" => Ok(UnaryOp::Max),
        "add" => Ok(UnaryOp::Add),
        "any" => Ok(UnaryOp::Any),
        "all" => Ok(UnaryOp::All),
        "transpose" => Ok(UnaryOp::Transpose),
        "to_entries" => Ok(UnaryOp::ToEntries),
        "from_entries" => Ok(UnaryOp::FromEntries),
        "gmtime" => Ok(UnaryOp::Gmtime),
        "localtime" => Ok(UnaryOp::Localtime),
        "mktime" => Ok(UnaryOp::Mktime),
        "now" => Ok(UnaryOp::Now),
        "abs" => Ok(UnaryOp::Abs),
        "not" => Ok(UnaryOp::Not),
        "isinfinite" => Ok(UnaryOp::IsInfinite),
        "isnan" => Ok(UnaryOp::IsNan),
        "isnormal" => Ok(UnaryOp::IsNormal),
        "isfinite" => Ok(UnaryOp::IsFinite),
        _ => bail!("unknown unary operation: {}", name),
    }
}

/// Peephole optimization for Pipe(left, right).
/// - `[a, b] | add` → `a + b` (avoid array construction)
/// - `[a, b, c, ...] | add` → `a + b + c + ...`
///
/// `keys | length` / `keys_unsorted | length` → `length` was removed (#220):
/// the prefix op errors on non-iterable input, while bare `length` happily
/// returns 0/1/N, so the rewrite swallowed the type error.
fn optimize_pipe(left: Expr, right: Expr) -> Expr {
    use crate::ir::{UnaryOp, BinOp};
    // Check for Collect(...) | UnaryOp(Add)
    if let Expr::UnaryOp { op: UnaryOp::Add, operand } = &right {
        if matches!(operand.as_ref(), Expr::Input) {
            if let Expr::Collect { generator } = &left {
                // Extract comma-separated elements
                let mut elems = Vec::new();
                fn collect_comma(e: &Expr, out: &mut Vec<Expr>) {
                    match e {
                        Expr::Comma { left, right } => {
                            collect_comma(left, out);
                            collect_comma(right, out);
                        }
                        _ => out.push(e.clone()),
                    }
                }
                collect_comma(generator, &mut elems);
                // Only safe when every branch yields exactly one value. `Empty`
                // contributes nothing (`x + empty` collapses the whole output) and
                // multi-valued branches like `range`/`.[]`/`recurse` would stream
                // every value through the surrounding fold instead of being
                // collected first (issue #152).
                let all_single = elems.iter().all(crate::interpreter::is_single_valued_expr);
                if all_single && elems.len() >= 2 {
                    // Rewrite [a, b, c, ...] | add → a + b + c + ...
                    let mut result = elems.remove(0);
                    for elem in elems {
                        result = Expr::BinOp {
                            op: BinOp::Add,
                            lhs: Box::new(result),
                            rhs: Box::new(elem),
                        };
                    }
                    return result;
                }
            }
        }
    }
    Expr::Pipe { left: Box::new(left), right: Box::new(right) }
}
