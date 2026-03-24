//! Recursive descent parser for jq filter expressions.
//!
//! Parses jq filter strings directly into our IR (`Expr`), bypassing libjq's
//! bytecode entirely. This gives us full control over execution.

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
}

impl Scope {
    fn new() -> Self {
        Scope {
            vars: Vec::new(),
            funcs: Vec::new(),
            next_var: 0,
            compiled_funcs: Vec::new(),
        }
    }

    fn alloc_var(&mut self, name: &str) -> u16 {
        let idx = self.next_var;
        self.next_var += 1;
        self.vars.push((name.to_string(), idx));
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
        self.compiled_funcs.push(CompiledFunc {
            name: Some(name.to_string()),
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
}

impl Lexer {
    fn new(input: &str) -> Self {
        Lexer {
            chars: input.chars().collect(),
            pos: 0,
            tokens: Vec::new(),
        }
    }

    fn tokenize(&mut self) -> Result<Vec<Token>> {
        while self.pos < self.chars.len() {
            self.skip_whitespace_and_comments();
            if self.pos >= self.chars.len() { break; }

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
        }
        self.tokens.push(Token::Eof);
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
        // Preserve original string for numbers that lose precision in f64
        let f64_repr = crate::value::format_jq_number(n);
        let repr = if f64_repr != num_str {
            Some(Rc::from(normalize_num_repr(&num_str)))
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
        let first_sig = digits.iter().position(|c| *c != '0').unwrap_or(0);

        if first_sig >= digits.len() {
            return "0".to_string();
        }

        // Compute normalized exponent
        let new_exp = exp + (int_part.len() as i64) - 1 - (first_sig as i64);

        // Build normalized mantissa from significant digits
        let sig_digits: String = digits[first_sig..].iter().collect();
        let sig_digits = sig_digits.trim_end_matches('0');

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
                                // High surrogate, look for \uXXXX low surrogate
                                if self.pos + 5 < self.chars.len()
                                    && self.chars[self.pos] == '\\'
                                    && self.chars[self.pos + 1] == 'u'
                                {
                                    self.pos += 2;
                                    let hex2: String = self.chars[self.pos..self.pos+4].iter().collect();
                                    self.pos += 4;
                                    let cp2 = u32::from_str_radix(&hex2, 16)
                                        .map_err(|_| anyhow::anyhow!("invalid unicode escape"))?;
                                    if (0xDC00..=0xDFFF).contains(&cp2) {
                                        let combined = ((cp - 0xD800) << 10) + (cp2 - 0xDC00) + 0x10000;
                                        if let Some(c) = char::from_u32(combined) {
                                            current.push(c);
                                        }
                                    }
                                }
                            } else if let Some(c) = char::from_u32(cp) {
                                current.push(c);
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
    pos: usize,
    scope: Scope,
    lib_dirs: Vec<String>,
}

/// Result of parsing: expression + compiled functions.
pub struct ParseResult {
    pub expr: Expr,
    pub funcs: Vec<CompiledFunc>,
}

impl Parser {
    pub fn parse(input: &str) -> Result<ParseResult> {
        Self::parse_with_libs(input, &[])
    }

    pub fn parse_with_libs(input: &str, lib_dirs: &[String]) -> Result<ParseResult> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser {
            tokens,
            pos: 0,
            scope: Scope::new(),
            lib_dirs: lib_dirs.to_vec(),
        };

        // Pre-register $ENV
        let _env_idx = parser.scope.alloc_var("ENV");

        let expr = parser.parse_program()?;
        if !parser.at_eof() {
            bail!("unexpected token {:?} at position {}", parser.current(), parser.pos);
        }
        Ok(ParseResult {
            expr,
            funcs: parser.scope.compiled_funcs,
        })
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
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
        let mut body = self.parse_pipe()?;
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
            pos: 0,
            scope: mod_scope,
            lib_dirs: mod_lib_dirs,
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
        let mut func_id_map: Vec<(usize, usize)> = Vec::new(); // (old, new)

        // First pass: register all functions with placeholder bodies to get new func_ids
        for (name, mod_func_id, nargs) in &mod_parser.scope.funcs {
            let func = &mod_parser.scope.compiled_funcs[*mod_func_id];
            let new_name = if namespace.is_empty() {
                name.clone()
            } else {
                format!("{}::{}", namespace, name)
            };
            let new_id = self.scope.define_func(&new_name, *nargs, Expr::Empty, func.param_vars.clone());
            func_id_map.push((*mod_func_id, new_id));
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
        let mut expr = self.parse_assign()?;

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
            // No ?// alternatives - simple binding
            let pattern = alt_patterns.into_iter().next().unwrap();
            let allocs = self.alloc_pattern_vars(&pattern);
            self.expect(&Token::Pipe)?;
            let body = self.parse_pipe()?;
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

        // Allocate once for shared variables
        let mut var_map: std::collections::HashMap<String, u16> = std::collections::HashMap::new();
        for name in &unique_vars {
            let idx = self.scope.alloc_var(name);
            var_map.insert(name.clone(), idx);
        }

        self.expect(&Token::Pipe)?;
        let body = self.parse_pipe()?;

        // Build: try (bind pattern1 | body) catch try (bind pattern2 | body) catch ... catch empty
        let tmp_idx = self.scope.alloc_var("__altdestruct_tmp__");
        let tmp_ref = Expr::LoadVar { var_index: tmp_idx };

        let mut result = Expr::Empty; // final fallback: empty
        for pattern in alt_patterns.into_iter().rev() {
            let binding = self.build_binding_with_varmap(tmp_ref.clone(), &pattern, &var_map, body.clone())?;
            result = Expr::TryCatch {
                try_expr: Box::new(binding),
                catch_expr: Box::new(result),
            };
        }

        Ok(Expr::LetBinding {
            var_index: tmp_idx,
            value: Box::new(value_expr),
            body: Box::new(result),
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

    fn build_object_destructure_varmap(&mut self, value: Expr, pats: &[(String, Pattern)], var_map: &std::collections::HashMap<String, u16>, body: Expr) -> Result<Expr> {
        let mut result = body;
        for (key, pat) in pats.iter().rev() {
            let field_expr = Expr::Index {
                expr: Box::new(value.clone()),
                key: Box::new(Expr::Literal(Literal::Str(key.clone()))),
            };
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
        match pattern {
            Pattern::Var(name) => {
                vec![self.scope.alloc_var(name)]
            }
            Pattern::Array(pats) => {
                pats.iter().flat_map(|p| self.alloc_pattern_vars(p)).collect()
            }
            Pattern::Object(pats) => {
                pats.iter().flat_map(|(_, p)| self.alloc_pattern_vars(p)).collect()
            }
            Pattern::VarAndSub(name, sub) => {
                let mut vars = vec![self.scope.alloc_var(name)];
                vars.extend(self.alloc_pattern_vars(sub));
                vars
            }
        }
    }

    fn parse_pattern(&mut self) -> Result<Pattern> {
        match self.current().clone() {
            Token::Variable(name) => {
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

    fn parse_obj_pattern_pair(&mut self) -> Result<(String, Pattern)> {
        match self.current().clone() {
            Token::Variable(name) => {
                self.advance();
                if self.eat(&Token::Colon) {
                    // $var: pattern — key is variable name, bind $var AND destructure
                    let sub_pat = self.parse_pattern()?;
                    Ok((name.clone(), Pattern::VarAndSub(name, Box::new(sub_pat))))
                } else {
                    // $var shorthand — key is variable name, bind to $var
                    Ok((name.clone(), Pattern::Var(name)))
                }
            }
            Token::Ident(key) | Token::Str(key) => {
                self.advance();
                self.expect(&Token::Colon)?;
                let pat = self.parse_pattern()?;
                Ok((key, pat))
            }
            Token::LParen => {
                // Computed key pattern: (expr): $var
                self.advance();
                let key_expr = self.parse_pipe()?;
                self.expect(&Token::RParen)?;
                self.expect(&Token::Colon)?;
                let pat = self.parse_pattern()?;
                // Try to evaluate key as a constant string
                let key = self.try_eval_const_string(&key_expr)?;
                Ok((key, pat))
            }
            ref tok if Self::keyword_as_string(tok).is_some() => {
                let key = Self::keyword_as_string(tok).unwrap().to_string();
                self.advance();
                self.expect(&Token::Colon)?;
                let pat = self.parse_pattern()?;
                Ok((key, pat))
            }
            _ => bail!("expected object pattern key, got {:?}", self.current()),
        }
    }

    /// Try to evaluate a constant string expression at parse time.
    fn try_eval_const_string(&self, expr: &Expr) -> Result<String> {
        match expr {
            Expr::Literal(Literal::Str(s)) => Ok(s.clone()),
            Expr::BinOp { op: BinOp::Add, lhs, rhs } => {
                let l = self.try_eval_const_string(lhs)?;
                let r = self.try_eval_const_string(rhs)?;
                Ok(format!("{}{}", l, r))
            }
            _ => bail!("cannot evaluate computed key pattern at parse time"),
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

    fn build_object_destructure(&mut self, value: Expr, pats: &[(String, Pattern)], allocs: &[u16], body: Expr) -> Result<Expr> {
        let mut result = body;
        let mut alloc_idx = allocs.len();
        for (key, pat) in pats.iter().rev() {
            let count = self.count_pattern_vars(pat);
            alloc_idx -= count;
            let field_expr = Expr::Index {
                expr: Box::new(value.clone()),
                key: Box::new(Expr::Literal(Literal::Str(key.clone()))),
            };
            result = self.build_pattern_binding(pat, field_expr, &allocs[alloc_idx..alloc_idx+count], result)?;
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
        let mut expr = self.parse_assign()?;
        while self.eat(&Token::Comma) {
            let right = self.parse_assign()?;
            expr = Expr::Comma {
                left: Box::new(expr),
                right: Box::new(right),
            };
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
        let mut expr = self.parse_alt()?;
        // Check for ?// (alternative destructuring)
        while self.at(&Token::AltDestructure) {
            self.advance();
            let right = self.parse_alt()?;
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
            let right = self.parse_alt()?;
            expr = Expr::BinOp {
                op,
                lhs: Box::new(expr),
                rhs: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_alt(&mut self) -> Result<Expr> {
        let mut expr = self.parse_add()?;
        while self.eat(&Token::Alt) {
            let right = self.parse_add()?;
            expr = Expr::Alternative {
                primary: Box::new(expr),
                fallback: Box::new(right),
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
            let operand = self.parse_postfix()?;
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
                // .. is recurse(. ; .[]?)
                // Can optionally be followed by .field
                Ok(Expr::Recurse { input_expr: Box::new(Expr::Input) })
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
                        let var_idx = self.scope.alloc_var(&name);
                        self.expect(&Token::Pipe)?;
                        let body = self.parse_pipe()?;
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
                        let var_idx = self.scope.lookup_var(&name)
                            .ok_or_else(|| anyhow::anyhow!("undefined label variable ${}", name))?;
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
                    Ok(Expr::Loc { file: "<top-level>".to_string(), line: 1 })
                } else if name == "ENV" {
                    Ok(Expr::Env)
                } else {
                    match self.scope.lookup_var(&name) {
                        Some(idx) => Ok(Expr::LoadVar { var_index: idx }),
                        None => {
                            let idx = self.scope.alloc_var(&name);
                            Ok(Expr::LoadVar { var_index: idx })
                        }
                    }
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
                self.advance();
                if self.eat(&Token::Colon) {
                    let val = self.parse_pipe_nocomma()?;
                    // $var: value — key is the variable's value converted to string
                    let idx = self.scope.lookup_var(&name).unwrap_or(0);
                    let key_expr = Expr::LoadVar { var_index: idx };
                    Ok((key_expr, val))
                } else {
                    // Shorthand: {$x} = {($x|tostring): $x}
                    let val_expr = if name == "__loc__" {
                        Expr::Loc { file: "<top-level>".to_string(), line: 1 }
                    } else if name == "ENV" {
                        Expr::Env
                    } else {
                        let idx = self.scope.lookup_var(&name).unwrap_or(0);
                        Expr::LoadVar { var_index: idx }
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
        // reduce SOURCE as $VAR|PATTERN (INIT; UPDATE)
        let source = self.parse_or()?;
        self.expect(&Token::As)?;

        if matches!(self.current(), Token::Variable(_)) && !matches!(self.peek(), Token::Comma | Token::Colon) {
            // Simple variable binding
            let var_name = match self.advance() {
                Token::Variable(name) => name,
                _ => unreachable!(),
            };
            let var_idx = self.scope.alloc_var(&var_name);
            let acc_idx = self.scope.alloc_var("__acc__");

            self.expect(&Token::LParen)?;
            let init = self.parse_pipe()?;
            self.expect(&Token::Semicolon)?;
            let update = self.parse_pipe()?;
            self.expect(&Token::RParen)?;

            Ok(Expr::Reduce {
                source: Box::new(source),
                init: Box::new(init),
                var_index: var_idx,
                acc_index: acc_idx,
                update: Box::new(update),
            })
        } else {
            // Destructuring pattern
            let pattern = self.parse_pattern()?;
            let allocs = self.alloc_pattern_vars(&pattern);
            let tmp_var = self.scope.alloc_var("__reduce_item__");
            let acc_idx = self.scope.alloc_var("__acc__");

            self.expect(&Token::LParen)?;
            let init = self.parse_pipe()?;
            self.expect(&Token::Semicolon)?;
            let update_raw = self.parse_pipe()?;
            self.expect(&Token::RParen)?;

            let update = self.build_binding(
                Expr::LoadVar { var_index: tmp_var },
                pattern,
                allocs,
                update_raw,
            )?;

            Ok(Expr::Reduce {
                source: Box::new(source),
                init: Box::new(init),
                var_index: tmp_var,
                acc_index: acc_idx,
                update: Box::new(update),
            })
        }
    }

    fn parse_foreach(&mut self) -> Result<Expr> {
        // foreach SOURCE as $VAR|PATTERN (INIT; UPDATE [; EXTRACT])
        let source = self.parse_or()?;
        self.expect(&Token::As)?;

        // Support both $var and destructuring patterns
        if matches!(self.current(), Token::Variable(_)) && !matches!(self.peek(), Token::Comma | Token::Colon) {
            // Simple variable binding
            let var_name = match self.advance() {
                Token::Variable(name) => name,
                _ => unreachable!(),
            };
            let var_idx = self.scope.alloc_var(&var_name);
            let acc_idx = self.scope.alloc_var("__acc__");

            self.expect(&Token::LParen)?;
            let init = self.parse_pipe()?;
            self.expect(&Token::Semicolon)?;
            let update = self.parse_pipe()?;
            let extract = if self.eat(&Token::Semicolon) {
                Some(Box::new(self.parse_pipe()?))
            } else {
                None
            };
            self.expect(&Token::RParen)?;

            Ok(Expr::Foreach {
                source: Box::new(source),
                init: Box::new(init),
                var_index: var_idx,
                acc_index: acc_idx,
                update: Box::new(update),
                extract,
            })
        } else {
            // Destructuring pattern
            let pattern = self.parse_pattern()?;
            let allocs = self.alloc_pattern_vars(&pattern);
            let tmp_var = self.scope.alloc_var("__foreach_item__");
            let acc_idx = self.scope.alloc_var("__acc__");

            self.expect(&Token::LParen)?;
            let init = self.parse_pipe()?;
            self.expect(&Token::Semicolon)?;
            let update_raw = self.parse_pipe()?;
            let extract_raw = if self.eat(&Token::Semicolon) {
                Some(self.parse_pipe()?)
            } else {
                None
            };
            self.expect(&Token::RParen)?;

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

            Ok(Expr::Foreach {
                source: Box::new(source),
                init: Box::new(init),
                var_index: tmp_var,
                acc_index: acc_idx,
                update: Box::new(update),
                extract,
            })
        }
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
            | "ascii" | "explode" | "implode"
            | "ascii_downcase" | "ascii_upcase" | "ltrim" | "rtrim" | "trim"
            | "floor" | "ceil" | "round" | "fabs" | "sqrt"
            | "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
            | "exp" | "exp2" | "exp10" | "log" | "log2" | "log10"
            | "cbrt" | "significand" | "exponent" | "logb"
            | "nearbyint" | "trunc" | "rint" | "j0" | "j1"
            | "keys" | "keys_unsorted" | "values" | "sort" | "reverse"
            | "unique" | "flatten" | "min" | "max" | "add" | "any" | "all"
            | "transpose" | "to_entries" | "from_entries"
            | "gmtime" | "mktime" | "now" | "abs"
            | "not" | "env" | "builtins" | "input" | "inputs"
            | "debug" | "stderr" | "modulemeta" | "path"
            | "with_entries" | "recurse" | "recurse_down" | "leaf_paths"
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

    fn compile_builtin_noargs(&self, name: &str) -> Result<Expr> {
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
                // paths = [path(..[])] but without the empty root path
                // Use Recurse with .[] step and then get paths via Each
                Ok(Expr::Pipe {
                    left: Box::new(Expr::PathExpr {
                        expr: Box::new(Expr::Recurse { input_expr: Box::new(Expr::Input) }),
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
            "leaf_paths" => {
                // leaf_paths = paths(scalars)
                // scalars = select(type != "array" and type != "object")
                let scalars_cond = Expr::BinOp {
                    op: BinOp::And,
                    lhs: Box::new(Expr::BinOp {
                        op: BinOp::Ne,
                        lhs: Box::new(Expr::UnaryOp { op: UnaryOp::Type, operand: Box::new(Expr::Input) }),
                        rhs: Box::new(Expr::Literal(Literal::Str("array".to_string()))),
                    }),
                    rhs: Box::new(Expr::BinOp {
                        op: BinOp::Ne,
                        lhs: Box::new(Expr::UnaryOp { op: UnaryOp::Type, operand: Box::new(Expr::Input) }),
                        rhs: Box::new(Expr::Literal(Literal::Str("object".to_string()))),
                    }),
                };
                Ok(Expr::PathExpr {
                    expr: Box::new(Expr::Pipe {
                        left: Box::new(Expr::Recurse { input_expr: Box::new(Expr::Input) }),
                        right: Box::new(Expr::IfThenElse {
                            cond: Box::new(scalars_cond),
                            then_branch: Box::new(Expr::Input),
                            else_branch: Box::new(Expr::Empty),
                        }),
                    }),
                })
            }
            "recurse" | "recurse_down" => {
                Ok(Expr::Recurse { input_expr: Box::new(Expr::Input) })
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
            "fromcsv" | "fromtsv" | "fromcsvh" | "fromtsvh" => {
                Ok(Expr::CallBuiltin { name: name.to_string(), args: vec![] })
            }
            _ => {
                // Check user-defined functions
                if let Some(func_id) = self.scope.lookup_func(name, 0) {
                    Ok(Expr::FuncCall { func_id, args: vec![] })
                } else {
                    // Check for filter parameter reference (bare identifier like `x` in `def f(x):`)
                    let fparam_name = format!("\x00fparam:{}", name);
                    if let Some(var_idx) = self.scope.lookup_var(&fparam_name) {
                        Ok(Expr::LoadVar { var_index: var_idx })
                    } else {
                        // Treat as a 0-arg builtin via runtime
                        Ok(Expr::UnaryOp {
                            op: name_to_unary_op(name)?,
                            operand: Box::new(Expr::Input),
                        })
                    }
                }
            }
        }
    }

    fn compile_funcall(&mut self, name: &str, args: Vec<Expr>) -> Result<Expr> {
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
            ("recurse", 2) => {
                let mut args = args.into_iter();
                let f = args.next().unwrap();
                let cond = args.next().unwrap();
                // recurse(f; cond) = def r: ., (select(cond) | f | r); r
                Ok(Expr::Recurse {
                    input_expr: Box::new(Expr::Pipe {
                        left: Box::new(Expr::IfThenElse {
                            cond: Box::new(cond),
                            then_branch: Box::new(Expr::Input),
                            else_branch: Box::new(Expr::Empty),
                        }),
                        right: Box::new(f),
                    }),
                })
            }
            ("path", 1) => {
                let f = args.into_iter().next().unwrap();
                Ok(Expr::PathExpr { expr: Box::new(f) })
            }
            ("paths", 1) => {
                let f = args.into_iter().next().unwrap();
                // paths(f) = path(recurse | select(f))... actually it's more complex
                // paths(node_filter) outputs paths to nodes matching filter
                Ok(Expr::PathExpr {
                    expr: Box::new(Expr::Pipe {
                        left: Box::new(Expr::Recurse { input_expr: Box::new(Expr::Input) }),
                        right: Box::new(Expr::IfThenElse {
                            cond: Box::new(f),
                            then_branch: Box::new(Expr::Input),
                            else_branch: Box::new(Expr::Empty),
                        }),
                    }),
                })
            }
            ("getpath", 1) => {
                let path = args.into_iter().next().unwrap();
                // Optimize getpath(["key1","key2",...]) → .key1.key2...
                if let Some(keys) = extract_literal_str_array(&path) {
                    if !keys.is_empty() {
                        let mut expr = Expr::Input;
                        for key in keys {
                            expr = Expr::Index {
                                expr: Box::new(expr),
                                key: Box::new(Expr::Literal(Literal::Str(key))),
                            };
                        }
                        return Ok(expr);
                    }
                }
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
                let code = args.into_iter().next().unwrap();
                Ok(Expr::Error { msg: Some(Box::new(code)) })
            }
            ("pow", 2) | ("atan2", 2) | ("fma", 3)
            | ("remainder", 2) | ("hypot", 2) | ("ldexp", 2)
            | ("scalb", 2) | ("scalbln", 2) => {
                Ok(Expr::CallBuiltin { name: name.to_string(), args })
            }
            ("nth", 1) | ("nth", 2) => {
                let mut args = args.into_iter();
                let n_expr = args.next().unwrap();
                let generator = args.next().unwrap_or(Expr::Each { input_expr: Box::new(Expr::Input) });
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
                            rhs: Box::new(Expr::LoadVar { var_index: n_var }),
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
            ("strftime", 1) | ("strptime", 1) | ("dateadd", 2) | ("datesub", 2)
            | ("todate", 0) | ("fromdate", 0) | ("date", 0) => {
                Ok(Expr::CallBuiltin { name: name.to_string(), args })
            }
            ("input", 0) => Ok(Expr::ReadInput),
            ("inputs", 0) => Ok(Expr::ReadInputs),
            ("genlabel", 0) => Ok(Expr::GenLabel),
            ("format", 1) => {
                let fmt = args.into_iter().next().unwrap();
                Ok(Expr::Format {
                    name: "text".to_string(),
                    expr: Box::new(fmt),
                })
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
            // IN/2: IN(s; b) = reduce s as $x (false; . or ($x | IN(b)))
            ("IN", 2) => {
                let mut args = args.into_iter();
                let s = args.next().unwrap();
                let b = args.next().unwrap();
                let x_var = self.scope.alloc_var("__in2_x__");
                let acc_var = self.scope.alloc_var("__in2_acc__");
                let check_var = self.scope.alloc_var("__in2_chk__");
                // IN($x; b) = $x | IN(b) = $x as $chk | first(b | if . == $chk then true else empty end) // false
                let in_check = Expr::LetBinding {
                    var_index: check_var,
                    value: Box::new(Expr::LoadVar { var_index: x_var }),
                    body: Box::new(Expr::Alternative {
                        primary: Box::new(Expr::Limit {
                            count: Box::new(Expr::Literal(Literal::Num(1.0, None))),
                            generator: Box::new(Expr::Pipe {
                                left: Box::new(b),
                                right: Box::new(Expr::IfThenElse {
                                    cond: Box::new(Expr::BinOp {
                                        op: BinOp::Eq,
                                        lhs: Box::new(Expr::Input),
                                        rhs: Box::new(Expr::LoadVar { var_index: check_var }),
                                    }),
                                    then_branch: Box::new(Expr::Literal(Literal::True)),
                                    else_branch: Box::new(Expr::Empty),
                                }),
                            }),
                        }),
                        fallback: Box::new(Expr::Literal(Literal::False)),
                    }),
                };
                // reduce s as $x (false; . or in_check)
                Ok(Expr::Reduce {
                    source: Box::new(s),
                    init: Box::new(Expr::Literal(Literal::False)),
                    var_index: x_var,
                    acc_index: acc_var,
                    update: Box::new(Expr::BinOp {
                        op: BinOp::Or,
                        lhs: Box::new(Expr::Input),
                        rhs: Box::new(in_check),
                    }),
                })
            }
            // INDEX/2: INDEX(stream; f) = reduce stream as $x ({}; . + {($x|f|tostring): $x})
            ("INDEX", 2) => {
                let mut args = args.into_iter();
                let stream = args.next().unwrap();
                let f = args.next().unwrap();
                let x_var = self.scope.alloc_var("__idx_x__");
                let acc_var = self.scope.alloc_var("__idx_acc__");
                Ok(Expr::Reduce {
                    source: Box::new(stream),
                    init: Box::new(Expr::ObjectConstruct { pairs: vec![] }),
                    var_index: x_var,
                    acc_index: acc_var,
                    update: Box::new(Expr::BinOp {
                        op: BinOp::Add,
                        lhs: Box::new(Expr::Input),
                        rhs: Box::new(Expr::ObjectConstruct {
                            pairs: vec![(
                                Expr::Pipe {
                                    left: Box::new(Expr::LoadVar { var_index: x_var }),
                                    right: Box::new(Expr::Pipe {
                                        left: Box::new(f),
                                        right: Box::new(Expr::UnaryOp { op: UnaryOp::ToString, operand: Box::new(Expr::Input) }),
                                    }),
                                },
                                Expr::LoadVar { var_index: x_var },
                            )],
                        }),
                    }),
                })
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
            // del/1: del(f) — delegate to eval for proper slice handling
            ("del", 1) => {
                let f = args.into_iter().next().unwrap();
                Ok(Expr::CallBuiltin { name: "del".to_string(), args: vec![f] })
            }
            _ => {
                // Check user-defined functions
                if let Some(func_id) = self.scope.lookup_func(name, args.len()) {
                    Ok(Expr::FuncCall { func_id, args })
                } else {
                    bail!("unknown function '{}' with {} args", name, args.len())
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
    Object(Vec<(String, Pattern)>),
    /// $var: sub_pattern — binds $var to whole value AND destructures via sub_pattern
    VarAndSub(String, Box<Pattern>),
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
        Expr::TryCatch { try_expr, catch_expr } => Expr::TryCatch {
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
        "mktime" => Ok(UnaryOp::Mktime),
        "now" => Ok(UnaryOp::Now),
        "abs" => Ok(UnaryOp::Abs),
        "not" => Ok(UnaryOp::Not),
        "isinfinite" => Ok(UnaryOp::IsInfinite),
        "isnan" => Ok(UnaryOp::IsNan),
        "isnormal" => Ok(UnaryOp::IsNormal),
        "isfinite" => Ok(UnaryOp::IsFinite),
        "ascii" => Ok(UnaryOp::Ascii),
        _ => bail!("unknown unary operation: {}", name),
    }
}

/// Extract a literal string array from an expression like `Collect(Comma("a", "b", ...))`.
/// Returns None if the expression isn't a constant string array.
fn extract_literal_str_array(expr: &Expr) -> Option<Vec<String>> {
    use crate::ir::Literal;
    match expr {
        Expr::Collect { generator } => {
            let mut keys = Vec::new();
            fn collect_strings(e: &Expr, out: &mut Vec<String>) -> bool {
                match e {
                    Expr::Literal(Literal::Str(s)) => { out.push(s.clone()); true }
                    Expr::Comma { left, right } => collect_strings(left, out) && collect_strings(right, out),
                    _ => false,
                }
            }
            if collect_strings(generator, &mut keys) {
                Some(keys)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Peephole optimization for Pipe(left, right).
/// - `keys | length` → `length` (same result for arrays and objects)
/// - `[a, b] | add` → `a + b` (avoid array construction)
/// - `[a, b, c, ...] | add` → `a + b + c + ...`
fn optimize_pipe(left: Expr, right: Expr) -> Expr {
    use crate::ir::{UnaryOp, BinOp};
    // keys | length → length (keys_unsorted | length too)
    if let Expr::UnaryOp { op: UnaryOp::Length, operand } = &right {
        if matches!(operand.as_ref(), Expr::Input) {
            if let Expr::UnaryOp { op: UnaryOp::Keys | UnaryOp::KeysUnsorted, operand: inner } = &left {
                if matches!(inner.as_ref(), Expr::Input) {
                    return Expr::UnaryOp { op: UnaryOp::Length, operand: Box::new(Expr::Input) };
                }
            }
        }
    }
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
                if elems.len() >= 2 {
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
