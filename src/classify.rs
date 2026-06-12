//! Fast-path classification: the shape enums (`RemapExpr`, `ArithExpr`,
//! `CondBranch`, ...) and the `detect_*` / `classify_*` / `is_*` methods on
//! [`Filter`] that recognize raw-byte fast-path candidates. Extracted from
//! `src/interpreter.rs` (#1029); pure code motion, no behavior change.
//!
//! The classifiers run against [`Filter::detect_expr`] (the simplified IR)
//! and return shape descriptors consumed by the raw-byte executors in
//! `src/bin/jq-jit.rs` and the typed paths in [`crate::fast_path`]. The
//! shape types are re-exported from [`crate::interpreter`] for existing
//! callers. See docs/maintenance.md § Fast path map for the catalog and
//! the invariants each shape must preserve.

use crate::interpreter::Filter;
use crate::ir::BuiltinOp;
use crate::simplify::contains_input;

/// Comparison value: either numeric or string.
#[derive(Debug, Clone)]
pub enum CmpVal {
    Num(f64),
    Str(String),
}

/// String function condition for if-then-else patterns.
pub enum StrFuncCond {
    Test(String, Option<String>), // regex pattern, optional flags
    Startswith(String),
    Endswith(String),
    Contains(String),
}

/// Describes how to compute one value in a computed remap fast path.
#[derive(Debug, Clone)]
pub enum RemapExpr {
    /// `.field` — raw byte copy
    Field(String),
    /// `.field op N` — numeric arithmetic with constant
    FieldOpConst(String, crate::ir::BinOp, f64),
    /// `.field1 op .field2` — numeric arithmetic with two fields
    FieldOpField(String, crate::ir::BinOp, String),
    /// `N op .field` — constant op field (e.g., `100 - .x`)
    ConstOpField(f64, crate::ir::BinOp, String),
    /// `.field cmp N` — boolean comparison result (true/false)
    FieldCmpConst(String, crate::ir::BinOp, f64),
    /// `.field1 cmp .field2` — boolean comparison between two fields
    FieldCmpField(String, crate::ir::BinOp, String),
    /// `.field | tostring` — convert field value to string
    FieldToString(String),
    /// `.field op N | tostring` — arithmetic then tostring
    FieldOpConstToString(String, crate::ir::BinOp, f64),
    /// Compound arithmetic expression over fields and constants.
    /// ArithExpr::Field(i) indexes into Vec<String> field names.
    Arith(ArithExpr, Vec<String>),
    /// `[.field1, .field2] | min` or `| max`
    FieldMinMax(String, String, bool), // true=max, false=min
    /// Pre-serialized JSON literal bytes (e.g., `"str"`, `123`, `null`)
    LiteralJson(Vec<u8>),
    /// `.field | length` — string length or array length
    FieldLength(String),
    /// String interpolation: `"\(.x):\(.y)"` — parts are literals and field references
    StringInterp(Vec<InterpPart>),
    /// `.field | split(sep) | join(rep)` — string replacement
    FieldSplitJoin(String, String, String), // field, split_sep, join_rep
    /// `if .field cmp N then A elif ... else B end` — conditional chain
    CondChain(Vec<CondBranch>, Box<BranchOutput>),
    /// `.field | ascii_upcase` or `.field | ascii_downcase`
    FieldStringCase(String, bool), // field, is_upper
    /// `.field | split(sep) | length` — count split segments
    FieldSplitLength(String, String), // field, separator
    /// `.field | builtin("arg")` — string builtin with one argument
    FieldStrBuiltin(String, StrBuiltin, String), // field, op, arg
    /// `.field | split(sep) | .[N]` — split then index
    FieldSplitIndex(String, String, i32), // field, separator, index
    /// `(.f1 op .f2) | tostring` — field op field then tostring
    FieldOpFieldToString(String, crate::ir::BinOp, String), // f1, op, f2
    /// `(arith_expr) | tostring` — compound arithmetic then tostring
    ArithToString(ArithExpr, Vec<String>),
    /// `(arith_expr) | math_unary` — compound arithmetic then sqrt/floor/ceil
    ArithUnary(MathUnary, ArithExpr, Vec<String>),
    /// `.field | .[from:to]` — string slice
    FieldSlice(String, Option<i64>, Option<i64>), // field, from, to
    /// `[expr1, expr2, ...]` — array of remap expressions
    FieldArray(Vec<RemapExpr>),
    /// `(cmp1) and/or (cmp2)` — boolean expression combining comparisons
    BoolExpr(Box<RemapExpr>, crate::ir::BinOp, Box<RemapExpr>), // lhs, And/Or, rhs
    /// `.field | type` — JSON type string
    FieldType(String),
    /// `-.field` — negation of a field
    FieldNegate(String),
    /// `(arith) cmp N` — compare compound arithmetic result to constant, emit true/false
    ArithCmp(ArithExpr, crate::ir::BinOp, f64, Vec<String>),
    /// `.name + ":" + (.x | tostring)` — string add chain
    StringChain(Vec<StringAddPart>),
}

/// Math unary operation for ArithUnary.
#[derive(Debug, Clone, Copy)]
pub enum MathUnary {
    Sqrt, Floor, Ceil, Fabs, Round,
}

/// String builtin for FieldStrBuiltin.
#[derive(Debug, Clone, Copy)]
pub enum StrBuiltin {
    Ltrimstr,
    Rtrimstr,
    Startswith,
    Endswith,
    Index,
    Contains,
}

/// Part of a string interpolation for raw byte remap emission.
#[derive(Debug, Clone)]
pub enum InterpPart {
    /// Literal text (decoded string content, needs JSON-escaping on output)
    Literal(String),
    /// `.field` — emit tostring of field value
    Field(String),
}

/// A pure numeric expression over fields and constants.
/// Used for raw byte fast path evaluation.
#[derive(Debug, Clone)]
pub enum ArithExpr {
    /// `.field` — extract numeric field
    Field(usize), // index into field list
    /// Numeric constant
    Const(f64),
    /// Binary operation
    BinOp(crate::ir::BinOp, Box<ArithExpr>, Box<ArithExpr>),
    /// Unary math operation (floor, ceil, sqrt, fabs, round)
    Unary(MathUnary, Box<ArithExpr>),
}

impl ArithExpr {
    pub fn eval(&self, fields: &[f64]) -> f64 {
        match self {
            ArithExpr::Field(idx) => fields[*idx],
            ArithExpr::Const(n) => *n,
            ArithExpr::BinOp(op, lhs, rhs) => {
                let l = lhs.eval(fields);
                let r = rhs.eval(fields);
                use crate::ir::BinOp;
                match op {
                    BinOp::Add => l + r,
                    BinOp::Sub => l - r,
                    BinOp::Mul => l * r,
                    BinOp::Div => l / r,
                    BinOp::Mod => crate::runtime::jq_mod_f64(l, r).unwrap_or(f64::NAN),
                    _ => unreachable!(),
                }
            }
            ArithExpr::Unary(op, operand) => {
                let v = operand.eval(fields);
                match op {
                    MathUnary::Floor => v.floor(),
                    MathUnary::Ceil => v.ceil(),
                    MathUnary::Sqrt => v.sqrt(),
                    MathUnary::Fabs => v.abs(),
                    MathUnary::Round => v.round(),
                }
            }
        }
    }
}

/// Output of a conditional branch — either a literal or a field access.
#[derive(Debug, Clone)]
pub enum BranchOutput {
    /// Pre-serialized JSON literal bytes
    Literal(Vec<u8>),
    /// `.field` — extract raw bytes from input
    Field(String),
    /// `empty` — produce no output
    Empty,
    /// `{key: value, ...}` — object construction with computed values
    Remap(Vec<(String, RemapExpr)>),
    /// Computed value (e.g., `.x - .y`)
    Computed(RemapExpr),
}

/// Right-hand side of a condition: either a constant or a field reference.
#[derive(Debug, Clone)]
pub enum CondRhs {
    Const(f64),
    Field(String),
    Str(String),
    Null,
    Bool(bool),
    /// `.field | startswith("str")` — true if field starts with str
    Startswith(String),
    /// `.field | endswith("str")` — true if field ends with str
    Endswith(String),
    /// `.field | contains("str")` — true if field contains str
    Contains(String),
    /// `.field | test("regex")` — true if field matches regex
    Test(String),
}

/// One branch in a conditional chain: if .field [arith_ops...] cmp (N | .field2) then output.
#[derive(Debug, Clone)]
pub struct CondBranch {
    pub cond_field: String,
    /// Arithmetic ops applied to field value before comparison (e.g., % 2 for modulo).
    pub cond_arith_ops: Vec<(crate::ir::BinOp, f64)>,
    pub cond_op: crate::ir::BinOp,
    pub cond_rhs: CondRhs,
    pub output: BranchOutput,
}

/// Condition type for if-then-else with array outputs.
#[derive(Debug, Clone)]
pub enum IfArrayCond {
    /// `.field cmp N`
    FieldConst(String, crate::ir::BinOp, f64),
    /// `.field1 cmp .field2`
    FieldField(String, crate::ir::BinOp, String),
}

/// One step in a chained string operation: `.field | op1 | op2 | ...`
#[derive(Debug, Clone)]
pub enum StringChainOp {
    AsciiDowncase,
    AsciiUpcase,
    Ltrimstr(String),
    Rtrimstr(String),
    /// `split(sep) | join(rep)` fused as a single op (string replace)
    SplitJoin(String, String),
    /// `split(sep) | reverse | join(rep)` fused
    SplitReverseJoin(String, String),
}

/// Terminal operation at the end of a string chain (returns bool or length, not string).
#[derive(Debug, Clone)]
pub enum StringChainTerminal {
    /// No terminal — output is a string
    None,
    /// startswith("str")
    Startswith(String),
    /// endswith("str")
    Endswith(String),
    /// contains("str")
    Contains(String),
    /// length
    Length,
    /// index("str") — first occurrence position (UTF-8 codepoints), or null
    Index(String),
}

/// Part of a string Add-chain: `.name + ": " + (.x | tostring)`.
#[derive(Debug, Clone)]
pub enum StringAddPart {
    /// Literal string
    Literal(String),
    /// `.field` — string field, raw bytes copy
    Field(String),
    /// `.field | tostring` — numeric field, format as string
    FieldToString(String),
    /// `.field * N + M ... | tostring` — arithmetic chain then tostring
    FieldArithToString(String, Vec<(crate::ir::BinOp, f64)>),
}

/// Part of a split-then-concat pattern: `.field | split(sep) | .[i] + "lit" + .[j]`
#[derive(Debug, Clone)]
pub enum SplitConcatPart {
    /// `.[N]` — index into split result
    Index(i32),
    /// `"literal"` — literal string
    Lit(String),
}

/// Step in a numeric chain update: `.field |= (. * 100 | floor | . / 100)`.
#[derive(Debug, Clone)]
pub enum NumChainStep {
    Arith(crate::ir::BinOp, f64),
    Unary(crate::ir::UnaryOp),
}

/// A single condition in a mixed compound select.
pub enum MixedCond {
    /// .field cmp N (numeric comparison)
    NumCmp(String, crate::ir::BinOp, f64),
    /// .field | str_op("arg") (startswith/endswith/contains/test/eq)
    StrTest(String, String, String),
}

/// Extract string function condition from an expression (test/startswith/endswith/contains on Input).
fn extract_strfunc_cond(expr: &crate::ir::Expr) -> Option<StrFuncCond> {
    use crate::ir::{Expr, Literal};
    match expr {
        Expr::RegexTest { input_expr, re, flags } => {
            if !matches!(input_expr.as_ref(), Expr::Input) { return None; }
            if let Expr::Literal(Literal::Str(pattern)) = re.as_ref() {
                let flags_str = match flags.as_ref() {
                    Expr::Literal(Literal::Null) => None,
                    Expr::Literal(Literal::Str(f)) => Some(f.clone()),
                    _ => return None,
                };
                return Some(StrFuncCond::Test(pattern.clone(), flags_str));
            }
        }
        Expr::CallBuiltin { op: name, args } => {
            if args.len() == 2 && matches!(args[0], Expr::Input) {
                if let Expr::Literal(Literal::Str(s)) = &args[1] {
                    match name {
                        BuiltinOp::StartsWith => return Some(StrFuncCond::Startswith(s.clone())),
                        BuiltinOp::EndsWith => return Some(StrFuncCond::Endswith(s.clone())),
                        BuiltinOp::Contains => return Some(StrFuncCond::Contains(s.clone())),
                        _ => {}
                    }
                }
            }
        }
        _ => {}
    }
    None
}

/// Collapse duplicate keys in an object-pair list: keep each key at the
/// position of its *first* occurrence and overwrite its value with the
/// value of the *last* occurrence. Matches jq's `{a:1, a:2}` → `{"a":2}`
/// object-literal semantics.
///
/// Every fast path that constructs an object from a static pair list must
/// route through this helper so new paths inherit the invariant. See
/// `docs/maintenance.md` §3 "オブジェクト重複キーの dedup".
pub(crate) fn normalize_object_pairs<K, V>(pairs: Vec<(K, V)>) -> Vec<(K, V)>
where
    K: PartialEq,
{
    let mut out: Vec<(K, V)> = Vec::with_capacity(pairs.len());
    for (k, v) in pairs {
        if let Some(existing) = out.iter_mut().find(|(ek, _)| *ek == k) {
            existing.1 = v;
        } else {
            out.push((k, v));
        }
    }
    out
}

/// Serialize a constant expression to compact JSON bytes.
/// Supports string, number, null, true, false, and constant ObjectConstruct/Collect.
fn const_expr_to_json(expr: &crate::ir::Expr) -> Option<Vec<u8>> {
    use crate::ir::{Expr, Literal};
    match expr {
        Expr::Literal(Literal::Str(s)) => {
            let mut v = Vec::with_capacity(s.len() + 2);
            push_const_json_string(&mut v, s);
            Some(v)
        }
        Expr::Literal(Literal::Num(n, repr)) => {
            if let Some(r) = repr.as_ref().filter(|r| crate::value::is_valid_json_number(r)) {
                Some(r.as_bytes().to_vec())
            } else {
                let mut buf = Vec::new();
                crate::value::push_jq_number_bytes(&mut buf, *n);
                Some(buf)
            }
        }
        Expr::Literal(Literal::Null) => Some(b"null".to_vec()),
        Expr::Literal(Literal::True) => Some(b"true".to_vec()),
        Expr::Literal(Literal::False) => Some(b"false".to_vec()),
        Expr::ObjectConstruct { pairs } => {
            // Mirror of `push_const_json`'s ObjectConstruct arm (#324).
            // Every value — including the ones that will be eliminated
            // by `normalize_object_pairs` — must be const-foldable;
            // otherwise an earlier `(key: input-touching)` pair's runtime
            // error gets silently dropped when a later same-key pair
            // rebinds the slot (#337).
            let mut extracted: Vec<(&str, &Expr)> = Vec::with_capacity(pairs.len());
            for (k, v) in pairs {
                if let Expr::Literal(Literal::Str(key)) = k {
                    extracted.push((key.as_str(), v));
                } else {
                    return None;
                }
            }
            for (_, v) in &extracted {
                const_expr_to_json(v)?;
            }
            let normalized = normalize_object_pairs(extracted);
            let mut buf = Vec::new();
            buf.push(b'{');
            for (i, (key, v)) in normalized.iter().enumerate() {
                if i > 0 { buf.push(b','); }
                push_const_json_string(&mut buf, key);
                buf.push(b':');
                buf.extend(const_expr_to_json(v)?);
            }
            buf.push(b'}');
            Some(buf)
        }
        Expr::Collect { generator } => {
            // Constant array: [lit, lit, ...]
            fn collect_comma_elems<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
                match e {
                    Expr::Comma { left, right } => {
                        collect_comma_elems(left, out);
                        collect_comma_elems(right, out);
                    }
                    _ => out.push(e),
                }
            }
            let mut elems = Vec::new();
            collect_comma_elems(generator, &mut elems);
            let mut buf = Vec::new();
            buf.push(b'[');
            for (i, elem) in elems.iter().enumerate() {
                if i > 0 { buf.push(b','); }
                buf.extend(const_expr_to_json(elem)?);
            }
            buf.push(b']');
            Some(buf)
        }
        _ => None,
    }
}

/// Serialize a constant expression to JSON bytes. Returns false if expression is not fully constant.
/// Emit `s` as a JSON string (quotes included) with jq's escaping: `"`, `\`,
/// the named control escapes (`\b \t \n \f \r`), `\u00xx` for the rest of
/// U+0000–U+001F, and `\u007f` for DEL. Mirrors the canonical
/// `push_json_string` serializer. Used for both string-literal *values* and
/// object *keys* on the constant fast path so the const path escapes
/// identically to the generic path — keys previously leaked verbatim (#975).
fn push_const_json_string(buf: &mut Vec<u8>, s: &str) {
    buf.push(b'"');
    for &b in s.as_bytes() {
        match b {
            b'"' => buf.extend_from_slice(b"\\\""),
            b'\\' => buf.extend_from_slice(b"\\\\"),
            b'\n' => buf.extend_from_slice(b"\\n"),
            b'\r' => buf.extend_from_slice(b"\\r"),
            b'\t' => buf.extend_from_slice(b"\\t"),
            0x08 => buf.extend_from_slice(b"\\b"),
            0x0c => buf.extend_from_slice(b"\\f"),
            c if c < 0x20 => {
                let hex = format!("\\u{:04x}", c);
                buf.extend_from_slice(hex.as_bytes());
            }
            0x7f => buf.extend_from_slice(b"\\u007f"),
            _ => buf.push(b),
        }
    }
    buf.push(b'"');
}

fn push_const_json(expr: &crate::ir::Expr, buf: &mut Vec<u8>) -> bool {
    use crate::ir::{Expr, Literal};
    match expr {
        Expr::Literal(Literal::Null) => { buf.extend_from_slice(b"null"); true }
        Expr::Literal(Literal::True) => { buf.extend_from_slice(b"true"); true }
        Expr::Literal(Literal::False) => { buf.extend_from_slice(b"false"); true }
        Expr::Literal(Literal::Num(n, Some(raw))) => {
            if crate::value::is_valid_json_number(raw) {
                buf.extend_from_slice(crate::value::canonical_repr_bytes(raw).as_bytes());
            } else {
                crate::value::push_jq_number_bytes(buf, *n);
            }
            true
        }
        Expr::Literal(Literal::Num(n, None)) => {
            crate::value::push_jq_number_bytes(buf, *n);
            true
        }
        Expr::Literal(Literal::Str(s)) => {
            push_const_json_string(buf, s);
            true
        }
        Expr::ObjectConstruct { pairs } => {
            // All keys must be string literals. Duplicates collapse via
            // `normalize_object_pairs` (last value wins, keeps first
            // position). Every value — including ones that will be
            // overwritten by a later duplicate — must be `push_const_json`-
            // emittable: jq evaluates each `(key: value)` pair in source
            // order, so an earlier pair's runtime error must still surface
            // even when a later pair would rebind the same key. If we
            // dedup first and only check the survivors, the eliminated
            // expression's error is silently dropped (#324).
            let mut extracted: Vec<(&str, &Expr)> = Vec::with_capacity(pairs.len());
            for (key, val) in pairs {
                match key {
                    Expr::Literal(Literal::Str(k)) => extracted.push((k.as_str(), val)),
                    _ => return false,
                }
            }
            for (_, val) in &extracted {
                let mut probe = Vec::new();
                if !push_const_json(val, &mut probe) { return false; }
            }
            let normalized = normalize_object_pairs(extracted);
            buf.push(b'{');
            for (i, (k, val)) in normalized.iter().enumerate() {
                if i > 0 { buf.push(b','); }
                push_const_json_string(buf, k);
                buf.push(b':');
                if !push_const_json(val, buf) { return false; }
            }
            buf.push(b'}');
            true
        }
        Expr::Collect { generator } => {
            // [expr] — could be a Comma list of constants
            buf.push(b'[');
            if !push_const_comma_list(generator, buf, true) { return false; }
            buf.push(b']');
            true
        }
        _ => false,
    }
}

fn push_const_comma_list(expr: &crate::ir::Expr, buf: &mut Vec<u8>, first: bool) -> bool {
    use crate::ir::Expr;
    if let Expr::Comma { left, right } = expr {
        if !push_const_comma_list(left, buf, first) { return false; }
        if !push_const_comma_list(right, buf, false) { return false; }
        true
    } else {
        if !first { buf.push(b','); }
        push_const_json(expr, buf)
    }
}

impl Filter {
    /// Returns true if this filter is a simple identity (`.`) that passes through input unchanged.
    /// Also recognizes semantic equivalences like `to_entries | from_entries`.
    pub fn is_identity(&self) -> bool {
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        Self::expr_is_identity(expr)
    }

    fn expr_is_identity(expr: &crate::ir::Expr) -> bool {
        use crate::ir::Expr;
        match expr {
            Expr::Input => true,
            Expr::Pipe { left, right } => {
                // . | X → X, X | . → X (recursive identity simplification)
                if Self::expr_is_identity(left) { return Self::expr_is_identity(right); }
                if Self::expr_is_identity(right) { return Self::expr_is_identity(left); }
                // NOTE: `to_entries | from_entries` is NOT universal identity — it's
                // identity only for string-keyed objects. Arrays get a type error
                // on numeric keys, and `[]` round-trips to `{}` (issue #73).
                false
            }
            _ => false,
        }
    }

    /// Detect `objects`, `arrays`, `strings`, `numbers`, `nulls`, `booleans` type filter.
    /// Returns the first bytes that match the type.
    pub fn detect_type_filter(&self) -> Option<Vec<u8>> {
        use crate::ir::{Expr, Literal, BinOp, UnaryOp};
        let expr = self.detect_expr()?;
        // select(type == "object") compiles to: IfThenElse { cond: BinOp(Eq, UnaryOp(Type,.), Literal(Str)), then: ., else: Empty }
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            if let Expr::BinOp { op: BinOp::Eq, lhs, rhs } = cond.as_ref() {
                let type_check = |operand: &Expr, type_str: &Expr| -> Option<Vec<u8>> {
                    if matches!(operand, Expr::UnaryOp { op: UnaryOp::Type, operand: inner } if matches!(inner.as_ref(), Expr::Input)) {
                        if let Expr::Literal(Literal::Str(t)) = type_str {
                            return match t.as_str() {
                                "object" => Some(vec![b'{']),
                                "array" => Some(vec![b'[']),
                                "string" => Some(vec![b'"']),
                                "number" => Some(vec![b'-', b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9']),
                                "boolean" => Some(vec![b't', b'f']),
                                "null" => Some(vec![b'n']),
                                _ => None,
                            };
                        }
                    }
                    None
                };
                if let Some(r) = type_check(lhs.as_ref(), rhs.as_ref()) { return Some(r); }
                if let Some(r) = type_check(rhs.as_ref(), lhs.as_ref()) { return Some(r); }
            }
        }
        None
    }

    /// Detect a literal filter that doesn't reference input.
    /// Returns the compact JSON bytes for the literal, or None.
    pub fn detect_literal_output(&self) -> Option<Vec<u8>> {
        let expr = self.detect_expr()?;
        let mut buf = Vec::new();
        if push_const_json(expr, &mut buf) {
            Some(buf)
        } else {
            None
        }
    }

    /// Like detect_literal_output but also handles input-free expressions
    /// (evaluated once with null input). Returns None if expression depends on input.
    /// The result is a list of JSON output lines (one per output value).
    pub fn detect_input_free_output(&self) -> Option<Vec<Vec<u8>>> {
        // Use the ORIGINAL parsed expr, not the simplified one. The simplifier
        // can rewrite a binding (e.g. the `. as $x` / `LIT as $x` that `IN(x)`
        // desugars to) into a shape whose input-dependence `contains_input`
        // no longer sees, so evaluating the simplified form against `null` lost
        // the binding and `IN(1)` returned `false`. The original expr is the
        // authoritative one the generic path evaluates. See #847.
        let expr = &self.parsed.0;
        if contains_input(expr) { return None; }
        // Already handled by literal_output?
        let mut buf = Vec::new();
        if push_const_json(expr, &mut buf) {
            return Some(vec![buf]);
        }
        // This fast path evaluates `self.simplified` against a `null` input
        // with an env that carries no user functions. That is only faithful
        // when the program defines none: a `FuncCall` would otherwise hit an
        // empty func table and raise "undefined function", which an enclosing
        // `try`/`catch` silently turns into bogus output (#777 — e.g.
        // `def f: label $o|break $o; try f catch "c"` wrongly yielded "c").
        // Const-foldable input-free exprs are already returned above via
        // `push_const_json`; anything left that references user defs must run
        // on the authoritative input-carrying path instead, so bail.
        if !self.parsed.1.is_empty() {
            return None;
        }
        let mut outputs = Vec::new();
        let env: crate::eval::EnvRef = std::rc::Rc::new(std::cell::RefCell::new(crate::eval::Env::new(vec![])));
        let result = crate::eval::eval(expr, crate::value::Value::Null, &env, &mut |v| {
            let json = crate::value::value_to_json_precise(&v);
            outputs.push(json.into_bytes());
            Ok(true)
        });
        if result.is_ok() && !outputs.is_empty() {
            Some(outputs)
        } else {
            None
        }
    }

    /// Returns true if this filter produces no output (e.g. `empty`, `. | empty`).
    pub fn is_empty(&self) -> bool {
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        matches!(expr, crate::ir::Expr::Empty)
    }

    /// Detect `select(.field > N)` pattern for fast-path select.
    /// Returns (field_name, comparison_op, threshold) if detected.
    pub fn detect_select_field_cmp(&self) -> Option<(String, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        // select(cond) compiles to if cond then . else empty end
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            // cond should be BinOp { .field, op, Literal::Num }
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                    return None;
                }
                // .field on lhs, literal on rhs
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            return Some((field.clone(), *op, *n));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.field <arith_ops> <cmp> N)` — select with arithmetic chain, outputting the full object.
    /// Returns (field, arith_ops, cmp_op, threshold).
    pub fn detect_select_arith_cmp(&self) -> Option<(String, Vec<(crate::ir::BinOp, f64)>, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                    return None;
                }
                if let Expr::Literal(Literal::Num(threshold, _)) = rhs.as_ref() {
                    let mut arith_ops = Vec::new();
                    let mut cur = lhs.as_ref();
                    loop {
                        if let Expr::BinOp { op: aop, lhs: al, rhs: ar } = cur {
                            if matches!(aop, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                                if let Expr::Literal(Literal::Num(n, _)) = ar.as_ref() {
                                    // jq raises on a zero divisor; keep the chain on generic eval (#1063)
                                    if matches!(aop, BinOp::Div | BinOp::Mod) && *n == 0.0 { break; }
                                    arith_ops.push((*aop, *n));
                                    cur = al.as_ref();
                                    continue;
                                }
                            }
                        }
                        break;
                    }
                    if arith_ops.is_empty() { return None; }
                    arith_ops.reverse();
                    if let Expr::Index { expr: base, key } = cur {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            return Some((field.clone(), arith_ops, *op, *threshold));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.field | length cmp N)` — select by string/array length.
    /// Returns (field_name, cmp_op, threshold).
    pub fn detect_select_field_length_cmp(&self) -> Option<(String, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        let try_extract = |cond: &Expr| -> Option<(String, BinOp, f64)> {
            if let Expr::BinOp { op, lhs, rhs } = cond {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                // LHS: .field | length (beta-reduced to UnaryOp(Length, Index(Input, field)))
                if let Expr::UnaryOp { op: UnaryOp::Length, operand } = lhs.as_ref() {
                    if let Expr::Index { expr: base, key } = operand.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((f.clone(), *op, *n));
                            }
                        }
                    }
                }
            }
            None
        };
        // select(cond) → if cond then . else empty end
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                return try_extract(cond);
            }
        }
        None
    }

    /// Detect `select(.field | length cmp N) | .out_field`.
    /// Returns (field_name, cmp_op, threshold, out_field).
    pub fn detect_select_field_length_cmp_then_field(&self) -> Option<(String, crate::ir::BinOp, f64, String)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, BinOp, f64, String)> {
            // Output must be .field
            let out_field = if let Expr::Index { expr: base, key } = output {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() } else { return None; }
            } else { return None; };
            // Condition: (.field | length) cmp N
            if let Expr::BinOp { op, lhs, rhs } = cond {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                if let Expr::UnaryOp { op: UnaryOp::Length, operand } = lhs.as_ref() {
                    if let Expr::Index { expr: base, key } = operand.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((f.clone(), *op, *n, out_field));
                            }
                        }
                    }
                }
            }
            None
        };
        // Form 1: Pipe(select(cond), .field)
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        // Form 2: if cond then .field else empty end
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `if .field|length cmp N then .f1 else .f2 end`.
    /// Returns (cond_field, cmp_op, threshold, then_field, else_field).
    pub fn detect_if_field_length_cmp_then_fields(&self) -> Option<(String, crate::ir::BinOp, f64, String, String)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                if let Expr::UnaryOp { op: UnaryOp::Length, operand } = lhs.as_ref() {
                    if let Expr::Index { expr: base, key } = operand.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(cond_field)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                // Then branch: .field
                                let then_f = if let Expr::Index { expr: tb, key: tk } = then_branch.as_ref() {
                                    if !matches!(tb.as_ref(), Expr::Input) { return None; }
                                    if let Expr::Literal(Literal::Str(f)) = tk.as_ref() { f.clone() } else { return None; }
                                } else { return None; };
                                // Else branch: .field
                                let else_f = if let Expr::Index { expr: eb, key: ek } = else_branch.as_ref() {
                                    if !matches!(eb.as_ref(), Expr::Input) { return None; }
                                    if let Expr::Literal(Literal::Str(f)) = ek.as_ref() { f.clone() } else { return None; }
                                } else { return None; };
                                return Some((cond_field.clone(), *op, *n, then_f, else_f));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.field|length cmp N) | {remap}`.
    /// Returns (cond_field, cmp_op, threshold, remap_fields).
    pub fn detect_select_field_length_cmp_then_remap(&self) -> Option<(String, crate::ir::BinOp, f64, Vec<(String, RemapExpr)>)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        let try_extract_cond = |cond: &Expr| -> Option<(String, BinOp, f64)> {
            if let Expr::BinOp { op, lhs, rhs } = cond {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                if let Expr::UnaryOp { op: UnaryOp::Length, operand } = lhs.as_ref() {
                    if let Expr::Index { expr: base, key } = operand.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((f.clone(), *op, *n));
                            }
                        }
                    }
                }
            }
            None
        };
        let try_extract_remap = |output: &Expr| -> Option<Vec<(String, RemapExpr)>> {
            if let Expr::ObjectConstruct { pairs } = output {
                if pairs.is_empty() { return None; }
                let mut result = Vec::with_capacity(pairs.len());
                for (k, v) in pairs {
                    let key = if let Expr::Literal(Literal::Str(s)) = k { s.clone() } else { return None; };
                    result.push((key, Self::classify_remap_value(v)?));
                }
                Some(result)
            } else { None }
        };
        // Form 1: Pipe(select(cond), {remap})
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some((cf, op, n)) = try_extract_cond(cond) {
                        if let Some(remap) = try_extract_remap(right) {
                            return Some((cf, op, n, remap));
                        }
                    }
                }
            }
        }
        // Form 2: if cond then {remap} else empty end
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some((cf, op, n)) = try_extract_cond(cond) {
                    if let Some(remap) = try_extract_remap(then_branch) {
                        return Some((cf, op, n, remap));
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | tostring | length`.
    /// Returns field_name.
    pub fn detect_field_tostring_length(&self) -> Option<String> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::UnaryOp { op: UnaryOp::Length, operand } = expr {
            if let Expr::UnaryOp { op: UnaryOp::ToString, operand: inner } = operand.as_ref() {
                if let Expr::Index { expr: base, key } = inner.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            return Some(f.clone());
                        }
                    }
                }
            }
        }
        // Pipe form: Pipe(.field, Pipe(tostring, length))
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        if let Expr::Pipe { left: pl, right: pr } = right.as_ref() {
                            if matches!(pl.as_ref(), Expr::UnaryOp { op: UnaryOp::ToString, operand } if matches!(operand.as_ref(), Expr::Input))
                                && matches!(pr.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input))
                            {
                                return Some(f.clone());
                            }
                        }
                        if let Expr::UnaryOp { op: UnaryOp::Length, operand } = right.as_ref() {
                            if let Expr::UnaryOp { op: UnaryOp::ToString, operand: inner } = operand.as_ref() {
                                if matches!(inner.as_ref(), Expr::Input) {
                                    return Some(f.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.field == null)` or `select(.field != null)` — output whole object.
    /// Returns (field_name, is_eq) where is_eq=true for ==null, false for !=null.
    pub fn detect_select_field_null(&self) -> Option<(String, bool)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                let is_eq = match op {
                    BinOp::Eq => true,
                    BinOp::Ne => false,
                    _ => return None,
                };
                // .field == null or .field != null
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            if matches!(rhs.as_ref(), Expr::Literal(Literal::Null)) {
                                return Some((f.clone(), is_eq));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.f1 > N and .f2 < M)` or `select(.f1 > N or .f2 < M)` pattern.
    /// Returns (conjunct, Vec<(field, op, threshold)>) where conjunct is And or Or.
    pub fn detect_select_compound_cmp(&self) -> Option<(crate::ir::BinOp, Vec<(String, crate::ir::BinOp, f64)>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            // cond = BinOp(And/Or, cmp1, cmp2)
            let extract_cmp = |e: &Expr| -> Option<(String, BinOp, f64)> {
                if let Expr::BinOp { op, lhs, rhs } = e {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                        return None;
                    }
                    if let Expr::Index { expr: base, key } = lhs.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((field.clone(), *op, *n));
                            }
                        }
                    }
                }
                None
            };
            // Flatten And/Or chains: (A and B) and C → [A, B, C]
            fn collect_conds<'a>(e: &'a Expr, conj: BinOp, out: &mut Vec<&'a Expr>) -> bool {
                if let Expr::BinOp { op, lhs, rhs } = e {
                    if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                        return collect_conds(lhs, conj, out) && collect_conds(rhs, conj, out);
                    }
                }
                out.push(e);
                true
            }
            for conj in [BinOp::And, BinOp::Or] {
                if let Expr::BinOp { op, .. } = cond.as_ref() {
                    if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                        let mut parts = Vec::new();
                        if collect_conds(cond, conj, &mut parts) && parts.len() >= 2 {
                            let cmps: Vec<_> = parts.iter().filter_map(|e| extract_cmp(e)).collect();
                            if cmps.len() == parts.len() {
                                return Some((conj, cmps));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.field1 cmp N and (.field2 | str_op("str")))` — mixed numeric + string compound select.
    /// Returns (num_field, cmp_op, threshold, str_field, str_op_name, str_arg).
    /// str_op_name is one of: "startswith", "endswith", "contains", "test", "eq".
    pub fn detect_select_num_and_str(&self) -> Option<(String, crate::ir::BinOp, f64, String, String, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            if let Expr::BinOp { op: BinOp::And, lhs, rhs } = cond.as_ref() {
                let extract_num_cmp = |e: &Expr| -> Option<(String, BinOp, f64)> {
                    if let Expr::BinOp { op, lhs, rhs } = e {
                        if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                        if let Expr::Index { expr: base, key } = lhs.as_ref() {
                            if !matches!(base.as_ref(), Expr::Input) { return None; }
                            if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                    return Some((f.clone(), *op, *n));
                                }
                            }
                        }
                    }
                    None
                };
                let extract_str_cond = |e: &Expr| -> Option<(String, String, String)> {
                    // Form: .field | str_op("arg") — as Pipe(Index, CallBuiltin)
                    if let Expr::Pipe { left, right } = e {
                        if let Expr::Index { expr: base, key } = left.as_ref() {
                            if !matches!(base.as_ref(), Expr::Input) { return None; }
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                // CallBuiltin(startswith/endswith/contains, [Literal(Str)])
                                if let Expr::CallBuiltin { op: name, args } = right.as_ref() {
                                    if args.len() == 1 {
                                        if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                            if matches!(name, BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains) {
                                                return Some((field.clone(), name.name().to_string(), arg.clone()));
                                            }
                                        }
                                    }
                                }
                                // RegexTest
                                if let Expr::RegexTest { input_expr, re, .. } = right.as_ref() {
                                    if matches!(input_expr.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Str(pat)) = re.as_ref() {
                                            return Some((field.clone(), "test".to_string(), pat.clone()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Form: .field == "str"
                    if let Expr::BinOp { op: BinOp::Eq, lhs: l, rhs: r } = e {
                        if let Expr::Index { expr: base, key } = l.as_ref() {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                    if let Expr::Literal(Literal::Str(val)) = r.as_ref() {
                                        return Some((field.clone(), "eq".to_string(), val.clone()));
                                    }
                                }
                            }
                        }
                    }
                    None
                };
                // Try both orderings: (num, str) and (str, num)
                if let (Some((nf, nop, nth)), Some((sf, sop, sarg))) = (extract_num_cmp(lhs), extract_str_cond(rhs)) {
                    return Some((nf, nop, nth, sf, sop, sarg));
                }
                if let (Some((sf, sop, sarg)), Some((nf, nop, nth))) = (extract_str_cond(lhs), extract_num_cmp(rhs)) {
                    return Some((nf, nop, nth, sf, sop, sarg));
                }
            }
        }
        None
    }

    /// Detect `select(.x > N and .y < M) | .output` — compound select then field access.
    /// Returns (conjunction, comparisons, output_field).
    pub fn detect_select_compound_cmp_then_field(&self) -> Option<(crate::ir::BinOp, Vec<(String, crate::ir::BinOp, f64)>, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let extract_cmp = |e: &Expr| -> Option<(String, BinOp, f64)> {
            if let Expr::BinOp { op, lhs, rhs } = e {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                    return None;
                }
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            return Some((field.clone(), *op, *n));
                        }
                    }
                }
            }
            None
        };
        fn collect_conds<'a>(e: &'a Expr, conj: BinOp, out: &mut Vec<&'a Expr>) -> bool {
            if let Expr::BinOp { op, lhs, rhs } = e {
                if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                    return collect_conds(lhs, conj, out) && collect_conds(rhs, conj, out);
                }
            }
            out.push(e);
            true
        }
        // Extract select condition and output field from Pipe(IfThenElse, .field)
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(BinOp, Vec<(String, BinOp, f64)>, String)> {
            let out_field = if let Expr::Index { expr: base, key } = output {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() } else { return None; }
            } else { return None; };
            for conj in [BinOp::And, BinOp::Or] {
                if let Expr::BinOp { op, .. } = cond {
                    if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                        let mut parts = Vec::new();
                        if collect_conds(cond, conj, &mut parts) && parts.len() >= 2 {
                            let cmps: Vec<_> = parts.iter().filter_map(|e| extract_cmp(e)).collect();
                            if cmps.len() == parts.len() {
                                return Some((conj, cmps, out_field));
                            }
                        }
                    }
                }
            }
            None
        };
        // Form 1: Pipe(select(compound), .field)
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    return try_extract(cond, right);
                }
            }
        }
        // Form 2: IfThenElse{compound_cond, .field, empty}
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                return try_extract(cond, then_branch);
            }
        }
        None
    }

    /// Detect `select(.x > N and .y < M) | {a:.f1, b:.f2}` — compound select then field remap.
    /// Returns (conjunction, comparisons, remap_pairs[(key, field)]).
    pub fn detect_select_compound_cmp_then_remap(&self) -> Option<(crate::ir::BinOp, Vec<(String, crate::ir::BinOp, f64)>, Vec<(String, String)>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let extract_cmps = |cond: &Expr| -> Option<(BinOp, Vec<(String, BinOp, f64)>)> {
            let extract_cmp = |e: &Expr| -> Option<(String, BinOp, f64)> {
                if let Expr::BinOp { op, lhs, rhs } = e {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                    if let Expr::Index { expr: base, key } = lhs.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((f.clone(), *op, *n));
                            }
                        }
                    }
                }
                None
            };
            fn collect_conds<'a>(e: &'a Expr, conj: BinOp, out: &mut Vec<&'a Expr>) -> bool {
                if let Expr::BinOp { op, lhs, rhs } = e {
                    if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                        return collect_conds(lhs, conj, out) && collect_conds(rhs, conj, out);
                    }
                }
                out.push(e);
                true
            }
            for conj in [BinOp::And, BinOp::Or] {
                if let Expr::BinOp { op, .. } = cond {
                    if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                        let mut parts = Vec::new();
                        if collect_conds(cond, conj, &mut parts) && parts.len() >= 2 {
                            let cmps: Vec<_> = parts.iter().filter_map(|e| extract_cmp(e)).collect();
                            if cmps.len() == parts.len() { return Some((conj, cmps)); }
                        }
                    }
                }
            }
            None
        };
        let extract_remap = |output: &Expr| -> Option<Vec<(String, String)>> {
            if let Expr::ObjectConstruct { pairs } = output {
                let mut remap = Vec::new();
                for (k, v) in pairs {
                    let key = if let Expr::Literal(Literal::Str(s)) = k { s.clone() } else { return None; };
                    if let Expr::Index { expr: base, key: fk } = v {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = fk.as_ref() {
                            remap.push((key, f.clone()));
                        } else { return None; }
                    } else { return None; }
                }
                if !remap.is_empty() { return Some(remap); }
            }
            None
        };
        // Form 1: Pipe(select(compound), {remap})
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let (Some((conj, cmps)), Some(remap)) = (extract_cmps(cond), extract_remap(right)) {
                        return Some((conj, cmps, remap));
                    }
                }
            }
        }
        // Form 2: IfThenElse{compound_cond, {remap}, empty}
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let (Some((conj, cmps)), Some(remap)) = (extract_cmps(cond), extract_remap(then_branch)) {
                    return Some((conj, cmps, remap));
                }
            }
        }
        None
    }

    /// Detect `select(.x > N and .y < M) | computed_value` — compound select then RemapExpr.
    pub fn detect_select_compound_cmp_then_computed(&self) -> Option<(crate::ir::BinOp, Vec<(String, crate::ir::BinOp, f64)>, RemapExpr)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let extract_cmps = |cond: &Expr| -> Option<(BinOp, Vec<(String, BinOp, f64)>)> {
            let extract_cmp = |e: &Expr| -> Option<(String, BinOp, f64)> {
                if let Expr::BinOp { op, lhs, rhs } = e {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                    if let Expr::Index { expr: base, key } = lhs.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((f.clone(), *op, *n));
                            }
                        }
                    }
                }
                None
            };
            fn collect_conds<'a>(e: &'a Expr, conj: BinOp, out: &mut Vec<&'a Expr>) -> bool {
                if let Expr::BinOp { op, lhs, rhs } = e {
                    if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                        return collect_conds(lhs, conj, out) && collect_conds(rhs, conj, out);
                    }
                }
                out.push(e);
                true
            }
            for conj in [BinOp::And, BinOp::Or] {
                if let Expr::BinOp { op, .. } = cond {
                    if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                        let mut parts = Vec::new();
                        if collect_conds(cond, conj, &mut parts) && parts.len() >= 2 {
                            let cmps: Vec<_> = parts.iter().filter_map(|e| extract_cmp(e)).collect();
                            if cmps.len() == parts.len() { return Some((conj, cmps)); }
                        }
                    }
                }
            }
            None
        };
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(BinOp, Vec<(String, BinOp, f64)>, RemapExpr)> {
            let (conj, cmps) = extract_cmps(cond)?;
            let rexpr = Self::classify_remap_value(output)?;
            // Must be a computed value (not just a field — that's already handled by compound_field)
            if matches!(rexpr, RemapExpr::Field(_)) { return None; }
            Some((conj, cmps, rexpr))
        };
        // Form 1: Pipe(select(compound), computed)
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    return try_extract(cond, right);
                }
            }
        }
        // Form 2: IfThenElse{compound_cond, computed, empty}
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                return try_extract(cond, then_branch);
            }
        }
        None
    }

    /// Detect `select(.x > N and .y < M) | {k: rexpr, ...}` — compound select then computed remap.
    pub fn detect_select_compound_cmp_then_cremap(&self) -> Option<(crate::ir::BinOp, Vec<(String, crate::ir::BinOp, f64)>, Vec<(String, RemapExpr)>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let extract_cmps = |cond: &Expr| -> Option<(BinOp, Vec<(String, BinOp, f64)>)> {
            let extract_cmp = |e: &Expr| -> Option<(String, BinOp, f64)> {
                if let Expr::BinOp { op, lhs, rhs } = e {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                    if let Expr::Index { expr: base, key } = lhs.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((f.clone(), *op, *n));
                            }
                        }
                    }
                }
                None
            };
            fn collect_conds2<'a>(e: &'a Expr, conj: BinOp, out: &mut Vec<&'a Expr>) -> bool {
                if let Expr::BinOp { op, lhs, rhs } = e {
                    if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                        return collect_conds2(lhs, conj, out) && collect_conds2(rhs, conj, out);
                    }
                }
                out.push(e);
                true
            }
            for conj in [BinOp::And, BinOp::Or] {
                if let Expr::BinOp { op, .. } = cond {
                    if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                        let mut parts = Vec::new();
                        if collect_conds2(cond, conj, &mut parts) && parts.len() >= 2 {
                            let cmps: Vec<_> = parts.iter().filter_map(|e| extract_cmp(e)).collect();
                            if cmps.len() == parts.len() { return Some((conj, cmps)); }
                        }
                    }
                }
            }
            None
        };
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(BinOp, Vec<(String, BinOp, f64)>, Vec<(String, RemapExpr)>)> {
            let (conj, cmps) = extract_cmps(cond)?;
            if let Expr::ObjectConstruct { pairs } = output {
                if pairs.is_empty() { return None; }
                let mut result = Vec::with_capacity(pairs.len());
                for (k, v) in pairs {
                    let key = if let Expr::Literal(Literal::Str(s)) = k { s.clone() } else { return None; };
                    let rexpr = Self::classify_remap_value(v)?;
                    result.push((key, rexpr));
                }
                return Some((conj, cmps, result));
            }
            None
        };
        // Form 1: Pipe(select(compound), {computed_remap})
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        // Form 2: IfThenElse{compound_cond, {computed_remap}, empty}
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.x > N and .y < M) | str_add_chain` — compound select then string chain.
    /// Returns (conjunction, conditions, string_add_parts).
    pub fn detect_select_compound_cmp_then_str_chain(&self) -> Option<(crate::ir::BinOp, Vec<(String, crate::ir::BinOp, f64)>, Vec<StringAddPart>)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        let extract_cmp = |e: &Expr| -> Option<(String, BinOp, f64)> {
            if let Expr::BinOp { op, lhs, rhs } = e {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            return Some((field.clone(), *op, *n));
                        }
                    }
                }
            }
            None
        };
        fn collect_conds_sc<'a>(e: &'a Expr, conj: BinOp, out: &mut Vec<&'a Expr>) -> bool {
            if let Expr::BinOp { op, lhs, rhs } = e {
                if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                    return collect_conds_sc(lhs, conj, out) && collect_conds_sc(rhs, conj, out);
                }
            }
            out.push(e); true
        }
        fn collect_chain_sc_ts(operand: &Expr, parts: &mut Vec<StringAddPart>) -> bool {
            if let Expr::Index { expr: base, key } = operand {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        parts.push(StringAddPart::FieldToString(f.clone()));
                        return true;
                    }
                }
            }
            let mut arith_ops = Vec::new();
            let mut cur = operand;
            loop {
                if let Expr::BinOp { op: aop, lhs, rhs } = cur {
                    if matches!(aop, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            // jq raises on a zero divisor; keep the chain on generic eval (#1063)
                            if matches!(aop, BinOp::Div | BinOp::Mod) && *n == 0.0 { break; }
                            arith_ops.push((*aop, *n));
                            cur = lhs.as_ref();
                            continue;
                        }
                    }
                }
                break;
            }
            if !arith_ops.is_empty() {
                arith_ops.reverse();
                if let Expr::Index { expr: base, key } = cur {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            parts.push(StringAddPart::FieldArithToString(f.clone(), arith_ops));
                            return true;
                        }
                    }
                }
            }
            false
        }
        fn collect_chain_sc(expr: &Expr, parts: &mut Vec<StringAddPart>) -> bool {
            match expr {
                Expr::BinOp { op: BinOp::Add, lhs, rhs } => {
                    collect_chain_sc(lhs, parts) && collect_chain_sc(rhs, parts)
                }
                Expr::Index { expr: base, key } if matches!(base.as_ref(), Expr::Input) => {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        parts.push(StringAddPart::Field(f.clone())); true
                    } else { false }
                }
                Expr::Literal(Literal::Str(s)) => {
                    parts.push(StringAddPart::Literal(s.clone())); true
                }
                Expr::UnaryOp { op: UnaryOp::ToString, operand } => {
                    collect_chain_sc_ts(operand, parts)
                }
                _ => false,
            }
        }
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(BinOp, Vec<(String, BinOp, f64)>, Vec<StringAddPart>)> {
            let mut parts = Vec::new();
            if !collect_chain_sc(output, &mut parts) || parts.len() < 2 { return None; }
            if !parts.iter().any(|p| !matches!(p, StringAddPart::Literal(_))) { return None; }
            for conj in [BinOp::And, BinOp::Or] {
                if let Expr::BinOp { op, .. } = cond {
                    if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                        let mut cond_parts = Vec::new();
                        if collect_conds_sc(cond, conj, &mut cond_parts) && cond_parts.len() >= 2 {
                            let cmps: Vec<_> = cond_parts.iter().filter_map(|e| extract_cmp(e)).collect();
                            if cmps.len() == cond_parts.len() {
                                return Some((conj, cmps, parts));
                            }
                        }
                    }
                }
            }
            None
        };
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.a.b.c > N)` pattern for nested field numeric comparison.
    /// Returns (field_path, comparison_op, threshold) if detected.
    pub fn detect_select_nested_cmp(&self) -> Option<(Vec<String>, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                    return None;
                }
                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                    // Extract nested field path
                    let mut fields = Vec::new();
                    let mut current = lhs.as_ref();
                    loop {
                        if let Expr::Index { expr: base, key } = current {
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                fields.push(field.clone());
                                current = base.as_ref();
                            } else { return None; }
                        } else if matches!(current, Expr::Input) {
                            break;
                        } else { return None; }
                    }
                    if fields.len() >= 2 {
                        fields.reverse();
                        return Some((fields, *op, *n));
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.field == "str")` pattern for string comparison select.
    /// Returns (field_name, op, string_value) if detected.
    pub fn detect_select_field_str(&self) -> Option<(String, crate::ir::BinOp, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le) { return None; }
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::Literal(Literal::Str(val)) = rhs.as_ref() {
                            return Some((field.clone(), *op, val.clone()));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.field | startswith/endswith/contains("str"))` pattern.
    /// Returns (field_name, builtin_name, string_arg) if detected.
    pub fn detect_select_field_str_test(&self) -> Option<(String, String, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            // cond = .field | builtin("str")
            if let Expr::Pipe { left, right } = cond.as_ref() {
                if let Expr::Index { expr: base, key } = left.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::CallBuiltin { op: name, args } = right.as_ref() {
                            if args.len() == 1 {
                                if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                    if matches!(name, BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains) {
                                        return Some((field.clone(), name.name().to_string(), arg.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.field | string_ops... | terminal)` where terminal is boolean.
    /// E.g., `select(.name | ascii_downcase | startswith("user"))`.
    /// Returns (field, ops, terminal) — same as field_string_chain but in select context.
    pub fn detect_select_string_chain(&self) -> Option<(String, Vec<StringChainOp>, StringChainTerminal)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            // cond must be: .field | ops... | terminal
            if let Expr::Pipe { left, right } = cond.as_ref() {
                if let Expr::Index { expr: base, key } = left.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        let mut ops = Vec::new();
                        let terminal = Self::collect_string_chain_ops_with_terminal(right, &mut ops);
                        // Must have at least one op + boolean terminal
                        if !ops.is_empty() && matches!(terminal, StringChainTerminal::Startswith(_) | StringChainTerminal::Endswith(_) | StringChainTerminal::Contains(_)) {
                            return Some((field.clone(), ops, terminal));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.f | startswith("a") and/or .f | endswith("b"))` — compound string test select.
    /// Returns (logic_op, Vec<(field, test_name, test_arg)>) where logic_op is And/Or.
    pub fn detect_select_compound_str_test(&self) -> Option<(crate::ir::BinOp, Vec<(String, String, String)>)> {
        use crate::ir::{BinOp, Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            fn extract_str_test(e: &Expr) -> Option<(String, String, String)> {
                if let Expr::Pipe { left, right } = e {
                    if let Expr::Index { expr: base, key } = left.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            if let Expr::CallBuiltin { op: name, args } = right.as_ref() {
                                if matches!(name, BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains) && args.len() == 1 {
                                    if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                        return Some((field.clone(), name.name().to_string(), arg.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
                // Also handle beta-reduced form: CallBuiltin("startswith", [Index(Input, "field"), Literal("str")])
                if let Expr::CallBuiltin { op: name, args } = e {
                    if matches!(name, BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains) && args.len() == 2 {
                        if let Expr::Index { expr: base, key } = &args[0] {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                    if let Expr::Literal(Literal::Str(arg)) = &args[1] {
                                        return Some((field.clone(), name.name().to_string(), arg.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
                None
            }
            // Collect And/Or chain of string tests
            fn collect_str_conds(e: &Expr, logic: &BinOp, out: &mut Vec<(String, String, String)>) -> bool {
                if let Expr::BinOp { op, lhs, rhs } = e {
                    if std::mem::discriminant(op) == std::mem::discriminant(logic) {
                        return collect_str_conds(lhs, logic, out) && collect_str_conds(rhs, logic, out);
                    }
                }
                if let Some(t) = extract_str_test(e) {
                    out.push(t);
                    true
                } else {
                    false
                }
            }
            if let Expr::BinOp { op, .. } = cond.as_ref() {
                if matches!(op, BinOp::And | BinOp::Or) {
                    let mut conds = Vec::new();
                    if collect_str_conds(cond, op, &mut conds) && conds.len() >= 2 {
                        return Some((*op, conds));
                    }
                }
            }
        }
        None
    }

    /// Detect mixed compound select: `select(A and B and ...)` where each condition is either
    /// a numeric comparison (.field cmp N) or a string test (.field | str_op("arg")).
    /// Returns (logic_op, conditions) where conditions is a Vec<MixedCond>.
    /// Only fires when both numeric and string conditions are present (otherwise use homogeneous detectors).
    pub fn detect_select_mixed_compound(&self) -> Option<(crate::ir::BinOp, Vec<MixedCond>)> {
        use crate::ir::{BinOp, Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            fn extract_mixed_cond(e: &Expr) -> Option<MixedCond> {
                // Numeric: .field cmp N
                if let Expr::BinOp { op, lhs, rhs } = e {
                    if matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                        if let Expr::Index { expr: base, key } = lhs.as_ref() {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                    if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                        return Some(MixedCond::NumCmp(field.clone(), *op, *n));
                                    }
                                }
                            }
                        }
                    }
                }
                // String test: .field | str_op("arg")
                if let Expr::Pipe { left, right } = e {
                    if let Expr::Index { expr: base, key } = left.as_ref() {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                if let Expr::CallBuiltin { op: name, args } = right.as_ref() {
                                    if matches!(name, BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains) && args.len() == 1 {
                                        if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                            return Some(MixedCond::StrTest(field.clone(), name.name().to_string(), arg.clone()));
                                        }
                                    }
                                }
                                // RegexTest
                                if let Expr::RegexTest { input_expr, re, .. } = right.as_ref() {
                                    if matches!(input_expr.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Str(pat)) = re.as_ref() {
                                            return Some(MixedCond::StrTest(field.clone(), "test".to_string(), pat.clone()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Beta-reduced: CallBuiltin("startswith", [Index, Literal])
                if let Expr::CallBuiltin { op: name, args } = e {
                    if matches!(name, BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains) && args.len() == 2 {
                        if let Expr::Index { expr: base, key } = &args[0] {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                    if let Expr::Literal(Literal::Str(arg)) = &args[1] {
                                        return Some(MixedCond::StrTest(field.clone(), name.name().to_string(), arg.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
                // .field == "str"
                if let Expr::BinOp { op: BinOp::Eq, lhs, rhs } = e {
                    if let Expr::Index { expr: base, key } = lhs.as_ref() {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                if let Expr::Literal(Literal::Str(val)) = rhs.as_ref() {
                                    return Some(MixedCond::StrTest(field.clone(), "eq".to_string(), val.clone()));
                                }
                            }
                        }
                    }
                }
                None
            }
            fn collect_mixed(e: &Expr, logic: &BinOp, out: &mut Vec<MixedCond>) -> bool {
                if let Expr::BinOp { op, lhs, rhs } = e {
                    if std::mem::discriminant(op) == std::mem::discriminant(logic) {
                        return collect_mixed(lhs, logic, out) && collect_mixed(rhs, logic, out);
                    }
                }
                if let Some(c) = extract_mixed_cond(e) {
                    out.push(c);
                    true
                } else {
                    false
                }
            }
            if let Expr::BinOp { op, .. } = cond.as_ref() {
                if matches!(op, BinOp::And | BinOp::Or) {
                    let mut conds = Vec::new();
                    if collect_mixed(cond, op, &mut conds) && conds.len() >= 2 {
                        // Only fire when mixed (has both numeric and string)
                        let has_num = conds.iter().any(|c| matches!(c, MixedCond::NumCmp(..)));
                        let has_str = conds.iter().any(|c| matches!(c, MixedCond::StrTest(..)));
                        if has_num && has_str {
                            return Some((*op, conds));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.field | test("regex"))` pattern.
    /// Returns (field_name, regex_pattern, flags_str) if detected.
    pub fn detect_select_field_regex_test(&self) -> Option<(String, String, Option<String>)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            if let Expr::Pipe { left, right } = cond.as_ref() {
                if let Expr::Index { expr: base, key } = left.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::RegexTest { input_expr, re, flags } = right.as_ref() {
                            if !matches!(input_expr.as_ref(), Expr::Input) { return None; }
                            if let Expr::Literal(Literal::Str(pattern)) = re.as_ref() {
                                let flags_str = match flags.as_ref() {
                                    Expr::Literal(Literal::Null) => None,
                                    Expr::Literal(Literal::Str(f)) => Some(f.clone()),
                                    _ => return None,
                                };
                                return Some((field.clone(), pattern.clone(), flags_str));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.field | test("re")) | value` pattern.
    /// Returns (cond_field, pattern, flags, output RemapExpr).
    pub fn detect_select_regex_then_value(&self) -> Option<(String, String, Option<String>, RemapExpr)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        let extract_regex_cond = |cond: &Expr| -> Option<(String, String, Option<String>)> {
            if let Expr::Pipe { left, right } = cond {
                if let Expr::Index { expr: base, key } = left.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::RegexTest { input_expr, re, flags } = right.as_ref() {
                            if !matches!(input_expr.as_ref(), Expr::Input) { return None; }
                            if let Expr::Literal(Literal::Str(pattern)) = re.as_ref() {
                                let flags_str = match flags.as_ref() {
                                    Expr::Literal(Literal::Null) => None,
                                    Expr::Literal(Literal::Str(f)) => Some(f.clone()),
                                    _ => return None,
                                };
                                return Some((field.clone(), pattern.clone(), flags_str));
                            }
                        }
                    }
                }
            }
            None
        };
        // Form 1: Pipe(select(.field|test("re")), output)
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some((field, pattern, flags)) = extract_regex_cond(cond) {
                        let rexpr = Self::classify_remap_value(right)?;
                        return Some((field, pattern, flags, rexpr));
                    }
                }
            }
        }
        // Form 2: IfThenElse { cond: .field|test("re"), then: output, else: empty }
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some((field, pattern, flags)) = extract_regex_cond(cond) {
                    let rexpr = Self::classify_remap_value(then_branch)?;
                    // Skip if output is identity (already handled by detect_select_field_regex_test)
                    if matches!(rexpr, RemapExpr::Field(ref f) if f == &field) { return None; }
                    return Some((field, pattern, flags, rexpr));
                }
            }
        }
        None
    }

    /// Detect `{a: .x, b: .y}` pattern (object construction from field access).
    /// Returns Vec of (output_key, input_field) pairs if detected.
    pub fn detect_field_remap(&self) -> Option<Vec<(String, String)>> {
        use crate::ir::{Expr, Literal, BinOp};
        let expr = self.detect_expr()?;
        // Helper to extract pairs from an ObjectConstruct
        fn extract_remap_pairs(expr: &Expr) -> Option<Vec<(String, String)>> {
            if let Expr::ObjectConstruct { pairs } = expr {
                let mut result = Vec::with_capacity(pairs.len());
                for (k, v) in pairs {
                    let key = if let Expr::Literal(Literal::Str(s)) = k {
                        s.clone()
                    } else { return None; };
                    if let Expr::Index { expr: base, key: field_key } = v {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = field_key.as_ref() {
                            result.push((key, f.clone()));
                        } else { return None; }
                    } else { return None; }
                }
                if result.is_empty() { return None; }
                return Some(result);
            }
            None
        }
        // Direct ObjectConstruct
        if let Some(r) = extract_remap_pairs(expr) { return Some(normalize_object_pairs(r)); }
        // {a:.x} + {b:.y} — merged object constructs
        if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = expr {
            if let (Some(mut left), Some(right)) = (extract_remap_pairs(lhs), extract_remap_pairs(rhs)) {
                left.extend(right);
                return Some(normalize_object_pairs(left));
            }
        }
        None
    }

    /// Detect `{a: .x, b: (.y * 2), c: (.x + .y)}` pattern — object construction with computed values.
    /// Each value can be: field ref, field op const, or field op field.
    /// Returns Vec of (output_key, RemapExpr) if detected.
    /// Only matches when detect_field_remap fails (i.e., at least one value is computed).
    pub fn detect_computed_remap(&self) -> Option<Vec<(String, RemapExpr)>> {
        use crate::ir::{Expr, Literal, BinOp};
        let expr = self.detect_expr()?;
        fn extract_computed_pairs(this: &Filter, expr: &Expr) -> Option<(Vec<(String, RemapExpr)>, bool)> {
            if let Expr::ObjectConstruct { pairs } = expr {
                if pairs.is_empty() { return None; }
                let mut result = Vec::with_capacity(pairs.len());
                let mut has_computed = false;
                for (k, v) in pairs {
                    let key = if let Expr::Literal(Literal::Str(s)) = k {
                        s.clone()
                    } else { return None; };
                    let rexpr = Filter::classify_remap_value(v)?;
                    if !matches!(rexpr, RemapExpr::Field(_)) { has_computed = true; }
                    result.push((key, rexpr));
                }
                return Some((result, has_computed));
            }
            let _ = this; // silence unused
            None
        }
        // Reject computed remaps with duplicate keys: `normalize_object_pairs`
        // collapses to last-wins, but jq evaluates every pair in source
        // order and aborts on the first error. If an earlier pair has a
        // computed (input-touching) value, that value's runtime error
        // would be silently elided when the later pair rebinds the same
        // key. Same family as #324 / #337; here the fold lives in
        // `detect_computed_remap` rather than `push_const_json` /
        // `const_expr_to_json`.
        let has_duplicate_keys = |pairs: &[(String, RemapExpr)]| -> bool {
            let mut seen = std::collections::HashSet::new();
            !pairs.iter().all(|(k, _)| seen.insert(k.clone()))
        };
        // Direct ObjectConstruct
        if let Some((result, has_computed)) = extract_computed_pairs(self, expr) {
            if !has_computed { return None; }
            if has_duplicate_keys(&result) { return None; }
            return Some(normalize_object_pairs(result));
        }
        // {a:.x,b:(.y*2)} + {c:.z} — merged object constructs
        if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = expr {
            if let (Some((mut left, lc)), Some((right, rc))) = (extract_computed_pairs(self, lhs), extract_computed_pairs(self, rhs)) {
                left.extend(right);
                if !(lc || rc) { return None; }
                if has_duplicate_keys(&left) { return None; }
                return Some(normalize_object_pairs(left));
            }
        }
        None
    }

    /// Detect standalone array collect: `[expr1, expr2, ...]` where each element is classifiable.
    /// Returns Vec<RemapExpr> for the elements.
    pub fn detect_standalone_array(&self) -> Option<Vec<RemapExpr>> {
        use crate::ir::Expr;
        let expr = self.detect_expr()?;
        if let Expr::Collect { generator } = expr {
            fn collect_comma_elements<'a>(expr: &'a Expr, result: &mut Vec<&'a Expr>) {
                match expr {
                    Expr::Comma { left, right } => {
                        collect_comma_elements(left, result);
                        collect_comma_elements(right, result);
                    }
                    _ => result.push(expr),
                }
            }
            let mut elements = Vec::new();
            collect_comma_elements(generator, &mut elements);
            if elements.len() < 2 { return None; } // single-element arrays not worth special-casing
            let mut rexprs = Vec::with_capacity(elements.len());
            for elem in &elements {
                rexprs.push(Self::classify_remap_value(elem)?);
            }
            return Some(rexprs);
        }
        None
    }

    /// Classify a single remap value expression.
    fn classify_remap_value(v: &crate::ir::Expr) -> Option<RemapExpr> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        // Literal value (string, number, null, true, false)
        if let Some(json_bytes) = const_expr_to_json(v) {
            return Some(RemapExpr::LiteralJson(json_bytes));
        }
        // .field
        if let Expr::Index { expr: base, key } = v {
            if matches!(base.as_ref(), Expr::Input) {
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                    return Some(RemapExpr::Field(f.clone()));
                }
            }
            return None;
        }
        // .field | ascii_upcase/downcase (beta-reduced: UnaryOp(AsciiUpcase/Downcase, Index(Input, field)))
        if let Expr::UnaryOp { op, operand } = v {
            let is_case = matches!(op, UnaryOp::AsciiUpcase | UnaryOp::AsciiDowncase);
            if is_case {
                if let Expr::Index { expr: base, key } = operand.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            return Some(RemapExpr::FieldStringCase(f.clone(), matches!(op, UnaryOp::AsciiUpcase)));
                        }
                    }
                }
            }
        }
        // .field | length (beta-reduced: UnaryOp(Length, Index(Input, field)))
        if let Expr::UnaryOp { op: UnaryOp::Length, operand } = v {
            if let Expr::Index { expr: base, key } = operand.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        return Some(RemapExpr::FieldLength(f.clone()));
                    }
                }
            }
        }
        // -.field (Negate(Index(Input, field)))
        if let Expr::Negate { operand } = v {
            if let Expr::Index { expr: base, key } = operand.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        return Some(RemapExpr::FieldNegate(f.clone()));
                    }
                }
            }
        }
        // .field | type (beta-reduced: UnaryOp(Type, Index(Input, field)))
        if let Expr::UnaryOp { op: UnaryOp::Type, operand } = v {
            if let Expr::Index { expr: base, key } = operand.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        return Some(RemapExpr::FieldType(f.clone()));
                    }
                }
            }
        }
        // .field | tostring (beta-reduced: UnaryOp(ToString, Index(Input, field)))
        if let Expr::UnaryOp { op: UnaryOp::ToString, operand } = v {
            if let Expr::Index { expr: base, key } = operand.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        return Some(RemapExpr::FieldToString(f.clone()));
                    }
                }
            }
            // .field op N | tostring (beta-reduced: UnaryOp(ToString, BinOp(op, Index, Num)))
            if let Expr::BinOp { op, lhs, rhs } = operand.as_ref() {
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                    if let Expr::Index { expr: base, key } = lhs.as_ref() {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                    if matches!(op, BinOp::Div | BinOp::Mod) && *n == 0.0 { return None; }
                                    return Some(RemapExpr::FieldOpConstToString(f.clone(), *op, *n));
                                }
                                // .f1 op .f2 | tostring
                                if let Expr::Index { expr: base2, key: key2 } = rhs.as_ref() {
                                    if matches!(base2.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                                            return Some(RemapExpr::FieldOpFieldToString(f.clone(), *op, f2.clone()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // compound arith | tostring
            {
                let mut fields = Vec::new();
                if let Some(arith) = Self::try_build_arith_expr(operand, &mut fields) {
                    if !fields.is_empty() {
                        return Some(RemapExpr::ArithToString(arith, fields));
                    }
                }
            }
        }
        // compound arith | sqrt/floor/ceil/fabs/round
        if let Expr::UnaryOp { op, operand } = v {
            let math_op = match op {
                UnaryOp::Sqrt => Some(MathUnary::Sqrt),
                UnaryOp::Floor => Some(MathUnary::Floor),
                UnaryOp::Ceil => Some(MathUnary::Ceil),
                UnaryOp::Fabs => Some(MathUnary::Fabs),
                UnaryOp::Round => Some(MathUnary::Round),
                _ => None,
            };
            if let Some(math_op) = math_op {
                let mut fields = Vec::new();
                if let Some(arith) = Self::try_build_arith_expr(operand, &mut fields) {
                    if !fields.is_empty() {
                        return Some(RemapExpr::ArithUnary(math_op, arith, fields));
                    }
                }
            }
        }
        // .field op N or .field op .field2
        if let Expr::BinOp { op, lhs, rhs } = v {
            // (cmp1) and/or (cmp2) — boolean compound
            if matches!(op, BinOp::And | BinOp::Or) {
                let l = Self::classify_remap_value(lhs);
                let r = Self::classify_remap_value(rhs);
                if let (Some(l), Some(r)) = (l, r) {
                    return Some(RemapExpr::BoolExpr(Box::new(l), *op, Box::new(r)));
                }
                return None;
            }
            let is_arith = matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod);
            let is_cmp = matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne);
            if !is_arith && !is_cmp { return None; }
            if let Expr::Index { expr: base, key } = lhs.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f1)) = key.as_ref() {
                    // .field op/cmp N
                    if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                        if is_arith {
                            if matches!(op, BinOp::Div | BinOp::Mod) && *n == 0.0 { return None; }
                            return Some(RemapExpr::FieldOpConst(f1.clone(), *op, *n));
                        } else {
                            return Some(RemapExpr::FieldCmpConst(f1.clone(), *op, *n));
                        }
                    }
                    // .field1 op/cmp .field2
                    if let Expr::Index { expr: base2, key: key2 } = rhs.as_ref() {
                        if !matches!(base2.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                            if is_arith {
                                return Some(RemapExpr::FieldOpField(f1.clone(), *op, f2.clone()));
                            } else {
                                return Some(RemapExpr::FieldCmpField(f1.clone(), *op, f2.clone()));
                            }
                        }
                    }
                }
            }
            // N op .field (e.g., 100 - .x), only for arithmetic
            if is_arith {
                if let Expr::Literal(Literal::Num(n, _)) = lhs.as_ref() {
                    if let Expr::Index { expr: base, key } = rhs.as_ref() {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                                return Some(RemapExpr::ConstOpField(*n, *op, f.clone()));
                            }
                        }
                    }
                }
            }
            // (compound_arith) cmp N — e.g. (.x % 2 == 0)
            if is_cmp {
                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                    let mut fields = Vec::new();
                    if let Some(arith) = Self::try_build_arith_expr(lhs, &mut fields) {
                        if !fields.is_empty() {
                            return Some(RemapExpr::ArithCmp(arith, *op, *n, fields));
                        }
                    }
                }
                // N cmp (compound_arith) — flip
                if let Expr::Literal(Literal::Num(n, _)) = lhs.as_ref() {
                    let mut fields = Vec::new();
                    if let Some(arith) = Self::try_build_arith_expr(rhs, &mut fields) {
                        if !fields.is_empty() {
                            // Flip: N cmp arith → arith flipped_cmp N
                            let flipped = match op {
                                BinOp::Gt => BinOp::Lt,
                                BinOp::Lt => BinOp::Gt,
                                BinOp::Ge => BinOp::Le,
                                BinOp::Le => BinOp::Ge,
                                _ => *op, // Eq, Ne are symmetric
                            };
                            return Some(RemapExpr::ArithCmp(arith, flipped, *n, fields));
                        }
                    }
                }
            }
        }
        // .field | Pipe patterns
        if let Expr::Pipe { left, right } = v {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        // .field | split(sep) | ...
                        if let Expr::Pipe { left: split_expr, right: tail_expr } = right.as_ref() {
                            if let Expr::CallBuiltin { op: sn, args: sa } = split_expr.as_ref() {
                                if *sn == BuiltinOp::Split && sa.len() == 1 {
                                    if let Expr::Literal(Literal::Str(sep)) = &sa[0] {
                                        // .field | split(sep) | length
                                        if matches!(tail_expr.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                                            return Some(RemapExpr::FieldSplitLength(field.clone(), sep.clone()));
                                        }
                                        // .field | split(sep) | .[N]
                                        if let Expr::Index { expr: ibase, key: ikey } = tail_expr.as_ref() {
                                            if matches!(ibase.as_ref(), Expr::Input) {
                                                if let Expr::Literal(Literal::Num(n, _)) = ikey.as_ref() {
                                                    return Some(RemapExpr::FieldSplitIndex(field.clone(), sep.clone(), *n as i32));
                                                }
                                                // Handle .[-N] parsed as Negate(Literal(N))
                                                if let Expr::Negate { operand } = ikey.as_ref() {
                                                    if let Expr::Literal(Literal::Num(n, _)) = operand.as_ref() {
                                                        return Some(RemapExpr::FieldSplitIndex(field.clone(), sep.clone(), -(*n as i32)));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // .field | builtin("arg") — string builtins
                        if let Expr::CallBuiltin { op: bn, args: ba } = right.as_ref() {
                            if ba.len() == 1 {
                                if let Expr::Literal(Literal::Str(arg)) = &ba[0] {
                                    let op = match bn {
                                        BuiltinOp::LtrimStr => Some(StrBuiltin::Ltrimstr),
                                        BuiltinOp::RtrimStr => Some(StrBuiltin::Rtrimstr),
                                        BuiltinOp::StartsWith => Some(StrBuiltin::Startswith),
                                        BuiltinOp::EndsWith => Some(StrBuiltin::Endswith),
                                        BuiltinOp::Index => Some(StrBuiltin::Index),
                                        BuiltinOp::Contains => Some(StrBuiltin::Contains),
                                        _ => None,
                                    };
                                    if let Some(op) = op {
                                        return Some(RemapExpr::FieldStrBuiltin(field.clone(), op, arg.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // [.field1, .field2] | min/max
        if let Expr::Pipe { left, right } = v {
            if let Expr::UnaryOp { op, operand } = right.as_ref() {
                if matches!(operand.as_ref(), Expr::Input) {
                    let is_max = match op {
                        UnaryOp::Max => Some(true),
                        UnaryOp::Min => Some(false),
                        _ => None,
                    };
                    if let Some(is_max) = is_max {
                        if let Expr::Collect { generator } = left.as_ref() {
                            if let Expr::Comma { left: cl, right: cr } = generator.as_ref() {
                                if let (
                                    Expr::Index { expr: base1, key: key1 },
                                    Expr::Index { expr: base2, key: key2 },
                                ) = (cl.as_ref(), cr.as_ref()) {
                                    if matches!(base1.as_ref(), Expr::Input) && matches!(base2.as_ref(), Expr::Input) {
                                        if let (
                                            Expr::Literal(Literal::Str(f1)),
                                            Expr::Literal(Literal::Str(f2)),
                                        ) = (key1.as_ref(), key2.as_ref()) {
                                            return Some(RemapExpr::FieldMinMax(f1.clone(), f2.clone(), is_max));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // .field | split(sep) | join(rep) — string replacement
        if let Expr::Pipe { left, right } = v {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::Pipe { left: split_expr, right: join_expr } = right.as_ref() {
                            if let Expr::CallBuiltin { op: sn, args: sa } = split_expr.as_ref() {
                                if *sn == BuiltinOp::Split && sa.len() == 1 {
                                    if let Expr::Literal(Literal::Str(sep)) = &sa[0] {
                                        if let Expr::CallBuiltin { op: jn, args: ja } = join_expr.as_ref() {
                                            if *jn == BuiltinOp::Join && ja.len() == 1 {
                                                if let Expr::Literal(Literal::Str(rep)) = &ja[0] {
                                                    return Some(RemapExpr::FieldSplitJoin(field.clone(), sep.clone(), rep.clone()));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Conditional chain: if .field cmp N then A elif ... else B end
        if let Expr::IfThenElse { .. } = v {
            if let Some((branches, else_out)) = Self::classify_remap_cond_chain(v) {
                if !branches.is_empty() {
                    return Some(RemapExpr::CondChain(branches, Box::new(else_out)));
                }
            }
        }
        // String interpolation: "\(.x):\(.y)" etc.
        if let Expr::StringInterpolation { parts } = v {
            use crate::ir::StringPart;
            let mut interp_parts = Vec::new();
            let mut has_field = false;
            for part in parts {
                match part {
                    StringPart::Literal(s) => {
                        interp_parts.push(InterpPart::Literal(s.clone()));
                    }
                    StringPart::Expr(Expr::Index { expr: base, key }) => {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            interp_parts.push(InterpPart::Field(f.clone()));
                            has_field = true;
                        } else { return None; }
                    }
                    _ => return None,
                }
            }
            if has_field {
                return Some(RemapExpr::StringInterp(interp_parts));
            }
        }
        // [expr1, expr2, ...] — array literal of remap values
        if let Expr::Collect { generator } = v {
            fn collect_comma_elements<'a>(expr: &'a Expr, result: &mut Vec<&'a Expr>) {
                match expr {
                    Expr::Comma { left, right } => {
                        collect_comma_elements(left, result);
                        collect_comma_elements(right, result);
                    }
                    _ => result.push(expr),
                }
            }
            let mut elements = Vec::new();
            collect_comma_elements(generator, &mut elements);
            let mut rexprs = Vec::with_capacity(elements.len());
            for elem in &elements {
                if let Some(rexpr) = Self::classify_remap_value(elem) {
                    rexprs.push(rexpr);
                } else {
                    return None;
                }
            }
            if !rexprs.is_empty() {
                return Some(RemapExpr::FieldArray(rexprs));
            }
        }
        // .field | .[from:to] (beta-reduced: Slice { expr: Index(Input, field), from, to })
        if let Expr::Slice { expr: base, from, to } = v {
            if let Expr::Index { expr: input, key } = base.as_ref() {
                if matches!(input.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        let from_val = match from {
                            Some(e) => match e.as_ref() {
                                Expr::Literal(Literal::Num(n, _)) => Some(*n as i64),
                                _ => return None,
                            },
                            None => None,
                        };
                        let to_val = match to {
                            Some(e) => match e.as_ref() {
                                Expr::Literal(Literal::Num(n, _)) => Some(*n as i64),
                                _ => return None,
                            },
                            None => None,
                        };
                        return Some(RemapExpr::FieldSlice(field.clone(), from_val, to_val));
                    }
                }
            }
        }
        // String add chain: .name + ":" + (.x | tostring) etc.
        {
            use crate::ir::UnaryOp;
            fn remap_tostring_arith(operand: &Expr, parts: &mut Vec<StringAddPart>) -> bool {
                if let Expr::Index { expr: base, key } = operand {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            parts.push(StringAddPart::FieldToString(f.clone()));
                            return true;
                        }
                    }
                }
                let mut arith_ops = Vec::new();
                let mut cur = operand;
                loop {
                    if let Expr::BinOp { op: aop, lhs, rhs } = cur {
                        if matches!(aop, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                // jq raises on a zero divisor; keep the chain on generic eval (#1063)
                                if matches!(aop, BinOp::Div | BinOp::Mod) && *n == 0.0 { break; }
                                arith_ops.push((*aop, *n));
                                cur = lhs.as_ref();
                                continue;
                            }
                        }
                    }
                    break;
                }
                if !arith_ops.is_empty() {
                    arith_ops.reverse();
                    if let Expr::Index { expr: base, key } = cur {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                                parts.push(StringAddPart::FieldArithToString(f.clone(), arith_ops));
                                return true;
                            }
                        }
                    }
                }
                false
            }
            fn collect_chain_rv(expr: &Expr, parts: &mut Vec<StringAddPart>) -> bool {
                match expr {
                    Expr::BinOp { op: BinOp::Add, lhs, rhs } => {
                        collect_chain_rv(lhs, parts) && collect_chain_rv(rhs, parts)
                    }
                    Expr::Index { expr: base, key } if matches!(base.as_ref(), Expr::Input) => {
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            parts.push(StringAddPart::Field(f.clone())); true
                        } else { false }
                    }
                    Expr::Literal(Literal::Str(s)) => {
                        parts.push(StringAddPart::Literal(s.clone())); true
                    }
                    Expr::UnaryOp { op: UnaryOp::ToString, operand } => {
                        remap_tostring_arith(operand, parts)
                    }
                    _ => false,
                }
            }
            let mut parts = Vec::new();
            if collect_chain_rv(v, &mut parts) && parts.len() >= 2
                && parts.iter().any(|p| !matches!(p, StringAddPart::Literal(_)))
            {
                return Some(RemapExpr::StringChain(parts));
            }
        }
        // Fallback: compound arithmetic expression tree over fields and constants
        {
            let mut fields = Vec::new();
            if let Some(arith) = Self::try_build_arith_expr(v, &mut fields) {
                if fields.len() >= 1 {
                    return Some(RemapExpr::Arith(arith, fields));
                }
            }
        }
        None
    }

    /// Classify a conditional chain (if-elif-else) as a remap value.
    fn classify_remap_cond_chain(v: &crate::ir::Expr) -> Option<(Vec<CondBranch>, BranchOutput)> {
        use crate::ir::{Expr, BinOp, Literal};

        fn expr_to_branch_output(e: &Expr) -> Option<BranchOutput> {
            // .field
            if let Expr::Index { expr: base, key } = e {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        return Some(BranchOutput::Field(f.clone()));
                    }
                }
                return None;
            }
            // Literal
            if let Some(json_bytes) = const_expr_to_json(e) {
                return Some(BranchOutput::Literal(json_bytes));
            }
            // ObjectConstruct → Remap
            if let Expr::ObjectConstruct { pairs } = e {
                if !pairs.is_empty() {
                    let mut result = Vec::with_capacity(pairs.len());
                    for (k, v) in pairs {
                        let key = if let Expr::Literal(Literal::Str(s)) = k { s.clone() } else { return None; };
                        let rexpr = Filter::classify_remap_value(v)?;
                        result.push((key, rexpr));
                    }
                    return Some(BranchOutput::Remap(result));
                }
            }
            // Fallback: try as computed value (e.g., .x - .y)
            if let Some(rexpr) = Filter::classify_remap_value(e) {
                return Some(BranchOutput::Computed(rexpr));
            }
            None
        }

        fn extract_cond(cond: &Expr) -> Option<(String, Vec<(BinOp, f64)>, BinOp, CondRhs)> {
            if let Expr::BinOp { op, lhs, rhs } = cond {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                    return None;
                }
                // Unwrap arithmetic chain from LHS
                let mut arith_ops = Vec::new();
                let mut cur = lhs.as_ref();
                loop {
                    if let Expr::BinOp { op: aop, lhs: al, rhs: ar } = cur {
                        if matches!(aop, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                            if let Expr::Literal(Literal::Num(n, _)) = ar.as_ref() {
                                // jq raises on a zero divisor; keep the chain on generic eval (#1063)
                                if matches!(aop, BinOp::Div | BinOp::Mod) && *n == 0.0 { break; }
                                arith_ops.push((*aop, *n));
                                cur = al.as_ref();
                                continue;
                            }
                        }
                    }
                    break;
                }
                arith_ops.reverse();
                if let Expr::Index { expr: base, key } = cur {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            // RHS: number or field
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((field.clone(), arith_ops, *op, CondRhs::Const(*n)));
                            }
                            if let Expr::Index { expr: base2, key: key2 } = rhs.as_ref() {
                                if matches!(base2.as_ref(), Expr::Input) {
                                    if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                                        return Some((field.clone(), arith_ops, *op, CondRhs::Field(f2.clone())));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            None
        }

        let mut branches = Vec::new();
        let mut cur = v;
        loop {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = cur {
                let (field, arith_ops, op, rhs) = extract_cond(cond)?;
                let output = expr_to_branch_output(then_branch)?;
                branches.push(CondBranch {
                    cond_field: field,
                    cond_arith_ops: arith_ops,
                    cond_op: op,
                    cond_rhs: rhs,
                    output,
                });
                cur = else_branch;
            } else {
                let else_out = expr_to_branch_output(cur)?;
                return Some((branches, else_out));
            }
        }
    }

    /// Try to build an ArithExpr from an expression tree.
    /// ArithExpr::Field(i) indexes into the `fields` vector.
    fn try_build_arith_expr(expr: &crate::ir::Expr, fields: &mut Vec<String>) -> Option<ArithExpr> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        match expr {
            Expr::Index { expr: base, key } if matches!(base.as_ref(), Expr::Input) => {
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                    let idx = if let Some(pos) = fields.iter().position(|x| x == f) {
                        pos
                    } else {
                        fields.push(f.clone());
                        fields.len() - 1
                    };
                    Some(ArithExpr::Field(idx))
                } else { None }
            }
            Expr::Literal(Literal::Num(n, _)) => Some(ArithExpr::Const(*n)),
            Expr::BinOp { op, lhs, rhs } if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) => {
                let l = Self::try_build_arith_expr(lhs, fields)?;
                let r = Self::try_build_arith_expr(rhs, fields)?;
                Some(ArithExpr::BinOp(*op, Box::new(l), Box::new(r)))
            }
            Expr::UnaryOp { op, operand } => {
                let math_op = match op {
                    UnaryOp::Floor => MathUnary::Floor,
                    UnaryOp::Ceil => MathUnary::Ceil,
                    UnaryOp::Sqrt => MathUnary::Sqrt,
                    UnaryOp::Fabs => MathUnary::Fabs,
                    UnaryOp::Round => MathUnary::Round,
                    _ => return None,
                };
                let inner = Self::try_build_arith_expr(operand, fields)?;
                Some(ArithExpr::Unary(math_op, Box::new(inner)))
            }
            _ => None,
        }
    }

    /// Detect `.field1 op .field2` pattern (binary arithmetic on two input fields).
    /// Returns (field1, op, field2) if detected.
    pub fn detect_field_binop(&self) -> Option<(String, crate::ir::BinOp, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::BinOp { op, lhs, rhs } = expr {
            if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
            if let Expr::Index { expr: base1, key: key1 } = lhs.as_ref() {
                if !matches!(base1.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f1)) = key1.as_ref() {
                    if let Expr::Index { expr: base2, key: key2 } = rhs.as_ref() {
                        if !matches!(base2.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                            return Some((f1.clone(), *op, f2.clone()));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `(.field1 op1 .field2) op2 const` pattern — two-field binop then constant op.
    /// Returns (field1, op1, field2, op2, const_val).
    pub fn detect_two_field_binop_const(&self) -> Option<(String, crate::ir::BinOp, String, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        // Form 2: Pipe(BinOp(.f1 op1 .f2), BinOp(op2, Input, Literal(N)))
        if let Expr::Pipe { left, right } = expr {
            if let Expr::BinOp { op: op1, lhs: inner_lhs, rhs: inner_rhs } = left.as_ref() {
                if matches!(op1, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                    if let Expr::Index { expr: base1, key: key1 } = inner_lhs.as_ref() {
                        if matches!(base1.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(f1)) = key1.as_ref() {
                                if let Expr::Index { expr: base2, key: key2 } = inner_rhs.as_ref() {
                                    if matches!(base2.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                                            if let Expr::BinOp { op: op2, lhs: binop_lhs, rhs: binop_rhs } = right.as_ref() {
                                                if matches!(op2, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                                                    if matches!(binop_lhs.as_ref(), Expr::Input) {
                                                        if let Expr::Literal(Literal::Num(n, _)) = binop_rhs.as_ref() {
                                                            return Some((f1.clone(), *op1, f2.clone(), *op2, *n));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Expr::BinOp { op: op2, lhs, rhs } = expr {
            if !matches!(op2, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                if let Expr::BinOp { op: op1, lhs: inner_lhs, rhs: inner_rhs } = lhs.as_ref() {
                    if !matches!(op1, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
                    if let Expr::Index { expr: base1, key: key1 } = inner_lhs.as_ref() {
                        if !matches!(base1.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f1)) = key1.as_ref() {
                            if let Expr::Index { expr: base2, key: key2 } = inner_rhs.as_ref() {
                                if !matches!(base2.as_ref(), Expr::Input) { return None; }
                                if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                                    return Some((f1.clone(), *op1, f2.clone(), *op2, *n));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field + "literal"` pattern (field access + string concatenation).
    /// Returns (field_name, suffix) if detected.
    pub fn detect_field_str_concat(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = expr {
            if let Expr::Index { expr: base, key } = lhs.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Literal(Literal::Str(suffix)) = rhs.as_ref() {
                        return Some((field.clone(), suffix.clone()));
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | unary_op` pattern (field access + unary op).
    /// Returns (field_name, op) if detected.
    /// Supports numeric ops (floor/ceil/sqrt/fabs/abs), tostring, and
    /// string ops (ascii_downcase/ascii_upcase).
    pub fn detect_field_unary_num(&self) -> Option<(String, crate::ir::UnaryOp)> {
        use crate::ir::{Expr, UnaryOp, Literal};
        let expr = self.detect_expr()?;
        // Utf8ByteLength is intentionally excluded: the raw-byte fast path
        // would dispatch it through the same code as length and silently
        // return a non-string-domain value (#159). Generic eval errors on
        // non-string input correctly.
        let is_supported = |op: &UnaryOp| matches!(op,
            UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Sqrt |
            UnaryOp::Fabs | UnaryOp::Abs | UnaryOp::ToString |
            UnaryOp::AsciiDowncase | UnaryOp::AsciiUpcase |
            UnaryOp::Length | UnaryOp::Explode);
        // Pipe form: .field | op
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::UnaryOp { op, operand } = right.as_ref() {
                            if matches!(operand.as_ref(), Expr::Input) && is_supported(op) {
                                return Some((field.clone(), *op));
                            }
                        }
                    }
                }
            }
        }
        // Beta-reduced form: op(.field) — from simplify_expr
        if let Expr::UnaryOp { op, operand } = expr {
            if is_supported(op) {
                if let Expr::Index { expr: base, key } = operand.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            return Some((field.clone(), *op));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | floor/ceil/round | arith_chain` pattern.
    /// Returns (field_name, unary_op, arith_steps) where arith_steps is [(op, const)].
    pub fn detect_field_unary_arith(&self) -> Option<(String, crate::ir::UnaryOp, Vec<(crate::ir::BinOp, f64)>)> {
        use crate::ir::{Expr, UnaryOp, BinOp, Literal};
        let expr = self.detect_expr()?;
        let is_numeric_unary = |op: &UnaryOp| matches!(op, UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Sqrt | UnaryOp::Fabs | UnaryOp::Abs | UnaryOp::Round | UnaryOp::Length);
        // Collect arith chain from the outermost pipe/binop
        fn collect_arith_tail(e: &Expr) -> Option<(Vec<(BinOp, f64)>, &Expr)> {
            // e is BinOp(inner, const) → arith step on top
            if let Expr::BinOp { op, lhs, rhs } = e {
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                    if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                        let (mut steps, inner) = collect_arith_tail(lhs)?;
                        steps.push((*op, *n));
                        return Some((steps, inner));
                    }
                }
            }
            // e is Pipe(inner, BinOp(Input, const))
            if let Expr::Pipe { left, right } = e {
                if let Expr::BinOp { op, lhs, rhs } = right.as_ref() {
                    if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                        if matches!(lhs.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                let (mut steps, inner) = collect_arith_tail(left)?;
                                steps.push((*op, *n));
                                return Some((steps, inner));
                            }
                        }
                    }
                }
            }
            Some((Vec::new(), e))
        }
        let (arith_steps, inner) = collect_arith_tail(expr)?;
        if arith_steps.is_empty() { return None; }
        // inner should be .field | unary
        // Pipe form: Pipe(.field, UnaryOp)
        if let Expr::Pipe { left, right } = inner {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::UnaryOp { op, operand } = right.as_ref() {
                            if matches!(operand.as_ref(), Expr::Input) && is_numeric_unary(op) {
                                return Some((field.clone(), *op, arith_steps));
                            }
                        }
                    }
                }
            }
        }
        // Beta-reduced form: UnaryOp(.field)
        if let Expr::UnaryOp { op, operand } = inner {
            if is_numeric_unary(op) {
                if let Expr::Index { expr: base, key } = operand.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            return Some((field.clone(), *op, arith_steps));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | startswith/endswith/ltrimstr/rtrimstr("str")` pattern.
    /// Returns (field_name, builtin_name, string_arg) if detected.
    pub fn detect_field_str_builtin(&self) -> Option<(String, String, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::CallBuiltin { op: name, args } = right.as_ref() {
                        if args.len() == 1 {
                            if matches!(name, BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::LtrimStr | BuiltinOp::RtrimStr | BuiltinOp::Split | BuiltinOp::Index | BuiltinOp::Rindex | BuiltinOp::Indices | BuiltinOp::Contains) {
                                if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                    return Some((field.clone(), name.name().to_string(), arg.clone()));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | index/rindex(str) op N`.
    /// Returns (field_name, search_str, is_rindex, arith_op, constant).
    pub fn detect_field_index_arith(&self) -> Option<(String, String, bool, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, Literal, BinOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::BinOp { op, lhs, rhs } = right.as_ref() {
                        if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) { return None; }
                        if let Expr::CallBuiltin { op: name, args } = lhs.as_ref() {
                            if (*name == BuiltinOp::Index || *name == BuiltinOp::Rindex) && args.len() == 1 {
                                if let Expr::Literal(Literal::Str(search)) = &args[0] {
                                    if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                        return Some((field.clone(), search.clone(), *name == BuiltinOp::Rindex, *op, *n));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | op1 | op2 | ...` chained string operations.
    /// Returns (field_name, [ops], terminal) with 1+ string ops + optional terminal,
    /// where the total chain length is 2+ (either 2+ string ops, or 1+ string ops + terminal).
    pub fn detect_field_string_chain(&self) -> Option<(String, Vec<StringChainOp>, StringChainTerminal)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        // Top level must be Pipe(.field, chain)
        if let Expr::Pipe { left, right } = expr {
            let field = if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() } else { return None; }
            } else { return None; };
            let mut ops = Vec::new();
            let terminal = Self::collect_string_chain_ops_with_terminal(right, &mut ops);
            let total = ops.len() + if matches!(terminal, StringChainTerminal::None) { 0 } else { 1 };
            if total >= 2 {
                return Some((field, ops, terminal));
            }
        }
        None
    }

    /// Recursively collect string ops from a right-associative pipe chain.
    /// Returns the terminal operation (if any) at the end of the chain.
    fn collect_string_chain_ops_with_terminal(expr: &crate::ir::Expr, ops: &mut Vec<StringChainOp>) -> StringChainTerminal {
        use crate::ir::Expr;
        match expr {
            Expr::Pipe { left, right } => {
                if Self::try_extract_string_op(left, ops) {
                    Self::collect_string_chain_ops_with_terminal(right, ops)
                } else {
                    StringChainTerminal::None
                }
            }
            _ => {
                // Try string op first, then try terminal
                if Self::try_extract_string_op(expr, ops) {
                    StringChainTerminal::None
                } else {
                    Self::try_extract_terminal(expr)
                }
            }
        }
    }

    fn try_extract_terminal(expr: &crate::ir::Expr) -> StringChainTerminal {
        use crate::ir::{Expr, Literal, UnaryOp};
        match expr {
            Expr::CallBuiltin { op: name, args } if args.len() == 1 => {
                if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                    match name {
                        BuiltinOp::StartsWith => return StringChainTerminal::Startswith(arg.clone()),
                        BuiltinOp::EndsWith => return StringChainTerminal::Endswith(arg.clone()),
                        BuiltinOp::Contains => return StringChainTerminal::Contains(arg.clone()),
                        BuiltinOp::Index => return StringChainTerminal::Index(arg.clone()),
                        _ => {}
                    }
                }
                StringChainTerminal::None
            }
            Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input) => {
                StringChainTerminal::Length
            }
            _ => StringChainTerminal::None,
        }
    }

    fn try_extract_string_op(expr: &crate::ir::Expr, ops: &mut Vec<StringChainOp>) -> bool {
        use crate::ir::{Expr, Literal, UnaryOp};
        match expr {
            Expr::UnaryOp { op: UnaryOp::AsciiDowncase, operand } if matches!(operand.as_ref(), Expr::Input) => {
                ops.push(StringChainOp::AsciiDowncase); true
            }
            Expr::UnaryOp { op: UnaryOp::AsciiUpcase, operand } if matches!(operand.as_ref(), Expr::Input) => {
                ops.push(StringChainOp::AsciiUpcase); true
            }
            Expr::CallBuiltin { op: name, args } if args.len() == 1 => {
                if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                    match name {
                        BuiltinOp::LtrimStr => { ops.push(StringChainOp::Ltrimstr(arg.clone())); true }
                        BuiltinOp::RtrimStr => { ops.push(StringChainOp::Rtrimstr(arg.clone())); true }
                        _ => false,
                    }
                } else { false }
            }
            // split(sep) | join(rep) — fused as SplitJoin
            Expr::Pipe { left, right } => {
                if let Expr::CallBuiltin { op: sn, args: sa } = left.as_ref() {
                    if *sn == BuiltinOp::Split && sa.len() == 1 {
                        if let Expr::Literal(Literal::Str(sep)) = &sa[0] {
                            // Check for split | join
                            if let Expr::CallBuiltin { op: jn, args: ja } = right.as_ref() {
                                if *jn == BuiltinOp::Join && ja.len() == 1 {
                                    if let Expr::Literal(Literal::Str(rep)) = &ja[0] {
                                        ops.push(StringChainOp::SplitJoin(sep.clone(), rep.clone()));
                                        return true;
                                    }
                                }
                            }
                            // Check for split | reverse | join
                            if let Expr::Pipe { left: rev, right: join_expr } = right.as_ref() {
                                if matches!(rev.as_ref(), Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                                    if let Expr::CallBuiltin { op: jn, args: ja } = join_expr.as_ref() {
                                        if *jn == BuiltinOp::Join && ja.len() == 1 {
                                            if let Expr::Literal(Literal::Str(rep)) = &ja[0] {
                                                ops.push(StringChainOp::SplitReverseJoin(sep.clone(), rep.clone()));
                                                return true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Detect `select(.field | string_test) | {computed_remap}`.
    /// String tests: startswith/endswith/contains("str"), .field == "str", .field != "str".
    /// Returns (field, test_name, test_arg, remap_pairs).
    pub fn detect_select_str_then_computed_remap(&self) -> Option<(String, String, String, Vec<(String, RemapExpr)>)> {
        use crate::ir::{Expr, Literal, BinOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            // Right: ObjectConstruct (computed remap)
            let remap = if let Expr::ObjectConstruct { pairs } = right.as_ref() {
                if pairs.is_empty() { return None; }
                let mut result = Vec::with_capacity(pairs.len());
                for (k, v) in pairs {
                    let key = if let Expr::Literal(Literal::Str(s)) = k { s.clone() } else { return None; };
                    let rexpr = Self::classify_remap_value(v)?;
                    result.push((key, rexpr));
                }
                result
            } else { return None; };
            // Left: select(cond) = IfThenElse { cond, then: Input, else: Empty }
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
                if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
                // Form A: .field == "str" / .field != "str"
                if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                    if matches!(op, BinOp::Eq | BinOp::Ne) {
                        if let Expr::Index { expr: base, key } = lhs.as_ref() {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                    if let Expr::Literal(Literal::Str(val)) = rhs.as_ref() {
                                        let test_type = if matches!(op, BinOp::Eq) { "eq" } else { "ne" };
                                        return Some((field.clone(), test_type.to_string(), val.clone(), remap));
                                    }
                                }
                            }
                        }
                    }
                }
                // Form B: .field | startswith/endswith/contains("str")
                if let Expr::Pipe { left: pl, right: pr } = cond.as_ref() {
                    if let Expr::Index { expr: base, key } = pl.as_ref() {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                if let Expr::CallBuiltin { op: name, args } = pr.as_ref() {
                                    if matches!(name, BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains) && args.len() == 1 {
                                        if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                            return Some((field.clone(), name.name().to_string(), arg.clone(), remap));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | test("regex")` pattern.
    /// Returns (field_name, regex_pattern, flags_str) if detected.
    pub fn detect_field_test(&self) -> Option<(String, String, Option<String>)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::RegexTest { input_expr, re, flags } = right.as_ref() {
                        if !matches!(input_expr.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(pattern)) = re.as_ref() {
                            let flags_str = match flags.as_ref() {
                                Expr::Literal(Literal::Null) => None,
                                Expr::Literal(Literal::Str(f)) => Some(f.clone()),
                                _ => return None,
                            };
                            return Some((field.clone(), pattern.clone(), flags_str));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | scan("regex")` pattern (generator, multiple outputs per input).
    /// Returns (field_name, regex_pattern) if detected.
    pub fn detect_field_scan(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::RegexScan { input_expr, re, flags } = right.as_ref() {
                        if !matches!(input_expr.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(pattern)) = re.as_ref() {
                            // Only support no-flags case for simplicity
                            if matches!(flags.as_ref(), Expr::Literal(Literal::Null)) {
                                return Some((field.clone(), pattern.clone()));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | match("regex")` pattern.
    /// Returns (field_name, regex_pattern, flags_opt) if detected.
    pub fn detect_field_match(&self) -> Option<(String, String, Option<String>)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::RegexMatch { input_expr, re, flags } = right.as_ref() {
                        if !matches!(input_expr.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(pattern)) = re.as_ref() {
                            let flags_str = match flags.as_ref() {
                                Expr::Literal(Literal::Null) => None,
                                Expr::Literal(Literal::Str(f)) => Some(f.clone()),
                                _ => return None,
                            };
                            return Some((field.clone(), pattern.clone(), flags_str));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | capture("regex")` pattern.
    /// Returns (field_name, regex_pattern, flags_opt) if detected.
    pub fn detect_field_capture(&self) -> Option<(String, String, Option<String>)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::RegexCapture { input_expr, re, flags } = right.as_ref() {
                        if !matches!(input_expr.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(pattern)) = re.as_ref() {
                            let flags_str = match flags.as_ref() {
                                Expr::Literal(Literal::Null) => None,
                                Expr::Literal(Literal::Str(f)) => Some(f.clone()),
                                _ => return None,
                            };
                            return Some((field.clone(), pattern.clone(), flags_str));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | @base64` (or other simple format operations).
    /// Returns (field_name, format_name) if detected.
    pub fn detect_field_format(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        use crate::ir::FormatKind;
        let is_supported = |kind: &FormatKind| matches!(
            kind,
            FormatKind::Base64 | FormatKind::Uri | FormatKind::Html | FormatKind::Json | FormatKind::Text
        );
        // Pipe form: .field | @format
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::Format { kind, expr: fmt_expr } = right.as_ref() {
                            if matches!(fmt_expr.as_ref(), Expr::Input) && is_supported(kind) {
                                return Some((field.clone(), kind.name().to_string()));
                            }
                        }
                    }
                }
            }
        }
        // Beta-reduced form: @format(.field)
        if let Expr::Format { kind, expr: fmt_expr } = expr {
            if is_supported(kind) {
                if let Expr::Index { expr: base, key } = fmt_expr.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            return Some((field.clone(), kind.name().to_string()));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | gsub("pattern"; "replacement")` or `.field | sub("pattern"; "replacement")`.
    /// Returns (field_name, is_global, regex_pattern, replacement, flags) if detected.
    pub fn detect_field_gsub(&self) -> Option<(String, bool, String, String, Option<String>)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    let (is_global, input_expr, re, tostr, flags) = match right.as_ref() {
                        Expr::RegexGsub { input_expr, re, tostr, flags } => (true, input_expr, re, tostr, flags),
                        Expr::RegexSub { input_expr, re, tostr, flags } => (false, input_expr, re, tostr, flags),
                        _ => return None,
                    };
                    if !matches!(input_expr.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(pattern)) = re.as_ref() {
                        if let Expr::Literal(Literal::Str(replacement)) = tostr.as_ref() {
                            let flags_str = match flags.as_ref() {
                                Expr::Literal(Literal::Null) => None,
                                Expr::Literal(Literal::Str(f)) => Some(f.clone()),
                                _ => return None,
                            };
                            return Some((field.clone(), is_global, pattern.clone(), replacement.clone(), flags_str));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | ascii_case | gsub/sub(re; rep)` pattern.
    /// Returns (field, is_upper, is_global, pattern, replacement, flags).
    pub fn detect_field_case_gsub(&self) -> Option<(String, bool, bool, String, String, Option<String>)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: case_expr, right: gsub_expr } = right.as_ref() {
                        let is_upper = match case_expr.as_ref() {
                            Expr::UnaryOp { op: UnaryOp::AsciiUpcase, operand } if matches!(operand.as_ref(), Expr::Input) => true,
                            Expr::UnaryOp { op: UnaryOp::AsciiDowncase, operand } if matches!(operand.as_ref(), Expr::Input) => false,
                            _ => return None,
                        };
                        let (is_global, input_expr, re, tostr, flags) = match gsub_expr.as_ref() {
                            Expr::RegexGsub { input_expr, re, tostr, flags } => (true, input_expr, re, tostr, flags),
                            Expr::RegexSub { input_expr, re, tostr, flags } => (false, input_expr, re, tostr, flags),
                            _ => return None,
                        };
                        if !matches!(input_expr.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(pattern)) = re.as_ref() {
                            if let Expr::Literal(Literal::Str(replacement)) = tostr.as_ref() {
                                let flags_str = match flags.as_ref() {
                                    Expr::Literal(Literal::Null) => None,
                                    Expr::Literal(Literal::Str(f)) => Some(f.clone()),
                                    _ => return None,
                                };
                                return Some((field.clone(), is_upper, is_global, pattern.clone(), replacement.clone(), flags_str));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | ascii_downcase/upcase | test("regex")`.
    /// Returns (field, is_upper, regex_pattern).
    pub fn detect_field_case_test(&self) -> Option<(String, bool, String)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: case_expr, right: test_expr } = right.as_ref() {
                        let is_upper = match case_expr.as_ref() {
                            Expr::UnaryOp { op: UnaryOp::AsciiUpcase, operand } if matches!(operand.as_ref(), Expr::Input) => true,
                            Expr::UnaryOp { op: UnaryOp::AsciiDowncase, operand } if matches!(operand.as_ref(), Expr::Input) => false,
                            _ => return None,
                        };
                        // Match test(regex) in both RegexTest and CallBuiltin forms
                        match test_expr.as_ref() {
                            Expr::RegexTest { input_expr, re, flags } => {
                                if !matches!(input_expr.as_ref(), Expr::Input) { return None; }
                                if let Expr::Literal(Literal::Str(pattern)) = re.as_ref() {
                                    if matches!(flags.as_ref(), Expr::Literal(Literal::Null) | Expr::Literal(Literal::Str(_))) {
                                        return Some((field.clone(), is_upper, pattern.clone()));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | ltrimstr("prefix") | tonumber` pattern.
    /// Returns (field_name, prefix) if detected.
    /// Returns (field, prefix, arith_ops).
    /// arith_ops is a list of (op, const) to apply after tonumber.
    pub fn detect_field_ltrimstr_tonumber(&self) -> Option<(String, String, Vec<(crate::ir::BinOp, f64)>)> {
        use crate::ir::{Expr, Literal, UnaryOp, BinOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: mid, right: rr } = right.as_ref() {
                        if let Expr::CallBuiltin { op: name, args } = mid.as_ref() {
                            if *name == BuiltinOp::LtrimStr && args.len() == 1 {
                                if let Expr::Literal(Literal::Str(prefix)) = &args[0] {
                                    // tonumber with no further ops
                                    if let Expr::UnaryOp { op: UnaryOp::ToNumber, operand } = rr.as_ref() {
                                        if matches!(operand.as_ref(), Expr::Input) {
                                            return Some((field.clone(), prefix.clone(), Vec::new()));
                                        }
                                    }
                                    // tonumber | arith chain (e.g., tonumber | . * 2 | . + 1)
                                    // Beta-reduced: BinOp(op, UnaryOp(ToNumber, Input), Num)
                                    // or Pipe(tonumber, arith_chain)
                                    let mut arith_ops = Vec::new();
                                    let mut cur: &Expr = rr.as_ref();
                                    // Peel off piped arithmetic: Pipe(lhs, BinOp(op, Input, N))
                                    loop {
                                        if let Expr::Pipe { left: pl, right: pr } = cur {
                                            cur = pl.as_ref();
                                            // pr should be BinOp(op, Input, N)
                                            if let Expr::BinOp { op, lhs, rhs } = pr.as_ref() {
                                                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                                                    if matches!(lhs.as_ref(), Expr::Input) {
                                                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                                            // jq raises on a zero divisor; keep the row on generic eval (#1063)
                                                            if matches!(op, BinOp::Div | BinOp::Mod) && *n == 0.0 { break; }
                                                            arith_ops.push((*op, *n));
                                                            continue;
                                                        }
                                                    }
                                                }
                                            }
                                            return None;
                                        }
                                        break;
                                    }
                                    // Beta-reduced: BinOp(op, BinOp(...(UnaryOp(ToNumber, Input))...), N)
                                    let mut bcur: &Expr = cur;
                                    let mut b_ops = Vec::new();
                                    loop {
                                        if let Expr::BinOp { op, lhs, rhs } = bcur {
                                            if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                                                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                                    b_ops.push((*op, *n));
                                                    bcur = lhs.as_ref();
                                                    continue;
                                                }
                                            }
                                        }
                                        break;
                                    }
                                    if let Expr::UnaryOp { op: UnaryOp::ToNumber, operand } = bcur {
                                        if matches!(operand.as_ref(), Expr::Input) {
                                            b_ops.reverse();
                                            b_ops.extend(arith_ops);
                                            return Some((field.clone(), prefix.clone(), b_ops));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `[.field, "lit", ...] | join("sep")` pattern.
    /// Each element must be a field access (.field) or a string literal.
    /// Returns (parts: Vec<(is_literal, name)>, separator).
    pub fn detect_array_join(&self) -> Option<(Vec<(bool, String)>, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            // right must be join("sep")
            if let Expr::CallBuiltin { op: name, args } = right.as_ref() {
                if *name != BuiltinOp::Join || args.len() != 1 { return None; }
                if let Expr::Literal(Literal::Str(sep)) = &args[0] {
                    // left must be [expr1, expr2, ...]
                    if let Expr::Collect { generator } = left.as_ref() {
                        let mut parts = Vec::new();
                        fn collect_comma_parts(e: &Expr, out: &mut Vec<(bool, String)>) -> bool {
                            use crate::ir::UnaryOp;
                            match e {
                                Expr::Comma { left, right } => {
                                    collect_comma_parts(left, out) && collect_comma_parts(right, out)
                                }
                                Expr::Index { expr: base, key } => {
                                    if !matches!(base.as_ref(), Expr::Input) { return false; }
                                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                        out.push((false, field.clone()));
                                        true
                                    } else { false }
                                }
                                // tostring(.field) — same as .field for join purposes
                                Expr::UnaryOp { op: UnaryOp::ToString, operand } => {
                                    if let Expr::Index { expr: base, key } = operand.as_ref() {
                                        if !matches!(base.as_ref(), Expr::Input) { return false; }
                                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                            out.push((false, field.clone()));
                                            return true;
                                        }
                                    }
                                    false
                                }
                                Expr::Literal(Literal::Str(s)) => {
                                    out.push((true, s.clone()));
                                    true
                                }
                                _ => false,
                            }
                        }
                        if collect_comma_parts(generator, &mut parts) && !parts.is_empty() {
                            return Some((parts, sep.clone()));
                        }
                    }
                } else { return None; }
            }
        }
        None
    }

    /// Detect `[remap_exprs] | map(tostring) | join("sep")` pattern.
    /// Returns (remap_exprs, separator) if detected.
    pub fn detect_remap_tostring_join(&self) -> Option<(Vec<RemapExpr>, String)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // Structure: Pipe(Collect(gen), Pipe(map_tostring, join(sep)))
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Collect { generator } = left.as_ref() {
                if let Expr::Pipe { left: map_expr, right: join_expr } = right.as_ref() {
                    // Check join(sep)
                    let sep = if let Expr::CallBuiltin { op: name, args } = join_expr.as_ref() {
                        if *name != BuiltinOp::Join || args.len() != 1 { return None; }
                        if let Expr::Literal(Literal::Str(s)) = &args[0] { s.clone() } else { return None; }
                    } else { return None; };
                    // Check map(tostring) = [.[] | tostring]
                    let is_map_tostring = if let Expr::Collect { generator: mg } = map_expr.as_ref() {
                        if let Expr::Pipe { left: ml, right: mr } = mg.as_ref() {
                            matches!(ml.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input))
                                && matches!(mr.as_ref(), Expr::UnaryOp { op: UnaryOp::ToString, operand } if matches!(operand.as_ref(), Expr::Input))
                        } else { false }
                    } else { false };
                    if !is_map_tostring { return None; }
                    // Collect remap expressions from the generator
                    let mut exprs = Vec::new();
                    if collect_comma_remap(generator, &mut exprs) && !exprs.is_empty() {
                        return Some((exprs, sep));
                    }
                }
            }
        }
        None
    }

    /// Detect `.field / N | floor` or `.field % N` pattern (field + binop + optional unary).
    /// Returns (field_name, binop, constant, optional unary op) if detected.
    /// Returns (field, op, const, unary_op, const_on_left).
    /// When const_on_left is true, the expression is `N op .field` instead of `.field op N`.
    pub fn detect_field_binop_const_unary(&self) -> Option<(String, crate::ir::BinOp, f64, Option<crate::ir::UnaryOp>, bool)> {
        use crate::ir::{Expr, BinOp, UnaryOp, Literal};
        let expr = self.detect_expr()?;
        // Case 1: `.field / N | floor` — Pipe { left: BinOp(.field, N), right: UnaryOp }
        if let Expr::Pipe { left, right } = expr {
            if let Expr::BinOp { op, lhs, rhs } = left.as_ref() {
                if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            if let Expr::UnaryOp { op: uop, operand } = right.as_ref() {
                                if !matches!(operand.as_ref(), Expr::Input) { return None; }
                                if matches!(uop, UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Sqrt | UnaryOp::Fabs | UnaryOp::Abs) {
                                    return Some((field.clone(), *op, *n, Some(*uop), false));
                                }
                            }
                        }
                    }
                }
            }
        }
        // Case 2: beta-reduced `.field / N | floor` → UnaryOp(floor, BinOp(.field, N))
        if let Expr::UnaryOp { op: uop, operand } = expr {
            if matches!(uop, UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Sqrt | UnaryOp::Fabs | UnaryOp::Abs) {
                if let Expr::BinOp { op, lhs, rhs } = operand.as_ref() {
                    if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                        if let Expr::Index { expr: base, key } = lhs.as_ref() {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                    if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                        return Some((field.clone(), *op, *n, Some(*uop), false));
                                    }
                                }
                            }
                        }
                        // Beta-reduced N op .field | unary
                        if let Expr::Index { expr: base, key } = rhs.as_ref() {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                    if let Expr::Literal(Literal::Num(n, _)) = lhs.as_ref() {
                                        return Some((field.clone(), *op, *n, Some(*uop), true));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Case 3: `.field op N` — top-level BinOp (all arithmetic ops)
        if let Expr::BinOp { op, lhs, rhs } = expr {
            if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                if !matches!(op, BinOp::Div | BinOp::Mod) || *n != 0.0 {
                                    return Some((field.clone(), *op, *n, None, false));
                                }
                            }
                        }
                    }
                }
                // Case 4: `N op .field` — constant on left (e.g. `100 - .x`)
                if let Expr::Literal(Literal::Num(n, _)) = lhs.as_ref() {
                    if let Expr::Index { expr: base, key } = rhs.as_ref() {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                return Some((field.clone(), *op, *n, None, true));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect chained field arithmetic: `.field op1 N1 op2 N2 ...` (e.g. `.x * 2 + 1`).
    /// Returns (field_name, [(op, val), ...]) if detected. Only matches chains ≥2 ops.
    pub fn detect_field_arith_chain(&self) -> Option<(String, Vec<(crate::ir::BinOp, f64)>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        // Walk the left-nested BinOp chain: BinOp(op2, BinOp(op1, .field, N1), N2)
        let mut ops = Vec::new();
        let mut cur = expr;
        loop {
            if let Expr::BinOp { op, lhs, rhs } = cur {
                if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                    if matches!(op, BinOp::Div | BinOp::Mod) && *n == 0.0 { return None; }
                    ops.push((*op, *n));
                    cur = lhs.as_ref();
                } else {
                    return None;
                }
            } else {
                break;
            }
        }
        if ops.len() < 2 { return None; } // Single op is handled by detect_field_binop_const_unary
        ops.reverse(); // Inner-first to outer → execution order
        // cur should be .field
        if let Expr::Index { expr: base, key } = cur {
            if !matches!(base.as_ref(), Expr::Input) { return None; }
            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                return Some((field.clone(), ops));
            }
        }
        None
    }

    /// Detect `.field arith_chain | tostring` — arithmetic chain followed by tostring.
    /// Returns (field, ops) where ops is the arithmetic chain.
    pub fn detect_field_arith_chain_tostring(&self) -> Option<(String, Vec<(crate::ir::BinOp, f64)>)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // Beta-reduced: UnaryOp(ToString, BinOp(Add, BinOp(Mul, .field, N), N2))
        if let Expr::UnaryOp { op: UnaryOp::ToString, operand } = expr {
            let mut ops = Vec::new();
            let mut cur = operand.as_ref();
            loop {
                if let Expr::BinOp { op, lhs, rhs } = cur {
                    if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
                    if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                        if matches!(op, BinOp::Div | BinOp::Mod) && *n == 0.0 { return None; }
                        ops.push((*op, *n));
                        cur = lhs.as_ref();
                    } else { return None; }
                } else { break; }
            }
            if ops.is_empty() { return None; }
            ops.reverse();
            if let Expr::Index { expr: base, key } = cur {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        return Some((field.clone(), ops));
                    }
                }
            }
        }
        // Non-reduced: Pipe(arith_chain, UnaryOp(ToString, Input))
        if let Expr::Pipe { left, right } = expr {
            if matches!(right.as_ref(), Expr::UnaryOp { op: UnaryOp::ToString, operand } if matches!(operand.as_ref(), Expr::Input)) {
                let mut ops = Vec::new();
                let mut cur = left.as_ref();
                loop {
                    if let Expr::BinOp { op, lhs, rhs } = cur {
                        if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            if matches!(op, BinOp::Div | BinOp::Mod) && *n == 0.0 { return None; }
                            ops.push((*op, *n));
                            cur = lhs.as_ref();
                        } else { return None; }
                    } else { break; }
                }
                if ops.is_empty() { return None; }
                ops.reverse();
                if let Expr::Index { expr: base, key } = cur {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            return Some((field.clone(), ops));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `(.field1 op .field2) | tostring` — field-field binop piped to tostring.
    /// Returns (field1, op, field2) if detected.
    pub fn detect_field_binop_tostring(&self) -> Option<(String, crate::ir::BinOp, String)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // Beta-reduced: UnaryOp(ToString, BinOp(.f1, op, .f2))
        if let Expr::UnaryOp { op: UnaryOp::ToString, operand } = expr {
            if let Expr::BinOp { op, lhs, rhs } = operand.as_ref() {
                if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
                if let Expr::Index { expr: base1, key: key1 } = lhs.as_ref() {
                    if matches!(base1.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(f1)) = key1.as_ref() {
                            if let Expr::Index { expr: base2, key: key2 } = rhs.as_ref() {
                                if matches!(base2.as_ref(), Expr::Input) {
                                    if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                                        return Some((f1.clone(), *op, f2.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Non-reduced: Pipe(BinOp(.f1, op, .f2), UnaryOp(ToString, Input))
        if let Expr::Pipe { left, right } = expr {
            if matches!(right.as_ref(), Expr::UnaryOp { op: UnaryOp::ToString, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if let Expr::BinOp { op, lhs, rhs } = left.as_ref() {
                    if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
                    if let Expr::Index { expr: base1, key: key1 } = lhs.as_ref() {
                        if matches!(base1.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(f1)) = key1.as_ref() {
                                if let Expr::Index { expr: base2, key: key2 } = rhs.as_ref() {
                                    if matches!(base2.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                                            return Some((f1.clone(), *op, f2.clone()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | split("") | reverse | join("")` — string reversal pattern.
    /// Returns field name if detected.
    pub fn detect_field_str_reverse(&self) -> Option<String> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // Right-associative: Pipe(.field, Pipe(split(""), Pipe(Reverse, join(""))))
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: split_expr, right: rest } = right.as_ref() {
                        if let Expr::CallBuiltin { op: name, args } = split_expr.as_ref() {
                            if *name == BuiltinOp::Split && args.len() == 1 {
                                if let Expr::Literal(Literal::Str(sep)) = &args[0] {
                                    if sep.is_empty() {
                                        if let Expr::Pipe { left: rev_expr, right: join_expr } = rest.as_ref() {
                                            if matches!(rev_expr.as_ref(), Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                                                if let Expr::CallBuiltin { op: jn, args: ja } = join_expr.as_ref() {
                                                    if *jn == BuiltinOp::Join && ja.len() == 1 {
                                                        if let Expr::Literal(Literal::Str(js)) = &ja[0] {
                                                            if js.is_empty() {
                                                                return Some(field.clone());
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | split(sep) | reverse | join(sep2)` — split-reverse-join pattern.
    /// Returns (field_name, split_sep, join_sep).
    pub fn detect_field_split_reverse_join(&self) -> Option<(String, String, String)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: split_expr, right: rest } = right.as_ref() {
                        if let Expr::CallBuiltin { op: name, args } = split_expr.as_ref() {
                            if *name == BuiltinOp::Split && args.len() == 1 {
                                if let Expr::Literal(Literal::Str(sep)) = &args[0] {
                                    if let Expr::Pipe { left: rev_expr, right: join_expr } = rest.as_ref() {
                                        if matches!(rev_expr.as_ref(), Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                                            if let Expr::CallBuiltin { op: jn, args: ja } = join_expr.as_ref() {
                                                if *jn == BuiltinOp::Join && ja.len() == 1 {
                                                    if let Expr::Literal(Literal::Str(js)) = &ja[0] {
                                                        if !sep.is_empty() {
                                                            return Some((field.clone(), sep.clone(), js.clone()));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | ascii_upcase/downcase | split(s) | join(r)`.
    /// Returns (field, is_upper, split_sep, join_sep).
    pub fn detect_field_case_split_join(&self) -> Option<(String, bool, String, String)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: case_expr, right: rest } = right.as_ref() {
                        let is_upper = match case_expr.as_ref() {
                            Expr::UnaryOp { op: UnaryOp::AsciiUpcase, operand } if matches!(operand.as_ref(), Expr::Input) => true,
                            Expr::UnaryOp { op: UnaryOp::AsciiDowncase, operand } if matches!(operand.as_ref(), Expr::Input) => false,
                            _ => return None,
                        };
                        if let Expr::Pipe { left: split_expr, right: join_expr } = rest.as_ref() {
                            if let Expr::CallBuiltin { op: name, args } = split_expr.as_ref() {
                                if *name == BuiltinOp::Split && args.len() == 1 {
                                    if let Expr::Literal(Literal::Str(sep)) = &args[0] {
                                        if let Expr::CallBuiltin { op: jn, args: ja } = join_expr.as_ref() {
                                            if *jn == BuiltinOp::Join && ja.len() == 1 {
                                                if let Expr::Literal(Literal::Str(js)) = &ja[0] {
                                                    if !sep.is_empty() {
                                                        return Some((field.clone(), is_upper, sep.clone(), js.clone()));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | ascii_upcase/downcase | split(s)`.
    /// Returns (field, is_upper, split_sep).
    pub fn detect_field_case_split(&self) -> Option<(String, bool, String)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: case_expr, right: split_expr } = right.as_ref() {
                        let is_upper = match case_expr.as_ref() {
                            Expr::UnaryOp { op: UnaryOp::AsciiUpcase, operand } if matches!(operand.as_ref(), Expr::Input) => true,
                            Expr::UnaryOp { op: UnaryOp::AsciiDowncase, operand } if matches!(operand.as_ref(), Expr::Input) => false,
                            _ => return None,
                        };
                        if let Expr::CallBuiltin { op: name, args } = split_expr.as_ref() {
                            if *name == BuiltinOp::Split && args.len() == 1 {
                                if let Expr::Literal(Literal::Str(sep)) = &args[0] {
                                    if !sep.is_empty() {
                                        return Some((field.clone(), is_upper, sep.clone()));
                                    }
                                }
                            }
                        }
                    }
                    // Also handle beta-reduced form: CallBuiltin(split, [UnaryOp(case, .field)])
                    if let Expr::CallBuiltin { op: name, args } = right.as_ref() {
                        if *name == BuiltinOp::Split && args.len() == 1 {
                            if let Expr::Literal(Literal::Str(_sep)) = &args[0] {
                                // This form doesn't include the case op, skip
                            }
                        }
                    }
                }
            }
        }
        // Beta-reduced: CallBuiltin(split, [UnaryOp(case, Index(.field))])
        if let Expr::CallBuiltin { op: name, args } = expr {
            if *name == BuiltinOp::Split && args.len() == 1 {
                if let Expr::Literal(Literal::Str(sep)) = &args[0] {
                    if !sep.is_empty() {
                        // The input to split would be the case-converted field — check operand
                        // This form would be: split(sep) with input being case(.field)
                        // Not typical, skip
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | ascii_upcase/downcase | split("s") | .[N]` pattern.
    /// Returns (field_name, is_upper, separator, index) if detected.
    pub fn detect_field_case_split_nth(&self) -> Option<(String, bool, String, i64)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // Pattern: Pipe(Index(.field), Pipe(case, Pipe(split, .[N])))
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: case_expr, right: rest } = right.as_ref() {
                        let is_upper = match case_expr.as_ref() {
                            Expr::UnaryOp { op: UnaryOp::AsciiUpcase, operand } if matches!(operand.as_ref(), Expr::Input) => true,
                            Expr::UnaryOp { op: UnaryOp::AsciiDowncase, operand } if matches!(operand.as_ref(), Expr::Input) => false,
                            _ => return None,
                        };
                        if let Expr::Pipe { left: split_expr, right: idx_expr } = rest.as_ref() {
                            if let Expr::CallBuiltin { op: name, args } = split_expr.as_ref() {
                                if *name == BuiltinOp::Split && args.len() == 1 {
                                    if let Expr::Literal(Literal::Str(sep)) = &args[0] {
                                        if !sep.is_empty() {
                                            if let Expr::Index { expr: ibase, key: ikey } = idx_expr.as_ref() {
                                                if matches!(ibase.as_ref(), Expr::Input) {
                                                    if let Expr::Literal(Literal::Num(n, _)) = ikey.as_ref() {
                                                        let idx = *n as i64;
                                                        return Some((field.clone(), is_upper, sep.clone(), idx));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.f1 cmp .f2) | value` — field-field select then computed value.
    /// Returns (field1, op, field2, value_rexpr).
    pub fn detect_select_ff_cmp_then_value(&self) -> Option<(String, crate::ir::BinOp, String, RemapExpr)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, BinOp, String, RemapExpr)> {
            let rexpr = Self::classify_remap_value(output)?;
            if let Expr::BinOp { op, lhs, rhs } = cond {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                if let (Expr::Index { expr: b1, key: k1 }, Expr::Index { expr: b2, key: k2 }) = (lhs.as_ref(), rhs.as_ref()) {
                    if !matches!(b1.as_ref(), Expr::Input) || !matches!(b2.as_ref(), Expr::Input) { return None; }
                    if let (Expr::Literal(Literal::Str(f1)), Expr::Literal(Literal::Str(f2))) = (k1.as_ref(), k2.as_ref()) {
                        return Some((f1.clone(), *op, f2.clone(), rexpr));
                    }
                }
            }
            None
        };
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.field1 cmp .field2) | {computed_remap}` — select with field-field comparison + computed remap.
    /// Returns (field1, op, field2, remap_entries).
    pub fn detect_select_ff_cmp_then_computed_remap(&self) -> Option<(String, crate::ir::BinOp, String, Vec<(String, RemapExpr)>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, BinOp, String, Vec<(String, RemapExpr)>)> {
            if let Expr::ObjectConstruct { pairs } = output {
                if pairs.is_empty() { return None; }
                let mut result = Vec::with_capacity(pairs.len());
                for (k, v) in pairs {
                    let key = if let Expr::Literal(Literal::Str(s)) = k { s.clone() } else { return None; };
                    let rexpr = Self::classify_remap_value(v)?;
                    result.push((key, rexpr));
                }
                if let Expr::BinOp { op, lhs, rhs } = cond {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                        return None;
                    }
                    if let Expr::Index { expr: base1, key: key1 } = lhs.as_ref() {
                        if !matches!(base1.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f1)) = key1.as_ref() {
                            if let Expr::Index { expr: base2, key: key2 } = rhs.as_ref() {
                                if !matches!(base2.as_ref(), Expr::Input) { return None; }
                                if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                                    return Some((f1.clone(), *op, f2.clone(), result));
                                }
                            }
                        }
                    }
                }
            }
            None
        };
        // Form 1: Pipe(select(.f1 > .f2), {computed_remap})
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        // Form 2: if .f1 > .f2 then {computed_remap} else empty end
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect a general numeric expression over multiple fields.
    /// Matches patterns like `.x + .y * 2`, `(.x + .y) / 2`, `.x * .x + .y * .y`.
    /// Returns (field_names, arith_expr) where arith_expr uses field indices.
    /// Only matches when simpler detectors don't (multi-field or complex trees).
    pub fn detect_numeric_expr(&self) -> Option<(Vec<String>, ArithExpr)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let mut fields: Vec<String> = Vec::new();
        fn build_arith(expr: &Expr, fields: &mut Vec<String>) -> Option<ArithExpr> {
            use crate::ir::UnaryOp;
            match expr {
                Expr::BinOp { op, lhs, rhs } => {
                    if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
                    let l = build_arith(lhs, fields)?;
                    let r = build_arith(rhs, fields)?;
                    Some(ArithExpr::BinOp(*op, Box::new(l), Box::new(r)))
                }
                Expr::UnaryOp { op, operand } => {
                    let math_op = match op {
                        UnaryOp::Floor => MathUnary::Floor,
                        UnaryOp::Ceil => MathUnary::Ceil,
                        UnaryOp::Sqrt => MathUnary::Sqrt,
                        UnaryOp::Fabs => MathUnary::Fabs,
                        UnaryOp::Round => MathUnary::Round,
                        _ => return None,
                    };
                    let inner = build_arith(operand, fields)?;
                    Some(ArithExpr::Unary(math_op, Box::new(inner)))
                }
                Expr::Index { expr: base, key } => {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        let idx = if let Some(pos) = fields.iter().position(|f| f == field) {
                            pos
                        } else {
                            fields.push(field.clone());
                            fields.len() - 1
                        };
                        Some(ArithExpr::Field(idx))
                    } else { None }
                }
                Expr::Literal(Literal::Num(n, _)) => Some(ArithExpr::Const(*n)),
                _ => None,
            }
        }
        let arith = build_arith(expr, &mut fields)?;
        if fields.is_empty() { return None; }
        // For single-field, only match complex exprs (e.g. .x * .x + 1)
        // Simple single-field exprs are already handled by field_binop/field_arith_chain
        if fields.len() == 1 {
            // Must have field used multiple times (otherwise simpler detectors handle it)
            fn count_field_refs(e: &ArithExpr) -> usize {
                match e {
                    ArithExpr::Field(_) => 1,
                    ArithExpr::Const(_) => 0,
                    ArithExpr::BinOp(_, l, r) => count_field_refs(l) + count_field_refs(r),
                    ArithExpr::Unary(_, inner) => count_field_refs(inner),
                }
            }
            if count_field_refs(&arith) < 2 { return None; }
        }
        Some((fields, arith))
    }

    /// Detect compound arithmetic with math unary: `(arith) | sqrt/floor/ceil/fabs/round`.
    /// Returns (fields, arith_expr, math_unary) if detected.
    pub fn detect_numeric_expr_unary(&self) -> Option<(Vec<String>, ArithExpr, MathUnary)> {
        use crate::ir::{Expr, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::UnaryOp { op, operand } = expr {
            let math_op = match op {
                UnaryOp::Sqrt => Some(MathUnary::Sqrt),
                UnaryOp::Floor => Some(MathUnary::Floor),
                UnaryOp::Ceil => Some(MathUnary::Ceil),
                UnaryOp::Fabs => Some(MathUnary::Fabs),
                UnaryOp::Round => Some(MathUnary::Round),
                _ => None,
            }?;
            let mut fields = Vec::new();
            let arith = Self::try_build_arith_expr(operand, &mut fields)?;
            if fields.is_empty() { return None; }
            return Some((fields, arith, math_op));
        }
        None
    }

    /// Detect two-field numeric comparison: `.x > .y`, `.x == .y`, etc.
    /// Returns (field1, cmp_op, field2) if detected.
    pub fn detect_field_field_cmp(&self) -> Option<(String, crate::ir::BinOp, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::BinOp { op, lhs, rhs } = expr {
            if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
            if let Expr::Index { expr: base1, key: key1 } = lhs.as_ref() {
                if !matches!(base1.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f1)) = key1.as_ref() {
                    if let Expr::Index { expr: base2, key: key2 } = rhs.as_ref() {
                        if !matches!(base2.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                            return Some((f1.clone(), *op, f2.clone()));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field cmp N` producing boolean output (not in select context).
    /// Returns (field, op, value) if detected.
    pub fn detect_field_const_cmp(&self) -> Option<(String, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::BinOp { op, lhs, rhs } = expr {
            if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
            if let Expr::Index { expr: base, key } = lhs.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                        return Some((field.clone(), *op, *n));
                    }
                }
            }
        }
        None
    }

    /// Detect `(.field * 2 + 1) cmp N` — arith chain then comparison.
    /// Returns (field, arith_ops, cmp_op, threshold).
    pub fn detect_arith_chain_cmp(&self) -> Option<(String, Vec<(crate::ir::BinOp, f64)>, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::BinOp { op, lhs, rhs } = expr {
            if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
            if let Expr::Literal(Literal::Num(threshold, _)) = rhs.as_ref() {
                // LHS should be an arith chain: BinOp(op2, BinOp(op1, .field, N1), N2)
                let mut ops = Vec::new();
                let mut cur = lhs.as_ref();
                loop {
                    if let Expr::BinOp { op: aop, lhs: al, rhs: ar } = cur {
                        if !matches!(aop, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
                        if let Expr::Literal(Literal::Num(n, _)) = ar.as_ref() {
                            ops.push((*aop, *n));
                            cur = al.as_ref();
                        } else {
                            return None;
                        }
                    } else {
                        break;
                    }
                }
                ops.reverse();
                // Leaf: plain .field
                if let Expr::Index { expr: base, key } = cur {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            if !ops.is_empty() {
                                return Some((field.clone(), ops, *op, *threshold));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.f1 cmp1 N1 and/or .f2 cmp2 N2` producing boolean output.
    /// Returns (conjunct, Vec<(field, op, threshold)>).
    pub fn detect_compound_field_cmp(&self) -> Option<(crate::ir::BinOp, Vec<(String, crate::ir::BinOp, f64)>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let extract_cmp = |e: &Expr| -> Option<(String, BinOp, f64)> {
            if let Expr::BinOp { op, lhs, rhs } = e {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            return Some((field.clone(), *op, *n));
                        }
                    }
                }
            }
            None
        };
        if let Expr::BinOp { op: conjunct @ (BinOp::And | BinOp::Or), lhs, rhs } = expr {
            let mut cmps = Vec::new();
            // Flatten nested and/or of same type
            fn collect_cmps(e: &Expr, conjunct: BinOp, extract: &dyn Fn(&Expr) -> Option<(String, BinOp, f64)>, out: &mut Vec<(String, BinOp, f64)>) -> bool {
                if let Expr::BinOp { op, lhs, rhs } = e {
                    if std::mem::discriminant(op) == std::mem::discriminant(&conjunct) {
                        return collect_cmps(lhs, conjunct, extract, out) && collect_cmps(rhs, conjunct, extract, out);
                    }
                }
                if let Some(cmp) = extract(e) { out.push(cmp); true } else { false }
            }
            if collect_cmps(lhs, *conjunct, &extract_cmp, &mut cmps) && collect_cmps(rhs, *conjunct, &extract_cmp, &mut cmps) {
                if cmps.len() >= 2 {
                    return Some((*conjunct, cmps));
                }
            }
        }
        None
    }

    /// Detect string interpolation with field accesses: `"\(.f1)lit\(.f2)..."`.
    /// Returns Vec<(is_literal, content)> where content is either the literal text
    /// or the field name for interpolation parts.
    pub fn detect_string_interp_fields(&self) -> Option<Vec<(bool, String)>> {
        use crate::ir::{Expr, Literal, StringPart};
        let expr = self.detect_expr()?;
        if let Expr::StringInterpolation { parts } = expr {
            let mut result = Vec::new();
            for part in parts {
                match part {
                    StringPart::Literal(s) => {
                        result.push((true, s.clone()));
                    }
                    StringPart::Expr(Expr::Index { expr: base, key }) => {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            result.push((false, field.clone()));
                        } else {
                            return None;
                        }
                    }
                    _ => return None,
                }
            }
            if result.iter().any(|(is_lit, _)| !is_lit) {
                return Some(result);
            }
        }
        None
    }

    /// Detect string concatenation chains: `.name + ": " + (.x | tostring)`.
    /// Returns parts: (is_literal, is_tostring, text_or_field_name).
    /// Parts are in concatenation order.
    pub fn detect_string_add_chain(&self) -> Option<Vec<StringAddPart>> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        fn collect_tostring_arith(operand: &Expr, parts: &mut Vec<StringAddPart>) -> bool {
            // Simple: .field | tostring
            if let Expr::Index { expr: base, key } = operand {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        parts.push(StringAddPart::FieldToString(f.clone()));
                        return true;
                    }
                }
            }
            // Arithmetic chain: .field * N + M ... | tostring
            let mut arith_ops = Vec::new();
            let mut cur = operand;
            loop {
                if let Expr::BinOp { op: aop, lhs, rhs } = cur {
                    if matches!(aop, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            // jq raises on a zero divisor; keep the chain on generic eval (#1063)
                            if matches!(aop, BinOp::Div | BinOp::Mod) && *n == 0.0 { break; }
                            arith_ops.push((*aop, *n));
                            cur = lhs.as_ref();
                            continue;
                        }
                    }
                }
                break;
            }
            if !arith_ops.is_empty() {
                arith_ops.reverse();
                if let Expr::Index { expr: base, key } = cur {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            parts.push(StringAddPart::FieldArithToString(f.clone(), arith_ops));
                            return true;
                        }
                    }
                }
            }
            false
        }
        fn collect(expr: &Expr, parts: &mut Vec<StringAddPart>) -> bool {
            match expr {
                Expr::BinOp { op: BinOp::Add, lhs, rhs } => {
                    if !collect(lhs, parts) { return false; }
                    if !collect(rhs, parts) { return false; }
                    true
                }
                Expr::Index { expr: base, key } if matches!(base.as_ref(), Expr::Input) => {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        parts.push(StringAddPart::Field(f.clone()));
                        true
                    } else { false }
                }
                Expr::Literal(Literal::Str(s)) => {
                    parts.push(StringAddPart::Literal(s.clone()));
                    true
                }
                Expr::UnaryOp { op: UnaryOp::ToString, operand } => {
                    collect_tostring_arith(operand, parts)
                }
                _ => false,
            }
        }
        let mut parts = Vec::new();
        if collect(expr, &mut parts) && parts.len() >= 2
            && parts.iter().any(|p| !matches!(p, StringAddPart::Literal(_)))
        {
            Some(parts)
        } else {
            None
        }
    }

    /// Detect `length` applied directly to input.
    pub fn is_length(&self) -> bool {
        use crate::ir::{Expr, UnaryOp};
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        // Direct: `length`
        //
        // The `to_entries | length` / `keys | length` / `keys_unsorted | length`
        // shortcuts were removed (#220): each prefix op has a stricter type
        // contract than `length` (e.g. `null | keys` errors, `null | length` is 0),
        // so collapsing them changes observable behaviour for non-iterable input.
        matches!(expr, Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input))
    }

    /// Detect `keys` applied directly to input.
    pub fn is_keys(&self) -> bool {
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        matches!(expr, crate::ir::Expr::UnaryOp { op: crate::ir::UnaryOp::Keys, operand } if matches!(operand.as_ref(), crate::ir::Expr::Input))
    }

    /// Detect `del(.field)` applied directly to input.
    /// Returns the field name to delete.
    pub fn detect_del_field(&self) -> Option<String> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::CallBuiltin { op: name, args } = expr {
            if *name == BuiltinOp::Del && args.len() == 1 {
                if let Expr::Index { expr: base, key } = &args[0] {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            return Some(field.clone());
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.field cmp N) | del(.field1, ...)`.
    /// Returns (cmp_field, op, threshold, del_fields).
    pub fn detect_select_cmp_del(&self) -> Option<(String, crate::ir::BinOp, f64, Vec<String>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        // select is desugared to if-then-else: IfThenElse { cond, then: del(...), else: empty }
        // or Pipe { left: IfThenElse { cond, then: ., else: empty }, right: del(...) }
        let (cond, del_expr) = if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    (cond.as_ref(), right.as_ref())
                } else { return None; }
            } else { return None; }
        } else if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                (cond.as_ref(), then_branch.as_ref())
            } else { return None; }
        } else { return None; };
        // Parse condition: .field cmp N
        let (field, op, threshold) = if let Expr::BinOp { op, lhs, rhs } = cond {
            if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
            if let Expr::Index { expr: base, key } = lhs.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                    if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                        (f.clone(), *op, *n)
                    } else { return None; }
                } else { return None; }
            } else { return None; }
        } else { return None; };
        // Parse del expression
        if let Expr::CallBuiltin { op: name, args } = del_expr {
            if *name != BuiltinOp::Del || args.len() != 1 { return None; }
            let mut del_fields = Vec::new();
            fn collect_del_fields(expr: &Expr, fields: &mut Vec<String>) -> bool {
                match expr {
                    Expr::Comma { left, right } => {
                        collect_del_fields(left, fields) && collect_del_fields(right, fields)
                    }
                    Expr::Index { expr: base, key } => {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                fields.push(field.clone());
                                return true;
                            }
                        }
                        false
                    }
                    _ => false,
                }
            }
            if collect_del_fields(&args[0], &mut del_fields) && !del_fields.is_empty() {
                return Some((field, op, threshold, del_fields));
            }
        }
        None
    }

    /// Detect `select(.field | startswith/endswith/test("str")) | del(.fields)`.
    /// Returns (cmp_field, str_op, str_arg, del_fields).
    pub fn detect_select_str_del(&self) -> Option<(String, String, String, Vec<String>)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        let (cond, del_expr) = if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    (cond.as_ref(), right.as_ref())
                } else { return None; }
            } else { return None; }
        } else if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                (cond.as_ref(), then_branch.as_ref())
            } else { return None; }
        } else { return None; };
        // Parse condition: .field | startswith/endswith/test("str")
        let (field, str_op, str_arg) = if let Expr::Pipe { left, right } = cond {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                    if let Expr::CallBuiltin { op: name, args } = right.as_ref() {
                        if args.len() == 1 {
                            if let Expr::Literal(Literal::Str(s)) = &args[0] {
                                match name {
                                    BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains => {
                                        (f.clone(), name.name().to_string(), s.clone())
                                    }
                                    _ => return None,
                                }
                            } else { return None; }
                        } else { return None; }
                    } else { return None; }
                } else { return None; }
            } else { return None; }
        } else if let Expr::BinOp { op: crate::ir::BinOp::Eq, lhs, rhs } = cond {
            // .field == "str"
            if let Expr::Index { expr: base, key } = lhs.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        if let Expr::Literal(Literal::Str(s)) = rhs.as_ref() {
                            (f.clone(), "eq".to_string(), s.clone())
                        } else { return None; }
                    } else { return None; }
                } else { return None; }
            } else { return None; }
        } else { return None; };
        // Parse del expression
        if let Expr::CallBuiltin { op: name, args } = del_expr {
            if *name != BuiltinOp::Del || args.len() != 1 { return None; }
            let mut del_fields = Vec::new();
            fn collect_del(expr: &Expr, fields: &mut Vec<String>) -> bool {
                match expr {
                    Expr::Comma { left, right } => collect_del(left, fields) && collect_del(right, fields),
                    Expr::Index { expr: base, key } => {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                fields.push(field.clone());
                                return true;
                            }
                        }
                        false
                    }
                    _ => false,
                }
            }
            if collect_del(&args[0], &mut del_fields) && !del_fields.is_empty() {
                return Some((field, str_op, str_arg, del_fields));
            }
        }
        None
    }

    /// Detect `select(.field cmp N) | .+{key: literal, ...}`.
    /// Returns (cmp_field, op, threshold, merge_pairs: Vec<(key, json_value_bytes)>).
    pub fn detect_select_cmp_merge(&self) -> Option<(String, crate::ir::BinOp, f64, Vec<(String, Vec<u8>)>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        // Extract (cond, merge_expr) from either:
        //   Pipe(IfThenElse{cond,.,empty}, BinOp(Add, ., {..}))  — unsimplified
        //   BinOp(Add, IfThenElse{cond,.,empty}, {..})           — simplified (Input substituted)
        let (cond, obj_pairs) = if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if !matches!(then_branch.as_ref(), Expr::Input) || !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
                if let Expr::BinOp { op: BinOp::Add | BinOp::Mul, lhs, rhs } = right.as_ref() {
                    if !matches!(lhs.as_ref(), Expr::Input) { return None; }
                    if let Expr::ObjectConstruct { pairs } = rhs.as_ref() {
                        (cond.as_ref(), pairs)
                    } else { return None; }
                } else { return None; }
            } else { return None; }
        } else if let Expr::BinOp { op: BinOp::Add | BinOp::Mul, lhs, rhs } = expr {
            // Simplified form: Add(IfThenElse{cond, Input, Empty}, ObjectConstruct)
            if let Expr::IfThenElse { cond, then_branch, else_branch } = lhs.as_ref() {
                if !matches!(then_branch.as_ref(), Expr::Input) || !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
                if let Expr::ObjectConstruct { pairs } = rhs.as_ref() {
                    (cond.as_ref(), pairs)
                } else { return None; }
            } else { return None; }
        } else { return None; };
        // Parse condition: .field cmp N
        let (field, op, threshold) = if let Expr::BinOp { op, lhs, rhs } = cond {
            if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
            if let Expr::Index { expr: base, key } = lhs.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                    if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                        (f.clone(), *op, *n)
                    } else { return None; }
                } else { return None; }
            } else { return None; }
        } else { return None; };
        // Parse merge pairs: {key: literal, ...}
        let mut merge_pairs = Vec::new();
        for (k, v) in obj_pairs {
            let key = if let Expr::Literal(Literal::Str(s)) = k { s.clone() } else { return None; };
            if let Some(json_bytes) = const_expr_to_json(v) {
                merge_pairs.push((key, json_bytes));
            } else { return None; }
        }
        if !merge_pairs.is_empty() {
            return Some((field, op, threshold, merge_pairs));
        }
        None
    }

    /// Detect `select(.field | startswith/endswith/test("str")) | .+{key: literal, ...}`.
    /// Returns (field, str_op, str_arg, merge_pairs).
    pub fn detect_select_str_merge(&self) -> Option<(String, String, String, Vec<(String, Vec<u8>)>)> {
        use crate::ir::{Expr, Literal, BinOp};
        let expr = self.detect_expr()?;
        // Extract (cond, obj_pairs) from either:
        //   Pipe(IfThenElse{cond,.,empty}, BinOp(Add, ., {..}))  — unsimplified
        //   BinOp(Add, IfThenElse{cond,.,empty}, {..})           — simplified
        let (cond, obj_pairs) = if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if !matches!(then_branch.as_ref(), Expr::Input) || !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
                if let Expr::BinOp { op: BinOp::Add | BinOp::Mul, lhs, rhs } = right.as_ref() {
                    if !matches!(lhs.as_ref(), Expr::Input) { return None; }
                    if let Expr::ObjectConstruct { pairs } = rhs.as_ref() {
                        (cond.as_ref(), pairs)
                    } else { return None; }
                } else { return None; }
            } else { return None; }
        } else if let Expr::BinOp { op: BinOp::Add | BinOp::Mul, lhs, rhs } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = lhs.as_ref() {
                if !matches!(then_branch.as_ref(), Expr::Input) || !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
                if let Expr::ObjectConstruct { pairs } = rhs.as_ref() {
                    (cond.as_ref(), pairs)
                } else { return None; }
            } else { return None; }
        } else { return None; };
        // Parse condition: .field | startswith/endswith/test("str") or .field == "str"
        let (field, str_op, str_arg) = if let Expr::Pipe { left, right } = cond {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                    if let Expr::CallBuiltin { op: name, args } = right.as_ref() {
                        if args.len() == 1 {
                            if let Expr::Literal(Literal::Str(s)) = &args[0] {
                                match name {
                                    BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains => {
                                        (f.clone(), name.name().to_string(), s.clone())
                                    }
                                    _ => return None,
                                }
                            } else { return None; }
                        } else { return None; }
                    } else if let Expr::RegexTest { input_expr, re, flags } = right.as_ref() {
                        if !matches!(input_expr.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(pattern)) = re.as_ref() {
                            if matches!(flags.as_ref(), Expr::Literal(Literal::Null) | Expr::Literal(Literal::Str(_))) {
                                (f.clone(), "test".to_string(), pattern.clone())
                            } else { return None; }
                        } else { return None; }
                    } else { return None; }
                } else { return None; }
            } else { return None; }
        } else if let Expr::BinOp { op: BinOp::Eq, lhs, rhs } = cond {
            if let Expr::Index { expr: base, key } = lhs.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        if let Expr::Literal(Literal::Str(s)) = rhs.as_ref() {
                            (f.clone(), "eq".to_string(), s.clone())
                        } else { return None; }
                    } else { return None; }
                } else { return None; }
            } else { return None; }
        } else { return None; };
        // Parse merge pairs
        let mut merge_pairs = Vec::new();
        for (k, v) in obj_pairs {
            let key = if let Expr::Literal(Literal::Str(s)) = k { s.clone() } else { return None; };
            if let Some(json_bytes) = const_expr_to_json(v) {
                merge_pairs.push((key, json_bytes));
            } else { return None; }
        }
        if !merge_pairs.is_empty() {
            return Some((field, str_op, str_arg, merge_pairs));
        }
        None
    }

    /// Detect `del(.field1, .field2, ...)` — multi-field deletion.
    /// Returns list of field names to delete.
    pub fn detect_del_fields(&self) -> Option<Vec<String>> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::CallBuiltin { op: name, args } = expr {
            if *name == BuiltinOp::Del && args.len() == 1 {
                let mut fields = Vec::new();
                fn collect_fields(expr: &Expr, fields: &mut Vec<String>) -> bool {
                    match expr {
                        Expr::Comma { left, right } => {
                            collect_fields(left, fields) && collect_fields(right, fields)
                        }
                        Expr::Index { expr: base, key } => {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                    fields.push(field.clone());
                                    return true;
                                }
                            }
                            false
                        }
                        _ => false,
                    }
                }
                if collect_fields(&args[0], &mut fields) && fields.len() >= 2 {
                    return Some(fields);
                }
            }
        }
        None
    }

    /// Detect `type` applied directly to input.
    pub fn is_type(&self) -> bool {
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        matches!(expr, crate::ir::Expr::UnaryOp { op: crate::ir::UnaryOp::Type, operand } if matches!(operand.as_ref(), crate::ir::Expr::Input))
    }

    /// Detect `has("field")` applied directly to input.
    /// Returns the field name if this is `has("literal_string")`.
    pub fn detect_has_field(&self) -> Option<String> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::CallBuiltin { op: name, args } = expr {
            if *name == BuiltinOp::Has && args.len() == 1 {
                if let Expr::Literal(Literal::Str(field)) = &args[0] {
                    return Some(field.clone());
                }
            }
        }
        None
    }

    /// Detect `has("a") and has("b") [and ...]` or `has("a") or has("b") [or ...]`.
    /// Returns (fields, is_and) where is_and=true means AND, false means OR.
    pub fn detect_has_multi_field(&self) -> Option<(Vec<String>, bool)> {
        use crate::ir::{Expr, Literal, BinOp};
        let expr = self.detect_expr()?;
        fn extract_has_chain(e: &Expr) -> Option<(Vec<String>, bool)> {
            if let Expr::BinOp { op: op @ (BinOp::And | BinOp::Or), .. } = e {
                let is_and = matches!(op, BinOp::And);
                let mut fields = Vec::new();
                fn collect(e: &Expr, fields: &mut Vec<String>, is_and: bool) -> bool {
                    if let Expr::BinOp { op, lhs, rhs } = e {
                        let same_op = if is_and { matches!(op, BinOp::And) } else { matches!(op, BinOp::Or) };
                        if same_op {
                            return collect(lhs, fields, is_and) && collect(rhs, fields, is_and);
                        }
                    }
                    if let Expr::CallBuiltin { op: name, args } = e {
                        if *name == BuiltinOp::Has && args.len() == 1 {
                            if let Expr::Literal(Literal::Str(f)) = &args[0] {
                                fields.push(f.clone());
                                return true;
                            }
                        }
                    }
                    false
                }
                if collect(e, &mut fields, is_and) && fields.len() >= 2 {
                    return Some((fields, is_and));
                }
            }
            None
        }
        extract_has_chain(expr)
    }

    /// Detect `select(has("a") and has("b") [and ...])` or with `or`.
    /// Returns (fields, is_and) if matched.
    pub fn detect_select_has_multi(&self) -> Option<(Vec<String>, bool)> {
        use crate::ir::Expr;
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            // Also handle single has: select(has("a"))
            if let Expr::CallBuiltin { op: name, args } = cond.as_ref() {
                if *name == BuiltinOp::Has && args.len() == 1 {
                    if let crate::ir::Literal::Str(f) = match &args[0] {
                        Expr::Literal(l) => l,
                        _ => return None,
                    } {
                        return Some((vec![f.clone()], true));
                    }
                }
            }
            // Try multi-has chain
            use crate::ir::{Literal, BinOp};
            fn collect_has(e: &Expr, fields: &mut Vec<String>, is_and: bool) -> bool {
                if let Expr::BinOp { op, lhs, rhs } = e {
                    let same_op = if is_and { matches!(op, BinOp::And) } else { matches!(op, BinOp::Or) };
                    if same_op {
                        return collect_has(lhs, fields, is_and) && collect_has(rhs, fields, is_and);
                    }
                }
                if let Expr::CallBuiltin { op: name, args } = e {
                    if *name == BuiltinOp::Has && args.len() == 1 {
                        if let Expr::Literal(Literal::Str(f)) = &args[0] {
                            fields.push(f.clone());
                            return true;
                        }
                    }
                }
                false
            }
            if let Expr::BinOp { op: op @ (BinOp::And | BinOp::Or), .. } = cond.as_ref() {
                let is_and = matches!(op, BinOp::And);
                let mut fields = Vec::new();
                if collect_has(cond, &mut fields, is_and) && !fields.is_empty() {
                    return Some((fields, is_and));
                }
            }
        }
        None
    }

    /// Detect `keys_unsorted` on input.
    pub fn is_keys_unsorted(&self) -> bool {
        use crate::ir::{Expr, UnaryOp};
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        matches!(expr, Expr::UnaryOp { op: UnaryOp::KeysUnsorted, operand } if matches!(operand.as_ref(), Expr::Input))
    }

    /// Detect `to_entries | sort_by(.key) | from_entries` pattern — sort object keys.
    pub fn is_sort_keys(&self) -> bool {
        use crate::ir::{Expr, UnaryOp, ClosureOpKind, Literal};
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        // Pattern: Pipe(Pipe(to_entries, sort_by(.key)), from_entries)
        // or: Pipe(to_entries, Pipe(sort_by(.key), from_entries))
        // After simplify_expr normalization, it could be either form.
        fn check(expr: &Expr) -> bool {
            // Try: Pipe(Pipe(to_entries, sort_by(.key)), from_entries)
            if let Expr::Pipe { left, right } = expr {
                if matches!(right.as_ref(), Expr::UnaryOp { op: UnaryOp::FromEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    if let Expr::Pipe { left: l2, right: r2 } = left.as_ref() {
                        if matches!(l2.as_ref(), Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                            return is_sort_by_key(r2);
                        }
                    }
                }
                // Try: Pipe(to_entries, Pipe(sort_by(.key), from_entries))
                if matches!(left.as_ref(), Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    if let Expr::Pipe { left: l2, right: r2 } = right.as_ref() {
                        if matches!(r2.as_ref(), Expr::UnaryOp { op: UnaryOp::FromEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                            return is_sort_by_key(l2);
                        }
                    }
                }
            }
            false
        }
        fn is_sort_by_key(expr: &Expr) -> bool {
            if let Expr::ClosureOp { op: ClosureOpKind::SortBy, input_expr, key_expr } = expr {
                if !matches!(input_expr.as_ref(), Expr::Input) { return false; }
                // key_expr should be .key (Index{Input, "key"})
                if let Expr::Index { expr: base, key } = key_expr.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(s)) = key.as_ref() {
                            return s == "key";
                        }
                    }
                }
            }
            false
        }
        check(expr)
    }

    /// Detect `to_entries` on input.
    pub fn is_to_entries(&self) -> bool {
        use crate::ir::{Expr, UnaryOp};
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        matches!(expr, Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input))
    }

    /// Detect `to_entries[] | "\(.key)SEP\(.value)"` pattern.
    /// Returns Vec<(is_literal, content)> where content is "key"/"value" for interpolated parts
    /// and literal text for literal parts.
    pub fn detect_to_entries_each_interp(&self) -> Option<Vec<(bool, String)>> {
        use crate::ir::{Expr, Literal, UnaryOp, StringPart};
        let expr = self.detect_expr()?;
        // Match: Pipe(to_entries_each, string_interp)
        // Forms:
        //   Pipe(Each(UnaryOp(ToEntries,Input)), StringInterp)
        //   Pipe(Pipe(UnaryOp(ToEntries,Input), Each(Input)), StringInterp)
        let (te_each, interp) = if let Expr::Pipe { left, right } = expr {
            (left.as_ref(), right.as_ref())
        } else { return None; };
        // Verify left is to_entries[]
        let is_te_each = match te_each {
            Expr::Each { input_expr } => {
                matches!(input_expr.as_ref(), Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input))
            }
            Expr::Pipe { left, right } => {
                matches!(left.as_ref(), Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input))
                && matches!(right.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input))
            }
            _ => false,
        };
        if !is_te_each { return None; }
        // Parse string interpolation with .key and .value references
        if let Expr::StringInterpolation { parts } = interp {
            let mut result = Vec::new();
            for part in parts {
                match part {
                    StringPart::Literal(s) => result.push((true, s.clone())),
                    StringPart::Expr(Expr::Index { expr: base, key }) => {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            if field == "key" || field == "value" {
                                result.push((false, field.clone()));
                            } else { return None; }
                        } else { return None; }
                    }
                    _ => return None,
                }
            }
            if result.iter().any(|(is_lit, _)| !is_lit) {
                return Some(result);
            }
        }
        None
    }

    /// Detect `{k1:.f1, k2:.f2, ...} | to_entries` pattern.
    /// Returns Vec of (output_key, source_field) pairs.
    pub fn detect_remap_to_entries(&self) -> Option<Vec<(String, String)>> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            // right must be to_entries on input
            if !matches!(right.as_ref(), Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                return None;
            }
            // left must be an ObjectConstruct with field refs
            if let Expr::ObjectConstruct { pairs } = left.as_ref() {
                let mut result = Vec::with_capacity(pairs.len());
                for (k, v) in pairs {
                    let key = if let Expr::Literal(Literal::Str(s)) = k {
                        s.clone()
                    } else { return None; };
                    if let Expr::Index { expr: base, key: field_key } = v {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = field_key.as_ref() {
                            result.push((key, f.clone()));
                        } else { return None; }
                    } else { return None; }
                }
                if result.is_empty() { return None; }
                return Some(result);
            }
        }
        None
    }

    /// Detect `with_entries(select(.value CMP N))` pattern.
    /// Returns (cmp_op, threshold) for numeric value comparison.
    /// This matches: to_entries | [.[] | select(.value CMP N)] | from_entries
    pub fn detect_with_entries_select_value_cmp(&self) -> Option<(crate::ir::BinOp, f64)> {
        use crate::ir::{BinOp, Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // Pattern: Pipe(UnaryOp(ToEntries), Pipe(Collect(Pipe(Each, IfThenElse(cond, Input, Empty))), UnaryOp(FromEntries)))
        if let Expr::Pipe { left: l1, right: r1 } = expr {
            // l1 = to_entries
            if !matches!(l1.as_ref(), Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                return None;
            }
            // r1 = Pipe(Collect(...), from_entries)
            if let Expr::Pipe { left: l2, right: r2 } = r1.as_ref() {
                // r2 = from_entries
                if !matches!(r2.as_ref(), Expr::UnaryOp { op: UnaryOp::FromEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    return None;
                }
                // l2 = Collect(Pipe(Each(Input), IfThenElse(...)))
                if let Expr::Collect { generator } = l2.as_ref() {
                    if let Expr::Pipe { left: l3, right: r3 } = generator.as_ref() {
                        if !matches!(l3.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            return None;
                        }
                        // r3 = IfThenElse(cond, Input, Empty) i.e. select(cond)
                        if let Expr::IfThenElse { cond, then_branch, else_branch } = r3.as_ref() {
                            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
                            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
                            // cond = BinOp(cmp, Index(Input, "value"), Literal(Num(n)))
                            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                                if matches!(op, BinOp::Gt | BinOp::Ge | BinOp::Lt | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                                    // .value CMP N
                                    if let Expr::Index { expr: base, key } = lhs.as_ref() {
                                        if matches!(base.as_ref(), Expr::Input) {
                                            if let Expr::Literal(Literal::Str(s)) = key.as_ref() {
                                                if s == "value" {
                                                    if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                                        return Some((*op, *n));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    // N CMP .value → flip
                                    if let Expr::Index { expr: base, key } = rhs.as_ref() {
                                        if matches!(base.as_ref(), Expr::Input) {
                                            if let Expr::Literal(Literal::Str(s)) = key.as_ref() {
                                                if s == "value" {
                                                    if let Expr::Literal(Literal::Num(n, _)) = lhs.as_ref() {
                                                        let flipped = match op {
                                                            BinOp::Gt => BinOp::Lt,
                                                            BinOp::Ge => BinOp::Le,
                                                            BinOp::Lt => BinOp::Gt,
                                                            BinOp::Le => BinOp::Ge,
                                                            _ => *op,
                                                        };
                                                        return Some((flipped, *n));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `with_entries(select(.value | type == "type_name"))` pattern.
    /// Returns the type name string if detected (e.g., "number", "string", "boolean", "array", "object", "null").
    pub fn detect_with_entries_select_value_type(&self) -> Option<String> {
        use crate::ir::{Expr, Literal, UnaryOp, BinOp};
        let expr = self.detect_expr()?;
        // Same structure as with_entries_select_value_cmp but condition is type(.value) == "typename"
        if let Expr::Pipe { left: l1, right: r1 } = expr {
            if !matches!(l1.as_ref(), Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                return None;
            }
            if let Expr::Pipe { left: l2, right: r2 } = r1.as_ref() {
                if !matches!(r2.as_ref(), Expr::UnaryOp { op: UnaryOp::FromEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    return None;
                }
                if let Expr::Collect { generator } = l2.as_ref() {
                    if let Expr::Pipe { left: l3, right: r3 } = generator.as_ref() {
                        if !matches!(l3.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            return None;
                        }
                        if let Expr::IfThenElse { cond, then_branch, else_branch } = r3.as_ref() {
                            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
                            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
                            // cond: BinOp(Eq, UnaryOp(Type, Index(Input, "value")), Literal(Str(type_name)))
                            // or beta-reduced: BinOp(Eq, UnaryOp(Type, Index(Input, "value")), Literal(Str(type_name)))
                            if let Expr::BinOp { op: BinOp::Eq, lhs, rhs } = cond.as_ref() {
                                // Check: type(.value) == "typename"
                                if let Expr::UnaryOp { op: UnaryOp::Type, operand } = lhs.as_ref() {
                                    if let Expr::Index { expr: base, key } = operand.as_ref() {
                                        if matches!(base.as_ref(), Expr::Input) {
                                            if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                                                if f == "value" {
                                                    if let Expr::Literal(Literal::Str(type_name)) = rhs.as_ref() {
                                                        return Some(type_name.clone());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                // Check reverse: "typename" == type(.value)
                                if let Expr::UnaryOp { op: UnaryOp::Type, operand } = rhs.as_ref() {
                                    if let Expr::Index { expr: base, key } = operand.as_ref() {
                                        if matches!(base.as_ref(), Expr::Input) {
                                            if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                                                if f == "value" {
                                                    if let Expr::Literal(Literal::Str(type_name)) = lhs.as_ref() {
                                                        return Some(type_name.clone());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `with_entries(select(.key | startswith/endswith/contains("str")))` or `with_entries(select(.key == "str"))`.
    /// Returns (test_op, test_string) where test_op is "startswith"/"endswith"/"contains"/"eq".
    pub fn detect_with_entries_select_key_str(&self) -> Option<(String, String)> {
        use crate::ir::{BinOp, Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left: l1, right: r1 } = expr {
            if !matches!(l1.as_ref(), Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                return None;
            }
            if let Expr::Pipe { left: l2, right: r2 } = r1.as_ref() {
                if !matches!(r2.as_ref(), Expr::UnaryOp { op: UnaryOp::FromEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    return None;
                }
                if let Expr::Collect { generator } = l2.as_ref() {
                    if let Expr::Pipe { left: l3, right: r3 } = generator.as_ref() {
                        if !matches!(l3.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            return None;
                        }
                        if let Expr::IfThenElse { cond, then_branch, else_branch } = r3.as_ref() {
                            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
                            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
                            // .key | startswith("str") — beta-reduced form
                            if let Expr::CallBuiltin { op: name, args } = cond.as_ref() {
                                if matches!(name, BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains) && args.len() == 2 {
                                    if let Expr::Index { expr: base, key } = &args[0] {
                                        if matches!(base.as_ref(), Expr::Input) {
                                            if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                                                if f == "key" {
                                                    if let Expr::Literal(Literal::Str(s)) = &args[1] {
                                                        return Some((name.name().to_string(), s.clone()));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // .key | startswith("str") — piped form: Pipe(Index(Input, "key"), CallBuiltin("startswith", [Literal("str")]))
                            // Note: Input is implicit in piped form (1 arg), or explicit (2 args)
                            if let Expr::Pipe { left: pl, right: pr } = cond.as_ref() {
                                if let Expr::Index { expr: base, key } = pl.as_ref() {
                                    if matches!(base.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                                            if f == "key" {
                                                if let Expr::CallBuiltin { op: name, args } = pr.as_ref() {
                                                    if matches!(name, BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains) {
                                                        // 1-arg form: CallBuiltin("startswith", [Literal("str")])
                                                        if args.len() == 1 {
                                                            if let Expr::Literal(Literal::Str(s)) = &args[0] {
                                                                return Some((name.name().to_string(), s.clone()));
                                                            }
                                                        }
                                                        // 2-arg form: CallBuiltin("startswith", [Input, Literal("str")])
                                                        if args.len() == 2 && matches!(args[0], Expr::Input) {
                                                            if let Expr::Literal(Literal::Str(s)) = &args[1] {
                                                                return Some((name.name().to_string(), s.clone()));
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // .key == "str" → BinOp(Eq, Index(Input, "key"), Literal(Str(s)))
                            if let Expr::BinOp { op: BinOp::Eq, lhs, rhs } = cond.as_ref() {
                                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                                    if matches!(base.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                                            if f == "key" {
                                                if let Expr::Literal(Literal::Str(s)) = rhs.as_ref() {
                                                    return Some(("eq".to_string(), s.clone()));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `with_entries(select(.key != "name"))` — equivalent to `del(.name)`.
    /// Returns list of excluded key names.
    pub fn detect_with_entries_del_keys(&self) -> Option<Vec<String>> {
        use crate::ir::{BinOp, Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left: l1, right: r1 } = expr {
            if !matches!(l1.as_ref(), Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                return None;
            }
            if let Expr::Pipe { left: l2, right: r2 } = r1.as_ref() {
                if !matches!(r2.as_ref(), Expr::UnaryOp { op: UnaryOp::FromEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    return None;
                }
                if let Expr::Collect { generator } = l2.as_ref() {
                    if let Expr::Pipe { left: l3, right: r3 } = generator.as_ref() {
                        if !matches!(l3.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            return None;
                        }
                        if let Expr::IfThenElse { cond, then_branch, else_branch } = r3.as_ref() {
                            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
                            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
                            // Single: .key != "name"
                            if let Expr::BinOp { op: BinOp::Ne, lhs, rhs } = cond.as_ref() {
                                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                                    if matches!(base.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Str(s)) = key.as_ref() {
                                            if s == "key" {
                                                if let Expr::Literal(Literal::Str(name)) = rhs.as_ref() {
                                                    return Some(vec![name.clone()]);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // Compound AND: .key != "a" and .key != "b"
                            fn collect_key_ne(cond: &Expr, keys: &mut Vec<String>) -> bool {
                                match cond {
                                    Expr::BinOp { op: BinOp::And, lhs, rhs } => {
                                        collect_key_ne(lhs, keys) && collect_key_ne(rhs, keys)
                                    }
                                    Expr::BinOp { op: BinOp::Ne, lhs, rhs } => {
                                        if let Expr::Index { expr: base, key } = lhs.as_ref() {
                                            if matches!(base.as_ref(), Expr::Input) {
                                                if let Expr::Literal(Literal::Str(s)) = key.as_ref() {
                                                    if s == "key" {
                                                        if let Expr::Literal(Literal::Str(name)) = rhs.as_ref() {
                                                            keys.push(name.clone());
                                                            return true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        false
                                    }
                                    _ => false,
                                }
                            }
                            let mut keys = Vec::new();
                            if collect_key_ne(cond, &mut keys) && !keys.is_empty() {
                                return Some(keys);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `with_entries(.value |= tostring)`.
    /// Returns true if matched.
    pub fn is_with_entries_tostring(&self) -> bool {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        // Pattern: Pipe(to_entries, Pipe(map(.value |= tostring), from_entries))
        if let Expr::Pipe { left: l1, right: r1 } = expr {
            if !matches!(l1.as_ref(), Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                return false;
            }
            if let Expr::Pipe { left: l2, right: r2 } = r1.as_ref() {
                if !matches!(r2.as_ref(), Expr::UnaryOp { op: UnaryOp::FromEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    return false;
                }
                if let Expr::Collect { generator } = l2.as_ref() {
                    if let Expr::Pipe { left: l3, right: r3 } = generator.as_ref() {
                        if !matches!(l3.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            return false;
                        }
                        // Body: Update { .value, tostring(.) }
                        if let Expr::Update { path_expr, update_expr } = r3.as_ref() {
                            if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                                if matches!(base.as_ref(), Expr::Input) {
                                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                        if field == "value" {
                                            if let Expr::UnaryOp { op: UnaryOp::ToString, operand } = update_expr.as_ref() {
                                                if matches!(operand.as_ref(), Expr::Input) {
                                                    return true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Also handle LetBinding wrapper
                        if let Expr::LetBinding { body, .. } = r3.as_ref() {
                            if let Expr::Update { path_expr, update_expr } = body.as_ref() {
                                if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                                    if matches!(base.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                            if field == "value" {
                                                if let Expr::UnaryOp { op: UnaryOp::ToString, operand } = update_expr.as_ref() {
                                                    if matches!(operand.as_ref(), Expr::Input) {
                                                        return true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Detect `.field = CONST` or `setpath(["field"]; CONST)` pattern.
    /// Returns (field_name, json_bytes_of_value) for raw byte replacement.
    pub fn detect_field_assign_const(&self) -> Option<(String, Vec<u8>)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        // .field = CONST
        if let Expr::Assign { path_expr, value_expr } = expr {
            let field = if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                else { return None; }
            } else { return None; };
            let val_bytes = literal_to_json_bytes(value_expr)?;
            return Some((field, val_bytes));
        }
        // setpath(["field"]; CONST) — path is Collect(Literal(Str(field)))
        if let Expr::SetPath { path, value, .. } = expr {
            if let Expr::Collect { generator } = path.as_ref() {
                if let Expr::Literal(Literal::Str(f)) = generator.as_ref() {
                    let val_bytes = literal_to_json_bytes(value)?;
                    return Some((f.clone(), val_bytes));
                }
            }
        }
        None
    }

    /// Detect `.dest = (.src op N)` — cross-field numeric assignment.
    /// Returns (dest_field, src_field, op, constant, is_const_on_left).
    pub fn detect_field_assign_field_arith(&self) -> Option<(String, String, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, Literal, BinOp};
        let expr = self.detect_expr()?;
        if let Expr::Assign { path_expr, value_expr } = expr {
            let dest = if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                else { return None; }
            } else { return None; };
            if let Expr::BinOp { op, lhs, rhs } = value_expr.as_ref() {
                // .dest = (.src op N)
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(src)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                match op {
                                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                                        return Some((dest, src.clone(), *op, *n));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                // .dest = (N op .src)
                if let Expr::Index { expr: base, key } = rhs.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(src)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = lhs.as_ref() {
                                match op {
                                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                                        // Normalize: N op .src → .src op N for commutative, keep as-is for non-commutative
                                        match op {
                                            BinOp::Add | BinOp::Mul => return Some((dest, src.clone(), *op, *n)),
                                            _ => {
                                                // For sub/div/mod, we can't simply swap, need special handling
                                                // Skip for now — less common
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
            // .dest = .src (direct field copy)
            if let Expr::Index { expr: base, key } = value_expr.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(src)) = key.as_ref() {
                        return Some((dest, src.clone(), BinOp::Add, 0.0)); // identity: .src + 0
                    }
                }
            }
        }
        None
    }

    /// Detect `.dest = (.src1 op .src2)` — cross-field two-field arithmetic assignment.
    /// Returns (dest_field, src1_field, src2_field, op).
    pub fn detect_field_assign_two_fields(&self) -> Option<(String, String, String, crate::ir::BinOp)> {
        use crate::ir::{Expr, Literal, BinOp};
        let expr = self.detect_expr()?;
        if let Expr::Assign { path_expr, value_expr } = expr {
            let dest = if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                else { return None; }
            } else { return None; };
            if let Expr::BinOp { op, lhs, rhs } = value_expr.as_ref() {
                if let Expr::Index { expr: bl, key: kl } = lhs.as_ref() {
                    if matches!(bl.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(src1)) = kl.as_ref() {
                            if let Expr::Index { expr: br, key: kr } = rhs.as_ref() {
                                if matches!(br.as_ref(), Expr::Input) {
                                    if let Expr::Literal(Literal::Str(src2)) = kr.as_ref() {
                                        match op {
                                            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                                                return Some((dest, src1.clone(), src2.clone(), *op));
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `tojson` on input.
    pub fn is_tojson(&self) -> bool {
        use crate::ir::{Expr, UnaryOp};
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        matches!(expr, Expr::UnaryOp { op: UnaryOp::ToJson, operand } if matches!(operand.as_ref(), Expr::Input))
            || matches!(expr, Expr::Format { kind: crate::ir::FormatKind::Json, expr: inner } if matches!(inner.as_ref(), Expr::Input))
    }

    /// Detect `tojson | fromjson` — identity for valid JSON input (from files).
    /// NaN/inf values are only produced by arithmetic, never present in parsed JSON,
    /// so this is safe as identity when processing file input.
    pub fn is_tojson_fromjson(&self) -> bool {
        use crate::ir::{Expr, UnaryOp};
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        // Unsimplified: Pipe(UnaryOp(ToJson, Input), UnaryOp(FromJson, Input))
        if let Expr::Pipe { left, right } = expr {
            matches!(left.as_ref(), Expr::UnaryOp { op: UnaryOp::ToJson, operand } if matches!(operand.as_ref(), Expr::Input))
                && matches!(right.as_ref(), Expr::UnaryOp { op: UnaryOp::FromJson, operand } if matches!(operand.as_ref(), Expr::Input))
        }
        // Simplified (beta-reduced): UnaryOp(FromJson, UnaryOp(ToJson, Input))
        else if let Expr::UnaryOp { op: UnaryOp::FromJson, operand } = expr {
            matches!(operand.as_ref(), Expr::UnaryOp { op: UnaryOp::ToJson, operand: inner } if matches!(inner.as_ref(), Expr::Input))
        } else {
            false
        }
    }

    /// Detect `{a:.x, b:.y} | tojson` — remap then serialize to JSON string.
    /// Returns Vec of (output_key, input_field) pairs if detected.
    pub fn detect_remap_tojson(&self) -> Option<Vec<(String, String)>> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        let (remap_expr, tojson_check) = if let Expr::Pipe { left, right } = expr {
            (left.as_ref(), right.as_ref())
        } else { return None; };
        // Check right is tojson
        let is_tojson = matches!(tojson_check,
            Expr::UnaryOp { op: UnaryOp::ToJson, operand } if matches!(operand.as_ref(), Expr::Input))
            || matches!(tojson_check,
            Expr::Format { kind: crate::ir::FormatKind::Json, expr: inner } if matches!(inner.as_ref(), Expr::Input));
        if !is_tojson { return None; }
        // Check left is {key: .field, ...}
        if let Expr::ObjectConstruct { pairs } = remap_expr {
            let mut result = Vec::with_capacity(pairs.len());
            for (k, v) in pairs {
                let key = if let Expr::Literal(Literal::Str(s)) = k { s.clone() } else { return None; };
                if let Expr::Index { expr: base, key: field_key } = v {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(f)) = field_key.as_ref() {
                        result.push((key, f.clone()));
                    } else { return None; }
                } else { return None; }
            }
            if !result.is_empty() { return Some(result); }
        }
        None
    }

    /// Detect `.[]` — each/iteration on input.
    pub fn is_each(&self) -> bool {
        use crate::ir::Expr;
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        matches!(expr, Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input))
    }

    /// Detect `.[] | strings/numbers/booleans/nulls/arrays/objects` — each with type filter.
    /// Returns the type name ("string", "number", etc.).
    pub fn detect_each_type_filter(&self) -> Option<String> {
        use crate::ir::{Expr, Literal, BinOp, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if matches!(left.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                // select(type == "T") = IfThenElse { cond: type == "T", then: ., else: empty }
                if let Expr::IfThenElse { cond, then_branch, else_branch } = right.as_ref() {
                    if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                        if let Expr::BinOp { op: BinOp::Eq, lhs, rhs } = cond.as_ref() {
                            if matches!(lhs.as_ref(), Expr::UnaryOp { op: UnaryOp::Type, operand } if matches!(operand.as_ref(), Expr::Input)) {
                                if let Expr::Literal(Literal::Str(ty)) = rhs.as_ref() {
                                    return Some(ty.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `[.[]]` — collect all values into array.
    pub fn is_collect_each(&self) -> bool {
        use crate::ir::Expr;
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        matches!(expr, Expr::Collect { generator } if matches!(generator.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)))
    }

    /// Detect `[.[] | . op N]` — map each value with arithmetic.
    /// Returns (BinOp, f64) for the operation applied to each value.
    pub fn detect_collect_each_arith(&self) -> Option<(crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, Literal, BinOp};
        let expr = self.detect_expr()?;
        if let Expr::Collect { generator } = expr {
            if let Expr::Pipe { left, right } = generator.as_ref() {
                if matches!(left.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                    // . op N  (e.g., . * 2, . + 1)
                    if let Expr::BinOp { op, lhs, rhs } = right.as_ref() {
                        if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                            if matches!(lhs.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                    return Some((*op, *n));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `[.[] | select(type == "TYPE")]` — collect values of given type.
    /// Returns the type string ("number", "string", "object", "array", "boolean", "null").
    pub fn detect_collect_each_select_type(&self) -> Option<String> {
        use crate::ir::{Expr, Literal, BinOp, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Collect { generator } = expr {
            if let Expr::Pipe { left, right } = generator.as_ref() {
                if matches!(left.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                    // select(type == "T") = IfThenElse { cond: type == "T", then: ., else: empty }
                    if let Expr::IfThenElse { cond, then_branch, else_branch } = right.as_ref() {
                        if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                            if let Expr::BinOp { op: BinOp::Eq, lhs, rhs } = cond.as_ref() {
                                if matches!(lhs.as_ref(), Expr::UnaryOp { op: UnaryOp::Type, operand } if matches!(operand.as_ref(), Expr::Input)) {
                                    if let Expr::Literal(Literal::Str(t)) = rhs.as_ref() {
                                        if matches!(t.as_str(), "number" | "string" | "object" | "array" | "boolean" | "null") {
                                            return Some(t.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `[.[] | select(. cmp N)]` — collect values passing numeric comparison.
    /// Returns (BinOp, f64).
    pub fn detect_collect_each_select_cmp(&self) -> Option<(crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, Literal, BinOp};
        let expr = self.detect_expr()?;
        if let Expr::Collect { generator } = expr {
            if let Expr::Pipe { left, right } = generator.as_ref() {
                if matches!(left.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                    // select(. cmp N) = IfThenElse { cond: . cmp N, then: ., else: empty }
                    if let Expr::IfThenElse { cond, then_branch, else_branch } = right.as_ref() {
                        if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                                if matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                                    if matches!(lhs.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                            return Some((*op, *n));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `first(.[] | select(type == "T"))` or `limit(1; .[] | select(type == "T"))`.
    /// Returns the type string.
    pub fn detect_first_each_select_type(&self) -> Option<String> {
        use crate::ir::{Expr, Literal, BinOp, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Limit { count, generator } = expr {
            if let Expr::Literal(Literal::Num(n, _)) = count.as_ref() {
                if *n == 1.0 {
                    if let Expr::Pipe { left, right } = generator.as_ref() {
                        if matches!(left.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            if let Expr::IfThenElse { cond, then_branch, else_branch } = right.as_ref() {
                                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                                    if let Expr::BinOp { op: BinOp::Eq, lhs, rhs } = cond.as_ref() {
                                        if matches!(lhs.as_ref(), Expr::UnaryOp { op: UnaryOp::Type, operand } if matches!(operand.as_ref(), Expr::Input)) {
                                            if let Expr::Literal(Literal::Str(t)) = rhs.as_ref() {
                                                if matches!(t.as_str(), "number" | "string" | "object" | "array" | "boolean" | "null") {
                                                    return Some(t.clone());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `[.[] | select(. cmp N)] | length` — count values passing comparison.
    /// Returns (BinOp, f64).
    pub fn detect_count_each_select_cmp(&self) -> Option<(crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, Literal, BinOp, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if matches!(right.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if let Expr::Collect { generator } = left.as_ref() {
                    if let Expr::Pipe { left: gen_left, right: gen_right } = generator.as_ref() {
                        if matches!(gen_left.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            if let Expr::IfThenElse { cond, then_branch, else_branch } = gen_right.as_ref() {
                                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                                    if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                                        if matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                                            if matches!(lhs.as_ref(), Expr::Input) {
                                                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                                    return Some((*op, *n));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `[.x, .y] | sort` — two-field sort into array.
    /// Returns (field1, field2).
    pub fn detect_sort_two_fields(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Collect { generator } = left.as_ref() {
                if let Expr::Comma { left: f1, right: f2 } = generator.as_ref() {
                    let field1 = if let Expr::Index { expr: base, key } = f1.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                        else { return None; }
                    } else { return None; };
                    let field2 = if let Expr::Index { expr: base, key } = f2.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                        else { return None; }
                    } else { return None; };
                    if matches!(right.as_ref(), Expr::UnaryOp { op: UnaryOp::Sort, operand } if matches!(operand.as_ref(), Expr::Input)) {
                        return Some((field1, field2));
                    }
                }
            }
        }
        None
    }

    /// Detect array-of-field-access `[.f1,.f2,...]` pattern.
    /// Returns the list of field names if this is Collect over comma field accesses.
    pub fn detect_array_field_access(&self) -> Option<Vec<String>> {
        use crate::ir::Expr;
        let expr = self.detect_expr()?;
        if let Expr::Collect { generator } = expr {
            let mut fields = Vec::new();
            if collect_comma_fields(generator, &mut fields) && fields.len() >= 2 {
                return Some(fields);
            }
        }
        None
    }

    /// Detect `[.x, .y, .x + .y]` — array construct with computed values.
    /// Returns Vec of RemapExpr if at least one value is computed.
    pub fn detect_computed_array(&self) -> Option<Vec<RemapExpr>> {
        use crate::ir::Expr;
        let expr = self.detect_expr()?;
        if let Expr::Collect { generator } = expr {
            let mut elems = Vec::new();
            if collect_comma_remap(generator, &mut elems) && elems.len() >= 2 {
                let has_computed = elems.iter().any(|e| !matches!(e, RemapExpr::Field(_)));
                if has_computed { return Some(elems); }
            }
        }
        None
    }

    /// Detect `[.f1,.f2,...] | @csv` or `@tsv` pattern.
    /// Returns (field_names, format) where format is "csv" or "tsv".
    pub fn detect_array_fields_format(&self) -> Option<(Vec<String>, String)> {
        use crate::ir::Expr;
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Format { kind, .. } = right.as_ref() {
                if matches!(kind, crate::ir::FormatKind::Csv | crate::ir::FormatKind::Tsv) {
                    if let Expr::Collect { generator } = left.as_ref() {
                        let mut fields = Vec::new();
                        if collect_comma_fields(generator, &mut fields) && fields.len() >= 2 {
                            return Some((fields, kind.name().to_string()));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | split(x) | join(y)` pattern (string replace).
    /// Returns (field_name, split_str, join_str) if detected.
    pub fn detect_field_split_join(&self) -> Option<(String, String, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        // .field | split("x") | join("y")
        // Right-associative pipes: Pipe(.field, Pipe(split, join))
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: split_expr, right: join_expr } = right.as_ref() {
                        if let Expr::CallBuiltin { op: split_name, args: split_args } = split_expr.as_ref() {
                            if *split_name != BuiltinOp::Split || split_args.len() != 1 { return None; }
                            if let Expr::Literal(Literal::Str(split_str)) = &split_args[0] {
                                if let Expr::CallBuiltin { op: join_name, args: join_args } = join_expr.as_ref() {
                                    if *join_name != BuiltinOp::Join || join_args.len() != 1 { return None; }
                                    if let Expr::Literal(Literal::Str(join_str)) = &join_args[0] {
                                        return Some((field.clone(), split_str.clone(), join_str.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | split(sep) | .[from:to] | join(sep2)` pattern.
    /// Returns (field_name, split_sep, from, to, join_sep).
    /// from/to are Option<i64> (None = unbounded).
    pub fn detect_field_split_slice_join(&self) -> Option<(String, String, Option<i64>, Option<i64>, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        // Pipe(.field, Pipe(split, Pipe(slice, join)))
        if let Expr::Pipe { left, right } = expr {
            let field = if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                else { return None; }
            } else { return None; };
            if let Expr::Pipe { left: l2, right: r2 } = right.as_ref() {
                let split_sep = if let Expr::CallBuiltin { op: name, args } = l2.as_ref() {
                    if *name != BuiltinOp::Split || args.len() != 1 { return None; }
                    if let Expr::Literal(Literal::Str(s)) = &args[0] { s.clone() }
                    else { return None; }
                } else { return None; };
                if let Expr::Pipe { left: l3, right: r3 } = r2.as_ref() {
                    let (from, to) = if let Expr::Slice { expr: base, from: f, to: t } = l3.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        fn extract_slice_idx(e: &Expr) -> Option<i64> {
                            match e {
                                Expr::Literal(Literal::Num(n, _)) => Some(*n as i64),
                                Expr::Negate { operand } => {
                                    if let Expr::Literal(Literal::Num(n, _)) = operand.as_ref() {
                                        Some(-(*n as i64))
                                    } else { None }
                                }
                                _ => None,
                            }
                        }
                        let from_val = match f {
                            Some(e) => Some(extract_slice_idx(e)?),
                            None => None,
                        };
                        let to_val = match t {
                            Some(e) => Some(extract_slice_idx(e)?),
                            None => None,
                        };
                        (from_val, to_val)
                    } else { return None; };
                    let join_sep = if let Expr::CallBuiltin { op: name, args } = r3.as_ref() {
                        if *name != BuiltinOp::Join || args.len() != 1 { return None; }
                        if let Expr::Literal(Literal::Str(s)) = &args[0] { s.clone() }
                        else { return None; }
                    } else { return None; };
                    return Some((field, split_sep, from, to, join_sep));
                }
            }
        }
        None
    }

    /// Detect `keys_unsorted | join(sep)` or `keys | join(sep)`.
    /// Returns (separator, is_sorted).
    pub fn detect_keys_join(&self) -> Option<(String, bool)> {
        use crate::ir::{Expr, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            let (is_sorted, operand) = match left.as_ref() {
                Expr::UnaryOp { op: UnaryOp::KeysUnsorted, operand } => (false, operand),
                Expr::UnaryOp { op: UnaryOp::Keys, operand } => (true, operand),
                _ => return None,
            };
            if !matches!(operand.as_ref(), Expr::Input) { return None; }
            if let Expr::CallBuiltin { op: name, args } = right.as_ref() {
                if *name == BuiltinOp::Join && args.len() == 1 {
                    if let Expr::Literal(crate::ir::Literal::Str(sep)) = &args[0] {
                        return Some((sep.clone(), is_sorted));
                    }
                }
            }
        }
        None
    }

    /// Detect `{(.field_key): .field_val}` — single dynamic-key object construction.
    /// Returns (key_field, value_field).
    pub fn detect_dynamic_key_obj(&self) -> Option<(String, RemapExpr)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::ObjectConstruct { pairs } = expr {
            if pairs.len() != 1 { return None; }
            let (k, v) = &pairs[0];
            // Key must be a field access (.field)
            let key_field = if let Expr::Index { expr: base, key } = k {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                else { return None; }
            } else { return None; };
            // Value: classify as RemapExpr
            let val_rexpr = Self::classify_remap_value(v)?;
            return Some((key_field, val_rexpr));
        }
        None
    }

    /// Detect `{(.name): .x, static_key: val, ...}` — object with one dynamic key and N static keys.
    ///
    /// Disabled: the generated output placed the dynamic key first regardless
    /// of source order and skipped duplicate-key collapse, producing both
    /// reordered output and invalid JSON when the dynamic key collided with a
    /// static one (issue #53). Falling back to the generic object-construct
    /// path preserves both invariants.
    pub fn detect_dynamic_key_mixed_obj(&self) -> Option<(String, RemapExpr, Vec<(String, RemapExpr)>)> {
        None
    }

    /// Detect `.field op= N` where op is +, -, *, /, %.
    /// Returns (field_name, BinOp, constant).
    pub fn detect_field_update_num(&self) -> Option<(String, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        // The parser wraps `.x += N` as LetBinding { var, N, Update { .x, . + LoadVar(var) } }.
        // Unwrap the LetBinding for constant RHS.
        let (update_expr_outer, let_var, let_val) = if let Expr::LetBinding { var_index, value, body } = expr {
            (body.as_ref(), Some(*var_index), Some(value.as_ref()))
        } else {
            (expr, None, None)
        };
        if let Expr::Update { path_expr, update_expr } = update_expr_outer {
            // path must be .field
            let field = if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                else { return None; }
            } else { return None; };
            // update must be . op N (either literal or LoadVar from LetBinding)
            if let Expr::BinOp { op, lhs, rhs } = update_expr.as_ref() {
                if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                    return None;
                }
                if !matches!(lhs.as_ref(), Expr::Input) { return None; }
                // Direct literal: Update { .x, . + N }
                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                    return Some((field, *op, *n));
                }
                // LetBinding-wrapped: LetBinding { var, N, Update { .x, . + LoadVar(var) } }
                if let (Some(var_idx), Some(val_expr)) = (let_var, let_val) {
                    if let Expr::LoadVar { var_index } = rhs.as_ref() {
                        if *var_index == var_idx {
                            if let Expr::Literal(Literal::Num(n, _)) = val_expr {
                                return Some((field, *op, *n));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field |= (. op1 N1 | op2 | . op3 N3 | ...)` — numeric chain update.
    /// Each step is either BinOp(op, Input, Num) or UnaryOp(floor/ceil/round/abs, Input).
    /// Returns (field_name, steps) where steps are (Option<(BinOp, f64)>, Option<UnaryOp>).
    pub fn detect_field_update_num_chain(&self) -> Option<(String, Vec<NumChainStep>)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        let update_expr_outer = if let Expr::LetBinding { body, .. } = expr { body.as_ref() } else { expr };
        if let Expr::Update { path_expr, update_expr } = update_expr_outer {
            let field = if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                else { return None; }
            } else { return None; };
            // Unwrap nested composition: BinOp(Div, UnaryOp(Floor, BinOp(Mul, Input, 100)), 100)
            // Each step wraps the previous result. We recurse into the inner expression
            // and push steps AFTER recursion so they end up in execution order.
            fn collect_steps(e: &Expr, steps: &mut Vec<NumChainStep>) -> bool {
                // Also handle Pipe chains (in case simplification didn't compose them)
                if let Expr::Pipe { left, right } = e {
                    if !collect_steps(left, steps) { return false; }
                    return collect_steps(right, steps);
                }
                if let Expr::BinOp { op, lhs, rhs } = e {
                    if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            // lhs is the previous stage's output
                            let ok = collect_steps(lhs, steps);
                            steps.push(NumChainStep::Arith(*op, *n));
                            return ok;
                        }
                    }
                }
                if let Expr::UnaryOp { op, operand } = e {
                    match op {
                        UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round |
                        UnaryOp::Fabs | UnaryOp::Sqrt | UnaryOp::Trunc => {
                            let ok = collect_steps(operand, steps);
                            steps.push(NumChainStep::Unary(*op));
                            return ok;
                        }
                        _ => {}
                    }
                }
                // Base case: Input means we've reached the start of the chain
                matches!(e, Expr::Input)
            }
            let mut steps = Vec::new();
            if collect_steps(update_expr, &mut steps) && steps.len() >= 2 {
                return Some((field, steps));
            }
        }
        None
    }

    /// Detect `.field |= (split("sep") | .[0])` — update field to first split component.
    /// Returns (field_name, separator).
    pub fn detect_field_update_split_first(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        let update_expr_outer = if let Expr::LetBinding { body, .. } = expr { body.as_ref() } else { expr };
        if let Expr::Update { path_expr, update_expr } = update_expr_outer {
            if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    fn check_split_first(e: &Expr) -> Option<String> {
                        // Form 1: Pipe(split, .[0])
                        if let Expr::Pipe { left, right } = e {
                            if let Some(sep) = extract_split(left) {
                                if is_first_index(right) { return Some(sep); }
                            }
                        }
                        // Form 2 (simplified): Index { expr: split(.,"_"), key: Literal(Num(0)) }
                        if let Expr::Index { expr: inner, key } = e {
                            if matches!(key.as_ref(), Expr::Literal(Literal::Num(n, _)) if *n == 0.0) {
                                if let Some(sep) = extract_split(inner) { return Some(sep); }
                            }
                        }
                        None
                    }
                    fn extract_split(e: &Expr) -> Option<String> {
                        if let Expr::CallBuiltin { op: name, args } = e {
                            if *name == BuiltinOp::Split && args.len() == 1 {
                                if let Expr::Literal(Literal::Str(sep)) = &args[0] {
                                    return Some(sep.clone());
                                }
                            }
                        }
                        // . / "sep" is string division = split
                        if let Expr::BinOp { op: crate::ir::BinOp::Div, lhs, rhs } = e {
                            if matches!(lhs.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(sep)) = rhs.as_ref() {
                                    return Some(sep.clone());
                                }
                            }
                        }
                        None
                    }
                    fn is_first_index(e: &Expr) -> bool {
                        if let Expr::Index { expr: inner, key } = e {
                            matches!(inner.as_ref(), Expr::Input) && matches!(key.as_ref(), Expr::Literal(Literal::Num(n, _)) if *n == 0.0)
                        } else { false }
                    }
                    if let Some(sep) = check_split_first(update_expr) {
                        return Some((field.clone(), sep));
                    }
                }
            }
        }
        None
    }

    /// Detect `.field |= (split("sep") | last)`.
    /// Returns (field_name, separator).
    pub fn detect_field_update_split_last(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        let update_expr_outer = if let Expr::LetBinding { body, .. } = expr { body.as_ref() } else { expr };
        if let Expr::Update { path_expr, update_expr } = update_expr_outer {
            if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    fn check_split_last(e: &Expr) -> Option<String> {
                        // Form 1: Pipe(split, last)
                        if let Expr::Pipe { left, right } = e {
                            if let Some(sep) = extract_split(left) {
                                if is_last(right) { return Some(sep); }
                            }
                        }
                        // Form 2 (simplified): Index { expr: split(.,"_"), key: Literal(Num(-1)) }
                        if let Expr::Index { expr: inner, key } = e {
                            if matches!(key.as_ref(), Expr::Literal(Literal::Num(n, _)) if *n == -1.0) {
                                if let Some(sep) = extract_split(inner) { return Some(sep); }
                            }
                        }
                        None
                    }
                    fn extract_split(e: &Expr) -> Option<String> {
                        if let Expr::CallBuiltin { op: name, args } = e {
                            if *name == BuiltinOp::Split && args.len() == 1 {
                                if let Expr::Literal(Literal::Str(sep)) = &args[0] {
                                    return Some(sep.clone());
                                }
                            }
                        }
                        // . / "sep" is string division = split
                        if let Expr::BinOp { op: crate::ir::BinOp::Div, lhs, rhs } = e {
                            if matches!(lhs.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(sep)) = rhs.as_ref() {
                                    return Some(sep.clone());
                                }
                            }
                        }
                        None
                    }
                    fn is_last(e: &Expr) -> bool {
                        // last is .[-1] or CallBuiltin("last", [Input])
                        if let Expr::Index { expr: inner, key } = e {
                            if matches!(inner.as_ref(), Expr::Input) && matches!(key.as_ref(), Expr::Literal(Literal::Num(n, _)) if *n == -1.0) {
                                return true;
                            }
                        }
                        false
                    }
                    if let Some(sep) = check_split_last(update_expr) {
                        return Some((field.clone(), sep));
                    }
                }
            }
        }
        None
    }

    /// Detect `.field |= gsub("re"; "replacement")` or `.field |= sub("re"; "replacement")`.
    /// Returns (field_name, regex_pattern, replacement, is_global).
    pub fn detect_field_update_gsub(&self) -> Option<(String, String, String, bool)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        let update_expr_outer = if let Expr::LetBinding { body, .. } = expr { body.as_ref() } else { expr };
        if let Expr::Update { path_expr, update_expr } = update_expr_outer {
            if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    let (is_global, input_expr, re, tostr) = match update_expr.as_ref() {
                        Expr::RegexGsub { input_expr, re, tostr, .. } => (true, input_expr, re, tostr),
                        Expr::RegexSub { input_expr, re, tostr, .. } => (false, input_expr, re, tostr),
                        _ => return None,
                    };
                    if !matches!(input_expr.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(pattern)) = re.as_ref() {
                        if let Expr::Literal(Literal::Str(replacement)) = tostr.as_ref() {
                            return Some((field.clone(), pattern.clone(), replacement.clone(), is_global));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field |= ascii_downcase/ascii_upcase`.
    /// Returns (field_name, is_upcase).
    pub fn detect_field_update_case(&self) -> Option<(String, bool)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // May be wrapped in LetBinding for the desugared form
        let update_expr = if let Expr::LetBinding { body, .. } = expr { body.as_ref() } else { expr };
        if let Expr::Update { path_expr, update_expr: upd } = update_expr {
            if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::UnaryOp { op, operand } = upd.as_ref() {
                        if matches!(operand.as_ref(), Expr::Input) {
                            match op {
                                UnaryOp::AsciiDowncase => return Some((field.clone(), false)),
                                UnaryOp::AsciiUpcase => return Some((field.clone(), true)),
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field |= ltrimstr("prefix")` or `.field |= rtrimstr("suffix")`.
    /// Returns (field_name, string_arg, is_rtrim).
    pub fn detect_field_update_trim(&self) -> Option<(String, String, bool)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        let update_expr = if let Expr::LetBinding { body, .. } = expr { body.as_ref() } else { expr };
        if let Expr::Update { path_expr, update_expr: upd } = update_expr {
            if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::CallBuiltin { op: name, args } = upd.as_ref() {
                        if args.len() == 1 {
                            if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                match name {
                                    BuiltinOp::LtrimStr => return Some((field.clone(), arg.clone(), false)),
                                    BuiltinOp::RtrimStr => return Some((field.clone(), arg.clone(), true)),
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field |= .[from:to]` (string slice update).
    /// Returns (field_name, from_opt, to_opt).
    pub fn detect_field_update_slice(&self) -> Option<(String, Option<i64>, Option<i64>)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        let update_expr = if let Expr::LetBinding { body, .. } = expr { body.as_ref() } else { expr };
        if let Expr::Update { path_expr, update_expr: upd } = update_expr {
            if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Slice { expr: slice_base, from, to } = upd.as_ref() {
                        if !matches!(slice_base.as_ref(), Expr::Input) { return None; }
                        let from_val = match from {
                            Some(e) => match e.as_ref() {
                                Expr::Literal(Literal::Num(n, _)) => Some(*n as i64),
                                _ => return None,
                            },
                            None => None,
                        };
                        let to_val = match to {
                            Some(e) => match e.as_ref() {
                                Expr::Literal(Literal::Num(n, _)) => Some(*n as i64),
                                _ => return None,
                            },
                            None => None,
                        };
                        return Some((field.clone(), from_val, to_val));
                    }
                }
            }
        }
        None
    }

    /// Detect `.field |= if . == "str" then "a" else "b" end`.
    /// Returns (field_name, cond_str, then_str, else_str).
    pub fn detect_field_update_str_map(&self) -> Option<(String, String, Vec<u8>, Vec<u8>)> {
        use crate::ir::{Expr, Literal, BinOp};
        let expr = self.detect_expr()?;
        let update_expr = if let Expr::LetBinding { body, .. } = expr { body.as_ref() } else { expr };
        if let Expr::Update { path_expr, update_expr: upd } = update_expr {
            if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::IfThenElse { cond, then_branch, else_branch } = upd.as_ref() {
                        // cond: . == "str"
                        if let Expr::BinOp { op: BinOp::Eq, lhs, rhs } = cond.as_ref() {
                            if matches!(lhs.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(cond_str)) = rhs.as_ref() {
                                    let then_json = literal_to_json_bytes(then_branch)?;
                                    let else_json = literal_to_json_bytes(else_branch)?;
                                    return Some((field.clone(), cond_str.clone(), then_json, else_json));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field |= (. + "literal")` or `.field |= ("literal" + .)`.
    /// Returns (field_name, prefix, suffix) — one of prefix/suffix may be empty.
    pub fn detect_field_update_str_concat(&self) -> Option<(String, String, String)> {
        use crate::ir::{Expr, Literal, BinOp};
        let expr = self.detect_expr()?;
        let update_expr = if let Expr::LetBinding { body, .. } = expr { body.as_ref() } else { expr };
        if let Expr::Update { path_expr, update_expr: upd } = update_expr {
            if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = upd.as_ref() {
                        // . + "suffix"
                        if matches!(lhs.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(s)) = rhs.as_ref() {
                                return Some((field.clone(), String::new(), s.clone()));
                            }
                        }
                        // "prefix" + .
                        if matches!(rhs.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(s)) = lhs.as_ref() {
                                return Some((field.clone(), s.clone(), String::new()));
                            }
                        }
                    }
                    // "prefix" + . + "suffix" — BinOp(Add, BinOp(Add, "prefix", .), "suffix")
                    if let Expr::BinOp { op: BinOp::Add, lhs: outer_lhs, rhs: outer_rhs } = upd.as_ref() {
                        if let Expr::Literal(Literal::Str(suffix)) = outer_rhs.as_ref() {
                            if let Expr::BinOp { op: BinOp::Add, lhs: inner_lhs, rhs: inner_rhs } = outer_lhs.as_ref() {
                                if matches!(inner_rhs.as_ref(), Expr::Input) {
                                    if let Expr::Literal(Literal::Str(prefix)) = inner_lhs.as_ref() {
                                        return Some((field.clone(), prefix.clone(), suffix.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field |= length`.
    /// Returns field_name.
    pub fn detect_field_update_length(&self) -> Option<String> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        let update_expr = if let Expr::LetBinding { body, .. } = expr { body.as_ref() } else { expr };
        if let Expr::Update { path_expr, update_expr: upd } = update_expr {
            if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::UnaryOp { op: UnaryOp::Length, operand } = upd.as_ref() {
                        if matches!(operand.as_ref(), Expr::Input) {
                            return Some(field.clone());
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field |= tostring`.
    /// Returns field_name.
    pub fn detect_field_update_tostring(&self) -> Option<String> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        let update_expr = if let Expr::LetBinding { body, .. } = expr { body.as_ref() } else { expr };
        if let Expr::Update { path_expr, update_expr: upd } = update_expr {
            if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::UnaryOp { op: UnaryOp::ToString, operand } = upd.as_ref() {
                        if matches!(operand.as_ref(), Expr::Input) {
                            return Some(field.clone());
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field |= test("regex")`.
    /// Returns (field_name, regex_pattern, flags_str).
    pub fn detect_field_update_test(&self) -> Option<(String, String, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        let update_expr = if let Expr::LetBinding { body, .. } = expr { body.as_ref() } else { expr };
        if let Expr::Update { path_expr, update_expr: upd } = update_expr {
            if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::RegexTest { input_expr, re, flags } = upd.as_ref() {
                        if !matches!(input_expr.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(pattern)) = re.as_ref() {
                            let flags_str = match flags.as_ref() {
                                Expr::Literal(Literal::Null) => String::new(),
                                Expr::Literal(Literal::Str(f)) => f.clone(),
                                _ => return None,
                            };
                            return Some((field.clone(), pattern.clone(), flags_str));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field[from:to]` with literal numeric bounds.
    /// Returns (field_name, from_opt, to_opt).
    pub fn detect_field_slice(&self) -> Option<(String, Option<i64>, Option<i64>)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Slice { expr: base, from, to } = expr {
            if let Expr::Index { expr: input, key } = base.as_ref() {
                if !matches!(input.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    let from_val = match from {
                        Some(e) => match e.as_ref() {
                            Expr::Literal(Literal::Num(n, _)) => Some(*n as i64),
                            _ => return None,
                        },
                        None => None,
                    };
                    let to_val = match to {
                        Some(e) => match e.as_ref() {
                            Expr::Literal(Literal::Num(n, _)) => Some(*n as i64),
                            _ => return None,
                        },
                        None => None,
                    };
                    return Some((field.clone(), from_val, to_val));
                }
            }
        }
        None
    }

    /// Detect `.field | split("s") | .[0]` or `.field | split("s") | first`.
    /// Returns (field_name, split_delimiter).
    pub fn detect_field_split_first(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        // Pipe(.field, Pipe(split("s"), .[0]))
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: split_expr, right: index_expr } = right.as_ref() {
                        if let Expr::CallBuiltin { op: name, args } = split_expr.as_ref() {
                            if *name != BuiltinOp::Split || args.len() != 1 { return None; }
                            if let Expr::Literal(Literal::Str(delim)) = &args[0] {
                                // Check for .[0]
                                let is_first = match index_expr.as_ref() {
                                    Expr::Index { expr: base, key } => {
                                        matches!(base.as_ref(), Expr::Input) &&
                                        matches!(key.as_ref(), Expr::Literal(Literal::Num(n, _)) if *n == 0.0)
                                    }
                                    _ => false,
                                };
                                if is_first {
                                    return Some((field.clone(), delim.clone()));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | split("s") | .[N]` for any integer N (positive or negative).
    /// Returns (field_name, split_delimiter, index).
    pub fn detect_field_split_index(&self) -> Option<(String, String, i32)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: split_expr, right: index_expr } = right.as_ref() {
                        if let Expr::CallBuiltin { op: name, args } = split_expr.as_ref() {
                            if *name != BuiltinOp::Split || args.len() != 1 { return None; }
                            if let Expr::Literal(Literal::Str(delim)) = &args[0] {
                                if let Expr::Index { expr: ibase, key: ikey } = index_expr.as_ref() {
                                    if matches!(ibase.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Num(n, _)) = ikey.as_ref() {
                                            let idx = *n as i32;
                                            // Skip 0 and -1 since those are handled by split_first / split_last
                                            if idx != 0 && idx != -1 {
                                                return Some((field.clone(), delim.clone(), idx));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | split("s") | last` or `.field | split("s") | .[-1]`.
    /// Returns (field_name, split_delimiter).
    pub fn detect_field_split_last(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: split_expr, right: last_expr } = right.as_ref() {
                        if let Expr::CallBuiltin { op: name, args } = split_expr.as_ref() {
                            if *name != BuiltinOp::Split || args.len() != 1 { return None; }
                            if let Expr::Literal(Literal::Str(delim)) = &args[0] {
                                // Check for .[-1] (last is parsed as .[-1])
                                let is_last = matches!(last_expr.as_ref(),
                                    Expr::Index { expr: base, key }
                                    if matches!(base.as_ref(), Expr::Input)
                                    && matches!(key.as_ref(), Expr::Literal(Literal::Num(n, _)) if *n == -1.0)
                                );
                                if is_last {
                                    return Some((field.clone(), delim.clone()));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | split("s") | last | tonumber` — returns (field_name, delimiter).
    pub fn detect_field_split_last_tonumber(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    // Pipe(split("s"), Pipe(last, tonumber)) or Pipe(split("s"), tonumber(last))
                    if let Expr::Pipe { left: split_expr, right: rest } = right.as_ref() {
                        if let Expr::CallBuiltin { op: name, args } = split_expr.as_ref() {
                            if *name != BuiltinOp::Split || args.len() != 1 { return None; }
                            if let Expr::Literal(Literal::Str(delim)) = &args[0] {
                                // rest should be Pipe(last, tonumber) or UnaryOp(Tonumber, last)
                                if let Expr::Pipe { left: last_expr, right: tonum_expr } = rest.as_ref() {
                                    let is_last = matches!(last_expr.as_ref(),
                                        Expr::Index { expr: b, key: k }
                                        if matches!(b.as_ref(), Expr::Input)
                                        && matches!(k.as_ref(), Expr::Literal(Literal::Num(n, _)) if *n == -1.0)
                                    );
                                    let is_tonum = matches!(tonum_expr.as_ref(),
                                        Expr::UnaryOp { op: UnaryOp::ToNumber, operand }
                                        if matches!(operand.as_ref(), Expr::Input)
                                    );
                                    if is_last && is_tonum {
                                        return Some((field.clone(), delim.clone()));
                                    }
                                }
                                // Also check UnaryOp(Tonumber, Index(Input, -1))
                                if let Expr::UnaryOp { op: UnaryOp::ToNumber, operand } = rest.as_ref() {
                                    if let Expr::Index { expr: b, key: k } = operand.as_ref() {
                                        if matches!(b.as_ref(), Expr::Input) {
                                            if let Expr::Literal(Literal::Num(n, _)) = k.as_ref() {
                                                if *n == -1.0 {
                                                    return Some((field.clone(), delim.clone()));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | split("s") | .[N] | tonumber` — returns (field_name, delimiter, index).
    pub fn detect_field_split_nth_tonumber(&self) -> Option<(String, String, i32)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: split_expr, right: rest } = right.as_ref() {
                        if let Expr::CallBuiltin { op: name, args } = split_expr.as_ref() {
                            if *name != BuiltinOp::Split || args.len() != 1 { return None; }
                            if let Expr::Literal(Literal::Str(delim)) = &args[0] {
                                // rest: Pipe(.[N], tonumber) or UnaryOp(ToNumber, .[N])
                                if let Expr::Pipe { left: idx_expr, right: tonum_expr } = rest.as_ref() {
                                    if let Expr::Index { expr: ib, key: ik } = idx_expr.as_ref() {
                                        if matches!(ib.as_ref(), Expr::Input) {
                                            if let Expr::Literal(Literal::Num(n, _)) = ik.as_ref() {
                                                let idx = *n as i32;
                                                if idx >= 0 {
                                                    let is_tonum = matches!(tonum_expr.as_ref(),
                                                        Expr::UnaryOp { op: UnaryOp::ToNumber, operand }
                                                        if matches!(operand.as_ref(), Expr::Input)
                                                    );
                                                    if is_tonum {
                                                        return Some((field.clone(), delim.clone(), idx));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                if let Expr::UnaryOp { op: UnaryOp::ToNumber, operand } = rest.as_ref() {
                                    if let Expr::Index { expr: ib, key: ik } = operand.as_ref() {
                                        if matches!(ib.as_ref(), Expr::Input) {
                                            if let Expr::Literal(Literal::Num(n, _)) = ik.as_ref() {
                                                let idx = *n as i32;
                                                if idx >= 0 {
                                                    return Some((field.clone(), delim.clone(), idx));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | split("str") | length` — returns (field_name, delimiter).
    pub fn detect_field_split_length(&self) -> Option<(String, String, Vec<(crate::ir::BinOp, f64)>)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // Pipe(.field, Pipe(split("s"), length_expr))
        // where length_expr is either Length(Input) or BinOp(op, Length(Input), N) chain
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: split_expr, right: len_expr } = right.as_ref() {
                        if let Expr::CallBuiltin { op: name, args } = split_expr.as_ref() {
                            if *name != BuiltinOp::Split || args.len() != 1 { return None; }
                            if let Expr::Literal(Literal::Str(delim)) = &args[0] {
                                // Plain length
                                if matches!(len_expr.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                                    return Some((field.clone(), delim.clone(), vec![]));
                                }
                                // length with arith chain: BinOp(op, ..., N)
                                let mut ops = Vec::new();
                                let mut cur = len_expr.as_ref();
                                loop {
                                    if let Expr::BinOp { op, lhs, rhs } = cur {
                                        if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
                                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                            ops.push((*op, *n));
                                            cur = lhs.as_ref();
                                        } else { return None; }
                                    } else { break; }
                                }
                                if !ops.is_empty() {
                                    if matches!(cur, Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                                        ops.reverse();
                                        return Some((field.clone(), delim.clone(), ops));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | split("sep") | length cmp N` — returns (field, delim, cmp_op, threshold).
    /// Counts split occurrences in raw bytes without constructing the array.
    pub fn detect_field_split_length_cmp(&self) -> Option<(String, String, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // Form 1: BinOp(cmp, Pipe(.field, Pipe(split, length)), Literal(N))
        if let Expr::BinOp { op, lhs, rhs } = expr {
            if matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                    if let Expr::Pipe { left, right } = lhs.as_ref() {
                        if let Expr::Index { expr: base, key } = left.as_ref() {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                    if let Expr::Pipe { left: split_expr, right: len_expr } = right.as_ref() {
                                        if let Expr::CallBuiltin { op: name, args } = split_expr.as_ref() {
                                            if *name == BuiltinOp::Split && args.len() == 1 {
                                                if let Expr::Literal(Literal::Str(delim)) = &args[0] {
                                                    if matches!(len_expr.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                                                        return Some((field.clone(), delim.clone(), *op, *n));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Beta-reduced: lhs might be Length(Split(.field, "sep"))
                    if let Expr::UnaryOp { op: UnaryOp::Length, operand: inner } = lhs.as_ref() {
                        if let Expr::CallBuiltin { op: name, args } = inner.as_ref() {
                            if *name == BuiltinOp::Split && args.len() == 2 {
                                if let Expr::Index { expr: base, key } = &args[0] {
                                    if matches!(base.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                            if let Expr::Literal(Literal::Str(delim)) = &args[1] {
                                                return Some((field.clone(), delim.clone(), *op, *n));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Form 2: Pipe(.field, Pipe(split("sep"), BinOp(cmp, Length(Input), N)))
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::Pipe { left: split_expr, right: cmp_expr } = right.as_ref() {
                            if let Expr::CallBuiltin { op: name, args } = split_expr.as_ref() {
                                if *name == BuiltinOp::Split && args.len() == 1 {
                                    if let Expr::Literal(Literal::Str(delim)) = &args[0] {
                                        if let Expr::BinOp { op, lhs: cmp_lhs, rhs: cmp_rhs } = cmp_expr.as_ref() {
                                            if matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                                                if matches!(cmp_lhs.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                                                    if let Expr::Literal(Literal::Num(n, _)) = cmp_rhs.as_ref() {
                                                        return Some((field.clone(), delim.clone(), *op, *n));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | split(sep) | .[i] + "lit" + .[j]` — split then concatenate indexed parts.
    /// Returns (field_name, delimiter, parts) where parts are SplitConcatPart.
    pub fn detect_field_split_concat(&self) -> Option<(String, String, Vec<SplitConcatPart>)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: split_expr, right: concat_expr } = right.as_ref() {
                        if let Expr::CallBuiltin { op: name, args } = split_expr.as_ref() {
                            if *name != BuiltinOp::Split || args.len() != 1 { return None; }
                            if let Expr::Literal(Literal::Str(delim)) = &args[0] {
                                let mut parts = Vec::new();
                                if Self::collect_split_concat_parts(concat_expr, &mut parts) && parts.len() >= 2 {
                                    return Some((field.clone(), delim.clone(), parts));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Recursively collect parts from a string add chain: .[i] + "lit" + .[j]
    fn collect_split_concat_parts(expr: &crate::ir::Expr, parts: &mut Vec<SplitConcatPart>) -> bool {
        use crate::ir::{Expr, BinOp, Literal};
        match expr {
            Expr::BinOp { op: BinOp::Add, lhs, rhs } => {
                if !Self::collect_split_concat_parts(lhs, parts) { return false; }
                Self::collect_split_concat_parts(rhs, parts)
            }
            Expr::Index { expr: base, key } if matches!(base.as_ref(), Expr::Input) => {
                if let Expr::Literal(Literal::Num(n, _)) = key.as_ref() {
                    parts.push(SplitConcatPart::Index(*n as i32));
                    true
                } else if let Expr::Negate { operand } = key.as_ref() {
                    // .[-N] → Negate(Num(N))
                    if let Expr::Literal(Literal::Num(n, _)) = operand.as_ref() {
                        parts.push(SplitConcatPart::Index(-(*n as i32)));
                        true
                    } else { false }
                } else { false }
            }
            Expr::Literal(Literal::Str(s)) => {
                parts.push(SplitConcatPart::Lit(s.clone()));
                true
            }
            _ => false,
        }
    }

    /// Detect `.field | str_op | length` chains — returns (field, op_name, op_arg).
    /// Handles ltrimstr, rtrimstr, ascii_downcase, ascii_upcase, explode.
    pub fn detect_field_strop_length(&self) -> Option<(String, String, Option<String>)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // Fully beta-reduced: UnaryOp(Length, UnaryOp(op, .field))
        if let Expr::UnaryOp { op: UnaryOp::Length, operand: inner } = expr {
            if let Expr::UnaryOp { op, operand: field_expr } = inner.as_ref() {
                if let Expr::Index { expr: base, key } = field_expr.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            match op {
                                UnaryOp::AsciiDowncase | UnaryOp::AsciiUpcase => {
                                    return Some((field.clone(), "identity_length".to_string(), None));
                                }
                                UnaryOp::Explode => {
                                    return Some((field.clone(), "explode".to_string(), None));
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        if let Expr::Pipe { left, right } = expr {
            let field = if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                else { return None; }
            } else { return None; };
            // Non-reduced: Pipe(.field, Pipe(str_op, length))
            if let Expr::Pipe { left: op_expr, right: len_expr } = right.as_ref() {
                if matches!(len_expr.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    match op_expr.as_ref() {
                        Expr::CallBuiltin { op: name, args } if args.len() == 1 => {
                            if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                match name {
                                    BuiltinOp::LtrimStr | BuiltinOp::RtrimStr => return Some((field, name.name().to_string(), Some(arg.clone()))),
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            // Beta-reduced: Pipe(.field, UnaryOp(Length, UnaryOp(op, Input)))
            if let Expr::UnaryOp { op: UnaryOp::Length, operand: inner } = right.as_ref() {
                match inner.as_ref() {
                    Expr::UnaryOp { op, operand } if matches!(operand.as_ref(), Expr::Input) => {
                        match op {
                            UnaryOp::AsciiDowncase | UnaryOp::AsciiUpcase => {
                                return Some((field, "identity_length".to_string(), None));
                            }
                            UnaryOp::Explode => {
                                return Some((field, "explode".to_string(), None));
                            }
                            _ => {}
                        }
                    }
                    Expr::Input => {
                        // .field | length — already handled by other fast paths
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Detect `.field | length | tostring` pattern.
    /// Returns field name if detected.
    pub fn detect_field_length_tostring(&self) -> Option<String> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // Beta-reduced: UnaryOp(ToString, UnaryOp(Length, Index(Input, field)))
        if let Expr::UnaryOp { op: UnaryOp::ToString, operand } = expr {
            if let Expr::UnaryOp { op: UnaryOp::Length, operand: inner } = operand.as_ref() {
                if let Expr::Index { expr: base, key } = inner.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            return Some(f.clone());
                        }
                    }
                }
            }
        }
        // Non-reduced: Pipe(.field, Pipe(Length(Input), ToString(Input)))
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        if let Expr::Pipe { left: rl, right: rr } = right.as_ref() {
                            if matches!(rl.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input))
                                && matches!(rr.as_ref(), Expr::UnaryOp { op: UnaryOp::ToString, operand } if matches!(operand.as_ref(), Expr::Input))
                            {
                                return Some(f.clone());
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `.field | length cmp N` — string length comparison.
    /// Returns (field, cmp_op, threshold).
    pub fn detect_field_length_cmp(&self) -> Option<(String, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // Beta-reduced: BinOp(cmp, UnaryOp(Length, Index(Input, field)), Literal(N))
        if let Expr::BinOp { op, lhs, rhs } = expr {
            if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
            if let Expr::UnaryOp { op: UnaryOp::Length, operand } = lhs.as_ref() {
                if let Expr::Index { expr: base, key } = operand.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((field.clone(), *op, *n));
                            }
                        }
                    }
                }
            }
        }
        // Non-reduced: Pipe(.field, Pipe(UnaryOp(Length, Input), BinOp(cmp, Input, Literal(N))))
        // or: Pipe(.field, BinOp(cmp, UnaryOp(Length, Input), Literal(N)))
        if let Expr::Pipe { left, right } = expr {
            let field = if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                else { return None; }
            } else { return None; };
            // Pipe(.field, Pipe(length, . > N))
            if let Expr::Pipe { left: len_expr, right: cmp_expr } = right.as_ref() {
                if matches!(len_expr.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    if let Expr::BinOp { op, lhs, rhs } = cmp_expr.as_ref() {
                        if matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                            if matches!(lhs.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                    return Some((field, *op, *n));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `[.x, .y] | sort | .[0]` — min of two numeric fields.
    /// Returns (field1, field2).
    pub fn detect_min_two_fields(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // Pipe(Collect(Comma(.x, .y)), Index(UnaryOp(Sort, Input), Literal(0)))
        // after beta-reduction: sort | .[0] → (sort)[0]
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Collect { generator } = left.as_ref() {
                if let Expr::Comma { left: f1, right: f2 } = generator.as_ref() {
                    let field1 = if let Expr::Index { expr: base, key } = f1.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                        else { return None; }
                    } else { return None; };
                    let field2 = if let Expr::Index { expr: base, key } = f2.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                        else { return None; }
                    } else { return None; };
                    // Beta-reduced form: Index(UnaryOp(Sort, Input), Literal(0))
                    if let Expr::Index { expr: sort_expr, key } = right.as_ref() {
                        if matches!(sort_expr.as_ref(), Expr::UnaryOp { op: UnaryOp::Sort, operand } if matches!(operand.as_ref(), Expr::Input)) {
                            if let Expr::Literal(Literal::Num(n, _)) = key.as_ref() {
                                if *n == 0.0 {
                                    return Some((field1, field2));
                                }
                            }
                        }
                    }
                    // Non-reduced form: Pipe(sort, .[0])
                    if let Expr::Pipe { left: sort_expr, right: idx_expr } = right.as_ref() {
                        if matches!(sort_expr.as_ref(), Expr::UnaryOp { op: UnaryOp::Sort, operand } if matches!(operand.as_ref(), Expr::Input)) {
                            if let Expr::Index { expr: base, key } = idx_expr.as_ref() {
                                if !matches!(base.as_ref(), Expr::Input) { return None; }
                                if let Expr::Literal(Literal::Num(n, _)) = key.as_ref() {
                                    if *n == 0.0 {
                                        return Some((field1, field2));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `[.x, .y] | max` or `[.x, .y] | min` — returns (field1, field2, is_max).
    pub fn detect_minmax_two_fields(&self) -> Option<(String, String, bool)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Collect { generator } = left.as_ref() {
                if let Expr::Comma { left: f1, right: f2 } = generator.as_ref() {
                    let field1 = if let Expr::Index { expr: base, key } = f1.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                        else { return None; }
                    } else { return None; };
                    let field2 = if let Expr::Index { expr: base, key } = f2.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() }
                        else { return None; }
                    } else { return None; };
                    match right.as_ref() {
                        Expr::UnaryOp { op: UnaryOp::Max, operand } if matches!(operand.as_ref(), Expr::Input) => {
                            return Some((field1, field2, true));
                        }
                        Expr::UnaryOp { op: UnaryOp::Min, operand } if matches!(operand.as_ref(), Expr::Input) => {
                            return Some((field1, field2, false));
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }

    /// Detect `[.f1, .f2, ...] | min` or `[.f1, .f2, ...] | max` with N >= 3 fields.
    /// Returns (fields, is_max).
    pub fn detect_minmax_n_fields(&self) -> Option<(Vec<String>, bool)> {
        use crate::ir::{Expr, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            let is_max = match right.as_ref() {
                Expr::UnaryOp { op: UnaryOp::Max, operand } if matches!(operand.as_ref(), Expr::Input) => true,
                Expr::UnaryOp { op: UnaryOp::Min, operand } if matches!(operand.as_ref(), Expr::Input) => false,
                _ => return None,
            };
            if let Expr::Collect { generator } = left.as_ref() {
                let mut fields = Vec::new();
                if collect_comma_fields(generator, &mut fields) && fields.len() >= 3 {
                    return Some((fields, is_max));
                }
            }
        }
        None
    }

    /// Detect comma-separated field access `.f1,.f2,...` pattern.
    /// Returns the list of field names if all branches are direct field accesses on input.
    pub fn detect_multi_field_access(&self) -> Option<Vec<String>> {
        let expr = self.detect_expr()?;
        let mut fields = Vec::new();
        if collect_comma_fields(expr, &mut fields) && fields.len() >= 2 {
            Some(fields)
        } else {
            None
        }
    }

    /// Detect `.field // literal` pattern (alternative with fallback).
    /// Returns (field_name, fallback_json_bytes).
    pub fn detect_field_alternative(&self) -> Option<(String, Vec<u8>)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Alternative { primary, fallback } = expr {
            if let Expr::Index { expr: base, key } = primary.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    let fb_bytes = match fallback.as_ref() {
                        Expr::Literal(Literal::Str(s)) => {
                            let mut v = Vec::with_capacity(s.len() + 2);
                            v.push(b'"');
                            for &b in s.as_bytes() {
                                match b {
                                    b'"' => v.extend_from_slice(b"\\\""),
                                    b'\\' => v.extend_from_slice(b"\\\\"),
                                    _ => v.push(b),
                                }
                            }
                            v.push(b'"');
                            v
                        }
                        Expr::Literal(Literal::Num(n, repr)) => {
                            if let Some(r) = repr.as_ref().filter(|r| crate::value::is_valid_json_number(r)) {
                                r.as_bytes().to_vec()
                            } else {
                                let i = *n as i64;
                                if i as f64 == *n {
                                    itoa::Buffer::new().format(i).as_bytes().to_vec()
                                } else {
                                    ryu::Buffer::new().format(*n).as_bytes().to_vec()
                                }
                            }
                        }
                        Expr::Literal(Literal::Null) => b"null".to_vec(),
                        Expr::Literal(Literal::True) => b"true".to_vec(),
                        Expr::Literal(Literal::False) => b"false".to_vec(),
                        _ => return None,
                    };
                    return Some((field.clone(), fb_bytes));
                }
            }
        }
        None
    }

    /// Detect `.field1 // .field2` pattern (field alternative with field fallback).
    /// Returns (primary_field, fallback_field) if detected.
    pub fn detect_field_field_alternative(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Alternative { primary, fallback } = expr {
            if let Expr::Index { expr: base1, key: key1 } = primary.as_ref() {
                if !matches!(base1.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f1)) = key1.as_ref() {
                    if let Expr::Index { expr: base2, key: key2 } = fallback.as_ref() {
                        if !matches!(base2.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                            return Some((f1.clone(), f2.clone()));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `if .field cmp N then literal_a else literal_b end` pattern.
    /// Returns (field, op, threshold, true_output_bytes, false_output_bytes).
    pub fn detect_cmp_branch_literals(&self) -> Option<(String, crate::ir::BinOp, f64, Vec<u8>, Vec<u8>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                    return None;
                }
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            if let (Some(t_bytes), Some(f_bytes)) = (const_expr_to_json(then_branch), const_expr_to_json(else_branch)) {
                                return Some((field.clone(), *op, *n, t_bytes, f_bytes));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `if (.field | unary_op) cmp N then LIT else LIT end`
    /// where unary_op is length, floor, ceil, round, fabs.
    /// Returns (field, unary_op, cmp_op, threshold, then_bytes, else_bytes).
    pub fn detect_field_unary_cmp_branch_literals(&self) -> Option<(String, crate::ir::UnaryOp, crate::ir::BinOp, f64, Vec<u8>, Vec<u8>)> {
        use crate::ir::{Expr, BinOp, UnaryOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                    return None;
                }
                if let Expr::UnaryOp { op: uop, operand } = lhs.as_ref() {
                    if !matches!(uop, UnaryOp::Length | UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Fabs | UnaryOp::Abs) {
                        return None;
                    }
                    if let Expr::Index { expr: base, key } = operand.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                if let (Some(t_bytes), Some(f_bytes)) = (const_expr_to_json(then_branch), const_expr_to_json(else_branch)) {
                                    return Some((field.clone(), *uop, *op, *n, t_bytes, f_bytes));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `if (.field | test("re")) then LIT else LIT end`
    /// Also handles startswith/endswith/contains as the condition.
    /// Returns (field, pattern, flags, then_bytes, else_bytes).
    pub fn detect_field_strfunc_cmp_branch_literals(&self) -> Option<(String, StrFuncCond, Vec<u8>, Vec<u8>)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            let (t_bytes, f_bytes) = match (const_expr_to_json(then_branch), const_expr_to_json(else_branch)) {
                (Some(t), Some(f)) => (t, f),
                _ => return None,
            };
            // Match Pipe(.field, test/startswith/endswith/contains)
            if let Expr::Pipe { left, right } = cond.as_ref() {
                if let Expr::Index { expr: base, key } = left.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Some(sf) = extract_strfunc_cond(right) {
                            return Some((field.clone(), sf, t_bytes, f_bytes));
                        }
                    }
                }
            }
            // Also match beta-reduced forms directly
            match cond.as_ref() {
                Expr::RegexTest { input_expr, re, flags } => {
                    if let Expr::Index { expr: base, key } = input_expr.as_ref() {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                if let Expr::Literal(Literal::Str(pattern)) = re.as_ref() {
                                    let flags_str = match flags.as_ref() {
                                        Expr::Literal(Literal::Null) => None,
                                        Expr::Literal(Literal::Str(f)) => Some(f.clone()),
                                        _ => return None,
                                    };
                                    return Some((field.clone(), StrFuncCond::Test(pattern.clone(), flags_str), t_bytes, f_bytes));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Detect `if .field cmp N then {remap} else {remap} end` where both branches
    /// are objects with all-field-access values. Condition compares field to constant.
    /// Returns (cmp_field, op, cmp_val, then_pairs, else_pairs).
    pub fn detect_cmp_branch_remaps(&self) -> Option<(String, crate::ir::BinOp, CmpVal, Vec<(String, String)>, Vec<(String, String)>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        fn extract_remap(e: &Expr) -> Option<Vec<(String, String)>> {
            if let Expr::ObjectConstruct { pairs } = e {
                let mut result = Vec::with_capacity(pairs.len());
                for (k, v) in pairs {
                    let key = if let Expr::Literal(Literal::Str(s)) = k { s.clone() } else { return None; };
                    if let Expr::Index { expr: base, key: fk } = v {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = fk.as_ref() {
                            result.push((key, f.clone()));
                        } else { return None; }
                    } else { return None; }
                }
                if result.is_empty() { return None; }
                Some(result)
            } else { None }
        }
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        let cmp_val = match rhs.as_ref() {
                            Expr::Literal(Literal::Num(n, _)) => CmpVal::Num(*n),
                            Expr::Literal(Literal::Str(s)) => CmpVal::Str(s.clone()),
                            _ => return None,
                        };
                        let then_pairs = extract_remap(then_branch)?;
                        let else_pairs = extract_remap(else_branch)?;
                        return Some((field.clone(), *op, cmp_val, then_pairs, else_pairs));
                    }
                }
            }
        }
        None
    }

    /// Detect `if .field op val then merge else . end` (conditional merge).
    /// Handles both prepend ({literal} + .) and append (. + {literal}), and both
    /// numeric (.field > N) and string (.field == "str") conditions.
    /// Returns (field, op, cmp_val, merge_pairs, is_prepend) where cmp_val is either
    /// CmpVal::Num(f64) or CmpVal::Str(String).
    pub fn detect_cmp_branch_merge(&self) -> Option<(String, crate::ir::BinOp, CmpVal, Vec<(String, Vec<u8>)>, bool)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(else_branch.as_ref(), Expr::Input) { return None; }
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                    return None;
                }
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        let cmp_val = match rhs.as_ref() {
                            Expr::Literal(Literal::Num(n, _)) => CmpVal::Num(*n),
                            Expr::Literal(Literal::Str(s)) => CmpVal::Str(s.clone()),
                            _ => return None,
                        };
                        // Check then branch: {literal} + . (prepend) or . + {literal} (append)
                        if let Expr::BinOp { op: BinOp::Add | BinOp::Mul, lhs: add_lhs, rhs: add_rhs } = then_branch.as_ref() {
                            let (obj_expr, is_prepend) = if matches!(add_rhs.as_ref(), Expr::Input) {
                                (add_lhs.as_ref(), true)
                            } else if matches!(add_lhs.as_ref(), Expr::Input) {
                                (add_rhs.as_ref(), false)
                            } else { return None; };
                            if let Expr::ObjectConstruct { pairs } = obj_expr {
                                let mut merge_pairs = Vec::new();
                                for (k, v) in pairs {
                                    let key_str = if let Expr::Literal(Literal::Str(s)) = k {
                                        s.clone()
                                    } else { return None; };
                                    let val_bytes = const_expr_to_json(v)?;
                                    merge_pairs.push((key_str, val_bytes));
                                }
                                if merge_pairs.is_empty() { return None; }
                                return Some((field.clone(), *op, cmp_val, merge_pairs, is_prepend));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `if .field == null then literal_a else literal_b end` (or `!= null`).
    /// Returns (field, is_eq_null, true_output_bytes, false_output_bytes).
    pub fn detect_field_null_branch_literals(&self) -> Option<(String, bool, Vec<u8>, Vec<u8>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Eq | BinOp::Ne) { return None; }
                // .field == null or .field != null
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if matches!(rhs.as_ref(), Expr::Literal(Literal::Null)) {
                            if let (Some(t_bytes), Some(f_bytes)) = (const_expr_to_json(then_branch), const_expr_to_json(else_branch)) {
                                return Some((field.clone(), matches!(op, BinOp::Eq), t_bytes, f_bytes));
                            }
                        }
                    }
                }
                // null == .field or null != .field (reversed)
                if let Expr::Index { expr: base, key } = rhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if matches!(lhs.as_ref(), Expr::Literal(Literal::Null)) {
                            if let (Some(t_bytes), Some(f_bytes)) = (const_expr_to_json(then_branch), const_expr_to_json(else_branch)) {
                                return Some((field.clone(), matches!(op, BinOp::Eq), t_bytes, f_bytes));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `if .field1 cmp .field2 then const else const end` pattern.
    /// Both branches must be constant (serializable to JSON).
    /// Returns (field1, op, field2, true_output_bytes, false_output_bytes).
    pub fn detect_field_field_cmp_branch(&self) -> Option<(String, crate::ir::BinOp, String, Vec<u8>, Vec<u8>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                    return None;
                }
                if let Expr::Index { expr: base1, key: key1 } = lhs.as_ref() {
                    if !matches!(base1.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field1)) = key1.as_ref() {
                        if let Expr::Index { expr: base2, key: key2 } = rhs.as_ref() {
                            if !matches!(base2.as_ref(), Expr::Input) { return None; }
                            if let Expr::Literal(Literal::Str(field2)) = key2.as_ref() {
                                if let (Some(t_bytes), Some(f_bytes)) = (const_expr_to_json(then_branch), const_expr_to_json(else_branch)) {
                                    return Some((field1.clone(), *op, field2.clone(), t_bytes, f_bytes));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `if .f1 cmp .f2 then .f3 else .f4 end` where branches are field accesses.
    /// Returns (cmp_f1, op, cmp_f2, then_field, else_field).
    pub fn detect_if_ff_cmp_then_fields(&self) -> Option<(String, crate::ir::BinOp, String, String, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                let f1 = if let Expr::Index { expr: b, key: k } = lhs.as_ref() {
                    if !matches!(b.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(s)) = k.as_ref() { s.clone() } else { return None; }
                } else { return None; };
                let f2 = if let Expr::Index { expr: b, key: k } = rhs.as_ref() {
                    if !matches!(b.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(s)) = k.as_ref() { s.clone() } else { return None; }
                } else { return None; };
                let then_f = if let Expr::Index { expr: b, key: k } = then_branch.as_ref() {
                    if !matches!(b.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(s)) = k.as_ref() { s.clone() } else { return None; }
                } else { return None; };
                let else_f = if let Expr::Index { expr: b, key: k } = else_branch.as_ref() {
                    if !matches!(b.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(s)) = k.as_ref() { s.clone() } else { return None; }
                } else { return None; };
                return Some((f1, *op, f2, then_f, else_f));
            }
        }
        None
    }

    /// Detect `if .f1 cmp .f2 then .f3 else .f4 end` where branches have arithmetic.
    /// E.g. `if .x > .y then .x - .y else .y - .x end`
    /// Returns (cmp_f1, op, cmp_f2, then_expr, else_expr) where exprs are RemapExpr.
    pub fn detect_if_ff_cmp_then_computed(&self) -> Option<(String, crate::ir::BinOp, String, RemapExpr, RemapExpr)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                let f1 = if let Expr::Index { expr: b, key: k } = lhs.as_ref() {
                    if !matches!(b.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(s)) = k.as_ref() { s.clone() } else { return None; }
                } else { return None; };
                let f2 = if let Expr::Index { expr: b, key: k } = rhs.as_ref() {
                    if !matches!(b.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(s)) = k.as_ref() { s.clone() } else { return None; }
                } else { return None; };
                let then_r = Self::classify_remap_value(then_branch)?;
                let else_r = Self::classify_remap_value(else_branch)?;
                // At least one branch must not be a simple field (otherwise detect_if_ff_cmp_then_fields handles it)
                if matches!(then_r, RemapExpr::Field(_)) && matches!(else_r, RemapExpr::Field(_)) { return None; }
                return Some((f1, *op, f2, then_r, else_r));
            }
        }
        None
    }

    /// Detect `if .f1 cmp .f2 then {remap} else {remap} end` where both branches
    /// are objects with all-field-access values.
    /// Returns (cmp_f1, op, cmp_f2, then_pairs, else_pairs).
    pub fn detect_if_ff_cmp_then_remaps(&self) -> Option<(String, crate::ir::BinOp, String, Vec<(String, String)>, Vec<(String, String)>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        fn extract_remap(e: &Expr) -> Option<Vec<(String, String)>> {
            if let Expr::ObjectConstruct { pairs } = e {
                let mut result = Vec::with_capacity(pairs.len());
                for (k, v) in pairs {
                    let key = if let Expr::Literal(Literal::Str(s)) = k { s.clone() } else { return None; };
                    if let Expr::Index { expr: base, key: fk } = v {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = fk.as_ref() {
                            result.push((key, f.clone()));
                        } else { return None; }
                    } else { return None; }
                }
                if result.is_empty() { return None; }
                Some(result)
            } else { None }
        }
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                let f1 = if let Expr::Index { expr: b, key: k } = lhs.as_ref() {
                    if !matches!(b.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(s)) = k.as_ref() { s.clone() } else { return None; }
                } else { return None; };
                let f2 = if let Expr::Index { expr: b, key: k } = rhs.as_ref() {
                    if !matches!(b.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(s)) = k.as_ref() { s.clone() } else { return None; }
                } else { return None; };
                let then_pairs = extract_remap(then_branch)?;
                let else_pairs = extract_remap(else_branch)?;
                return Some((f1, *op, f2, then_pairs, else_pairs));
            }
        }
        None
    }

    /// Detect `if .field cmp N then "\(.f1) lit" else "\(.f2) lit" end`
    /// Both branches are string interpolations referencing input fields.
    /// Returns (cmp_field, op, threshold, then_parts, else_parts).
    pub fn detect_cmp_branch_string_interp(&self) -> Option<(String, crate::ir::BinOp, f64, Vec<(bool, String)>, Vec<(bool, String)>)> {
        use crate::ir::{Expr, BinOp, Literal, StringPart};
        let expr = self.detect_expr()?;
        fn extract_interp_parts(e: &Expr) -> Option<Vec<(bool, String)>> {
            if let Expr::StringInterpolation { parts } = e {
                let mut result = Vec::new();
                for part in parts {
                    match part {
                        StringPart::Literal(s) => result.push((true, s.clone())),
                        StringPart::Expr(Expr::Index { expr: base, key }) => {
                            if !matches!(base.as_ref(), Expr::Input) { return None; }
                            if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                                result.push((false, f.clone()));
                            } else { return None; }
                        }
                        _ => return None,
                    }
                }
                if result.iter().any(|(is_lit, _)| !is_lit) { return Some(result); }
            }
            None
        }
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                    return None;
                }
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            if let (Some(t_parts), Some(f_parts)) = (extract_interp_parts(then_branch), extract_interp_parts(else_branch)) {
                                return Some((field.clone(), *op, *n, t_parts, f_parts));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `if cond then [arr1] else [arr2] end` where cond is a field comparison
    /// and both branches are arrays of classifiable remap expressions.
    /// Condition types: .field cmp N, .f1 cmp .f2
    /// Returns (IfArrayCond, then_elems, else_elems).
    pub fn detect_if_cmp_then_arrays(&self) -> Option<(IfArrayCond, Vec<RemapExpr>, Vec<RemapExpr>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let try_collect_arr = |e: &Expr| -> Option<Vec<RemapExpr>> {
            if let Expr::Collect { generator } = e {
                fn collect_e<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
                    match e {
                        Expr::Comma { left, right } => { collect_e(left, out); collect_e(right, out); }
                        _ => out.push(e),
                    }
                }
                let mut elems = Vec::new();
                collect_e(generator, &mut elems);
                if elems.len() < 2 { return None; }
                let mut rexprs = Vec::with_capacity(elems.len());
                for elem in &elems { rexprs.push(Self::classify_remap_value(elem)?); }
                Some(rexprs)
            } else { None }
        };
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            let then_arr = try_collect_arr(then_branch)?;
            let else_arr = try_collect_arr(else_branch)?;
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                // .field cmp N
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((IfArrayCond::FieldConst(field.clone(), *op, *n), then_arr, else_arr));
                            }
                            // .f1 cmp .f2
                            if let Expr::Index { expr: base2, key: key2 } = rhs.as_ref() {
                                if matches!(base2.as_ref(), Expr::Input) {
                                    if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                                        return Some((IfArrayCond::FieldField(field.clone(), *op, f2.clone()), then_arr, else_arr));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `if .field <arith_ops> <cmp> N then literal else literal end`.
    /// E.g. `if .x % 2 == 0 then "even" else "odd" end`
    /// Returns (field, arith_ops, cmp_op, threshold, true_bytes, false_bytes).
    pub fn detect_arith_cmp_branch_literals(&self) -> Option<(String, Vec<(crate::ir::BinOp, f64)>, crate::ir::BinOp, f64, Vec<u8>, Vec<u8>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if let Expr::BinOp { op: cmp_op, lhs, rhs } = cond.as_ref() {
                if !matches!(cmp_op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                    return None;
                }
                if let Expr::Literal(Literal::Num(threshold, _)) = rhs.as_ref() {
                    // LHS should be an arith chain ending in .field
                    let mut ops = Vec::new();
                    let mut cur = lhs.as_ref();
                    loop {
                        if let Expr::BinOp { op: aop, lhs: al, rhs: ar } = cur {
                            if !matches!(aop, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
                            if let Expr::Literal(Literal::Num(n, _)) = ar.as_ref() {
                                ops.push((*aop, *n));
                                cur = al.as_ref();
                            } else {
                                return None;
                            }
                        } else {
                            break;
                        }
                    }
                    if ops.is_empty() { return None; }
                    ops.reverse();
                    if let Expr::Index { expr: base, key } = cur {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            if let (Some(t_bytes), Some(f_bytes)) = (const_expr_to_json(then_branch), const_expr_to_json(else_branch)) {
                                return Some((field.clone(), ops, *cmp_op, *threshold, t_bytes, f_bytes));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect chained if-elif-else with field comparisons and field/literal outputs.
    /// `if .x > N then .x elif .x > M then .y else 0 end`
    /// Returns (branches, else_output). Only matches if it extends beyond detect_cmp_branch_literals
    /// (i.e., has >1 branch, or has a field output).
    pub fn detect_cond_chain(&self) -> Option<(Vec<CondBranch>, BranchOutput)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;

        let expr_to_output = |e: &Expr| -> Option<BranchOutput> {
            // empty
            if matches!(e, Expr::Empty) {
                return Some(BranchOutput::Empty);
            }
            // .field
            if let Expr::Index { expr: base, key } = e {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        return Some(BranchOutput::Field(f.clone()));
                    }
                }
                return None;
            }
            // Literal
            match e {
                Expr::Literal(Literal::Num(n, repr)) => {
                    let mut buf = Vec::new();
                    if let Some(r) = repr.as_ref().filter(|r| crate::value::is_valid_json_number(r)) {
                        buf.extend_from_slice(r.as_bytes());
                    } else {
                        let i = *n as i64;
                        if i as f64 == *n { buf.extend_from_slice(itoa::Buffer::new().format(i).as_bytes()); }
                        else { buf.extend_from_slice(ryu::Buffer::new().format(*n).as_bytes()); }
                    }
                    Some(BranchOutput::Literal(buf))
                }
                Expr::Literal(Literal::Str(s)) => {
                    let mut v = Vec::with_capacity(s.len() + 2);
                    v.push(b'"');
                    for &b in s.as_bytes() {
                        match b {
                            b'"' => v.extend_from_slice(b"\\\""),
                            b'\\' => v.extend_from_slice(b"\\\\"),
                            b'\n' => v.extend_from_slice(b"\\n"),
                            b'\r' => v.extend_from_slice(b"\\r"),
                            b'\t' => v.extend_from_slice(b"\\t"),
                            c if c < 0x20 => { v.extend_from_slice(format!("\\u{:04x}", c).as_bytes()); }
                            _ => v.push(b),
                        }
                    }
                    v.push(b'"');
                    Some(BranchOutput::Literal(v))
                }
                Expr::Literal(Literal::Null) => Some(BranchOutput::Literal(b"null".to_vec())),
                Expr::Literal(Literal::True) => Some(BranchOutput::Literal(b"true".to_vec())),
                Expr::Literal(Literal::False) => Some(BranchOutput::Literal(b"false".to_vec())),
                Expr::ObjectConstruct { pairs } if !pairs.is_empty() => {
                    let mut result = Vec::with_capacity(pairs.len());
                    for (k, v) in pairs {
                        let key = if let Expr::Literal(Literal::Str(s)) = k { s.clone() } else { return None; };
                        let rexpr = Self::classify_remap_value(v)?;
                        result.push((key, rexpr));
                    }
                    Some(BranchOutput::Remap(result))
                }
                _ => {
                    // Fallback: try to classify as a computed RemapExpr
                    if let Some(rexpr) = Self::classify_remap_value(e) {
                        Some(BranchOutput::Computed(rexpr))
                    } else {
                        None
                    }
                }
            }
        };

        let extract_cond = |cond: &Expr| -> Option<(String, Vec<(BinOp, f64)>, BinOp, CondRhs)> {
            if let Expr::BinOp { op, lhs, rhs } = cond {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                    return None;
                }
                // Unwrap arithmetic chain from LHS (e.g., .x % 2 → ops=[(Mod,2)], field="x")
                let mut arith_ops = Vec::new();
                let mut cur = lhs.as_ref();
                loop {
                    if let Expr::BinOp { op: aop, lhs: al, rhs: ar } = cur {
                        if matches!(aop, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                            if let Expr::Literal(Literal::Num(n, _)) = ar.as_ref() {
                                // jq raises on a zero divisor; keep the chain on generic eval (#1063)
                                if matches!(aop, BinOp::Div | BinOp::Mod) && *n == 0.0 { break; }
                                arith_ops.push((*aop, *n));
                                cur = al.as_ref();
                                continue;
                            }
                        }
                    }
                    break;
                }
                arith_ops.reverse();
                if let Expr::Index { expr: base, key } = cur {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        // .field [arith_ops...] cmp N
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            return Some((field.clone(), arith_ops, *op, CondRhs::Const(*n)));
                        }
                        // .field [arith_ops...] cmp "str"
                        if arith_ops.is_empty() {
                            if let Expr::Literal(Literal::Str(s)) = rhs.as_ref() {
                                return Some((field.clone(), arith_ops, *op, CondRhs::Str(s.clone())));
                            }
                        }
                        // .field == null / .field != null
                        if arith_ops.is_empty() && matches!(op, BinOp::Eq | BinOp::Ne) {
                            if matches!(rhs.as_ref(), Expr::Literal(Literal::Null)) {
                                return Some((field.clone(), arith_ops, *op, CondRhs::Null));
                            }
                            if matches!(rhs.as_ref(), Expr::Literal(Literal::True)) {
                                return Some((field.clone(), arith_ops, *op, CondRhs::Bool(true)));
                            }
                            if matches!(rhs.as_ref(), Expr::Literal(Literal::False)) {
                                return Some((field.clone(), arith_ops, *op, CondRhs::Bool(false)));
                            }
                        }
                        // .field1 [arith_ops...] cmp .field2 (only without arith ops)
                        if arith_ops.is_empty() {
                            if let Expr::Index { expr: base2, key: key2 } = rhs.as_ref() {
                                if matches!(base2.as_ref(), Expr::Input) {
                                    if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                                        return Some((field.clone(), arith_ops, *op, CondRhs::Field(f2.clone())));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // .field | startswith/endswith/contains("str")
            if let Expr::Pipe { left, right } = cond {
                if let Expr::Index { expr: base, key } = left.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            if let Expr::CallBuiltin { op: name, args } = right.as_ref() {
                                if args.len() == 1 {
                                    if let Expr::Literal(Literal::Str(s)) = &args[0] {
                                        let rhs = match name {
                                            BuiltinOp::StartsWith => Some(CondRhs::Startswith(s.clone())),
                                            BuiltinOp::EndsWith => Some(CondRhs::Endswith(s.clone())),
                                            BuiltinOp::Contains => Some(CondRhs::Contains(s.clone())),
                                            _ => None,
                                        };
                                        if let Some(r) = rhs {
                                            // Use BinOp::Eq as a dummy — the actual test is in the CondRhs
                                            return Some((field.clone(), Vec::new(), BinOp::Eq, r));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            None
        };

        // Recursively collect branches from nested IfThenElse
        let mut branches = Vec::new();
        let mut current = expr;
        loop {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = current {
                let (field, arith_ops, op, rhs) = extract_cond(cond)?;
                let output = expr_to_output(then_branch)?;
                branches.push(CondBranch { cond_field: field, cond_arith_ops: arith_ops, cond_op: op, cond_rhs: rhs, output });
                current = else_branch;
            } else {
                let else_output = expr_to_output(current)?;
                // Only use this if it adds value over detect_cmp_branch_literals / detect_arith_cmp_branch_literals
                let has_field_output = branches.iter().any(|b| matches!(b.output, BranchOutput::Field(_)))
                    || matches!(else_output, BranchOutput::Field(_));
                let has_remap_output = branches.iter().any(|b| matches!(b.output, BranchOutput::Remap(_)))
                    || matches!(else_output, BranchOutput::Remap(_));
                let has_field_rhs = branches.iter().any(|b| matches!(b.cond_rhs, CondRhs::Field(_)));
                let has_arith_ops = branches.iter().any(|b| !b.cond_arith_ops.is_empty());
                let has_str_func = branches.iter().any(|b| matches!(b.cond_rhs, CondRhs::Startswith(_) | CondRhs::Endswith(_) | CondRhs::Contains(_) | CondRhs::Test(_)));
                if branches.len() < 2 && !has_field_output && !has_remap_output && !has_field_rhs && !has_arith_ops && !has_str_func { return None; }
                // Single-branch arith with all-literal outputs → defer to detect_arith_cmp_branch_literals (faster handler)
                let has_computed_output = branches.iter().any(|b| matches!(b.output, BranchOutput::Computed(_)))
                    || matches!(else_output, BranchOutput::Computed(_));
                if branches.len() == 1 && has_arith_ops && !has_field_output && !has_field_rhs && !has_remap_output && !has_computed_output && !has_str_func
                    && matches!(branches[0].cond_rhs, CondRhs::Const(_))
                { return None; }
                return Some((branches, else_output));
            }
        }
    }

    /// Detect `select(.field > N) | .output_field` or `if .field > N then .output_field else empty end`.
    /// Returns (select_field, op, threshold, output_field).
    pub fn detect_select_cmp_then_field(&self) -> Option<(String, crate::ir::BinOp, f64, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        // Helper to extract (sel_field, op, threshold, output_field) from a cond+output pair
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, BinOp, f64, String)> {
            if let Expr::Index { expr: base, key } = output {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                let output_field = if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                    f.clone()
                } else { return None; };
                if let Expr::BinOp { op, lhs, rhs } = cond {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                        return None;
                    }
                    if let Expr::Index { expr: base2, key: key2 } = lhs.as_ref() {
                        if !matches!(base2.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(sel_field)) = key2.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((sel_field.clone(), *op, *n, output_field));
                            }
                        }
                    }
                }
            }
            None
        };
        // Form 1: select(.field > N) | .output_field = Pipe(IfThenElse{cond, then:Input, else:Empty}, Index)
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        // Form 2: if .field > N then .output_field else empty end = IfThenElse{cond, then:Index, else:Empty}
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.field <arith_ops> <cmp> N) | .output_field`.
    /// E.g. `select(.x % 2 == 0) | .name`
    /// Returns (cond_field, arith_ops, cmp_op, threshold, output_field).
    pub fn detect_select_arith_cmp_then_field(&self) -> Option<(String, Vec<(crate::ir::BinOp, f64)>, crate::ir::BinOp, f64, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, Vec<(BinOp, f64)>, BinOp, f64, String)> {
            if let Expr::Index { expr: base, key } = output {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                let output_field = if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() } else { return None; };
                if let Expr::BinOp { op, lhs, rhs } = cond {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                    if let Expr::Literal(Literal::Num(threshold, _)) = rhs.as_ref() {
                        // Unwrap arithmetic chain
                        let mut arith_ops = Vec::new();
                        let mut cur = lhs.as_ref();
                        loop {
                            if let Expr::BinOp { op: aop, lhs: al, rhs: ar } = cur {
                                if matches!(aop, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                                    if let Expr::Literal(Literal::Num(n, _)) = ar.as_ref() {
                                        // jq raises on a zero divisor; keep the chain on generic eval (#1063)
                                        if matches!(aop, BinOp::Div | BinOp::Mod) && *n == 0.0 { break; }
                                        arith_ops.push((*aop, *n));
                                        cur = al.as_ref();
                                        continue;
                                    }
                                }
                            }
                            break;
                        }
                        if arith_ops.is_empty() { return None; } // Plain .field cmp N handled by detect_select_cmp_then_field
                        arith_ops.reverse();
                        if let Expr::Index { expr: base2, key: key2 } = cur {
                            if !matches!(base2.as_ref(), Expr::Input) { return None; }
                            if let Expr::Literal(Literal::Str(field)) = key2.as_ref() {
                                return Some((field.clone(), arith_ops, *op, *threshold, output_field));
                            }
                        }
                    }
                }
            }
            None
        };
        // Form 1: select(.field arith cmp N) | .output = Pipe(IfThenElse, Index)
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        // Form 2: if .field arith cmp N then .output else empty end
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.field cmp N) | .output_field | unary_op`.
    /// Returns (select_field, op, threshold, output_field, unary_op).
    pub fn detect_select_cmp_then_field_unary(&self) -> Option<(String, crate::ir::BinOp, f64, String, crate::ir::UnaryOp)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // See `detect_field_unary_num`: Utf8ByteLength must not enter this
        // fast path because the dispatch would silently use length's value
        // for non-string inputs (#159).
        let is_supported = |op: &UnaryOp| matches!(op,
            UnaryOp::Length | UnaryOp::ToString |
            UnaryOp::AsciiDowncase | UnaryOp::AsciiUpcase |
            UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Sqrt | UnaryOp::Fabs | UnaryOp::Abs);
        // Form: Pipe(IfThenElse{cond, then: Input, else: Empty}, Pipe(.field, UnaryOp))
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    // Right is Pipe(.field, UnaryOp(op, Input))
                    if let Expr::Pipe { left: field_expr, right: unary_expr } = right.as_ref() {
                        if let Expr::Index { expr: base, key } = field_expr.as_ref() {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(out_field)) = key.as_ref() {
                                    if let Expr::UnaryOp { op: uop, operand } = unary_expr.as_ref() {
                                        if matches!(operand.as_ref(), Expr::Input) && is_supported(uop) {
                                            // Extract select condition
                                            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                                                if matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                                                    if let Expr::Index { expr: base2, key: key2 } = lhs.as_ref() {
                                                        if matches!(base2.as_ref(), Expr::Input) {
                                                            if let Expr::Literal(Literal::Str(sel_f)) = key2.as_ref() {
                                                                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                                                    return Some((sel_f.clone(), *op, *n, out_field.clone(), *uop));
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Beta-reduced: right is UnaryOp(op, Index(.field, Input))
                    if let Expr::UnaryOp { op: uop, operand } = right.as_ref() {
                        if is_supported(uop) {
                            if let Expr::Index { expr: base, key } = operand.as_ref() {
                                if matches!(base.as_ref(), Expr::Input) {
                                    if let Expr::Literal(Literal::Str(out_field)) = key.as_ref() {
                                        if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                                            if matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                                                if let Expr::Index { expr: base2, key: key2 } = lhs.as_ref() {
                                                    if matches!(base2.as_ref(), Expr::Input) {
                                                        if let Expr::Literal(Literal::Str(sel_f)) = key2.as_ref() {
                                                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                                                return Some((sel_f.clone(), *op, *n, out_field.clone(), *uop));
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.sel_field cmp N) | .upd_field |= (. arith M)` — select then field update.
    /// Returns (sel_field, cmp_op, threshold, upd_field, arith_op, arith_val).
    pub fn detect_select_cmp_then_update_num(&self) -> Option<(String, crate::ir::BinOp, f64, String, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            // Left: select = IfThenElse { cond, then: Input, else: Empty }
            let (sel_field, cmp_op, threshold) = if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
                if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
                if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                    if let Expr::Index { expr: base, key } = lhs.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                (f.clone(), *op, *n)
                            } else { return None; }
                        } else { return None; }
                    } else { return None; }
                } else { return None; }
            } else { return None; };
            // Right: Update { path: .field, update: BinOp(op, Input, Num) }
            if let Expr::Update { path_expr, update_expr } = right.as_ref() {
                if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(uf)) = key.as_ref() {
                        if let Expr::BinOp { op, lhs, rhs } = update_expr.as_ref() {
                            if matches!(lhs.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                    match op {
                                        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                                            return Some((sel_field, cmp_op, threshold, uf.clone(), *op, *n));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.sel_field cmp N) | .upd_field |= . + "str"` — select then string concat update.
    /// Returns (sel_field, cmp_op, threshold, upd_field, prefix, suffix).
    pub fn detect_select_cmp_then_update_str_concat(&self) -> Option<(String, crate::ir::BinOp, f64, String, String, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            let (sel_field, cmp_op, threshold) = if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
                if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
                if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                    if let Expr::Index { expr: base, key } = lhs.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                (f.clone(), *op, *n)
                            } else { return None; }
                        } else { return None; }
                    } else { return None; }
                } else { return None; }
            } else { return None; };
            if let Expr::Update { path_expr, update_expr } = right.as_ref() {
                if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(uf)) = key.as_ref() {
                        // . + "suffix"
                        if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = update_expr.as_ref() {
                            if matches!(lhs.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(s)) = rhs.as_ref() {
                                    return Some((sel_field, cmp_op, threshold, uf.clone(), String::new(), s.clone()));
                                }
                            }
                            if matches!(rhs.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(s)) = lhs.as_ref() {
                                    return Some((sel_field, cmp_op, threshold, uf.clone(), s.clone(), String::new()));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.f1 cmp N and .f2 cmp M) | .upd_field |= (. arith V)` — compound select then numeric update.
    /// Returns (logic_op, conds, upd_field, arith_op, arith_val).
    pub fn detect_select_compound_then_update_num(&self) -> Option<(crate::ir::BinOp, Vec<(String, crate::ir::BinOp, f64)>, String, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            // Left: compound select
            let (logic_op, conds) = if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
                if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
                let extract_cmp = |e: &Expr| -> Option<(String, BinOp, f64)> {
                    if let Expr::BinOp { op, lhs, rhs } = e {
                        if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                        if let Expr::Index { expr: base, key } = lhs.as_ref() {
                            if !matches!(base.as_ref(), Expr::Input) { return None; }
                            if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                    return Some((f.clone(), *op, *n));
                                }
                            }
                        }
                    }
                    None
                };
                fn collect_conds_u<'a>(e: &'a Expr, conj: BinOp, out: &mut Vec<&'a Expr>) -> bool {
                    if let Expr::BinOp { op, lhs, rhs } = e {
                        if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                            return collect_conds_u(lhs, conj, out) && collect_conds_u(rhs, conj, out);
                        }
                    }
                    out.push(e);
                    true
                }
                let mut found = None;
                for conj in [BinOp::And, BinOp::Or] {
                    if let Expr::BinOp { op, .. } = cond.as_ref() {
                        if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                            let mut parts = Vec::new();
                            if collect_conds_u(cond, conj, &mut parts) && parts.len() >= 2 {
                                let cmps: Vec<_> = parts.iter().filter_map(|e| extract_cmp(e)).collect();
                                if cmps.len() == parts.len() {
                                    found = Some((conj, cmps));
                                    break;
                                }
                            }
                        }
                    }
                }
                found?
            } else { return None; };
            // Right: Update { path: .field, update: BinOp(arith, Input, Num) }
            if let Expr::Update { path_expr, update_expr } = right.as_ref() {
                if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(uf)) = key.as_ref() {
                        if let Expr::BinOp { op, lhs, rhs } = update_expr.as_ref() {
                            if matches!(lhs.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                    match op {
                                        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                                            return Some((logic_op, conds, uf.clone(), *op, *n));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.f1 cmp .f2) | .output_field` — field-field comparison select then field.
    /// Returns (cmp_field1, op, cmp_field2, output_field).
    pub fn detect_select_field_cmp_field_then_field(&self) -> Option<(String, crate::ir::BinOp, String, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, BinOp, String, String)> {
            if let Expr::Index { expr: base, key } = output {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                let output_field = if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() } else { return None; };
                if let Expr::BinOp { op, lhs, rhs } = cond {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                    if let Expr::Index { expr: base1, key: key1 } = lhs.as_ref() {
                        if !matches!(base1.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(f1)) = key1.as_ref() {
                            if let Expr::Index { expr: base2, key: key2 } = rhs.as_ref() {
                                if !matches!(base2.as_ref(), Expr::Input) { return None; }
                                if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                                    return Some((f1.clone(), *op, f2.clone(), output_field));
                                }
                            }
                        }
                    }
                }
            }
            None
        };
        // Form 1: Pipe(IfThenElse{cond, then:Input, else:Empty}, Index)
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        // Form 2: IfThenElse{cond, then:Index, else:Empty}
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.field1 cmp .field2)` — field-to-field comparison in select, outputting whole object.
    /// Returns (field1, op, field2).
    pub fn detect_select_field_field_cmp(&self) -> Option<(String, crate::ir::BinOp, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
            if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
            if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                    return None;
                }
                if let Expr::Index { expr: base1, key: key1 } = lhs.as_ref() {
                    if !matches!(base1.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(f1)) = key1.as_ref() {
                        if let Expr::Index { expr: base2, key: key2 } = rhs.as_ref() {
                            if !matches!(base2.as_ref(), Expr::Input) { return None; }
                            if let Expr::Literal(Literal::Str(f2)) = key2.as_ref() {
                                return Some((f1.clone(), *op, f2.clone()));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.field > N) | RemapExpr` — select then single computed value.
    /// Returns (sel_field, op, threshold, output_expr).
    /// Only matches when the output is a computed expression (not a simple .field, which
    /// is handled by detect_select_cmp_then_field).
    pub fn detect_select_cmp_then_value(&self) -> Option<(String, crate::ir::BinOp, f64, RemapExpr)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, BinOp, f64, RemapExpr)> {
            let rexpr = Self::classify_remap_value(output)?;
            if matches!(rexpr, RemapExpr::Field(_)) { return None; } // handled by detect_select_cmp_then_field
            if let Expr::BinOp { op, lhs, rhs } = cond {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(sel_field)) = key.as_ref() {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            return Some((sel_field.clone(), *op, *n, rexpr));
                        }
                    }
                }
            }
            None
        };
        // Form 1: Pipe(select, expr)
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        // Form 2: if cond then expr else empty end
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.field > N) | {a:.x, b:.y}` or `if .field > N then {a:.x, b:.y} else empty end`.
    /// Returns (select_field, op, threshold, output_fields).
    pub fn detect_select_cmp_then_remap(&self) -> Option<(String, crate::ir::BinOp, f64, Vec<(String, String)>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let try_extract_remap = |cond: &Expr, output: &Expr| -> Option<(String, BinOp, f64, Vec<(String, String)>)> {
            let mut out_pairs = Vec::new();
            if let Expr::ObjectConstruct { pairs: entries } = output {
                for (k, v) in entries {
                    if let Expr::Literal(Literal::Str(key)) = k {
                        if let Expr::Index { expr: base, key: vk } = v {
                            if !matches!(base.as_ref(), Expr::Input) { return None; }
                            if let Expr::Literal(Literal::Str(field)) = vk.as_ref() {
                                out_pairs.push((key.clone(), field.clone()));
                            } else { return None; }
                        } else { return None; }
                    } else { return None; }
                }
                if out_pairs.is_empty() { return None; }
            } else { return None; }
            if let Expr::BinOp { op, lhs, rhs } = cond {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                    return None;
                }
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(sel_field)) = key.as_ref() {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            return Some((sel_field.clone(), *op, *n, out_pairs));
                        }
                    }
                }
            }
            None
        };
        // Form 1: Pipe(select(.field > N), {remap})
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract_remap(cond, right) { return Some(r); }
                }
            }
        }
        // Form 2: if .field > N then {remap} else empty end
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract_remap(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.field > N) | {a:.x, b:(.y*2)}` — select then computed remap.
    /// Returns (select_field, op, threshold, computed_remap_pairs).
    pub fn detect_select_cmp_then_computed_remap(&self) -> Option<(String, crate::ir::BinOp, f64, Vec<(String, RemapExpr)>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, BinOp, f64, Vec<(String, RemapExpr)>)> {
            if let Expr::ObjectConstruct { pairs } = output {
                if pairs.is_empty() { return None; }
                let mut result = Vec::with_capacity(pairs.len());
                let mut has_computed = false;
                for (k, v) in pairs {
                    let key = if let Expr::Literal(Literal::Str(s)) = k { s.clone() } else { return None; };
                    let rexpr = Self::classify_remap_value(v)?;
                    if !matches!(rexpr, RemapExpr::Field(_)) { has_computed = true; }
                    result.push((key, rexpr));
                }
                if !has_computed { return None; }
                if let Expr::BinOp { op, lhs, rhs } = cond {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                        return None;
                    }
                    if let Expr::Index { expr: base, key } = lhs.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(sel_field)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((sel_field.clone(), *op, *n, result));
                            }
                        }
                    }
                }
            }
            None
        };
        // Form 1: Pipe(select(.field > N), {computed_remap})
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        // Form 2: if .field > N then {computed_remap} else empty end
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.field > N) | [remap_expr, ...]` — select with array output.
    /// Returns (sel_field, cmp_op, threshold, array_elements).
    pub fn detect_select_cmp_then_array(&self) -> Option<(String, crate::ir::BinOp, f64, Vec<RemapExpr>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, BinOp, f64, Vec<RemapExpr>)> {
            if let Expr::Collect { generator } = output {
                fn collect_elems<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
                    match e {
                        Expr::Comma { left, right } => { collect_elems(left, out); collect_elems(right, out); }
                        _ => out.push(e),
                    }
                }
                let mut elems = Vec::new();
                collect_elems(generator, &mut elems);
                if elems.len() < 2 { return None; }
                let mut rexprs = Vec::with_capacity(elems.len());
                for elem in &elems {
                    rexprs.push(Self::classify_remap_value(elem)?);
                }
                if let Expr::BinOp { op, lhs, rhs } = cond {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) {
                        return None;
                    }
                    if let Expr::Index { expr: base, key } = lhs.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(sel_field)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((sel_field.clone(), *op, *n, rexprs));
                            }
                        }
                    }
                }
            }
            None
        };
        // Form 1: Pipe(select(.field > N), [arr])
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        // Form 2: if .field > N then [arr] else empty end
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.field > N) | {(.dynkey): rexpr}` — select then dynamic-key single-pair object.
    /// Returns (sel_field, cmp_op, threshold, dynkey_field, val_rexpr).
    pub fn detect_select_cmp_then_dynkey(&self) -> Option<(String, crate::ir::BinOp, f64, String, RemapExpr)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, BinOp, f64, String, RemapExpr)> {
            // Output: {(.field): rexpr}
            if let Expr::ObjectConstruct { pairs } = output {
                if pairs.len() != 1 { return None; }
                let (k, v) = &pairs[0];
                let dk = if let Expr::Index { expr: base, key } = k {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() } else { return None; }
                } else { return None; };
                let rexpr = Self::classify_remap_value(v)?;
                // Condition: .field cmp N
                if let Expr::BinOp { op, lhs, rhs } = cond {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                    if let Expr::Index { expr: base, key } = lhs.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(sf)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((sf.clone(), *op, *n, dk, rexpr));
                            }
                        }
                    }
                }
            }
            None
        };
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.field > N) | {(.dynkey): rexpr, static_key: rexpr2, ...}`.
    /// Returns (sel_field, cmp_op, threshold, dynkey_field, dynval_rexpr, static_pairs).
    pub fn detect_select_cmp_then_dynkey_mixed(&self) -> Option<(String, crate::ir::BinOp, f64, String, RemapExpr, Vec<(String, RemapExpr)>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, BinOp, f64, String, RemapExpr, Vec<(String, RemapExpr)>)> {
            if let Expr::ObjectConstruct { pairs } = output {
                if pairs.len() < 2 { return None; }
                let mut dyn_key: Option<(String, RemapExpr)> = None;
                let mut static_pairs: Vec<(String, RemapExpr)> = Vec::new();
                for (k, v) in pairs {
                    match k {
                        Expr::Index { expr: base, key } if matches!(base.as_ref(), Expr::Input) => {
                            if dyn_key.is_some() { return None; }
                            if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                                let rexpr = Self::classify_remap_value(v)?;
                                dyn_key = Some((f.clone(), rexpr));
                            } else { return None; }
                        }
                        Expr::Literal(Literal::Str(key_name)) => {
                            let rexpr = Self::classify_remap_value(v)?;
                            static_pairs.push((key_name.clone(), rexpr));
                        }
                        _ => return None,
                    }
                }
                let (dk_field, dk_val) = dyn_key?;
                if static_pairs.is_empty() { return None; }
                if let Expr::BinOp { op, lhs, rhs } = cond {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                    if let Expr::Index { expr: base, key } = lhs.as_ref() {
                        if !matches!(base.as_ref(), Expr::Input) { return None; }
                        if let Expr::Literal(Literal::Str(sf)) = key.as_ref() {
                            if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                return Some((sf.clone(), *op, *n, dk_field, dk_val, static_pairs));
                            }
                        }
                    }
                }
            }
            None
        };
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.field arith cmp N) | [array]` — arith select then array output.
    /// Returns (field, arith_ops, cmp_op, threshold, array_elements).
    pub fn detect_select_arith_cmp_then_array(&self) -> Option<(String, Vec<(crate::ir::BinOp, f64)>, crate::ir::BinOp, f64, Vec<RemapExpr>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, Vec<(BinOp, f64)>, BinOp, f64, Vec<RemapExpr>)> {
            if let Expr::Collect { generator } = output {
                fn collect_elems_ac<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
                    match e {
                        Expr::Comma { left, right } => { collect_elems_ac(left, out); collect_elems_ac(right, out); }
                        _ => out.push(e),
                    }
                }
                let mut elems = Vec::new();
                collect_elems_ac(generator, &mut elems);
                if elems.len() < 2 { return None; }
                let mut rexprs = Vec::with_capacity(elems.len());
                for elem in &elems { rexprs.push(Self::classify_remap_value(elem)?); }
                if let Expr::BinOp { op, lhs, rhs } = cond {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                    if let Expr::Literal(Literal::Num(threshold, _)) = rhs.as_ref() {
                        let mut arith_ops = Vec::new();
                        let mut cur = lhs.as_ref();
                        loop {
                            if let Expr::BinOp { op: aop, lhs: al, rhs: ar } = cur {
                                if matches!(aop, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                                    if let Expr::Literal(Literal::Num(n, _)) = ar.as_ref() {
                                        // jq raises on a zero divisor; keep the chain on generic eval (#1063)
                                        if matches!(aop, BinOp::Div | BinOp::Mod) && *n == 0.0 { break; }
                                        arith_ops.push((*aop, *n));
                                        cur = al.as_ref();
                                        continue;
                                    }
                                }
                            }
                            break;
                        }
                        if arith_ops.is_empty() { return None; }
                        arith_ops.reverse();
                        if let Expr::Index { expr: base, key } = cur {
                            if !matches!(base.as_ref(), Expr::Input) { return None; }
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                return Some((field.clone(), arith_ops, *op, *threshold, rexprs));
                            }
                        }
                    }
                }
            }
            None
        };
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.x > N and .y < M) | [array]` — compound select then array output.
    pub fn detect_select_compound_cmp_then_array(&self) -> Option<(crate::ir::BinOp, Vec<(String, crate::ir::BinOp, f64)>, Vec<RemapExpr>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let extract_cmp = |e: &Expr| -> Option<(String, BinOp, f64)> {
            if let Expr::BinOp { op, lhs, rhs } = e {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            return Some((field.clone(), *op, *n));
                        }
                    }
                }
            }
            None
        };
        fn collect_conds2<'a>(e: &'a Expr, conj: BinOp, out: &mut Vec<&'a Expr>) -> bool {
            if let Expr::BinOp { op, lhs, rhs } = e {
                if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                    return collect_conds2(lhs, conj, out) && collect_conds2(rhs, conj, out);
                }
            }
            out.push(e);
            true
        }
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(BinOp, Vec<(String, BinOp, f64)>, Vec<RemapExpr>)> {
            if let Expr::Collect { generator } = output {
                fn collect_elems2<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
                    match e {
                        Expr::Comma { left, right } => { collect_elems2(left, out); collect_elems2(right, out); }
                        _ => out.push(e),
                    }
                }
                let mut elems = Vec::new();
                collect_elems2(generator, &mut elems);
                if elems.len() < 2 { return None; }
                let mut rexprs = Vec::with_capacity(elems.len());
                for elem in &elems { rexprs.push(Self::classify_remap_value(elem)?); }
                for conj in [BinOp::And, BinOp::Or] {
                    if let Expr::BinOp { op, .. } = cond {
                        if std::mem::discriminant(op) == std::mem::discriminant(&conj) {
                            let mut parts = Vec::new();
                            if collect_conds2(cond, conj, &mut parts) && parts.len() >= 2 {
                                let cmps: Vec<_> = parts.iter().filter_map(|e| extract_cmp(e)).collect();
                                if cmps.len() == parts.len() {
                                    return Some((conj, cmps, rexprs));
                                }
                            }
                        }
                    }
                }
            }
            None
        };
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.x > .y) | [array]` — field-field compare select then array output.
    pub fn detect_select_ff_cmp_then_array(&self) -> Option<(String, crate::ir::BinOp, String, Vec<RemapExpr>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, BinOp, String, Vec<RemapExpr>)> {
            if let Expr::Collect { generator } = output {
                fn collect_elems3<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
                    match e {
                        Expr::Comma { left, right } => { collect_elems3(left, out); collect_elems3(right, out); }
                        _ => out.push(e),
                    }
                }
                let mut elems = Vec::new();
                collect_elems3(generator, &mut elems);
                if elems.len() < 2 { return None; }
                let mut rexprs = Vec::with_capacity(elems.len());
                for elem in &elems { rexprs.push(Self::classify_remap_value(elem)?); }
                if let Expr::BinOp { op, lhs, rhs } = cond {
                    if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                    if let (Expr::Index { expr: base1, key: key1 }, Expr::Index { expr: base2, key: key2 }) = (lhs.as_ref(), rhs.as_ref()) {
                        if !matches!(base1.as_ref(), Expr::Input) || !matches!(base2.as_ref(), Expr::Input) { return None; }
                        if let (Expr::Literal(Literal::Str(f1)), Expr::Literal(Literal::Str(f2))) = (key1.as_ref(), key2.as_ref()) {
                            return Some((f1.clone(), *op, f2.clone(), rexprs));
                        }
                    }
                }
            }
            None
        };
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.field == "str"|startswith/endswith/contains("str")) | [array]`.
    pub fn detect_select_str_then_array(&self) -> Option<(String, String, String, Vec<RemapExpr>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, String, String, Vec<RemapExpr>)> {
            if let Expr::Collect { generator } = output {
                fn collect_elems4<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
                    match e {
                        Expr::Comma { left, right } => { collect_elems4(left, out); collect_elems4(right, out); }
                        _ => out.push(e),
                    }
                }
                let mut elems = Vec::new();
                collect_elems4(generator, &mut elems);
                if elems.len() < 2 { return None; }
                let mut rexprs = Vec::with_capacity(elems.len());
                for elem in &elems { rexprs.push(Self::classify_remap_value(elem)?); }
                // Form A: .field == "str" / .field != "str"
                if let Expr::BinOp { op, lhs, rhs } = cond {
                    if matches!(op, BinOp::Eq | BinOp::Ne) {
                        if let Expr::Index { expr: base, key } = lhs.as_ref() {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                    if let Expr::Literal(Literal::Str(val)) = rhs.as_ref() {
                                        let test_type = if matches!(op, BinOp::Eq) { "eq" } else { "ne" };
                                        return Some((field.clone(), test_type.to_string(), val.clone(), rexprs));
                                    }
                                }
                            }
                        }
                    }
                }
                // Form B: .field | startswith/endswith/contains("str")
                if let Expr::Pipe { left: pl, right: pr } = cond {
                    if let Expr::Index { expr: base, key } = pl.as_ref() {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                if let Expr::CallBuiltin { op: name, args } = pr.as_ref() {
                                    if matches!(name, BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains) && args.len() == 1 {
                                        if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                            return Some((field.clone(), name.name().to_string(), arg.clone(), rexprs));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            None
        };
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.field == "str") | .output_field` or `select(.field | startswith("str")) | .output_field`.
    /// Returns (select_field, test_type, test_arg, output_field).
    /// test_type: "eq", "ne", "startswith", "endswith", "contains"
    pub fn detect_select_str_then_field(&self) -> Option<(String, String, String, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        // Must be Pipe(select, .field)
        if let Expr::Pipe { left, right } = expr {
            // Right side: .output_field
            let output_field = if let Expr::Index { expr: base, key } = right.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(f)) = key.as_ref() { f.clone() } else { return None; }
            } else { return None; };
            // Left side: select(cond) = IfThenElse { cond, then: Input, else: Empty }
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
                if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
                // Form A: select(.field == "str")
                if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                    if matches!(op, BinOp::Eq | BinOp::Ne) {
                        if let Expr::Index { expr: base, key } = lhs.as_ref() {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                    if let Expr::Literal(Literal::Str(val)) = rhs.as_ref() {
                                        let test_type = if matches!(op, BinOp::Eq) { "eq" } else { "ne" };
                                        return Some((field.clone(), test_type.to_string(), val.clone(), output_field));
                                    }
                                }
                            }
                        }
                    }
                }
                // Form B: select(.field | startswith/endswith/contains("str"))
                if let Expr::Pipe { left: pipe_left, right: pipe_right } = cond.as_ref() {
                    if let Expr::Index { expr: base, key } = pipe_left.as_ref() {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                if let Expr::CallBuiltin { op: name, args } = pipe_right.as_ref() {
                                    if matches!(name, BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains) && args.len() == 1 {
                                        if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                            return Some((field.clone(), name.name().to_string(), arg.clone(), output_field));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.field == "str"|startswith|endswith|contains("str")) | .upd_field |= (. arith N)`.
    /// Returns (cond_field, test_type, test_arg, upd_field, arith_op, arith_val).
    pub fn detect_select_str_then_update_num(&self) -> Option<(String, String, String, String, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            // Right side: Update { path: .field, update: BinOp(arith, Input, Num) }
            let (upd_field, arith_op, arith_val) = if let Expr::Update { path_expr, update_expr } = right.as_ref() {
                if let Expr::Index { expr: base, key } = path_expr.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(uf)) = key.as_ref() {
                        if let Expr::BinOp { op, lhs, rhs } = update_expr.as_ref() {
                            if matches!(lhs.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                    match op {
                                        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                                            (uf.clone(), *op, *n)
                                        }
                                        _ => return None,
                                    }
                                } else { return None; }
                            } else { return None; }
                        } else { return None; }
                    } else { return None; }
                } else { return None; }
            } else { return None; };
            // Left side: select(cond) = IfThenElse { cond, then: Input, else: Empty }
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if !matches!(then_branch.as_ref(), Expr::Input) { return None; }
                if !matches!(else_branch.as_ref(), Expr::Empty) { return None; }
                // Form A: select(.field == "str")
                if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                    if matches!(op, BinOp::Eq) {
                        if let Expr::Index { expr: base, key } = lhs.as_ref() {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                    if let Expr::Literal(Literal::Str(val)) = rhs.as_ref() {
                                        return Some((field.clone(), "eq".to_string(), val.clone(), upd_field, arith_op, arith_val));
                                    }
                                }
                            }
                        }
                    }
                }
                // Form B: select(.field | startswith/endswith/contains("str"))
                if let Expr::Pipe { left: pipe_left, right: pipe_right } = cond.as_ref() {
                    if let Expr::Index { expr: base, key } = pipe_left.as_ref() {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                if let Expr::CallBuiltin { op: name, args } = pipe_right.as_ref() {
                                    if matches!(name, BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains) && args.len() == 1 {
                                        if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                            return Some((field.clone(), name.name().to_string(), arg.clone(), upd_field, arith_op, arith_val));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect `select(.field == "str"|startswith|endswith|contains("str")) | str_add_chain`.
    /// Returns (cond_field, test_type, test_arg, string_add_parts).
    pub fn detect_select_str_then_str_chain(&self) -> Option<(String, String, String, Vec<StringAddPart>)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        fn collect_tostring_arith2(operand: &Expr, parts: &mut Vec<StringAddPart>) -> bool {
            if let Expr::Index { expr: base, key } = operand {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        parts.push(StringAddPart::FieldToString(f.clone()));
                        return true;
                    }
                }
            }
            let mut arith_ops = Vec::new();
            let mut cur = operand;
            loop {
                if let Expr::BinOp { op: aop, lhs, rhs } = cur {
                    if matches!(aop, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            // jq raises on a zero divisor; keep the chain on generic eval (#1063)
                            if matches!(aop, BinOp::Div | BinOp::Mod) && *n == 0.0 { break; }
                            arith_ops.push((*aop, *n));
                            cur = lhs.as_ref();
                            continue;
                        }
                    }
                }
                break;
            }
            if !arith_ops.is_empty() {
                arith_ops.reverse();
                if let Expr::Index { expr: base, key } = cur {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            parts.push(StringAddPart::FieldArithToString(f.clone(), arith_ops));
                            return true;
                        }
                    }
                }
            }
            false
        }
        fn collect_chain2(expr: &Expr, parts: &mut Vec<StringAddPart>) -> bool {
            match expr {
                Expr::BinOp { op: BinOp::Add, lhs, rhs } => {
                    collect_chain2(lhs, parts) && collect_chain2(rhs, parts)
                }
                Expr::Index { expr: base, key } if matches!(base.as_ref(), Expr::Input) => {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        parts.push(StringAddPart::Field(f.clone())); true
                    } else { false }
                }
                Expr::Literal(Literal::Str(s)) => {
                    parts.push(StringAddPart::Literal(s.clone())); true
                }
                Expr::UnaryOp { op: UnaryOp::ToString, operand } => {
                    collect_tostring_arith2(operand, parts)
                }
                _ => false,
            }
        }
        let extract_str_cond = |cond: &Expr| -> Option<(String, String, String)> {
            // Form A: .field == "str" / .field != "str"
            if let Expr::BinOp { op, lhs, rhs } = cond {
                if matches!(op, BinOp::Eq | BinOp::Ne) {
                    if let Expr::Index { expr: base, key } = lhs.as_ref() {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                if let Expr::Literal(Literal::Str(val)) = rhs.as_ref() {
                                    let tt = if matches!(op, BinOp::Eq) { "eq" } else { "ne" };
                                    return Some((field.clone(), tt.to_string(), val.clone()));
                                }
                            }
                        }
                    }
                }
            }
            // Form B: .field | startswith/endswith/contains("str")
            if let Expr::Pipe { left, right } = cond {
                if let Expr::Index { expr: base, key } = left.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            if let Expr::CallBuiltin { op: name, args } = right.as_ref() {
                                if matches!(name, BuiltinOp::StartsWith | BuiltinOp::EndsWith | BuiltinOp::Contains) && args.len() == 1 {
                                    if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                        return Some((field.clone(), name.name().to_string(), arg.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            None
        };
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, String, String, Vec<StringAddPart>)> {
            let mut parts = Vec::new();
            if !collect_chain2(output, &mut parts) || parts.len() < 2 { return None; }
            if !parts.iter().any(|p| !matches!(p, StringAddPart::Literal(_))) { return None; }
            let (f, tt, ta) = extract_str_cond(cond)?;
            Some((f, tt, ta, parts))
        };
        // Form 1: Pipe(select(str_cond), str_chain)
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        // Form 2: if str_cond then str_chain else empty end
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect `select(.field op N) | str_add_chain` pattern.
    /// Returns (select_field, op, threshold, string_add_parts).
    pub fn detect_select_cmp_then_str_chain(&self) -> Option<(String, crate::ir::BinOp, f64, Vec<StringAddPart>)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        fn collect_str_chain_tostring(operand: &Expr, parts: &mut Vec<StringAddPart>) -> bool {
            if let Expr::Index { expr: base, key } = operand {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        parts.push(StringAddPart::FieldToString(f.clone()));
                        return true;
                    }
                }
            }
            let mut arith_ops = Vec::new();
            let mut cur = operand;
            loop {
                if let Expr::BinOp { op: aop, lhs, rhs } = cur {
                    if matches!(aop, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            // jq raises on a zero divisor; keep the chain on generic eval (#1063)
                            if matches!(aop, BinOp::Div | BinOp::Mod) && *n == 0.0 { break; }
                            arith_ops.push((*aop, *n));
                            cur = lhs.as_ref();
                            continue;
                        }
                    }
                }
                break;
            }
            if !arith_ops.is_empty() {
                arith_ops.reverse();
                if let Expr::Index { expr: base, key } = cur {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                            parts.push(StringAddPart::FieldArithToString(f.clone(), arith_ops));
                            return true;
                        }
                    }
                }
            }
            false
        }
        fn collect_str_chain(expr: &Expr, parts: &mut Vec<StringAddPart>) -> bool {
            match expr {
                Expr::BinOp { op: BinOp::Add, lhs, rhs } => {
                    if !collect_str_chain(lhs, parts) { return false; }
                    if !collect_str_chain(rhs, parts) { return false; }
                    true
                }
                Expr::Index { expr: base, key } if matches!(base.as_ref(), Expr::Input) => {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        parts.push(StringAddPart::Field(f.clone()));
                        true
                    } else { false }
                }
                Expr::Literal(Literal::Str(s)) => {
                    parts.push(StringAddPart::Literal(s.clone()));
                    true
                }
                Expr::UnaryOp { op: UnaryOp::ToString, operand } => {
                    collect_str_chain_tostring(operand, parts)
                }
                _ => false,
            }
        }
        let try_extract = |cond: &Expr, output: &Expr| -> Option<(String, BinOp, f64, Vec<StringAddPart>)> {
            let mut parts = Vec::new();
            if !collect_str_chain(output, &mut parts) || parts.len() < 2 { return None; }
            if !parts.iter().any(|p| !matches!(p, StringAddPart::Literal(_))) { return None; }
            if let Expr::BinOp { op, lhs, rhs } = cond {
                if !matches!(op, BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne) { return None; }
                if let Expr::Index { expr: base, key } = lhs.as_ref() {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(sel_field)) = key.as_ref() {
                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                            return Some((sel_field.clone(), *op, *n, parts));
                        }
                    }
                }
            }
            None
        };
        // Form 1: Pipe(select(.f > N), str_chain)
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some(r) = try_extract(cond, right) { return Some(r); }
                }
            }
        }
        // Form 2: if .f > N then str_chain else empty end
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some(r) = try_extract(cond, then_branch) { return Some(r); }
            }
        }
        None
    }

    /// Detect simple field access `.field` pattern.
    /// Returns the field name if this is a direct field access on input.
    pub fn detect_field_access(&self) -> Option<String> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Index { expr: base, key } = expr {
            if !matches!(base.as_ref(), Expr::Input) { return None; }
            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                return Some(field.clone());
            }
        }
        None
    }

    /// Detect `. + {key: literal, ...}` or `. * {key: literal, ...}` — merge literal object into input.
    /// Returns list of (key, json_bytes) pairs for each literal entry.
    pub fn detect_obj_merge_literal(&self) -> Option<Vec<(String, Vec<u8>)>> {
        use crate::ir::{Expr, Literal, BinOp};
        let expr = self.detect_expr()?;
        if let Expr::BinOp { op: BinOp::Add | BinOp::Mul, lhs, rhs } = expr {
            if !matches!(lhs.as_ref(), Expr::Input) { return None; }
            if let Expr::ObjectConstruct { pairs } = rhs.as_ref() {
                let mut result = Vec::new();
                for (key_expr, val_expr) in pairs {
                    // Key must be a string literal
                    let key = match key_expr {
                        Expr::Literal(Literal::Str(s)) => s.clone(),
                        _ => return None,
                    };
                    // Value must be a literal
                    let json_bytes = match val_expr {
                        Expr::Literal(Literal::Num(n, _)) => {
                            let mut buf = Vec::new();
                            crate::value::push_jq_number_bytes(&mut buf, *n);
                            buf
                        }
                        Expr::Literal(Literal::Str(s)) => {
                            // JSON-encode the string
                            let mut buf = Vec::new();
                            buf.push(b'"');
                            for ch in s.bytes() {
                                match ch {
                                    b'"' => buf.extend_from_slice(b"\\\""),
                                    b'\\' => buf.extend_from_slice(b"\\\\"),
                                    b'\n' => buf.extend_from_slice(b"\\n"),
                                    b'\r' => buf.extend_from_slice(b"\\r"),
                                    b'\t' => buf.extend_from_slice(b"\\t"),
                                    c if c < 0x20 => {
                                        buf.extend_from_slice(format!("\\u{:04x}", c).as_bytes());
                                    }
                                    c => buf.push(c),
                                }
                            }
                            buf.push(b'"');
                            buf
                        }
                        Expr::Literal(Literal::Null) => b"null".to_vec(),
                        Expr::Literal(Literal::True) => b"true".to_vec(),
                        Expr::Literal(Literal::False) => b"false".to_vec(),
                        _ => return None,
                    };
                    result.push((key, json_bytes));
                }
                if !result.is_empty() {
                    return Some(result);
                }
            }
        }
        None
    }

    /// Detect `. + {key: numeric_expr(.fields)}` — object enrichment with computed numeric field.
    /// Returns (output_key, needed_fields, arith_expr) if detected.
    /// The raw byte handler scans for existing key, falls back to JIT if found.
    pub fn detect_obj_merge_computed(&self) -> Option<(String, Vec<String>, ArithExpr)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::BinOp { op: BinOp::Add | BinOp::Mul, lhs, rhs } = expr {
            if !matches!(lhs.as_ref(), Expr::Input) { return None; }
            if let Expr::ObjectConstruct { pairs } = rhs.as_ref() {
                if pairs.len() != 1 { return None; }
                let (key_expr, val_expr) = &pairs[0];
                let key = if let Expr::Literal(Literal::Str(k)) = key_expr { k.clone() } else { return None; };
                // Build ArithExpr from value expression
                let mut fields: Vec<String> = Vec::new();
                fn build_arith(expr: &Expr, fields: &mut Vec<String>) -> Option<ArithExpr> {
                    match expr {
                        Expr::BinOp { op, lhs, rhs } => {
                            if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
                            let l = build_arith(lhs, fields)?;
                            let r = build_arith(rhs, fields)?;
                            Some(ArithExpr::BinOp(*op, Box::new(l), Box::new(r)))
                        }
                        Expr::Index { expr: base, key } => {
                            if !matches!(base.as_ref(), Expr::Input) { return None; }
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                let idx = if let Some(pos) = fields.iter().position(|f| f == field) {
                                    pos
                                } else {
                                    fields.push(field.clone());
                                    fields.len() - 1
                                };
                                Some(ArithExpr::Field(idx))
                            } else { None }
                        }
                        Expr::Literal(Literal::Num(n, _)) => Some(ArithExpr::Const(*n)),
                        _ => None,
                    }
                }
                let arith = build_arith(val_expr, &mut fields)?;
                if fields.is_empty() { return None; } // All constants → detect_obj_merge_literal handles
                return Some((key, fields, arith));
            }
        }
        None
    }

    /// Detect `. + {k1: arith1, k2: arith2, ...}` — multi-field object enrichment with computed values.
    /// Returns Vec<(output_key, arith_expr)> and the shared list of input fields.
    pub fn detect_obj_merge_multi_computed(&self) -> Option<(Vec<(String, ArithExpr)>, Vec<String>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let expr = self.detect_expr()?;
        if let Expr::BinOp { op: BinOp::Add | BinOp::Mul, lhs, rhs } = expr {
            if !matches!(lhs.as_ref(), Expr::Input) { return None; }
            if let Expr::ObjectConstruct { pairs } = rhs.as_ref() {
                if pairs.len() < 2 { return None; } // single field → detect_obj_merge_computed handles
                let mut fields: Vec<String> = Vec::new();
                let mut result = Vec::with_capacity(pairs.len());
                fn build_arith(expr: &Expr, fields: &mut Vec<String>) -> Option<ArithExpr> {
                    match expr {
                        Expr::BinOp { op, lhs, rhs } => {
                            if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
                            let l = build_arith(lhs, fields)?;
                            let r = build_arith(rhs, fields)?;
                            Some(ArithExpr::BinOp(*op, Box::new(l), Box::new(r)))
                        }
                        Expr::Index { expr: base, key } => {
                            if !matches!(base.as_ref(), Expr::Input) { return None; }
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                let idx = if let Some(pos) = fields.iter().position(|f| f == field) {
                                    pos
                                } else {
                                    fields.push(field.clone());
                                    fields.len() - 1
                                };
                                Some(ArithExpr::Field(idx))
                            } else { None }
                        }
                        Expr::Literal(Literal::Num(n, _)) => Some(ArithExpr::Const(*n)),
                        _ => None,
                    }
                }
                for (key_expr, val_expr) in pairs {
                    let key = if let Expr::Literal(Literal::Str(k)) = key_expr { k.clone() } else { return None; };
                    let arith = build_arith(val_expr, &mut fields)?;
                    result.push((key, arith));
                }
                if fields.is_empty() { return None; }
                return Some((result, fields));
            }
        }
        None
    }

    /// Detect `walk(if type == "number" then . op N else . end)` pattern.
    /// Returns (op, N) for the numeric transformation.
    pub fn detect_walk_num_op(&self) -> Option<(crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::CallBuiltin { op: name, args } = expr {
            if *name != BuiltinOp::Walk || args.len() != 1 { return None; }
            // The body should be: if type == "number" then . op N else . end
            if let Expr::IfThenElse { cond, then_branch, else_branch } = &args[0] {
                // else_branch must be identity (.)
                if !matches!(else_branch.as_ref(), Expr::Input) { return None; }
                // cond: type == "number" (which is BinOp(Eq, UnaryOp(Type, Input), Literal(Str("number"))))
                if let Expr::BinOp { op: BinOp::Eq, lhs, rhs } = cond.as_ref() {
                    let is_type_number = match (lhs.as_ref(), rhs.as_ref()) {
                        (Expr::UnaryOp { op: UnaryOp::Type, operand }, Expr::Literal(Literal::Str(s)))
                            if matches!(operand.as_ref(), Expr::Input) && s == "number" => true,
                        (Expr::Literal(Literal::Str(s)), Expr::UnaryOp { op: UnaryOp::Type, operand })
                            if matches!(operand.as_ref(), Expr::Input) && s == "number" => true,
                        _ => false,
                    };
                    if !is_type_number { return None; }
                    // then_branch: . op N
                    if let Expr::BinOp { op, lhs: tl, rhs: tr } = then_branch.as_ref() {
                        if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) { return None; }
                        if matches!(tl.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Num(n, _)) = tr.as_ref() {
                                return Some((*op, *n));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect nested field access `.a.b` or `.a.b.c` pattern.
    /// Returns the chain of field names if this is chained field access on input.
    pub fn detect_nested_field_access(&self) -> Option<Vec<String>> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        let mut fields = Vec::new();
        let mut current = expr;
        loop {
            if let Expr::Index { expr: base, key } = current {
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    fields.push(field.clone());
                    current = base.as_ref();
                } else {
                    return None;
                }
            } else if matches!(current, Expr::Input) {
                break;
            } else {
                return None;
            }
        }
        if fields.len() >= 2 {
            fields.reverse(); // .a.b parses as Index(Index(Input, "a"), "b"), so reverse
            Some(fields)
        } else {
            None
        }
    }

    /// Returns the set of input fields accessed by the filter, if it can be statically determined.
    /// Returns None if the filter might access any/all fields (e.g., identity, iteration).
    pub fn needed_input_fields(&self) -> Option<Vec<String>> {
        // Use simplified expression (beta-reduced) since the raw parsed expression
        // has Input references that refer to pipe inputs, not the original input.
        // After beta-reduction, all Input references refer to the actual top-level input.
        let mut fields = Vec::new();
        if collect_input_fields(&self.simplified, &mut fields) {
            fields.sort();
            fields.dedup();
            // The projecting parser tracks found keys in a u64 bitset; past 64
            // distinct fields the missing-field backfill could duplicate keys.
            if !fields.is_empty() && fields.len() <= 64 {
                return Some(fields);
            }
        }
        None
    }
}

/// Recursively collect field names accessed from the input. Returns false if the filter
/// accesses the input in a way that requires the full object (e.g., bare `.`, `.[]`, `keys`).
/// Collect field names from a comma expression tree where all leaves are .field on input.
fn collect_comma_remap(expr: &crate::ir::Expr, elems: &mut Vec<RemapExpr>) -> bool {
    use crate::ir::Expr;
    match expr {
        Expr::Comma { left, right } => {
            collect_comma_remap(left, elems) && collect_comma_remap(right, elems)
        }
        _ => {
            if let Some(rexpr) = Filter::classify_remap_value(expr) {
                elems.push(rexpr);
                true
            } else {
                false
            }
        }
    }
}

/// Convert a literal Expr to its JSON byte representation.
fn literal_to_json_bytes(expr: &crate::ir::Expr) -> Option<Vec<u8>> {
    use crate::ir::{Expr, Literal};
    match expr {
        Expr::Literal(Literal::Num(n, _)) => {
            let mut buf = Vec::new();
            crate::value::push_jq_number_bytes(&mut buf, *n);
            Some(buf)
        }
        Expr::Literal(Literal::Str(s)) => {
            let mut buf = Vec::with_capacity(s.len() + 2);
            buf.push(b'"');
            for &b in s.as_bytes() {
                match b {
                    b'"' => buf.extend_from_slice(b"\\\""),
                    b'\\' => buf.extend_from_slice(b"\\\\"),
                    b'\n' => buf.extend_from_slice(b"\\n"),
                    b'\r' => buf.extend_from_slice(b"\\r"),
                    b'\t' => buf.extend_from_slice(b"\\t"),
                    _ if b < 0x20 => {
                        buf.extend_from_slice(format!("\\u{:04x}", b).as_bytes());
                    }
                    _ => buf.push(b),
                }
            }
            buf.push(b'"');
            Some(buf)
        }
        Expr::Literal(Literal::Null) => Some(b"null".to_vec()),
        Expr::Literal(Literal::True) => Some(b"true".to_vec()),
        Expr::Literal(Literal::False) => Some(b"false".to_vec()),
        _ => None,
    }
}

fn collect_comma_fields(expr: &crate::ir::Expr, fields: &mut Vec<String>) -> bool {
    use crate::ir::{Expr, Literal};
    match expr {
        Expr::Comma { left, right } => {
            collect_comma_fields(left, fields) && collect_comma_fields(right, fields)
        }
        Expr::Index { expr: base, key } => {
            if matches!(base.as_ref(), Expr::Input) {
                if let Expr::Literal(Literal::Str(s)) = key.as_ref() {
                    fields.push(s.clone());
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

fn collect_input_fields(expr: &crate::ir::Expr, fields: &mut Vec<String>) -> bool {
    use crate::ir::{Expr, Literal, StringPart};
    match expr {
        // Accessing a specific field of input: .foo
        Expr::Index { expr: base, key } => {
            if matches!(base.as_ref(), Expr::Input) {
                if let Expr::Literal(Literal::Str(s)) = key.as_ref() {
                    fields.push(s.clone());
                    return true;
                }
            }
            // General index: recurse
            collect_input_fields(base, fields) && collect_input_fields(key, fields)
        }
        Expr::IndexOpt { expr: base, key } => {
            if matches!(base.as_ref(), Expr::Input) {
                if let Expr::Literal(Literal::Str(s)) = key.as_ref() {
                    fields.push(s.clone());
                    return true;
                }
            }
            collect_input_fields(base, fields) && collect_input_fields(key, fields)
        }
        // Bare input access — the record itself reaches the output (or an
        // expression we cannot see through). This arm is the safety anchor:
        // every other arm may only pass when its outputs are literals or
        // fully-parsed derived values, never the (projected) record.
        Expr::Input => false,
        // Literals, variables and input-independent leaves
        Expr::Literal(_) | Expr::LoadVar { .. } => true,
        Expr::Empty | Expr::Not | Expr::Env | Expr::Loc { .. } | Expr::Builtins => true,
        // Pipe: when the left side passes the record through unchanged (bare
        // `.`, or the `select(cond)` desugar `if cond then . else empty end`),
        // the right side still reads top-level fields — walk it strictly.
        // Otherwise the left side's outputs are complete derived values, so
        // the right side needs no field collection at all; it only must not
        // touch the input stream.
        Expr::Pipe { left, right } => {
            if pipe_passes_record_through(left, fields) {
                collect_input_fields(right, fields)
            } else {
                collect_input_fields(left, fields) && projection_stream_safe(right)
            }
        }
        Expr::Comma { left, right } => {
            collect_input_fields(left, fields) && collect_input_fields(right, fields)
        }
        // Array construct and iteration over a derived value. Bare `.[]`
        // (`Each { input_expr: Input }`) is anchored by the Input arm.
        // NOTE: `Recurse { input_expr }` must NOT get this treatment:
        // `recurse(f)` yields the record itself before f's outputs.
        Expr::Collect { generator } => collect_input_fields(generator, fields),
        Expr::Each { input_expr } | Expr::EachOpt { input_expr } => {
            collect_input_fields(input_expr, fields)
        }
        // Object construct: check keys and values
        Expr::ObjectConstruct { pairs } => {
            pairs.iter().all(|(k, v)| collect_input_fields(k, fields) && collect_input_fields(v, fields))
        }
        // Conditionals
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            collect_input_fields(cond, fields) && collect_input_fields(then_branch, fields) && collect_input_fields(else_branch, fields)
        }
        // Binary/unary ops
        Expr::BinOp { lhs, rhs, .. } => collect_input_fields(lhs, fields) && collect_input_fields(rhs, fields),
        Expr::UnaryOp { operand, .. } => collect_input_fields(operand, fields),
        Expr::Negate { operand } => collect_input_fields(operand, fields),
        // Let binding (`e as $x | body` keeps `.` bound to the record in body)
        Expr::LetBinding { value, body, .. } => collect_input_fields(value, fields) && collect_input_fields(body, fields),
        // Select, alternative
        Expr::Alternative { primary, fallback } => collect_input_fields(primary, fields) && collect_input_fields(fallback, fields),
        // try/catch: the catch body sees the error value, which a strict-
        // passing try can only build from message strings or derived values —
        // never the record. The `?//` desugar (restore_dot) re-runs the catch
        // against the try's input, i.e. the record, so it stays strict.
        Expr::TryCatch { try_expr, catch_expr, restore_dot } => {
            collect_input_fields(try_expr, fields)
                && if *restore_dot {
                    collect_input_fields(catch_expr, fields)
                } else {
                    projection_stream_safe(catch_expr)
                }
        }
        Expr::StringInterpolation { parts } => parts.iter().all(|p| match p {
            StringPart::Literal(_) => true,
            StringPart::Expr(e) => collect_input_fields(e, fields),
        }),
        Expr::Format { expr, .. } => collect_input_fields(expr, fields),
        // Slice bounds are evaluated against the same `.` as the sliced expr
        Expr::Slice { expr, from, to } => {
            collect_input_fields(expr, fields)
                && from.as_ref().is_none_or(|e| collect_input_fields(e, fields))
                && to.as_ref().is_none_or(|e| collect_input_fields(e, fields))
        }
        Expr::Limit { count, generator } => {
            collect_input_fields(count, fields) && collect_input_fields(generator, fields)
        }
        Expr::Range { from, to, step } => {
            collect_input_fields(from, fields)
                && collect_input_fields(to, fields)
                && step.as_ref().is_none_or(|e| collect_input_fields(e, fields))
        }
        // reduce/foreach: source and init run against the record; update and
        // extract run against the accumulator/element, which are complete.
        Expr::Reduce { source, init, update, .. } => {
            collect_input_fields(source, fields)
                && collect_input_fields(init, fields)
                && projection_stream_safe(update)
        }
        Expr::Foreach { source, init, update, extract, .. } => {
            collect_input_fields(source, fields)
                && collect_input_fields(init, fields)
                && projection_stream_safe(update)
                && extract.as_ref().is_none_or(|e| projection_stream_safe(e))
        }
        Expr::Label { body, .. } => collect_input_fields(body, fields),
        Expr::Break { value, .. } => collect_input_fields(value, fields),
        // all/any predicates run against the generated elements (complete)
        Expr::AllShort { generator, predicate } | Expr::AnyShort { generator, predicate } => {
            collect_input_fields(generator, fields) && projection_stream_safe(predicate)
        }
        // NOTE: `Debug`/`Stderr` print their expr but pass the record
        // through, so they stay in the catch-all bail below.
        // Bare `error` throws the record itself as the error payload
        Expr::Error { msg } => msg.as_ref().is_some_and(|m| collect_input_fields(m, fields)),
        // Anything else: assume full input needed
        _ => false,
    }
}

/// True for pipe left-hand sides that yield the raw record itself (or
/// nothing): bare `.`, the `select(cond)` desugar (`if cond then . else
/// empty end`, either branch order), and chains of those. Field reads in
/// the conditions are collected; the caller then walks the right-hand side
/// strictly because the record flows into it.
fn pipe_passes_record_through(expr: &crate::ir::Expr, fields: &mut Vec<String>) -> bool {
    use crate::ir::Expr;
    match expr {
        Expr::Input => true,
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            matches!(then_branch.as_ref(), Expr::Input | Expr::Empty)
                && matches!(else_branch.as_ref(), Expr::Input | Expr::Empty)
                && collect_input_fields(cond, fields)
        }
        Expr::Pipe { left, right } => {
            pipe_passes_record_through(left, fields) && pipe_passes_record_through(right, fields)
        }
        _ => false,
    }
}

/// Scan a sub-expression whose `.` is bound to a *derived* (complete) value —
/// a pipe output, accumulator, generated element or error payload — rather
/// than the raw top-level record. Such an expression cannot read record
/// fields, so it needs no field collection; it only must not (a) pull more
/// records from the host input stream (`input`/`inputs` would observe
/// projected records), (b) read parser bookkeeping the projecting scanner is
/// not guaranteed to maintain (`input_line_number`/`input_filename`), or
/// (c) call an opaque function whose body this scan cannot see (recursive
/// user functions survive beta-reduction). The match is exhaustive on
/// purpose: a new `Expr` variant must be reviewed against (a)–(c) here.
fn projection_stream_safe(expr: &crate::ir::Expr) -> bool {
    use crate::ir::{BuiltinOp, Expr, StringPart};
    match expr {
        Expr::ReadInput | Expr::ReadInputs | Expr::FuncCall { .. } => false,
        Expr::CallBuiltin { op, args } => {
            !matches!(op, BuiltinOp::InputFilename | BuiltinOp::InputLineNumber)
                && args.iter().all(projection_stream_safe)
        }
        Expr::Input
        | Expr::Literal(_)
        | Expr::LoadVar { .. }
        | Expr::Empty
        | Expr::Not
        | Expr::Loc { .. }
        | Expr::Env
        | Expr::Builtins
        | Expr::ModuleMeta
        | Expr::GenLabel => true,
        Expr::BinOp { lhs, rhs, .. } => projection_stream_safe(lhs) && projection_stream_safe(rhs),
        Expr::UnaryOp { operand, .. } | Expr::Negate { operand } => projection_stream_safe(operand),
        Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => {
            projection_stream_safe(expr) && projection_stream_safe(key)
        }
        Expr::Pipe { left, right } | Expr::Comma { left, right } => {
            projection_stream_safe(left) && projection_stream_safe(right)
        }
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            projection_stream_safe(cond)
                && projection_stream_safe(then_branch)
                && projection_stream_safe(else_branch)
        }
        Expr::TryCatch { try_expr, catch_expr, .. } => {
            projection_stream_safe(try_expr) && projection_stream_safe(catch_expr)
        }
        Expr::Each { input_expr } | Expr::EachOpt { input_expr } | Expr::Recurse { input_expr } => {
            projection_stream_safe(input_expr)
        }
        Expr::LetBinding { value, body, .. } => {
            projection_stream_safe(value) && projection_stream_safe(body)
        }
        Expr::Reduce { source, init, update, .. } => {
            projection_stream_safe(source)
                && projection_stream_safe(init)
                && projection_stream_safe(update)
        }
        Expr::Foreach { source, init, update, extract, .. } => {
            projection_stream_safe(source)
                && projection_stream_safe(init)
                && projection_stream_safe(update)
                && extract.as_ref().is_none_or(|e| projection_stream_safe(e))
        }
        Expr::Collect { generator } => projection_stream_safe(generator),
        Expr::ObjectConstruct { pairs } => pairs
            .iter()
            .all(|(k, v)| projection_stream_safe(k) && projection_stream_safe(v)),
        Expr::Alternative { primary, fallback } => {
            projection_stream_safe(primary) && projection_stream_safe(fallback)
        }
        Expr::Range { from, to, step } => {
            projection_stream_safe(from)
                && projection_stream_safe(to)
                && step.as_ref().is_none_or(|e| projection_stream_safe(e))
        }
        Expr::Label { body, .. } => projection_stream_safe(body),
        Expr::Break { value, .. } => projection_stream_safe(value),
        Expr::Update { path_expr, update_expr } => {
            projection_stream_safe(path_expr) && projection_stream_safe(update_expr)
        }
        Expr::Assign { path_expr, value_expr } | Expr::Mutate { path_expr, value_expr, .. } => {
            projection_stream_safe(path_expr) && projection_stream_safe(value_expr)
        }
        Expr::PathExpr { expr } => projection_stream_safe(expr),
        Expr::SetPath { path, value } => {
            projection_stream_safe(path) && projection_stream_safe(value)
        }
        Expr::GetPath { path } => projection_stream_safe(path),
        Expr::DelPaths { paths } => projection_stream_safe(paths),
        Expr::StringInterpolation { parts } => parts.iter().all(|p| match p {
            StringPart::Literal(_) => true,
            StringPart::Expr(e) => projection_stream_safe(e),
        }),
        Expr::Limit { count, generator } => {
            projection_stream_safe(count) && projection_stream_safe(generator)
        }
        Expr::While { cond, update } | Expr::Until { cond, update } => {
            projection_stream_safe(cond) && projection_stream_safe(update)
        }
        Expr::Repeat { update } => projection_stream_safe(update),
        Expr::AllShort { generator, predicate } | Expr::AnyShort { generator, predicate } => {
            projection_stream_safe(generator) && projection_stream_safe(predicate)
        }
        Expr::Error { msg } => msg.as_ref().is_none_or(|m| projection_stream_safe(m)),
        Expr::Format { expr, .. } => projection_stream_safe(expr),
        Expr::ClosureOp { input_expr, key_expr, .. } => {
            projection_stream_safe(input_expr) && projection_stream_safe(key_expr)
        }
        Expr::RegexTest { input_expr, re, flags }
        | Expr::RegexMatch { input_expr, re, flags }
        | Expr::RegexCapture { input_expr, re, flags }
        | Expr::RegexScan { input_expr, re, flags } => {
            projection_stream_safe(input_expr)
                && projection_stream_safe(re)
                && projection_stream_safe(flags)
        }
        Expr::RegexSub { input_expr, re, tostr, flags }
        | Expr::RegexGsub { input_expr, re, tostr, flags } => {
            projection_stream_safe(input_expr)
                && projection_stream_safe(re)
                && projection_stream_safe(tostr)
                && projection_stream_safe(flags)
        }
        Expr::AlternativeDestructure { alternatives } => {
            alternatives.iter().all(projection_stream_safe)
        }
        Expr::Slice { expr, from, to } => {
            projection_stream_safe(expr)
                && from.as_ref().is_none_or(|e| projection_stream_safe(e))
                && to.as_ref().is_none_or(|e| projection_stream_safe(e))
        }
        Expr::Debug { expr } | Expr::Stderr { expr } => projection_stream_safe(expr),
        Expr::Memoize { key, body, .. } => {
            key.as_ref().is_none_or(|k| projection_stream_safe(k)) && projection_stream_safe(body)
        }
    }
}
