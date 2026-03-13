//! Filter execution: parser → IR → tree-walking interpreter.
//!
//! Primary path: our own parser + eval (full control, correct behavior).
//! Fallback: libjq execution (for filters we can't parse yet).

use std::rc::Rc;

use anyhow::{Result, bail};

use crate::ir::CompiledFunc;
use crate::value::Value;

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
                    BinOp::Mod => l % r,
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
}

/// A compiled jq filter, ready to execute.
pub struct Filter {
    program: String,
    /// Our parsed IR (if parsing succeeded).
    parsed: Option<(crate::ir::Expr, Vec<CompiledFunc>)>,
    /// Simplified expression for fast path detection (identity pipes stripped).
    simplified: Option<crate::ir::Expr>,
    /// JIT-compiled function (if JIT compilation succeeded).
    jit_fn: Option<crate::jit::JitFilterFn>,
    /// JIT compiler kept alive to own the compiled code.
    _jit_compiler: Option<Box<crate::jit::JitCompiler>>,
    lib_dirs: Vec<String>,
    /// Cached eval environment to avoid re-allocating per call.
    cached_env: std::cell::RefCell<Option<crate::eval::EnvRef>>,
}

/// Serialize a constant expression to compact JSON bytes.
/// Supports string, number, null, true, false, and constant ObjectConstruct/Collect.
fn const_expr_to_json(expr: &crate::ir::Expr) -> Option<Vec<u8>> {
    use crate::ir::{Expr, Literal};
    match expr {
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
                    b if b < 0x20 => { v.extend_from_slice(format!("\\u{:04x}", b).as_bytes()); }
                    _ => v.push(b),
                }
            }
            v.push(b'"');
            Some(v)
        }
        Expr::Literal(Literal::Num(n, repr)) => {
            if let Some(r) = repr {
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
            let mut buf = Vec::new();
            buf.push(b'{');
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 { buf.push(b','); }
                if let Expr::Literal(Literal::Str(key)) = k {
                    buf.push(b'"');
                    buf.extend_from_slice(key.as_bytes());
                    buf.push(b'"');
                    buf.push(b':');
                    buf.extend(const_expr_to_json(v)?);
                } else { return None; }
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

/// Recursively strip identity pipes and beta-reduce for fast path detection.
/// Pipe(Input, X) → X, Pipe(X, Input) → X.
/// Pipe(E, F) → F[E/.] when E is scalar and F has free Input.
/// Also applies semantic rewrites (to_entries|from_entries → identity).
fn simplify_expr(expr: &crate::ir::Expr) -> crate::ir::Expr {
    use crate::ir::{Expr, UnaryOp};
    match expr {
        Expr::Pipe { left, right } => {
            let sl = simplify_expr(left);
            let sr = simplify_expr(right);
            if matches!(&sl, Expr::Input) { return sr; }
            if matches!(&sr, Expr::Input) { return sl; }
            // Semantic: to_entries | from_entries → identity
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::ToEntries, .. })
                && matches!(&sr, Expr::UnaryOp { op: UnaryOp::FromEntries, operand } if matches!(operand.as_ref(), Expr::Input))
            {
                return Expr::Input;
            }
            // Semantic: to_entries | map(.key) → keys_unsorted
            // to_entries | map(.value) → [.[]]
            // Semantic: to_entries | map(.key) → keys_unsorted, to_entries | map(.value) → [.[]]
            // Also handles: to_entries | map(.key) | sort → keys (composed with trailing pipe)
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                // Helper: check if expr is map(.key) or map(.value) — returns Some("key") or Some("value")
                fn is_map_entry_field(e: &Expr) -> Option<&str> {
                    if let Expr::Collect { generator } = e {
                        if let Expr::Pipe { left: gl, right: gr } = generator.as_ref() {
                            if matches!(gl.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                                if let Expr::Index { expr: base, key } = gr.as_ref() {
                                    if matches!(base.as_ref(), Expr::Input) {
                                        if let Expr::Literal(crate::ir::Literal::Str(field)) = key.as_ref() {
                                            if field == "key" || field == "value" {
                                                return Some(field.as_str());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    None
                }
                // Direct: to_entries | map(.key) or to_entries | map(.value)
                if let Some(field) = is_map_entry_field(&sr) {
                    if field == "key" {
                        return Expr::UnaryOp { op: UnaryOp::KeysUnsorted, operand: Box::new(Expr::Input) };
                    } else {
                        return Expr::Collect { generator: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }) };
                    }
                }
                // Composed: to_entries | Pipe(map(.key/.value), tail) → rewrite left, keep tail
                if let Expr::Pipe { left: pl, right: pr } = &sr {
                    if let Some(field) = is_map_entry_field(pl) {
                        let rewritten = if field == "key" {
                            Expr::UnaryOp { op: UnaryOp::KeysUnsorted, operand: Box::new(Expr::Input) }
                        } else {
                            Expr::Collect { generator: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }) }
                        };
                        // Recursively simplify the new pipe
                        return simplify_expr(&Expr::Pipe {
                            left: Box::new(rewritten),
                            right: pr.clone(),
                        });
                    }
                }
            }
            // Semantic: keys_unsorted | sort → keys
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::KeysUnsorted, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Sort, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    return Expr::UnaryOp { op: UnaryOp::Keys, operand: Box::new(Expr::Input) };
                }
            }
            // Semantic: cmp_expr | not → inverted cmp_expr
            if matches!(&sr, Expr::Not) {
                if let Expr::BinOp { op, lhs, rhs } = &sl {
                    if let Some(inv) = op.invert_cmp() {
                        return Expr::BinOp { op: inv, lhs: lhs.clone(), rhs: rhs.clone() };
                    }
                }
            }
            // Beta-reduction: .x | . + 1 → .x + 1
            if sl.is_simple_scalar() && sr.is_input_free() {
                return sr.substitute_input(&sl);
            }
            // [gen] | map(f) = [gen] | [.[] | f] → [gen | f]
            // where gen is a comma of simple scalars and f is input-free
            if let Expr::Collect { generator: ref lg } = sl {
                if let Expr::Collect { generator: ref rg } = sr {
                    if let Expr::Pipe { left: ref rl, right: ref rr } = rg.as_ref() {
                        if matches!(rl.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            if rr.is_input_free() {
                                // Distribute f over each element of gen
                                fn distribute_map(gen: &Expr, f: &Expr) -> Expr {
                                    match gen {
                                        Expr::Comma { left, right } => {
                                            Expr::Comma {
                                                left: Box::new(distribute_map(left, f)),
                                                right: Box::new(distribute_map(right, f)),
                                            }
                                        }
                                        other => {
                                            let piped = Expr::Pipe {
                                                left: Box::new(other.clone()),
                                                right: Box::new(f.clone()),
                                            };
                                            simplify_expr(&piped)
                                        }
                                    }
                                }
                                return Expr::Collect { generator: Box::new(distribute_map(lg, rr)) };
                            }
                        }
                    }
                }
            }
            Expr::Pipe { left: Box::new(sl), right: Box::new(sr) }
        }
        // Recurse into IfThenElse conditions (select patterns)
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            let sc = simplify_expr(cond);
            let st = simplify_expr(then_branch);
            let se = simplify_expr(else_branch);
            // if .field then .field else F end → .field // F
            if let Expr::Index { expr: base_c, key: key_c } = &sc {
                if matches!(base_c.as_ref(), Expr::Input) {
                    if let Expr::Literal(crate::ir::Literal::Str(fc)) = key_c.as_ref() {
                        if let Expr::Index { expr: base_t, key: key_t } = &st {
                            if matches!(base_t.as_ref(), Expr::Input) {
                                if let Expr::Literal(crate::ir::Literal::Str(ft)) = key_t.as_ref() {
                                    if fc == ft {
                                        return Expr::Alternative {
                                            primary: Box::new(sc),
                                            fallback: Box::new(se),
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Expr::IfThenElse {
                cond: Box::new(sc),
                then_branch: Box::new(st),
                else_branch: Box::new(se),
            }
        }
        // Inline LetBinding: (E as $x | F) → F[$x := E] when E is simple
        Expr::LetBinding { var_index, value, body } => {
            let sv = simplify_expr(value);
            let sb = simplify_expr(body);
            if sv.is_simple_scalar() {
                return sb.substitute_var(*var_index, &sv);
            }
            Expr::LetBinding { var_index: *var_index, value: Box::new(sv), body: Box::new(sb) }
        }
        // Recurse into ObjectConstruct, BinOp, etc.
        Expr::ObjectConstruct { pairs } => {
            Expr::ObjectConstruct {
                pairs: pairs.iter().map(|(k, v)| (simplify_expr(k), simplify_expr(v))).collect(),
            }
        }
        Expr::BinOp { op, lhs, rhs } => {
            let sl = simplify_expr(lhs);
            let sr = simplify_expr(rhs);
            // {a:.x} + {b:.y} → {a:.x, b:.y} (merge object constructions)
            // Only when all keys are distinct string literals (no dedup needed)
            if matches!(op, crate::ir::BinOp::Add) {
                if let (Expr::ObjectConstruct { pairs: p1 }, Expr::ObjectConstruct { pairs: p2 }) = (&sl, &sr) {
                    let all_literal_keys = p1.iter().chain(p2.iter()).all(|(k, _)| {
                        matches!(k, Expr::Literal(crate::ir::Literal::Str(_)))
                    });
                    if all_literal_keys {
                        let mut keys: Vec<&str> = Vec::new();
                        let mut all_distinct = true;
                        for (k, _) in p1.iter().chain(p2.iter()) {
                            if let Expr::Literal(crate::ir::Literal::Str(s)) = k {
                                if keys.contains(&s.as_str()) { all_distinct = false; break; }
                                keys.push(s.as_str());
                            }
                        }
                        if all_distinct {
                            let mut merged = p1.clone();
                            merged.extend(p2.iter().cloned());
                            return Expr::ObjectConstruct { pairs: merged };
                        }
                    }
                }
            }
            Expr::BinOp { op: *op, lhs: Box::new(sl), rhs: Box::new(sr) }
        }
        Expr::StringInterpolation { parts } => {
            use crate::ir::StringPart;
            Expr::StringInterpolation {
                parts: parts.iter().map(|p| match p {
                    StringPart::Literal(s) => StringPart::Literal(s.clone()),
                    StringPart::Expr(e) => StringPart::Expr(simplify_expr(e)),
                }).collect(),
            }
        }
        Expr::UnaryOp { op, operand } => {
            Expr::UnaryOp { op: *op, operand: Box::new(simplify_expr(operand)) }
        }
        Expr::Collect { generator } => {
            Expr::Collect { generator: Box::new(simplify_expr(generator)) }
        }
        Expr::Comma { left, right } => {
            Expr::Comma { left: Box::new(simplify_expr(left)), right: Box::new(simplify_expr(right)) }
        }
        Expr::Index { expr, key } => {
            Expr::Index { expr: Box::new(simplify_expr(expr)), key: Box::new(simplify_expr(key)) }
        }
        Expr::IndexOpt { expr, key } => {
            Expr::IndexOpt { expr: Box::new(simplify_expr(expr)), key: Box::new(simplify_expr(key)) }
        }
        Expr::CallBuiltin { name, args } => {
            let sargs: Vec<_> = args.iter().map(|a| simplify_expr(a)).collect();
            // walk(.) → identity
            if name == "walk" && sargs.len() == 1 && matches!(&sargs[0], Expr::Input) {
                return Expr::Input;
            }
            Expr::CallBuiltin { name: name.clone(), args: sargs }
        }
        Expr::Alternative { primary, fallback } => {
            Expr::Alternative { primary: Box::new(simplify_expr(primary)), fallback: Box::new(simplify_expr(fallback)) }
        }
        Expr::Each { input_expr } => {
            Expr::Each { input_expr: Box::new(simplify_expr(input_expr)) }
        }
        Expr::EachOpt { input_expr } => {
            Expr::EachOpt { input_expr: Box::new(simplify_expr(input_expr)) }
        }
        Expr::Negate { operand } => {
            Expr::Negate { operand: Box::new(simplify_expr(operand)) }
        }
        Expr::Slice { expr, from, to } => {
            Expr::Slice {
                expr: Box::new(simplify_expr(expr)),
                from: from.as_ref().map(|e| Box::new(simplify_expr(e))),
                to: to.as_ref().map(|e| Box::new(simplify_expr(e))),
            }
        }
        Expr::Update { path_expr, update_expr } => {
            Expr::Update { path_expr: Box::new(simplify_expr(path_expr)), update_expr: Box::new(simplify_expr(update_expr)) }
        }
        Expr::Assign { path_expr, value_expr } => {
            Expr::Assign { path_expr: Box::new(simplify_expr(path_expr)), value_expr: Box::new(simplify_expr(value_expr)) }
        }
        Expr::TryCatch { try_expr, catch_expr } => {
            Expr::TryCatch { try_expr: Box::new(simplify_expr(try_expr)), catch_expr: Box::new(simplify_expr(catch_expr)) }
        }
        // delpaths([["field"]]) → del(.field)
        Expr::DelPaths { paths } => {
            use crate::ir::Literal;
            if let Expr::Collect { generator } = paths.as_ref() {
                if let Expr::Collect { generator: inner } = generator.as_ref() {
                    if let Expr::Literal(Literal::Str(field)) = inner.as_ref() {
                        return Expr::CallBuiltin {
                            name: "del".into(),
                            args: vec![Expr::Index {
                                expr: Box::new(Expr::Input),
                                key: Box::new(Expr::Literal(Literal::Str(field.clone()))),
                            }],
                        };
                    }
                }
            }
            Expr::DelPaths { paths: Box::new(simplify_expr(paths)) }
        }
        _ => expr.clone(),
    }
}

/// Serialize a constant expression to JSON bytes. Returns false if expression is not fully constant.
fn push_const_json(expr: &crate::ir::Expr, buf: &mut Vec<u8>) -> bool {
    use crate::ir::{Expr, Literal};
    match expr {
        Expr::Literal(Literal::Null) => { buf.extend_from_slice(b"null"); true }
        Expr::Literal(Literal::True) => { buf.extend_from_slice(b"true"); true }
        Expr::Literal(Literal::False) => { buf.extend_from_slice(b"false"); true }
        Expr::Literal(Literal::Num(_, Some(raw))) => { buf.extend_from_slice(raw.as_bytes()); true }
        Expr::Literal(Literal::Num(n, None)) => {
            crate::value::push_jq_number_bytes(buf, *n);
            true
        }
        Expr::Literal(Literal::Str(s)) => {
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
                    },
                    _ => buf.push(b),
                }
            }
            buf.push(b'"');
            true
        }
        Expr::ObjectConstruct { pairs } => {
            buf.push(b'{');
            for (i, (key, val)) in pairs.iter().enumerate() {
                if i > 0 { buf.push(b','); }
                // Key must be a literal string
                if let Expr::Literal(Literal::Str(k)) = key {
                    buf.push(b'"');
                    buf.extend_from_slice(k.as_bytes());
                    buf.push(b'"');
                } else {
                    return false;
                }
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
    /// Get the inner expression for pattern detection, unwrapping top-level
    /// `try EXPR` (TryCatch with Empty catch) since the raw byte fast paths
    /// handle missing fields gracefully (return null/nothing).
    fn detect_expr(&self) -> Option<&crate::ir::Expr> {
        // Use simplified expression (identity pipes stripped) for pattern detection
        if let Some(ref simplified) = self.simplified {
            if let crate::ir::Expr::TryCatch { try_expr, catch_expr } = simplified {
                if matches!(catch_expr.as_ref(), crate::ir::Expr::Empty) {
                    return Some(try_expr);
                }
            }
            return Some(simplified);
        }
        let (ref expr, _) = self.parsed.as_ref()?;
        if let crate::ir::Expr::TryCatch { try_expr, catch_expr } = expr {
            if matches!(catch_expr.as_ref(), crate::ir::Expr::Empty) {
                return Some(try_expr);
            }
        }
        Some(expr)
    }

    pub fn new(program: &str) -> Result<Self> {
        Self::with_options(program, &[], true)
    }

    pub fn with_lib_dirs(program: &str, lib_dirs: &[String]) -> Result<Self> {
        Self::with_options(program, lib_dirs, true)
    }

    pub fn with_options(program: &str, lib_dirs: &[String], use_jit: bool) -> Result<Self> {
        // Try our parser first
        let parsed = match crate::parser::Parser::parse_with_libs(program, lib_dirs) {
            Ok(result) => Some((result.expr, result.funcs)),
            Err(_e) => {
                // Fall back to libjq for compilation check
                let mut jq = crate::bytecode::JqState::new()?;
                let _bc = jq.compile(program)?;
                None
            }
        };

        // Try JIT compilation for the parsed expression
        let mut jit_fn = None;
        let mut jit_compiler = None;
        if use_jit {
            if let Some((ref expr, ref funcs)) = parsed {
                if crate::jit::is_jit_compilable_with_funcs(expr, funcs) {
                    if let Ok(mut compiler) = crate::jit::JitCompiler::new() {
                        if let Ok(func) = compiler.compile_with_funcs(expr, funcs) {
                            jit_fn = Some(func);
                            jit_compiler = Some(Box::new(compiler));
                        }
                    }
                }
            }
        }

        let simplified = parsed.as_ref().map(|(expr, _)| simplify_expr(expr));

        Ok(Filter {
            program: program.to_string(),
            parsed,
            simplified,
            jit_fn,
            _jit_compiler: jit_compiler,
            lib_dirs: lib_dirs.to_vec(),
            cached_env: std::cell::RefCell::new(None),
        })
    }

    /// Try to JIT-compile this filter if not already JIT'd.
    /// Call this after determining the input is large enough to justify compilation.
    pub fn compile_jit(&mut self) {
        if self.jit_fn.is_some() { return; }
        if let Some((ref expr, ref funcs)) = self.parsed {
            if crate::jit::is_jit_compilable_with_funcs(expr, funcs) {
                if let Ok(mut compiler) = crate::jit::JitCompiler::new() {
                    if let Ok(func) = compiler.compile_with_funcs(expr, funcs) {
                        self.jit_fn = Some(func);
                        self._jit_compiler = Some(Box::new(compiler));
                    }
                }
            }
        }
    }

    /// Returns true if this filter has a JIT-compiled function.
    pub fn has_jit(&self) -> bool {
        self.jit_fn.is_some()
    }

    /// Returns true if this filter has loop constructs that benefit from JIT.
    /// Specifically: Update (.[] |= f), While/Until/Repeat, and Reduce/Foreach
    /// whose source references the input (e.g. `.[]` but not `range(N)`).
    /// For constant-range reduces on small inputs, eval.rs handles them efficiently.
    pub fn has_loop_constructs(&self) -> bool {
        use crate::ir::Expr;
        fn references_input(e: &Expr) -> bool {
            match e {
                Expr::Input => true,
                Expr::Pipe { left, right } | Expr::Comma { left, right }
                | Expr::BinOp { lhs: left, rhs: right, .. } => {
                    references_input(left) || references_input(right)
                }
                Expr::UnaryOp { operand, .. } | Expr::Negate { operand }
                | Expr::Collect { generator: operand } => references_input(operand),
                Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => {
                    references_input(expr) || references_input(key)
                }
                Expr::Range { from, to, step } => {
                    references_input(from) || references_input(to)
                    || step.as_ref().map_or(false, |s| references_input(s))
                }
                Expr::CallBuiltin { args, .. } => args.iter().any(|a| references_input(a)),
                _ => false,
            }
        }
        fn check(e: &Expr) -> bool {
            match e {
                Expr::Update { .. } => true,
                Expr::While { .. } | Expr::Until { .. } | Expr::Repeat { .. } => true,
                Expr::Reduce { source, .. } | Expr::Foreach { source, .. } => {
                    references_input(source)
                }
                _ => false,
            }
        }
        fn walk(e: &Expr) -> bool {
            if check(e) { return true; }
            match e {
                Expr::Pipe { left, right } | Expr::Comma { left, right }
                | Expr::BinOp { lhs: left, rhs: right, .. }
                | Expr::Alternative { primary: left, fallback: right } => {
                    walk(left) || walk(right)
                }
                Expr::UnaryOp { operand, .. } | Expr::Negate { operand }
                | Expr::Collect { generator: operand } | Expr::Each { input_expr: operand }
                | Expr::EachOpt { input_expr: operand } | Expr::Recurse { input_expr: operand } => {
                    walk(operand)
                }
                Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => walk(expr) || walk(key),
                Expr::IfThenElse { cond, then_branch, else_branch } => {
                    walk(cond) || walk(then_branch) || walk(else_branch)
                }
                Expr::TryCatch { try_expr, catch_expr } => walk(try_expr) || walk(catch_expr),
                Expr::Reduce { source, init, update, .. } => walk(source) || walk(init) || walk(update),
                Expr::Foreach { source, init, update, extract, .. } => {
                    walk(source) || walk(init) || walk(update) || extract.as_ref().map_or(false, |e| walk(e))
                }
                Expr::Slice { expr, from, to } => {
                    walk(expr) || from.as_ref().map_or(false, |e| walk(e))
                    || to.as_ref().map_or(false, |e| walk(e))
                }
                Expr::ObjectConstruct { pairs } => pairs.iter().any(|(k, v)| walk(k) || walk(v)),
                Expr::LetBinding { value, body, .. } => walk(value) || walk(body),
                Expr::Label { body, .. } => walk(body),
                Expr::CallBuiltin { args, .. } => args.iter().any(|a| walk(a)),
                Expr::Update { path_expr, update_expr } | Expr::Assign { path_expr, value_expr: update_expr } => {
                    walk(path_expr) || walk(update_expr)
                }
                _ => false,
            }
        }
        if let Some((ref expr, _)) = self.parsed {
            walk(expr)
        } else {
            false
        }
    }

    /// Returns true if the filter uses `input` or `inputs` anywhere.
    pub fn uses_inputs(&self) -> bool {
        use crate::ir::Expr;
        fn walk(e: &Expr) -> bool {
            match e {
                Expr::ReadInput | Expr::ReadInputs => true,
                Expr::Pipe { left, right } | Expr::Comma { left, right }
                | Expr::BinOp { lhs: left, rhs: right, .. }
                | Expr::Alternative { primary: left, fallback: right } => walk(left) || walk(right),
                Expr::UnaryOp { operand, .. } | Expr::Negate { operand }
                | Expr::Collect { generator: operand } | Expr::Each { input_expr: operand }
                | Expr::EachOpt { input_expr: operand } | Expr::Recurse { input_expr: operand } => walk(operand),
                Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => walk(expr) || walk(key),
                Expr::IfThenElse { cond, then_branch, else_branch } => walk(cond) || walk(then_branch) || walk(else_branch),
                Expr::TryCatch { try_expr, catch_expr } => walk(try_expr) || walk(catch_expr),
                Expr::Reduce { source, init, update, .. } => walk(source) || walk(init) || walk(update),
                Expr::Foreach { source, init, update, extract, .. } => walk(source) || walk(init) || walk(update) || extract.as_ref().map_or(false, |e| walk(e)),
                Expr::Slice { expr, from, to } => walk(expr) || from.as_ref().map_or(false, |e| walk(e)) || to.as_ref().map_or(false, |e| walk(e)),
                Expr::ObjectConstruct { pairs } => pairs.iter().any(|(k, v)| walk(k) || walk(v)),
                Expr::LetBinding { value, body, .. } => walk(value) || walk(body),
                Expr::Label { body, .. } => walk(body),
                Expr::CallBuiltin { args, .. } => args.iter().any(|a| walk(a)),
                Expr::Update { path_expr, update_expr } | Expr::Assign { path_expr, value_expr: update_expr } => walk(path_expr) || walk(update_expr),
                Expr::ClosureOp { input_expr, key_expr, .. } => walk(input_expr) || walk(key_expr),
                Expr::Format { expr: e, .. } => walk(e),
                Expr::Limit { count, generator } => walk(count) || walk(generator),
                Expr::While { cond, update, .. } | Expr::Until { cond, update } => walk(cond) || walk(update),
                Expr::Repeat { update, .. } => walk(update),
                Expr::Range { from, to, step } => {
                    walk(from) || walk(to) || step.as_ref().map_or(false, |s| walk(s))
                }
                _ => false,
            }
        }
        if let Some((ref expr, _)) = self.parsed {
            walk(expr)
        } else {
            false
        }
    }

    /// Returns true if this filter is a simple identity (`.`) that passes through input unchanged.
    /// Also recognizes semantic equivalences like `to_entries | from_entries`.
    pub fn is_identity(&self) -> bool {
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        Self::expr_is_identity(expr)
    }

    fn expr_is_identity(expr: &crate::ir::Expr) -> bool {
        use crate::ir::{Expr, UnaryOp};
        match expr {
            Expr::Input => true,
            Expr::Pipe { left, right } => {
                // . | X → X, X | . → X (recursive identity simplification)
                if Self::expr_is_identity(left) { return Self::expr_is_identity(right); }
                if Self::expr_is_identity(right) { return Self::expr_is_identity(left); }
                // to_entries | from_entries → identity
                matches!(left.as_ref(), Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input))
                && matches!(right.as_ref(), Expr::UnaryOp { op: UnaryOp::FromEntries, operand } if matches!(operand.as_ref(), Expr::Input))
            }
            _ => false,
        }
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
                if !matches!(op, BinOp::Eq | BinOp::Ne) { return None; }
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
                        if let Expr::CallBuiltin { name, args } = right.as_ref() {
                            if args.len() == 1 {
                                if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                    if matches!(name.as_str(), "startswith" | "endswith" | "contains") {
                                        return Some((field.clone(), name.clone(), arg.clone()));
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
        if let Some(r) = extract_remap_pairs(expr) { return Some(r); }
        // {a:.x} + {b:.y} — merged object constructs
        if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = expr {
            if let (Some(mut left), Some(right)) = (extract_remap_pairs(lhs), extract_remap_pairs(rhs)) {
                left.extend(right);
                return Some(left);
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
        // Direct ObjectConstruct
        if let Some((result, has_computed)) = extract_computed_pairs(self, expr) {
            if has_computed { return Some(result); }
            return None;
        }
        // {a:.x,b:(.y*2)} + {c:.z} — merged object constructs
        if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = expr {
            if let (Some((mut left, lc)), Some((right, rc))) = (extract_computed_pairs(self, lhs), extract_computed_pairs(self, rhs)) {
                left.extend(right);
                if lc || rc { return Some(left); }
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
                            if let Expr::CallBuiltin { name: sn, args: sa } = split_expr.as_ref() {
                                if sn == "split" && sa.len() == 1 {
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
                        if let Expr::CallBuiltin { name: bn, args: ba } = right.as_ref() {
                            if ba.len() == 1 {
                                if let Expr::Literal(Literal::Str(arg)) = &ba[0] {
                                    let op = match bn.as_str() {
                                        "ltrimstr" => Some(StrBuiltin::Ltrimstr),
                                        "rtrimstr" => Some(StrBuiltin::Rtrimstr),
                                        "startswith" => Some(StrBuiltin::Startswith),
                                        "endswith" => Some(StrBuiltin::Endswith),
                                        "index" => Some(StrBuiltin::Index),
                                        "contains" => Some(StrBuiltin::Contains),
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
                            if let Expr::CallBuiltin { name: sn, args: sa } = split_expr.as_ref() {
                                if sn == "split" && sa.len() == 1 {
                                    if let Expr::Literal(Literal::Str(sep)) = &sa[0] {
                                        if let Expr::CallBuiltin { name: jn, args: ja } = join_expr.as_ref() {
                                            if jn == "join" && ja.len() == 1 {
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
        let is_supported = |op: &UnaryOp| matches!(op,
            UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Sqrt |
            UnaryOp::Fabs | UnaryOp::Abs | UnaryOp::ToString |
            UnaryOp::AsciiDowncase | UnaryOp::AsciiUpcase |
            UnaryOp::Length | UnaryOp::Utf8ByteLength);
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

    /// Detect `.field | startswith/endswith/ltrimstr/rtrimstr("str")` pattern.
    /// Returns (field_name, builtin_name, string_arg) if detected.
    pub fn detect_field_str_builtin(&self) -> Option<(String, String, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::CallBuiltin { name, args } = right.as_ref() {
                        if args.len() == 1 {
                            if matches!(name.as_str(), "startswith" | "endswith" | "ltrimstr" | "rtrimstr" | "split") {
                                if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                    return Some((field.clone(), name.clone(), arg.clone()));
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
            Expr::CallBuiltin { name, args } if args.len() == 1 => {
                if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                    match name.as_str() {
                        "startswith" => return StringChainTerminal::Startswith(arg.clone()),
                        "endswith" => return StringChainTerminal::Endswith(arg.clone()),
                        "contains" => return StringChainTerminal::Contains(arg.clone()),
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
            Expr::CallBuiltin { name, args } if args.len() == 1 => {
                if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                    match name.as_str() {
                        "ltrimstr" => { ops.push(StringChainOp::Ltrimstr(arg.clone())); true }
                        "rtrimstr" => { ops.push(StringChainOp::Rtrimstr(arg.clone())); true }
                        _ => false,
                    }
                } else { false }
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
                                if let Expr::CallBuiltin { name, args } = pr.as_ref() {
                                    if matches!(name.as_str(), "startswith" | "endswith" | "contains") && args.len() == 1 {
                                        if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                            return Some((field.clone(), name.clone(), arg.clone(), remap));
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

    /// Detect `.field | @base64` (or other simple format operations).
    /// Returns (field_name, format_name) if detected.
    pub fn detect_field_format(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, Literal};
        let expr = self.detect_expr()?;
        let is_supported = |name: &str| matches!(name, "base64" | "uri" | "html" | "json" | "text");
        // Pipe form: .field | @format
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        if let Expr::Format { name, expr: fmt_expr } = right.as_ref() {
                            if matches!(fmt_expr.as_ref(), Expr::Input) && is_supported(name) {
                                return Some((field.clone(), name.clone()));
                            }
                        }
                    }
                }
            }
        }
        // Beta-reduced form: @format(.field)
        if let Expr::Format { name, expr: fmt_expr } = expr {
            if is_supported(name) {
                if let Expr::Index { expr: base, key } = fmt_expr.as_ref() {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                            return Some((field.clone(), name.clone()));
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

    /// Detect `.field | ltrimstr("prefix") | tonumber` pattern.
    /// Returns (field_name, prefix) if detected.
    pub fn detect_field_ltrimstr_tonumber(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: mid, right: rr } = right.as_ref() {
                        if let Expr::CallBuiltin { name, args } = mid.as_ref() {
                            if name == "ltrimstr" && args.len() == 1 {
                                if let Expr::Literal(Literal::Str(prefix)) = &args[0] {
                                    if let Expr::UnaryOp { op: UnaryOp::ToNumber, operand } = rr.as_ref() {
                                        if matches!(operand.as_ref(), Expr::Input) {
                                            return Some((field.clone(), prefix.clone()));
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
            if let Expr::CallBuiltin { name, args } = right.as_ref() {
                if name != "join" || args.len() != 1 { return None; }
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
                    let sep = if let Expr::CallBuiltin { name, args } = join_expr.as_ref() {
                        if name != "join" || args.len() != 1 { return None; }
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
                        if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                            if name == "split" && args.len() == 1 {
                                if let Expr::Literal(Literal::Str(sep)) = &args[0] {
                                    if sep.is_empty() {
                                        if let Expr::Pipe { left: rev_expr, right: join_expr } = rest.as_ref() {
                                            if matches!(rev_expr.as_ref(), Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                                                if let Expr::CallBuiltin { name: jn, args: ja } = join_expr.as_ref() {
                                                    if jn == "join" && ja.len() == 1 {
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
                if ops.len() < 1 { return None; }
                ops.reverse();
                if let Expr::Index { expr: base, key } = cur {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                        return Some((field.clone(), ops, *op, *threshold));
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
                    if let Expr::Index { expr: base, key } = operand.as_ref() {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                                parts.push(StringAddPart::FieldToString(f.clone()));
                                return true;
                            }
                        }
                    }
                    false
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
        if matches!(expr, Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
            return true;
        }
        // Pipe form: `to_entries | length`, `keys | length`
        if let Expr::Pipe { left, right } = expr {
            if matches!(right.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if matches!(left.as_ref(), Expr::UnaryOp { op: UnaryOp::ToEntries | UnaryOp::Keys | UnaryOp::KeysUnsorted, .. }) {
                    return true;
                }
            }
        }
        // Beta-reduced form: `length(to_entries(.))`, `length(keys(.))`
        if let Expr::UnaryOp { op: UnaryOp::Length, operand } = expr {
            if let Expr::UnaryOp { op: UnaryOp::ToEntries | UnaryOp::Keys | UnaryOp::KeysUnsorted, operand: inner } = operand.as_ref() {
                if matches!(inner.as_ref(), Expr::Input) {
                    return true;
                }
            }
        }
        false
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
        if let Expr::CallBuiltin { name, args } = expr {
            if name == "del" && args.len() == 1 {
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
        if let Expr::CallBuiltin { name, args } = expr {
            if name == "has" && args.len() == 1 {
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
                    if let Expr::CallBuiltin { name, args } = e {
                        if name == "has" && args.len() == 1 {
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
            if let Expr::CallBuiltin { name, args } = cond.as_ref() {
                if name == "has" && args.len() == 1 {
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
                if let Expr::CallBuiltin { name, args } = e {
                    if name == "has" && args.len() == 1 {
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

    /// Detect `tojson` on input.
    pub fn is_tojson(&self) -> bool {
        use crate::ir::{Expr, UnaryOp};
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        matches!(expr, Expr::UnaryOp { op: UnaryOp::ToJson, operand } if matches!(operand.as_ref(), Expr::Input))
            || matches!(expr, Expr::Format { name, expr: inner } if (name == "json" || name == "text") && matches!(inner.as_ref(), Expr::Input))
    }

    /// Detect `.[]` — each/iteration on input.
    pub fn is_each(&self) -> bool {
        use crate::ir::Expr;
        let expr = match self.detect_expr() { Some(e) => e, None => return false };
        matches!(expr, Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input))
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
            if let Expr::Format { name, .. } = right.as_ref() {
                if matches!(name.as_str(), "csv" | "tsv") {
                    if let Expr::Collect { generator } = left.as_ref() {
                        let mut fields = Vec::new();
                        if collect_comma_fields(generator, &mut fields) && fields.len() >= 2 {
                            return Some((fields, name.clone()));
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
                        if let Expr::CallBuiltin { name: split_name, args: split_args } = split_expr.as_ref() {
                            if split_name != "split" || split_args.len() != 1 { return None; }
                            if let Expr::Literal(Literal::Str(split_str)) = &split_args[0] {
                                if let Expr::CallBuiltin { name: join_name, args: join_args } = join_expr.as_ref() {
                                    if join_name != "join" || join_args.len() != 1 { return None; }
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
                        if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                            if name != "split" || args.len() != 1 { return None; }
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
                        if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                            if name != "split" || args.len() != 1 { return None; }
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
                        if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                            if name != "split" || args.len() != 1 { return None; }
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

    /// Detect `.field | split("str") | length` — returns (field_name, delimiter).
    pub fn detect_field_split_length(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        // Pipe(.field, Pipe(split("s"), length))
        if let Expr::Pipe { left, right } = expr {
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Pipe { left: split_expr, right: len_expr } = right.as_ref() {
                        if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                            if name != "split" || args.len() != 1 { return None; }
                            if let Expr::Literal(Literal::Str(delim)) = &args[0] {
                                if matches!(len_expr.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
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
                        Expr::CallBuiltin { name, args } if args.len() == 1 => {
                            if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                match name.as_str() {
                                    "ltrimstr" | "rtrimstr" => return Some((field, name.clone(), Some(arg.clone()))),
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

    /// Detect `select(.field | length cmp N) | .output_field`.
    /// Returns (cond_field, cmp_op, threshold, output_field).
    pub fn detect_select_field_length_cmp_then_field(&self) -> Option<(String, crate::ir::BinOp, f64, String)> {
        use crate::ir::{Expr, BinOp, Literal, UnaryOp};
        let expr = self.detect_expr()?;
        let extract_length_cmp = |cond: &Expr| -> Option<(String, BinOp, f64)> {
            // BinOp(cmp, UnaryOp(Length, Index(Input, field)), Literal(N))
            if let Expr::BinOp { op, lhs, rhs } = cond {
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
            None
        };
        let extract_output_field = |out: &Expr| -> Option<String> {
            if let Expr::Index { expr: base, key } = out {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                        return Some(f.clone());
                    }
                }
            }
            None
        };
        // Form 1: Pipe(IfThenElse{cond, then:Input, else:Empty}, Index(.output))
        if let Expr::Pipe { left, right } = expr {
            if let Expr::IfThenElse { cond, then_branch, else_branch } = left.as_ref() {
                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                    if let Some((field, op, n)) = extract_length_cmp(cond) {
                        if let Some(out_field) = extract_output_field(right) {
                            return Some((field, op, n, out_field));
                        }
                    }
                }
            }
        }
        // Form 2: IfThenElse{cond, then:Index(.output), else:Empty}
        if let Expr::IfThenElse { cond, then_branch, else_branch } = expr {
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Some((field, op, n)) = extract_length_cmp(cond) {
                    if let Some(out_field) = extract_output_field(then_branch) {
                        return Some((field, op, n, out_field));
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
                        Expr::CallBuiltin { name, args } if args.is_empty() => {
                            if name == "max" { return Some((field1, field2, true)); }
                            if name == "min" { return Some((field1, field2, false)); }
                        }
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
                            if let Some(r) = repr {
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
                    if let Some(r) = repr { buf.extend_from_slice(r.as_bytes()); }
                    else {
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
                _ => None,
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
                            if let Expr::CallBuiltin { name, args } = right.as_ref() {
                                if args.len() == 1 {
                                    if let Expr::Literal(Literal::Str(s)) = &args[0] {
                                        let rhs = match name.as_str() {
                                            "startswith" => Some(CondRhs::Startswith(s.clone())),
                                            "endswith" => Some(CondRhs::Endswith(s.clone())),
                                            "contains" => Some(CondRhs::Contains(s.clone())),
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
                let has_str_func = branches.iter().any(|b| matches!(b.cond_rhs, CondRhs::Startswith(_) | CondRhs::Endswith(_) | CondRhs::Contains(_)));
                if branches.len() < 2 && !has_field_output && !has_remap_output && !has_field_rhs && !has_arith_ops && !has_str_func { return None; }
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
        let is_supported = |op: &UnaryOp| matches!(op,
            UnaryOp::Length | UnaryOp::Utf8ByteLength | UnaryOp::ToString |
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
                                if let Expr::CallBuiltin { name, args } = pr.as_ref() {
                                    if matches!(name.as_str(), "startswith" | "endswith" | "contains") && args.len() == 1 {
                                        if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                            return Some((field.clone(), name.clone(), arg.clone(), rexprs));
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
                                if let Expr::CallBuiltin { name, args } = pipe_right.as_ref() {
                                    if matches!(name.as_str(), "startswith" | "endswith" | "contains") && args.len() == 1 {
                                        if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                            return Some((field.clone(), name.clone(), arg.clone(), output_field));
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

    /// Detect `. + {key: literal, ...}` — merge literal object into input.
    /// Returns list of (key, json_bytes) pairs for each literal entry.
    pub fn detect_obj_merge_literal(&self) -> Option<Vec<(String, Vec<u8>)>> {
        use crate::ir::{Expr, Literal, BinOp};
        let expr = self.detect_expr()?;
        if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = expr {
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
        if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = expr {
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

    /// Execute the filter against an input value, collecting all results.
    pub fn execute(&self, input: &Value) -> Result<Vec<Value>> {
        // Try JIT execution first
        if let Some(jit_fn) = self.jit_fn {
            return crate::jit::execute_jit(jit_fn, input);
        }

        if let Some((ref expr, ref funcs)) = self.parsed {
            // Use our own interpreter
            crate::eval::execute_ir_with_libs(expr, input.clone(), funcs.clone(), self.lib_dirs.clone())
        } else {
            // Fall back to libjq
            execute_via_libjq(&self.program, input)
        }
    }

    /// Execute the filter with a callback for each result (avoids Vec allocation).
    /// Returns Ok(true) if all values were processed, Ok(false) if stopped early.
    pub fn execute_cb(&self, input: &Value, cb: &mut dyn FnMut(&Value) -> Result<bool>) -> Result<bool> {
        if let Some(jit_fn) = self.jit_fn {
            return crate::jit::execute_jit_cb(jit_fn, input, cb);
        }

        if let Some((ref expr, ref funcs)) = self.parsed {
            // Use cached env to avoid re-allocation per call
            let env = {
                let mut cached = self.cached_env.borrow_mut();
                if let Some(ref env) = *cached {
                    env.clone()
                } else {
                    let env = std::rc::Rc::new(std::cell::RefCell::new(
                        crate::eval::Env::with_lib_dirs(funcs.clone(), self.lib_dirs.clone())
                    ));
                    *cached = Some(env.clone());
                    env
                }
            };
            return crate::eval::execute_ir_with_env_cb(
                expr, input.clone(), &env,
                &mut |val| {
                    if let Value::Error(e) = &val {
                        eprintln!("jq: error: {}", e.as_str());
                        return Ok(true);
                    }
                    cb(&val)
                },
            );
        }

        // Fall back to libjq: collect results and iterate
        let results = self.execute(input)?;
        for result in &results {
            if !cb(result)? { return Ok(false); }
        }
        Ok(true)
    }

    /// Returns the set of input fields accessed by the filter, if it can be statically determined.
    /// Returns None if the filter might access any/all fields (e.g., identity, iteration).
    pub fn needed_input_fields(&self) -> Option<Vec<String>> {
        // Use simplified expression (beta-reduced) if available, since the raw parsed
        // expression has Input references that refer to pipe inputs, not the original input.
        // After beta-reduction, all Input references refer to the actual top-level input.
        let expr = if let Some(ref simplified) = self.simplified {
            simplified
        } else if let Some((ref expr, _)) = self.parsed {
            expr
        } else {
            return None;
        };
        let mut fields = Vec::new();
        if collect_input_fields(expr, &mut fields) {
            fields.sort();
            fields.dedup();
            if !fields.is_empty() {
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
    use crate::ir::{Expr, Literal};
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
        // Bare input access — needs full object
        Expr::Input => false,
        // Literals and variables don't access input
        Expr::Literal(_) | Expr::LoadVar { .. } => true,
        // Pipe: both sides
        Expr::Pipe { left, right } => collect_input_fields(left, fields) && collect_input_fields(right, fields),
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
        Expr::Not => true,
        // Let binding
        Expr::LetBinding { value, body, .. } => collect_input_fields(value, fields) && collect_input_fields(body, fields),
        // Select, alternative
        Expr::Alternative { primary, fallback } => collect_input_fields(primary, fields) && collect_input_fields(fallback, fields),
        // Anything else: assume full input needed
        _ => false,
    }
}

/// Execute a jq filter using libjq directly.
fn execute_via_libjq(program: &str, input: &Value) -> Result<Vec<Value>> {
    use crate::jq_ffi;
    let mut jq = crate::bytecode::JqState::new()?;
    let _bc = jq.compile(program)?;

    let input_jv = value_to_jv(input)?;

    unsafe {
        jq_ffi::jq_start(jq.as_ptr(), input_jv, 0);

        let mut results = Vec::new();
        loop {
            let result = jq_ffi::jq_next(jq.as_ptr());
            if jq_ffi::jv_get_kind(result) == jq_ffi::JvKind::Invalid {
                let msg = jq_ffi::jv_invalid_get_msg(jq_ffi::jv_copy(result));
                let kind = jq_ffi::jv_get_kind(msg);
                if kind == jq_ffi::JvKind::Null {
                    jq_ffi::jv_free(msg);
                    jq_ffi::jv_free(result);
                    break;
                } else if kind == jq_ffi::JvKind::String {
                    let cstr = jq_ffi::jv_string_value(msg);
                    let err = std::ffi::CStr::from_ptr(cstr)
                        .to_string_lossy()
                        .into_owned();
                    jq_ffi::jv_free(msg);
                    jq_ffi::jv_free(result);
                    results.push(Value::Error(Rc::new(err)));
                    continue;
                } else {
                    jq_ffi::jv_free(msg);
                    jq_ffi::jv_free(result);
                    break;
                }
            }
            let val = crate::value::jv_to_value(result)?;
            results.push(val);
        }

        Ok(results)
    }
}

/// Convert a Value to a libjq jv.
fn value_to_jv(v: &Value) -> Result<crate::jq_ffi::Jv> {
    use crate::jq_ffi;
    use std::ffi::CString;

    let json = crate::value::value_to_json(v);
    let c_json = CString::new(json)?;
    let jv = unsafe { jq_ffi::jv_parse(c_json.as_ptr()) };
    let kind = unsafe { jq_ffi::jv_get_kind(jv) };
    if kind == jq_ffi::JvKind::Invalid {
        unsafe { jq_ffi::jv_free(jv) };
        bail!("Failed to convert Value to jv");
    }
    Ok(jv)
}
