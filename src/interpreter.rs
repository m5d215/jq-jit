//! Filter execution: parser → IR → tree-walking interpreter.
//!
//! Primary path: our own parser + eval (full control, correct behavior).
//! Fallback: libjq execution (for filters we can't parse yet).


use anyhow::Result;

use crate::ir::CompiledFunc;
use crate::value::Value;

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

/// A compiled jq filter, ready to execute.
pub struct Filter {
    /// Our parsed IR.
    parsed: (crate::ir::Expr, Vec<CompiledFunc>),
    /// Simplified expression for fast path detection (identity pipes stripped).
    simplified: crate::ir::Expr,
    /// JIT-compiled function (if JIT compilation succeeded).
    jit_fn: Option<crate::jit::JitFilterFn>,
    /// JIT compiler kept alive to own the compiled code.
    _jit_compiler: Option<Box<crate::jit::JitCompiler>>,
    lib_dirs: Vec<String>,
    /// Cached eval environment to avoid re-allocating per call.
    cached_env: std::cell::RefCell<Option<crate::eval::EnvRef>>,
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
        Expr::CallBuiltin { name, args } => {
            if args.len() == 2 && matches!(args[0], Expr::Input) {
                if let Expr::Literal(Literal::Str(s)) = &args[1] {
                    match name.as_str() {
                        "startswith" => return Some(StrFuncCond::Startswith(s.clone())),
                        "endswith" => return Some(StrFuncCond::Endswith(s.clone())),
                        "contains" => return Some(StrFuncCond::Contains(s.clone())),
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
            let mut extracted: Vec<(&str, &Expr)> = Vec::with_capacity(pairs.len());
            for (k, v) in pairs {
                if let Expr::Literal(Literal::Str(key)) = k {
                    extracted.push((key.as_str(), v));
                } else {
                    return None;
                }
            }
            let normalized = normalize_object_pairs(extracted);
            let mut buf = Vec::new();
            buf.push(b'{');
            for (i, (key, v)) in normalized.iter().enumerate() {
                if i > 0 { buf.push(b','); }
                buf.push(b'"');
                buf.extend_from_slice(key.as_bytes());
                buf.push(b'"');
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

/// Recursively strip identity pipes and beta-reduce for fast path detection.
/// Pipe(Input, X) → X, Pipe(X, Input) → X.
/// Pipe(E, F) → F[E/.] when E is scalar and F has free Input.
/// Also applies semantic rewrites (to_entries|from_entries → identity).
fn simplify_expr(expr: &crate::ir::Expr) -> crate::ir::Expr {
    use crate::ir::{Expr, Literal, UnaryOp};
    match expr {
        Expr::Pipe { left, right } => {
            let sl = simplify_expr(left);
            let sr = simplify_expr(right);
            if matches!(&sl, Expr::Input) { return sr; }
            if matches!(&sr, Expr::Input) { return sl; }
            // Beta-reduce: X | UnaryOp(op, .) → UnaryOp(op, X)
            // Only for numeric unary ops that don't have specialized detectors
            // in Pipe(.field, UnaryOp) form
            // NOTE: Disabled — too aggressive, breaks other detectors that match
            // Pipe(.field, UnaryOp(op, Input)). Instead, extend specific detectors.
            // Beta-reduce: X | (. binop N) → (X binop N) when N is input-free
            // This flattens pipes like `.x | floor | . > N` into `floor(.x) > N`
            // enabling existing arith_chain_cmp and field_cmp detectors
            if let Expr::BinOp { op, lhs, rhs } = &sr {
                if matches!(lhs.as_ref(), Expr::Input) && rhs.is_input_free() {
                    return simplify_expr(&Expr::BinOp {
                        op: *op,
                        lhs: Box::new(sl),
                        rhs: rhs.clone(),
                    });
                }
            }
            // Beta-reduce: X | (N binop .) → (N binop X) when N is input-free
            if let Expr::BinOp { op, lhs, rhs } = &sr {
                if matches!(rhs.as_ref(), Expr::Input) && lhs.is_input_free() {
                    return simplify_expr(&Expr::BinOp {
                        op: *op,
                        lhs: lhs.clone(),
                        rhs: Box::new(sl),
                    });
                }
            }
            // Beta-reduce: X | (. binop .) → (X binop X) when both sides are input
            if let Expr::BinOp { op, lhs, rhs } = &sr {
                if matches!(lhs.as_ref(), Expr::Input) && matches!(rhs.as_ref(), Expr::Input) {
                    return simplify_expr(&Expr::BinOp {
                        op: *op,
                        lhs: Box::new(sl.clone()),
                        rhs: Box::new(sl),
                    });
                }
            }
            // Beta-reduce: X | if (cond_with_.) then A else B end → if (cond_with_X) then A else B end
            // when A and B are constants (no Input refs), cond is substitutable,
            // and X is a single-output expression (not a generator like range/each/comma)
            if let Expr::IfThenElse { cond, then_branch, else_branch } = &sr {
                let branch_no_input = |b: &Expr| matches!(b, Expr::Literal(_) | Expr::Empty);
                if branch_no_input(then_branch) && branch_no_input(else_branch) && cond.is_input_free() && sl.is_single_output() {
                    return simplify_expr(&Expr::IfThenElse {
                        cond: Box::new(cond.substitute_input(&sl)),
                        then_branch: then_branch.clone(),
                        else_branch: else_branch.clone(),
                    });
                }
            }
            // Semantic: to_entries | from_entries → identity
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::ToEntries, .. })
                && matches!(&sr, Expr::UnaryOp { op: UnaryOp::FromEntries, operand } if matches!(operand.as_ref(), Expr::Input))
            {
                return Expr::Input;
            }
            // NOTE: tojson | fromjson is NOT identity — tojson normalizes nan/inf to null.
            // E.g., {a:nan} | tojson | fromjson → {a:null}. Do not simplify.
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
            // Semantic: {pairs} | length → N (number of keys)
            // Also: {pairs} | to_entries | ... gets simplified via to_entries | length → constant
            // Only safe when all keys are literal strings — otherwise we can't know whether two
            // pairs refer to the same key. After normalization duplicate keys collapse to one,
            // so the pair count matches jq's output.
            if let Expr::ObjectConstruct { pairs } = &sl {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    let mut extracted: Vec<(&str, ())> = Vec::with_capacity(pairs.len());
                    let mut all_literal = true;
                    for (k, _) in pairs {
                        if let Expr::Literal(Literal::Str(s)) = k {
                            extracted.push((s.as_str(), ()));
                        } else {
                            all_literal = false;
                            break;
                        }
                    }
                    if all_literal {
                        let n = normalize_object_pairs(extracted).len();
                        return Expr::Literal(Literal::Num(n as f64, None));
                    }
                }
            }
            // Semantic: [elements] | length → N (number of elements, if known at compile time)
            // Only when each element is known to produce exactly one output.
            if let Expr::Collect { generator } = &sl {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    fn is_single_output(e: &Expr) -> bool {
                        match e {
                            Expr::Input | Expr::Literal(_) | Expr::Negate { .. } => true,
                            Expr::Index { expr: base, .. } => matches!(base.as_ref(), Expr::Input),
                            Expr::BinOp { lhs, rhs, .. } => is_single_output(lhs) && is_single_output(rhs),
                            Expr::UnaryOp { operand, .. } => is_single_output(operand),
                            Expr::ObjectConstruct { .. } => true,
                            Expr::Pipe { left, right } => is_single_output(left) && is_single_output(right),
                            _ => false, // generators, conditionals, etc. may produce 0 or many
                        }
                    }
                    fn count_comma_elements(e: &Expr) -> Option<usize> {
                        match e {
                            Expr::Comma { left, right } => {
                                Some(count_comma_elements(left)? + count_comma_elements(right)?)
                            }
                            _ if is_single_output(e) => Some(1),
                            _ => None,
                        }
                    }
                    if let Some(n) = count_comma_elements(generator) {
                        return Expr::Literal(Literal::Num(n as f64, None));
                    }
                }
            }
            // Semantic: to_entries | length → length (same count)
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    return Expr::UnaryOp { op: UnaryOp::Length, operand: Box::new(Expr::Input) };
                }
            }
            // Semantic: keys | length → length, keys_unsorted | length → length
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::Keys | UnaryOp::KeysUnsorted, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    return Expr::UnaryOp { op: UnaryOp::Length, operand: Box::new(Expr::Input) };
                }
            }
            // Semantic: values | length → length (|values| == |keys| for objects and arrays)
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::Values, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    return Expr::UnaryOp { op: UnaryOp::Length, operand: Box::new(Expr::Input) };
                }
            }
            // Semantic: to_entries | length → length (|entries| == number of keys)
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    return Expr::UnaryOp { op: UnaryOp::Length, operand: Box::new(Expr::Input) };
                }
            }
            // Semantic: reverse | length → length (reverse doesn't change length)
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    return Expr::UnaryOp { op: UnaryOp::Length, operand: Box::new(Expr::Input) };
                }
            }
            // Semantic: sort | length → length (sort doesn't change length)
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::Sort, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    return Expr::UnaryOp { op: UnaryOp::Length, operand: Box::new(Expr::Input) };
                }
            }
            // Semantic: unique | length → unique | length (can't simplify, unique changes length)
            // Semantic: flatten | length — can't simplify, changes length
            // Semantic: keys_unsorted | sort → keys
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::KeysUnsorted, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Sort, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    return Expr::UnaryOp { op: UnaryOp::Keys, operand: Box::new(Expr::Input) };
                }
            }
            // Semantic: sort | reverse → sort | reverse (fused at runtime via SortReverse)
            // sort | reverse | .[0] → max, sort | reverse | .[-1] → min
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::Sort, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    // Keep as sort|reverse — JIT/eval can fuse sort+reverse into sort_unstable_by(|a,b| b.cmp(a))
                    // But optimize sort|reverse|.[0] → max and sort|reverse|.[-1] → min
                }
                // sort | reverse | .[0] → max (largest element)
                if let Expr::Pipe { left: ref pl, right: ref pr } = sr {
                    if matches!(pl.as_ref(), Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                        if let Expr::Index { expr: base, key } = pr.as_ref() {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Num(n, _)) = key.as_ref() {
                                    if *n == 0.0 {
                                        return Expr::UnaryOp { op: UnaryOp::Max, operand: Box::new(Expr::Input) };
                                    } else if *n == -1.0 {
                                        return Expr::UnaryOp { op: UnaryOp::Min, operand: Box::new(Expr::Input) };
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Semantic: sort | .[0] → min, sort | .[-1] → max
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::Sort, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if let Expr::Index { expr: base, key } = &sr {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Num(n, _)) = key.as_ref() {
                            if *n == 0.0 {
                                return Expr::UnaryOp { op: UnaryOp::Min, operand: Box::new(Expr::Input) };
                            } else if *n == -1.0 {
                                return Expr::UnaryOp { op: UnaryOp::Max, operand: Box::new(Expr::Input) };
                            }
                        }
                    }
                }
            }
            // Semantic: reverse | .[0] → .[-1], reverse | .[-1] → .[0]
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if let Expr::Index { expr: base, key } = &sr {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Num(n, _)) = key.as_ref() {
                            if *n == 0.0 {
                                return Expr::Index {
                                    expr: Box::new(Expr::Input),
                                    key: Box::new(Expr::Literal(Literal::Num(-1.0, None))),
                                };
                            } else if *n == -1.0 {
                                return Expr::Index {
                                    expr: Box::new(Expr::Input),
                                    key: Box::new(Expr::Literal(Literal::Num(0.0, None))),
                                };
                            }
                        }
                    }
                }
            }
            // NOTE: explode | implode is NOT identity — explode errors on non-strings
            // Semantic: explode | map(. + N) | implode → __shift_codepoints__(N)
            // Also: explode | map(. - N) | implode
            // Helper: check if expr is map(. + N) pattern, return shift amount
            fn is_map_shift(expr: &Expr) -> Option<f64> {
                if let Expr::Collect { generator } = expr {
                    if let Expr::Pipe { left: gl, right: gr } = generator.as_ref() {
                        if matches!(gl.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            if let Expr::BinOp { op, lhs, rhs } = gr.as_ref() {
                                if matches!(lhs.as_ref(), Expr::Input) {
                                    if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                        return match op {
                                            crate::ir::BinOp::Add => Some(*n),
                                            crate::ir::BinOp::Sub => Some(-*n),
                                            _ => None,
                                        };
                                    }
                                }
                                if matches!(op, crate::ir::BinOp::Add) && matches!(rhs.as_ref(), Expr::Input) {
                                    if let Expr::Literal(Literal::Num(n, _)) = lhs.as_ref() {
                                        return Some(*n);
                                    }
                                }
                            }
                        }
                    }
                }
                None
            }
            // Case 1: Pipe(Pipe(explode, map(.+N)), implode) — left-associative
            if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Implode, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if let Expr::Pipe { left: el, right: er } = &sl {
                    if matches!(el.as_ref(), Expr::UnaryOp { op: UnaryOp::Explode, operand } if matches!(operand.as_ref(), Expr::Input)) {
                        if let Some(shift) = is_map_shift(er) {
                            return Expr::CallBuiltin {
                                name: "__shift_codepoints__".to_string(),
                                args: vec![Expr::Literal(Literal::Num(shift, None))],
                            };
                        }
                    }
                }
            }
            // Case 2: Pipe(explode, Pipe(map(.+N), implode)) — right-associative
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::Explode, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if let Expr::Pipe { left: mr, right: ir } = &sr {
                    if matches!(ir.as_ref(), Expr::UnaryOp { op: UnaryOp::Implode, operand } if matches!(operand.as_ref(), Expr::Input)) {
                        if let Some(shift) = is_map_shift(mr) {
                            return Expr::CallBuiltin {
                                name: "__shift_codepoints__".to_string(),
                                args: vec![Expr::Literal(Literal::Num(shift, None))],
                            };
                        }
                    }
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
            // Semantic: [a, b, c] | reverse → [c, b, a]
            if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if let Expr::Collect { generator } = &sl {
                    fn collect_comma_elements(expr: &Expr, out: &mut Vec<Expr>) {
                        match expr {
                            Expr::Comma { left, right } => {
                                collect_comma_elements(left, out);
                                collect_comma_elements(right, out);
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    let mut elements = Vec::new();
                    collect_comma_elements(generator, &mut elements);
                    if elements.len() >= 2 {
                        elements.reverse();
                        let mut gen = elements.pop().unwrap();
                        while let Some(e) = elements.pop() {
                            gen = Expr::Comma { left: Box::new(e), right: Box::new(gen) };
                        }
                        return Expr::Collect { generator: Box::new(gen) };
                    }
                }
            }
            // Semantic: reverse | .[0] → .[-1], reverse | .[-1] → .[0]
            // reverse then index = access from the other end
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if let Expr::Index { expr: base, key } = &sr {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(crate::ir::Literal::Num(n, _)) = key.as_ref() {
                            let idx = *n as i64;
                            let new_idx = if idx >= 0 { -(idx + 1) } else { -idx - 1 };
                            return Expr::Index {
                                expr: Box::new(Expr::Input),
                                key: Box::new(Expr::Literal(crate::ir::Literal::Num(new_idx as f64, None))),
                            };
                        }
                    }
                }
            }
            // Semantic: [.f1, .f2, ...] | add op X → (.f1 + .f2 + ... + .fN) op X
            // Handles: add * N, add - N, add / N, add / length, add + N
            if let Expr::Collect { generator: ref lg } = sl {
                // Check if rhs is BinOp(op, add, something) where add = UnaryOp(Add, Input)
                let add_binop = if let Expr::BinOp { op: ref bop, lhs: ref blhs, rhs: ref brhs } = sr {
                    let is_add = matches!(blhs.as_ref(), Expr::UnaryOp { op: UnaryOp::Add, operand } if matches!(operand.as_ref(), Expr::Input));
                    if is_add && matches!(bop, crate::ir::BinOp::Add | crate::ir::BinOp::Sub | crate::ir::BinOp::Mul | crate::ir::BinOp::Div | crate::ir::BinOp::Mod) {
                        // Determine the right operand: either a literal or length
                        let rhs_expr = if let Expr::Literal(crate::ir::Literal::Num(n, _)) = brhs.as_ref() {
                            Some((*bop, Expr::Literal(crate::ir::Literal::Num(*n, None))))
                        } else if matches!(brhs.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                            // add / length → sum / N (length is count of elements)
                            None // handled specially below
                        } else {
                            None
                        };
                        if let Some((op, rhs_val)) = rhs_expr {
                            Some((op, rhs_val, false))
                        } else if matches!(bop, crate::ir::BinOp::Div) && matches!(brhs.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                            Some((*bop, Expr::Literal(crate::ir::Literal::Num(0.0, None)), true)) // placeholder, will use N
                        } else {
                            None
                        }
                    } else { None }
                } else { None };
                if let Some((outer_op, rhs_val, use_length)) = add_binop {
                    fn collect_comma_elems2(e: &Expr, out: &mut Vec<Expr>) {
                        match e {
                            Expr::Comma { left, right } => {
                                collect_comma_elems2(left, out);
                                collect_comma_elems2(right, out);
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    let mut elems = Vec::new();
                    collect_comma_elems2(lg, &mut elems);
                    let n = elems.len();
                    if n >= 2 {
                        let mut sum = elems.remove(0);
                        for elem in elems {
                            sum = Expr::BinOp {
                                op: crate::ir::BinOp::Add,
                                lhs: Box::new(sum),
                                rhs: Box::new(elem),
                            };
                        }
                        let actual_rhs = if use_length {
                            Expr::Literal(crate::ir::Literal::Num(n as f64, None))
                        } else {
                            rhs_val
                        };
                        return Expr::BinOp {
                            op: outer_op,
                            lhs: Box::new(sum),
                            rhs: Box::new(actual_rhs),
                        };
                    }
                }
            }
            // Semantic: [.f1, .f2, ...] | add → .f1 + .f2 + ... + .fN
            // This catches cases where the parser's pipe-level optimization missed it
            // (e.g., from simplify_expr creating new Collect | add patterns)
            if let Expr::Collect { generator: ref lg } = sl {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Add, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    fn collect_elems_for_add(e: &Expr, out: &mut Vec<Expr>) {
                        match e {
                            Expr::Comma { left, right } => {
                                collect_elems_for_add(left, out);
                                collect_elems_for_add(right, out);
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    // Rewrite is only valid when every element yields exactly
                    // one value. `Empty` yields zero; `.[]`, `recurse`, and
                    // other generators yield many. `[.[]] | add` was collapsing
                    // to `.[]` (issue #56) because a single-element list with
                    // a generator inside was treated as "identity".
                    fn is_single_valued(e: &Expr) -> bool {
                        match e {
                            Expr::Empty => false,
                            Expr::Each { .. } | Expr::EachOpt { .. }
                            | Expr::Comma { .. } | Expr::Recurse { .. }
                            | Expr::Range { .. } | Expr::Limit { .. }
                            | Expr::RegexMatch { .. } | Expr::RegexScan { .. }
                            | Expr::RegexCapture { .. } => false,
                            Expr::Pipe { left, right } => is_single_valued(left) && is_single_valued(right),
                            Expr::IfThenElse { cond, then_branch, else_branch } => {
                                is_single_valued(cond) && is_single_valued(then_branch) && is_single_valued(else_branch)
                            }
                            Expr::TryCatch { try_expr, catch_expr } => {
                                is_single_valued(try_expr) && is_single_valued(catch_expr)
                            }
                            Expr::Alternative { primary, fallback } => {
                                is_single_valued(primary) && is_single_valued(fallback)
                            }
                            Expr::LetBinding { value, body, .. } => {
                                is_single_valued(value) && is_single_valued(body)
                            }
                            Expr::Collect { .. } => true,
                            Expr::Input | Expr::Literal(_) | Expr::LoadVar { .. }
                            | Expr::Not | Expr::Negate { .. }
                            | Expr::Index { .. } | Expr::IndexOpt { .. }
                            | Expr::Slice { .. } | Expr::UnaryOp { .. } | Expr::BinOp { .. }
                            | Expr::StringInterpolation { .. } | Expr::ObjectConstruct { .. }
                            | Expr::RegexTest { .. } | Expr::RegexSub { .. } | Expr::RegexGsub { .. } => true,
                            _ => false,
                        }
                    }
                    let mut elems = Vec::new();
                    collect_elems_for_add(lg, &mut elems);
                    let all_single = !elems.is_empty()
                        && elems.iter().all(|e| is_single_valued(e));
                    if all_single {
                        if elems.len() == 1 {
                            // [expr] | add → expr (single element, add is identity)
                            return elems.remove(0);
                        } else if elems.len() >= 2 {
                            let mut result = elems.remove(0);
                            for elem in elems {
                                result = Expr::BinOp {
                                    op: crate::ir::BinOp::Add,
                                    lhs: Box::new(result),
                                    rhs: Box::new(elem),
                                };
                            }
                            return result;
                        }
                    }
                }
            }
            // Semantic: [A, [B, C], D] | flatten → [A, B, C, D]
            // Unwrap inner Collect elements one level
            if let Expr::Collect { generator: ref lg } = sl {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Flatten, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    fn collect_for_flatten(e: &Expr, out: &mut Vec<Expr>) {
                        match e {
                            Expr::Comma { left, right } => {
                                collect_for_flatten(left, out);
                                collect_for_flatten(right, out);
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    let mut top_elems = Vec::new();
                    collect_for_flatten(lg, &mut top_elems);
                    // Check if all elements are Collect (inner arrays) or simple exprs
                    let has_inner_collect = top_elems.iter().any(|e| matches!(e, Expr::Collect { .. }));
                    if has_inner_collect {
                        let mut flat_elems = Vec::new();
                        for elem in &top_elems {
                            match elem {
                                Expr::Collect { generator } => {
                                    collect_for_flatten(generator, &mut flat_elems);
                                }
                                other => flat_elems.push(other.clone()),
                            }
                        }
                        if flat_elems.len() >= 2 {
                            let mut gen = flat_elems.remove(0);
                            for e in flat_elems {
                                gen = Expr::Comma { left: Box::new(gen), right: Box::new(e) };
                            }
                            return Expr::Collect { generator: Box::new(gen) };
                        }
                    }
                }
            }
            // Semantic: [e0, e1, ...] | .[N] → eN (extract Nth element at compile time)
            // Also handles .[−1] → last element
            if let Expr::Collect { generator: ref lg } = sl {
                if let Expr::Index { expr: base, key } = &sr {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Num(n, _)) = key.as_ref() {
                            if n.is_nan() || !n.is_finite() { /* skip NaN/Inf indices */ }
                            else {
                            let idx = *n as i64;
                            fn collect_comma_for_idx(e: &Expr, out: &mut Vec<Expr>) {
                                match e {
                                    Expr::Comma { left, right } => {
                                        collect_comma_for_idx(left, out);
                                        collect_comma_for_idx(right, out);
                                    }
                                    other => out.push(other.clone()),
                                }
                            }
                            let mut elems = Vec::new();
                            collect_comma_for_idx(lg, &mut elems);
                            // Only constant-fold when the negative index
                            // actually lands inside the array. Previously the
                            // negative branch clamped to 0 via `.max(0)`,
                            // returning the first element for any out-of-range
                            // negative index (issue #42). Falling through for
                            // the out-of-range cases lets the runtime emit the
                            // correct null result.
                            let effective: Option<usize> = if idx >= 0 {
                                if (idx as usize) < elems.len() { Some(idx as usize) } else { None }
                            } else {
                                let v = elems.len() as i64 + idx;
                                if v >= 0 && (v as usize) < elems.len() { Some(v as usize) } else { None }
                            };
                            if let Some(i) = effective {
                                return elems.swap_remove(i);
                            }
                            }
                        }
                    }
                }
            }
            // Semantic: [a, b] | min → if a <= b then a else b end
            // Semantic: [a, b] | max → if a > b then a else b end
            if let Expr::Collect { generator: ref lg } = sl {
                let is_min = matches!(&sr, Expr::UnaryOp { op: UnaryOp::Min, operand } if matches!(operand.as_ref(), Expr::Input));
                let is_max = matches!(&sr, Expr::UnaryOp { op: UnaryOp::Max, operand } if matches!(operand.as_ref(), Expr::Input));
                if is_min || is_max {
                    // Collect all elements from comma-chain
                    fn collect_elems(e: &Expr, out: &mut Vec<Expr>) {
                        match e {
                            Expr::Comma { left, right } => {
                                collect_elems(left, out);
                                collect_elems(right, out);
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    let mut elems = Vec::new();
                    collect_elems(lg, &mut elems);
                    if elems.len() >= 2 {
                        // Fold: min(a,b,c) = min(min(a,b), c)
                        let cmp_op = if is_min { crate::ir::BinOp::Le } else { crate::ir::BinOp::Gt };
                        let mut result = elems.remove(0);
                        for elem in elems {
                            result = Expr::IfThenElse {
                                cond: Box::new(Expr::BinOp { op: cmp_op, lhs: Box::new(result.clone()), rhs: Box::new(elem.clone()) }),
                                then_branch: Box::new(result),
                                else_branch: Box::new(elem),
                            };
                        }
                        return simplify_expr(&result);
                    }
                }
            }
            // Semantic: [a, b] | sort → if a <= b then [a,b] else [b,a] end
            if let Expr::Collect { generator: ref lg } = sl {
                let is_sort = matches!(&sr, Expr::UnaryOp { op: UnaryOp::Sort, operand } if matches!(operand.as_ref(), Expr::Input));
                if is_sort {
                    if let Expr::Comma { left, right } = lg.as_ref() {
                        if !matches!(left.as_ref(), Expr::Comma { .. }) && !matches!(right.as_ref(), Expr::Comma { .. }) {
                            let a = left.as_ref().clone();
                            let b = right.as_ref().clone();
                            return simplify_expr(&Expr::IfThenElse {
                                cond: Box::new(Expr::BinOp { op: crate::ir::BinOp::Le, lhs: Box::new(a.clone()), rhs: Box::new(b.clone()) }),
                                then_branch: Box::new(Expr::Collect { generator: Box::new(Expr::Comma { left: Box::new(a.clone()), right: Box::new(b.clone()) }) }),
                                else_branch: Box::new(Expr::Collect { generator: Box::new(Expr::Comma { left: Box::new(b), right: Box::new(a) }) }),
                            });
                        }
                    }
                }
            }
            // Semantic: [e1, e2, ...] | add → e1 + e2 + ... (avoids array construction)
            // Also: [e1, e2, ...] | add / length → (e1 + e2 + ...) / N
            if let Expr::Collect { generator: ref lg } = sl {
                // Check for add or add / length
                let is_add = matches!(&sr, Expr::UnaryOp { op: UnaryOp::Add, operand } if matches!(operand.as_ref(), Expr::Input));
                let is_add_div_length = if !is_add {
                    if let Expr::BinOp { op: crate::ir::BinOp::Div, lhs, rhs } = &sr {
                        matches!(lhs.as_ref(), Expr::UnaryOp { op: UnaryOp::Add, operand } if matches!(operand.as_ref(), Expr::Input))
                        && matches!(rhs.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input))
                    } else { false }
                } else { false };
                if is_add || is_add_div_length {
                    fn collect_comma_elems(e: &Expr, out: &mut Vec<Expr>) {
                        match e {
                            Expr::Comma { left, right } => {
                                collect_comma_elems(left, out);
                                collect_comma_elems(right, out);
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    let mut elems = Vec::new();
                    collect_comma_elems(lg, &mut elems);
                    if elems.len() >= 2 && elems.len() <= 16 {
                        let n = elems.len();
                        let mut result = elems.remove(0);
                        for elem in elems {
                            result = Expr::BinOp {
                                op: crate::ir::BinOp::Add,
                                lhs: Box::new(result),
                                rhs: Box::new(elem),
                            };
                        }
                        if is_add_div_length {
                            result = Expr::BinOp {
                                op: crate::ir::BinOp::Div,
                                lhs: Box::new(result),
                                rhs: Box::new(Expr::Literal(Literal::Num(n as f64, None))),
                            };
                        }
                        return simplify_expr(&result);
                    }
                }
            }
            // Semantic: [e1, e2, ...] | any(f) → (e1|f) or (e2|f) or ...
            // Semantic: [e1, e2, ...] | all(f) → (e1|f) and (e2|f) and ...
            if let Expr::Collect { generator: ref lg } = sl {
                let (is_any_all, predicate) = match &sr {
                    Expr::AnyShort { generator: gen, predicate } => {
                        if matches!(gen.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            (Some(true), Some(predicate.as_ref()))
                        } else { (None, None) }
                    }
                    Expr::AllShort { generator: gen, predicate } => {
                        if matches!(gen.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            (Some(false), Some(predicate.as_ref()))
                        } else { (None, None) }
                    }
                    // Also: any without explicit predicate = any(.)
                    Expr::UnaryOp { op: UnaryOp::Any, operand } if matches!(operand.as_ref(), Expr::Input) => {
                        (Some(true), Some(&Expr::Input as &Expr))
                    }
                    Expr::UnaryOp { op: UnaryOp::All, operand } if matches!(operand.as_ref(), Expr::Input) => {
                        (Some(false), Some(&Expr::Input as &Expr))
                    }
                    _ => (None, None),
                };
                if let (Some(is_any), Some(pred)) = (is_any_all, predicate) {
                    fn collect_comma_for_any(e: &Expr, out: &mut Vec<Expr>) {
                        match e {
                            Expr::Comma { left, right } => {
                                collect_comma_for_any(left, out);
                                collect_comma_for_any(right, out);
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    let mut elems = Vec::new();
                    collect_comma_for_any(lg, &mut elems);
                    if elems.len() >= 2 && elems.len() <= 8 {
                        let combiner = if is_any { crate::ir::BinOp::Or } else { crate::ir::BinOp::And };
                        let mut result = simplify_expr(&Expr::Pipe {
                            left: Box::new(elems.remove(0)),
                            right: Box::new(pred.clone()),
                        });
                        for elem in elems {
                            let applied = simplify_expr(&Expr::Pipe {
                                left: Box::new(elem),
                                right: Box::new(pred.clone()),
                            });
                            result = Expr::BinOp {
                                op: combiner,
                                lhs: Box::new(result),
                                rhs: Box::new(applied),
                            };
                        }
                        return result;
                    }
                }
            }
            // Beta-reduction: .x | . + 1 → .x + 1
            if sl.is_simple_scalar() && sr.is_input_free() {
                return sr.substitute_input(&sl);
            }
            // [gen] | map(f) = [gen] | [.[] | f] → [gen | f]
            // Distributes f over each element of gen via beta-reduction.
            // Each element gets piped through f and simplified (beta-reduced).
            // [gen] | map(f) distribution helper
            fn is_comma_of_simple_scalars(e: &Expr) -> bool {
                match e {
                    Expr::Comma { left, right } => {
                        is_comma_of_simple_scalars(left) && is_comma_of_simple_scalars(right)
                    }
                    other => other.is_simple_scalar(),
                }
            }
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
            fn try_extract_map_body(expr: &Expr) -> Option<Expr> {
                // [.[] | f] → Some(f)
                if let Expr::Collect { generator } = expr {
                    if let Expr::Pipe { left, right } = generator.as_ref() {
                        if matches!(left.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            return Some(right.as_ref().clone());
                        }
                    }
                    // Also: [f(.[])] where .[] appears inside f — e.g., [.[] * 2]
                    // Replace Each(Input) with Input to get the body
                    fn replace_each_with_input(e: &Expr) -> Option<Expr> {
                        match e {
                            Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input) => {
                                Some(Expr::Input)
                            }
                            Expr::BinOp { op, lhs, rhs } => {
                                let new_lhs = replace_each_with_input(lhs);
                                let new_rhs = replace_each_with_input(rhs);
                                if new_lhs.is_some() || new_rhs.is_some() {
                                    Some(Expr::BinOp {
                                        op: *op,
                                        lhs: Box::new(new_lhs.unwrap_or_else(|| lhs.as_ref().clone())),
                                        rhs: Box::new(new_rhs.unwrap_or_else(|| rhs.as_ref().clone())),
                                    })
                                } else {
                                    None
                                }
                            }
                            Expr::UnaryOp { op, operand } => {
                                replace_each_with_input(operand).map(|o| Expr::UnaryOp {
                                    op: *op, operand: Box::new(o),
                                })
                            }
                            Expr::Negate { operand } => {
                                replace_each_with_input(operand).map(|o| Expr::Negate { operand: Box::new(o) })
                            }
                            _ => None,
                        }
                    }
                    if let Some(body) = replace_each_with_input(generator) {
                        return Some(body);
                    }
                }
                None
            }
            if let Expr::Collect { generator: ref lg } = sl {
                if is_comma_of_simple_scalars(lg) {
                    // [gen] | map(f) → [gen distributed with f]
                    if let Some(f) = try_extract_map_body(&sr) {
                        return Expr::Collect { generator: Box::new(distribute_map(lg, &f)) };
                    }
                    // [gen] | Pipe(map(f), rest) → [gen distributed with f] | rest
                    if let Expr::Pipe { left: ref pl, right: ref pr } = sr {
                        if let Some(f) = try_extract_map_body(pl) {
                            let distributed = Expr::Collect { generator: Box::new(distribute_map(lg, &f)) };
                            let result = Expr::Pipe { left: Box::new(distributed), right: pr.clone() };
                            return simplify_expr(&result);
                        }
                    }
                }
            }
            // split("s") | length > 1 → contains("s")  (more efficient, enables raw byte path)
            if let Expr::CallBuiltin { name, args } = &sl {
                if name == "split" && args.len() == 1 {
                    if let Expr::Literal(crate::ir::Literal::Str(delim)) = &args[0] {
                        if let Expr::BinOp { op, lhs: cmp_lhs, rhs: cmp_rhs } = &sr {
                            if let Expr::UnaryOp { op: crate::ir::UnaryOp::Length, operand } = cmp_lhs.as_ref() {
                                if matches!(operand.as_ref(), Expr::Input) {
                                    if let Expr::Literal(crate::ir::Literal::Num(n, _)) = cmp_rhs.as_ref() {
                                        // split(S) | length > 1 means "contains S"
                                        // split(S) | length > 0 is always true for any finite string
                                        if !delim.is_empty() && *n == 1.0 && matches!(op, crate::ir::BinOp::Gt) {
                                            return Expr::CallBuiltin {
                                                name: "contains".to_string(),
                                                args: vec![Expr::Literal(crate::ir::Literal::Str(delim.clone()))],
                                            };
                                        }
                                        if !delim.is_empty() && *n == 0.0 && matches!(op, crate::ir::BinOp::Gt) {
                                            return Expr::Literal(crate::ir::Literal::True);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // select(A) | select(B) → select(A and B)
            if let Expr::IfThenElse { cond: cond_a, then_branch: then_a, else_branch: else_a } = &sl {
                if matches!(then_a.as_ref(), Expr::Input) && matches!(else_a.as_ref(), Expr::Empty) {
                    if let Expr::IfThenElse { cond: cond_b, then_branch: then_b, else_branch: else_b } = &sr {
                        if matches!(then_b.as_ref(), Expr::Input) && matches!(else_b.as_ref(), Expr::Empty) {
                            return simplify_expr(&Expr::IfThenElse {
                                cond: Box::new(Expr::BinOp { op: crate::ir::BinOp::And, lhs: cond_a.clone(), rhs: cond_b.clone() }),
                                then_branch: Box::new(Expr::Input),
                                else_branch: Box::new(Expr::Empty),
                            });
                        }
                    }
                    // select(A) | Pipe(select(B), rest) → select(A and B) | rest
                    if let Expr::Pipe { left: pl, right: pr } = &sr {
                        if let Expr::IfThenElse { cond: cond_b, then_branch: then_b, else_branch: else_b } = pl.as_ref() {
                            if matches!(then_b.as_ref(), Expr::Input) && matches!(else_b.as_ref(), Expr::Empty) {
                                let merged_select = Expr::IfThenElse {
                                    cond: Box::new(Expr::BinOp { op: crate::ir::BinOp::And, lhs: cond_a.clone(), rhs: cond_b.clone() }),
                                    then_branch: Box::new(Expr::Input),
                                    else_branch: Box::new(Expr::Empty),
                                };
                                return simplify_expr(&Expr::Pipe {
                                    left: Box::new(merged_select),
                                    right: pr.clone(),
                                });
                            }
                        }
                    }
                }
            }
            // map(f) | map(g) → map(f | g) — eliminate intermediate array allocation
            if let Some(f) = try_extract_map_body(&sl) {
                if let Some(g) = try_extract_map_body(&sr) {
                    let fused = Expr::Pipe { left: Box::new(f), right: Box::new(g) };
                    return simplify_expr(&Expr::Collect {
                        generator: Box::new(Expr::Pipe {
                            left: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                            right: Box::new(fused),
                        }),
                    });
                }
                // map(f) | Pipe(map(g), rest) → map(f | g) | rest
                if let Expr::Pipe { left: ref pl, right: ref pr } = sr {
                    if let Some(g) = try_extract_map_body(pl) {
                        let fused = Expr::Pipe { left: Box::new(f.clone()), right: Box::new(g) };
                        let fused_map = Expr::Collect {
                            generator: Box::new(Expr::Pipe {
                                left: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                                right: Box::new(fused),
                            }),
                        };
                        return simplify_expr(&Expr::Pipe {
                            left: Box::new(fused_map),
                            right: pr.clone(),
                        });
                    }
                }
            }
            // group_by(.f) | map(.[0]) → unique_by(.f)
            if let Expr::ClosureOp { op: crate::ir::ClosureOpKind::GroupBy, input_expr, key_expr } = &sl {
                if let Some(body) = try_extract_map_body(&sr) {
                    // Check if body is .[0]
                    if let Expr::Index { expr: base, key } = &body {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(crate::ir::Literal::Num(n, _)) = key.as_ref() {
                                if *n == 0.0 {
                                    return Expr::ClosureOp {
                                        op: crate::ir::ClosureOpKind::UniqueBy,
                                        input_expr: input_expr.clone(),
                                        key_expr: key_expr.clone(),
                                    };
                                }
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
            // Constant condition folding: if true then A else B end → A
            match &sc {
                Expr::Literal(crate::ir::Literal::Null) | Expr::Literal(crate::ir::Literal::False) => return se,
                Expr::Literal(crate::ir::Literal::True) | Expr::Literal(crate::ir::Literal::Num(_, _)) | Expr::Literal(crate::ir::Literal::Str(_)) => return st,
                _ => {}
            }
            // if A then (if B then X else empty end) else empty end → if (A and B) then X else empty end
            if matches!(se, Expr::Empty) {
                if let Expr::IfThenElse { cond: cond_inner, then_branch: then_inner, else_branch: else_inner } = &st {
                    if matches!(else_inner.as_ref(), Expr::Empty) {
                        return Expr::IfThenElse {
                            cond: Box::new(Expr::BinOp { op: crate::ir::BinOp::And, lhs: Box::new(sc), rhs: cond_inner.clone() }),
                            then_branch: then_inner.clone(),
                            else_branch: Box::new(Expr::Empty),
                        };
                    }
                }
            }
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
            // For `+`: right-side keys win on collision = same as last-key-wins in single construct.
            // So {A} + {B} → {A, B} is always valid for `+`.
            // For `*`: only safe when all keys are distinct literal strings (no nested merge).
            if matches!(op, crate::ir::BinOp::Add) {
                if let (Expr::ObjectConstruct { pairs: p1 }, Expr::ObjectConstruct { pairs: p2 }) = (&sl, &sr) {
                    let mut merged = p1.clone();
                    merged.extend(p2.iter().cloned());
                    return Expr::ObjectConstruct { pairs: merged };
                }
            }
            if matches!(op, crate::ir::BinOp::Mul) {
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
            // Constant fold: Num op Num → Num
            if let (Expr::Literal(Literal::Num(a, _)), Expr::Literal(Literal::Num(b, _))) = (&sl, &sr) {
                let result = match op {
                    crate::ir::BinOp::Add => Some(a + b),
                    crate::ir::BinOp::Sub => Some(a - b),
                    crate::ir::BinOp::Mul => Some(a * b),
                    crate::ir::BinOp::Div if *b != 0.0 => Some(a / b),
                    crate::ir::BinOp::Mod if *b != 0.0 && b.is_finite() => Some(a % b),
                    _ => None,
                };
                if let Some(r) = result {
                    return Expr::Literal(Literal::Num(r, None));
                }
                // Comparison ops
                let cmp_result = match op {
                    crate::ir::BinOp::Eq => Some(*a == *b),
                    crate::ir::BinOp::Ne => Some(*a != *b),
                    crate::ir::BinOp::Lt => Some(*a < *b),
                    crate::ir::BinOp::Gt => Some(*a > *b),
                    crate::ir::BinOp::Le => Some(*a <= *b),
                    crate::ir::BinOp::Ge => Some(*a >= *b),
                    _ => None,
                };
                if let Some(r) = cmp_result {
                    return if r { Expr::Literal(Literal::True) } else { Expr::Literal(Literal::False) };
                }
            }
            // Constant fold: Str + Str → Str
            if matches!(op, crate::ir::BinOp::Add) {
                if let (Expr::Literal(Literal::Str(a)), Expr::Literal(Literal::Str(b))) = (&sl, &sr) {
                    return Expr::Literal(Literal::Str(format!("{}{}", a, b)));
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
            let so = simplify_expr(operand);
            // Normalize f|op → f | op(.) when f is not input
            if !matches!(&so, Expr::Input) {
                return simplify_expr(&Expr::Pipe {
                    left: Box::new(so),
                    right: Box::new(Expr::UnaryOp { op: *op, operand: Box::new(Expr::Input) }),
                });
            }
            Expr::UnaryOp { op: *op, operand: Box::new(so) }
        }
        Expr::Collect { generator } => {
            Expr::Collect { generator: Box::new(simplify_expr(generator)) }
        }
        Expr::Comma { left, right } => {
            Expr::Comma { left: Box::new(simplify_expr(left)), right: Box::new(simplify_expr(right)) }
        }
        Expr::Index { expr, key } => {
            let se = simplify_expr(expr);
            let sk = simplify_expr(key);
            // Normalize f[k] → f | .[k] ONLY when k is a literal (doesn't reference input)
            // f[.baz] is NOT f | .[.baz] because .baz binds to different inputs
            if !matches!(&se, Expr::Input) && matches!(&sk, Expr::Literal(_)) {
                return simplify_expr(&Expr::Pipe {
                    left: Box::new(se),
                    right: Box::new(Expr::Index { expr: Box::new(Expr::Input), key: Box::new(sk) }),
                });
            }
            Expr::Index { expr: Box::new(se), key: Box::new(sk) }
        }
        Expr::IndexOpt { expr, key } => {
            let se = simplify_expr(expr);
            let sk = simplify_expr(key);
            if !matches!(&se, Expr::Input) && matches!(&sk, Expr::Literal(_)) {
                return simplify_expr(&Expr::Pipe {
                    left: Box::new(se),
                    right: Box::new(Expr::IndexOpt { expr: Box::new(Expr::Input), key: Box::new(sk) }),
                });
            }
            Expr::IndexOpt { expr: Box::new(se), key: Box::new(sk) }
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
            let se = simplify_expr(input_expr);
            // Normalize f[] → f | .[] when f is not input
            if !matches!(&se, Expr::Input) {
                return simplify_expr(&Expr::Pipe {
                    left: Box::new(se),
                    right: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                });
            }
            Expr::Each { input_expr: Box::new(se) }
        }
        Expr::EachOpt { input_expr } => {
            let se = simplify_expr(input_expr);
            if !matches!(&se, Expr::Input) {
                return simplify_expr(&Expr::Pipe {
                    left: Box::new(se),
                    right: Box::new(Expr::EachOpt { input_expr: Box::new(Expr::Input) }),
                });
            }
            Expr::EachOpt { input_expr: Box::new(se) }
        }
        Expr::Negate { operand } => {
            let s = simplify_expr(operand);
            if let Expr::Literal(Literal::Num(n, _)) = &s {
                Expr::Literal(Literal::Num(-n, None))
            } else {
                Expr::Negate { operand: Box::new(s) }
            }
        }
        Expr::Slice { expr, from, to } => {
            let se = simplify_expr(expr);
            let sf = from.as_ref().map(|e| Box::new(simplify_expr(e)));
            let st = to.as_ref().map(|e| Box::new(simplify_expr(e)));
            // Normalize f[from:to] → f | .[from:to] when f is not input
            // Only safe when from/to are literals (don't reference input)
            if !matches!(&se, Expr::Input)
                && sf.as_ref().map_or(true, |e| matches!(e.as_ref(), Expr::Literal(_)))
                && st.as_ref().map_or(true, |e| matches!(e.as_ref(), Expr::Literal(_)))
            {
                return simplify_expr(&Expr::Pipe {
                    left: Box::new(se),
                    right: Box::new(Expr::Slice { expr: Box::new(Expr::Input), from: sf, to: st }),
                });
            }
            Expr::Slice {
                expr: Box::new(se),
                from: sf,
                to: st,
            }
        }
        Expr::Update { path_expr, update_expr } => {
            Expr::Update { path_expr: Box::new(simplify_expr(path_expr)), update_expr: Box::new(simplify_expr(update_expr)) }
        }
        Expr::Assign { path_expr, value_expr } => {
            let sp = simplify_expr(path_expr);
            let sv = simplify_expr(value_expr);
            // .field = f(.field) → .field |= f(.) when value only references .field
            if let Expr::Index { expr: base, key } = &sp {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(crate::ir::Literal::Str(field)) = key.as_ref() {
                        fn replace_field_with_input(e: &Expr, field: &str) -> Option<Expr> {
                            match e {
                                Expr::Index { expr: base, key } if matches!(base.as_ref(), Expr::Input)
                                    && matches!(key.as_ref(), Expr::Literal(crate::ir::Literal::Str(f)) if f == field) => {
                                    Some(Expr::Input)
                                }
                                Expr::BinOp { op, lhs, rhs } => {
                                    let nl = replace_field_with_input(lhs, field);
                                    let nr = replace_field_with_input(rhs, field);
                                    if nl.is_some() || nr.is_some() {
                                        Some(Expr::BinOp {
                                            op: *op,
                                            lhs: Box::new(nl.unwrap_or_else(|| lhs.as_ref().clone())),
                                            rhs: Box::new(nr.unwrap_or_else(|| rhs.as_ref().clone())),
                                        })
                                    } else { None }
                                }
                                Expr::UnaryOp { op, operand } => {
                                    replace_field_with_input(operand, field).map(|o| Expr::UnaryOp { op: *op, operand: Box::new(o) })
                                }
                                Expr::CallBuiltin { name, args } => {
                                    let mut any_replaced = false;
                                    let new_args: Vec<_> = args.iter().map(|a| {
                                        if let Some(r) = replace_field_with_input(a, field) { any_replaced = true; r }
                                        else { a.clone() }
                                    }).collect();
                                    if any_replaced { Some(Expr::CallBuiltin { name: name.clone(), args: new_args }) }
                                    else { None }
                                }
                                Expr::RegexTest { input_expr, re, flags } => {
                                    replace_field_with_input(input_expr, field).map(|ie| Expr::RegexTest {
                                        input_expr: Box::new(ie), re: re.clone(), flags: flags.clone()
                                    })
                                }
                                _ => None,
                            }
                        }
                        // Only convert if value_expr ONLY references .field (no other fields or bare Input)
                        fn only_uses_field(e: &Expr, field: &str) -> bool {
                            match e {
                                Expr::Input => false, // bare . reference = other field context
                                Expr::Index { expr: base, key } if matches!(base.as_ref(), Expr::Input) => {
                                    matches!(key.as_ref(), Expr::Literal(crate::ir::Literal::Str(f)) if f == field)
                                }
                                Expr::BinOp { lhs, rhs, .. } => only_uses_field(lhs, field) && only_uses_field(rhs, field),
                                Expr::UnaryOp { operand, .. } => only_uses_field(operand, field),
                                Expr::CallBuiltin { args, .. } => args.iter().all(|a| only_uses_field(a, field)),
                                Expr::RegexTest { input_expr, re, flags } => only_uses_field(input_expr, field) && only_uses_field(re, field) && only_uses_field(flags, field),
                                Expr::Literal(_) => true,
                                _ => false, // unknown expr types = don't optimize
                            }
                        }
                        if only_uses_field(&sv, field) {
                            if let Some(update) = replace_field_with_input(&sv, field) {
                                return Expr::Update {
                                    path_expr: Box::new(sp),
                                    update_expr: Box::new(update),
                                };
                            }
                        }
                    }
                }
            }
            Expr::Assign { path_expr: Box::new(sp), value_expr: Box::new(sv) }
        }
        Expr::TryCatch { try_expr, catch_expr } => {
            Expr::TryCatch { try_expr: Box::new(simplify_expr(try_expr)), catch_expr: Box::new(simplify_expr(catch_expr)) }
        }
        // delpaths([["field"]]) → del(.field)
        Expr::Limit { count, generator } => {
            let sc = simplify_expr(count);
            let sg = simplify_expr(generator);
            // first(scalar_expr) → scalar_expr: if count >= 1 and generator produces exactly 1 output
            if let Expr::Literal(crate::ir::Literal::Num(n, _)) = &sc {
                if *n >= 1.0 {
                    fn is_single_output(e: &Expr) -> bool {
                        match e {
                            Expr::Input | Expr::Literal(_) | Expr::Not => true,
                            Expr::Index { expr, key } => is_single_output(expr) && is_single_output(key),
                            Expr::BinOp { lhs, rhs, .. } => is_single_output(lhs) && is_single_output(rhs),
                            Expr::UnaryOp { operand, .. } => is_single_output(operand),
                            Expr::Negate { operand } => is_single_output(operand),
                            Expr::Pipe { left, right } => is_single_output(left) && is_single_output(right),
                            Expr::IfThenElse { cond, then_branch, else_branch } => {
                                is_single_output(cond) && is_single_output(then_branch) && is_single_output(else_branch)
                            }
                            Expr::LoadVar { .. } => true,
                            Expr::LetBinding { value, body, .. } => is_single_output(value) && is_single_output(body),
                            Expr::CallBuiltin { args, .. } => args.iter().all(|a| is_single_output(a)),
                            Expr::ObjectConstruct { pairs } => pairs.iter().all(|(k, v)| is_single_output(k) && is_single_output(v)),
                            Expr::RegexTest { input_expr, re, flags } => is_single_output(input_expr) && is_single_output(re) && is_single_output(flags),
                            Expr::RegexSub { input_expr, re, tostr, flags } | Expr::RegexGsub { input_expr, re, tostr, flags } => {
                                is_single_output(input_expr) && is_single_output(re) && is_single_output(tostr) && is_single_output(flags)
                            }
                            Expr::Update { path_expr, update_expr } => is_single_output(path_expr) && is_single_output(update_expr),
                            Expr::Assign { path_expr, value_expr } => is_single_output(path_expr) && is_single_output(value_expr),
                            Expr::Alternative { primary, fallback } => is_single_output(primary) && is_single_output(fallback),
                            _ => false,
                        }
                    }
                    if is_single_output(&sg) {
                        return sg;
                    }
                    // first(a, b, ...) where a is single-output → a. Only valid for
                    // limit(1; ...): for larger counts we must keep the full generator.
                    if *n == 1.0 {
                        let mut g = &sg;
                        while let Expr::Comma { left, .. } = g {
                            if is_single_output(left) {
                                return simplify_expr(left);
                            }
                            g = left;
                        }
                    }
                }
            }
            Expr::Limit { count: Box::new(sc), generator: Box::new(sg) }
        }
        Expr::PathExpr { expr: pe } => {
            let sp = simplify_expr(pe);
            // Note: `path(.field)` cannot be folded to `["field"]` at
            // compile time — jq errors when the input type is not
            // indexable, so the type check must happen at runtime
            // (issue #46). Only comma distributivity is safe to fold.
            if let Expr::Comma { left, right } = &sp {
                let lp = simplify_expr(&Expr::PathExpr { expr: left.clone() });
                let rp = simplify_expr(&Expr::PathExpr { expr: right.clone() });
                return Expr::Comma { left: Box::new(lp), right: Box::new(rp) };
            }
            Expr::PathExpr { expr: Box::new(sp) }
        }
        Expr::GetPath { path } => {
            let sp = simplify_expr(path);
            // getpath(["field"]) → .field
            if let Expr::Collect { generator } = &sp {
                if let Expr::Literal(Literal::Str(field)) = generator.as_ref() {
                    return Expr::Index {
                        expr: Box::new(Expr::Input),
                        key: Box::new(Expr::Literal(Literal::Str(field.clone()))),
                    };
                }
            }
            Expr::GetPath { path: Box::new(sp) }
        }
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

/// Returns true if the expression contains any Expr::Input node (i.e., references `.`).
fn contains_input(expr: &crate::ir::Expr) -> bool {
    use crate::ir::{Expr, StringPart};
    match expr {
        Expr::Input => true,
        Expr::Literal(_) | Expr::Empty | Expr::Env | Expr::Builtins
        | Expr::ReadInput | Expr::ReadInputs | Expr::ModuleMeta
        | Expr::GenLabel | Expr::Loc { .. } | Expr::Break { .. } => false,
        // `not` negates the truthiness of current input
        Expr::Not => true,
        Expr::LoadVar { .. } => false,
        Expr::BinOp { lhs, rhs, .. } => contains_input(lhs) || contains_input(rhs),
        Expr::UnaryOp { operand, .. } | Expr::Negate { operand } => contains_input(operand),
        Expr::Index { expr: e, key } | Expr::IndexOpt { expr: e, key } => contains_input(e) || contains_input(key),
        Expr::Collect { generator } => contains_input(generator),
        Expr::Comma { left, right } => contains_input(left) || contains_input(right),
        Expr::Each { input_expr } | Expr::EachOpt { input_expr } => contains_input(input_expr),
        Expr::Pipe { left, right } => contains_input(left) || contains_input(right),
        Expr::IfThenElse { cond, then_branch, else_branch } => contains_input(cond) || contains_input(then_branch) || contains_input(else_branch),
        Expr::ObjectConstruct { pairs } => pairs.iter().any(|(k, v)| contains_input(k) || contains_input(v)),
        Expr::Alternative { primary, fallback } => contains_input(primary) || contains_input(fallback),
        Expr::Format { expr: e, .. } => contains_input(e),
        Expr::Slice { expr: e, from, to } => contains_input(e) || from.as_ref().map_or(false, |f| contains_input(f)) || to.as_ref().map_or(false, |t| contains_input(t)),
        Expr::StringInterpolation { parts } => parts.iter().any(|p| matches!(p, StringPart::Expr(e) if contains_input(e))),
        Expr::LetBinding { value, body, .. } => contains_input(value) || contains_input(body),
        Expr::Reduce { source, init, update, .. } => contains_input(source) || contains_input(init) || contains_input(update),
        Expr::Foreach { source, init, update, extract, .. } => contains_input(source) || contains_input(init) || contains_input(update) || extract.as_ref().map_or(false, |e| contains_input(e)),
        Expr::While { cond, update } | Expr::Until { cond, update } => contains_input(cond) || contains_input(update),
        Expr::Repeat { update } => contains_input(update),
        Expr::TryCatch { try_expr, catch_expr } => contains_input(try_expr) || contains_input(catch_expr),
        // CallBuiltin implicitly operates on the current input (passed as first arg)
        Expr::CallBuiltin { .. } => true,
        Expr::Range { from, to, step } => contains_input(from) || contains_input(to) || step.as_ref().map_or(false, |s| contains_input(s)),
        Expr::Limit { count, generator } => contains_input(count) || contains_input(generator),
        Expr::Error { msg } => msg.as_ref().map_or(false, |m| contains_input(m)),
        Expr::Update { path_expr, update_expr } | Expr::Assign { path_expr, value_expr: update_expr } => contains_input(path_expr) || contains_input(update_expr),
        // GetPath/SetPath/DelPaths/PathExpr implicitly operate on the current input
        Expr::GetPath { .. } | Expr::SetPath { .. } | Expr::DelPaths { .. } | Expr::PathExpr { .. } => true,
        Expr::Recurse { input_expr: e } => contains_input(e),
        // debug/stderr pass through the current input
        Expr::Debug { .. } | Expr::Stderr { .. } => true,
        Expr::Label { body, .. } => contains_input(body),
        // Conservative: assume these reference input
        Expr::RegexTest { input_expr, .. } | Expr::RegexMatch { input_expr, .. }
        | Expr::RegexCapture { input_expr, .. } | Expr::RegexScan { input_expr, .. }
        | Expr::RegexSub { input_expr, .. } | Expr::RegexGsub { input_expr, .. } => contains_input(input_expr),
        Expr::FuncCall { args, .. } => args.iter().any(contains_input),
        Expr::ClosureOp { .. } | Expr::AnyShort { .. } | Expr::AllShort { .. }
        | Expr::AlternativeDestructure { .. } => true, // conservative
    }
}

/// Serialize a constant expression to JSON bytes. Returns false if expression is not fully constant.
fn push_const_json(expr: &crate::ir::Expr, buf: &mut Vec<u8>) -> bool {
    use crate::ir::{Expr, Literal};
    match expr {
        Expr::Literal(Literal::Null) => { buf.extend_from_slice(b"null"); true }
        Expr::Literal(Literal::True) => { buf.extend_from_slice(b"true"); true }
        Expr::Literal(Literal::False) => { buf.extend_from_slice(b"false"); true }
        Expr::Literal(Literal::Num(n, Some(raw))) => {
            if crate::value::is_valid_json_number(raw) {
                buf.extend_from_slice(raw.as_bytes());
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
            // All keys must be string literals. Duplicates collapse via
            // `normalize_object_pairs` (last value wins, keeps first position).
            let mut extracted: Vec<(&str, &Expr)> = Vec::with_capacity(pairs.len());
            for (key, val) in pairs {
                match key {
                    Expr::Literal(Literal::Str(k)) => extracted.push((k.as_str(), val)),
                    _ => return false,
                }
            }
            let normalized = normalize_object_pairs(extracted);
            buf.push(b'{');
            for (i, (k, val)) in normalized.iter().enumerate() {
                if i > 0 { buf.push(b','); }
                buf.push(b'"');
                buf.extend_from_slice(k.as_bytes());
                buf.push(b'"');
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
    /// Get the inner expression for pattern detection.
    ///
    /// Previously this stripped top-level `try EXPR` (TryCatch with Empty
    /// catch) on the assumption that the raw byte fast paths handled missing
    /// fields gracefully. They don't: `(.a)?` on a non-object needs to emit
    /// nothing (the error is caught), while the fast path emitted `null`.
    /// Leave TryCatch visible so the fast paths that can't honour `?`
    /// semantics simply don't match, and eval handles it correctly (see
    /// issue #50).
    fn detect_expr(&self) -> Option<&crate::ir::Expr> {
        Some(&self.simplified)
    }

    pub fn new(program: &str) -> Result<Self> {
        Self::with_options(program, &[], true)
    }

    pub fn with_lib_dirs(program: &str, lib_dirs: &[String]) -> Result<Self> {
        Self::with_options(program, lib_dirs, true)
    }

    pub fn with_options(program: &str, lib_dirs: &[String], use_jit: bool) -> Result<Self> {
        let result = crate::parser::Parser::parse_with_libs(program, lib_dirs)?;
        let parsed = (result.expr, result.funcs);

        // Try JIT compilation for the parsed expression.
        let mut jit_fn = None;
        let mut jit_compiler = None;
        if use_jit {
            let (ref expr, ref funcs) = parsed;
            if crate::jit::is_jit_compilable_with_funcs(expr, funcs) {
                if let Ok(mut compiler) = crate::jit::JitCompiler::new() {
                    if let Ok(func) = compiler.compile_with_funcs(expr, funcs) {
                        jit_fn = Some(func);
                        jit_compiler = Some(Box::new(compiler));
                    }
                }
            }
        }

        let simplified = simplify_expr(&parsed.0);

        let _ = program;
        Ok(Filter {
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
        {
            let (ref expr, ref funcs) = self.parsed;
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
        walk(&self.parsed.0)
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
        walk(&self.parsed.0)
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
        let expr = self.detect_expr()?;
        if contains_input(expr) { return None; }
        // Already handled by literal_output?
        let mut buf = Vec::new();
        if push_const_json(expr, &mut buf) {
            return Some(vec![buf]);
        }
        // Evaluate with null input using eval
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
                                if let Expr::CallBuiltin { name, args } = right.as_ref() {
                                    if args.len() == 1 {
                                        if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                            if matches!(name.as_str(), "startswith" | "endswith" | "contains") {
                                                return Some((field.clone(), name.clone(), arg.clone()));
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
                                // CallBuiltin("test", [Literal(Str)])
                                if let Expr::CallBuiltin { name, args } = right.as_ref() {
                                    if name == "test" && args.len() == 1 {
                                        if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                            return Some((field.clone(), "test".to_string(), arg.clone()));
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
                            if let Expr::CallBuiltin { name, args } = right.as_ref() {
                                if matches!(name.as_str(), "startswith" | "endswith" | "contains") && args.len() == 1 {
                                    if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                        return Some((field.clone(), name.clone(), arg.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
                // Also handle beta-reduced form: CallBuiltin("startswith", [Index(Input, "field"), Literal("str")])
                if let Expr::CallBuiltin { name, args } = e {
                    if matches!(name.as_str(), "startswith" | "endswith" | "contains") && args.len() == 2 {
                        if let Expr::Index { expr: base, key } = &args[0] {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                    if let Expr::Literal(Literal::Str(arg)) = &args[1] {
                                        return Some((field.clone(), name.clone(), arg.clone()));
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
                                if let Expr::CallBuiltin { name, args } = right.as_ref() {
                                    if matches!(name.as_str(), "startswith" | "endswith" | "contains") && args.len() == 1 {
                                        if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                            return Some(MixedCond::StrTest(field.clone(), name.clone(), arg.clone()));
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
                if let Expr::CallBuiltin { name, args } = e {
                    if matches!(name.as_str(), "startswith" | "endswith" | "contains") && args.len() == 2 {
                        if let Expr::Index { expr: base, key } = &args[0] {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                    if let Expr::Literal(Literal::Str(arg)) = &args[1] {
                                        return Some(MixedCond::StrTest(field.clone(), name.clone(), arg.clone()));
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
        // Direct ObjectConstruct
        if let Some((result, has_computed)) = extract_computed_pairs(self, expr) {
            if has_computed { return Some(normalize_object_pairs(result)); }
            return None;
        }
        // {a:.x,b:(.y*2)} + {c:.z} — merged object constructs
        if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = expr {
            if let (Some((mut left, lc)), Some((right, rc))) = (extract_computed_pairs(self, lhs), extract_computed_pairs(self, rhs)) {
                left.extend(right);
                if lc || rc { return Some(normalize_object_pairs(left)); }
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
        let is_supported = |op: &UnaryOp| matches!(op,
            UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Sqrt |
            UnaryOp::Fabs | UnaryOp::Abs | UnaryOp::ToString |
            UnaryOp::AsciiDowncase | UnaryOp::AsciiUpcase |
            UnaryOp::Length | UnaryOp::Utf8ByteLength | UnaryOp::Explode);
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
                    if let Expr::CallBuiltin { name, args } = right.as_ref() {
                        if args.len() == 1 {
                            if matches!(name.as_str(), "startswith" | "endswith" | "ltrimstr" | "rtrimstr" | "split" | "index" | "rindex" | "indices" | "contains") {
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
                        if let Expr::CallBuiltin { name, args } = lhs.as_ref() {
                            if (name == "index" || name == "rindex") && args.len() == 1 {
                                if let Expr::Literal(Literal::Str(search)) = &args[0] {
                                    if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                        return Some((field.clone(), search.clone(), name == "rindex", *op, *n));
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
            Expr::CallBuiltin { name, args } if args.len() == 1 => {
                if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                    match name.as_str() {
                        "startswith" => return StringChainTerminal::Startswith(arg.clone()),
                        "endswith" => return StringChainTerminal::Endswith(arg.clone()),
                        "contains" => return StringChainTerminal::Contains(arg.clone()),
                        "index" => return StringChainTerminal::Index(arg.clone()),
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
            // split(sep) | join(rep) — fused as SplitJoin
            Expr::Pipe { left, right } => {
                if let Expr::CallBuiltin { name: sn, args: sa } = left.as_ref() {
                    if sn == "split" && sa.len() == 1 {
                        if let Expr::Literal(Literal::Str(sep)) = &sa[0] {
                            // Check for split | join
                            if let Expr::CallBuiltin { name: jn, args: ja } = right.as_ref() {
                                if jn == "join" && ja.len() == 1 {
                                    if let Expr::Literal(Literal::Str(rep)) = &ja[0] {
                                        ops.push(StringChainOp::SplitJoin(sep.clone(), rep.clone()));
                                        return true;
                                    }
                                }
                            }
                            // Check for split | reverse | join
                            if let Expr::Pipe { left: rev, right: join_expr } = right.as_ref() {
                                if matches!(rev.as_ref(), Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                                    if let Expr::CallBuiltin { name: jn, args: ja } = join_expr.as_ref() {
                                        if jn == "join" && ja.len() == 1 {
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
                            Expr::CallBuiltin { name, args } if name == "test" && args.len() == 1 => {
                                if let Expr::Literal(Literal::Str(pattern)) = &args[0] {
                                    return Some((field.clone(), is_upper, pattern.clone()));
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
                        if let Expr::CallBuiltin { name, args } = mid.as_ref() {
                            if name == "ltrimstr" && args.len() == 1 {
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
                        if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                            if name == "split" && args.len() == 1 {
                                if let Expr::Literal(Literal::Str(sep)) = &args[0] {
                                    if let Expr::Pipe { left: rev_expr, right: join_expr } = rest.as_ref() {
                                        if matches!(rev_expr.as_ref(), Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                                            if let Expr::CallBuiltin { name: jn, args: ja } = join_expr.as_ref() {
                                                if jn == "join" && ja.len() == 1 {
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
                            if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                                if name == "split" && args.len() == 1 {
                                    if let Expr::Literal(Literal::Str(sep)) = &args[0] {
                                        if let Expr::CallBuiltin { name: jn, args: ja } = join_expr.as_ref() {
                                            if jn == "join" && ja.len() == 1 {
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
                        if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                            if name == "split" && args.len() == 1 {
                                if let Expr::Literal(Literal::Str(sep)) = &args[0] {
                                    if !sep.is_empty() {
                                        return Some((field.clone(), is_upper, sep.clone()));
                                    }
                                }
                            }
                        }
                    }
                    // Also handle beta-reduced form: CallBuiltin(split, [UnaryOp(case, .field)])
                    if let Expr::CallBuiltin { name, args } = right.as_ref() {
                        if name == "split" && args.len() == 1 {
                            if let Expr::Literal(Literal::Str(_sep)) = &args[0] {
                                // This form doesn't include the case op, skip
                            }
                        }
                    }
                }
            }
        }
        // Beta-reduced: CallBuiltin(split, [UnaryOp(case, Index(.field))])
        if let Expr::CallBuiltin { name, args } = expr {
            if name == "split" && args.len() == 1 {
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
                            if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                                if name == "split" && args.len() == 1 {
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
                                            // Check for first/last
                                            if let Expr::CallBuiltin { name: fn_name, args: fn_args } = idx_expr.as_ref() {
                                                if fn_args.is_empty() || (fn_args.len() == 1 && matches!(fn_args[0], Expr::Input)) {
                                                    if fn_name == "first" {
                                                        return Some((field.clone(), is_upper, sep.clone(), 0));
                                                    } else if fn_name == "last" {
                                                        return Some((field.clone(), is_upper, sep.clone(), -1));
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
        if let Expr::CallBuiltin { name, args } = del_expr {
            if name != "del" || args.len() != 1 { return None; }
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
                    if let Expr::CallBuiltin { name, args } = right.as_ref() {
                        if args.len() == 1 {
                            if let Expr::Literal(Literal::Str(s)) = &args[0] {
                                match name.as_str() {
                                    "startswith" | "endswith" | "test" | "contains" => {
                                        (f.clone(), name.clone(), s.clone())
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
        if let Expr::CallBuiltin { name, args } = del_expr {
            if name != "del" || args.len() != 1 { return None; }
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
                    if let Expr::CallBuiltin { name, args } = right.as_ref() {
                        if args.len() == 1 {
                            if let Expr::Literal(Literal::Str(s)) = &args[0] {
                                match name.as_str() {
                                    "startswith" | "endswith" | "test" | "contains" => {
                                        (f.clone(), name.clone(), s.clone())
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
        if let Expr::CallBuiltin { name, args } = expr {
            if name == "del" && args.len() == 1 {
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
                            if let Expr::CallBuiltin { name, args } = cond.as_ref() {
                                if matches!(name.as_str(), "startswith" | "endswith" | "contains") && args.len() == 2 {
                                    if let Expr::Index { expr: base, key } = &args[0] {
                                        if matches!(base.as_ref(), Expr::Input) {
                                            if let Expr::Literal(Literal::Str(f)) = key.as_ref() {
                                                if f == "key" {
                                                    if let Expr::Literal(Literal::Str(s)) = &args[1] {
                                                        return Some((name.clone(), s.clone()));
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
                                                if let Expr::CallBuiltin { name, args } = pr.as_ref() {
                                                    if matches!(name.as_str(), "startswith" | "endswith" | "contains") {
                                                        // 1-arg form: CallBuiltin("startswith", [Literal("str")])
                                                        if args.len() == 1 {
                                                            if let Expr::Literal(Literal::Str(s)) = &args[0] {
                                                                return Some((name.clone(), s.clone()));
                                                            }
                                                        }
                                                        // 2-arg form: CallBuiltin("startswith", [Input, Literal("str")])
                                                        if args.len() == 2 && matches!(args[0], Expr::Input) {
                                                            if let Expr::Literal(Literal::Str(s)) = &args[1] {
                                                                return Some((name.clone(), s.clone()));
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
            || matches!(expr, Expr::Format { name, expr: inner } if name == "json" && matches!(inner.as_ref(), Expr::Input))
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
            Expr::Format { name, expr: inner } if name == "json" && matches!(inner.as_ref(), Expr::Input));
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
                let split_sep = if let Expr::CallBuiltin { name, args } = l2.as_ref() {
                    if name != "split" || args.len() != 1 { return None; }
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
                    let join_sep = if let Expr::CallBuiltin { name, args } = r3.as_ref() {
                        if name != "join" || args.len() != 1 { return None; }
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
            if let Expr::CallBuiltin { name, args } = right.as_ref() {
                if name == "join" && args.len() == 1 {
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
                        if let Expr::CallBuiltin { name, args } = e {
                            if name == "split" && args.len() == 1 {
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
                        // Form 3: CallBuiltin("last", [split])
                        if let Expr::CallBuiltin { name, args } = e {
                            if name == "last" && args.len() == 1 {
                                if let Some(sep) = extract_split(&args[0]) { return Some(sep); }
                            }
                        }
                        None
                    }
                    fn extract_split(e: &Expr) -> Option<String> {
                        if let Expr::CallBuiltin { name, args } = e {
                            if name == "split" && args.len() == 1 {
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
                        if let Expr::CallBuiltin { name, args } = e {
                            if name == "last" && args.len() == 1 && matches!(args[0], Expr::Input) {
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
                    if let Expr::CallBuiltin { name, args } = upd.as_ref() {
                        if args.len() == 1 {
                            if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                match name.as_str() {
                                    "ltrimstr" => return Some((field.clone(), arg.clone(), false)),
                                    "rtrimstr" => return Some((field.clone(), arg.clone(), true)),
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
                        if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                            if name != "split" || args.len() != 1 { return None; }
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
                        if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                            if name != "split" || args.len() != 1 { return None; }
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
                        if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                            if name != "split" || args.len() != 1 { return None; }
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
                                        if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                                            if name == "split" && args.len() == 1 {
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
                        if let Expr::CallBuiltin { name, args } = inner.as_ref() {
                            if name == "split" && args.len() == 2 {
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
                            if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                                if name == "split" && args.len() == 1 {
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
                        if let Expr::CallBuiltin { name, args } = split_expr.as_ref() {
                            if name != "split" || args.len() != 1 { return None; }
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
                                            "test" => Some(CondRhs::Test(s.clone())),
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
                                if let Expr::CallBuiltin { name, args } = pipe_right.as_ref() {
                                    if matches!(name.as_str(), "startswith" | "endswith" | "contains") && args.len() == 1 {
                                        if let Expr::Literal(Literal::Str(arg)) = &args[0] {
                                            return Some((field.clone(), name.clone(), arg.clone(), upd_field, arith_op, arith_val));
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
                            if let Expr::CallBuiltin { name, args } = right.as_ref() {
                                if matches!(name.as_str(), "startswith" | "endswith" | "contains") && args.len() == 1 {
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
        if let Expr::CallBuiltin { name, args } = expr {
            if name != "walk" || args.len() != 1 { return None; }
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

    /// Execute the filter against an input value, collecting all results.
    pub fn execute(&self, input: &Value) -> Result<Vec<Value>> {
        // Try JIT execution first
        if let Some(jit_fn) = self.jit_fn {
            return crate::jit::execute_jit(jit_fn, input);
        }

        let (ref expr, ref funcs) = self.parsed;
        crate::eval::execute_ir_with_libs(expr, input.clone(), funcs.clone(), self.lib_dirs.clone())
    }

    /// Execute the filter with a callback for each result (avoids Vec allocation).
    /// Returns Ok(true) if all values were processed, Ok(false) if stopped early.
    pub fn execute_cb(&self, input: &Value, cb: &mut dyn FnMut(&Value) -> Result<bool>) -> Result<bool> {
        if let Some(jit_fn) = self.jit_fn {
            return crate::jit::execute_jit_cb(jit_fn, input, cb);
        }

        let (ref expr, ref funcs) = self.parsed;
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
        crate::eval::execute_ir_with_env_cb(
            expr, input.clone(), &env,
            &mut |val| {
                if let Value::Error(e) = &val {
                    eprintln!("jq: error: {}", e.as_str());
                    return Ok(true);
                }
                cb(&val)
            },
        )
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

