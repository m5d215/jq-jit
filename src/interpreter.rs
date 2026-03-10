//! Filter execution: parser → IR → tree-walking interpreter.
//!
//! Primary path: our own parser + eval (full control, correct behavior).
//! Fallback: libjq execution (for filters we can't parse yet).

use std::rc::Rc;

use anyhow::{Result, bail};

use crate::ir::CompiledFunc;
use crate::value::Value;

/// A compiled jq filter, ready to execute.
pub struct Filter {
    program: String,
    /// Our parsed IR (if parsing succeeded).
    parsed: Option<(crate::ir::Expr, Vec<CompiledFunc>)>,
    /// JIT-compiled function (if JIT compilation succeeded).
    jit_fn: Option<crate::jit::JitFilterFn>,
    /// JIT compiler kept alive to own the compiled code.
    _jit_compiler: Option<Box<crate::jit::JitCompiler>>,
    lib_dirs: Vec<String>,
}

impl Filter {
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

        Ok(Filter {
            program: program.to_string(),
            parsed,
            jit_fn,
            _jit_compiler: jit_compiler,
            lib_dirs: lib_dirs.to_vec(),
        })
    }

    /// Returns true if this filter is a simple identity (`.`) that passes through input unchanged.
    pub fn is_identity(&self) -> bool {
        if let Some((ref expr, _)) = self.parsed {
            matches!(expr, crate::ir::Expr::Input)
        } else {
            false
        }
    }

    /// Detect a literal filter that doesn't reference input.
    /// Returns the compact JSON bytes for the literal, or None.
    pub fn detect_literal_output(&self) -> Option<Vec<u8>> {
        use crate::ir::{Expr, Literal};
        let (ref expr, _) = self.parsed.as_ref()?;
        match expr {
            Expr::Literal(Literal::Null) => Some(b"null".to_vec()),
            Expr::Literal(Literal::True) => Some(b"true".to_vec()),
            Expr::Literal(Literal::False) => Some(b"false".to_vec()),
            Expr::Literal(Literal::Num(_, Some(raw))) => Some(raw.as_bytes().to_vec()),
            Expr::Literal(Literal::Num(n, None)) => {
                let mut buf = Vec::new();
                crate::value::push_jq_number_bytes(&mut buf, *n);
                Some(buf)
            }
            Expr::Literal(Literal::Str(s)) => {
                let mut buf = Vec::new();
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
                Some(buf)
            }
            _ => None,
        }
    }

    /// Returns true if this filter produces no output (e.g. `empty`).
    pub fn is_empty(&self) -> bool {
        if let Some((ref expr, _)) = self.parsed {
            matches!(expr, crate::ir::Expr::Empty)
        } else {
            false
        }
    }

    /// Detect `select(.field > N)` pattern for fast-path select.
    /// Returns (field_name, comparison_op, threshold) if detected.
    pub fn detect_select_field_cmp(&self) -> Option<(String, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal};
        let (ref expr, _) = self.parsed.as_ref()?;
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

    /// Detect `select(.f1 > N and .f2 < M)` or `select(.f1 > N or .f2 < M)` pattern.
    /// Returns (conjunct, Vec<(field, op, threshold)>) where conjunct is And or Or.
    pub fn detect_select_compound_cmp(&self) -> Option<(crate::ir::BinOp, Vec<(String, crate::ir::BinOp, f64)>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let (ref expr, _) = self.parsed.as_ref()?;
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

    /// Detect `select(.a.b.c > N)` pattern for nested field numeric comparison.
    /// Returns (field_path, comparison_op, threshold) if detected.
    pub fn detect_select_nested_cmp(&self) -> Option<(Vec<String>, crate::ir::BinOp, f64)> {
        use crate::ir::{Expr, BinOp, Literal};
        let (ref expr, _) = self.parsed.as_ref()?;
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
        let (ref expr, _) = self.parsed.as_ref()?;
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
        let (ref expr, _) = self.parsed.as_ref()?;
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

    /// Detect `{a: .x, b: .y}` pattern (object construction from field access).
    /// Returns Vec of (output_key, input_field) pairs if detected.
    pub fn detect_field_remap(&self) -> Option<Vec<(String, String)>> {
        use crate::ir::{Expr, Literal};
        let (ref expr, _) = self.parsed.as_ref()?;
        if let Expr::ObjectConstruct { pairs } = expr {
            if pairs.is_empty() { return None; }
            let mut result = Vec::with_capacity(pairs.len());
            for (k, v) in pairs {
                // Key must be a literal string
                let key = if let Expr::Literal(Literal::Str(s)) = k {
                    s.clone()
                } else { return None; };
                // Value must be .field (Index on Input with literal string key)
                if let Expr::Index { expr: base, key: field_key } = v {
                    if !matches!(base.as_ref(), Expr::Input) { return None; }
                    if let Expr::Literal(Literal::Str(f)) = field_key.as_ref() {
                        result.push((key, f.clone()));
                    } else { return None; }
                } else { return None; }
            }
            return Some(result);
        }
        None
    }

    /// Detect `.field1 op .field2` pattern (binary arithmetic on two input fields).
    /// Returns (field1, op, field2) if detected.
    pub fn detect_field_binop(&self) -> Option<(String, crate::ir::BinOp, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let (ref expr, _) = self.parsed.as_ref()?;
        if let Expr::BinOp { op, lhs, rhs } = expr {
            if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul) { return None; }
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
        let (ref expr, _) = self.parsed.as_ref()?;
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
        let (ref expr, _) = self.parsed.as_ref()?;
        if let Expr::Pipe { left, right } = expr {
            // left = .field
            if let Expr::Index { expr: base, key } = left.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::UnaryOp { op, operand } = right.as_ref() {
                        if !matches!(operand.as_ref(), Expr::Input) { return None; }
                        if matches!(op, UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Sqrt |
                            UnaryOp::Fabs | UnaryOp::Abs | UnaryOp::ToString |
                            UnaryOp::AsciiDowncase | UnaryOp::AsciiUpcase |
                            UnaryOp::Length | UnaryOp::Utf8ByteLength) {
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
        let (ref expr, _) = self.parsed.as_ref()?;
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

    /// Detect `.field | ltrimstr("prefix") | tonumber` pattern.
    /// Returns (field_name, prefix) if detected.
    pub fn detect_field_ltrimstr_tonumber(&self) -> Option<(String, String)> {
        use crate::ir::{Expr, Literal, UnaryOp};
        let (ref expr, _) = self.parsed.as_ref()?;
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
        let (ref expr, _) = self.parsed.as_ref()?;
        if let Expr::Pipe { left, right } = expr {
            // right must be join("sep")
            if let Expr::CallBuiltin { name, args } = right.as_ref() {
                if name != "join" || args.len() != 1 { return None; }
                if let Expr::Literal(Literal::Str(sep)) = &args[0] {
                    // left must be [expr1, expr2, ...]
                    if let Expr::Collect { generator } = left.as_ref() {
                        let mut parts = Vec::new();
                        fn collect_comma_parts(e: &Expr, out: &mut Vec<(bool, String)>) -> bool {
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

    /// Detect `.field / N | floor` or `.field % N` pattern (field + binop + optional unary).
    /// Returns (field_name, binop, constant, optional unary op) if detected.
    pub fn detect_field_binop_const_unary(&self) -> Option<(String, crate::ir::BinOp, f64, Option<crate::ir::UnaryOp>)> {
        use crate::ir::{Expr, BinOp, UnaryOp, Literal};
        let (ref expr, _) = self.parsed.as_ref()?;
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
                                    return Some((field.clone(), *op, *n, Some(*uop)));
                                }
                            }
                        }
                    }
                }
            }
        }
        // Case 2: `.field op N` — top-level BinOp (all arithmetic ops)
        if let Expr::BinOp { op, lhs, rhs } = expr {
            if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) { return None; }
            if let Expr::Index { expr: base, key } = lhs.as_ref() {
                if !matches!(base.as_ref(), Expr::Input) { return None; }
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                        if !matches!(op, BinOp::Div | BinOp::Mod) || *n != 0.0 {
                            return Some((field.clone(), *op, *n, None));
                        }
                    }
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
        let (ref expr, _) = self.parsed.as_ref()?;
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

    /// Detect `length` applied directly to input.
    pub fn is_length(&self) -> bool {
        if let Some((ref expr, _)) = self.parsed {
            matches!(expr, crate::ir::Expr::UnaryOp { op: crate::ir::UnaryOp::Length, operand } if matches!(operand.as_ref(), crate::ir::Expr::Input))
        } else {
            false
        }
    }

    /// Detect `keys` applied directly to input.
    pub fn is_keys(&self) -> bool {
        if let Some((ref expr, _)) = self.parsed {
            matches!(expr, crate::ir::Expr::UnaryOp { op: crate::ir::UnaryOp::Keys, operand } if matches!(operand.as_ref(), crate::ir::Expr::Input))
        } else {
            false
        }
    }

    /// Detect `del(.field)` applied directly to input.
    /// Returns the field name to delete.
    pub fn detect_del_field(&self) -> Option<String> {
        use crate::ir::{Expr, Literal};
        let (ref expr, _) = self.parsed.as_ref()?;
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
        if let Some((ref expr, _)) = self.parsed {
            matches!(expr, crate::ir::Expr::UnaryOp { op: crate::ir::UnaryOp::Type, operand } if matches!(operand.as_ref(), crate::ir::Expr::Input))
        } else {
            false
        }
    }

    /// Detect `has("field")` applied directly to input.
    /// Returns the field name if this is `has("literal_string")`.
    pub fn detect_has_field(&self) -> Option<String> {
        use crate::ir::{Expr, Literal};
        let (ref expr, _) = self.parsed.as_ref()?;
        if let Expr::CallBuiltin { name, args } = expr {
            if name == "has" && args.len() == 1 {
                if let Expr::Literal(Literal::Str(field)) = &args[0] {
                    return Some(field.clone());
                }
            }
        }
        None
    }

    /// Detect `keys_unsorted` on input.
    pub fn is_keys_unsorted(&self) -> bool {
        use crate::ir::{Expr, UnaryOp};
        if let Some((ref expr, _)) = self.parsed {
            matches!(expr, Expr::UnaryOp { op: UnaryOp::KeysUnsorted, operand } if matches!(operand.as_ref(), Expr::Input))
        } else { false }
    }

    /// Detect `to_entries` on input.
    pub fn is_to_entries(&self) -> bool {
        use crate::ir::{Expr, UnaryOp};
        if let Some((ref expr, _)) = self.parsed {
            matches!(expr, Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input))
        } else { false }
    }

    /// Detect `.[]` — each/iteration on input.
    pub fn is_each(&self) -> bool {
        use crate::ir::Expr;
        if let Some((ref expr, _)) = self.parsed {
            matches!(expr, Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input))
        } else { false }
    }

    /// Detect array-of-field-access `[.f1,.f2,...]` pattern.
    /// Returns the list of field names if this is Collect over comma field accesses.
    pub fn detect_array_field_access(&self) -> Option<Vec<String>> {
        use crate::ir::Expr;
        let (ref expr, _) = self.parsed.as_ref()?;
        if let Expr::Collect { generator } = expr {
            let mut fields = Vec::new();
            if collect_comma_fields(generator, &mut fields) && fields.len() >= 2 {
                return Some(fields);
            }
        }
        None
    }

    /// Detect `[.f1,.f2,...] | @csv` or `@tsv` pattern.
    /// Returns (field_names, format) where format is "csv" or "tsv".
    pub fn detect_array_fields_format(&self) -> Option<(Vec<String>, String)> {
        use crate::ir::Expr;
        let (ref expr, _) = self.parsed.as_ref()?;
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
        let (ref expr, _) = self.parsed.as_ref()?;
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

    /// Detect comma-separated field access `.f1,.f2,...` pattern.
    /// Returns the list of field names if all branches are direct field accesses on input.
    pub fn detect_multi_field_access(&self) -> Option<Vec<String>> {
        let (ref expr, _) = self.parsed.as_ref()?;
        let mut fields = Vec::new();
        if collect_comma_fields(expr, &mut fields) && fields.len() >= 2 {
            Some(fields)
        } else {
            None
        }
    }

    /// Detect `select(.field > N) | .output_field` or `if .field > N then .output_field else empty end`.
    /// Returns (select_field, op, threshold, output_field).
    pub fn detect_select_cmp_then_field(&self) -> Option<(String, crate::ir::BinOp, f64, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let (ref expr, _) = self.parsed.as_ref()?;
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

    /// Detect `select(.field > N) | {a:.x, b:.y}` or `if .field > N then {a:.x, b:.y} else empty end`.
    /// Returns (select_field, op, threshold, output_fields).
    pub fn detect_select_cmp_then_remap(&self) -> Option<(String, crate::ir::BinOp, f64, Vec<(String, String)>)> {
        use crate::ir::{Expr, BinOp, Literal};
        let (ref expr, _) = self.parsed.as_ref()?;
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

    /// Detect `select(.field == "str") | .output_field` or `select(.field | startswith("str")) | .output_field`.
    /// Returns (select_field, test_type, test_arg, output_field).
    /// test_type: "eq", "ne", "startswith", "endswith", "contains"
    pub fn detect_select_str_then_field(&self) -> Option<(String, String, String, String)> {
        use crate::ir::{Expr, BinOp, Literal};
        let (ref expr, _) = self.parsed.as_ref()?;
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
        let (ref expr, _) = self.parsed.as_ref()?;
        if let Expr::Index { expr: base, key } = expr {
            if !matches!(base.as_ref(), Expr::Input) { return None; }
            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                return Some(field.clone());
            }
        }
        None
    }

    /// Detect nested field access `.a.b` or `.a.b.c` pattern.
    /// Returns the chain of field names if this is chained field access on input.
    pub fn detect_nested_field_access(&self) -> Option<Vec<String>> {
        use crate::ir::{Expr, Literal};
        let (ref expr, _) = self.parsed.as_ref()?;
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
            // Stream results directly via eval callback
            return crate::eval::execute_ir_with_libs_cb(
                expr, input.clone(), funcs.clone(),
                self.lib_dirs.clone(),
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
        if let Some((ref expr, _)) = self.parsed {
            let mut fields = Vec::new();
            if collect_input_fields(expr, &mut fields) {
                // Deduplicate
                fields.sort();
                fields.dedup();
                if !fields.is_empty() {
                    return Some(fields);
                }
            }
        }
        None
    }
}

/// Recursively collect field names accessed from the input. Returns false if the filter
/// accesses the input in a way that requires the full object (e.g., bare `.`, `.[]`, `keys`).
/// Collect field names from a comma expression tree where all leaves are .field on input.
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
