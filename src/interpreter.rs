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
