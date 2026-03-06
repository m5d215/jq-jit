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
}

impl Filter {
    pub fn new(program: &str) -> Result<Self> {
        // Try our parser first
        let parsed = match crate::parser::Parser::parse(program) {
            Ok(result) => Some((result.expr, result.funcs)),
            Err(_e) => {
                // Fall back to libjq for compilation check
                let mut jq = crate::bytecode::JqState::new()?;
                let _bc = jq.compile(program)?;
                None
            }
        };

        Ok(Filter {
            program: program.to_string(),
            parsed,
        })
    }

    /// Execute the filter against an input value, collecting all results.
    pub fn execute(&self, input: &Value) -> Result<Vec<Value>> {
        if let Some((ref expr, ref funcs)) = self.parsed {
            // Use our own interpreter
            crate::eval::execute_ir(expr, input.clone(), funcs.clone())
        } else {
            // Fall back to libjq
            execute_via_libjq(&self.program, input)
        }
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
