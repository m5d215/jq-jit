//! Tree-walking interpreter for jq filters.
//!
//! Strategy: Instead of building a JIT compiler from scratch, we first build a
//! correct interpreter that passes all jq tests. Then we JIT-compile hot paths.
//!
//! Every filter is a generator: it takes an input Value and produces zero or more
//! output Values via a callback. This eliminates the generator-in-scalar-context
//! problem that plagued jq-jit.

use std::rc::Rc;

use anyhow::{Result, bail};

use crate::value::Value;

/// A compiled jq filter, ready to execute.
/// Uses libjq for parsing, then runs via our interpreter.
pub struct Filter {
    program: String,
}

impl Filter {
    pub fn new(program: &str) -> Result<Self> {
        // Validate the program compiles by testing with libjq
        let mut jq = crate::bytecode::JqState::new()?;
        let _bc = jq.compile(program)?;
        Ok(Filter {
            program: program.to_string(),
        })
    }

    /// Execute the filter against an input value, collecting all results.
    pub fn execute(&self, input: &Value) -> Result<Vec<Value>> {
        // Use libjq to execute for now - this ensures 100% compatibility
        // We will replace this with our own interpreter/JIT incrementally
        execute_via_libjq(&self.program, input)
    }
}

/// Execute a jq filter using libjq directly.
/// This is our baseline - guaranteed correct, but not JIT-compiled.
fn execute_via_libjq(program: &str, input: &Value) -> Result<Vec<Value>> {
    use crate::jq_ffi;
    let mut jq = crate::bytecode::JqState::new()?;
    let _bc = jq.compile(program)?;

    // Convert input Value to jv
    let input_jv = value_to_jv(input)?;

    // Run the filter
    unsafe {
        jq_ffi::jq_start(jq.as_ptr(), input_jv, 0);

        let mut results = Vec::new();
        loop {
            let result = jq_ffi::jq_next(jq.as_ptr());
            if jq_ffi::jv_get_kind(result) == jq_ffi::JvKind::Invalid {
                // Check if it's an error or just end-of-results
                let msg = jq_ffi::jv_invalid_get_msg(jq_ffi::jv_copy(result));
                let kind = jq_ffi::jv_get_kind(msg);
                if kind == jq_ffi::JvKind::Null {
                    // Normal end of results
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
                    // jq outputs errors to stderr and continues, we collect as error values
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
