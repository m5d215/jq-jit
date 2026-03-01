//! Run jq filters via libjq for differential testing.
//!
//! This module uses libjq's public API to execute a jq filter on a JSON input
//! string and return the results as JSON strings.  It serves as the "reference
//! implementation" side of the differential testing framework.

use std::ffi::{CStr, CString};

use anyhow::{Context, Result, bail};

use crate::jq_ffi::{self, JvKind};

/// Run a jq filter on a JSON input string using libjq.
///
/// Returns a `Vec<String>` of JSON-encoded results.  For Phase 1 filters
/// (no generators), this will typically contain exactly one result.
///
/// # Example
///
/// ```ignore
/// let results = run_jq(". + 1", "5")?;
/// assert_eq!(results, vec!["6"]);
/// ```
pub fn run_jq(filter: &str, input_json: &str) -> Result<Vec<String>> {
    let c_filter = CString::new(filter).context("filter contains null byte")?;
    let c_input = CString::new(input_json).context("input_json contains null byte")?;

    unsafe {
        // 1. Initialize jq state
        let mut state = jq_ffi::jq_init();
        if state.is_null() {
            bail!("jq_init() returned null");
        }

        // 2. Compile the filter
        let compile_result = jq_ffi::jq_compile(state, c_filter.as_ptr());
        if compile_result == 0 {
            jq_ffi::jq_teardown(&mut state);
            bail!("jq_compile({:?}) failed", filter);
        }

        // 3. Parse the input JSON
        let input_jv = jq_ffi::jv_parse(c_input.as_ptr());
        let input_kind = jq_ffi::jv_get_kind(input_jv);
        if input_kind == JvKind::Invalid {
            jq_ffi::jv_free(input_jv);
            jq_ffi::jq_teardown(&mut state);
            bail!("jv_parse({:?}) returned invalid", input_json);
        }

        // 4. Start execution (jq_start consumes input_jv)
        jq_ffi::jq_start(state, input_jv, 0);

        // 5. Collect results
        let mut results = Vec::new();
        loop {
            let result_jv = jq_ffi::jq_next(state);
            let kind = jq_ffi::jv_get_kind(result_jv);

            if kind == JvKind::Invalid {
                jq_ffi::jv_free(result_jv);
                break;
            }

            // Convert result to JSON string via jv_dump_string.
            // jv_dump_string consumes the jv and returns a jv string.
            let dump_jv = jq_ffi::jv_dump_string(result_jv, 0);
            let dump_kind = jq_ffi::jv_get_kind(dump_jv);

            if dump_kind == JvKind::String {
                let cstr = jq_ffi::jv_string_value(dump_jv);
                let s = CStr::from_ptr(cstr).to_string_lossy().into_owned();
                // jv_string_value returns a pointer into the jv, so read it
                // before freeing.
                jq_ffi::jv_free(dump_jv);
                results.push(s);
            } else {
                jq_ffi::jv_free(dump_jv);
                results.push("<dump failed>".to_string());
            }
        }

        // 6. Teardown
        jq_ffi::jq_teardown(&mut state);

        Ok(results)
    }
}
