//! Axis fuzz: `.f = expr` / `.f |= expr` / `.f += expr` re-enabled
//! (#686 axis 4 — substitutes for the issue's defunct "dup_keys" axis;
//! object-key dedup is now generated freely in
//! [`common::filter_strategy::base_json_strategy`] and tracked by #233 /
//! #325, leaving assignments as the next major real exclusion).
//!
//! [`tests/fuzz_restricted.rs`] has no AST variant for any assignment
//! form. The shared [`common::filter_strategy::FilterExpr`] omits them
//! because the conservative single-valued contract holds for *value*
//! filters only; assignment is a path-context operation with its own
//! fast-path family (`detect_assign_*`, `detect_update_*`,
//! `detect_arith_update_*`). This axis re-enables the three most common
//! forms in isolation.
//!
//! Filter shape: `((.f <op> <rhs>) | .)?` where `.f` draws from
//! [`common::filter_strategy::IDENT_POOL`] and `<rhs>` is the shared
//! single-valued base. The trailing `| .` is just to force the assigned
//! value out through the pipeline; the outer `?` swallows path-error
//! messages on inputs whose top-level type doesn't admit field
//! assignment (numbers, strings).
//!
//! ## Knobs
//!
//! * `JQJIT_PROPTEST_CASES` — case budget (default 200)
//! * `JQJIT_PROPTEST_TIMEOUT_SECS` — per-subprocess cap (default 3)
//! * `JQ_BIN` — override the reference jq binary

mod common;

use std::time::Duration;

use proptest::prelude::*;

use common::diff_harness::{jq_jit_path, require_jq, run_filter};
use common::filter_strategy::{
    base_filter_strategy, base_json_strategy, conservative_json_leaf,
    conservative_leaf_strategy, ident_strategy, render, render_json,
};
use common::json_normalize::normalize;

const TEST_LABEL: &str = "fuzz_axis_assign";

/// The three assignment operators jq treats as a closed family. `=`
/// (plain), `|=` (update-with-pipeline, applies `expr` to the existing
/// value at the path), `+=` (arithmetic-update). Adding `-=`/`*=`/`/=`
/// is the natural next widening once these three land clean.
const ASSIGN_OPS: &[&str] = &["=", "|=", "+="];

#[test]
fn fuzz_axis_assign_against_jq_1_8() {
    let Some(jq) = require_jq(TEST_LABEL) else { return };
    let jq_jit = jq_jit_path().to_string();

    let cases: u32 = std::env::var("JQJIT_PROPTEST_CASES")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let timeout_secs: u64 = std::env::var("JQJIT_PROPTEST_TIMEOUT_SECS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let timeout = Duration::from_secs(timeout_secs);

    let compared = std::sync::atomic::AtomicUsize::new(0);
    let both_error = std::sync::atomic::AtomicUsize::new(0);

    let cfg = ProptestConfig {
        cases, failure_persistence: None, max_shrink_time: 15_000,
        ..ProptestConfig::default()
    };

    let mut runner = proptest::test_runner::TestRunner::new(cfg);
    let strategy = (
        ident_strategy(),
        prop::sample::select(ASSIGN_OPS),
        base_filter_strategy(conservative_leaf_strategy(), 3, 16, 3),
        base_json_strategy(conservative_json_leaf(), 3, 12, 3),
    );

    let result = runner.run(&strategy, |(lhs_field, op, rhs, input_shape)| {
        let filter = format!("((.{} {} ({})) | .)?", lhs_field, op, render(&rhs));
        let input = render_json(&input_shape);

        let Some(r_jq) = run_filter(&jq, &filter, &input, timeout) else { return Ok(()); };
        let Some(r_jit) = run_filter(&jq_jit, &filter, &input, timeout) else { return Ok(()); };

        let crash_markers = ["panicked", "SIGSEGV", "Assertion failed", "stack overflow", "RUST_BACKTRACE"];
        if crash_markers.iter().any(|m| r_jit.stdout.contains(m)) {
            return Err(TestCaseError::fail(format!(
                "jq-jit crashed\n  filter: {}\n  input:  {}\n  stderr: {}",
                filter, input, r_jit.stdout.trim()
            )));
        }

        if r_jq.is_error && r_jit.is_error {
            both_error.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        if r_jq.is_error != r_jit.is_error {
            return Err(TestCaseError::fail(format!(
                "error mismatch (jq error={}, jit error={})\n  filter: {}\n  input:  {}\n  jq:  {}\n  jit: {}",
                r_jq.is_error, r_jit.is_error, filter, input,
                r_jq.stdout.trim(), r_jit.stdout.trim()
            )));
        }

        let a_norm = match normalize(&r_jq.stdout) { Ok(s) => s, Err(_) => return Ok(()) };
        let b_norm = match normalize(&r_jit.stdout) {
            Ok(s) => s,
            Err(e) => return Err(TestCaseError::fail(format!(
                "jq-jit emitted non-JSON\n  filter: {}\n  input:  {}\n  err: {}\n  jit: {}",
                filter, input, e, r_jit.stdout.trim()
            ))),
        };

        compared.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if a_norm != b_norm {
            return Err(TestCaseError::fail(format!(
                "value mismatch\n  filter: {}\n  input:  {}\n  jq:  {}\n  jit: {}",
                filter, input, a_norm, b_norm
            )));
        }
        Ok(())
    });

    eprintln!(
        "=== fuzz_axis_assign (vs {}) ===\n  compared: {}\n  both_errored: {}",
        jq,
        compared.load(std::sync::atomic::Ordering::Relaxed),
        both_error.load(std::sync::atomic::Ordering::Relaxed),
    );

    if let Err(e) = result {
        panic!("fuzz_axis_assign failed:\n{}", e);
    }
}
