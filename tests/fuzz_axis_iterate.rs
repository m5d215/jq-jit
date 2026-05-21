//! Axis fuzz: `.[]` (iterate-all) re-enabled (#686 axis 3).
//!
//! [`tests/fuzz_restricted.rs`] has no AST variant for `.[]` — the
//! shared [`common::filter_strategy::FilterExpr`] omits it because the
//! conservative single-valued base is the contract that strategy
//! provides, and `.[]` is the canonical multi-valued shape. Random
//! recursion in fuzz_restricted produces something *close* via
//! `Map(Identity)` (`map(.)` → `[.[]]`) and `Limit(n; …)`, but the bare
//! `.[]` placed at the head of the pipeline is a different fast-path
//! shape: it touches the raw-byte iterate-all detector
//! (`detect_iterate_all_then_*` family) without the array-construct
//! wrapping. This axis re-enables that bare form.
//!
//! Filter shape: `((.[]) | (<inner>))?` where `<inner>` is the shared
//! single-valued base. Outer `?` swallows per-element type errors from
//! iterating into mixed inputs.
//!
//! Note: the iteration order on objects in jq is grammar-defined as
//! lexicographic-by-key (jq sorts on print), but the *yield* order
//! through `.[]` is insertion order. The shared
//! [`common::json_normalize::normalize`] sorts object keys for the final
//! value-level compare; for top-level array iteration, both
//! implementations yield in array order. So normalization handles both
//! cases without needing harness-specific reordering.
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
    conservative_leaf_strategy, render, render_json,
};
use common::json_normalize::normalize;

const TEST_LABEL: &str = "fuzz_axis_iterate";

#[test]
fn fuzz_axis_iterate_against_jq_1_8() {
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
        base_filter_strategy(conservative_leaf_strategy(), 3, 16, 3),
        base_json_strategy(conservative_json_leaf(), 3, 12, 3),
    );

    let result = runner.run(&strategy, |(inner, input_shape)| {
        // `.[]` placed at the head — each element / value of the input
        // becomes the input to `<inner>`. Outer `?` swallows per-element
        // type errors.
        let filter = format!("((.[]) | ({}))?", render(&inner));
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
        "=== fuzz_axis_iterate (vs {}) ===\n  compared: {}\n  both_errored: {}",
        jq,
        compared.load(std::sync::atomic::Ordering::Relaxed),
        both_error.load(std::sync::atomic::Ordering::Relaxed),
    );

    if let Err(e) = result {
        panic!("fuzz_axis_iterate failed:\n{}", e);
    }
}
