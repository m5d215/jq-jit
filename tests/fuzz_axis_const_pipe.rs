//! Axis fuzz: `Pipe(<lhs>, <const-rhs>)` biased (#686 axis 5).
//!
//! [`tests/fuzz_restricted.rs`]'s recursive generator produces Pipe
//! shapes freely, but the constant-RHS subset is statistically rare —
//! a `<const>` leaf has the same probability as any other leaf, so most
//! random Pipes have RHSs that reference input.
//!
//! Constant-RHS Pipe (`(<lhs>) | <constant>`) is its own fast-path
//! shape: jq evaluates `<lhs>` for cardinality (errors propagate,
//! multi-valued LHS emits the constant N times) but the RHS is
//! input-independent. The simplify layer (#685 axis b) tries to fold
//! `_ | const` into `const`; that fold is only safe when `<lhs>` is
//! statically known to be single-valued and side-effect-free, and
//! getting the analysis wrong is exactly the kind of divergence this
//! axis aims to surface.
//!
//! Filter shape: `((<lhs>) | <const>)?` where `<lhs>` is the shared
//! single-valued-safe base and `<const>` is a literal leaf
//! (`null`, bool, integer, string, empty array / object). The outer
//! `?` swallows the per-LHS-value errors so the property under test is
//! whether the constant emits with the *cardinality* and *value* jq
//! expects.
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

const TEST_LABEL: &str = "fuzz_axis_const_pipe";

/// Literal RHS pool. Every entry is a complete jq filter that ignores
/// its input and emits exactly one canonical value. The empty `[]` /
/// `{}` shapes exercise the array / object construct fast paths in the
/// constant subset; `null` / `true` / `false` and the integer / string
/// pool catch the literal-fast-path detectors.
const CONST_RHS: &[&str] = &[
    "null", "true", "false",
    "0", "1", "-1", "42",
    "\"\"", "\"x\"", "\"hello\"",
    "[]", "{}",
];

#[test]
fn fuzz_axis_const_pipe_against_jq_1_8() {
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
        prop::sample::select(CONST_RHS),
        base_json_strategy(conservative_json_leaf(), 3, 12, 3),
    );

    let result = runner.run(&strategy, |(lhs, rhs, input_shape)| {
        let filter = format!("(({}) | {})?", render(&lhs), rhs);
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
        "=== fuzz_axis_const_pipe (vs {}) ===\n  compared: {}\n  both_errored: {}",
        jq,
        compared.load(std::sync::atomic::Ordering::Relaxed),
        both_error.load(std::sync::atomic::Ordering::Relaxed),
    );

    if let Err(e) = result {
        panic!("fuzz_axis_const_pipe failed:\n{}", e);
    }
}
