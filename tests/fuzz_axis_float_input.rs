//! Axis fuzz: float values in JSON *input* re-enabled (#686 axis 2).
//!
//! [`tests/fuzz_restricted.rs`] excludes input-side float literals
//! ("jq's number printer normalizes formatting in ways the harness's
//! `serde_json` re-parse can mask, leading to false-positive shrinks").
//! Filter-side float literals stayed in via
//! [`fuzz_restricted::FLOAT_LITERALS`], but the input-side path is its
//! own surface: parser → value factory → fast-path numeric ops →
//! printer.
//!
//! This axis re-enables float input values in isolation, with a narrow
//! pool that round-trips through `serde_json`'s `f64` parser cleanly
//! (no overflow forms, no non-finite, no denormals) so a shrunk failure
//! points at a *value-level* divergence rather than a known printer
//! asymmetry. The shared
//! [`common::json_normalize::normalize_value`] already folds
//! integer-valued `f64` into integers, so `1.0` vs `1` comparisons are
//! handled at the harness level.
//!
//! Filter shape: `(<inner>)?` with `<inner>` drawn from the shared
//! single-valued-safe base. Outer `?` swallows the type-error message
//! divergences that arise when a filter expecting an array hits a float.
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
    base_filter_strategy, conservative_leaf_strategy, ident_strategy, render,
};
use common::json_normalize::normalize;

const TEST_LABEL: &str = "fuzz_axis_float_input";

/// Float values that round-trip through `serde_json::from_str::<Value>`
/// → `Number::as_f64` cleanly on both jq-1.8 and jq-jit. All are finite,
/// in-range, and have unambiguous canonical formatting. The integer-
/// valued samples (`0.0`, `1.0`, `-2.0`) catch the `f64 == i64` fold path
/// in `normalize_value`; the fractional samples exercise the genuine
/// floating arithmetic path.
const INPUT_FLOATS: &[f64] = &[
    0.0, 1.0, -1.0, 2.0, -2.0,
    0.5, -0.5, 1.5, -1.5,
    0.125, 1024.0, 0.001, -0.001,
];

#[derive(Debug, Clone)]
enum FloatJson {
    Null,
    Bool(bool),
    IntN(i32),
    FloatN(f64),
    Str(String),
    Arr(Vec<FloatJson>),
    Obj(Vec<(String, FloatJson)>),
}

fn render_float_json(v: &FloatJson) -> String {
    match v {
        FloatJson::Null => "null".into(),
        FloatJson::Bool(b) => b.to_string(),
        FloatJson::IntN(n) => n.to_string(),
        FloatJson::FloatN(f) => {
            // Force a trailing `.0` so integer-valued floats render as
            // `1.0`, not `1` — otherwise the f64 pool would collapse into
            // the IntN distribution for this axis and defeat the purpose.
            let mut s = format!("{}", f);
            if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                s.push_str(".0");
            }
            s
        }
        FloatJson::Str(s) => serde_json::to_string(s).unwrap(),
        FloatJson::Arr(items) => {
            let parts: Vec<String> = items.iter().map(render_float_json).collect();
            format!("[{}]", parts.join(","))
        }
        FloatJson::Obj(pairs) => {
            let parts: Vec<String> = pairs
                .iter()
                .map(|(k, v)| format!("{}:{}", serde_json::to_string(k).unwrap(), render_float_json(v)))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

fn json_leaf() -> impl Strategy<Value = FloatJson> {
    // Float-heavy leaf distribution (3:1 over the other leaves) so every
    // case has at least one float somewhere, which is the axis thesis.
    prop_oneof![
        3 => prop::sample::select(INPUT_FLOATS).prop_map(FloatJson::FloatN),
        1 => prop_oneof![
            Just(FloatJson::Null),
            any::<bool>().prop_map(FloatJson::Bool),
            (-3i32..=3).prop_map(FloatJson::IntN),
            prop::sample::select(vec!["", "a", "ab", "0", "hello"])
                .prop_map(|s| FloatJson::Str(s.to_string())),
        ],
    ]
}

fn json_strategy() -> impl Strategy<Value = FloatJson> {
    json_leaf().prop_recursive(3, 12, 3, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..=3).prop_map(FloatJson::Arr),
            prop::collection::vec((ident_strategy(), inner.clone()), 0..=3)
                .prop_map(FloatJson::Obj),
        ]
    })
}

#[test]
fn fuzz_axis_float_input_against_jq_1_8() {
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
        json_strategy(),
    );

    let result = runner.run(&strategy, |(inner, input_shape)| {
        let filter = format!("({})?", render(&inner));
        let input = render_float_json(&input_shape);

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
        "=== fuzz_axis_float_input (vs {}) ===\n  compared: {}\n  both_errored: {}",
        jq,
        compared.load(std::sync::atomic::Ordering::Relaxed),
        both_error.load(std::sync::atomic::Ordering::Relaxed),
    );

    if let Err(e) = result {
        panic!("fuzz_axis_float_input failed:\n{}", e);
    }
}
