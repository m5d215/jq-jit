//! Issue #919: a literal `-0` in program source must normalize to `0` under
//! `-n` (null-input mode), matching both jq and jq-jit's own piped-input path.
//! The JIT `-n` path previously const-folded `Negate` via a raw IEEE negate,
//! leaking a signed zero (`-0`) and dropping a negated literal's decimal repr
//! (`-1.0` -> `-1`). The regression-test harness only pipes input (interpreter
//! path), so the `-n` JIT path is exercised by shelling out directly.

use std::process::Command;

fn run(filter: &str) -> String {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let out = Command::new(jq_jit)
        .args(["-nc", filter])
        .output()
        .expect("failed to spawn jq-jit");
    String::from_utf8(out.stdout)
        .expect("non-utf8 stdout")
        .trim_end()
        .to_string()
}

#[test]
fn literal_neg_zero_normalizes_under_null_input() {
    assert_eq!(run("-0"), "0");
}

#[test]
fn literal_neg_zero_decimal_keeps_decimal_drops_sign() {
    assert_eq!(run("-0.0"), "0.0");
}

#[test]
fn literal_neg_zero_in_array() {
    assert_eq!(run("[-0]"), "[0]");
}

#[test]
fn literal_neg_zero_in_object_value() {
    assert_eq!(run("{\"a\":-0}"), "{\"a\":0}");
}

#[test]
fn negated_decimal_literal_keeps_decimal_repr() {
    // The broader gap the #919 fix closed: the JIT Negate fast path carried no
    // repr, so `-1.0` rendered "-1" under -n.
    assert_eq!(run("-1.0"), "-1.0");
}

#[test]
fn ordinary_negation_unaffected() {
    assert_eq!(run("-5"), "-5");
    assert_eq!(run("-3.5"), "-3.5");
    assert_eq!(run("0|-."), "0");
    assert_eq!(run("5|-."), "-5");
}
