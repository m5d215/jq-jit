//! The object-construct computed-remap fast path for `{key: (.field | length)}`
//! must not mask jq's `<type> has no length` error when the field value is a
//! boolean. The inline emitter coerced booleans to a `null` literal, so
//! `{a: (.b | length)}` on `{"b":false}` emitted `{"a":null}` while every other
//! path (`.b | length`, `[.b|length]`, `reduce …`) correctly errored.
//!
//! Surfaced by the metamorphic equivalence harness (error-class mismatch
//! between the object-construct path and the generic path). The regression-test
//! harness only compares stdout (an error and the pre-fix silent-null both pass
//! some checks), so assert the process exit status here.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(filter: &str, stdin: &str) -> (String, bool) {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .args(["-c", filter])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn jq-jit");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    let out = child.wait_with_output().expect("wait failed");
    (
        String::from_utf8_lossy(&out.stdout).trim_end().to_string(),
        out.status.success(),
    )
}

#[test]
fn boolean_field_length_errors_in_object_construct() {
    for input in [r#"{"b":false}"#, r#"{"b":true}"#] {
        let (out, ok) = run("{a: (.b | length)}", input);
        assert!(!ok, "expected `{{a:.b|length}}` on `{input}` to error, got {out:?}");
        assert!(out.is_empty(), "expected no stdout for `{input}`, got {out:?}");
    }
}

#[test]
fn lengthable_field_types_still_fast_path() {
    let cases = [
        (r#"{"b":5}"#, r#"{"a":5}"#),
        (r#"{"b":-3}"#, r#"{"a":3}"#),
        (r#"{"b":null}"#, r#"{"a":0}"#),
        (r#"{"b":"hi"}"#, r#"{"a":2}"#),
        (r#"{"b":[1,2,3]}"#, r#"{"a":3}"#),
        (r#"{"b":{"x":1}}"#, r#"{"a":1}"#),
    ];
    for (input, expected) in cases {
        let (out, ok) = run("{a: (.b | length)}", input);
        assert!(ok, "expected success for `{input}`");
        assert_eq!(out, expected, "for input `{input}`");
    }
}
