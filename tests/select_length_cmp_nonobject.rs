//! Issue #998: the fused `select(.field | length CMP const)` raw fast path
//! must not mask the index type error on non-object inputs. jq raises
//! `Cannot index <type> with string` when `.field` is applied to a
//! non-object; jq-jit's raw scanner silently emitted nothing (exit 0).
//!
//! The regression-test harness only compares stdout (an error case and the
//! pre-fix silent-drop both produce empty stdout), so it can't distinguish
//! the masking — assert the process exit status here instead.

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
fn nonobject_input_raises_index_error() {
    // Every comparison operator and every non-object input must error,
    // matching jq's `Cannot index <type> with string`.
    for op in [">", "<", "==", ">=", "<=", "!="] {
        let filter = format!("select(.a | length {op} 0)");
        for input in ["5", "\"s\"", "true", "[1]"] {
            let (out, ok) = run(&filter, input);
            assert!(
                !ok,
                "expected `{filter}` on `{input}` to error, got success with {out:?}"
            );
            assert!(out.is_empty(), "expected no stdout for `{filter}` on `{input}`");
        }
    }
}

#[test]
fn null_input_does_not_error() {
    // `null | .a` is `null` in jq, so `null | length` is 0 and the select
    // simply produces nothing — it must not raise a type error.
    let (out, ok) = run("select(.a | length > 0)", "null");
    assert!(ok, "expected null input to succeed, got error");
    assert!(out.is_empty(), "expected no output for null input, got {out:?}");
}

#[test]
fn object_missing_field_compares_against_zero() {
    // An object missing the field has `.field == null`, `null | length == 0`,
    // so `== 0` passes and the whole object is emitted.
    let (out, ok) = run("select(.a | length == 0)", r#"{"b":1}"#);
    assert!(ok, "expected success");
    assert_eq!(out, r#"{"b":1}"#);
}

#[test]
fn string_field_fast_path_still_works() {
    let (out, ok) = run("select(.a | length > 3)", r#"{"a":"hello"}"#);
    assert!(ok);
    assert_eq!(out, r#"{"a":"hello"}"#);
    let (out, ok) = run("select(.a | length > 3)", r#"{"a":"hi"}"#);
    assert!(ok);
    assert!(out.is_empty());
}
