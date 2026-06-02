//! Issue #850: the top-level `@json`/`tojson` raw-byte fast path
//! (`push_tojson_raw`) used to copy the input document's bytes verbatim,
//! leaking `\uXXXX` escapes that jq normalizes on the real parse path.
//!
//! A lone *high* surrogate (`"\ud83d"`) is a parse error in jq (exit 5 with no
//! stdout), so it can't be expressed in the 3-line regression-test format — it
//! is pinned here instead. The value-producing cases (printable decode, lone
//! low surrogate -> U+FFFD, valid pair) live in `tests/regression.test`.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(filter: &str, stdin: &str) -> (String, Option<i32>) {
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
        out.status.code(),
    )
}

#[test]
fn lone_high_surrogate_is_a_parse_error() {
    // `"\ud83d"` — a JSON string with a lone high surrogate escape.
    for filter in ["@json", "tojson"] {
        let (out, code) = run(filter, "\"\\ud83d\"");
        assert!(out.is_empty(), "{filter}: expected no stdout, got {out:?}");
        assert_eq!(code, Some(5), "{filter}: expected exit 5 (parse error)");
    }
}

#[test]
fn lone_high_surrogate_in_nested_object_is_a_parse_error() {
    let (out, code) = run("@json", "{\"k\":\"\\ud83d\"}");
    assert!(out.is_empty());
    assert_eq!(code, Some(5));
}

#[test]
fn value_producing_escapes_match_jq() {
    // Mirror of the regression-test cases, asserted here too for locality.
    let cases = [
        ("@json", "\"\\u0041\"", "\"\\\"A\\\"\""),
        ("@json", "\"\\udc00\"", "\"\\\"\u{fffd}\\\"\""),
        ("tojson", "\"\\ud83d\\ude00\"", "\"\\\"\u{1f600}\\\"\""),
    ];
    for (filter, input, expected) in cases {
        let (out, code) = run(filter, input);
        assert_eq!(out, expected, "{filter} on {input}");
        assert_eq!(code, Some(0));
    }
}

#[test]
fn escape_free_input_is_unchanged() {
    let (out, code) = run("tojson", "{\"x\":1,\"y\":\"hi\"}");
    assert_eq!(out, "\"{\\\"x\\\":1,\\\"y\\\":\\\"hi\\\"}\"");
    assert_eq!(code, Some(0));
}
