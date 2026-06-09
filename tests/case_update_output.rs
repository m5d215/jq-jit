//! Issue #996: the top-level `.field |= ascii_upcase` / `ascii_downcase` raw
//! fast path produced byte-incorrect stdout that the line-normalizing
//! regression harness can't see:
//!
//!   1. On a non-string field value it leaked the partial, unterminated object
//!      prefix (`{"a":5`) to stdout before erroring (the value-side helper
//!      appends bytes as it scans, then bails when the value isn't a string).
//!   2. On every successful record it emitted a stray blank line — the helper
//!      terminated with `}\n` while the compact apply arm pushed its own `\n`.
//!
//! These assert the exact stdout bytes, so they catch both regressions.

use std::io::Write;
use std::process::{Command, Stdio};

fn run_bytes(filter: &str, stdin: &str) -> (Vec<u8>, bool) {
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
    (out.stdout, out.status.success())
}

#[test]
fn successful_update_has_no_stray_blank_line() {
    let (out, ok) = run_bytes(".a |= ascii_upcase", r#"{"a":"hi"}"#);
    assert!(ok);
    assert_eq!(out, b"{\"a\":\"HI\"}\n", "exactly one trailing newline");

    let (out, ok) = run_bytes(".a |= ascii_downcase", r#"{"a":"HELLO"}"#);
    assert!(ok);
    assert_eq!(out, b"{\"a\":\"hello\"}\n");
}

#[test]
fn nonstring_value_leaks_nothing_to_stdout() {
    for input in [r#"{"a":5}"#, r#"{"a":[1]}"#, r#"{"a":{}}"#, r#"{"a":null}"#, r#"{"a":true}"#] {
        let (out, ok) = run_bytes(".a |= ascii_upcase", input);
        assert!(!ok, "expected error on `{input}`");
        assert!(
            out.is_empty(),
            "expected no stdout for `{input}`, got {:?}",
            String::from_utf8_lossy(&out)
        );
    }
}

#[test]
fn multi_record_stream_no_blank_lines_and_no_leak() {
    // First and third records succeed; the middle errors. jq writes only the
    // two valid objects (the stream continues past the error), each on its own
    // line with no blank lines and no partial prefix from the error record.
    let (out, _ok) = run_bytes(".a |= ascii_upcase", "{\"a\":\"hi\"}\n{\"a\":5}\n{\"a\":\"yo\"}");
    assert_eq!(out, b"{\"a\":\"HI\"}\n{\"a\":\"YO\"}\n");
}
