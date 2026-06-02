//! Issue #845: `halt_error` with a non-string payload writes the JSON form to
//! stderr followed by a trailing newline (a string payload is written verbatim
//! with no newline; a `null` payload writes nothing). jq-jit previously omitted
//! the newline for the JSON case.
//!
//! `regression.test` / `corpus.test` can't cover this: both harnesses compare
//! stdout, while `halt_error` writes to stderr. Shell out and inspect the raw
//! stderr bytes instead.

use std::io::Write;
use std::process::{Command, Stdio};

fn halt_error_stderr(input: &str, filter: &str) -> Vec<u8> {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .arg(filter)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn jq-jit");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let out = child.wait_with_output().expect("wait failed");
    out.stderr
}

#[test]
fn non_string_payload_gets_trailing_newline() {
    assert_eq!(halt_error_stderr("5", "halt_error(0)"), b"5\n");
    assert_eq!(halt_error_stderr("true", "halt_error(0)"), b"true\n");
    assert_eq!(halt_error_stderr("[1,2]", "halt_error(0)"), b"[1,2]\n");
    assert_eq!(
        halt_error_stderr("{\"m\":\"x\"}", "halt_error(0)"),
        b"{\"m\":\"x\"}\n"
    );
}

#[test]
fn string_payload_written_verbatim_without_newline() {
    assert_eq!(halt_error_stderr("\"oops\"", "halt_error(0)"), b"oops");
}

#[test]
fn null_payload_writes_nothing() {
    assert_eq!(halt_error_stderr("null", "halt_error(0)"), b"");
}

#[test]
fn halt_error_zero_arity_matches() {
    assert_eq!(halt_error_stderr("5", "halt_error"), b"5\n");
}
