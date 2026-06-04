//! Issue #856: jq parses the input document stream lazily and emits each valid
//! document's result to stdout *before* it reaches a later malformed token,
//! then exits 5. jq-jit previously buffered the whole stream's output and
//! discarded it (emitting nothing) when any later token failed to parse,
//! exiting 2.
//!
//! The regression-test harness parses a single input per case, so a stream that
//! is valid up front and malformed later can't be expressed there — shell out
//! instead.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(args: &[&str], stdin: &str) -> (String, Option<i32>) {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .args(args)
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
fn leading_documents_flush_before_parse_error() {
    // Identity over a stream that goes bad after two valid documents.
    let (out, code) = run(&["-c", "."], "1 2 xx");
    assert_eq!(out, "1\n2", "valid leading documents must reach stdout");
    assert_eq!(code, Some(5), "jq exits 5 on a stream parse error");
}

#[test]
fn fast_path_filter_flushes_leading_documents() {
    // A detect_* fast path (field alternative) must flush the same way.
    let (out, code) = run(&["-c", ".a // .b"], "{\"a\":1}\n{\"b\":2}\ngarbage");
    assert_eq!(out, "1\n2");
    assert_eq!(code, Some(5));
}

#[test]
fn pretty_output_flushes_leading_documents() {
    // Non-compact (pretty) output path takes the same exit/flush route.
    let (out, code) = run(&["."], "1 2 xx");
    assert_eq!(out, "1\n2");
    assert_eq!(code, Some(5));
}

#[test]
fn clean_stream_still_succeeds() {
    let (out, code) = run(&["-c", "."], "1 2 3");
    assert_eq!(out, "1\n2\n3");
    assert_eq!(code, Some(0));
}
