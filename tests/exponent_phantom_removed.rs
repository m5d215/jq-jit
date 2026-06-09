//! Issue #959: `exponent/0` was a callable phantom — defined in jq-jit but
//! undefined in jq 1.8.1 (and absent from both `builtins` lists and the
//! documented jqx extension set). It shadowed a name jq rejects at compile
//! time, so a program valid in jq-jit was rejected by jq. Removed so jq-jit
//! matches jq's `exponent/0 is not defined` compile error; the IEEE binary
//! exponent stays reachable through the standard `logb` (which jq also has).
//!
//! This is a compile-time "not defined" error (exit 3), not a runtime error a
//! `try`/`catch` can capture, so shell out to the binary.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(filter: &str, stdin: &str) -> (String, String, Option<i32>) {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .args(["-c", filter])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
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
        String::from_utf8_lossy(&out.stderr).to_string(),
        out.status.code(),
    )
}

#[test]
fn exponent_is_not_defined() {
    let (stdout, stderr, code) = run("exponent", "8");
    assert!(stdout.is_empty(), "exponent must not produce output, got {stdout:?}");
    assert!(
        stderr.contains("exponent/0 is not defined"),
        "expected jq's compile error, got stderr: {stderr:?}"
    );
    assert_eq!(code, Some(3), "jq rejects undefined builtins with exit 3");
}

#[test]
fn logb_is_the_supported_equivalent() {
    // The IEEE binary exponent jq-jit's `exponent` returned is `logb` in jq.
    let (stdout, _stderr, code) = run("logb", "8");
    assert_eq!(stdout, "3");
    assert_eq!(code, Some(0));
}

#[test]
fn significand_still_defined() {
    // `significand/0` exists in both jq and jq-jit and must be unaffected.
    let (stdout, _stderr, code) = run("significand", "8");
    assert_eq!(stdout, "1");
    assert_eq!(code, Some(0));
}
