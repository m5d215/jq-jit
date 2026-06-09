//! Issue #980: under `--raw-output0`, jq refuses to emit a string whose content
//! contains an embedded NUL (NUL is the output record separator there) and
//! errors `Cannot dump a string containing NUL with --raw-output0 option`
//! (exit 5). jq-jit previously wrote the NUL bytes through with exit 0.
//!
//! The guard is specific to `--raw-output0`: plain `-r` and `-j` (without
//! `--raw-output0`) must still pass NUL through. `regression.test` /
//! `corpus.test` can't cover this (CLI flag + exit-code + stderr), so shell
//! out and inspect the raw bytes/status instead.

use std::io::Write;
use std::process::{Command, Stdio};

struct Run {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    code: i32,
}

fn run(args: &[&str], input: &str) -> Run {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
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
    Run {
        stdout: out.stdout,
        stderr: out.stderr,
        code: out.status.code().unwrap_or(-1),
    }
}

// `"AA==" | @base64d` decodes to a single NUL byte.
const NUL_FILTER: &str = "\"AA==\" | @base64d";

#[test]
fn raw_output0_rejects_embedded_nul() {
    let r = run(&["--raw-output0", NUL_FILTER], "null");
    assert_eq!(r.code, 5, "should exit 5 like jq");
    assert!(r.stdout.is_empty(), "no NUL bytes should be written");
    assert_eq!(
        String::from_utf8_lossy(&r.stderr).trim_end(),
        "jq: error (at <stdin>:0): Cannot dump a string containing NUL with --raw-output0 option"
    );
}

#[test]
fn raw_output0_join_output_also_rejects_nul() {
    // `-j --raw-output0` still applies the NUL-content guard in jq.
    let r = run(&["-j", "--raw-output0", NUL_FILTER], "null");
    assert_eq!(r.code, 5);
    assert!(r.stdout.is_empty());
}

#[test]
fn raw_output0_passes_non_nul_string() {
    let r = run(&["--raw-output0", "\"hello\""], "null");
    assert_eq!(r.code, 0);
    assert_eq!(r.stdout, b"hello\0");
}

#[test]
fn plain_raw_output_passes_nul_through() {
    // `-r` alone has no NUL guard: the byte goes through, then a newline.
    let r = run(&["-r", NUL_FILTER], "null");
    assert_eq!(r.code, 0);
    assert_eq!(r.stdout, b"\0\n");
}

#[test]
fn join_output_passes_nul_through() {
    // `-j` alone (no --raw-output0) passes NUL through with no separator.
    let r = run(&["-j", NUL_FILTER], "null");
    assert_eq!(r.code, 0);
    assert_eq!(r.stdout, b"\0");
}
