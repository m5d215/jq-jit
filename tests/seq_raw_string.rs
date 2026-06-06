//! Issue #890: under `--seq`, jq omits the RS (0x1e) record-separator prefix for
//! a RAW STRING output (`-r` / `-j` applied to a string), since that bypasses the
//! JSON dumper. Numbers/bools/arrays/objects (JSON-encoded even under `-r`) and
//! default JSON output keep the RS. The regression-test harness can't express
//! `--seq`, so shell out and assert exact stdout bytes.

use std::io::Write;
use std::process::{Command, Stdio};

/// Run jq-jit with the given args, feeding `stdin`, returning raw stdout bytes.
fn run(args: &[&str], stdin: &[u8]) -> Vec<u8> {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn jq-jit");
    child.stdin.take().unwrap().write_all(stdin).unwrap();
    child.wait_with_output().expect("wait failed").stdout
}

// RS-framed input so the --seq parser is happy.
const IN_STR: &[u8] = b"\x1e\"a\"\n";
const IN_NUM: &[u8] = b"\x1e5\n";

#[test]
fn raw_string_output_omits_rs() {
    // -j on a string: just the bytes, no RS.
    assert_eq!(run(&["--seq", "-j", "."], IN_STR), b"a");
    // -r on a string: bytes + newline, no RS.
    assert_eq!(run(&["--seq", "-r", "."], IN_STR), b"a\n");
}

#[test]
fn raw_number_output_keeps_rs() {
    // A number is JSON-encoded even under -r, so the RS stays.
    assert_eq!(run(&["--seq", "-r", "."], IN_NUM), b"\x1e5\n");
}

#[test]
fn default_json_string_keeps_rs() {
    // No -r/-j: default JSON output keeps the RS.
    assert_eq!(run(&["--seq", "-c", "."], IN_STR), b"\x1e\"a\"\n");
}

#[test]
fn multiple_raw_strings_have_no_separators() {
    assert_eq!(
        run(&["--seq", "-j", "."], b"\x1e\"a\"\n\x1e\"b\"\n"),
        b"ab"
    );
}
