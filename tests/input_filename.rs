//! Issue #926: `input_filename` must report the source of the current input —
//! the file path for a named file argument, "<stdin>" for the stdin stream,
//! and `null` before any input has been consumed (e.g. `-n` mode before the
//! first `input`). It was previously hardcoded to the literal "<stdin>".
//!
//! These behaviours depend on file arguments and `-n` mode, which the 3-line
//! `regression.test` harness (single stdin document) cannot drive, so we shell
//! out and verify against the expectations captured from jq 1.8.1.

use std::io::Write;
use std::process::{Command, Stdio};

fn jq_jit() -> &'static str {
    env!("CARGO_BIN_EXE_jq-jit")
}

/// Run with explicit file arguments (no stdin).
fn run_files(args: &[&str]) -> String {
    let out = Command::new(jq_jit())
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("failed to spawn jq-jit");
    String::from_utf8(out.stdout).expect("non-utf8 stdout").trim_end().to_string()
}

/// Run feeding `stdin`, returning trimmed stdout.
fn run_stdin(args: &[&str], stdin: &str) -> String {
    let mut child = Command::new(jq_jit())
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn jq-jit");
    child.stdin.take().unwrap().write_all(stdin.as_bytes()).unwrap();
    let out = child.wait_with_output().expect("wait failed");
    String::from_utf8(out.stdout).expect("non-utf8 stdout").trim_end().to_string()
}

fn write_tmp(name: &str, content: &str) -> String {
    let mut path = std::env::temp_dir();
    path.push(format!("jqjit_ifn_{}_{}", std::process::id(), name));
    std::fs::write(&path, content).unwrap();
    path.to_str().unwrap().to_string()
}

#[test]
fn stdin_reports_stdin_literal() {
    assert_eq!(run_stdin(&["-c", "input_filename"], "1"), "\"<stdin>\"");
}

#[test]
fn null_input_before_read_is_null() {
    assert_eq!(run_stdin(&["-cn", "input_filename"], "null"), "null");
}

#[test]
fn single_file_reports_its_path() {
    let a = write_tmp("a", "1\n");
    assert_eq!(run_files(&["-c", "input_filename", &a]), format!("{:?}", a));
}

#[test]
fn two_files_report_their_respective_paths() {
    let a = write_tmp("two_a", "1\n");
    let b = write_tmp("two_b", "2\n");
    let expected = format!("{:?}\n{:?}", a, b);
    assert_eq!(run_files(&["-c", "input_filename", &a, &b]), expected);
}

#[test]
fn null_input_with_file_is_null_until_first_read() {
    let a = write_tmp("nfa", "1\n");
    assert_eq!(run_files(&["-cn", "input_filename", &a]), "null");
}

#[test]
fn null_input_filename_follows_first_read() {
    let a = write_tmp("nffa", "1\n");
    assert_eq!(run_files(&["-cn", "input | input_filename", &a]), format!("{:?}", a));
}

#[test]
fn input_filename_tracks_source_across_file_boundary() {
    let a = write_tmp("xb_a", "1\n");
    let b = write_tmp("xb_b", "2\n");
    // null (no read), then read 1 from a, then read 2 from b — the filename
    // tracks the source of the document the cursor last consumed.
    let expected = format!("[null,1,{:?},2,{:?}]", a, b);
    assert_eq!(
        run_files(&["-cn", "[input_filename, input, input_filename, input, input_filename]", &a, &b]),
        expected
    );
}

#[test]
fn multi_document_file_reports_same_path_each_time() {
    let m = write_tmp("multi", "1\n2\n3\n");
    let expected = format!("{0:?}\n{0:?}\n{0:?}", m);
    assert_eq!(run_files(&["-c", "input_filename", &m]), expected);
}

#[test]
fn slurp_two_files_reports_last_source() {
    // --slurp folds every file into one document; jq reports the LAST source.
    let a = write_tmp("slurp_a", "1\n");
    let b = write_tmp("slurp_b", "2\n");
    assert_eq!(run_files(&["-c", "input_filename", "--slurp", &a, &b]), format!("{:?}", b));
}

#[test]
fn dash_argument_reports_stdin() {
    assert_eq!(run_stdin(&["-c", "input_filename", "-"], "9"), "\"<stdin>\"");
}
