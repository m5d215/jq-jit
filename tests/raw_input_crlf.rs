//! Issue #889: `-R` (raw input) must split on `\n` only and keep a trailing
//! `\r` as line content (a CRLF file yields `"a\r"`), matching jq. Rust's
//! `lines()` strips the CR. The regression-test harness can't express the `-R`
//! flag, so shell out directly and assert exact stdout (CR included).

use std::io::Write;
use std::process::{Command, Stdio};

fn run(args: &[&str], stdin: &[u8]) -> String {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn jq-jit");
    child.stdin.take().unwrap().write_all(stdin).unwrap();
    let out = child.wait_with_output().expect("wait failed");
    String::from_utf8(out.stdout)
        .expect("non-utf8 stdout")
        .trim_end_matches('\n')
        .to_string()
}

#[test]
fn crlf_line_keeps_trailing_cr() {
    // "a\r\n" -> one line "a\r" (jq escapes the CR as \r in JSON output).
    assert_eq!(run(&["-Rc", "."], b"a\r\n"), r#""a\r""#);
}

#[test]
fn crlf_line_length_counts_the_cr() {
    assert_eq!(run(&["-Rc", ".|length"], b"a\r\n"), "2");
}

#[test]
fn multiple_crlf_lines_each_keep_cr() {
    assert_eq!(
        run(&["-Rc", "."], b"a\r\nb\r\n"),
        "\"a\\r\"\n\"b\\r\""
    );
}

#[test]
fn lone_cr_mid_line_is_preserved() {
    assert_eq!(run(&["-Rc", "."], b"a\rb\r\n"), r#""a\rb\r""#);
}

#[test]
fn lf_only_lines_are_unaffected() {
    assert_eq!(run(&["-Rc", "."], b"a\nb"), "\"a\"\n\"b\"");
}

#[test]
fn trailing_newline_yields_no_empty_final_line() {
    assert_eq!(run(&["-Rc", "."], b"a\n"), "\"a\"");
}

#[test]
fn empty_input_yields_no_lines() {
    assert_eq!(run(&["-Rc", "."], b""), "");
}

#[test]
fn bare_newline_yields_one_empty_line() {
    assert_eq!(run(&["-Rc", "."], b"\n"), "\"\"");
}

#[test]
fn slurp_raw_keeps_cr_unchanged() {
    // -Rs reads the whole input as one string; the CR was already preserved.
    assert_eq!(run(&["-Rsc", ".|length"], b"a\r\nb"), "4");
}
