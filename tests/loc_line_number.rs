//! Issue #778: `$__loc__` must report the actual source line of the token,
//! counting newlines in the program text — jq does, jq-jit previously always
//! returned `line: 1` because the lexer never tracked the source line.
//!
//! `regression.test` / `corpus.test` cannot cover this: both harnesses are
//! line-based (one line per filter), so a multi-line program can't be
//! expressed. Shell out with multi-line program strings instead.

use std::process::Command;

fn run(filter: &str) -> String {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let out = Command::new(jq_jit)
        .arg("-nc")
        .arg(filter)
        .output()
        .expect("failed to spawn jq-jit");
    assert!(
        out.status.success(),
        "jq-jit -nc {filter:?} exited with {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .expect("non-utf8 stdout")
        .trim_end()
        .to_string()
}

#[test]
fn loc_reports_line_one_at_program_start() {
    assert_eq!(run("$__loc__"), r#"{"file":"<top-level>","line":1}"#);
}

#[test]
fn loc_counts_leading_newlines() {
    assert_eq!(run("\n$__loc__"), r#"{"file":"<top-level>","line":2}"#);
    assert_eq!(run("\n\n\n$__loc__"), r#"{"file":"<top-level>","line":4}"#);
}

#[test]
fn loc_counts_newlines_through_pipes() {
    assert_eq!(run("1\n|2\n|$__loc__"), r#"{"file":"<top-level>","line":3}"#);
}

#[test]
fn loc_counts_newlines_past_comments() {
    assert_eq!(
        run("1 |\n# comment\n$__loc__"),
        r#"{"file":"<top-level>","line":3}"#
    );
}

#[test]
fn loc_in_object_shorthand_tracks_line() {
    assert_eq!(
        run("\n{x: $__loc__}"),
        r#"{"x":{"file":"<top-level>","line":2}}"#
    );
}
