//! Issue #853: `input`/`inputs` used inside a path-expression argument
//! (`getpath([input])`, `setpath([input];v)`, `delpaths([[input]])`,
//! `path(input|...)`) — or hidden behind a user-defined function — must seed
//! the input queue. `uses_inputs()` previously didn't descend into the
//! path-expression forms (or `FuncCall` bodies), so the binary skipped queue
//! setup and `input` raised a bogus `break`.
//!
//! These need a multi-document stream, which the 3-line regression harness
//! (one input per case) can't express, so shell out to the binary.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(args: &[&str], stdin: &str) -> (String, bool) {
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
        out.status.success(),
    )
}

#[test]
fn input_inside_getpath_arg() {
    let (out, ok) = run(&["-c", "getpath([input])"], r#"{"a":5} "a""#);
    assert!(ok, "getpath([input]) should not error");
    assert_eq!(out, "5");
}

#[test]
fn input_as_literal_path_array() {
    let (out, ok) = run(&["-c", "getpath(input)"], r#"{"a":5} ["a"]"#);
    assert!(ok);
    assert_eq!(out, "5");
}

#[test]
fn input_inside_setpath_arg() {
    let (out, ok) = run(&["-c", "setpath([input];9)"], r#"{"a":5} "a""#);
    assert!(ok);
    assert_eq!(out, r#"{"a":9}"#);
}

#[test]
fn input_inside_delpaths_arg() {
    let (out, ok) = run(&["-c", "delpaths([[input]])"], r#"{"a":5} "a""#);
    assert!(ok);
    assert_eq!(out, "{}");
}

#[test]
fn input_hidden_behind_user_def() {
    // The stream read is inside a `def` body — uses_inputs() must follow the call.
    let (out, ok) = run(&["-c", "def f: input; f"], "1 2");
    assert!(ok);
    assert_eq!(out, "2");
    let (out, ok) = run(&["-c", "def g: getpath([input]); g"], r#"{"a":5} "a""#);
    assert!(ok);
    assert_eq!(out, "5");
}

#[test]
fn inputs_plural_in_path_arg_still_ok() {
    // Regression guard: the plural form already worked; keep it working.
    let (out, ok) = run(
        &["-c", "{p:getpath([input]),rest:[inputs]}"],
        r#"{"a":5} "a" "b""#,
    );
    assert!(ok);
    assert_eq!(out, r#"{"p":5,"rest":["b"]}"#);
}
