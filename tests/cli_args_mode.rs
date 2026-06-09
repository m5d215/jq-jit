//! Issue #957: `--args` / `--jsonargs` are *mode* flags, not terminal
//! separators. After one, jq keeps parsing option flags (tokens starting with
//! `-`) and only collects non-flag tokens as positional arguments; a bare `--`
//! ends option parsing entirely. jq-jit previously swallowed everything after
//! `--args` into `$ARGS.positional` and stopped parsing options.
//!
//! These exercise CLI argument parsing, which the 3-line regression harness
//! cannot express, so shell out to the binary.

use std::process::Command;

fn run(args: &[&str]) -> (String, bool) {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let out = Command::new(jq_jit)
        .args(args)
        .output()
        .expect("failed to spawn jq-jit");
    (
        String::from_utf8_lossy(&out.stdout).trim_end().to_string(),
        out.status.success(),
    )
}

#[test]
fn args_keeps_parsing_option_flags() {
    // `--arg a 1` after `--args x y` is still parsed as a named arg.
    let (out, ok) = run(&["-cn", "$ARGS", "--args", "x", "y", "--arg", "a", "1"]);
    assert!(ok);
    assert_eq!(out, r#"{"positional":["x","y"],"named":{"a":"1"}}"#);
}

#[test]
fn args_consumes_n_flag_not_as_positional() {
    let (out, ok) = run(&["-cn", "$ARGS.positional", "--args", "a", "-n", "b"]);
    assert!(ok);
    assert_eq!(out, r#"["a","b"]"#);
}

#[test]
fn args_unknown_flag_errors() {
    // A stray `-y` in args mode is an unknown option, not a positional.
    let (_out, ok) = run(&["-cn", "$ARGS.positional", "--args", "x", "-y"]);
    assert!(!ok, "stray -y in args mode must error");
}

#[test]
fn args_double_dash_ends_option_parsing() {
    // After `--`, `-n` is a positional, not the null-input flag.
    let (out, ok) = run(&["-cn", "$ARGS.positional", "--args", "--", "a", "-n"]);
    assert!(ok);
    assert_eq!(out, r#"["a","-n"]"#);
}

#[test]
fn jsonargs_consumes_n_flag() {
    let (out, ok) = run(&["-cn", "$ARGS.positional", "--jsonargs", "1", "-n", "2"]);
    assert!(ok);
    assert_eq!(out, "[1,2]");
}

#[test]
fn jsonargs_keeps_parsing_named_args() {
    let (out, ok) = run(&["-cn", "$ARGS", "--jsonargs", "1", "2", "--argjson", "z", "3"]);
    assert!(ok);
    assert_eq!(out, r#"{"positional":[1,2],"named":{"z":3}}"#);
}

#[test]
fn first_operand_in_args_mode_is_the_filter() {
    // With no filter before `--args`, the first non-option token fills the
    // filter slot; subsequent tokens are positionals. `.x` is the program.
    let (out, ok) = run(&["-cn", "--args", "$ARGS.positional", "extra"]);
    assert!(ok);
    assert_eq!(out, r#"["extra"]"#);
}

#[test]
fn bare_double_dash_is_end_of_options_without_args_mode() {
    // `--` with no args mode just ends option parsing; remaining tokens are
    // the filter / input files, not an "Unknown option" error.
    let (out, ok) = run(&["-cn", "--", "1+1"]);
    assert!(ok);
    assert_eq!(out, "2");
}
