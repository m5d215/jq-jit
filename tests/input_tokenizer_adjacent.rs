//! Issue #854: the top-level input tokenizer must reject two bare-word/number
//! tokens jammed together with no separator (`nullnull`, `true123`, `123true`,
//! `false0`, `1-2`, `false-1`), the way jq does — jq-jit previously split them
//! into separate documents. Self-delimiting values (`{}{}`, `[][]`, `"a""b"`,
//! `"a"-1`, `[1]-2`) and whitespace-separated documents stay valid.
//!
//! The regression-test harness parses a single input per case, so a malformed
//! multi-token stream can't be expressed there — shell out instead.

use std::io::Write;
use std::process::{Command, Stdio};

fn run_with_args(args: &[&str], stdin: &str) -> (String, bool) {
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

fn run_inputs(stdin: &str) -> (String, bool) {
    run_with_args(&["-cn", "[inputs]"], stdin)
}

/// The default main-loop document reader (plain `jq .`) — a separate raw
/// streamer from the `inputs` builtin (#1000).
fn run_main_loop(stdin: &str) -> (String, bool) {
    run_with_args(&["-c", "."], stdin)
}

#[test]
fn adjacent_word_tokens_error() {
    for bad in [
        "nullnull", "truefalse", "true123", "123true", "false0", "1true", "1abc",
        "1-2", "1+2", "1.2.3", "1e2e3", "false-1", "null-1", "-1-2", "false.x",
    ] {
        let (_, ok) = run_inputs(bad);
        assert!(!ok, "expected `{bad}` to be rejected as adjacent tokens");
    }
}

#[test]
fn well_separated_and_structural_values_ok() {
    let cases = [
        ("null null", "[null,null]"),
        ("1 2", "[1,2]"),
        ("1\n2\n", "[1,2]"),
        (r#"{"a":1}{"b":2}"#, r#"[{"a":1},{"b":2}]"#),
        ("[1][2]", "[[1],[2]]"),
        (r#""a""b""#, r#"["a","b"]"#),
        (r#""a"-1"#, r#"["a",-1]"#),
        ("[1]-2", "[[1],-2]"),
        ("-1 -2", "[-1,-2]"),
    ];
    for (input, expected) in cases {
        let (out, ok) = run_inputs(input);
        assert!(ok, "expected `{input}` to parse");
        assert_eq!(out, expected, "for input `{input}`");
    }
}

/// #1000: the plain `jq .` main-loop reader (a different code path from
/// `inputs`) must also reject adjacent word/number tokens instead of splitting
/// them into separate documents.
#[test]
fn main_loop_rejects_adjacent_word_tokens() {
    for bad in [
        "nullnull", "truefalse", "true1", "1true", "false0", "false-1", "1-2", "1.2.3",
    ] {
        let (out, ok) = run_main_loop(bad);
        assert!(!ok, "expected main loop to reject `{bad}`, got {out:?}");
    }
}

#[test]
fn main_loop_keeps_structural_and_separated_values() {
    let cases = [
        ("{}{}", "{}\n{}"),
        ("[1][2]", "[1]\n[2]"),
        (r#""a""b""#, "\"a\"\n\"b\""),
        (r#""a"-1"#, "\"a\"\n-1"),
        ("[1]-2", "[1]\n-2"),
        ("1 2", "1\n2"),
        ("1\n2\n", "1\n2"),
    ];
    for (input, expected) in cases {
        let (out, ok) = run_main_loop(input);
        assert!(ok, "expected main loop to accept `{input}`");
        assert_eq!(out, expected, "for input `{input}`");
    }
}
