//! #1034: the CLI's uncaught-error report is built from the typed
//! `ErrorValue` signal (no sentinel-string parsing). Pin jq 1.8.1's format:
//! a string payload prints bare, any other payload gets the
//! ` (not a string)` marker with its JSON form, and a plain message error
//! prints its display text — identically across every execution backend.
//!
//! `regression.test` / `corpus.test` can't cover this: both harnesses
//! compare stdout, while the error report goes to stderr. Shell out and
//! inspect the raw stderr line instead.

use std::io::Write;
use std::process::{Command, Stdio};

const KNOBS: [&[(&str, &str)]; 4] = [
    &[],
    &[("JQJIT_FORCE_INTERPRETER", "1")],
    &[("JQJIT_FORCE_JITOP_INTERP", "1")],
    &[("JQJIT_FORCE_CRANELIFT", "1")],
];

fn first_stderr_line(filter: &str, input: &str, envs: &[(&str, &str)]) -> String {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut cmd = Command::new(jq_jit);
    cmd.arg("-c").arg(filter);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
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
    String::from_utf8_lossy(&out.stderr)
        .lines()
        .next()
        .unwrap_or_default()
        .to_string()
}

fn assert_all_backends(filter: &str, input: &str, expected: &str) {
    for envs in KNOBS {
        assert_eq!(
            first_stderr_line(filter, input, envs),
            expected,
            "filter={filter:?} envs={envs:?}"
        );
    }
}

#[test]
fn string_payload_prints_bare() {
    assert_all_backends(
        r#"error("boom")"#,
        "1\n",
        "jq: error (at <stdin>:1): boom",
    );
}

#[test]
fn non_string_payload_gets_marker_with_json() {
    assert_all_backends(
        r#"error({"a":1})"#,
        "1\n",
        r#"jq: error (at <stdin>:1) (not a string): {"a":1}"#,
    );
    assert_all_backends(
        "error(null)",
        "1\n",
        "jq: error (at <stdin>:1) (not a string): null",
    );
}

#[test]
fn bare_error_rethrows_the_input() {
    assert_all_backends(
        "error",
        "{\"x\":1}\n",
        r#"jq: error (at <stdin>:1) (not a string): {"x":1}"#,
    );
}

#[test]
fn plain_message_error_prints_display_text() {
    assert_all_backends(
        r#"1 + "x""#,
        "1\n",
        r#"jq: error (at <stdin>:1): number (1) and string ("x") cannot be added"#,
    );
}
