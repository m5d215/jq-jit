//! Issue #1087: the fused `FieldCmpNum` JitOp treated a non-numeric field
//! as a comparison miss, dropping jq's sort total order (null < false <
//! true < numbers < strings < arrays < objects) — `select(.x > 3)` on
//! `{"x":"hi"}` produced no output where jq keeps the object. The helper
//! now falls back to compare_values for non-numeric fields; the numeric
//! fast arm is unchanged.

use std::io::Write;
use std::process::{Command, Stdio};

fn run_backend(filter: &str, stdin: &str, backend_env: &str) -> (String, Option<i32>) {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .args(["-c", filter])
        .env(backend_env, "1")
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
        out.status.code(),
    )
}

fn assert_jit(filter: &str, stdin: &str, expected: &str) {
    for backend in ["JQJIT_FORCE_CRANELIFT", "JQJIT_FORCE_JITOP_INTERP"] {
        let (stdout, code) = run_backend(filter, stdin, backend);
        assert_eq!(
            stdout, expected,
            "{backend}: filter {filter:?} on {stdin:?} gave {stdout:?}, want {expected:?}"
        );
        assert_eq!(code, Some(0), "{backend}: filter {filter:?} exited nonzero");
    }
}

#[test]
fn string_field_ranks_above_numbers() {
    assert_jit("select(.x > 3)", "{\"x\":\"hi\"}", "{\"x\":\"hi\"}");
    assert_jit("select(.x < 3)", "{\"x\":\"hi\"}", "");
    assert_jit("if .x < 3 then \"lo\" else \"hi\" end", "{\"x\":\"s\"}", "\"hi\"");
}

#[test]
fn booleans_rank_below_numbers() {
    assert_jit("select(.x > 3)", "{\"x\":true}", "");
    assert_jit("select(.x <= 3)", "{\"x\":false}", "{\"x\":false}");
}

#[test]
fn containers_rank_above_numbers() {
    assert_jit("select(.x >= 3)", "{\"x\":{\"a\":1}}", "{\"x\":{\"a\":1}}");
    assert_jit("select(.x != 3)", "{\"x\":[1]}", "{\"x\":[1]}");
    assert_jit("select(.x == 3)", "{\"x\":\"hi\"}", "");
}

#[test]
fn numeric_fast_arm_unchanged() {
    assert_jit("select(.x > 3)", "{\"x\":5}", "{\"x\":5}");
    assert_jit("select(.x > 3)", "{\"x\":1}", "");
    assert_jit("select(.x > 3)", "{}", "");
}
