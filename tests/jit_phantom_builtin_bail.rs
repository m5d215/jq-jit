//! Issue #1082: `combinations` and `modf` flattened to a `CallBuiltin` JitOp
//! but `jit_rt_call_builtin` has no dispatch arm for them, so the JIT path
//! errored with `unknown builtin` at runtime. Default dispatch masked the bug
//! for small inputs (< 4KB routes to eval) but large inputs hit it.
//!
//! The fix makes the flattener bail on any `CallBuiltin` whose op has no
//! generic runtime dispatch arm (`RtBuiltin::from_builtin` returns `None`),
//! so the filter falls back to eval. These tests pin the behavior under
//! `--force-jit`, which routes everything flattenable through the JIT.

use std::io::Write;
use std::process::{Command, Stdio};

fn run_force_jit(filter: &str, stdin: &str) -> (String, String, Option<i32>) {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .args(["--force-jit", "-c", filter])
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
        String::from_utf8_lossy(&out.stderr).to_string(),
        out.status.code(),
    )
}

#[test]
fn combinations_collect_force_jit() {
    let (stdout, stderr, code) = run_force_jit("[combinations]", "[[1,2],[3,4]]");
    assert_eq!(stdout, "[[1,3],[1,4],[2,3],[2,4]]", "stderr: {stderr:?}");
    assert_eq!(code, Some(0));
}

#[test]
fn combinations_stream_force_jit() {
    let (stdout, stderr, code) = run_force_jit("combinations", "[[1,2],[3,4]]");
    assert_eq!(stdout, "[1,3]\n[1,4]\n[2,3]\n[2,4]", "stderr: {stderr:?}");
    assert_eq!(code, Some(0));
}

#[test]
fn combinations_n_force_jit() {
    let (stdout, stderr, code) = run_force_jit("[combinations(2)]", "[[1,2]]");
    assert_eq!(stdout, "[[[1,2],[1,2]]]", "stderr: {stderr:?}");
    assert_eq!(code, Some(0));
}

#[test]
fn modf_force_jit() {
    let (stdout, stderr, code) = run_force_jit("modf", "1.5");
    assert_eq!(stdout, "[0.5,1]", "stderr: {stderr:?}");
    assert_eq!(code, Some(0));
}

#[test]
fn modf_negative_tojson_force_jit() {
    let (stdout, stderr, code) = run_force_jit("modf | tojson", "-1");
    assert_eq!(stdout, "\"[-0,-1]\"", "stderr: {stderr:?}");
    assert_eq!(code, Some(0));
}
