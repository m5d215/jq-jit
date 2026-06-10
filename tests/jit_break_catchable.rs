//! Issue #1086: label/break unwinding diverged on the JIT path. jq
//! implements `break $label` as a special error value carrying `__jq`,
//! which `try/catch` intercepts and `?` suppresses (only the break itself
//! — sibling outputs keep running). The JIT lowered break as a direct jump
//! to the label end, bypassing any catch in between.
//!
//! Break now throws the `{"__jq": id}` object when a try sits between the
//! break and its label (depth-tracked at label registration); a label
//! inside the current try keeps the plain jump, so `def f: label $o |
//! (10, break $o, 20); try (f, 99) catch "c"` still yields 10, 99.

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

fn assert_jit(filter: &str, expected: &str) {
    for backend in ["JQJIT_FORCE_CRANELIFT", "JQJIT_FORCE_JITOP_INTERP"] {
        let (stdout, code) = run_backend(filter, "null", backend);
        assert_eq!(
            stdout, expected,
            "{backend}: filter {filter:?} gave {stdout:?}, want {expected:?}"
        );
        assert_eq!(code, Some(0), "{backend}: filter {filter:?} exited nonzero");
    }
}

#[test]
fn break_is_catchable() {
    assert_jit("label $out | try break $out catch \"c\"", "\"c\"");
    assert_jit(
        "[label $out | try break $out catch (type, has(\"__jq\"))]",
        "[\"object\",true]",
    );
    assert_jit(
        "[label $out | try (1, break $out, 2) catch \"c\"]",
        "[1,\"c\"]",
    );
}

#[test]
fn optional_suppresses_only_the_break() {
    assert_jit("[label $out | (break $out)?, 1]", "[1]");
}

#[test]
fn caught_break_does_not_exit_enclosing_loops() {
    assert_jit(
        "[label $out | range(5) | try (if .==2 then break $out else . end) catch \"x\"]",
        "[0,1,\"x\",3,4]",
    );
}

#[test]
fn label_inside_try_still_unwinds_by_jump() {
    // The label sits inside the try: the break unwinds to it normally and
    // is NOT intercepted by the outer catch.
    assert_jit(
        "def f: label $o | (10, break $o, 20); [try (f, 99) catch \"c\"]",
        "[10,99]",
    );
    assert_jit(
        "[label $a | try (label $b | (1, break $a, 2)) catch \"c\"]",
        "[1,\"c\"]",
    );
}

#[test]
fn plain_break_unchanged() {
    assert_jit("[label $out | 1, break $out, 2]", "[1]");
}
