//! Issue #1088: CallBuiltin argument generators were evaluated eagerly and
//! in the wrong cartesian order on the JIT path.
//!
//! - The generator-arg rewrite nested the LetBindings reversed, so
//!   `[pow(10,20; 1,2)]` produced [10,100,20,400] instead of eval's
//!   first-argument-fastest [10,20,100,400].
//! - flatten_gen_with_each_output's generic fallback collected EVERY
//!   output before iterating, defeating short-circuiting consumers:
//!   `IN(2, error("x"); 2)` desugars to any((2,error("x")) == 2; .) and
//!   must stop at the first match without evaluating the error. BinOps
//!   with one generator operand now stream per item.

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
fn cartesian_order_matches_eval() {
    assert_jit("[pow(10,20; 1,2)]", "null", "[10,20,100,400]");
    assert_jit("[ldexp(1,3; 0,4)]", "null", "[1,3,16,48]");
    assert_jit(
        "[fma(1,2; 10,20; 100,200)]",
        "null",
        "[110,120,120,140,210,220,220,240]",
    );
    assert_jit("[pow(2; 1,2,3)]", "null", "[2,4,8]");
}

#[test]
fn in_short_circuits_past_errors() {
    assert_jit("IN(2, error(\"x\"); 2)", "null", "true");
    assert_jit(
        "[.[] | select(.x | IN(1, error; 1))]",
        "[{\"x\":1},{\"x\":9}]",
        "[{\"x\":1},{\"x\":9}]",
    );
}

#[test]
fn in_plain_membership_unchanged() {
    assert_jit("IN(1,2,3; 2)", "null", "true");
    assert_jit("IN(1; 2)", "null", "false");
}
