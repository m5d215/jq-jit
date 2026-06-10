//! Issue #1043: `halt` / `halt_error` must work on every dispatch engine and
//! produce jq's exit codes, and halt must propagate past `try ... catch`
//! (#182) instead of being bound as a catch value.
//!
//! `regression.test` / `corpus.test` can't cover this: both harnesses compare
//! stdout only and tolerate nonzero exits, so wrong exit codes and the
//! try/catch propagation are invisible. Shell out and assert the full
//! (stdout, stderr, exit code) triple instead, driving both the stdin path
//! (eval dispatch) and the `-n` path (JIT dispatch via runtime::call_builtin).

use std::io::Write;
use std::process::{Command, Stdio};

struct Run {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    code: i32,
}

fn run_stdin(input: &str, filter: &str) -> Run {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .arg(filter)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
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
    Run {
        stdout: out.stdout,
        stderr: out.stderr,
        code: out.status.code().expect("no exit code"),
    }
}

fn run_null_input(filter: &str) -> Run {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let out = Command::new(jq_jit)
        .arg("-n")
        .arg(filter)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("failed to spawn jq-jit");
    Run {
        stdout: out.stdout,
        stderr: out.stderr,
        code: out.status.code().expect("no exit code"),
    }
}

/// Both dispatch paths for the same program: stdin-driven (eval) and `-n`
/// (JIT). The stdin path feeds `null` so the two see the same input.
fn both_paths(filter: &str) -> [(&'static str, Run); 2] {
    [
        ("stdin", run_stdin("null", filter)),
        ("-n", run_null_input(filter)),
    ]
}

fn assert_run(label: &str, filter: &str, r: &Run, stdout: &[u8], stderr: &[u8], code: i32) {
    assert_eq!(
        (r.stdout.as_slice(), r.stderr.as_slice(), r.code),
        (stdout, stderr, code),
        "[{}] {}",
        label,
        filter
    );
}

#[test]
fn halt_exits_zero_silently() {
    for (label, r) in both_paths("halt") {
        assert_run(label, "halt", &r, b"", b"", 0);
    }
}

#[test]
fn halt_after_yield_keeps_prior_output() {
    for (label, r) in both_paths("1,halt,2") {
        assert_run(label, "1,halt,2", &r, b"1\n", b"", 0);
    }
}

#[test]
fn halt_propagates_past_try_catch() {
    for f in ["try halt catch .", "[try halt catch .]", "(try halt catch .) // 9"] {
        for (label, r) in both_paths(f) {
            assert_run(label, f, &r, b"", b"", 0);
        }
    }
}

#[test]
fn halt_error_zero_arity_exits_five_with_payload() {
    for (label, r) in both_paths("\"x\" | halt_error") {
        assert_run(label, "halt_error/0", &r, b"", b"x", 5);
    }
}

#[test]
fn halt_error_code_seven() {
    for (label, r) in both_paths("\"x\" | halt_error(7)") {
        assert_run(label, "halt_error(7)", &r, b"", b"x", 7);
    }
}

#[test]
fn halt_error_negative_code_clamps_to_zero() {
    // jq clamps negative halt_error codes to 0 (#979) while still emitting
    // the stderr payload.
    for (label, r) in both_paths("\"x\" | halt_error(-3)") {
        assert_run(label, "halt_error(-3)", &r, b"", b"x", 0);
    }
}

#[test]
fn halt_error_propagates_past_try_catch() {
    for (label, r) in both_paths("try (\"x\" | halt_error(7)) catch .") {
        assert_run(label, "try halt_error(7) catch .", &r, b"", b"x", 7);
    }
}

#[test]
fn halt_error_non_number_code_is_catchable_error() {
    // A non-number code is an ordinary (catchable) error, not a halt: jq
    // reports `string ("x") halt_error/1: number required` and exits 5.
    for (label, r) in both_paths("\"x\" | halt_error(\"s\")") {
        assert_eq!(r.stdout, b"", "[{}] stdout", label);
        assert_eq!(r.code, 5, "[{}] exit code", label);
        let stderr = String::from_utf8_lossy(&r.stderr);
        assert!(
            stderr.contains("string (\"x\") halt_error/1: number required"),
            "[{}] stderr: {}",
            label,
            stderr
        );
    }
}
