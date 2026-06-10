//! Issue #1084: errors from JIT-delegated builtins were swallowed or
//! reworded inside try/catch.
//!
//! - The match/capture generator lowering caught *every* error to map the
//!   internal no-match signal to "no output", so genuine type errors
//!   vanished (catch never ran). The catch now rethrows anything that is
//!   not the internal "match failed" payload.
//! - `limit` unboxed its count with a silent f64 coercion, so a
//!   non-numeric count produced empty output instead of jq's type errors;
//!   non-numeric-literal counts now bail to eval (which implements jq's
//!   lazy validation, #806).
//! - `range` bounds were coerced the same way; bounds that are not numeric
//!   literals now unbox through a checked conversion raising "Range bounds
//!   must be numeric". Provably non-numeric literal steps bail to eval for
//!   its lazy add-error semantics.
//! - The runtime sub/gsub arm produced its own message ("sub/gsub requires
//!   string, regex, and replacement") instead of eval's canonical wording,
//!   and its replace_all-based splicing missed zero-width matches adjacent
//!   to non-empty ones (`gsub("a*"; "X")` on `"abc"`).
//!
//! Default dispatch masked all of this for small inputs; these tests force
//! both JitOp backends.

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
fn match_type_errors_are_catchable() {
    assert_jit(
        "try (.a | match(\"x\")) catch \"err\"",
        "{\"a\":1}",
        "\"err\"",
    );
    assert_jit(
        "try (match(\"a\")) catch .",
        "null",
        "\"null (null) cannot be matched, as it is not a string\"",
    );
    assert_jit(
        "try (capture(\"(?<x>.)\")) catch .",
        "null",
        "\"null (null) cannot be matched, as it is not a string\"",
    );
    assert_jit(
        "\"abc\" | try match(123) catch .",
        "null",
        "\"number not a string or array\"",
    );
}

#[test]
fn match_no_match_is_still_empty() {
    assert_jit("[match(\"z\"; \"g\")] | length", "\"banana\"", "0");
    assert_jit("[match(\"a\"; \"g\")] | length", "\"banana\"", "3");
}

#[test]
fn limit_count_type_errors_match_eval() {
    assert_jit(
        "try (limit(\"a\"; .[])) catch .",
        "[1,2,3]",
        "\"string (\\\"a\\\") and number (1) cannot be subtracted\"",
    );
    assert_jit(
        "try (limit(null; .[])) catch .",
        "[1,2,3]",
        "\"limit doesn't support negative count\"",
    );
    // Numeric literal counts stay on the JIT path.
    assert_jit("[limit(2; 1,2,3)]", "null", "[1,2]");
    assert_jit("[limit(0; 1,2)]", "null", "[]");
}

#[test]
fn range_bounds_raise_instead_of_coercing() {
    assert_jit(
        "try (range(\"x\"; 5)) catch .",
        "null",
        "\"Range bounds must be numeric\"",
    );
    assert_jit(
        "try (range(0; \"x\")) catch .",
        "null",
        "\"Range bounds must be numeric\"",
    );
    assert_jit(
        "\"x\" | try [range(.)] catch .",
        "null",
        "\"Range bounds must be numeric\"",
    );
    // Dynamic numeric bounds keep working.
    assert_jit(". as $n | [range($n)]", "3", "[0,1,2]");
}

#[test]
fn range_non_numeric_literal_step_uses_eval_lazy_semantics() {
    assert_jit(
        "[try (range(0; 10; \"a\")) catch .]",
        "null",
        "[0,\"number (0) and string (\\\"a\\\") cannot be added\"]",
    );
    assert_jit(
        "[try (range(0; 10; [])) catch .]",
        "null",
        "[0,\"number (0) and array ([]) cannot be added\"]",
    );
}

#[test]
fn sub_gsub_type_errors_match_eval() {
    assert_jit(
        "try ([1] | sub(\"a\"; \"b\")) catch .",
        "null",
        "\"array ([1]) cannot be matched, as it is not a string\"",
    );
    assert_jit(
        "try (null | gsub(\"a\"; \"b\")) catch .",
        "null",
        "\"null (null) cannot be matched, as it is not a string\"",
    );
}

#[test]
fn gsub_enumerates_empty_matches_adjacent_to_nonempty() {
    assert_jit("gsub(\"a*\"; \"X\")", "\"abc\"", "\"XXbXcX\"");
    // `n` flag still drops the zero-width matches.
    assert_jit("gsub(\"a*\"; \"X\"; \"n\")", "\"abc\"", "\"Xbc\"");
    assert_jit("sub(\"a\"; \"X\")", "\"abc\"", "\"Xbc\"");
}
