//! Issue #1085: path()/setpath validation gaps on the JIT path returned
//! wrong values where eval/jq raise, and `limit(n; path(recurse(.a)))`
//! hung. The worst class from the #1059 backend self-diff: silently wrong
//! data, not just wrong error text.
//!
//! - `path(<simple path>)` emitted the static path array without walking
//!   the input, skipping eval's navigation errors. A getpath probe now
//!   performs the identical walk for effect.
//! - Complex path expressions were delegated to an eager collect-all,
//!   which hung on infinite streams eval cuts lazily and predated the
//!   eval-side provenance fixes (#880/#953) for rootless anchors like
//!   `. as $x | path($x)`. They now bail the whole filter to eval.
//! - The in-place reduce setpath (SetPathMut) discarded navigation errors,
//!   leaving the accumulator untouched.
//! - The jit_rt_unaryop transpose and from_entries arms drifted from the
//!   shared rt_ helpers (null-padding instead of "Cannot index", pre-#976
//!   `key_` alias and missing null-key error).

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
fn simple_path_validates_navigation() {
    assert_jit(
        "try path(.[0]) catch .",
        "5",
        "\"Cannot index number with number (0)\"",
    );
    assert_jit(
        "try path(.x) catch .",
        "5",
        "\"Cannot index number with string (\\\"x\\\")\"",
    );
    // Valid navigations still yield the path.
    assert_jit("path(.a)", "{\"a\":1}", "[\"a\"]");
    assert_jit("path(.a.b)", "{\"a\":{\"b\":1}}", "[\"a\",\"b\"]");
    assert_jit("path(.missing)", "{}", "[\"missing\"]");
}

#[test]
fn invalid_path_keys_raise() {
    assert_jit(
        "try (path(.[true])) catch .",
        "null",
        "\"Cannot index null with boolean (true)\"",
    );
    assert_jit(
        "try (path(.[null])) catch .",
        "null",
        "\"Cannot index null with null (null)\"",
    );
}

#[test]
fn rootless_var_anchored_path_yields_empty_prefix() {
    assert_jit(". as $x | path($x)", "{\"a\":1}", "[]");
    assert_jit(". as $x | path($x | .a)", "{\"a\":1}", "[\"a\"]");
}

#[test]
fn limit_of_infinite_path_stream_terminates() {
    assert_jit(
        "[limit(5; path(recurse(.a)))]",
        "{\"a\":{\"b\":9}}",
        "[[],[\"a\"],[\"a\",\"a\"],[\"a\",\"a\",\"a\"],[\"a\",\"a\",\"a\",\"a\"]]",
    );
}

#[test]
fn reduce_setpath_surfaces_navigation_errors() {
    assert_jit(
        "try (reduce range(1) as $_ (.; setpath([\"a\"]; 99))) catch .",
        "5",
        "\"Cannot index number with string (\\\"a\\\")\"",
    );
    assert_jit(
        "try (reduce range(1) as $_ (.; setpath([null]; 99))) catch .",
        "5",
        "\"Cannot index number with null (null)\"",
    );
    // The happy path keeps working in place.
    assert_jit(
        "reduce range(3) as $i ({}; setpath([\"k\\($i)\"]; $i)) | length",
        "null",
        "3",
    );
}

#[test]
fn transpose_raises_on_non_array_elements() {
    assert_jit(
        "try transpose catch .",
        "[1,2]",
        "\"Cannot index number with number (0)\"",
    );
    assert_jit("transpose", "[[1,2],[3,4]]", "[[1,3],[2,4]]");
}

#[test]
fn from_entries_key_precedence_and_null_key() {
    // jq 1.8.1 resolves the key via .key // .Key // .name // .Name;
    // `key_` is NOT an alias (#976).
    assert_jit(
        "from_entries",
        "[{\"key_\":\"K_\",\"name\":\"NAME\",\"value\":1}]",
        "{\"NAME\":1}",
    );
    assert_jit(
        "try from_entries catch .",
        "[{\"key_\":\"x\",\"value\":1}]",
        "\"Cannot use null (null) as object key\"",
    );
    assert_jit(
        "from_entries",
        "[{\"key\":\"a\",\"value\":1}]",
        "{\"a\":1}",
    );
}
