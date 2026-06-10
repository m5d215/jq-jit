//! Issue #1090: when a JIT-executed filter yields its input unchanged
//! (e.g. a passing select), the identity-output raw passthrough copied the
//! original bytes verbatim — without the last-wins duplicate-key dedup jq
//! applies at parse time (#233 class). The select verdict itself was
//! computed on the parsed (deduped) Value, so only the emitted bytes
//! diverged. The passthrough now routes through the Value serializer when
//! the raw bytes contain duplicate keys, like the detect_* fast paths.

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
fn passing_select_dedupes_duplicate_keys() {
    assert_jit(
        "select(.a > .b)",
        "{\"a\":1,\"a\":5,\"b\":3}",
        "{\"a\":5,\"b\":3}",
    );
    assert_jit(
        "select(.b > .a)",
        "{\"b\":1,\"a\":5,\"a\":1,\"b\":7}",
        "{\"b\":7,\"a\":1}",
    );
}

#[test]
fn clean_input_still_passes_through() {
    assert_jit("select(.a > .b)", "{\"a\":5,\"b\":3}", "{\"a\":5,\"b\":3}");
    assert_jit("select(.a > .b)", "{\"a\":1,\"b\":3}", "");
}
