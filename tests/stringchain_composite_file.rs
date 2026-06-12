//! Issue #1127: the file-input StringChain remap fast path
//! (`{a: (.name + "_" + (.x | tostring)), b: .a}`) must re-render a *composite*
//! tostring operand instead of copying its raw lexeme verbatim:
//!
//!   - a nested non-canonical number lexeme must canonicalise (`[1e3]` → `[1E+3]`), and
//!   - the composite's inner `"`/`\` must be escaped for string embedding so the
//!     surrounding string stays valid JSON.
//!
//! The bug only fires through the file-argument fast path (`ResolvedRemap::StringChain`);
//! the stdin streaming twin already bailed composites to generic eval. The
//! single-stdin-document regression harness routes through the stdin twin and so
//! can't observe it — hence this shell-out test drives a real file argument and
//! cross-checks the stdin twin for parity. Expectations captured from jq 1.8.1.

use std::process::{Command, Stdio};

fn jq_jit() -> &'static str {
    env!("CARGO_BIN_EXE_jq-jit")
}

fn write_tmp(name: &str, content: &str) -> String {
    let mut path = std::env::temp_dir();
    path.push(format!("jqjit_sc1127_{}_{}", std::process::id(), name));
    std::fs::write(&path, content).unwrap();
    path.to_str().unwrap().to_string()
}

/// Run with `input` as a file argument (drives the file-input fast path).
fn run_file(filter: &str, input: &str, name: &str) -> String {
    let path = write_tmp(name, input);
    let out = Command::new(jq_jit())
        .args(["-c", filter, &path])
        .stdin(Stdio::null())
        .output()
        .expect("failed to spawn jq-jit");
    String::from_utf8(out.stdout).expect("non-utf8 stdout").trim_end().to_string()
}

/// Run the same filter feeding `input` on stdin (the streaming twin).
fn run_stdin(filter: &str, input: &str) -> String {
    use std::io::Write;
    let mut child = Command::new(jq_jit())
        .args(["-c", filter])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn jq-jit");
    child.stdin.take().unwrap().write_all(input.as_bytes()).unwrap();
    let out = child.wait_with_output().expect("wait failed");
    String::from_utf8(out.stdout).expect("non-utf8 stdout").trim_end().to_string()
}

const FILTER: &str = r#"{a: (.name + "_" + (.x | tostring)), b: .a}"#;

#[test]
fn nested_noncanonical_number_canonicalises() {
    let input = r#"{"a":[1e3,"s"],"name":"n","x":[1e3]}"#;
    let expected = r#"{"a":"n_[1E+3]","b":[1E+3,"s"]}"#;
    assert_eq!(run_file(FILTER, input, "num"), expected);
    // The stdin twin must agree (it was already correct; guard parity).
    assert_eq!(run_stdin(FILTER, input), expected);
}

#[test]
fn composite_inner_quotes_escaped_valid_json() {
    let input = r#"{"a":{"k":1e3},"name":"n","x":{"k":nan}}"#;
    let expected = r#"{"a":"n_{\"k\":null}","b":{"k":1E+3}}"#;
    assert_eq!(run_file(FILTER, input, "obj"), expected);
    assert_eq!(run_stdin(FILTER, input), expected);
}
