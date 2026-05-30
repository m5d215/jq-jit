//! Issue #731: `"" | split(sep)` must return `[]` (an empty array) for an
//! empty input string, matching jq, even when the program runs through the
//! JIT codepath reached by `-n` (null input).
//!
//! The interpreter's `rt_split` already special-cases the empty input string,
//! and so does jq itself. The JIT fast path for single-arg `split` did not:
//! Rust's `"".split(",")` yields one empty segment (`[""]`), so the JIT path
//! emitted `[""]` while every other path returned `[]`.
//!
//! `regression.test` cannot cover this: its harness always feeds input on
//! stdin (no `-n`), which routes through the interpreter/eval path and never
//! reaches the JIT `split` fast path. So shell out with `-n` directly.

use std::process::Command;

fn run_null_input(filter: &str) -> String {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let out = Command::new(jq_jit)
        .arg("-nc")
        .arg(filter)
        .output()
        .expect("failed to spawn jq-jit");
    assert!(
        out.status.success(),
        "jq-jit -nc {filter:?} exited with {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .expect("non-utf8 stdout")
        .trim_end()
        .to_string()
}

#[test]
fn empty_input_string_splits_to_empty_array_on_null_input() {
    // The core bug: empty string with a non-empty separator -> [].
    assert_eq!(run_null_input(r#""" | split(",")"#), "[]");
    assert_eq!(run_null_input(r#""" | split(",x,")"#), "[]");
    // Empty separator on empty string is also [] (already correct, pinned here).
    assert_eq!(run_null_input(r#""" | split("")"#), "[]");
}

#[test]
fn non_empty_input_string_split_unaffected_on_null_input() {
    // Guard against over-eager emptying: non-empty inputs keep their segments.
    assert_eq!(run_null_input(r#""abc" | split(",")"#), r#"["abc"]"#);
    assert_eq!(run_null_input(r#""a,b,c" | split(",")"#), r#"["a","b","c"]"#);
    assert_eq!(run_null_input(r#""a,," | split(",")"#), r#"["a","",""]"#);
    assert_eq!(run_null_input(r#""abc" | split("")"#), r#"["a","b","c"]"#);
}
