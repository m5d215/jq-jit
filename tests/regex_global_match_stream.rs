//! Issue #748: `match`/`capture` with the `g` (global) flag must emit one
//! match per value as a *stream*, matching jq, rather than a single array
//! containing all matches. `scan`/`splits`/`gsub` already behave correctly.
//!
//! This was a JIT-only divergence reached on the `-n` (null-input) codepath:
//! the interpreter's regex path already streamed the global array via the
//! callback, but the JIT generator flattener emitted the whole array slot once.
//! A non-global match yields a single match *object* (never an array), so the
//! fix dispatches on the runtime result kind: array -> stream each element,
//! object/other -> yield once.
//!
//! `regression.test` cannot cover this: its harness always feeds input on
//! stdin (no `-n`), which routes through the eval/interpreter path and never
//! reaches the JIT regex generator path. So shell out with `-n` directly.

use std::process::Command;

/// Run `jq-jit -nc <filter>` and return stdout, one output value per line,
/// joined with '|' so a streamed result is distinguishable from a single array.
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
        .replace('\n', "|")
}

#[test]
fn global_match_streams_one_object_per_match() {
    // The core bug: `g` flag must stream, not pack into a single array.
    assert_eq!(
        run_null_input(r#""aXbXc" | match("X";"g")"#),
        r#"{"offset":1,"length":1,"string":"X","captures":[]}|{"offset":3,"length":1,"string":"X","captures":[]}"#
    );
    // Collecting the stream yields one element per match.
    assert_eq!(run_null_input(r#"[ "aXbXc" | match("X";"g") ] | length"#), "2");
    assert_eq!(run_null_input(r#"[ "aXbXcXd" | match("X";"g") ] | length"#), "3");
}

#[test]
fn global_capture_streams_one_object_per_match() {
    assert_eq!(
        run_null_input(r#""aXbYc" | capture("(?<c>[A-Z])";"g")"#),
        r#"{"c":"X"}|{"c":"Y"}"#
    );
    assert_eq!(
        run_null_input(r#"[ "aXbYc" | capture("(?<c>[A-Z])";"g") ]"#),
        r#"[{"c":"X"},{"c":"Y"}]"#
    );
}

#[test]
fn array_form_regex_with_global_flag_streams() {
    // The `g` flag can also ride inside the `[regex, flags]` array argument
    // (flags == null but the array carries "g"). Must still stream.
    assert_eq!(run_null_input(r#"[ "aXbXcXd" | match(["X","g"]) ] | length"#), "3");
}

#[test]
fn non_global_match_yields_single_object() {
    // Without `g`, match/capture yield exactly one object (a stream of 1),
    // never wrapped in an array.
    assert_eq!(
        run_null_input(r#""aXbXc" | match("X")"#),
        r#"{"offset":1,"length":1,"string":"X","captures":[]}"#
    );
    assert_eq!(run_null_input(r#"[ "aXbXc" | match("X") ] | length"#), "1");
    assert_eq!(run_null_input(r#""aXbYc" | capture("(?<c>[A-Z])")"#), r#"{"c":"X"}"#);
}

#[test]
fn no_match_global_is_empty_stream() {
    // No matches -> empty stream (collects to []), not an error or [null].
    assert_eq!(run_null_input(r#"[ "abc" | match("X";"g") ] | length"#), "0");
    assert_eq!(run_null_input(r#"[ "abc" | capture("(?<c>[A-Z])";"g") ] | length"#), "0");
}
