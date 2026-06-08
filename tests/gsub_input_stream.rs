//! Issue #930: when the replacement of `sub`/`gsub` pulls from `input`/`inputs`
//! (`gsub(re; input)`), the input queue must be seeded so the read consumes the
//! shared stream instead of raising a bogus `break` at EOF. The bug: the
//! `uses_inputs()` walker had no arm for the Regex* expressions, so the binary
//! never seeded the queue. `input` then hit EOF (printing a leaked `break`,
//! exit 5) and the main loop replayed each remaining document as its own
//! top-level input, producing the wrong output shape (`"X"`,`"Y"` instead of
//! the joined `"XY"`). The regression-test harness feeds a single document, so
//! we shell out to drive a multi-document stream.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(args: &[&str], stdin: &str) -> (String, i32) {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn jq-jit");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    let out = child.wait_with_output().expect("wait failed");
    let code = out.status.code().unwrap_or(-1);
    let s = String::from_utf8(out.stdout)
        .expect("non-utf8 stdout")
        .trim_end()
        .to_string();
    (s, code)
}

#[test]
fn gsub_single_match_consumes_one_input() {
    // "a" matches once; replacement reads the next document "X". Clean exit.
    assert_eq!(run(&["-c", r#"gsub("a"; input)"#], "\"a\"\n\"X\""), ("\"X\"".to_string(), 0));
}

#[test]
fn gsub_two_matches_lockstep_joins_into_one_string() {
    // Two matches each pull one input in lockstep -> a single joined "XY",
    // NOT two separate top-level outputs.
    assert_eq!(
        run(&["-c", r#"gsub("a"; input)"#], "\"aa\"\n\"X\"\n\"Y\""),
        ("\"XY\"".to_string(), 0)
    );
}

#[test]
fn sub_single_match_consumes_one_input() {
    assert_eq!(run(&["-c", r#"sub("a"; input)"#], "\"a\"\n\"R\""), ("\"R\"".to_string(), 0));
}

#[test]
fn gsub_with_capture_and_input() {
    assert_eq!(
        run(&["-c", r#"gsub("(?<x>a)"; input)"#], "\"a\"\n\"Z\""),
        ("\"Z\"".to_string(), 0)
    );
}

#[test]
fn gsub_inputs_drains_remaining_stream() {
    // `inputs` is a generator: the first match drains the whole remaining
    // stream (["X","Y"]); the second match's `inputs` yields nothing. The
    // lockstep length is 2, so output index 0 -> "X", index 1 -> "Y" (the
    // empty second-match generator drops out at each index). This matches
    // jq exactly; the point of the test is no leaked `break`/exit 5.
    assert_eq!(
        run(&["-c", r#"gsub("a"; inputs)"#], "\"aa\"\n\"X\"\n\"Y\""),
        ("\"X\"\n\"Y\"".to_string(), 0)
    );
}

#[test]
fn gsub_genuine_eof_still_breaks() {
    // No second document: the read genuinely hits EOF. jq surfaces an
    // uncaught break (exit 5); we must match that, not silently swallow it.
    let (_out, code) = run(&["-c", r#"gsub("a"; input)"#], "\"a\"");
    assert_eq!(code, 5);
}

#[test]
fn gsub_no_input_replacement_unaffected() {
    // The replacement does not read the stream: the second document must
    // remain a separate top-level input, each producing its own output.
    assert_eq!(
        run(&["-c", r#"gsub("a"; "b")"#], "\"aaa\"\n\"junk\""),
        ("\"bbb\"\n\"junk\"".to_string(), 0)
    );
}
