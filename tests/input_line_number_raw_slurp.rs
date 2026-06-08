//! Issue #925: `input_line_number` must advance in the `-R` (raw) main loop
//! and the `-s` / `-Rs` slurp paths (it returned 0), and the raw `inputs`
//! line number must not bump on an unterminated final line. jq reports the
//! count of `\n` consumed when the value was emitted; a terminated line N gets
//! N, an unterminated final line keeps the prior count. The regression-test
//! harness can't express these flags/streams, so shell out directly.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(args: &[&str], stdin: &str) -> String {
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
    String::from_utf8(out.stdout)
        .expect("non-utf8 stdout")
        .trim_end()
        .to_string()
}

// --- -R raw main loop ---

#[test]
fn raw_main_loop_unterminated_final_line() {
    // a->1, b->2, c->2 (c has no trailing newline, so no extra \n consumed).
    assert_eq!(run(&["-Rc", "input_line_number"], "a\nb\nc"), "1\n2\n2");
}

#[test]
fn raw_main_loop_trailing_newline() {
    assert_eq!(run(&["-Rc", "input_line_number"], "a\nb\nc\n"), "1\n2\n3");
}

#[test]
fn raw_main_loop_single_unterminated_line() {
    assert_eq!(run(&["-Rc", "input_line_number"], "x"), "0");
}

// --- -s JSON slurp ---

#[test]
fn json_slurp_reports_total_newlines() {
    assert_eq!(run(&["-sc", "input_line_number"], "1\n2\n3\n"), "3");
    assert_eq!(run(&["-sc", "input_line_number"], "1\n2\n3"), "2");
    assert_eq!(run(&["-sc", "input_line_number"], "1 2 3"), "0");
}

// --- -Rs raw slurp ---

#[test]
fn raw_slurp_reports_total_newlines() {
    assert_eq!(run(&["-Rsc", "input_line_number"], "a\nb\n"), "2");
    assert_eq!(run(&["-Rsc", "input_line_number"], "a\nb"), "1");
}

// --- raw inputs (-Rn) off-by-one on the unterminated final line ---

#[test]
fn raw_inputs_unterminated_final_line_no_bump() {
    assert_eq!(run(&["-Rnc", "[inputs] | input_line_number"], "a\nb\nc"), "2");
    assert_eq!(run(&["-Rnc", "[inputs] | input_line_number"], "a\nb\nc\n"), "3");
    assert_eq!(run(&["-Rnc", "[inputs|input_line_number]"], "a\nb\nc"), "[1,2,2]");
}

// --- regression guard: the JSON streaming main loop still works ---

#[test]
fn json_stream_main_loop_unaffected() {
    assert_eq!(run(&["-c", "input_line_number"], "1\n2\n3"), "1\n2\n2");
}
