//! Issue #855: `input_line_number` must advance when a document is consumed via
//! the `input` / `inputs` builtins, not only by the main per-document loop
//! (a residual of #117). The regression-test harness can't express the `-n`
//! flag plus a multi-document stream, so shell out directly.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(args: &[&str], stdin: &str) -> String {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .args(args)
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
    String::from_utf8(out.stdout)
        .expect("non-utf8 stdout")
        .trim_end()
        .to_string()
}

#[test]
fn input_advances_line_number() {
    assert_eq!(run(&["-nc", "input | input_line_number"], "1\n2\n"), "1");
}

#[test]
fn inputs_advance_line_number_each_step() {
    assert_eq!(
        run(&["-nc", "[inputs as $x | input_line_number]"], "10\n20\n30\n"),
        "[1,2,3]"
    );
}

#[test]
fn main_loop_counter_still_works() {
    // The original #117 fix (main per-document loop) must remain intact.
    assert_eq!(run(&["-c", "input_line_number"], "11\n22\n"), "1\n2");
}

#[test]
fn input_then_line_number_interleaved() {
    // Document 1 pulls document 2 via `input` (line 2); document 3 has no
    // further `input` to read (the program then errors, but the first line's
    // stdout is what we assert on).
    assert_eq!(
        run(&["-c", "[., input, input_line_number]"], "1\n2\n3\n"),
        "[1,2,2]"
    );
}
