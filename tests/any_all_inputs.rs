//! Issue #928: `any(inputs; cond)` / `all(inputs; cond)` (and an
//! input-consuming generator like `any(inputs == 5; .)`) must drive the input
//! stream. Two defects combined to drop it: `uses_inputs()` didn't recurse into
//! `AnyShort`/`AllShort` (so `-n` never seeded the input queue), and the JIT
//! `any/all` compile ignored an uncompilable generator (silently emitting no
//! loop body, collapsing any/2 to `false` and all/2 to `true`). The
//! regression-test harness can't express `-n` plus a multi-document stream, so
//! shell out directly. Run both the default (JIT) and forced-interpreter paths.

use std::io::Write;
use std::process::{Command, Stdio};

fn run_env(args: &[&str], stdin: &str, force_interp: bool) -> String {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut cmd = Command::new(jq_jit);
    cmd.args(args);
    if force_interp {
        cmd.env("JQJIT_FORCE_INTERPRETER", "1");
    }
    let mut child = cmd
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

/// Assert both the JIT and forced-interpreter paths produce `expected`.
fn both(args: &[&str], stdin: &str, expected: &str) {
    assert_eq!(run_env(args, stdin, false), expected, "JIT path: {:?}", args);
    assert_eq!(run_env(args, stdin, true), expected, "interp path: {:?}", args);
}

#[test]
fn any_inputs_true_when_a_value_matches() {
    both(&["-nc", "any(inputs; .>2)"], "1 2 3", "true");
}

#[test]
fn all_inputs_false_when_a_value_fails() {
    both(&["-nc", "all(inputs; .>0)"], "1 2 -3", "false");
}

#[test]
fn any_inputs_true_predicate_sees_every_value() {
    both(&["-nc", "any(inputs; true)"], "1 2 3", "true");
}

#[test]
fn all_inputs_false_predicate_sees_every_value() {
    both(&["-nc", "all(inputs; false)"], "1 2 3", "false");
}

#[test]
fn any_inputs_truthy_value() {
    both(&["-nc", "any(inputs; .)"], "0 0 1", "true");
}

#[test]
fn input_consuming_generator_expression() {
    // `inputs == 5` is the generator; the stream's 5 must reach it.
    both(&["-nc", "any(inputs == 5; .)"], "5", "true");
}

#[test]
fn any_inputs_false_when_none_match() {
    both(&["-nc", "any(inputs; .>10)"], "1 2 3", "false");
}

#[test]
fn non_input_generators_still_work() {
    // The JIT must still compile these (the bailout is generator-specific).
    both(&["-nc", "any(range(2000); .>1998)"], "", "true");
    both(&["-nc", "all(range(2000); .>=0)"], "", "true");
    both(&["-nc", "any(1,2,3; .>2)"], "", "true");
}
