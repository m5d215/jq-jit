//! Issue #977: `?//` is the destructuring-alternative separator, valid only
//! between patterns in `EXPR as PAT ?// PAT | body` (and the `reduce`/`foreach`
//! pattern chains). jq raises a *compile* error when it appears as a bare
//! binary operator anywhere else; jq-jit previously accepted it (running
//! `try LHS catch (LHS // RHS)`).
//!
//! `regression.test` cannot cover the rejection: its harness skips programs
//! that exit 3 (compile error). Assert directly against `Filter::with_options`,
//! which surfaces the compile error in the CLI.

use jq_jit::interpreter::Filter;

fn is_compile_error(program: &str) -> bool {
    Filter::with_options(program, &[], false).is_err()
}

fn parses(program: &str) {
    if let Err(e) = Filter::with_options(program, &[], false) {
        panic!("expected `{program}` to parse, got compile error: {e}");
    }
}

#[test]
fn bare_altdestructure_is_a_compile_error() {
    assert!(is_compile_error("1 ?// 2"), "top-level bare ?//");
    assert!(is_compile_error("{a: (1 ?// 2)}"), "object value");
    assert!(is_compile_error("[1 ?// 2]"), "array element");
    assert!(is_compile_error(".x ?// .y"), "after a field");
    assert!(is_compile_error(r#"(.[0]) ?// "fallback""#), "parenthesized lhs");
    assert!(is_compile_error("1 | 2 ?// 3"), "after a pipe");
    assert!(is_compile_error("def f: 1 ?// 2; f"), "in a def body");
}

#[test]
fn destructuring_alternative_still_parses() {
    // The legitimate `?//` positions must keep working.
    parses(". as [$a] ?// $a | $a");
    parses(". as {a:$x} ?// $x | $x");
    parses(". as [$a] ?// {a:$a} ?// $a | $a");
    parses("reduce .[] as [$a] ?// $a (0; . + $a)");
    parses("foreach .[] as [$a] ?// $a (0; . + $a; .)");
}
