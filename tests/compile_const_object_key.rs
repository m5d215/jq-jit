//! Issue #726: a constant non-string object key is a *compile-time* error in
//! jq (rejected before any input is read), so it cannot be intercepted by
//! `try`/`?`. jq folds the key the way its compiler does — literals, binary
//! arithmetic/comparison operators, and constant array/object literals — and
//! rejects a non-string result at compile time. Runtime-computed keys
//! (`{(.k):2}`), unary operators (`{(-1):2}`), pipes, `and`/`or`, and anything
//! input- or variable-dependent keep their *runtime* error and parse fine.
//!
//! `regression.test` cannot cover this: its harness skips programs that exit 3
//! (compile error). So assert directly against `Filter::with_options`, which
//! is what surfaces the compile error in the CLI (`jq: error: ...`, exit 3).

use jq_jit::interpreter::Filter;

fn compile_err(program: &str) -> String {
    match Filter::with_options(program, &[], false) {
        Ok(_) => panic!("expected compile error for `{program}`, but it parsed"),
        Err(e) => e.to_string(),
    }
}

fn parses(program: &str) {
    if let Err(e) = Filter::with_options(program, &[], false) {
        panic!("expected `{program}` to parse (defer to runtime), got compile error: {e}");
    }
}

#[test]
fn scalar_constant_non_string_keys_are_compile_errors() {
    assert!(compile_err("{(1):2}").contains("Cannot use number (1) as object key"));
    assert!(compile_err("{(null):2}").contains("Cannot use null (null) as object key"));
    assert!(compile_err("{(true):2}").contains("Cannot use boolean (true) as object key"));
    assert!(compile_err("{(false):2}").contains("Cannot use boolean (false) as object key"));
}

#[test]
fn constant_key_compile_error_is_not_catchable() {
    // The whole point of #726: `try`/`?` must not be able to swallow it,
    // because the error happens before the program runs.
    compile_err(r#"try {(1):2} catch "c""#);
    compile_err("{(1):2}?");
}

#[test]
fn folded_binop_constant_keys_are_compile_errors() {
    // jq constant-folds arithmetic and comparison operators over literals.
    assert!(compile_err("{(1+1):2}").contains("Cannot use number (2) as object key"));
    assert!(compile_err("{(2*3):2}").contains("Cannot use number (6) as object key"));
    assert!(compile_err("{(3-1):2}").contains("Cannot use number (2) as object key"));
    assert!(compile_err("{(6/2):2}").contains("Cannot use number (3) as object key"));
    assert!(compile_err("{(7%3):2}").contains("Cannot use number (1) as object key"));
    assert!(compile_err("{(1+2*3):2}").contains("Cannot use number (7) as object key"));
    assert!(compile_err("{(1==1):2}").contains("Cannot use boolean (true) as object key"));
    assert!(compile_err("{(1<2):2}").contains("Cannot use boolean (true) as object key"));
    assert!(compile_err("{(1>=2):2}").contains("Cannot use boolean (false) as object key"));
}

#[test]
fn composite_constant_literal_keys_are_compile_errors() {
    assert!(compile_err("{([]):2}").contains("Cannot use array ([]) as object key"));
    assert!(compile_err("{([1,2]):2}").contains("Cannot use array ([1,2]) as object key"));
    assert!(compile_err("{([1+1]):2}").contains("Cannot use array ([2]) as object key"));
    assert!(compile_err("{({}):2}").contains("Cannot use object ({}) as object key"));
    assert!(compile_err("{({a:1}):2}").contains("as object key"));
}

#[test]
fn unused_def_with_constant_key_is_a_compile_error() {
    // jq validates constant keys regardless of reachability.
    compile_err("def f: {(1):2}; 1");
}

#[test]
fn string_constant_keys_parse() {
    parses(r#"{("a"):2}"#);
    parses(r#"{("a"+"b"):1}"#); // folds to a string -> allowed
    parses(r#"{("a"|ascii_upcase):1}"#);
}

#[test]
fn runtime_computed_keys_defer_to_runtime() {
    // These are *runtime* errors in jq (catchable), so they must parse.
    parses("{(.k):2}");
    parses("{(-1):2}"); // unary negate is not folded by jq
    parses("{(1|.+1):2}");
    parses("{(1,2):3}");
    parses("{(if true then 1 else 2 end):2}");
    parses("{(true and false):2}");
    parses("{(true or false):2}");
    parses("{(null//5):2}");
    parses("1 as $x | {($x):2}");
    parses("{([range(3)]):2}");
    parses("{([.]):2}");
    parses("{([1,.]):2}");
}
