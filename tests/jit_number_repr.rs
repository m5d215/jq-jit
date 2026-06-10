//! Issue #1083: number literal repr and canonical uppercase-E formatting
//! were lost on the JIT path. jq's contract: a number still carrying its
//! literal lexeme renders with the canonicalized form (uppercase `E`,
//! preserved trailing `.0`), computed numbers render in the lowercase
//! shortest form. The eval path implements this; the JIT path dropped the
//! repr in several places:
//!
//! - `jit_rt_unaryop` hand-rolled arms (length/tonumber/tojson/fromjson/
//!   abs/fabs/add and the inline math `keep_repr` policy) drifted from the
//!   shared rt_ implementations,
//! - range loops re-boxed every item from the f64 counter, losing the seed
//!   repr eval preserves on the first yielded value,
//! - the constant-folded `[range(...)]` and fused f64 reduce/foreach paths
//!   had the same seed/init repr gaps.
//!
//! Default dispatch masked all of this for small inputs (< 4KB routes to
//! eval), so these tests force each JitOp backend explicitly.

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

/// Assert both forced JitOp backends produce the expected output.
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
fn tojson_canonicalizes_literal_repr() {
    assert_jit("1.0 | tojson", "null", "\"1.0\"");
    assert_jit("1e10 | tojson", "null", "\"1E+10\"");
    assert_jit("5e-324 | tojson", "null", "\"5E-324\"");
    assert_jit("[1.0] | tojson", "null", "\"[1.0]\"");
}

#[test]
fn tonumber_preserves_string_lexeme() {
    assert_jit("\"1e10\" | tonumber", "null", "1E+10");
}

#[test]
fn fromjson_preserves_number_lexeme() {
    assert_jit("\"0.00001\" | fromjson", "null", "0.00001");
}

#[test]
fn abs_keeps_repr_fabs_drops_it() {
    assert_jit("-1.0 | abs", "null", "1.0");
    assert_jit("-1.50 | abs", "null", "1.50");
    assert_jit("-0.0 | fabs", "null", "0");
}

#[test]
fn length_keeps_repr() {
    assert_jit("-0.0 | length", "null", "0.0");
    assert_jit("-1e10 | length", "null", "1E+10");
}

#[test]
fn math_ops_drop_repr_like_eval() {
    // eval's rt_floor returns the canonical f64 form even when the value
    // is unchanged; the JIT kept the repr when result == input.
    assert_jit("1.0 | floor", "null", "1");
}

#[test]
fn add_single_element_passes_value_through() {
    assert_jit("[1.0] | add | tojson", "null", "\"1.0\"");
    assert_jit("[1.5e2] | add | tojson", "null", "\"1.5E+2\"");
    assert_jit("[1.0, null] | add | tojson", "null", "\"1.0\"");
    // An actual addition is computed: repr drops.
    assert_jit("[1.0, 0] | add", "null", "1");
}

#[test]
fn range_first_item_keeps_seed_repr() {
    assert_jit("[range(0.0; 3)]", "null", "[0.0,1,2]");
    assert_jit("[range(0.0; 1.0; 0.25)]", "null", "[0.0,0.25,0.5,0.75]");
    assert_jit("[range(3.0; 0; -1)]", "null", "[3.0,2,1]");
    assert_jit("range(0.0; 2)", "null", "0.0\n1");
    assert_jit("1.0 as $x | [range($x; 4)]", "null", "[1.0,2,3]");
    assert_jit("[range(.a; 3)]", "{\"a\":1.50}", "[1.50,2.5]");
    assert_jit("[range(0.0; 3) | . * 2]", "null", "[0,2,4]");
}

#[test]
fn fused_reduce_foreach_respect_seed_and_init_repr() {
    assert_jit("[foreach range(0.0; 2) as $i (0; $i)]", "null", "[0.0,1]");
    assert_jit("reduce range(0.0; 1) as $x (0; $x)", "null", "0.0");
    // Empty range yields the init verbatim.
    assert_jit("reduce range(0) as $x (0.0; . + $x)", "null", "0.0");
    // Computed accumulators stay reprless.
    assert_jit("[foreach range(0.0; 2) as $i (0; . + $i)]", "null", "[0,1]");
    assert_jit("reduce range(0.0; 3) as $x (0; . + $x)", "null", "3");
}
