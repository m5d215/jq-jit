//! Issue #751: an arithmetic update (`+=`) whose RHS is `if COND then GEN
//! else SCALAR end` (with a *generator* then-branch) must evaluate `COND`
//! with the same error semantics as every other path. A condition that
//! indexes a non-object/non-null value — e.g. `.a` on a boolean — raises
//! `Cannot index <type> with string ("<field>")`, and `?`-less callers surface
//! it.
//!
//! The JIT compiled the if-condition through the fused `FieldCmpNum` /
//! `FieldIsTruthy` ops, which silently returned `false` for a non-indexable
//! base instead of erroring, so the `else` branch was taken and the error
//! vanished — but only on the `+=` codepath (a scalar then-branch kept the
//! `map` value scalar and took the erroring path). The fused ops now set the
//! pending error and the if-codegen bails via `JumpIfError`.
//!
//! `regression.test` cannot cover this: an erroring filter produces empty
//! stdout, and an empty expected line is the suite's group delimiter. So
//! assert against stderr / exit status directly.

use std::io::Write;
use std::process::{Command, Stdio};

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

fn run(filter: &str, input: &str) -> Run {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .arg("-c")
        .arg(filter)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn jq-jit");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(format!("{input}\n").as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).trim_end().to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).trim_end().to_string(),
    }
}

#[test]
fn plus_assign_generator_then_propagates_cond_index_error() {
    // The core bug: the `+=` RHS must raise, not silently take `else`.
    for filter in [
        r#".x += map(if (.a > 0) then values else 0 end)"#,
        r#".x += map(if .a then values else 0 end)"#,
        r#".x += map(if (.a|. > 0) then values else 0 end)"#,
        r#".x += map(if (.a == 1) then values else 0 end)"#,
    ] {
        let r = run(filter, r#"{"a":false}"#);
        assert_ne!(r.code, 0, "expected error for `{filter}`, got stdout {:?}", r.stdout);
        assert!(r.stdout.is_empty(), "expected no output for `{filter}`, got {:?}", r.stdout);
        assert!(
            r.stderr.contains(r#"Cannot index boolean with string ("a")"#),
            "wrong error for `{filter}`: {:?}",
            r.stderr
        );
    }
}

#[test]
fn error_message_includes_field_name() {
    // Secondary fix: the JIT field-access error text used to be truncated
    // ("... with string" with no field name).
    let r = run(r#".x += map(if (.a > 0) then 1 else 0 end)"#, r#"{"a":false}"#);
    assert!(
        r.stderr.contains(r#"Cannot index boolean with string ("a")"#),
        "truncated/wrong error: {:?}",
        r.stderr
    );
    // Number base reports its own type.
    let r = run(r#".x += map(if (.a > 0) then values else 0 end)"#, r#"{"a":7}"#);
    assert!(
        r.stderr.contains(r#"Cannot index number with string ("a")"#),
        "wrong type in error: {:?}",
        r.stderr
    );
}

#[test]
fn cond_index_error_is_catchable() {
    // `try`/`?` must still be able to swallow it — the error is a normal
    // runtime error, not a compile error.
    let r = run(
        r#".x += [.[] | try (if (.a > 0) then values else 0 end) catch "caught"]"#,
        r#"{"a":false}"#,
    );
    assert_eq!(r.code, 0, "try/catch should succeed: {:?}", r.stderr);
    assert_eq!(r.stdout, r#"{"a":false,"x":["caught"]}"#);

    let r = run(
        r#".x += [.[] | (if (.a > 0) then values else 0 end)?]"#,
        r#"{"a":false}"#,
    );
    assert_eq!(r.code, 0, "? should suppress: {:?}", r.stderr);
    assert_eq!(r.stdout, r#"{"a":false,"x":[]}"#);
}

#[test]
fn fused_cond_fast_path_unaffected_on_valid_input() {
    // The fused FieldCmpNum/FieldIsTruthy fast paths must stay correct for
    // object/null bases (no spurious errors).
    let cases = [
        // jq truthiness: 0 is truthy, so {"a":0} → "T"; missing field → null → "F".
        (r#"map(if .a then "T" else "F" end)"#, r#"[{"a":1},{"a":0},{"b":2}]"#, r#"["T","T","F"]"#),
        (r#"[.[] | if .a >= 0 then .a else empty end]"#, r#"[{"a":5},{"a":-3}]"#, "[5]"),
        (r#"if .a > 0 then "y" else "n" end"#, r#"{"a":3}"#, r#""y""#),
        (r#"if .a > 0 then "y" else "n" end"#, r#"{"b":1}"#, r#""n""#),
        (r#"if .a > 0 then "y" else "n" end"#, "null", r#""n""#),
        (r#".x += map(if (.a > 0) then [.a] else [] end)"#, r#"{"k":{"a":5}}"#, r#"{"k":{"a":5},"x":[[5]]}"#),
    ];
    for (filter, input, expected) in cases {
        let r = run(filter, input);
        assert_eq!(r.code, 0, "`{filter}` on `{input}` errored: {:?}", r.stderr);
        assert_eq!(r.stdout, expected, "`{filter}` on `{input}`");
    }
}
