//! Issue #999: the destructuring-pattern parser over-accepted empty patterns
//! (`. as []`, `. as {}`) and trailing commas (`. as [$a,]`, `. as {a:$x,}`,
//! `. as {$a,}`) that jq's grammar rejects as compile errors. These are
//! compile-time syntax errors (exit 3), which the diff/regression harnesses
//! skip, so assert acceptance/rejection at the process level here.

use std::io::Write;
use std::process::{Command, Stdio};

/// Returns true if jq-jit *compiled* the program (it may still fail at
/// runtime). jq/jq-jit use exit code 3 for compile errors, so rejection is
/// exactly exit 3 — a runtime type error (exit 5) still counts as accepted,
/// since the pattern was syntactically valid.
fn accepts(filter: &str) -> bool {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .args(["-c", filter])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn jq-jit");
    child.stdin.take().unwrap().write_all(b"[1,2]").unwrap();
    let code = child.wait_with_output().expect("wait failed").status.code();
    code != Some(3)
}

#[test]
fn rejects_empty_and_trailing_comma_patterns() {
    let bad = [
        r#". as [] | "ok""#,
        r#". as {} | "ok""#,
        r#". as [$a,] | $a"#,
        r#". as [$a,$b,] | $a"#,
        r#". as {a:$x,} | $x"#,
        r#". as {$a,} | $a"#,
        r#". as {a:{}} | "ok""#,
        r#". as {a:[]} | "ok""#,
        r#"reduce .[] as {} (0; .)"#,
        r#"reduce .[] as [] (0; .)"#,
        r#"foreach .[] as [$a,] (0; .; .)"#,
    ];
    for f in bad {
        assert!(!accepts(f), "expected jq-jit to reject `{f}`");
    }
}

#[test]
fn still_accepts_valid_patterns() {
    let good = [
        r#"[1,2] as [$a,$b] | $a"#,
        r#"{"a":1} as {a:$x} | $x"#,
        r#"{"a":1} as {$a} | $a"#,
        r#"[[1]] as [[$a]] | $a"#,
        r#"{"a":[1]} as {a:[$x]} | $x"#,
        r#"[1] as [$a] ?// $a | $a"#,
        r#"{"a":1,"b":2} as {a:$x,b:$y} | $x"#,
        r#"foreach .[] as [$a] (0; .; .)"#,
        r#"reduce .[] as [$a] (0; .)"#,
    ];
    for f in good {
        assert!(accepts(f), "expected jq-jit to accept `{f}`");
    }
}
