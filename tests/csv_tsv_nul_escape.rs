//! Issue #929: `@csv`/`@tsv` must escape an embedded NUL to the two-char `\0`
//! (the #849 NUL fix covered `@html`/`@sh` but missed the CSV/TSV formatters).
//! While fixing it, the `-r` raw CSV/TSV field fast path was also found to emit
//! TSV special bytes (`\t \n \r \\`) and NUL verbatim instead of escaping them;
//! that path is now aligned with jq / the value formatter too.
//!
//! The regression-test harness only runs without `-r`, so the raw fast path is
//! exercised here by shelling out with `-r` and a JSON input carrying escapes.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(args: &[&str], stdin: &str) -> Vec<u8> {
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
    let mut bytes = out.stdout;
    while bytes.last() == Some(&b'\n') {
        bytes.pop();
    }
    bytes
}

// --- Value-formatter path (@csv/@tsv applied to a constructed array) ---

#[test]
fn csv_escapes_embedded_nul() {
    // [[97,0,98]|implode] -> "a\0b", @csv cell = "a\0b" (NUL -> backslash-zero).
    // -r so we see the raw @csv string, not its JSON re-encoding.
    assert_eq!(
        run(&["-rc", "[[97,0,98]|implode]|@csv"], "null"),
        b"\"a\\0b\""
    );
}

#[test]
fn tsv_escapes_embedded_nul() {
    assert_eq!(run(&["-rc", "[[97,0,98]|implode]|@tsv"], "null"), b"a\\0b");
}

// --- Raw `-r` field fast path (`[.a,.b]|@csv`/`@tsv`) ---

#[test]
fn raw_csv_escapes_nul_field() {
    assert_eq!(
        run(&["-rc", "[.a,.b]|@csv"], "{\"a\":\"\\u0000\",\"b\":\"z\"}"),
        b"\"\\0\",\"z\""
    );
}

#[test]
fn raw_tsv_escapes_nul_field() {
    assert_eq!(
        run(&["-rc", "[.a,.b]|@tsv"], "{\"a\":\"\\u0000\",\"b\":\"z\"}"),
        b"\\0\tz"
    );
}

#[test]
fn raw_tsv_escapes_tab_newline_cr_backslash() {
    // These leaked unescaped on the raw fast path before the #929 fix.
    assert_eq!(
        run(&["-rc", "[.a,.b]|@tsv"], "{\"a\":\"x\\ty\",\"b\":\"z\"}"),
        b"x\\ty\tz"
    );
    assert_eq!(
        run(&["-rc", "[.a,.b]|@tsv"], "{\"a\":\"x\\ny\",\"b\":\"z\"}"),
        b"x\\ny\tz"
    );
    assert_eq!(
        run(&["-rc", "[.a,.b]|@tsv"], "{\"a\":\"x\\\\y\",\"b\":\"z\"}"),
        b"x\\\\y\tz"
    );
}

#[test]
fn raw_csv_quote_doubling_unaffected() {
    assert_eq!(
        run(&["-rc", "[.a,.b]|@csv"], "{\"a\":\"x\\\"y\",\"b\":\"z\"}"),
        b"\"x\"\"y\",\"z\""
    );
}
