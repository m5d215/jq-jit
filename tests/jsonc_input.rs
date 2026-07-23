//! Issue #1141: `--jsonc` accepts `//` line and `/* */` block comments in JSON
//! input by blanking them with spaces at the input boundary, so the downstream
//! parser (and every raw-byte fast path) sees standard JSON with unchanged
//! byte offsets and line numbers. Comment-only scope — no trailing commas or
//! other JSON5. Default behavior (flag off) is unchanged.

use std::io::Write;
use std::process::{Command, Stdio};

use jq_jit::value::strip_json_comments;

fn run_stdin(args: &[&str], input: &str) -> (String, String, Option<i32>) {
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
        .write_all(input.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
        out.status.code(),
    )
}

// --- strip_json_comments unit-level coverage ---------------------------------

fn stripped(s: &str) -> String {
    let mut owned = s.to_string();
    strip_json_comments(&mut owned);
    owned
}

#[test]
fn strip_blanks_line_and_block_comments() {
    assert_eq!(stripped("// c\n{\"a\":1}"), "    \n{\"a\":1}");
    assert_eq!(stripped("{\"a\":/*x*/1}"), "{\"a\":     1}");
}

#[test]
fn strip_preserves_length_and_newlines() {
    let src = "/* a\n b */\n{\"a\":1} // t\n";
    let out = stripped(src);
    assert_eq!(out.len(), src.len());
    let nl = |s: &str| s.bytes().filter(|&b| b == b'\n').count();
    assert_eq!(nl(&out), nl(src));
    assert_eq!(out, "    \n     \n{\"a\":1}     \n");
}

#[test]
fn strip_leaves_string_contents_alone() {
    assert_eq!(stripped("{\"u\":\"http://x\"}"), "{\"u\":\"http://x\"}");
    assert_eq!(stripped("\"a//b\""), "\"a//b\"");
    assert_eq!(stripped("\"/* not a comment */\""), "\"/* not a comment */\"");
}

#[test]
fn strip_handles_escaped_quotes_in_strings() {
    // The escaped quote must not end the string; the // inside stays.
    assert_eq!(stripped("\"a\\\"//b\""), "\"a\\\"//b\"");
    // A string ending in a literal backslash-escape, then a real comment.
    assert_eq!(stripped("\"a\\\\\" // c"), "\"a\\\\\"     ");
}

#[test]
fn strip_blanks_multibyte_comment_text_to_valid_utf8() {
    let out = stripped("// コメント\n1");
    assert!(std::str::from_utf8(out.as_bytes()).is_ok());
    assert_eq!(out.trim_start(), "\n1".trim_start());
}

#[test]
fn strip_unterminated_block_comment_blanks_to_eof() {
    assert_eq!(stripped("1 /* open"), "1        ");
}

#[test]
fn strip_lone_slash_is_untouched() {
    assert_eq!(stripped("1 / 2"), "1 / 2");
}

// --- CLI behavior ------------------------------------------------------------

#[test]
fn jsonc_stdin_line_and_block_comments() {
    let input = "// header\n{\n  \"a\": 1, /* inline */\n  \"b\": \"http://x\"\n}\n";
    let (stdout, _, code) = run_stdin(&["-c", "--jsonc", "."], input);
    assert_eq!(stdout.trim_end(), "{\"a\":1,\"b\":\"http://x\"}");
    assert_eq!(code, Some(0));
}

#[test]
fn without_flag_comments_are_a_parse_error() {
    let (stdout, _, code) = run_stdin(&["-c", "."], "// c\n{\"a\":1}\n");
    assert_eq!(stdout, "");
    assert_eq!(code, Some(5));
}

#[test]
fn jsonc_preserves_input_line_number() {
    // Blanking keeps every newline, so the value on line 3 still reports 3.
    let input = "/* a\n b */\n{\"a\":1}\n";
    let (stdout, _, code) = run_stdin(&["--jsonc", "-c", "input_line_number"], input);
    assert_eq!(stdout.trim_end(), "3");
    assert_eq!(code, Some(0));
}

#[test]
fn jsonc_with_unbuffered_json_stdin_is_rejected() {
    let (stdout, stderr, code) = run_stdin(&["-c", "--jsonc", "--unbuffered", "."], "{}\n");
    assert_eq!(stdout, "");
    assert!(stderr.contains("--jsonc"), "stderr: {stderr}");
    assert_eq!(code, Some(2));
}

#[test]
fn jsonc_file_input() {
    let dir = std::env::temp_dir().join(format!("jq-jit-jsonc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.jsonc");
    std::fs::write(&path, "{\n  // port to bind\n  \"port\": 8080 /* tcp */\n}\n").unwrap();

    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let out = Command::new(jq_jit)
        .args(["-c", "--jsonc", ".port", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim_end(), "8080");
    assert_eq!(out.status.code(), Some(0));

    // --unbuffered with a FILE stays supported — only streaming stdin is not.
    let out = Command::new(jq_jit)
        .args(["-c", "--jsonc", "--unbuffered", ".port", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim_end(), "8080");
    assert_eq!(out.status.code(), Some(0));

    // -n + input reads the same file through the inputs queue.
    let out = Command::new(jq_jit)
        .args(["-c", "-n", "--jsonc", "input.port", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim_end(), "8080");
    assert_eq!(out.status.code(), Some(0));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn jsonc_slurp_and_multiple_documents() {
    let input = "// two docs\n1\n/* between */\n2\n";
    let (stdout, _, code) = run_stdin(&["-c", "--jsonc", "-s", "add"], input);
    assert_eq!(stdout.trim_end(), "3");
    assert_eq!(code, Some(0));
}

#[test]
fn jsonc_raw_input_is_unaffected() {
    // -R reads lines as strings; comment syntax must pass through verbatim.
    let (stdout, _, code) = run_stdin(&["-c", "--jsonc", "-R", "."], "// not json\n");
    assert_eq!(stdout.trim_end(), "\"// not json\"");
    assert_eq!(code, Some(0));
}
