//! Issue #891: `-C` colored output must honour the `JQ_COLORS` environment
//! variable (8 colon-separated SGR slots: null:false:true:numbers:strings:
//! arrays:objects:object-keys), falling back to jq's defaults for missing or
//! invalid input. jq-jit previously ignored it and always used the default
//! palette. Shell out so the env var and `-C`/`-Cc` flags are exercised.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(jq_colors: Option<&str>, args: &[&str], stdin: &[u8]) -> String {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut cmd = Command::new(jq_jit);
    cmd.args(args);
    match jq_colors {
        Some(v) => {
            cmd.env("JQ_COLORS", v);
        }
        None => {
            cmd.env_remove("JQ_COLORS");
        }
    }
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn jq-jit");
    child.stdin.take().unwrap().write_all(stdin).unwrap();
    let out = child.wait_with_output().expect("wait failed").stdout;
    // Render ESC as `E` so the expected strings stay readable.
    String::from_utf8(out)
        .expect("non-utf8 stdout")
        .replace('\u{1b}', "E")
        .trim_end_matches('\n')
        .to_string()
}

#[test]
fn default_palette_unchanged_when_unset() {
    assert_eq!(
        run(None, &["-Cc", "."], b"{\"a\":1}"),
        "E[1;39m{E[0mE[1;34m\"a\"E[0mE[1;39m:E[0mE[0;39m1E[0mE[1;39m}E[0m"
    );
}

#[test]
fn full_palette_is_honored() {
    // null:false:true:numbers:strings:arrays:objects:object-keys
    let colors = "1;30:0;31:0;32:0;33:0;34:0;35:0;36:1;41";
    assert_eq!(
        run(Some(colors), &["-Cc", "."], b"{\"a\":1}"),
        // object=0;36, key=1;41, number=0;33
        "E[0;36m{E[0mE[1;41m\"a\"E[0mE[0;36m:E[0mE[0;33m1E[0mE[0;36m}E[0m"
    );
}

#[test]
fn individual_slots() {
    // number slot (index 3)
    assert_eq!(
        run(Some("1;30:0;30:0;30:0;33:0;30:0;30:0;30:0;30"), &["-Cc", "."], b"5"),
        "E[0;33m5E[0m"
    );
    // null slot (index 0)
    assert_eq!(run(Some("1;31"), &["-Cc", "."], b"null"), "E[1;31mnullE[0m");
    // string slot (index 4)
    assert_eq!(
        run(Some("0;30:0;30:0;30:0;30:1;35:0;30:0;30:0;30"), &["-Cc", "."], b"\"s\""),
        "E[1;35m\"s\"E[0m"
    );
}

#[test]
fn missing_slots_keep_defaults() {
    // Only the null slot is given; the string slot keeps its default 0;32.
    assert_eq!(run(Some("1;31"), &["-Cc", "."], b"\"s\""), "E[0;32m\"s\"E[0m");
}

#[test]
fn invalid_value_falls_back_to_defaults() {
    // A non-digit field makes jq reject the whole variable -> default null 0;90.
    assert_eq!(run(Some("zzz"), &["-Cc", "."], b"null"), "E[0;90mnullE[0m");
}

#[test]
fn empty_container_is_one_colored_span() {
    assert_eq!(run(None, &["-Cc", "."], b"[]"), "E[1;39m[]E[0m");
    assert_eq!(run(None, &["-Cc", "."], b"{}"), "E[1;39m{}E[0m");
}
