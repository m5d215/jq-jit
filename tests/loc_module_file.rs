//! Issue #1004: `$__loc__` inside a function defined in an imported/included
//! module must report the module's source file in `.file`, not "<top-level>".
//! The exact path is canonicalised (realpath), so the test computes the
//! expected path at runtime; the `line` field was already correct.

use std::io::Write;
use std::process::{Command, Stdio};

/// JSON-escape a filesystem path for embedding in an expected output
/// string. Windows canonical paths (`\\?\C:\...`) contain backslashes,
/// which the `$__loc__` output renders as `\\` — embedding the raw path
/// made these assertions Windows-only failures (caught by release.yml,
/// the only workflow that tests on Windows).
fn json_escaped(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn run(args: &[&str]) -> (String, bool) {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn jq-jit");
    child.stdin.take().unwrap().write_all(b"null").unwrap();
    let out = child.wait_with_output().expect("wait failed");
    (
        String::from_utf8_lossy(&out.stdout).trim_end().to_string(),
        out.status.success(),
    )
}

#[test]
fn loc_file_attributes_to_module() {
    let dir = std::env::temp_dir().join(format!("jqjit_loc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("w.jq"), "def w: $__loc__;").unwrap();
    let dir_s = dir.to_str().unwrap();
    let canon = std::fs::canonicalize(dir.join("w.jq")).unwrap();
    let canon_s = json_escaped(canon.to_str().unwrap());
    let expected = format!(r#"{{"file":"{canon_s}","line":1}}"#);

    // import
    let (out, ok) = run(&["-L", dir_s, "-c", "import \"w\" as m; m::w"]);
    assert!(ok, "import failed: {out:?}");
    assert_eq!(out, expected, "import module $__loc__.file");

    // include
    let (out, ok) = run(&["-L", dir_s, "-c", "include \"w\"; w"]);
    assert!(ok, "include failed: {out:?}");
    assert_eq!(out, expected, "include module $__loc__.file");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn loc_file_stays_top_level_for_program() {
    let (out, ok) = run(&["-c", "$__loc__"]);
    assert!(ok);
    assert_eq!(out, r#"{"file":"<top-level>","line":1}"#);

    // shorthand object pattern path goes through the second Expr::Loc site
    let (out, ok) = run(&["-c", "{$__loc__} | .__loc__"]);
    assert!(ok);
    assert_eq!(out, r#"{"file":"<top-level>","line":1}"#);
}

#[test]
fn loc_line_correct_for_later_module_def() {
    let dir = std::env::temp_dir().join(format!("jqjit_loc2_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // `g` is defined on line 2, so $__loc__.line must be 2.
    std::fs::write(dir.join("m2.jq"), "def f: 1;\ndef g: $__loc__;").unwrap();
    let dir_s = dir.to_str().unwrap();
    let canon = std::fs::canonicalize(dir.join("m2.jq")).unwrap();
    let canon_s = json_escaped(canon.to_str().unwrap());

    let (out, ok) = run(&["-L", dir_s, "-c", "import \"m2\" as m; m::g"]);
    assert!(ok, "import failed: {out:?}");
    assert_eq!(out, format!(r#"{{"file":"{canon_s}","line":2}}"#));

    std::fs::remove_dir_all(&dir).ok();
}
