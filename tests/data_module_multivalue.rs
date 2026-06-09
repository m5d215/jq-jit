//! Issue #1002: a JSON data module (`import "name" as $var;`) is a JSON text
//! *sequence* — jq reads every whitespace-separated value in the file and binds
//! the array of them. jq-jit wrapped the raw file content in `[...]` and let
//! `fromjson` choke on any multi-value file (`[1 2]`). Verify the array binding
//! across single, multi, object, empty, and nested-array files.

use std::io::Write;
use std::process::{Command, Stdio};

fn import_var(dir: &std::path::Path, module: &str) -> (String, bool) {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let prog = format!("import \"{module}\" as $d; $d");
    let mut child = Command::new(jq_jit)
        .args(["-L", dir.to_str().unwrap(), "-c", &prog])
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
fn data_module_binds_array_of_all_values() {
    let dir = std::env::temp_dir().join(format!("jqjit_data_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cases = [
        ("two", "1 2", "[1,2]"),
        ("one", "42", "[42]"),
        ("objs", "{\"a\":1}\n{\"b\":2}\n", r#"[{"a":1},{"b":2}]"#),
        ("empty", "", "[]"),
        ("arr", "[1,2,3]", "[[1,2,3]]"),
        ("mix", "1 2 3\n4", "[1,2,3,4]"),
        ("strs", "\"x\"\n\"y\"", r#"["x","y"]"#),
    ];
    for (name, content, expected) in cases {
        std::fs::write(dir.join(format!("{name}.json")), content).unwrap();
        let (out, ok) = import_var(&dir, name);
        assert!(ok, "import of `{name}` failed: {out:?}");
        assert_eq!(out, expected, "data module `{name}` (content {content:?})");
    }
    std::fs::remove_dir_all(&dir).ok();
}
