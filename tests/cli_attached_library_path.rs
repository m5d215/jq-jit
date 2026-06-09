//! Issue #1001: jq-jit must accept the GNU-style attached short-option form
//! `-L<path>` (no space) for the module library path, like jq, in addition to
//! the separated `-L <path>` / `--library-path <path>` forms.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(args: &[&str], cwd: Option<&std::path::Path>) -> (String, bool) {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut cmd = Command::new(jq_jit);
    cmd.args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let mut child = cmd.spawn().expect("failed to spawn jq-jit");
    child.stdin.take().unwrap().write_all(b"null").unwrap();
    let out = child.wait_with_output().expect("wait failed");
    (
        String::from_utf8_lossy(&out.stdout).trim_end().to_string(),
        out.status.success(),
    )
}

#[test]
fn attached_and_separated_library_path_forms() {
    let dir = std::env::temp_dir().join(format!("jqjit_l_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("m.jq"), "def greet: \"hi\";").unwrap();
    let dir_s = dir.to_str().unwrap();
    let prog = "import \"m\" as m; m::greet";

    // Attached `-L<path>` (the bug).
    let attached = format!("-L{dir_s}");
    let (out, ok) = run(&[&attached, "-c", prog], None);
    assert!(ok, "attached -L<path> failed: {out:?}");
    assert_eq!(out, "\"hi\"");

    // Separated `-L <path>` still works.
    let (out, ok) = run(&["-L", dir_s, "-c", prog], None);
    assert!(ok, "separated -L <path> failed");
    assert_eq!(out, "\"hi\"");

    // Long `--library-path <path>` still works.
    let (out, ok) = run(&["--library-path", dir_s, "-c", prog], None);
    assert!(ok, "--library-path failed");
    assert_eq!(out, "\"hi\"");

    // Attached relative `-L.` resolved against the cwd.
    let (out, ok) = run(&["-L.", "-c", prog], Some(&dir));
    assert!(ok, "attached -L. failed: {out:?}");
    assert_eq!(out, "\"hi\"");

    std::fs::remove_dir_all(&dir).ok();
}
