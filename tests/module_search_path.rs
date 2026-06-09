//! Issue #1003: module search-path handling was incomplete.
//!   * The `{search:"..."}` import/include metadata was dropped (for `include`
//!     entirely, and for `import` it was only ever joined onto `-L` dirs, so
//!     with no `-L` it was lost), making `{search}`-resolvable modules
//!     unfindable.
//!   * `get_search_list` returned a static default list, ignoring `-L`.
//!
//! Exact default path strings and realpath canonicalisation are
//! platform/build dependent, so the assertions compute expectations at runtime.

use std::io::Write;
use std::process::{Command, Stdio};

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
fn search_metadata_resolves_modules_without_dash_l() {
    let dir = std::env::temp_dir().join(format!("jqjit_search_{}", std::process::id()));
    let sd = dir.join("sd");
    std::fs::create_dir_all(&sd).unwrap();
    std::fs::write(sd.join("sm.jq"), "def g: 7;").unwrap();
    let sd_s = sd.to_str().unwrap();

    // import with absolute {search}, no -L
    let prog = format!("import \"sm\" as s {{search:\"{sd_s}\"}}; s::g");
    let (out, ok) = run(&["-c", &prog]);
    assert!(ok, "import {{search}} failed: {out:?}");
    assert_eq!(out, "7");

    // include with absolute {search}, no -L
    let prog = format!("include \"sm\" {{search:\"{sd_s}\"}}; g");
    let (out, ok) = run(&["-c", &prog]);
    assert!(ok, "include {{search}} failed: {out:?}");
    assert_eq!(out, "7");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn get_search_list_reflects_dash_l() {
    let dir = std::env::temp_dir().join(format!("jqjit_gsl_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dir_s = dir.to_str().unwrap();
    // jq canonicalises a resolvable -L dir via realpath; mirror that here.
    let canon = std::fs::canonicalize(&dir).unwrap();
    let canon_s = canon.to_str().unwrap();

    let (out, ok) = run(&["-L", dir_s, "-c", "get_search_list"]);
    assert!(ok);
    assert_eq!(out, format!("[\"{canon_s}\"]"));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn get_search_list_defaults_without_dash_l() {
    let (out, ok) = run(&["-c", "get_search_list"]);
    assert!(ok);
    assert_eq!(out, r#"["~/.jq","$ORIGIN/../lib/jq","$ORIGIN/../lib"]"#);
}

#[test]
fn get_search_list_keeps_unresolvable_dash_l_verbatim() {
    let (out, ok) = run(&["-L", "/nonexistent_zzz_jqjit", "-c", "get_search_list"]);
    assert!(ok);
    assert_eq!(out, r#"["/nonexistent_zzz_jqjit"]"#);
}
