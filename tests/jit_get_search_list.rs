//! Issue #1089: the JIT's get_search_list dispatch returned the static
//! compile-time defaults, ignoring `-L` — eval reports the *effective*
//! search list (#1003). The CLI now publishes the `-L` dirs to the runtime
//! so both backends agree.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(filter: &str, extra_args: &[&str], backend_env: &str) -> String {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut cmd = Command::new(jq_jit);
    cmd.args(extra_args).args(["-c", filter]).env(backend_env, "1");
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn jq-jit");
    child.stdin.take().unwrap().write_all(b"0\n").unwrap();
    let out = child.wait_with_output().expect("wait failed");
    String::from_utf8_lossy(&out.stdout).trim_end().to_string()
}

#[test]
fn search_list_honors_lib_dirs_on_jit_path() {
    let lib_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/modules");
    let eval = run("get_search_list", &["-L", lib_dir], "JQJIT_FORCE_INTERPRETER");
    for backend in ["JQJIT_FORCE_CRANELIFT", "JQJIT_FORCE_JITOP_INTERP"] {
        let jit = run("get_search_list", &["-L", lib_dir], backend);
        assert_eq!(jit, eval, "{backend} diverges from eval");
    }
    let count = run("get_search_list | length", &["-L", lib_dir], "JQJIT_FORCE_CRANELIFT");
    assert_eq!(count, "1");
}

#[test]
fn search_list_defaults_without_lib_dirs() {
    let jit = run("get_search_list", &[], "JQJIT_FORCE_CRANELIFT");
    assert_eq!(jit, "[\"~/.jq\",\"$ORIGIN/../lib/jq\",\"$ORIGIN/../lib\"]");
}
