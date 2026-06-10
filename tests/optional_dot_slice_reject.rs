//! Issue #1023: jq's postfix dotted bracket form (`.a.[...]`) only carries an
//! index or iterate — the slice form (`.a.[lo:hi]`) is a compile error in
//! jq 1.8, while the undotted `.a[lo:hi]` and the bare leading `.[lo:hi]`
//! both parse. The regression harness skips exit-3 cases, so assert the
//! compile rejection by shelling out.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(filter: &str) -> i32 {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .arg("-c")
        .arg(filter)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn jq-jit");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"{\"a\":[1,2,3]}")
        .unwrap();
    child
        .wait_with_output()
        .expect("wait failed")
        .status
        .code()
        .expect("no exit code")
}

#[test]
fn dotted_bracket_slice_is_a_compile_error() {
    for f in [
        ".a.[0:1]",
        ".a.[1:]",
        ".a.[:2]",
        ".a.b.[0:1]",
        ".[0:1].[0:1]",
        ".a.[0:1]?",
    ] {
        assert_eq!(run(f), 3, "{} must be rejected at compile time", f);
    }
}

#[test]
fn undotted_slice_and_dotted_index_iterate_still_parse() {
    for f in [".a[0:1]", ".a | .[0:1]", ".a.[0]", ".a.[]", ".a.[]?", ".a.[0]?"] {
        assert_eq!(run(f), 0, "{} must keep parsing and running", f);
    }
}
