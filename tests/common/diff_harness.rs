//! Spawn `jq-jit` and reference `jq-1.8.x` and capture their output.
//!
//! The reference binary is resolved in this order: `$JQ_BIN` → Homebrew
//! canonical path (`/opt/homebrew/opt/jq/bin/jq`) → `jq` in `$PATH`.
//! `--version` must report `jq-1.8.x`.
//!
//! Tests that depend on jq-1.8.x should call [`require_jq`] at the top of
//! their `#[test]` body. It panics on CI (so missing-binary configuration
//! errors fail loud) and returns `None` locally (so `cargo test` is usable
//! without 1.8.1 installed).

use std::ffi::OsStr;
use std::process::Command;
use std::time::Duration;

#[derive(Debug)]
pub struct RunOutput {
    pub stdout: String,
    pub is_error: bool,
}

/// Run `bin` with `-c <filter>`, piping `input` to stdin. Trailing newline
/// is appended to stdin (jq tolerates it, and some inputs come from
/// generators that don't).
pub fn run_filter(bin: &str, filter: &str, input: &str, timeout: Duration) -> Option<RunOutput> {
    run_with_args(bin, ["-c", filter], input, timeout)
}

/// Run `bin -c -f <script_path>`, piping `input` to stdin. Used by the
/// real-world script corpus where filters live in `.jq` files.
pub fn run_script_file(
    bin: &str,
    script_path: &std::path::Path,
    input: &str,
    timeout: Duration,
) -> Option<RunOutput> {
    run_with_args(
        bin,
        ["-c".as_ref(), "-f".as_ref(), script_path.as_os_str()],
        input,
        timeout,
    )
}

fn run_with_args<I, S>(bin: &str, args: I, stdin: &str, timeout: Duration) -> Option<RunOutput>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut cmd = Command::new(bin);
    cmd.args(args);
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn().ok()?;
    {
        use std::io::Write;
        let mut h = child.stdin.take()?;
        let _ = h.write_all(stdin.as_bytes());
        let _ = h.write_all(b"\n");
    }

    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = child.wait_with_output().ok()?;
                let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    if let Some(sig) = status.signal() {
                        return Some(RunOutput {
                            stdout: format!(
                                "<killed by signal {}> stderr: {}",
                                sig,
                                stderr.trim()
                            ),
                            is_error: true,
                        });
                    }
                }
                return Some(RunOutput {
                    stdout,
                    is_error: !status.success(),
                });
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Some(RunOutput {
                        stdout: "<timeout>".to_string(),
                        is_error: true,
                    });
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => return None,
        }
    }
}

/// Locate a jq-1.8.x binary, or return `None`.
pub fn resolve_jq(test_label: &str) -> Option<String> {
    let candidates: Vec<String> = std::env::var("JQ_BIN")
        .ok()
        .into_iter()
        .chain(std::iter::once(
            "/opt/homebrew/opt/jq/bin/jq".to_string(),
        ))
        .chain(std::iter::once("jq".to_string()))
        .collect();

    for cand in &candidates {
        let Ok(output) = Command::new(cand).arg("--version").output() else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let ver = String::from_utf8_lossy(&output.stdout);
        let ver = ver.trim();
        if ver.starts_with("jq-1.8.") {
            return Some(cand.clone());
        }
        eprintln!(
            "SKIP {}: candidate `{}` is `{}`, need jq-1.8.x",
            test_label, cand, ver
        );
    }
    None
}

/// Resolve a jq-1.8.x binary or, if none is found, panic on CI / log+return
/// `None` locally. The intended usage in a test body is:
///
/// ```ignore
/// let Some(jq) = require_jq("my_test") else { return; };
/// ```
pub fn require_jq(test_label: &str) -> Option<String> {
    if let Some(j) = resolve_jq(test_label) {
        return Some(j);
    }
    let msg = "no jq-1.8.x binary found. Set JQ_BIN to a jq-1.8.x binary.";
    if std::env::var_os("CI").is_some() {
        panic!("{}: {}", test_label, msg);
    }
    eprintln!("SKIP {}: {}", test_label, msg);
    None
}

/// Path to the freshly-built `jq-jit` binary. Wraps the cargo-injected
/// `CARGO_BIN_EXE_jq-jit` env var so tests don't need to remember it.
pub fn jq_jit_path() -> &'static str {
    env!("CARGO_BIN_EXE_jq-jit")
}
