//! Self-diff harness (issue #323): run every regression case through the JIT
//! / fast-path dispatch *and* through the generic tree-walking interpreter,
//! and assert identical stdout + exit-code class.
//!
//! Differential testing against `jq-1.8.1` (`tests/diff_corpus.rs`) catches
//! external divergences only. This harness catches the *internal* class —
//! the JIT path and the interpreter path inside jq-jit drifting apart on the
//! same filter — without depending on an external `jq` binary.
//!
//! The runtime knob is `JQJIT_FORCE_INTERPRETER=1`: the binary then disables
//! all raw-byte fast paths, skips JIT compilation, and routes
//! `Filter::execute` / `Filter::execute_cb` through the generic interpreter
//! (see `jq_jit::interpreter::set_force_interpreter`).
//!
//! Set `JIT_INTERP_DIFF_LIMIT=N` to truncate the corpus during local
//! development; the default runs every case in `tests/regression.test`.

mod common;

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use common::json_normalize::{normalize_value, serialize_sorted};

struct Case {
    filter: String,
    input: String,
    line: usize,
}

fn parse_corpus(content: &str) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut block: Vec<(String, usize)> = Vec::new();

    let flush = |block: &mut Vec<(String, usize)>, cases: &mut Vec<Case>| {
        if block.len() >= 3 {
            let (filter, line) = block[0].clone();
            let input = block[1].0.clone();
            cases.push(Case { filter, input, line });
        }
        block.clear();
    };

    for (idx, line) in content.lines().enumerate() {
        let line_no = idx + 1;
        if line.trim_start().starts_with('#') {
            continue;
        }
        if line.trim().is_empty() {
            flush(&mut block, &mut cases);
            continue;
        }
        block.push((line.to_string(), line_no));
    }
    flush(&mut block, &mut cases);
    cases
}

#[derive(Debug)]
struct RunOutput {
    stdout: String,
    is_error: bool,
}

fn run_once(bin: &str, filter: &str, input: &str, force_interp: bool) -> Option<RunOutput> {
    let lib_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/modules");
    let mut cmd = Command::new(bin);
    cmd.arg("-L").arg(lib_dir).arg("-c").arg(filter);
    cmd.env_remove("JQJIT_FORCE_INTERPRETER");
    if force_interp {
        cmd.env("JQJIT_FORCE_INTERPRETER", "1");
    }
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::null());

    let mut child = cmd.spawn().ok()?;
    {
        use std::io::Write;
        let mut stdin = child.stdin.take()?;
        let _ = stdin.write_all(input.as_bytes());
        let _ = stdin.write_all(b"\n");
    }

    let timeout = Duration::from_secs(10);
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = child.wait_with_output().ok()?;
                let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    if let Some(sig) = status.signal() {
                        return Some(RunOutput {
                            stdout: format!("<killed by signal {}>", sig),
                            is_error: true,
                        });
                    }
                }
                let is_error = !status.success();
                return Some(RunOutput { stdout, is_error });
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Some(RunOutput {
                        stdout: "<timeout after 10s>".to_string(),
                        is_error: true,
                    });
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(_) => return None,
        }
    }
}

/// Lossy variant of `common::json_normalize::normalize`: filters that
/// emit non-JSON lines (raw error text via `error("…")`) fall through to
/// the raw string branch so their output still compares directly.
fn normalize(output: &str) -> String {
    let mut lines = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(val) => lines.push(serialize_sorted(&normalize_value(val))),
            Err(_) => lines.push(trimmed.to_string()),
        }
    }
    lines.join("\n")
}

/// Cases the harness flags but does not fail on. Each entry pins a specific
/// regression-test line to the underlying-divergence note that explains it.
/// New divergences are NOT silently allowed — they have to be added here with
/// rationale, which is the point: the allowlist is the audit trail of "we
/// know about this, here's why we haven't fixed it yet."
///
/// Current entries (#323):
/// - `tojson` on numbers that overflow `f64` (`1e1000` → `±INFINITY`):
///   the raw-byte fast path on stdin lexes the digits and emits the
///   canonicalised literal (`"1E+1000"`); the interpreter routes through
///   `Value::Num(f64, repr)` and `push_jq_number_str` saturates non-finite
///   values to `±1.7976931348623157e+308`. Plumbing the original `repr`
///   through `value_to_json_tojson` is the obvious local fix, but doing so
///   without also flipping `have_decnum` to `true` breaks the upstream
///   `tests/official/jq.test` decnum consistency check (#443). Tracked
///   in #415.
const KNOWN_DIVERGENCES: &[usize] = &[2197, 2202, 2207, 2333, 2338];

#[test]
fn jit_vs_interpreter_self_diff() {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");

    let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/regression.test");
    let content = std::fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", corpus_path.display(), e));
    let mut cases = parse_corpus(&content);
    assert!(!cases.is_empty(), "regression corpus is empty");

    if let Ok(limit) = std::env::var("JIT_INTERP_DIFF_LIMIT") {
        if let Ok(n) = limit.parse::<usize>() {
            cases.truncate(n);
        }
    }

    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut spawn_fail = 0usize;
    let mut known_diverged = 0usize;
    let mut unexpected_pass: Vec<usize> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for case in &cases {
        let known = KNOWN_DIVERGENCES.contains(&case.line);
        let jit = run_once(jq_jit, &case.filter, &case.input, false);
        let interp = run_once(jq_jit, &case.filter, &case.input, true);

        let (Some(jit), Some(interp)) = (jit, interp) else {
            spawn_fail += 1;
            failures.push(format!(
                "  line {}: spawn failure\n    filter: {}\n    input:  {}",
                case.line, case.filter, case.input
            ));
            continue;
        };

        if jit.is_error && interp.is_error {
            if known { unexpected_pass.push(case.line); }
            pass += 1;
            continue;
        }
        if jit.is_error != interp.is_error {
            if known {
                known_diverged += 1;
                continue;
            }
            fail += 1;
            failures.push(format!(
                "  line {}: error-class mismatch (jit error={}, interp error={})\n    filter: {}\n    input:  {}\n    jit:    {}\n    interp: {}",
                case.line, jit.is_error, interp.is_error, case.filter, case.input,
                jit.stdout.trim(), interp.stdout.trim()
            ));
            continue;
        }

        let jit_norm = normalize(&jit.stdout);
        let interp_norm = normalize(&interp.stdout);
        if jit_norm == interp_norm {
            if known { unexpected_pass.push(case.line); }
            pass += 1;
        } else if known {
            known_diverged += 1;
        } else {
            fail += 1;
            failures.push(format!(
                "  line {}: value mismatch\n    filter: {}\n    input:  {}\n    jit:    {}\n    interp: {}",
                case.line, case.filter, case.input, jit_norm, interp_norm
            ));
        }
    }

    eprintln!();
    eprintln!("=== JIT vs interpreter self-diff ===");
    eprintln!("PASS:        {}", pass);
    eprintln!("FAIL:        {}", fail);
    eprintln!("SPAWN:       {}", spawn_fail);
    eprintln!("KNOWN-DIV:   {}", known_diverged);
    eprintln!("TOTAL:       {}", cases.len());

    if !failures.is_empty() {
        eprintln!();
        eprintln!("=== Divergences ===");
        for f in &failures {
            eprintln!("{}", f);
        }
    }

    assert_eq!(
        fail + spawn_fail,
        0,
        "{} self-diff divergences out of {}",
        fail + spawn_fail,
        cases.len()
    );

    // If a previously-known-divergent case now agrees, the allowlist entry is
    // stale: shrinking the list is the goal, so flag it loudly.
    assert!(
        unexpected_pass.is_empty(),
        "KNOWN_DIVERGENCES is stale — these line(s) now agree and should be removed: {:?}",
        unexpected_pass
    );
}
