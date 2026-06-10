//! Backend self-diff harness (issue #1059 Phase 1): run every regression
//! case through BOTH backends of the shared JitOp lowering — the direct
//! JitOp interpreter (`JQJIT_FORCE_JITOP_INTERP=1`) and the Cranelift
//! codegen (`JQJIT_FORCE_CRANELIFT=1`) — and assert identical stdout +
//! exit-code class.
//!
//! The two knobs configure *identical* routing (raw-byte fast paths off,
//! typed fast path off, non-flattenable filters falling back to eval) and
//! differ only in which backend executes the flattened JitOp sequence, so
//! any divergence here is a true backend bug: one of the two executions of
//! the same linear op sequence is wrong.
//!
//! This complements `tests/selfdiff_jit_interp.rs` (default dispatch vs
//! tree-walking eval): that harness pins the lowering + fast-path layers
//! against eval semantics, this one pins the two lowering backends against
//! each other.
//!
//! Set `JITOP_BACKEND_DIFF_LIMIT=N` to truncate the corpus during local
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

/// Backend selector for one subprocess run.
#[derive(Clone, Copy)]
enum Backend {
    JitopInterp,
    Cranelift,
}

fn run_once(bin: &str, filter: &str, input: &str, backend: Backend) -> Option<RunOutput> {
    let lib_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/modules");
    let mut cmd = Command::new(bin);
    cmd.arg("-L").arg(lib_dir).arg("-c").arg(filter);
    cmd.env_remove("JQJIT_FORCE_INTERPRETER");
    cmd.env_remove("JQJIT_FORCE_JITOP_INTERP");
    cmd.env_remove("JQJIT_FORCE_CRANELIFT");
    match backend {
        Backend::JitopInterp => { cmd.env("JQJIT_FORCE_JITOP_INTERP", "1"); }
        Backend::Cranelift => { cmd.env("JQJIT_FORCE_CRANELIFT", "1"); }
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

/// Cases the harness flags but does not fail on. Both sides execute the
/// same JitOp sequence, so this list is expected to stay empty — a true
/// backend divergence is a bug in one of the two executions and should be
/// fixed, not allowlisted. The mechanism is kept for parity with the other
/// self-diff harnesses in case a platform-specific divergence ever needs a
/// documented waiver.
const KNOWN_DIVERGENCES: &[usize] = &[];

/// Per-case verdict, produced in parallel and folded sequentially so the
/// counters and reporting stay identical across the self-diff harnesses.
enum Outcome {
    /// Agreed (cranelift == jitop interp). `Some(line)` if the case is on the
    /// allowlist yet agreed anyway — counts as a pass *and* flags a stale entry.
    Pass(Option<usize>),
    KnownDiverged,
    SpawnFail(String),
    Fail(String),
}

#[test]
fn jitop_interp_vs_cranelift_self_diff() {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");

    let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/regression.test");
    let content = std::fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", corpus_path.display(), e));
    let mut cases = parse_corpus(&content);
    assert!(!cases.is_empty(), "regression corpus is empty");

    if let Ok(limit) = std::env::var("JITOP_BACKEND_DIFF_LIMIT") {
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

    // Each case spawns two isolated jq-jit subprocesses (Cranelift vs JitOp
    // interpreter); no shared in-process state, so fan out across cores and
    // fold the verdicts back in input order.
    let outcomes = common::parallel::par_map(&cases, |case| {
        let known = KNOWN_DIVERGENCES.contains(&case.line);
        let cranelift = run_once(jq_jit, &case.filter, &case.input, Backend::Cranelift);
        let jitop = run_once(jq_jit, &case.filter, &case.input, Backend::JitopInterp);

        let (Some(cranelift), Some(jitop)) = (cranelift, jitop) else {
            return Outcome::SpawnFail(format!(
                "  line {}: spawn failure\n    filter: {}\n    input:  {}",
                case.line, case.filter, case.input
            ));
        };

        if cranelift.is_error && jitop.is_error {
            return Outcome::Pass(known.then_some(case.line));
        }
        if cranelift.is_error != jitop.is_error {
            if known {
                return Outcome::KnownDiverged;
            }
            return Outcome::Fail(format!(
                "  line {}: error-class mismatch (cranelift error={}, jitop error={})\n    filter: {}\n    input:  {}\n    cranelift: {}\n    jitop:     {}",
                case.line, cranelift.is_error, jitop.is_error, case.filter, case.input,
                cranelift.stdout.trim(), jitop.stdout.trim()
            ));
        }

        let cranelift_norm = normalize(&cranelift.stdout);
        let jitop_norm = normalize(&jitop.stdout);
        if cranelift_norm == jitop_norm {
            Outcome::Pass(known.then_some(case.line))
        } else if known {
            Outcome::KnownDiverged
        } else {
            Outcome::Fail(format!(
                "  line {}: value mismatch\n    filter: {}\n    input:  {}\n    cranelift: {}\n    jitop:     {}",
                case.line, case.filter, case.input, cranelift_norm, jitop_norm
            ))
        }
    });

    for outcome in outcomes {
        match outcome {
            Outcome::Pass(maybe_line) => {
                pass += 1;
                if let Some(line) = maybe_line {
                    unexpected_pass.push(line);
                }
            }
            Outcome::KnownDiverged => known_diverged += 1,
            Outcome::SpawnFail(msg) => {
                spawn_fail += 1;
                failures.push(msg);
            }
            Outcome::Fail(msg) => {
                fail += 1;
                failures.push(msg);
            }
        }
    }

    eprintln!();
    eprintln!("=== JitOp interpreter vs Cranelift backend self-diff ===");
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
        "{} backend self-diff divergences out of {}",
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
