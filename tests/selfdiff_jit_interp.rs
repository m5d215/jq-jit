//! Self-diff harness (issue #323): run every regression case through the JIT
//! / fast-path dispatch *and* through the generic tree-walking interpreter,
//! and assert identical stdout + exit-code class.
//!
//! Differential testing against `jq-1.8.1` (`tests/diff_corpus.rs`) catches
//! external divergences only. This harness catches the *internal* class —
//! the JIT path and the interpreter path inside jq-jit drifting apart on the
//! same filter — without depending on an external `jq` binary.
//!
//! Post-#1059 role: the shared JitOp lowering covers the full corpus and
//! `tests/selfdiff_jitop_backend.rs` is the primary backend diff
//! (jitop-interp vs Cranelift over identical op sequences). Eval survives
//! as the delegated-op engine and as the A/B-preferred route for
//! delegate-dominant / interleaved-IO programs — this harness pins those
//! delegation and routing boundaries against eval semantics.
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
    cmd.env_remove("JQJIT_FORCE_JITOP_INTERP");
    cmd.env_remove("JQJIT_FORCE_CRANELIFT");
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
/// case to the underlying-divergence note that explains it. Entries are
/// keyed by `(filter, input)` content — not `tests/regression.test` line
/// numbers — so the corpus can be edited anywhere without renumbering the
/// allowlist (#1026). A content key matches every corpus case with that
/// exact filter and input line (behaviour is deterministic per content, so
/// duplicates share one verdict).
/// New divergences are NOT silently allowed — they have to be added here with
/// rationale, which is the point: the allowlist is the audit trail of "we
/// know about this, here's why we haven't fixed it yet."
///
/// Currently empty (#1149). The last entries were `tojson` on literals that
/// overflow `f64` (`1e1000`): the raw-byte fast path lexed the digits and
/// emitted the canonical literal while the interpreter saturated to
/// `±1.7976931348623157e+308`. Both paths now keep the preserved repr, so the
/// two agree — and they agree with jq 1.8.2 as well.
const KNOWN_DIVERGENCES: &[(&str, &str)] = &[];

/// Index of the allowlist entry matching this case's content, if any.
fn known_divergence_idx(case: &Case) -> Option<usize> {
    KNOWN_DIVERGENCES
        .iter()
        .position(|&(f, i)| f == case.filter && i == case.input)
}

/// Per-case verdict, produced in parallel and folded sequentially so the
/// counters and reporting stay identical to the original serial loop.
enum Outcome {
    /// Agreed (jit == interp). `Some(idx)` if the case matches allowlist
    /// entry `idx` yet agreed anyway — counts as a pass *and* flags a stale
    /// entry.
    Pass(Option<usize>),
    KnownDiverged,
    SpawnFail(String),
    Fail(String),
}

#[test]
fn jit_vs_interpreter_self_diff() {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");

    let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/regression.test");
    let content = std::fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", corpus_path.display(), e));
    let mut cases = parse_corpus(&content);
    assert!(!cases.is_empty(), "regression corpus is empty");

    let mut limited = false;
    if let Ok(limit) = std::env::var("JIT_INTERP_DIFF_LIMIT") {
        if let Ok(n) = limit.parse::<usize>() {
            cases.truncate(n);
            limited = true;
        }
    }

    // Every allowlist entry must match at least one corpus case, or it is
    // dead weight (e.g. the case was rewritten without updating the entry).
    // Skipped under JIT_INTERP_DIFF_LIMIT — truncation drops cases.
    if !limited {
        let dangling: Vec<&(&str, &str)> = KNOWN_DIVERGENCES
            .iter()
            .filter(|(f, i)| !cases.iter().any(|c| c.filter == *f && c.input == *i))
            .collect();
        assert!(
            dangling.is_empty(),
            "KNOWN_DIVERGENCES entries match no corpus case (filter, input): {:?}",
            dangling
        );
    }

    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut spawn_fail = 0usize;
    let mut known_diverged = 0usize;
    let mut unexpected_pass: Vec<usize> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    // Each case spawns two isolated jq-jit subprocesses (JIT vs forced
    // interpreter); no shared in-process state, so fan out across cores and
    // fold the verdicts back in input order.
    let outcomes = common::parallel::par_map(&cases, |case| {
        let known = known_divergence_idx(case);
        let jit = run_once(jq_jit, &case.filter, &case.input, false);
        let interp = run_once(jq_jit, &case.filter, &case.input, true);

        let (Some(jit), Some(interp)) = (jit, interp) else {
            return Outcome::SpawnFail(format!(
                "  line {}: spawn failure\n    filter: {}\n    input:  {}",
                case.line, case.filter, case.input
            ));
        };

        if jit.is_error && interp.is_error {
            return Outcome::Pass(known);
        }
        if jit.is_error != interp.is_error {
            if known.is_some() {
                return Outcome::KnownDiverged;
            }
            return Outcome::Fail(format!(
                "  line {}: error-class mismatch (jit error={}, interp error={})\n    filter: {}\n    input:  {}\n    jit:    {}\n    interp: {}",
                case.line, jit.is_error, interp.is_error, case.filter, case.input,
                jit.stdout.trim(), interp.stdout.trim()
            ));
        }

        let jit_norm = normalize(&jit.stdout);
        let interp_norm = normalize(&interp.stdout);
        if jit_norm == interp_norm {
            Outcome::Pass(known)
        } else if known.is_some() {
            Outcome::KnownDiverged
        } else {
            Outcome::Fail(format!(
                "  line {}: value mismatch\n    filter: {}\n    input:  {}\n    jit:    {}\n    interp: {}",
                case.line, case.filter, case.input, jit_norm, interp_norm
            ))
        }
    });

    for outcome in outcomes {
        match outcome {
            Outcome::Pass(maybe_idx) => {
                pass += 1;
                if let Some(idx) = maybe_idx {
                    unexpected_pass.push(idx);
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
    unexpected_pass.sort_unstable();
    unexpected_pass.dedup();
    let stale: Vec<&(&str, &str)> = unexpected_pass
        .iter()
        .map(|&idx| &KNOWN_DIVERGENCES[idx])
        .collect();
    assert!(
        stale.is_empty(),
        "KNOWN_DIVERGENCES is stale — these entries now agree and should be removed (filter, input): {:?}",
        stale
    );
}
