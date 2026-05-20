//! 4-way layer-pinned self-diff harness (issue #685): run every regression
//! case through four configurations of the internal fast-path stack and
//! assert identical output. When the diff disagrees, the matrix points at
//! the offending fast-path layer in `docs/maintenance.md` §2.
//!
//! | config       | raw-byte | simplify_expr | JIT |
//! |--------------|----------|---------------|-----|
//! | baseline     | on       | on            | on  |
//! | no-raw-byte  | off      | on            | on  |
//! | no-simplify  | on       | off           | on  |
//! | pure-interp  | off      | off           | off |
//!
//! `JQJIT_DISABLE_RAW_BYTE=1` (issue #685) gates layer (a) at the
//! `use_raw_fast_paths` check in `src/bin/jq-jit.rs`.
//! `JQJIT_DISABLE_SIMPLIFY=1` gates layer (b) at the top of `simplify_expr`
//! in `src/interpreter.rs`. `JQJIT_FORCE_INTERPRETER=1` (issue #323) gates
//! (a)+(d) by routing `Filter::execute*` through the generic interpreter.
//!
//! Set `JIT_INTERP_DIFF_LIMIT=N` to truncate the corpus during local
//! development; the default runs every case in `tests/regression.test`,
//! the same convention as `tests/selfdiff_jit_interp.rs`.
//!
//! This harness does not replace `tests/selfdiff_jit_interp.rs` — that one
//! tests the 2-way diff and ships a `KNOWN_DIVERGENCES` allowlist tied to
//! specific upstream issues. The 4-way harness uses the same allowlist
//! semantics so a known 2-way divergence does not double-fail here.

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Config {
    Baseline,
    NoRawByte,
    NoSimplify,
    PureInterp,
}

impl Config {
    fn label(self) -> &'static str {
        match self {
            Config::Baseline => "baseline",
            Config::NoRawByte => "no-raw-byte",
            Config::NoSimplify => "no-simplify",
            Config::PureInterp => "pure-interp",
        }
    }

    /// Env vars that switch this config on. Always sets/clears all three
    /// knobs so a leaked env from the caller can't poison the matrix.
    fn env(self) -> [(&'static str, Option<&'static str>); 3] {
        match self {
            Config::Baseline => [
                ("JQJIT_DISABLE_RAW_BYTE", None),
                ("JQJIT_DISABLE_SIMPLIFY", None),
                ("JQJIT_FORCE_INTERPRETER", None),
            ],
            Config::NoRawByte => [
                ("JQJIT_DISABLE_RAW_BYTE", Some("1")),
                ("JQJIT_DISABLE_SIMPLIFY", None),
                ("JQJIT_FORCE_INTERPRETER", None),
            ],
            Config::NoSimplify => [
                ("JQJIT_DISABLE_RAW_BYTE", None),
                ("JQJIT_DISABLE_SIMPLIFY", Some("1")),
                ("JQJIT_FORCE_INTERPRETER", None),
            ],
            Config::PureInterp => [
                ("JQJIT_DISABLE_RAW_BYTE", Some("1")),
                ("JQJIT_DISABLE_SIMPLIFY", Some("1")),
                ("JQJIT_FORCE_INTERPRETER", Some("1")),
            ],
        }
    }
}

const CONFIGS: &[Config] = &[
    Config::Baseline,
    Config::NoRawByte,
    Config::NoSimplify,
    Config::PureInterp,
];

#[derive(Debug)]
struct RunOutput {
    stdout: String,
    is_error: bool,
}

fn run_once(bin: &str, filter: &str, input: &str, config: Config) -> Option<RunOutput> {
    let lib_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/modules");
    let mut cmd = Command::new(bin);
    cmd.arg("-L").arg(lib_dir).arg("-c").arg(filter);
    for (key, val) in config.env() {
        match val {
            Some(v) => { cmd.env(key, v); }
            None => { cmd.env_remove(key); }
        }
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

/// Lossy variant of `common::json_normalize::normalize` — matches the
/// semantics in `selfdiff_jit_interp.rs` so both harnesses treat
/// raw-error stderr-shaped stdout lines the same way.
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

/// Cases that diverge in the underlying 2-way self-diff
/// (`tests/selfdiff_jit_interp.rs`). The 4-way harness inherits these so
/// known upstream-tracked divergences do not double-fail. Keep in sync
/// with `selfdiff_jit_interp.rs::KNOWN_DIVERGENCES`.
const KNOWN_DIVERGENCES: &[usize] = &[2197, 2202, 2207, 2333, 2338];

#[test]
fn layer_pinned_self_diff() {
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

        // Run all four configs. Baseline is the oracle every other config
        // is compared against.
        let mut runs: Vec<(Config, RunOutput)> = Vec::with_capacity(CONFIGS.len());
        let mut spawn_failed = false;
        for &config in CONFIGS {
            match run_once(jq_jit, &case.filter, &case.input, config) {
                Some(out) => runs.push((config, out)),
                None => { spawn_failed = true; break; }
            }
        }
        if spawn_failed {
            spawn_fail += 1;
            failures.push(format!(
                "  line {}: spawn failure\n    filter: {}\n    input:  {}",
                case.line, case.filter, case.input
            ));
            continue;
        }

        // Per-config normalized output (stdout) and error class. The
        // oracle is config[0] (Baseline).
        let normalized: Vec<(Config, String, bool)> = runs.iter()
            .map(|(c, r)| (*c, normalize(&r.stdout), r.is_error))
            .collect();
        let (_, ref oracle_out, oracle_err) = normalized[0];

        // Detect any disagreement.
        let mut disagreeing: Vec<&(Config, String, bool)> = Vec::new();
        for entry in normalized.iter().skip(1) {
            let (_, ref out, err) = *entry;
            if out != oracle_out || err != oracle_err {
                disagreeing.push(entry);
            }
        }

        if disagreeing.is_empty() {
            if known { unexpected_pass.push(case.line); }
            pass += 1;
            continue;
        }
        if known {
            known_diverged += 1;
            continue;
        }

        fail += 1;
        let mut buf = String::new();
        buf.push_str(&format!(
            "  line {}: layer divergence\n    filter: {}\n    input:  {}\n",
            case.line, case.filter, case.input,
        ));
        buf.push_str(&format!(
            "    baseline    (error={}): {}\n",
            oracle_err, oracle_out,
        ));
        for &&(config, ref out, err) in disagreeing.iter() {
            buf.push_str(&format!(
                "    {:<11} (error={}): {}\n",
                config.label(), err, out,
            ));
        }
        // Layer-localisation hint: which layer is implicated by the
        // disagreeing config set?
        let configs_off: Vec<&'static str> = disagreeing.iter()
            .map(|(c, _, _)| c.label())
            .collect();
        buf.push_str(&format!(
            "    → disagreeing configs: [{}]\n",
            configs_off.join(", "),
        ));
        buf.push_str(&format!(
            "      (only `{}` differs ⇒ that layer is implicated;\n",
            configs_off.join(", "),
        ));
        buf.push_str("       multiple configs differ ⇒ inspect baseline output for the canonical answer)\n");
        failures.push(buf);
    }

    eprintln!();
    eprintln!("=== 4-way layer-pinned self-diff ===");
    eprintln!("configs:     {} (baseline / no-raw-byte / no-simplify / pure-interp)", CONFIGS.len());
    eprintln!("PASS:        {}", pass);
    eprintln!("FAIL:        {}", fail);
    eprintln!("SPAWN:       {}", spawn_fail);
    eprintln!("KNOWN-DIV:   {}", known_diverged);
    eprintln!("TOTAL:       {}", cases.len());

    if !failures.is_empty() {
        eprintln!();
        eprintln!("=== Layer divergences ===");
        for f in &failures {
            eprintln!("{}", f);
        }
    }

    assert_eq!(
        fail + spawn_fail,
        0,
        "{} layer-pinned divergences out of {}",
        fail + spawn_fail,
        cases.len(),
    );

    assert!(
        unexpected_pass.is_empty(),
        "KNOWN_DIVERGENCES is stale — these line(s) now agree across all four configs and should be removed (also check `selfdiff_jit_interp.rs::KNOWN_DIVERGENCES`): {:?}",
        unexpected_pass,
    );
}
