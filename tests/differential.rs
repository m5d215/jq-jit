//! Differential test harness: run `(filter, input)` pairs through both
//! `jq-jit` and reference `jq` (jq-1.8.x) and flag value-level divergences.
//!
//! Error-message wording differs between implementations, so cases where both
//! sides error are treated as agreement. One-sided errors and any value-level
//! difference are failures.
//!
//! The reference `jq` binary is resolved via, in order:
//!   1. `$JQ_BIN` — explicit override (used by CI / local dev).
//!   2. `/opt/homebrew/opt/jq/bin/jq` — Homebrew canonical path (see
//!      `docs/maintenance.md`).
//!   3. `jq` in `$PATH`.
//!
//! If no binary whose `--version` matches `jq-1.8.` can be found, the test is
//! skipped with a diagnostic instead of failing — this keeps local `cargo
//! test` usable without 1.8.1 installed.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

struct Case {
    filter: String,
    input: String,
    line: usize,
}

fn parse_corpus(content: &str) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut filter: Option<(String, usize)> = None;
    let mut input: Option<String> = None;

    for (idx, raw) in content.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw;
        if line.trim_start().starts_with('#') {
            continue;
        }
        if line.trim().is_empty() {
            if let (Some((f, fl)), Some(i)) = (filter.take(), input.take()) {
                cases.push(Case { filter: f, input: i, line: fl });
            } else {
                filter = None;
                input = None;
            }
            continue;
        }
        if filter.is_none() {
            filter = Some((line.to_string(), line_no));
        } else if input.is_none() {
            input = Some(line.to_string());
        } else {
            panic!("corpus line {}: expected blank line between cases", line_no);
        }
    }
    if let (Some((f, fl)), Some(i)) = (filter, input) {
        cases.push(Case { filter: f, input: i, line: fl });
    }
    cases
}

#[derive(Debug)]
struct RunOutput {
    stdout: String,
    is_error: bool,
}

fn run_once(bin: &str, filter: &str, input: &str) -> Option<RunOutput> {
    let mut cmd = Command::new(bin);
    cmd.arg("-c").arg(filter);
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn().ok()?;
    {
        use std::io::Write;
        let mut stdin = child.stdin.take()?;
        let _ = stdin.write_all(input.as_bytes());
        let _ = stdin.write_all(b"\n");
    }

    let timeout = Duration::from_secs(5);
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
                            stdout: format!("<killed by signal {}> stderr: {}", sig, stderr.trim()),
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
                        stdout: "<timeout after 5s>".to_string(),
                        is_error: true,
                    });
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => return None,
        }
    }
}

/// Parse one JSON value per output line and re-serialise compactly with sorted
/// keys, so output equality is value-level (not format-level).
fn normalize(output: &str) -> Result<String, String> {
    let mut normalized_lines = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| format!("non-JSON line `{}`: {}", trimmed, e))?;
        normalized_lines.push(serialize_sorted(&normalize_value(val)));
    }
    Ok(normalized_lines.join("\n"))
}

fn normalize_value(val: serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match val {
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                if f.is_finite() && f == (f as i64) as f64 && f.abs() < (1i64 << 53) as f64 {
                    return Value::Number(serde_json::Number::from(f as i64));
                }
            }
            Value::Number(n)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(normalize_value).collect()),
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| (k, normalize_value(v)))
                .collect(),
        ),
        other => other,
    }
}

fn serialize_sorted(val: &serde_json::Value) -> String {
    use serde_json::Value;
    match val {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap(),
        Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(serialize_sorted).collect();
            format!("[{}]", items.join(","))
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            let items: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}:{}", serde_json::to_string(k).unwrap(), serialize_sorted(v)))
                .collect();
            format!("{{{}}}", items.join(","))
        }
    }
}

fn resolve_jq() -> Option<String> {
    let candidates: Vec<String> = std::env::var("JQ_BIN")
        .ok()
        .into_iter()
        .chain(std::iter::once("/opt/homebrew/opt/jq/bin/jq".to_string()))
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
        eprintln!("SKIP differential: candidate `{}` is `{}`, need jq-1.8.x", cand, ver);
    }
    None
}

#[test]
fn differential_against_jq_1_8() {
    let Some(jq) = resolve_jq() else {
        let msg = "no jq-1.8.x binary found. Set JQ_BIN to a jq-1.8.x binary.";
        if std::env::var_os("CI").is_some() {
            panic!("differential: {}", msg);
        }
        eprintln!("SKIP differential: {}", msg);
        return;
    };
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");

    let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/differential/corpus.test");
    let content = std::fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", corpus_path.display(), e));
    let cases = parse_corpus(&content);
    assert!(!cases.is_empty(), "corpus is empty");

    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for case in &cases {
        let jq_out = run_once(&jq, &case.filter, &case.input);
        let jit_out = run_once(jq_jit, &case.filter, &case.input);

        let (Some(a), Some(b)) = (jq_out, jit_out) else {
            fail += 1;
            failures.push(format!(
                "  line {}: spawn failure\n    filter: {}\n    input:  {}",
                case.line, case.filter, case.input
            ));
            continue;
        };

        // Both errored → agreement, skip value comparison.
        if a.is_error && b.is_error {
            pass += 1;
            continue;
        }
        // One side errored → divergence.
        if a.is_error != b.is_error {
            fail += 1;
            failures.push(format!(
                "  line {}: error mismatch (jq error={}, jit error={})\n    filter: {}\n    input:  {}\n    jq:  {}\n    jit: {}",
                case.line, a.is_error, b.is_error, case.filter, case.input,
                a.stdout.trim(), b.stdout.trim()
            ));
            continue;
        }

        // Both succeeded → normalise and compare.
        let a_norm = match normalize(&a.stdout) {
            Ok(s) => s,
            Err(e) => {
                fail += 1;
                failures.push(format!(
                    "  line {}: jq output not JSON-parsable ({})\n    filter: {}\n    input:  {}\n    jq:  {}",
                    case.line, e, case.filter, case.input, a.stdout.trim()
                ));
                continue;
            }
        };
        let b_norm = match normalize(&b.stdout) {
            Ok(s) => s,
            Err(e) => {
                fail += 1;
                failures.push(format!(
                    "  line {}: jit output not JSON-parsable ({})\n    filter: {}\n    input:  {}\n    jit: {}",
                    case.line, e, case.filter, case.input, b.stdout.trim()
                ));
                continue;
            }
        };

        if a_norm == b_norm {
            pass += 1;
        } else {
            fail += 1;
            failures.push(format!(
                "  line {}: value mismatch\n    filter: {}\n    input:  {}\n    jq:  {}\n    jit: {}",
                case.line, case.filter, case.input, a_norm, b_norm
            ));
        }
    }

    eprintln!();
    eprintln!("=== Differential (vs {}) ===", jq);
    eprintln!("PASS: {}", pass);
    eprintln!("FAIL: {}", fail);
    eprintln!("TOTAL: {}", cases.len());

    if !failures.is_empty() {
        eprintln!();
        eprintln!("=== Divergences ===");
        for f in &failures {
            eprintln!("{}", f);
        }
    }

    assert_eq!(fail, 0, "{} differential divergences out of {}", fail, cases.len());
}
