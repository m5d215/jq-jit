//! Real-world script corpus differential test (#322).
//!
//! Each case in `tests/corpus/` is a pair of files:
//!   - `<name>.jq`   — the filter script.
//!   - `<name>.json` — the input fixture.
//!
//! Both files are passed (via `-f` and stdin) to jq-jit and to a reference
//! `jq-1.8.x` binary; outputs are normalised and compared value-level.
//!
//! Adding a new case is "one file + one fixture, no harness changes".
//!
//! The reference `jq` binary is resolved via, in order:
//!   1. `$JQ_BIN` — explicit override.
//!   2. `/opt/homebrew/opt/jq/bin/jq` — Homebrew canonical path.
//!   3. `jq` in `$PATH`.
//!
//! If no jq-1.8.x binary is found, the test is skipped (panics on CI to
//! prevent silent gaps; `eprintln!` + return locally to keep
//! `cargo test --release` usable without 1.8.1 installed). Same policy as
//! `tests/differential.rs`.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

struct Case {
    name: String,
    script_path: PathBuf,
    input_path: PathBuf,
}

fn collect_cases() -> Vec<Case> {
    let corpus_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus");
    let entries = std::fs::read_dir(&corpus_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", corpus_dir.display(), e));

    let mut cases: Vec<Case> = Vec::new();
    for entry in entries {
        let entry = entry.expect("read_dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("jq") {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("non-utf8 corpus filename: {}", path.display()));
        let input_path = corpus_dir.join(format!("{}.json", stem));
        assert!(
            input_path.exists(),
            "corpus case `{}` is missing its `.json` fixture at {}",
            stem, input_path.display()
        );
        cases.push(Case {
            name: stem.to_string(),
            script_path: path,
            input_path,
        });
    }
    cases.sort_by(|a, b| a.name.cmp(&b.name));
    cases
}

#[derive(Debug)]
struct RunOutput {
    stdout: String,
    is_error: bool,
}

fn run_once(bin: &str, script_path: &std::path::Path, input: &str) -> Option<RunOutput> {
    let mut cmd = Command::new(bin);
    cmd.arg("-c").arg("-f").arg(script_path);
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn().ok()?;
    {
        use std::io::Write;
        let mut stdin = child.stdin.take()?;
        let _ = stdin.write_all(input.as_bytes());
    }

    let timeout = Duration::from_secs(10);
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
                        stdout: "<timeout after 10s>".to_string(),
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
/// keys, so output equality is value-level (not format-level). `@csv` and
/// other string-producing filters yield bare lines that are not JSON-parsable
/// on their own; in that case we fall back to byte-for-byte comparison.
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
        eprintln!("SKIP corpus_diff: candidate `{}` is `{}`, need jq-1.8.x", cand, ver);
    }
    None
}

#[test]
fn corpus_diff_against_jq_1_8() {
    let Some(jq) = resolve_jq() else {
        let msg = "no jq-1.8.x binary found. Set JQ_BIN to a jq-1.8.x binary.";
        if std::env::var_os("CI").is_some() {
            panic!("corpus_diff: {}", msg);
        }
        eprintln!("SKIP corpus_diff: {}", msg);
        return;
    };
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");

    let cases = collect_cases();
    assert!(!cases.is_empty(), "corpus is empty (tests/corpus/)");

    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for case in &cases {
        let input = std::fs::read_to_string(&case.input_path)
            .unwrap_or_else(|e| panic!("read {}: {}", case.input_path.display(), e));

        let jq_out = run_once(&jq, &case.script_path, &input);
        let jit_out = run_once(jq_jit, &case.script_path, &input);

        let (Some(a), Some(b)) = (jq_out, jit_out) else {
            fail += 1;
            failures.push(format!("  case `{}`: spawn failure", case.name));
            continue;
        };

        if a.is_error && b.is_error {
            pass += 1;
            continue;
        }
        if a.is_error != b.is_error {
            fail += 1;
            failures.push(format!(
                "  case `{}`: error mismatch (jq error={}, jit error={})\n    jq:  {}\n    jit: {}",
                case.name, a.is_error, b.is_error, a.stdout.trim(), b.stdout.trim()
            ));
            continue;
        }

        // Try value-level normalisation first; fall back to byte-for-byte for
        // string-producing filters like `@csv` whose output isn't JSON.
        let cmp = match (normalize(&a.stdout), normalize(&b.stdout)) {
            (Ok(an), Ok(bn)) => an == bn,
            _ => a.stdout == b.stdout,
        };
        if cmp {
            pass += 1;
        } else {
            fail += 1;
            failures.push(format!(
                "  case `{}`: value mismatch\n    jq:  {}\n    jit: {}",
                case.name, a.stdout.trim(), b.stdout.trim()
            ));
        }
    }

    eprintln!();
    eprintln!("=== Corpus diff (vs {}) ===", jq);
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

    assert_eq!(fail, 0, "{} corpus divergences out of {}", fail, cases.len());
}
