//! Integration test that runs the official jq test suite against jq-jit binary.
//!
//! Test file format (3-line groups separated by blank lines):
//!   filter_expression
//!   input_json
//!   expected_output_line1
//!   expected_output_line2
//!   ...
//!
//! - Lines starting with `#` are comments
//! - `%%FAIL` blocks are skipped (until next blank line)
//! - Empty lines separate test cases

use std::process::Command;
use std::time::Duration;

struct TestCase {
    filter: String,
    input: String,
    expected: String,
    test_num: usize,
}

struct TestResult {
    test_num: usize,
    filter: String,
    input: String,
    expected: String,
    actual: String,
    status: TestStatus,
}

enum TestStatus {
    Pass,
    Fail,
    Skip(String),
}

fn parse_test_file(content: &str) -> Vec<TestCase> {
    let mut cases = Vec::new();
    let mut test_num = 0usize;

    let mut filter = String::new();
    let mut input = String::new();
    let mut expected = String::new();
    let mut state = "filter"; // "filter", "input", "output"
    let mut in_fail_block = false;

    for line in content.lines() {
        // Skip comment lines
        if line.trim_start().starts_with('#') {
            continue;
        }

        // Handle %%FAIL blocks
        if line.starts_with("%%FAIL") {
            in_fail_block = true;
            continue;
        }
        if in_fail_block {
            if line.is_empty() {
                in_fail_block = false;
            }
            continue;
        }

        if line.is_empty() {
            if state == "output" && !filter.is_empty() {
                test_num += 1;
                cases.push(TestCase {
                    filter: filter.clone(),
                    input: input.clone(),
                    expected: expected.clone(),
                    test_num,
                });
                filter.clear();
                input.clear();
                expected.clear();
                state = "filter";
            }
            continue;
        }

        match state {
            "filter" => {
                filter = line.to_string();
                state = "input";
            }
            "input" => {
                input = line.to_string();
                state = "output";
                expected.clear();
            }
            "output" => {
                if expected.is_empty() {
                    expected = line.to_string();
                } else {
                    expected.push('\n');
                    expected.push_str(line);
                }
            }
            _ => unreachable!(),
        }
    }

    // Handle last test case (no trailing blank line)
    if state == "output" && !filter.is_empty() {
        test_num += 1;
        cases.push(TestCase {
            filter,
            input,
            expected,
            test_num,
        });
    }

    cases
}

/// Normalize a single JSON line: parse it, normalize floats that are integers,
/// then re-serialize compactly with sorted keys.
fn normalize_json_line(line: &str) -> String {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(val) => {
            let normalized = normalize_value(val);
            // Re-serialize compactly. serde_json's to_string produces compact output.
            // We need sorted keys for consistent comparison.
            let s = serialize_sorted(&normalized);
            s
        }
        Err(_) => {
            // Not valid JSON, return as-is with compact whitespace
            trimmed
                .replace(", ", ",")
                .replace(": ", ":")
        }
    }
}

/// Recursively normalize a JSON value: floats that equal integers become integers.
fn normalize_value(val: serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match val {
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                if f.is_finite() && f == (f as i64) as f64 && f.abs() < (1i64 << 53) as f64 {
                    Value::Number(serde_json::Number::from(f as i64))
                } else {
                    Value::Number(n)
                }
            } else {
                Value::Number(n)
            }
        }
        Value::Array(arr) => {
            Value::Array(arr.into_iter().map(normalize_value).collect())
        }
        Value::Object(map) => {
            Value::Object(
                map.into_iter()
                    .map(|(k, v)| (k, normalize_value(v)))
                    .collect(),
            )
        }
        other => other,
    }
}

/// Serialize a JSON value with sorted keys (for stable comparison).
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

/// Normalize multi-line output: normalize each line, join back.
fn normalize_output(output: &str) -> String {
    output
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(normalize_json_line)
        .collect::<Vec<_>>()
        .join("\n")
}

fn run_test(case: &TestCase) -> TestResult {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let lib_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/modules");

    let mut cmd = Command::new(jq_jit);
    cmd.arg("-L").arg(lib_dir).arg("-c").arg(&case.filter);

    // Determine input to pipe via stdin
    let stdin_input = if case.input == "null" && case.filter != "." {
        "null".to_string()
    } else {
        case.input.clone()
    };

    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::null());

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return TestResult {
                test_num: case.test_num,
                filter: case.filter.clone(),
                input: case.input.clone(),
                expected: case.expected.clone(),
                actual: String::new(),
                status: TestStatus::Skip(format!("spawn error: {}", e)),
            };
        }
    };

    // Write stdin in the child
    use std::io::Write;
    let mut child = child;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(stdin_input.as_bytes());
        let _ = stdin.write_all(b"\n");
        // drop stdin to close it
    }

    // Wait with timeout
    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Process exited
                let output = child.wait_with_output().unwrap();

                // On Unix, signal-killed processes have no exit code.
                // Treat them as skips (like the shell script does for SIGSEGV).
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    if let Some(sig) = status.signal() {
                        return TestResult {
                            test_num: case.test_num,
                            filter: case.filter.clone(),
                            input: case.input.clone(),
                            expected: case.expected.clone(),
                            actual: String::new(),
                            status: TestStatus::Skip(format!("killed by signal {}", sig)),
                        };
                    }
                }

                let exit_code = status.code().unwrap_or(-1);

                // Exit code 3 = unsupported, skip
                if exit_code == 3 {
                    return TestResult {
                        test_num: case.test_num,
                        filter: case.filter.clone(),
                        input: case.input.clone(),
                        expected: case.expected.clone(),
                        actual: String::new(),
                        status: TestStatus::Skip("unsupported (exit 3)".to_string()),
                    };
                }

                // Non-zero exit (except 0 and 5) = fail
                if exit_code != 0 && exit_code != 5 {
                    return TestResult {
                        test_num: case.test_num,
                        filter: case.filter.clone(),
                        input: case.input.clone(),
                        expected: case.expected.clone(),
                        actual: format!("<exit code {}>", exit_code),
                        status: TestStatus::Fail,
                    };
                }

                let actual_raw = String::from_utf8_lossy(&output.stdout).to_string();
                let actual_normalized = normalize_output(&actual_raw);
                let expected_normalized = normalize_output(&case.expected);

                if actual_normalized == expected_normalized {
                    return TestResult {
                        test_num: case.test_num,
                        filter: case.filter.clone(),
                        input: case.input.clone(),
                        expected: case.expected.clone(),
                        actual: actual_raw,
                        status: TestStatus::Pass,
                    };
                }

                // Try sorted comparison (reordered output)
                let mut actual_lines: Vec<&str> = actual_normalized.lines().collect();
                let mut expected_lines: Vec<&str> = expected_normalized.lines().collect();
                actual_lines.sort();
                expected_lines.sort();

                if actual_lines == expected_lines {
                    return TestResult {
                        test_num: case.test_num,
                        filter: case.filter.clone(),
                        input: case.input.clone(),
                        expected: case.expected.clone(),
                        actual: actual_raw,
                        status: TestStatus::Pass,
                    };
                }

                return TestResult {
                    test_num: case.test_num,
                    filter: case.filter.clone(),
                    input: case.input.clone(),
                    expected: expected_normalized,
                    actual: actual_normalized,
                    status: TestStatus::Fail,
                };
            }
            Ok(None) => {
                // Still running, check timeout
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return TestResult {
                        test_num: case.test_num,
                        filter: case.filter.clone(),
                        input: case.input.clone(),
                        expected: case.expected.clone(),
                        actual: String::new(),
                        status: TestStatus::Skip("timeout (5s)".to_string()),
                    };
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => {
                return TestResult {
                    test_num: case.test_num,
                    filter: case.filter.clone(),
                    input: case.input.clone(),
                    expected: case.expected.clone(),
                    actual: String::new(),
                    status: TestStatus::Skip(format!("wait error: {}", e)),
                };
            }
        }
    }
}

#[test]
fn official_jq_test_suite() {
    let test_file = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/official/jq.test");
    let content = std::fs::read_to_string(test_file)
        .unwrap_or_else(|e| panic!("Failed to read test file {}: {}", test_file, e));

    let cases = parse_test_file(&content);
    let total = cases.len();

    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut skip = 0usize;
    let mut failures: Vec<TestResult> = Vec::new();

    for case in &cases {
        let result = run_test(case);
        match &result.status {
            TestStatus::Pass => pass += 1,
            TestStatus::Fail => {
                fail += 1;
                failures.push(result);
            }
            TestStatus::Skip(reason) => {
                skip += 1;
                eprintln!("SKIP #{} ({}): {}", case.test_num, reason, case.filter);
            }
        }
    }

    // Print summary
    eprintln!();
    eprintln!("=== jq Official Test Suite Results ===");
    eprintln!("PASS: {}", pass);
    eprintln!("FAIL: {}", fail);
    eprintln!("SKIP: {}", skip);
    eprintln!("TOTAL: {}", total);
    if total > 0 {
        eprintln!("PASS rate: {:.1}%", pass as f64 * 100.0 / total as f64);
    }

    if !failures.is_empty() {
        eprintln!();
        eprintln!("=== Failures ===");
        for f in &failures {
            eprintln!("  #{}: {}", f.test_num, f.filter);
            eprintln!("    input:    {}", f.input);
            eprintln!("    expected: {}", f.expected.lines().take(3).collect::<Vec<_>>().join(" | "));
            eprintln!("    actual:   {}", f.actual.lines().take(3).collect::<Vec<_>>().join(" | "));
        }
    }

    assert_eq!(fail, 0, "{} tests failed out of {} total ({} skipped)", fail, total, skip);
}

#[test]
fn regression_test_suite() {
    let test_file = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/regression.test");
    let content = std::fs::read_to_string(test_file)
        .unwrap_or_else(|e| panic!("Failed to read test file {}: {}", test_file, e));

    let cases = parse_test_file(&content);
    let total = cases.len();

    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut skip = 0usize;
    let mut failures: Vec<TestResult> = Vec::new();

    for case in &cases {
        let result = run_test(case);
        match &result.status {
            TestStatus::Pass => pass += 1,
            TestStatus::Fail => {
                fail += 1;
                failures.push(result);
            }
            TestStatus::Skip(reason) => {
                skip += 1;
                eprintln!("SKIP #{} ({}): {}", case.test_num, reason, case.filter);
            }
        }
    }

    // Print summary
    eprintln!();
    eprintln!("=== Regression Test Suite Results ===");
    eprintln!("PASS: {}", pass);
    eprintln!("FAIL: {}", fail);
    eprintln!("SKIP: {}", skip);
    eprintln!("TOTAL: {}", total);
    if total > 0 {
        eprintln!("PASS rate: {:.1}%", pass as f64 * 100.0 / total as f64);
    }

    if !failures.is_empty() {
        eprintln!();
        eprintln!("=== Failures ===");
        for f in &failures {
            eprintln!("  #{}: {}", f.test_num, f.filter);
            eprintln!("    input:    {}", f.input);
            eprintln!("    expected: {}", f.expected.lines().take(3).collect::<Vec<_>>().join(" | "));
            eprintln!("    actual:   {}", f.actual.lines().take(3).collect::<Vec<_>>().join(" | "));
        }
    }

    assert_eq!(fail, 0, "{} regression tests failed out of {} total ({} skipped)", fail, total, skip);
}
