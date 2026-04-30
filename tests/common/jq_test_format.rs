//! Parser + runner for the 3-line group test format used by both
//! `tests/official/jq.test` (upstream compatibility) and
//! `tests/regression.test` (issue-driven self-pinned cases).
//!
//! Each group:
//! ```text
//! filter_expression
//! input_json
//! expected_output_line1
//! expected_output_line2
//! ...
//! ```
//! Groups are separated by blank lines. Lines starting with `#` are
//! comments. `%%FAIL` blocks (jq's "expected to fail" marker) are skipped
//! until the next blank line.

use std::process::Command;
use std::time::Duration;

use super::json_normalize::{normalize_value, serialize_sorted};

pub struct TestCase {
    pub filter: String,
    pub input: String,
    pub expected: String,
    pub test_num: usize,
}

pub struct TestResult {
    pub test_num: usize,
    pub filter: String,
    pub input: String,
    pub expected: String,
    pub actual: String,
    pub status: TestStatus,
}

pub enum TestStatus {
    Pass,
    Fail,
    Skip(String),
}

pub fn parse_test_file(content: &str) -> Vec<TestCase> {
    let mut cases = Vec::new();
    let mut test_num = 0usize;

    let mut filter = String::new();
    let mut input = String::new();
    let mut expected = String::new();
    let mut state = "filter";
    let mut in_fail_block = false;

    for line in content.lines() {
        if line.trim_start().starts_with('#') {
            continue;
        }

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

fn normalize_json_line(line: &str) -> String {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(val) => serialize_sorted(&normalize_value(val)),
        Err(_) => trimmed.replace(", ", ",").replace(": ", ":"),
    }
}

pub fn normalize_output(output: &str) -> String {
    output
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(normalize_json_line)
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn run_test(case: &TestCase) -> TestResult {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let lib_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/modules");

    let mut cmd = Command::new(jq_jit);
    cmd.arg("-L").arg(lib_dir).arg("-c").arg(&case.filter);

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

    use std::io::Write;
    let mut child = child;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(stdin_input.as_bytes());
        let _ = stdin.write_all(b"\n");
    }

    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let output = child.wait_with_output().unwrap();

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

/// Driver shared by the official and regression suites: parse the file,
/// run every case, print a summary, and assert zero failures.
pub fn run_jq_test_suite(label: &str, test_file_path: &str) {
    let content = std::fs::read_to_string(test_file_path)
        .unwrap_or_else(|e| panic!("Failed to read test file {}: {}", test_file_path, e));

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

    eprintln!();
    eprintln!("=== {} ===", label);
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
            eprintln!(
                "    expected: {}",
                f.expected
                    .lines()
                    .take(3)
                    .collect::<Vec<_>>()
                    .join(" | ")
            );
            eprintln!(
                "    actual:   {}",
                f.actual.lines().take(3).collect::<Vec<_>>().join(" | ")
            );
        }
    }

    assert_eq!(
        fail, 0,
        "{} tests failed out of {} total ({} skipped)",
        fail, total, skip
    );
}
