//! Differential test against jq-1.8.x using the hand-curated corpus at
//! `tests/differential/corpus.test` — pairs of `(filter, input)` whose
//! expected output is whatever the reference implementation produces.
//!
//! Error-message wording differs between implementations, so cases where
//! both sides error are treated as agreement. One-sided errors and any
//! value-level difference are failures.
//!
//! Skipped (locally) / failing (on CI) when no jq-1.8.x binary is found —
//! see `tests/common/diff_harness.rs` for the resolution policy.

mod common;

use std::path::PathBuf;
use std::time::Duration;

use common::diff_harness::{jq_jit_path, require_jq, run_filter};
use common::json_normalize::normalize;

const TEST_LABEL: &str = "diff_corpus";
const TIMEOUT: Duration = Duration::from_secs(5);
/// Retry budget for cases whose first run involved a `<timeout>` (#1053).
const RETRY_TIMEOUT: Duration = Duration::from_secs(20);

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

#[test]
fn differential_against_jq_1_8() {
    let Some(jq) = require_jq(TEST_LABEL) else { return };
    let jq_jit = jq_jit_path();

    let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/differential/corpus.test");
    let content = std::fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", corpus_path.display(), e));
    let cases = parse_corpus(&content);
    assert!(!cases.is_empty(), "corpus is empty");

    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut failures: Vec<String> = Vec::new();

    // Run one case at the given per-process timeout. The bool in the error
    // marks whether either side reported `<timeout>` — under full-suite
    // parallelism the reference jq occasionally exceeds the budget on
    // trivially fast filters, which would otherwise surface as a spurious
    // one-sided divergence (#1053).
    fn run_case(
        jq: &str,
        jq_jit: &str,
        case: &Case,
        timeout: Duration,
    ) -> Result<(), (String, bool)> {
        let jq_out = run_filter(jq, &case.filter, &case.input, timeout);
        let jit_out = run_filter(jq_jit, &case.filter, &case.input, timeout);

        let (Some(a), Some(b)) = (jq_out, jit_out) else {
            return Err((format!(
                "  line {}: spawn failure\n    filter: {}\n    input:  {}",
                case.line, case.filter, case.input
            ), false));
        };
        let timed_out = a.stdout == "<timeout>" || b.stdout == "<timeout>";

        if a.is_error && b.is_error {
            return Ok(());
        }
        if a.is_error != b.is_error {
            return Err((format!(
                "  line {}: error mismatch (jq error={}, jit error={})\n    filter: {}\n    input:  {}\n    jq:  {}\n    jit: {}",
                case.line, a.is_error, b.is_error, case.filter, case.input,
                a.stdout.trim(), b.stdout.trim()
            ), timed_out));
        }

        let a_norm = match normalize(&a.stdout) {
            Ok(s) => s,
            Err(e) => {
                return Err((format!(
                    "  line {}: jq output not JSON-parsable ({})\n    filter: {}\n    input:  {}\n    jq:  {}",
                    case.line, e, case.filter, case.input, a.stdout.trim()
                ), timed_out));
            }
        };
        let b_norm = match normalize(&b.stdout) {
            Ok(s) => s,
            Err(e) => {
                return Err((format!(
                    "  line {}: jit output not JSON-parsable ({})\n    filter: {}\n    input:  {}\n    jit: {}",
                    case.line, e, case.filter, case.input, b.stdout.trim()
                ), timed_out));
            }
        };

        if a_norm == b_norm {
            Ok(())
        } else {
            Err((format!(
                "  line {}: value mismatch\n    filter: {}\n    input:  {}\n    jq:  {}\n    jit: {}",
                case.line, case.filter, case.input, a_norm, b_norm
            ), timed_out))
        }
    }

    // Each case spawns its own jq and jq-jit subprocesses with no shared
    // in-process state, so fan out across cores; verdicts are folded back in
    // input order to keep the failure report deterministic. A failure that
    // involved a per-process timeout is retried once with a larger budget:
    // a load-induced timeout heals, a genuine hang times out again (#1053).
    let verdicts = common::parallel::par_map(&cases, |case| -> Result<(), String> {
        match run_case(&jq, jq_jit, case, TIMEOUT) {
            Ok(()) => Ok(()),
            Err((_, true)) => run_case(&jq, jq_jit, case, RETRY_TIMEOUT).map_err(|(m, _)| m),
            Err((m, false)) => Err(m),
        }
    });

    for verdict in verdicts {
        match verdict {
            Ok(()) => pass += 1,
            Err(msg) => {
                fail += 1;
                failures.push(msg);
            }
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
