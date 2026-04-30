//! Differential test against jq-1.8.x using realistic script scenarios
//! under `tests/corpus/` — each scenario is a `(<name>.jq, <name>.json)`
//! pair. Adding a new scenario is "one filter file + one fixture, no
//! harness changes" (#322).
//!
//! Falls back to byte-for-byte comparison for `@csv`-style filters whose
//! output isn't JSON-parsable. Skipped (locally) / failing (on CI) when
//! no jq-1.8.x binary is found.

mod common;

use std::path::PathBuf;
use std::time::Duration;

use common::diff_harness::{jq_jit_path, require_jq, run_script_file};
use common::json_normalize::normalize;

const TEST_LABEL: &str = "diff_scenarios";
const TIMEOUT: Duration = Duration::from_secs(10);

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
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("non-utf8 corpus filename: {}", path.display()));
        let input_path = corpus_dir.join(format!("{}.json", stem));
        assert!(
            input_path.exists(),
            "corpus case `{}` is missing its `.json` fixture at {}",
            stem,
            input_path.display()
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

#[test]
fn corpus_diff_against_jq_1_8() {
    let Some(jq) = require_jq(TEST_LABEL) else { return };
    let jq_jit = jq_jit_path();

    let cases = collect_cases();
    assert!(!cases.is_empty(), "corpus is empty (tests/corpus/)");

    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for case in &cases {
        let input = std::fs::read_to_string(&case.input_path)
            .unwrap_or_else(|e| panic!("read {}: {}", case.input_path.display(), e));

        let jq_out = run_script_file(&jq, &case.script_path, &input, TIMEOUT);
        let jit_out = run_script_file(jq_jit, &case.script_path, &input, TIMEOUT);

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

        // Value-level normalisation first; fall back to byte-for-byte for
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
