//! Fast-path coverage test: every named `detect_*` fast path in
//! `src/bin/jq-jit.rs` must be exercised by at least one entry in
//! `tests/differential/corpus.test`, unless it is explicitly grandfathered
//! into `tests/coverage_fast_path.allowlist`. A new, non-allowlisted fast
//! path with zero corpus coverage blocks merge.
//!
//! Works by running each corpus case with `JQJIT_TRACE=1`, parsing the
//! `[trace] filter='…' matched=<name>` line from stderr, and asserting the
//! union of observed names covers the full set of known names minus the
//! allowlist.
//!
//! The known-names set is scraped from `src/bin/jq-jit.rs` at test time
//! (`trace_detect!(filter, <name>)` and `trace_is!(filter, <name>)` macro
//! invocations), so adding a new fast path and a new corpus probe in the
//! same PR is all that's needed to keep this test green — no macro_rules!
//! registry or `inventory` dependency required.
//!
//! The allowlist (`tests/coverage_fast_path.allowlist`) is a grandfathered
//! baseline of currently uncovered paths: it only shrinks. Adding new
//! corpus probes that hit an allowlisted name forces its removal (the test
//! fails if an allowlisted name is actually covered), so the list decays
//! toward empty over time.
//!
//! Reports per-path invocation counts as a leading stderr block so
//! reviewers can see the hot/cold distribution without opening a CI
//! artifact.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

fn known_fast_paths() -> Vec<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/bin/jq-jit.rs");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));

    let mut names: std::collections::BTreeSet<String> = Default::default();
    for line in src.lines() {
        let trimmed = line.trim_start();
        for prefix in ["trace_detect!(filter, ", "trace_is!(filter, "] {
            if let Some(rest) = trimmed.find(prefix) {
                let after = &trimmed[rest + prefix.len()..];
                if let Some(end) = after.find(')') {
                    let name = after[..end].trim();
                    if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                        names.insert(name.to_string());
                    }
                }
            }
        }
    }
    assert!(
        !names.is_empty(),
        "no fast-path names scraped from src/bin/jq-jit.rs — did the macro names change?"
    );
    names.into_iter().collect()
}

struct Case {
    filter: String,
    input: String,
}

fn parse_corpus(content: &str) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut filter: Option<String> = None;
    let mut input: Option<String> = None;
    for (idx, raw) in content.lines().enumerate() {
        let line_no = idx + 1;
        if raw.trim_start().starts_with('#') { continue; }
        if raw.trim().is_empty() {
            if let (Some(f), Some(i)) = (filter.take(), input.take()) {
                cases.push(Case { filter: f, input: i });
            } else {
                filter = None;
                input = None;
            }
            continue;
        }
        if filter.is_none() { filter = Some(raw.to_string()); }
        else if input.is_none() { input = Some(raw.to_string()); }
        else { panic!("corpus line {}: expected blank between cases", line_no); }
    }
    if let (Some(f), Some(i)) = (filter, input) {
        cases.push(Case { filter: f, input: i });
    }
    cases
}

fn trace_one(bin: &str, filter: &str, input: &str) -> Option<String> {
    let mut cmd = Command::new(bin);
    cmd.arg("-c").arg(filter);
    cmd.env("JQJIT_TRACE", "1");
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
            Ok(Some(_)) => {
                let out = child.wait_with_output().ok()?;
                let stderr = String::from_utf8_lossy(&out.stderr);
                for line in stderr.lines() {
                    // Format: [trace] filter='...' matched=<name>
                    if let Some(rest) = line.rsplit_once("matched=") {
                        let name = rest.1.trim();
                        if !name.is_empty() {
                            return Some(name.to_string());
                        }
                    }
                }
                return None;
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(_) => return None,
        }
    }
}

#[test]
fn fast_paths_have_nonzero_corpus_coverage() {
    let known = known_fast_paths();
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");

    let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/differential/corpus.test");
    let content = std::fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", corpus_path.display(), e));
    let cases = parse_corpus(&content);
    assert!(!cases.is_empty(), "corpus is empty");

    // path name -> invocation count across corpus
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    // "generic" fallbacks, reported separately
    let mut generic_counts: BTreeMap<String, usize> = BTreeMap::new();

    for case in &cases {
        if let Some(name) = trace_one(jq_jit, &case.filter, &case.input) {
            if name == "jit" || name == "eval" {
                *generic_counts.entry(name).or_insert(0) += 1;
            } else {
                *counts.entry(name).or_insert(0) += 1;
            }
        }
    }

    let allowlist = load_allowlist();

    let known_set: std::collections::BTreeSet<&String> = known.iter().collect();
    let covered_set: std::collections::BTreeSet<&String> = counts.keys().collect();

    let uncovered: std::collections::BTreeSet<&String> =
        known_set.difference(&covered_set).copied().collect();

    // Three error classes:
    //  (A) uncovered AND not in allowlist → new fast path without a probe
    //  (B) in allowlist AND not in known set → stale allowlist entry
    //  (C) in allowlist AND covered → allowlist should shrink
    let allowlist_refs: std::collections::BTreeSet<&String> = allowlist.iter().collect();
    let new_uncovered: Vec<&&String> =
        uncovered.difference(&allowlist_refs).collect();
    let stale_allowlist: Vec<&&String> =
        allowlist_refs.difference(&known_set).collect();
    let covered_allowlist: Vec<&&String> = allowlist_refs
        .intersection(&covered_set)
        .collect();
    let unknown: Vec<&&String> = covered_set.difference(&known_set).collect();

    eprintln!();
    eprintln!("=== Fast-path coverage ===");
    eprintln!("known fast paths:     {}", known.len());
    eprintln!("corpus cases:         {}", cases.len());
    eprintln!("covered paths:        {}", counts.len());
    eprintln!("uncovered paths:      {}", uncovered.len());
    eprintln!("allowlist size:       {}", allowlist.len());
    eprintln!("generic dispatch:     {:?}", generic_counts);

    // Report hot (top 15) and cold (bottom 15) for quick diagnostic signal.
    let mut sorted: Vec<(&String, &usize)> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    eprintln!();
    eprintln!("hot paths (top 15):");
    for (n, c) in sorted.iter().take(15) {
        eprintln!("  {:>6}  {}", c, n);
    }
    eprintln!();
    eprintln!("cold covered paths (bottom 15):");
    for (n, c) in sorted.iter().rev().take(15) {
        eprintln!("  {:>6}  {}", c, n);
    }

    if !unknown.is_empty() {
        eprintln!();
        eprintln!("=== Traced names not in known set ({}) ===", unknown.len());
        for n in &unknown {
            eprintln!("  {}", n);
        }
        eprintln!("\nThis means `src/bin/jq-jit.rs` emits a trace name that the coverage");
        eprintln!("scraper did not pick up. Update known_fast_paths() in this test.");
    }

    if !new_uncovered.is_empty() {
        eprintln!();
        eprintln!("=== New uncovered fast paths ({}) ===", new_uncovered.len());
        for n in &new_uncovered {
            eprintln!("  {}", n);
        }
        eprintln!("\nEach new uncovered path needs at least one (filter, input) probe in");
        eprintln!("tests/differential/corpus.test. Use `JQJIT_TRACE=1 ./target/release/jq-jit`");
        eprintln!("to confirm the new probe actually hits the path. If the path is");
        eprintln!("genuinely unreachable from user filters, document why in");
        eprintln!("tests/coverage_fast_path.allowlist instead.");
    }

    if !stale_allowlist.is_empty() {
        eprintln!();
        eprintln!("=== Stale allowlist entries ({}) ===", stale_allowlist.len());
        for n in &stale_allowlist {
            eprintln!("  {}", n);
        }
        eprintln!("\nThese names are in tests/coverage_fast_path.allowlist but no longer");
        eprintln!("exist in src/bin/jq-jit.rs. Remove them.");
    }

    if !covered_allowlist.is_empty() {
        eprintln!();
        eprintln!("=== Allowlist entries now covered ({}) ===", covered_allowlist.len());
        for n in &covered_allowlist {
            eprintln!("  {}", n);
        }
        eprintln!("\nThese names are in tests/coverage_fast_path.allowlist but the corpus");
        eprintln!("already exercises them — remove them from the allowlist.");
    }

    assert!(unknown.is_empty(), "{} traced names not in the scraped known set", unknown.len());
    assert!(
        new_uncovered.is_empty(),
        "{} new fast paths have zero corpus coverage (not in allowlist)",
        new_uncovered.len(),
    );
    assert!(
        stale_allowlist.is_empty(),
        "{} stale allowlist entries (no longer in src)",
        stale_allowlist.len(),
    );
    assert!(
        covered_allowlist.is_empty(),
        "{} allowlist entries are now covered — remove them",
        covered_allowlist.len(),
    );
}

fn load_allowlist() -> Vec<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/coverage_fast_path.allowlist");
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let mut out: Vec<String> = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        out.push(trimmed.to_string());
    }
    out
}
