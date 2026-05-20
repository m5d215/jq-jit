//! Enforce maintenance.md §3 "Fast path と `?` / `try` の相互作用": every
//! `pub fn apply_*_raw` in `src/fast_path.rs` must return `RawApplyOutcome`
//! AND mention `RawApplyOutcome::Bail` in its body.
//!
//! The `RawApplyOutcome` return shape (#83 Phase B) replaced implicit
//! `match raw[0]` fall-throughs that silently emitted `null` for
//! non-expected input types — the null-masking class of #50. Any new
//! raw apply-site that elides the `Bail` outcome regresses the
//! invariant: the missing type-check ships as a silent `null` cell.
//!
//! This test enumerates `pub fn apply_*_raw` from `src/fast_path.rs` and
//! verifies the contract structurally. It is a static contract check,
//! not a semantic correctness check — a function that returns
//! `RawApplyOutcome::Emit` for all paths still fails to encode "no
//! type-mismatch is possible", so the assertion catches the canonical
//! "forgot the bail" shape without false positives.

use std::path::PathBuf;

const TARGET_FILE: &str = "src/fast_path.rs";

fn read_target() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TARGET_FILE);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {}", TARGET_FILE, e))
}

/// One `pub fn apply_*_raw` declaration in source order.
struct ApplySite {
    name: String,
    /// Concatenated signature text from `pub fn` to the opening `{` of the body.
    signature: String,
    /// Body text between the opening `{` and matching `}` (exclusive).
    body: String,
}

/// Enumerate `pub fn apply_*_raw` sites by line, terminating each at the
/// next line that begins with `}` at column 0 (the canonical rustfmt
/// shape for top-level function bodies). This avoids the need to
/// brace-balance through string literals, char literals, and comments.
fn enumerate_apply_sites(src: &str) -> Vec<ApplySite> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if !line.starts_with("pub fn apply_") {
            i += 1;
            continue;
        }
        let after_fn = &line[7..];
        let name: String = after_fn.chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !name.starts_with("apply_") || !name.ends_with("_raw") {
            i += 1;
            continue;
        }
        // Signature = lines from `pub fn` up to and including the line
        // containing the opening `{` of the body.
        let mut sig_end = i;
        while sig_end < lines.len() && !lines[sig_end].contains('{') {
            sig_end += 1;
        }
        let signature = lines[i..=sig_end.min(lines.len() - 1)].join("\n");
        // Body = lines after sig_end up to the first line starting with
        // `}` at column 0.
        let body_start = sig_end + 1;
        let mut body_end = body_start;
        while body_end < lines.len() && !lines[body_end].starts_with('}') {
            body_end += 1;
        }
        let body = lines[body_start..body_end].join("\n");
        out.push(ApplySite { name, signature, body });
        i = body_end + 1;
    }
    out
}

#[test]
fn raw_apply_sites_use_bail_outcome() {
    let src = read_target();
    let sites = enumerate_apply_sites(&src);

    assert!(
        sites.len() >= 50,
        "raw-apply enumerator sanity: only found {} `pub fn apply_*_raw` in {} (expected many more)",
        sites.len(), TARGET_FILE,
    );

    eprintln!();
    eprintln!("=== Raw-apply Bail-outcome enforcement ===");
    eprintln!("target:                 {}", TARGET_FILE);
    eprintln!("pub fn apply_*_raw:     {}", sites.len());

    let mut missing_return_type: Vec<String> = Vec::new();
    let mut missing_bail: Vec<String> = Vec::new();

    for site in &sites {
        if !site.signature.contains("RawApplyOutcome") {
            missing_return_type.push(site.name.clone());
        }
        if !site.body.contains("RawApplyOutcome::Bail") {
            missing_bail.push(site.name.clone());
        }
    }

    if !missing_return_type.is_empty() {
        eprintln!();
        eprintln!("=== `apply_*_raw` not returning `RawApplyOutcome` ===");
        for n in &missing_return_type {
            eprintln!("  {}", n);
        }
        eprintln!();
        eprintln!("Raw apply-sites must declare `-> RawApplyOutcome` (or");
        eprintln!("`-> Result<RawApplyOutcome, _>` etc.) so callers can");
        eprintln!("distinguish `Emit` from `Bail` and route the bail through");
        eprintln!("`process_input` → `Filter::execute_cb` (#83 Phase B).");
    }

    if !missing_bail.is_empty() {
        eprintln!();
        eprintln!("=== `apply_*_raw` body missing `RawApplyOutcome::Bail` ===");
        for n in &missing_bail {
            eprintln!("  {}", n);
        }
        eprintln!();
        eprintln!("Every raw fast path must have an explicit bail exit: any input");
        eprintln!("shape the apply-site can't guarantee jq-compatible semantics");
        eprintln!("for MUST `return RawApplyOutcome::Bail`. Implicit");
        eprintln!("`match raw[0]` fall-through that emits `null` for non-expected");
        eprintln!("types is the null-masking bug class (#50). If the function");
        eprintln!("genuinely never bails (e.g. its input was pre-classified by a");
        eprintln!("detector), document it with a doc-comment that includes the");
        eprintln!("token `RawApplyOutcome::Bail` so the discipline is still");
        eprintln!("visible at the apply-site.");
    }

    assert!(
        missing_return_type.is_empty() && missing_bail.is_empty(),
        "raw-apply Bail discipline: {} missing return type, {} missing Bail",
        missing_return_type.len(), missing_bail.len(),
    );
}
