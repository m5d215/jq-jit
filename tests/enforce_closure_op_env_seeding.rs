//! Enforce maintenance.md §3 "JIT → eval 委譲時の env seeding": every
//! `__<op>__:` runtime dispatcher in `src/jit.rs` that delegates to the
//! eval interpreter must seed its `eval::Env` via `new_delegated_env`
//! or `reset_delegated_env`.
//!
//! `__<op>__:` tags surface in two places:
//!
//! * **emitters** — `format!("__<op>__:...")` calls inside JIT op
//!   construction, which embed the tag in a `JitOp::CallBuiltin` name.
//! * **dispatchers** — `name.strip_prefix("__<op>__:")` arms in the
//!   runtime trampoline, which decode the tag and invoke the matching
//!   eval-side `eval_*_standalone` helper.
//!
//! Dispatchers that build a fresh `Env` must call `new_delegated_env`
//! (or `reset_delegated_env` for cached envs) so JIT-set let-bindings
//! are seeded into the delegated env. A bare `Env::new(...)` here is
//! how the `(.a, .b) += 100` regression slipped in.
//!
//! Some tags are not eval-delegation dispatchers — `__loc__:` returns a
//! constant from a JIT-side parser literal, `__jqerror__:` is a runtime
//! error-message marker — and they live in
//! `tests/enforce_closure_op_env_seeding.allowlist` with the reason.

use std::collections::BTreeSet;
use std::path::PathBuf;

const TARGET_FILE: &str = "src/jit.rs";

fn read_target() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TARGET_FILE);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {}", TARGET_FILE, e))
}

/// Collect every `__<name>__:` tag literal from string-quoted contexts
/// (`"__loc__:..."`, `format!("__loc__:...")`, `strip_prefix("__loc__:")`).
fn collect_tags(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let bytes = src.as_bytes();
    let needle = b"\"__";
    let mut i = 0;
    while i + needle.len() < bytes.len() {
        if &bytes[i..i + needle.len()] == needle {
            // Scan forward to find the closing `__` followed by `:`.
            let mut j = i + 3;
            while j + 3 < bytes.len() {
                if &bytes[j..j + 3] == b"__:" {
                    let name = &src[i + 3..j];
                    if name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                        && !name.is_empty()
                    {
                        out.insert(name.to_string());
                    }
                    break;
                }
                if bytes[j] == b'"' || bytes[j] == b'\n' { break; }
                j += 1;
            }
            i = j + 1;
            continue;
        }
        i += 1;
    }
    out
}

/// Find the dispatcher handler block for a given tag: the block of code
/// starting at `name.strip_prefix("__<tag>__:")` and ending at the next
/// `if let Some(rest) = name.strip_prefix(` or `if name ==` line, or EOF.
fn extract_handler_block<'a>(src: &'a str, tag: &str) -> Option<&'a str> {
    let needle = format!("name.strip_prefix(\"__{}__:\")", tag);
    let start = src.find(&needle)?;
    // Walk forward from `start` to find the matching `}` at brace depth 0
    // relative to this `if let` block, OR until we hit the next dispatcher
    // arm at the outer level.
    let bytes = src.as_bytes();
    let mut depth: i32 = 0;
    let mut seen_brace = false;
    let mut i = start;
    while i < bytes.len() {
        let ch = bytes[i];
        if ch == b'{' { depth += 1; seen_brace = true; }
        else if ch == b'}' { depth -= 1; }
        if seen_brace && depth == 0 { return Some(&src[start..=i]); }
        i += 1;
    }
    Some(&src[start..])
}

fn load_allowlist() -> BTreeSet<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/enforce_closure_op_env_seeding.allowlist");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read allowlist: {}", e));
    let mut out = BTreeSet::new();
    for (n, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        // Allowlist line format: `<tag>` (rest of line is comment/justification).
        let tag = line.split_whitespace().next()
            .unwrap_or_else(|| panic!("allowlist line {}: empty", n + 1));
        out.insert(tag.to_string());
    }
    out
}

#[test]
fn closure_op_dispatchers_seed_delegated_env() {
    let src = read_target();
    let tags = collect_tags(&src);
    let allowed = load_allowlist();

    eprintln!();
    eprintln!("=== closure-op dispatcher env-seeding enforcement ===");
    eprintln!("target:      {}", TARGET_FILE);
    eprintln!("tags found:  {:?}", tags);
    eprintln!("allowlisted: {:?}", allowed);

    // Sanity: known tags must be present (catches a parser breakage).
    for required in ["assign", "update", "path", "paths_filtered", "closure_op"] {
        assert!(
            tags.contains(required),
            "tag parser sanity: expected to find `__{}__:` literal in {}",
            required, TARGET_FILE,
        );
    }

    let mut missing_seeding: Vec<String> = Vec::new();
    let mut no_dispatcher: Vec<String> = Vec::new();
    let mut stale_allowlist: Vec<String> = Vec::new();

    for tag in &tags {
        if allowed.contains(tag) { continue; }
        let Some(block) = extract_handler_block(&src, tag) else {
            // No `strip_prefix` dispatcher for this tag — it's an emitter-only
            // literal (e.g. a regex pattern or a comment). Skip silently.
            no_dispatcher.push(tag.clone());
            continue;
        };
        let has_seed = block.contains("new_delegated_env(")
            || block.contains("reset_delegated_env(");
        if !has_seed {
            missing_seeding.push(tag.clone());
        }
    }

    for tag in &allowed {
        if !tags.contains(tag) {
            stale_allowlist.push(tag.clone());
        }
    }

    if !missing_seeding.is_empty() {
        eprintln!();
        eprintln!("=== Closure-op dispatchers missing env seeding ===");
        for tag in &missing_seeding {
            eprintln!("  __{}__:", tag);
        }
        eprintln!();
        eprintln!("Each dispatcher must construct its delegated `eval::Env`");
        eprintln!("via `new_delegated_env(&[&delegated_expr, ...])` (fresh) or");
        eprintln!("`reset_delegated_env(&env, &[&delegated_expr, ...])` (cached)");
        eprintln!("so JIT-set let-bindings are seeded into the env.");
        eprintln!();
        eprintln!("If a tag genuinely does not delegate to eval (e.g. a constant");
        eprintln!("emitter like `__loc__:` or an error-message marker like");
        eprintln!("`__jqerror__:`), add it to the allowlist with a justifying");
        eprintln!("comment.");
        eprintln!();
        eprintln!("See maintenance.md §3 \"JIT → eval 委譲時の env seeding\".");
    }

    if !no_dispatcher.is_empty() {
        eprintln!();
        eprintln!("=== Tags with no dispatcher arm (informational) ===");
        for tag in &no_dispatcher {
            eprintln!("  __{}__:", tag);
        }
    }

    if !stale_allowlist.is_empty() {
        eprintln!();
        eprintln!("=== Stale allowlist entries ===");
        for tag in &stale_allowlist {
            eprintln!("  __{}__:  no longer found in {}", tag, TARGET_FILE);
        }
    }

    assert!(
        missing_seeding.is_empty() && stale_allowlist.is_empty(),
        "closure-op env seeding: {} dispatcher(s) missing seeding, {} stale allowlist entries",
        missing_seeding.len(), stale_allowlist.len(),
    );
}
