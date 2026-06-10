//! Enforce maintenance.md §3 "JIT → eval 委譲時の env seeding": every
//! `JitBuiltin` dispatcher in `src/jit.rs` that delegates to the eval
//! interpreter must seed its `eval::Env` via `new_delegated_env` or
//! `reset_delegated_env`.
//!
//! `JitBuiltin` variants surface in two places:
//!
//! * **emitters** — `JitOp::CallBuiltin { builtin: JitBuiltin::<V> ... }`
//!   constructions inside JIT op lowering.
//! * **dispatchers** — `if let JitBuiltin::<V> ...` / `matches!(b,
//!   JitBuiltin::<V>)` arms in the runtime trampoline
//!   (`jit_rt_call_builtin`), which invoke the matching eval-side
//!   `eval_*_standalone` helper.
//!
//! Dispatchers that build a fresh `Env` must call `new_delegated_env`
//! (or `reset_delegated_env` for cached envs) so JIT-set let-bindings
//! are seeded into the delegated env. A bare `Env::new(...)` here is
//! how the `(.a, .b) += 100` regression slipped in.
//!
//! Variants that are not eval-delegation dispatchers — constant values,
//! fast paths over already-evaluated arguments, the generic `Rt` runtime
//! dispatch — live in `tests/enforce_closure_op_env_seeding.allowlist`
//! with the reason. A new variant must either seed its env or be
//! allowlisted explicitly.

use std::collections::BTreeSet;
use std::path::PathBuf;

const TARGET_FILE: &str = "src/jit.rs";

fn read_target() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TARGET_FILE);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {}", TARGET_FILE, e))
}

/// Collect the `JitBuiltin` variant names from the enum definition.
fn collect_variants(src: &str) -> BTreeSet<String> {
    let start = src
        .find("enum JitBuiltin {")
        .unwrap_or_else(|| panic!("enum JitBuiltin not found in {}", TARGET_FILE));
    let body = &src[start..];
    let mut out = BTreeSet::new();
    let mut depth = 0;
    for line in body.lines() {
        depth += line.matches('{').count() as i32;
        depth -= line.matches('}').count() as i32;
        if depth <= 0 && !out.is_empty() {
            break;
        }
        let t = line.trim();
        if t.starts_with("///") || t.starts_with("//") || t.starts_with('#') {
            continue;
        }
        // A variant line starts with an UpperCamelCase ident at depth 1.
        if depth == 1 {
            let ident: String = t
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if ident
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_uppercase())
                && ident != "JitBuiltin"
            {
                out.insert(ident);
            }
        }
    }
    out
}

/// Find the dispatcher handler block for a variant inside the runtime
/// trampoline: the block of code starting at `if let JitBuiltin::<V>` or
/// `matches!(b, JitBuiltin::<V>)` and ending at its closing brace.
fn extract_handler_block<'a>(src: &'a str, variant: &str) -> Option<&'a str> {
    let tramp_start = src.find("extern \"C\" fn jit_rt_call_builtin")?;
    let tramp = &src[tramp_start..];
    let needle_a = format!("if let JitBuiltin::{variant}");
    let needle_b = format!("matches!(b, JitBuiltin::{variant})");
    let start = tramp.find(&needle_a).or_else(|| tramp.find(&needle_b))?;
    let bytes = tramp.as_bytes();
    // The handler body opens at the `{` that ends the `if` line; a payload
    // destructure (`if let JitBuiltin::Assign { path_idx, .. } = b {`) puts
    // earlier braces on the same line, so find the line-trailing `{` first.
    let mut body_open = None;
    let mut i = start;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\r') {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'\n' {
                body_open = Some(i);
                break;
            }
        }
        if bytes[i] == b'\n' {
            break;
        }
        i += 1;
    }
    let body_open = body_open?;
    let mut depth: i32 = 0;
    let mut i = body_open;
    while i < bytes.len() {
        let ch = bytes[i];
        if ch == b'{' {
            depth += 1;
        } else if ch == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(&tramp[start..=i]);
            }
        }
        i += 1;
    }
    Some(&tramp[start..])
}

fn load_allowlist() -> BTreeSet<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/enforce_closure_op_env_seeding.allowlist");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read allowlist: {}", e));
    let mut out = BTreeSet::new();
    for (n, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Allowlist line format: `<variant>` (rest of line is comment).
        let tag = line
            .split_whitespace()
            .next()
            .unwrap_or_else(|| panic!("allowlist line {}: empty", n + 1));
        out.insert(tag.to_string());
    }
    out
}

#[test]
fn closure_op_dispatchers_seed_delegated_env() {
    let src = read_target();
    let variants = collect_variants(&src);
    let allowed = load_allowlist();

    eprintln!();
    eprintln!("=== closure-op dispatcher env-seeding enforcement ===");
    eprintln!("target:      {}", TARGET_FILE);
    eprintln!("variants:    {:?}", variants);
    eprintln!("allowlisted: {:?}", allowed);

    // Sanity: the known eval-delegating variants must be present (catches
    // a parser breakage or a rename that would silently skip enforcement).
    // PathExpr was removed in #1085 — complex path() now bails the whole
    // filter to eval instead of delegating an eager collection.
    for required in ["Assign", "Update", "PathsFiltered", "ClosureOp"] {
        assert!(
            variants.contains(required),
            "variant parser sanity: expected JitBuiltin::{} in {}",
            required, TARGET_FILE,
        );
    }

    let mut missing_seeding: Vec<String> = Vec::new();
    let mut no_dispatcher: Vec<String> = Vec::new();
    let mut stale_allowlist: Vec<String> = Vec::new();

    for variant in &variants {
        if allowed.contains(variant) {
            continue;
        }
        let Some(block) = extract_handler_block(&src, variant) else {
            // No dispatcher arm in the trampoline — emitter-only or handled
            // by the final generic match. Informational.
            no_dispatcher.push(variant.clone());
            continue;
        };
        let has_seed = block.contains("new_delegated_env(")
            || block.contains("reset_delegated_env(");
        if !has_seed {
            missing_seeding.push(variant.clone());
        }
    }

    for tag in &allowed {
        if !variants.contains(tag) {
            stale_allowlist.push(tag.clone());
        }
    }

    if !missing_seeding.is_empty() {
        eprintln!();
        eprintln!("=== Closure-op dispatchers missing env seeding ===");
        for v in &missing_seeding {
            eprintln!("  JitBuiltin::{}", v);
        }
        eprintln!();
        eprintln!("Each dispatcher must construct its delegated `eval::Env`");
        eprintln!("via `new_delegated_env(&[&delegated_expr, ...])` (fresh) or");
        eprintln!("`reset_delegated_env(&env, &[&delegated_expr, ...])` (cached)");
        eprintln!("so JIT-set let-bindings are seeded into the env.");
        eprintln!();
        eprintln!("If a variant genuinely does not delegate to eval (constant");
        eprintln!("values, fast paths over evaluated args, generic Rt dispatch),");
        eprintln!("add it to the allowlist with a justifying comment.");
        eprintln!();
        eprintln!("See maintenance.md §3 \"JIT → eval 委譲時の env seeding\".");
    }

    if !no_dispatcher.is_empty() {
        eprintln!();
        eprintln!("=== Variants with no dedicated dispatcher arm (informational) ===");
        for v in &no_dispatcher {
            eprintln!("  JitBuiltin::{}", v);
        }
    }

    if !stale_allowlist.is_empty() {
        eprintln!();
        eprintln!("=== Stale allowlist entries ===");
        for tag in &stale_allowlist {
            eprintln!("  {}  no longer a JitBuiltin variant in {}", tag, TARGET_FILE);
        }
    }

    assert!(
        missing_seeding.is_empty() && stale_allowlist.is_empty(),
        "closure-op env seeding: {} dispatcher(s) missing seeding, {} stale allowlist entries",
        missing_seeding.len(),
        stale_allowlist.len(),
    );
}
