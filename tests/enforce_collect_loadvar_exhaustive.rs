//! Enforce maintenance.md §3 "JIT → eval 委譲時の env seeding":
//! `Flattener::collect_loadvar_indices` (in `src/jit.rs`) must mention every
//! `Expr` variant defined in `src/ir.rs`.
//!
//! Every JIT→eval closure-op dispatcher walks the delegated expression with
//! this helper to seed `$var`s into the freshly-built `eval::Env`. A missing
//! variant silently drops any `LoadVar` buried inside (an Index, an
//! ObjectConstruct pair, a StringInterpolation expression part, a CallBuiltin
//! arg, ...) and the closure op runs against a `null`-shaped env.
//!
//! This test parses the `Expr` enum out of `src/ir.rs` and the body of
//! `collect_loadvar_indices` out of `src/jit.rs`, then asserts that every
//! variant name appears at least once in the function body — either in an
//! explicit recurse arm or in a no-children leaf arm. Add a new variant and
//! forget to extend the walker → this test fails with the missing names.

use std::path::PathBuf;

fn read_source(rel: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {}", rel, e))
}

/// Extract every variant identifier from `pub enum Expr { ... }` in ir.rs.
///
/// Tracks brace depth so we only consider lines at the enum body's top
/// level (depth 0). Struct-variant fields like `cond: Box<Expr>` live at
/// deeper depth and are skipped.
fn parse_expr_variants(src: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut in_enum = false;
    let mut depth: i32 = 0;
    for line in src.lines() {
        if !in_enum {
            if line.contains("pub enum Expr {") {
                in_enum = true;
            }
            continue;
        }
        let code = match line.find("//") {
            Some(i) => &line[..i],
            None => line,
        };
        let trimmed = code.trim();
        if depth == 0 && !trimmed.is_empty() {
            let first = trimmed.chars().next().unwrap();
            if first == '}' { break; }
            if first.is_ascii_uppercase() {
                let name: String = trimmed.chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() && !out.contains(&name) {
                    out.push(name);
                }
            }
        }
        for ch in code.chars() {
            if ch == '{' { depth += 1; }
            if ch == '}' { depth -= 1; }
        }
    }
    out
}

/// Extract the body text of a function definition by name.
///
/// Scans for `fn <name>(` (any leading visibility / qualifiers), then walks
/// braces from the opening `{` to its matching `}` and returns the slice
/// between them.
fn extract_fn_body<'a>(src: &'a str, name: &str) -> &'a str {
    let needle = format!("fn {}(", name);
    let start = src.find(&needle)
        .unwrap_or_else(|| panic!("could not find `fn {}(` in source", name));
    let brace = src[start..].find('{')
        .unwrap_or_else(|| panic!("no opening brace after `fn {}`", name));
    let body_start = start + brace + 1;
    let mut depth: i32 = 1;
    let bytes = src.as_bytes();
    let mut i = body_start;
    while i < bytes.len() && depth > 0 {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            _ => {}
        }
        if depth == 0 { return &src[body_start..i]; }
        i += 1;
    }
    panic!("could not find matching `}}` for `fn {}`", name);
}

#[test]
fn collect_loadvar_indices_covers_every_expr_variant() {
    let ir_src = read_source("src/ir.rs");
    let jit_src = read_source("src/jit.rs");

    let variants = parse_expr_variants(&ir_src);
    assert!(
        variants.len() > 20,
        "Expr variant parser sanity check failed: only {} variants extracted ({:?})",
        variants.len(), variants,
    );

    let body = extract_fn_body(&jit_src, "collect_loadvar_indices");

    let mut missing: Vec<&str> = Vec::new();
    for v in &variants {
        let token = format!("Expr::{}", v);
        if !body.contains(&token) {
            missing.push(v.as_str());
        }
    }

    eprintln!();
    eprintln!("=== `collect_loadvar_indices` Expr exhaustiveness ===");
    eprintln!("parsed Expr variants:  {}", variants.len());
    eprintln!("walker mentions:       {}", variants.len() - missing.len());

    if !missing.is_empty() {
        eprintln!();
        eprintln!("=== Expr variants missing from `collect_loadvar_indices` ===");
        for v in &missing {
            eprintln!("  Expr::{}", v);
        }
        eprintln!();
        eprintln!("Every variant must appear at least once in");
        eprintln!("`Flattener::collect_loadvar_indices` (src/jit.rs). For variants");
        eprintln!("that hold sub-expressions, recurse into each one. For leaf");
        eprintln!("variants without sub-expressions, add them to the no-children");
        eprintln!("arm. Skipping a variant silently drops any `$var` buried inside");
        eprintln!("the delegated expression, which degrades JIT→eval closure ops");
        eprintln!("to a null-seeded env (see maintenance.md §3 \"JIT → eval");
        eprintln!("委譲時の env seeding\").");
    }

    assert!(
        missing.is_empty(),
        "collect_loadvar_indices missing {} Expr variant(s): {:?}",
        missing.len(), missing,
    );
}
