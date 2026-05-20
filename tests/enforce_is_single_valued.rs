//! Enforce maintenance.md §3 "`[gen] | add` は single-valued 要素のみ畳める":
//! every `Expr` variant must have a deliberate classification in
//! `is_single_valued_expr` (`src/interpreter.rs`), driven by an external
//! allowlist that classifies each variant as:
//!
//! * `reject`           — explicit reject arm returning `false`, for variants
//!                        that yield 0 or many values (Empty, Each, Comma,
//!                        Recurse, Range, Limit, IndexOpt, TryCatch, regex
//!                        multi-emitters, ...).
//! * `accept_explicit`  — explicit accept arm returning `true`, for leaf
//!                        variants known to always yield exactly one value
//!                        (Input, Literal, LoadVar, Collect, ...).
//! * `recurse`          — explicit recurse arm that delegates the decision
//!                        to sub-expressions (Pipe, IfThenElse, BinOp,
//!                        Alternative, ...).
//! * `default_false`    — falls through to the catch-all `_ => false`. Safe
//!                        default for "we don't know, conservatively assume
//!                        multi-valued" so a new variant can't accidentally
//!                        enable a generator-folding rewrite.
//!
//! Adding a new `Expr` variant without updating the allowlist fails this
//! test, forcing a deliberate classification choice. Forgetting to add the
//! variant to `is_single_valued_expr` itself when the allowlist says
//! `reject`/`accept_explicit`/`recurse` also fails, with a useful message.

use std::collections::BTreeMap;
use std::path::PathBuf;

fn read_source(rel: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {}", rel, e))
}

fn parse_expr_variants(src: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut in_enum = false;
    let mut depth: i32 = 0;
    for line in src.lines() {
        if !in_enum {
            if line.contains("pub enum Expr {") { in_enum = true; }
            continue;
        }
        let code = match line.find("//") { Some(i) => &line[..i], None => line };
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Classification { Reject, AcceptExplicit, Recurse, DefaultFalse }

fn load_allowlist() -> BTreeMap<String, Classification> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/enforce_is_single_valued.allowlist");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read allowlist: {}", e));
    let mut out: BTreeMap<String, Classification> = BTreeMap::new();
    for (n, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let mut iter = line.split_whitespace();
        let variant = iter.next()
            .unwrap_or_else(|| panic!("allowlist line {}: empty variant", n + 1));
        let kind = iter.next()
            .unwrap_or_else(|| panic!("allowlist line {}: missing classification", n + 1));
        let classification = match kind {
            "reject" => Classification::Reject,
            "accept_explicit" => Classification::AcceptExplicit,
            "recurse" => Classification::Recurse,
            "default_false" => Classification::DefaultFalse,
            other => panic!(
                "allowlist line {}: unknown classification `{}` (expected reject / accept_explicit / recurse / default_false)",
                n + 1, other,
            ),
        };
        if out.insert(variant.to_string(), classification).is_some() {
            panic!("allowlist line {}: duplicate entry for `{}`", n + 1, variant);
        }
    }
    out
}

#[test]
fn is_single_valued_expr_classifies_every_variant() {
    let ir_src = read_source("src/ir.rs");
    let interp_src = read_source("src/interpreter.rs");
    let variants = parse_expr_variants(&ir_src);
    let allowlist = load_allowlist();
    let body = extract_fn_body(&interp_src, "is_single_valued_expr");

    eprintln!();
    eprintln!("=== `is_single_valued_expr` variant-classification enforcement ===");
    eprintln!("parsed Expr variants: {}", variants.len());
    eprintln!("allowlist entries:    {}", allowlist.len());

    let mut unclassified: Vec<String> = Vec::new();
    let mut stale_allowlist: Vec<String> = Vec::new();
    let mut missing_in_body: Vec<(String, Classification)> = Vec::new();
    let mut unexpected_in_body: Vec<String> = Vec::new();

    for v in &variants {
        if !allowlist.contains_key(v) {
            unclassified.push(v.clone());
        }
    }
    for v in allowlist.keys() {
        if !variants.contains(v) {
            stale_allowlist.push(v.clone());
        }
    }

    // Catch-all sanity: the body must end with `_ => false` so the
    // `default_false` classification is meaningfully enforced.
    assert!(
        body.contains("_ => false"),
        "`is_single_valued_expr` must include a `_ => false` catch-all arm",
    );

    for v in &variants {
        let token = format!("Expr::{}", v);
        let appears = body.contains(&token);
        match allowlist.get(v) {
            None => continue, // already flagged as unclassified
            Some(Classification::Reject)
            | Some(Classification::AcceptExplicit)
            | Some(Classification::Recurse) => {
                if !appears {
                    missing_in_body.push((v.clone(), *allowlist.get(v).unwrap()));
                }
            }
            Some(Classification::DefaultFalse) => {
                if appears {
                    unexpected_in_body.push(v.clone());
                }
            }
        }
    }

    if !unclassified.is_empty() {
        eprintln!();
        eprintln!("=== Expr variants missing from allowlist ===");
        for v in &unclassified {
            eprintln!("  {}", v);
        }
        eprintln!();
        eprintln!("Each new variant needs a deliberate classification in");
        eprintln!("`tests/enforce_is_single_valued.allowlist`:");
        eprintln!("  reject           — explicit `false` arm (generator-like)");
        eprintln!("  accept_explicit  — explicit `true` arm (always 1 value)");
        eprintln!("  recurse          — explicit recurse arm (delegates to children)");
        eprintln!("  default_false    — falls through to `_ => false` (safe default)");
    }

    if !stale_allowlist.is_empty() {
        eprintln!();
        eprintln!("=== Stale allowlist entries (Expr variant no longer exists) ===");
        for v in &stale_allowlist {
            eprintln!("  {}", v);
        }
    }

    if !missing_in_body.is_empty() {
        eprintln!();
        eprintln!("=== Variants classified non-default but missing from `is_single_valued_expr` ===");
        for (v, c) in &missing_in_body {
            eprintln!("  Expr::{:<26}  classified `{:?}`", v, c);
        }
        eprintln!();
        eprintln!("Either add the explicit arm to `is_single_valued_expr` in");
        eprintln!("`src/interpreter.rs`, or downgrade the classification to");
        eprintln!("`default_false` in the allowlist.");
    }

    if !unexpected_in_body.is_empty() {
        eprintln!();
        eprintln!("=== Variants classified `default_false` but explicitly mentioned in body ===");
        for v in &unexpected_in_body {
            eprintln!("  Expr::{}", v);
        }
        eprintln!();
        eprintln!("Either remove the explicit arm or upgrade the classification");
        eprintln!("(reject / accept_explicit / recurse) in the allowlist.");
    }

    assert!(
        unclassified.is_empty()
            && stale_allowlist.is_empty()
            && missing_in_body.is_empty()
            && unexpected_in_body.is_empty(),
        "is_single_valued_expr classification: {} unclassified, {} stale, {} missing-in-body, {} unexpected-in-body",
        unclassified.len(), stale_allowlist.len(), missing_in_body.len(), unexpected_in_body.len(),
    );
}
