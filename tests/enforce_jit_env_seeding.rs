//! Enforce maintenance.md §3 "JIT → eval 委譲時の env seeding": every
//! delegated `eval::Env` constructed inside `src/jit.rs` must go through
//! `new_delegated_env` / `reset_delegated_env`, not a bare
//! `Rc::new(RefCell::new(crate::eval::Env::new(...)))`.
//!
//! The seeding helpers walk the delegated expression with
//! `Flattener::collect_loadvar_indices` and copy every `$var` it references
//! out of the JIT env. Skipping them is the failure mode behind the
//! `(.a, .b) += 100` regression where `$r` from the parser-emitted
//! `let $r = 100 in update(...)` was silently lost and the update no-op'd.
//!
//! A new bare `Rc::new(RefCell::new(crate::eval::Env::new(...)))` in
//! `src/jit.rs` makes this test fail with a useful message. The three
//! currently grandfathered sites (the body of `new_delegated_env` itself
//! plus the two `thread_local!` cache `get_or_insert_with` blocks that
//! call `reset_delegated_env` immediately after) live in
//! `tests/enforce_jit_env_seeding.allowlist`.

use std::path::PathBuf;

const GUARDED_PATTERN: &str = "Rc::new(RefCell::new(crate::eval::Env::new(";
const TARGET_FILE: &str = "src/jit.rs";

fn count_hits() -> usize {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TARGET_FILE);
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {}", TARGET_FILE, e));
    content.matches(GUARDED_PATTERN).count()
}

fn load_allowed_count() -> usize {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/enforce_jit_env_seeding.allowlist");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read allowlist: {}", e));
    let mut out: Option<usize> = None;
    for (n, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let Some((file, count_str)) = line.rsplit_once(char::is_whitespace) else {
            panic!("allowlist line {}: expected `<path>  <count>` (got `{}`)", n + 1, raw);
        };
        let file = file.trim();
        let count: usize = count_str.trim().parse()
            .unwrap_or_else(|_| panic!("allowlist line {}: `{}` is not a valid count", n + 1, count_str));
        if file != TARGET_FILE {
            panic!("allowlist line {}: only {} is allowed, got `{}`", n + 1, TARGET_FILE, file);
        }
        if out.is_some() {
            panic!("allowlist has multiple entries for {}", TARGET_FILE);
        }
        out = Some(count);
    }
    out.unwrap_or_else(|| panic!("allowlist missing entry for {}", TARGET_FILE))
}

#[test]
fn jit_delegated_env_construction_routes_through_helpers() {
    let actual = count_hits();
    let allowed = load_allowed_count();

    eprintln!();
    eprintln!("=== JIT → eval delegated Env construction enforcement ===");
    eprintln!("pattern:     {}", GUARDED_PATTERN);
    eprintln!("target:      {}", TARGET_FILE);
    eprintln!("allowlisted: {}", allowed);
    eprintln!("actual:      {}", actual);

    if actual > allowed {
        eprintln!();
        eprintln!("=== New bare `Env::new(...)` construction in {} ===", TARGET_FILE);
        eprintln!("Was {} sites, now {}.\n", allowed, actual);
        eprintln!("Route through one of:");
        eprintln!("  new_delegated_env(&[&delegated_expr, ...])     // fresh Env, auto-seeds $vars");
        eprintln!("  reset_delegated_env(&env, &[&delegated_expr])  // cached Env, re-seeds $vars");
        eprintln!();
        eprintln!("These helpers walk the delegated expression with");
        eprintln!("Flattener::collect_loadvar_indices and copy every $var the");
        eprintln!("delegated expression references out of the JIT env. Skipping");
        eprintln!("them is what broke `(.a, .b) += 100` (see maintenance.md §3");
        eprintln!("\"JIT → eval 委譲時の env seeding\").");
        eprintln!();
        eprintln!("If a new bare site is genuinely necessary (e.g. a new cached-Env");
        eprintln!("thread_local that calls reset_delegated_env immediately after");
        eprintln!("get_or_insert_with), bump the allowlist count with a justifying");
        eprintln!("comment.");
    } else if actual < allowed {
        eprintln!();
        eprintln!("=== Fewer bare `Env::new(...)` sites than allowlisted ===");
        eprintln!("Was {}, now {} — update tests/enforce_jit_env_seeding.allowlist.", allowed, actual);
    }

    assert_eq!(
        actual, allowed,
        "JIT env seeding enforcement: expected {} bare `Env::new(...)` in {}, found {}",
        allowed, TARGET_FILE, actual,
    );
}
