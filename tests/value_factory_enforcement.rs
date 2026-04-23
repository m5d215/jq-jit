//! Enforce that new code routes `Value::Obj` construction through the
//! `Value::object_from_pairs` / `Value::object_from_normalized_pairs` /
//! `Value::object_from_map` factories in `src/value.rs`.
//!
//! Direct `Value::Obj(Rc::new(…))` construction is the failure mode behind
//! #52 / #53 / #73 / #75: each site has to remember to run the pair list
//! through `normalize_object_pairs` before constructing the Value, and any
//! site that forgets silently ships duplicate-key output. The factories
//! fold dedup into the construction call, so "the only way to build an
//! object Value routes through normalize".
//!
//! Since migrating the ~50 existing call sites is its own refactor, this
//! test grandfathers each current site into
//! `tests/value_factory_enforcement.allowlist` (file + line-count).
//! The allowlist is unidirectional:
//!
//!   - New `Value::Obj(Rc::new(…))` site not in allowlist  →  FAIL
//!   - Existing file grew above its grandfathered count    →  FAIL
//!   - Existing file shrank below its grandfathered count  →  FAIL (update the count)
//!   - File in allowlist no longer exists / has zero hits  →  FAIL
//!
//! To add a legitimate new construction site: prefer one of the factories.
//! If you genuinely need raw `Value::Obj(Rc::new(…))` — e.g. inside
//! `src/value.rs` where the factory itself lives — update the allowlist.

use std::collections::BTreeMap;
use std::path::PathBuf;

const GUARDED_PATTERN: &str = "Value::Obj(Rc::new";

fn scan_src() -> BTreeMap<String, usize> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    walk(&root, &root, &mut counts);
    counts
}

fn walk(base: &PathBuf, dir: &PathBuf, counts: &mut BTreeMap<String, usize>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(base, &path, counts);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else { continue };
        let hits = content.matches(GUARDED_PATTERN).count();
        if hits == 0 { continue; }
        let rel = path.strip_prefix(base).unwrap_or(&path).to_string_lossy().into_owned();
        counts.insert(format!("src/{}", rel), hits);
    }
}

fn load_allowlist() -> BTreeMap<String, usize> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/value_factory_enforcement.allowlist");
    let Ok(content) = std::fs::read_to_string(&path) else {
        return BTreeMap::new();
    };
    let mut out: BTreeMap<String, usize> = BTreeMap::new();
    for (n, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let Some((file, count_str)) = line.rsplit_once(char::is_whitespace) else {
            panic!(
                "allowlist line {}: expected `<path>  <count>` (got: `{}`)",
                n + 1, raw,
            );
        };
        let count: usize = count_str.trim().parse().unwrap_or_else(|_| {
            panic!(
                "allowlist line {}: `{}` is not a valid count",
                n + 1, count_str,
            )
        });
        out.insert(file.trim().to_string(), count);
    }
    out
}

#[test]
fn value_obj_construction_goes_through_factories() {
    let actual = scan_src();
    let allowed = load_allowlist();

    let mut new_hits: Vec<(String, usize)> = Vec::new();
    let mut grew: Vec<(String, usize, usize)> = Vec::new();
    let mut shrank: Vec<(String, usize, usize)> = Vec::new();
    let mut stale: Vec<String> = Vec::new();

    for (file, hits) in &actual {
        match allowed.get(file) {
            None => new_hits.push((file.clone(), *hits)),
            Some(limit) if hits > limit => grew.push((file.clone(), *limit, *hits)),
            Some(limit) if hits < limit => shrank.push((file.clone(), *limit, *hits)),
            _ => {}
        }
    }
    for file in allowed.keys() {
        if !actual.contains_key(file) {
            stale.push(file.clone());
        }
    }

    eprintln!();
    eprintln!("=== Value::Obj(Rc::new(…)) enforcement ===");
    eprintln!("pattern:       {}", GUARDED_PATTERN);
    eprintln!("allowlisted:   {} files, {} hits total",
        allowed.len(), allowed.values().sum::<usize>());
    eprintln!("actual:        {} files, {} hits total",
        actual.len(), actual.values().sum::<usize>());

    if !new_hits.is_empty() {
        eprintln!();
        eprintln!("=== New files with direct Value::Obj construction ===");
        for (file, hits) in &new_hits {
            eprintln!("  {}  (+{} hits)", file, hits);
        }
        eprintln!("\nRoute through one of:");
        eprintln!("  Value::object_from_pairs(pairs)             // dedupes duplicate keys");
        eprintln!("  Value::object_from_normalized_pairs(pairs)  // asserts pre-deduped");
        eprintln!("  Value::object_from_map(map)                 // reuse an ObjMap built via insert/push_unique");
        eprintln!("\nIf a raw construction is genuinely necessary (e.g. inside value.rs");
        eprintln!("itself), add the file to tests/value_factory_enforcement.allowlist.");
    }
    if !grew.is_empty() {
        eprintln!();
        eprintln!("=== Files that grew past their grandfathered count ===");
        for (file, limit, now) in &grew {
            eprintln!("  {}  was {}, now {}", file, limit, now);
        }
        eprintln!("\nNew raw construction sites were added to an existing file. Either");
        eprintln!("route them through a factory or bump the allowlist count with a");
        eprintln!("justifying comment.");
    }
    if !shrank.is_empty() {
        eprintln!();
        eprintln!("=== Files with fewer hits than allowlisted ===");
        for (file, limit, now) in &shrank {
            eprintln!("  {}  was {}, now {}  — update the allowlist", file, limit, now);
        }
    }
    if !stale.is_empty() {
        eprintln!();
        eprintln!("=== Stale allowlist entries ===");
        for file in &stale {
            eprintln!("  {}", file);
        }
        eprintln!("\nThese files are in the allowlist but no longer contain any direct");
        eprintln!("construction. Remove them.");
    }

    assert!(
        new_hits.is_empty() && grew.is_empty() && shrank.is_empty() && stale.is_empty(),
        "value-factory enforcement: {} new, {} grew, {} shrank, {} stale",
        new_hits.len(), grew.len(), shrank.len(), stale.len(),
    );
}
