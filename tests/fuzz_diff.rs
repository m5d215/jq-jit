//! Property-based differential fuzz harness (#319).
//!
//! Goal: surface jq-compat divergences that no hand-curated test would
//! think to enumerate. Generates random `(filter, input)` pairs from a
//! deliberately *conservative* AST/JSON distribution, runs both jq-jit
//! and reference `jq 1.8.x`, and fails on any value-level divergence.
//!
//! ## Why "conservative"
//!
//! `tests/differential_proptest.rs` already exists as a heavier opt-in
//! variant (`#[ignore]`) that exercises the full grammar and trips on
//! every fast-path divergence as it falls out. This file is the
//! *default-on* counterpart — its generators omit shapes whose
//! divergence is already tracked elsewhere so it can be wired into
//! `cargo test --release` without becoming a stuck-CI nuisance.
//!
//! Tracked exclusions (each is or has been its own follow-up — see
//! the inline comments where each is filtered out):
//!
//! * `try ... catch …` and `(…)?` — these turn raised errors into
//!   in-band values, exposing minor error-message wording divergences
//!   (parens around quoted keys, value-tagged numbers, etc.) that are
//!   their own bug class. The opt-in `differential_proptest.rs` covers
//!   that surface; this harness keeps errors on stderr so they remain
//!   in the "both errored → skip" branch.
//! * `.[:]` — slice with both endpoints absent. jq treats this as a
//!   syntax error; jq-jit's parser accepts it. Distinct from the
//!   runtime fast-path bug class this harness chases.
//! * `..` (recurse) — output ordering is grammar-defined and the
//!   permutations explode the search space without finding new bugs.
//! * Float literals in input — jq's number printer normalizes
//!   formatting (`1e10`, `+5`) in ways the harness's `serde_json`
//!   re-parse can mask, leading to false-positive shrinks. Restrict
//!   numeric inputs to integers and let the *filter* introduce
//!   floating arithmetic when needed.
//! * Duplicate keys in input objects — `parse_json_object` dedupes
//!   (#233), but the raw-byte `keys` / `length` / `to_entries`
//!   /iteration fast paths skip that dedup (#325). The shape stays
//!   excluded until #325 is closed.
//! * Duplicate keys in `{k: v, k: v'}` literals — jq-jit's optimizer
//!   sometimes drops the earlier value's evaluation when a later one
//!   will rebind the same key, losing any error it would have raised
//!   (#324).
//!
//! ## Knobs
//!
//! * `JQJIT_PROPTEST_CASES` — case budget (default 256, ≤30s on dev hw)
//! * `JQJIT_PROPTEST_TIMEOUT_SECS` — per-subprocess cap (default 3)
//! * `JQ_BIN` — override the reference jq binary (must be jq-1.8.x)
//!
//! Bigger runs:
//!
//! ```bash
//! JQJIT_PROPTEST_CASES=100000 cargo test --release --test fuzz_diff
//! ```
//!
//! When a failure shrinks, paste the minimal `(FilterExpr, JsonShape)`
//! into `tests/regression.test` and start the bug fix from there.
//!
//! ## Extending the generators
//!
//! When a new fast-path lands, decide whether to:
//!
//! 1. **Include it here** — add the new builtin / shape to the lists
//!    below if its grammar is single-valued and its divergence surface
//!    is small.
//! 2. **Leave it to `differential_proptest.rs`** — if the shape needs
//!    its own dedicated contract test (e.g. multi-stream forms), let
//!    the heavier opt-in harness cover it for now.
//!
//! Err on the side of (1): more coverage here means more bugs caught
//! by `cargo test`. Only fall back to (2) when a shape is genuinely
//! divergence-prone in a way that can't be fixed in the same PR.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use proptest::prelude::*;

const IDENT_POOL: &[&str] = &["a", "b", "c", "x", "y"];

/// Single-valued unary builtins with stable jq behaviour. Each has been
/// observed to round-trip via `serde_json` re-parse cleanly across
/// 100k+ proptest cases. When extending, prefer builtins that emit
/// canonical JSON (no `1e10` / `+5` quirks) and stay deterministic on
/// the `(int, bool, str, arr, obj)` input distribution below.
const BUILTIN_UNARY: &[&str] = &[
    "length", "keys", "keys_unsorted", "values", "type",
    "tostring", "to_entries", "reverse", "sort", "min", "max",
    "floor", "ceil", "fabs", "not", "add", "empty", "any", "all",
    "ascii_downcase", "ascii_upcase", "utf8bytelength",
];

#[derive(Debug, Clone)]
enum FilterExpr {
    Identity,
    Field(String),
    Index(i32),
    /// Half-open slice. `.[:]` (both endpoints absent) is excluded —
    /// see module docs.
    SliceLo(i32, Option<i32>),
    SliceHi(Option<i32>, i32),
    ArrayConstruct(Vec<FilterExpr>),
    ObjectConstruct(Vec<(String, FilterExpr)>),
    Pipe(Box<FilterExpr>, Box<FilterExpr>),
    Comma(Box<FilterExpr>, Box<FilterExpr>),
    If(Box<FilterExpr>, Box<FilterExpr>, Box<FilterExpr>),
    Slash(Box<FilterExpr>, Box<FilterExpr>),
    Limit(u32, Box<FilterExpr>),
    Map(Box<FilterExpr>),
    Select(Box<FilterExpr>),
    UnaryBuiltin(&'static str),
    Reduce(Box<FilterExpr>),
    RangeN(u32),
    IntLiteral(i32),
}

fn render(expr: &FilterExpr) -> String {
    match expr {
        FilterExpr::Identity => ".".into(),
        FilterExpr::Field(name) => format!(".{}", name),
        FilterExpr::Index(n) => format!(".[{}]", n),
        FilterExpr::SliceLo(a, b) => {
            let hi = b.map(|v| v.to_string()).unwrap_or_default();
            format!(".[{}:{}]", a, hi)
        }
        FilterExpr::SliceHi(a, b) => {
            let lo = a.map(|v| v.to_string()).unwrap_or_default();
            format!(".[{}:{}]", lo, b)
        }
        FilterExpr::ArrayConstruct(items) => {
            if items.is_empty() { return "[]".into(); }
            let parts: Vec<String> = items.iter().map(render).collect();
            format!("[{}]", parts.join(","))
        }
        FilterExpr::ObjectConstruct(pairs) => {
            if pairs.is_empty() { return "{}".into(); }
            let parts: Vec<String> = pairs
                .iter()
                .map(|(k, v)| format!("{}: ({})", k, render(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        FilterExpr::Pipe(a, b) => format!("({}) | ({})", render(a), render(b)),
        FilterExpr::Comma(a, b) => format!("({}), ({})", render(a), render(b)),
        FilterExpr::If(c, t, e) => {
            format!("if ({}) then ({}) else ({}) end", render(c), render(t), render(e))
        }
        FilterExpr::Slash(a, b) => format!("({}) // ({})", render(a), render(b)),
        FilterExpr::Limit(n, g) => format!("limit({}; {})", n, render(g)),
        FilterExpr::Map(f) => format!("map({})", render(f)),
        FilterExpr::Select(f) => format!("select({})", render(f)),
        FilterExpr::UnaryBuiltin(name) => (*name).to_string(),
        FilterExpr::Reduce(g) => format!(
            "reduce ({}) as $x (0; . + ($x | tonumber? // 0))",
            render(g)
        ),
        FilterExpr::RangeN(n) => format!("range({})", n),
        FilterExpr::IntLiteral(n) => n.to_string(),
    }
}

fn ident_strategy() -> impl Strategy<Value = String> {
    prop::sample::select(IDENT_POOL).prop_map(|s| s.to_string())
}

fn leaf_strategy() -> impl Strategy<Value = FilterExpr> {
    prop_oneof![
        Just(FilterExpr::Identity),
        ident_strategy().prop_map(FilterExpr::Field),
        (-3i32..=3).prop_map(FilterExpr::Index),
        prop::sample::select(BUILTIN_UNARY).prop_map(FilterExpr::UnaryBuiltin),
        (0u32..5).prop_map(FilterExpr::RangeN),
        (-3i32..=3).prop_map(FilterExpr::IntLiteral),
        prop_oneof![
            (-3i32..=3, prop::option::of(-3i32..=3))
                .prop_map(|(a, b)| FilterExpr::SliceLo(a, b)),
            (prop::option::of(-3i32..=3), -3i32..=3)
                .prop_map(|(a, b)| FilterExpr::SliceHi(a, b)),
        ],
    ]
}

fn filter_strategy() -> impl Strategy<Value = FilterExpr> {
    leaf_strategy().prop_recursive(
        4,   // depth
        32,  // total size budget
        4,   // max items per collection / branches
        |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..=3).prop_map(FilterExpr::ArrayConstruct),
                // Duplicate keys are deduped to last-wins by jq's parser
                // before evaluation. jq-jit's optimizer occasionally
                // skips earlier values' evaluation when a later one will
                // rebind the same key, dropping any errors they would
                // have raised (#324). To keep that bug class out of the
                // default-on harness, generate unique keys only — the
                // single-key case is just last-wins by tautology.
                prop::collection::vec(
                    (ident_strategy(), inner.clone()),
                    0..=3,
                ).prop_map(|mut pairs| {
                    let mut seen = std::collections::HashSet::new();
                    pairs.retain(|(k, _)| seen.insert(k.clone()));
                    FilterExpr::ObjectConstruct(pairs)
                }),
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| FilterExpr::Pipe(Box::new(a), Box::new(b))),
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| FilterExpr::Comma(Box::new(a), Box::new(b))),
                (inner.clone(), inner.clone(), inner.clone()).prop_map(|(a, b, c)| {
                    FilterExpr::If(Box::new(a), Box::new(b), Box::new(c))
                }),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| FilterExpr::Slash(Box::new(a), Box::new(b))),
                (1u32..=4, inner.clone()).prop_map(|(n, g)| FilterExpr::Limit(n, Box::new(g))),
                inner.clone().prop_map(|f| FilterExpr::Map(Box::new(f))),
                inner.clone().prop_map(|f| FilterExpr::Select(Box::new(f))),
                inner.clone().prop_map(|g| FilterExpr::Reduce(Box::new(g))),
            ]
        },
    )
}

#[derive(Debug, Clone)]
enum JsonShape {
    Null,
    Bool(bool),
    IntN(i32),
    Str(String),
    Arr(Vec<JsonShape>),
    Obj(Vec<(String, JsonShape)>),
}

fn render_json(v: &JsonShape) -> String {
    match v {
        JsonShape::Null => "null".into(),
        JsonShape::Bool(b) => b.to_string(),
        JsonShape::IntN(n) => n.to_string(),
        JsonShape::Str(s) => serde_json::to_string(s).unwrap(),
        JsonShape::Arr(items) => {
            let parts: Vec<String> = items.iter().map(render_json).collect();
            format!("[{}]", parts.join(","))
        }
        JsonShape::Obj(pairs) => {
            let parts: Vec<String> = pairs
                .iter()
                .map(|(k, v)| format!("{}:{}", serde_json::to_string(k).unwrap(), render_json(v)))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

fn json_leaf() -> impl Strategy<Value = JsonShape> {
    prop_oneof![
        Just(JsonShape::Null),
        any::<bool>().prop_map(JsonShape::Bool),
        (-5i32..=5).prop_map(JsonShape::IntN),
        prop::sample::select(vec!["", "a", "ab", "0", "hello"])
            .prop_map(|s| JsonShape::Str(s.to_string())),
    ]
}

fn json_strategy() -> impl Strategy<Value = JsonShape> {
    json_leaf().prop_recursive(3, 12, 3, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..=3).prop_map(JsonShape::Arr),
            // Input-side dedup is enforced by `parse_json_object`
            // (#233), but the raw-byte fast paths for keys / length /
            // to_entries / iteration scan the bytes directly and still
            // see every duplicate (#325). Until that is closed,
            // dedupe at generation time so the harness exercises the
            // unique-key shape only.
            prop::collection::vec((ident_strategy(), inner.clone()), 0..=3)
                .prop_map(|mut pairs| {
                    let mut seen = std::collections::HashSet::new();
                    pairs.retain(|(k, _)| seen.insert(k.clone()));
                    JsonShape::Obj(pairs)
                }),
        ]
    })
}

struct RunOutput {
    stdout: String,
    is_error: bool,
}

fn run_once(bin: &str, filter: &str, input: &str, timeout: Duration) -> Option<RunOutput> {
    let mut cmd = Command::new(bin);
    cmd.arg("-c").arg(filter);
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
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = child.wait_with_output().ok()?;
                let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    if let Some(sig) = status.signal() {
                        return Some(RunOutput {
                            stdout: format!("<killed by signal {}> stderr: {}", sig, stderr.trim()),
                            is_error: true,
                        });
                    }
                }
                return Some(RunOutput { stdout, is_error: !status.success() });
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Some(RunOutput { stdout: "<timeout>".to_string(), is_error: true });
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(_) => return None,
        }
    }
}

fn normalize(output: &str) -> Result<String, String> {
    let mut lines = Vec::new();
    for line in output.lines() {
        let t = line.trim();
        if t.is_empty() { continue; }
        let v: serde_json::Value = serde_json::from_str(t)
            .map_err(|e| format!("non-JSON `{}`: {}", t, e))?;
        lines.push(serialize_sorted(&normalize_value(v)));
    }
    Ok(lines.join("\n"))
}

fn normalize_value(val: serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match val {
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                if f.is_finite() && f == (f as i64) as f64 && f.abs() < (1i64 << 53) as f64 {
                    return Value::Number(serde_json::Number::from(f as i64));
                }
            }
            Value::Number(n)
        }
        Value::Array(a) => Value::Array(a.into_iter().map(normalize_value).collect()),
        Value::Object(m) => Value::Object(m.into_iter().map(|(k, v)| (k, normalize_value(v))).collect()),
        other => other,
    }
}

fn serialize_sorted(val: &serde_json::Value) -> String {
    use serde_json::Value;
    match val {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap(),
        Value::Array(a) => {
            let items: Vec<String> = a.iter().map(serialize_sorted).collect();
            format!("[{}]", items.join(","))
        }
        Value::Object(m) => {
            let mut entries: Vec<(&String, &Value)> = m.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            let items: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}:{}", serde_json::to_string(k).unwrap(), serialize_sorted(v)))
                .collect();
            format!("{{{}}}", items.join(","))
        }
    }
}

fn resolve_jq() -> Option<String> {
    let candidates: Vec<String> = std::env::var("JQ_BIN")
        .ok()
        .into_iter()
        .chain(std::iter::once("/opt/homebrew/opt/jq/bin/jq".to_string()))
        .chain(std::iter::once("jq".to_string()))
        .collect();
    for cand in &candidates {
        let Ok(out) = Command::new(cand).arg("--version").output() else { continue };
        if !out.status.success() { continue; }
        let ver = String::from_utf8_lossy(&out.stdout);
        if ver.trim().starts_with("jq-1.8.") { return Some(cand.clone()); }
    }
    None
}

#[test]
fn fuzz_diff_against_jq_1_8() {
    let Some(jq) = resolve_jq() else {
        let msg = "no jq-1.8.x binary found. Set JQ_BIN to a jq-1.8.x binary.";
        if std::env::var_os("CI").is_some() {
            panic!("fuzz_diff: {}", msg);
        }
        eprintln!("SKIP fuzz_diff: {}", msg);
        return;
    };
    let jq_jit: PathBuf = env!("CARGO_BIN_EXE_jq-jit").into();
    let jq_jit = jq_jit.to_string_lossy().into_owned();

    let cases: u32 = std::env::var("JQJIT_PROPTEST_CASES")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let timeout_secs: u64 = std::env::var("JQJIT_PROPTEST_TIMEOUT_SECS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let timeout = Duration::from_secs(timeout_secs);

    let compared = std::sync::atomic::AtomicUsize::new(0);
    let both_error = std::sync::atomic::AtomicUsize::new(0);

    let cfg = ProptestConfig {
        cases,
        failure_persistence: None,
        max_shrink_time: 15_000,
        ..ProptestConfig::default()
    };

    let mut runner = proptest::test_runner::TestRunner::new(cfg);
    let strategy = (filter_strategy(), json_strategy());

    let result = runner.run(&strategy, |(expr, input_shape)| {
        let filter = render(&expr);
        let input = render_json(&input_shape);

        let Some(r_jq) = run_once(&jq, &filter, &input, timeout) else {
            return Ok(());
        };
        let Some(r_jit) = run_once(&jq_jit, &filter, &input, timeout) else {
            return Ok(());
        };

        let crash_markers = ["panicked", "SIGSEGV", "Assertion failed", "stack overflow", "RUST_BACKTRACE"];
        if crash_markers.iter().any(|m| r_jit.stdout.contains(m)) {
            return Err(TestCaseError::fail(format!(
                "jq-jit crashed\n  filter: {}\n  input:  {}\n  stderr: {}",
                filter, input, r_jit.stdout.trim()
            )));
        }

        if r_jq.is_error && r_jit.is_error {
            both_error.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        if r_jq.is_error != r_jit.is_error {
            return Err(TestCaseError::fail(format!(
                "error mismatch (jq error={}, jit error={})\n  filter: {}\n  input:  {}\n  jq:  {}\n  jit: {}",
                r_jq.is_error, r_jit.is_error, filter, input,
                r_jq.stdout.trim(), r_jit.stdout.trim()
            )));
        }

        let a_norm = match normalize(&r_jq.stdout) {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };
        let b_norm = match normalize(&r_jit.stdout) {
            Ok(s) => s,
            Err(e) => return Err(TestCaseError::fail(format!(
                "jq-jit emitted non-JSON\n  filter: {}\n  input:  {}\n  err: {}\n  jit: {}",
                filter, input, e, r_jit.stdout.trim()
            ))),
        };

        compared.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if a_norm != b_norm {
            return Err(TestCaseError::fail(format!(
                "value mismatch\n  filter: {}\n  input:  {}\n  jq:  {}\n  jit: {}",
                filter, input, a_norm, b_norm
            )));
        }
        Ok(())
    });

    eprintln!(
        "=== fuzz_diff (vs {}) ===\n  compared: {}\n  both_errored: {}",
        jq,
        compared.load(std::sync::atomic::Ordering::Relaxed),
        both_error.load(std::sync::atomic::Ordering::Relaxed),
    );

    if let Err(e) = result {
        panic!("fuzz_diff failed:\n{}", e);
    }
}
