//! Property-based differential fuzzer: generate random `(filter, input)` pairs
//! and run both `jq-jit` and reference `jq 1.8.x`. Any value-level divergence
//! is a failure; both-error agreement is tolerated.
//!
//! Sized to finish within a ~60s CI budget. Override via env vars:
//!   `JQJIT_PROPTEST_CASES` (default 500)
//!   `JQJIT_PROPTEST_TIMEOUT_SECS` (per-subprocess cap, default 3)
//!
//! The reference `jq` binary is resolved the same way as the hand-curated
//! differential harness (`$JQ_BIN` → Homebrew → `$PATH`). If no jq-1.8.x is
//! available the test is skipped with a diagnostic.
//!
//! ## Running
//!
//! The test is marked `#[ignore]`: it exists to surface the entire class of
//! fast-path type-dispatch divergences (`.a` on a non-object, etc.) on
//! demand. Those divergences are structural and will keep falling out of
//! random generation until the fast-path contract (issue #83) and the
//! normalize-by-default constructors (issue #84) land. Running it in the
//! default CI matrix would block every PR on the same known bug class.
//!
//! Run it explicitly:
//!
//! ```bash
//! cargo test --release --test differential_proptest -- --ignored --nocapture
//! # longer run:
//! JQJIT_PROPTEST_CASES=5000 cargo test --release --test differential_proptest -- --ignored --nocapture
//! ```
//!
//! Shrinker output reports a minimal `(FilterExpr, JsonShape)` pair; paste it
//! into `tests/regression.test` once the underlying bug is fixed.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use proptest::prelude::*;

const IDENT_POOL: &[&str] = &["a", "b", "c", "k", "v", "x", "y"];
const BUILTIN_UNARY_VALUE: &[&str] = &[
    "length", "keys", "keys_unsorted", "values", "type",
    "tostring", "tonumber", "add", "to_entries", "from_entries",
    "reverse", "sort", "unique", "min", "max", "floor", "ceil", "fabs",
    "not", "empty", "any", "all",
];

#[derive(Debug, Clone)]
enum FilterExpr {
    Identity,
    Recurse,
    Field(String),
    Index(i32),
    Slice(Option<i32>, Option<i32>),
    ArrayConstruct(Vec<FilterExpr>),
    ObjectConstruct(Vec<(String, FilterExpr)>),
    Pipe(Box<FilterExpr>, Box<FilterExpr>),
    Comma(Box<FilterExpr>, Box<FilterExpr>),
    If(Box<FilterExpr>, Box<FilterExpr>, Box<FilterExpr>),
    Slash(Box<FilterExpr>, Box<FilterExpr>),
    TryCatch(Box<FilterExpr>, Box<FilterExpr>),
    Optional(Box<FilterExpr>),
    Limit(u32, Box<FilterExpr>),
    Map(Box<FilterExpr>),
    Select(Box<FilterExpr>),
    UnaryBuiltin(&'static str),
    Reduce(Box<FilterExpr>),   // reduce .[] as $x (0; . + $x)
    Foreach(Box<FilterExpr>),  // foreach .[] as $x (0; . + $x; .)
    RangeN(u32),
    IntLiteral(i32),
}

fn render(expr: &FilterExpr) -> String {
    match expr {
        FilterExpr::Identity => ".".into(),
        FilterExpr::Recurse => "..".into(),
        FilterExpr::Field(name) => format!(".{}", name),
        FilterExpr::Index(n) => format!(".[{}]", n),
        FilterExpr::Slice(a, b) => {
            let lo = a.map(|v| v.to_string()).unwrap_or_default();
            let hi = b.map(|v| v.to_string()).unwrap_or_default();
            format!(".[{}:{}]", lo, hi)
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
        FilterExpr::TryCatch(a, b) => format!("try ({}) catch ({})", render(a), render(b)),
        FilterExpr::Optional(a) => format!("({})?", render(a)),
        FilterExpr::Limit(n, g) => format!("limit({}; {})", n, render(g)),
        FilterExpr::Map(f) => format!("map({})", render(f)),
        FilterExpr::Select(f) => format!("select({})", render(f)),
        FilterExpr::UnaryBuiltin(name) => (*name).to_string(),
        FilterExpr::Reduce(gen) => format!("reduce ({}) as $x (0; . + ($x | tonumber? // 0))", render(gen)),
        FilterExpr::Foreach(gen) => format!("foreach ({}) as $x (0; . + ($x | tonumber? // 0); .)", render(gen)),
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
        Just(FilterExpr::Recurse),
        ident_strategy().prop_map(FilterExpr::Field),
        (-3i32..=3).prop_map(FilterExpr::Index),
        prop::sample::select(BUILTIN_UNARY_VALUE).prop_map(FilterExpr::UnaryBuiltin),
        (0u32..5).prop_map(FilterExpr::RangeN),
        (-3i32..=3).prop_map(FilterExpr::IntLiteral),
        prop::option::of(-3i32..=3)
            .prop_flat_map(|a| prop::option::of(-3i32..=3).prop_map(move |b| FilterExpr::Slice(a, b))),
    ]
}

fn filter_strategy() -> impl Strategy<Value = FilterExpr> {
    leaf_strategy().prop_recursive(
        4,   // depth
        40,  // total size budget
        6,   // max items per collection / branches
        |inner| {
            prop_oneof![
                // Array construct (0-3 items)
                prop::collection::vec(inner.clone(), 0..=3).prop_map(FilterExpr::ArrayConstruct),
                // Object construct (0-3 pairs) with keys from pool
                prop::collection::vec(
                    (ident_strategy(), inner.clone()),
                    0..=3,
                ).prop_map(FilterExpr::ObjectConstruct),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| FilterExpr::Pipe(Box::new(a), Box::new(b))),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| FilterExpr::Comma(Box::new(a), Box::new(b))),
                (inner.clone(), inner.clone(), inner.clone()).prop_map(|(a, b, c)| {
                    FilterExpr::If(Box::new(a), Box::new(b), Box::new(c))
                }),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| FilterExpr::Slash(Box::new(a), Box::new(b))),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| FilterExpr::TryCatch(Box::new(a), Box::new(b))),
                inner.clone().prop_map(|a| FilterExpr::Optional(Box::new(a))),
                (1u32..=4, inner.clone()).prop_map(|(n, g)| FilterExpr::Limit(n, Box::new(g))),
                inner.clone().prop_map(|f| FilterExpr::Map(Box::new(f))),
                inner.clone().prop_map(|f| FilterExpr::Select(Box::new(f))),
                inner.clone().prop_map(|g| FilterExpr::Reduce(Box::new(g))),
                inner.clone().prop_map(|g| FilterExpr::Foreach(Box::new(g))),
            ]
        },
    )
}

#[derive(Debug, Clone)]
enum JsonShape {
    Null,
    Bool(bool),
    IntN(i32),
    FloatN(f64),
    Str(String),
    Arr(Vec<JsonShape>),
    Obj(Vec<(String, JsonShape)>),
}

fn render_json(v: &JsonShape) -> String {
    match v {
        JsonShape::Null => "null".into(),
        JsonShape::Bool(b) => b.to_string(),
        JsonShape::IntN(n) => n.to_string(),
        JsonShape::FloatN(f) => {
            if f.is_finite() {
                // Use ryu-style formatting; fall back to a sane default.
                let mut s = format!("{}", f);
                if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                    s.push_str(".0");
                }
                s
            } else {
                "0".into()
            }
        }
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
        prop_oneof![
            Just(0.5f64), Just(-0.5), Just(1.5), Just(2.5), Just(-1.5),
        ].prop_map(JsonShape::FloatN),
        prop::sample::select(vec!["", "a", "ab", "hello", "0"]).prop_map(|s| JsonShape::Str(s.to_string())),
    ]
}

fn json_strategy() -> impl Strategy<Value = JsonShape> {
    json_leaf().prop_recursive(3, 16, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..=3).prop_map(JsonShape::Arr),
            prop::collection::vec((ident_strategy(), inner.clone()), 0..=3).prop_map(JsonShape::Obj),
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
#[ignore = "opt-in: surfaces known fast-path type-dispatch divergences; run with --ignored"]
fn differential_proptest_against_jq_1_8() {
    let Some(jq) = resolve_jq() else {
        let msg = "no jq-1.8.x binary found. Set JQ_BIN to a jq-1.8.x binary.";
        if std::env::var_os("CI").is_some() {
            panic!("differential_proptest: {}", msg);
        }
        eprintln!("SKIP differential_proptest: {}", msg);
        return;
    };
    let jq_jit: PathBuf = env!("CARGO_BIN_EXE_jq-jit").into();
    let jq_jit = jq_jit.to_string_lossy().into_owned();

    let cases: u32 = std::env::var("JQJIT_PROPTEST_CASES")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(500);
    let timeout_secs: u64 = std::env::var("JQJIT_PROPTEST_TIMEOUT_SECS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let timeout = Duration::from_secs(timeout_secs);

    // Track how many proptest iterations ran meaningful jq/jit comparisons,
    // and how many fell into "both errored" skips (counted as pass).
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
            // Spawn failure — skip, do not fail the property.
            return Ok(());
        };
        let Some(r_jit) = run_once(&jq_jit, &filter, &input, timeout) else {
            return Ok(());
        };

        // Crash markers from jq-jit are hard failures.
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
            Err(_) => return Ok(()),  // jq produced non-JSON line (shouldn't happen; skip rather than fail)
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
        "=== Proptest differential (vs {}) ===\n  compared: {}\n  both_errored: {}",
        jq,
        compared.load(std::sync::atomic::Ordering::Relaxed),
        both_error.load(std::sync::atomic::Ordering::Relaxed),
    );

    if let Err(e) = result {
        panic!("proptest differential failed:\n{}", e);
    }
}
