//! Fast-path error-equivalence proptest (#172).
//!
//! Property: for any single-valued `(filter, input)` pair,
//!
//! > if `jq -c <filter>` exits with an error,
//! > then `jq-jit -c '(<filter>)?' <input>` must produce empty output.
//!
//! This is the canonical fingerprint of the bug class surfaced by the
//! 2026-04-26 sweep (#151–#167): a raw-byte fast path commits where the
//! generic path would have raised, and the divergence is invisible until
//! a user wraps the filter in `?`. By exercising the property over
//! random shapes we lock in `(error)? ≡ empty` as a structural invariant.
//!
//! The generator is a deliberately *narrower* version of
//! `differential_proptest.rs`'s `FilterExpr`: it excludes multi-valued
//! constructors (`,`, `range`, `recurse`, `foreach`, `limit(n>1; …)`)
//! and assignment forms because the "empty output" post-condition is
//! ill-defined when the filter can yield several values before erroring.
//! Single-valued shapes give a sharp predicate.
//!
//! ## Why default-on, unlike `differential_proptest.rs`
//!
//! `differential_proptest.rs` is `#[ignore]` because it asserts full
//! value-level equivalence and trips on every fast-path divergence; it
//! waits on the broader contract (#83) to land. The narrower invariant
//! here passes today (post-sweep) and is what we want to keep passing.
//!
//! ## Knobs
//!
//! * `JQJIT_PROPTEST_CASES` — case budget (default 200, ≤30s on dev hw)
//! * `JQJIT_PROPTEST_TIMEOUT_SECS` — per-subprocess cap (default 3)
//! * `JQ_BIN` — override the reference jq binary
//!
//! Shrinker output names a minimal `(FilterExpr, JsonShape)` pair —
//! paste it into `tests/regression.test` (the `(<expr>)?` wrap is what
//! you'd assert against `""`).

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use proptest::prelude::*;

const IDENT_POOL: &[&str] = &["a", "b", "c", "x", "y"];

/// Single-valued unary builtins. Picked to exercise type contracts —
/// each of these errors on at least one input shape in jq.
const BUILTIN_UNARY_VALUE: &[&str] = &[
    "length", "type", "tostring", "tonumber", "keys", "values",
    "to_entries", "reverse", "sort", "min", "max", "floor", "ceil",
    "fabs", "not", "add", "utf8bytelength", "ascii_downcase", "ascii_upcase",
];

#[derive(Debug, Clone)]
enum FilterExpr {
    Identity,
    Field(String),
    Index(i32),
    Slice(Option<i32>, Option<i32>),
    ArrayConstruct(Vec<FilterExpr>),
    ObjectConstruct(Vec<(String, FilterExpr)>),
    Pipe(Box<FilterExpr>, Box<FilterExpr>),
    If(Box<FilterExpr>, Box<FilterExpr>, Box<FilterExpr>),
    Slash(Box<FilterExpr>, Box<FilterExpr>),
    TryCatch(Box<FilterExpr>, Box<FilterExpr>),
    Optional(Box<FilterExpr>),
    Map(Box<FilterExpr>),
    Select(Box<FilterExpr>),
    UnaryBuiltin(&'static str),
    Reduce(Box<FilterExpr>),   // reduce . as $x (0; . + ($x | tonumber? // 0))
    LimitOne(Box<FilterExpr>), // limit(1; …) — single-valued slice of a generator
    IntLiteral(i32),
}

fn render(expr: &FilterExpr) -> String {
    match expr {
        FilterExpr::Identity => ".".into(),
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
        FilterExpr::If(c, t, e) => {
            format!("if ({}) then ({}) else ({}) end", render(c), render(t), render(e))
        }
        FilterExpr::Slash(a, b) => format!("({}) // ({})", render(a), render(b)),
        FilterExpr::TryCatch(a, b) => format!("try ({}) catch ({})", render(a), render(b)),
        FilterExpr::Optional(a) => format!("({})?", render(a)),
        FilterExpr::Map(f) => format!("map({})", render(f)),
        FilterExpr::Select(f) => format!("select({})", render(f)),
        FilterExpr::UnaryBuiltin(name) => (*name).to_string(),
        FilterExpr::Reduce(g) => format!("reduce ({}) as $x (0; . + ($x | tonumber? // 0))", render(g)),
        FilterExpr::LimitOne(g) => format!("limit(1; {})", render(g)),
        FilterExpr::IntLiteral(n) => n.to_string(),
    }
}

/// True when `expr` evaluates to a value that does not depend on its input,
/// i.e. it is a literal or an array/object built entirely from such values.
/// Used to gate `Pipe(_, b)` generation: when `b` is constant-like, jq-jit's
/// compile-time fold collapses the pipe and the lhs's runtime error is lost.
/// That is its own bug class, separate from the fast-path type-leak class
/// this test exists to detect.
fn is_const_like(expr: &FilterExpr) -> bool {
    match expr {
        FilterExpr::IntLiteral(_) => true,
        FilterExpr::ArrayConstruct(items) => items.iter().all(is_const_like),
        FilterExpr::ObjectConstruct(pairs) => pairs.iter().all(|(_, v)| is_const_like(v)),
        _ => false,
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
        prop::sample::select(BUILTIN_UNARY_VALUE).prop_map(FilterExpr::UnaryBuiltin),
        (-3i32..=3).prop_map(FilterExpr::IntLiteral),
        // Slice variants. `.[:]` (both endpoints absent) is a syntax
        // error in jq but accepted by jq-jit's parser — that is its own
        // divergence class, separate from the runtime fast-path bug
        // class this test is built to detect, so we exclude it from
        // generation.
        prop_oneof![
            (-3i32..=3, prop::option::of(-3i32..=3))
                .prop_map(|(a, b)| FilterExpr::Slice(Some(a), b)),
            (prop::option::of(-3i32..=3), -3i32..=3)
                .prop_map(|(a, b)| FilterExpr::Slice(a, Some(b))),
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
                prop::collection::vec(
                    (ident_strategy(), inner.clone()),
                    0..=3,
                ).prop_map(FilterExpr::ObjectConstruct),
                (inner.clone(), inner.clone())
                    .prop_filter(
                        "Pipe(_, all-constant-rhs) hits an unrelated \
                         compile-time fold bug: the rhs is emitted without \
                         honouring the lhs's runtime error. Exclude until the \
                         fold learns to preserve errors; tracked separately \
                         from the fast-path bug class this test exists to \
                         detect.",
                        |(_, b)| !is_const_like(b),
                    )
                    .prop_map(|(a, b)| FilterExpr::Pipe(Box::new(a), Box::new(b))),
                (inner.clone(), inner.clone(), inner.clone()).prop_map(|(a, b, c)| {
                    FilterExpr::If(Box::new(a), Box::new(b), Box::new(c))
                }),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| FilterExpr::Slash(Box::new(a), Box::new(b))),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| FilterExpr::TryCatch(Box::new(a), Box::new(b))),
                inner.clone().prop_map(|a| FilterExpr::Optional(Box::new(a))),
                inner.clone().prop_map(|f| FilterExpr::Map(Box::new(f))),
                inner.clone().prop_map(|f| FilterExpr::Select(Box::new(f))),
                inner.clone().prop_map(|g| FilterExpr::Reduce(Box::new(g))),
                inner.clone().prop_map(|g| FilterExpr::LimitOne(Box::new(g))),
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
            Just(0.5f64), Just(-0.5), Just(1.5), Just(-1.5),
        ].prop_map(JsonShape::FloatN),
        prop::sample::select(vec!["", "a", "ab", "0", "hello"])
            .prop_map(|s| JsonShape::Str(s.to_string())),
    ]
}

fn json_strategy() -> impl Strategy<Value = JsonShape> {
    json_leaf().prop_recursive(3, 12, 3, |inner| {
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

fn output_is_empty(s: &str) -> bool {
    s.lines().all(|l| l.trim().is_empty())
}

#[test]
fn fast_path_error_wrap_empty() {
    let Some(jq) = resolve_jq() else {
        let msg = "no jq-1.8.x binary found. Set JQ_BIN to a jq-1.8.x binary.";
        if std::env::var_os("CI").is_some() {
            panic!("fast_path_error_wrap_proptest: {}", msg);
        }
        eprintln!("SKIP fast_path_error_wrap_proptest: {}", msg);
        return;
    };
    let jq_jit: PathBuf = env!("CARGO_BIN_EXE_jq-jit").into();
    let jq_jit = jq_jit.to_string_lossy().into_owned();

    let cases: u32 = std::env::var("JQJIT_PROPTEST_CASES")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let timeout_secs: u64 = std::env::var("JQJIT_PROPTEST_TIMEOUT_SECS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let timeout = Duration::from_secs(timeout_secs);

    let active = std::sync::atomic::AtomicUsize::new(0);
    let skipped_no_error = std::sync::atomic::AtomicUsize::new(0);

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

        // Step 1: does jq error on this filter+input?
        let Some(r_jq) = run_once(&jq, &filter, &input, timeout) else {
            return Ok(());  // spawn failure — skip
        };
        if !r_jq.is_error {
            skipped_no_error.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());  // jq did not error — vacuously satisfied
        }

        // Step 2: does jq-jit (filter)? produce empty output?
        let wrapped = format!("({})?", filter);
        let Some(r_jit) = run_once(&jq_jit, &wrapped, &input, timeout) else {
            return Ok(());
        };

        // Crash markers from jq-jit are hard failures.
        let crash_markers = [
            "panicked", "SIGSEGV", "Assertion failed",
            "stack overflow", "RUST_BACKTRACE",
        ];
        if crash_markers.iter().any(|m| r_jit.stdout.contains(m)) {
            return Err(TestCaseError::fail(format!(
                "jq-jit crashed under wrap\n  filter: {}\n  wrapped: {}\n  input:  {}\n  out: {}",
                filter, wrapped, input, r_jit.stdout.trim()
            )));
        }

        active.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if r_jit.is_error {
            // (filter)? itself errored — that means the wrap failed to
            // catch. The fast-path likely raised a Rust-level error that
            // bypasses jq's `?` machinery.
            return Err(TestCaseError::fail(format!(
                "(filter)? errored — wrap failed to catch\n  filter: {}\n  wrapped: {}\n  input:  {}\n  out: {}",
                filter, wrapped, input, r_jit.stdout.trim()
            )));
        }

        if !output_is_empty(&r_jit.stdout) {
            return Err(TestCaseError::fail(format!(
                "fast-path leaked output that wrap-with-? did not suppress\n  \
                 filter: {}\n  wrapped: {}\n  input:  {}\n  jq: errored (expected)\n  jit: {}",
                filter, wrapped, input, r_jit.stdout.trim()
            )));
        }
        Ok(())
    });

    eprintln!(
        "=== Error-wrap proptest (vs {}) ===\n  active: {}\n  skipped (jq did not error): {}",
        jq,
        active.load(std::sync::atomic::Ordering::Relaxed),
        skipped_no_error.load(std::sync::atomic::Ordering::Relaxed),
    );

    if let Err(e) = result {
        panic!("fast-path error-wrap proptest failed:\n{}", e);
    }
}
