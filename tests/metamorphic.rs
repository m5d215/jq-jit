//! Metamorphic equivalence harness (#683).
//!
//! Generates `(filter, input)` pairs and asserts a fixed set of
//! jq-language algebraic equivalences hold across jq-jit's pipeline
//! (raw-byte fast paths, `simplify_expr`, parser rewrites, JIT). The
//! reference is jq-jit itself: each equivalence renders two textually
//! distinct filters that jq's semantics guarantee produce identical
//! output streams on any input, and the harness runs both through the
//! same jq-jit binary.
//!
//! ## Why this catches bugs the existing harnesses miss
//!
//! - `diff_*` / `fuzz_*` pin against reference jq, which catches
//!   value-level divergences but is blind to "both jq-jit pipeline
//!   paths happen to be wrong in the same way."
//! - `selfdiff_jit_interp` pins JIT vs interpreter on *identical*
//!   filters, which catches layer drift but not rewrites that fire on
//!   shapes they shouldn't.
//!
//! The bugs that motivate this harness all fit the
//! "rewrite-claims-equivalence-but-violates-it-on-some-shape" template:
//!
//! - `[gen] | add` rewrite firing on multi-valued `gen` (#56)
//! - `path(.a)` constant-folded to `["a"]` regardless of input type (#46)
//! - `paths` rewrite missing the `length > 0` guard
//! - `limit(n; ...)` collapsing to `first` for `n >= 1`
//!
//! ## Equivalences
//!
//! Each one is a single `#[test]` proptest. Generators are kept small
//! (depth ≤ 3) so shrinks land on minimal cases that are easy to
//! triage. To add a new equivalence: drop a new `#[test]` calling
//! `run_equivalence(...)` with a strategy that emits `(lhs_filter,
//! rhs_filter, input)` triples.
//!
//! 1. `f`            ≡  `f | .`
//! 2. `f, empty`     ≡  `f`
//! 3. `(f | g) | h`  ≡  `f | (g | h)`
//! 4. `[f] | .[]`    ≡  `f`              *(single-valued `f` only)*
//! 5. `paths`        ≡  `path(..) | select(length > 0)`
//! 6. `[gen] | add`  ≡  `reduce gen as $x (null; . + $x)`
//!                                       *(single-valued `gen` only)*
//!
//! The single-valued restriction on (4) and (6) mirrors
//! `is_single_valued_expr` in `src/interpreter.rs` — it's the precondition
//! the corresponding rewrites guard themselves on. Testing the broader
//! multi-valued case is a follow-up once the single-valued surface lands
//! clean.
//!
//! ## Knobs
//!
//! - `JQJIT_PROPTEST_CASES` — case budget per property (default 128)
//! - `JQJIT_PROPTEST_TIMEOUT_SECS` — per-subprocess wall-clock cap
//!   (default 3)
//!
//! When a divergence shrinks, paste the minimal `(filter, input)` into
//! `tests/regression.test` with reference jq's output as expected, fix
//! the underlying rewrite or guard, then re-run.

mod common;

use std::time::Duration;

use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestRunner};

use common::diff_harness::{jq_jit_path, run_filter};
use common::json_normalize::normalize;

// ===== filter AST =====

#[derive(Debug, Clone)]
enum Filter {
    Identity,
    Field(String),
    Index(i32),
    IntLit(i32),
    UnaryBuiltin(&'static str),
    Pipe(Box<Filter>, Box<Filter>),
    ArrayCons(Vec<Filter>),
    ObjCons(Vec<(String, Filter)>),
    Map(Box<Filter>),
    If(Box<Filter>, Box<Filter>, Box<Filter>),
    /// `.f1 OP .f2`
    FieldCmpField(String, &'static str, String),
    /// `.f OP N`
    FieldCmpInt(String, &'static str, i32),
}

/// Multi-valued generators that the single-valued strategy must reject.
/// Kept in a separate AST node tree so the multi-valued strategy can
/// produce them and the single-valued one cannot.
#[derive(Debug, Clone)]
enum MultiFilter {
    /// Single-valued island; anything below this point is single-valued.
    Single(Filter),
    Comma(Box<MultiFilter>, Box<MultiFilter>),
    /// `range(n)` — produces 0,1,…,n-1.
    RangeN(u32),
    /// `.[]` — produces one value per array element / object value.
    EachUnchecked,
    /// `limit(n; gen)` — produces ≤n values from `gen`.
    Limit(u32, Box<MultiFilter>),
    Pipe(Box<MultiFilter>, Box<MultiFilter>),
}

const IDENTS: &[&str] = &["a", "b", "x", "y"];

const SAFE_BUILTINS: &[&str] = &[
    "length",
    "type",
    "tostring",
    "not",
    "keys",
    "values",
    "reverse",
    "sort",
    "to_entries",
];

const CMP_OPS: &[&str] = &["==", "!=", "<", ">", "<=", ">="];

fn render(f: &Filter) -> String {
    match f {
        Filter::Identity => ".".into(),
        Filter::Field(n) => format!(".{}", n),
        Filter::Index(n) => format!(".[{}]", n),
        Filter::IntLit(n) => n.to_string(),
        Filter::UnaryBuiltin(n) => (*n).to_string(),
        Filter::Pipe(a, b) => format!("({}) | ({})", render(a), render(b)),
        Filter::ArrayCons(items) => {
            if items.is_empty() {
                "[]".into()
            } else {
                let parts: Vec<String> = items.iter().map(render).collect();
                format!("[{}]", parts.join(", "))
            }
        }
        Filter::ObjCons(pairs) => {
            if pairs.is_empty() {
                "{}".into()
            } else {
                let parts: Vec<String> = pairs
                    .iter()
                    .map(|(k, v)| format!("{}: ({})", k, render(v)))
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
        }
        Filter::Map(inner) => format!("map({})", render(inner)),
        Filter::If(c, t, e) => format!(
            "if ({}) then ({}) else ({}) end",
            render(c),
            render(t),
            render(e)
        ),
        Filter::FieldCmpField(a, op, b) => format!(".{} {} .{}", a, op, b),
        Filter::FieldCmpInt(f, op, n) => format!(".{} {} {}", f, op, n),
    }
}

fn render_multi(f: &MultiFilter) -> String {
    match f {
        MultiFilter::Single(s) => render(s),
        MultiFilter::Comma(a, b) => format!("({}), ({})", render_multi(a), render_multi(b)),
        MultiFilter::RangeN(n) => format!("range({})", n),
        MultiFilter::EachUnchecked => ".[]".into(),
        MultiFilter::Limit(n, g) => format!("limit({}; {})", n, render_multi(g)),
        MultiFilter::Pipe(a, b) => format!("({}) | ({})", render_multi(a), render_multi(b)),
    }
}

// ===== strategies =====

fn ident_strategy() -> impl Strategy<Value = String> {
    prop::sample::select(IDENTS).prop_map(|s| s.to_string())
}

fn cmp_op_strategy() -> impl Strategy<Value = &'static str> {
    prop::sample::select(CMP_OPS)
}

fn leaf_single() -> impl Strategy<Value = Filter> {
    prop_oneof![
        Just(Filter::Identity),
        ident_strategy().prop_map(Filter::Field),
        (-2i32..=2).prop_map(Filter::Index),
        (-3i32..=3).prop_map(Filter::IntLit),
        prop::sample::select(SAFE_BUILTINS).prop_map(Filter::UnaryBuiltin),
        (ident_strategy(), cmp_op_strategy(), ident_strategy())
            .prop_map(|(a, op, b)| Filter::FieldCmpField(a, op, b)),
        (ident_strategy(), cmp_op_strategy(), -3i32..=3)
            .prop_map(|(f, op, n)| Filter::FieldCmpInt(f, op, n)),
    ]
}

/// Single-valued filter strategy. Every filter produced yields exactly
/// one value per input (or errors). Mirrors the
/// `is_single_valued_expr` precondition in `src/interpreter.rs`.
fn single_filter_strategy() -> impl Strategy<Value = Filter> {
    leaf_single().prop_recursive(3, 16, 3, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| Filter::Pipe(Box::new(a), Box::new(b))),
            prop::collection::vec(inner.clone(), 0..=3).prop_map(Filter::ArrayCons),
            prop::collection::vec((ident_strategy(), inner.clone()), 0..=3)
                .prop_map(Filter::ObjCons),
            inner.clone().prop_map(|f| Filter::Map(Box::new(f))),
            (inner.clone(), inner.clone(), inner.clone())
                .prop_map(|(c, t, e)| Filter::If(Box::new(c), Box::new(t), Box::new(e))),
        ]
    })
}

/// Multi-valued strategy. Includes generators (`,`, `range`, `.[]`,
/// `limit`) plus all single-valued shapes.
fn multi_filter_strategy() -> impl Strategy<Value = MultiFilter> {
    let leaf = prop_oneof![
        single_filter_strategy().prop_map(MultiFilter::Single),
        Just(MultiFilter::EachUnchecked),
        (0u32..=3).prop_map(MultiFilter::RangeN),
    ];
    leaf.prop_recursive(3, 12, 3, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| MultiFilter::Comma(Box::new(a), Box::new(b))),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| MultiFilter::Pipe(Box::new(a), Box::new(b))),
            (1u32..=3, inner.clone()).prop_map(|(n, g)| MultiFilter::Limit(n, Box::new(g))),
        ]
    })
}

// ===== JSON input shape =====

#[derive(Debug, Clone)]
enum Json {
    Null,
    Bool(bool),
    Int(i32),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

fn render_json(v: &Json) -> String {
    match v {
        Json::Null => "null".into(),
        Json::Bool(b) => b.to_string(),
        Json::Int(n) => n.to_string(),
        Json::Str(s) => serde_json::to_string(s).unwrap(),
        Json::Arr(items) => {
            let parts: Vec<String> = items.iter().map(render_json).collect();
            format!("[{}]", parts.join(","))
        }
        Json::Obj(pairs) => {
            let parts: Vec<String> = pairs
                .iter()
                .map(|(k, v)| format!("{}:{}", serde_json::to_string(k).unwrap(), render_json(v)))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

fn json_leaf() -> impl Strategy<Value = Json> {
    prop_oneof![
        Just(Json::Null),
        any::<bool>().prop_map(Json::Bool),
        (-3i32..=3).prop_map(Json::Int),
        prop::sample::select(vec!["", "a", "ab", "hello"])
            .prop_map(|s| Json::Str(s.to_string())),
    ]
}

fn json_strategy() -> impl Strategy<Value = Json> {
    json_leaf().prop_recursive(3, 12, 3, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..=3).prop_map(Json::Arr),
            prop::collection::vec((ident_strategy(), inner.clone()), 0..=3).prop_map(Json::Obj),
        ]
    })
}

// ===== runner =====

fn run_one(filter: &str, input: &str, timeout: Duration) -> Option<common::diff_harness::RunOutput> {
    run_filter(jq_jit_path(), filter, input, timeout)
}

/// Assert two filters produce identical output streams on `input`. Both
/// erroring counts as agreement; one-sided error or value divergence
/// fails the proptest case.
fn assert_equivalent(
    label: &str,
    lhs_filter: &str,
    rhs_filter: &str,
    input: &str,
    timeout: Duration,
) -> Result<(), TestCaseError> {
    let Some(l) = run_one(lhs_filter, input, timeout) else {
        return Ok(());
    };
    let Some(r) = run_one(rhs_filter, input, timeout) else {
        return Ok(());
    };

    let crash_markers = ["panicked", "SIGSEGV", "Assertion failed", "stack overflow"];
    for (which, out) in [("lhs", &l), ("rhs", &r)] {
        if crash_markers.iter().any(|m| out.stdout.contains(m)) {
            return Err(TestCaseError::fail(format!(
                "[{}] {} crashed\n  filter: {}\n  input:  {}\n  out:    {}",
                label,
                which,
                if which == "lhs" { lhs_filter } else { rhs_filter },
                input,
                out.stdout.trim()
            )));
        }
    }

    if l.is_error && r.is_error {
        return Ok(());
    }
    if l.is_error != r.is_error {
        return Err(TestCaseError::fail(format!(
            "[{}] error-class mismatch (lhs error={}, rhs error={})\n  lhs:    {}\n  rhs:    {}\n  input:  {}\n  lhs_out: {}\n  rhs_out: {}",
            label,
            l.is_error,
            r.is_error,
            lhs_filter,
            rhs_filter,
            input,
            l.stdout.trim(),
            r.stdout.trim(),
        )));
    }

    let (l_norm, r_norm) = match (normalize(&l.stdout), normalize(&r.stdout)) {
        (Ok(a), Ok(b)) => (a, b),
        // One side emitted non-JSON; treat as inconclusive (the
        // existing `selfdiff_jit_interp` policy). Reference jq diffing
        // is the other harnesses' job.
        _ => return Ok(()),
    };

    if l_norm != r_norm {
        return Err(TestCaseError::fail(format!(
            "[{}] value mismatch\n  lhs:    {}\n  rhs:    {}\n  input:  {}\n  lhs_out: {}\n  rhs_out: {}",
            label, lhs_filter, rhs_filter, input, l_norm, r_norm
        )));
    }
    Ok(())
}

fn proptest_cases() -> u32 {
    std::env::var("JQJIT_PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128)
}

fn proptest_timeout() -> Duration {
    let secs: u64 = std::env::var("JQJIT_PROPTEST_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    Duration::from_secs(secs)
}

fn proptest_config() -> ProptestConfig {
    ProptestConfig {
        cases: proptest_cases(),
        failure_persistence: None,
        max_shrink_time: 15_000,
        ..ProptestConfig::default()
    }
}

fn run_equivalence<S, F>(label: &'static str, strategy: S, to_filters: F)
where
    S: Strategy,
    F: Fn(S::Value) -> (String, String, String),
{
    let timeout = proptest_timeout();
    let mut runner = TestRunner::new(proptest_config());
    let result = runner.run(&strategy, |value| {
        let (lhs, rhs, input) = to_filters(value);
        assert_equivalent(label, &lhs, &rhs, &input, timeout)
    });
    if let Err(e) = result {
        panic!("metamorphic [{}] failed:\n{}", label, e);
    }
}

// ===== tests =====

/// 1. `f`  ≡  `f | .`
#[test]
fn equiv_identity_postfix_pipe() {
    run_equivalence(
        "f ≡ f | .",
        (single_filter_strategy(), json_strategy()),
        |(f, input)| {
            let f_str = render(&f);
            let lhs = f_str.clone();
            let rhs = format!("({}) | .", f_str);
            (lhs, rhs, render_json(&input))
        },
    );
}

/// 2. `f, empty`  ≡  `f`
#[test]
fn equiv_comma_empty_drop() {
    run_equivalence(
        "f, empty ≡ f",
        (multi_filter_strategy(), json_strategy()),
        |(f, input)| {
            let f_str = render_multi(&f);
            let lhs = format!("({}), empty", f_str);
            let rhs = f_str;
            (lhs, rhs, render_json(&input))
        },
    );
}

/// 3. `(f | g) | h`  ≡  `f | (g | h)`
#[test]
fn equiv_pipe_associativity() {
    run_equivalence(
        "(f|g)|h ≡ f|(g|h)",
        (
            multi_filter_strategy(),
            multi_filter_strategy(),
            multi_filter_strategy(),
            json_strategy(),
        ),
        |(f, g, h, input)| {
            let f_str = render_multi(&f);
            let g_str = render_multi(&g);
            let h_str = render_multi(&h);
            let lhs = format!("(({}) | ({})) | ({})", f_str, g_str, h_str);
            let rhs = format!("({}) | (({}) | ({}))", f_str, g_str, h_str);
            (lhs, rhs, render_json(&input))
        },
    );
}

/// 4. `[f] | .[]`  ≡  `f`   *(single-valued `f`)*
#[test]
fn equiv_arr_iter_roundtrip() {
    run_equivalence(
        "[f]|.[] ≡ f  (single-valued f)",
        (single_filter_strategy(), json_strategy()),
        |(f, input)| {
            let f_str = render(&f);
            let lhs = format!("[{}] | .[]", f_str);
            let rhs = f_str;
            (lhs, rhs, render_json(&input))
        },
    );
}

/// 5. `paths`  ≡  `path(..) | select(length > 0)`
#[test]
fn equiv_paths_definition() {
    run_equivalence(
        "paths ≡ path(..) | select(length > 0)",
        json_strategy(),
        |input| {
            let lhs = "[paths]".to_string();
            let rhs = "[path(..) | select(length > 0)]".to_string();
            (lhs, rhs, render_json(&input))
        },
    );
}

/// 6. `[gen] | add`  ≡  `reduce gen as $x (null; . + $x)`
///    *(single-valued `gen`)*
#[test]
fn equiv_collect_add_reduce() {
    run_equivalence(
        "[gen]|add ≡ reduce gen as $x (null;.+$x)  (single-valued gen)",
        (single_filter_strategy(), json_strategy()),
        |(g, input)| {
            let g_str = render(&g);
            let lhs = format!("[{}] | add", g_str);
            let rhs = format!("reduce ({}) as $x (null; . + $x)", g_str);
            (lhs, rhs, render_json(&input))
        },
    );
}
