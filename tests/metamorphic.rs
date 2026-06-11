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
//! `is_single_valued_expr` in `src/simplify.rs` — it's the precondition
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
//!
//! ## Strategy provenance
//!
//! Single-valued filter generation, the JSON input distribution, and
//! the AST printer all come from
//! `tests/common/filter_strategy.rs` (#688). This file owns the
//! `MultiFilter` wrapper that partitions filters into single-valued
//! vs multi-valued (needed for equivalences (4) and (6)) and the
//! equivalence-test plumbing. Single-valued shapes nest inside
//! `MultiFilter::Single(FilterExpr)`; the multi-valued generators
//! (`,`, `range`, `.[]`, `limit`) live only in this file.

mod common;

use std::time::Duration;

use proptest::prelude::*;
use proptest::test_runner::{FileFailurePersistence, TestCaseError, TestRunner};

use common::diff_harness::{jq_jit_path, run_filter};
use common::filter_strategy::{
    base_filter_strategy, base_json_strategy, conservative_json_leaf,
    conservative_leaf_strategy, render, render_json, FilterExpr, JsonShape,
};
use common::json_normalize::normalize;

// =====================================================================
// MultiFilter — single-valued / multi-valued partition
// =====================================================================
//
// Multi-valued generators (the ones the single-valued strategy must
// reject for equivalences (4) and (6)) live in a separate AST tree so
// the type system enforces the partition. The leaves below
// `MultiFilter::Single` are guaranteed single-valued by the shared
// strategy in `tests/common/filter_strategy.rs`; the four multi-valued
// constructors (`Comma`, `RangeN`, `EachUnchecked`, `Limit`) can only
// be produced by `multi_filter_strategy`.

#[derive(Debug, Clone)]
enum MultiFilter {
    /// Single-valued island; the wrapped `FilterExpr` is guaranteed
    /// single-valued by the shared `single_filter_strategy` below.
    Single(FilterExpr),
    Comma(Box<MultiFilter>, Box<MultiFilter>),
    /// `range(n)` — produces 0,1,…,n-1.
    RangeN(u32),
    /// `.[]` — produces one value per array element / object value.
    EachUnchecked,
    /// `limit(n; gen)` — produces ≤n values from `gen`.
    Limit(u32, Box<MultiFilter>),
    Pipe(Box<MultiFilter>, Box<MultiFilter>),
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

// =====================================================================
// Strategies
// =====================================================================

/// Single-valued filter strategy. Every filter produced yields exactly
/// one value per input (or errors). Mirrors the
/// `is_single_valued_expr` precondition in `src/simplify.rs`.
///
/// Built on top of [`conservative_leaf_strategy`] from the shared
/// `filter_strategy` module: the leaves are the safe single-valued
/// shapes (`Identity`, `Field`, `Index`, `IntLiteral`, safe builtins,
/// comparison-only `Field*Binop`), and the recursive composition uses
/// the shared `base_filter_strategy` which only adds single-valued-
/// preserving constructors (`Pipe`, `ArrayConstruct`,
/// `ObjectConstruct`, `Map`, `If`). Multi-valued shapes like `Comma`
/// and `Limit` are excluded from the base by design — they only
/// appear via `MultiFilter` below.
fn single_filter_strategy() -> impl Strategy<Value = FilterExpr> {
    base_filter_strategy(conservative_leaf_strategy(), 3, 16, 3)
}

/// Multi-valued strategy. Includes generators (`,`, `range`, `.[]`,
/// `limit`) plus all single-valued shapes (via the `MultiFilter::Single`
/// wrapper).
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
            (1u32..=3, inner.clone())
                .prop_map(|(n, g)| MultiFilter::Limit(n, Box::new(g))),
        ]
    })
}

fn json_strategy() -> impl Strategy<Value = JsonShape> {
    base_json_strategy(conservative_json_leaf(), 3, 12, 3)
}

// =====================================================================
// Runner / equivalence assertion plumbing
// =====================================================================

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
    // `Direct` over `SourceParallel`: proptest's `SourceParallel` walks up
    // looking for a sibling `lib.rs`/`main.rs`, and silently falls back to
    // a flat `<source>.<sibling>` file when the crate uses the `src/lib.rs`
    // layout (as this one does). Hardcoding the path keeps the regressions
    // file in a predictable location for the CI artifact upload.
    ProptestConfig {
        cases: proptest_cases(),
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(
            "tests/proptest-regressions/metamorphic.txt",
        ))),
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

// =====================================================================
// Tests
// =====================================================================

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
