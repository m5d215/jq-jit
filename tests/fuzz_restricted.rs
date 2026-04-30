//! Property-based differential fuzz harness (#319).
//!
//! Goal: surface jq-compat divergences that no hand-curated test would
//! think to enumerate. Generates random `(filter, input)` pairs from a
//! deliberately *conservative* AST/JSON distribution, runs both jq-jit
//! and reference `jq 1.8.x`, and fails on any value-level divergence.
//!
//! ## Why "conservative"
//!
//! `tests/fuzz_full.rs` already exists as a heavier opt-in
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
//!   their own bug class. The opt-in `fuzz_full.rs` covers
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
//!
//! ## Expected-divergence classes
//!
//! Some shapes legitimately produce different stdout on jq vs jq-jit
//! without either being wrong. The harness is designed to land those
//! in the "both errored → skip" branch (or to mask the difference via
//! `normalize`); when adding new adversarial shapes, check the value
//! against this list before widening the generator:
//!
//! * **Non-finite floats** (`NaN`, `±Infinity`, `-0.0`, denormals,
//!   values beyond the f64 mantissa boundary). jq's printer emits
//!   non-canonical forms (`1.797e+308`, `nan`) that the harness's
//!   `serde_json` re-parse rejects, while jq-jit may print canonical
//!   JSON. The integer adversarial pool deliberately caps at `±2^53`;
//!   floats beyond that are left to a future phase that extends
//!   `normalize` to recognise both spellings.
//! * **Broken UTF-8** (lone surrogates such as `"\uD83D"` without the
//!   trailing low surrogate, embedded NUL bytes). jq-1.8 rejects these
//!   at input parse while jq-jit's parser accepts them, producing an
//!   error-vs-success mismatch the harness would flag as a bug. The
//!   adversarial string pool stays inside well-formed UTF-8 — multi-
//!   byte text, RTL, combining marks, supplementary-plane emoji, and
//!   mid-string BOM all round-trip cleanly through both implementations
//!   and are safe to include.
//! * **Empty input** — both implementations agree to error, so the
//!   `both_error` branch covers it.
//!
//! When a new adversarial generator surfaces a true divergence, mint a
//! permanent regression case in `tests/regression.test` and the value
//! stays in the generator. When it surfaces an *expected* divergence,
//! either extend `normalize` to fold both sides together or filter the
//! shape out of the generator with an inline comment pointing back
//! here.
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
//! JQJIT_PROPTEST_CASES=100000 cargo test --release --test fuzz_restricted
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
//! 2. **Leave it to `fuzz_full.rs`** — if the shape needs
//!    its own dedicated contract test (e.g. multi-stream forms), let
//!    the heavier opt-in harness cover it for now.
//!
//! Err on the side of (1): more coverage here means more bugs caught
//! by `cargo test`. Only fall back to (2) when a shape is genuinely
//! divergence-prone in a way that can't be fixed in the same PR.

mod common;

use std::time::Duration;

use proptest::prelude::*;

use common::diff_harness::{jq_jit_path, require_jq, run_filter};
use common::json_normalize::normalize;

const TEST_LABEL: &str = "fuzz_restricted";

const IDENT_POOL: &[&str] = &["a", "b", "c", "x", "y"];

/// String-literal pool for `select(.f == "lit")` shapes. Overlaps with
/// the JSON leaf string pool so matches occur often enough for the
/// select_str_* fast paths to actually fire.
const STR_LITERAL_POOL: &[&str] = &["", "a", "ab", "0", "hello"];

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
    // Non-finite producers (#321 phase 3b). Both ignore input and
    // emit the same canonical form on jq-1.8 and jq-jit:
    //   infinite -> 1.7976931348623157e+308 (jq stores Infinity as
    //               f64::MAX in its number printer)
    //   nan      -> null (NaN serializes as JSON null)
    // Combinations like `infinite + 1`, `infinite > 0`, `nan == nan`
    // were verified identical via probe before adding to the pool.
    "infinite", "nan",
];

/// Float literals used as filter leaves (#321 phase 3b). Covers the
/// f64 boundary cross-section the issue calls out: `f64::MAX`,
/// `f64::MIN`, denormals, and overflow spellings. Inputs that
/// overflow to Infinity (`1e500`) print as `1E+500` on both
/// implementations; `serde_json` rejects that as out-of-range, so
/// the harness lands the case in the jq-side normalize-Err branch and
/// skips. The eval/JIT path is still exercised, which is the goal.
/// Pure float literals stay in the *filter* per the module-doc
/// exclusion on input-side floats.
const FLOAT_LITERALS: &[&str] = &[
    "1.7976931348623157e+308",  // exact f64::MAX
    "-1.7976931348623157e+308", // exact -f64::MAX
    "1e308",                    // just under f64::MAX
    "-1e308",
    "1e500",                    // overflows to Infinity printer
    "-1e500",
    "1e-300",                   // small normal
    "5e-324",                   // smallest subnormal
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
    /// Float literal as a static spelling, drawn from
    /// `FLOAT_LITERALS` (#321 phase 3b). Stored as `&'static str`
    /// rather than `f64` so the rendered filter matches the source
    /// spelling exactly — relevant for overflow forms (`1e500`)
    /// where `f64::to_string` would emit `inf` instead.
    FloatLiteral(&'static str),
    /// `.f1 op .f2` — exercises the FieldCmpField / FieldOpField fast
    /// paths the leaf shapes don't otherwise reach (#347).
    FieldFieldBinop(String, BinopOp, String),
    /// `.field op N` and `N op .field` — exercises FieldCmpConst /
    /// FieldOpConst / ConstOpField shapes, which were already in the
    /// distribution via leaf composition but worth a direct hit.
    FieldConstBinop(String, BinopOp, i32),
    ConstFieldBinop(i32, BinopOp, String),
    /// `.field op "lit"` — exercises the select_str_* family
    /// (#394 / #396 / #398).
    FieldStrConstBinop(String, BinopOp, String),
    /// `(<binop>) <and|or> (<binop>)` — compound boolean condition.
    /// Used inside `select(...)` to exercise the
    /// `select_compound_*` fast paths.
    CompoundCond(Box<FilterExpr>, AndOr, Box<FilterExpr>),
}

#[derive(Debug, Clone, Copy)]
enum AndOr { And, Or }

impl AndOr {
    fn render(self) -> &'static str { match self { AndOr::And => "and", AndOr::Or => "or" } }
}

#[derive(Debug, Clone, Copy)]
enum BinopOp { Add, Sub, Mul, Div, Mod, Gt, Lt, Ge, Le, Eq, Ne, And, Or }

impl BinopOp {
    fn render(self) -> &'static str {
        match self {
            BinopOp::Add => "+", BinopOp::Sub => "-", BinopOp::Mul => "*",
            BinopOp::Div => "/", BinopOp::Mod => "%",
            BinopOp::Gt => ">", BinopOp::Lt => "<",
            BinopOp::Ge => ">=", BinopOp::Le => "<=",
            BinopOp::Eq => "==", BinopOp::Ne => "!=",
            BinopOp::And => "and", BinopOp::Or => "or",
        }
    }
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
        FilterExpr::FloatLiteral(s) => (*s).to_string(),
        FilterExpr::FieldFieldBinop(f1, op, f2) => format!(".{} {} .{}", f1, op.render(), f2),
        FilterExpr::FieldConstBinop(f, op, n) => format!(".{} {} {}", f, op.render(), n),
        FilterExpr::ConstFieldBinop(n, op, f) => format!("{} {} .{}", n, op.render(), f),
        FilterExpr::FieldStrConstBinop(f, op, s) => {
            format!(".{} {} {}", f, op.render(), serde_json::to_string(s).unwrap())
        }
        FilterExpr::CompoundCond(l, ao, r) => format!("({}) {} ({})", render(l), ao.render(), render(r)),
    }
}

fn binop_strategy() -> impl Strategy<Value = BinopOp> {
    prop_oneof![
        Just(BinopOp::Add), Just(BinopOp::Sub), Just(BinopOp::Mul), Just(BinopOp::Div), Just(BinopOp::Mod),
        Just(BinopOp::Gt), Just(BinopOp::Lt), Just(BinopOp::Ge), Just(BinopOp::Le), Just(BinopOp::Eq), Just(BinopOp::Ne),
        Just(BinopOp::And), Just(BinopOp::Or),
    ]
}

fn ident_strategy() -> impl Strategy<Value = String> {
    prop::sample::select(IDENT_POOL).prop_map(|s| s.to_string())
}

fn str_literal_strategy() -> impl Strategy<Value = String> {
    prop::sample::select(STR_LITERAL_POOL).prop_map(|s| s.to_string())
}

fn cmp_binop_strategy() -> impl Strategy<Value = BinopOp> {
    prop_oneof![
        Just(BinopOp::Gt), Just(BinopOp::Lt), Just(BinopOp::Ge),
        Just(BinopOp::Le), Just(BinopOp::Eq), Just(BinopOp::Ne),
    ]
}

fn andor_strategy() -> impl Strategy<Value = AndOr> {
    prop_oneof![Just(AndOr::And), Just(AndOr::Or)]
}

/// Field-vs-field / field-vs-const binops as a standalone strategy.
/// Reused by `composition_biased_pipe` so the select-side and the
/// post-select-side can both draw from the same pool independently.
fn binop_expr_strategy() -> impl Strategy<Value = FilterExpr> {
    prop_oneof![
        (ident_strategy(), binop_strategy(), ident_strategy())
            .prop_map(|(a, op, b)| FilterExpr::FieldFieldBinop(a, op, b)),
        (ident_strategy(), binop_strategy(), -3i32..=3)
            .prop_map(|(f, op, n)| FilterExpr::FieldConstBinop(f, op, n)),
        (-3i32..=3, binop_strategy(), ident_strategy())
            .prop_map(|(n, op, f)| FilterExpr::ConstFieldBinop(n, op, f)),
        (ident_strategy(), cmp_binop_strategy(), str_literal_strategy())
            .prop_map(|(f, op, s)| FilterExpr::FieldStrConstBinop(f, op, s)),
    ]
}

/// Composition-biased generator (#320). Produces `Pipe(L, R)` where
/// `L` is a `select(<binop>)` shape and `R` is a reader (binop, field,
/// array-construct, object-construct) over the surviving input. This
/// is the cross-product shape that has repeatedly surfaced
/// multi-fast-path bugs when both halves hit independent detectors and
/// the apply-site invariant fails on one side: #358 (Pipe-substitution
/// over `and`/`or`), #366 (`select_cmp_then_value`), #373
/// (`select_cmp_then_computed_remap`), #375 (short-circuit elision).
/// The random recursion in `filter_strategy` already produces these by
/// accident; biasing this branch higher exercises them deterministically.
fn composition_biased_pipe() -> impl Strategy<Value = FilterExpr> {
    // Select gate: plain binop *or* a compound (and/or) of two
    // binops, which is the shape that exercises the
    // `select_compound_*` fast paths.
    let plain_binop = binop_expr_strategy();
    let compound_cond = (binop_expr_strategy(), andor_strategy(), binop_expr_strategy())
        .prop_map(|(l, ao, r)| FilterExpr::CompoundCond(Box::new(l), ao, Box::new(r)));
    let select_inner = prop_oneof![2 => plain_binop, 1 => compound_cond];
    let select_shape = select_inner.prop_map(|inner| FilterExpr::Select(Box::new(inner)));

    let post_select_shape = prop_oneof![
        binop_expr_strategy(),
        ident_strategy().prop_map(FilterExpr::Field),
        prop::collection::vec(ident_strategy().prop_map(FilterExpr::Field), 1..=3)
            .prop_map(FilterExpr::ArrayConstruct),
        prop::collection::vec(
            (ident_strategy(), ident_strategy().prop_map(FilterExpr::Field)),
            1..=3,
        )
            .prop_map(FilterExpr::ObjectConstruct),
    ];

    (select_shape, post_select_shape)
        .prop_map(|(l, r)| FilterExpr::Pipe(Box::new(l), Box::new(r)))
}

fn leaf_strategy() -> impl Strategy<Value = FilterExpr> {
    prop_oneof![
        Just(FilterExpr::Identity),
        ident_strategy().prop_map(FilterExpr::Field),
        (-3i32..=3).prop_map(FilterExpr::Index),
        prop::sample::select(BUILTIN_UNARY).prop_map(FilterExpr::UnaryBuiltin),
        (0u32..5).prop_map(FilterExpr::RangeN),
        (-3i32..=3).prop_map(FilterExpr::IntLiteral),
        prop::sample::select(FLOAT_LITERALS).prop_map(FilterExpr::FloatLiteral),
        prop_oneof![
            (-3i32..=3, prop::option::of(-3i32..=3))
                .prop_map(|(a, b)| FilterExpr::SliceLo(a, b)),
            (prop::option::of(-3i32..=3), -3i32..=3)
                .prop_map(|(a, b)| FilterExpr::SliceHi(a, b)),
        ],
        (ident_strategy(), binop_strategy(), ident_strategy())
            .prop_map(|(a, op, b)| FilterExpr::FieldFieldBinop(a, op, b)),
        (ident_strategy(), binop_strategy(), -3i32..=3)
            .prop_map(|(f, op, n)| FilterExpr::FieldConstBinop(f, op, n)),
        (-3i32..=3, binop_strategy(), ident_strategy())
            .prop_map(|(n, op, f)| FilterExpr::ConstFieldBinop(n, op, f)),
    ]
}

fn filter_strategy() -> impl Strategy<Value = FilterExpr> {
    leaf_strategy().prop_recursive(
        4,   // depth
        32,  // total size budget
        4,   // max items per collection / branches
        |inner| {
            prop_oneof![
                // Composition-biased Pipe (#320) — see
                // `composition_biased_pipe` for the rationale.
                3 => composition_biased_pipe(),
                1 => prop_oneof![
                    prop::collection::vec(inner.clone(), 0..=3).prop_map(FilterExpr::ArrayConstruct),
                    prop::collection::vec(
                        (ident_strategy(), inner.clone()),
                        0..=3,
                    ).prop_map(FilterExpr::ObjectConstruct),
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
                ],
            ]
        },
    )
}

#[derive(Debug, Clone)]
enum JsonShape {
    Null,
    Bool(bool),
    /// Integer literal. Widened from `i32` (#321) so the adversarial
    /// pool below can stress the f64 mantissa boundary (`±2^53`) and
    /// the i32 boundary (`±2^31`). Anything ≤ 2^53 in absolute value
    /// round-trips as an integer through both jq's number printer and
    /// the harness's `serde_json` re-parse; values strictly beyond
    /// that boundary become floats and would trip the float-formatting
    /// asymmetry called out in the module doc, so the adversarial pool
    /// caps at ±2^53.
    IntN(i64),
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

/// Boundary integer pool (#321). Each value round-trips as an
/// integer through jq's printer and `serde_json`. The largest
/// magnitude here is `±2^53`, the f64 mantissa limit; anything
/// beyond that becomes a float in jq's output and trips the
/// float-formatting asymmetry described in the module doc.
const ADVERSARIAL_INTS: &[i64] = &[
    i32::MIN as i64,
    i32::MAX as i64,
    (i32::MAX as i64) + 1,
    -((i32::MAX as i64) + 1),
    1 << 53,
    -(1i64 << 53),
];

/// String pool with shapes that have surfaced bugs historically:
/// empty string, single ASCII, runs that brush against the small
/// SSO threshold, a longer ASCII string that exercises allocation
/// paths on both runtimes, and a multi-byte UTF-8 cross-section
/// (#321 phase 3a) covering 2-byte / 3-byte / 4-byte sequences,
/// RTL text, combining marks, BMP-supplementary emoji, and a BOM
/// in mid-string position. All values are well-formed UTF-8 — lone
/// surrogates and embedded NUL are deliberately excluded because
/// jq-1.8 rejects them at input parse and the harness would surface
/// an error-vs-success mismatch (see "Expected-divergence classes"
/// in the module doc).
const ADVERSARIAL_STRS: &[&str] = &[
    "",
    "0",
    "true",
    "null",
    "          ",
    "abcdefghijklmnopqrstuvwxyz",
    // 3-byte CJK
    "日本語",
    // RTL Hebrew
    "שלום",
    // Combining mark: e + U+0301
    "e\u{0301}",
    // 4-byte BMP-supplementary emoji
    "😀",
    // 4-byte musical symbol
    "𝄞",
    // BOM (U+FEFF) in mid-string — exercises the path where the
    // byte-order mark is *not* at offset 0 and must be preserved.
    "a\u{FEFF}b",
    // Long multi-byte run — stresses allocation / SSO boundaries
    // for codepoint vs byte indexing.
    "日本語日本語日本語日本語日本語日本語日本語日本語",
];

fn json_leaf() -> impl Strategy<Value = JsonShape> {
    // Mix the conservative leaf set with a small adversarial pool
    // (#321). Weighted ~5:1 so the adversarial values are exercised
    // every round without crowding out the normal distribution.
    prop_oneof![
        5 => prop_oneof![
            Just(JsonShape::Null),
            any::<bool>().prop_map(JsonShape::Bool),
            (-5i64..=5).prop_map(JsonShape::IntN),
            prop::sample::select(vec!["", "a", "ab", "0", "hello"])
                .prop_map(|s| JsonShape::Str(s.to_string())),
        ],
        1 => prop_oneof![
            prop::sample::select(ADVERSARIAL_INTS).prop_map(JsonShape::IntN),
            prop::sample::select(ADVERSARIAL_STRS)
                .prop_map(|s| JsonShape::Str(s.to_string())),
        ],
    ]
}

/// Adversarial object-key pool (#321 phase 2). Empty string is the
/// notable shape — `{"": v}` is valid JSON, valid jq input, and
/// touches a different code path on key lookup, key sort, and key
/// serialization than any non-empty key. Mixed with one ident-pool
/// key so duplicate-vs-unique permutations both occur.
const ADVERSARIAL_OBJ_KEYS: &[&str] = &["", "a"];

/// Single-element nested chain `[[[...x]]]` or `{"a":{"a":{...x}}}`,
/// depth `1..=10`. The conservative recursive generator caps at
/// depth 3, so this is the only path that exercises field/index
/// access through depth >3 on shapes the parser still accepts as a
/// single value. Stresses recursion-depth assumptions in the eval
/// and JIT layers without invoking `..`, which is excluded by the
/// module doc.
fn deep_chain_strategy() -> impl Strategy<Value = JsonShape> {
    (1usize..=10, any::<bool>(), json_leaf()).prop_map(|(depth, use_arr, leaf)| {
        let mut v = leaf;
        for _ in 0..depth {
            v = if use_arr {
                JsonShape::Arr(vec![v])
            } else {
                JsonShape::Obj(vec![("a".to_string(), v)])
            };
        }
        v
    })
}

/// Sparse array: length 4–12, all `null` except one position with a
/// leaf value. Real-world telemetry and timeseries pad with nulls,
/// and this is the shape most likely to surface a fast path that
/// short-circuits on the first non-null index instead of iterating
/// the whole array.
fn sparse_array_strategy() -> impl Strategy<Value = JsonShape> {
    (4usize..=12, json_leaf()).prop_flat_map(|(len, leaf)| {
        (Just(len), Just(leaf), 0usize..len).prop_map(|(len, leaf, pos)| {
            let mut items = vec![JsonShape::Null; len];
            items[pos] = leaf;
            JsonShape::Arr(items)
        })
    })
}

/// Object with adversarial keys (including `""`) and 1–3 entries.
/// Duplicate `""` keys are deduped last-wins by jq; the harness
/// already tolerates that for the ident pool (#233 / #325), and the
/// same applies to the empty key.
fn adversarial_obj_strategy() -> impl Strategy<Value = JsonShape> {
    prop::collection::vec(
        (
            prop::sample::select(ADVERSARIAL_OBJ_KEYS).prop_map(|s| s.to_string()),
            json_leaf(),
        ),
        1..=3,
    )
    .prop_map(JsonShape::Obj)
}

fn json_strategy() -> impl Strategy<Value = JsonShape> {
    let recursive = json_leaf().prop_recursive(3, 12, 3, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..=3).prop_map(JsonShape::Arr),
            // Duplicate input keys are deduped last-wins-first-position
            // by both the value-level parse path (#233) and the
            // raw-byte fast paths (#325). Generate freely.
            prop::collection::vec((ident_strategy(), inner.clone()), 0..=3)
                .prop_map(JsonShape::Obj),
        ]
    });

    // Mix the conservative recursive generator with adversarial
    // container shapes (#321 phase 2). Weighted ~5:1 to mirror the
    // adversarial-leaf split — frequent enough that every multi-k run
    // hits each shape, rare enough not to crowd out normal coverage.
    prop_oneof![
        5 => recursive,
        1 => prop_oneof![
            deep_chain_strategy(),
            sparse_array_strategy(),
            adversarial_obj_strategy(),
        ],
    ]
}

#[test]
fn fuzz_restricted_against_jq_1_8() {
    let Some(jq) = require_jq(TEST_LABEL) else { return };
    let jq_jit = jq_jit_path().to_string();

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

        let Some(r_jq) = run_filter(&jq, &filter, &input, timeout) else {
            return Ok(());
        };
        let Some(r_jit) = run_filter(&jq_jit, &filter, &input, timeout) else {
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
        "=== fuzz_restricted (vs {}) ===\n  compared: {}\n  both_errored: {}",
        jq,
        compared.load(std::sync::atomic::Ordering::Relaxed),
        both_error.load(std::sync::atomic::Ordering::Relaxed),
    );

    if let Err(e) = result {
        panic!("fuzz_restricted failed:\n{}", e);
    }
}
