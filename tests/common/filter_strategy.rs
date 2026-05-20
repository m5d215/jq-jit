//! Shared filter / JSON proptest strategies (#688).
//!
//! Single source of truth for the `FilterExpr` AST, the `JsonShape` input
//! AST, their printers, and the base proptest combinators that
//! `tests/fuzz_restricted.rs`, `tests/metamorphic.rs`, and `#686`'s
//! per-axis fuzz harnesses all build on.
//!
//! ## What lives here vs. in each harness
//!
//! * **Here**: the AST union (every variant any harness uses), the
//!   printers, the conservative single-valued leaf strategies, and the
//!   conservative recursive base. These are the parts that every
//!   harness needs to agree on so a shrunk failure in one harness can be
//!   pasted into another's regression-test format without rewriting.
//! * **In each harness**: the weights and the "exotic" leaves. The
//!   composition-biased shapes, boundary-stressing builtins, float
//!   literals, adversarial JSON pools, and multi-valued-island AST
//!   wrappers stay with their callers because they encode that harness's
//!   particular bug-hunt thesis.
//!
//! ## Extension points
//!
//! Harnesses extend the base by:
//!
//! 1. Adding their own constants (`fuzz_restricted::FLOAT_LITERALS`,
//!    `fuzz_restricted::ADVERSARIAL_INTS`, …) that draw from the AST
//!    variants defined here. The variants themselves are already in
//!    [`FilterExpr`] / [`JsonShape`]; harnesses don't add new variants,
//!    only new *weights* and *constants*.
//! 2. Wrapping [`conservative_leaf_strategy`] / [`base_filter_strategy`]
//!    with a `prop_oneof![n => ..., m => ...]` that biases distinct
//!    shapes. The metamorphic harness instead restricts to a
//!    single-valued subset and wraps it in a `MultiFilter` AST in its
//!    own file.
//! 3. Wrapping [`base_json_strategy`] with their own adversarial
//!    leaf / container mix when the conservative shape isn't enough.
//!
//! When a new fast-path detector lands and needs proptest coverage,
//! prefer adding constants to the harness that already targets that
//! axis. Add new variants to [`FilterExpr`] / [`JsonShape`] *here* only
//! when a shape genuinely cannot be expressed via composition of
//! existing variants — every added variant has to be implemented in
//! `render` and considered by every harness.
//!
//! ## Naming
//!
//! Variant names follow `fuzz_restricted.rs`'s pre-extraction
//! convention (the richer original) so the diff stays minimal there.
//! Where `metamorphic.rs` used different names (`FieldCmpField` vs
//! `FieldFieldBinop`, `IntLit` vs `IntLiteral`, `Json::Int(i32)` vs
//! `JsonShape::IntN(i64)`), the harness was rebuilt to use the shared
//! names.

#![allow(dead_code)]

use proptest::prelude::*;

// =====================================================================
// Identifier / operator / literal pools
// =====================================================================

/// Field-name pool. Five identifiers (`a,b,c,x,y`) — small enough that
/// matches between filter-side and input-side keys are common, large
/// enough to exercise multi-key shapes. Both filter `.f` and input
/// object keys draw from this pool, by design.
pub const IDENT_POOL: &[&str] = &["a", "b", "c", "x", "y"];

/// String-literal pool for `select(.f == "lit")` and similar shapes.
/// Overlaps with the JSON leaf string pool so matches occur often
/// enough for the `select_str_*` fast paths to actually fire.
pub const STR_LITERAL_POOL: &[&str] = &["", "a", "ab", "0", "hello"];

/// Single-valued unary builtins that are safe to mix freely in any
/// harness. Each yields exactly one value per input (or errors),
/// emits canonical JSON (no `1e10` / `+5` quirks), and round-trips
/// cleanly through `serde_json` re-parse across 100k+ proptest
/// cases. Harnesses with a more aggressive thesis layer their own
/// builtin pools (boundary-stressing, non-finite-producing) on top.
pub const SAFE_BUILTINS: &[&str] = &[
    "length",
    "keys",
    "keys_unsorted",
    "values",
    "type",
    "tostring",
    "to_entries",
    "reverse",
    "sort",
    "not",
];

// =====================================================================
// Filter AST + printer
// =====================================================================

#[derive(Debug, Clone, Copy)]
pub enum BinopOp { Add, Sub, Mul, Div, Mod, Gt, Lt, Ge, Le, Eq, Ne, And, Or }

impl BinopOp {
    pub fn render(self) -> &'static str {
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

#[derive(Debug, Clone, Copy)]
pub enum AndOr { And, Or }

impl AndOr {
    pub fn render(self) -> &'static str {
        match self { AndOr::And => "and", AndOr::Or => "or" }
    }
}

#[derive(Debug, Clone)]
pub enum FilterExpr {
    Identity,
    Field(String),
    Index(i32),
    /// Half-open slice. The both-bounds-absent form `.[:]` is a parse
    /// error in both jq and jq-jit (#438), so the lo bound is always
    /// present in the `SliceLo` variant.
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
    /// Float literal as a static spelling, drawn from a harness-owned
    /// pool (fuzz_restricted uses `FLOAT_LITERALS` for f64-boundary
    /// stress). Stored as `&'static str` rather than `f64` so the
    /// rendered filter matches the source spelling exactly — relevant
    /// for overflow forms (`1e500`) where `f64::to_string` would emit
    /// `inf` instead.
    FloatLiteral(&'static str),
    /// `.f1 op .f2` — exercises the FieldCmpField / FieldOpField fast
    /// paths leaf shapes don't otherwise reach (#347).
    FieldFieldBinop(String, BinopOp, String),
    /// `.field op N` and `N op .field` — exercises FieldCmpConst /
    /// FieldOpConst / ConstOpField shapes.
    FieldConstBinop(String, BinopOp, i32),
    ConstFieldBinop(i32, BinopOp, String),
    /// `.field op "lit"` — exercises the `select_str_*` family
    /// (#394 / #396 / #398).
    FieldStrConstBinop(String, BinopOp, String),
    /// `(<binop>) <and|or> (<binop>)` — compound boolean condition,
    /// used inside `select(...)` to exercise `select_compound_*`.
    CompoundCond(Box<FilterExpr>, AndOr, Box<FilterExpr>),
}

pub fn render(expr: &FilterExpr) -> String {
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
        FilterExpr::CompoundCond(l, ao, r) => {
            format!("({}) {} ({})", render(l), ao.render(), render(r))
        }
    }
}

// =====================================================================
// Base strategy combinators
// =====================================================================

pub fn ident_strategy() -> impl Strategy<Value = String> {
    prop::sample::select(IDENT_POOL).prop_map(|s| s.to_string())
}

pub fn str_literal_strategy() -> impl Strategy<Value = String> {
    prop::sample::select(STR_LITERAL_POOL).prop_map(|s| s.to_string())
}

pub fn safe_builtin_strategy() -> impl Strategy<Value = &'static str> {
    prop::sample::select(SAFE_BUILTINS)
}

pub fn binop_strategy() -> impl Strategy<Value = BinopOp> {
    prop_oneof![
        Just(BinopOp::Add), Just(BinopOp::Sub), Just(BinopOp::Mul),
        Just(BinopOp::Div), Just(BinopOp::Mod),
        Just(BinopOp::Gt), Just(BinopOp::Lt),
        Just(BinopOp::Ge), Just(BinopOp::Le),
        Just(BinopOp::Eq), Just(BinopOp::Ne),
        Just(BinopOp::And), Just(BinopOp::Or),
    ]
}

pub fn cmp_binop_strategy() -> impl Strategy<Value = BinopOp> {
    prop_oneof![
        Just(BinopOp::Gt), Just(BinopOp::Lt),
        Just(BinopOp::Ge), Just(BinopOp::Le),
        Just(BinopOp::Eq), Just(BinopOp::Ne),
    ]
}

pub fn andor_strategy() -> impl Strategy<Value = AndOr> {
    prop_oneof![Just(AndOr::And), Just(AndOr::Or)]
}

/// Conservative single-valued leaf. Every variant returned by this
/// strategy yields exactly one value per input (or errors). Suitable
/// for harnesses that need single-valued generators (metamorphic
/// equivalences guarded on `is_single_valued_expr`, future
/// `fuzz_axis_*.rs` harnesses).
///
/// Pool: `Identity`, `Field`, `Index`, `IntLiteral`,
/// `UnaryBuiltin(SAFE_BUILTINS)`, `FieldFieldBinop(cmp)`,
/// `FieldConstBinop(cmp)`. Harnesses widen by `prop_oneof!`-ing in
/// extra leaves rather than redefining the function.
pub fn conservative_leaf_strategy() -> impl Strategy<Value = FilterExpr> {
    prop_oneof![
        Just(FilterExpr::Identity),
        ident_strategy().prop_map(FilterExpr::Field),
        (-2i32..=2).prop_map(FilterExpr::Index),
        (-3i32..=3).prop_map(FilterExpr::IntLiteral),
        safe_builtin_strategy().prop_map(FilterExpr::UnaryBuiltin),
        (ident_strategy(), cmp_binop_strategy(), ident_strategy())
            .prop_map(|(a, op, b)| FilterExpr::FieldFieldBinop(a, op, b)),
        (ident_strategy(), cmp_binop_strategy(), -3i32..=3)
            .prop_map(|(f, op, n)| FilterExpr::FieldConstBinop(f, op, n)),
    ]
}

/// Single-valued-safe recursive base. Wraps `leaf` (assumed
/// single-valued) with the composition shapes that *preserve* the
/// single-valued property: Pipe, ArrayConstruct, ObjectConstruct,
/// Map, If. Multi-valued constructors (`Comma`, `Limit`, generator
/// builtins, `.[]`, `range`) are deliberately excluded — harnesses
/// that want them either add them in their own recursion
/// (fuzz_restricted) or wrap the output in a multi-valued AST
/// (metamorphic's `MultiFilter::Single`).
///
/// `branches` controls the max element count per array / object
/// construct AND the recursion branch factor (passed verbatim to
/// `prop_recursive`). `depth` and `size` are the standard
/// `prop_recursive` parameters.
pub fn base_filter_strategy(
    leaf: impl Strategy<Value = FilterExpr> + 'static,
    depth: u32,
    size: u32,
    branches: u32,
) -> impl Strategy<Value = FilterExpr> {
    leaf.prop_recursive(depth, size, branches, move |inner| {
        let max_items = branches as usize;
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..=max_items)
                .prop_map(FilterExpr::ArrayConstruct),
            prop::collection::vec(
                (ident_strategy(), inner.clone()),
                0..=max_items,
            ).prop_map(FilterExpr::ObjectConstruct),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| FilterExpr::Pipe(Box::new(a), Box::new(b))),
            inner.clone().prop_map(|f| FilterExpr::Map(Box::new(f))),
            (inner.clone(), inner.clone(), inner.clone()).prop_map(|(a, b, c)| {
                FilterExpr::If(Box::new(a), Box::new(b), Box::new(c))
            }),
        ]
    })
}

// =====================================================================
// JSON shape AST + printer + base strategies
// =====================================================================

#[derive(Debug, Clone)]
pub enum JsonShape {
    Null,
    Bool(bool),
    /// Integer literal. Widened to `i64` so harnesses can include
    /// adversarial boundary pools (`±2^53`, `±2^31`) without redefining
    /// the AST. Conservative leaves stay inside `i32` range, which the
    /// printer handles identically.
    IntN(i64),
    Str(String),
    Arr(Vec<JsonShape>),
    Obj(Vec<(String, JsonShape)>),
}

pub fn render_json(v: &JsonShape) -> String {
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
                .map(|(k, v)| {
                    format!("{}:{}", serde_json::to_string(k).unwrap(), render_json(v))
                })
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

/// Conservative JSON leaf. Null, bool, small integer, short string.
/// Identical to both pre-extraction harnesses' leaf distribution.
pub fn conservative_json_leaf() -> impl Strategy<Value = JsonShape> {
    prop_oneof![
        Just(JsonShape::Null),
        any::<bool>().prop_map(JsonShape::Bool),
        (-3i64..=3).prop_map(JsonShape::IntN),
        prop::sample::select(vec!["", "a", "ab", "0", "hello"])
            .prop_map(|s| JsonShape::Str(s.to_string())),
    ]
}

/// Conservative recursive JSON. Wraps `leaf` with array / object
/// constructors using `IDENT_POOL` keys. `branches` plays the same
/// role as in [`base_filter_strategy`].
pub fn base_json_strategy(
    leaf: impl Strategy<Value = JsonShape> + 'static,
    depth: u32,
    size: u32,
    branches: u32,
) -> impl Strategy<Value = JsonShape> {
    leaf.prop_recursive(depth, size, branches, move |inner| {
        let max_items = branches as usize;
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..=max_items)
                .prop_map(JsonShape::Arr),
            // Duplicate input keys are deduped last-wins-first-position
            // by both the value-level parse path (#233) and the
            // raw-byte fast paths (#325). Generate freely.
            prop::collection::vec((ident_strategy(), inner.clone()), 0..=max_items)
                .prop_map(JsonShape::Obj),
        ]
    })
}
