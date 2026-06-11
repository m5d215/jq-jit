//! Expression simplification: the `simplify_expr` peephole-rewrite passes
//! and the predicates that guard them (`is_single_valued_expr`, the
//! `Input`-usage walkers, variable-reference checks). Extracted from
//! `src/interpreter.rs` (#1029); pure code motion, no behavior change.
//!
//! The rewrites run once at `Filter` construction (see
//! [`crate::interpreter::Filter`]) and feed both the fast-path classifiers
//! in [`crate::classify`] and the generic eval / JIT paths. Invariants the
//! guards protect are catalogued in docs/maintenance.md (§3 generator
//! folding, § simplify layer).

use crate::interpreter::disable_simplify;
use crate::classify::normalize_object_pairs;
use crate::ir::BuiltinOp;

/// Conservatively decide whether `e` is guaranteed to yield exactly one value
/// per evaluation. Used by `[gen0, gen1, …] | F` fold rewrites
/// (`add`/`min`/`max`/`sort`/`reverse`/`any`/`all`) to bail when a non-single-valued
/// branch would otherwise be promoted into the surrounding fold and stream
/// every element instead of being collected first (issue #152). Safe default
/// when uncertain: false.
/// Conservative check that an expression cannot raise a runtime error. Used
/// to decide when a LetBinding can be eliminated even if its body doesn't
/// reference the bound variable — `(E as $x | F)` may not drop E unless E is
/// guaranteed not to error (#521).
fn expr_is_pure_scalar(e: &crate::ir::Expr) -> bool {
    use crate::ir::Expr;
    match e {
        Expr::Literal(_) | Expr::Input | Expr::LoadVar { .. }
        | Expr::Empty | Expr::Env | Expr::Builtins
        | Expr::Loc { .. } => true,
        // `not` and `type` never error on any value.
        Expr::Not => true,
        Expr::UnaryOp { op, operand } => {
            matches!(op, crate::ir::UnaryOp::Type) && expr_is_pure_scalar(operand)
        }
        _ => false,
    }
}

/// True if `expr` contains a `LoadVar` reference to `var_index`. Used by the
/// LetBinding inliner to decide whether substitution would actually use the
/// bound value or only need to keep it for its side effect.
fn expr_references_var(expr: &crate::ir::Expr, var_index: crate::ir::VarIdx) -> bool {
    use crate::ir::Expr;
    match expr {
        Expr::LoadVar { var_index: idx } => *idx == var_index,
        Expr::Input | Expr::Empty | Expr::Not | Expr::Env | Expr::Builtins
        | Expr::ReadInput | Expr::ReadInputs | Expr::ModuleMeta | Expr::GenLabel
        | Expr::Literal(_) | Expr::Loc { .. } => false,
        Expr::Pipe { left, right } | Expr::Comma { left, right }
        | Expr::BinOp { lhs: left, rhs: right, .. }
        | Expr::Alternative { primary: left, fallback: right }
        | Expr::While { cond: left, update: right }
        | Expr::Until { cond: left, update: right }
        | Expr::Limit { count: left, generator: right }
        | Expr::Index { expr: left, key: right }
        | Expr::IndexOpt { expr: left, key: right }
        | Expr::Update { path_expr: left, update_expr: right }
        | Expr::Assign { path_expr: left, value_expr: right }
        | Expr::SetPath { path: left, value: right }
        | Expr::TryCatch { try_expr: left, catch_expr: right, .. } => {
            expr_references_var(left, var_index) || expr_references_var(right, var_index)
        }
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            expr_references_var(cond, var_index)
                || expr_references_var(then_branch, var_index)
                || expr_references_var(else_branch, var_index)
        }
        Expr::LetBinding { value, body, .. } => {
            expr_references_var(value, var_index) || expr_references_var(body, var_index)
        }
        Expr::Each { input_expr } | Expr::EachOpt { input_expr }
        | Expr::Recurse { input_expr } | Expr::Repeat { update: input_expr }
        | Expr::Negate { operand: input_expr } | Expr::UnaryOp { operand: input_expr, .. }
        | Expr::Collect { generator: input_expr }
        | Expr::PathExpr { expr: input_expr } | Expr::GetPath { path: input_expr }
        | Expr::DelPaths { paths: input_expr } | Expr::Debug { expr: input_expr }
        | Expr::Stderr { expr: input_expr } | Expr::Format { expr: input_expr, .. } => {
            expr_references_var(input_expr, var_index)
        }
        Expr::Reduce { source, init, update, .. }
        | Expr::Foreach { source, init, update, .. } => {
            expr_references_var(source, var_index)
                || expr_references_var(init, var_index)
                || expr_references_var(update, var_index)
        }
        Expr::Range { from, to, step } => {
            expr_references_var(from, var_index)
                || expr_references_var(to, var_index)
                || step.as_ref().is_some_and(|s| expr_references_var(s, var_index))
        }
        Expr::Slice { expr, .. } => expr_references_var(expr, var_index),
        Expr::ObjectConstruct { pairs } => {
            pairs.iter().any(|(k, v)| expr_references_var(k, var_index) || expr_references_var(v, var_index))
        }
        Expr::AllShort { generator, predicate } | Expr::AnyShort { generator, predicate } => {
            expr_references_var(generator, var_index) || expr_references_var(predicate, var_index)
        }
        Expr::AlternativeDestructure { alternatives } => {
            alternatives.iter().any(|a| expr_references_var(a, var_index))
        }
        Expr::StringInterpolation { parts } => parts.iter().any(|p| match p {
            crate::ir::StringPart::Expr(e) => expr_references_var(e, var_index),
            crate::ir::StringPart::Literal(_) => false,
        }),
        // Conservative for variants we don't enumerate: assume they may use the var.
        _ => true,
    }
}

/// Returns true if `var_index` is referenced anywhere inside `expr` in a
/// position where `.` has been rebound away from the value it had at `expr`'s
/// entry. The LetBinding inliner uses this to refuse substituting a
/// replacement that reads `.` (e.g. `.foo`, or `.` itself) into such a
/// position — `. as $v | map($v)` must keep `$v` bound to the outer `.`, not
/// the per-element `.` that `map`/`.[] |` rebinds (#818).
///
/// Sound by construction: only clearly dot-preserving constructs propagate
/// `dot_same`; every other (or unenumerated) construct is treated as rebinding
/// `.`, so a reference found there is reported (the inliner just declines the
/// optimization and keeps the binding).
fn var_in_rebound_dot_scope(expr: &crate::ir::Expr, var_index: crate::ir::VarIdx) -> bool {
    use crate::ir::{Expr, StringPart};
    fn walk(e: &Expr, var: crate::ir::VarIdx, dot_same: bool) -> bool {
        match e {
            Expr::LoadVar { var_index: v } => *v == var && !dot_same,
            // Dot-preserving: every sub-expression sees the same `.`.
            Expr::BinOp { lhs, rhs, .. } => walk(lhs, var, dot_same) || walk(rhs, var, dot_same),
            Expr::UnaryOp { operand, .. }
            | Expr::Negate { operand } => walk(operand, var, dot_same),
            Expr::Format { expr, .. } => walk(expr, var, dot_same),
            Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => {
                walk(expr, var, dot_same) || walk(key, var, dot_same)
            }
            Expr::Comma { left, right } => walk(left, var, dot_same) || walk(right, var, dot_same),
            Expr::Collect { generator } => walk(generator, var, dot_same),
            Expr::Each { input_expr } | Expr::EachOpt { input_expr } => walk(input_expr, var, dot_same),
            Expr::IfThenElse { cond, then_branch, else_branch } => {
                walk(cond, var, dot_same)
                    || walk(then_branch, var, dot_same)
                    || walk(else_branch, var, dot_same)
            }
            Expr::Alternative { primary, fallback } => {
                walk(primary, var, dot_same) || walk(fallback, var, dot_same)
            }
            Expr::ObjectConstruct { pairs } => {
                pairs.iter().any(|(k, v)| walk(k, var, dot_same) || walk(v, var, dot_same))
            }
            Expr::StringInterpolation { parts } => parts.iter().any(|p| {
                matches!(p, StringPart::Expr(x) if walk(x, var, dot_same))
            }),
            Expr::Slice { expr, from, to } => {
                walk(expr, var, dot_same)
                    || from.as_ref().is_some_and(|x| walk(x, var, dot_same))
                    || to.as_ref().is_some_and(|x| walk(x, var, dot_same))
            }
            Expr::Range { from, to, step } => {
                walk(from, var, dot_same)
                    || walk(to, var, dot_same)
                    || step.as_ref().is_some_and(|x| walk(x, var, dot_same))
            }
            // `limit(n; g)` runs both n and g against the entry `.`.
            Expr::Limit { count, generator } => {
                walk(count, var, dot_same) || walk(generator, var, dot_same)
            }
            // `a | b`: `a` sees the entry `.`; `b` sees it only when `a` is the
            // identity `.` (otherwise `b`'s `.` is whatever `a` produced).
            Expr::Pipe { left, right } => {
                walk(left, var, dot_same)
                    || walk(right, var, dot_same && matches!(left.as_ref(), Expr::Input))
            }
            // A nested binding keeps `.`; the body is skipped if it shadows our var.
            Expr::LetBinding { var_index: vi, value, body } => {
                walk(value, var, dot_same) || (*vi != var && walk(body, var, dot_same))
            }
            // Everything else rebinds `.` (reduce/foreach update, while/until,
            // try/catch handler, path updates, …) or is not clearly
            // dot-preserving: any reference inside is treated as rebound.
            other => expr_references_var(other, var),
        }
    }
    walk(expr, var_index, true)
}

pub(crate) fn is_single_valued_expr(e: &crate::ir::Expr) -> bool {
    use crate::ir::Expr;
    match e {
        Expr::Empty => false,
        Expr::Each { .. } | Expr::EachOpt { .. }
        | Expr::Comma { .. } | Expr::Recurse { .. }
        | Expr::Range { .. } | Expr::Limit { .. }
        | Expr::RegexMatch { .. } | Expr::RegexScan { .. }
        | Expr::RegexCapture { .. } => false,
        // `.x?` / `.[]?` swallow type errors and yield empty for
        // mismatched inputs — that's a 0-or-1 value count, not exactly
        // one. Treat them as multi-valued so the all/any short-circuit
        // rewrite (and any other optimisation that distributes pipe over
        // a comma list) doesn't drop the empty branches and skew the
        // result (#519).
        Expr::IndexOpt { .. } => false,
        // `try/catch` can swallow errors into the catch branch; if the
        // catch is `empty` the whole thing yields nothing. Conservatively
        // treat it as not exactly-one.
        Expr::TryCatch { .. } => false,
        Expr::Pipe { left, right } => is_single_valued_expr(left) && is_single_valued_expr(right),
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            is_single_valued_expr(cond)
                && is_single_valued_expr(then_branch)
                && is_single_valued_expr(else_branch)
        }
        Expr::Alternative { primary, fallback } => {
            is_single_valued_expr(primary) && is_single_valued_expr(fallback)
        }
        Expr::LetBinding { value, body, .. } => {
            is_single_valued_expr(value) && is_single_valued_expr(body)
        }
        Expr::Collect { .. } => true,
        Expr::BinOp { lhs, rhs, .. } => is_single_valued_expr(lhs) && is_single_valued_expr(rhs),
        Expr::UnaryOp { operand, .. } => is_single_valued_expr(operand),
        Expr::Negate { operand } => is_single_valued_expr(operand),
        Expr::Index { expr, key } => {
            is_single_valued_expr(expr) && is_single_valued_expr(key)
        }
        Expr::Input | Expr::Literal(_) | Expr::LoadVar { .. }
        | Expr::Not
        | Expr::Slice { .. }
        | Expr::StringInterpolation { .. } | Expr::ObjectConstruct { .. }
        | Expr::RegexTest { .. } | Expr::RegexSub { .. } | Expr::RegexGsub { .. } => true,
        _ => false,
    }
}

/// Recursively strip identity pipes and beta-reduce for fast path detection.
/// Pipe(Input, X) → X, Pipe(X, Input) → X.
/// Pipe(E, F) → F[E/.] when E is scalar and F has free Input.
/// Also applies semantic rewrites (to_entries|from_entries → identity).
///
/// Short-circuits to identity when [`disable_simplify`] is on (issue #685
/// layer-pinning knob `JQJIT_DISABLE_SIMPLIFY`). All recursive calls also
/// route through this guard, so a single top-level guard suffices.
pub(crate) fn simplify_expr(expr: &crate::ir::Expr) -> crate::ir::Expr {
    if disable_simplify() {
        return expr.clone();
    }
    use crate::ir::{Expr, Literal, UnaryOp};
    match expr {
        Expr::Pipe { left, right } => {
            let sl = simplify_expr(left);
            let sr = simplify_expr(right);
            if matches!(&sl, Expr::Input) { return sr; }
            if matches!(&sr, Expr::Input) { return sl; }
            // Beta-reduce: X | UnaryOp(op, .) → UnaryOp(op, X)
            // Only for numeric unary ops that don't have specialized detectors
            // in Pipe(.field, UnaryOp) form
            // NOTE: Disabled — too aggressive, breaks other detectors that match
            // Pipe(.field, UnaryOp(op, Input)). Instead, extend specific detectors.
            // Beta-reduce: X | (. binop N) → (X binop N) when N is input-free
            // This flattens pipes like `.x | floor | . > N` into `floor(.x) > N`
            // enabling existing arith_chain_cmp and field_cmp detectors
            if let Expr::BinOp { op, lhs, rhs } = &sr {
                if matches!(lhs.as_ref(), Expr::Input) && rhs.is_input_free() {
                    return simplify_expr(&Expr::BinOp {
                        op: *op,
                        lhs: Box::new(sl),
                        rhs: rhs.clone(),
                    });
                }
            }
            // Beta-reduce: X | (N binop .) → (N binop X) when N is input-free
            if let Expr::BinOp { op, lhs, rhs } = &sr {
                if matches!(rhs.as_ref(), Expr::Input) && lhs.is_input_free() {
                    return simplify_expr(&Expr::BinOp {
                        op: *op,
                        lhs: lhs.clone(),
                        rhs: Box::new(sl),
                    });
                }
            }
            // Beta-reduce: X | (. binop .) → (X binop X) when both sides are input
            if let Expr::BinOp { op, lhs, rhs } = &sr {
                if matches!(lhs.as_ref(), Expr::Input) && matches!(rhs.as_ref(), Expr::Input) {
                    return simplify_expr(&Expr::BinOp {
                        op: *op,
                        lhs: Box::new(sl.clone()),
                        rhs: Box::new(sl),
                    });
                }
            }
            // Beta-reduce: X | if (cond_with_.) then A else B end → if (cond_with_X) then A else B end
            // when A and B are constants (no Input refs), cond is substitutable,
            // and X is a single-output expression (not a generator like range/each/comma)
            if let Expr::IfThenElse { cond, then_branch, else_branch } = &sr {
                let branch_no_input = |b: &Expr| matches!(b, Expr::Literal(_) | Expr::Empty);
                if branch_no_input(then_branch) && branch_no_input(else_branch) && cond.is_input_free() && sl.is_single_output() {
                    return simplify_expr(&Expr::IfThenElse {
                        cond: Box::new(cond.substitute_input(&sl)),
                        then_branch: then_branch.clone(),
                        else_branch: else_branch.clone(),
                    });
                }
            }
            // NOTE: `to_entries | from_entries` is NOT identity. It's identity only
            // for string-keyed objects; arrays and other inputs must flow through
            // `from_entries`'s type check (issue #73). Do not rewrite.
            // NOTE: tojson | fromjson is NOT identity — tojson normalizes nan/inf to null.
            // E.g., {a:nan} | tojson | fromjson → {a:null}. Do not simplify.
            // Semantic: to_entries | map(.key) → keys_unsorted
            // to_entries | map(.value) → [.[]]
            // Semantic: to_entries | map(.key) → keys_unsorted, to_entries | map(.value) → [.[]]
            // Also handles: to_entries | map(.key) | sort → keys (composed with trailing pipe)
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::ToEntries, operand } if matches!(operand.as_ref(), Expr::Input)) {
                // Helper: check if expr is map(.key) or map(.value) — returns Some("key") or Some("value")
                fn is_map_entry_field(e: &Expr) -> Option<&str> {
                    if let Expr::Collect { generator } = e {
                        if let Expr::Pipe { left: gl, right: gr } = generator.as_ref() {
                            if matches!(gl.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                                if let Expr::Index { expr: base, key } = gr.as_ref() {
                                    if matches!(base.as_ref(), Expr::Input) {
                                        if let Expr::Literal(crate::ir::Literal::Str(field)) = key.as_ref() {
                                            if field == "key" || field == "value" {
                                                return Some(field.as_str());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    None
                }
                // Direct: to_entries | map(.key) or to_entries | map(.value)
                if let Some(field) = is_map_entry_field(&sr) {
                    if field == "key" {
                        return Expr::UnaryOp { op: UnaryOp::KeysUnsorted, operand: Box::new(Expr::Input) };
                    } else {
                        return Expr::Collect { generator: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }) };
                    }
                }
                // Composed: to_entries | Pipe(map(.key/.value), tail) → rewrite left, keep tail
                if let Expr::Pipe { left: pl, right: pr } = &sr {
                    if let Some(field) = is_map_entry_field(pl) {
                        let rewritten = if field == "key" {
                            Expr::UnaryOp { op: UnaryOp::KeysUnsorted, operand: Box::new(Expr::Input) }
                        } else {
                            Expr::Collect { generator: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }) }
                        };
                        // Recursively simplify the new pipe
                        return simplify_expr(&Expr::Pipe {
                            left: Box::new(rewritten),
                            right: pr.clone(),
                        });
                    }
                }
            }
            // Semantic: {pairs} | length → N (number of keys).
            // Safe only when every value is *single-output* and
            // *input-free*. A multi-output value (e.g. `range(2)`)
            // would produce multiple objects, each with its own length;
            // folding to a bare integer eats both the multiplicity and
            // any value-time error (#220 / #324 / #333). Same-key
            // duplicates still collapse via `normalize_object_pairs`,
            // which is correct *because* every value is single-output —
            // an earlier pair's runtime error is gone for both
            // implementations once dedup applies.
            if let Expr::ObjectConstruct { pairs } = &sl {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    let mut extracted: Vec<(&str, ())> = Vec::with_capacity(pairs.len());
                    let mut all_static = true;
                    for (k, v) in pairs {
                        if let Expr::Literal(Literal::Str(s)) = k {
                            if contains_input(v) || !expr_is_single_output(v) {
                                all_static = false;
                                break;
                            }
                            extracted.push((s.as_str(), ()));
                        } else {
                            all_static = false;
                            break;
                        }
                    }
                    if all_static {
                        let n = normalize_object_pairs(extracted).len();
                        return Expr::Literal(Literal::Num(n as f64, None));
                    }
                }
            }
            // Semantic: [elements] | length → N (number of elements, if known at compile time)
            // Only safe when no element references Input — otherwise `.a` etc. may
            // raise a type error against the actual input that the rewrite would skip
            // (issue #220). `is_input_free` here is too lax (it considers `Expr::Input`
            // itself free for substitution), so we walk the AST for any Input mention.
            if let Expr::Collect { generator } = &sl {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    fn mentions_input(e: &Expr) -> bool {
                        match e {
                            Expr::Input => true,
                            Expr::Literal(_) | Expr::Empty | Expr::Env | Expr::Builtins
                            | Expr::LoadVar { .. } | Expr::ReadInput | Expr::ReadInputs
                            | Expr::ModuleMeta | Expr::GenLabel | Expr::Loc { .. } => false,
                            Expr::BinOp { lhs, rhs, .. } => mentions_input(lhs) || mentions_input(rhs),
                            Expr::UnaryOp { operand, .. } => mentions_input(operand),
                            Expr::Index { expr, key } => mentions_input(expr) || mentions_input(key),
                            Expr::IndexOpt { expr, key } => mentions_input(expr) || mentions_input(key),
                            Expr::Negate { operand } => mentions_input(operand),
                            Expr::Comma { left, right } => mentions_input(left) || mentions_input(right),
                            _ => true, // conservative: any unknown shape may use input
                        }
                    }
                    fn count_comma_elements_no_input(e: &Expr) -> Option<usize> {
                        match e {
                            // `[]` and `[empty]` both lower to
                            // `Collect { generator: Empty }`. The
                            // catch-all branch below would rewrite them
                            // to `1`; Empty produces zero outputs.
                            Expr::Empty => Some(0),
                            Expr::Comma { left, right } => {
                                Some(count_comma_elements_no_input(left)? + count_comma_elements_no_input(right)?)
                            }
                            _ if !mentions_input(e) => Some(1),
                            _ => None,
                        }
                    }
                    if let Some(n) = count_comma_elements_no_input(generator) {
                        return Expr::Literal(Literal::Num(n as f64, None));
                    }
                }
            }
            // NOTE: `OP | length → length` rewrites for `to_entries`, `keys`,
            // `keys_unsorted`, `values`, `reverse`, `sort` were removed (#220).
            // Each prefix op's type-error contract differs from `length`'s, so the
            // rewrite would silently turn jq errors (e.g. `null | keys`,
            // `"x" | reverse`, `1 | sort`) into 0/1/N. The runtime cost saving was
            // marginal compared to the correctness violation.
            // Semantic: unique | length → unique | length (can't simplify, unique changes length)
            // Semantic: flatten | length — can't simplify, changes length
            // Semantic: keys_unsorted | sort → keys
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::KeysUnsorted, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Sort, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    return Expr::UnaryOp { op: UnaryOp::Keys, operand: Box::new(Expr::Input) };
                }
            }
            // Semantic: sort | reverse → sort | reverse (fused at runtime via SortReverse)
            // sort | reverse | .[0] → max, sort | reverse | .[-1] → min
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::Sort, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    // Keep as sort|reverse — JIT/eval can fuse sort+reverse into sort_unstable_by(|a,b| b.cmp(a))
                    // But optimize sort|reverse|.[0] → max and sort|reverse|.[-1] → min
                }
                // sort | reverse | .[0] → max (largest element)
                if let Expr::Pipe { left: ref pl, right: ref pr } = sr {
                    if matches!(pl.as_ref(), Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                        if let Expr::Index { expr: base, key } = pr.as_ref() {
                            if matches!(base.as_ref(), Expr::Input) {
                                if let Expr::Literal(Literal::Num(n, _)) = key.as_ref() {
                                    if *n == 0.0 {
                                        return Expr::UnaryOp { op: UnaryOp::Max, operand: Box::new(Expr::Input) };
                                    } else if *n == -1.0 {
                                        return Expr::UnaryOp { op: UnaryOp::Min, operand: Box::new(Expr::Input) };
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Semantic: sort | .[0] → min, sort | .[-1] → max
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::Sort, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if let Expr::Index { expr: base, key } = &sr {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Num(n, _)) = key.as_ref() {
                            if *n == 0.0 {
                                return Expr::UnaryOp { op: UnaryOp::Min, operand: Box::new(Expr::Input) };
                            } else if *n == -1.0 {
                                return Expr::UnaryOp { op: UnaryOp::Max, operand: Box::new(Expr::Input) };
                            }
                        }
                    }
                }
            }
            // Semantic: reverse | .[0] → .[-1], reverse | .[-1] → .[0]
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if let Expr::Index { expr: base, key } = &sr {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Num(n, _)) = key.as_ref() {
                            if *n == 0.0 {
                                return Expr::Index {
                                    expr: Box::new(Expr::Input),
                                    key: Box::new(Expr::Literal(Literal::Num(-1.0, None))),
                                };
                            } else if *n == -1.0 {
                                return Expr::Index {
                                    expr: Box::new(Expr::Input),
                                    key: Box::new(Expr::Literal(Literal::Num(0.0, None))),
                                };
                            }
                        }
                    }
                }
            }
            // NOTE: explode | implode is NOT identity — explode errors on non-strings
            // Semantic: explode | map(. + N) | implode → __shift_codepoints__(N)
            // Also: explode | map(. - N) | implode
            // Helper: check if expr is map(. + N) pattern, return shift amount
            fn is_map_shift(expr: &Expr) -> Option<f64> {
                if let Expr::Collect { generator } = expr {
                    if let Expr::Pipe { left: gl, right: gr } = generator.as_ref() {
                        if matches!(gl.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            if let Expr::BinOp { op, lhs, rhs } = gr.as_ref() {
                                if matches!(lhs.as_ref(), Expr::Input) {
                                    if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                        return match op {
                                            crate::ir::BinOp::Add => Some(*n),
                                            crate::ir::BinOp::Sub => Some(-*n),
                                            _ => None,
                                        };
                                    }
                                }
                                if matches!(op, crate::ir::BinOp::Add) && matches!(rhs.as_ref(), Expr::Input) {
                                    if let Expr::Literal(Literal::Num(n, _)) = lhs.as_ref() {
                                        return Some(*n);
                                    }
                                }
                            }
                        }
                    }
                }
                None
            }
            // Case 1: Pipe(Pipe(explode, map(.+N)), implode) — left-associative
            if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Implode, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if let Expr::Pipe { left: el, right: er } = &sl {
                    if matches!(el.as_ref(), Expr::UnaryOp { op: UnaryOp::Explode, operand } if matches!(operand.as_ref(), Expr::Input)) {
                        if let Some(shift) = is_map_shift(er) {
                            return Expr::CallBuiltin {
                                op: BuiltinOp::ShiftCodepoints,
                                args: vec![Expr::Literal(Literal::Num(shift, None))],
                            };
                        }
                    }
                }
            }
            // Case 2: Pipe(explode, Pipe(map(.+N), implode)) — right-associative
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::Explode, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if let Expr::Pipe { left: mr, right: ir } = &sr {
                    if matches!(ir.as_ref(), Expr::UnaryOp { op: UnaryOp::Implode, operand } if matches!(operand.as_ref(), Expr::Input)) {
                        if let Some(shift) = is_map_shift(mr) {
                            return Expr::CallBuiltin {
                                op: BuiltinOp::ShiftCodepoints,
                                args: vec![Expr::Literal(Literal::Num(shift, None))],
                            };
                        }
                    }
                }
            }
            // Semantic: cmp_expr | not → inverted cmp_expr
            if matches!(&sr, Expr::Not) {
                if let Expr::BinOp { op, lhs, rhs } = &sl {
                    if let Some(inv) = op.invert_cmp() {
                        return Expr::BinOp { op: inv, lhs: lhs.clone(), rhs: rhs.clone() };
                    }
                }
            }
            // Semantic: [a, b, c] | reverse → [c, b, a]
            if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if let Expr::Collect { generator } = &sl {
                    fn collect_comma_elements(expr: &Expr, out: &mut Vec<Expr>) {
                        match expr {
                            Expr::Comma { left, right } => {
                                collect_comma_elements(left, out);
                                collect_comma_elements(right, out);
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    let mut elements = Vec::new();
                    collect_comma_elements(generator, &mut elements);
                    // Bail if any branch is multi-valued — otherwise the rewrite
                    // promotes the generator out of the array (issue #152).
                    if elements.len() >= 2 && elements.iter().all(is_single_valued_expr) {
                        elements.reverse();
                        let mut gen = elements.pop().unwrap();
                        while let Some(e) = elements.pop() {
                            gen = Expr::Comma { left: Box::new(e), right: Box::new(gen) };
                        }
                        return Expr::Collect { generator: Box::new(gen) };
                    }
                }
            }
            // Semantic: reverse | .[0] → .[-1], reverse | .[-1] → .[0]
            // reverse then index = access from the other end
            if matches!(&sl, Expr::UnaryOp { op: UnaryOp::Reverse, operand } if matches!(operand.as_ref(), Expr::Input)) {
                if let Expr::Index { expr: base, key } = &sr {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(crate::ir::Literal::Num(n, _)) = key.as_ref() {
                            let idx = *n as i64;
                            let new_idx = if idx >= 0 { -(idx + 1) } else { -idx - 1 };
                            return Expr::Index {
                                expr: Box::new(Expr::Input),
                                key: Box::new(Expr::Literal(crate::ir::Literal::Num(new_idx as f64, None))),
                            };
                        }
                    }
                }
            }
            // Semantic: [.f1, .f2, ...] | add op X → (.f1 + .f2 + ... + .fN) op X
            // Handles: add * N, add - N, add / N, add / length, add + N
            if let Expr::Collect { generator: ref lg } = sl {
                // Check if rhs is BinOp(op, add, something) where add = UnaryOp(Add, Input)
                let add_binop = if let Expr::BinOp { op: ref bop, lhs: ref blhs, rhs: ref brhs } = sr {
                    let is_add = matches!(blhs.as_ref(), Expr::UnaryOp { op: UnaryOp::Add, operand } if matches!(operand.as_ref(), Expr::Input));
                    if is_add && matches!(bop, crate::ir::BinOp::Add | crate::ir::BinOp::Sub | crate::ir::BinOp::Mul | crate::ir::BinOp::Div | crate::ir::BinOp::Mod) {
                        // Determine the right operand: either a literal or length
                        let rhs_expr = if let Expr::Literal(crate::ir::Literal::Num(n, _)) = brhs.as_ref() {
                            Some((*bop, Expr::Literal(crate::ir::Literal::Num(*n, None))))
                        } else if matches!(brhs.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                            // add / length → sum / N (length is count of elements)
                            None // handled specially below
                        } else {
                            None
                        };
                        if let Some((op, rhs_val)) = rhs_expr {
                            Some((op, rhs_val, false))
                        } else if matches!(bop, crate::ir::BinOp::Div) && matches!(brhs.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input)) {
                            Some((*bop, Expr::Literal(crate::ir::Literal::Num(0.0, None)), true)) // placeholder, will use N
                        } else {
                            None
                        }
                    } else { None }
                } else { None };
                if let Some((outer_op, rhs_val, use_length)) = add_binop {
                    fn collect_comma_elems2(e: &Expr, out: &mut Vec<Expr>) {
                        match e {
                            Expr::Comma { left, right } => {
                                collect_comma_elems2(left, out);
                                collect_comma_elems2(right, out);
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    let mut elems = Vec::new();
                    collect_comma_elems2(lg, &mut elems);
                    let n = elems.len();
                    // Bail when a branch is multi-valued: the array literal
                    // would otherwise stream rather than fold (issue #152).
                    if n >= 2 && elems.iter().all(is_single_valued_expr) {
                        let mut sum = elems.remove(0);
                        for elem in elems {
                            sum = Expr::BinOp {
                                op: crate::ir::BinOp::Add,
                                lhs: Box::new(sum),
                                rhs: Box::new(elem),
                            };
                        }
                        let actual_rhs = if use_length {
                            Expr::Literal(crate::ir::Literal::Num(n as f64, None))
                        } else {
                            rhs_val
                        };
                        return Expr::BinOp {
                            op: outer_op,
                            lhs: Box::new(sum),
                            rhs: Box::new(actual_rhs),
                        };
                    }
                }
            }
            // Semantic: [.f1, .f2, ...] | add → .f1 + .f2 + ... + .fN
            // This catches cases where the parser's pipe-level optimization missed it
            // (e.g., from simplify_expr creating new Collect | add patterns)
            if let Expr::Collect { generator: ref lg } = sl {
                if matches!(&sr, Expr::UnaryOp { op: UnaryOp::Add, operand } if matches!(operand.as_ref(), Expr::Input)) {
                    fn collect_elems_for_add(e: &Expr, out: &mut Vec<Expr>) {
                        match e {
                            Expr::Comma { left, right } => {
                                collect_elems_for_add(left, out);
                                collect_elems_for_add(right, out);
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    // Rewrite is only valid when every element yields exactly
                    // one value. `Empty` yields zero; `.[]`, `recurse`, and
                    // other generators yield many. `[.[]] | add` was collapsing
                    // to `.[]` (issue #56) because a single-element list with
                    // a generator inside was treated as "identity".
                    fn is_single_valued(e: &Expr) -> bool {
                        match e {
                            Expr::Empty => false,
                            Expr::Each { .. } | Expr::EachOpt { .. }
                            | Expr::Comma { .. } | Expr::Recurse { .. }
                            | Expr::Range { .. } | Expr::Limit { .. }
                            | Expr::RegexMatch { .. } | Expr::RegexScan { .. }
                            | Expr::RegexCapture { .. } => false,
                            Expr::Pipe { left, right } => is_single_valued(left) && is_single_valued(right),
                            Expr::IfThenElse { cond, then_branch, else_branch } => {
                                is_single_valued(cond) && is_single_valued(then_branch) && is_single_valued(else_branch)
                            }
                            Expr::TryCatch { try_expr, catch_expr, .. } => {
                                is_single_valued(try_expr) && is_single_valued(catch_expr)
                            }
                            Expr::Alternative { primary, fallback } => {
                                is_single_valued(primary) && is_single_valued(fallback)
                            }
                            Expr::LetBinding { value, body, .. } => {
                                is_single_valued(value) && is_single_valued(body)
                            }
                            Expr::Collect { .. } => true,
                            // Compound nodes can hide generators in their operands
                            // (e.g. simplify_expr beta-reduces `gen | .*k` into
                            // `BinOp(gen, k)` — issue #102). Recurse so the fold
                            // doesn't promote that buried generator.
                            Expr::BinOp { lhs, rhs, .. } => is_single_valued(lhs) && is_single_valued(rhs),
                            Expr::UnaryOp { operand, .. } => is_single_valued(operand),
                            Expr::Negate { operand } => is_single_valued(operand),
                            Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => {
                                is_single_valued(expr) && is_single_valued(key)
                            }
                            Expr::Input | Expr::Literal(_) | Expr::LoadVar { .. }
                            | Expr::Not
                            | Expr::Slice { .. }
                            | Expr::StringInterpolation { .. } | Expr::ObjectConstruct { .. }
                            | Expr::RegexTest { .. } | Expr::RegexSub { .. } | Expr::RegexGsub { .. } => true,
                            _ => false,
                        }
                    }
                    let mut elems = Vec::new();
                    collect_elems_for_add(lg, &mut elems);
                    let all_single = !elems.is_empty()
                        && elems.iter().all(|e| is_single_valued(e));
                    if all_single {
                        if elems.len() == 1 {
                            // [expr] | add → expr (single element, add is identity)
                            return elems.remove(0);
                        } else if elems.len() >= 2 {
                            let mut result = elems.remove(0);
                            for elem in elems {
                                result = Expr::BinOp {
                                    op: crate::ir::BinOp::Add,
                                    lhs: Box::new(result),
                                    rhs: Box::new(elem),
                                };
                            }
                            return result;
                        }
                    }
                }
            }
            // NOTE: `[A, [B, C], D] | flatten → [A, B, C, D]` rewrite was removed
            // (#221). Bare `flatten` is recursive in jq 1.8.1, so unwrapping just
            // one literal level produced wrong results when the inner elements
            // were themselves arrays (`[1, [.b, 3], 4] | flatten` with `.b=[10,20]`
            // should yield `[1,10,20,3,4]`, not `[1,[10,20],3,4]`). The old
            // rewrite couldn't know the runtime shape of `.b`, so it's unsafe
            // by construction.
            // Semantic: [e0, e1, ...] | .[N] → eN (extract Nth element at compile time)
            // Also handles .[−1] → last element
            if let Expr::Collect { generator: ref lg } = sl {
                if let Expr::Index { expr: base, key } = &sr {
                    if matches!(base.as_ref(), Expr::Input) {
                        if let Expr::Literal(Literal::Num(n, _)) = key.as_ref() {
                            if n.is_nan() || !n.is_finite() { /* skip NaN/Inf indices */ }
                            else {
                            fn collect_comma_for_idx(e: &Expr, out: &mut Vec<Expr>) {
                                match e {
                                    Expr::Comma { left, right } => {
                                        collect_comma_for_idx(left, out);
                                        collect_comma_for_idx(right, out);
                                    }
                                    other => out.push(other.clone()),
                                }
                            }
                            // Only constant-fold when every element yields
                            // exactly one value — otherwise the rewrite promotes
                            // a generator (range/recurse/.[] /limit/...) to the
                            // top level and streams every element instead of
                            // returning the indexed one (issue #78).
                            fn is_single_valued_idx(e: &Expr) -> bool {
                                match e {
                                    Expr::Empty => false,
                                    Expr::Each { .. } | Expr::EachOpt { .. }
                                    | Expr::Comma { .. } | Expr::Recurse { .. }
                                    | Expr::Range { .. } | Expr::Limit { .. }
                                    | Expr::RegexMatch { .. } | Expr::RegexScan { .. }
                                    | Expr::RegexCapture { .. } => false,
                                    Expr::Pipe { left, right } => is_single_valued_idx(left) && is_single_valued_idx(right),
                                    Expr::IfThenElse { cond, then_branch, else_branch } =>
                                        is_single_valued_idx(cond) && is_single_valued_idx(then_branch) && is_single_valued_idx(else_branch),
                                    Expr::TryCatch { try_expr, catch_expr, .. } =>
                                        is_single_valued_idx(try_expr) && is_single_valued_idx(catch_expr),
                                    Expr::Alternative { primary, fallback } =>
                                        is_single_valued_idx(primary) && is_single_valued_idx(fallback),
                                    Expr::LetBinding { value, body, .. } =>
                                        is_single_valued_idx(value) && is_single_valued_idx(body),
                                    Expr::Collect { .. } => true,
                                    // Compound nodes can hide generators in their
                                    // operands (e.g. simplify_expr beta-reduces
                                    // `gen | .*k` into `BinOp(gen, k)` — issue #102).
                                    // Recurse so the fold doesn't promote it.
                                    Expr::BinOp { lhs, rhs, .. } => is_single_valued_idx(lhs) && is_single_valued_idx(rhs),
                                    Expr::UnaryOp { operand, .. } => is_single_valued_idx(operand),
                                    Expr::Negate { operand } => is_single_valued_idx(operand),
                                    Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => {
                                        is_single_valued_idx(expr) && is_single_valued_idx(key)
                                    }
                                    Expr::Input | Expr::Literal(_) | Expr::LoadVar { .. }
                                    | Expr::Not
                                    | Expr::Slice { .. }
                                    | Expr::StringInterpolation { .. } | Expr::ObjectConstruct { .. }
                                    | Expr::RegexTest { .. } | Expr::RegexSub { .. } | Expr::RegexGsub { .. } => true,
                                    _ => false,
                                }
                            }
                            let mut elems = Vec::new();
                            collect_comma_for_idx(lg, &mut elems);
                            let all_single = elems.iter().all(is_single_valued_idx);
                            // Only constant-fold when the negative index
                            // actually lands inside the array. Previously the
                            // negative branch clamped to 0 via `.max(0)`,
                            // returning the first element for any out-of-range
                            // negative index (issue #42). Falling through for
                            // the out-of-range cases lets the runtime emit the
                            // correct null result.
                            let effective: Option<usize> =
                                crate::value::resolve_array_index(*n, elems.len());
                            if all_single {
                                if let Some(i) = effective {
                                    // jq evaluates every element when building
                                    // the array literal, so a non-selected
                                    // element that references Input may raise
                                    // and must propagate. Skip the fold when
                                    // any sibling touches Input — otherwise
                                    // `[.[0], 0] | .[1]` on a non-indexable
                                    // input would silently return `0` instead
                                    // of erroring (#234).
                                    let any_sibling_touches_input = elems
                                        .iter()
                                        .enumerate()
                                        .any(|(j, e)| j != i && contains_input(e));
                                    if !any_sibling_touches_input {
                                        return elems.swap_remove(i);
                                    }
                                }
                            }
                            }
                        }
                    }
                }
            }
            // Semantic: [a, b] | min → if a <= b then a else b end
            // Semantic: [a, b] | max → if a > b then a else b end
            if let Expr::Collect { generator: ref lg } = sl {
                let is_min = matches!(&sr, Expr::UnaryOp { op: UnaryOp::Min, operand } if matches!(operand.as_ref(), Expr::Input));
                let is_max = matches!(&sr, Expr::UnaryOp { op: UnaryOp::Max, operand } if matches!(operand.as_ref(), Expr::Input));
                if is_min || is_max {
                    // Collect all elements from comma-chain
                    fn collect_elems(e: &Expr, out: &mut Vec<Expr>) {
                        match e {
                            Expr::Comma { left, right } => {
                                collect_elems(left, out);
                                collect_elems(right, out);
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    let mut elems = Vec::new();
                    collect_elems(lg, &mut elems);
                    // Bail on multi-valued branches — issue #152: the rewrite would
                    // otherwise stream every value through the cmp instead of
                    // folding the collected array.
                    if elems.len() >= 2 && elems.iter().all(is_single_valued_expr) {
                        // Fold: min(a,b,c) = min(min(a,b), c)
                        let cmp_op = if is_min { crate::ir::BinOp::Le } else { crate::ir::BinOp::Gt };
                        let mut result = elems.remove(0);
                        for elem in elems {
                            result = Expr::IfThenElse {
                                cond: Box::new(Expr::BinOp { op: cmp_op, lhs: Box::new(result.clone()), rhs: Box::new(elem.clone()) }),
                                then_branch: Box::new(result),
                                else_branch: Box::new(elem),
                            };
                        }
                        return simplify_expr(&result);
                    }
                }
            }
            // Semantic: [a, b] | sort → if a <= b then [a,b] else [b,a] end
            if let Expr::Collect { generator: ref lg } = sl {
                let is_sort = matches!(&sr, Expr::UnaryOp { op: UnaryOp::Sort, operand } if matches!(operand.as_ref(), Expr::Input));
                if is_sort {
                    if let Expr::Comma { left, right } = lg.as_ref() {
                        if !matches!(left.as_ref(), Expr::Comma { .. }) && !matches!(right.as_ref(), Expr::Comma { .. })
                            && is_single_valued_expr(left) && is_single_valued_expr(right)
                        {
                            // Bail on multi-valued branches (issue #152).
                            let a = left.as_ref().clone();
                            let b = right.as_ref().clone();
                            return simplify_expr(&Expr::IfThenElse {
                                cond: Box::new(Expr::BinOp { op: crate::ir::BinOp::Le, lhs: Box::new(a.clone()), rhs: Box::new(b.clone()) }),
                                then_branch: Box::new(Expr::Collect { generator: Box::new(Expr::Comma { left: Box::new(a.clone()), right: Box::new(b.clone()) }) }),
                                else_branch: Box::new(Expr::Collect { generator: Box::new(Expr::Comma { left: Box::new(b), right: Box::new(a) }) }),
                            });
                        }
                    }
                }
            }
            // Semantic: [e1, e2, ...] | add → e1 + e2 + ... (avoids array construction)
            // Also: [e1, e2, ...] | add / length → (e1 + e2 + ...) / N
            if let Expr::Collect { generator: ref lg } = sl {
                // Check for add or add / length
                let is_add = matches!(&sr, Expr::UnaryOp { op: UnaryOp::Add, operand } if matches!(operand.as_ref(), Expr::Input));
                let is_add_div_length = if !is_add {
                    if let Expr::BinOp { op: crate::ir::BinOp::Div, lhs, rhs } = &sr {
                        matches!(lhs.as_ref(), Expr::UnaryOp { op: UnaryOp::Add, operand } if matches!(operand.as_ref(), Expr::Input))
                        && matches!(rhs.as_ref(), Expr::UnaryOp { op: UnaryOp::Length, operand } if matches!(operand.as_ref(), Expr::Input))
                    } else { false }
                } else { false };
                if is_add || is_add_div_length {
                    fn collect_comma_elems(e: &Expr, out: &mut Vec<Expr>) {
                        match e {
                            Expr::Comma { left, right } => {
                                collect_comma_elems(left, out);
                                collect_comma_elems(right, out);
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    let mut elems = Vec::new();
                    collect_comma_elems(lg, &mut elems);
                    // Bail on multi-valued branches (issue #152).
                    if elems.len() >= 2 && elems.len() <= 16
                        && elems.iter().all(is_single_valued_expr)
                    {
                        let n = elems.len();
                        let mut result = elems.remove(0);
                        for elem in elems {
                            result = Expr::BinOp {
                                op: crate::ir::BinOp::Add,
                                lhs: Box::new(result),
                                rhs: Box::new(elem),
                            };
                        }
                        if is_add_div_length {
                            result = Expr::BinOp {
                                op: crate::ir::BinOp::Div,
                                lhs: Box::new(result),
                                rhs: Box::new(Expr::Literal(Literal::Num(n as f64, None))),
                            };
                        }
                        return simplify_expr(&result);
                    }
                }
            }
            // Semantic: [e1, e2, ...] | any(f) → (e1|f) or (e2|f) or ...
            // Semantic: [e1, e2, ...] | all(f) → (e1|f) and (e2|f) and ...
            if let Expr::Collect { generator: ref lg } = sl {
                let (is_any_all, predicate) = match &sr {
                    Expr::AnyShort { generator: gen, predicate } => {
                        if matches!(gen.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            (Some(true), Some(predicate.as_ref()))
                        } else { (None, None) }
                    }
                    Expr::AllShort { generator: gen, predicate } => {
                        if matches!(gen.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            (Some(false), Some(predicate.as_ref()))
                        } else { (None, None) }
                    }
                    // Also: any without explicit predicate = any(.)
                    Expr::UnaryOp { op: UnaryOp::Any, operand } if matches!(operand.as_ref(), Expr::Input) => {
                        (Some(true), Some(&Expr::Input as &Expr))
                    }
                    Expr::UnaryOp { op: UnaryOp::All, operand } if matches!(operand.as_ref(), Expr::Input) => {
                        (Some(false), Some(&Expr::Input as &Expr))
                    }
                    _ => (None, None),
                };
                if let (Some(is_any), Some(pred)) = (is_any_all, predicate) {
                    fn collect_comma_for_any(e: &Expr, out: &mut Vec<Expr>) {
                        match e {
                            Expr::Comma { left, right } => {
                                collect_comma_for_any(left, out);
                                collect_comma_for_any(right, out);
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    let mut elems = Vec::new();
                    collect_comma_for_any(lg, &mut elems);
                    // Bail on multi-valued branches (issue #152) AND on
                    // multi-valued predicates: `(e1, e2) and (e3, e4)` is
                    // a 4-valued cross product, while jq's `all/any` short-
                    // circuits over the flattened predicate stream and
                    // returns one bool. Without this guard, e.g.
                    // `[1,2] | all((true, false))` rewrites to
                    // `(true, false) and (true, false)` and emits 3 values
                    // instead of jq's single `false`.
                    if elems.len() >= 2 && elems.len() <= 8
                        && elems.iter().all(is_single_valued_expr)
                        && is_single_valued_expr(pred)
                    {
                        let combiner = if is_any { crate::ir::BinOp::Or } else { crate::ir::BinOp::And };
                        let mut result = simplify_expr(&Expr::Pipe {
                            left: Box::new(elems.remove(0)),
                            right: Box::new(pred.clone()),
                        });
                        for elem in elems {
                            let applied = simplify_expr(&Expr::Pipe {
                                left: Box::new(elem),
                                right: Box::new(pred.clone()),
                            });
                            result = Expr::BinOp {
                                op: combiner,
                                lhs: Box::new(result),
                                rhs: Box::new(applied),
                            };
                        }
                        return result;
                    }
                }
            }
            // Beta-reduction: .x | . + 1 → .x + 1
            // `sr` must actually reference Input. When it doesn't, the
            // substitution is a no-op and folding away `sl |` would
            // silently swallow lhs runtime errors — e.g. `.a | 0` on a
            // non-object array must raise "Cannot index array with
            // string a", not collapse to a bare `0` (#172).
            //
            // Additionally, refuse substitution when any Input position
            // in `sr` is reached only through a short-circuiting wrapper
            // (`Alternative.fallback`, `TryCatch.*`, `Alternative.primary`
            // when fallback would be reached on error). Substituting `sl`
            // into a fallback / catch slot can elide a runtime error from
            // `sl`'s evaluation that the original Pipe semantics would
            // have raised eagerly (#354).
            if sl.is_simple_scalar()
                && sr.is_input_free()
                && contains_input(&sr)
                && !input_behind_short_circuit(&sr)
            {
                return sr.substitute_input(&sl);
            }
            // [gen] | map(f) = [gen] | [.[] | f] → [gen | f]
            // Distributes f over each element of gen via beta-reduction.
            // Each element gets piped through f and simplified (beta-reduced).
            // [gen] | map(f) distribution helper
            fn is_comma_of_simple_scalars(e: &Expr) -> bool {
                match e {
                    Expr::Comma { left, right } => {
                        is_comma_of_simple_scalars(left) && is_comma_of_simple_scalars(right)
                    }
                    other => other.is_simple_scalar(),
                }
            }
            fn distribute_map(gen: &Expr, f: &Expr) -> Expr {
                match gen {
                    Expr::Comma { left, right } => {
                        Expr::Comma {
                            left: Box::new(distribute_map(left, f)),
                            right: Box::new(distribute_map(right, f)),
                        }
                    }
                    other => {
                        let piped = Expr::Pipe {
                            left: Box::new(other.clone()),
                            right: Box::new(f.clone()),
                        };
                        simplify_expr(&piped)
                    }
                }
            }
            fn try_extract_map_body(expr: &Expr) -> Option<Expr> {
                // [.[] | f] → Some(f)
                if let Expr::Collect { generator } = expr {
                    if let Expr::Pipe { left, right } = generator.as_ref() {
                        if matches!(left.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                            return Some(right.as_ref().clone());
                        }
                    }
                    // Also: [f(.[])] where .[] appears inside f — e.g., [.[] * 2]
                    // Replace Each(Input) with Input to get the body
                    fn replace_each_with_input(e: &Expr) -> Option<Expr> {
                        match e {
                            Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input) => {
                                Some(Expr::Input)
                            }
                            Expr::BinOp { op, lhs, rhs } => {
                                let new_lhs = replace_each_with_input(lhs);
                                let new_rhs = replace_each_with_input(rhs);
                                if new_lhs.is_some() || new_rhs.is_some() {
                                    Some(Expr::BinOp {
                                        op: *op,
                                        lhs: Box::new(new_lhs.unwrap_or_else(|| lhs.as_ref().clone())),
                                        rhs: Box::new(new_rhs.unwrap_or_else(|| rhs.as_ref().clone())),
                                    })
                                } else {
                                    None
                                }
                            }
                            Expr::UnaryOp { op, operand } => {
                                replace_each_with_input(operand).map(|o| Expr::UnaryOp {
                                    op: *op, operand: Box::new(o),
                                })
                            }
                            Expr::Negate { operand } => {
                                replace_each_with_input(operand).map(|o| Expr::Negate { operand: Box::new(o) })
                            }
                            _ => None,
                        }
                    }
                    if let Some(body) = replace_each_with_input(generator) {
                        return Some(body);
                    }
                }
                None
            }
            if let Expr::Collect { generator: ref lg } = sl {
                if is_comma_of_simple_scalars(lg) {
                    // [gen] | map(f) → [gen distributed with f]
                    if let Some(f) = try_extract_map_body(&sr) {
                        return Expr::Collect { generator: Box::new(distribute_map(lg, &f)) };
                    }
                    // [gen] | Pipe(map(f), rest) → [gen distributed with f] | rest
                    if let Expr::Pipe { left: ref pl, right: ref pr } = sr {
                        if let Some(f) = try_extract_map_body(pl) {
                            let distributed = Expr::Collect { generator: Box::new(distribute_map(lg, &f)) };
                            let result = Expr::Pipe { left: Box::new(distributed), right: pr.clone() };
                            return simplify_expr(&result);
                        }
                    }
                }
            }
            // split("s") | length > 1 → contains("s")  (more efficient, enables raw byte path)
            if let Expr::CallBuiltin { op: name, args } = &sl {
                if *name == BuiltinOp::Split && args.len() == 1 {
                    if let Expr::Literal(crate::ir::Literal::Str(delim)) = &args[0] {
                        if let Expr::BinOp { op, lhs: cmp_lhs, rhs: cmp_rhs } = &sr {
                            if let Expr::UnaryOp { op: crate::ir::UnaryOp::Length, operand } = cmp_lhs.as_ref() {
                                if matches!(operand.as_ref(), Expr::Input) {
                                    if let Expr::Literal(crate::ir::Literal::Num(n, _)) = cmp_rhs.as_ref() {
                                        // split(S) | length > 1 means "contains S"
                                        // split(S) | length > 0 is always true for any finite string
                                        if !delim.is_empty() && *n == 1.0 && matches!(op, crate::ir::BinOp::Gt) {
                                            return Expr::CallBuiltin {
                                                op: BuiltinOp::Contains,
                                                args: vec![Expr::Literal(crate::ir::Literal::Str(delim.clone()))],
                                            };
                                        }
                                        if !delim.is_empty() && *n == 0.0 && matches!(op, crate::ir::BinOp::Gt) {
                                            return Expr::Literal(crate::ir::Literal::True);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // select(A) | select(B) → select(A and B)
            if let Expr::IfThenElse { cond: cond_a, then_branch: then_a, else_branch: else_a } = &sl {
                if matches!(then_a.as_ref(), Expr::Input) && matches!(else_a.as_ref(), Expr::Empty) {
                    if let Expr::IfThenElse { cond: cond_b, then_branch: then_b, else_branch: else_b } = &sr {
                        if matches!(then_b.as_ref(), Expr::Input) && matches!(else_b.as_ref(), Expr::Empty) {
                            return simplify_expr(&Expr::IfThenElse {
                                cond: Box::new(Expr::BinOp { op: crate::ir::BinOp::And, lhs: cond_a.clone(), rhs: cond_b.clone() }),
                                then_branch: Box::new(Expr::Input),
                                else_branch: Box::new(Expr::Empty),
                            });
                        }
                    }
                    // select(A) | Pipe(select(B), rest) → select(A and B) | rest
                    if let Expr::Pipe { left: pl, right: pr } = &sr {
                        if let Expr::IfThenElse { cond: cond_b, then_branch: then_b, else_branch: else_b } = pl.as_ref() {
                            if matches!(then_b.as_ref(), Expr::Input) && matches!(else_b.as_ref(), Expr::Empty) {
                                let merged_select = Expr::IfThenElse {
                                    cond: Box::new(Expr::BinOp { op: crate::ir::BinOp::And, lhs: cond_a.clone(), rhs: cond_b.clone() }),
                                    then_branch: Box::new(Expr::Input),
                                    else_branch: Box::new(Expr::Empty),
                                };
                                return simplify_expr(&Expr::Pipe {
                                    left: Box::new(merged_select),
                                    right: pr.clone(),
                                });
                            }
                        }
                    }
                }
            }
            // map(f) | map(g) → map(f | g) — eliminate intermediate array allocation
            if let Some(f) = try_extract_map_body(&sl) {
                if let Some(g) = try_extract_map_body(&sr) {
                    let fused = Expr::Pipe { left: Box::new(f), right: Box::new(g) };
                    return simplify_expr(&Expr::Collect {
                        generator: Box::new(Expr::Pipe {
                            left: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                            right: Box::new(fused),
                        }),
                    });
                }
                // map(f) | Pipe(map(g), rest) → map(f | g) | rest
                if let Expr::Pipe { left: ref pl, right: ref pr } = sr {
                    if let Some(g) = try_extract_map_body(pl) {
                        let fused = Expr::Pipe { left: Box::new(f.clone()), right: Box::new(g) };
                        let fused_map = Expr::Collect {
                            generator: Box::new(Expr::Pipe {
                                left: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                                right: Box::new(fused),
                            }),
                        };
                        return simplify_expr(&Expr::Pipe {
                            left: Box::new(fused_map),
                            right: pr.clone(),
                        });
                    }
                }
            }
            // group_by(.f) | map(.[0]) → unique_by(.f)
            if let Expr::ClosureOp { op: crate::ir::ClosureOpKind::GroupBy, input_expr, key_expr } = &sl {
                if let Some(body) = try_extract_map_body(&sr) {
                    // Check if body is .[0]
                    if let Expr::Index { expr: base, key } = &body {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(crate::ir::Literal::Num(n, _)) = key.as_ref() {
                                if *n == 0.0 {
                                    return Expr::ClosureOp {
                                        op: crate::ir::ClosureOpKind::UniqueBy,
                                        input_expr: input_expr.clone(),
                                        key_expr: key_expr.clone(),
                                    };
                                }
                            }
                        }
                    }
                }
            }
            Expr::Pipe { left: Box::new(sl), right: Box::new(sr) }
        }
        // Recurse into IfThenElse conditions (select patterns)
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            let sc = simplify_expr(cond);
            let st = simplify_expr(then_branch);
            let se = simplify_expr(else_branch);
            // Constant condition folding: if true then A else B end → A
            match &sc {
                Expr::Literal(crate::ir::Literal::Null) | Expr::Literal(crate::ir::Literal::False) => return se,
                Expr::Literal(crate::ir::Literal::True) | Expr::Literal(crate::ir::Literal::Num(_, _)) | Expr::Literal(crate::ir::Literal::Str(_)) => return st,
                _ => {}
            }
            // if A then (if B then X else empty end) else empty end → if (A and B) then X else empty end
            if matches!(se, Expr::Empty) {
                if let Expr::IfThenElse { cond: cond_inner, then_branch: then_inner, else_branch: else_inner } = &st {
                    if matches!(else_inner.as_ref(), Expr::Empty) {
                        return Expr::IfThenElse {
                            cond: Box::new(Expr::BinOp { op: crate::ir::BinOp::And, lhs: Box::new(sc), rhs: cond_inner.clone() }),
                            then_branch: then_inner.clone(),
                            else_branch: Box::new(Expr::Empty),
                        };
                    }
                }
            }
            // if .field then .field else F end → .field // F
            if let Expr::Index { expr: base_c, key: key_c } = &sc {
                if matches!(base_c.as_ref(), Expr::Input) {
                    if let Expr::Literal(crate::ir::Literal::Str(fc)) = key_c.as_ref() {
                        if let Expr::Index { expr: base_t, key: key_t } = &st {
                            if matches!(base_t.as_ref(), Expr::Input) {
                                if let Expr::Literal(crate::ir::Literal::Str(ft)) = key_t.as_ref() {
                                    if fc == ft {
                                        return Expr::Alternative {
                                            primary: Box::new(sc),
                                            fallback: Box::new(se),
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Expr::IfThenElse {
                cond: Box::new(sc),
                then_branch: Box::new(st),
                else_branch: Box::new(se),
            }
        }
        // Inline LetBinding: (E as $x | F) → F[$x := E] when E is simple.
        //
        // When F doesn't reference $x, the substitution drops E entirely. That
        // is only safe when E is guaranteed not to error: jq's `as` is
        // documented to evaluate the right-hand side eagerly and propagate any
        // error (the destructuring `. as {a:$a} | "lit"` parses to nested
        // LetBindings whose RHSes are `Index{Input, "a"}` — those must error
        // on non-objects even when the body doesn't read $a). Without this
        // guard, the LetBinding gets eliminated and the catch never fires
        // (#521).
        Expr::LetBinding { var_index, value, body } => {
            let sv = simplify_expr(value);
            let sb = simplify_expr(body);
            if sv.is_simple_scalar() {
                let body_uses_var = expr_references_var(&sb, *var_index);
                if body_uses_var {
                    // Substituting the var's value is unsafe when the value
                    // reads `.` and a reference sits where `.` was rebound —
                    // `. as $v | map($v)` would otherwise become `map(.)`,
                    // reading the per-element `.` instead of the bound one
                    // (#818). A value that doesn't read `.` is dot-independent
                    // and always safe to substitute.
                    let dot_safe = !contains_input(&sv)
                        || !var_in_rebound_dot_scope(&sb, *var_index);
                    if dot_safe {
                        return sb.substitute_var(*var_index, &sv);
                    }
                } else if expr_is_pure_scalar(&sv) {
                    // Unused binding with a side-effect-free value: drop it.
                    return sb.substitute_var(*var_index, &sv);
                }
            }
            Expr::LetBinding { var_index: *var_index, value: Box::new(sv), body: Box::new(sb) }
        }
        // Recurse into ObjectConstruct, BinOp, etc.
        Expr::ObjectConstruct { pairs } => {
            Expr::ObjectConstruct {
                pairs: pairs.iter().map(|(k, v)| (simplify_expr(k), simplify_expr(v))).collect(),
            }
        }
        Expr::BinOp { op, lhs, rhs } => {
            let sl = simplify_expr(lhs);
            let sr = simplify_expr(rhs);
            // {a:.x} + {b:.y} → {a:.x, b:.y} (merge object constructions)
            // For `+`: right-side keys win on collision = same as last-key-wins in single construct.
            // So {A} + {B} → {A, B} is always valid for `+`.
            // For `*`: only safe when all keys are distinct literal strings (no nested merge).
            if matches!(op, crate::ir::BinOp::Add) {
                if let (Expr::ObjectConstruct { pairs: p1 }, Expr::ObjectConstruct { pairs: p2 }) = (&sl, &sr) {
                    let mut merged = p1.clone();
                    merged.extend(p2.iter().cloned());
                    return Expr::ObjectConstruct { pairs: merged };
                }
            }
            if matches!(op, crate::ir::BinOp::Mul) {
                if let (Expr::ObjectConstruct { pairs: p1 }, Expr::ObjectConstruct { pairs: p2 }) = (&sl, &sr) {
                    let all_literal_keys = p1.iter().chain(p2.iter()).all(|(k, _)| {
                        matches!(k, Expr::Literal(crate::ir::Literal::Str(_)))
                    });
                    if all_literal_keys {
                        let mut keys: Vec<&str> = Vec::new();
                        let mut all_distinct = true;
                        for (k, _) in p1.iter().chain(p2.iter()) {
                            if let Expr::Literal(crate::ir::Literal::Str(s)) = k {
                                if keys.contains(&s.as_str()) { all_distinct = false; break; }
                                keys.push(s.as_str());
                            }
                        }
                        if all_distinct {
                            let mut merged = p1.clone();
                            merged.extend(p2.iter().cloned());
                            return Expr::ObjectConstruct { pairs: merged };
                        }
                    }
                }
            }
            // Constant fold: Num op Num → Num
            if let (Expr::Literal(Literal::Num(a, _)), Expr::Literal(Literal::Num(b, _))) = (&sl, &sr) {
                let result = match op {
                    crate::ir::BinOp::Add => Some(a + b),
                    crate::ir::BinOp::Sub => Some(a - b),
                    crate::ir::BinOp::Mul => Some(a * b),
                    crate::ir::BinOp::Div if *b != 0.0 => Some(a / b),
                    crate::ir::BinOp::Mod if a.is_finite() && b.is_finite() => {
                        // jq truncates both operands to int before %, so 1 % 1.5 = 0.
                        let yi = *b as i64;
                        let xi = *a as i64;
                        if yi == 0 { None }
                        else if xi == i64::MIN && yi == -1 { Some(0.0) }
                        else { Some((xi % yi) as f64) }
                    }
                    _ => None,
                };
                if let Some(r) = result {
                    return Expr::Literal(Literal::Num(r, None));
                }
                // Comparison ops. jq treats NaN as below every number
                // (#115) so ordering folds must mirror runtime semantics
                // — IEEE 754's "all false" would have NaN sort scattered
                // and `nan < nan` come out false. `==`/`!=` keep IEEE 754
                // inequality (`nan == nan` is still false).
                let cmp_result = match op {
                    crate::ir::BinOp::Eq => Some(*a == *b),
                    crate::ir::BinOp::Ne => Some(*a != *b),
                    crate::ir::BinOp::Lt => Some(crate::eval::jq_num_lt(*a, *b)),
                    crate::ir::BinOp::Gt => Some(crate::eval::jq_num_gt(*a, *b)),
                    crate::ir::BinOp::Le => Some(crate::eval::jq_num_le(*a, *b)),
                    crate::ir::BinOp::Ge => Some(crate::eval::jq_num_ge(*a, *b)),
                    _ => None,
                };
                if let Some(r) = cmp_result {
                    return if r { Expr::Literal(Literal::True) } else { Expr::Literal(Literal::False) };
                }
            }
            // Constant fold: Str + Str → Str
            if matches!(op, crate::ir::BinOp::Add) {
                if let (Expr::Literal(Literal::Str(a)), Expr::Literal(Literal::Str(b))) = (&sl, &sr) {
                    return Expr::Literal(Literal::Str(format!("{}{}", a, b)));
                }
            }
            Expr::BinOp { op: *op, lhs: Box::new(sl), rhs: Box::new(sr) }
        }
        Expr::StringInterpolation { parts } => {
            use crate::ir::StringPart;
            Expr::StringInterpolation {
                parts: parts.iter().map(|p| match p {
                    StringPart::Literal(s) => StringPart::Literal(s.clone()),
                    StringPart::Expr(e) => StringPart::Expr(simplify_expr(e)),
                }).collect(),
            }
        }
        Expr::UnaryOp { op, operand } => {
            let so = simplify_expr(operand);
            // Normalize f|op → f | op(.) when f is not input
            if !matches!(&so, Expr::Input) {
                return simplify_expr(&Expr::Pipe {
                    left: Box::new(so),
                    right: Box::new(Expr::UnaryOp { op: *op, operand: Box::new(Expr::Input) }),
                });
            }
            Expr::UnaryOp { op: *op, operand: Box::new(so) }
        }
        Expr::Collect { generator } => {
            Expr::Collect { generator: Box::new(simplify_expr(generator)) }
        }
        Expr::Comma { left, right } => {
            Expr::Comma { left: Box::new(simplify_expr(left)), right: Box::new(simplify_expr(right)) }
        }
        Expr::Index { expr, key } => {
            let se = simplify_expr(expr);
            let sk = simplify_expr(key);
            // Normalize f[k] → f | .[k] ONLY when k is a literal (doesn't reference input)
            // f[.baz] is NOT f | .[.baz] because .baz binds to different inputs
            if !matches!(&se, Expr::Input) && matches!(&sk, Expr::Literal(_)) {
                return simplify_expr(&Expr::Pipe {
                    left: Box::new(se),
                    right: Box::new(Expr::Index { expr: Box::new(Expr::Input), key: Box::new(sk) }),
                });
            }
            Expr::Index { expr: Box::new(se), key: Box::new(sk) }
        }
        Expr::IndexOpt { expr, key } => {
            let se = simplify_expr(expr);
            let sk = simplify_expr(key);
            if !matches!(&se, Expr::Input) && matches!(&sk, Expr::Literal(_)) {
                return simplify_expr(&Expr::Pipe {
                    left: Box::new(se),
                    right: Box::new(Expr::IndexOpt { expr: Box::new(Expr::Input), key: Box::new(sk) }),
                });
            }
            Expr::IndexOpt { expr: Box::new(se), key: Box::new(sk) }
        }
        Expr::CallBuiltin { op: name, args } => {
            let sargs: Vec<_> = args.iter().map(|a| simplify_expr(a)).collect();
            // walk(.) → identity
            if *name == BuiltinOp::Walk && sargs.len() == 1 && matches!(&sargs[0], Expr::Input) {
                return Expr::Input;
            }
            Expr::CallBuiltin { op: *name, args: sargs }
        }
        Expr::Alternative { primary, fallback } => {
            Expr::Alternative { primary: Box::new(simplify_expr(primary)), fallback: Box::new(simplify_expr(fallback)) }
        }
        Expr::Each { input_expr } => {
            let se = simplify_expr(input_expr);
            // Normalize f[] → f | .[] when f is not input
            if !matches!(&se, Expr::Input) {
                return simplify_expr(&Expr::Pipe {
                    left: Box::new(se),
                    right: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                });
            }
            Expr::Each { input_expr: Box::new(se) }
        }
        Expr::EachOpt { input_expr } => {
            let se = simplify_expr(input_expr);
            if !matches!(&se, Expr::Input) {
                return simplify_expr(&Expr::Pipe {
                    left: Box::new(se),
                    right: Box::new(Expr::EachOpt { input_expr: Box::new(Expr::Input) }),
                });
            }
            Expr::EachOpt { input_expr: Box::new(se) }
        }
        Expr::Negate { operand } => {
            let s = simplify_expr(operand);
            if let Expr::Literal(Literal::Num(n, repr)) = &s {
                // jq normalises `-0` (Negate of zero) back to `+0`. Only IEEE
                // arithmetic (`0 * -1`, `0 - 0`) lands a signed zero. Issue #110.
                let new_n = if *n == 0.0 { 0.0 } else { -n };
                let new_repr = crate::value::Value::negate_repr(repr.clone());
                Expr::Literal(Literal::Num(new_n, new_repr))
            } else {
                Expr::Negate { operand: Box::new(s) }
            }
        }
        Expr::Slice { expr, from, to } => {
            let se = simplify_expr(expr);
            let sf = from.as_ref().map(|e| Box::new(simplify_expr(e)));
            let st = to.as_ref().map(|e| Box::new(simplify_expr(e)));
            // Normalize f[from:to] → f | .[from:to] when f is not input
            // Only safe when from/to are literals (don't reference input)
            if !matches!(&se, Expr::Input)
                && sf.as_ref().map_or(true, |e| matches!(e.as_ref(), Expr::Literal(_)))
                && st.as_ref().map_or(true, |e| matches!(e.as_ref(), Expr::Literal(_)))
            {
                return simplify_expr(&Expr::Pipe {
                    left: Box::new(se),
                    right: Box::new(Expr::Slice { expr: Box::new(Expr::Input), from: sf, to: st }),
                });
            }
            Expr::Slice {
                expr: Box::new(se),
                from: sf,
                to: st,
            }
        }
        Expr::Update { path_expr, update_expr } => {
            let sp = simplify_expr(path_expr);
            let su = simplify_expr(update_expr);
            // `path |= empty` deletes the path and yields the modified
            // container exactly once — equivalent to `del(path)`. The
            // generic JIT generator-update branch silently produces zero
            // outputs because the closure is never invoked, so rewrite to
            // `del` at compile time (issue #155).
            if matches!(&su, Expr::Empty) {
                return Expr::CallBuiltin {
                    op: BuiltinOp::Del,
                    args: vec![sp],
                };
            }
            Expr::Update { path_expr: Box::new(sp), update_expr: Box::new(su) }
        }
        Expr::Assign { path_expr, value_expr } => {
            let sp = simplify_expr(path_expr);
            let sv = simplify_expr(value_expr);
            // .field = f(.field) → .field |= f(.) when value only references .field
            if let Expr::Index { expr: base, key } = &sp {
                if matches!(base.as_ref(), Expr::Input) {
                    if let Expr::Literal(crate::ir::Literal::Str(field)) = key.as_ref() {
                        fn replace_field_with_input(e: &Expr, field: &str) -> Option<Expr> {
                            match e {
                                Expr::Index { expr: base, key } if matches!(base.as_ref(), Expr::Input)
                                    && matches!(key.as_ref(), Expr::Literal(crate::ir::Literal::Str(f)) if f == field) => {
                                    Some(Expr::Input)
                                }
                                Expr::BinOp { op, lhs, rhs } => {
                                    let nl = replace_field_with_input(lhs, field);
                                    let nr = replace_field_with_input(rhs, field);
                                    if nl.is_some() || nr.is_some() {
                                        Some(Expr::BinOp {
                                            op: *op,
                                            lhs: Box::new(nl.unwrap_or_else(|| lhs.as_ref().clone())),
                                            rhs: Box::new(nr.unwrap_or_else(|| rhs.as_ref().clone())),
                                        })
                                    } else { None }
                                }
                                Expr::UnaryOp { op, operand } => {
                                    replace_field_with_input(operand, field).map(|o| Expr::UnaryOp { op: *op, operand: Box::new(o) })
                                }
                                Expr::CallBuiltin { op: name, args } => {
                                    let mut any_replaced = false;
                                    let new_args: Vec<_> = args.iter().map(|a| {
                                        if let Some(r) = replace_field_with_input(a, field) { any_replaced = true; r }
                                        else { a.clone() }
                                    }).collect();
                                    if any_replaced { Some(Expr::CallBuiltin { op: *name, args: new_args }) }
                                    else { None }
                                }
                                Expr::RegexTest { input_expr, re, flags } => {
                                    replace_field_with_input(input_expr, field).map(|ie| Expr::RegexTest {
                                        input_expr: Box::new(ie), re: re.clone(), flags: flags.clone()
                                    })
                                }
                                _ => None,
                            }
                        }
                        // Only convert if value_expr ONLY references .field (no other fields or bare Input)
                        fn only_uses_field(e: &Expr, field: &str) -> bool {
                            match e {
                                Expr::Input => false, // bare . reference = other field context
                                Expr::Index { expr: base, key } if matches!(base.as_ref(), Expr::Input) => {
                                    matches!(key.as_ref(), Expr::Literal(crate::ir::Literal::Str(f)) if f == field)
                                }
                                Expr::BinOp { lhs, rhs, .. } => only_uses_field(lhs, field) && only_uses_field(rhs, field),
                                Expr::UnaryOp { operand, .. } => only_uses_field(operand, field),
                                Expr::CallBuiltin { args, .. } => args.iter().all(|a| only_uses_field(a, field)),
                                Expr::RegexTest { input_expr, re, flags } => only_uses_field(input_expr, field) && only_uses_field(re, field) && only_uses_field(flags, field),
                                Expr::Literal(_) => true,
                                _ => false, // unknown expr types = don't optimize
                            }
                        }
                        if only_uses_field(&sv, field) {
                            if let Some(update) = replace_field_with_input(&sv, field) {
                                return Expr::Update {
                                    path_expr: Box::new(sp),
                                    update_expr: Box::new(update),
                                };
                            }
                        }
                    }
                }
            }
            Expr::Assign { path_expr: Box::new(sp), value_expr: Box::new(sv) }
        }
        Expr::TryCatch { try_expr, catch_expr, restore_dot } => {
            Expr::TryCatch { try_expr: Box::new(simplify_expr(try_expr)), catch_expr: Box::new(simplify_expr(catch_expr)), restore_dot: *restore_dot }
        }
        // delpaths([["field"]]) → del(.field)
        Expr::Limit { count, generator } => {
            let sc = simplify_expr(count);
            let sg = simplify_expr(generator);
            // first(scalar_expr) → scalar_expr: if count >= 1 and generator produces exactly 1 output
            if let Expr::Literal(crate::ir::Literal::Num(n, _)) = &sc {
                if *n >= 1.0 {
                    fn is_single_output(e: &Expr) -> bool {
                        match e {
                            Expr::Input | Expr::Literal(_) | Expr::Not => true,
                            Expr::Index { expr, key } => is_single_output(expr) && is_single_output(key),
                            Expr::BinOp { lhs, rhs, .. } => is_single_output(lhs) && is_single_output(rhs),
                            Expr::UnaryOp { operand, .. } => is_single_output(operand),
                            Expr::Negate { operand } => is_single_output(operand),
                            Expr::Pipe { left, right } => is_single_output(left) && is_single_output(right),
                            Expr::IfThenElse { cond, then_branch, else_branch } => {
                                is_single_output(cond) && is_single_output(then_branch) && is_single_output(else_branch)
                            }
                            Expr::LoadVar { .. } => true,
                            Expr::LetBinding { value, body, .. } => is_single_output(value) && is_single_output(body),
                            Expr::CallBuiltin { args, .. } => args.iter().all(|a| is_single_output(a)),
                            Expr::ObjectConstruct { pairs } => pairs.iter().all(|(k, v)| is_single_output(k) && is_single_output(v)),
                            Expr::RegexTest { input_expr, re, flags } => is_single_output(input_expr) && is_single_output(re) && is_single_output(flags),
                            Expr::RegexSub { input_expr, re, tostr, flags } | Expr::RegexGsub { input_expr, re, tostr, flags } => {
                                is_single_output(input_expr) && is_single_output(re) && is_single_output(tostr) && is_single_output(flags)
                            }
                            Expr::Update { path_expr, update_expr } => is_single_output(path_expr) && is_single_output(update_expr),
                            Expr::Assign { path_expr, value_expr } => is_single_output(path_expr) && is_single_output(value_expr),
                            Expr::Mutate { path_expr, value_expr, .. } => is_single_output(path_expr) && is_single_output(value_expr),
                            Expr::Alternative { primary, fallback } => is_single_output(primary) && is_single_output(fallback),
                            _ => false,
                        }
                    }
                    if is_single_output(&sg) {
                        return sg;
                    }
                    // first(a, b, ...) where a is single-output → a. Only valid for
                    // limit(1; ...): for larger counts we must keep the full generator.
                    if *n == 1.0 {
                        let mut g = &sg;
                        while let Expr::Comma { left, .. } = g {
                            if is_single_output(left) {
                                return simplify_expr(left);
                            }
                            g = left;
                        }
                    }
                }
            }
            Expr::Limit { count: Box::new(sc), generator: Box::new(sg) }
        }
        Expr::PathExpr { expr: pe } => {
            let sp = simplify_expr(pe);
            // Note: `path(.field)` cannot be folded to `["field"]` at
            // compile time — jq errors when the input type is not
            // indexable, so the type check must happen at runtime
            // (issue #46). Only comma distributivity is safe to fold.
            if let Expr::Comma { left, right } = &sp {
                let lp = simplify_expr(&Expr::PathExpr { expr: left.clone() });
                let rp = simplify_expr(&Expr::PathExpr { expr: right.clone() });
                return Expr::Comma { left: Box::new(lp), right: Box::new(rp) };
            }
            Expr::PathExpr { expr: Box::new(sp) }
        }
        Expr::GetPath { path } => {
            let sp = simplify_expr(path);
            // getpath(["field"]) → .field
            if let Expr::Collect { generator } = &sp {
                if let Expr::Literal(Literal::Str(field)) = generator.as_ref() {
                    return Expr::Index {
                        expr: Box::new(Expr::Input),
                        key: Box::new(Expr::Literal(Literal::Str(field.clone()))),
                    };
                }
            }
            Expr::GetPath { path: Box::new(sp) }
        }
        Expr::DelPaths { paths } => {
            use crate::ir::Literal;
            if let Expr::Collect { generator } = paths.as_ref() {
                if let Expr::Collect { generator: inner } = generator.as_ref() {
                    if let Expr::Literal(Literal::Str(field)) = inner.as_ref() {
                        return Expr::CallBuiltin {
                            op: BuiltinOp::Del,
                            args: vec![Expr::Index {
                                expr: Box::new(Expr::Input),
                                key: Box::new(Expr::Literal(Literal::Str(field.clone()))),
                            }],
                        };
                    }
                }
            }
            Expr::DelPaths { paths: Box::new(simplify_expr(paths)) }
        }
        _ => expr.clone(),
    }
}

/// Returns true if the expression contains any Expr::Input node (i.e., references `.`).
pub(crate) fn contains_input(expr: &crate::ir::Expr) -> bool {
    use crate::ir::{Expr, StringPart};
    match expr {
        Expr::Input => true,
        Expr::Literal(_) | Expr::Empty | Expr::Env | Expr::Builtins
        | Expr::ReadInput | Expr::ReadInputs | Expr::ModuleMeta
        | Expr::GenLabel | Expr::Loc { .. } | Expr::Break { .. } => false,
        // `not` negates the truthiness of current input
        Expr::Not => true,
        Expr::LoadVar { .. } => false,
        Expr::BinOp { lhs, rhs, .. } => contains_input(lhs) || contains_input(rhs),
        Expr::UnaryOp { operand, .. } | Expr::Negate { operand } => contains_input(operand),
        Expr::Index { expr: e, key } | Expr::IndexOpt { expr: e, key } => contains_input(e) || contains_input(key),
        Expr::Collect { generator } => contains_input(generator),
        Expr::Comma { left, right } => contains_input(left) || contains_input(right),
        Expr::Each { input_expr } | Expr::EachOpt { input_expr } => contains_input(input_expr),
        Expr::Pipe { left, right } => contains_input(left) || contains_input(right),
        Expr::IfThenElse { cond, then_branch, else_branch } => contains_input(cond) || contains_input(then_branch) || contains_input(else_branch),
        Expr::ObjectConstruct { pairs } => pairs.iter().any(|(k, v)| contains_input(k) || contains_input(v)),
        Expr::Alternative { primary, fallback } => contains_input(primary) || contains_input(fallback),
        Expr::Format { expr: e, .. } => contains_input(e),
        Expr::Slice { expr: e, from, to } => contains_input(e) || from.as_ref().map_or(false, |f| contains_input(f)) || to.as_ref().map_or(false, |t| contains_input(t)),
        Expr::StringInterpolation { parts } => parts.iter().any(|p| matches!(p, StringPart::Expr(e) if contains_input(e))),
        Expr::LetBinding { value, body, .. } => contains_input(value) || contains_input(body),
        Expr::Reduce { source, init, update, .. } => contains_input(source) || contains_input(init) || contains_input(update),
        Expr::Foreach { source, init, update, extract, .. } => contains_input(source) || contains_input(init) || contains_input(update) || extract.as_ref().map_or(false, |e| contains_input(e)),
        // `while(cond; update)` / `until(cond; update)` always emit the current
        // value (the input) — jq desugars them to `if cond then ., (update|_loop)
        // …` (while) and `if cond then . else …` (until), both of which yield `.`.
        // So they are input-dependent regardless of whether cond/update mention
        // `.`. Classifying e.g. `while(true; empty)` by its sub-exprs marked it
        // input-free, so the fast path evaluated it against `null` and emitted
        // `null` instead of echoing the input. Same family as `recurse` (#713)
        // and the assignment forms (#716) below.
        Expr::While { .. } | Expr::Until { .. } => true,
        Expr::Repeat { update } => contains_input(update),
        Expr::TryCatch { try_expr, catch_expr, .. } => contains_input(try_expr) || contains_input(catch_expr),
        // CallBuiltin implicitly operates on the current input (passed as first arg)
        Expr::CallBuiltin { .. } => true,
        Expr::Range { from, to, step } => contains_input(from) || contains_input(to) || step.as_ref().map_or(false, |s| contains_input(s)),
        Expr::Limit { count, generator } => contains_input(count) || contains_input(generator),
        Expr::Error { msg } => msg.as_ref().map_or(false, |m| contains_input(m)),
        // Assignment forms (`p = v`, `p |= f`, `p += v`, …) always read the
        // current input as the document being updated and return it (possibly
        // modified), regardless of whether the path/value sub-exprs mention
        // `.`. Classifying e.g. `empty = 9` by its sub-exprs marked it
        // input-free, so the fast path evaluated it against `null` and emitted
        // `null` instead of echoing the input unchanged (#716). Same family as
        // the SetPath/GetPath/PathExpr arm below and eval.rs's #556 fix.
        Expr::Update { .. } | Expr::Assign { .. } | Expr::Mutate { .. } => true,
        // GetPath/SetPath/DelPaths/PathExpr implicitly operate on the current input
        Expr::GetPath { .. } | Expr::SetPath { .. } | Expr::DelPaths { .. } | Expr::PathExpr { .. } => true,
        // `recurse(f)` always emits the input value first (`def recurse(f):
        // ., (f | recurse(f))`), so it is input-dependent regardless of `f`.
        // Classifying by the body alone wrongly marked `recurse(empty)` (and
        // any input-free `f`) as input-free, evaluating it against `null` and
        // emitting `null` instead of the input (#713).
        Expr::Recurse { .. } => true,
        // debug/stderr pass through the current input
        Expr::Debug { .. } | Expr::Stderr { .. } => true,
        Expr::Label { body, .. } => contains_input(body),
        // Conservative: assume these reference input
        Expr::RegexTest { input_expr, .. } | Expr::RegexMatch { input_expr, .. }
        | Expr::RegexCapture { input_expr, .. } | Expr::RegexScan { input_expr, .. }
        | Expr::RegexSub { input_expr, .. } | Expr::RegexGsub { input_expr, .. } => contains_input(input_expr),
        Expr::FuncCall { args, .. } => args.iter().any(contains_input),
        Expr::ClosureOp { .. } | Expr::AnyShort { .. } | Expr::AllShort { .. }
        | Expr::AlternativeDestructure { .. } => true, // conservative
        Expr::Memoize { key, body, .. } => {
            key.as_ref().is_some_and(|k| contains_input(k)) || contains_input(body)
        }
    }
}

/// Returns true when at least one `Expr::Input` reference inside `e`
/// is reached only through a short-circuiting wrapper — currently
/// `Alternative.fallback`, `TryCatch.try_expr`, or `TryCatch.catch_expr`.
/// Beta-substituting a side-effecting (or error-producing) replacement
/// into one of those positions can silently elide a runtime error,
/// because the wrapper's semantics deliberately re-route around errors
/// in the inner expression (#354 family).
///
/// The `Alternative.primary` slot is *not* short-circuiting — it always
/// evaluates first — so Input there is safe to substitute.
fn input_behind_short_circuit(e: &crate::ir::Expr) -> bool {
    use crate::ir::Expr;
    match e {
        Expr::Alternative { primary, fallback } => {
            input_behind_short_circuit(primary) || contains_input(fallback)
                || input_behind_short_circuit(fallback)
        }
        Expr::TryCatch { try_expr, catch_expr, .. } => {
            // try_expr's errors are caught; the post-substitution
            // result no longer raises. catch_expr only fires on error;
            // its evaluation order is conditional.
            contains_input(try_expr) || contains_input(catch_expr)
                || input_behind_short_circuit(try_expr)
                || input_behind_short_circuit(catch_expr)
        }
        Expr::Pipe { left, right } => {
            input_behind_short_circuit(left) || input_behind_short_circuit(right)
        }
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            input_behind_short_circuit(cond)
                || input_behind_short_circuit(then_branch)
                || input_behind_short_circuit(else_branch)
        }
        Expr::BinOp { op, lhs, rhs } => {
            // jq's `and` / `or` short-circuit on the lhs's truthiness:
            // `0 or X` returns true without evaluating X, `false and X`
            // returns false without evaluating X. Substituting an
            // input-side expression (with potential errors / side
            // effects) into the rhs of an `and` / `or` therefore lets
            // the simplifier silently elide that evaluation when the
            // lhs's value is statically determinable. Treat any Input
            // reference inside the rhs as behind a short-circuit so
            // the Pipe-substitution at `simplify_expr` line ~1281
            // refuses (#375, sibling of the Alternative.fallback /
            // TryCatch.* guards #354).
            use crate::ir::BinOp;
            if matches!(op, BinOp::And | BinOp::Or) {
                input_behind_short_circuit(lhs)
                    || contains_input(rhs)
                    || input_behind_short_circuit(rhs)
            } else {
                input_behind_short_circuit(lhs) || input_behind_short_circuit(rhs)
            }
        }
        Expr::UnaryOp { operand, .. } | Expr::Negate { operand } => input_behind_short_circuit(operand),
        Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => {
            input_behind_short_circuit(expr) || input_behind_short_circuit(key)
        }
        Expr::Comma { left, right } => {
            input_behind_short_circuit(left) || input_behind_short_circuit(right)
        }
        Expr::Each { input_expr } | Expr::EachOpt { input_expr } => input_behind_short_circuit(input_expr),
        Expr::ObjectConstruct { pairs } => {
            pairs.iter().any(|(k, v)| input_behind_short_circuit(k) || input_behind_short_circuit(v))
        }
        Expr::Collect { generator } => input_behind_short_circuit(generator),
        Expr::Format { expr, .. } => input_behind_short_circuit(expr),
        Expr::Slice { expr, from, to } => {
            input_behind_short_circuit(expr)
                || from.as_ref().is_some_and(|e| input_behind_short_circuit(e))
                || to.as_ref().is_some_and(|e| input_behind_short_circuit(e))
        }
        Expr::StringInterpolation { parts } => parts.iter().any(|p| match p {
            crate::ir::StringPart::Expr(e) => input_behind_short_circuit(e),
            _ => false,
        }),
        Expr::LetBinding { value, body, .. } => {
            input_behind_short_circuit(value) || input_behind_short_circuit(body)
        }
        // Conservative leaf: no Input here, or no short-circuit.
        _ => false,
    }
}

/// Conservative check that an expression yields exactly one output for
/// any input — the *cardinality* sibling of [`contains_input`]. A
/// `false` answer is always safe; a `true` answer means the simplifier
/// can rely on single-output semantics (e.g. when folding
/// `{pairs} | length` to a numeric literal, or `first(g)` to `g`).
///
/// Generators (`,`, `range`, `..`, `each`, `empty`, `foreach`/`reduce`
/// streams) and anything that recursively contains one return `false`.
fn expr_is_single_output(e: &crate::ir::Expr) -> bool {
    use crate::ir::Expr;
    match e {
        Expr::Input | Expr::Literal(_) | Expr::Not | Expr::LoadVar { .. } => true,
        Expr::Index { expr, key } => expr_is_single_output(expr) && expr_is_single_output(key),
        Expr::BinOp { lhs, rhs, .. } => expr_is_single_output(lhs) && expr_is_single_output(rhs),
        Expr::UnaryOp { operand, .. } | Expr::Negate { operand } => expr_is_single_output(operand),
        Expr::Pipe { left, right } => expr_is_single_output(left) && expr_is_single_output(right),
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            expr_is_single_output(cond) && expr_is_single_output(then_branch) && expr_is_single_output(else_branch)
        }
        Expr::LetBinding { value, body, .. } => expr_is_single_output(value) && expr_is_single_output(body),
        Expr::CallBuiltin { args, .. } => args.iter().all(expr_is_single_output),
        Expr::ObjectConstruct { pairs } => pairs.iter().all(|(k, v)| expr_is_single_output(k) && expr_is_single_output(v)),
        Expr::RegexTest { input_expr, re, flags } => {
            expr_is_single_output(input_expr) && expr_is_single_output(re) && expr_is_single_output(flags)
        }
        Expr::RegexSub { input_expr, re, tostr, flags } | Expr::RegexGsub { input_expr, re, tostr, flags } => {
            expr_is_single_output(input_expr) && expr_is_single_output(re) && expr_is_single_output(tostr) && expr_is_single_output(flags)
        }
        Expr::Update { path_expr, update_expr } => expr_is_single_output(path_expr) && expr_is_single_output(update_expr),
        Expr::Assign { path_expr, value_expr } => expr_is_single_output(path_expr) && expr_is_single_output(value_expr),
        Expr::Mutate { path_expr, value_expr, .. } => expr_is_single_output(path_expr) && expr_is_single_output(value_expr),
        Expr::Alternative { primary, fallback } => expr_is_single_output(primary) && expr_is_single_output(fallback),
        _ => false,
    }
}
