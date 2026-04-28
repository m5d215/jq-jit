//! Coverage for the [`jq_jit::fast_path::FastPath`] contract introduced
//! in #83. Verifies that:
//!
//! - the pilot `FieldAccessPath` succeeds on object and null inputs,
//! - the pilot bails with `None` on every other input type (so the
//!   generic path can raise the jq-compatible "Cannot index …" error
//!   instead of leaking null-masking divergence #50),
//! - `Filter::try_typed_fast_path` routes `.field` filters through the
//!   pilot and returns `None` for filters that aren't yet migrated.

use jq_jit::fast_path::{
    FastPath, FieldAccessPath, RawApplyOutcome, apply_arith_chain_cmp_raw, apply_field_access_raw,
    apply_field_alternative_raw, apply_field_arith_chain_raw, apply_field_binop_raw,
    apply_field_case_gsub_raw, apply_field_case_test_raw, apply_field_const_cmp_raw,
    apply_field_field_alternative_raw, apply_field_field_cmp_raw,
    apply_field_format_raw, apply_field_gsub_raw, apply_field_ltrimstr_tonumber_raw,
    apply_field_match_raw, apply_field_scan_raw, apply_field_str_builtin_raw,
    apply_field_str_concat_raw, apply_field_str_reverse_raw, apply_field_test_raw,
    apply_full_object_fields_raw, apply_has_field_raw, apply_has_multi_field_raw,
    apply_multi_field_access_raw, apply_nested_field_access_raw, apply_object_compute_raw,
    apply_del_field_raw, apply_del_fields_raw, apply_field_update_case_raw,
    apply_field_update_gsub_raw, apply_field_update_length_raw, apply_field_update_slice_raw,
    apply_field_update_split_first_raw, apply_field_update_split_last_raw,
    apply_field_update_str_concat_raw, apply_field_update_str_map_raw,
    apply_field_update_test_raw, apply_field_update_tostring_raw,
    apply_compound_field_cmp_raw,
    apply_field_binop_const_unary_raw, apply_field_index_arith_raw,
    apply_field_unary_arith_raw, apply_field_unary_num_raw,
    apply_field_update_trim_raw, apply_is_length_raw, apply_is_type_raw,
    apply_numeric_expr_raw, apply_obj_assign_field_arith_raw,
    apply_collect_each_select_type_raw, apply_each_type_filter_raw,
    apply_first_each_select_type_raw,
    apply_field_cmp_val_raw, apply_null_branch_lit_raw,
    apply_obj_assign_two_fields_arith_raw, apply_obj_merge_computed_raw,
    apply_obj_merge_lit_raw, apply_select_nested_cmp_raw, apply_select_num_str_raw,
    apply_select_arith_cmp_raw, apply_select_cmp_raw,
    apply_select_field_null_raw, apply_select_str_raw, apply_select_str_test_raw,
    apply_to_entries_each_interp_raw, apply_two_field_binop_const_raw,
};
use jq_jit::interpreter::{ArithExpr, CmpVal, Filter, MathUnary};
use jq_jit::ir::{BinOp, UnaryOp};
use jq_jit::value::Value;

#[test]
fn field_access_on_object_hits() {
    let path = FieldAccessPath::new("x");
    let obj = Value::from_pairs(vec![
        ("x".to_string(), Value::from_f64(42.0)),
        ("y".to_string(), Value::from_f64(7.0)),
    ]);
    let out = path.run(&obj).expect("must produce a verdict").expect("ok");
    match out {
        Value::Num(n, _) => assert_eq!(n, 42.0),
        _ => panic!("expected Num, got {:?}", out),
    }
}

#[test]
fn field_access_on_object_missing_key_returns_null() {
    let path = FieldAccessPath::new("absent");
    let obj = Value::from_pairs(vec![("present".to_string(), Value::from_f64(1.0))]);
    let out = path.run(&obj).expect("must produce a verdict").expect("ok");
    assert!(matches!(out, Value::Null), "expected null, got {:?}", out);
}

#[test]
fn field_access_on_null_returns_null() {
    // jq semantics: `.x` on null is null.
    let path = FieldAccessPath::new("x");
    let out = path.run(&Value::Null).expect("must produce a verdict").expect("ok");
    assert!(matches!(out, Value::Null), "expected null, got {:?}", out);
}

#[test]
fn field_access_on_non_object_bails_to_generic() {
    // Every non-object non-null input must return None so the generic
    // path can raise the correct type-error. This is the #50 /
    // null-masking divergence the typed contract exists to prevent.
    for input in [
        Value::from_bool(true),
        Value::from_bool(false),
        Value::from_f64(1.0),
        Value::from_str("hello"),
        Value::Arr(std::rc::Rc::new(vec![Value::from_f64(1.0)])),
    ] {
        let path = FieldAccessPath::new("x");
        let verdict = path.run(&input);
        assert!(
            verdict.is_none(),
            "FieldAccessPath must bail on non-object non-null input, got {:?} for input {:?}",
            verdict, input,
        );
    }
}

#[test]
fn filter_try_typed_fast_path_routes_field_access() {
    let f = Filter::new(".x").expect("parse");
    let obj = Value::from_pairs(vec![("x".to_string(), Value::from_f64(99.0))]);
    let verdict = f.try_typed_fast_path(&obj).expect("must route");
    let v = verdict.expect("ok");
    match v {
        Value::Num(n, _) => assert_eq!(n, 99.0),
        _ => panic!("expected Num, got {:?}", v),
    }
}

#[test]
fn filter_try_typed_fast_path_bails_on_non_object() {
    // `.x` on a non-object non-null input must not short-circuit — the
    // contract defers to the generic path which raises the correct error.
    let f = Filter::new(".x").expect("parse");
    let verdict = f.try_typed_fast_path(&Value::from_bool(true));
    assert!(verdict.is_none(), "expected bail to generic, got {:?}", verdict);
}

#[test]
fn filter_try_typed_fast_path_returns_none_for_unmigrated_filter() {
    // A filter shape that the pilot doesn't recognise must return None
    // so the existing raw-byte / eval / jit dispatch stays in charge.
    let f = Filter::new("[.x, .y] | add").expect("parse");
    let obj = Value::from_pairs(vec![
        ("x".to_string(), Value::from_f64(1.0)),
        ("y".to_string(), Value::from_f64(2.0)),
    ]);
    let verdict = f.try_typed_fast_path(&obj);
    assert!(verdict.is_none(), "pilot should decline unmigrated shapes, got {:?}", verdict);
}

// ---------------------------------------------------------------------------
// Integration coverage — `Filter::execute` / `execute_cb` routing.
//
// The tests above pin the pilot's verdict surface in isolation. These ones
// close the loop: the typed fast path is only useful once it actually fires
// from the public dispatch. If `execute` is ever refactored and forgets to
// probe `try_typed_fast_path`, these break before a compat bug ships.

#[test]
fn execute_field_access_on_object_returns_value() {
    let f = Filter::new(".x").expect("parse");
    let obj = Value::from_pairs(vec![("x".to_string(), Value::from_f64(99.0))]);
    let out = f.execute(&obj).expect("ok");
    assert_eq!(out.len(), 1, "expected single value, got {:?}", out);
    match &out[0] {
        Value::Num(n, _) => assert_eq!(*n, 99.0),
        v => panic!("expected Num, got {:?}", v),
    }
}

#[test]
fn execute_field_access_on_null_returns_null() {
    let f = Filter::new(".x").expect("parse");
    let out = f.execute(&Value::Null).expect("ok");
    assert_eq!(out, vec![Value::Null]);
}

#[test]
fn execute_field_access_on_non_object_falls_through_to_generic() {
    // The typed fast path bails with `None` on boolean input, which is its
    // contract: it MUST NOT short-circuit with `Some(Ok(Value::Null))` or it
    // re-introduces the null-masking bug class (#50). Here we only confirm
    // that `execute` does not panic or return the typed verdict directly —
    // the generic eval / jit path takes over. The generic path's compat with
    // jq (whether it raises "Cannot index ...") is tracked separately as
    // part of #83's broader migration and is not asserted here.
    let f = Filter::new(".x").expect("parse");
    let _ = f.execute(&Value::from_bool(true));
}

#[test]
fn execute_cb_field_access_invokes_callback_once() {
    let f = Filter::new(".x").expect("parse");
    let obj = Value::from_pairs(vec![("x".to_string(), Value::from_f64(7.0))]);
    let mut seen: Vec<Value> = Vec::new();
    let all = f.execute_cb(&obj, &mut |v| {
        seen.push(v.clone());
        Ok(true)
    }).expect("ok");
    assert!(all, "callback must report full completion");
    assert_eq!(seen.len(), 1, "typed path should emit exactly one value");
    match &seen[0] {
        Value::Num(n, _) => assert_eq!(*n, 7.0),
        v => panic!("expected Num, got {:?}", v),
    }
}

// ---------------------------------------------------------------------------
// Raw-byte apply-site contract (#83 Phase B)
//
// The CLI keeps its raw-byte fast paths for perf (parsing JSON into Value is
// 2.5× slower on the `.name` benchmark — see #83's revert table). Phase B's
// contribution is making the bail decision structurally explicit at the
// apply-site instead of hiding it inside an inline `match raw[0]` arm.
// `apply_field_access_raw` is the pilot; these tests pin the verdict surface
// so a future regression that lets a non-object input slip through (the #50
// null-masking class) fails CI immediately.

#[test]
fn raw_field_access_object_emits_value_bytes() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_access_raw(b"{\"x\":42,\"y\":7}", "x", |b| emitted.push(b.to_vec()));
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"42".to_vec()]);
}

#[test]
fn raw_field_access_object_missing_key_emits_null_literal() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_access_raw(b"{\"y\":7}", "x", |b| emitted.push(b.to_vec()));
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"null".to_vec()]);
}

#[test]
fn raw_field_access_null_input_emits_null_literal() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_access_raw(b"null", "x", |b| emitted.push(b.to_vec()));
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"null".to_vec()]);
}

#[test]
fn raw_field_access_non_object_non_null_bails() {
    // The whole point of #83 Phase B: every non-{object,null} input must
    // return Bail so the caller routes to `process_input` →
    // `Filter::execute_cb`. Emitting `null` (or anything else) here would
    // re-introduce the null-masking divergence (#50).
    for raw in [
        b"42".as_slice(),
        b"\"hello\"".as_slice(),
        b"true".as_slice(),
        b"false".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_field_access_raw(raw, "x", |b| emitted.push(b.to_vec()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(
            emitted.is_empty(),
            "Bail must not emit anything (got {:?} for input {:?})",
            emitted,
            std::str::from_utf8(raw).unwrap(),
        );
    }
}

// ---------------------------------------------------------------------------
// Nested field access (`.a.b.c`) — same Bail discipline as the single-field
// pilot, with one extra wrinkle: when the path doesn't resolve, the helper
// can't tell apart a missing key (jq → null) from a non-object intermediate
// (jq → type error). Both cases bail to the generic path.

#[test]
fn raw_nested_field_full_path_emits_value_bytes() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_nested_field_access_raw(
        b"{\"a\":{\"b\":42}}",
        &["a", "b"],
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"42".to_vec()]);
}

#[test]
fn raw_nested_field_null_input_emits_null_literal() {
    // jq: `.a.b` on null is null. Emit the literal directly.
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_nested_field_access_raw(b"null", &["a", "b"], |b| emitted.push(b.to_vec()));
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"null".to_vec()]);
}

#[test]
fn raw_nested_field_missing_key_bails() {
    // Top-level key missing — jq produces null, but the raw scanner can't
    // distinguish this from "intermediate is non-object" (which jq errors on),
    // so it bails and the generic path picks the right semantics.
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_nested_field_access_raw(b"{}", &["a", "b"], |bytes| {
        emitted.push(bytes.to_vec())
    });
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_nested_field_non_object_intermediate_bails() {
    // `.a.b` on `{"a":"hello"}` — intermediate is string, jq raises a
    // type error. The helper bails so the generic path raises it.
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_nested_field_access_raw(
        b"{\"a\":\"hello\"}",
        &["a", "b"],
        |bytes| emitted.push(bytes.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_nested_field_non_object_non_null_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hello\"".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_nested_field_access_raw(raw, &["a", "b"], |bytes| {
            emitted.push(bytes.to_vec())
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// Multi-field comma access (`.a, .b, .c`) — fires only on objects that contain
// every requested field. Anything else bails (null input, partial object,
// non-object) so jq's per-field semantics (`null` for null input, mix of
// values/nulls for partial objects, type error for non-object non-null) come
// from the generic path.

#[test]
fn raw_multi_field_complete_object_emits_each_field() {
    let mut ranges_buf = vec![(0usize, 0usize); 3];
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_multi_field_access_raw(
        b"{\"a\":1,\"b\":2,\"c\":3}",
        &["a", "b", "c"],
        &mut ranges_buf,
        |bytes| emitted.push(bytes.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"1".to_vec(), b"2".to_vec(), b"3".to_vec()]);
}

#[test]
fn raw_multi_field_partial_object_bails() {
    // jq emits a mix of values and nulls; the raw scanner is all-or-nothing,
    // so the helper bails to the generic path.
    let mut ranges_buf = vec![(0usize, 0usize); 3];
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_multi_field_access_raw(
        b"{\"a\":1,\"c\":3}",
        &["a", "b", "c"],
        &mut ranges_buf,
        |bytes| emitted.push(bytes.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_multi_field_null_input_bails() {
    // jq emits one `null` per field; the raw scanner bails so the generic
    // path produces the per-field nulls.
    let mut ranges_buf = vec![(0usize, 0usize); 2];
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_multi_field_access_raw(
        b"null",
        &["a", "b"],
        &mut ranges_buf,
        |bytes| emitted.push(bytes.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_multi_field_non_object_non_null_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hello\"".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut ranges_buf = vec![(0usize, 0usize); 2];
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_multi_field_access_raw(
            raw,
            &["a", "b"],
            &mut ranges_buf,
            |bytes| emitted.push(bytes.to_vec()),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// Array-of-fields collect (`[.a, .b, .c]`) — same Bail shape as multi-field
// (object with every field present, otherwise Bail). Serialisation is left to
// the caller's closure so pretty / compact framing stays at the apply-site.

#[test]
fn raw_full_object_fields_complete_object_invokes_emit_once() {
    let mut ranges_buf = vec![(0usize, 0usize); 3];
    let mut calls = 0usize;
    let mut collected: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_full_object_fields_raw(
        b"{\"a\":1,\"b\":2,\"c\":3}",
        &["a", "b", "c"],
        &mut ranges_buf,
        |ranges, raw| {
            calls += 1;
            for (vs, ve) in ranges {
                collected.push(raw[*vs..*ve].to_vec());
            }
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(calls, 1, "emit_array must be invoked exactly once");
    assert_eq!(collected, vec![b"1".to_vec(), b"2".to_vec(), b"3".to_vec()]);
}

#[test]
fn raw_full_object_fields_partial_object_bails_without_calling_emit() {
    let mut ranges_buf = vec![(0usize, 0usize); 3];
    let mut calls = 0usize;
    let outcome = apply_full_object_fields_raw(
        b"{\"a\":1,\"c\":3}",
        &["a", "b", "c"],
        &mut ranges_buf,
        |_, _| { calls += 1; },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(calls, 0, "Bail must not invoke emit_array");
}

#[test]
fn raw_full_object_fields_null_input_bails_without_calling_emit() {
    let mut ranges_buf = vec![(0usize, 0usize); 2];
    let mut calls = 0usize;
    let outcome = apply_full_object_fields_raw(
        b"null",
        &["a", "b"],
        &mut ranges_buf,
        |_, _| { calls += 1; },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(calls, 0);
}

#[test]
fn raw_full_object_fields_non_object_non_null_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hello\"".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut ranges_buf = vec![(0usize, 0usize); 2];
        let mut calls = 0usize;
        let outcome = apply_full_object_fields_raw(
            raw,
            &["a", "b"],
            &mut ranges_buf,
            |_, _| { calls += 1; },
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert_eq!(calls, 0);
    }
}

// ---------------------------------------------------------------------------
// has("x") / has("x") and|or has("y") — emit a literal `true` / `false` for
// object inputs; bail to generic for any non-object input so jq's
// type-sensitive `has(IDX)` semantics (arrays accept indices, others error)
// surface correctly.

#[test]
fn raw_has_field_present_emits_true() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_has_field_raw(b"{\"x\":1,\"y\":2}", "x", |b| emitted.push(b.to_vec()));
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"true".to_vec()]);
}

#[test]
fn raw_has_field_absent_emits_false() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_has_field_raw(b"{\"y\":2}", "x", |b| emitted.push(b.to_vec()));
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"false".to_vec()]);
}

#[test]
fn raw_has_field_non_object_input_bails() {
    for raw in [
        b"null".as_slice(),
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_has_field_raw(raw, "x", |b| emitted.push(b.to_vec()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

#[test]
fn raw_has_multi_field_and_returns_true_when_all_present() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_has_multi_field_raw(
        b"{\"a\":1,\"b\":2,\"c\":3}",
        &["a", "b"],
        true,
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"true".to_vec()]);
}

#[test]
fn raw_has_multi_field_and_returns_false_when_one_missing() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_has_multi_field_raw(
        b"{\"a\":1}",
        &["a", "b"],
        true,
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"false".to_vec()]);
}

#[test]
fn raw_has_multi_field_or_returns_true_when_any_present() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_has_multi_field_raw(
        b"{\"a\":1}",
        &["a", "b"],
        false,
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"true".to_vec()]);
}

#[test]
fn raw_has_multi_field_or_returns_false_when_all_missing() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_has_multi_field_raw(
        b"{\"c\":3}",
        &["a", "b"],
        false,
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"false".to_vec()]);
}

#[test]
fn raw_has_multi_field_non_object_input_bails() {
    for is_and in [true, false] {
        for raw in [
            b"null".as_slice(),
            b"42".as_slice(),
            b"\"hi\"".as_slice(),
            b"[1,2,3]".as_slice(),
        ] {
            let mut emitted: Vec<Vec<u8>> = Vec::new();
            let outcome =
                apply_has_multi_field_raw(raw, &["a", "b"], is_and, |b| emitted.push(b.to_vec()));
            assert!(
                matches!(outcome, RawApplyOutcome::Bail),
                "expected Bail for input {:?} (is_and={}), got {:?}",
                std::str::from_utf8(raw).unwrap(),
                is_and,
                outcome,
            );
            assert!(emitted.is_empty());
        }
    }
}

// ---------------------------------------------------------------------------
// Object-compute (`{a: .x + 1, b: .y * 2}` / `[.x + 1, .y * 2]` / standalone-
// array variants). The helper exposes a two-stage Bail: outer for non-object
// or partially-missing inputs, inner for the per-cell type-error check (#163,
// e.g. `str + num`). Both must invoke neither `bail_check` nor `emit` once
// they decide to bail.

#[test]
fn raw_object_compute_full_object_invokes_emit_after_inner_check() {
    let mut ranges_buf = vec![(0usize, 0usize); 2];
    let mut bail_calls = 0usize;
    let mut emit_calls = 0usize;
    let outcome = apply_object_compute_raw(
        b"{\"a\":1,\"b\":2}",
        &["a", "b"],
        &mut ranges_buf,
        |_, _| { bail_calls += 1; false }, // inner check passes
        |_, _| { emit_calls += 1; },
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(bail_calls, 1, "bail_check must run exactly once on outer success");
    assert_eq!(emit_calls, 1, "emit must run exactly once when both checks pass");
}

#[test]
fn raw_object_compute_outer_bail_skips_inner_and_emit() {
    // Non-object input — outer bail at fields_raw_buf, neither closure runs.
    let mut ranges_buf = vec![(0usize, 0usize); 2];
    let mut bail_calls = 0usize;
    let mut emit_calls = 0usize;
    let outcome = apply_object_compute_raw(
        b"42",
        &["a", "b"],
        &mut ranges_buf,
        |_, _| { bail_calls += 1; false },
        |_, _| { emit_calls += 1; },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(bail_calls, 0, "outer bail must skip the inner check");
    assert_eq!(emit_calls, 0, "outer bail must skip emit");
}

#[test]
fn raw_object_compute_inner_bail_skips_emit() {
    // Outer succeeds, inner says "would error" — bail without emitting.
    let mut ranges_buf = vec![(0usize, 0usize); 2];
    let mut emit_calls = 0usize;
    let outcome = apply_object_compute_raw(
        b"{\"a\":1,\"b\":2}",
        &["a", "b"],
        &mut ranges_buf,
        |_, _| true, // inner says bail
        |_, _| { emit_calls += 1; },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(emit_calls, 0, "inner bail must skip emit");
}

#[test]
fn raw_object_compute_partial_object_outer_bails() {
    // Object missing one field — `json_object_get_fields_raw_buf` returns
    // false, so it's an outer bail (jq emits a mix of values and nulls,
    // generic produces it).
    let mut ranges_buf = vec![(0usize, 0usize); 2];
    let mut bail_calls = 0usize;
    let mut emit_calls = 0usize;
    let outcome = apply_object_compute_raw(
        b"{\"a\":1}",
        &["a", "b"],
        &mut ranges_buf,
        |_, _| { bail_calls += 1; false },
        |_, _| { emit_calls += 1; },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(bail_calls, 0);
    assert_eq!(emit_calls, 0);
}

// ---------------------------------------------------------------------------
// `.field // fallback` and `.field // .other` — `//` MUST NOT swallow type
// errors (#198). Both helpers Bail for non-object non-null inputs so the
// generic path raises `Cannot index ... with string`. On object / null they
// emit per jq's `//` falsey gate (`null` / `false` -> fallback).

#[test]
fn raw_field_alt_present_truthy_emits_value() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_alternative_raw(
        b"{\"x\":42}",
        "x",
        b"\"fallback\"",
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"42".to_vec()]);
}

#[test]
fn raw_field_alt_present_falsey_emits_fallback() {
    for falsey in [&b"null"[..], &b"false"[..]] {
        let raw = format!("{{\"x\":{}}}", std::str::from_utf8(falsey).unwrap());
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_field_alternative_raw(
            raw.as_bytes(),
            "x",
            b"99",
            |b| emitted.push(b.to_vec()),
        );
        assert!(matches!(outcome, RawApplyOutcome::Emit));
        assert_eq!(
            emitted,
            vec![b"99".to_vec()],
            "input {} (field is falsey) must emit fallback",
            raw,
        );
    }
}

#[test]
fn raw_field_alt_missing_emits_fallback() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_alternative_raw(
        b"{\"y\":1}",
        "x",
        b"\"missing\"",
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"\"missing\"".to_vec()]);
}

#[test]
fn raw_field_alt_null_input_emits_fallback() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_alternative_raw(
        b"null",
        "x",
        b"77",
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"77".to_vec()]);
}

#[test]
fn raw_field_alt_non_object_non_null_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_field_alternative_raw(raw, "x", b"99", |b| emitted.push(b.to_vec()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

#[test]
fn raw_field_field_alt_primary_truthy_emits_primary() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_field_alternative_raw(
        b"{\"a\":1,\"b\":2}",
        "a",
        "b",
        |bytes| emitted.push(bytes.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"1".to_vec()]);
}

#[test]
fn raw_field_field_alt_primary_falsey_emits_fallback_value() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_field_alternative_raw(
        b"{\"a\":null,\"b\":2}",
        "a",
        "b",
        |bytes| emitted.push(bytes.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"2".to_vec()]);
}

#[test]
fn raw_field_field_alt_both_missing_emits_null_literal() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_field_alternative_raw(
        b"{\"c\":3}",
        "a",
        "b",
        |bytes| emitted.push(bytes.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"null".to_vec()]);
}

#[test]
fn raw_field_field_alt_non_object_non_null_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome =
            apply_field_field_alternative_raw(raw, "a", "b", |b| emitted.push(b.to_vec()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `.field | test("pattern")` — raw scanner only handles quoted strings with
// no backslash escapes; everything else bails. The helper hands the regex
// match verdict (`true`/`false`) back through the emit closure.

#[test]
fn raw_field_test_match_emits_true() {
    let re = regex::Regex::new(r"^foo").unwrap();
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_test_raw(
        b"{\"x\":\"foobar\"}",
        "x",
        &re,
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"true".to_vec()]);
}

#[test]
fn raw_field_test_no_match_emits_false() {
    let re = regex::Regex::new(r"^foo").unwrap();
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_test_raw(
        b"{\"x\":\"barfoo\"}",
        "x",
        &re,
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"false".to_vec()]);
}

#[test]
fn raw_field_test_field_missing_bails() {
    let re = regex::Regex::new(r"^foo").unwrap();
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_test_raw(
        b"{\"y\":\"hi\"}",
        "x",
        &re,
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_test_non_string_field_bails() {
    let re = regex::Regex::new(r"^foo").unwrap();
    for inner in [&b"{\"x\":42}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1,2]}"[..]] {
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_field_test_raw(inner, "x", &re, |b| emitted.push(b.to_vec()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-string field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

#[test]
fn raw_field_test_escaped_string_bails() {
    // Backslash escapes need decoding; the raw scanner can't, so it bails.
    let re = regex::Regex::new(r"\n").unwrap();
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_test_raw(
        br#"{"x":"a\nb"}"#,
        "x",
        &re,
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_test_non_object_input_bails() {
    let re = regex::Regex::new(r".*").unwrap();
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_field_test_raw(raw, "x", &re, |b| emitted.push(b.to_vec()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `.field | gsub("p"; "r")` / `sub` — same Bail discipline as `field_test`,
// with the helper handing the regex replacement result back as `&str` so the
// apply-site owns JSON-escape framing.

#[test]
fn raw_field_gsub_global_replaces_all_matches() {
    let re = regex::Regex::new("a").unwrap();
    let mut emitted: Vec<String> = Vec::new();
    let outcome = apply_field_gsub_raw(
        b"{\"x\":\"banana\"}",
        "x",
        &re,
        "X",
        true,
        |s| emitted.push(s.to_string()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec!["bXnXnX".to_string()]);
}

#[test]
fn raw_field_gsub_non_global_replaces_first_only() {
    let re = regex::Regex::new("a").unwrap();
    let mut emitted: Vec<String> = Vec::new();
    let outcome = apply_field_gsub_raw(
        b"{\"x\":\"banana\"}",
        "x",
        &re,
        "X",
        false,
        |s| emitted.push(s.to_string()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec!["bXnana".to_string()]);
}

#[test]
fn raw_field_gsub_no_match_emits_unchanged() {
    let re = regex::Regex::new("z").unwrap();
    let mut emitted: Vec<String> = Vec::new();
    let outcome = apply_field_gsub_raw(
        b"{\"x\":\"banana\"}",
        "x",
        &re,
        "X",
        true,
        |s| emitted.push(s.to_string()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec!["banana".to_string()]);
}

#[test]
fn raw_field_gsub_field_missing_or_non_string_bails() {
    let re = regex::Regex::new("a").unwrap();
    for inner in [
        &b"{\"y\":\"hi\"}"[..],
        &b"{\"x\":42}"[..],
        &b"{\"x\":null}"[..],
        &b"{\"x\":[1,2,3]}"[..],
    ] {
        let mut emitted: Vec<String> = Vec::new();
        let outcome =
            apply_field_gsub_raw(inner, "x", &re, "X", true, |s| emitted.push(s.to_string()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

#[test]
fn raw_field_gsub_escaped_string_bails() {
    let re = regex::Regex::new("a").unwrap();
    let mut emitted: Vec<String> = Vec::new();
    let outcome = apply_field_gsub_raw(
        br#"{"x":"a\nb"}"#,
        "x",
        &re,
        "X",
        true,
        |s| emitted.push(s.to_string()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_gsub_non_object_input_bails() {
    let re = regex::Regex::new(".*").unwrap();
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<String> = Vec::new();
        let outcome =
            apply_field_gsub_raw(raw, "x", &re, "X", true, |s| emitted.push(s.to_string()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `.field | ascii_downcase|ascii_upcase | test("p")` — case fold then regex
// test. Same Bail discipline as `apply_field_test_raw`; the case fold is
// byte-wise ASCII over the un-decoded string content.

#[test]
fn raw_field_case_test_downcase_match_emits_true() {
    let re = regex::Regex::new("^foo").unwrap();
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_case_test_raw(
        b"{\"x\":\"FOObar\"}",
        "x",
        false,
        &re,
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"true".to_vec()]);
}

#[test]
fn raw_field_case_test_upcase_match_emits_true() {
    let re = regex::Regex::new("^FOO").unwrap();
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_case_test_raw(
        b"{\"x\":\"foobar\"}",
        "x",
        true,
        &re,
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"true".to_vec()]);
}

#[test]
fn raw_field_case_test_no_match_emits_false() {
    let re = regex::Regex::new("^foo").unwrap();
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_case_test_raw(
        b"{\"x\":\"BARbaz\"}",
        "x",
        false,
        &re,
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"false".to_vec()]);
}

#[test]
fn raw_field_case_test_field_missing_bails() {
    let re = regex::Regex::new("^foo").unwrap();
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_case_test_raw(
        b"{\"y\":\"FOO\"}",
        "x",
        false,
        &re,
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_case_test_non_string_field_bails() {
    let re = regex::Regex::new(".*").unwrap();
    for inner in [&b"{\"x\":42}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1,2]}"[..]] {
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome =
            apply_field_case_test_raw(inner, "x", false, &re, |b| emitted.push(b.to_vec()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-string field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

#[test]
fn raw_field_case_test_escaped_string_bails() {
    let re = regex::Regex::new(".*").unwrap();
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_case_test_raw(
        br#"{"x":"a\nb"}"#,
        "x",
        false,
        &re,
        |b| emitted.push(b.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_case_test_non_object_input_bails() {
    let re = regex::Regex::new(".*").unwrap();
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome =
            apply_field_case_test_raw(raw, "x", false, &re, |b| emitted.push(b.to_vec()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `.field | ascii_downcase|ascii_upcase | gsub/sub("p"; "r")` — same Bail
// discipline as `apply_field_gsub_raw`; case fold runs first, then regex
// replacement on the folded bytes.

#[test]
fn raw_field_case_gsub_downcase_global_replaces_all() {
    let re = regex::Regex::new("a").unwrap();
    let mut emitted: Vec<String> = Vec::new();
    let outcome = apply_field_case_gsub_raw(
        b"{\"x\":\"BAnAnA\"}",
        "x",
        false,
        &re,
        "X",
        true,
        |s| emitted.push(s.to_string()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec!["bXnXnX".to_string()]);
}

#[test]
fn raw_field_case_gsub_upcase_first_match_only() {
    let re = regex::Regex::new("A").unwrap();
    let mut emitted: Vec<String> = Vec::new();
    let outcome = apply_field_case_gsub_raw(
        b"{\"x\":\"banana\"}",
        "x",
        true,
        &re,
        "X",
        false,
        |s| emitted.push(s.to_string()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec!["BXNANA".to_string()]);
}

#[test]
fn raw_field_case_gsub_no_match_emits_folded_string() {
    let re = regex::Regex::new("z").unwrap();
    let mut emitted: Vec<String> = Vec::new();
    let outcome = apply_field_case_gsub_raw(
        b"{\"x\":\"BANANA\"}",
        "x",
        false,
        &re,
        "X",
        true,
        |s| emitted.push(s.to_string()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec!["banana".to_string()]);
}

#[test]
fn raw_field_case_gsub_field_missing_or_non_string_bails() {
    let re = regex::Regex::new("a").unwrap();
    for inner in [
        &b"{\"y\":\"hi\"}"[..],
        &b"{\"x\":42}"[..],
        &b"{\"x\":null}"[..],
        &b"{\"x\":[1,2,3]}"[..],
    ] {
        let mut emitted: Vec<String> = Vec::new();
        let outcome = apply_field_case_gsub_raw(
            inner, "x", false, &re, "X", true, |s| emitted.push(s.to_string()),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

#[test]
fn raw_field_case_gsub_escaped_string_bails() {
    let re = regex::Regex::new("a").unwrap();
    let mut emitted: Vec<String> = Vec::new();
    let outcome = apply_field_case_gsub_raw(
        br#"{"x":"a\nb"}"#,
        "x",
        false,
        &re,
        "X",
        true,
        |s| emitted.push(s.to_string()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_case_gsub_non_object_input_bails() {
    let re = regex::Regex::new(".*").unwrap();
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<String> = Vec::new();
        let outcome = apply_field_case_gsub_raw(
            raw, "x", false, &re, "X", true, |s| emitted.push(s.to_string()),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `.field | match("p")` — same Bail discipline as `field_test`. The helper
// hands `(content_str, captures)` back to the apply-site so it can build jq's
// `{offset,length,string,captures:[...]}` output bytes; on no match the
// closure is not invoked (jq's `match` emits no output for a non-match).

#[test]
fn raw_field_match_invokes_closure_on_match() {
    let re = regex::Regex::new(r"f(o+)").unwrap();
    let mut hits: Vec<(String, String)> = Vec::new();
    let outcome = apply_field_match_raw(
        b"{\"x\":\"foobar\"}",
        "x",
        &re,
        |content, caps| {
            let m = caps.get(0).unwrap();
            let g1 = caps.get(1).unwrap();
            hits.push((content.to_string(), format!("{}|{}", m.as_str(), g1.as_str())));
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(hits, vec![("foobar".to_string(), "foo|oo".to_string())]);
}

#[test]
fn raw_field_match_no_match_skips_closure() {
    let re = regex::Regex::new(r"^foo").unwrap();
    let mut called = 0u32;
    let outcome = apply_field_match_raw(
        b"{\"x\":\"barfoo\"}",
        "x",
        &re,
        |_, _| {
            called += 1;
        },
    );
    // Emit verdict (the helper handled the input — jq emits nothing on no
    // match), but the closure is never invoked.
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(called, 0);
}

#[test]
fn raw_field_match_field_missing_bails() {
    let re = regex::Regex::new(r"^foo").unwrap();
    let mut called = 0u32;
    let outcome = apply_field_match_raw(
        b"{\"y\":\"hi\"}",
        "x",
        &re,
        |_, _| {
            called += 1;
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(called, 0);
}

#[test]
fn raw_field_match_non_string_field_bails() {
    let re = regex::Regex::new(r".").unwrap();
    for inner in [&b"{\"x\":42}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1,2]}"[..]] {
        let mut called = 0u32;
        let outcome = apply_field_match_raw(inner, "x", &re, |_, _| {
            called += 1;
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-string field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert_eq!(called, 0);
    }
}

#[test]
fn raw_field_match_escaped_string_bails() {
    // Backslash escapes need decoding; the raw scanner can't, so it bails.
    let re = regex::Regex::new(r"\n").unwrap();
    let mut called = 0u32;
    let outcome = apply_field_match_raw(
        br#"{"x":"a\nb"}"#,
        "x",
        &re,
        |_, _| {
            called += 1;
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(called, 0);
}

#[test]
fn raw_field_match_non_object_input_bails() {
    let re = regex::Regex::new(r".*").unwrap();
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut called = 0u32;
        let outcome = apply_field_match_raw(raw, "x", &re, |_, _| {
            called += 1;
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert_eq!(called, 0);
    }
}

// ---------------------------------------------------------------------------
// `.field | scan("p")` — same Bail discipline as `field_match`. The helper
// iterates `re.captures_iter` and invokes the closure once per non-overlapping
// match (zero invocations for no-match-on-string, still Emit verdict).

#[test]
fn raw_field_scan_emits_one_per_match() {
    let re = regex::Regex::new("a").unwrap();
    let mut hits: Vec<String> = Vec::new();
    let outcome = apply_field_scan_raw(
        b"{\"x\":\"banana\"}",
        "x",
        &re,
        |_content, caps| {
            hits.push(caps.get(0).unwrap().as_str().to_string());
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(hits, vec!["a".to_string(), "a".to_string(), "a".to_string()]);
}

#[test]
fn raw_field_scan_with_capture_groups_yields_all_captures() {
    // Optional group simulates jq's "unmatched group → null" semantic; the
    // helper just hands each `Captures` to the apply-site so it can decide.
    let re = regex::Regex::new("(z)?(a)").unwrap();
    let mut hits: Vec<(Option<String>, Option<String>)> = Vec::new();
    let outcome = apply_field_scan_raw(
        b"{\"x\":\"banana\"}",
        "x",
        &re,
        |_content, caps| {
            hits.push((
                caps.get(1).map(|m| m.as_str().to_string()),
                caps.get(2).map(|m| m.as_str().to_string()),
            ));
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(
        hits,
        vec![
            (None, Some("a".to_string())),
            (None, Some("a".to_string())),
            (None, Some("a".to_string())),
        ],
    );
}

#[test]
fn raw_field_scan_no_match_skips_closure() {
    let re = regex::Regex::new("z").unwrap();
    let mut called = 0u32;
    let outcome = apply_field_scan_raw(
        b"{\"x\":\"banana\"}",
        "x",
        &re,
        |_, _| {
            called += 1;
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(called, 0);
}

#[test]
fn raw_field_scan_field_missing_bails() {
    let re = regex::Regex::new("a").unwrap();
    let mut called = 0u32;
    let outcome = apply_field_scan_raw(
        b"{\"y\":\"banana\"}",
        "x",
        &re,
        |_, _| {
            called += 1;
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(called, 0);
}

#[test]
fn raw_field_scan_non_string_field_bails() {
    let re = regex::Regex::new(r".").unwrap();
    for inner in [&b"{\"x\":42}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1,2]}"[..]] {
        let mut called = 0u32;
        let outcome = apply_field_scan_raw(inner, "x", &re, |_, _| {
            called += 1;
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-string field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert_eq!(called, 0);
    }
}

#[test]
fn raw_field_scan_escaped_string_bails() {
    let re = regex::Regex::new("a").unwrap();
    let mut called = 0u32;
    let outcome = apply_field_scan_raw(
        br#"{"x":"a\nb"}"#,
        "x",
        &re,
        |_, _| {
            called += 1;
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(called, 0);
}

#[test]
fn raw_field_scan_non_object_input_bails() {
    let re = regex::Regex::new(r".*").unwrap();
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut called = 0u32;
        let outcome = apply_field_scan_raw(raw, "x", &re, |_, _| {
            called += 1;
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert_eq!(called, 0);
    }
}

// ---------------------------------------------------------------------------
// `.field | @<format>` — the helper handles only the *structural* type-guard
// (field present + non-escape-bearing quoted string check). The
// format-specific Emit-vs-Bail decision is delegated to the closure via its
// returned `RawApplyOutcome`. These tests pin the structural surface.

#[test]
fn raw_field_format_string_value_emits() {
    let mut seen: Vec<(Vec<u8>, Option<Vec<u8>>)> = Vec::new();
    let outcome = apply_field_format_raw(
        b"{\"x\":\"hi\"}",
        "x",
        |val, content| {
            seen.push((val.to_vec(), content.map(|c| c.to_vec())));
            RawApplyOutcome::Emit
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(
        seen,
        vec![(b"\"hi\"".to_vec(), Some(b"hi".to_vec()))],
    );
}

#[test]
fn raw_field_format_non_string_scalar_invokes_with_none() {
    // The helper hands the closure `(val, None)` for non-string scalars; the
    // closure decides whether to Emit (e.g. @json wraps raw bytes) or Bail
    // (e.g. @base64 raises on non-string).
    for (input, expected_val) in [
        (&b"{\"x\":42}"[..], &b"42"[..]),
        (&b"{\"x\":null}"[..], &b"null"[..]),
        (&b"{\"x\":true}"[..], &b"true"[..]),
    ] {
        let mut seen: Vec<(Vec<u8>, Option<Vec<u8>>)> = Vec::new();
        let outcome = apply_field_format_raw(input, "x", |val, content| {
            seen.push((val.to_vec(), content.map(|c| c.to_vec())));
            RawApplyOutcome::Emit
        });
        assert!(matches!(outcome, RawApplyOutcome::Emit));
        assert_eq!(seen, vec![(expected_val.to_vec(), None)]);
    }
}

#[test]
fn raw_field_format_propagates_closure_bail_verdict() {
    // The closure can choose to Bail even on a structurally-OK input
    // (this is what `@base64`/`@uri`/`@html` do for non-string values).
    let outcome = apply_field_format_raw(
        b"{\"x\":42}",
        "x",
        |_, _| RawApplyOutcome::Bail,
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_field_format_field_missing_bails_without_invoking_closure() {
    let mut called = 0u32;
    let outcome = apply_field_format_raw(
        b"{\"y\":\"hi\"}",
        "x",
        |_, _| {
            called += 1;
            RawApplyOutcome::Emit
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(called, 0);
}

#[test]
fn raw_field_format_escaped_string_bails_without_invoking_closure() {
    let mut called = 0u32;
    let outcome = apply_field_format_raw(
        br#"{"x":"a\nb"}"#,
        "x",
        |_, _| {
            called += 1;
            RawApplyOutcome::Emit
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(called, 0);
}

#[test]
fn raw_field_format_non_object_input_bails_without_invoking_closure() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut called = 0u32;
        let outcome = apply_field_format_raw(raw, "x", |_, _| {
            called += 1;
            RawApplyOutcome::Emit
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for format input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert_eq!(called, 0);
    }
}

// ---------------------------------------------------------------------------
// `.field | ltrimstr("p") | tonumber [ | <arith> ]*` — same Bail discipline
// as the regex helpers (object input + quoted-string field with no escapes),
// plus a numeric-parse Bail (#160) so the generic path produces jq's
// "Invalid numeric literal" error on non-numeric strings.

#[test]
fn raw_field_ltrimstr_tonumber_strips_prefix_and_parses() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_ltrimstr_tonumber_raw(
        b"{\"x\":\"id-42\"}",
        "x",
        b"id-",
        &[],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![42.0]);
}

#[test]
fn raw_field_ltrimstr_tonumber_no_prefix_match_keeps_content() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_ltrimstr_tonumber_raw(
        b"{\"x\":\"42\"}",
        "x",
        b"id-",
        &[],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![42.0]);
}

#[test]
fn raw_field_ltrimstr_tonumber_applies_arith_chain() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_ltrimstr_tonumber_raw(
        b"{\"x\":\"10\"}",
        "x",
        b"",
        &[(BinOp::Add, 5.0), (BinOp::Mul, 2.0)],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![30.0]);
}

#[test]
fn raw_field_ltrimstr_tonumber_non_numeric_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_ltrimstr_tonumber_raw(
        b"{\"x\":\"abc\"}",
        "x",
        b"",
        &[],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_ltrimstr_tonumber_field_missing_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_ltrimstr_tonumber_raw(
        b"{\"y\":\"42\"}",
        "x",
        b"",
        &[],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_ltrimstr_tonumber_non_string_field_bails() {
    for inner in [&b"{\"x\":42}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1,2]}"[..]] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_ltrimstr_tonumber_raw(inner, "x", b"", &[], |n| emitted.push(n));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-string field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

#[test]
fn raw_field_ltrimstr_tonumber_escaped_string_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_ltrimstr_tonumber_raw(
        br#"{"x":"4\n2"}"#,
        "x",
        b"",
        &[],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_ltrimstr_tonumber_non_object_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_ltrimstr_tonumber_raw(raw, "x", b"", &[], |n| emitted.push(n));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for ltrimstr_tonumber input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `.field + "<literal-suffix>"` — the helper hands the closure
// `Some(content)` for a present-and-valid string, `None` for a missing
// field (jq's `null + "s" == "s"`), and Bails on every other shape.

#[test]
fn raw_field_str_concat_present_string_yields_content() {
    let mut seen: Vec<Option<Vec<u8>>> = Vec::new();
    let outcome = apply_field_str_concat_raw(
        b"{\"x\":\"hi\"}",
        "x",
        |c| seen.push(c.map(|s| s.to_vec())),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(seen, vec![Some(b"hi".to_vec())]);
}

#[test]
fn raw_field_str_concat_missing_field_yields_none() {
    let mut seen: Vec<Option<Vec<u8>>> = Vec::new();
    let outcome = apply_field_str_concat_raw(
        b"{\"y\":\"hi\"}",
        "x",
        |c| seen.push(c.map(|s| s.to_vec())),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(seen, vec![None]);
}

#[test]
fn raw_field_str_concat_non_string_field_bails() {
    for inner in [&b"{\"x\":42}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1,2]}"[..]] {
        let mut called = 0u32;
        let outcome = apply_field_str_concat_raw(inner, "x", |_| {
            called += 1;
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-string field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert_eq!(called, 0);
    }
}

#[test]
fn raw_field_str_concat_escaped_string_bails() {
    let mut called = 0u32;
    let outcome = apply_field_str_concat_raw(
        br#"{"x":"a\nb"}"#,
        "x",
        |_| {
            called += 1;
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(called, 0);
}

#[test]
fn raw_field_str_concat_non_object_input_bails() {
    // Includes `null`, even though `null + "s" == "s"` in jq — the helper
    // delegates that fold to the generic path.
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut called = 0u32;
        let outcome = apply_field_str_concat_raw(raw, "x", |_| {
            called += 1;
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for str_concat input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert_eq!(called, 0);
    }
}

// ---------------------------------------------------------------------------
// `.field | split("") | reverse | join("")` — structural-only Bail at the
// helper boundary; the closure receives the raw inner bytes (still possibly
// containing JSON escapes) and decides between ASCII-fast and decode-rebuild
// strategies, returning its own `RawApplyOutcome` for failures (e.g. invalid
// UTF-8 after escape decode).

#[test]
fn raw_field_str_reverse_hands_inner_bytes_to_closure() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_str_reverse_raw(
        b"{\"x\":\"hello\"}",
        "x",
        |inner| {
            seen.push(inner.to_vec());
            RawApplyOutcome::Emit
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(seen, vec![b"hello".to_vec()]);
}

#[test]
fn raw_field_str_reverse_propagates_closure_bail_verdict() {
    // Closure may signal a per-input Bail — e.g. the apply-site cannot
    // decode an escape-bearing string; the helper must propagate without
    // emitting anything.
    let outcome = apply_field_str_reverse_raw(
        br#"{"x":"a\u00ZZ"}"#,
        "x",
        |_| RawApplyOutcome::Bail,
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_field_str_reverse_field_missing_bails_without_invoking_closure() {
    let mut called = 0u32;
    let outcome = apply_field_str_reverse_raw(
        b"{\"y\":\"hi\"}",
        "x",
        |_| {
            called += 1;
            RawApplyOutcome::Emit
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(called, 0);
}

#[test]
fn raw_field_str_reverse_non_string_field_bails_without_invoking_closure() {
    for inner in [&b"{\"x\":42}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1,2]}"[..]] {
        let mut called = 0u32;
        let outcome = apply_field_str_reverse_raw(inner, "x", |_| {
            called += 1;
            RawApplyOutcome::Emit
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-string field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert_eq!(called, 0);
    }
}

#[test]
fn raw_field_str_reverse_non_object_input_bails_without_invoking_closure() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut called = 0u32;
        let outcome = apply_field_str_reverse_raw(raw, "x", |_| {
            called += 1;
            RawApplyOutcome::Emit
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for str_reverse input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert_eq!(called, 0);
    }
}

// ---------------------------------------------------------------------------
// `.field | <string-builtin>(arg)` — the helper enforces the structural
// type-guard (object input + non-escape quoted-string field) and lets the
// closure dispatch on builtin name. The closure can return Bail for an
// unknown name (defensive fallback).

#[test]
fn raw_field_str_builtin_invokes_closure_with_inner_bytes() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_field_str_builtin_raw(
        b"{\"x\":\"banana\"}",
        "x",
        |content| {
            seen.push(content.to_vec());
            RawApplyOutcome::Emit
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(seen, vec![b"banana".to_vec()]);
}

#[test]
fn raw_field_str_builtin_propagates_closure_bail_verdict() {
    // The closure can choose to Bail (e.g. unknown builtin name fallback).
    let outcome = apply_field_str_builtin_raw(
        b"{\"x\":\"banana\"}",
        "x",
        |_| RawApplyOutcome::Bail,
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_field_str_builtin_field_missing_bails() {
    let mut called = 0u32;
    let outcome = apply_field_str_builtin_raw(
        b"{\"y\":\"hi\"}",
        "x",
        |_| {
            called += 1;
            RawApplyOutcome::Emit
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(called, 0);
}

#[test]
fn raw_field_str_builtin_non_string_field_bails() {
    for inner in [&b"{\"x\":42}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1,2]}"[..]] {
        let mut called = 0u32;
        let outcome = apply_field_str_builtin_raw(inner, "x", |_| {
            called += 1;
            RawApplyOutcome::Emit
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-string field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert_eq!(called, 0);
    }
}

#[test]
fn raw_field_str_builtin_escaped_string_bails() {
    let mut called = 0u32;
    let outcome = apply_field_str_builtin_raw(
        br#"{"x":"a\nb"}"#,
        "x",
        |_| {
            called += 1;
            RawApplyOutcome::Emit
        },
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(called, 0);
}

#[test]
fn raw_field_str_builtin_non_object_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut called = 0u32;
        let outcome = apply_field_str_builtin_raw(raw, "x", |_| {
            called += 1;
            RawApplyOutcome::Emit
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for str_builtin input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert_eq!(called, 0);
    }
}

// ---------------------------------------------------------------------------
// `.x <op> .y` — both fields must be JSON numbers; the helper computes the
// arithmetic and Bails on missing/non-numeric fields, on a non-arithmetic op
// (defensive), and on a non-finite result (so the generic path raises jq's
// `/ by zero` error).

#[test]
fn raw_field_binop_add_emits_sum() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_binop_raw(
        b"{\"x\":3,\"y\":4}",
        "x",
        "y",
        BinOp::Add,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![7.0]);
}

#[test]
fn raw_field_binop_handles_all_arith_ops() {
    for (op, expected) in [
        (BinOp::Add, 13.0),
        (BinOp::Sub, 7.0),
        (BinOp::Mul, 30.0),
        (BinOp::Div, 10.0 / 3.0),
        (BinOp::Mod, 1.0),
    ] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_binop_raw(
            b"{\"x\":10,\"y\":3}",
            "x",
            "y",
            op,
            |n| emitted.push(n),
        );
        assert!(matches!(outcome, RawApplyOutcome::Emit));
        assert_eq!(emitted, vec![expected], "op={:?}", op);
    }
}

#[test]
fn raw_field_binop_div_by_zero_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_binop_raw(
        b"{\"x\":1,\"y\":0}",
        "x",
        "y",
        BinOp::Div,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_binop_non_arith_op_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_binop_raw(
        b"{\"x\":1,\"y\":2}",
        "x",
        "y",
        BinOp::Eq,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_binop_field_missing_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_binop_raw(
        b"{\"x\":1}",
        "x",
        "y",
        BinOp::Add,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_binop_non_numeric_field_bails() {
    for inner in [
        &b"{\"x\":\"hi\",\"y\":1}"[..],
        &b"{\"x\":1,\"y\":null}"[..],
        &b"{\"x\":[1],\"y\":2}"[..],
    ] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_binop_raw(inner, "x", "y", BinOp::Add, |n| emitted.push(n));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-numeric field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

#[test]
fn raw_field_binop_non_object_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_binop_raw(raw, "x", "y", BinOp::Add, |n| emitted.push(n));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for binop input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `.field <op1> <c1> <op2> <c2> ...` — left-fold of `(BinOp, f64)` pairs over a
// single numeric field. The detector excludes div/mod-by-zero constants at
// compile time, so the helper trusts the chain's divisors are non-zero.

#[test]
fn raw_field_arith_chain_folds_ops_left_to_right() {
    // ((2 + 3) * 4) - 1 == 19
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_arith_chain_raw(
        b"{\"x\":2}",
        "x",
        &[(BinOp::Add, 3.0), (BinOp::Mul, 4.0), (BinOp::Sub, 1.0)],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![19.0]);
}

#[test]
fn raw_field_arith_chain_empty_ops_emits_input() {
    // Edge case: zero-length chain; helper should pass through.
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_arith_chain_raw(
        b"{\"x\":7}",
        "x",
        &[],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![7.0]);
}

#[test]
fn raw_field_arith_chain_field_missing_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_arith_chain_raw(
        b"{\"y\":1}",
        "x",
        &[(BinOp::Add, 1.0)],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_arith_chain_non_numeric_field_bails() {
    for inner in [
        &b"{\"x\":\"hi\"}"[..],
        &b"{\"x\":null}"[..],
        &b"{\"x\":[1,2]}"[..],
    ] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_arith_chain_raw(inner, "x", &[(BinOp::Add, 1.0)], |n| {
            emitted.push(n)
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-numeric field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

#[test]
fn raw_field_arith_chain_non_arith_op_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_arith_chain_raw(
        b"{\"x\":1}",
        "x",
        &[(BinOp::Eq, 1.0)],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_arith_chain_non_object_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_arith_chain_raw(raw, "x", &[(BinOp::Add, 1.0)], |n| {
            emitted.push(n)
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for arith_chain input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `(.x <op1> .y) <op2> <const>` — two-field-and-const arithmetic. Helper
// Bails on missing / non-numeric / non-arith op / non-finite result /
// non-object so cross-type and runtime errors route through the generic
// path.

#[test]
fn raw_two_field_binop_const_emits_finite_result() {
    let mut emitted: Vec<f64> = Vec::new();
    // (3 + 4) * 2 = 14
    let outcome = apply_two_field_binop_const_raw(
        b"{\"x\":3,\"y\":4}",
        "x", "y", BinOp::Add, BinOp::Mul, 2.0,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![14.0]);
}

#[test]
fn raw_two_field_binop_const_field_missing_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_two_field_binop_const_raw(
        b"{\"x\":3}",
        "x", "y", BinOp::Add, BinOp::Mul, 2.0,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_two_field_binop_const_non_numeric_field_bails() {
    for inner in [&b"{\"x\":3,\"y\":\"hi\"}"[..], &b"{\"x\":null,\"y\":4}"[..]] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_two_field_binop_const_raw(
            inner, "x", "y", BinOp::Add, BinOp::Mul, 2.0, |n| emitted.push(n),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-numeric input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

#[test]
fn raw_two_field_binop_const_non_arith_op_bails() {
    for op in [BinOp::Eq, BinOp::Lt, BinOp::And] {
        let mut emitted: Vec<f64> = Vec::new();
        // op1 is non-arith
        let outcome = apply_two_field_binop_const_raw(
            b"{\"x\":3,\"y\":4}",
            "x", "y", op, BinOp::Mul, 2.0, |n| emitted.push(n),
        );
        assert!(matches!(outcome, RawApplyOutcome::Bail), "op1={:?}", op);
        // op2 is non-arith
        let mut emitted2: Vec<f64> = Vec::new();
        let outcome2 = apply_two_field_binop_const_raw(
            b"{\"x\":3,\"y\":4}",
            "x", "y", BinOp::Add, op, 2.0, |n| emitted2.push(n),
        );
        assert!(matches!(outcome2, RawApplyOutcome::Bail), "op2={:?}", op);
    }
}

#[test]
fn raw_two_field_binop_const_div_by_zero_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    // (1 / 0) * 2 = inf → Bail
    let outcome = apply_two_field_binop_const_raw(
        b"{\"x\":1,\"y\":0}",
        "x", "y", BinOp::Div, BinOp::Mul, 2.0,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_two_field_binop_const_inner_finite_outer_div_zero_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    // (3 + 4) / 0 = inf → Bail
    let outcome = apply_two_field_binop_const_raw(
        b"{\"x\":3,\"y\":4}",
        "x", "y", BinOp::Add, BinOp::Div, 0.0,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_two_field_binop_const_non_object_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_two_field_binop_const_raw(
            raw, "x", "y", BinOp::Add, BinOp::Mul, 2.0, |n| emitted.push(n),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for two_field_binop_const input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `.field <op> <const> [| unary]` — single-field-with-const arithmetic,
// optional final unary math op. Helper Bails on missing / non-numeric /
// non-arith op / unsupported unary / non-finite / non-object so jq's runtime
// errors (div-by-zero, type errors) surface through the generic path.

#[test]
fn raw_field_binop_const_unary_no_unary() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_binop_const_unary_raw(
        b"{\"x\":5}", "x", BinOp::Add, 3.0, None, false, |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![8.0]);
}

#[test]
fn raw_field_binop_const_unary_const_left_subtraction() {
    let mut emitted: Vec<f64> = Vec::new();
    // 2 - .x with x=5 → -3, then abs → 3
    let outcome = apply_field_binop_const_unary_raw(
        b"{\"x\":5}", "x", BinOp::Sub, 2.0, Some(UnaryOp::Abs), true,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![3.0]);
}

#[test]
fn raw_field_binop_const_unary_floor_ceil() {
    let mut floored: Vec<f64> = Vec::new();
    let outcome = apply_field_binop_const_unary_raw(
        b"{\"x\":3.7}", "x", BinOp::Add, 0.0, Some(UnaryOp::Floor), false,
        |n| floored.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(floored, vec![3.0]);

    let mut ceilinged: Vec<f64> = Vec::new();
    let outcome = apply_field_binop_const_unary_raw(
        b"{\"x\":3.2}", "x", BinOp::Add, 0.0, Some(UnaryOp::Ceil), false,
        |n| ceilinged.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(ceilinged, vec![4.0]);
}

#[test]
fn raw_field_binop_const_unary_field_missing_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_binop_const_unary_raw(
        b"{\"y\":3}", "x", BinOp::Add, 1.0, None, false, |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_binop_const_unary_non_numeric_field_bails() {
    for inner in [&b"{\"x\":\"hi\"}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1]}"[..]] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_binop_const_unary_raw(
            inner, "x", BinOp::Add, 1.0, None, false, |n| emitted.push(n),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-numeric input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

#[test]
fn raw_field_binop_const_unary_non_arith_op_bails() {
    for op in [BinOp::Eq, BinOp::Lt, BinOp::And] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_binop_const_unary_raw(
            b"{\"x\":5}", "x", op, 3.0, None, false, |n| emitted.push(n),
        );
        assert!(matches!(outcome, RawApplyOutcome::Bail), "op={:?}", op);
    }
}

#[test]
fn raw_field_binop_const_unary_unsupported_unary_bails() {
    // Detector should never produce these for this shape, but defensive Bail.
    for uop in [UnaryOp::Length, UnaryOp::ToString, UnaryOp::Type] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_binop_const_unary_raw(
            b"{\"x\":5}", "x", BinOp::Add, 3.0, Some(uop), false,
            |n| emitted.push(n),
        );
        assert!(matches!(outcome, RawApplyOutcome::Bail), "uop={:?}", uop);
    }
}

#[test]
fn raw_field_binop_const_unary_div_by_zero_bails() {
    // .x / 0 → inf → Bail (fixes #83-class divergence: prior fast path
    // emitted "1.7976931348623157e+308" instead of jq's "/ by zero" error).
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_binop_const_unary_raw(
        b"{\"x\":4}", "x", BinOp::Div, 0.0, None, false, |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_binop_const_unary_sqrt_negative_bails() {
    // (.x + 0) | sqrt with x=-1 → NaN → Bail (generic produces null,
    // matching jq).
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_binop_const_unary_raw(
        b"{\"x\":-1}", "x", BinOp::Add, 0.0, Some(UnaryOp::Sqrt), false,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_binop_const_unary_non_object_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_binop_const_unary_raw(
            raw, "x", BinOp::Add, 1.0, None, false, |n| emitted.push(n),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `. + {k: v}` — object-merge-literal. Thin wrapper around
// `json_object_merge_literal`; Bails on non-object input.

#[test]
fn raw_obj_merge_lit_appends_new_key() {
    let mut buf = Vec::new();
    let pairs = vec![("y".to_string(), b"42".to_vec())];
    let outcome = apply_obj_merge_lit_raw(b"{\"x\":1}", &pairs, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"{\"x\":1,\"y\":42}");
}

#[test]
fn raw_obj_merge_lit_replaces_existing_key() {
    let mut buf = Vec::new();
    let pairs = vec![("x".to_string(), b"99".to_vec())];
    let outcome = apply_obj_merge_lit_raw(b"{\"x\":1,\"y\":2}", &pairs, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"{\"x\":99,\"y\":2}");
}

#[test]
fn raw_obj_merge_lit_non_object_bails() {
    for raw in [b"42".as_slice(), b"\"hi\"".as_slice(), b"null".as_slice(), b"[1]".as_slice()] {
        let mut buf = Vec::new();
        let pairs = vec![("y".to_string(), b"42".to_vec())];
        let outcome = apply_obj_merge_lit_raw(raw, &pairs, &mut buf);
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "raw={:?}", std::str::from_utf8(raw).unwrap(),
        );
    }
}

// ---------------------------------------------------------------------------
// `. + {key: <arith>}` — object-merge-computed. Reads numeric fields,
// evaluates the ArithExpr, formats as JSON-number bytes, merges. Bails on
// non-object/missing-or-non-numeric/non-finite/buffer-shape mismatch.

#[test]
fn raw_obj_merge_computed_emits_merged() {
    // . + {sum: .x + .y} on {"x":3,"y":4} → {"x":3,"y":4,"sum":7}
    let arith = ArithExpr::BinOp(
        BinOp::Add,
        Box::new(ArithExpr::Field(0)),
        Box::new(ArithExpr::Field(1)),
    );
    let nfields = ["x", "y"];
    let mut vals = vec![0.0; 2];
    let mut merge_pair = vec![("sum".to_string(), Vec::new())];
    let mut buf = Vec::new();
    let outcome = apply_obj_merge_computed_raw(
        b"{\"x\":3,\"y\":4}", &nfields, &arith, &mut vals, &mut merge_pair, &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"{\"x\":3,\"y\":4,\"sum\":7}");
}

#[test]
fn raw_obj_merge_computed_missing_field_bails() {
    let arith = ArithExpr::Field(0);
    let nfields = ["x"];
    let mut vals = vec![0.0; 1];
    let mut merge_pair = vec![("dbl".to_string(), Vec::new())];
    let mut buf = Vec::new();
    let outcome = apply_obj_merge_computed_raw(
        b"{\"y\":1}", &nfields, &arith, &mut vals, &mut merge_pair, &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_obj_merge_computed_div_by_zero_bails() {
    let arith = ArithExpr::BinOp(
        BinOp::Div,
        Box::new(ArithExpr::Field(0)),
        Box::new(ArithExpr::Field(1)),
    );
    let nfields = ["x", "y"];
    let mut vals = vec![0.0; 2];
    let mut merge_pair = vec![("q".to_string(), Vec::new())];
    let mut buf = Vec::new();
    let outcome = apply_obj_merge_computed_raw(
        b"{\"x\":1,\"y\":0}", &nfields, &arith, &mut vals, &mut merge_pair, &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_obj_merge_computed_non_object_bails() {
    let arith = ArithExpr::Field(0);
    let nfields = ["x"];
    for raw in [b"42".as_slice(), b"null".as_slice(), b"[1]".as_slice()] {
        let mut vals = vec![0.0; 1];
        let mut merge_pair = vec![("dbl".to_string(), Vec::new())];
        let mut buf = Vec::new();
        let outcome = apply_obj_merge_computed_raw(
            raw, &nfields, &arith, &mut vals, &mut merge_pair, &mut buf,
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "raw={:?}", std::str::from_utf8(raw).unwrap(),
        );
    }
}

#[test]
fn raw_obj_merge_computed_buffer_shape_mismatch_bails() {
    let arith = ArithExpr::Field(0);
    let nfields = ["x"];
    let mut vals = vec![0.0; 1];
    // merge_pair_buf must have exactly 1 element
    let mut merge_pair: Vec<(String, Vec<u8>)> = vec![];
    let mut buf = Vec::new();
    let outcome = apply_obj_merge_computed_raw(
        b"{\"x\":3}", &nfields, &arith, &mut vals, &mut merge_pair, &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

// ---------------------------------------------------------------------------
// `.dest = (.src op N)` — object-assign single-field arithmetic. Thin
// wrapper around `json_object_assign_field_arith`. Bails on missing/non-
// numeric src, non-finite result, or non-arith op.

#[test]
fn raw_obj_assign_field_arith_emits_rewritten_object() {
    // .y = (.x + 1) on {"x":3,"y":0} → {"x":3,"y":4}
    let mut buf = Vec::new();
    let outcome = apply_obj_assign_field_arith_raw(
        b"{\"x\":3,\"y\":0}", "y", "x", BinOp::Add, 1.0, &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"{\"x\":3,\"y\":4}");
}

#[test]
fn raw_obj_assign_field_arith_missing_src_bails() {
    let mut buf = Vec::new();
    let outcome = apply_obj_assign_field_arith_raw(
        b"{\"y\":0}", "y", "x", BinOp::Add, 1.0, &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_obj_assign_field_arith_non_numeric_src_bails() {
    let mut buf = Vec::new();
    let outcome = apply_obj_assign_field_arith_raw(
        b"{\"x\":\"hi\",\"y\":0}", "y", "x", BinOp::Add, 1.0, &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_obj_assign_field_arith_div_by_zero_bails() {
    let mut buf = Vec::new();
    let outcome = apply_obj_assign_field_arith_raw(
        b"{\"x\":3,\"y\":0}", "y", "x", BinOp::Div, 0.0, &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_obj_assign_field_arith_non_arith_op_bails() {
    for op in [BinOp::Eq, BinOp::Lt, BinOp::And] {
        let mut buf = Vec::new();
        let outcome = apply_obj_assign_field_arith_raw(
            b"{\"x\":3,\"y\":0}", "y", "x", op, 1.0, &mut buf,
        );
        assert!(matches!(outcome, RawApplyOutcome::Bail), "op={:?}", op);
    }
}

#[test]
fn raw_obj_assign_field_arith_non_object_input_bails() {
    for raw in [b"42".as_slice(), b"\"hi\"".as_slice(), b"null".as_slice(), b"[1,2]".as_slice()] {
        let mut buf = Vec::new();
        let outcome = apply_obj_assign_field_arith_raw(
            raw, "y", "x", BinOp::Add, 1.0, &mut buf,
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "raw={:?}", std::str::from_utf8(raw).unwrap(),
        );
    }
}

// ---------------------------------------------------------------------------
// `.dest = (.src1 op .src2)` — object-assign two-field arithmetic.

#[test]
fn raw_obj_assign_two_fields_arith_emits_rewritten() {
    // .z = (.x + .y) on {"x":3,"y":4,"z":0} → {"x":3,"y":4,"z":7}
    let mut buf = Vec::new();
    let outcome = apply_obj_assign_two_fields_arith_raw(
        b"{\"x\":3,\"y\":4,\"z\":0}", "z", "x", "y", BinOp::Add, &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"{\"x\":3,\"y\":4,\"z\":7}");
}

#[test]
fn raw_obj_assign_two_fields_arith_missing_field_bails() {
    let mut buf = Vec::new();
    let outcome = apply_obj_assign_two_fields_arith_raw(
        b"{\"x\":3,\"z\":0}", "z", "x", "y", BinOp::Add, &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_obj_assign_two_fields_arith_div_by_zero_bails() {
    let mut buf = Vec::new();
    let outcome = apply_obj_assign_two_fields_arith_raw(
        b"{\"x\":3,\"y\":0,\"z\":0}", "z", "x", "y", BinOp::Div, &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_obj_assign_two_fields_arith_non_arith_op_bails() {
    let mut buf = Vec::new();
    let outcome = apply_obj_assign_two_fields_arith_raw(
        b"{\"x\":3,\"y\":4,\"z\":0}", "z", "x", "y", BinOp::Eq, &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_obj_assign_two_fields_arith_non_object_input_bails() {
    for raw in [b"42".as_slice(), b"null".as_slice(), b"[1,2,3]".as_slice()] {
        let mut buf = Vec::new();
        let outcome = apply_obj_assign_two_fields_arith_raw(
            raw, "z", "x", "y", BinOp::Add, &mut buf,
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "raw={:?}", std::str::from_utf8(raw).unwrap(),
        );
    }
}

// ---------------------------------------------------------------------------
// `type` — no-Bail helper. The output is uniquely determined by the first
// byte of the parsed JSON value, with no type-error case in jq.

#[test]
fn raw_is_type_object() {
    let mut buf = Vec::new();
    let outcome = apply_is_type_raw(b"{\"x\":1}", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"\"object\"\n");
}

#[test]
fn raw_is_type_each_kind() {
    for (raw, expected) in [
        (&b"[1,2]"[..], &b"\"array\"\n"[..]),
        (&b"\"hi\""[..], &b"\"string\"\n"[..]),
        (&b"true"[..], &b"\"boolean\"\n"[..]),
        (&b"false"[..], &b"\"boolean\"\n"[..]),
        (&b"null"[..], &b"\"null\"\n"[..]),
        (&b"42"[..], &b"\"number\"\n"[..]),
        (&b"-3.14"[..], &b"\"number\"\n"[..]),
    ] {
        let mut buf = Vec::new();
        let outcome = apply_is_type_raw(raw, &mut buf);
        assert!(matches!(outcome, RawApplyOutcome::Emit));
        assert_eq!(buf, expected, "input={:?}", std::str::from_utf8(raw).unwrap());
    }
}

#[test]
fn raw_is_type_empty_input_bails() {
    let mut buf = Vec::new();
    let outcome = apply_is_type_raw(b"", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

// ---------------------------------------------------------------------------
// `length` — polymorphic. The raw scanner only handles container/null lengths
// directly; strings and numbers Bail to the generic path (which has the
// unicode-aware string-length and abs-of-number paths). Booleans Bail too —
// jq raises `boolean has no length` and the generic path produces it.

#[test]
fn raw_is_length_array() {
    let mut buf = Vec::new();
    let outcome = apply_is_length_raw(b"[1,2,3]", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"3\n");
}

#[test]
fn raw_is_length_array_empty() {
    let mut buf = Vec::new();
    let outcome = apply_is_length_raw(b"[]", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"0\n");
}

#[test]
fn raw_is_length_object() {
    let mut buf = Vec::new();
    let outcome = apply_is_length_raw(b"{\"a\":1,\"b\":2}", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"2\n");
}

#[test]
fn raw_is_length_null_is_zero() {
    let mut buf = Vec::new();
    let outcome = apply_is_length_raw(b"null", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"0\n");
}

#[test]
fn raw_is_length_string_bails() {
    // The raw scanner punts on strings (`json_value_length` only handles
    // containers + null); generic path counts unicode code points.
    let mut buf = Vec::new();
    let outcome = apply_is_length_raw(b"\"hello\"", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_is_length_number_bails() {
    // Same — generic path produces `abs(n)` for number length.
    let mut buf = Vec::new();
    let outcome = apply_is_length_raw(b"-7", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_is_length_boolean_bails() {
    // jq: `boolean (true) has no length`
    for raw in [&b"true"[..], &b"false"[..]] {
        let mut buf = Vec::new();
        let outcome = apply_is_length_raw(raw, &mut buf);
        assert!(matches!(outcome, RawApplyOutcome::Bail), "raw={:?}", std::str::from_utf8(raw).unwrap());
        assert!(buf.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `.[] | <type>s` — each-element type filter. Bails on non-iterable input
// or unrecognised type name.

#[test]
fn raw_each_type_filter_array_strings() {
    let mut buf = Vec::new();
    let outcome = apply_each_type_filter_raw(b"[1,\"a\",2,\"b\",null]", "string", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"\"a\"\n\"b\"\n");
}

#[test]
fn raw_each_type_filter_object_numbers() {
    let mut buf = Vec::new();
    let outcome = apply_each_type_filter_raw(b"{\"a\":1,\"b\":\"hi\",\"c\":3}", "number", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"1\n3\n");
}

#[test]
fn raw_each_type_filter_no_match_emits_nothing() {
    let mut buf = Vec::new();
    let outcome = apply_each_type_filter_raw(b"[1,2,3]", "string", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert!(buf.is_empty());
}

#[test]
fn raw_each_type_filter_non_iterable_bails() {
    for raw in [&b"42"[..], &b"\"hi\""[..], &b"null"[..], &b"true"[..]] {
        let mut buf = Vec::new();
        let outcome = apply_each_type_filter_raw(raw, "string", &mut buf);
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "raw={:?}", std::str::from_utf8(raw).unwrap(),
        );
    }
}

#[test]
fn raw_each_type_filter_unknown_type_bails() {
    let mut buf = Vec::new();
    let outcome = apply_each_type_filter_raw(b"[1,2]", "unknown", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

// ---------------------------------------------------------------------------
// `[.[] | select(type == "T")]` — collect elements of given type into array.

#[test]
fn raw_collect_each_select_type_emits_array() {
    let mut buf = Vec::new();
    let outcome = apply_collect_each_select_type_raw(b"[1,\"a\",2,\"b\"]", "string", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"[\"a\",\"b\"]\n");
}

#[test]
fn raw_collect_each_select_type_no_match_empty_array() {
    let mut buf = Vec::new();
    let outcome = apply_collect_each_select_type_raw(b"[1,2,3]", "string", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"[]\n");
}

#[test]
fn raw_collect_each_select_type_truncates_buf_on_bail() {
    // Caller's buffer should be restored to pre-call state on Bail.
    let mut buf = Vec::from(&b"prefix"[..]);
    let outcome = apply_collect_each_select_type_raw(b"42", "string", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(buf, b"prefix");
}

#[test]
fn raw_collect_each_select_type_unknown_type_bails() {
    let mut buf = Vec::new();
    let outcome = apply_collect_each_select_type_raw(b"[1,2]", "unknown", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

// ---------------------------------------------------------------------------
// `first(.[] | select(type == "T"))` — first element of given type.

#[test]
fn raw_first_each_select_type_emits_first_match() {
    let mut buf = Vec::new();
    let outcome = apply_first_each_select_type_raw(b"[1,\"a\",2,\"b\"]", "string", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"\"a\"\n");
}

#[test]
fn raw_first_each_select_type_no_match_emits_nothing() {
    // first(...) on empty stream: no output, no error.
    let mut buf = Vec::new();
    let outcome = apply_first_each_select_type_raw(b"[1,2,3]", "string", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert!(buf.is_empty());
}

#[test]
fn raw_first_each_select_type_non_iterable_bails() {
    let mut buf = Vec::new();
    let outcome = apply_first_each_select_type_raw(b"42", "string", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_first_each_select_type_unknown_type_bails() {
    let mut buf = Vec::new();
    let outcome = apply_first_each_select_type_raw(b"[1,2]", "unknown", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

// ---------------------------------------------------------------------------
// `apply_field_cmp_val_raw` — generic `.field cmp <num|str>` predicate
// reporting bool via emit closure. Bails on non-object/missing/wrong-type/
// escape-bearing-string/non-cmp-op.

#[test]
fn raw_field_cmp_val_numeric() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_field_cmp_val_raw(
        b"{\"x\":5}", "x", BinOp::Gt, &CmpVal::Num(3.0), |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![true]);
}

#[test]
fn raw_field_cmp_val_string_eq() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_field_cmp_val_raw(
        b"{\"x\":\"hi\"}", "x", BinOp::Eq, &CmpVal::Str("hi".to_string()), |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![true]);
}

#[test]
fn raw_field_cmp_val_string_lt() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_field_cmp_val_raw(
        b"{\"x\":\"abc\"}", "x", BinOp::Lt, &CmpVal::Str("abd".to_string()), |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![true]);
}

#[test]
fn raw_field_cmp_val_field_missing_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_field_cmp_val_raw(
        b"{\"y\":1}", "x", BinOp::Gt, &CmpVal::Num(3.0), |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_cmp_val_num_field_non_numeric_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_field_cmp_val_raw(
        b"{\"x\":\"hi\"}", "x", BinOp::Gt, &CmpVal::Num(3.0), |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_field_cmp_val_str_field_non_string_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_field_cmp_val_raw(
        b"{\"x\":42}", "x", BinOp::Eq, &CmpVal::Str("42".to_string()), |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_field_cmp_val_str_field_escape_bearing_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_field_cmp_val_raw(
        br#"{"x":"a\nb"}"#, "x", BinOp::Eq, &CmpVal::Str("a".to_string()), |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_field_cmp_val_non_cmp_op_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_field_cmp_val_raw(
        b"{\"x\":5}", "x", BinOp::Add, &CmpVal::Num(3.0), |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_field_cmp_val_non_object_input_bails() {
    for raw in [b"42".as_slice(), b"\"hi\"".as_slice(), b"null".as_slice(), b"[1]".as_slice()] {
        let mut emitted: Vec<bool> = Vec::new();
        let outcome = apply_field_cmp_val_raw(
            raw, "x", BinOp::Gt, &CmpVal::Num(3.0), |b| emitted.push(b),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "raw={:?}", std::str::from_utf8(raw).unwrap(),
        );
    }
}

// ---------------------------------------------------------------------------
// `if .field == null then T else F end` — null branch literal. Bails on
// non-object input.

#[test]
fn raw_null_branch_lit_eq_null_present_field() {
    let mut buf = Vec::new();
    let outcome = apply_null_branch_lit_raw(
        b"{\"x\":42}", "x", true, b"\"yes\"", b"\"no\"", &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"\"no\"\n");
}

#[test]
fn raw_null_branch_lit_eq_null_missing_field() {
    // jq: missing field is null
    let mut buf = Vec::new();
    let outcome = apply_null_branch_lit_raw(
        b"{\"y\":1}", "x", true, b"\"yes\"", b"\"no\"", &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"\"yes\"\n");
}

#[test]
fn raw_null_branch_lit_eq_null_explicit_null() {
    let mut buf = Vec::new();
    let outcome = apply_null_branch_lit_raw(
        b"{\"x\":null}", "x", true, b"\"yes\"", b"\"no\"", &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"\"yes\"\n");
}

#[test]
fn raw_null_branch_lit_ne_null_inverted() {
    let mut buf = Vec::new();
    let outcome = apply_null_branch_lit_raw(
        b"{\"x\":null}", "x", false, b"\"yes\"", b"\"no\"", &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"\"no\"\n");
    let mut buf2 = Vec::new();
    let outcome2 = apply_null_branch_lit_raw(
        b"{\"x\":42}", "x", false, b"\"yes\"", b"\"no\"", &mut buf2,
    );
    assert!(matches!(outcome2, RawApplyOutcome::Emit));
    assert_eq!(buf2, b"\"yes\"\n");
}

#[test]
fn raw_null_branch_lit_non_object_bails() {
    // Pre-existing #83-class divergence: non-object root silently emitted T
    // (treating as field-missing → null) instead of jq's `Cannot index <type>`.
    for raw in [b"42".as_slice(), b"\"hi\"".as_slice(), b"null".as_slice(), b"[1,2]".as_slice()] {
        let mut buf = Vec::new();
        let outcome = apply_null_branch_lit_raw(
            raw, "x", true, b"\"yes\"", b"\"no\"", &mut buf,
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "raw={:?}", std::str::from_utf8(raw).unwrap(),
        );
        assert!(buf.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `select(.x.y.z cmp N)` — nested-field numeric comparison. Bails on
// non-object/missing-or-non-numeric/non-cmp-op.

#[test]
fn raw_select_nested_cmp_emits_match() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_nested_cmp_raw(
        b"{\"a\":{\"b\":{\"c\":5}}}", &["a", "b", "c"], BinOp::Gt, 3.0,
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"{\"a\":{\"b\":{\"c\":5}}}".to_vec()]);
}

#[test]
fn raw_select_nested_cmp_no_emit_on_predicate_fail() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_nested_cmp_raw(
        b"{\"a\":{\"b\":{\"c\":1}}}", &["a", "b", "c"], BinOp::Gt, 3.0,
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert!(emitted.is_empty());
}

#[test]
fn raw_select_nested_cmp_intermediate_non_object_bails() {
    // .a.b.c on .a being a non-object: jq raises type error.
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_nested_cmp_raw(
        b"{\"a\":42}", &["a", "b", "c"], BinOp::Gt, 0.0,
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_select_nested_cmp_leaf_missing_bails() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_nested_cmp_raw(
        b"{\"a\":{\"b\":{}}}", &["a", "b", "c"], BinOp::Gt, 0.0,
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_select_nested_cmp_leaf_non_numeric_bails() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_nested_cmp_raw(
        b"{\"a\":{\"b\":{\"c\":\"hi\"}}}", &["a", "b", "c"], BinOp::Gt, 0.0,
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_select_nested_cmp_non_cmp_op_bails() {
    for op in [BinOp::Add, BinOp::Mul, BinOp::And] {
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_select_nested_cmp_raw(
            b"{\"a\":{\"b\":{\"c\":5}}}", &["a", "b", "c"], op, 0.0,
            |raw| emitted.push(raw.to_vec()),
        );
        assert!(matches!(outcome, RawApplyOutcome::Bail), "op={:?}", op);
    }
}

#[test]
fn raw_select_nested_cmp_non_object_input_bails() {
    for raw in [b"42".as_slice(), b"\"hi\"".as_slice(), b"null".as_slice(), b"[1,2]".as_slice()] {
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_select_nested_cmp_raw(
            raw, &["a", "b"], BinOp::Gt, 0.0, |raw| emitted.push(raw.to_vec()),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "raw={:?}", std::str::from_utf8(raw).unwrap(),
        );
    }
}

// ---------------------------------------------------------------------------
// `select((.numfield cmp N) and (.strfield op_str arg))` — numeric+string
// compound predicate. Bails on non-object/missing-or-non-numeric/non-string/
// escape-bearing-with-test/unknown-op.

#[test]
fn raw_select_num_str_emits_match_startswith() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_num_str_raw(
        b"{\"n\":5,\"s\":\"hello\"}", "n", BinOp::Gt, 0.0, "s", "startswith", b"he", None,
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![b"{\"n\":5,\"s\":\"hello\"}".to_vec()]);
}

#[test]
fn raw_select_num_str_no_emit_on_num_fail() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_num_str_raw(
        b"{\"n\":-1,\"s\":\"hello\"}", "n", BinOp::Gt, 0.0, "s", "startswith", b"he", None,
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert!(emitted.is_empty());
}

#[test]
fn raw_select_num_str_no_emit_on_str_fail() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_num_str_raw(
        b"{\"n\":5,\"s\":\"world\"}", "n", BinOp::Gt, 0.0, "s", "startswith", b"he", None,
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert!(emitted.is_empty());
}

#[test]
fn raw_select_num_str_contains_eq_endswith() {
    for (op, arg, expected) in [
        ("contains", &b"ell"[..], true),
        ("endswith", &b"llo"[..], true),
        ("eq", &b"hello"[..], true),
        ("eq", &b"world"[..], false),
    ] {
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_select_num_str_raw(
            b"{\"n\":5,\"s\":\"hello\"}", "n", BinOp::Gt, 0.0, "s", op, arg, None,
            |raw| emitted.push(raw.to_vec()),
        );
        assert!(matches!(outcome, RawApplyOutcome::Emit));
        assert_eq!(emitted.len(), if expected { 1 } else { 0 }, "op={} arg={:?}", op, std::str::from_utf8(arg).unwrap());
    }
}

#[test]
fn raw_select_num_str_test_with_regex() {
    let re = regex::Regex::new("^h.l").unwrap();
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_num_str_raw(
        b"{\"n\":5,\"s\":\"hello\"}", "n", BinOp::Gt, 0.0, "s", "test", b"^h.l", Some(&re),
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted.len(), 1);
}

#[test]
fn raw_select_num_str_test_without_regex_bails() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_num_str_raw(
        b"{\"n\":5,\"s\":\"hello\"}", "n", BinOp::Gt, 0.0, "s", "test", b"^h.l", None,
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_select_num_str_num_field_missing_bails() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_num_str_raw(
        b"{\"s\":\"hello\"}", "n", BinOp::Gt, 0.0, "s", "startswith", b"he", None,
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_select_num_str_str_field_missing_bails() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_num_str_raw(
        b"{\"n\":5}", "n", BinOp::Gt, 0.0, "s", "startswith", b"he", None,
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_select_num_str_str_field_non_string_bails() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_num_str_raw(
        b"{\"n\":5,\"s\":42}", "n", BinOp::Gt, 0.0, "s", "startswith", b"he", None,
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_select_num_str_test_escape_bearing_string_bails() {
    let re = regex::Regex::new("a").unwrap();
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_num_str_raw(
        br#"{"n":5,"s":"a\nb"}"#, "n", BinOp::Gt, 0.0, "s", "test", b"a", Some(&re),
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_select_num_str_unknown_str_op_bails() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_num_str_raw(
        b"{\"n\":5,\"s\":\"hello\"}", "n", BinOp::Gt, 0.0, "s", "unknown", b"he", None,
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_select_num_str_non_cmp_op_bails() {
    let mut emitted: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_num_str_raw(
        b"{\"n\":5,\"s\":\"hello\"}", "n", BinOp::Add, 0.0, "s", "startswith", b"he", None,
        |raw| emitted.push(raw.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_select_num_str_non_object_input_bails() {
    for raw in [b"42".as_slice(), b"\"hi\"".as_slice(), b"null".as_slice(), b"[1,2]".as_slice()] {
        let mut emitted: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_select_num_str_raw(
            raw, "n", BinOp::Gt, 0.0, "s", "startswith", b"he", None,
            |raw| emitted.push(raw.to_vec()),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "raw={:?}", std::str::from_utf8(raw).unwrap(),
        );
    }
}

// ---------------------------------------------------------------------------
// `(.x cmp1 N1) and/or (.y cmp2 N2) ...` — compound numeric predicate.
// Helper short-circuits AND/OR; Bails on non-object / missing-or-non-numeric
// field / non-cmp-op / non-and-or conjunct / buffer-mismatch.

#[test]
fn raw_compound_field_cmp_and_short_circuits() {
    // (.x > 0) and (.y > 0) with x=5, y=3 → true
    let mut emitted: Vec<bool> = Vec::new();
    let fields = ["x", "y"];
    let spec = vec![(0, BinOp::Gt, 0.0), (1, BinOp::Gt, 0.0)];
    let mut vals = vec![0.0; 2];
    let outcome = apply_compound_field_cmp_raw(
        b"{\"x\":5,\"y\":3}", &fields, &spec, BinOp::And, &mut vals,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![true]);
}

#[test]
fn raw_compound_field_cmp_and_short_circuit_false() {
    // (.x > 0) and (.y > 0) with x=5, y=-3 → false (short-circuits at second)
    let mut emitted: Vec<bool> = Vec::new();
    let fields = ["x", "y"];
    let spec = vec![(0, BinOp::Gt, 0.0), (1, BinOp::Gt, 0.0)];
    let mut vals = vec![0.0; 2];
    let outcome = apply_compound_field_cmp_raw(
        b"{\"x\":5,\"y\":-3}", &fields, &spec, BinOp::And, &mut vals,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![false]);
}

#[test]
fn raw_compound_field_cmp_or_short_circuits_true() {
    // (.x > 100) or (.y < 0) with x=5, y=-3 → true (second matches)
    let mut emitted: Vec<bool> = Vec::new();
    let fields = ["x", "y"];
    let spec = vec![(0, BinOp::Gt, 100.0), (1, BinOp::Lt, 0.0)];
    let mut vals = vec![0.0; 2];
    let outcome = apply_compound_field_cmp_raw(
        b"{\"x\":5,\"y\":-3}", &fields, &spec, BinOp::Or, &mut vals,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![true]);
}

#[test]
fn raw_compound_field_cmp_or_all_false() {
    // (.x > 100) or (.y > 100) with x=5, y=3 → false
    let mut emitted: Vec<bool> = Vec::new();
    let fields = ["x", "y"];
    let spec = vec![(0, BinOp::Gt, 100.0), (1, BinOp::Gt, 100.0)];
    let mut vals = vec![0.0; 2];
    let outcome = apply_compound_field_cmp_raw(
        b"{\"x\":5,\"y\":3}", &fields, &spec, BinOp::Or, &mut vals,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![false]);
}

#[test]
fn raw_compound_field_cmp_three_fields() {
    // (.a > 0) and (.b > 0) and (.c > 0) with a=1, b=2, c=3 → true
    let mut emitted: Vec<bool> = Vec::new();
    let fields = ["a", "b", "c"];
    let spec = vec![(0, BinOp::Gt, 0.0), (1, BinOp::Gt, 0.0), (2, BinOp::Gt, 0.0)];
    let mut vals = vec![0.0; 3];
    let outcome = apply_compound_field_cmp_raw(
        b"{\"a\":1,\"b\":2,\"c\":3}", &fields, &spec, BinOp::And, &mut vals,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![true]);
}

#[test]
fn raw_compound_field_cmp_field_missing_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let fields = ["x", "y"];
    let spec = vec![(0, BinOp::Gt, 0.0), (1, BinOp::Gt, 0.0)];
    let mut vals = vec![0.0; 2];
    let outcome = apply_compound_field_cmp_raw(
        b"{\"x\":5}", &fields, &spec, BinOp::And, &mut vals,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_compound_field_cmp_non_numeric_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let fields = ["x", "y"];
    let spec = vec![(0, BinOp::Gt, 0.0), (1, BinOp::Gt, 0.0)];
    let mut vals = vec![0.0; 2];
    let outcome = apply_compound_field_cmp_raw(
        b"{\"x\":5,\"y\":\"hi\"}", &fields, &spec, BinOp::And, &mut vals,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_compound_field_cmp_non_cmp_op_bails() {
    for op in [BinOp::Add, BinOp::Mul] {
        let mut emitted: Vec<bool> = Vec::new();
        let fields = ["x"];
        let spec = vec![(0, op, 0.0)];
        let mut vals = vec![0.0; 1];
        let outcome = apply_compound_field_cmp_raw(
            b"{\"x\":5}", &fields, &spec, BinOp::And, &mut vals,
            |b| emitted.push(b),
        );
        assert!(matches!(outcome, RawApplyOutcome::Bail), "op={:?}", op);
    }
}

#[test]
fn raw_compound_field_cmp_non_and_or_conjunct_bails() {
    for conj in [BinOp::Add, BinOp::Eq] {
        let mut emitted: Vec<bool> = Vec::new();
        let fields = ["x"];
        let spec = vec![(0, BinOp::Gt, 0.0)];
        let mut vals = vec![0.0; 1];
        let outcome = apply_compound_field_cmp_raw(
            b"{\"x\":5}", &fields, &spec, conj, &mut vals, |b| emitted.push(b),
        );
        assert!(matches!(outcome, RawApplyOutcome::Bail), "conj={:?}", conj);
    }
}

#[test]
fn raw_compound_field_cmp_non_object_input_bails() {
    let fields = ["x"];
    let spec = vec![(0, BinOp::Gt, 0.0)];
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<bool> = Vec::new();
        let mut vals = vec![0.0; 1];
        let outcome = apply_compound_field_cmp_raw(
            raw, &fields, &spec, BinOp::And, &mut vals, |b| emitted.push(b),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
    }
}

// ---------------------------------------------------------------------------
// `apply_numeric_expr_raw` — generic ArithExpr over named fields, optional
// trailing math unary. Bails on non-object / missing-or-non-numeric field /
// non-finite final result / buffer-mismatch.

#[test]
fn raw_numeric_expr_two_fields_add() {
    // .x + .y with x=3, y=4
    let arith = ArithExpr::BinOp(
        BinOp::Add,
        Box::new(ArithExpr::Field(0)),
        Box::new(ArithExpr::Field(1)),
    );
    let fields = ["x", "y"];
    let mut ranges = vec![(0, 0); 2];
    let mut vals = vec![0.0; 2];
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_numeric_expr_raw(
        b"{\"x\":3,\"y\":4}", &fields, &arith, None,
        &mut ranges, &mut vals, |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![7.0]);
}

#[test]
fn raw_numeric_expr_three_fields() {
    // .a * .b + .c with a=2, b=3, c=4 → 10
    let arith = ArithExpr::BinOp(
        BinOp::Add,
        Box::new(ArithExpr::BinOp(
            BinOp::Mul,
            Box::new(ArithExpr::Field(0)),
            Box::new(ArithExpr::Field(1)),
        )),
        Box::new(ArithExpr::Field(2)),
    );
    let fields = ["a", "b", "c"];
    let mut ranges = vec![(0, 0); 3];
    let mut vals = vec![0.0; 3];
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_numeric_expr_raw(
        b"{\"a\":2,\"b\":3,\"c\":4}", &fields, &arith, None,
        &mut ranges, &mut vals, |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![10.0]);
}

#[test]
fn raw_numeric_expr_with_unary_floor() {
    // floor((.x + .y) / 2) with x=3, y=4 → floor(3.5) = 3
    let arith = ArithExpr::BinOp(
        BinOp::Div,
        Box::new(ArithExpr::BinOp(
            BinOp::Add,
            Box::new(ArithExpr::Field(0)),
            Box::new(ArithExpr::Field(1)),
        )),
        Box::new(ArithExpr::Const(2.0)),
    );
    let fields = ["x", "y"];
    let mut ranges = vec![(0, 0); 2];
    let mut vals = vec![0.0; 2];
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_numeric_expr_raw(
        b"{\"x\":3,\"y\":4}", &fields, &arith, Some(MathUnary::Floor),
        &mut ranges, &mut vals, |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![3.0]);
}

#[test]
fn raw_numeric_expr_field_missing_bails() {
    let arith = ArithExpr::Field(0);
    let fields = ["x"];
    let mut ranges = vec![(0, 0); 1];
    let mut vals = vec![0.0; 1];
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_numeric_expr_raw(
        b"{\"y\":1}", &fields, &arith, None,
        &mut ranges, &mut vals, |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_numeric_expr_non_numeric_field_bails() {
    let arith = ArithExpr::Field(0);
    let fields = ["x"];
    let mut ranges = vec![(0, 0); 1];
    let mut vals = vec![0.0; 1];
    for inner in [&b"{\"x\":\"hi\"}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1]}"[..]] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_numeric_expr_raw(
            inner, &fields, &arith, None,
            &mut ranges, &mut vals, |n| emitted.push(n),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-numeric input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
    }
}

#[test]
fn raw_numeric_expr_div_by_zero_bails() {
    // .x / .y with x=1, y=0 → inf → Bail
    let arith = ArithExpr::BinOp(
        BinOp::Div,
        Box::new(ArithExpr::Field(0)),
        Box::new(ArithExpr::Field(1)),
    );
    let fields = ["x", "y"];
    let mut ranges = vec![(0, 0); 2];
    let mut vals = vec![0.0; 2];
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_numeric_expr_raw(
        b"{\"x\":1,\"y\":0}", &fields, &arith, None,
        &mut ranges, &mut vals, |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_numeric_expr_non_object_input_bails() {
    let arith = ArithExpr::Field(0);
    let fields = ["x"];
    let mut ranges = vec![(0, 0); 1];
    let mut vals = vec![0.0; 1];
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_numeric_expr_raw(
            raw, &fields, &arith, None,
            &mut ranges, &mut vals, |n| emitted.push(n),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
    }
}

#[test]
fn raw_numeric_expr_buffer_too_small_bails() {
    // Defensive: caller bug
    let arith = ArithExpr::BinOp(
        BinOp::Add,
        Box::new(ArithExpr::Field(0)),
        Box::new(ArithExpr::Field(1)),
    );
    let fields = ["x", "y"];
    let mut ranges = vec![(0, 0); 1];
    let mut vals = vec![0.0; 1];
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_numeric_expr_raw(
        b"{\"x\":3,\"y\":4}", &fields, &arith, None,
        &mut ranges, &mut vals, |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

// ---------------------------------------------------------------------------
// `(.field | index/rindex("str")) <op> <const>` — match position + arith.
// Helper handles jq's `null + N = N` rule on no-match for Add; Bails on
// non-object / non-string / escape-bearing string / non-arith op /
// no-match-with-non-Add-op.

#[test]
fn raw_field_index_arith_index_first_match_codepoint_pos() {
    let mut emitted: Vec<f64> = Vec::new();
    // .x | index("ab") + 5 with x="cabd" → match at byte 1, cp=1, +5 = 6
    let outcome = apply_field_index_arith_raw(
        b"{\"x\":\"cabd\"}", "x", b"ab", false, BinOp::Add, 5.0,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![6.0]);
}

#[test]
fn raw_field_index_arith_rindex_last_match() {
    let mut emitted: Vec<f64> = Vec::new();
    // .x | rindex("a") + 0 with x="cabad" → last 'a' at byte 3, cp=3
    let outcome = apply_field_index_arith_raw(
        b"{\"x\":\"cabad\"}", "x", b"a", true, BinOp::Add, 0.0,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![3.0]);
}

#[test]
fn raw_field_index_arith_no_match_add_emits_n() {
    let mut emitted: Vec<f64> = Vec::new();
    // .x | index("zz") + 5 with x="abc" → no match, null+5=5
    let outcome = apply_field_index_arith_raw(
        b"{\"x\":\"abc\"}", "x", b"zz", false, BinOp::Add, 5.0,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![5.0]);
}

#[test]
fn raw_field_index_arith_no_match_non_add_bails() {
    // For Sub/Mul/Div on null, jq raises type error; helper Bails.
    for op in [BinOp::Sub, BinOp::Mul, BinOp::Div] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_index_arith_raw(
            b"{\"x\":\"abc\"}", "x", b"zz", false, op, 5.0,
            |n| emitted.push(n),
        );
        assert!(matches!(outcome, RawApplyOutcome::Bail), "op={:?}", op);
        assert!(emitted.is_empty());
    }
}

#[test]
fn raw_field_index_arith_missing_field_add_emits_n() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_index_arith_raw(
        b"{\"y\":1}", "x", b"ab", false, BinOp::Add, 7.0,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![7.0]);
}

#[test]
fn raw_field_index_arith_missing_field_non_add_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_index_arith_raw(
        b"{\"y\":1}", "x", b"ab", false, BinOp::Sub, 7.0,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_index_arith_non_string_field_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_index_arith_raw(
        b"{\"x\":42}", "x", b"a", false, BinOp::Add, 0.0,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_index_arith_escape_bearing_string_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_index_arith_raw(
        br#"{"x":"a\nb"}"#, "x", b"b", false, BinOp::Add, 0.0,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_index_arith_non_arith_op_bails() {
    for op in [BinOp::Eq, BinOp::Lt, BinOp::And] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_index_arith_raw(
            b"{\"x\":\"abc\"}", "x", b"a", false, op, 0.0,
            |n| emitted.push(n),
        );
        assert!(matches!(outcome, RawApplyOutcome::Bail), "op={:?}", op);
    }
}

#[test]
fn raw_field_index_arith_div_by_zero_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_index_arith_raw(
        b"{\"x\":\"abc\"}", "x", b"a", false, BinOp::Div, 0.0,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_index_arith_codepoint_position_for_multibyte() {
    // .x | index("c") with x="日本cd" → byte position 6, cp position 2
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_index_arith_raw(
        "{\"x\":\"日本cd\"}".as_bytes(), "x", b"c", false, BinOp::Add, 0.0,
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![2.0]);
}

#[test]
fn raw_field_index_arith_non_object_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_index_arith_raw(
            raw, "x", b"a", false, BinOp::Add, 0.0, |n| emitted.push(n),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `.field | <unary>` — multi-modal: Length/Utf8ByteLength/Explode/AsciiCase/
// Floor/Ceil/Sqrt/Abs/Fabs/ToString. Helper writes raw output bytes + `\n`
// directly into the caller buffer on Emit.

#[test]
fn raw_field_unary_num_explode_ascii() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(b"{\"x\":\"abc\"}", "x", UnaryOp::Explode, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"[97,98,99]\n");
}

#[test]
fn raw_field_unary_num_explode_non_string_bails() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(b"{\"x\":42}", "x", UnaryOp::Explode, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_field_unary_num_explode_escaped_string_bails() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(br#"{"x":"a\nb"}"#, "x", UnaryOp::Explode, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_field_unary_num_ascii_downcase() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(b"{\"x\":\"ABC\"}", "x", UnaryOp::AsciiDowncase, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"\"abc\"\n");
}

#[test]
fn raw_field_unary_num_ascii_upcase() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(b"{\"x\":\"abc\"}", "x", UnaryOp::AsciiUpcase, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"\"ABC\"\n");
}

#[test]
fn raw_field_unary_num_ascii_case_non_string_bails() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(b"{\"x\":42}", "x", UnaryOp::AsciiUpcase, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_field_unary_num_length_string() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(b"{\"x\":\"abcd\"}", "x", UnaryOp::Length, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"4\n");
}

#[test]
fn raw_field_unary_num_length_array() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(b"{\"x\":[1,2,3]}", "x", UnaryOp::Length, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"3\n");
}

#[test]
fn raw_field_unary_num_length_null_field() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(b"{\"x\":null}", "x", UnaryOp::Length, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"0\n");
}

#[test]
fn raw_field_unary_num_length_missing_field_is_zero() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(b"{\"y\":1}", "x", UnaryOp::Length, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"0\n");
}

#[test]
fn raw_field_unary_num_length_number_is_abs() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(b"{\"x\":-7}", "x", UnaryOp::Length, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"7\n");
}

#[test]
fn raw_field_unary_num_length_boolean_bails() {
    // jq: `boolean has no length`
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(b"{\"x\":true}", "x", UnaryOp::Length, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_field_unary_num_length_escaped_string_bails() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(br#"{"x":"a\nb"}"#, "x", UnaryOp::Length, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_field_unary_num_utf8_byte_length() {
    // 4-byte UTF-8 char "🦀" (0xF0 0x9F 0xA6 0x80) → utf8_byte_length=4, length=1
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw("{\"x\":\"🦀\"}".as_bytes(), "x", UnaryOp::Utf8ByteLength, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"4\n");
    let mut buf2 = Vec::new();
    let outcome2 = apply_field_unary_num_raw("{\"x\":\"🦀\"}".as_bytes(), "x", UnaryOp::Length, &mut buf2);
    assert!(matches!(outcome2, RawApplyOutcome::Emit));
    assert_eq!(buf2, b"1\n");
}

#[test]
fn raw_field_unary_num_floor() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(b"{\"x\":3.7}", "x", UnaryOp::Floor, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"3\n");
}

#[test]
fn raw_field_unary_num_floor_non_numeric_bails() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(b"{\"x\":\"hi\"}", "x", UnaryOp::Floor, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_field_unary_num_tostring_numeric() {
    let mut buf = Vec::new();
    let outcome = apply_field_unary_num_raw(b"{\"x\":42}", "x", UnaryOp::ToString, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf, b"\"42\"\n");
}

#[test]
fn raw_field_unary_num_tostring_non_numeric_bails() {
    // generic path knows `tostring` of strings/arrays/etc
    for inner in [&b"{\"x\":\"hi\"}"[..], &b"{\"x\":[1,2]}"[..], &b"{\"x\":null}"[..]] {
        let mut buf = Vec::new();
        let outcome = apply_field_unary_num_raw(inner, "x", UnaryOp::ToString, &mut buf);
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for tostring on {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(), outcome,
        );
    }
}

#[test]
fn raw_field_unary_num_unsupported_unary_bails() {
    for uop in [UnaryOp::ToNumber, UnaryOp::Type, UnaryOp::FromJson] {
        let mut buf = Vec::new();
        let outcome = apply_field_unary_num_raw(b"{\"x\":1}", "x", uop, &mut buf);
        assert!(matches!(outcome, RawApplyOutcome::Bail), "uop={:?}", uop);
    }
}

#[test]
fn raw_field_unary_num_non_object_input_bails() {
    // Was a #83-class divergence: file-mode silently emitted "0" for
    // length on non-object roots.
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        for uop in [UnaryOp::Length, UnaryOp::Explode, UnaryOp::AsciiDowncase, UnaryOp::Floor, UnaryOp::ToString] {
            let mut buf = Vec::new();
            let outcome = apply_field_unary_num_raw(raw, "x", uop, &mut buf);
            assert!(
                matches!(outcome, RawApplyOutcome::Bail),
                "expected Bail for input {:?} with uop {:?}, got {:?}",
                std::str::from_utf8(raw).unwrap(), uop, outcome,
            );
            assert!(buf.is_empty());
        }
    }
}

// ---------------------------------------------------------------------------
// `(.field | <unary>) <op1> <c1> ...` — unary then arith chain. Helper Bails
// on non-object / unsupported unary / non-numeric-for-numeric-unary /
// escape-bearing string for length / non-finite final.

#[test]
fn raw_field_unary_arith_length_string() {
    // (.x | length) + 5 with x="abc" → 3 + 5 = 8
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_unary_arith_raw(
        b"{\"x\":\"abc\"}", "x", UnaryOp::Length, &[(BinOp::Add, 5.0)],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![8.0]);
}

#[test]
fn raw_field_unary_arith_length_array() {
    // (.x | length) * 2 with x=[1,2,3] → 3 * 2 = 6
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_unary_arith_raw(
        b"{\"x\":[1,2,3]}", "x", UnaryOp::Length, &[(BinOp::Mul, 2.0)],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![6.0]);
}

#[test]
fn raw_field_unary_arith_length_null_value() {
    // jq: `null | length = 0`. (.x | length) + 1 with x=null → 1
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_unary_arith_raw(
        b"{\"x\":null}", "x", UnaryOp::Length, &[(BinOp::Add, 1.0)],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![1.0]);
}

#[test]
fn raw_field_unary_arith_length_missing_field_is_zero() {
    // jq: missing field is null, null | length = 0. (.x | length) + 7 → 7
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_unary_arith_raw(
        b"{\"y\":1}", "x", UnaryOp::Length, &[(BinOp::Add, 7.0)],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![7.0]);
}

#[test]
fn raw_field_unary_arith_length_number_field_uses_abs() {
    // jq: number | length = abs. (.x | length) + 0 with x=-7 → 7
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_unary_arith_raw(
        b"{\"x\":-7}", "x", UnaryOp::Length, &[(BinOp::Add, 0.0)],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![7.0]);
}

#[test]
fn raw_field_unary_arith_length_escape_bearing_string_bails() {
    // Raw scanner can't decode `\n` to count code points correctly.
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_unary_arith_raw(
        br#"{"x":"a\nb"}"#, "x", UnaryOp::Length, &[(BinOp::Add, 0.0)],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_unary_arith_floor_then_chain() {
    // (.x | floor) + 1 with x=3.7 → 4
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_unary_arith_raw(
        b"{\"x\":3.7}", "x", UnaryOp::Floor, &[(BinOp::Add, 1.0)],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![4.0]);
}

#[test]
fn raw_field_unary_arith_numeric_op_non_numeric_field_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_unary_arith_raw(
        b"{\"x\":\"hi\"}", "x", UnaryOp::Floor, &[(BinOp::Add, 1.0)],
        |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_unary_arith_unsupported_unary_bails() {
    for uop in [UnaryOp::ToString, UnaryOp::Type, UnaryOp::Explode] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_unary_arith_raw(
            b"{\"x\":1}", "x", uop, &[(BinOp::Add, 0.0)], |n| emitted.push(n),
        );
        assert!(matches!(outcome, RawApplyOutcome::Bail), "uop={:?}", uop);
    }
}

#[test]
fn raw_field_unary_arith_non_arith_step_bails() {
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_unary_arith_raw(
        b"{\"x\":1}", "x", UnaryOp::Abs, &[(BinOp::Eq, 1.0)], |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_field_unary_arith_div_by_zero_bails() {
    // ((.x | abs) / 0) → inf → Bail (was emitting saturated 1.797e+308 pre-fix)
    let mut emitted: Vec<f64> = Vec::new();
    let outcome = apply_field_unary_arith_raw(
        b"{\"x\":4}", "x", UnaryOp::Abs, &[(BinOp::Div, 0.0)], |n| emitted.push(n),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_unary_arith_non_object_input_bails() {
    // Pre-existing #83-class bug: prior fast path returned `0` for length on
    // non-object roots instead of jq's `Cannot index <type>` error.
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<f64> = Vec::new();
        let outcome = apply_field_unary_arith_raw(
            raw, "x", UnaryOp::Length, &[(BinOp::Add, 0.0)], |n| emitted.push(n),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `.x <cmp> .y` — numeric comparison fast path. Helper Bails on missing /
// non-numeric / non-comparison-op / non-object so cross-type equality (e.g.
// `null == null`) doesn't silently take the numeric branch.

#[test]
fn raw_field_field_cmp_handles_all_ops() {
    for (op, expected) in [
        (BinOp::Gt, true),  // 5 > 3
        (BinOp::Lt, false),
        (BinOp::Ge, true),
        (BinOp::Le, false),
        (BinOp::Eq, false),
        (BinOp::Ne, true),
    ] {
        let mut emitted: Vec<bool> = Vec::new();
        let outcome = apply_field_field_cmp_raw(
            b"{\"x\":5,\"y\":3}",
            "x",
            "y",
            op,
            |b| emitted.push(b),
        );
        assert!(matches!(outcome, RawApplyOutcome::Emit));
        assert_eq!(emitted, vec![expected], "op={:?}", op);
    }
}

#[test]
fn raw_field_field_cmp_non_cmp_op_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_field_field_cmp_raw(
        b"{\"x\":1,\"y\":2}",
        "x",
        "y",
        BinOp::Add,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_field_cmp_field_missing_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_field_field_cmp_raw(
        b"{\"x\":1}",
        "x",
        "y",
        BinOp::Eq,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_field_cmp_non_numeric_field_bails() {
    // Both nulls would compare equal in jq, but the helper bails so the
    // generic path takes over (so the fast path doesn't shadow non-numeric
    // equality semantics).
    for inner in [
        &b"{\"x\":\"a\",\"y\":\"a\"}"[..],
        &b"{\"x\":null,\"y\":null}"[..],
        &b"{\"x\":[1],\"y\":[1]}"[..],
        &b"{\"x\":1,\"y\":\"1\"}"[..],
    ] {
        let mut emitted: Vec<bool> = Vec::new();
        let outcome = apply_field_field_cmp_raw(inner, "x", "y", BinOp::Eq, |b| emitted.push(b));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-numeric field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

#[test]
fn raw_field_field_cmp_non_object_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<bool> = Vec::new();
        let outcome = apply_field_field_cmp_raw(raw, "x", "y", BinOp::Eq, |b| emitted.push(b));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for field_field_cmp input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `.field <cmp> N` — numeric comparison vs compile-time constant. Same Bail
// surface as `apply_field_field_cmp_raw` but only one field needs to be
// numeric.

#[test]
fn raw_field_const_cmp_handles_all_ops() {
    for (op, expected) in [
        (BinOp::Gt, true),
        (BinOp::Lt, false),
        (BinOp::Ge, true),
        (BinOp::Le, false),
        (BinOp::Eq, false),
        (BinOp::Ne, true),
    ] {
        let mut emitted: Vec<bool> = Vec::new();
        let outcome = apply_field_const_cmp_raw(
            b"{\"x\":5}",
            "x",
            op,
            3.0,
            |b| emitted.push(b),
        );
        assert!(matches!(outcome, RawApplyOutcome::Emit));
        assert_eq!(emitted, vec![expected], "op={:?}", op);
    }
}

#[test]
fn raw_field_const_cmp_non_cmp_op_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_field_const_cmp_raw(
        b"{\"x\":1}",
        "x",
        BinOp::Add,
        2.0,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_const_cmp_field_missing_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_field_const_cmp_raw(
        b"{\"y\":1}",
        "x",
        BinOp::Eq,
        1.0,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_field_const_cmp_non_numeric_field_bails() {
    for inner in [
        &b"{\"x\":\"hi\"}"[..],
        &b"{\"x\":null}"[..],
        &b"{\"x\":[1]}"[..],
    ] {
        let mut emitted: Vec<bool> = Vec::new();
        let outcome =
            apply_field_const_cmp_raw(inner, "x", BinOp::Eq, 1.0, |b| emitted.push(b));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-numeric field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

#[test]
fn raw_field_const_cmp_non_object_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<bool> = Vec::new();
        let outcome =
            apply_field_const_cmp_raw(raw, "x", BinOp::Eq, 1.0, |b| emitted.push(b));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for field_const_cmp input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `.field <arith_chain> <cmp> <threshold>` — fold a `(BinOp, f64)` chain over
// a numeric field, then compare against `threshold`.

#[test]
fn raw_arith_chain_cmp_emits_correct_boolean() {
    // (5 * 2) - 1 == 9 → true
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_arith_chain_cmp_raw(
        b"{\"x\":5}",
        "x",
        &[(BinOp::Mul, 2.0), (BinOp::Sub, 1.0)],
        BinOp::Eq,
        9.0,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(emitted, vec![true]);
}

#[test]
fn raw_arith_chain_cmp_field_missing_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_arith_chain_cmp_raw(
        b"{\"y\":5}",
        "x",
        &[(BinOp::Add, 1.0)],
        BinOp::Eq,
        6.0,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_arith_chain_cmp_non_numeric_field_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_arith_chain_cmp_raw(
        b"{\"x\":\"hi\"}",
        "x",
        &[(BinOp::Add, 1.0)],
        BinOp::Eq,
        2.0,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_arith_chain_cmp_non_arith_op_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_arith_chain_cmp_raw(
        b"{\"x\":5}",
        "x",
        &[(BinOp::Eq, 5.0)],
        BinOp::Eq,
        1.0,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_arith_chain_cmp_non_cmp_op_bails() {
    let mut emitted: Vec<bool> = Vec::new();
    let outcome = apply_arith_chain_cmp_raw(
        b"{\"x\":5}",
        "x",
        &[(BinOp::Add, 1.0)],
        BinOp::Add,
        6.0,
        |b| emitted.push(b),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(emitted.is_empty());
}

#[test]
fn raw_arith_chain_cmp_non_object_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut emitted: Vec<bool> = Vec::new();
        let outcome = apply_arith_chain_cmp_raw(
            raw,
            "x",
            &[(BinOp::Add, 1.0)],
            BinOp::Eq,
            2.0,
            |b| emitted.push(b),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for arith_chain_cmp input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(emitted.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `select(.field <cmp> N)` — predicate fast path. The helper passes the raw
// record bytes to the closure when the predicate fires; non-object inputs and
// non-numeric / missing fields Bail to the generic path so jq's total-order
// rules and #199 type errors surface.

#[test]
fn raw_select_cmp_emits_record_when_predicate_fires() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_cmp_raw(
        b"{\"x\":5}",
        "x",
        BinOp::Gt,
        3.0,
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(seen, vec![b"{\"x\":5}".to_vec()]);
}

#[test]
fn raw_select_cmp_no_output_when_predicate_fails() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_cmp_raw(
        b"{\"x\":5}",
        "x",
        BinOp::Lt,
        3.0,
        |r| seen.push(r.to_vec()),
    );
    // Helper still returns Emit (it handled the input — predicate just
    // didn't fire), but the closure is never invoked.
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert!(seen.is_empty());
}

#[test]
fn raw_select_cmp_non_object_bails_for_type_error() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let outcome =
            apply_select_cmp_raw(raw, "x", BinOp::Eq, 1.0, |r| seen.push(r.to_vec()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for select_cmp input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(seen.is_empty());
    }
}

#[test]
fn raw_select_cmp_field_missing_bails_for_total_order() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_cmp_raw(
        b"{\"y\":5}",
        "x",
        BinOp::Lt,
        0.0,
        |r| seen.push(r.to_vec()),
    );
    // Bails so the generic path applies jq's total-order
    // (null < 0 == true → emits the record).
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(seen.is_empty());
}

#[test]
fn raw_select_cmp_non_numeric_field_bails_for_total_order() {
    for inner in [&b"{\"x\":\"hi\"}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1]}"[..]] {
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let outcome =
            apply_select_cmp_raw(inner, "x", BinOp::Eq, 1.0, |r| seen.push(r.to_vec()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-numeric field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(seen.is_empty());
    }
}

#[test]
fn raw_select_cmp_non_cmp_op_bails() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_cmp_raw(
        b"{\"x\":5}",
        "x",
        BinOp::Add,
        1.0,
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(seen.is_empty());
}

// ---------------------------------------------------------------------------
// `select(.field == null)` / `select(.field != null)` — null-equality
// predicate fast path. Missing field counts as null.

#[test]
fn raw_select_field_null_eq_emits_when_null_or_missing() {
    for input in [&b"{\"x\":null}"[..], &b"{\"y\":1}"[..]] {
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let outcome =
            apply_select_field_null_raw(input, "x", true, |r| seen.push(r.to_vec()));
        assert!(matches!(outcome, RawApplyOutcome::Emit));
        assert_eq!(
            seen,
            vec![input.to_vec()],
            "input {:?}",
            std::str::from_utf8(input).unwrap()
        );
    }
}

#[test]
fn raw_select_field_null_eq_skips_when_field_present_non_null() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_field_null_raw(
        b"{\"x\":1}",
        "x",
        true,
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert!(seen.is_empty());
}

#[test]
fn raw_select_field_null_ne_emits_when_field_present_non_null() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_field_null_raw(
        b"{\"x\":1}",
        "x",
        false,
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(seen, vec![b"{\"x\":1}".to_vec()]);
}

#[test]
fn raw_select_field_null_ne_skips_when_null_or_missing() {
    for input in [&b"{\"x\":null}"[..], &b"{\"y\":1}"[..]] {
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let outcome =
            apply_select_field_null_raw(input, "x", false, |r| seen.push(r.to_vec()));
        assert!(matches!(outcome, RawApplyOutcome::Emit));
        assert!(seen.is_empty());
    }
}

#[test]
fn raw_select_field_null_non_object_bails_for_type_error() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"true".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let outcome =
            apply_select_field_null_raw(raw, "x", true, |r| seen.push(r.to_vec()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for select_field_null input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(seen.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `select(.field <arith_chain> <cmp> N)` — same Bail discipline as
// select_cmp plus the arithmetic chain. Fixes silent-skip on non-object
// (jq raises) and non-numeric field (jq raises for non-add ops).

#[test]
fn raw_select_arith_cmp_emits_record_when_predicate_fires() {
    // (5 * 2) > 8 → true
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_arith_cmp_raw(
        b"{\"x\":5}",
        "x",
        &[(BinOp::Mul, 2.0)],
        BinOp::Gt,
        8.0,
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(seen, vec![b"{\"x\":5}".to_vec()]);
}

#[test]
fn raw_select_arith_cmp_skips_when_predicate_fails() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_arith_cmp_raw(
        b"{\"x\":1}",
        "x",
        &[(BinOp::Mul, 2.0)],
        BinOp::Gt,
        8.0,
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert!(seen.is_empty());
}

#[test]
fn raw_select_arith_cmp_non_object_bails_for_type_error() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_select_arith_cmp_raw(
            raw,
            "x",
            &[(BinOp::Add, 1.0)],
            BinOp::Gt,
            5.0,
            |r| seen.push(r.to_vec()),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for select_arith_cmp input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(seen.is_empty());
    }
}

#[test]
fn raw_select_arith_cmp_field_missing_bails() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_arith_cmp_raw(
        b"{\"y\":5}",
        "x",
        &[(BinOp::Add, 1.0)],
        BinOp::Gt,
        5.0,
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(seen.is_empty());
}

#[test]
fn raw_select_arith_cmp_non_numeric_field_bails() {
    for inner in [&b"{\"x\":\"hi\"}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1]}"[..]] {
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_select_arith_cmp_raw(
            inner,
            "x",
            &[(BinOp::Add, 1.0)],
            BinOp::Gt,
            5.0,
            |r| seen.push(r.to_vec()),
        );
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-numeric field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(seen.is_empty());
    }
}

#[test]
fn raw_select_arith_cmp_non_arith_op_bails() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_arith_cmp_raw(
        b"{\"x\":5}",
        "x",
        &[(BinOp::Eq, 5.0)],
        BinOp::Gt,
        5.0,
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(seen.is_empty());
}

#[test]
fn raw_select_arith_cmp_non_cmp_op_bails() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_arith_cmp_raw(
        b"{\"x\":5}",
        "x",
        &[(BinOp::Add, 1.0)],
        BinOp::Add,
        5.0,
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(seen.is_empty());
}

// ---------------------------------------------------------------------------
// `select(.field <cmp> "value")` — string equality / ordering predicate.
// Eq/Ne can byte-compare on plain strings and non-strings (jq's cross-type
// equality is "never equal" for non-string vs string, matching the byte
// comparison result). Ordering ops (Gt/Lt/Ge/Le) require an escape-free
// quoted string — anything else bails so jq's cross-type ordering applies.

#[test]
fn raw_select_str_eq_emits_match() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_str_raw(
        b"{\"x\":\"foo\"}",
        "x",
        BinOp::Eq,
        "foo",
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(seen, vec![b"{\"x\":\"foo\"}".to_vec()]);
}

#[test]
fn raw_select_str_eq_skips_when_different() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_str_raw(
        b"{\"x\":\"bar\"}",
        "x",
        BinOp::Eq,
        "foo",
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert!(seen.is_empty());
}

#[test]
fn raw_select_str_eq_non_string_field_emits_when_ne() {
    // Eq on non-string field byte-compares: 42 != "foo" so Ne fires.
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_str_raw(
        b"{\"x\":42}",
        "x",
        BinOp::Ne,
        "foo",
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(seen, vec![b"{\"x\":42}".to_vec()]);
}

#[test]
fn raw_select_str_eq_escape_bearing_string_bails() {
    // The raw scanner can't safely byte-compare strings with `\` escapes
    // (they'd compare unequal even when the decoded text matches), so
    // bail. Build the bytes explicitly to avoid editor-side escape
    // interpretation.
    let raw: Vec<u8> = b"{\"x\":\"a\\nb\"}".to_vec();
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome =
        apply_select_str_raw(&raw, "x", BinOp::Eq, "a\nb", |r| seen.push(r.to_vec()));
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(seen.is_empty());
}

#[test]
fn raw_select_str_ordering_emits_on_match() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_str_raw(
        b"{\"x\":\"foo\"}",
        "x",
        BinOp::Gt,
        "bar",
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(seen, vec![b"{\"x\":\"foo\"}".to_vec()]);
}

#[test]
fn raw_select_str_ordering_non_string_field_bails() {
    // jq's cross-type ordering: array > string. Bail so generic decides.
    for inner in [&b"{\"x\":42}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1]}"[..]] {
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let outcome =
            apply_select_str_raw(inner, "x", BinOp::Gt, "bar", |r| seen.push(r.to_vec()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-string field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(seen.is_empty());
    }
}

#[test]
fn raw_select_str_ordering_escape_bearing_field_bails() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_str_raw(
        br#"{"x":"a\nb"}"#,
        "x",
        BinOp::Gt,
        "bar",
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(seen.is_empty());
}

#[test]
fn raw_select_str_field_missing_bails() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_str_raw(
        b"{\"y\":\"foo\"}",
        "x",
        BinOp::Eq,
        "foo",
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(seen.is_empty());
}

#[test]
fn raw_select_str_non_object_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let outcome =
            apply_select_str_raw(raw, "x", BinOp::Eq, "foo", |r| seen.push(r.to_vec()));
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for select_str input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(seen.is_empty());
    }
}

#[test]
fn raw_select_str_non_cmp_op_bails() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_str_raw(
        b"{\"x\":\"foo\"}",
        "x",
        BinOp::Add,
        "foo",
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(seen.is_empty());
}

// ---------------------------------------------------------------------------
// `select(.field | startswith/endswith/contains("arg"))` — string-test
// predicate fast path. jq raises `<builtin>() requires string inputs` on
// non-string fields, so the helper bails on every non-string-no-escape shape
// (fixing the silent-skip bug in the old apply-site).

#[test]
fn raw_select_str_test_startswith_emits_match() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_str_test_raw(
        b"{\"x\":\"foobar\"}",
        "x",
        "startswith",
        "foo",
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(seen, vec![b"{\"x\":\"foobar\"}".to_vec()]);
}

#[test]
fn raw_select_str_test_endswith_emits_match() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_str_test_raw(
        b"{\"x\":\"foobar\"}",
        "x",
        "endswith",
        "bar",
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(seen, vec![b"{\"x\":\"foobar\"}".to_vec()]);
}

#[test]
fn raw_select_str_test_contains_emits_match() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_str_test_raw(
        b"{\"x\":\"foobar\"}",
        "x",
        "contains",
        "oba",
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(seen, vec![b"{\"x\":\"foobar\"}".to_vec()]);
}

#[test]
fn raw_select_str_test_skips_when_predicate_fails() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_str_test_raw(
        b"{\"x\":\"foobar\"}",
        "x",
        "startswith",
        "zzz",
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert!(seen.is_empty());
}

#[test]
fn raw_select_str_test_unknown_builtin_bails() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_str_test_raw(
        b"{\"x\":\"foo\"}",
        "x",
        "test",
        "foo",
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(seen.is_empty());
}

#[test]
fn raw_select_str_test_field_missing_bails() {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_str_test_raw(
        b"{\"y\":\"hi\"}",
        "x",
        "startswith",
        "h",
        |r| seen.push(r.to_vec()),
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(seen.is_empty());
}

#[test]
fn raw_select_str_test_non_string_field_bails() {
    for inner in [&b"{\"x\":42}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1]}"[..]] {
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_select_str_test_raw(inner, "x", "startswith", "h", |r| {
            seen.push(r.to_vec())
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-string field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
        assert!(seen.is_empty());
    }
}

#[test]
fn raw_select_str_test_escape_bearing_field_bails() {
    let raw: Vec<u8> = b"{\"x\":\"a\\nb\"}".to_vec();
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let outcome = apply_select_str_test_raw(&raw, "x", "startswith", "a", |r| {
        seen.push(r.to_vec())
    });
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(seen.is_empty());
}

#[test]
fn raw_select_str_test_non_object_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let outcome = apply_select_str_test_raw(raw, "x", "startswith", "h", |r| {
            seen.push(r.to_vec())
        });
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for select_str_test input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(seen.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `.field |= length` — string-length update fast path. Only commits when the
// field value is a JSON string; everything else bails so the generic path can
// apply jq's polymorphic length (array length, object key count, abs(number),
// 0 for null).

#[test]
fn raw_field_update_length_emits_string_codepoint_count() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_length_raw(b"{\"x\":\"hello\"}", "x", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":5}");
}

#[test]
fn raw_field_update_length_counts_codepoints_not_bytes() {
    // 3 multi-byte chars (UTF-8 = 9 bytes), but length is 3.
    let mut buf = Vec::new();
    let outcome = apply_field_update_length_raw(
        "{\"x\":\"あいう\"}".as_bytes(),
        "x",
        &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), "{\"x\":3}".as_bytes());
}

#[test]
fn raw_field_update_length_field_missing_bails() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_length_raw(b"{\"y\":\"hi\"}", "x", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_field_update_length_non_string_field_bails() {
    // jq's length on non-strings: array→len, object→key-count,
    // number→abs, null→0. The fast path bails so generic produces the
    // right verdict.
    for inner in [&b"{\"x\":42}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1,2]}"[..]] {
        let mut buf = Vec::new();
        let outcome = apply_field_update_length_raw(inner, "x", &mut buf);
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-string field input {:?}, got {:?}",
            std::str::from_utf8(inner).unwrap(),
            outcome,
        );
    }
}

#[test]
fn raw_field_update_length_non_object_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut buf = Vec::new();
        let outcome = apply_field_update_length_raw(raw, "x", &mut buf);
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for field_update_length input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
    }
}

// ---------------------------------------------------------------------------
// `.field |= tostring` — string-coerce update fast path. Strings pass
// through; numbers / booleans / null are wrapped in quotes. Arrays and
// objects bail since their stringification needs the recursive JSON encoder.

#[test]
fn raw_field_update_tostring_string_passthrough() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_tostring_raw(b"{\"x\":\"hi\"}", "x", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":\"hi\"}");
}

#[test]
fn raw_field_update_tostring_number_wraps_in_quotes() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_tostring_raw(b"{\"x\":42}", "x", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":\"42\"}");
}

#[test]
fn raw_field_update_tostring_null_wraps_in_quotes() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_tostring_raw(b"{\"x\":null}", "x", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":\"null\"}");
}

#[test]
fn raw_field_update_tostring_field_missing_bails() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_tostring_raw(b"{\"y\":\"hi\"}", "x", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_field_update_tostring_non_object_input_bails() {
    for raw in [
        b"42".as_slice(),
        b"\"hi\"".as_slice(),
        b"null".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let mut buf = Vec::new();
        let outcome = apply_field_update_tostring_raw(raw, "x", &mut buf);
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for field_update_tostring input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
    }
}

// ---------------------------------------------------------------------------
// `.field |= test("p")` — boolean-coerce predicate update.

#[test]
fn raw_field_update_test_emits_match() {
    let re = regex::Regex::new("^foo").unwrap();
    let mut buf = Vec::new();
    let outcome = apply_field_update_test_raw(b"{\"x\":\"foobar\"}", "x", &re, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":true}");
}

#[test]
fn raw_field_update_test_emits_false_on_no_match() {
    let re = regex::Regex::new("^zz").unwrap();
    let mut buf = Vec::new();
    let outcome = apply_field_update_test_raw(b"{\"x\":\"foobar\"}", "x", &re, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":false}");
}

#[test]
fn raw_field_update_test_non_string_field_bails() {
    let re = regex::Regex::new(".").unwrap();
    for inner in [&b"{\"x\":42}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1]}"[..]] {
        let mut buf = Vec::new();
        let outcome = apply_field_update_test_raw(inner, "x", &re, &mut buf);
        assert!(matches!(outcome, RawApplyOutcome::Bail));
    }
}

#[test]
fn raw_field_update_test_non_object_input_bails() {
    let re = regex::Regex::new(".").unwrap();
    for raw in [b"42".as_slice(), b"\"hi\"".as_slice(), b"null".as_slice()] {
        let mut buf = Vec::new();
        let outcome = apply_field_update_test_raw(raw, "x", &re, &mut buf);
        assert!(matches!(outcome, RawApplyOutcome::Bail));
    }
}

// ---------------------------------------------------------------------------
// `.field |= sub/gsub("p"; "r")` — regex replacement update.

#[test]
fn raw_field_update_gsub_global_emits_replaced() {
    let re = regex::Regex::new("a").unwrap();
    let mut buf = Vec::new();
    let outcome = apply_field_update_gsub_raw(
        b"{\"x\":\"banana\"}",
        "x",
        &re,
        "X",
        true,
        &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":\"bXnXnX\"}");
}

#[test]
fn raw_field_update_gsub_non_global_replaces_first() {
    let re = regex::Regex::new("a").unwrap();
    let mut buf = Vec::new();
    let outcome = apply_field_update_gsub_raw(
        b"{\"x\":\"banana\"}",
        "x",
        &re,
        "X",
        false,
        &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":\"bXnana\"}");
}

#[test]
fn raw_field_update_gsub_non_string_field_bails() {
    let re = regex::Regex::new("a").unwrap();
    for inner in [&b"{\"x\":42}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1]}"[..]] {
        let mut buf = Vec::new();
        let outcome =
            apply_field_update_gsub_raw(inner, "x", &re, "X", true, &mut buf);
        assert!(matches!(outcome, RawApplyOutcome::Bail));
    }
}

#[test]
fn raw_field_update_gsub_non_object_input_bails() {
    let re = regex::Regex::new("a").unwrap();
    for raw in [b"42".as_slice(), b"\"hi\"".as_slice(), b"null".as_slice()] {
        let mut buf = Vec::new();
        let outcome = apply_field_update_gsub_raw(raw, "x", &re, "X", true, &mut buf);
        assert!(matches!(outcome, RawApplyOutcome::Bail));
    }
}

// ---------------------------------------------------------------------------
// `.field |= ascii_downcase / ascii_upcase` — ASCII case update.

#[test]
fn raw_field_update_case_downcase_emits() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_case_raw(b"{\"x\":\"AbC\"}", "x", false, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    let s = std::str::from_utf8(&buf).unwrap();
    assert!(s.contains("\"x\":\"abc\""), "got: {}", s);
}

#[test]
fn raw_field_update_case_upcase_emits() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_case_raw(b"{\"x\":\"AbC\"}", "x", true, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    let s = std::str::from_utf8(&buf).unwrap();
    assert!(s.contains("\"x\":\"ABC\""), "got: {}", s);
}

#[test]
fn raw_field_update_case_non_string_field_bails() {
    for inner in [&b"{\"x\":42}"[..], &b"{\"x\":null}"[..], &b"{\"x\":[1]}"[..]] {
        let mut buf = Vec::new();
        let outcome = apply_field_update_case_raw(inner, "x", false, &mut buf);
        assert!(matches!(outcome, RawApplyOutcome::Bail));
    }
}

#[test]
fn raw_field_update_case_non_object_input_bails() {
    for raw in [b"42".as_slice(), b"\"hi\"".as_slice(), b"null".as_slice()] {
        let mut buf = Vec::new();
        let outcome = apply_field_update_case_raw(raw, "x", false, &mut buf);
        assert!(matches!(outcome, RawApplyOutcome::Bail));
    }
}

// ---------------------------------------------------------------------------
// Six more `.field |=` thin-wrappers: split_first / split_last / trim /
// slice / str_map / str_concat. All share the same Bail discipline:
// non-object, missing/non-string field.

#[test]
fn raw_field_update_split_first_emits_prefix() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_split_first_raw(b"{\"x\":\"a/b/c\"}", "x", "/", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":\"a\"}");
}

#[test]
fn raw_field_update_split_first_non_object_bails() {
    for raw in [b"42".as_slice(), b"null".as_slice(), b"[1]".as_slice()] {
        let mut buf = Vec::new();
        let outcome = apply_field_update_split_first_raw(raw, "x", "/", &mut buf);
        assert!(matches!(outcome, RawApplyOutcome::Bail));
    }
}

#[test]
fn raw_field_update_split_last_emits_suffix() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_split_last_raw(b"{\"x\":\"a/b/c\"}", "x", "/", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":\"c\"}");
}

#[test]
fn raw_field_update_split_last_non_string_field_bails() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_split_last_raw(b"{\"x\":42}", "x", "/", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_field_update_trim_ltrim_emits() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_trim_raw(b"{\"x\":\"abcabc\"}", "x", "ab", false, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":\"cabc\"}");
}

#[test]
fn raw_field_update_trim_rtrim_emits() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_trim_raw(b"{\"x\":\"abcabc\"}", "x", "bc", true, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":\"abca\"}");
}

#[test]
fn raw_field_update_trim_non_object_bails() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_trim_raw(b"42", "x", "ab", false, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_field_update_slice_emits() {
    let mut buf = Vec::new();
    let outcome =
        apply_field_update_slice_raw(b"{\"x\":\"abcdef\"}", "x", Some(1), Some(4), &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":\"bcd\"}");
}

#[test]
fn raw_field_update_slice_non_string_field_bails() {
    let mut buf = Vec::new();
    let outcome =
        apply_field_update_slice_raw(b"{\"x\":[1,2,3]}", "x", Some(1), Some(2), &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_field_update_str_map_emits_then_branch() {
    // .x |= if . == "yes" then "Y" else "N" end
    let mut buf = Vec::new();
    let outcome = apply_field_update_str_map_raw(
        b"{\"x\":\"yes\"}",
        "x",
        b"yes",
        b"\"Y\"",
        b"\"N\"",
        &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":\"Y\"}");
}

#[test]
fn raw_field_update_str_map_emits_else_branch() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_str_map_raw(
        b"{\"x\":\"no\"}",
        "x",
        b"yes",
        b"\"Y\"",
        b"\"N\"",
        &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":\"N\"}");
}

#[test]
fn raw_field_update_str_map_non_object_bails() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_str_map_raw(
        b"\"yes\"",
        "x",
        b"yes",
        b"\"Y\"",
        b"\"N\"",
        &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

#[test]
fn raw_field_update_str_concat_wraps_value() {
    let mut buf = Vec::new();
    let outcome = apply_field_update_str_concat_raw(
        b"{\"x\":\"hi\"}",
        "x",
        b"<",
        b">",
        &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"x\":\"<hi>\"}");
}

#[test]
fn raw_field_update_str_concat_non_string_field_bails() {
    let mut buf = Vec::new();
    let outcome =
        apply_field_update_str_concat_raw(b"{\"x\":42}", "x", b"<", b">", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
}

// ---------------------------------------------------------------------------
// `del(.field)` / `del(.a, .b, ...)` — field deletion. Object input
// commits; non-object input bails (jq's `del(.x)` on `null` is `null`,
// on numbers it raises — both routed through generic).

#[test]
fn raw_del_field_emits_object_without_field() {
    let mut buf = Vec::new();
    let outcome = apply_del_field_raw(b"{\"x\":1,\"y\":2}", "x", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"y\":2}");
}

#[test]
fn raw_del_field_field_absent_passthrough() {
    // jq's `del(.x)` on `{}` returns `{}` — no error.
    let mut buf = Vec::new();
    let outcome = apply_del_field_raw(b"{\"y\":2}", "x", &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"y\":2}");
}

#[test]
fn raw_del_field_non_object_bails() {
    for raw in [b"42".as_slice(), b"\"hi\"".as_slice(), b"null".as_slice(), b"[1]".as_slice()] {
        let mut buf = Vec::new();
        let outcome = apply_del_field_raw(raw, "x", &mut buf);
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for del_field input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
    }
}

#[test]
fn raw_del_fields_emits_object_without_fields() {
    let mut buf = Vec::new();
    let outcome = apply_del_fields_raw(b"{\"x\":1,\"y\":2,\"z\":3}", &["x", "z"], &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"y\":2}");
}

#[test]
fn raw_del_fields_missing_silently_skipped() {
    let mut buf = Vec::new();
    let outcome = apply_del_fields_raw(b"{\"y\":2}", &["x", "z"], &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"{\"y\":2}");
}

#[test]
fn raw_del_fields_non_object_bails() {
    for raw in [b"42".as_slice(), b"null".as_slice(), b"[1]".as_slice()] {
        let mut buf = Vec::new();
        let outcome = apply_del_fields_raw(raw, &["x"], &mut buf);
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for del_fields input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
    }
}

fn te_interp_key_eq_value() -> Vec<(bool, String)> {
    vec![
        (false, "key".to_string()),
        (true, "=".to_string()),
        (false, "value".to_string()),
    ]
}

#[test]
fn raw_to_entries_each_interp_object_emits() {
    let parts = te_interp_key_eq_value();
    let mut buf = Vec::new();
    let outcome = apply_to_entries_each_interp_raw(b"{\"a\":1,\"b\":\"x\"}", &parts, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"\"a=1\"\n\"b=x\"\n");
}

#[test]
fn raw_to_entries_each_interp_empty_object_emits_nothing() {
    let parts = te_interp_key_eq_value();
    let mut buf = Vec::new();
    let outcome = apply_to_entries_each_interp_raw(b"{}", &parts, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert!(buf.is_empty());
}

#[test]
fn raw_to_entries_each_interp_non_object_bails() {
    let parts = te_interp_key_eq_value();
    for raw in [b"[1,2]".as_slice(), b"42".as_slice(), b"null".as_slice(), b"\"s\"".as_slice()] {
        let mut buf = Vec::new();
        let outcome = apply_to_entries_each_interp_raw(raw, &parts, &mut buf);
        assert!(
            matches!(outcome, RawApplyOutcome::Bail),
            "expected Bail for non-object input {:?}, got {:?}",
            std::str::from_utf8(raw).unwrap(),
            outcome,
        );
        assert!(buf.is_empty(), "buf should be untouched on Bail");
    }
}

#[test]
fn raw_to_entries_each_interp_non_canonical_number_bails() {
    let parts = te_interp_key_eq_value();
    let mut buf = Vec::new();
    let outcome = apply_to_entries_each_interp_raw(b"{\"x\":1e10}", &parts, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_to_entries_each_interp_leading_plus_number_bails() {
    let parts = te_interp_key_eq_value();
    let mut buf = Vec::new();
    let outcome = apply_to_entries_each_interp_raw(b"{\"x\":+5}", &parts, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_to_entries_each_interp_nested_array_value_bails() {
    let parts = te_interp_key_eq_value();
    let mut buf = Vec::new();
    let outcome = apply_to_entries_each_interp_raw(b"{\"x\":[1,2]}", &parts, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_to_entries_each_interp_nested_object_value_bails() {
    let parts = te_interp_key_eq_value();
    let mut buf = Vec::new();
    let outcome = apply_to_entries_each_interp_raw(b"{\"x\":{\"y\":1}}", &parts, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_to_entries_each_interp_escaped_value_string_bails() {
    let parts = te_interp_key_eq_value();
    let mut buf = Vec::new();
    let outcome = apply_to_entries_each_interp_raw(b"{\"x\":\"a\\u00e9\"}", &parts, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_to_entries_each_interp_escaped_key_bails() {
    let parts = te_interp_key_eq_value();
    let mut buf = Vec::new();
    let outcome = apply_to_entries_each_interp_raw(b"{\"a\\nb\":1}", &parts, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert!(buf.is_empty());
}

#[test]
fn raw_to_entries_each_interp_bool_null_values_emit() {
    let parts = te_interp_key_eq_value();
    let mut buf = Vec::new();
    let outcome = apply_to_entries_each_interp_raw(
        b"{\"a\":true,\"b\":false,\"c\":null}",
        &parts,
        &mut buf,
    );
    assert!(matches!(outcome, RawApplyOutcome::Emit));
    assert_eq!(buf.as_slice(), b"\"a=true\"\n\"b=false\"\n\"c=null\"\n");
}

#[test]
fn raw_to_entries_each_interp_preserves_existing_buf_on_bail() {
    let parts = te_interp_key_eq_value();
    let mut buf = b"prefix-".to_vec();
    let outcome = apply_to_entries_each_interp_raw(b"{\"x\":1e10}", &parts, &mut buf);
    assert!(matches!(outcome, RawApplyOutcome::Bail));
    assert_eq!(buf.as_slice(), b"prefix-");
}
