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
    apply_field_const_cmp_raw, apply_field_field_alternative_raw, apply_field_field_cmp_raw,
    apply_field_format_raw, apply_field_gsub_raw, apply_field_ltrimstr_tonumber_raw,
    apply_field_match_raw, apply_field_scan_raw, apply_field_str_builtin_raw,
    apply_field_str_concat_raw, apply_field_str_reverse_raw, apply_field_test_raw,
    apply_full_object_fields_raw, apply_has_field_raw, apply_has_multi_field_raw,
    apply_multi_field_access_raw, apply_nested_field_access_raw, apply_object_compute_raw,
    apply_field_update_length_raw, apply_select_arith_cmp_raw, apply_select_cmp_raw,
    apply_select_field_null_raw, apply_select_str_raw, apply_select_str_test_raw,
};
use jq_jit::interpreter::Filter;
use jq_jit::ir::BinOp;
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
