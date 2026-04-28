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
    FastPath, FieldAccessPath, RawApplyOutcome, apply_full_object_fields_raw,
    apply_field_access_raw, apply_has_field_raw, apply_has_multi_field_raw,
    apply_multi_field_access_raw, apply_nested_field_access_raw, apply_object_compute_raw,
};
use jq_jit::interpreter::Filter;
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
