//! Coverage for the [`jq_jit::fast_path::FastPath`] contract introduced
//! in #83. Verifies that:
//!
//! - the pilot `FieldAccessPath` succeeds on object and null inputs,
//! - the pilot bails with `None` on every other input type (so the
//!   generic path can raise the jq-compatible "Cannot index …" error
//!   instead of leaking null-masking divergence #50),
//! - `Filter::try_typed_fast_path` routes `.field` filters through the
//!   pilot and returns `None` for filters that aren't yet migrated.

use jq_jit::fast_path::{FastPath, FieldAccessPath};
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
