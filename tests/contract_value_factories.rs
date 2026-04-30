//! Unit coverage for the Value factories introduced alongside #84.

use jq_jit::value::Value;

#[test]
fn object_from_pairs_dedupes_last_wins_first_position() {
    // { a: 1, b: 2, a: 3 } → { a: 3, b: 2 } with `a` kept at position 0.
    let obj = Value::object_from_pairs(vec![
        ("a", Value::number(1.0)),
        ("b", Value::number(2.0)),
        ("a", Value::number(3.0)),
    ]);
    let Value::Obj(rc) = obj else { panic!("expected Obj") };
    let keys: Vec<&str> = rc.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, vec!["a", "b"]);
    let (_, a_val) = rc.iter().find(|(k, _)| k.as_str() == "a").unwrap();
    let Value::Num(n, _) = a_val else { panic!("expected Num") };
    assert_eq!(*n, 3.0);
}

#[test]
fn object_from_pairs_empty() {
    let obj = Value::object_from_pairs::<&str, _>(Vec::<(&str, Value)>::new());
    let Value::Obj(rc) = obj else { panic!("expected Obj") };
    assert_eq!(rc.len(), 0);
}

#[test]
fn object_from_normalized_pairs_trusts_caller() {
    let obj = Value::object_from_normalized_pairs(vec![
        ("x", Value::number(1.0)),
        ("y", Value::number(2.0)),
    ]);
    let Value::Obj(rc) = obj else { panic!("expected Obj") };
    assert_eq!(rc.len(), 2);
}

#[test]
fn number_factory_drops_repr() {
    let n = Value::number(1.0);
    match n {
        Value::Num(v, repr) => {
            assert_eq!(v, 1.0);
            assert!(repr.is_none(), "bare number should have no repr");
        }
        _ => panic!("expected Num"),
    }
}

#[test]
fn number_with_repr_factory_preserves_repr() {
    let raw: std::rc::Rc<str> = std::rc::Rc::from("1.0");
    let n = Value::number_with_repr(1.0, raw.clone());
    match n {
        Value::Num(v, repr) => {
            assert_eq!(v, 1.0);
            assert_eq!(repr.as_deref(), Some("1.0"));
        }
        _ => panic!("expected Num"),
    }
}
