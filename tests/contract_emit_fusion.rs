//! #1058: output fusion. A tail-position static construct compiled with the
//! host's output-fusion knob stages its compact JSON bytes in the JIT env
//! emit buffer (drained by the host) instead of streaming Values through
//! the yield callback. These contracts pin the three load-bearing
//! properties:
//!
//!  1. byte parity — drained bytes are identical to serializing the
//!     non-fused program's outputs with the shared compact serializer,
//!     on both backends (JitOp interpreter and Cranelift);
//!  2. all-or-nothing — a program mixing fused and callback yields
//!     re-lowers without fusion, so output order can never be scrambled;
//!  3. opt-in — filters compiled without the knob never stage bytes.

use jq_jit::interpreter::Filter;
use jq_jit::jit;
use jq_jit::value::{json_to_value, push_compact_line, Value};

/// Corpus of fusable filters × inputs covering the plan-builder surface:
/// nesting, literal folding (repr canonicalization, string escapes),
/// string/composite/missing holes, arrays of objects, branches.
const CASES: &[(&str, &str)] = &[
    ("{a: {b: .x}}", r#"{"x":1}"#),
    ("[{a: .x, b: .y}]", r#"{"x":1,"y":2}"#),
    ("{a: .x, b: {c: [.y, {d: .name}], e: 1}}", r#"{"x":1,"y":2,"name":"n"}"#),
    (r#"{a: 1e3, b: [0.1], c: "li\"t\n", d: null}"#, "null"),
    ("{k: .s}", r#"{"s":"a\/bA\t"}"#),
    ("{n: .v}", r#"{"v":1e3}"#),
    ("{o: .v}", r#"{"v":{"q":"x\"y","a":[1,"\/"]}}"#),
    ("{m: .missing}", "{}"),
    ("[{a: .x}, {b: 2}, [3, {c: .y}]]", r#"{"x":1,"y":4}"#),
    ("if .x > 1 then {hit: .x} else empty end", r#"{"x":2}"#),
    ("{s: (.x | tostring), t: [.x * 2]}", r#"{"x":3}"#),
    // scalar pipe prefixes peel into the fused tail
    ("select(.x > 1) | {a: {b: .x}}", r#"{"x":2}"#),
    (".x + 1 | {v: ., w: [.]}", r#"{"x":2}"#),
];

/// Reset the thread-local knob even on panic so a failing assertion does
/// not leak fusion into other tests running on the same thread.
struct KnobGuard;
impl Drop for KnobGuard {
    fn drop(&mut self) {
        jit::set_output_fusion(false);
    }
}

enum Backend {
    Jitop,
    Cranelift,
}

/// Compile `filter` with the given knob state and backend, run it on
/// `input`, and return (values seen by the yield callback, drained bytes).
fn run(filter: &str, input: &str, fused: bool, backend: Backend) -> (Vec<Value>, Vec<u8>) {
    let _guard = KnobGuard;
    jit::set_output_fusion(fused);
    let mut f = Filter::with_options(filter, &[], false).expect("parse");
    match backend {
        Backend::Jitop => {
            f.compile_jitop_program();
            assert!(f.has_jitop_program(), "{filter:?} must flatten");
        }
        Backend::Cranelift => {
            f.compile_jit_with_delegates();
            assert!(f.has_jit(), "{filter:?} must codegen");
        }
    }
    jit::set_output_fusion(false);
    let inp = json_to_value(input).expect("parse input");
    let mut seen = Vec::new();
    // Pre-drain any stale bytes so the assertion below sees only this run.
    let mut stale = Vec::new();
    jit::drain_emit_buf(&mut stale);
    f.execute_cb(&inp, &mut |v| {
        seen.push(v.clone());
        Ok(true)
    })
    .expect("execute");
    let mut drained = Vec::new();
    jit::drain_emit_buf(&mut drained);
    (seen, drained)
}

fn serialize(values: &[Value]) -> Vec<u8> {
    let mut buf = Vec::new();
    for v in values {
        push_compact_line(&mut buf, v);
    }
    buf
}

#[test]
fn fused_bytes_match_serialized_plain_outputs_jitop() {
    for &(filter, input) in CASES {
        let (plain_vals, plain_bytes) = run(filter, input, false, Backend::Jitop);
        assert!(plain_bytes.is_empty(), "{filter:?}: knob off must not stage bytes");
        let (fused_vals, fused_bytes) = run(filter, input, true, Backend::Jitop);
        assert!(fused_vals.is_empty(), "{filter:?}: fused program must not stream via cb");
        assert_eq!(
            String::from_utf8(fused_bytes).unwrap(),
            String::from_utf8(serialize(&plain_vals)).unwrap(),
            "{filter:?} on {input:?}"
        );
    }
}

#[test]
fn fused_bytes_match_serialized_plain_outputs_cranelift() {
    for &(filter, input) in CASES {
        let (plain_vals, _) = run(filter, input, false, Backend::Cranelift);
        let (fused_vals, fused_bytes) = run(filter, input, true, Backend::Cranelift);
        assert!(fused_vals.is_empty(), "{filter:?}: fused program must not stream via cb");
        assert_eq!(
            String::from_utf8(fused_bytes).unwrap(),
            String::from_utf8(serialize(&plain_vals)).unwrap(),
            "{filter:?} on {input:?}"
        );
    }
}

/// A program with both a fused construct and a plain yield site must
/// retreat to the non-fused lowering entirely — partial fusion would let
/// the host's append-after-execute drain reorder outputs.
#[test]
fn mixed_yield_program_retreats_from_fusion() {
    for filter in ["({a: .x}, .x)", "(.x, {a: .x})", "({a: .x}, {b: .x} | .b)"] {
        let (vals, bytes) = run(filter, r#"{"x":7}"#, true, Backend::Jitop);
        assert_eq!(vals.len(), 2, "{filter:?}: both outputs stream via cb");
        assert!(bytes.is_empty(), "{filter:?}: no bytes may be staged");
    }
}

/// A hole error aborts before any byte of the record is emitted: the
/// emit buffer stays empty (whole-record atomicity), and the error
/// surfaces exactly like the non-fused lowering.
#[test]
fn hole_error_stages_no_partial_record() {
    let _guard = KnobGuard;
    jit::set_output_fusion(true);
    let mut f = Filter::with_options("{a: .x, b: (.x + \"s\")}", &[], false).expect("parse");
    f.compile_jitop_program();
    assert!(f.has_jitop_program());
    jit::set_output_fusion(false);
    let inp = json_to_value(r#"{"x":1}"#).expect("parse input");
    let mut stale = Vec::new();
    jit::drain_emit_buf(&mut stale);
    let err = f.execute_cb(&inp, &mut |_| Ok(true)).expect_err("hole must error");
    assert!(err.to_string().contains("cannot be added"), "got: {err}");
    let mut drained = Vec::new();
    jit::drain_emit_buf(&mut drained);
    assert!(drained.is_empty(), "partial record staged: {:?}", String::from_utf8_lossy(&drained));
}
