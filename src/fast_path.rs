//! Structural contract for fast paths — a staged replacement for the
//! "match-and-commit" pattern in `bin/jq-jit.rs::detect_*` dispatch.
//!
//! # Background
//!
//! Today every raw-byte fast path in `bin/jq-jit.rs` and every rewrite
//! in `simplify_expr` (inside [`crate::interpreter`]) commits once the
//! filter shape matches: it must produce the jq-compatible answer for
//! every possible input, or the user sees garbage. There is no "give
//! up" exit. When a matcher forgets to check an input type (#50), an
//! error-propagation boundary (#43, #48), or a number-repr invariant
//! (#75), the unhandled case surfaces as a silent compat bug in
//! release.
//!
//! # The contract
//!
//! A [`FastPath`] is a concrete optimisation that runs on a
//! [`crate::value::Value`] input and returns one of three verdicts:
//!
//! | return            | meaning                                         |
//! |-------------------|-------------------------------------------------|
//! | `Some(Ok(v))`     | success — `v` is the jq-compatible output value |
//! | `Some(Err(e))`    | the same error jq would raise on this input     |
//! | `None`            | "not my problem, run the generic path"          |
//!
//! The generic path ([`crate::eval`] / [`crate::jit`]) is authoritative,
//! so returning `None` is always safe. The review signal is at the
//! trait boundary: *any* fast path that `Some(Ok(_))`'s an input it
//! shouldn't handle is visibly wrong.
//!
//! # Non-goals
//!
//! - Not removing the existing raw-byte fast paths: they stay, because
//!   materialising a [`Value`] is strictly more expensive than copying
//!   JSON bytes directly.
//! - Not forcing every fast path to handle every input type: `None` is
//!   always available and is what the type-guard pattern exists for.
//!
//! # Pilot: [`FieldAccessPath`]
//!
//! `.field` — the single-field access fast path exercised by
//! `bin/jq-jit.rs::detect_field_access`. It has a clean type-check:
//! object input succeeds, null input returns null (matching jq), every
//! other input shape bails out with `None` for the generic path to
//! raise `Cannot index <type> with "<field>"`.
//!
//! # Migration plan (for future PRs)
//!
//! 1. Wire [`Filter::try_typed_fast_path`] into the dispatch in
//!    `bin/jq-jit.rs` *ahead of* the existing raw-byte detector cascade.
//!    When the trait returns `None`, fall through to the existing
//!    raw-byte path (unchanged behaviour). When the trait returns
//!    `Some`, serialise the resulting `Value` and short-circuit the
//!    remaining detectors.
//! 2. Benchmark the pilot with `./bench/comprehensive.sh --quick`.
//!    Document the cost of `Value` materialisation for the chosen path
//!    so reviewers can weigh correctness-vs-perf on future migrations.
//! 3. Port the next `detect_*` that is known to miss a type-dispatch
//!    case (`detect_field_remap`, `detect_obj_merge_literal`, …).
//!    Each port removes a compat-bug pattern structurally, not by yet
//!    another defensive `match` added to the raw-byte emitter.

use anyhow::Result;

use crate::value::{
    KeyStr, Value, ObjInner, json_object_get_field_raw, json_object_get_fields_raw_buf,
    json_object_get_nested_field_raw, json_object_has_all_keys, json_object_has_any_key,
    json_object_has_key,
};

/// A fast path whose type-dispatch obligations are encoded in its
/// `run` signature. See the module docs for the contract.
pub trait FastPath {
    /// Run this fast path on `input` and return its verdict.
    ///
    /// Implementations **must** return `None` (not `Some(Ok(Value::Null))`
    /// or `Some(Err(_))`) for any input type they are not confident
    /// they handle identically to the generic path. The generic path
    /// is authoritative.
    fn run(&self, input: &Value) -> Option<Result<Value>>;
}

/// Pilot fast path: single `.field` access on input.
///
/// * Object input — returns the field value or `null` if absent.
/// * `null` input — returns `null` (jq semantics: `.x` on null is null).
/// * Any other input — returns `None` so the generic path can raise the
///   "Cannot index <type> with \"<field>\"" error jq does. This is the
///   divergence #50 / the null-masking class would otherwise produce.
pub struct FieldAccessPath {
    pub field: KeyStr,
}

impl FieldAccessPath {
    pub fn new(field: impl Into<KeyStr>) -> Self {
        FieldAccessPath { field: field.into() }
    }
}

impl FastPath for FieldAccessPath {
    fn run(&self, input: &Value) -> Option<Result<Value>> {
        match input {
            Value::Null => Some(Ok(Value::Null)),
            Value::Obj(ObjInner(obj)) => {
                let v = obj.get(self.field.as_str()).cloned().unwrap_or(Value::Null);
                Some(Ok(v))
            }
            // Non-object, non-null input: bail to the generic path so it
            // can raise the correct type-error. Returning `Some(Ok(Value::Null))`
            // here would reintroduce the null-masking bug class (#50).
            _ => None,
        }
    }
}

// =============================================================================
// Raw-byte apply-site contract (#83 Phase B)
// =============================================================================
//
// The CLI in `bin/jq-jit.rs` keeps its raw-byte fast paths (parsing JSON into a
// `Value` is strictly more expensive than copying bytes — see #83's revert
// table: `.name` on 2M-line NDJSON is 2.5× slower with `Value` materialisation).
//
// What the typed [`FastPath`] contract gives us — a named exit point so a
// missing type-check is visible at review — is brought to the raw paths via
// [`RawApplyOutcome`]. Apply-sites no longer commit implicitly inside
// `match raw[0]` arms; they return an explicit verdict, and the caller routes
// `Bail` through `process_input` → `Filter::execute_cb`, where the typed probe
// (Phase A) and generic eval take over.

/// The verdict returned by a raw-byte fast path apply-site for a single JSON
/// record.
///
/// See [`apply_field_access_raw`] for the canonical pilot. New raw fast paths
/// should follow the same shape: type-guard at entry, emit on match, otherwise
/// `Bail`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawApplyOutcome {
    /// The apply-site has written its output bytes (including any trailing
    /// newline) to the caller's buffer. The caller can move on to the next
    /// input record.
    Emit,
    /// The apply-site declines: the input shape does not match what this fast
    /// path can guarantee jq-compatible semantics for. The caller MUST fall
    /// through to `process_input` → `Filter::execute_cb` so the generic path
    /// produces the answer. Returning [`Self::Emit`] for an input the fast
    /// path cannot handle is the bug class #83 is designed to end.
    Bail,
}

/// Apply the `.field` raw-byte fast path on a single JSON record.
///
/// * Object input — invokes `emit` with the field value's raw bytes (or with
///   `b"null"` if the field is absent).
/// * `null` input — invokes `emit` with `b"null"` (jq semantics: `.x` on null
///   is null).
/// * Any other input — returns [`RawApplyOutcome::Bail`] so the caller routes
///   to the generic path, which raises the correct
///   `Cannot index <type> with "<field>"` error.
///
/// The single-closure shape lets the apply-site route every emit through its
/// dup-key / pretty-print pipeline (e.g. the `emit_raw_ln!` macro in
/// `bin/jq-jit.rs`) without forcing two simultaneous `&mut` borrows of the
/// output buffer.
pub fn apply_field_access_raw<E>(
    raw: &[u8],
    field: &str,
    mut emit: E,
) -> RawApplyOutcome
where
    E: FnMut(&[u8]),
{
    match raw.first().copied() {
        Some(b'{') => {
            match json_object_get_field_raw(raw, 0, field) {
                Some((vs, ve)) => emit(&raw[vs..ve]),
                None => emit(b"null"),
            }
            RawApplyOutcome::Emit
        }
        Some(b'n') => {
            emit(b"null");
            RawApplyOutcome::Emit
        }
        // Non-object, non-null: jq raises a type error. Hand off to the
        // generic path; do NOT silently emit `null` (that's #50).
        _ => RawApplyOutcome::Bail,
    }
}

/// Apply the multi-field `.a, .b, .c` raw-byte fast path on a single JSON
/// record.
///
/// The fast path can only emit when the input is an object that contains
/// **every** requested field — those are the only inputs where the right
/// answer is exactly the sequence of raw byte ranges in `ranges_buf` after
/// `json_object_get_fields_raw_buf` succeeds. For everything else the helper
/// returns [`RawApplyOutcome::Bail`]:
///
/// * `null` input — jq emits `null` per field. The generic path produces that
///   sequence; the raw scanner can't because it has no per-field literal to
///   emit and falls back to bytes copying.
/// * partially-missing object — jq emits a mix of values and `null`s. Same
///   reason as above: the raw scanner is all-or-nothing.
/// * non-object non-null input — jq raises a type error (one per field, but
///   it stops at the first); the generic path produces the right error.
///
/// `ranges_buf` is borrowed from the caller so the apply-site can keep one
/// allocation hot across the input stream. Its length must be at least
/// `fields.len()`.
pub fn apply_multi_field_access_raw<E>(
    raw: &[u8],
    fields: &[&str],
    ranges_buf: &mut [(usize, usize)],
    mut emit: E,
) -> RawApplyOutcome
where
    E: FnMut(&[u8]),
{
    if json_object_get_fields_raw_buf(raw, 0, fields, ranges_buf) {
        for (vs, ve) in ranges_buf.iter() {
            emit(&raw[*vs..*ve]);
        }
        RawApplyOutcome::Emit
    } else {
        // Includes: non-object inputs, malformed JSON, and any object that
        // doesn't contain every requested field. The generic path picks the
        // right jq verdict in each case.
        RawApplyOutcome::Bail
    }
}

/// Apply the "object → resolved compute" raw-byte fast path on a single
/// JSON record.
///
/// Used by the apply-sites for `[.x + 1, .y * 2]` (`computed_array` /
/// `standalone_array`) and `{a: .x + 1, b: .y * 2}` (`computed_remap`),
/// which all share the same two-stage bail discipline:
///
/// 1. **Outer bail** — `json_object_get_fields_raw_buf` fails (input isn't
///    an object, or some required field is missing). `RawApplyOutcome::Bail`
///    so the generic path produces jq's per-input verdict.
/// 2. **Inner bail** — `bail_check` returns `true`. This is the #163 hook:
///    the raw scanner can only emit `num+num` and `str+str` arithmetic;
///    anything else (e.g. `str + num`) is a type error in jq, and silently
///    writing `null` would re-introduce the null-masking bug class. The
///    apply-site's `bail_check` walks the resolved cells once with the
///    actual byte ranges and returns `true` if any cell would raise.
///
/// When both bails clear, `emit` is invoked with the filled ranges and the
/// raw input bytes; the apply-site writes the array or object form into its
/// captured buffer.
///
/// `ranges_buf` must have length `>= fields.len()`.
pub fn apply_object_compute_raw<C, E>(
    raw: &[u8],
    fields: &[&str],
    ranges_buf: &mut [(usize, usize)],
    bail_check: C,
    emit: E,
) -> RawApplyOutcome
where
    C: FnOnce(&[(usize, usize)], &[u8]) -> bool,
    E: FnOnce(&[(usize, usize)], &[u8]),
{
    if !json_object_get_fields_raw_buf(raw, 0, fields, ranges_buf) {
        return RawApplyOutcome::Bail;
    }
    let ranges = &ranges_buf[..fields.len()];
    if bail_check(ranges, raw) {
        return RawApplyOutcome::Bail;
    }
    emit(ranges, raw);
    RawApplyOutcome::Emit
}

/// Apply the `has("x")` raw-byte fast path on a single JSON record.
///
/// * Object input — invokes `emit` with `b"true"` if the key is present and
///   `b"false"` otherwise.
/// * Any non-object input — returns [`RawApplyOutcome::Bail`]. jq's
///   `has(IDX)` is type-sensitive (arrays accept integer indices, strings
///   error, …); the generic path raises the right verdict.
pub fn apply_has_field_raw<E>(raw: &[u8], field: &str, mut emit: E) -> RawApplyOutcome
where
    E: FnMut(&[u8]),
{
    match json_object_has_key(raw, 0, field) {
        Some(true) => {
            emit(b"true");
            RawApplyOutcome::Emit
        }
        Some(false) => {
            emit(b"false");
            RawApplyOutcome::Emit
        }
        None => RawApplyOutcome::Bail,
    }
}

/// Apply the multi-key `has("a") and has("b")` / `has("a") or has("b")`
/// raw-byte fast path on a single JSON record.
///
/// `is_and` selects between AND-folding (`all`) and OR-folding (`any`) of
/// the per-key results. Bail discipline matches [`apply_has_field_raw`]:
/// non-object inputs route to the generic path.
pub fn apply_has_multi_field_raw<E>(
    raw: &[u8],
    fields: &[&str],
    is_and: bool,
    mut emit: E,
) -> RawApplyOutcome
where
    E: FnMut(&[u8]),
{
    let result = if is_and {
        json_object_has_all_keys(raw, 0, fields)
    } else {
        json_object_has_any_key(raw, 0, fields)
    };
    match result {
        Some(true) => {
            emit(b"true");
            RawApplyOutcome::Emit
        }
        Some(false) => {
            emit(b"false");
            RawApplyOutcome::Emit
        }
        None => RawApplyOutcome::Bail,
    }
}

/// Apply an "all fields required" raw-byte fast path on a single JSON
/// record.
///
/// Used by every shape that emits a structural form built from the input
/// object's field values: `[.a, .b, .c]` (`array_field`),
/// `{a: .x, b: .y}` (`field_remap`), and any future fast path with the
/// same all-or-nothing fetch semantics. Bail discipline matches
/// [`apply_multi_field_access_raw`]: emit only when the input is an
/// object containing every requested field; anything else routes to the
/// generic path.
///
/// Serialisation is left to the caller: when the fields resolve, the
/// helper hands the filled `ranges_buf` and the raw input bytes to
/// `emit_structural`, which writes whatever shape the apply-site wants
/// (array form, object form, pretty / compact framing, nested-value
/// pretty-print) into its captured buffer. This keeps the helper
/// independent of `use_pretty_buf` / colour / per-shape flags while
/// still naming the commit point at the function boundary.
///
/// `ranges_buf` must have length `>= fields.len()`.
pub fn apply_full_object_fields_raw<E>(
    raw: &[u8],
    fields: &[&str],
    ranges_buf: &mut [(usize, usize)],
    emit_structural: E,
) -> RawApplyOutcome
where
    E: FnOnce(&[(usize, usize)], &[u8]),
{
    if json_object_get_fields_raw_buf(raw, 0, fields, ranges_buf) {
        emit_structural(&ranges_buf[..fields.len()], raw);
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the nested `.a.b.c` raw-byte fast path on a single JSON record.
///
/// * Object input, fully-resolvable nested path — invokes `emit` with the
///   final value's raw bytes.
/// * Object input, path doesn't resolve cleanly — returns
///   [`RawApplyOutcome::Bail`]. The cause may be a missing key (jq says
///   `null`) *or* a non-object intermediate (jq says
///   `Cannot index <type> with "<field>"`); the raw scanner can't
///   distinguish, so the caller hands off to the generic path which produces
///   the correct verdict in either case.
/// * `null` input — invokes `emit` with `b"null"` (jq semantics: any nested
///   `.a.b…` on null is null).
/// * Any other input — returns [`RawApplyOutcome::Bail`] for the same reason
///   as [`apply_field_access_raw`]: the generic path raises the correct
///   type-error.
pub fn apply_nested_field_access_raw<E>(
    raw: &[u8],
    fields: &[&str],
    mut emit: E,
) -> RawApplyOutcome
where
    E: FnMut(&[u8]),
{
    match raw.first().copied() {
        Some(b'{') => match json_object_get_nested_field_raw(raw, 0, fields) {
            Some((vs, ve)) => {
                emit(&raw[vs..ve]);
                RawApplyOutcome::Emit
            }
            None => RawApplyOutcome::Bail,
        },
        Some(b'n') => {
            emit(b"null");
            RawApplyOutcome::Emit
        }
        _ => RawApplyOutcome::Bail,
    }
}
