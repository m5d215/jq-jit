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

use crate::interpreter::{ArithExpr, CmpVal, MathUnary, MixedCond, StringChainOp, StringChainTerminal};
use crate::ir::{BinOp, UnaryOp};
use crate::runtime::jq_mod_f64;
use crate::value::{
    KeyStr, Value, ObjInner, json_each_value_cb, json_object_assign_field_arith,
    json_object_assign_two_fields_arith, json_object_del_field, json_object_del_fields,
    json_object_filter_by_key_str, json_object_filter_by_value_type,
    json_object_get_field_raw, json_object_get_fields_raw_buf, json_object_merge_literal,
    json_object_values_tostring, json_with_entries_select_value_cmp, push_tojson_raw,
    json_object_get_nested_field_raw, json_object_get_num, json_object_get_two_nums,
    json_object_has_all_keys, json_object_has_any_key, json_object_has_key,
    json_object_update_field_case, json_object_update_field_gsub,
    json_object_update_field_length, json_object_update_field_slice,
    json_object_update_field_split_first, json_object_update_field_split_last,
    json_object_update_field_str_concat, json_object_update_field_str_map,
    json_object_update_field_test, json_object_update_field_tostring,
    json_object_update_field_trim, json_type_byte, json_value_length,
    json_object_extract_keys_only, json_object_keys_to_buf_reuse, parse_json_num,
    push_jq_number_bytes, push_json_pretty_raw, skip_json_value,
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

/// Apply the `.field | test("pattern"; flags)` raw-byte fast path on a
/// single JSON record.
///
/// The raw scanner can only run the regex over a quoted-string field with
/// **no backslash escapes** (decoding escapes would defeat the
/// no-materialise contract); everything else bails:
///
/// * Object input, field present, value is a quoted string with no `\`
///   escapes — invokes `emit` with `b"true"` or `b"false"`.
/// * Field absent, value isn't a quoted string, or the string contains
///   any backslash escape — returns [`RawApplyOutcome::Bail`] so the
///   generic path produces jq's verdict (which knows how to decode
///   escapes and how to raise on non-string inputs).
/// * Non-object input — returns [`RawApplyOutcome::Bail`] (the field
///   fetch fails, so this is collapsed into the same Bail branch).
///
/// The caller owns the compiled `regex::Regex` so it can be reused
/// across the input stream.
pub fn apply_field_test_raw<E>(
    raw: &[u8],
    field: &str,
    re: &regex::Regex,
    mut emit: E,
) -> RawApplyOutcome
where
    E: FnMut(&[u8]),
{
    let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
        Some(r) => r,
        None => return RawApplyOutcome::Bail,
    };
    let val = &raw[vs..ve];
    // Only handle the no-escape ASCII-or-UTF-8 quoted-string fast lane.
    // Anything else (non-string, escaped string, missing trailing quote)
    // bails — the generic path can decode escapes and raise the right
    // error for non-strings.
    if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
        || val[1..val.len() - 1].contains(&b'\\')
    {
        return RawApplyOutcome::Bail;
    }
    let content = &val[1..val.len() - 1];
    let content_str = unsafe { std::str::from_utf8_unchecked(content) };
    if re.is_match(content_str) {
        emit(b"true");
    } else {
        emit(b"false");
    }
    RawApplyOutcome::Emit
}

/// Apply the `.field | gsub/sub("pattern"; "replacement"; flags)` raw-byte
/// fast path on a single JSON record.
///
/// Bail discipline matches [`apply_field_test_raw`] (only quoted-string
/// fields with no backslash escapes are handled by the raw scanner; the
/// generic path takes everything else, including the type errors jq raises
/// on non-strings). When the bail clears, `emit` is invoked with the
/// `Cow<str>` produced by the regex replacement; the caller is responsible
/// for JSON-escaping the result and writing the surrounding quotes /
/// trailing newline.
///
/// `is_global` selects between `replace_all` (gsub) and `replace` (sub).
pub fn apply_field_gsub_raw<E>(
    raw: &[u8],
    field: &str,
    re: &regex::Regex,
    replacement: &str,
    is_global: bool,
    mut emit: E,
) -> RawApplyOutcome
where
    E: FnMut(&str),
{
    let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
        Some(r) => r,
        None => return RawApplyOutcome::Bail,
    };
    let val = &raw[vs..ve];
    if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
        || val[1..val.len() - 1].contains(&b'\\')
    {
        return RawApplyOutcome::Bail;
    }
    let content = unsafe { std::str::from_utf8_unchecked(&val[1..val.len() - 1]) };
    let result = if is_global {
        re.replace_all(content, replacement)
    } else {
        re.replace(content, replacement)
    };
    emit(&result);
    RawApplyOutcome::Emit
}

/// Apply the `.field | ascii_downcase|ascii_upcase | test("pattern")`
/// raw-byte fast path on a single JSON record.
///
/// `upper = true` selects `ascii_upcase`, `false` selects `ascii_downcase`.
/// The case fold is byte-wise ASCII (`a..z` ↔ `A..Z`), matching jq's
/// behaviour for ASCII characters in the no-escape lane.
///
/// Bail discipline matches [`apply_field_test_raw`]. Strings with any `\`
/// escape Bail because the raw scanner can't decode them, and the case
/// fold is only safe on bytes the scanner can already see verbatim. Non-
/// object input or missing field also Bail so the generic path produces
/// jq's verdict.
pub fn apply_field_case_test_raw<E>(
    raw: &[u8],
    field: &str,
    upper: bool,
    re: &regex::Regex,
    mut emit: E,
) -> RawApplyOutcome
where
    E: FnMut(&[u8]),
{
    let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
        Some(r) => r,
        None => return RawApplyOutcome::Bail,
    };
    let val = &raw[vs..ve];
    if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
        || val[1..val.len() - 1].contains(&b'\\')
    {
        return RawApplyOutcome::Bail;
    }
    let inner = &val[1..val.len() - 1];
    let converted: Vec<u8> = inner.iter().map(|&b| {
        if upper {
            if b.is_ascii_lowercase() { b - 32 } else { b }
        } else if b.is_ascii_uppercase() { b + 32 } else { b }
    }).collect();
    let content = unsafe { std::str::from_utf8_unchecked(&converted) };
    if re.is_match(content) {
        emit(b"true");
    } else {
        emit(b"false");
    }
    RawApplyOutcome::Emit
}

/// Apply the `.field | ascii_downcase|ascii_upcase | gsub/sub("pattern";
/// "replacement"; flags)` raw-byte fast path on a single JSON record.
///
/// `upper = true` selects `ascii_upcase`, `false` selects `ascii_downcase`.
/// `is_global` selects between `replace_all` (gsub) and `replace` (sub).
///
/// Bail discipline matches [`apply_field_gsub_raw`]. Only the no-escape
/// quoted-string lane is handled; everything else returns
/// [`RawApplyOutcome::Bail`] for the generic path.
pub fn apply_field_case_gsub_raw<E>(
    raw: &[u8],
    field: &str,
    upper: bool,
    re: &regex::Regex,
    replacement: &str,
    is_global: bool,
    mut emit: E,
) -> RawApplyOutcome
where
    E: FnMut(&str),
{
    let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
        Some(r) => r,
        None => return RawApplyOutcome::Bail,
    };
    let val = &raw[vs..ve];
    if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
        || val[1..val.len() - 1].contains(&b'\\')
    {
        return RawApplyOutcome::Bail;
    }
    let inner = &val[1..val.len() - 1];
    let converted: Vec<u8> = inner.iter().map(|&b| {
        if upper {
            if b.is_ascii_lowercase() { b - 32 } else { b }
        } else if b.is_ascii_uppercase() { b + 32 } else { b }
    }).collect();
    let content = unsafe { std::str::from_utf8_unchecked(&converted) };
    let result = if is_global {
        re.replace_all(content, replacement)
    } else {
        re.replace(content, replacement)
    };
    emit(&result);
    RawApplyOutcome::Emit
}

/// Apply a single-call `.field | re.captures(content)` raw-byte fast path
/// on one JSON record. Used by both `match` and `capture` apply-sites:
/// both run a single `captures()` call (not an iterator), emit one output
/// on a successful match, and emit nothing on a non-match — the only
/// difference is the bytes the apply-site builds in `on_match` (`match`
/// emits `{offset,length,string,captures:[...]}`, `capture` emits an
/// object keyed by the named groups).
///
/// Bail discipline matches [`apply_field_test_raw`] (only quoted-string
/// fields with no backslash escapes are handled by the raw scanner; the
/// generic path takes everything else, including the type errors jq raises
/// on non-strings):
///
/// * Object input, field present, value is a quoted string with no `\`
///   escapes — runs `re.captures(content)`. On a match, invokes
///   `on_match(content, captures)` so the caller can build jq's output
///   bytes. On no match, `on_match` is **not** called.
/// * Field absent, value isn't a quoted string, or the string contains
///   any backslash escape — returns [`RawApplyOutcome::Bail`] so the
///   generic path produces jq's verdict (decoded escapes, type errors).
/// * Non-object input — returns [`RawApplyOutcome::Bail`] (the field
///   fetch fails, so this is collapsed into the same Bail branch).
///
/// The caller owns the compiled `regex::Regex` and any
/// `capture_names`-style metadata (typically built once outside the
/// stream loop) so they can be reused across the input stream.
pub fn apply_field_match_raw<F>(
    raw: &[u8],
    field: &str,
    re: &regex::Regex,
    mut on_match: F,
) -> RawApplyOutcome
where
    F: FnMut(&str, &regex::Captures<'_>),
{
    let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
        Some(r) => r,
        None => return RawApplyOutcome::Bail,
    };
    let val = &raw[vs..ve];
    if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
        || val[1..val.len() - 1].contains(&b'\\')
    {
        return RawApplyOutcome::Bail;
    }
    let content = unsafe { std::str::from_utf8_unchecked(&val[1..val.len() - 1]) };
    if let Some(caps) = re.captures(content) {
        on_match(content, &caps);
    }
    RawApplyOutcome::Emit
}

/// Apply the `.field | scan("pattern")` raw-byte fast path on a single
/// JSON record.
///
/// `scan` is a *multi-output* filter: it emits one value per
/// non-overlapping regex match. Bail discipline matches
/// [`apply_field_match_raw`] (object input + quoted-string field with no
/// backslash escapes); on a passing type-guard the helper iterates
/// `re.captures_iter(content)` and invokes `on_match(content,
/// captures)` for **each** match. On zero matches the closure is not
/// called — the helper still returns [`RawApplyOutcome::Emit`] (jq emits
/// no output for `scan` with no match, and that is the fast path's
/// intended semantics).
///
/// `captures_iter` is used uniformly for both the no-capture-group and
/// capture-group cases — for a no-group regex, each `Captures` exposes
/// only group 0 (the full match), so the apply-site can branch on
/// `caps.len()` to choose between scalar and array output without the
/// helper needing to know about it.
pub fn apply_field_scan_raw<F>(
    raw: &[u8],
    field: &str,
    re: &regex::Regex,
    mut on_match: F,
) -> RawApplyOutcome
where
    F: FnMut(&str, &regex::Captures<'_>),
{
    let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
        Some(r) => r,
        None => return RawApplyOutcome::Bail,
    };
    let val = &raw[vs..ve];
    if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
        || val[1..val.len() - 1].contains(&b'\\')
    {
        return RawApplyOutcome::Bail;
    }
    let content = unsafe { std::str::from_utf8_unchecked(&val[1..val.len() - 1]) };
    for caps in re.captures_iter(content) {
        on_match(content, &caps);
    }
    RawApplyOutcome::Emit
}

/// Apply the `.field | @<format>` raw-byte fast path on a single JSON
/// record. Used by `@text`, `@json`, `@base64`, `@uri`, `@html`.
///
/// jq's `@<format>` family doesn't share a single Bail discipline
/// across formats: `@base64`/`@uri`/`@html` raise on non-string input,
/// while `@json`/`@text` accept any scalar (and wrap its raw bytes in
/// quotes for output). The helper therefore handles only the
/// **structural** type-guard — field present and the value is not an
/// escape-bearing quoted string — and delegates the
/// format-specific Emit-vs-Bail decision to the closure via its
/// returned [`RawApplyOutcome`].
///
/// * Object input, field present, value is a quoted string with no `\`
///   escapes — invokes `on_value(val_with_quotes, Some(content))`.
/// * Object input, field present, value is a non-string scalar
///   (number / bool / null / array / object literal) — invokes
///   `on_value(val_bytes, None)`. The closure decides whether to emit
///   or to return [`RawApplyOutcome::Bail`] (e.g. `@base64` returns
///   Bail, `@json` returns Emit after wrapping the raw bytes).
/// * Field absent, or value is a quoted string with at least one `\`
///   escape — returns [`RawApplyOutcome::Bail`] without invoking the
///   closure (the generic path decodes the escape / raises the error).
/// * Non-object input — returns [`RawApplyOutcome::Bail`] (the field
///   fetch fails).
pub fn apply_field_format_raw<F>(
    raw: &[u8],
    field: &str,
    on_value: F,
) -> RawApplyOutcome
where
    F: FnOnce(&[u8], Option<&[u8]>) -> RawApplyOutcome,
{
    let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
        Some(r) => r,
        None => return RawApplyOutcome::Bail,
    };
    let val = &raw[vs..ve];
    let content = if val.len() >= 2 && val[0] == b'"' && val[val.len() - 1] == b'"' {
        if val[1..val.len() - 1].contains(&b'\\') {
            return RawApplyOutcome::Bail;
        }
        Some(&val[1..val.len() - 1])
    } else {
        None
    };
    on_value(val, content)
}

/// Apply the `.field | ltrimstr("prefix") | tonumber [ | <arith> ]*`
/// raw-byte fast path on a single JSON record.
///
/// Bail discipline matches [`apply_field_test_raw`] for the structural
/// type-guard (object input + quoted-string field with no `\` escapes),
/// plus a numeric-parse Bail (#160): if the post-prefix-strip content
/// can't be parsed as an `f64`, jq raises `... cannot be parsed as a
/// number`, so the fast path returns [`RawApplyOutcome::Bail`] and lets
/// the generic path produce the real error.
///
/// On success, the helper applies the post-`tonumber` arithmetic chain
/// `arith_ops` (each `(BinOp, f64)` is folded left over the parsed
/// number) and invokes `emit` with the final `f64` so the apply-site
/// owns JSON-number formatting (see `push_jq_number_bytes`).
///
/// `arith_ops` may be empty (a bare `ltrimstr | tonumber` chain). Only
/// the arithmetic ops jq actually emits in this lowering are honoured —
/// `Add`, `Sub`, `Mul`, `Div`, `Mod`. Comparison/logical ops would
/// require Boolean output and are not part of the fast path's contract;
/// any other variant is treated as identity (defensive).
pub fn apply_field_ltrimstr_tonumber_raw<F>(
    raw: &[u8],
    field: &str,
    prefix: &[u8],
    arith_ops: &[(BinOp, f64)],
    mut emit: F,
) -> RawApplyOutcome
where
    F: FnMut(f64),
{
    let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
        Some(r) => r,
        None => return RawApplyOutcome::Bail,
    };
    let val = &raw[vs..ve];
    if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
        || val[1..val.len() - 1].contains(&b'\\')
    {
        return RawApplyOutcome::Bail;
    }
    let content = &val[1..val.len() - 1];
    let num_bytes = if content.len() >= prefix.len() && &content[..prefix.len()] == prefix {
        &content[prefix.len()..]
    } else {
        content
    };
    let num_str = unsafe { std::str::from_utf8_unchecked(num_bytes) };
    let mut n: f64 = match fast_float::parse(num_str) {
        Ok(n) => n,
        Err(_) => return RawApplyOutcome::Bail,
    };
    for (op, c) in arith_ops {
        n = match op {
            BinOp::Add => n + c,
            BinOp::Sub => n - c,
            BinOp::Mul => n * c,
            BinOp::Div => n / c,
            BinOp::Mod => jq_mod_f64(n, *c).unwrap_or(f64::NAN),
            _ => n,
        };
    }
    emit(n);
    RawApplyOutcome::Emit
}

/// Apply the `.field + "<literal-suffix>"` raw-byte fast path on a single
/// JSON record.
///
/// jq's `+` on strings concatenates, and `null + "x"` returns `"x"` (jq
/// treats `null` as a left-identity for most operations), so this fast
/// path's discipline is not just "string field, otherwise bail":
///
/// * Object input, field absent — invokes `on_field(None)`. The
///   apply-site emits `"<suffix>"` (matching `null + "suffix"` →
///   `"suffix"`).
/// * Object input, field present, value is a quoted string with no `\`
///   escapes — invokes `on_field(Some(content))` where `content` is the
///   field bytes without the surrounding quotes. The apply-site emits
///   `"<content><suffix>"`.
/// * Object input, field present, value is a non-string scalar (number,
///   bool, array, etc.) — returns [`RawApplyOutcome::Bail`] so the
///   generic path raises jq's `<type> and string cannot be added`.
/// * Object input, field present, value is an escape-bearing string —
///   returns [`RawApplyOutcome::Bail`] (the raw scanner can't decode
///   escapes safely; the generic path will).
/// * Non-object input — returns [`RawApplyOutcome::Bail`]. Notably this
///   includes `null`, even though `null | (.x + "s") == "s"` is jq's
///   answer; the helper conservatively delegates the lookup-and-fold to
///   the generic path.
pub fn apply_field_str_concat_raw<E>(
    raw: &[u8],
    field: &str,
    on_field: E,
) -> RawApplyOutcome
where
    E: FnOnce(Option<&[u8]>),
{
    if raw.first() != Some(&b'{') {
        return RawApplyOutcome::Bail;
    }
    match json_object_get_field_raw(raw, 0, field) {
        None => {
            on_field(None);
            RawApplyOutcome::Emit
        }
        Some((vs, ve)) => {
            let val = &raw[vs..ve];
            if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
                || val[1..val.len() - 1].contains(&b'\\')
            {
                return RawApplyOutcome::Bail;
            }
            on_field(Some(&val[1..val.len() - 1]));
            RawApplyOutcome::Emit
        }
    }
}

/// Apply the `.field | split("") | reverse | join("")` raw-byte fast
/// path (string reversal) on a single JSON record.
///
/// The bail discipline here is *structural-only*: the helper guarantees
/// the input is an object whose `field` is a quoted string, and hands
/// the caller the raw inner bytes (still possibly containing JSON
/// escapes). The caller decides between two emit strategies and signals
/// the verdict via the closure's return value:
///
/// 1. ASCII-only with no `\` escapes — emit reversed bytes directly
///    (returns [`RawApplyOutcome::Emit`]).
/// 2. Any other string — decode JSON escapes, reverse Unicode code
///    points, re-encode (returns [`RawApplyOutcome::Emit`]) — or, if
///    decode fails (e.g. invalid UTF-8 after escape unescape), return
///    [`RawApplyOutcome::Bail`] so the generic path handles it.
///
/// Helper-side Bail: field absent, value isn't a quoted string, input
/// isn't an object. (`null` input bails too — `null | split("")`
/// raises in jq.)
pub fn apply_field_str_reverse_raw<E>(
    raw: &[u8],
    field: &str,
    on_string: E,
) -> RawApplyOutcome
where
    E: FnOnce(&[u8]) -> RawApplyOutcome,
{
    let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
        Some(r) => r,
        None => return RawApplyOutcome::Bail,
    };
    let val = &raw[vs..ve];
    if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"' {
        return RawApplyOutcome::Bail;
    }
    on_string(&val[1..val.len() - 1])
}

/// Apply the `.x <op> .y` raw-byte arithmetic fast path on a single
/// JSON record where both `.x` and `.y` resolve to JSON numbers.
///
/// Bail discipline:
/// * Either field is absent, or either value isn't a JSON number —
///   [`RawApplyOutcome::Bail`] (the generic path raises jq's
///   `<type> and <type> cannot be added` / similar).
/// * `op` is anything other than `Add`/`Sub`/`Mul`/`Div`/`Mod` —
///   [`RawApplyOutcome::Bail`] (the detector should never produce a
///   non-arithmetic op for this shape, but the helper rejects it
///   defensively rather than wrap the comparison ops in a coerced
///   `f64`).
/// * The arithmetic result is non-finite (div-by-zero, mod-by-zero,
///   IEEE NaN/∞) — [`RawApplyOutcome::Bail`] so the generic path
///   produces jq's runtime error (`/ by zero`, etc.).
/// * Non-object input — [`RawApplyOutcome::Bail`] (number lookup
///   fails).
///
/// On a passing type-guard with a finite result, invokes `emit(n)` so
/// the apply-site owns JSON-number formatting.
pub fn apply_field_binop_raw<F>(
    raw: &[u8],
    field_a: &str,
    field_b: &str,
    op: BinOp,
    mut emit: F,
) -> RawApplyOutcome
where
    F: FnMut(f64),
{
    let (a, b) = match json_object_get_two_nums(raw, 0, field_a, field_b) {
        Some(p) => p,
        None => return RawApplyOutcome::Bail,
    };
    let result = match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Mod => jq_mod_f64(a, b).unwrap_or(f64::NAN),
        _ => return RawApplyOutcome::Bail,
    };
    if !result.is_finite() {
        return RawApplyOutcome::Bail;
    }
    emit(result);
    RawApplyOutcome::Emit
}

/// Apply the `.x <cmp> .y` raw-byte numeric-comparison fast path on a
/// single JSON record (`<cmp>` ∈ Gt/Lt/Ge/Le/Eq/Ne) where both fields
/// resolve to JSON numbers.
///
/// jq's `==`/`!=` work on every type (`null == null`, `"a" == "a"`),
/// but this fast path only commits when both fields are numbers; the
/// generic path handles every other type combination so cross-type
/// equality (e.g. `null == null`) doesn't silently take the numeric
/// branch.
///
/// Bail discipline:
/// * Either field absent or non-numeric — [`RawApplyOutcome::Bail`].
/// * Non-comparison op (`Add`/`And`/etc.) — [`RawApplyOutcome::Bail`]
///   (defensive — the detector should never produce these).
/// * Non-object input — [`RawApplyOutcome::Bail`].
///
/// On success, invokes `emit(result)` with the boolean comparison.
pub fn apply_field_field_cmp_raw<F>(
    raw: &[u8],
    field_a: &str,
    field_b: &str,
    cmp_op: BinOp,
    mut emit: F,
) -> RawApplyOutcome
where
    F: FnMut(bool),
{
    let (a, b) = match json_object_get_two_nums(raw, 0, field_a, field_b) {
        Some(p) => p,
        None => return RawApplyOutcome::Bail,
    };
    let result = match cmp_op {
        BinOp::Gt => a > b,
        BinOp::Lt => a < b,
        BinOp::Ge => a >= b,
        BinOp::Le => a <= b,
        BinOp::Eq => a == b,
        BinOp::Ne => a != b,
        _ => return RawApplyOutcome::Bail,
    };
    emit(result);
    RawApplyOutcome::Emit
}

/// Apply the `. + {key: <arith>}` raw-byte object-merge fast path on
/// a single JSON record. Reads numeric fields, evaluates the
/// `ArithExpr`, formats the result as JSON-number bytes, and merges
/// `{key: <result>}` into the input object.
///
/// Caller provides:
/// * `out_key` — the merge key.
/// * `nfields` — fields referenced by the arith expression.
/// * `arith` — the compiled numeric expression.
/// * `vals_buf` — scratch sized to `nfields.len()`.
/// * `merge_pair_buf` — single-element scratch (shape:
///   `[(out_key.clone(), Vec::new())]`); element 0's value is
///   overwritten each call with the JSON-number bytes.
///
/// Bail discipline:
/// * Non-object input — Bail.
/// * Any referenced field absent or non-numeric — Bail.
/// * Non-finite result (div-by-zero / overflow / NaN) — Bail so
///   `null` / saturating numbers don't leak into the merged object.
/// * `merge_pair_buf` has wrong shape (defensive) — Bail.
/// * Underlying `json_object_merge_literal` returning false — Bail.
///
/// Writes the merged-object bytes (without trailing `\n`) to `buf` on
/// Emit.
pub fn apply_obj_merge_computed_raw(
    raw: &[u8],
    nfields: &[&str],
    arith: &ArithExpr,
    vals_buf: &mut [f64],
    merge_pair_buf: &mut [(String, Vec<u8>)],
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    if merge_pair_buf.len() != 1 || vals_buf.len() < nfields.len() {
        return RawApplyOutcome::Bail;
    }
    let nf = nfields.len();
    let ok = if nf == 1 {
        match json_object_get_num(raw, 0, nfields[0]) {
            Some(v) => { vals_buf[0] = v; true }
            None => false,
        }
    } else if nf == 2 {
        match json_object_get_two_nums(raw, 0, nfields[0], nfields[1]) {
            Some((a, b)) => { vals_buf[0] = a; vals_buf[1] = b; true }
            None => false,
        }
    } else {
        // 3+ fields not supported by the existing apply-site; the
        // detector should never produce this shape, but Bail
        // defensively rather than emit garbage.
        false
    };
    if !ok {
        return RawApplyOutcome::Bail;
    }
    let result = arith.eval(&vals_buf[..nf]);
    if !result.is_finite() {
        return RawApplyOutcome::Bail;
    }
    merge_pair_buf[0].1.clear();
    push_jq_number_bytes(&mut merge_pair_buf[0].1, result);
    if json_object_merge_literal(raw, 0, merge_pair_buf, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Test whether a JSON value's first byte indicates the given jq
/// type name. Returns None for unrecognised type names so callers
/// can Bail on detector regression.
#[inline]
/// Walk `bytes` and return true when it contains a non-canonical numeric
/// literal — `1e10` (lowercase `e`), `+1`, or the special-float literals
/// `nan` / `inf` / `infinity` (case-insensitive). String contents are
/// skipped (a stray `e` inside a string isn't a number's exponent
/// marker). Mirrors the helper in `bin/jq-jit.rs`. See #598.
fn raw_contains_non_canonical_number(bytes: &[u8]) -> bool {
    static LUT: [bool; 256] = {
        let mut t = [false; 256];
        let chars: &[u8] = &[b'"', b'+', b'e', b'E', b'n', b'N', b'i', b'I', b'.'];
        let mut k = 0;
        while k < chars.len() { t[chars[k] as usize] = true; k += 1; }
        t
    };
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && !LUT[bytes[i] as usize] { i += 1; }
        if i >= bytes.len() { return false; }
        match bytes[i] {
            b'"' => {
                i += 1;
                while i < bytes.len() {
                    match memchr::memchr2(b'"', b'\\', &bytes[i..]) {
                        Some(off) => {
                            i += off;
                            if bytes[i] == b'"' { i += 1; break; }
                            // \uD[8-F]XX is a surrogate codepoint — defer to
                            // the slow parser to handle lone surrogates and
                            // pair decoding (#615).
                            if i + 5 < bytes.len()
                                && bytes[i + 1] == b'u'
                                && (bytes[i + 2] == b'D' || bytes[i + 2] == b'd')
                                && matches!(bytes[i + 3],
                                    b'8' | b'9' | b'A' | b'B' | b'C' | b'D' | b'E' | b'F'
                                         | b'a' | b'b' | b'c' | b'd' | b'e' | b'f')
                            {
                                return true;
                            }
                            i = i.saturating_add(2);
                        }
                        None => return false,
                    }
                }
            }
            b'+' => return true,
            b'n' | b'N' if bytes.get(i..i+3).is_some_and(|s| s.eq_ignore_ascii_case(b"nan")) => return true,
            b'i' | b'I' if bytes.get(i..i+8).is_some_and(|s| s.eq_ignore_ascii_case(b"infinity"))
                || bytes.get(i..i+3).is_some_and(|s| s.eq_ignore_ascii_case(b"inf")) => return true,
            b'e' | b'E' => {
                if i > 0 {
                    let prev = bytes[i - 1];
                    if prev.is_ascii_digit() || prev == b'.' {
                        return true;
                    }
                }
                i += 1;
            }
            b'.' => {
                // 6+ leading zeros in the fractional part means effective
                // exponent < -6, which jq's decnum normalises to scientific
                // (#611). Need the previous byte to be a digit so we don't
                // mistake `[.foo]` style tokens for numeric literals.
                if bytes.get(i + 1..i + 7) == Some(&[b'0'; 6][..])
                    && i > 0 && bytes[i - 1].is_ascii_digit()
                {
                    return true;
                }
                i += 1;
            }
            _ => i += 1,
        }
    }
    false
}

fn type_byte_matches(type_name: &str, b: u8) -> Option<bool> {
    Some(match type_name {
        "string" => b == b'"',
        "number" => b == b'-' || b.is_ascii_digit(),
        "boolean" => b == b't' || b == b'f',
        "null" => b == b'n',
        "object" => b == b'{',
        "array" => b == b'[',
        _ => return None,
    })
}

/// Apply the `.[] | <type>s` raw-byte fast path on a single JSON
/// record (where `<type>s` is one of jq's type-filter built-ins —
/// `strings`/`numbers`/`booleans`/`nulls`/`objects`/`arrays`).
/// Iterates each element of the input array/object and emits the
/// element bytes (with a trailing `\n`) for every element whose
/// first byte indicates the given type.
///
/// Bail discipline:
/// * Unrecognised type name (defensive) — Bail.
/// * Non-iterable input (`json_each_value_cb` returns false) —
///   Bail so jq's `Cannot iterate over <type>` error surfaces.
///
/// The helper writes element-bytes-plus-`\n` directly to `buf` for
/// every match. On a passing iterable, returns `Emit` even if no
/// element matched (no output is the correct `select` semantics for
/// no-match).
pub fn apply_each_type_filter_raw(
    raw: &[u8],
    type_name: &str,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if type_byte_matches(type_name, b'\0').is_none() {
        return RawApplyOutcome::Bail;
    }
    // Number canonicalisation (#598): values containing non-canonical
    // numeric literals (`1e10`, `+1`, `nan`, `inf`) must be re-rendered
    // through Value. For type=string/boolean/null the matched value can
    // never contain such a literal, so we skip the check entirely.
    let needs_canon_check = !matches!(type_name, "string" | "boolean" | "null");
    let mut needs_bail = false;
    let save_len = buf.len();
    let ok = json_each_value_cb(raw, 0, |vs, ve| {
        if needs_bail { return; }
        let val = &raw[vs..ve];
        if val.is_empty() { return; }
        if matches!(type_byte_matches(type_name, val[0]), Some(true)) {
            if needs_canon_check && raw_contains_non_canonical_number(val) {
                needs_bail = true;
                return;
            }
            buf.extend_from_slice(val);
            buf.push(b'\n');
        }
    });
    if !ok || needs_bail {
        buf.truncate(save_len);
        return RawApplyOutcome::Bail;
    }
    RawApplyOutcome::Emit
}

/// Apply the `[.[] | select(type == "<type>")]` raw-byte fast path
/// on a single JSON record. Collects elements of the matching type
/// into a single JSON array and emits the array with a trailing `\n`.
///
/// Bail discipline matches [`apply_each_type_filter_raw`]:
/// unrecognised type → Bail; non-iterable input → Bail. The helper
/// truncates `buf` back to its pre-call length on Bail so the caller
/// doesn't need to manage the partial-write rollback itself.
pub fn apply_collect_each_select_type_raw(
    raw: &[u8],
    type_name: &str,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if type_byte_matches(type_name, b'\0').is_none() {
        return RawApplyOutcome::Bail;
    }
    let needs_canon_check = !matches!(type_name, "string" | "boolean" | "null");
    let mut needs_bail = false;
    let save_len = buf.len();
    buf.push(b'[');
    let mut first_elem = true;
    let ok = json_each_value_cb(raw, 0, |vs, ve| {
        if needs_bail { return; }
        let val = &raw[vs..ve];
        if val.is_empty() { return; }
        if matches!(type_byte_matches(type_name, val[0]), Some(true)) {
            if needs_canon_check && raw_contains_non_canonical_number(val) {
                needs_bail = true;
                return;
            }
            if !first_elem { buf.push(b','); }
            first_elem = false;
            buf.extend_from_slice(val);
        }
    });
    if ok && !needs_bail {
        buf.extend_from_slice(b"]\n");
        RawApplyOutcome::Emit
    } else {
        buf.truncate(save_len);
        RawApplyOutcome::Bail
    }
}

/// Apply the `first(.[] | select(type == "<type>"))` raw-byte fast
/// path on a single JSON record. Emits the first element of the
/// matching type with a trailing `\n`. If no element matches, emits
/// nothing (returns `Emit` — that's jq's behaviour for `first` on an
/// empty stream — no output, no error).
///
/// Bail discipline matches [`apply_each_type_filter_raw`].
pub fn apply_first_each_select_type_raw(
    raw: &[u8],
    type_name: &str,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if type_byte_matches(type_name, b'\0').is_none() {
        return RawApplyOutcome::Bail;
    }
    let needs_canon_check = !matches!(type_name, "string" | "boolean" | "null");
    let mut found_range: Option<(usize, usize)> = None;
    let ok = json_each_value_cb(raw, 0, |vs, ve| {
        if found_range.is_some() { return; }
        let val = &raw[vs..ve];
        if val.is_empty() { return; }
        if matches!(type_byte_matches(type_name, val[0]), Some(true)) {
            found_range = Some((vs, ve));
        }
    });
    if !ok {
        return RawApplyOutcome::Bail;
    }
    if let Some((vs, ve)) = found_range {
        let val = &raw[vs..ve];
        if needs_canon_check && raw_contains_non_canonical_number(val) {
            return RawApplyOutcome::Bail;
        }
        buf.extend_from_slice(val);
        buf.push(b'\n');
    }
    RawApplyOutcome::Emit
}

/// Apply the `. + {k1: v1, k2: v2, ...}` raw-byte object-merge fast
/// path on a single JSON record. Each `(k, v)` is a literal key-name
/// + JSON-encoded value-bytes pair (the parser pre-encodes each
/// branch's JSON value).
///
/// Bail discipline: delegates to `json_object_merge_literal`, which
/// returns `false` on non-object input. The wrapper documents the
/// structural shape at the helper boundary.
///
/// Writes the merged-object bytes (without trailing `\n`) to `buf` on
/// Emit. Caller appends the newline.
pub fn apply_obj_merge_lit_raw(
    raw: &[u8],
    merge_pairs: &[(String, Vec<u8>)],
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_merge_literal(raw, 0, merge_pairs, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `.dest = (.src <op> N)` raw-byte object-assign fast path
/// on a single JSON record. Reads `.src`'s numeric value, applies the
/// arithmetic, and writes the result to `.dest` (creating the field
/// if absent), preserving the rest of the object's structure.
///
/// Bail discipline (delegates to `json_object_assign_field_arith`):
/// * Non-object input — Bail.
/// * `.src` absent or non-numeric — Bail (jq raises type error).
/// * `op` is `Mod` with divisor zero — Bail (jq raises).
/// * Non-finite arithmetic result — Bail.
/// * Non-arithmetic op (defensive) — Bail (the underlying call falls
///   through to a no-op for non-arith ops, but the structural shape
///   is documented).
///
/// Writes the rewritten object bytes (without trailing `\n`) to `buf`
/// on Emit. Caller appends the newline.
pub fn apply_obj_assign_field_arith_raw(
    raw: &[u8],
    dest_field: &str,
    src_field: &str,
    op: BinOp,
    n: f64,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
        return RawApplyOutcome::Bail;
    }
    if json_object_assign_field_arith(raw, 0, dest_field, src_field, op, n, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `.dest = (.src1 <op> .src2)` raw-byte object-assign fast
/// path on a single JSON record. Reads two source fields' numeric
/// values, applies the arithmetic, and writes the result to `.dest`.
///
/// Bail discipline (delegates to `json_object_assign_two_fields_arith`):
/// * Non-object input — Bail.
/// * Either source field absent or non-numeric — Bail.
/// * Non-finite arithmetic result — Bail (div-by-zero etc.).
/// * Non-arithmetic op — Bail (defensive).
///
/// Writes the rewritten object bytes (without trailing `\n`) to `buf`
/// on Emit.
pub fn apply_obj_assign_two_fields_arith_raw(
    raw: &[u8],
    dest_field: &str,
    src1_field: &str,
    src2_field: &str,
    op: BinOp,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
        return RawApplyOutcome::Bail;
    }
    if json_object_assign_two_fields_arith(raw, 0, dest_field, src1_field, src2_field, op, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `type` raw-byte fast path on a single JSON record.
///
/// jq's `type` returns the static type name (`"object"` / `"array"` /
/// `"string"` / `"boolean"` / `"null"` / `"number"`) of the input.
/// The output is determined entirely by the first byte of the parsed
/// JSON value, which `json_stream_raw` guarantees is one of the
/// valid JSON-value starts.
///
/// **No-Bail helper.** Unlike most apply helpers in this module, this
/// one never returns [`RawApplyOutcome::Bail`] for well-formed input
/// — the first byte uniquely determines the output, and there is no
/// type-error case for `type` in jq. The structural shape is pinned
/// here so a future reader doesn't add a Bail branch and accidentally
/// route some input through the generic path with a different
/// behaviour.
///
/// Writes the type-name bytes (with surrounding `"` quotes) plus a
/// trailing newline to `buf`.
pub fn apply_is_type_raw(raw: &[u8], buf: &mut Vec<u8>) -> RawApplyOutcome {
    if raw.is_empty() {
        return RawApplyOutcome::Bail;
    }
    buf.extend_from_slice(json_type_byte(raw[0]));
    buf.push(b'\n');
    RawApplyOutcome::Emit
}

/// Apply the `length` raw-byte fast path on a single JSON record.
///
/// The raw scanner only handles container and null lengths directly:
/// * Object — number of keys.
/// * Array — number of elements.
/// * Null — `0`.
///
/// Strings, numbers, booleans, and any malformed input return
/// [`RawApplyOutcome::Bail`] so the generic path produces jq's
/// verdict (unicode-aware string length, `abs(n)` for numbers,
/// `boolean has no length` error). Adding fast-path support for
/// strings/numbers is a separate optimisation.
pub fn apply_is_length_raw(raw: &[u8], buf: &mut Vec<u8>) -> RawApplyOutcome {
    match json_value_length(raw, 0) {
        Some(len) => {
            push_jq_number_bytes(buf, len as f64);
            buf.push(b'\n');
            RawApplyOutcome::Emit
        }
        None => RawApplyOutcome::Bail,
    }
}

/// Apply a `.field <cmp> <val>` predicate (where `<val>` is either
/// numeric or a quoted string) and report the boolean verdict via
/// `emit`. Used by branch fast paths whose predicate shape is more
/// general than `apply_field_const_cmp_raw` (which is numeric-only)
/// or `apply_select_str_raw` (which is string-only).
///
/// Bail discipline:
/// * Non-object input — Bail (jq's `Cannot index <type>` surfaces
///   via the generic path).
/// * Field absent — Bail (so jq's null comparison or cross-type
///   ordering applies via the generic path).
/// * For `CmpVal::Num`: field non-numeric — Bail.
/// * For `CmpVal::Str`: field non-string or escape-bearing — Bail.
/// * Non-comparison op (`Add`/`And`/etc.) — Bail (defensive).
///
/// On a passing type-guard, invokes `emit(verdict)` with the boolean
/// result. The string comparison uses byte ordering (jq's lexicographic
/// string comparison agrees with byte ordering for ASCII; non-ASCII
/// strings without escapes also agree because UTF-8 byte-ordering
/// preserves Unicode code-point ordering).
pub fn apply_field_cmp_val_raw<F>(
    raw: &[u8],
    field: &str,
    cmp_op: BinOp,
    cmp_val: &CmpVal,
    mut emit: F,
) -> RawApplyOutcome
where
    F: FnMut(bool),
{
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    if !matches!(
        cmp_op,
        BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne,
    ) {
        return RawApplyOutcome::Bail;
    }
    let pass = match cmp_val {
        CmpVal::Num(threshold) => {
            let n = match json_object_get_num(raw, 0, field) {
                Some(v) => v,
                None => return RawApplyOutcome::Bail,
            };
            match cmp_op {
                BinOp::Gt => n > *threshold,
                BinOp::Lt => n < *threshold,
                BinOp::Ge => n >= *threshold,
                BinOp::Le => n <= *threshold,
                BinOp::Eq => n == *threshold,
                BinOp::Ne => n != *threshold,
                _ => unreachable!(),
            }
        }
        CmpVal::Str(s) => {
            let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
                Some(r) => r,
                None => return RawApplyOutcome::Bail,
            };
            let val = &raw[vs..ve];
            if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
                || val[1..val.len() - 1].contains(&b'\\')
            {
                return RawApplyOutcome::Bail;
            }
            let inner = &val[1..val.len() - 1];
            let arg = s.as_bytes();
            match cmp_op {
                BinOp::Eq => inner == arg,
                BinOp::Ne => inner != arg,
                BinOp::Gt => inner > arg,
                BinOp::Lt => inner < arg,
                BinOp::Ge => inner >= arg,
                BinOp::Le => inner <= arg,
                _ => unreachable!(),
            }
        }
    };
    emit(pass);
    RawApplyOutcome::Emit
}

/// Apply the `{k1: .f1, k2: .f2, ...} | tojson` raw-byte fast path
/// on a single JSON record. Builds the inner object from the named
/// source fields, then emits the JSON-string-escaped representation.
///
/// Caller provides:
/// * `pairs` — `(out_key, src_field)` mapping.
/// * `inner_key_prefixes` — pre-encoded key prefixes for the inner
///   object (e.g. `[b"{\"a\":", b",\"b\":"]`); passed in to avoid
///   re-allocating per record.
/// * `inner_buf` — scratch buffer for the inner object (cleared
///   inside the helper).
/// * `ranges_buf` — scratch sized to `pairs.len()` for the field
///   range fetch.
/// * `out_buf` — output buffer for the tojson result (no trailing
///   `\n`; caller appends).
///
/// Bail discipline:
/// * Non-object input — Bail (jq's `Cannot index <type>` for the
///   underlying `.f1` access surfaces via the generic path).
/// * Buffer length mismatch (defensive) — Bail.
pub fn apply_remap_tojson_raw(
    raw: &[u8],
    pairs: &[(String, String)],
    inner_key_prefixes: &[Vec<u8>],
    inner_buf: &mut Vec<u8>,
    ranges_buf: &mut [(usize, usize)],
    out_buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    let n = pairs.len();
    if ranges_buf.len() < n || inner_key_prefixes.len() < n {
        return RawApplyOutcome::Bail;
    }
    let input_fields: Vec<&str> = pairs.iter().map(|(_, f)| f.as_str()).collect();
    if !json_object_get_fields_raw_buf(raw, 0, &input_fields, ranges_buf) {
        return RawApplyOutcome::Bail;
    }
    inner_buf.clear();
    for (i, (vs, ve)) in ranges_buf[..n].iter().enumerate() {
        inner_buf.extend_from_slice(&inner_key_prefixes[i]);
        inner_buf.extend_from_slice(&raw[*vs..*ve]);
    }
    inner_buf.push(b'}');
    push_tojson_raw(out_buf, inner_buf);
    RawApplyOutcome::Emit
}

/// Apply the `{k1: .f1, k2: .f2, ...} | to_entries` raw-byte fast
/// path on a single JSON record. Builds a JSON array of `{"key":
/// k, "value": v}` entries from the named source fields. Missing
/// source fields emit `"value": null` (matching jq's `to_entries`
/// for `null` values).
///
/// Bail discipline:
/// * Non-object input — Bail (jq raises `Cannot index <type> with
///   "<field>"` for the underlying `.f1` access).
///
/// Writes the resulting JSON array bytes plus a trailing `\n` to
/// `buf` on Emit.
pub fn apply_remap_to_entries_raw(
    raw: &[u8],
    pairs: &[(String, String)],
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    buf.push(b'[');
    let mut first = true;
    for (out_key, src) in pairs {
        if !first { buf.push(b','); }
        first = false;
        buf.extend_from_slice(b"{\"key\":\"");
        for &b in out_key.as_bytes() {
            match b {
                b'"' => buf.extend_from_slice(b"\\\""),
                b'\\' => buf.extend_from_slice(b"\\\\"),
                _ => buf.push(b),
            }
        }
        buf.extend_from_slice(b"\",\"value\":");
        match json_object_get_field_raw(raw, 0, src) {
            Some((vs, ve)) => buf.extend_from_slice(&raw[vs..ve]),
            None => buf.extend_from_slice(b"null"),
        }
        buf.push(b'}');
    }
    buf.extend_from_slice(b"]\n");
    RawApplyOutcome::Emit
}

/// Apply the `to_entries[] | "\(.key)SEP\(.value)..."` raw-byte fast
/// path. Iterates kv pairs of the input object and emits one
/// interpolated string (with surrounding quotes and trailing `\n`)
/// per entry.
///
/// `interp_parts` is the parsed interpolation: `(is_lit, content)`
/// pairs where `is_lit=true` means literal text and `is_lit=false`
/// means a reference to either `"key"` or `"value"`.
///
/// Bail discipline (truncates `buf` back to its entry-checkpoint on
/// any per-entry violation, leaving caller state untouched):
/// - Non-object input → Bail. jq accepts `to_entries` on arrays, so
///   the slow path handles that.
/// - Empty object → Emit (no output, matches jq).
/// - Per entry, key bytes containing `\` → Bail. jq normalizes
///   `\u00XX` etc. to the actual character; copying the JSON-quoted
///   form would diverge.
/// - Per entry, value (computed via `skip_json_value`):
///   * Quoted string containing `\` → Bail (same escape-norm reason).
///   * Number containing non-canonical bytes (`+`, `e`/`E` after
///     digit/`.`) → Bail. jq normalizes `1e10`→`1E+10` and `+5`→`5`.
///   * Array/object → Bail. jq compacts whitespace inside; raw bytes
///     can preserve internal whitespace.
///   * Bool/null/canonical-number/escape-free string → Emit.
pub fn apply_to_entries_each_interp_raw(
    raw: &[u8],
    interp_parts: &[(bool, String)],
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    let entries_checkpoint = buf.len();
    let mut i = 1usize;
    while i < raw.len() && matches!(raw[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < raw.len() && raw[i] == b'}' {
        return RawApplyOutcome::Emit;
    }
    loop {
        if i >= raw.len() || raw[i] != b'"' {
            buf.truncate(entries_checkpoint);
            return RawApplyOutcome::Bail;
        }
        let ks = i + 1;
        i += 1;
        while i < raw.len() {
            match raw[i] {
                b'"' => break,
                b'\\' => { i = i.saturating_add(2); continue; }
                _ => i += 1,
            }
        }
        if i >= raw.len() {
            buf.truncate(entries_checkpoint);
            return RawApplyOutcome::Bail;
        }
        let ke = i;
        i += 1;
        while i < raw.len() && matches!(raw[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= raw.len() || raw[i] != b':' {
            buf.truncate(entries_checkpoint);
            return RawApplyOutcome::Bail;
        }
        i += 1;
        while i < raw.len() && matches!(raw[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        let vs = i;
        i = match skip_json_value(raw, i) {
            Ok(e) => e,
            Err(_) => {
                buf.truncate(entries_checkpoint);
                return RawApplyOutcome::Bail;
            }
        };
        let ve = i;
        let key_bytes = &raw[ks..ke];
        let val_bytes = &raw[vs..ve];
        if key_bytes.contains(&b'\\') {
            buf.truncate(entries_checkpoint);
            return RawApplyOutcome::Bail;
        }
        if val_bytes.is_empty() {
            buf.truncate(entries_checkpoint);
            return RawApplyOutcome::Bail;
        }
        let v0 = val_bytes[0];
        match v0 {
            b'"' => {
                let inner_end = val_bytes.len().saturating_sub(1);
                if val_bytes[1..inner_end].contains(&b'\\') {
                    buf.truncate(entries_checkpoint);
                    return RawApplyOutcome::Bail;
                }
            }
            b'[' | b'{' => {
                buf.truncate(entries_checkpoint);
                return RawApplyOutcome::Bail;
            }
            b't' | b'f' | b'n' => {}
            b'+' => {
                buf.truncate(entries_checkpoint);
                return RawApplyOutcome::Bail;
            }
            b'-' | b'0'..=b'9' => {
                let mut idx = 0;
                while idx < val_bytes.len() {
                    let b = val_bytes[idx];
                    if b == b'+' {
                        buf.truncate(entries_checkpoint);
                        return RawApplyOutcome::Bail;
                    }
                    if (b == b'e' || b == b'E') && idx > 0 {
                        let prev = val_bytes[idx - 1];
                        if prev.is_ascii_digit() || prev == b'.' {
                            buf.truncate(entries_checkpoint);
                            return RawApplyOutcome::Bail;
                        }
                    }
                    idx += 1;
                }
            }
            _ => {
                buf.truncate(entries_checkpoint);
                return RawApplyOutcome::Bail;
            }
        }
        buf.push(b'"');
        for (is_lit, content) in interp_parts {
            if *is_lit {
                for &b in content.as_bytes() {
                    match b {
                        b'"' => buf.extend_from_slice(b"\\\""),
                        b'\\' => buf.extend_from_slice(b"\\\\"),
                        _ => buf.push(b),
                    }
                }
            } else if content == "key" {
                for &b in key_bytes {
                    match b {
                        b'"' => buf.extend_from_slice(b"\\\""),
                        b'\\' => buf.extend_from_slice(b"\\\\"),
                        _ => buf.push(b),
                    }
                }
            } else if v0 == b'"' {
                buf.extend_from_slice(&val_bytes[1..val_bytes.len() - 1]);
            } else {
                for &b in val_bytes {
                    match b {
                        b'"' => buf.extend_from_slice(b"\\\""),
                        b'\\' => buf.extend_from_slice(b"\\\\"),
                        _ => buf.push(b),
                    }
                }
            }
        }
        buf.extend_from_slice(b"\"\n");
        while i < raw.len() && matches!(raw[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= raw.len() {
            buf.truncate(entries_checkpoint);
            return RawApplyOutcome::Bail;
        }
        if raw[i] == b'}' { break; }
        if raw[i] != b',' {
            buf.truncate(entries_checkpoint);
            return RawApplyOutcome::Bail;
        }
        i += 1;
        while i < raw.len() && matches!(raw[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
    RawApplyOutcome::Emit
}

/// Apply the `select(.field | <string-chain> | <terminal>)` raw-byte
/// fast path. Reads the field, runs an in-place ASCII byte-level
/// transformation chain (lowercase/uppercase/ltrimstr/rtrimstr/
/// split-join/split-reverse-join), then evaluates a boolean terminal
/// (`startswith`/`endswith`/`contains`). On a true verdict invokes
/// `emit_pass()` with the input record's raw bytes (select passes the
/// input through unchanged); on a false verdict returns `Emit`
/// silently with no buffer write (matching `select`'s empty-on-false
/// semantics).
///
/// `tmp_str` is a caller-owned scratch buffer (cleared on entry).
///
/// Bail discipline:
/// - Non-object input → Bail. jq's generic path raises "Cannot index
///   ... with string" — the in-place implementation previously
///   silently skipped these inputs (#83-class bug).
/// - Predicate field absent → Bail. jq's generic raises e.g.
///   "null cannot be matched"; previously silent.
/// - Predicate field is non-string → Bail. jq raises e.g. "explode
///   input must be a string"; previously silent.
/// - Predicate field is a string containing any backslash escape →
///   Bail. The raw chain operates on byte slices and cannot decode
///   `\u00XX`/`\n` etc.; the generic path handles those.
/// - Unsupported terminal (e.g. `Length`/`Index`/`None`) → Bail
///   defensively (the detector should not produce these for select,
///   but the helper guards against them at the type boundary).
pub fn apply_select_string_chain_raw<F>(
    raw: &[u8],
    field: &str,
    ops: &[StringChainOp],
    terminal: &StringChainTerminal,
    tmp_str: &mut Vec<u8>,
    mut emit_pass: F,
) -> RawApplyOutcome
where
    F: FnMut(&[u8]),
{
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
        Some(p) => p,
        None => return RawApplyOutcome::Bail,
    };
    let val = &raw[vs..ve];
    if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"' {
        return RawApplyOutcome::Bail;
    }
    let inner = &val[1..val.len() - 1];
    if memchr::memchr(b'\\', inner).is_some() {
        return RawApplyOutcome::Bail;
    }
    tmp_str.clear();
    tmp_str.extend_from_slice(inner);
    for op in ops {
        match op {
            StringChainOp::AsciiDowncase => tmp_str.make_ascii_lowercase(),
            StringChainOp::AsciiUpcase => tmp_str.make_ascii_uppercase(),
            StringChainOp::Ltrimstr(s) => {
                let sb = s.as_bytes();
                if tmp_str.starts_with(sb) {
                    tmp_str.drain(..sb.len());
                }
            }
            StringChainOp::Rtrimstr(s) => {
                let sb = s.as_bytes();
                if sb.is_empty() {
                    tmp_str.clear();
                } else if tmp_str.ends_with(sb) {
                    let new_len = tmp_str.len() - sb.len();
                    tmp_str.truncate(new_len);
                }
            }
            StringChainOp::SplitJoin(sep, rep) => {
                let sb = sep.as_bytes();
                let rb = rep.as_bytes();
                let mut result = Vec::new();
                let mut pos = 0usize;
                let mut first = true;
                while pos <= tmp_str.len() {
                    let end_pos = if sb.is_empty() {
                        pos + 1
                    } else {
                        tmp_str[pos..]
                            .windows(sb.len())
                            .position(|w| w == sb)
                            .map(|p| pos + p)
                            .unwrap_or(tmp_str.len())
                    };
                    if !first {
                        result.extend_from_slice(rb);
                    }
                    result.extend_from_slice(&tmp_str[pos..end_pos]);
                    first = false;
                    if end_pos >= tmp_str.len() {
                        break;
                    }
                    pos = end_pos + sb.len();
                }
                *tmp_str = result;
            }
            StringChainOp::SplitReverseJoin(sep, rep) => {
                let sb = sep.as_bytes();
                let rb = rep.as_bytes();
                let mut segments: Vec<&[u8]> = Vec::new();
                let mut pos = 0usize;
                while pos <= tmp_str.len() {
                    let end_pos = if sb.is_empty() {
                        pos + 1
                    } else {
                        tmp_str[pos..]
                            .windows(sb.len())
                            .position(|w| w == sb)
                            .map(|p| pos + p)
                            .unwrap_or(tmp_str.len())
                    };
                    segments.push(&tmp_str[pos..end_pos]);
                    if end_pos >= tmp_str.len() {
                        break;
                    }
                    pos = end_pos + sb.len();
                }
                segments.reverse();
                let mut result = Vec::new();
                for (i, seg) in segments.iter().enumerate() {
                    if i > 0 {
                        result.extend_from_slice(rb);
                    }
                    result.extend_from_slice(seg);
                }
                *tmp_str = result;
            }
        }
    }
    let pass = match terminal {
        StringChainTerminal::Startswith(arg) => tmp_str.starts_with(arg.as_bytes()),
        StringChainTerminal::Endswith(arg) => tmp_str.ends_with(arg.as_bytes()),
        StringChainTerminal::Contains(arg) => {
            let needle = arg.as_bytes();
            needle.is_empty() || memchr::memmem::find(tmp_str, needle).is_some()
        }
        // None / Length / Index — not boolean, defensive Bail.
        _ => return RawApplyOutcome::Bail,
    };
    if pass {
        emit_pass(raw);
    }
    RawApplyOutcome::Emit
}

/// Apply the `select((.f1 | startswith/endswith/contains(arg)) AND/OR
/// (.f2 | ...) AND/OR ...)` raw-byte fast path on a single JSON
/// record. On a true verdict invokes `emit_pass(raw)` (select passes
/// the input through unchanged); on a false verdict returns `Emit`
/// silently with no buffer write.
///
/// `is_and = true` for AND-combination, `false` for OR. `str_conds`
/// is a slice of `(field, test_name, test_arg)` triples where
/// `test_name` is one of `"startswith"`, `"endswith"`, `"contains"`.
///
/// jq's `and`/`or` short-circuit: AND breaks on first false, OR
/// breaks on first true. The helper reproduces that — once a verdict
/// is determined by a successfully-evaluated condition, later
/// conditions (which might error) are not evaluated, matching jq.
///
/// Bail discipline (only kicks in for conditions actually evaluated
/// — short-circuited conditions never trigger Bail):
/// - Non-object input → Bail. jq raises "Cannot index ..."; previously
///   silent (#83-class bug).
/// - Field absent → Bail. jq raises "startswith() requires string
///   inputs" on `null`; previously silent.
/// - Field is non-string or string with backslash escapes → Bail.
/// - Unsupported test name (defensive) → Bail.
pub fn apply_select_compound_str_test_raw<F>(
    raw: &[u8],
    is_and: bool,
    str_conds: &[(String, String, String)],
    mut emit_pass: F,
) -> RawApplyOutcome
where
    F: FnMut(&[u8]),
{
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    let mut result = is_and;
    for (field, test_name, test_arg) in str_conds {
        let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
            Some(p) => p,
            None => return RawApplyOutcome::Bail,
        };
        let val = &raw[vs..ve];
        if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
            || val[1..val.len() - 1].contains(&b'\\')
        {
            return RawApplyOutcome::Bail;
        }
        let inner = &val[1..val.len() - 1];
        let arg_bytes = test_arg.as_bytes();
        let pass = match test_name.as_str() {
            "startswith" => inner.starts_with(arg_bytes),
            "endswith" => inner.ends_with(arg_bytes),
            "contains" => arg_bytes.is_empty()
                || memchr::memmem::find(inner, arg_bytes).is_some(),
            _ => return RawApplyOutcome::Bail,
        };
        if is_and {
            if !pass {
                result = false;
                break;
            }
        } else if pass {
            result = true;
            break;
        }
    }
    if result {
        emit_pass(raw);
    }
    RawApplyOutcome::Emit
}

/// Apply the `select(<cond1> AND/OR <cond2> AND/OR ...)` raw-byte
/// fast path where each condition is either a numeric comparison
/// (`MixedCond::NumCmp(field, op, threshold)`) or a string-test
/// builtin (`MixedCond::StrTest(field, name, arg)` with `name` ∈
/// `"startswith"`/`"endswith"`/`"contains"`/`"eq"`/`"test"`).
///
/// `compiled_re` is a slice parallel to `mixed_conds` holding
/// pre-compiled `regex::Regex` for `"test"` conditions; entries for
/// non-test conditions are `None`. The caller compiles once outside
/// the per-record loop.
///
/// jq's `and`/`or` short-circuit is preserved: AND breaks on first
/// false, OR on first true, so later conditions that would otherwise
/// error are not evaluated.
///
/// Bail discipline (only kicks in for *evaluated* conditions —
/// short-circuited ones never trigger Bail):
/// - Non-object input → Bail.
/// - NumCmp: field absent / non-numeric → Bail. Non-comparison op
///   (defensive) → Bail.
/// - StrTest: field absent / non-string / string with `\` escapes →
///   Bail. `"test"` without a compiled regex (compile failure) →
///   Bail. Unsupported test name → Bail.
pub fn apply_select_mixed_compound_raw<F>(
    raw: &[u8],
    is_and: bool,
    mixed_conds: &[MixedCond],
    compiled_re: &[Option<regex::Regex>],
    mut emit_pass: F,
) -> RawApplyOutcome
where
    F: FnMut(&[u8]),
{
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    let mut result = is_and;
    for (i, cond) in mixed_conds.iter().enumerate() {
        let pass = match cond {
            MixedCond::NumCmp(field, op, threshold) => {
                let n = match json_object_get_num(raw, 0, field) {
                    Some(v) => v,
                    None => return RawApplyOutcome::Bail,
                };
                match op {
                    BinOp::Gt => n > *threshold,
                    BinOp::Lt => n < *threshold,
                    BinOp::Ge => n >= *threshold,
                    BinOp::Le => n <= *threshold,
                    BinOp::Eq => n == *threshold,
                    BinOp::Ne => n != *threshold,
                    _ => return RawApplyOutcome::Bail,
                }
            }
            MixedCond::StrTest(field, test_name, test_arg) => {
                let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
                    Some(p) => p,
                    None => return RawApplyOutcome::Bail,
                };
                let val = &raw[vs..ve];
                if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
                    || val[1..val.len() - 1].contains(&b'\\')
                {
                    return RawApplyOutcome::Bail;
                }
                let inner = &val[1..val.len() - 1];
                let arg_bytes = test_arg.as_bytes();
                match test_name.as_str() {
                    "startswith" => inner.starts_with(arg_bytes),
                    "endswith" => inner.ends_with(arg_bytes),
                    "contains" => arg_bytes.is_empty()
                        || memchr::memmem::find(inner, arg_bytes).is_some(),
                    "eq" => inner == arg_bytes,
                    "test" => match compiled_re.get(i).and_then(|r| r.as_ref()) {
                        Some(re) => {
                            let s = unsafe { std::str::from_utf8_unchecked(inner) };
                            re.is_match(s)
                        }
                        None => return RawApplyOutcome::Bail,
                    },
                    _ => return RawApplyOutcome::Bail,
                }
            }
        };
        if is_and {
            if !pass {
                result = false;
                break;
            }
        } else if pass {
            result = true;
            break;
        }
    }
    if result {
        emit_pass(raw);
    }
    RawApplyOutcome::Emit
}

/// Apply the `keys` raw-byte fast path on a single JSON record. Emits
/// a sorted JSON array literal of the input object's top-level keys.
///
/// The helper layers a per-stream key-set cache around the extraction:
/// when consecutive records share the same (unsorted) key set, the
/// cached sorted-output bytes are reused without re-sorting. The
/// caller owns the cache state (`cached_output`, `cached_keys`,
/// `keys_buf`) plus a scratch `tmp` buffer for pretty-printing.
///
/// `use_pretty` selects the formatting branch: when `true` the output
/// is collected in `tmp` then re-emitted into `out_buf` via
/// `push_json_pretty_raw` (with a trailing `\n`); when `false` the
/// raw compact bytes are written directly to `out_buf` and the cache
/// is captured from the same range. Both branches end with a single
/// `\n` appended to `out_buf`.
///
/// Bail discipline (delegates to `json_object_keys_to_buf_reuse`):
/// - Non-object input → Bail. jq's `keys` works on arrays too
///   (returns indices), strings/numbers raise — the generic path
///   handles both correctly.
/// - Malformed input (escaped key boundary, unmatched braces, etc.)
///   → Bail.
pub fn apply_is_keys_raw(
    raw: &[u8],
    cached_output: &mut Vec<u8>,
    cached_keys: &mut Vec<Vec<u8>>,
    keys_buf: &mut Vec<(usize, usize)>,
    tmp: &mut Vec<u8>,
    use_pretty: bool,
    out_buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    // Try cached permutation: if the unsorted key set matches the cache,
    // reuse the previously-sorted output bytes without re-sorting.
    if !cached_keys.is_empty() {
        if let Some(extracted) = json_object_extract_keys_only(raw, 0, keys_buf) {
            if extracted == cached_keys.len() {
                let mut same = true;
                for (i, (ks, ke)) in keys_buf.iter().enumerate() {
                    if &raw[*ks..*ke] != cached_keys[i].as_slice() {
                        same = false;
                        break;
                    }
                }
                if same {
                    if use_pretty {
                        push_json_pretty_raw(out_buf, cached_output, 2, false);
                    } else {
                        out_buf.extend_from_slice(cached_output);
                    }
                    out_buf.push(b'\n');
                    return RawApplyOutcome::Emit;
                }
            }
        }
    }
    // Full path: extract + sort + emit. On non-object / malformed input,
    // `json_object_keys_to_buf_reuse` returns false → Bail.
    if use_pretty {
        tmp.clear();
        if !json_object_keys_to_buf_reuse(raw, 0, tmp, keys_buf) {
            return RawApplyOutcome::Bail;
        }
        let len = tmp.len();
        if len > 0 && tmp[len - 1] == b'\n' {
            tmp.truncate(len - 1);
        }
        if cached_keys.is_empty() {
            *cached_output = tmp.clone();
            let mut unsorted: Vec<(usize, usize)> = Vec::new();
            if json_object_extract_keys_only(raw, 0, &mut unsorted).is_some() {
                *cached_keys = unsorted.iter().map(|(s, e)| raw[*s..*e].to_vec()).collect();
            }
        }
        push_json_pretty_raw(out_buf, tmp, 2, false);
        out_buf.push(b'\n');
    } else {
        let before = out_buf.len();
        if !json_object_keys_to_buf_reuse(raw, 0, out_buf, keys_buf) {
            // The helper truncates internal state on early failure, but
            // it may have appended partial bytes before discovering a
            // malformed structure — restore caller state.
            out_buf.truncate(before);
            return RawApplyOutcome::Bail;
        }
        if cached_keys.is_empty() {
            let end_pos = out_buf.len();
            // Output is "[...]\n"; cache the "[...]" portion (strip \n).
            *cached_output = out_buf[before..end_pos - 1].to_vec();
            let mut unsorted: Vec<(usize, usize)> = Vec::new();
            if json_object_extract_keys_only(raw, 0, &mut unsorted).is_some() {
                *cached_keys = unsorted.iter().map(|(s, e)| raw[*s..*e].to_vec()).collect();
            }
        }
    }
    RawApplyOutcome::Emit
}

/// Apply the `with_entries(select(.value cmp N))` raw-byte fast
/// path on a single JSON record. Filters object entries by value
/// against a numeric threshold.
///
/// `cmp_byte` is the encoded comparison op:
/// `b'>'`=Gt, `b'G'`=Ge, `b'<'`=Lt, `b'L'`=Le, `b'='`=Eq, `b'!'`=Ne.
///
/// Bail discipline (delegates to `json_with_entries_select_value_cmp`,
/// which returns `false` on non-object input so the wrapper Bails to
/// generic). Writes the filtered-object bytes including trailing `\n`
/// to `buf` on Emit.
pub fn apply_with_entries_select_raw(
    raw: &[u8],
    cmp_byte: u8,
    threshold: f64,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_with_entries_select_value_cmp(raw, 0, cmp_byte, threshold, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `with_entries(select(.key != "name"))` raw-byte fast
/// path — equivalent to `del(.name, ...)` over the listed keys.
///
/// Bail discipline (delegates to `json_object_del_fields`): non-object
/// input → Bail. Writes the rewritten-object bytes to `buf`
/// (without trailing `\n`).
pub fn apply_with_entries_del_raw(
    raw: &[u8],
    keys: &[&str],
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_del_fields(raw, 0, keys, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `with_entries(select(.value | type == "T"))` raw-byte
/// fast path — filter object entries by value type. `type_name` is
/// one of `"string"`/`"number"`/`"boolean"`/`"null"`/`"object"`/
/// `"array"`.
///
/// Bail discipline (delegates to `json_object_filter_by_value_type`):
/// non-object input → Bail. Writes the filtered-object bytes to `buf`.
pub fn apply_with_entries_type_raw(
    raw: &[u8],
    type_name: &str,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_filter_by_value_type(raw, 0, type_name, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `with_entries(select(.key | startswith/endswith/test
/// (...)))` raw-byte fast path — filter object entries by key
/// matching a string predicate.
///
/// Bail discipline (delegates to `json_object_filter_by_key_str`):
/// non-object input → Bail. Writes the filtered-object bytes to `buf`.
pub fn apply_with_entries_key_str_raw(
    raw: &[u8],
    test_op: &str,
    test_arg: &str,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_filter_by_key_str(raw, 0, test_op, test_arg, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `with_entries(.value |= tostring)` raw-byte fast path —
/// stringify all object values.
///
/// Bail discipline (delegates to `json_object_values_tostring`):
/// non-object input → Bail. Writes the rewritten-object bytes to
/// `buf`.
pub fn apply_with_entries_tostring_raw(
    raw: &[u8],
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_values_tostring(raw, 0, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `if .field == null then T else F end` (or `!= null`)
/// raw-byte fast path on a single JSON record. Both branches are
/// pre-encoded byte literals.
///
/// `is_eq_null = true` selects `== null` (T when field is null or
/// missing); `false` selects `!= null` (T when field is present and
/// non-null).
///
/// Bail discipline:
/// * Non-object input — Bail (jq's `Cannot index <type> with "<field>"`
///   surfaces via the generic path; the prior fast path silently
///   treated non-object input as "field missing → null", emitting the
///   wrong branch — a #83-class divergence).
///
/// On Emit, writes `t_bytes` or `f_bytes` (whichever the predicate
/// selects) followed by a trailing `\n` directly to `buf`.
pub fn apply_null_branch_lit_raw(
    raw: &[u8],
    field: &str,
    is_eq_null: bool,
    t_bytes: &[u8],
    f_bytes: &[u8],
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    let is_null = match json_object_get_field_raw(raw, 0, field) {
        Some((vs, ve)) => ve - vs == 4 && &raw[vs..ve] == b"null",
        None => true, // missing field is null in jq
    };
    let pass = if is_eq_null { is_null } else { !is_null };
    buf.extend_from_slice(if pass { t_bytes } else { f_bytes });
    buf.push(b'\n');
    RawApplyOutcome::Emit
}

/// Apply the `select(.x.y.z <cmp> N)` raw-byte fast path on a single
/// JSON record. Walks the nested field path, then runs a numeric
/// comparison against a compile-time constant; emits the input
/// bytes on a passing predicate (`select` semantics).
///
/// Bail discipline:
/// * Non-object input or any nested step's intermediate value
///   isn't an object — Bail (jq's `Cannot index <type> with "<key>"`
///   surfaces via the generic path).
/// * Leaf value absent or non-numeric — Bail (so jq's cross-type
///   ordering / type-error on null applies via the generic path).
/// * Non-comparison op (`Add`/`And`/etc.) — Bail (defensive).
///
/// On a passing predicate, calls `emit_match(raw)` so the apply-site
/// emits the matching record. On a failing predicate, returns
/// [`RawApplyOutcome::Emit`] without invoking the closure.
pub fn apply_select_nested_cmp_raw<F>(
    raw: &[u8],
    fields: &[&str],
    cmp_op: BinOp,
    threshold: f64,
    mut emit_match: F,
) -> RawApplyOutcome
where
    F: FnMut(&[u8]),
{
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    if !matches!(
        cmp_op,
        BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne,
    ) {
        return RawApplyOutcome::Bail;
    }
    let (vs, ve) = match crate::value::json_object_get_nested_field_raw(raw, 0, fields) {
        Some(r) => r,
        None => return RawApplyOutcome::Bail,
    };
    let val = match parse_json_num(&raw[vs..ve]) {
        Some(v) => v,
        None => return RawApplyOutcome::Bail,
    };
    let pass = match cmp_op {
        BinOp::Gt => val > threshold,
        BinOp::Lt => val < threshold,
        BinOp::Ge => val >= threshold,
        BinOp::Le => val <= threshold,
        BinOp::Eq => val == threshold,
        BinOp::Ne => val != threshold,
        _ => unreachable!(),
    };
    if pass {
        emit_match(raw);
    }
    RawApplyOutcome::Emit
}

/// Apply the `select((.numfield <cmp> N) and (.strfield <op_str> arg))`
/// raw-byte fast path on a single JSON record. Combines a numeric
/// comparison on one field with a string predicate
/// (`startswith`/`endswith`/`contains`/`test`/`eq`) on another.
///
/// `select` semantics: emit the input record unchanged on a passing
/// conjunction; emit nothing on a failing predicate. The helper
/// passes the raw bytes to `emit_match` only when both predicates
/// pass.
///
/// Bail discipline:
/// * Non-object input — [`RawApplyOutcome::Bail`] (jq raises
///   `Cannot index <type>`).
/// * Numeric field absent or non-numeric — Bail (so jq's cross-type
///   ordering / type error surfaces via the generic path).
/// * String field absent, non-string, or escape-bearing — Bail (raw
///   scanner can't decode escapes; for non-strings jq raises a type
///   error from the string predicate).
/// * `num_op` is non-comparison (`Add`/`And`/etc.) — Bail (defensive).
/// * `str_op` is not a known string op (`startswith`/`endswith`/
///   `contains`/`test`/`eq`) — Bail (defensive).
/// * `str_op == "test"` with `str_re` is `None` — Bail (regex
///   compilation failed; generic path will raise the right error).
///
/// On a passing predicate, calls `emit_match(raw)` so the caller
/// emits the matching record. On a failing predicate, returns
/// `RawApplyOutcome::Emit` without invoking the closure.
pub fn apply_select_num_str_raw<F>(
    raw: &[u8],
    num_field: &str,
    num_op: BinOp,
    num_threshold: f64,
    str_field: &str,
    str_op: &str,
    str_arg: &[u8],
    str_re: Option<&regex::Regex>,
    mut emit_match: F,
) -> RawApplyOutcome
where
    F: FnMut(&[u8]),
{
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    if !matches!(
        num_op,
        BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq | BinOp::Ne,
    ) {
        return RawApplyOutcome::Bail;
    }
    let known_str_op = matches!(str_op, "startswith" | "endswith" | "contains" | "test" | "eq");
    if !known_str_op {
        return RawApplyOutcome::Bail;
    }
    if str_op == "test" && str_re.is_none() {
        return RawApplyOutcome::Bail;
    }
    let n = match json_object_get_num(raw, 0, num_field) {
        Some(v) => v,
        None => return RawApplyOutcome::Bail,
    };
    let num_pass = match num_op {
        BinOp::Gt => n > num_threshold,
        BinOp::Lt => n < num_threshold,
        BinOp::Ge => n >= num_threshold,
        BinOp::Le => n <= num_threshold,
        BinOp::Eq => n == num_threshold,
        BinOp::Ne => n != num_threshold,
        _ => unreachable!(),
    };
    if !num_pass {
        return RawApplyOutcome::Emit;
    }
    let (vs, ve) = match json_object_get_field_raw(raw, 0, str_field) {
        Some(r) => r,
        None => return RawApplyOutcome::Bail,
    };
    let val = &raw[vs..ve];
    if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"' {
        return RawApplyOutcome::Bail;
    }
    let inner = &val[1..val.len() - 1];
    if str_op == "test" {
        if inner.contains(&b'\\') {
            return RawApplyOutcome::Bail;
        }
    }
    let str_pass = match str_op {
        "startswith" => inner.starts_with(str_arg),
        "endswith" => inner.ends_with(str_arg),
        "contains" => {
            if str_arg.len() <= inner.len() {
                memchr::memmem::find(inner, str_arg).is_some()
            } else {
                false
            }
        }
        "test" => {
            let re = str_re.unwrap();
            let s = unsafe { std::str::from_utf8_unchecked(inner) };
            re.is_match(s)
        }
        "eq" => inner == str_arg,
        _ => unreachable!(),
    };
    if str_pass {
        emit_match(raw);
    }
    RawApplyOutcome::Emit
}

/// Apply the `(.x cmp1 N1) <conjunct> (.y cmp2 N2) <conjunct> ...`
/// raw-byte compound numeric-comparison fast path on a single JSON
/// record. Each comparison is a `<field> <cmp> <const>` predicate
/// against a single numeric field; the predicates are joined by a
/// single conjunct (`And` for `and`, `Or` for `or`) with short-
/// circuiting.
///
/// Caller passes pre-deduplicated buffers:
/// * `field_names` — unique field names referenced by the
///   comparisons (`json_object_get_num` is called once per name).
/// * `cmp_spec` — `(field_index, cmp_op, threshold)`. The
///   `field_index` is into `field_names`.
/// * `vals_buf` — reusable scratch sized to `field_names.len()`.
///
/// Bail discipline:
/// * Non-object input — [`RawApplyOutcome::Bail`].
/// * Any referenced field absent or non-numeric — Bail.
/// * Any `cmp_op` is not `Gt`/`Lt`/`Ge`/`Le`/`Eq`/`Ne` — Bail
///   (defensive — the detector should never produce these).
/// * `conjunct` is not `And`/`Or` — Bail (defensive).
/// * Buffer length mismatch — Bail (defensive).
///
/// On Emit, invokes `emit(result)` with the boolean conjunction.
pub fn apply_compound_field_cmp_raw<F>(
    raw: &[u8],
    field_names: &[&str],
    cmp_spec: &[(usize, BinOp, f64)],
    conjunct: BinOp,
    vals_buf: &mut [f64],
    mut emit: F,
) -> RawApplyOutcome
where
    F: FnMut(bool),
{
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    let is_and = match conjunct {
        BinOp::And => true,
        BinOp::Or => false,
        _ => return RawApplyOutcome::Bail,
    };
    let nf = field_names.len();
    if vals_buf.len() < nf {
        return RawApplyOutcome::Bail;
    }
    if nf == 2 {
        match json_object_get_two_nums(raw, 0, field_names[0], field_names[1]) {
            Some((a, b)) => { vals_buf[0] = a; vals_buf[1] = b; }
            None => return RawApplyOutcome::Bail,
        }
    } else {
        for (i, fname) in field_names.iter().enumerate() {
            match json_object_get_num(raw, 0, fname) {
                Some(n) => vals_buf[i] = n,
                None => return RawApplyOutcome::Bail,
            }
        }
    }
    let mut result = is_and;
    for (idx, op, threshold) in cmp_spec {
        let v = vals_buf[*idx];
        let cmp_result = match op {
            BinOp::Gt => v > *threshold,
            BinOp::Lt => v < *threshold,
            BinOp::Ge => v >= *threshold,
            BinOp::Le => v <= *threshold,
            BinOp::Eq => v == *threshold,
            BinOp::Ne => v != *threshold,
            _ => return RawApplyOutcome::Bail,
        };
        if is_and {
            if !cmp_result { result = false; break; }
        } else if cmp_result { result = true; break; }
    }
    emit(result);
    RawApplyOutcome::Emit
}

/// Apply the `.field <cmp> <const>` raw-byte numeric-comparison fast
/// path on a single JSON record (`<cmp>` ∈ Gt/Lt/Ge/Le/Eq/Ne) where
/// the field resolves to a JSON number and the right-hand side is a
/// compile-time numeric constant.
///
/// Bail discipline mirrors [`apply_field_field_cmp_raw`]:
/// * Field absent or non-numeric — [`RawApplyOutcome::Bail`].
/// * Non-comparison op (`Add`/`And`/etc.) — [`RawApplyOutcome::Bail`]
///   (defensive — the detector should never produce these).
/// * Non-object input — [`RawApplyOutcome::Bail`].
///
/// On success, invokes `emit(result)` with the boolean comparison.
pub fn apply_field_const_cmp_raw<F>(
    raw: &[u8],
    field: &str,
    cmp_op: BinOp,
    cval: f64,
    mut emit: F,
) -> RawApplyOutcome
where
    F: FnMut(bool),
{
    let n = match json_object_get_num(raw, 0, field) {
        Some(n) => n,
        None => return RawApplyOutcome::Bail,
    };
    let result = match cmp_op {
        BinOp::Gt => n > cval,
        BinOp::Lt => n < cval,
        BinOp::Ge => n >= cval,
        BinOp::Le => n <= cval,
        BinOp::Eq => n == cval,
        BinOp::Ne => n != cval,
        _ => return RawApplyOutcome::Bail,
    };
    emit(result);
    RawApplyOutcome::Emit
}

/// Apply the `(.field | index/rindex("str")) <op> <const>` raw-byte
/// fast path on a single JSON record.
///
/// `is_rindex = true` selects `rindex` (last match) over `index` (first
/// match). On a successful match, the helper computes the
/// **codepoint-position** (jq semantics — not byte position) of the
/// match in the field's UTF-8 content, then folds the arithmetic
/// `(op, n)` over it. The helper handles the field-missing /
/// no-match case with jq's `null + N = N` rule for `Add`; for
/// `Sub`/`Mul`/`Div` jq raises a type error on `null`, so the
/// helper Bails to let the generic path produce it.
///
/// Bail discipline:
/// * Non-object input — [`RawApplyOutcome::Bail`] (jq raises
///   `Cannot index <type> with "<field>"`).
/// * Field absent and `op` isn't `Add` — Bail.
/// * Field is non-string or escape-bearing string — Bail (the raw
///   scanner can't decode escapes; for non-strings jq raises a
///   type error).
/// * `op` is non-arithmetic (`Eq`/`And`/etc.) — Bail (defensive).
///
/// On Emit, invokes `emit(result)` so the apply-site owns
/// JSON-number formatting.
pub fn apply_field_index_arith_raw<F>(
    raw: &[u8],
    field: &str,
    search: &[u8],
    is_rindex: bool,
    op: BinOp,
    n: f64,
    mut emit: F,
) -> RawApplyOutcome
where
    F: FnMut(f64),
{
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
        return RawApplyOutcome::Bail;
    }
    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, field) {
        let val = &raw[vs..ve];
        if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
            || val[1..val.len() - 1].contains(&b'\\')
        {
            return RawApplyOutcome::Bail;
        }
        let inner = &val[1..val.len() - 1];
        let pos = if search.is_empty() {
            None
        } else if is_rindex {
            inner.windows(search.len()).rposition(|w| w == search)
        } else {
            inner.windows(search.len()).position(|w| w == search)
        };
        if let Some(p) = pos {
            let cp_pos = inner[..p].iter().filter(|&&b| (b & 0xC0) != 0x80).count() as f64;
            let result = match op {
                BinOp::Add => cp_pos + n,
                BinOp::Sub => cp_pos - n,
                BinOp::Mul => cp_pos * n,
                BinOp::Div => cp_pos / n,
                BinOp::Mod => jq_mod_f64(cp_pos, n).unwrap_or(f64::NAN),
                _ => return RawApplyOutcome::Bail,
            };
            if !result.is_finite() {
                return RawApplyOutcome::Bail;
            }
            emit(result);
        } else if matches!(op, BinOp::Add) {
            // index returned null; jq's `null + N = N`.
            emit(n);
        } else {
            // For Sub/Mul/Div jq raises on null; Bail to generic.
            return RawApplyOutcome::Bail;
        }
    } else if matches!(op, BinOp::Add) {
        // Object input + missing field → index returns null → `null + N = N`.
        emit(n);
    } else {
        return RawApplyOutcome::Bail;
    }
    RawApplyOutcome::Emit
}

/// Apply the `.field | <unary>` raw-byte multi-modal fast path on a
/// single JSON record. Output type depends on the unary op:
///
/// * `Length` / `Utf8ByteLength` → number (polymorphic — works on
///   string/array/object/null/number; jq raises on boolean).
/// * `Explode` → array of code points (string only).
/// * `AsciiDowncase` / `AsciiUpcase` → string (string only).
/// * `Floor` / `Ceil` / `Sqrt` / `Fabs` / `Abs` → number (number only).
/// * `ToString` → string (number only — strings/arrays/objects need
///   the generic path's `tostring` semantics).
///
/// The helper writes the raw output bytes plus a trailing `\n`
/// directly to `buf` on Emit, matching the field-update helper
/// pattern (the apply-site doesn't need to format anything itself).
///
/// Bail discipline:
/// * Non-object input — [`RawApplyOutcome::Bail`].
/// * Field absent and op isn't `Length`/`Utf8ByteLength` — Bail.
///   (Length on a missing field is `null | length = 0`, which is
///   safe to emit.)
/// * Field is a string with backslash escapes — Bail (the raw
///   scanner can't decode escapes; for `Length` it would miscount
///   code points, for `Explode`/`AsciiCase` it would emit wrong
///   bytes).
/// * Field is non-string for `Explode` / `AsciiDowncase` /
///   `AsciiUpcase` — Bail (jq raises type error).
/// * Field is non-numeric for the numeric unary set —
///   Bail.
/// * Field is boolean for `Length` — Bail (jq raises
///   `boolean has no length`).
/// * `Length` on malformed array/object (parse failure) — Bail.
/// * `ToString` on non-numeric field — Bail (the generic path
///   knows how to stringify strings / arrays / objects / null).
/// * Unsupported unary op (defensive) — Bail.
///
/// On Emit, writes the result bytes + `\n` to `buf`. Caller is
/// responsible for flushing `buf` to output as needed.
pub fn apply_field_unary_num_raw(
    raw: &[u8],
    field: &str,
    uop: UnaryOp,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    match uop {
        UnaryOp::Explode => {
            let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
                Some(r) => r,
                None => return RawApplyOutcome::Bail,
            };
            let val = &raw[vs..ve];
            if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
                || val[1..val.len() - 1].contains(&b'\\')
            {
                return RawApplyOutcome::Bail;
            }
            let inner = &val[1..val.len() - 1];
            buf.push(b'[');
            if inner.is_ascii() {
                for (i, &byte) in inner.iter().enumerate() {
                    if i > 0 { buf.push(b','); }
                    buf.extend_from_slice(itoa::Buffer::new().format(byte as i64).as_bytes());
                }
            } else {
                let mut first = true;
                for ch in unsafe { std::str::from_utf8_unchecked(inner) }.chars() {
                    if !first { buf.push(b','); }
                    first = false;
                    buf.extend_from_slice(itoa::Buffer::new().format(ch as i64).as_bytes());
                }
            }
            buf.extend_from_slice(b"]\n");
        }
        UnaryOp::AsciiDowncase | UnaryOp::AsciiUpcase => {
            let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
                Some(r) => r,
                None => return RawApplyOutcome::Bail,
            };
            let val = &raw[vs..ve];
            if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
                || val[1..val.len() - 1].contains(&b'\\')
            {
                return RawApplyOutcome::Bail;
            }
            buf.push(b'"');
            for &byte in &val[1..val.len() - 1] {
                buf.push(match uop {
                    UnaryOp::AsciiDowncase => if byte.is_ascii_uppercase() { byte + 32 } else { byte },
                    UnaryOp::AsciiUpcase => if byte.is_ascii_lowercase() { byte - 32 } else { byte },
                    _ => unreachable!(),
                });
            }
            buf.extend_from_slice(b"\"\n");
        }
        UnaryOp::Length | UnaryOp::Utf8ByteLength => {
            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, field) {
                let val = &raw[vs..ve];
                match val[0] {
                    b'"' => {
                        let inner = &val[1..val.len() - 1];
                        if inner.contains(&b'\\') {
                            return RawApplyOutcome::Bail;
                        }
                        let len = if matches!(uop, UnaryOp::Utf8ByteLength) {
                            inner.len()
                        } else if inner.is_ascii() {
                            inner.len()
                        } else {
                            unsafe { std::str::from_utf8_unchecked(inner) }.chars().count()
                        };
                        push_jq_number_bytes(buf, len as f64);
                        buf.push(b'\n');
                    }
                    b'[' | b'{' => match json_value_length(val, 0) {
                        Some(len) => {
                            push_jq_number_bytes(buf, len as f64);
                            buf.push(b'\n');
                        }
                        None => return RawApplyOutcome::Bail,
                    },
                    b'n' => buf.extend_from_slice(b"0\n"),
                    b't' | b'f' => return RawApplyOutcome::Bail,
                    _ => match json_object_get_num(raw, 0, field) {
                        Some(n) => {
                            push_jq_number_bytes(buf, n.abs());
                            buf.push(b'\n');
                        }
                        None => return RawApplyOutcome::Bail,
                    },
                }
            } else {
                // Object input + missing field: jq's `null | length = 0`.
                buf.extend_from_slice(b"0\n");
            }
        }
        UnaryOp::ToString => {
            // Only the numeric-field shape stays in the fast path; strings,
            // arrays, etc. need the generic path's `tostring` semantics.
            match json_object_get_num(raw, 0, field) {
                Some(n) => {
                    buf.push(b'"');
                    push_jq_number_bytes(buf, n);
                    buf.extend_from_slice(b"\"\n");
                }
                None => return RawApplyOutcome::Bail,
            }
        }
        UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Sqrt | UnaryOp::Fabs | UnaryOp::Abs => {
            let n = match json_object_get_num(raw, 0, field) {
                Some(v) => v,
                None => return RawApplyOutcome::Bail,
            };
            let result = match uop {
                UnaryOp::Floor => n.floor(),
                UnaryOp::Ceil => n.ceil(),
                UnaryOp::Sqrt => n.sqrt(),
                UnaryOp::Fabs | UnaryOp::Abs => n.abs(),
                _ => unreachable!(),
            };
            push_jq_number_bytes(buf, result);
            buf.push(b'\n');
        }
        _ => return RawApplyOutcome::Bail,
    }
    RawApplyOutcome::Emit
}

/// Apply the `(.field | <unary>) <op1> <c1> <op2> <c2> ...` raw-byte
/// fast path on a single JSON record. The unary op is applied to the
/// field value first, then a left-fold of `(BinOp, f64)` arithmetic
/// pairs runs on the result.
///
/// Supported unary ops:
/// * `Length` — works on string/array/object/null/number (matches jq's
///   polymorphic `length`). On a numeric field, evaluates to `abs(n)`
///   per jq semantics.
/// * `Floor`/`Ceil`/`Sqrt`/`Fabs`/`Abs`/`Round` — numeric, requires a
///   number-valued field (Bail otherwise).
///
/// Bail discipline:
/// * Non-object input — [`RawApplyOutcome::Bail`]. The prior fast path
///   silently emitted `0` when invoked on `(.x | length)` against a
///   number/string root (a #83-class divergence; jq raises
///   `Cannot index <type> with "x"`).
/// * For `Length` on a string field: backslash-escaped strings Bail
///   (the raw scanner can't decode the escape to count code points).
/// * For `Length` on an array/object field: malformed value Bails
///   (`json_value_length` returns None).
/// * For numeric unary ops on a non-numeric field — Bail.
/// * Unsupported unary op (defensive — detector should never produce
///   these for this shape) — Bail.
/// * Non-arith op in `arith_steps` — Bail (defensive).
/// * Non-finite final result (div-by-zero / overflow / NaN) — Bail so
///   the generic path raises jq's `/ by zero` etc.
///
/// On success, invokes `emit(result)` so the apply-site owns
/// JSON-number formatting.
pub fn apply_field_unary_arith_raw<F>(
    raw: &[u8],
    field: &str,
    uop: UnaryOp,
    arith_steps: &[(BinOp, f64)],
    mut emit: F,
) -> RawApplyOutcome
where
    F: FnMut(f64),
{
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    let is_length = matches!(uop, UnaryOp::Length);
    let base = if is_length {
        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, field) {
            let val = &raw[vs..ve];
            match val[0] {
                b'"' => {
                    let inner = &val[1..val.len() - 1];
                    if inner.contains(&b'\\') {
                        return RawApplyOutcome::Bail;
                    }
                    if inner.is_ascii() {
                        inner.len() as f64
                    } else {
                        unsafe { std::str::from_utf8_unchecked(inner) }.chars().count() as f64
                    }
                }
                b'[' | b'{' => match json_value_length(val, 0) {
                    Some(l) => l as f64,
                    None => return RawApplyOutcome::Bail,
                },
                b'n' => 0.0,
                _ => match json_object_get_num(raw, 0, field) {
                    Some(n) => n.abs(),
                    None => return RawApplyOutcome::Bail,
                },
            }
        } else {
            // Object input + missing field: jq's `null | length = 0`.
            0.0
        }
    } else {
        let n = match json_object_get_num(raw, 0, field) {
            Some(v) => v,
            None => return RawApplyOutcome::Bail,
        };
        match uop {
            UnaryOp::Floor => n.floor(),
            UnaryOp::Ceil => n.ceil(),
            UnaryOp::Sqrt => n.sqrt(),
            UnaryOp::Fabs | UnaryOp::Abs => n.abs(),
            UnaryOp::Round => n.round(),
            _ => return RawApplyOutcome::Bail,
        }
    };
    let mut v = base;
    for (op, c) in arith_steps {
        v = match op {
            BinOp::Add => v + c,
            BinOp::Sub => v - c,
            BinOp::Mul => v * c,
            BinOp::Div => v / c,
            BinOp::Mod => jq_mod_f64(v, *c).unwrap_or(f64::NAN),
            _ => return RawApplyOutcome::Bail,
        };
    }
    if !v.is_finite() {
        return RawApplyOutcome::Bail;
    }
    emit(v);
    RawApplyOutcome::Emit
}

/// Apply the `.field <op> <const>` (or `<const> <op> .field`) raw-byte
/// fast path on a single JSON record, optionally followed by a unary
/// math op (`floor`/`ceil`/`sqrt`/`fabs`/`abs`).
///
/// `const_left = true` swaps operands so the constant is on the left
/// (e.g. `2 - .x`). `uop_opt = None` skips the unary stage.
///
/// Bail discipline:
/// * Field absent or non-numeric — [`RawApplyOutcome::Bail`] so the
///   generic path raises jq's type error.
/// * Non-arithmetic op (`Eq`/`And`/etc.) — [`RawApplyOutcome::Bail`]
///   (defensive — the detector should never produce these).
/// * Unary op not in the supported math set
///   (`Floor`/`Ceil`/`Sqrt`/`Fabs`/`Abs`) — [`RawApplyOutcome::Bail`]
///   (defensive — same reason).
/// * Final result non-finite (div-by-zero, sqrt of negative, etc.) —
///   [`RawApplyOutcome::Bail`] so the generic path raises jq's
///   `/ by zero` etc. Without this guard the fast path silently emits
///   `Infinity`/`NaN`-formatted bytes (#83-class bug in the prior
///   apply-site).
/// * Non-object input — [`RawApplyOutcome::Bail`].
///
/// On success, invokes `emit(result)` so the apply-site owns
/// JSON-number formatting.
pub fn apply_field_binop_const_unary_raw<F>(
    raw: &[u8],
    field: &str,
    bop: BinOp,
    cval: f64,
    uop_opt: Option<UnaryOp>,
    const_left: bool,
    mut emit: F,
) -> RawApplyOutcome
where
    F: FnMut(f64),
{
    let n = match json_object_get_num(raw, 0, field) {
        Some(v) => v,
        None => return RawApplyOutcome::Bail,
    };
    let (a, b) = if const_left { (cval, n) } else { (n, cval) };
    let mid = match bop {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Mod => jq_mod_f64(a, b).unwrap_or(f64::NAN),
        _ => return RawApplyOutcome::Bail,
    };
    let result = if let Some(uop) = uop_opt {
        match uop {
            UnaryOp::Floor => mid.floor(),
            UnaryOp::Ceil => mid.ceil(),
            UnaryOp::Sqrt => mid.sqrt(),
            UnaryOp::Fabs | UnaryOp::Abs => mid.abs(),
            _ => return RawApplyOutcome::Bail,
        }
    } else {
        mid
    };
    if !result.is_finite() {
        return RawApplyOutcome::Bail;
    }
    emit(result);
    RawApplyOutcome::Emit
}

/// Apply the `(.x <op1> .y) <op2> <const>` raw-byte two-field-and-const
/// arithmetic fast path on a single JSON record. Both `.x` and `.y`
/// must resolve to JSON numbers; `<op1>` and `<op2>` are arithmetic
/// (`Add`/`Sub`/`Mul`/`Div`/`Mod`).
///
/// Bail discipline:
/// * Either field absent or non-numeric — [`RawApplyOutcome::Bail`]
///   so the generic path raises jq's type error.
/// * Either op is non-arithmetic (`Eq`/`And`/etc.) —
///   [`RawApplyOutcome::Bail`] (defensive — the detector should never
///   produce these, but the helper rejects them at the boundary).
/// * Inner or final result non-finite (div-by-zero / mod-by-zero,
///   IEEE NaN/∞) — [`RawApplyOutcome::Bail`] so the generic path
///   raises jq's `/ by zero` etc.
/// * Non-object input — [`RawApplyOutcome::Bail`].
///
/// On success, invokes `emit(result)` so the apply-site owns
/// JSON-number formatting.
pub fn apply_two_field_binop_const_raw<F>(
    raw: &[u8],
    field_a: &str,
    field_b: &str,
    op1: BinOp,
    op2: BinOp,
    cval: f64,
    mut emit: F,
) -> RawApplyOutcome
where
    F: FnMut(f64),
{
    let (a, b) = match json_object_get_two_nums(raw, 0, field_a, field_b) {
        Some(p) => p,
        None => return RawApplyOutcome::Bail,
    };
    let inner = match op1 {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Mod => jq_mod_f64(a, b).unwrap_or(f64::NAN),
        _ => return RawApplyOutcome::Bail,
    };
    let result = match op2 {
        BinOp::Add => inner + cval,
        BinOp::Sub => inner - cval,
        BinOp::Mul => inner * cval,
        BinOp::Div => inner / cval,
        BinOp::Mod => jq_mod_f64(inner, cval).unwrap_or(f64::NAN),
        _ => return RawApplyOutcome::Bail,
    };
    if !result.is_finite() {
        return RawApplyOutcome::Bail;
    }
    emit(result);
    RawApplyOutcome::Emit
}

/// Apply the `.field <op1> <c1> <op2> <c2> ...` raw-byte arithmetic chain
/// fast path on a single JSON record (a left-fold of `(BinOp, f64)` pairs
/// over a single numeric field).
///
/// The detector rejects compile-time div-by-zero / mod-by-zero constants
/// (`detect_field_arith_chain` in `src/interpreter.rs`), so the chain's
/// `Div`/`Mod` ops are guaranteed to have a non-zero divisor. The
/// helper trusts this invariant — non-finite results (e.g. overflow)
/// are emitted as-is, matching `push_jq_number_bytes`'s saturating
/// behaviour.
///
/// Bail discipline:
/// * Field absent or non-numeric — [`RawApplyOutcome::Bail`].
/// * Non-arithmetic op (`Eq`/`And`/etc.) — [`RawApplyOutcome::Bail`]
///   (defensive — the detector should never produce these).
/// * Non-object input — [`RawApplyOutcome::Bail`] (number lookup
///   fails).
///
/// On a passing type-guard, invokes `emit(result)` so the apply-site
/// owns JSON-number formatting.
pub fn apply_field_arith_chain_raw<F>(
    raw: &[u8],
    field: &str,
    ops: &[(BinOp, f64)],
    mut emit: F,
) -> RawApplyOutcome
where
    F: FnMut(f64),
{
    let n = match json_object_get_num(raw, 0, field) {
        Some(n) => n,
        None => return RawApplyOutcome::Bail,
    };
    let mut result = n;
    for (op, c) in ops {
        result = match op {
            BinOp::Add => result + c,
            BinOp::Sub => result - c,
            BinOp::Mul => result * c,
            BinOp::Div => result / c,
            BinOp::Mod => jq_mod_f64(result, *c).unwrap_or(f64::NAN),
            _ => return RawApplyOutcome::Bail,
        };
    }
    emit(result);
    RawApplyOutcome::Emit
}

/// Apply a generic `ArithExpr` (a pure numeric expression over named
/// fields and constants) raw-byte fast path on a single JSON record,
/// optionally followed by a math unary op (`floor`/`ceil`/`sqrt`/
/// `fabs`/`round`).
///
/// Covers both `detect_numeric_expr` (no unary) and
/// `detect_numeric_expr_unary`. Caller provides:
/// * `fields` — list of field names referenced by the expression
///   (`ArithExpr::Field(idx)` resolves into this slice).
/// * `arith` — the compiled expression (built by `simplify_expr`).
/// * `math_op` — `Some(_)` to apply the trailing unary op, `None` to
///   skip it.
/// * `ranges_buf` and `vals_buf` — reusable scratch buffers sized
///   to `fields.len()`. Hoisting them across the input stream avoids
///   per-record allocation; required for the 3-or-more-fields
///   path that uses `json_object_get_fields_raw_buf`.
///
/// Bail discipline:
/// * Non-object input — [`RawApplyOutcome::Bail`].
/// * Any field absent or non-numeric — Bail.
/// * Buffer length mismatch (caller bug) — Bail (defensive).
/// * Non-finite final result — Bail so the generic path raises
///   jq's `/ by zero` etc. (The prior fast path silently emitted
///   `1.7976931348623157e+308` / `null` for ±Infinity / NaN —
///   another #83-class divergence the structural Bail closes.)
///
/// On Emit, invokes `emit(result)` so the apply-site owns
/// JSON-number formatting.
pub fn apply_numeric_expr_raw<F>(
    raw: &[u8],
    fields: &[&str],
    arith: &ArithExpr,
    math_op: Option<MathUnary>,
    ranges_buf: &mut [(usize, usize)],
    vals_buf: &mut [f64],
    mut emit: F,
) -> RawApplyOutcome
where
    F: FnMut(f64),
{
    if raw.is_empty() || raw[0] != b'{' {
        return RawApplyOutcome::Bail;
    }
    let nf = fields.len();
    if ranges_buf.len() < nf || vals_buf.len() < nf {
        return RawApplyOutcome::Bail;
    }
    let ok = if nf == 1 {
        match json_object_get_num(raw, 0, fields[0]) {
            Some(v) => { vals_buf[0] = v; true }
            None => false,
        }
    } else if nf == 2 {
        match json_object_get_two_nums(raw, 0, fields[0], fields[1]) {
            Some((a, b)) => { vals_buf[0] = a; vals_buf[1] = b; true }
            None => false,
        }
    } else if json_object_get_fields_raw_buf(raw, 0, fields, ranges_buf) {
        let mut all_ok = true;
        for (i, &(s, e)) in ranges_buf[..nf].iter().enumerate() {
            match fast_float::parse::<f64, _>(unsafe { std::str::from_utf8_unchecked(&raw[s..e]) }) {
                Ok(n) => vals_buf[i] = n,
                Err(_) => { all_ok = false; break; }
            }
        }
        all_ok
    } else {
        false
    };
    if !ok {
        return RawApplyOutcome::Bail;
    }
    let base = arith.eval(&vals_buf[..nf]);
    let result = match math_op {
        Some(MathUnary::Sqrt) => base.sqrt(),
        Some(MathUnary::Floor) => base.floor(),
        Some(MathUnary::Ceil) => base.ceil(),
        Some(MathUnary::Fabs) => base.abs(),
        Some(MathUnary::Round) => base.round(),
        None => base,
    };
    if !result.is_finite() {
        return RawApplyOutcome::Bail;
    }
    emit(result);
    RawApplyOutcome::Emit
}

/// Apply the `select(.field <cmp> N)` raw-byte fast path on a single
/// JSON record.
///
/// `select` is a *filter* — it emits the input unchanged when the
/// predicate is true, and emits nothing when it's false. The helper
/// passes the raw record bytes to `emit_match` only on a passing
/// comparison; the closure typically writes the record + newline to
/// the output buffer.
///
/// Bail discipline (#199):
/// * Non-object input — [`RawApplyOutcome::Bail`] so jq's
///   `Cannot index <type> with "<field>"` surfaces.
/// * Field absent or non-numeric — [`RawApplyOutcome::Bail`] (the
///   generic path applies jq's total-order semantics, e.g.
///   `null < N == true`, which the raw path can't represent without
///   re-implementing the ordering).
/// * Non-comparison op (`Add`/`And`/etc.) — [`RawApplyOutcome::Bail`]
///   (defensive).
///
/// On success the helper returns [`RawApplyOutcome::Emit`] regardless
/// of whether the predicate fired (no output is the right answer
/// when the predicate is false).
pub fn apply_select_cmp_raw<F>(
    raw: &[u8],
    field: &str,
    op: BinOp,
    threshold: f64,
    mut emit_match: F,
) -> RawApplyOutcome
where
    F: FnMut(&[u8]),
{
    if raw.first() != Some(&b'{') {
        return RawApplyOutcome::Bail;
    }
    let val = match json_object_get_num(raw, 0, field) {
        Some(v) => v,
        None => return RawApplyOutcome::Bail,
    };
    let pass = match op {
        BinOp::Gt => val > threshold,
        BinOp::Lt => val < threshold,
        BinOp::Ge => val >= threshold,
        BinOp::Le => val <= threshold,
        BinOp::Eq => val == threshold,
        BinOp::Ne => val != threshold,
        _ => return RawApplyOutcome::Bail,
    };
    if pass {
        emit_match(raw);
    }
    RawApplyOutcome::Emit
}

/// Apply the `select(.field == null)` / `select(.field != null)` raw-byte
/// fast path on a single JSON record.
///
/// `is_eq = true` corresponds to `== null`; `is_eq = false` to `!= null`.
/// A missing field counts as null (jq's `null | .x == null` is true).
///
/// Bail discipline:
/// * Non-object input — [`RawApplyOutcome::Bail`] so jq's
///   `Cannot index <type>` surfaces (#199 sibling).
///
/// On a passing type-guard the helper returns
/// [`RawApplyOutcome::Emit`] regardless of whether the predicate
/// fires; the closure is invoked only when it does.
pub fn apply_select_field_null_raw<F>(
    raw: &[u8],
    field: &str,
    is_eq: bool,
    mut emit_match: F,
) -> RawApplyOutcome
where
    F: FnMut(&[u8]),
{
    if raw.first() != Some(&b'{') {
        return RawApplyOutcome::Bail;
    }
    let is_null = if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, field) {
        &raw[vs..ve] == b"null"
    } else {
        true
    };
    let pass = if is_eq { is_null } else { !is_null };
    if pass {
        emit_match(raw);
    }
    RawApplyOutcome::Emit
}

/// Apply the `select(.field <cmp> "value")` raw-byte fast path on a
/// single JSON record (`<cmp>` ∈ Gt/Lt/Ge/Le/Eq/Ne).
///
/// Bail discipline:
/// * Non-object input — [`RawApplyOutcome::Bail`] (jq raises
///   `Cannot index <type>`).
/// * Field absent — [`RawApplyOutcome::Bail`] (jq's `null cmp "x"`
///   has cross-type-order semantics the raw path can't represent).
/// * For Eq/Ne: field value is a quoted string with any `\` escape —
///   [`RawApplyOutcome::Bail`] (raw byte equality would say "no
///   match" while jq decodes the escape and may match).
/// * For Gt/Lt/Ge/Le: field value isn't an escape-free quoted string —
///   [`RawApplyOutcome::Bail`] (jq's cross-type ordering applies, and
///   the raw scanner can't decode escape-bearing strings).
/// * Non-comparison op (`Add`/`And`/etc.) — [`RawApplyOutcome::Bail`]
///   (defensive).
///
/// On a passing predicate the helper invokes `emit_match(raw)` with
/// the original record bytes.
pub fn apply_select_str_raw<F>(
    raw: &[u8],
    field: &str,
    cmp_op: BinOp,
    expected: &str,
    mut emit_match: F,
) -> RawApplyOutcome
where
    F: FnMut(&[u8]),
{
    if raw.first() != Some(&b'{') {
        return RawApplyOutcome::Bail;
    }
    let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
        Some(r) => r,
        None => return RawApplyOutcome::Bail,
    };
    let val_bytes = &raw[vs..ve];
    let pass = match cmp_op {
        BinOp::Eq | BinOp::Ne => {
            // Eq/Ne can byte-compare directly *unless* the field is an
            // escape-bearing string — then the raw bytes wouldn't match
            // a literal that decodes equal.
            if val_bytes.len() >= 2 && val_bytes[0] == b'"' && val_bytes[val_bytes.len() - 1] == b'"'
                && val_bytes[1..val_bytes.len() - 1].contains(&b'\\')
            {
                return RawApplyOutcome::Bail;
            }
            // Build expected JSON: "value"
            // For non-string field values (numbers, etc.), jq's
            // cross-type equality says they're never equal to a string,
            // which matches our byte comparison (val_bytes ≠ "value").
            let mut want = Vec::with_capacity(expected.len() + 2);
            want.push(b'"');
            want.extend_from_slice(expected.as_bytes());
            want.push(b'"');
            let eq = val_bytes == want.as_slice();
            if matches!(cmp_op, BinOp::Eq) { eq } else { !eq }
        }
        BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le => {
            // Ordering only works on string-with-no-escape; everything
            // else (non-string, escape-bearing) routes through generic
            // for jq's cross-type ordering rules.
            if val_bytes.len() < 2 || val_bytes[0] != b'"'
                || val_bytes[val_bytes.len() - 1] != b'"'
                || val_bytes[1..val_bytes.len() - 1].contains(&b'\\')
            {
                return RawApplyOutcome::Bail;
            }
            let inner = &val_bytes[1..val_bytes.len() - 1];
            let cmp = inner.cmp(expected.as_bytes());
            match cmp_op {
                BinOp::Gt => cmp == std::cmp::Ordering::Greater,
                BinOp::Lt => cmp == std::cmp::Ordering::Less,
                BinOp::Ge => cmp != std::cmp::Ordering::Less,
                BinOp::Le => cmp != std::cmp::Ordering::Greater,
                _ => unreachable!(),
            }
        }
        _ => return RawApplyOutcome::Bail,
    };
    if pass {
        emit_match(raw);
    }
    RawApplyOutcome::Emit
}

/// Apply the `.field |= length` raw-byte update fast path on a single
/// JSON record, writing the updated object bytes to `buf`.
///
/// The fast path commits only when the input is an object whose `field`
/// is a JSON string — `length` is then the codepoint count of that
/// string (decoding any `\` escapes). For every other shape jq's
/// `length` has a different definition (array length, object key
/// count, abs(number), etc.), so the helper bails:
///
/// * Non-object input — [`RawApplyOutcome::Bail`].
/// * Field absent or value isn't a JSON string —
///   [`RawApplyOutcome::Bail`].
/// * Escape-bearing string that fails JSON unescape (very rare; e.g.
///   broken `\uXXXX`) — [`RawApplyOutcome::Bail`].
///
/// On success, `buf` is appended the updated object bytes (without a
/// trailing newline — the caller handles framing for pretty vs
/// compact output).
pub fn apply_field_update_length_raw(raw: &[u8], field: &str, buf: &mut Vec<u8>) -> RawApplyOutcome {
    if json_object_update_field_length(raw, 0, field, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `.field |= tostring` raw-byte update fast path on a single
/// JSON record, writing the updated object bytes to `buf`.
///
/// jq's `tostring` is identity on strings and JSON-stringifies every
/// other type (numbers → `"42"`, booleans → `"true"`, `null` → `"null"`,
/// arrays/objects → their pretty/compact JSON representation). The
/// underlying value-side function handles strings + scalars; arrays and
/// objects bail to generic since stringifying them requires the full
/// recursive JSON encoder.
///
/// Bail discipline:
/// * Non-object input — [`RawApplyOutcome::Bail`].
/// * Field absent — [`RawApplyOutcome::Bail`].
/// * Field value is an array or object literal —
///   [`RawApplyOutcome::Bail`] (generic does the recursive
///   stringification).
///
/// On success, `buf` is appended the updated object bytes (without a
/// trailing newline).
pub fn apply_field_update_tostring_raw(raw: &[u8], field: &str, buf: &mut Vec<u8>) -> RawApplyOutcome {
    if json_object_update_field_tostring(raw, 0, field, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `.field |= test("pattern")` raw-byte update fast path on a
/// single JSON record, writing the updated object bytes (with `field`
/// replaced by `true` / `false`) to `buf`.
///
/// Bail discipline (driven by the value-side
/// `json_object_update_field_test`):
/// * Non-object input — [`RawApplyOutcome::Bail`].
/// * Field absent or value isn't a JSON string —
///   [`RawApplyOutcome::Bail`] (jq raises on `null|test` /
///   `number|test` etc.).
///
/// On success, `buf` is appended the updated object bytes.
pub fn apply_field_update_test_raw(
    raw: &[u8],
    field: &str,
    re: &regex::Regex,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_update_field_test(raw, 0, field, re, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `.field |= sub/gsub("pattern"; "replacement")` raw-byte
/// update fast path on a single JSON record, writing the updated object
/// bytes to `buf`.
///
/// `is_global` selects between `replace_all` (gsub) and `replace` (sub).
///
/// Bail discipline (driven by the value-side
/// `json_object_update_field_gsub`):
/// * Non-object input — [`RawApplyOutcome::Bail`].
/// * Field absent or value isn't a JSON string —
///   [`RawApplyOutcome::Bail`].
///
/// On success, `buf` is appended the updated object bytes.
pub fn apply_field_update_gsub_raw(
    raw: &[u8],
    field: &str,
    re: &regex::Regex,
    replacement: &str,
    is_global: bool,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_update_field_gsub(raw, 0, field, re, replacement, is_global, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `.field |= ascii_downcase / ascii_upcase` raw-byte update
/// fast path on a single JSON record, writing the updated object bytes
/// to `buf`. `is_upcase = true` selects upcase.
///
/// Bail discipline (driven by the value-side
/// `json_object_update_field_case`):
/// * Non-object input — [`RawApplyOutcome::Bail`].
/// * Field absent or value isn't a JSON string —
///   [`RawApplyOutcome::Bail`].
///
/// On success, `buf` is appended the updated object bytes.
pub fn apply_field_update_case_raw(
    raw: &[u8],
    field: &str,
    is_upcase: bool,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_update_field_case(raw, 0, field, is_upcase, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `.field |= split("sep") | .[0]` raw-byte update fast path
/// on a single JSON record (drop everything after the first `sep`,
/// keeping the prefix as the new field value).
///
/// Bails on non-object input, missing/non-string field. The generic
/// path handles other shapes and any escape-decode quirks.
pub fn apply_field_update_split_first_raw(
    raw: &[u8],
    field: &str,
    sep: &str,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_update_field_split_first(raw, 0, field, sep, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `.field |= split("sep") | .[-1]` raw-byte update fast path
/// (keep everything after the last `sep`).
///
/// Bails on non-object input, missing/non-string field.
pub fn apply_field_update_split_last_raw(
    raw: &[u8],
    field: &str,
    sep: &str,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_update_field_split_last(raw, 0, field, sep, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `.field |= ltrimstr/rtrimstr("s")` raw-byte update fast
/// path. `is_rtrim` selects between left- and right-trim.
///
/// Bails on non-object input, missing/non-string field.
pub fn apply_field_update_trim_raw(
    raw: &[u8],
    field: &str,
    trim_str: &str,
    is_rtrim: bool,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_update_field_trim(raw, 0, field, trim_str, is_rtrim, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `.field |= .[from:to]` raw-byte update fast path.
///
/// Bails on non-object input, missing/non-string field; the generic
/// path handles arrays and other types.
pub fn apply_field_update_slice_raw(
    raw: &[u8],
    field: &str,
    from: Option<i64>,
    to: Option<i64>,
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_update_field_slice(raw, 0, field, from, to, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the fused `.field |= if . == "cond" then THEN else ELSE end`
/// raw-byte update fast path (`json_object_update_field_str_map`).
/// `cond_str` is the literal-string condition; `then_json` / `else_json`
/// are pre-encoded JSON values.
///
/// Bails on non-object input, missing/non-string field.
pub fn apply_field_update_str_map_raw(
    raw: &[u8],
    field: &str,
    cond_str: &[u8],
    then_json: &[u8],
    else_json: &[u8],
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_update_field_str_map(raw, 0, field, cond_str, then_json, else_json, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the fused `.field |= "PREFIX" + . + "SUFFIX"` raw-byte update
/// fast path. `prefix` and `suffix` are JSON-escaped strings (without
/// surrounding quotes).
///
/// Bails on non-object input, missing/non-string field.
pub fn apply_field_update_str_concat_raw(
    raw: &[u8],
    field: &str,
    prefix: &[u8],
    suffix: &[u8],
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_update_field_str_concat(raw, 0, field, prefix, suffix, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `del(.field)` raw-byte fast path on a single JSON record,
/// writing the object with `field` removed to `buf`.
///
/// Bail discipline:
/// * Non-object input — [`RawApplyOutcome::Bail`] (`del(.field)` on
///   `null` returns `null`; on numbers / arrays jq raises).
///
/// Field absent emits the input object unchanged (jq's `del(.x)` on
/// `{}` returns `{}` — no error).
pub fn apply_del_field_raw(raw: &[u8], field: &str, buf: &mut Vec<u8>) -> RawApplyOutcome {
    if json_object_del_field(raw, 0, field, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `del(.a, .b, ...)` raw-byte fast path on a single JSON
/// record, writing the object with all listed fields removed to `buf`.
///
/// Bail discipline mirrors [`apply_del_field_raw`]: non-object input
/// bails. Missing fields are silently skipped.
pub fn apply_del_fields_raw(
    raw: &[u8],
    fields: &[&str],
    buf: &mut Vec<u8>,
) -> RawApplyOutcome {
    if json_object_del_fields(raw, 0, fields, buf) {
        RawApplyOutcome::Emit
    } else {
        RawApplyOutcome::Bail
    }
}

/// Apply the `select(.field | startswith/endswith/contains("arg"))`
/// raw-byte fast path on a single JSON record.
///
/// All three jq builtins require string inputs; jq raises
/// `<builtin>() requires string inputs` on a non-string. The helper
/// therefore bails on every non-string-no-escape shape so the generic
/// path can produce the real error.
///
/// Bail discipline:
/// * Non-object input — [`RawApplyOutcome::Bail`].
/// * Field absent, value isn't a quoted string, or contains any
///   `\` escape — [`RawApplyOutcome::Bail`].
/// * Unknown builtin name (defensive — only `startswith`, `endswith`,
///   `contains` are handled) — [`RawApplyOutcome::Bail`].
///
/// On a passing predicate the helper invokes `emit_match(raw)` with
/// the original record bytes.
pub fn apply_select_str_test_raw<F>(
    raw: &[u8],
    field: &str,
    builtin: &str,
    arg: &str,
    mut emit_match: F,
) -> RawApplyOutcome
where
    F: FnMut(&[u8]),
{
    if raw.first() != Some(&b'{') {
        return RawApplyOutcome::Bail;
    }
    let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
        Some(r) => r,
        None => return RawApplyOutcome::Bail,
    };
    let val = &raw[vs..ve];
    if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
        || val[1..val.len() - 1].contains(&b'\\')
    {
        return RawApplyOutcome::Bail;
    }
    let inner = &val[1..val.len() - 1];
    let arg_bytes = arg.as_bytes();
    let pass = match builtin {
        "startswith" => inner.starts_with(arg_bytes),
        "endswith" => inner.ends_with(arg_bytes),
        "contains" => {
            if arg_bytes.is_empty() {
                true
            } else if arg_bytes.len() == 1 {
                memchr::memchr(arg_bytes[0], inner).is_some()
            } else {
                inner
                    .windows(arg_bytes.len())
                    .any(|w| w == arg_bytes)
            }
        }
        _ => return RawApplyOutcome::Bail,
    };
    if pass {
        emit_match(raw);
    }
    RawApplyOutcome::Emit
}

/// Apply the `select(.field <arith_chain> <cmp> N)` raw-byte fast path
/// on a single JSON record.
///
/// The detector excludes compile-time div/mod-by-zero in the chain
/// (`detect_select_arith_cmp` in `src/interpreter.rs`); the helper
/// trusts this.
///
/// Bail discipline:
/// * Non-object input — [`RawApplyOutcome::Bail`] so jq's
///   `Cannot index <type>` surfaces.
/// * Field absent or non-numeric — [`RawApplyOutcome::Bail`] so the
///   generic path produces jq's verdict (the chain may raise on
///   `string + N`, or silently emit no output on `null + N` ≤ threshold,
///   but that's not the raw path's call to make).
/// * Non-arithmetic op in the chain or non-comparison `cmp_op`
///   (defensive) — [`RawApplyOutcome::Bail`].
///
/// On a passing predicate the helper invokes `emit_match(raw)` with
/// the original record bytes.
pub fn apply_select_arith_cmp_raw<F>(
    raw: &[u8],
    field: &str,
    arith_ops: &[(BinOp, f64)],
    cmp_op: BinOp,
    threshold: f64,
    mut emit_match: F,
) -> RawApplyOutcome
where
    F: FnMut(&[u8]),
{
    if raw.first() != Some(&b'{') {
        return RawApplyOutcome::Bail;
    }
    let mut val = match json_object_get_num(raw, 0, field) {
        Some(v) => v,
        None => return RawApplyOutcome::Bail,
    };
    for (op, n) in arith_ops {
        val = match op {
            BinOp::Add => val + n,
            BinOp::Sub => val - n,
            BinOp::Mul => val * n,
            BinOp::Div => val / n,
            BinOp::Mod => jq_mod_f64(val, *n).unwrap_or(f64::NAN),
            _ => return RawApplyOutcome::Bail,
        };
    }
    let pass = match cmp_op {
        BinOp::Gt => val > threshold,
        BinOp::Lt => val < threshold,
        BinOp::Ge => val >= threshold,
        BinOp::Le => val <= threshold,
        BinOp::Eq => val == threshold,
        BinOp::Ne => val != threshold,
        _ => return RawApplyOutcome::Bail,
    };
    if pass {
        emit_match(raw);
    }
    RawApplyOutcome::Emit
}

/// Apply the `.field <arith_chain> <cmp> <threshold>` raw-byte fast path
/// on a single JSON record — fold a `(BinOp, f64)` arithmetic chain over
/// a numeric field, then compare the result against `threshold`.
///
/// The detector excludes compile-time div/mod-by-zero constants in the
/// arithmetic chain (`detect_arith_chain_cmp` in
/// `src/interpreter.rs`), so the helper trusts the chain's divisors are
/// non-zero. Non-finite results from overflow are passed through to the
/// comparison directly (matching the existing apply-site).
///
/// Bail discipline mirrors [`apply_field_const_cmp_raw`] plus the chain
/// helper:
/// * Field absent or non-numeric — [`RawApplyOutcome::Bail`].
/// * Non-arithmetic op in the chain or non-comparison `cmp_op`
///   (defensive) — [`RawApplyOutcome::Bail`].
/// * Non-object input — [`RawApplyOutcome::Bail`].
///
/// On success, invokes `emit(result)` with the boolean comparison.
pub fn apply_arith_chain_cmp_raw<F>(
    raw: &[u8],
    field: &str,
    arith_ops: &[(BinOp, f64)],
    cmp_op: BinOp,
    threshold: f64,
    mut emit: F,
) -> RawApplyOutcome
where
    F: FnMut(bool),
{
    let mut n = match json_object_get_num(raw, 0, field) {
        Some(v) => v,
        None => return RawApplyOutcome::Bail,
    };
    for (op, c) in arith_ops {
        n = match op {
            BinOp::Add => n + c,
            BinOp::Sub => n - c,
            BinOp::Mul => n * c,
            BinOp::Div => n / c,
            BinOp::Mod => jq_mod_f64(n, *c).unwrap_or(f64::NAN),
            _ => return RawApplyOutcome::Bail,
        };
    }
    let result = match cmp_op {
        BinOp::Gt => n > threshold,
        BinOp::Lt => n < threshold,
        BinOp::Ge => n >= threshold,
        BinOp::Le => n <= threshold,
        BinOp::Eq => n == threshold,
        BinOp::Ne => n != threshold,
        _ => return RawApplyOutcome::Bail,
    };
    emit(result);
    RawApplyOutcome::Emit
}

/// Apply the `.field | <string-builtin>(arg)` raw-byte fast path on a
/// single JSON record. The detector (`detect_field_str_builtin`) fuses
/// nine string builtins under one shape — `startswith`, `endswith`,
/// `ltrimstr`, `rtrimstr`, `split`, `index`, `rindex`, `indices`,
/// `contains` — all of which share the same input contract: an object
/// with a quoted-string field whose content has no `\` escapes.
///
/// The helper handles only the structural type-guard and hands the
/// inner bytes (without quotes) to the closure. The closure dispatches
/// on builtin name and emits the appropriate output. If the closure
/// encounters an unknown builtin name (defensive fallback) it returns
/// [`RawApplyOutcome::Bail`] to delegate to the generic path.
///
/// Bail discipline:
/// * Field absent — [`RawApplyOutcome::Bail`].
/// * Value isn't a quoted string, or contains any backslash escape —
///   [`RawApplyOutcome::Bail`] (the generic path decodes escapes and
///   raises type errors).
/// * Non-object input — [`RawApplyOutcome::Bail`].
/// * Closure returns [`RawApplyOutcome::Bail`] — the helper propagates.
pub fn apply_field_str_builtin_raw<E>(
    raw: &[u8],
    field: &str,
    on_string: E,
) -> RawApplyOutcome
where
    E: FnOnce(&[u8]) -> RawApplyOutcome,
{
    let (vs, ve) = match json_object_get_field_raw(raw, 0, field) {
        Some(r) => r,
        None => return RawApplyOutcome::Bail,
    };
    let val = &raw[vs..ve];
    if val.len() < 2 || val[0] != b'"' || val[val.len() - 1] != b'"'
        || val[1..val.len() - 1].contains(&b'\\')
    {
        return RawApplyOutcome::Bail;
    }
    on_string(&val[1..val.len() - 1])
}

/// Apply the `.field // fallback` raw-byte fast path on a single JSON
/// record.
///
/// `//` (jq's alternative operator) **must not swallow type errors**
/// (#198): when `.field` itself would raise (i.e. the input isn't an
/// object and isn't the literal `null`), the alternative fallback does
/// not apply — jq still raises. The helper enforces this with an outer
/// type-guard:
///
/// * Object input — extract the field. If the value is `null` / `false`
///   (jq's "false-y" gate for `//`) or the field is absent, invoke
///   `emit` with `fallback_bytes`. Otherwise `emit` with the field's
///   raw bytes.
/// * Literal `null` input — there's no field to extract, so emit
///   `fallback_bytes`.
/// * Anything else — return [`RawApplyOutcome::Bail`] so the generic
///   path raises `Cannot index <type> with "<field>"`.
pub fn apply_field_alternative_raw<E>(
    raw: &[u8],
    field: &str,
    fallback_bytes: &[u8],
    mut emit: E,
) -> RawApplyOutcome
where
    E: FnMut(&[u8]),
{
    // Outer guard: only object or the literal `null` proceed; non-
    // object non-null inputs bail so jq's type error surfaces (#198).
    if raw.is_empty() || (raw.first() != Some(&b'{') && raw != b"null") {
        return RawApplyOutcome::Bail;
    }
    match json_object_get_field_raw(raw, 0, field) {
        Some((vs, ve)) => {
            let val = &raw[vs..ve];
            if val == b"null" || val == b"false" {
                emit(fallback_bytes);
            } else {
                emit(val);
            }
        }
        None => emit(fallback_bytes),
    }
    RawApplyOutcome::Emit
}

/// Apply the `.field1 // .field2` raw-byte fast path on a single JSON
/// record.
///
/// Same outer type-guard as [`apply_field_alternative_raw`]: non-object,
/// non-null inputs bail so the generic path raises `Cannot index ...`
/// (#198). When the input is an object (or the literal `null`):
///
/// * If `primary_field` is present and its value is **not** `null` or
///   `false`, `emit` is invoked with the primary's raw bytes.
/// * Otherwise `emit` is invoked with `fallback_field`'s raw bytes if
///   present, else with `b"null"` (jq's `null` for missing fallback).
pub fn apply_field_field_alternative_raw<E>(
    raw: &[u8],
    primary_field: &str,
    fallback_field: &str,
    mut emit: E,
) -> RawApplyOutcome
where
    E: FnMut(&[u8]),
{
    if raw.is_empty() || (raw.first() != Some(&b'{') && raw != b"null") {
        return RawApplyOutcome::Bail;
    }
    let primary_emitted = match json_object_get_field_raw(raw, 0, primary_field) {
        Some((vs, ve)) => {
            let pval = &raw[vs..ve];
            if pval != b"null" && pval != b"false" {
                emit(pval);
                true
            } else {
                false
            }
        }
        None => false,
    };
    if !primary_emitted {
        match json_object_get_field_raw(raw, 0, fallback_field) {
            Some((vs, ve)) => emit(&raw[vs..ve]),
            None => emit(b"null"),
        }
    }
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
