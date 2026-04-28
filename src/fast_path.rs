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

use crate::ir::BinOp;
use crate::runtime::jq_mod_f64;
use crate::value::{
    KeyStr, Value, ObjInner, json_object_get_field_raw, json_object_get_fields_raw_buf,
    json_object_get_nested_field_raw, json_object_get_num, json_object_get_two_nums,
    json_object_has_all_keys, json_object_has_any_key, json_object_has_key,
    json_object_update_field_length, json_object_update_field_tostring,
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
