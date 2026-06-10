//! Typed control-flow signals threaded through the anyhow error channel.
//!
//! eval propagates non-error control flow (halt, label/break, `error(value)`
//! payloads, path-expression results) as `anyhow::Error` so it rides the
//! ordinary `?`/`Result` plumbing. Historically each signal was encoded in
//! the Display string and recovered by prefix parsing — fragile, and lossy
//! for non-finite numbers (#844). The types in this module make recovery a
//! typed downcast instead (#1034).
//!
//! The Display strings keep the legacy sentinel forms byte-for-byte: they
//! are still the wire format across the JIT FFI boundary (jit.rs serializes
//! errors with `format!("{}", e)` into its thread-local string channel), so
//! a signal that crosses JIT-compiled code arrives at the consumer as plain
//! text. Consumers therefore downcast first and keep a string fallback only
//! where JIT-crossed signals can land. Once the JIT channel itself carries a
//! typed payload, those fallbacks and the sentinel Display forms can go.

use std::fmt;

/// `halt` / `halt_error`: non-recoverable, propagates past `try`/`catch`
/// (#182); the CLI flushes buffered output and exits with `code`.
/// `halt_error`'s stderr payload is written at raise time, not carried here.
#[derive(Debug)]
pub struct HaltSignal {
    pub code: i32,
}

impl HaltSignal {
    pub fn raise(code: i32) -> anyhow::Error {
        HaltSignal { code }.into()
    }
}

impl fmt::Display for HaltSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Legacy sentinel form — still the JIT-boundary wire format.
        write!(f, "__halt__:{}", self.code)
    }
}

impl std::error::Error for HaltSignal {}

/// Typed error for label/break to avoid string formatting/parsing overhead.
#[derive(Debug)]
pub(crate) struct BreakError(pub(crate) u64);
impl fmt::Display for BreakError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "__break__:{}:", self.0)
    }
}
impl std::error::Error for BreakError {}

thread_local! {
    /// Lossless payload slot for the most recently raised `error(value)`
    /// (#844). `Value` holds `Rc`s, so it cannot live inside an
    /// `anyhow::Error` (which requires `Send + Sync`); instead `error`
    /// stashes the value here and bails with the `Send + Sync` [`ErrorValue`]
    /// marker. The matching `catch` takes the value straight back. This
    /// mirrors jit.rs's `JIT_ERROR_VALUE` channel. The slot is safe as a
    /// single cell because error propagation is synchronous LIFO unwinding:
    /// only one `error(value)` is ever in flight, and the catching `try`
    /// takes it immediately. A stale value left by an uncaught error is
    /// overwritten by the next `error` and is never read by the string path
    /// (which only fires when the `ErrorValue` downcast fails).
    static ERROR_PAYLOAD: std::cell::RefCell<Option<crate::value::Value>> =
        const { std::cell::RefCell::new(None) };
}

/// Typed marker error carrying an `error(value)` payload losslessly (#844).
///
/// The string channel (`__jqerror__:<JSON>`) serializes the payload with
/// `value_to_json_precise`, which is lossy for non-finite numbers anywhere
/// in the value: `nan` becomes `null` and `±infinite` saturates to the
/// nearest finite f64. Round-tripping that JSON back in a `catch` branch
/// therefore corrupts the caught value. The exact `Value` rides in
/// [`ERROR_PAYLOAD`] instead; `Display` still emits the `__jqerror__:<JSON>`
/// form so uncaught errors print exactly as before.
#[derive(Debug)]
pub(crate) struct ErrorValue {
    display: String,
}
impl ErrorValue {
    /// Stash `value` in the thread-local payload slot and build the marker
    /// error whose `Display` reproduces the legacy `__jqerror__:<JSON>` text.
    pub(crate) fn raise(value: crate::value::Value) -> anyhow::Error {
        let display = format!(
            "__jqerror__:{}",
            crate::value::value_to_json_precise(&value)
        );
        ERROR_PAYLOAD.with(|slot| *slot.borrow_mut() = Some(value));
        ErrorValue { display }.into()
    }
}
impl fmt::Display for ErrorValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display)
    }
}
impl std::error::Error for ErrorValue {}

/// Take the payload stashed by the most recent `error(value)`. Falls back to
/// re-parsing the marker's `display` JSON if the slot is somehow empty.
pub(crate) fn take_error_payload(ev: &ErrorValue) -> crate::value::Value {
    if let Some(v) = ERROR_PAYLOAD.with(|slot| slot.borrow_mut().take()) {
        return v;
    }
    match ev.display.strip_prefix("__jqerror__:") {
        Some(json) => crate::value::json_to_value(json)
            .unwrap_or_else(|_| crate::value::Value::from_str(&ev.display)),
        None => crate::value::Value::from_str(&ev.display),
    }
}

thread_local! {
    /// Payload slot for the most recent [`PathResultSignal`]. `Value` holds
    /// `Rc`s, so it cannot live inside an `anyhow::Error` (`Send + Sync`);
    /// the value rides here and the marker error carries a serial number to
    /// detect when a nested raise overwrote the slot before the outer signal
    /// was consumed (the consumer then falls back to the display JSON, which
    /// is exactly what the legacy string channel always did).
    static PATH_RESULT: std::cell::RefCell<(u64, Option<crate::value::Value>)> =
        const { std::cell::RefCell::new((0, None)) };
}

/// Path-mode value smuggling: a value-producing (non-path) expression under
/// `path()` / assignment tracking surfaces its *result* through the error
/// channel, and the enclosing path machinery either swallows it, rewrites it
/// into the user-facing "Invalid path expression …" error, or re-raises it.
/// Strictly eval-internal: every consumer converts it before the value
/// crosses a public entry point (`eval_*_standalone`), so unlike
/// [`HaltSignal`] it never rides the JIT string channel in normal operation.
#[derive(Debug)]
pub struct PathResultSignal {
    serial: u64,
    display: String,
}

impl PathResultSignal {
    /// Stash `value` in the payload slot and build the marker error. The
    /// Display keeps the legacy `__pathexpr_result__:<JSON>` sentinel form
    /// (serialized with the same lossy `value_to_json` as before) so any
    /// uncaught signal prints exactly as it used to.
    pub(crate) fn raise(value: &crate::value::Value) -> anyhow::Error {
        let display = format!(
            "__pathexpr_result__:{}",
            crate::value::value_to_json(value)
        );
        let serial = PATH_RESULT.with(|s| {
            let mut s = s.borrow_mut();
            s.0 += 1;
            s.1 = Some(value.clone());
            s.0
        });
        PathResultSignal { serial, display }.into()
    }
}

impl fmt::Display for PathResultSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display)
    }
}

impl std::error::Error for PathResultSignal {}

/// Whether `e` is a path-result signal (without consuming its payload).
pub(crate) fn is_path_result(e: &anyhow::Error) -> bool {
    e.downcast_ref::<PathResultSignal>().is_some()
}

/// Recover the smuggled value from a path-result signal; `None` if `e` is a
/// different error. Takes the payload slot when the serial still matches;
/// a slot overwritten by a nested raise falls back to re-parsing the display
/// JSON (lossy for non-finite numbers — the legacy channel's behavior).
pub(crate) fn take_path_result(e: &anyhow::Error) -> Option<crate::value::Value> {
    let sig = e.downcast_ref::<PathResultSignal>()?;
    let slot = PATH_RESULT.with(|s| {
        let mut s = s.borrow_mut();
        if s.0 == sig.serial { s.1.take() } else { None }
    });
    if let Some(v) = slot {
        return Some(v);
    }
    let json = sig
        .display
        .strip_prefix("__pathexpr_result__:")
        .unwrap_or(&sig.display);
    Some(
        crate::value::json_to_value(json)
            .unwrap_or_else(|_| crate::value::Value::from_str(json)),
    )
}
