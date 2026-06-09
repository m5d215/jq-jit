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
