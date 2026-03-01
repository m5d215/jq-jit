//! Value type for JIT-compiled jq filters.
//!
//! This module defines the `Value` enum used by both Rust runtime helpers and
//! JIT-generated native code.  The enum uses `#[repr(C, u64)]` to guarantee a
//! predictable memory layout:
//!
//! - **Bytes 0..7**: tag (u64 discriminant)
//! - **Bytes 8..15**: payload (f64 / bool / Rc pointer)
//!
//! Total size: 16 bytes, alignment: 8 bytes.
//!
//! # Discriminant values
//!
//! | Tag | Variant |
//! |-----|---------|
//! | 0   | Null    |
//! | 1   | Bool    |
//! | 2   | Num     |
//! | 3   | Str     |
//! | 4   | Arr     |
//! | 5   | Obj     |

use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::fmt;
use std::rc::Rc;

use anyhow::{Context, Result, bail};

use crate::jq_ffi::{self, Jv, JvKind};

/// Tag discriminant constants, matching the `#[repr(C, u64)]` layout.
pub const TAG_NULL: u64 = 0;
pub const TAG_BOOL: u64 = 1;
pub const TAG_NUM: u64 = 2;
#[allow(dead_code)]
pub const TAG_STR: u64 = 3;
#[allow(dead_code)]
pub const TAG_ARR: u64 = 4;
#[allow(dead_code)]
pub const TAG_OBJ: u64 = 5;
#[allow(dead_code)]
pub const TAG_ERROR: u64 = 6;

/// A jq value.
///
/// 16-byte tagged union with `#[repr(C, u64)]` for JIT code compatibility.
/// Heap-allocated variants (Str, Arr, Obj) use `Rc` for reference counting.
#[repr(C, u64)]
#[allow(dead_code)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(Rc<String>),
    Arr(Rc<Vec<Value>>),
    Obj(Rc<BTreeMap<String, Value>>),
    /// An error value produced by runtime operations (e.g., indexing a non-object).
    /// Used for try-catch: the error message is available to the catch expression.
    Error(Rc<String>),
}

// Compile-time size and alignment assertions.
const _: () = assert!(std::mem::size_of::<Value>() == 16);
const _: () = assert!(std::mem::align_of::<Value>() == 8);

// ---------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------

impl Clone for Value {
    fn clone(&self) -> Self {
        match self {
            Value::Null => Value::Null,
            Value::Bool(b) => Value::Bool(*b),
            Value::Num(n) => Value::Num(*n),
            Value::Str(s) => Value::Str(Rc::clone(s)),
            Value::Arr(a) => Value::Arr(Rc::clone(a)),
            Value::Obj(o) => Value::Obj(Rc::clone(o)),
            Value::Error(e) => Value::Error(Rc::clone(e)),
        }
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Num(n) => {
                if *n == n.trunc() && n.abs() < 1e15 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{}", n)
                }
            }
            Value::Str(s) => write!(f, "\"{}\"", s),
            Value::Arr(a) => {
                write!(f, "[")?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Value::Obj(o) => {
                write!(f, "{{")?;
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "\"{}\":{}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Error(e) => write!(f, "<error: {}>", e),
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "Null"),
            Value::Bool(b) => write!(f, "Bool({})", b),
            Value::Num(n) => write!(f, "Num({})", n),
            Value::Str(s) => write!(f, "Str({:?})", s.as_str()),
            Value::Arr(a) => write!(f, "Arr({:?})", a.as_ref()),
            Value::Obj(o) => write!(f, "Obj({:?})", o.as_ref()),
            Value::Error(e) => write!(f, "Error({:?})", e.as_str()),
        }
    }
}

// ---------------------------------------------------------------------------
// PartialEq
// ---------------------------------------------------------------------------

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Num(a), Value::Num(b)) => a == b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::Arr(a), Value::Arr(b)) => a == b,
            (Value::Obj(a), Value::Obj(b)) => a == b,
            (Value::Error(a), Value::Error(b)) => a == b,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Constructors / accessors
// ---------------------------------------------------------------------------

#[allow(dead_code)]
impl Value {
    /// Create a `Value::Num` from an f64.
    pub fn from_f64(n: f64) -> Self {
        Value::Num(n)
    }

    /// Extract the f64 payload if this is a `Num`, otherwise `None`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// Create a `Value::Str` from a Rust string.
    pub fn from_str(s: &str) -> Self {
        Value::Str(Rc::new(s.to_string()))
    }

    /// Create a `Value::Obj` from key-value pairs.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, Value)>) -> Self {
        Value::Obj(Rc::new(pairs.into_iter().collect()))
    }

    /// Returns the tag discriminant value for this value.
    pub fn tag(&self) -> u64 {
        // SAFETY: #[repr(C, u64)] guarantees the first 8 bytes are the discriminant.
        unsafe { *(self as *const Value as *const u64) }
    }
}

// ---------------------------------------------------------------------------
// JSON ↔ Value conversion (using libjq's jv_parse)
// ---------------------------------------------------------------------------

/// Parse a JSON string into a `Value` using libjq's `jv_parse`.
///
/// Supports: numbers, strings, booleans, null, arrays, and objects.
/// Does NOT use serde_json — relies solely on libjq for parsing.
pub fn json_to_value(json: &str) -> Result<Value> {
    let c_json = CString::new(json).context("JSON string contains null byte")?;
    unsafe {
        let jv = jq_ffi::jv_parse(c_json.as_ptr());
        let kind = jq_ffi::jv_get_kind(jv);
        if kind == JvKind::Invalid {
            // Try to extract jq's error message
            let msg_jv = jq_ffi::jv_invalid_get_msg(jq_ffi::jv_copy(jv));
            let msg_kind = jq_ffi::jv_get_kind(msg_jv);
            let err_msg = if msg_kind == JvKind::String {
                let c_str = jq_ffi::jv_string_value(jq_ffi::jv_copy(msg_jv));
                let msg = std::ffi::CStr::from_ptr(c_str).to_string_lossy().into_owned();
                jq_ffi::jv_free(msg_jv);
                msg
            } else {
                jq_ffi::jv_free(msg_jv);
                format!("jv_parse({:?}) returned invalid", json)
            };
            jq_ffi::jv_free(jv);
            bail!("{}", err_msg);
        }
        let value = jv_to_value(jv)?;
        Ok(value)
    }
}

/// Convert a jv into a `Value`.  Consumes the jv (calls jv_free).
pub unsafe fn jv_to_value(jv: Jv) -> Result<Value> {
    unsafe {
        let kind = jq_ffi::jv_get_kind(jv);
        match kind {
            JvKind::Null => {
                jq_ffi::jv_free(jv);
                Ok(Value::Null)
            }
            JvKind::True => {
                jq_ffi::jv_free(jv);
                Ok(Value::Bool(true))
            }
            JvKind::False => {
                jq_ffi::jv_free(jv);
                Ok(Value::Bool(false))
            }
            JvKind::Number => {
                let n = jq_ffi::jv_number_value(jv);
                jq_ffi::jv_free(jv);
                Ok(Value::Num(n))
            }
            JvKind::String => {
                let cstr = jq_ffi::jv_string_value(jq_ffi::jv_copy(jv));
                let len = jq_ffi::jv_string_length_bytes(jq_ffi::jv_copy(jv)) as usize;
                // Use raw pointer + length to handle embedded null bytes
                let bytes = std::slice::from_raw_parts(cstr as *const u8, len);
                let s = String::from_utf8_lossy(bytes).into_owned();
                jq_ffi::jv_free(jv);
                Ok(Value::Str(Rc::new(s)))
            }
            JvKind::Array => {
                let len = jq_ffi::jv_array_length(jq_ffi::jv_copy(jv));
                let mut items = Vec::with_capacity(len as usize);
                for i in 0..len {
                    let elem = jq_ffi::jv_array_get(jq_ffi::jv_copy(jv), i);
                    items.push(jv_to_value(elem)?);
                }
                jq_ffi::jv_free(jv);
                Ok(Value::Arr(Rc::new(items)))
            }
            JvKind::Object => {
                let mut map = BTreeMap::new();
                let mut iter = jq_ffi::jv_object_iter(jv);
                while jq_ffi::jv_object_iter_valid(jv, iter) != 0 {
                    let key_jv = jq_ffi::jv_object_iter_key(jv, iter);
                    let val_jv = jq_ffi::jv_object_iter_value(jv, iter);

                    // key_jv is a jv string
                    let key_cstr = jq_ffi::jv_string_value(key_jv);
                    let key = CStr::from_ptr(key_cstr).to_string_lossy().into_owned();
                    jq_ffi::jv_free(key_jv);

                    let val = jv_to_value(val_jv)?;
                    map.insert(key, val);

                    iter = jq_ffi::jv_object_iter_next(jv, iter);
                }
                jq_ffi::jv_free(jv);
                Ok(Value::Obj(Rc::new(map)))
            }
            JvKind::Invalid => {
                jq_ffi::jv_free(jv);
                bail!("jv_to_value: invalid jv");
            }
        }
    }
}

/// Format an f64 the way jq does.
///
/// jq uses a format similar to C's `%.17g`: integers are printed without decimal
/// point, and scientific notation is used for very large/small numbers.
pub fn format_jq_number(n: f64) -> String {
    if n.is_infinite() {
        if n.is_sign_positive() {
            "1.7976931348623157e+308".to_string()
        } else {
            "-1.7976931348623157e+308".to_string()
        }
    } else if n.is_nan() {
        "null".to_string()
    } else if n == 0.0 {
        if n.is_sign_negative() {
            "-0".to_string()
        } else {
            "0".to_string()
        }
    } else {
        // Use a format similar to C's %.17g
        // This matches jq's number formatting for computed results
        format_g17(n)
    }
}

/// Format f64 like jq does.
///
/// Uses integer format for exact integers that round-trip through i64,
/// otherwise uses Rust's default f64 Display (which is similar to %g).
fn format_g17(n: f64) -> String {
    // Use C-like %.17g formatting:
    // - For integers that are short enough, use integer format
    // - For large integers, use scientific notation if shorter
    // - For non-integers, use shortest representation
    if n == n.trunc() && n.abs() < 1e18 {
        let i = n as i64;
        if i as f64 == n {
            let int_str = format!("{}", i);
            // Check if scientific notation would be shorter (like %.17g does)
            // %.17g switches to scientific notation when exponent >= 17
            let sci_str = format!("{:e}", n);
            // Normalize scientific notation: 1e17 -> 1e+17, matching jq format
            let sci_str = if !sci_str.contains('+') && !sci_str.contains('-') {
                sci_str.replacen("e", "e+", 1)
            } else {
                sci_str
            };
            if sci_str.len() < int_str.len() {
                return sci_str;
            }
            return int_str;
        }
    }
    // Default Rust f64 formatting (similar to shortest-representation)
    format!("{}", n)
}

/// Convert a `Value` to a JSON string.
///
/// Produces compact JSON output (no extra whitespace).
pub fn value_to_json(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Value::Num(n) => {
            format_jq_number(*n)
        }
        Value::Str(s) => {
            // JSON string with escaped characters
            let mut out = String::with_capacity(s.len() + 2);
            out.push('"');
            for ch in s.chars() {
                match ch {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if c < '\x20' => {
                        out.push_str(&format!("\\u{:04x}", c as u32));
                    }
                    c => out.push(c),
                }
            }
            out.push('"');
            out
        }
        Value::Arr(a) => {
            let mut out = String::from("[");
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&value_to_json(item));
            }
            out.push(']');
            out
        }
        Value::Obj(o) => {
            let mut out = String::from("{");
            for (i, (k, v)) in o.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                // Key is always a string
                out.push('"');
                out.push_str(k);
                out.push('"');
                out.push(':');
                out.push_str(&value_to_json(v));
            }
            out.push('}');
            out
        }
        Value::Error(e) => {
            // Errors shouldn't normally reach JSON output, but format as string for debugging
            let mut out = String::with_capacity(e.len() + 2);
            out.push('"');
            out.push_str(e);
            out.push('"');
            out
        }
    }
}
