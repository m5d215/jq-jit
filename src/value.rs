//! Value type for JIT-compiled jq filters.
//!
//! Key improvement over jq-jit: uses IndexMap for objects (preserves insertion order).

use std::ffi::{CStr, CString};
use std::fmt;
use std::rc::Rc;

use anyhow::{Context, Result, bail};
use indexmap::IndexMap;

use crate::jq_ffi::{self, Jv, JvKind};

/// Tag discriminant constants, matching `#[repr(C, u64)]` layout.
pub const TAG_NULL: u64 = 0;
pub const TAG_FALSE: u64 = 1;
pub const TAG_TRUE: u64 = 2;
pub const TAG_NUM: u64 = 3;
pub const TAG_STR: u64 = 4;
pub const TAG_ARR: u64 = 5;
pub const TAG_OBJ: u64 = 6;
pub const TAG_ERROR: u64 = 7;

/// A jq value.
pub enum Value {
    Null,
    False,
    True,
    /// Numeric value. Optional Rc<str> preserves original representation for precision.
    Num(f64, Option<Rc<str>>),
    Str(Rc<String>),
    Arr(Rc<Vec<Value>>),
    Obj(Rc<IndexMap<String, Value>>),
    Error(Rc<String>),
}

impl Clone for Value {
    fn clone(&self) -> Self {
        match self {
            Value::Null => Value::Null,
            Value::False => Value::False,
            Value::True => Value::True,
            Value::Num(n, _) => Value::Num(*n, None),
            Value::Str(s) => Value::Str(Rc::clone(s)),
            Value::Arr(a) => Value::Arr(Rc::clone(a)),
            Value::Obj(o) => Value::Obj(Rc::clone(o)),
            Value::Error(e) => Value::Error(Rc::clone(e)),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::False, Value::False) => true,
            (Value::True, Value::True) => true,
            (Value::Num(a, _), Value::Num(b, _)) => a == b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::Arr(a), Value::Arr(b)) => a == b,
            (Value::Obj(a), Value::Obj(b)) => a == b,
            _ => false,
        }
    }
}

impl Value {
    pub fn from_f64(n: f64) -> Self {
        Value::Num(n, None)
    }

    pub fn from_bool(b: bool) -> Self {
        if b { Value::True } else { Value::False }
    }

    pub fn from_str(s: &str) -> Self {
        Value::Str(Rc::new(s.to_string()))
    }

    pub fn from_string(s: String) -> Self {
        Value::Str(Rc::new(s))
    }

    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, Value)>) -> Self {
        Value::Obj(Rc::new(pairs.into_iter().collect()))
    }

    pub fn is_true(&self) -> bool {
        !matches!(self, Value::Null | Value::False)
    }

    pub fn is_truthy(&self) -> bool {
        self.is_true()
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Num(n, _) => Some(*n),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&Rc<Vec<Value>>> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&Rc<IndexMap<String, Value>>> {
        match self {
            Value::Obj(o) => Some(o),
            _ => None,
        }
    }

    pub fn tag(&self) -> u64 {
        unsafe { *(self as *const Value as *const u64) }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::False | Value::True => "boolean",
            Value::Num(_, _) => "number",
            Value::Str(_) => "string",
            Value::Arr(_) => "array",
            Value::Obj(_) => "object",
            Value::Error(_) => "error",
        }
    }

    pub fn length(&self) -> Result<Value> {
        match self {
            Value::Null => Ok(Value::Num(0.0, None)),
            Value::False => Ok(Value::Num(0.0, None)),
            Value::True => Ok(Value::Num(1.0, None)),
            Value::Num(n, _) => Ok(Value::Num(n.abs(), None)),
            Value::Str(s) => {
                // jq counts Unicode codepoints
                Ok(Value::Num(s.chars().count() as f64, None))
            }
            Value::Arr(a) => Ok(Value::Num(a.len() as f64, None)),
            Value::Obj(o) => Ok(Value::Num(o.len() as f64, None)),
            Value::Error(_) => bail!("error has no length"),
        }
    }
}

// ---------------------------------------------------------------------------
// Display / Debug
// ---------------------------------------------------------------------------

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", value_to_json(self))
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "Null"),
            Value::False => write!(f, "False"),
            Value::True => write!(f, "True"),
            Value::Num(n, _) => write!(f, "Num({})", n),
            Value::Str(s) => write!(f, "Str({:?})", s.as_str()),
            Value::Arr(a) => write!(f, "Arr({:?})", a.as_ref()),
            Value::Obj(o) => write!(f, "Obj({:?})", o.as_ref()),
            Value::Error(e) => write!(f, "Error({:?})", e.as_str()),
        }
    }
}

// ---------------------------------------------------------------------------
// JSON conversion
// ---------------------------------------------------------------------------

pub fn json_to_value(json: &str) -> Result<Value> {
    let c_json = CString::new(json).context("JSON string contains null byte")?;
    unsafe {
        let jv = jq_ffi::jv_parse(c_json.as_ptr());
        let kind = jq_ffi::jv_get_kind(jv);
        if kind == JvKind::Invalid {
            let msg_jv = jq_ffi::jv_invalid_get_msg(jq_ffi::jv_copy(jv));
            let msg_kind = jq_ffi::jv_get_kind(msg_jv);
            let err_msg = if msg_kind == JvKind::String {
                let c_str = jq_ffi::jv_string_value(jq_ffi::jv_copy(msg_jv));
                let msg = CStr::from_ptr(c_str).to_string_lossy().into_owned();
                jq_ffi::jv_free(msg_jv);
                msg
            } else {
                jq_ffi::jv_free(msg_jv);
                format!("jv_parse({:?}) returned invalid", json)
            };
            jq_ffi::jv_free(jv);
            bail!("{}", err_msg);
        }
        jv_to_value(jv)
    }
}

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
                Ok(Value::True)
            }
            JvKind::False => {
                jq_ffi::jv_free(jv);
                Ok(Value::False)
            }
            JvKind::Number => {
                let n = jq_ffi::jv_number_value(jv);
                jq_ffi::jv_free(jv);
                Ok(Value::Num(n, None))
            }
            JvKind::String => {
                let cstr = jq_ffi::jv_string_value(jq_ffi::jv_copy(jv));
                let len = jq_ffi::jv_string_length_bytes(jq_ffi::jv_copy(jv)) as usize;
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
                let mut map = IndexMap::new();
                let mut iter = jq_ffi::jv_object_iter(jv);
                while jq_ffi::jv_object_iter_valid(jv, iter) != 0 {
                    let key_jv = jq_ffi::jv_object_iter_key(jv, iter);
                    let val_jv = jq_ffi::jv_object_iter_value(jv, iter);
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

/// Format f64 the way jq does (%.17g equivalent).
pub fn format_jq_number(n: f64) -> String {
    if n.is_nan() {
        return "null".to_string();
    }
    if n.is_infinite() {
        return if n.is_sign_positive() {
            "1.7976931348623157e+308".to_string()
        } else {
            "-1.7976931348623157e+308".to_string()
        };
    }
    if n == 0.0 {
        return if n.is_sign_negative() {
            "-0".to_string()
        } else {
            "0".to_string()
        };
    }

    // Use C-style %.17g formatting
    // Integer values: no decimal point
    if n == n.trunc() && n.abs() < 1e18 {
        let i = n as i64;
        if i as f64 == n {
            let int_str = format!("{}", i);
            // %.17g uses scientific notation when significant digits > 17
            let digit_count = int_str.chars().filter(|c| c.is_ascii_digit()).count();
            if digit_count > 17 {
                let sci = format!("{:e}", n);
                let sci = if !sci.contains('+') && !sci.contains('-') {
                    sci.replacen("e", "e+", 1)
                } else {
                    sci
                };
                return sci;
            }
            return int_str;
        }
    }

    format!("{}", n)
}

/// Convert Value to compact JSON string.
pub fn value_to_json(v: &Value) -> String {
    value_to_json_depth(v, 0)
}

const MAX_JSON_DEPTH: usize = 10000;

fn value_to_json_depth(v: &Value, depth: usize) -> String {
    if depth > MAX_JSON_DEPTH {
        return "\"<skipped: too deep>\"".to_string();
    }
    match v {
        Value::Null => "null".to_string(),
        Value::False => "false".to_string(),
        Value::True => "true".to_string(),
        Value::Num(n, _) => format_jq_number(*n),
        Value::Str(s) => json_encode_string(s),
        Value::Arr(a) => {
            let mut out = String::from("[");
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&value_to_json_depth(item, depth + 1));
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
                out.push_str(&json_encode_string(k));
                out.push(':');
                out.push_str(&value_to_json_depth(v, depth + 1));
            }
            out.push('}');
            out
        }
        Value::Error(e) => json_encode_string(e),
    }
}

/// Convert Value to pretty-printed JSON string.
pub fn value_to_json_pretty(v: &Value, indent: usize) -> String {
    match v {
        Value::Arr(a) if a.is_empty() => "[]".to_string(),
        Value::Obj(o) if o.is_empty() => "{}".to_string(),
        Value::Arr(a) => {
            let mut out = String::from("[\n");
            let inner = " ".repeat(indent + 2);
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(",\n");
                }
                out.push_str(&inner);
                out.push_str(&value_to_json_pretty(item, indent + 2));
            }
            out.push('\n');
            out.push_str(&" ".repeat(indent));
            out.push(']');
            out
        }
        Value::Obj(o) => {
            let mut out = String::from("{\n");
            let inner = " ".repeat(indent + 2);
            for (i, (k, v)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(",\n");
                }
                out.push_str(&inner);
                out.push_str(&json_encode_string(k));
                out.push_str(": ");
                out.push_str(&value_to_json_pretty(v, indent + 2));
            }
            out.push('\n');
            out.push_str(&" ".repeat(indent));
            out.push('}');
            out
        }
        _ => value_to_json(v),
    }
}

fn json_encode_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
