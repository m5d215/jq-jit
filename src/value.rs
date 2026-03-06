//! Value type for JIT-compiled jq filters.
//!
//! Key improvement over jq-jit: uses IndexMap for objects (preserves insertion order).

use std::ffi::{CStr, CString};
use std::fmt;
use std::rc::Rc;

use anyhow::{Context, Result, bail};
use indexmap::IndexMap;

use crate::jq_ffi::{self, Jv, JvKind};

/// Object map type — IndexMap with ahash for faster lookups.
pub type ObjMap = IndexMap<String, Value, ahash::RandomState>;

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
    Obj(Rc<ObjMap>),
    Error(Rc<String>),
}

impl Clone for Value {
    fn clone(&self) -> Self {
        match self {
            Value::Null => Value::Null,
            Value::False => Value::False,
            Value::True => Value::True,
            Value::Num(n, repr) => Value::Num(*n, repr.clone()),
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

    pub fn as_obj(&self) -> Option<&Rc<ObjMap>> {
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
            Value::Num(n, repr) => {
                if *n >= 0.0 { Ok(Value::Num(*n, repr.clone())) }
                else { Ok(Value::Num(n.abs(), None)) }
            }
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
    let bytes = json.as_bytes();
    let mut pos = 0;
    // Skip BOM
    if bytes.len() >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF { pos = 3; }
    pos = skip_ws(bytes, pos);
    if pos >= bytes.len() {
        bail!("Invalid numeric literal at line 1, column 1 (while parsing 'jq: ')");
    }
    let (val, end) = parse_json_value(bytes, pos, 0)?;
    pos = skip_ws(bytes, end);
    if pos < bytes.len() {
        bail!("Invalid numeric literal at line 1, column {} (while parsing 'jq: ')", pos + 1);
    }
    Ok(val)
}

fn skip_ws(b: &[u8], mut pos: usize) -> usize {
    while pos < b.len() && matches!(b[pos], b' ' | b'\t' | b'\n' | b'\r') { pos += 1; }
    pos
}

const MAX_JSON_DEPTH: usize = 10000;

fn parse_json_value(b: &[u8], pos: usize, depth: usize) -> Result<(Value, usize)> {
    if pos >= b.len() { bail!("Unexpected end of JSON input"); }
    if depth > MAX_JSON_DEPTH { bail!("Exceeds depth limit for parsing"); }
    match b[pos] {
        b'n' => {
            if b.get(pos..pos+4) == Some(b"null") { Ok((Value::Null, pos + 4)) }
            else if b.get(pos..pos+3) == Some(b"nan") { Ok((Value::Num(f64::NAN, None), pos + 3)) }
            else { bail!("Invalid JSON at position {}", pos) }
        }
        b't' => {
            if b.get(pos..pos+4) == Some(b"true") { Ok((Value::True, pos + 4)) }
            else { bail!("Invalid JSON at position {}", pos) }
        }
        b'f' => {
            if b.get(pos..pos+5) == Some(b"false") { Ok((Value::False, pos + 5)) }
            else { bail!("Invalid JSON at position {}", pos) }
        }
        b'N' => {
            if b.get(pos..pos+3) == Some(b"NaN") { Ok((Value::Num(f64::NAN, None), pos + 3)) }
            else { bail!("Invalid JSON at position {}", pos) }
        }
        b'I' => {
            if b.get(pos..pos+8) == Some(b"Infinity") { Ok((Value::Num(f64::INFINITY, None), pos + 8)) }
            else { bail!("Invalid JSON at position {}", pos) }
        }
        b'"' => parse_json_string(b, pos),
        b'[' => parse_json_array(b, pos, depth),
        b'{' => parse_json_object(b, pos, depth),
        b'-' => {
            // Could be -Infinity, -NaN, or negative number
            if b.get(pos..pos+9) == Some(b"-Infinity") { Ok((Value::Num(f64::NEG_INFINITY, None), pos + 9)) }
            else if b.get(pos..pos+4) == Some(b"-NaN") { Ok((Value::Num(f64::NAN, None), pos + 4)) }
            else { parse_json_number(b, pos) }
        }
        b'0'..=b'9' => parse_json_number(b, pos),
        c => bail!("Unexpected character '{}' at position {}", c as char, pos),
    }
}

fn parse_json_string(b: &[u8], pos: usize) -> Result<(Value, usize)> {
    debug_assert_eq!(b[pos], b'"');
    let mut i = pos + 1;
    // Fast path: scan for end of simple string (no escapes, all ASCII)
    let start = i;
    let mut has_escape = false;
    while i < b.len() {
        match b[i] {
            b'"' if !has_escape => {
                let s = std::str::from_utf8(&b[start..i]).unwrap_or("").to_string();
                return Ok((Value::Str(Rc::new(s)), i + 1));
            }
            b'\\' => { has_escape = true; break; }
            b'"' => break,
            _ => i += 1,
        }
    }
    // Slow path: handle escapes and multi-byte UTF-8
    i = pos + 1;
    let mut buf = Vec::new();
    while i < b.len() {
        match b[i] {
            b'"' => {
                let s = String::from_utf8_lossy(&buf).into_owned();
                return Ok((Value::Str(Rc::new(s)), i + 1));
            }
            b'\\' => {
                i += 1;
                if i >= b.len() { bail!("Unterminated string escape"); }
                match b[i] {
                    b'"' => buf.push(b'"'),
                    b'\\' => buf.push(b'\\'),
                    b'/' => buf.push(b'/'),
                    b'b' => buf.push(0x08),
                    b'f' => buf.push(0x0C),
                    b'n' => buf.push(b'\n'),
                    b'r' => buf.push(b'\r'),
                    b't' => buf.push(b'\t'),
                    b'u' => {
                        if i + 4 >= b.len() { bail!("Invalid unicode escape"); }
                        let hex = std::str::from_utf8(&b[i+1..i+5]).unwrap_or("0000");
                        let cp = u16::from_str_radix(hex, 16).unwrap_or(0);
                        i += 4;
                        if (0xD800..=0xDBFF).contains(&cp) {
                            if i + 6 <= b.len() && b[i+1] == b'\\' && b[i+2] == b'u' {
                                let hex2 = std::str::from_utf8(&b[i+3..i+7]).unwrap_or("0000");
                                let cp2 = u16::from_str_radix(hex2, 16).unwrap_or(0);
                                if (0xDC00..=0xDFFF).contains(&cp2) {
                                    let full = 0x10000 + ((cp as u32 - 0xD800) << 10) + (cp2 as u32 - 0xDC00);
                                    if let Some(c) = char::from_u32(full) {
                                        let mut tmp = [0u8; 4];
                                        buf.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
                                    }
                                    i += 6;
                                } else {
                                    buf.extend_from_slice("\u{FFFD}".as_bytes());
                                }
                            } else {
                                buf.extend_from_slice("\u{FFFD}".as_bytes());
                            }
                        } else if let Some(c) = char::from_u32(cp as u32) {
                            let mut tmp = [0u8; 4];
                            buf.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
                        }
                    }
                    _ => { buf.push(b'\\'); buf.push(b[i]); }
                }
                i += 1;
            }
            c => { buf.push(c); i += 1; }
        }
    }
    bail!("Unterminated string")
}

fn parse_json_number(b: &[u8], pos: usize) -> Result<(Value, usize)> {
    let mut i = pos;
    let is_neg = i < b.len() && b[i] == b'-';
    if is_neg { i += 1; }
    let digits_start = i;
    if i < b.len() && b[i] == b'0' { i += 1; }
    else { while i < b.len() && b[i].is_ascii_digit() { i += 1; } }
    let has_dot = i < b.len() && b[i] == b'.';
    if has_dot { i += 1; while i < b.len() && b[i].is_ascii_digit() { i += 1; } }
    let has_exp = i < b.len() && (b[i] == b'e' || b[i] == b'E');
    if has_exp {
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') { i += 1; }
        while i < b.len() && b[i].is_ascii_digit() { i += 1; }
    }
    let num_str = std::str::from_utf8(&b[pos..i]).unwrap_or("0");
    let n: f64 = num_str.parse().unwrap_or(0.0);
    // Fast path: simple integers that round-trip exactly don't need repr
    let repr = if !has_dot && !has_exp && (i - digits_start) <= 15 {
        // Simple integer with <= 15 digits always round-trips via format_jq_number
        None
    } else {
        // Check if f64 round-trip matches the original text
        let f64_repr = format_jq_number(n);
        if f64_repr == num_str { None } else { Some(Rc::from(num_str)) }
    };
    Ok((Value::Num(n, repr), i))
}

fn parse_json_array(b: &[u8], pos: usize, depth: usize) -> Result<(Value, usize)> {
    debug_assert_eq!(b[pos], b'[');
    let mut i = skip_ws(b, pos + 1);
    let mut items = Vec::new();
    if i < b.len() && b[i] == b']' { return Ok((Value::Arr(Rc::new(items)), i + 1)); }
    loop {
        let (val, end) = parse_json_value(b, i, depth + 1)?;
        items.push(val);
        i = skip_ws(b, end);
        if i >= b.len() { bail!("Unterminated array"); }
        if b[i] == b']' { return Ok((Value::Arr(Rc::new(items)), i + 1)); }
        if b[i] != b',' { bail!("Expected ',' or ']' at position {}", i); }
        i = skip_ws(b, i + 1);
    }
}

fn parse_json_object(b: &[u8], pos: usize, depth: usize) -> Result<(Value, usize)> {
    debug_assert_eq!(b[pos], b'{');
    let mut i = skip_ws(b, pos + 1);
    let mut map = ObjMap::default();
    if i < b.len() && b[i] == b'}' { return Ok((Value::Obj(Rc::new(map)), i + 1)); }
    loop {
        if i >= b.len() || b[i] != b'"' { bail!("Expected string key at position {}", i); }
        let (key_val, end) = parse_json_string(b, i)?;
        let key = match key_val { Value::Str(s) => (*s).clone(), _ => unreachable!() };
        i = skip_ws(b, end);
        if i >= b.len() || b[i] != b':' { bail!("Expected ':' at position {}", i); }
        i = skip_ws(b, i + 1);
        let (val, end) = parse_json_value(b, i, depth + 1)?;
        map.insert(key, val);
        i = skip_ws(b, end);
        if i >= b.len() { bail!("Unterminated object"); }
        if b[i] == b'}' { return Ok((Value::Obj(Rc::new(map)), i + 1)); }
        if b[i] != b',' { bail!("Expected ',' or '}}' at position {}", i); }
        i = skip_ws(b, i + 1);
    }
}

/// Parse JSON using libjq (for fromjson — produces libjq-compatible error messages)
pub fn json_to_value_libjq(json: &str) -> Result<Value> {
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
                let n = jq_ffi::jv_number_value(jq_ffi::jv_copy(jv));
                // Get precise string representation via jv_dump_string
                let dump_jv = jq_ffi::jv_dump_string(jv, 0);
                let repr = if jq_ffi::jv_get_kind(dump_jv) == JvKind::String {
                    let cstr = jq_ffi::jv_string_value(jq_ffi::jv_copy(dump_jv));
                    let s = CStr::from_ptr(cstr).to_string_lossy().into_owned();
                    jq_ffi::jv_free(dump_jv);
                    // Check if the precise repr differs from f64 round-trip
                    let f64_repr = format_jq_number(n);
                    if s != f64_repr {
                        Some(Rc::from(s.as_str()))
                    } else {
                        None
                    }
                } else {
                    jq_ffi::jv_free(dump_jv);
                    None
                };
                Ok(Value::Num(n, repr))
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
                let mut map = ObjMap::default();
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

/// Convert Value to compact JSON string (always uses f64 for numbers).
pub fn value_to_json(v: &Value) -> String {
    value_to_json_depth(v, 0, false)
}

/// Convert Value to compact JSON string, using precise repr when available.
pub fn value_to_json_precise(v: &Value) -> String {
    value_to_json_depth(v, 0, true)
}

fn value_to_json_depth(v: &Value, depth: usize, precise: bool) -> String {
    if depth > MAX_JSON_DEPTH {
        return "\"<skipped: too deep>\"".to_string();
    }
    match v {
        Value::Null => "null".to_string(),
        Value::False => "false".to_string(),
        Value::True => "true".to_string(),
        Value::Num(n, repr) => {
            if precise {
                if let Some(r) = repr {
                    return r.to_string();
                }
            }
            format_jq_number(*n)
        }
        Value::Str(s) => json_encode_string(s),
        Value::Arr(a) => {
            let mut out = String::from("[");
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&value_to_json_depth(item, depth + 1, precise));
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
                out.push_str(&value_to_json_depth(v, depth + 1, precise));
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
        _ => value_to_json_precise(v),
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

// ============================================================================
// Streaming JSON output — writes directly to io::Write, avoiding intermediate Strings
// ============================================================================

use std::io;

fn write_json_string(w: &mut dyn io::Write, s: &str) -> io::Result<()> {
    w.write_all(b"\"")?;
    for ch in s.chars() {
        match ch {
            '"' => w.write_all(b"\\\"")?,
            '\\' => w.write_all(b"\\\\")?,
            '\n' => w.write_all(b"\\n")?,
            '\r' => w.write_all(b"\\r")?,
            '\t' => w.write_all(b"\\t")?,
            '\u{08}' => w.write_all(b"\\b")?,
            '\u{0c}' => w.write_all(b"\\f")?,
            c if (c as u32) < 0x20 => {
                write!(w, "\\u{:04x}", c as u32)?;
            }
            c => {
                let mut buf = [0u8; 4];
                w.write_all(c.encode_utf8(&mut buf).as_bytes())?;
            }
        }
    }
    w.write_all(b"\"")?;
    Ok(())
}

fn write_jq_number(w: &mut dyn io::Write, n: f64) -> io::Result<()> {
    // Reuse format_jq_number to guarantee consistent output
    w.write_all(format_jq_number(n).as_bytes())
}

/// Write a Value as compact JSON directly to an io::Write, using precise repr when available.
pub fn write_value_compact(w: &mut dyn io::Write, v: &Value) -> io::Result<()> {
    match v {
        Value::Null => w.write_all(b"null"),
        Value::False => w.write_all(b"false"),
        Value::True => w.write_all(b"true"),
        Value::Num(n, repr) => {
            if let Some(r) = repr {
                w.write_all(r.as_bytes())
            } else {
                write_jq_number(w, *n)
            }
        }
        Value::Str(s) => write_json_string(w, s),
        Value::Arr(a) => {
            w.write_all(b"[")?;
            for (i, item) in a.iter().enumerate() {
                if i > 0 { w.write_all(b",")?; }
                write_value_compact(w, item)?;
            }
            w.write_all(b"]")
        }
        Value::Obj(o) => {
            w.write_all(b"{")?;
            for (i, (k, val)) in o.iter().enumerate() {
                if i > 0 { w.write_all(b",")?; }
                write_json_string(w, k)?;
                w.write_all(b":")?;
                write_value_compact(w, val)?;
            }
            w.write_all(b"}")
        }
        Value::Error(e) => write_json_string(w, e),
    }
}
