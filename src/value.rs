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

/// Parse a JSON string, returning the raw String and end position.
/// Used by both parse_json_string (wraps in Value) and parse_json_object (uses String directly as key).
fn parse_json_string_raw(b: &[u8], pos: usize) -> Result<(String, usize)> {
    debug_assert_eq!(b[pos], b'"');
    let mut i = pos + 1;
    // Fast path: scan for end of simple string (no escapes)
    // Safety: input comes from json_to_value(&str), so bytes are valid UTF-8
    let start = i;
    while i < b.len() {
        match b[i] {
            b'"' => {
                let s = unsafe { std::str::from_utf8_unchecked(&b[start..i]) }.to_string();
                return Ok((s, i + 1));
            }
            b'\\' => break,
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
                return Ok((s, i + 1));
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

fn parse_json_string(b: &[u8], pos: usize) -> Result<(Value, usize)> {
    let (s, end) = parse_json_string_raw(b, pos)?;
    Ok((Value::Str(Rc::new(s)), end))
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
    // Fast path for simple integers: parse directly without f64::from_str overhead
    if !has_dot && !has_exp && (i - digits_start) <= 15 {
        let mut n: i64 = 0;
        for &c in &b[digits_start..i] {
            n = n * 10 + (c - b'0') as i64;
        }
        if is_neg { n = -n; }
        return Ok((Value::Num(n as f64, None), i));
    }
    // Safety: number bytes are ASCII digits/signs/dots, always valid UTF-8
    let num_str = unsafe { std::str::from_utf8_unchecked(&b[pos..i]) };
    let n: f64 = num_str.parse().unwrap_or(0.0);
    let f64_repr = format_jq_number(n);
    let repr = if f64_repr == num_str { None } else { Some(Rc::from(num_str)) };
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
        let (key, end) = parse_json_string_raw(b, i)?;
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

/// Format f64 the way jq does — shortest representation with scientific notation for large/small values.
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

    // Integer fast path for common values (fits in i64, reasonable length)
    if n == n.trunc() && n.abs() < 1e16 {
        let i = n as i64;
        if i as f64 == n {
            return format!("{}", i);
        }
    }

    // Use Rust's shortest-representation (like jq's jvp_dtoa)
    let s = format!("{}", n);
    let abs = n.abs();

    // For very small numbers (abs < 1e-4), always use scientific notation (matching %g)
    if abs != 0.0 && abs < 1e-4 {
        return format_as_scientific(n);
    }

    // For large numbers: compare fixed vs scientific length, prefer shorter
    if abs >= 1e16 {
        let sci = format_as_scientific(n);
        if sci.len() < s.len() {
            return sci;
        }
    }

    s
}

/// Format a number in scientific notation matching jq's style.
fn format_as_scientific(n: f64) -> String {
    let sci = format!("{:e}", n);
    let sci = if sci.contains("e-") { sci } else { sci.replacen("e", "e+", 1) };
    normalize_scientific(&sci)
}

/// Normalize scientific notation to match jq's format (e.g., e+07 not e+7 for small exponents).
fn normalize_scientific(s: &str) -> String {
    // Split at 'e'
    if let Some(idx) = s.find('e') {
        let mantissa = &s[..idx];
        let exp_str = &s[idx+1..]; // includes sign
        let (sign, digits) = if exp_str.starts_with('-') {
            ("-", &exp_str[1..])
        } else if exp_str.starts_with('+') {
            ("+", &exp_str[1..])
        } else {
            ("+", exp_str)
        };
        // Parse exponent and re-format with at least 2 digits for < 100
        let exp: i32 = digits.parse().unwrap_or(0);
        if exp.abs() < 100 {
            format!("{}e{}{:02}", mantissa, sign, exp.abs())
        } else {
            format!("{}e{}{}", mantissa, sign, exp.abs())
        }
    } else {
        s.to_string()
    }
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
    // Fast path: check if string needs any escaping
    let bytes = s.as_bytes();
    let needs_escape = bytes.iter().any(|&b| b == b'"' || b == b'\\' || b < 0x20);
    if !needs_escape {
        w.write_all(bytes)?;
    } else {
        // Slow path: write segments between special characters
        let mut start = 0;
        for (i, &b) in bytes.iter().enumerate() {
            let escape = match b {
                b'"' => b"\\\"",
                b'\\' => b"\\\\",
                b'\n' => b"\\n",
                b'\r' => b"\\r",
                b'\t' => b"\\t",
                0x08 => b"\\b",
                0x0c => b"\\f",
                c if c < 0x20 => {
                    // Flush pending
                    if start < i { w.write_all(&bytes[start..i])?; }
                    write!(w, "\\u{:04x}", c)?;
                    start = i + 1;
                    continue;
                }
                _ => { continue; }
            };
            if start < i { w.write_all(&bytes[start..i])?; }
            w.write_all(escape)?;
            start = i + 1;
        }
        if start < bytes.len() { w.write_all(&bytes[start..])?; }
    }
    w.write_all(b"\"")?;
    Ok(())
}

fn write_jq_number(w: &mut dyn io::Write, n: f64) -> io::Result<()> {
    // Fast path for common small integers: avoid String allocation
    if n == n.trunc() && n.abs() < 1e15 {
        let i = n as i64;
        if i as f64 == n {
            let mut buf = itoa::Buffer::new();
            return w.write_all(buf.format(i).as_bytes());
        }
    }
    // Fallback: use format_jq_number for general case
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
