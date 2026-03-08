//! Value type for JIT-compiled jq filters.
//!
//! Uses Vec-backed ordered map for objects (optimal for typical small JSON objects).

use std::ffi::{CStr, CString};
use std::fmt;
use std::rc::Rc;

use anyhow::{Context, Result, bail};

use crate::jq_ffi::{self, Jv, JvKind};

/// Inline-optimized string for object keys (≤24 bytes stored on stack, no heap alloc).
pub type KeyStr = compact_str::CompactString;

// Global pool of Vec buffers for ObjMap reuse.
// Safe because jq-jit is single-threaded (uses Rc, not Arc).
struct ObjMapPool(std::cell::UnsafeCell<Vec<Vec<(KeyStr, Value)>>>);
unsafe impl Sync for ObjMapPool {}

static OBJMAP_POOL: ObjMapPool = ObjMapPool(std::cell::UnsafeCell::new(Vec::new()));

const OBJMAP_POOL_MAX: usize = 64;

// Global pool of Rc<ObjMap> to avoid repeated Rc alloc/dealloc.
// When an Rc<ObjMap> with refcount=1 is dropped, we clear entries and pool the Rc.
// On next alloc, we reuse the Rc memory instead of calling malloc.
struct RcObjMapPool(std::cell::UnsafeCell<Vec<*const ObjMap>>);
unsafe impl Sync for RcObjMapPool {}

static RC_OBJMAP_POOL: RcObjMapPool = RcObjMapPool(std::cell::UnsafeCell::new(Vec::new()));

const RC_OBJMAP_POOL_MAX: usize = 32;

/// Try to recycle an Rc<ObjMap> instead of allocating a new one.
#[inline]
pub fn rc_objmap_pool_get(cap: usize) -> Rc<ObjMap> {
    let pool = unsafe { &mut *RC_OBJMAP_POOL.0.get() };
    if let Some(raw) = pool.pop() {
        // Safety: raw was produced by Rc::into_raw and has refcount=1
        let mut rc = unsafe { Rc::from_raw(raw) };
        let map = Rc::get_mut(&mut rc).unwrap();
        map.entries = pool_get(cap);
        rc
    } else {
        Rc::new(ObjMap::with_capacity(cap))
    }
}

/// Try to pool an Rc<ObjMap> with refcount=1 instead of dropping it.
/// Returns true if pooled, false if dropped normally.
#[inline]
pub fn rc_objmap_pool_return(mut rc: Rc<ObjMap>) -> bool {
    if Rc::strong_count(&rc) != 1 {
        return false;
    }
    let pool = unsafe { &mut *RC_OBJMAP_POOL.0.get() };
    if pool.len() >= RC_OBJMAP_POOL_MAX {
        return false;
    }
    // Clear entries and pool the Vec buffer
    let map = Rc::get_mut(&mut rc).unwrap();
    let old_entries = std::mem::take(&mut map.entries);
    pool_return(old_entries);
    // Pool the Rc itself (prevents deallocation)
    let raw = Rc::into_raw(rc);
    pool.push(raw);
    true
}

#[inline]
fn pool_get(cap: usize) -> Vec<(KeyStr, Value)> {
    let pool = unsafe { &mut *OBJMAP_POOL.0.get() };
    if let Some(v) = pool.pop() {
        if v.capacity() >= cap {
            return v;
        }
    }
    Vec::with_capacity(cap)
}

#[inline]
fn pool_return(mut v: Vec<(KeyStr, Value)>) {
    // Fast path: if all entries are trivially droppable (inline strings, simple numbers),
    // skip the per-entry destructor calls and just zero the length.
    let trivial = v.iter().all(|(k, val)| {
        !k.is_heap_allocated() && match val {
            Value::Null | Value::True | Value::False => true,
            Value::Num(_, repr) => repr.is_none(),
            Value::Str(s) => !s.is_heap_allocated(),
            _ => false,
        }
    });
    if trivial {
        unsafe { v.set_len(0); }
    } else {
        v.clear();
    }
    let pool = unsafe { &mut *OBJMAP_POOL.0.get() };
    if pool.len() < OBJMAP_POOL_MAX {
        pool.push(v);
    }
}

/// Recycle Rc<ObjMap> from a consumed Value to avoid repeated alloc/dealloc.
/// Call this instead of letting a Value drop when you know it won't be used again.
#[inline]
pub fn pool_value(v: Value) {
    if let Value::Obj(rc) = v {
        rc_objmap_pool_return(rc);
    }
}

/// Vec-backed ordered map for JSON objects.
/// Preserves insertion order, last-key-wins on duplicate insert.
/// Optimized for small objects (typical JSON ≤10 keys): no hashing overhead.
/// Vec buffers are pooled thread-locally to avoid repeated allocation.
#[derive(Debug, PartialEq)]
pub struct ObjMap {
    entries: Vec<(KeyStr, Value)>,
}

impl Clone for ObjMap {
    fn clone(&self) -> Self {
        let mut entries = pool_get(self.entries.len());
        entries.extend(self.entries.iter().cloned());
        ObjMap { entries }
    }
}

impl Drop for ObjMap {
    fn drop(&mut self) {
        if self.entries.capacity() > 0 {
            pool_return(std::mem::take(&mut self.entries));
        }
    }
}

impl Default for ObjMap {
    fn default() -> Self {
        ObjMap::new()
    }
}

impl ObjMap {
    #[inline]
    pub fn new() -> Self {
        ObjMap { entries: Vec::new() }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        ObjMap { entries: pool_get(cap) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert a key-value pair. If the key exists, updates the value and returns the old one.
    #[inline]
    pub fn insert(&mut self, key: KeyStr, value: Value) -> Option<Value> {
        for entry in &mut self.entries {
            if entry.0 == key {
                let old = std::mem::replace(&mut entry.1, value);
                return Some(old);
            }
        }
        self.entries.push((key, value));
        None
    }

    /// Push a key-value pair without checking for duplicates.
    /// Caller must ensure the key does not already exist (e.g. during JSON parsing).
    #[inline]
    pub fn push_unique(&mut self, key: KeyStr, value: Value) {
        self.entries.push((key, value));
    }

    /// Get a value by key (linear scan).
    #[inline]
    pub fn get(&self, key: &str) -> Option<&Value> {
        for (k, v) in &self.entries {
            if k.as_str() == key {
                return Some(v);
            }
        }
        None
    }

    /// Get a key-value pair by index.
    #[inline]
    pub fn get_index(&self, index: usize) -> Option<(&KeyStr, &Value)> {
        self.entries.get(index).map(|(k, v)| (k, v))
    }

    /// Remove a key, shifting subsequent entries to preserve order.
    pub fn shift_remove(&mut self, key: &str) -> Option<Value> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k.as_str() == key) {
            Some(self.entries.remove(pos).1)
        } else {
            None
        }
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, (KeyStr, Value)> {
        self.entries.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, (KeyStr, Value)> {
        self.entries.iter_mut()
    }

    #[inline]
    pub fn keys(&self) -> ObjMapKeys<'_> {
        ObjMapKeys(self.entries.iter())
    }

    #[inline]
    pub fn values(&self) -> ObjMapValues<'_> {
        ObjMapValues(self.entries.iter())
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k.as_str() == key)
    }
}

impl std::iter::FromIterator<(KeyStr, Value)> for ObjMap {
    fn from_iter<I: IntoIterator<Item = (KeyStr, Value)>>(iter: I) -> Self {
        ObjMap { entries: iter.into_iter().collect() }
    }
}

impl IntoIterator for ObjMap {
    type Item = (KeyStr, Value);
    type IntoIter = std::vec::IntoIter<(KeyStr, Value)>;
    fn into_iter(mut self) -> Self::IntoIter {
        // Take entries out before Drop runs (which would return the Vec to pool)
        std::mem::take(&mut self.entries).into_iter()
    }
}

impl<'a> IntoIterator for &'a ObjMap {
    type Item = &'a (KeyStr, Value);
    type IntoIter = std::slice::Iter<'a, (KeyStr, Value)>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

/// Create a new empty ObjMap.
#[inline]
pub fn new_objmap() -> ObjMap {
    ObjMap::new()
}

/// Create a new ObjMap with pre-allocated capacity.
#[inline]
pub fn new_objmap_with_capacity(cap: usize) -> ObjMap {
    ObjMap::with_capacity(cap)
}

pub struct ObjMapKeys<'a>(std::slice::Iter<'a, (KeyStr, Value)>);
impl<'a> Iterator for ObjMapKeys<'a> {
    type Item = &'a KeyStr;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> { self.0.next().map(|(k, _)| k) }
    fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}
impl<'a> DoubleEndedIterator for ObjMapKeys<'a> {
    fn next_back(&mut self) -> Option<Self::Item> { self.0.next_back().map(|(k, _)| k) }
}
impl<'a> ExactSizeIterator for ObjMapKeys<'a> {}

pub struct ObjMapValues<'a>(std::slice::Iter<'a, (KeyStr, Value)>);
impl<'a> Iterator for ObjMapValues<'a> {
    type Item = &'a Value;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> { self.0.next().map(|(_, v)| v) }
    fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}
impl<'a> DoubleEndedIterator for ObjMapValues<'a> {
    fn next_back(&mut self) -> Option<Self::Item> { self.0.next_back().map(|(_, v)| v) }
}
impl<'a> ExactSizeIterator for ObjMapValues<'a> {}

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
    Str(KeyStr),
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
            Value::Str(s) => Value::Str(s.clone()),
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

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        Value::Str(KeyStr::from(s))
    }

    pub fn from_string(s: String) -> Self {
        Value::Str(KeyStr::from(s))
    }

    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, Value)>) -> Self {
        Value::Obj(Rc::new(pairs.into_iter().map(|(k, v)| (KeyStr::from(k), v)).collect()))
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

/// Parse multiple JSON values from a single string, calling a callback for each.
/// Avoids the double-scan of split_json_values + json_to_value.
pub fn json_stream<F>(input: &str, mut cb: F) -> Result<()>
where F: FnMut(Value) -> Result<()> {
    let bytes = input.as_bytes();
    let mut pos = 0;
    // Skip BOM
    if bytes.len() >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF { pos = 3; }
    pos = skip_ws(bytes, pos);
    while pos < bytes.len() {
        let (val, end) = parse_json_value(bytes, pos, 0)?;
        cb(val)?;
        pos = skip_ws(bytes, end);
    }
    Ok(())
}

/// Like json_stream but also provides the byte range [start, end) for each value.
pub fn json_stream_offsets<F>(input: &str, mut cb: F) -> Result<()>
where F: FnMut(Value, usize, usize) -> Result<()> {
    let bytes = input.as_bytes();
    let mut pos = 0;
    if bytes.len() >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF { pos = 3; }
    pos = skip_ws(bytes, pos);
    while pos < bytes.len() {
        let (val, end) = parse_json_value(bytes, pos, 0)?;
        cb(val, pos, end)?;
        pos = skip_ws(bytes, end);
    }
    Ok(())
}

/// Stream raw JSON byte ranges without parsing values.
/// For identity filters in compact mode — skips parsing entirely, just validates structure.
pub fn json_stream_raw<F>(input: &str, mut cb: F) -> Result<()>
where F: FnMut(usize, usize) -> Result<()> {
    let bytes = input.as_bytes();
    let mut pos = 0;
    if bytes.len() >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF { pos = 3; }
    // Try NDJSON fast path: if first value ends at a newline, use line-based scanning.
    pos = skip_ws(bytes, pos);
    if pos < bytes.len() {
        let end = skip_json_value(bytes, pos)?;
        let ws_after = skip_ws(bytes, end);
        if end < bytes.len() && bytes[end] == b'\n' && ws_after == end + 1 {
            // First value ends exactly at a newline — use NDJSON line scanning for rest
            cb(pos, end)?;
            return json_stream_ndjson(bytes, end + 1, cb);
        }
        cb(pos, end)?;
        pos = ws_after;
    }
    while pos < bytes.len() {
        let end = skip_json_value(bytes, pos)?;
        cb(pos, end)?;
        pos = skip_ws(bytes, end);
    }
    Ok(())
}

/// NDJSON fast path: use memchr to find newlines, trimming trailing whitespace.
/// Each line is assumed to be a single JSON value. Falls back to skip_json_value
/// if a line looks suspicious (starts with whitespace within the line).
fn json_stream_ndjson<F>(bytes: &[u8], start: usize, mut cb: F) -> Result<()>
where F: FnMut(usize, usize) -> Result<()> {
    use memchr::memchr;
    let mut pos = start;
    while pos < bytes.len() {
        // Skip leading whitespace/newlines
        while pos < bytes.len() && matches!(bytes[pos], b' ' | b'\t' | b'\n' | b'\r') { pos += 1; }
        if pos >= bytes.len() { break; }
        let line_end = match memchr(b'\n', &bytes[pos..]) {
            Some(offset) => pos + offset,
            None => bytes.len(),
        };
        // Trim trailing whitespace
        let mut end = line_end;
        while end > pos && matches!(bytes[end - 1], b' ' | b'\t' | b'\r') { end -= 1; }
        if end > pos {
            cb(pos, end)?;
        }
        pos = line_end + 1;
    }
    Ok(())
}

/// Extract a numeric field value from a JSON object without full parsing.
/// Returns Some(f64) if the field exists and is numeric, None otherwise.
/// Used for select fast paths to avoid parsing discarded objects.
pub fn json_object_get_num(b: &[u8], pos: usize, field: &str) -> Option<f64> {
    if pos >= b.len() || b[pos] != b'{' { return None; }
    let field_bytes = field.as_bytes();
    let mut i = pos + 1;
    // Skip whitespace
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { return None; }
    loop {
        if i >= b.len() || b[i] != b'"' { return None; }
        // Scan key
        let key_start = i + 1;
        let mut j = key_start;
        while j < b.len() {
            match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 }
        }
        let key_matches = (j - key_start) == field_bytes.len()
            && b[key_start..j] == *field_bytes;
        i = j + 1;
        // Skip ws + colon
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if key_matches {
            // Parse the numeric value inline
            if i >= b.len() { return None; }
            let neg = b[i] == b'-';
            let start = if neg { i + 1 } else { i };
            if start >= b.len() || !b[start].is_ascii_digit() { return None; }
            // Fast integer path
            let mut n: i64 = (b[start] - b'0') as i64;
            let mut k = start + 1;
            while k < b.len() && b[k].is_ascii_digit() {
                n = n * 10 + (b[k] - b'0') as i64;
                k += 1;
            }
            if k < b.len() && (b[k] == b'.' || b[k] == b'e' || b[k] == b'E') {
                // Has decimal/exponent — use fast-float
                let end = {
                    let mut e = k;
                    if b[e] == b'.' { e += 1; while e < b.len() && b[e].is_ascii_digit() { e += 1; } }
                    if e < b.len() && (b[e] == b'e' || b[e] == b'E') {
                        e += 1;
                        if e < b.len() && (b[e] == b'+' || b[e] == b'-') { e += 1; }
                        while e < b.len() && b[e].is_ascii_digit() { e += 1; }
                    }
                    e
                };
                let num_str = unsafe { std::str::from_utf8_unchecked(&b[i..end]) };
                return fast_float::parse::<f64, _>(num_str).ok();
            }
            if (k - start) > 15 { return None; }
            let val = if neg { -(n as f64) } else { n as f64 };
            return Some(val);
        }
        // Skip value
        i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return None };
        // Skip ws
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return None; }
        if b[i] == b'}' { return None; }
        if b[i] != b',' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
}

/// Extract the raw byte range of a field value from a JSON object without full parsing.
/// Returns Some((start, end)) byte offsets of the field's value, or None if not found
/// or the input isn't a JSON object. Used for field access fast paths.
pub fn json_object_get_field_raw(b: &[u8], pos: usize, field: &str) -> Option<(usize, usize)> {
    if pos >= b.len() || b[pos] != b'{' { return None; }
    let field_bytes = field.as_bytes();
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { return None; }
    loop {
        if i >= b.len() || b[i] != b'"' { return None; }
        let key_start = i + 1;
        let mut j = key_start;
        while j < b.len() {
            match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 }
        }
        let key_matches = (j - key_start) == field_bytes.len()
            && b[key_start..j] == *field_bytes;
        i = j + 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if key_matches {
            let val_start = i;
            let val_end = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return None };
            return Some((val_start, val_end));
        }
        i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return None };
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return None; }
        if b[i] == b'}' { return None; }
        if b[i] != b',' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
}

/// Count the number of key-value pairs in a JSON object without full parsing.
/// Returns None if the input isn't a JSON object.
pub fn json_object_length(b: &[u8], pos: usize) -> Option<usize> {
    if pos >= b.len() || b[pos] != b'{' { return None; }
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { return Some(0); }
    let mut count = 0usize;
    loop {
        if i >= b.len() || b[i] != b'"' { return None; }
        // Skip key string
        i += 1;
        while i < b.len() {
            match b[i] { b'"' => break, b'\\' => i += 2, _ => i += 1 }
        }
        if i >= b.len() { return None; }
        i += 1; // past closing quote
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        // Skip value
        i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return None };
        count += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return None; }
        if b[i] == b'}' { return Some(count); }
        if b[i] != b',' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
}

/// Count the length of a JSON value without full parsing.
/// - object: number of keys
/// - array: number of elements
/// - string: character count (requires scanning for multi-byte chars)
/// - null: returns 0
/// Returns None if we can't determine the length quickly.
pub fn json_value_length(b: &[u8], pos: usize) -> Option<usize> {
    if pos >= b.len() { return None; }
    match b[pos] {
        b'{' => json_object_length(b, pos),
        b'[' => {
            let mut i = pos + 1;
            while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
            if i < b.len() && b[i] == b']' { return Some(0); }
            let mut count = 0usize;
            loop {
                i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return None };
                count += 1;
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
                if i >= b.len() { return None; }
                if b[i] == b']' { return Some(count); }
                if b[i] != b',' { return None; }
                i += 1;
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
            }
        }
        b'n' => Some(0), // null → 0
        _ => None, // numbers return their value (not a usize), strings need unicode counting
    }
}

/// Extract all keys from a JSON object and write them as a sorted JSON array into buf.
/// Returns true if successful, false if the input isn't a JSON object.
pub fn json_object_keys_to_buf(b: &[u8], pos: usize, buf: &mut Vec<u8>) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' {
        buf.extend_from_slice(b"[]\n");
        return true;
    }
    // Collect raw key byte ranges
    let mut keys: Vec<(usize, usize)> = Vec::new();
    loop {
        if i >= b.len() || b[i] != b'"' { return false; }
        let key_start = i; // include the opening quote
        i += 1;
        while i < b.len() {
            match b[i] { b'"' => break, b'\\' => i += 2, _ => i += 1 }
        }
        if i >= b.len() { return false; }
        let key_end = i + 1; // include the closing quote
        keys.push((key_start, key_end));
        i = key_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return false };
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return false; }
        if b[i] == b'}' { break; }
        if b[i] != b',' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
    // Sort keys by their string content (between quotes)
    keys.sort_by(|a, c| b[a.0+1..a.1-1].cmp(&b[c.0+1..c.1-1]));
    buf.push(b'[');
    for (idx, (ks, ke)) in keys.iter().enumerate() {
        if idx > 0 { buf.push(b','); }
        buf.extend_from_slice(&b[*ks..*ke]);
    }
    buf.extend_from_slice(b"]\n");
    true
}

/// Check if a JSON object contains a given key without full parsing.
pub fn json_object_has_key(b: &[u8], pos: usize, field: &str) -> Option<bool> {
    if pos >= b.len() || b[pos] != b'{' { return None; }
    let field_bytes = field.as_bytes();
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { return Some(false); }
    loop {
        if i >= b.len() || b[i] != b'"' { return None; }
        let key_start = i + 1;
        let mut j = key_start;
        while j < b.len() {
            match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 }
        }
        if (j - key_start) == field_bytes.len() && b[key_start..j] == *field_bytes {
            return Some(true);
        }
        i = j + 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return None };
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return None; }
        if b[i] == b'}' { return Some(false); }
        if b[i] != b',' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
}

/// Get the jq type name for a JSON value by its first byte, without parsing.
/// Returns the type string literal (e.g., "object", "array", "string", "number", "boolean", "null").
pub fn json_type_byte(first: u8) -> &'static [u8] {
    match first {
        b'{' => b"\"object\"",
        b'[' => b"\"array\"",
        b'"' => b"\"string\"",
        b't' | b'f' => b"\"boolean\"",
        b'n' => b"\"null\"",
        b'-' | b'0'..=b'9' => b"\"number\"",
        _ => b"\"null\"",
    }
}

/// Extract two numeric field values from a JSON object without full parsing.
/// More efficient than calling json_object_get_num twice (single scan).
pub fn json_object_get_two_nums(b: &[u8], pos: usize, field1: &str, field2: &str) -> Option<(f64, f64)> {
    if pos >= b.len() || b[pos] != b'{' { return None; }
    let f1 = field1.as_bytes();
    let f2 = field2.as_bytes();
    let mut val1: Option<f64> = None;
    let mut val2: Option<f64> = None;
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { return None; }
    loop {
        if i >= b.len() || b[i] != b'"' { return None; }
        let key_start = i + 1;
        let mut j = key_start;
        while j < b.len() {
            match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 }
        }
        let key_len = j - key_start;
        let match1 = val1.is_none() && key_len == f1.len() && b[key_start..j] == *f1;
        let match2 = !match1 && val2.is_none() && key_len == f2.len() && b[key_start..j] == *f2;
        i = j + 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if match1 || match2 {
            // Parse numeric value inline
            if i >= b.len() { return None; }
            let neg = b[i] == b'-';
            let start = if neg { i + 1 } else { i };
            if start >= b.len() || !b[start].is_ascii_digit() { return None; }
            let mut n: i64 = (b[start] - b'0') as i64;
            let mut k = start + 1;
            while k < b.len() && b[k].is_ascii_digit() {
                n = n * 10 + (b[k] - b'0') as i64;
                k += 1;
            }
            let val = if k < b.len() && (b[k] == b'.' || b[k] == b'e' || b[k] == b'E') {
                let end = {
                    let mut e = k;
                    if b[e] == b'.' { e += 1; while e < b.len() && b[e].is_ascii_digit() { e += 1; } }
                    if e < b.len() && (b[e] == b'e' || b[e] == b'E') {
                        e += 1;
                        if e < b.len() && (b[e] == b'+' || b[e] == b'-') { e += 1; }
                        while e < b.len() && b[e].is_ascii_digit() { e += 1; }
                    }
                    e
                };
                let num_str = unsafe { std::str::from_utf8_unchecked(&b[i..end]) };
                i = end;
                fast_float::parse::<f64, _>(num_str).ok()?
            } else {
                if (k - start) > 15 { return None; }
                i = k;
                if neg { -(n as f64) } else { n as f64 }
            };
            if match1 { val1 = Some(val); } else { val2 = Some(val); }
            if val1.is_some() && val2.is_some() {
                return Some((val1.unwrap(), val2.unwrap()));
            }
        } else {
            i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return None };
        }
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return None; }
        if b[i] == b'}' { return None; }
        if b[i] != b',' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
}

/// Extract raw byte ranges for multiple fields from a JSON object.
/// `fields` is a list of (output_key, input_field) pairs.
/// Returns a Vec of (start, end) pairs for each field value, in the same order as `fields`.
/// Returns None if any field is missing or input isn't an object.
pub fn json_object_get_fields_raw(b: &[u8], pos: usize, input_fields: &[&str]) -> Option<Vec<(usize, usize)>> {
    if pos >= b.len() || b[pos] != b'{' { return None; }
    let n = input_fields.len();
    let mut results: Vec<Option<(usize, usize)>> = vec![None; n];
    let mut found = 0;
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' {
        return None; // empty object, fields can't be found
    }
    loop {
        if i >= b.len() || b[i] != b'"' { return None; }
        let key_start = i + 1;
        let mut j = key_start;
        while j < b.len() {
            match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 }
        }
        let key_len = j - key_start;
        // Check which field this key matches
        let mut matched_idx = None;
        for (idx, field) in input_fields.iter().enumerate() {
            if results[idx].is_none() && key_len == field.len() && b[key_start..j] == *field.as_bytes() {
                matched_idx = Some(idx);
                break;
            }
        }
        i = j + 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if let Some(idx) = matched_idx {
            let val_start = i;
            let val_end = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return None };
            results[idx] = Some((val_start, val_end));
            found += 1;
            if found == n { break; }
            i = val_end;
        } else {
            i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return None };
        }
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return None; }
        if b[i] == b'}' { break; }
        if b[i] != b',' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
    // Convert Options to values, using None → missing field → return None
    let mut out = Vec::with_capacity(n);
    for r in results {
        out.push(r?);
    }
    Some(out)
}

/// Compact a JSON value by stripping whitespace outside strings.
/// Copies to `buf` directly, avoiding Value construction.
pub fn push_json_compact_raw(buf: &mut Vec<u8>, b: &[u8]) {
    let mut i = 0;
    let len = b.len();
    while i < len {
        match b[i] {
            b' ' | b'\t' | b'\n' | b'\r' => { i += 1; }
            b'"' => {
                // Copy entire string including quotes
                let str_start = i;
                i += 1;
                while i < len {
                    match b[i] {
                        b'"' => { i += 1; break; }
                        b'\\' => { i += 2; }
                        _ => { i += 1; }
                    }
                }
                buf.extend_from_slice(&b[str_start..i]);
            }
            c => { buf.push(c); i += 1; }
        }
    }
}

/// Check if raw JSON bytes are compact (no whitespace outside of strings).
/// For objects, checks after `{` and after the first `:` for whitespace.
/// For arrays, checks after `[` for whitespace. Scalars are always compact.
#[inline]
pub fn is_json_compact(bytes: &[u8]) -> bool {
    if bytes.len() < 2 { return true; }
    let b0 = bytes[0];
    if b0 == b'[' {
        if bytes[1] <= b' ' { return false; }
        // Check after first comma: [1,2] is compact, [1, 2] is not
        // Skip the first value to find the comma
        if let Ok(end) = skip_json_value(bytes, 1) {
            if end < bytes.len() && bytes[end] == b',' {
                if end + 1 < bytes.len() && bytes[end + 1] <= b' ' { return false; }
            }
        }
        return true;
    }
    if b0 != b'{' { return true; }
    // Object: check after { for whitespace
    if bytes[1] <= b' ' { return false; }
    // Check after first colon: {"key":v is compact, {"key": v is not
    let mut i = 1;
    if i < bytes.len() && bytes[i] == b'"' {
        i += 1;
        while i < bytes.len() {
            if bytes[i] == b'\\' { i += 2; continue; }
            if bytes[i] == b'"' { i += 1; break; }
            i += 1;
        }
    }
    // i should be at ':'
    if i < bytes.len() && bytes[i] == b':' {
        i += 1;
        if i < bytes.len() && bytes[i] <= b' ' { return false; }
    }
    true
}

#[inline(always)]
fn skip_ws(b: &[u8], mut pos: usize) -> usize {
    while pos < b.len() && matches!(b[pos], b' ' | b'\t' | b'\n' | b'\r') { pos += 1; }
    pos
}

const MAX_JSON_DEPTH: usize = 10000;

/// Skip past a JSON value without constructing it. Returns the position after the value.
fn skip_json_value(b: &[u8], pos: usize) -> Result<usize> {
    if pos >= b.len() { bail!("Unexpected end of JSON"); }
    match b[pos] {
        b'n' => { if b.get(pos..pos+4) == Some(b"null") { Ok(pos + 4) } else { bail!("Invalid JSON at position {}", pos) } }
        b't' => { if b.get(pos..pos+4) == Some(b"true") { Ok(pos + 4) } else { bail!("Invalid JSON at position {}", pos) } }
        b'f' => { if b.get(pos..pos+5) == Some(b"false") { Ok(pos + 5) } else { bail!("Invalid JSON at position {}", pos) } }
        b'"' => {
            let mut i = pos + 1;
            while i < b.len() {
                match b[i] {
                    b'"' => return Ok(i + 1),
                    b'\\' => i += 2,
                    _ => i += 1,
                }
            }
            bail!("Unterminated string")
        }
        b'[' => {
            let mut i = skip_ws(b, pos + 1);
            if i < b.len() && b[i] == b']' { return Ok(i + 1); }
            loop {
                i = skip_json_value(b, i)?;
                i = skip_ws(b, i);
                if i >= b.len() { bail!("Unterminated array"); }
                if b[i] == b']' { return Ok(i + 1); }
                if b[i] != b',' { bail!("Expected ',' or ']' at position {}", i); }
                i = skip_ws(b, i + 1);
            }
        }
        b'{' => {
            let mut i = skip_ws(b, pos + 1);
            if i < b.len() && b[i] == b'}' { return Ok(i + 1); }
            loop {
                if i >= b.len() || b[i] != b'"' { bail!("Expected string key at position {}", i); }
                // Skip key
                let mut j = i + 1;
                while j < b.len() {
                    match b[j] { b'"' => break, b'\\' => j += 2, _ => j += 1 }
                }
                i = skip_ws(b, j + 1);
                if i >= b.len() || b[i] != b':' { bail!("Expected ':' at position {}", i); }
                i = skip_ws(b, i + 1);
                i = skip_json_value(b, i)?;
                i = skip_ws(b, i);
                if i >= b.len() { bail!("Unterminated object"); }
                if b[i] == b'}' { return Ok(i + 1); }
                if b[i] != b',' { bail!("Expected ',' or '}}' at position {}", i); }
                i = skip_ws(b, i + 1);
            }
        }
        b'-' | b'0'..=b'9' => {
            let mut i = pos;
            if b[i] == b'-' { i += 1; }
            while i < b.len() && b[i].is_ascii_digit() { i += 1; }
            if i < b.len() && b[i] == b'.' { i += 1; while i < b.len() && b[i].is_ascii_digit() { i += 1; } }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                i += 1;
                if i < b.len() && (b[i] == b'+' || b[i] == b'-') { i += 1; }
                while i < b.len() && b[i].is_ascii_digit() { i += 1; }
            }
            Ok(i)
        }
        c => bail!("Unexpected character '{}' at position {}", c as char, pos),
    }
}

/// Parse a JSON object, only parsing values for keys in `fields`. Other values are skipped.
/// Uses the same key parsing as parse_json_object but skips value parsing for non-matching keys.
fn parse_json_object_project(b: &[u8], pos: usize, depth: usize, fields: &[&str]) -> Result<(Value, usize)> {
    debug_assert_eq!(b[pos], b'{');
    let mut i = skip_ws(b, pos + 1);
    let n = fields.len();
    let mut map = new_objmap_with_capacity(n);
    if i < b.len() && b[i] == b'}' {
        // Empty object: fill with nulls
        for &f in fields { map.push_unique(KeyStr::from(f), Value::Null); }
        return Ok((Value::Obj(Rc::new(map)), i + 1));
    }
    let mut found = 0u64;
    let all_found: u64 = if n < 64 { (1u64 << n) - 1 } else { u64::MAX };
    loop {
        if i >= b.len() || b[i] != b'"' { bail!("Expected string key at position {}", i); }
        let (key, end) = parse_json_key(b, i)?;
        i = skip_ws(b, end);
        if i >= b.len() || b[i] != b':' { bail!("Expected ':' at position {}", i); }
        i = skip_ws(b, i + 1);
        // Check if key is in projection set
        let mut matched = false;
        if found != all_found {
            let key_str = key.as_str();
            for (fi, &f) in fields.iter().enumerate() {
                if fi < 64 && (found & (1u64 << fi)) != 0 { continue; }
                if key_str == f {
                    let (val, end) = parse_json_value(b, i, depth + 1)?;
                    map.push_unique(key, val);
                    found |= 1u64 << fi;
                    i = end;
                    matched = true;
                    break;
                }
            }
        }
        if !matched {
            i = skip_json_value(b, i)?;
        }
        i = skip_ws(b, i);
        if i >= b.len() { bail!("Unterminated object"); }
        if b[i] == b'}' { break; }
        if b[i] != b',' { bail!("Expected ',' or '}}' at position {}", i); }
        i = skip_ws(b, i + 1);
        // Early exit: all needed fields found — skip rest of object
        if found == all_found {
            // Find closing brace, handling nesting
            let mut nest = 1u32;
            while i < b.len() && nest > 0 {
                match b[i] {
                    b'{' => { nest += 1; i += 1; }
                    b'}' => { nest -= 1; if nest == 0 { break; } i += 1; }
                    b'"' => { i += 1; while i < b.len() { match b[i] { b'"' => { i += 1; break; } b'\\' => i += 2, _ => i += 1 } } }
                    _ => i += 1,
                }
            }
            break;
        }
    }
    // Add null for missing fields (in field order)
    for (fi, &f) in fields.iter().enumerate() {
        if fi < 64 && (found & (1u64 << fi)) != 0 { continue; }
        map.push_unique(KeyStr::from(f), Value::Null);
    }
    Ok((Value::Obj(Rc::new(map)), i + 1))
}

/// Stream JSON values from input, only parsing specified fields from top-level objects.
/// Other fields are skipped without constructing values.
pub fn json_stream_project<F>(input: &str, fields: &[&str], mut cb: F) -> Result<()>
where F: FnMut(Value) -> Result<()> {
    let bytes = input.as_bytes();
    let mut pos = 0;
    if bytes.len() >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF { pos = 3; }
    pos = skip_ws(bytes, pos);
    while pos < bytes.len() {
        let (val, end) = if bytes[pos] == b'{' {
            parse_json_object_project(bytes, pos, 0, fields)?
        } else {
            parse_json_value(bytes, pos, 0)?
        };
        cb(val)?;
        pos = skip_ws(bytes, end);
    }
    Ok(())
}

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
            // Fast path for negative integers (most common case for '-')
            let j = pos + 1;
            if j < b.len() && b[j] >= b'1' && b[j] <= b'9' {
                let mut n: i64 = -((b[j] - b'0') as i64);
                let mut k = j + 1;
                while k < b.len() && b[k].is_ascii_digit() {
                    n = n * 10 - (b[k] - b'0') as i64;
                    k += 1;
                }
                if (k >= b.len() || (b[k] != b'.' && b[k] != b'e' && b[k] != b'E')) && (k - j) <= 15 {
                    return Ok((Value::Num(n as f64, None), k));
                }
                // Has decimal/exponent — fall through to full parser
                parse_json_number(b, pos)
            }
            // -Infinity, -NaN, or other
            else if b.get(pos..pos+9) == Some(b"-Infinity") { Ok((Value::Num(f64::NEG_INFINITY, None), pos + 9)) }
            else if b.get(pos..pos+4) == Some(b"-NaN") { Ok((Value::Num(f64::NAN, None), pos + 4)) }
            else { parse_json_number(b, pos) }
        }
        b'0' => {
            // Fast path for zero or numbers starting with 0 (must be just "0" unless "0." etc)
            let j = pos + 1;
            if j >= b.len() || (b[j] != b'.' && b[j] != b'e' && b[j] != b'E' && !b[j].is_ascii_digit()) {
                Ok((Value::Num(0.0, None), j))
            } else {
                parse_json_number(b, pos)
            }
        }
        b'1'..=b'9' => {
            // Fast inline path: accumulate digits in single pass for non-negative integers
            let mut n: i64 = (b[pos] - b'0') as i64;
            let mut j = pos + 1;
            while j < b.len() && b[j].is_ascii_digit() {
                n = n * 10 + (b[j] - b'0') as i64;
                j += 1;
            }
            if (j >= b.len() || (b[j] != b'.' && b[j] != b'e' && b[j] != b'E')) && (j - pos) <= 15 {
                Ok((Value::Num(n as f64, None), j))
            } else {
                parse_json_number(b, pos)
            }
        }
        c => bail!("Unexpected character '{}' at position {}", c as char, pos),
    }
}

/// Parse a JSON string, returning the raw String and end position.
/// Used by both parse_json_string (wraps in Value) and parse_json_object (uses String directly as key).
fn parse_json_string_raw(b: &[u8], pos: usize) -> Result<(String, usize)> {
    debug_assert_eq!(b[pos], b'"');
    let mut i = pos + 1;
    // Fast path: scan for end of simple string (no escapes)
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

#[inline]
fn parse_json_string(b: &[u8], pos: usize) -> Result<(Value, usize)> {
    // Fast path: no escapes — create CompactString directly from byte slice (no intermediate String)
    debug_assert_eq!(b[pos], b'"');
    let mut i = pos + 1;
    let start = i;
    while i < b.len() {
        match b[i] {
            b'"' => {
                let s = KeyStr::from(unsafe { std::str::from_utf8_unchecked(&b[start..i]) });
                return Ok((Value::Str(s), i + 1));
            }
            b'\\' => break,
            _ => i += 1,
        }
    }
    // Slow path: has escapes — go through parse_json_string_raw
    let (s, end) = parse_json_string_raw(b, pos)?;
    Ok((Value::Str(KeyStr::from(s)), end))
}

/// Parse a JSON key, returning KeyStr (inline for short keys ≤ 24 bytes — no heap allocation).
#[inline]
fn parse_json_key(b: &[u8], pos: usize) -> Result<(KeyStr, usize)> {
    debug_assert_eq!(b[pos], b'"');
    let mut i = pos + 1;
    let start = i;
    while i < b.len() {
        match b[i] {
            b'"' => {
                let s = KeyStr::from(unsafe { std::str::from_utf8_unchecked(&b[start..i]) });
                return Ok((s, i + 1));
            }
            b'\\' => break,
            _ => i += 1,
        }
    }
    // Slow path: fall back to full string parser
    let (s, end) = parse_json_string_raw(b, pos)?;
    Ok((KeyStr::from(s), end))
}

#[inline]
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
    let n: f64 = fast_float::parse(num_str).unwrap_or(0.0);
    // Fast path: for simple decimals (no exponent, no trailing zeros after dot, ≤15 sig digits),
    // Rust's Display roundtrips identically — skip format_jq_number to avoid String allocation.
    if !has_exp && has_dot && (i - digits_start) <= 16 {
        let last = b[i - 1];
        if last != b'0' && (b[digits_start] != b'0' || digits_start + 1 == i || b[digits_start + 1] == b'.') {
            return Ok((Value::Num(n, None), i));
        }
        // Integer value with decimal notation (e.g., "1.0", "2.00") — format_jq_number would
        // return just the integer, so repr always differs. Skip format call.
        if n == n.trunc() && n.abs() < 1e16 {
            let iv = n as i64;
            if iv as f64 == n {
                return Ok((Value::Num(n, Some(Rc::from(num_str))), i));
            }
        }
    }
    let f64_repr = format_jq_number(n);
    let repr = if f64_repr == num_str { None } else { Some(Rc::from(num_str)) };
    Ok((Value::Num(n, repr), i))
}

fn parse_json_array(b: &[u8], pos: usize, depth: usize) -> Result<(Value, usize)> {
    debug_assert_eq!(b[pos], b'[');
    let mut i = skip_ws(b, pos + 1);
    if i < b.len() && b[i] == b']' { return Ok((Value::Arr(Rc::new(Vec::new())), i + 1)); }
    let mut items = Vec::with_capacity(4);
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
    if i < b.len() && b[i] == b'}' { return Ok((Value::Obj(Rc::new(ObjMap::new())), i + 1)); }
    // Use Rc pool to recycle both the Rc allocation and the Vec buffer
    let mut rc = rc_objmap_pool_get(4);
    let map = Rc::get_mut(&mut rc).unwrap();
    loop {
        if i >= b.len() || b[i] != b'"' { bail!("Expected string key at position {}", i); }
        let (key, end) = parse_json_key(b, i)?;
        i = skip_ws(b, end);
        if i >= b.len() || b[i] != b':' { bail!("Expected ':' at position {}", i); }
        i = skip_ws(b, i + 1);
        let (val, end) = parse_json_value(b, i, depth + 1)?;
        // push_unique skips the linear scan for duplicate keys — safe for JSON parsing
        // where duplicate keys are extremely rare and "first wins" is acceptable
        map.push_unique(key, val);
        i = skip_ws(b, end);
        if i >= b.len() { bail!("Unterminated object"); }
        if b[i] == b'}' { return Ok((Value::Obj(rc), i + 1)); }
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

/// # Safety
/// `jv` must be a valid `Jv` value obtained from libjq.
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
                Ok(Value::Str(KeyStr::from(s)))
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
                let mut map = new_objmap();
                let mut iter = jq_ffi::jv_object_iter(jv);
                while jq_ffi::jv_object_iter_valid(jv, iter) != 0 {
                    let key_jv = jq_ffi::jv_object_iter_key(jv, iter);
                    let val_jv = jq_ffi::jv_object_iter_value(jv, iter);
                    let key_cstr = jq_ffi::jv_string_value(key_jv);
                    let key = KeyStr::from(CStr::from_ptr(key_cstr).to_string_lossy().as_ref());
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
            let mut buf = itoa::Buffer::new();
            return buf.format(i).to_string();
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
        let (sign, digits) = if let Some(rest) = exp_str.strip_prefix('-') {
            ("-", rest)
        } else if let Some(rest) = exp_str.strip_prefix('+') {
            ("+", rest)
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
        Value::Str(s) => {
            let mut out = String::with_capacity(s.len() + 2);
            push_json_string(&mut out, s);
            out
        }
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
                push_json_string(&mut out, k);
                out.push(':');
                out.push_str(&value_to_json_depth(v, depth + 1, precise));
            }
            out.push('}');
            out
        }
        Value::Error(e) => {
            let mut out = String::with_capacity(e.len() + 2);
            push_json_string(&mut out, e);
            out
        }
    }
}

/// Convert Value to pretty-printed JSON string.
pub fn value_to_json_pretty(v: &Value, indent: usize) -> String {
    value_to_json_pretty_ext(v, indent, 2, false, false)
}

/// Convert Value to pretty-printed JSON string with formatting options.
///
/// - `depth`: current indentation depth (in units of indent chars)
/// - `step`: number of indent chars per level (e.g. 2 for 2-space indent)
/// - `use_tab`: use tab characters instead of spaces
/// - `sort_keys`: sort object keys alphabetically
pub fn value_to_json_pretty_ext(v: &Value, depth: usize, step: usize, use_tab: bool, sort_keys: bool) -> String {
    let mut out = String::with_capacity(128);
    write_pretty_to_string(&mut out, v, depth, step, use_tab, sort_keys);
    out
}

// Pre-computed indent buffers (avoid per-call allocation from repeat())
const SPACES_256: &str = "                                                                                                                                                                                                                                                                ";
const TABS_64: &str = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";

#[inline]
fn push_indent(out: &mut String, n: usize, use_tab: bool) {
    if n == 0 { return; }
    let buf = if use_tab { TABS_64 } else { SPACES_256 };
    if n <= buf.len() {
        out.push_str(&buf[..n]);
    } else {
        // Unlikely: very deep nesting
        let ch = if use_tab { '\t' } else { ' ' };
        for _ in 0..n { out.push(ch); }
    }
}

#[inline]
fn push_json_string(out: &mut String, s: &str) {
    let bytes = s.as_bytes();
    out.push('"');
    // Single-pass: copy safe chunks between escape chars
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        let esc = match b {
            b'"' => b'\"' as u16 | 0x100,
            b'\\' => b'\\' as u16 | 0x100,
            b'\n' => b'n' as u16 | 0x100,
            b'\r' => b'r' as u16 | 0x100,
            b'\t' => b't' as u16 | 0x100,
            0x08 => b'b' as u16 | 0x100,
            0x0c => b'f' as u16 | 0x100,
            c if c < 0x20 => c as u16,
            _ => continue,
        };
        if start < i {
            out.push_str(unsafe { std::str::from_utf8_unchecked(&bytes[start..i]) });
        }
        if esc & 0x100 != 0 {
            let pair = [b'\\', esc as u8];
            out.push_str(unsafe { std::str::from_utf8_unchecked(&pair) });
        } else {
            use std::fmt::Write;
            let _ = write!(out, "\\u{:04x}", esc);
        }
        start = i + 1;
    }
    if start < bytes.len() {
        out.push_str(unsafe { std::str::from_utf8_unchecked(&bytes[start..]) });
    }
    out.push('"');
}

#[inline]
fn push_jq_number(out: &mut String, n: f64) {
    if n == n.trunc() && n.abs() < 1e15 {
        let i = n as i64;
        if i as f64 == n {
            let mut buf = itoa::Buffer::new();
            out.push_str(buf.format(i));
            return;
        }
    }
    out.push_str(&format_jq_number(n));
}

fn write_pretty_to_string(out: &mut String, v: &Value, depth: usize, step: usize, use_tab: bool, sort_keys: bool) {
    match v {
        Value::Null => out.push_str("null"),
        Value::False => out.push_str("false"),
        Value::True => out.push_str("true"),
        Value::Num(n, repr) => {
            if let Some(r) = repr {
                out.push_str(r);
            } else {
                push_jq_number(out, *n);
            }
        }
        Value::Str(s) => push_json_string(out, s),
        Value::Error(e) => push_json_string(out, e),
        Value::Arr(a) if a.is_empty() => out.push_str("[]"),
        Value::Obj(o) if o.is_empty() => out.push_str("{}"),
        Value::Arr(a) => {
            let inner_depth = depth + step;
            out.push_str("[\n");
            for (i, item) in a.iter().enumerate() {
                if i > 0 { out.push_str(",\n"); }
                push_indent(out, inner_depth, use_tab);
                write_pretty_to_string(out, item, inner_depth, step, use_tab, sort_keys);
            }
            out.push('\n');
            push_indent(out, depth, use_tab);
            out.push(']');
        }
        Value::Obj(o) => {
            let inner_depth = depth + step;
            out.push_str("{\n");
            if sort_keys {
                let mut entries: Vec<_> = o.iter().collect();
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                for (i, (k, val)) in entries.iter().enumerate() {
                    if i > 0 { out.push_str(",\n"); }
                    push_indent(out, inner_depth, use_tab);
                    push_json_string(out, k);
                    out.push_str(": ");
                    write_pretty_to_string(out, val, inner_depth, step, use_tab, sort_keys);
                }
            } else {
                for (i, (k, val)) in o.iter().enumerate() {
                    if i > 0 { out.push_str(",\n"); }
                    push_indent(out, inner_depth, use_tab);
                    push_json_string(out, k);
                    out.push_str(": ");
                    write_pretty_to_string(out, val, inner_depth, step, use_tab, sort_keys);
                }
            }
            out.push('\n');
            push_indent(out, depth, use_tab);
            out.push('}');
        }
    }
}

// ============================================================================
// Streaming JSON output — writes directly to io::Write, avoiding intermediate Strings
// ============================================================================

use std::io;

fn write_json_string(w: &mut dyn io::Write, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    // Find first byte needing escape
    let first_esc = bytes.iter().position(|&b| b == b'"' || b == b'\\' || b < 0x20);
    if first_esc.is_none() {
        // No escapes: write "string" in minimal calls
        if bytes.len() <= 126 {
            let mut buf = [0u8; 128];
            buf[0] = b'"';
            buf[1..1 + bytes.len()].copy_from_slice(bytes);
            buf[1 + bytes.len()] = b'"';
            return w.write_all(&buf[..bytes.len() + 2]);
        } else {
            w.write_all(b"\"")?;
            w.write_all(bytes)?;
            return w.write_all(b"\"");
        }
    }
    // Has escapes: write prefix then scan remainder
    let first = first_esc.unwrap();
    w.write_all(b"\"")?;
    if first > 0 { w.write_all(&bytes[..first])?; }
    let mut start = first;
    for (i, &b) in bytes[first..].iter().enumerate() {
        let i = i + first;
        let escape = match b {
            b'"' => b"\\\"",
            b'\\' => b"\\\\",
            b'\n' => b"\\n",
            b'\r' => b"\\r",
            b'\t' => b"\\t",
            0x08 => b"\\b",
            0x0c => b"\\f",
            c if c < 0x20 => {
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
    w.write_all(b"\"")
}

fn write_jq_number(w: &mut dyn io::Write, n: f64) -> io::Result<()> {
    if n.is_nan() { return w.write_all(b"null"); }
    if n.is_infinite() {
        return if n.is_sign_positive() { w.write_all(b"1.7976931348623157e+308") }
        else { w.write_all(b"-1.7976931348623157e+308") };
    }
    if n == 0.0 {
        return if n.is_sign_negative() { w.write_all(b"-0") } else { w.write_all(b"0") };
    }
    // Integer fast path
    if n == n.trunc() && n.abs() < 1e16 {
        let i = n as i64;
        if i as f64 == n {
            let mut buf = itoa::Buffer::new();
            return w.write_all(buf.format(i).as_bytes());
        }
    }
    let abs = n.abs();
    // Very small or very large numbers need special formatting
    if (abs != 0.0 && abs < 1e-4) || abs >= 1e16 {
        return w.write_all(format_jq_number(n).as_bytes());
    }
    // Common case: write directly without String allocation
    write!(w, "{}", n)
}

// Reusable String buffer for pretty-print output.
struct PrettyBuf(std::cell::UnsafeCell<String>);
unsafe impl Sync for PrettyBuf {}
static PRETTY_BUF: PrettyBuf = PrettyBuf(std::cell::UnsafeCell::new(String::new()));

/// Write a Value as pretty JSON + newline, directly to writer.
/// For scalar values and small containers, uses stack buffer to avoid String allocation.
pub fn write_value_pretty_line(w: &mut dyn io::Write, v: &Value, indent: usize, use_tab: bool, sort_keys: bool) -> io::Result<()> {
    match v {
        Value::Null | Value::True | Value::False | Value::Num(..) | Value::Str(_) | Value::Error(_) => {
            // Scalar values: pretty == compact
            let mut buf = [0u8; 512];
            if let Some(n) = write_compact_to_buf(v, &mut buf) {
                buf[n] = b'\n';
                return w.write_all(&buf[..n + 1]);
            }
            // Fallback for long strings needing escapes
            write_value_compact_ext_inner(w, v, false)?;
            w.write_all(b"\n")
        }
        Value::Arr(a) if a.is_empty() => w.write_all(b"[]\n"),
        Value::Obj(o) if o.is_empty() => w.write_all(b"{}\n"),
        _ => {
            // Reuse a global String buffer to avoid per-value allocation
            let out = unsafe { &mut *PRETTY_BUF.0.get() };
            out.clear();
            write_pretty_to_string(out, v, 0, indent, use_tab, sort_keys);
            w.write_all(out.as_bytes())?;
            w.write_all(b"\n")
        }
    }
}

/// Write a Value as compact JSON directly to an io::Write, using precise repr when available.
pub fn write_value_compact(w: &mut dyn io::Write, v: &Value) -> io::Result<()> {
    write_value_compact_ext(w, v, false)
}

/// Write a Value as compact JSON with optional key sorting.
/// Uses a stack buffer for small values to minimize write_all calls.
pub fn write_value_compact_ext(w: &mut dyn io::Write, v: &Value, sort_keys: bool) -> io::Result<()> {
    if !sort_keys {
        // Try fast path: build in stack buffer, single write
        let mut buf = [0u8; 512];
        if let Some(n) = write_compact_to_buf(v, &mut buf) {
            return w.write_all(&buf[..n]);
        }
    }
    write_value_compact_ext_inner(w, v, sort_keys)
}

/// Write compact JSON + newline in a single operation.
/// Fuses the JSON serialization and newline into one write_all call when possible.
pub fn write_value_compact_line(w: &mut dyn io::Write, v: &Value, sort_keys: bool) -> io::Result<()> {
    if !sort_keys {
        let mut buf = [0u8; 512];
        if let Some(n) = write_compact_to_buf(v, &mut buf) {
            buf[n] = b'\n';
            return w.write_all(&buf[..n + 1]);
        }
    }
    write_value_compact_ext_inner(w, v, sort_keys)?;
    w.write_all(b"\n")
}

/// Push compact JSON + newline directly into a Vec<u8> buffer.
/// Avoids per-value write_all calls — caller flushes the buffer periodically.
pub fn push_compact_line(buf: &mut Vec<u8>, v: &Value) {
    push_compact_value(buf, v);
    buf.push(b'\n');
}

/// Push pretty-printed JSON + newline directly into a Vec<u8> buffer.
/// Mirrors push_compact_line but with indentation. Avoids per-value write_all overhead.
pub fn push_pretty_line(buf: &mut Vec<u8>, v: &Value, indent: usize, use_tab: bool) {
    push_pretty_value(buf, v, 0, indent, use_tab);
    buf.push(b'\n');
}

fn push_pretty_value(buf: &mut Vec<u8>, v: &Value, depth: usize, step: usize, use_tab: bool) {
    match v {
        Value::Null => buf.extend_from_slice(b"null"),
        Value::False => buf.extend_from_slice(b"false"),
        Value::True => buf.extend_from_slice(b"true"),
        Value::Num(n, repr) => {
            if let Some(r) = repr {
                buf.extend_from_slice(r.as_bytes());
            } else {
                push_jq_number_bytes(buf, *n);
            }
        }
        Value::Str(s) => push_json_string_to_vec(buf, s.as_str()),
        Value::Error(e) => push_json_string_to_vec(buf, e.as_str()),
        Value::Arr(a) if a.is_empty() => buf.extend_from_slice(b"[]"),
        Value::Obj(o) if o.is_empty() => buf.extend_from_slice(b"{}"),
        Value::Arr(a) => {
            let inner = depth + step;
            buf.extend_from_slice(b"[\n");
            for (i, item) in a.iter().enumerate() {
                if i > 0 { buf.extend_from_slice(b",\n"); }
                push_indent_bytes(buf, inner, use_tab);
                push_pretty_value(buf, item, inner, step, use_tab);
            }
            buf.push(b'\n');
            push_indent_bytes(buf, depth, use_tab);
            buf.push(b']');
        }
        Value::Obj(o) => {
            let inner = depth + step;
            buf.extend_from_slice(b"{\n");
            for (i, (k, val)) in o.iter().enumerate() {
                if i > 0 { buf.extend_from_slice(b",\n"); }
                push_indent_bytes(buf, inner, use_tab);
                push_json_string_to_vec(buf, k.as_str());
                buf.extend_from_slice(b": ");
                push_pretty_value(buf, val, inner, step, use_tab);
            }
            buf.push(b'\n');
            push_indent_bytes(buf, depth, use_tab);
            buf.push(b'}');
        }
    }
}

#[inline]
fn push_indent_bytes(buf: &mut Vec<u8>, n: usize, use_tab: bool) {
    if n == 0 { return; }
    let src = if use_tab { TABS_64.as_bytes() } else { SPACES_256.as_bytes() };
    if n <= src.len() {
        buf.extend_from_slice(&src[..n]);
    } else {
        let ch = if use_tab { b'\t' } else { b' ' };
        for _ in 0..n { buf.push(ch); }
    }
}

fn push_compact_value(buf: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Null => buf.extend_from_slice(b"null"),
        Value::False => buf.extend_from_slice(b"false"),
        Value::True => buf.extend_from_slice(b"true"),
        Value::Num(n, repr) => {
            if let Some(r) = repr {
                buf.extend_from_slice(r.as_bytes());
            } else {
                push_jq_number_bytes(buf, *n);
            }
        }
        Value::Str(s) => {
            push_json_string_to_vec(buf, s.as_str());
        }
        Value::Arr(a) => {
            buf.push(b'[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 { buf.push(b','); }
                push_compact_value(buf, item);
            }
            buf.push(b']');
        }
        Value::Obj(o) => {
            buf.push(b'{');
            for (i, (k, val)) in o.iter().enumerate() {
                let kb = k.as_bytes();
                let klen = kb.len();
                let key_needs_escape = kb.iter().any(|&b| b == b'"' || b == b'\\' || b < 0x20);
                if !key_needs_escape {
                    // Write ,"key": (or "key": for first) in a single memcpy
                    let prefix = if i > 0 { 1 } else { 0 }; // comma
                    buf.reserve(prefix + klen + 3); // [,]"key":
                    unsafe {
                        let dst = buf.as_mut_ptr().add(buf.len());
                        if i > 0 { *dst = b','; }
                        *dst.add(prefix) = b'"';
                        std::ptr::copy_nonoverlapping(kb.as_ptr(), dst.add(prefix + 1), klen);
                        *dst.add(prefix + 1 + klen) = b'"';
                        *dst.add(prefix + 2 + klen) = b':';
                        buf.set_len(buf.len() + prefix + klen + 3);
                    }
                } else {
                    if i > 0 { buf.push(b','); }
                    push_json_string_to_vec(buf, k.as_str());
                    buf.push(b':');
                }
                push_compact_value(buf, val);
            }
            buf.push(b'}');
        }
        Value::Error(e) => {
            push_json_string_to_vec(buf, e.as_str());
        }
    }
}

/// Write a jq-formatted number directly to a Vec<u8> buffer, avoiding intermediate String allocation.
#[inline]
pub fn push_jq_number_bytes(buf: &mut Vec<u8>, n: f64) {
    use std::io::Write;
    // Fast path: exact integer in displayable range (covers vast majority of JSON numbers).
    // The i64 roundtrip check (i as f64 == n) naturally rejects NaN, infinity, and non-integers.
    let i = n as i64;
    if i as f64 == n && i.unsigned_abs() < 10_000_000_000_000_000 {
        if i == 0 && n.is_sign_negative() {
            buf.extend_from_slice(b"-0");
        } else {
            let mut ibuf = itoa::Buffer::new();
            buf.extend_from_slice(ibuf.format(i).as_bytes());
        }
        return;
    }
    // Slow path: NaN, infinity, decimals, very large numbers
    if n.is_nan() { buf.extend_from_slice(b"null"); return; }
    if n.is_infinite() {
        if n.is_sign_positive() { buf.extend_from_slice(b"1.7976931348623157e+308"); }
        else { buf.extend_from_slice(b"-1.7976931348623157e+308"); }
        return;
    }
    let abs = n.abs();
    if (abs != 0.0 && abs < 1e-4) || abs >= 1e16 {
        let s = format_jq_number(n);
        buf.extend_from_slice(s.as_bytes());
        return;
    }
    // Common decimal case: write Display format directly to buffer
    let _ = write!(buf, "{}", n);
}

#[inline]
fn push_json_string_to_vec(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let len = bytes.len();
    // Ultra-fast path for single-byte strings (common object keys like "x", "y")
    if len == 1 {
        let c = bytes[0];
        if c >= 0x20 && c != b'"' && c != b'\\' {
            buf.reserve(3);
            unsafe {
                let dst = buf.as_mut_ptr().add(buf.len());
                *dst = b'"';
                *dst.add(1) = c;
                *dst.add(2) = b'"';
                buf.set_len(buf.len() + 3);
            }
            return;
        }
    }
    // Fast path: no escapes needed — write "str" in one contiguous block
    let needs_escape = bytes.iter().any(|&b| b == b'"' || b == b'\\' || b < 0x20);
    if !needs_escape {
        buf.reserve(len + 2);
        // Safety: we just reserved enough space
        unsafe {
            let dst = buf.as_mut_ptr().add(buf.len());
            *dst = b'"';
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.add(1), len);
            *dst.add(1 + len) = b'"';
            buf.set_len(buf.len() + len + 2);
        }
        return;
    }
    // Slow path: has escapes — single-pass scan+copy
    buf.reserve(len + 2);
    buf.push(b'"');
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        let esc = match b {
            b'"' | b'\\' => b,
            b'\n' => b'n',
            b'\r' => b'r',
            b'\t' => b't',
            0x08 => b'b',
            0x0c => b'f',
            c if c < 0x20 => {
                if start < i { buf.extend_from_slice(&bytes[start..i]); }
                let hex = [
                    b'\\', b'u', b'0', b'0',
                    b"0123456789abcdef"[(c >> 4) as usize],
                    b"0123456789abcdef"[(c & 0xf) as usize],
                ];
                buf.extend_from_slice(&hex);
                start = i + 1;
                continue;
            }
            _ => continue,
        };
        if start < i { buf.extend_from_slice(&bytes[start..i]); }
        buf.extend_from_slice(&[b'\\', esc]);
        start = i + 1;
    }
    if start < len { buf.extend_from_slice(&bytes[start..]); }
    buf.push(b'"');
}

/// Try to write compact JSON to a fixed buffer. Returns Some(len) on success, None if buffer too small.
fn write_compact_to_buf(v: &Value, buf: &mut [u8]) -> Option<usize> {
    let mut pos = 0;
    write_compact_buf_inner(v, buf, &mut pos).then_some(pos)
}

fn write_compact_buf_inner(v: &Value, buf: &mut [u8], pos: &mut usize) -> bool {
    macro_rules! push {
        ($bytes:expr) => {{
            let b: &[u8] = $bytes;
            if *pos + b.len() > buf.len() { return false; }
            buf[*pos..*pos + b.len()].copy_from_slice(b);
            *pos += b.len();
        }};
    }
    match v {
        Value::Null => push!(b"null"),
        Value::False => push!(b"false"),
        Value::True => push!(b"true"),
        Value::Num(n, repr) => {
            if let Some(r) = repr {
                push!(r.as_bytes());
            } else if *n == n.trunc() && n.abs() < 1e15 {
                let i = *n as i64;
                if i as f64 == *n {
                    let mut ibuf = itoa::Buffer::new();
                    push!(ibuf.format(i).as_bytes());
                } else {
                    let s = format_jq_number(*n);
                    push!(s.as_bytes());
                }
            } else {
                let s = format_jq_number(*n);
                push!(s.as_bytes());
            }
        }
        Value::Str(s) => {
            let bytes = s.as_bytes();
            let needs_escape = bytes.iter().any(|&b| b == b'"' || b == b'\\' || b < 0x20);
            if needs_escape { return false; } // fall back to slow path
            if *pos + bytes.len() + 2 > buf.len() { return false; }
            buf[*pos] = b'"';
            *pos += 1;
            buf[*pos..*pos + bytes.len()].copy_from_slice(bytes);
            *pos += bytes.len();
            buf[*pos] = b'"';
            *pos += 1;
        }
        Value::Arr(a) => {
            push!(b"[");
            for (i, item) in a.iter().enumerate() {
                if i > 0 { push!(b","); }
                if !write_compact_buf_inner(item, buf, pos) { return false; }
            }
            push!(b"]");
        }
        Value::Obj(o) => {
            push!(b"{");
            for (i, (k, val)) in o.iter().enumerate() {
                if i > 0 { push!(b","); }
                // Write key
                let kb = k.as_bytes();
                let key_escape = kb.iter().any(|&b| b == b'"' || b == b'\\' || b < 0x20);
                if key_escape { return false; }
                if *pos + kb.len() + 3 > buf.len() { return false; } // "key":
                buf[*pos] = b'"';
                *pos += 1;
                buf[*pos..*pos + kb.len()].copy_from_slice(kb);
                *pos += kb.len();
                buf[*pos] = b'"';
                *pos += 1;
                buf[*pos] = b':';
                *pos += 1;
                if !write_compact_buf_inner(val, buf, pos) { return false; }
            }
            push!(b"}");
        }
        Value::Error(e) => {
            let bytes = e.as_bytes();
            if *pos + bytes.len() + 2 > buf.len() { return false; }
            buf[*pos] = b'"';
            *pos += 1;
            buf[*pos..*pos + bytes.len()].copy_from_slice(bytes);
            *pos += bytes.len();
            buf[*pos] = b'"';
            *pos += 1;
        }
    }
    true
}

fn write_value_compact_ext_inner(w: &mut dyn io::Write, v: &Value, sort_keys: bool) -> io::Result<()> {
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
                write_value_compact_ext(w, item, sort_keys)?;
            }
            w.write_all(b"]")
        }
        Value::Obj(o) => {
            w.write_all(b"{")?;
            if sort_keys {
                let mut entries: Vec<_> = o.iter().collect();
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                for (i, (k, val)) in entries.iter().enumerate() {
                    if i > 0 { w.write_all(b",")?; }
                    write_json_string(w, k)?;
                    w.write_all(b":")?;
                    write_value_compact_ext(w, val, sort_keys)?;
                }
            } else {
                for (i, (k, val)) in o.iter().enumerate() {
                    if i > 0 { w.write_all(b",")?; }
                    write_json_string(w, k)?;
                    w.write_all(b":")?;
                    write_value_compact_ext(w, val, sort_keys)?;
                }
            }
            w.write_all(b"}")
        }
        Value::Error(e) => write_json_string(w, e),
    }
}
