//! Value type for JIT-compiled jq filters.
//!
//! Uses Vec-backed ordered map for objects (optimal for typical small JSON objects).

use std::cell::{RefCell, UnsafeCell};
use std::fmt;
use std::rc::Rc;

use anyhow::{Result, bail};

/// Inline-optimized string for object keys (≤24 bytes stored on stack, no heap alloc).
pub type KeyStr = compact_str::CompactString;

// Per-thread pool of Vec buffers for ObjMap reuse. Each thread keeps its own
// pool — `cargo test` may run multiple test threads concurrently, and a shared
// global pool was racy (the previous implementation hid this behind
// `--test-threads=1` in CI). Thread-locality also matches the runtime's
// `Rc` / `!Send` discipline.
//
// The bare `UnsafeCell` (vs `RefCell`) is a deliberate hot-path optimisation:
// pool ops are millions-per-bench and `RefCell::borrow_mut`'s flag check
// shows up as a measurable regression on `field_access` / `array construct`.
// Safety: each thread has its own cell, and the access patterns inside
// `pool_get` / `pool_return` / `rc_objmap_pool_*` never re-enter the same
// pool while a `&mut` is live (drops in `pool_return` only touch the
// `RC_OBJMAP_POOL` via `pool_value`, which is a different cell).
thread_local! {
    static OBJMAP_POOL: UnsafeCell<Vec<Vec<(KeyStr, Value)>>> = const { UnsafeCell::new(Vec::new()) };
}

const OBJMAP_POOL_MAX: usize = 64;

// Per-thread pool of Rc<ObjMap> to avoid repeated Rc alloc/dealloc.
// When an Rc<ObjMap> with refcount=1 is dropped, we clear entries and pool the Rc.
// On next alloc, we reuse the Rc memory instead of calling malloc.
thread_local! {
    static RC_OBJMAP_POOL: UnsafeCell<Vec<*const ObjMap>> = const { UnsafeCell::new(Vec::new()) };
}

const RC_OBJMAP_POOL_MAX: usize = 32;

/// Try to recycle an Rc<ObjMap> instead of allocating a new one.
#[inline]
pub fn rc_objmap_pool_get(cap: usize) -> Rc<ObjMap> {
    let popped = RC_OBJMAP_POOL.with(|cell| unsafe { (*cell.get()).pop() });
    if let Some(raw) = popped {
        // Safety: raw was produced by Rc::into_raw and has refcount=1
        let mut rc = unsafe { Rc::from_raw(raw) };
        let map = Rc::get_mut(&mut rc).unwrap();
        map.entries = pool_get(cap);
        map.index = None;
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
    if RC_OBJMAP_POOL.with(|cell| unsafe { (*cell.get()).len() }) >= RC_OBJMAP_POOL_MAX {
        return false;
    }
    // Clear entries and pool the Vec buffer
    let map = Rc::get_mut(&mut rc).unwrap();
    let old_entries = std::mem::take(&mut map.entries);
    pool_return(old_entries);
    // Pool the Rc itself (prevents deallocation)
    let raw = Rc::into_raw(rc);
    RC_OBJMAP_POOL.with(|cell| unsafe { (*cell.get()).push(raw) });
    true
}

#[inline]
fn pool_get(cap: usize) -> Vec<(KeyStr, Value)> {
    OBJMAP_POOL.with(|cell| {
        let pool = unsafe { &mut *cell.get() };
        if let Some(v) = pool.pop() {
            if v.capacity() >= cap {
                return v;
            }
        }
        Vec::with_capacity(cap)
    })
}

#[inline]
fn pool_return(mut v: Vec<(KeyStr, Value)>) {
    // Fast path: if all entries are trivially droppable (inline strings, simple numbers),
    // skip the per-entry destructor calls and just zero the length.
    let trivial = v.iter().all(|(k, val)| {
        !k.is_heap_allocated() && match val {
            Value::Null | Value::True | Value::False => true,
            Value::Num(_, NumRepr(repr)) => repr.is_none(),
            Value::Str(s) => !s.is_heap_allocated(),
            _ => false,
        }
    });
    if trivial {
        unsafe { v.set_len(0); }
    } else {
        v.clear();
    }
    // try_with: during process shutdown the TLS may already be destroyed
    // (e.g. an env-derived Rc<ObjMap> dropping after main returns). On
    // AccessError, fall through and let `v` drop normally.
    let _ = OBJMAP_POOL.try_with(|cell| {
        let pool = unsafe { &mut *cell.get() };
        if pool.len() < OBJMAP_POOL_MAX {
            pool.push(v);
        }
    });
}

/// Recycle Rc<ObjMap> from a consumed Value to avoid repeated alloc/dealloc.
/// Call this instead of letting a Value drop when you know it won't be used again.
#[inline]
pub fn pool_value(v: Value) {
    if let Value::Obj(ObjInner(rc)) = v {
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
    /// Lazily-built hash index for O(1) key lookup on large objects.
    index: Option<Box<std::collections::HashMap<KeyStr, usize>>>,
}

/// Threshold: build hash index when entries exceed this size.
const OBJMAP_INDEX_THRESHOLD: usize = 32;

impl Clone for ObjMap {
    fn clone(&self) -> Self {
        let mut entries = pool_get(self.entries.len());
        entries.extend(self.entries.iter().cloned());
        ObjMap { entries, index: self.index.clone() }
    }
}

impl Drop for ObjMap {
    fn drop(&mut self) {
        self.index = None;
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
        ObjMap { entries: Vec::new(), index: None }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        ObjMap { entries: pool_get(cap), index: None }
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
        // Use hash index for O(1) lookup on large objects
        if let Some(ref mut idx) = self.index {
            if let Some(&pos) = idx.get(&key) {
                let old = std::mem::replace(&mut self.entries[pos].1, value);
                return Some(old);
            }
            let pos = self.entries.len();
            idx.insert(key.clone(), pos);
            self.entries.push((key, value));
            return None;
        }
        // Linear scan for small objects
        for entry in &mut self.entries {
            if entry.0 == key {
                let old = std::mem::replace(&mut entry.1, value);
                return Some(old);
            }
        }
        self.entries.push((key, value));
        // Build hash index when crossing threshold
        if self.entries.len() == OBJMAP_INDEX_THRESHOLD {
            self.build_index();
        }
        None
    }

    /// Push a key-value pair without checking for duplicates.
    /// Caller must ensure the key does not already exist (e.g. during JSON parsing).
    #[inline]
    pub fn push_unique(&mut self, key: KeyStr, value: Value) {
        if let Some(ref mut idx) = self.index {
            idx.insert(key.clone(), self.entries.len());
        }
        self.entries.push((key, value));
    }

    /// Get a value by key. Uses hash index on large objects for O(1) lookup.
    #[inline]
    pub fn get(&self, key: &str) -> Option<&Value> {
        if let Some(ref idx) = self.index {
            return idx.get(key).map(|&pos| &self.entries[pos].1);
        }
        for (k, v) in &self.entries {
            if k.as_str() == key {
                return Some(v);
            }
        }
        None
    }

    /// Get a mutable reference to a value by key.
    #[inline]
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        if let Some(ref idx) = self.index {
            return idx.get(key).map(|&pos| &mut self.entries[pos].1);
        }
        for (k, v) in &mut self.entries {
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

    /// Get a mutable reference to value at index (O(1) direct Vec access).
    #[inline]
    pub fn get_value_mut_by_index(&mut self, index: usize) -> Option<&mut Value> {
        self.entries.get_mut(index).map(|(_, v)| v)
    }

    /// Remove a key, shifting subsequent entries to preserve order.
    pub fn shift_remove(&mut self, key: &str) -> Option<Value> {
        let pos = if let Some(ref idx) = self.index {
            idx.get(key).copied()
        } else {
            self.entries.iter().position(|(k, _)| k.as_str() == key)
        };
        if let Some(pos) = pos {
            let val = self.entries.remove(pos).1;
            // Invalidate index — cheaper to rebuild lazily than update all positions
            self.index = None;
            if self.entries.len() >= OBJMAP_INDEX_THRESHOLD {
                self.build_index();
            }
            Some(val)
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
        if let Some(ref idx) = self.index {
            return idx.contains_key(key);
        }
        self.entries.iter().any(|(k, _)| k.as_str() == key)
    }

    /// Build hash index from current entries.
    fn build_index(&mut self) {
        let mut idx = std::collections::HashMap::with_capacity(self.entries.len());
        for (i, (k, _)) in self.entries.iter().enumerate() {
            idx.insert(k.clone(), i);
        }
        self.index = Some(Box::new(idx));
    }
}

impl std::iter::FromIterator<(KeyStr, Value)> for ObjMap {
    fn from_iter<I: IntoIterator<Item = (KeyStr, Value)>>(iter: I) -> Self {
        ObjMap { entries: iter.into_iter().collect(), index: None }
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

/// Opaque wrapper around the optional source-repr `Rc<str>` of `Value::Num`.
///
/// The inner field is `pub(crate)`, so external crates can name the type
/// (needed to write the `Value::Num(_, _)` pattern) but cannot construct it
/// nor access its inner Option. This routes external `Value::Num` construction
/// through `Value::number` / `Value::number_with_repr` / `Value::number_opt` /
/// `Value::from_f64`, which preserve the f64↔repr invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumRepr(pub(crate) Option<Rc<str>>);

impl std::ops::Deref for NumRepr {
    type Target = Option<Rc<str>>;
    #[inline]
    fn deref(&self) -> &Self::Target { &self.0 }
}

/// Opaque wrapper around the `Rc<ObjMap>` of `Value::Obj`.
///
/// The inner field is `pub(crate)`, so external crates can name the type
/// (needed to write the `Value::Obj(_)` pattern) but cannot construct it nor
/// access its inner `Rc<ObjMap>`. This routes external `Value::Obj`
/// construction through `Value::object_from_pairs` /
/// `Value::object_from_normalized_pairs` / `Value::object_from_map` /
/// `Value::from_pairs`, which fold the dedup invariant into construction
/// instead of leaving it to `ObjMap::push_unique` callers.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjInner(pub(crate) Rc<ObjMap>);

impl std::ops::Deref for ObjInner {
    type Target = Rc<ObjMap>;
    #[inline]
    fn deref(&self) -> &Self::Target { &self.0 }
}

/// A jq value.
pub enum Value {
    Null,
    False,
    True,
    /// Numeric value. Optional Rc<str> preserves original representation for precision.
    Num(f64, NumRepr),
    Str(KeyStr),
    Arr(Rc<Vec<Value>>),
    Obj(ObjInner),
    Error(Rc<String>),
}

impl Clone for Value {
    fn clone(&self) -> Self {
        match self {
            Value::Null => Value::Null,
            Value::False => Value::False,
            Value::True => Value::True,
            Value::Num(n, NumRepr(repr)) => Value::number_opt(*n, repr.clone()),
            Value::Str(s) => Value::Str(s.clone()),
            Value::Arr(a) => Value::Arr(Rc::clone(a)),
            Value::Obj(ObjInner(o)) => Value::Obj(ObjInner(Rc::clone(o))),
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
            (Value::Obj(ObjInner(a)), Value::Obj(ObjInner(b))) => a == b,
            _ => false,
        }
    }
}

impl Value {
    pub fn from_f64(n: f64) -> Self {
        Value::Num(n, NumRepr(None))
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
        Value::Obj(ObjInner(Rc::new(pairs.into_iter().map(|(k, v)| (KeyStr::from(k), v)).collect())))
    }

    /// Canonical object factory: dedupes duplicate keys (last value wins,
    /// earliest position preserved) to match jq's `{a:1, a:2}` → `{"a":2}`
    /// object-literal semantics. Every Obj construction that builds a pair
    /// list from user-controlled / fast-path sources should route through
    /// this function so the dedup invariant is enforced structurally instead
    /// of by a per-site "remember to call the helper" convention. See
    /// `docs/maintenance.md` §3.
    pub fn object_from_pairs<K, I>(pairs: I) -> Self
    where
        K: Into<KeyStr>,
        I: IntoIterator<Item = (K, Value)>,
    {
        let iter = pairs.into_iter();
        let (lo, _) = iter.size_hint();
        let mut map = ObjMap::with_capacity(lo);
        for (k, v) in iter {
            map.insert(k.into(), v);
        }
        Value::Obj(ObjInner(Rc::new(map)))
    }

    /// Bypass variant of [`Value::object_from_pairs`]: trusts the caller to
    /// provide an already-normalized pair list (no duplicate keys, order
    /// reflects insertion order). Prefer the normalizing variant unless the
    /// source is provably unique — parser output, a clone of an existing
    /// `Rc<ObjMap>`, or a rebuild of a structure that went through dedup
    /// earlier in the same pipeline. In debug builds this asserts the
    /// caller's contract.
    pub fn object_from_normalized_pairs<K, I>(pairs: I) -> Self
    where
        K: Into<KeyStr>,
        I: IntoIterator<Item = (K, Value)>,
    {
        let iter = pairs.into_iter();
        let (lo, _) = iter.size_hint();
        let mut map = ObjMap::with_capacity(lo);
        for (k, v) in iter {
            let key: KeyStr = k.into();
            debug_assert!(
                !map.contains_key(key.as_str()),
                "object_from_normalized_pairs: duplicate key `{}`", key
            );
            map.push_unique(key, v);
        }
        Value::Obj(ObjInner(Rc::new(map)))
    }

    /// Wrap an existing `ObjMap` that was built with invariant-preserving
    /// operations (`insert` / `push_unique` / JSON parsing). Reuses the
    /// caller's allocation — does not dedupe.
    pub fn object_from_map(map: ObjMap) -> Self {
        Value::Obj(ObjInner(Rc::new(map)))
    }

    /// Numeric factory that drops the repr annotation. Equivalent to
    /// [`Value::from_f64`], named consistently with [`Value::object_from_pairs`].
    #[inline]
    pub fn number(n: f64) -> Self {
        Value::Num(n, NumRepr(None))
    }

    /// Numeric factory that preserves the original textual representation.
    /// Callers that read a number from source (parser, literal fold,
    /// identity-preserving round-trips) should prefer this so `1.0` stays
    /// `1.0` and `-1.0` stays `-1.0`. Operations that produce a fresh number
    /// value (arithmetic, length, etc.) should use [`Value::number`] so the
    /// repr does not carry over misleadingly.
    #[inline]
    pub fn number_with_repr(n: f64, repr: Rc<str>) -> Self {
        Value::Num(n, NumRepr(Some(repr)))
    }

    /// Numeric factory that takes an already-built repr option. Convenience
    /// wrapper for sites that hold a `Option<Rc<str>>` (cloned from another
    /// number, carried through a pipeline) and want to avoid pattern-matching
    /// on the option before dispatching to [`Value::number`] or
    /// [`Value::number_with_repr`].
    #[inline]
    pub fn number_opt(n: f64, repr: Option<Rc<str>>) -> Self {
        Value::Num(n, NumRepr(repr))
    }

    /// Flip the sign of a numeric repr, preserving the original textual form
    /// (so `1.0` ↔ `-1.0`, `1e10` ↔ `-1e10`). Returns `None` if the input is
    /// `None` or if the repr is not a JSON-valid number.
    #[inline]
    pub fn negate_repr(repr: Option<Rc<str>>) -> Option<Rc<str>> {
        let r = repr?;
        if !is_valid_json_number(&r) { return None; }
        let s: &str = &r;
        // jq normalises negative zero to positive in the literal/eval path,
        // so the negation must drop (not flip) the leading sign for any
        // zero-valued repr. `-0.0` literal → `0.0`, `0.0 * -1` → `0.0`.
        if r.parse::<f64>().map(|n| n == 0.0).unwrap_or(false) {
            return Some(s.strip_prefix('-').map(Rc::from).unwrap_or_else(|| r.clone()));
        }
        if let Some(rest) = s.strip_prefix('-') {
            Some(Rc::from(rest))
        } else {
            let mut out = String::with_capacity(s.len() + 1);
            out.push('-');
            out.push_str(s);
            Some(Rc::from(out.as_str()))
        }
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
            Value::Obj(ObjInner(o)) => Some(o),
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
            Value::Null => Ok(Value::number(0.0)),
            Value::True | Value::False => {
                bail!("{} ({}) has no length", self.type_name(), crate::value::value_to_json(self))
            }
            Value::Num(n, NumRepr(repr)) => {
                if *n >= 0.0 { Ok(Value::number_opt(*n, repr.clone())) }
                else { Ok(Value::number(n.abs())) }
            }
            Value::Str(s) => {
                // jq counts Unicode codepoints
                Ok(Value::number(s.chars().count() as f64))
            }
            Value::Arr(a) => Ok(Value::number(a.len() as f64)),
            Value::Obj(ObjInner(o)) => Ok(Value::number(o.len() as f64)),
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
            Value::Obj(ObjInner(o)) => write!(f, "Obj({:?})", o.as_ref()),
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

/// Cheap "no remaining duplicate" check used inside scan loops *after*
/// every target key has been seen at least once. Returns `true` when
/// the first byte of every target field is absent from `&b[start..]`,
/// which proves no duplicate of those keys exists in the rest of the
/// buffer (every duplicate of a key starting with byte `c` requires
/// `c` to appear at least once more in the bytes). False positives —
/// `c` inside a string value or shared with another key's first byte —
/// only route to the slow scan-to-end path, never to incorrect
/// early-exit.
///
/// The original #410 patch did this check **upfront** on the entire
/// buffer (`obj_no_dup_target_keys`) before scanning. That paid
/// `2×fields.len()` `memchr_iter` startups per record even when no
/// early-exit savings were possible — namely when the last requested
/// key sits at the end of the row, so the scan had to reach the end
/// anyway. The in-loop variant defers the check until after all keys
/// are matched and only scans the trailing bytes (`b.len() - start`),
/// dropping the per-row overhead from ~20ns to a couple of ns on
/// NDJSON shapes like `{"x":…,"y":…,"name":…}` reading `name+x`.
/// Fixes the v1.4.5 string-interpolation / `@csv` / `to_entries`
/// regressions (#422) without giving back the v1.4.5 wins on
/// single-field readers and `.x + .y`-shaped two-field readers.
///
/// Only `pos == 0` callers participate. Per-record buffers from
/// `json_stream_raw` keep `b[start..]` bounded by the current row's
/// trailing whitespace; `pos != 0` (nested) call sites would scan
/// into unrelated sibling data, so they keep the scan-to-end
/// behavior — correctness equivalent, no perf regression vs v1.4.4.
///
/// Used by the hot read helpers (`json_object_get_num`,
/// `json_object_get_two_nums`, `json_object_get_field_raw`,
/// `json_object_get_fields_raw_buf`).
#[inline]
fn no_target_first_byte_in_remainder(b: &[u8], start: usize, first_bytes: &[u8]) -> bool {
    if start > b.len() { return false; }
    let rem = &b[start..];
    for &c in first_bytes {
        if memchr::memchr(c, rem).is_some() { return false; }
    }
    true
}

/// Extract a numeric field value from a JSON object without full parsing.
/// Returns Some(f64) if the field exists and is numeric, None otherwise.
/// Used for select fast paths to avoid parsing discarded objects.
pub fn json_object_get_num(b: &[u8], pos: usize, field: &str) -> Option<f64> {
    // jq dedupes duplicate input keys last-wins (#233 / #325 / #360).
    // For correctness in the worst case we have to scan to the end of
    // the object and keep the LAST matching key's value (returning None
    // when that final value isn't numeric, even if an earlier same-key
    // value was). The scan-to-end pays a steep regression for the
    // overwhelmingly common dup-free case (#410). After the first
    // match, we run the cheap `no_target_first_byte_in_remainder`
    // proof on the bytes left in the row and early-exit when the
    // target key cannot reappear. The check is skipped for `pos != 0`
    // callers (nested) where `b[i..]` would extend into unrelated
    // sibling data and for empty field names.
    if pos >= b.len() || b[pos] != b'{' { return None; }
    let field_bytes = field.as_bytes();
    let allow_early_exit = pos == 0 && !field_bytes.is_empty();
    let mut early_exit_attempted = false;
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { return None; }
    let mut last_match: Option<f64> = None;
    let mut last_was_match: bool = false;
    loop {
        if i >= b.len() || b[i] != b'"' { return last_match; }
        let key_start = i + 1;
        let mut j = key_start;
        while j < b.len() {
            match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 }
        }
        let key_matches = (j - key_start) == field_bytes.len()
            && b[key_start..j] == *field_bytes;
        i = j + 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return last_match; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if key_matches {
            // Parse the numeric value inline; record None when the
            // current value isn't numeric so a later non-match doesn't
            // resurrect an earlier numeric value.
            last_was_match = true;
            let mut value: Option<f64> = None;
            if i < b.len() {
                let neg = b[i] == b'-';
                let start = if neg { i + 1 } else { i };
                if start < b.len() && b[start].is_ascii_digit() {
                    let mut n: i64 = (b[start] - b'0') as i64;
                    let mut k = start + 1;
                    while k < b.len() && b[k].is_ascii_digit() {
                        n = n * 10 + (b[k] - b'0') as i64;
                        k += 1;
                    }
                    if k < b.len() && (b[k] == b'.' || b[k] == b'e' || b[k] == b'E') {
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
                        value = fast_float::parse::<f64, _>(num_str).ok();
                    } else if (k - start) <= 15 {
                        value = Some(if neg { -(n as f64) } else { n as f64 });
                    }
                }
            }
            last_match = value;
        }
        // Skip past the value (whether we parsed or not) so we keep scanning.
        i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return if last_was_match { last_match } else { None } };
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return last_match; }
        if b[i] == b'}' { return last_match; }
        if b[i] != b',' { return last_match; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        // After the first match (and only at the top level), once
        // we've consumed the comma and there are more keys ahead,
        // check whether the target key's first byte still appears
        // in the remaining bytes. If not, no duplicate is possible
        // and the first-wins read agrees with last-wins (#410 /
        // #422). Run at most once per call. Skipping when the next
        // byte was `}` keeps the last-key-in-row case (e.g.
        // `.name | startswith(…)` on `{"x":…,"y":…,"name":…}`)
        // free of memchr overhead.
        if allow_early_exit && last_was_match && !early_exit_attempted {
            early_exit_attempted = true;
            if no_target_first_byte_in_remainder(b, i, &[field_bytes[0]]) {
                return last_match;
            }
        }
    }
}

/// Extract the raw byte range of a field value from a JSON object without full parsing.
/// Returns Some((start, end)) byte offsets of the field's value, or None if not found
/// or the input isn't a JSON object. Used for field access fast paths.
pub fn json_object_get_field_raw(b: &[u8], pos: usize, field: &str) -> Option<(usize, usize)> {
    if pos >= b.len() || b[pos] != b'{' { return None; }
    let field_bytes = field.as_bytes();
    // jq dedupes duplicate input keys last-wins (#233 / #325). After
    // the first match (and only at the top level) we run the cheap
    // `no_target_first_byte_in_remainder` check to early-exit when
    // no duplicate is possible (#410 / #422); otherwise we keep
    // scanning to honor last-wins.
    let allow_early_exit = pos == 0 && !field_bytes.is_empty();
    let mut early_exit_attempted = false;
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { return None; }
    let mut last_match: Option<(usize, usize)> = None;
    loop {
        if i >= b.len() || b[i] != b'"' { return last_match; }
        let key_start = i + 1;
        let mut j = key_start;
        while j < b.len() {
            match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 }
        }
        let key_matches = (j - key_start) == field_bytes.len()
            && b[key_start..j] == *field_bytes;
        i = j + 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return last_match; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        let val_start = i;
        let val_end = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return last_match };
        if key_matches {
            last_match = Some((val_start, val_end));
        }
        i = val_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return last_match; }
        if b[i] == b'}' { return last_match; }
        if b[i] != b',' { return last_match; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        // After the first match (and only at the top level), once
        // we've consumed the comma and there are more keys ahead,
        // check whether the target key's first byte still appears
        // in the remaining bytes. If not, no duplicate is possible
        // and the first-wins read agrees with last-wins (#410 /
        // #422). Run at most once per call.
        if allow_early_exit && last_match.is_some() && !early_exit_attempted {
            early_exit_attempted = true;
            if no_target_first_byte_in_remainder(b, i, &[field_bytes[0]]) {
                return last_match;
            }
        }
    }
}

/// Extract a nested field from a JSON object: `.a.b.c` traversal on raw bytes.
/// Returns (start, end) byte offsets of the final value within `b`.
pub fn json_object_get_nested_field_raw(b: &[u8], pos: usize, fields: &[&str]) -> Option<(usize, usize)> {
    if fields.is_empty() { return None; }
    let (mut vs, mut ve) = json_object_get_field_raw(b, pos, fields[0])?;
    for field in &fields[1..] {
        let (s, e) = json_object_get_field_raw(b, vs, field)?;
        vs = s;
        ve = e;
    }
    Some((vs, ve))
}

/// Parse a JSON number from raw bytes. Returns the f64 value or None if not a number.
pub fn parse_json_num(b: &[u8]) -> Option<f64> {
    if b.is_empty() { return None; }
    let neg = b[0] == b'-';
    let start = if neg { 1 } else { 0 };
    if start >= b.len() || !b[start].is_ascii_digit() { return None; }
    let mut n: i64 = (b[start] - b'0') as i64;
    let mut k = start + 1;
    while k < b.len() && b[k].is_ascii_digit() {
        n = n * 10 + (b[k] - b'0') as i64;
        k += 1;
    }
    if k < b.len() && (b[k] == b'.' || b[k] == b'e' || b[k] == b'E') {
        let num_str = unsafe { std::str::from_utf8_unchecked(b) };
        return fast_float::parse::<f64, _>(num_str).ok();
    }
    if k != b.len() { return None; } // trailing garbage
    if (k - start) > 15 { return None; }
    Some(if neg { -(n as f64) } else { n as f64 })
}

/// Count the number of key-value pairs in a JSON object without full parsing.
/// Returns None if the input isn't a JSON object.
pub fn json_object_length(b: &[u8], pos: usize) -> Option<usize> {
    if pos >= b.len() || b[pos] != b'{' { return None; }
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { return Some(0); }
    let mut seen: [(usize, usize); 16] = [(0, 0); 16];
    let mut seen_count: usize = 0;
    let mut count = 0usize;
    loop {
        if i >= b.len() || b[i] != b'"' { return None; }
        let key_start = i;
        i += 1;
        while i < b.len() {
            match b[i] { b'"' => break, b'\\' => i += 2, _ => i += 1 }
        }
        if i >= b.len() { return None; }
        let key_end = i + 1;

        let key_bytes = &b[key_start + 1 .. key_end - 1];
        let mut is_dup = false;
        for j in 0..seen_count {
            let (ks, ke) = seen[j];
            if &b[ks + 1 .. ke - 1] == key_bytes { is_dup = true; break; }
        }
        if is_dup {
            // skip — not counted
        } else if seen_count < seen.len() {
            seen[seen_count] = (key_start, key_end);
            seen_count += 1;
            count += 1;
        } else {
            // Object has more than 16 unique keys; fall back to the
            // allocating dedup helper.
            let mut pairs: Vec<(usize, usize, usize, usize)> = Vec::new();
            json_object_dedup_pairs(b, pos, &mut pairs)?;
            return Some(pairs.len());
        }

        i = key_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return None };
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

/// Walk a raw JSON object collecting `(key_range, value_range)` pairs
/// with last-wins-first-position dedup applied. Each range is a
/// half-open `(start, end)` byte index pair into `b`; key ranges
/// include the surrounding quotes.
///
/// This matches jq 1.8.x's input parse semantics (the same dedup
/// `parse_json_object` enforces in-memory, #233). Raw-byte fast paths
/// over object iteration / keys / length / to_entries skip that
/// in-memory dedup; routing them through this helper keeps the two
/// code paths consistent (#325).
///
/// Returns the byte index past the closing `}` on success, or `None`
/// on a malformed object.
///
/// Key comparison is byte-level *between the surrounding quotes* —
/// equivalent to jq's behaviour for keys without escape sequences.
/// Escape-equivalent keys (e.g. `"a"` vs `"a"`) are not folded;
/// jq does fold them after unescape. That edge case is uncommon in
/// real inputs and not currently surfaced by `tests/fuzz_restricted.rs`.
pub fn json_object_dedup_pairs(
    b: &[u8],
    pos: usize,
    pairs: &mut Vec<(usize, usize, usize, usize)>,
) -> Option<usize> {
    pairs.clear();
    if pos >= b.len() || b[pos] != b'{' { return None; }
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { return Some(i + 1); }
    loop {
        if i >= b.len() || b[i] != b'"' { return None; }
        let key_start = i;
        i += 1;
        while i < b.len() {
            match b[i] { b'"' => break, b'\\' => i += 2, _ => i += 1 }
        }
        if i >= b.len() { return None; }
        let key_end = i + 1;
        i = key_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        let val_start = i;
        i = match skip_json_value(b, i) { Ok(e) => e, Err(_) => return None };
        let val_end = i;

        let key_bytes = &b[key_start + 1 .. key_end - 1];
        let mut found = false;
        for entry in pairs.iter_mut() {
            if &b[entry.0 + 1 .. entry.1 - 1] == key_bytes {
                entry.2 = val_start;
                entry.3 = val_end;
                found = true;
                break;
            }
        }
        if !found {
            pairs.push((key_start, key_end, val_start, val_end));
        }

        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return None; }
        if b[i] == b'}' { return Some(i + 1); }
        if b[i] != b',' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
}

/// Extract all keys from a JSON object and write them as a sorted JSON array into buf.
/// Returns true if successful, false if the input isn't a JSON object.
pub fn json_object_keys_to_buf(b: &[u8], pos: usize, buf: &mut Vec<u8>) -> bool {
    let mut keys: Vec<(usize, usize)> = Vec::new();
    json_object_keys_to_buf_reuse(b, pos, buf, &mut keys)
}

/// Extract sorted keys from a JSON object, reusing a key-range buffer to avoid allocations.
pub fn json_object_keys_to_buf_reuse(b: &[u8], pos: usize, buf: &mut Vec<u8>, keys: &mut Vec<(usize, usize)>) -> bool {
    keys.clear();
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' {
        buf.extend_from_slice(b"[]\n");
        return true;
    }
    let mut seen: [(usize, usize); 16] = [(0, 0); 16];
    let mut seen_count: usize = 0;
    let mut had_dup = false;
    loop {
        if i >= b.len() || b[i] != b'"' { return false; }
        let key_start = i;
        i += 1;
        while i < b.len() {
            match b[i] { b'"' => break, b'\\' => i += 2, _ => i += 1 }
        }
        if i >= b.len() { return false; }
        let key_end = i + 1;

        let key_bytes = &b[key_start + 1 .. key_end - 1];
        let mut is_dup = false;
        for j in 0..seen_count {
            let (ks, ke) = seen[j];
            if &b[ks + 1 .. ke - 1] == key_bytes { is_dup = true; break; }
        }
        if is_dup || seen_count >= seen.len() {
            had_dup = true;
            break;
        }
        seen[seen_count] = (key_start, key_end);
        seen_count += 1;
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
    if had_dup {
        keys.clear();
        let mut pairs: Vec<(usize, usize, usize, usize)> = Vec::new();
        if json_object_dedup_pairs(b, pos, &mut pairs).is_none() { return false; }
        for (ks, ke, _, _) in &pairs { keys.push((*ks, *ke)); }
    }
    if keys.is_empty() {
        buf.extend_from_slice(b"[]\n");
        return true;
    }
    keys.sort_unstable_by(|a, c| b[a.0+1..a.1-1].cmp(&b[c.0+1..c.1-1]));
    buf.push(b'[');
    for (idx, (ks, ke)) in keys.iter().enumerate() {
        if idx > 0 { buf.push(b','); }
        buf.extend_from_slice(&b[*ks..*ke]);
    }
    buf.extend_from_slice(b"]\n");
    true
}

/// Extract unsorted key byte ranges from a JSON object without output.
/// Returns Some(count) if successful, populating `keys` with (start, end) ranges
/// including the surrounding quotes. Returns None on non-object.
pub fn json_object_extract_keys_only(b: &[u8], pos: usize, keys: &mut Vec<(usize, usize)>) -> Option<usize> {
    keys.clear();
    if pos >= b.len() || b[pos] != b'{' { return None; }
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { return Some(0); }
    let mut seen: [(usize, usize); 16] = [(0, 0); 16];
    let mut seen_count: usize = 0;
    let mut had_dup = false;
    loop {
        if i >= b.len() || b[i] != b'"' { return None; }
        let key_start = i;
        i += 1;
        while i < b.len() { match b[i] { b'"' => break, b'\\' => i += 2, _ => i += 1 } }
        if i >= b.len() { return None; }
        let key_end = i + 1;

        let key_bytes = &b[key_start + 1 .. key_end - 1];
        let mut is_dup = false;
        for j in 0..seen_count {
            let (ks, ke) = seen[j];
            if &b[ks + 1 .. ke - 1] == key_bytes { is_dup = true; break; }
        }
        if is_dup || seen_count >= seen.len() {
            had_dup = true;
            break;
        }
        seen[seen_count] = (key_start, key_end);
        seen_count += 1;
        keys.push((key_start, key_end));

        i = key_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return None };
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return None; }
        if b[i] == b'}' { break; }
        if b[i] != b',' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
    if had_dup {
        keys.clear();
        let mut pairs: Vec<(usize, usize, usize, usize)> = Vec::new();
        json_object_dedup_pairs(b, pos, &mut pairs)?;
        for (ks, ke, _, _) in &pairs { keys.push((*ks, *ke)); }
    }
    Some(keys.len())
}

/// Join keys of a JSON object with a separator, outputting a JSON string.
/// For keys_unsorted|join(sep): outputs `"key1<sep>key2<sep>..."\n`.
/// For keys|join(sep): sorts keys before joining.
/// Key content between quotes is extracted directly (handles escapes).
/// Returns false on non-object.
pub fn json_object_keys_join_to_buf(b: &[u8], pos: usize, sep: &[u8], sorted: bool, buf: &mut Vec<u8>, keys_buf: &mut Vec<(usize, usize)>) -> bool {
    let count = match json_object_extract_keys_only(b, pos, keys_buf) {
        Some(c) => c,
        None => return false,
    };
    if count == 0 {
        buf.extend_from_slice(b"\"\"\n");
        return true;
    }
    if sorted {
        keys_buf.sort_by(|a, b_range| {
            // Compare key content (between quotes, may have escapes)
            let ak = &b[a.0+1..a.1-1];
            let bk = &b[b_range.0+1..b_range.1-1];
            ak.cmp(bk)
        });
    }
    buf.push(b'"');
    // Check if separator needs JSON escaping
    // jq escapes 0x7F (DEL) too, in addition to U+0000..U+001F (#446).
    let sep_needs_escape = sep.iter().any(|&c| c == b'"' || c == b'\\' || c < 0x20 || c == 0x7F);
    for (idx, &(ks, ke)) in keys_buf.iter().enumerate() {
        if idx > 0 {
            if sep_needs_escape {
                for &c in sep {
                    match c {
                        b'"' => buf.extend_from_slice(b"\\\""),
                        b'\\' => buf.extend_from_slice(b"\\\\"),
                        c if c < 0x20 || c == 0x7F => {
                            buf.extend_from_slice(b"\\u00");
                            buf.push(b"0123456789abcdef"[(c >> 4) as usize]);
                            buf.push(b"0123456789abcdef"[(c & 0xf) as usize]);
                        }
                        _ => buf.push(c),
                    }
                }
            } else {
                buf.extend_from_slice(sep);
            }
        }
        // Key content is between quotes: b[ks+1..ke-1]
        // Already valid JSON string content (escapes preserved)
        buf.extend_from_slice(&b[ks+1..ke-1]);
    }
    buf.extend_from_slice(b"\"\n");
    true
}

/// Extract unsorted keys from a JSON object without full parsing.
/// Writes `["key1","key2",...]\n` to buf. Returns false on non-object.
pub fn json_object_keys_unsorted_to_buf(b: &[u8], pos: usize, buf: &mut Vec<u8>) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' {
        buf.extend_from_slice(b"[]\n");
        return true;
    }
    let buf_start = buf.len();
    let mut seen: [(usize, usize); 16] = [(0, 0); 16];
    let mut seen_count: usize = 0;
    buf.push(b'[');
    let mut first = true;
    loop {
        if i >= b.len() || b[i] != b'"' { return false; }
        let key_start = i;
        i += 1;
        while i < b.len() { match b[i] { b'"' => break, b'\\' => i += 2, _ => i += 1 } }
        if i >= b.len() { return false; }
        let key_end = i + 1;

        let key_bytes = &b[key_start + 1 .. key_end - 1];
        let mut is_dup = false;
        for j in 0..seen_count {
            let (ks, ke) = seen[j];
            if &b[ks + 1 .. ke - 1] == key_bytes { is_dup = true; break; }
        }
        if is_dup || seen_count >= seen.len() {
            buf.truncate(buf_start);
            let mut pairs: Vec<(usize, usize, usize, usize)> = Vec::new();
            if json_object_dedup_pairs(b, pos, &mut pairs).is_none() { return false; }
            if pairs.is_empty() {
                buf.extend_from_slice(b"[]\n");
                return true;
            }
            buf.push(b'[');
            for (idx, (ks, ke, _, _)) in pairs.iter().enumerate() {
                if idx > 0 { buf.push(b','); }
                buf.extend_from_slice(&b[*ks..*ke]);
            }
            buf.extend_from_slice(b"]\n");
            return true;
        }
        seen[seen_count] = (key_start, key_end);
        seen_count += 1;

        if !first { buf.push(b','); }
        first = false;
        buf.extend_from_slice(&b[key_start..key_end]);
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

/// Check if a raw JSON object has ALL of the given keys in a single pass.
/// Returns Some(true) if all keys found, Some(false) if any missing, None if not a valid object.
pub fn json_object_has_all_keys(b: &[u8], pos: usize, fields: &[&str]) -> Option<bool> {
    if pos >= b.len() || b[pos] != b'{' { return None; }
    let n = fields.len();
    if n == 0 { return Some(true); }
    let mut found = 0u64; // bitmask of found fields (supports up to 64 fields)
    if n > 64 { return None; }
    let all_found = (1u64 << n) - 1;
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
        let key_len = j - key_start;
        for (idx, f) in fields.iter().enumerate() {
            if key_len == f.len() && b[key_start..j] == *f.as_bytes() {
                found |= 1u64 << idx;
                if found == all_found { return Some(true); }
                break;
            }
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

/// Check if a raw JSON object has ANY of the given keys in a single pass.
/// Returns Some(true) if any key found, Some(false) if none found, None if not a valid object.
pub fn json_object_has_any_key(b: &[u8], pos: usize, fields: &[&str]) -> Option<bool> {
    if pos >= b.len() || b[pos] != b'{' { return None; }
    if fields.is_empty() { return Some(false); }
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
        let key_len = j - key_start;
        for f in fields.iter() {
            if key_len == f.len() && b[key_start..j] == *f.as_bytes() {
                return Some(true);
            }
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

/// Delete a field from a JSON object by copying raw bytes, skipping the field.
/// Writes a compact result (without trailing newline) to buf.
/// Returns true if successful, false if not a JSON object.
pub fn json_object_del_field(b: &[u8], pos: usize, field: &str, buf: &mut Vec<u8>) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let field_bytes = field.as_bytes();

    // Fast path for compact input: stream pairs directly, skip target field (no Vec alloc)
    if is_json_compact(&b[pos..]) {
        buf.push(b'{');
        let mut i = pos + 1;
        if i < b.len() && b[i] == b'}' { buf.push(b'}'); return true; }
        let mut first_out = true;
        loop {
            if i >= b.len() || b[i] != b'"' { break; }
            let key_start = i;
            i += 1;
            let mut j = i;
            while j < b.len() {
                match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 }
            }
            let key_matches = (j - i) == field_bytes.len() && b[i..j] == *field_bytes;
            i = j + 1;
            if i >= b.len() || b[i] != b':' { break; }
            i += 1;
            let val_end = match skip_json_value(b, i) { Ok(e) => e, Err(_) => break };
            if !key_matches {
                if !first_out { buf.push(b','); }
                first_out = false;
                buf.extend_from_slice(&b[key_start..val_end]); // "key":value as one chunk
            }
            i = val_end;
            if i < b.len() && b[i] == b',' { i += 1; } else { break; }
        }
        buf.push(b'}');
        return true;
    }

    // General path for non-compact input: stream directly (no Vec alloc)
    buf.push(b'{');
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { buf.push(b'}'); return true; }
    let mut first_out = true;
    loop {
        if i >= b.len() || b[i] != b'"' { break; }
        let key_start = i;
        i += 1;
        let mut j = i;
        while j < b.len() {
            match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 }
        }
        let key_matches = (j - i) == field_bytes.len() && b[i..j] == *field_bytes;
        let key_end = j + 1;
        i = key_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { break; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        let val_end = match skip_json_value(b, i) { Ok(e) => e, Err(_) => break };
        if !key_matches {
            if !first_out { buf.push(b','); }
            first_out = false;
            buf.extend_from_slice(&b[key_start..key_end]);
            buf.push(b':');
            buf.extend_from_slice(&b[i..val_end]);
        }
        i = val_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { break; }
        if b[i] == b'}' { break; }
        if b[i] != b',' { break; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
    buf.push(b'}');
    true
}

/// Delete multiple fields from a JSON object, writing compact output.
pub fn json_object_del_fields(b: &[u8], pos: usize, fields: &[&str], buf: &mut Vec<u8>) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    buf.push(b'{');
    let mut i = pos + 1;
    // Skip whitespace
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { buf.push(b'}'); return true; }
    let mut first_out = true;
    loop {
        // Skip whitespace before key
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b'"' { break; }
        let key_start = i;
        i += 1;
        let mut j = i;
        while j < b.len() {
            match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 }
        }
        let key_content = &b[i..j];
        let key_matches = fields.iter().any(|f| key_content == f.as_bytes());
        let key_end = j + 1;
        i = key_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { break; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        let val_end = match skip_json_value(b, i) { Ok(e) => e, Err(_) => break };
        if !key_matches {
            if !first_out { buf.push(b','); }
            first_out = false;
            buf.extend_from_slice(&b[key_start..key_end]);
            buf.push(b':');
            buf.extend_from_slice(&b[i..val_end]);
        }
        i = val_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { break; }
        if b[i] == b'}' { break; }
        if b[i] != b',' { break; }
        i += 1;
    }
    buf.push(b'}');
    true
}

/// Filter object entries by key string test, writing compact output.
/// test_op: "startswith", "endswith", "contains", "eq".
/// Includes entries where key passes the test.
pub fn json_object_filter_by_key_str(b: &[u8], pos: usize, test_op: &str, test_arg: &str, buf: &mut Vec<u8>) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let arg_bytes = test_arg.as_bytes();
    buf.push(b'{');
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { buf.push(b'}'); return true; }
    let mut first_out = true;
    loop {
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b'"' { break; }
        let key_start = i;
        i += 1;
        let mut j = i;
        while j < b.len() {
            match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 }
        }
        let key_content = &b[i..j];
        let key_matches = match test_op {
            "startswith" => key_content.len() >= arg_bytes.len() && &key_content[..arg_bytes.len()] == arg_bytes,
            "endswith" => key_content.len() >= arg_bytes.len() && &key_content[key_content.len()-arg_bytes.len()..] == arg_bytes,
            "contains" => {
                if arg_bytes.is_empty() { true }
                else {
                    key_content.windows(arg_bytes.len()).any(|w| w == arg_bytes)
                }
            }
            "eq" => key_content == arg_bytes,
            _ => false,
        };
        let key_end = j + 1;
        i = key_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { break; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        let val_end = match skip_json_value(b, i) { Ok(e) => e, Err(_) => break };
        if key_matches {
            if !first_out { buf.push(b','); }
            first_out = false;
            buf.extend_from_slice(&b[key_start..key_end]);
            buf.push(b':');
            buf.extend_from_slice(&b[i..val_end]);
        }
        i = val_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { break; }
        if b[i] == b'}' { break; }
        if b[i] != b',' { break; }
        i += 1;
    }
    buf.push(b'}');
    true
}

/// Replace a single field's value in a JSON object, writing compact output.
/// Copies the object structure, substituting only the target field's value with `new_val`.
/// If the field is not found, writes the object unchanged (compact).
/// Returns true on success.
pub fn json_object_replace_field(b: &[u8], pos: usize, field: &str, new_val: &[u8], buf: &mut Vec<u8>) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let field_bytes = field.as_bytes();
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' {
        buf.extend_from_slice(b"{}");
        return true;
    }
    let mut first = true;
    buf.push(b'{');
    loop {
        if i >= b.len() || b[i] != b'"' { return false; }
        let key_start = i;
        i += 1;
        let mut j = i;
        while j < b.len() { match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 } }
        if j >= b.len() { return false; }
        let key_matches = (j - i) == field_bytes.len() && b[i..j] == *field_bytes;
        let key_end = j + 1;
        i = key_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        let val_start = i;
        i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return false };
        let val_end = i;
        if !first { buf.push(b','); }
        first = false;
        buf.extend_from_slice(&b[key_start..key_end]);
        buf.push(b':');
        if key_matches {
            buf.extend_from_slice(new_val);
        } else {
            buf.extend_from_slice(&b[val_start..val_end]);
        }
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return false; }
        if b[i] == b'}' { break; }
        if b[i] != b',' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
    buf.push(b'}');
    true
}

/// Update a single numeric field in a JSON object in one pass: find + compute + replace.
/// Writes compact output to `buf`. Returns true on success (field found and was numeric).
pub fn json_object_update_field_num(b: &[u8], pos: usize, field: &str, op: crate::ir::BinOp, n: f64, buf: &mut Vec<u8>) -> bool {
    use crate::ir::BinOp;
    if pos >= b.len() || b[pos] != b'{' { return false; }
    // Find the target field value range, then do 3-way copy:
    // (bytes before value) + (new number) + (bytes after value including closing brace)
    if let Some((val_start, val_end)) = json_object_get_field_raw(b, pos, field) {
        if let Some(a) = parse_json_num(&b[val_start..val_end]) {
            let r = match op { BinOp::Add => a + n, BinOp::Sub => a - n, BinOp::Mul => a * n, BinOp::Div => a / n, BinOp::Mod => match crate::runtime::jq_mod_f64(a, n) { Some(v) => v, None => return false }, _ => a };
            if !r.is_finite() { return false; }
            // Find object end by scanning backward for '}' — avoids full re-parse
            let mut obj_end = b.len();
            while obj_end > val_end && b[obj_end - 1] != b'}' { obj_end -= 1; }
            if obj_end <= val_end { return false; }
            // 3-way bulk copy: prefix + new number + suffix
            buf.extend_from_slice(&b[pos..val_start]);
            push_jq_number_bytes(buf, r);
            buf.extend_from_slice(&b[val_end..obj_end]);
            return true;
        }
    }
    false
}

/// Update a numeric field using a chain of operations.
/// Each step is either an arithmetic operation (. op N) or a unary op (floor/ceil/etc).
pub fn json_object_update_field_num_chain(
    b: &[u8], pos: usize, field: &str,
    steps: &[crate::interpreter::NumChainStep], buf: &mut Vec<u8>,
) -> bool {
    use crate::ir::BinOp;
    use crate::ir::UnaryOp;
    use crate::interpreter::NumChainStep;
    if pos >= b.len() || b[pos] != b'{' { return false; }
    if let Some((val_start, val_end)) = json_object_get_field_raw(b, pos, field) {
        if let Some(mut v) = parse_json_num(&b[val_start..val_end]) {
            for step in steps {
                match step {
                    NumChainStep::Arith(op, n) => {
                        v = match op {
                            BinOp::Add => v + n, BinOp::Sub => v - n,
                            BinOp::Mul => v * n, BinOp::Div => v / n,
                            BinOp::Mod => match crate::runtime::jq_mod_f64(v, *n) { Some(r) => r, None => return false },
                            _ => v,
                        };
                    }
                    NumChainStep::Unary(op) => {
                        v = match op {
                            UnaryOp::Floor => v.floor(),
                            UnaryOp::Ceil => v.ceil(),
                            UnaryOp::Round => v.round(),
                            UnaryOp::Fabs => v.abs(),
                            UnaryOp::Sqrt => v.sqrt(),
                            UnaryOp::Trunc => v.trunc(),
                            _ => v,
                        };
                    }
                }
            }
            if !v.is_finite() { return false; }
            let mut obj_end = b.len();
            while obj_end > val_end && b[obj_end - 1] != b'}' { obj_end -= 1; }
            if obj_end <= val_end { return false; }
            buf.extend_from_slice(&b[pos..val_start]);
            push_jq_number_bytes(buf, v);
            buf.extend_from_slice(&b[val_end..obj_end]);
            return true;
        }
    }
    false
}

/// Update a string field by applying regex gsub replacement.
/// Extracts the field value, applies regex replacement, and writes back.
/// For strings without JSON escapes, operates directly on raw bytes.
/// `is_global`: true for gsub (replace all), false for sub (replace first).
pub fn json_object_update_field_gsub(
    b: &[u8], pos: usize, field: &str, re: &regex::Regex, replacement: &str, is_global: bool, buf: &mut Vec<u8>,
) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    if let Some((val_start, val_end)) = json_object_get_field_raw(b, pos, field) {
        if val_start >= val_end || b[val_start] != b'"' { return false; }
        let inner = &b[val_start + 1..val_end - 1];
        // Check for JSON escapes — if none, we can operate on raw bytes directly
        let has_escapes = memchr::memchr(b'\\', inner).is_some();
        let result_str;
        let result_bytes: &[u8];
        if has_escapes {
            // Must unescape, apply, re-escape
            let s: String = match serde_json::from_slice(&b[val_start..val_end]) {
                Ok(s) => s,
                Err(_) => return false,
            };
            result_str = if is_global {
                re.replace_all(&s, replacement).into_owned()
            } else {
                re.replace(&s, replacement).into_owned()
            };
            result_bytes = result_str.as_bytes();
        } else {
            // Fast: no escapes, work directly on raw bytes
            let s = unsafe { std::str::from_utf8_unchecked(inner) };
            result_str = if is_global {
                re.replace_all(s, replacement).into_owned()
            } else {
                re.replace(s, replacement).into_owned()
            };
            result_bytes = result_str.as_bytes();
        }
        // Find object end
        let mut obj_end = b.len();
        while obj_end > val_end && b[obj_end - 1] != b'}' { obj_end -= 1; }
        if obj_end <= val_end { return false; }
        // 3-way copy: prefix + new string value + suffix
        buf.extend_from_slice(&b[pos..val_start]);
        buf.push(b'"');
        for &ch in result_bytes {
            match ch {
                b'"' => buf.extend_from_slice(b"\\\""),
                b'\\' => buf.extend_from_slice(b"\\\\"),
                b'\n' => buf.extend_from_slice(b"\\n"),
                b'\r' => buf.extend_from_slice(b"\\r"),
                b'\t' => buf.extend_from_slice(b"\\t"),
                // jq also escapes 0x7F (DEL) as `` (#446).
                c if c < 0x20 || c == 0x7F => {
                    buf.extend_from_slice(format!("\\u{:04x}", c).as_bytes());
                }
                _ => buf.push(ch),
            }
        }
        buf.push(b'"');
        buf.extend_from_slice(&b[val_end..obj_end]);
        return true;
    }
    false
}

/// Combined select + field update: select(.sel_field cmp threshold) | .upd_field |= (. arith val).
/// Returns true and writes to buf if select matches and update succeeds; false otherwise (skip).
pub fn json_object_select_then_update_num(
    b: &[u8], pos: usize,
    sel_field: &str, cmp_op: crate::ir::BinOp, threshold: f64,
    upd_field: &str, arith_op: crate::ir::BinOp, arith_val: f64,
    buf: &mut Vec<u8>,
) -> Option<bool> {
    // Returns Some(true) = matched & updated, Some(false) = matched but update failed, None = not matched
    use crate::ir::BinOp;
    if pos >= b.len() || b[pos] != b'{' { return None; }
    // 1. Check select condition
    let sel_val = if let Some((vs, ve)) = json_object_get_field_raw(b, pos, sel_field) {
        match parse_json_num(&b[vs..ve]) { Some(v) => v, None => return None }
    } else { return None; };
    let cond = match cmp_op {
        BinOp::Gt => sel_val > threshold,
        BinOp::Lt => sel_val < threshold,
        BinOp::Ge => sel_val >= threshold,
        BinOp::Le => sel_val <= threshold,
        BinOp::Eq => sel_val == threshold,
        BinOp::Ne => sel_val != threshold,
        _ => return None,
    };
    if !cond { return None; } // select didn't match
    // 2. Apply update
    if json_object_update_field_num(b, pos, upd_field, arith_op, arith_val, buf) {
        Some(true)
    } else {
        Some(false)
    }
}

/// Update a field by replacing its value with the boolean result of a regex test.
/// `.field |= test("regex")` — extract string field, run regex, write true/false.
pub fn json_object_update_field_test(
    b: &[u8], pos: usize, field: &str, re: &regex::Regex, buf: &mut Vec<u8>,
) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    if let Some((val_start, val_end)) = json_object_get_field_raw(b, pos, field) {
        if val_start >= val_end || b[val_start] != b'"' { return false; }
        let inner = &b[val_start + 1..val_end - 1];
        let matched = if memchr::memchr(b'\\', inner).is_some() {
            let s: String = match serde_json::from_slice(&b[val_start..val_end]) {
                Ok(s) => s,
                Err(_) => return false,
            };
            re.is_match(&s)
        } else {
            let s = unsafe { std::str::from_utf8_unchecked(inner) };
            re.is_match(s)
        };
        let mut obj_end = b.len();
        while obj_end > val_end && b[obj_end - 1] != b'}' { obj_end -= 1; }
        if obj_end <= val_end { return false; }
        buf.extend_from_slice(&b[pos..val_start]);
        buf.extend_from_slice(if matched { b"true" } else { b"false" });
        buf.extend_from_slice(&b[val_end..obj_end]);
        return true;
    }
    false
}

/// Cross-field numeric assignment: `.dest = (.src op N)`.
/// Reads src field's number, applies arithmetic, writes result to dest field.
pub fn json_object_assign_field_arith(
    b: &[u8], pos: usize, dest_field: &str, src_field: &str,
    op: crate::ir::BinOp, n: f64, buf: &mut Vec<u8>,
) -> bool {
    use crate::ir::BinOp;
    if pos >= b.len() || b[pos] != b'{' { return false; }
    // 1. Read src field value
    let src_val = if let Some((vs, ve)) = json_object_get_field_raw(b, pos, src_field) {
        match parse_json_num(&b[vs..ve]) { Some(v) => v, None => return false }
    } else { return false; };
    // 2. Compute result
    let r = match op {
        BinOp::Add => src_val + n,
        BinOp::Sub => src_val - n,
        BinOp::Mul => src_val * n,
        BinOp::Div => src_val / n,
        BinOp::Mod => match crate::runtime::jq_mod_f64(src_val, n) { Some(v) => v, None => return false },
        _ => src_val,
    };
    if !r.is_finite() { return false; }
    // 3. Write result to dest field using 3-way copy (if dest exists)
    if let Some((dv_start, dv_end)) = json_object_get_field_raw(b, pos, dest_field) {
        let mut obj_end = b.len();
        while obj_end > dv_end && b[obj_end - 1] != b'}' { obj_end -= 1; }
        if obj_end <= dv_end { return false; }
        buf.extend_from_slice(&b[pos..dv_start]);
        push_jq_number_bytes(buf, r);
        buf.extend_from_slice(&b[dv_end..obj_end]);
        return true;
    }
    // dest field doesn't exist — use set_field_raw for add
    let mut num_buf = Vec::with_capacity(32);
    push_jq_number_bytes(&mut num_buf, r);
    return json_object_set_field_raw(b, pos, dest_field, &num_buf, buf);
}

/// Cross-field two-field arithmetic: `.dest = (.src1 op .src2)`.
/// Reads two numeric fields, applies arithmetic, writes result to dest field.
pub fn json_object_assign_two_fields_arith(
    b: &[u8], pos: usize, dest_field: &str, src1_field: &str, src2_field: &str,
    op: crate::ir::BinOp, buf: &mut Vec<u8>,
) -> bool {
    use crate::ir::BinOp;
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let v1 = if let Some((vs, ve)) = json_object_get_field_raw(b, pos, src1_field) {
        match parse_json_num(&b[vs..ve]) { Some(v) => v, None => return false }
    } else { return false; };
    let v2 = if let Some((vs, ve)) = json_object_get_field_raw(b, pos, src2_field) {
        match parse_json_num(&b[vs..ve]) { Some(v) => v, None => return false }
    } else { return false; };
    let r = match op {
        BinOp::Add => v1 + v2,
        BinOp::Sub => v1 - v2,
        BinOp::Mul => v1 * v2,
        BinOp::Div => v1 / v2,
        BinOp::Mod => match crate::runtime::jq_mod_f64(v1, v2) { Some(v) => v, None => return false },
        _ => return false,
    };
    if !r.is_finite() { return false; }
    if let Some((dv_start, dv_end)) = json_object_get_field_raw(b, pos, dest_field) {
        let mut obj_end = b.len();
        while obj_end > dv_end && b[obj_end - 1] != b'}' { obj_end -= 1; }
        if obj_end <= dv_end { return false; }
        buf.extend_from_slice(&b[pos..dv_start]);
        push_jq_number_bytes(buf, r);
        buf.extend_from_slice(&b[dv_end..obj_end]);
        return true;
    }
    let mut num_buf = Vec::with_capacity(32);
    push_jq_number_bytes(&mut num_buf, r);
    return json_object_set_field_raw(b, pos, dest_field, &num_buf, buf);
}

/// Update a field by taking the first element after splitting by a separator.
/// `.field |= (split("sep")|.[0])` — extract string field, find first separator, write prefix.
pub fn json_object_update_field_split_first(
    b: &[u8], pos: usize, field: &str, sep: &str, buf: &mut Vec<u8>,
) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    if let Some((val_start, val_end)) = json_object_get_field_raw(b, pos, field) {
        if val_start >= val_end || b[val_start] != b'"' { return false; }
        let inner = &b[val_start + 1..val_end - 1];
        let has_escapes = memchr::memchr(b'\\', inner).is_some();
        // Find object end
        let mut obj_end = b.len();
        while obj_end > val_end && b[obj_end - 1] != b'}' { obj_end -= 1; }
        if obj_end <= val_end { return false; }
        buf.extend_from_slice(&b[pos..val_start]);
        buf.push(b'"');
        if has_escapes {
            // Must unescape, split, re-escape
            let s: String = match serde_json::from_slice(&b[val_start..val_end]) {
                Ok(s) => s,
                Err(_) => return false,
            };
            let first = s.split(sep).next().unwrap_or("");
            for &ch in first.as_bytes() {
                match ch {
                    b'"' => buf.extend_from_slice(b"\\\""),
                    b'\\' => buf.extend_from_slice(b"\\\\"),
                    b'\n' => buf.extend_from_slice(b"\\n"),
                    b'\r' => buf.extend_from_slice(b"\\r"),
                    b'\t' => buf.extend_from_slice(b"\\t"),
                    // jq also escapes 0x7F (DEL) (#446).
                    c if c < 0x20 || c == 0x7F => {
                        buf.extend_from_slice(format!("\\u{:04x}", c).as_bytes());
                    }
                    _ => buf.push(ch),
                }
            }
        } else {
            // Fast: no escapes, find separator directly in raw bytes
            let sep_bytes = sep.as_bytes();
            if sep_bytes.is_empty() {
                // split("") produces individual characters — first is first char
                if !inner.is_empty() {
                    let ch_len = if inner[0] < 0x80 { 1 }
                        else if inner[0] < 0xE0 { 2 }
                        else if inner[0] < 0xF0 { 3 }
                        else { 4 };
                    let end = ch_len.min(inner.len());
                    buf.extend_from_slice(&inner[..end]);
                }
            } else if let Some(idx) = inner.windows(sep_bytes.len()).position(|w| w == sep_bytes) {
                buf.extend_from_slice(&inner[..idx]);
            } else {
                // No separator found — result is unchanged (whole string)
                buf.extend_from_slice(inner);
            }
        }
        buf.push(b'"');
        buf.extend_from_slice(&b[val_end..obj_end]);
        return true;
    }
    false
}

/// Update a field by taking the last element after splitting by a separator.
/// `.field |= (split("sep")|last)` — extract string field, find last separator, write suffix.
pub fn json_object_update_field_split_last(
    b: &[u8], pos: usize, field: &str, sep: &str, buf: &mut Vec<u8>,
) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    if let Some((val_start, val_end)) = json_object_get_field_raw(b, pos, field) {
        if val_start >= val_end || b[val_start] != b'"' { return false; }
        let inner = &b[val_start + 1..val_end - 1];
        let has_escapes = memchr::memchr(b'\\', inner).is_some();
        // Find object end
        let mut obj_end = b.len();
        while obj_end > val_end && b[obj_end - 1] != b'}' { obj_end -= 1; }
        if obj_end <= val_end { return false; }
        buf.extend_from_slice(&b[pos..val_start]);
        buf.push(b'"');
        if has_escapes {
            let s: String = match serde_json::from_slice(&b[val_start..val_end]) {
                Ok(s) => s,
                Err(_) => return false,
            };
            let last = s.rsplit(sep).next().unwrap_or("");
            for &ch in last.as_bytes() {
                match ch {
                    b'"' => buf.extend_from_slice(b"\\\""),
                    b'\\' => buf.extend_from_slice(b"\\\\"),
                    b'\n' => buf.extend_from_slice(b"\\n"),
                    b'\r' => buf.extend_from_slice(b"\\r"),
                    b'\t' => buf.extend_from_slice(b"\\t"),
                    // jq also escapes 0x7F (DEL) (#446).
                    c if c < 0x20 || c == 0x7F => {
                        buf.extend_from_slice(format!("\\u{:04x}", c).as_bytes());
                    }
                    _ => buf.push(ch),
                }
            }
        } else {
            let sep_bytes = sep.as_bytes();
            if sep_bytes.is_empty() {
                // split("") | last → last character
                if !inner.is_empty() {
                    // Find last UTF-8 character
                    let mut start = inner.len() - 1;
                    while start > 0 && inner[start] >= 0x80 && inner[start] < 0xC0 {
                        start -= 1;
                    }
                    buf.extend_from_slice(&inner[start..]);
                }
            } else if let Some(idx) = inner.windows(sep_bytes.len()).rposition(|w| w == sep_bytes) {
                buf.extend_from_slice(&inner[idx + sep_bytes.len()..]);
            } else {
                // No separator found — result is unchanged (whole string)
                buf.extend_from_slice(inner);
            }
        }
        buf.push(b'"');
        buf.extend_from_slice(&b[val_end..obj_end]);
        return true;
    }
    false
}

/// Update a field by trimming a prefix or suffix.
/// `.field |= ltrimstr("prefix")` or `.field |= rtrimstr("suffix")`.
pub fn json_object_update_field_trim(
    b: &[u8], pos: usize, field: &str, trim_str: &str, is_rtrim: bool, buf: &mut Vec<u8>,
) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    if let Some((val_start, val_end)) = json_object_get_field_raw(b, pos, field) {
        if val_start >= val_end || b[val_start] != b'"' { return false; }
        let inner = &b[val_start + 1..val_end - 1];
        let has_escapes = memchr::memchr(b'\\', inner).is_some();
        let mut obj_end = b.len();
        while obj_end > val_end && b[obj_end - 1] != b'}' { obj_end -= 1; }
        if obj_end <= val_end { return false; }
        if has_escapes {
            let s: String = match serde_json::from_slice(&b[val_start..val_end]) {
                Ok(s) => s,
                Err(_) => return false,
            };
            let result: &str = if is_rtrim {
                // rtrimstr("") reduces to .[:-0], which is "" in jq slice semantics.
                if trim_str.is_empty() { "" } else { s.strip_suffix(trim_str).unwrap_or(&s) }
            } else {
                s.strip_prefix(trim_str).unwrap_or(&s)
            };
            buf.extend_from_slice(&b[pos..val_start]);
            buf.push(b'"');
            for &ch in result.as_bytes() {
                match ch {
                    b'"' => buf.extend_from_slice(b"\\\""),
                    b'\\' => buf.extend_from_slice(b"\\\\"),
                    b'\n' => buf.extend_from_slice(b"\\n"),
                    b'\r' => buf.extend_from_slice(b"\\r"),
                    b'\t' => buf.extend_from_slice(b"\\t"),
                    // jq also escapes 0x7F (DEL) (#446).
                    c if c < 0x20 || c == 0x7F => buf.extend_from_slice(format!("\\u{:04x}", c).as_bytes()),
                    _ => buf.push(ch),
                }
            }
            buf.push(b'"');
        } else {
            let trim_bytes = trim_str.as_bytes();
            buf.extend_from_slice(&b[pos..val_start]);
            buf.push(b'"');
            if is_rtrim {
                if trim_bytes.is_empty() {
                    // rtrimstr("") yields "" regardless of input (jq slice -0 quirk).
                } else if inner.ends_with(trim_bytes) {
                    buf.extend_from_slice(&inner[..inner.len() - trim_bytes.len()]);
                } else {
                    buf.extend_from_slice(inner);
                }
            } else {
                if inner.starts_with(trim_bytes) {
                    buf.extend_from_slice(&inner[trim_bytes.len()..]);
                } else {
                    buf.extend_from_slice(inner);
                }
            }
            buf.push(b'"');
        }
        buf.extend_from_slice(&b[val_end..obj_end]);
        return true;
    }
    false
}

/// Update a field by slicing a string: `.field |= .[from:to]`.
pub fn json_object_update_field_slice(
    b: &[u8], pos: usize, field: &str, from: Option<i64>, to: Option<i64>, buf: &mut Vec<u8>,
) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    if let Some((val_start, val_end)) = json_object_get_field_raw(b, pos, field) {
        if val_start >= val_end || b[val_start] != b'"' { return false; }
        let inner = &b[val_start + 1..val_end - 1];
        let has_escapes = memchr::memchr(b'\\', inner).is_some();
        let mut obj_end = b.len();
        while obj_end > val_end && b[obj_end - 1] != b'}' { obj_end -= 1; }
        if obj_end <= val_end { return false; }
        // jq string slicing works on codepoints, so we need to decode
        let s: String = if has_escapes {
            match serde_json::from_slice(&b[val_start..val_end]) {
                Ok(s) => s,
                Err(_) => return false,
            }
        } else {
            unsafe { std::str::from_utf8_unchecked(inner) }.to_string()
        };
        let chars: Vec<char> = s.chars().collect();
        let len = chars.len() as i64;
        let f = match from {
            Some(v) => if v < 0 { (len + v).max(0) as usize } else { (v as usize).min(len as usize) },
            None => 0,
        };
        let t = match to {
            Some(v) => if v < 0 { (len + v).max(0) as usize } else { (v as usize).min(len as usize) },
            None => len as usize,
        };
        let result: String = if f < t { chars[f..t].iter().collect() } else { String::new() };
        buf.extend_from_slice(&b[pos..val_start]);
        buf.push(b'"');
        for &ch in result.as_bytes() {
            match ch {
                b'"' => buf.extend_from_slice(b"\\\""),
                b'\\' => buf.extend_from_slice(b"\\\\"),
                b'\n' => buf.extend_from_slice(b"\\n"),
                b'\r' => buf.extend_from_slice(b"\\r"),
                b'\t' => buf.extend_from_slice(b"\\t"),
                // jq also escapes 0x7F (DEL) (#446).
                c if c < 0x20 || c == 0x7F => buf.extend_from_slice(format!("\\u{:04x}", c).as_bytes()),
                _ => buf.push(ch),
            }
        }
        buf.push(b'"');
        buf.extend_from_slice(&b[val_end..obj_end]);
        return true;
    }
    false
}

/// Convert all values in a JSON object to strings.
/// `with_entries(.value |= tostring)` — strings stay, numbers/bools/null get wrapped in quotes.
pub fn json_object_values_tostring(
    b: &[u8], pos: usize, buf: &mut Vec<u8>,
) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    buf.push(b'{');
    let mut i = pos + 1;
    let mut first = true;
    loop {
        // Skip whitespace
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return false; }
        if b[i] == b'}' { break; }
        if !first { buf.push(b','); }
        first = false;
        // Key (must be string)
        if b[i] != b'"' { return false; }
        let key_start = i;
        i += 1;
        while i < b.len() {
            if b[i] == b'\\' { i += 2; continue; }
            if b[i] == b'"' { i += 1; break; }
            i += 1;
        }
        buf.extend_from_slice(&b[key_start..i]);
        // Colon
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return false; }
        i += 1;
        buf.push(b':');
        // Skip whitespace before value
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return false; }
        // Value
        let val_start = i;
        if b[i] == b'"' {
            // Already a string — copy as-is
            i += 1;
            while i < b.len() {
                if b[i] == b'\\' { i += 2; continue; }
                if b[i] == b'"' { i += 1; break; }
                i += 1;
            }
            buf.extend_from_slice(&b[val_start..i]);
        } else {
            // Number, bool, null — find end, wrap in quotes
            let end = match skip_json_value(b, val_start) {
                Ok(e) => e,
                Err(_) => return false,
            };
            let val_bytes = &b[val_start..end];
            buf.push(b'"');
            // For numbers, use push_jq_number_bytes for jq-compatible formatting
            if b[val_start] == b'-' || b[val_start].is_ascii_digit() {
                if let Ok(n) = unsafe { std::str::from_utf8_unchecked(val_bytes) }.parse::<f64>() {
                    push_jq_number_bytes(buf, n);
                } else {
                    buf.extend_from_slice(val_bytes);
                }
            } else {
                // true, false, null
                buf.extend_from_slice(val_bytes);
            }
            buf.push(b'"');
            i = end;
        }
        // Comma or end
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i < b.len() && b[i] == b',' { i += 1; }
    }
    buf.push(b'}');
    true
}

/// Update a field by replacing its string value with its length.
/// `.field |= length` — replace string with character count.
pub fn json_object_update_field_length(
    b: &[u8], pos: usize, field: &str, buf: &mut Vec<u8>,
) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    if let Some((val_start, val_end)) = json_object_get_field_raw(b, pos, field) {
        if val_start >= val_end || b[val_start] != b'"' { return false; }
        let inner = &b[val_start + 1..val_end - 1];
        let has_escapes = memchr::memchr(b'\\', inner).is_some();
        // Count characters (codepoints), not bytes
        let char_len = if has_escapes {
            let s: String = match serde_json::from_slice(&b[val_start..val_end]) {
                Ok(s) => s,
                Err(_) => return false,
            };
            s.chars().count()
        } else {
            // Count UTF-8 codepoints: count bytes that are NOT continuation bytes (10xxxxxx)
            inner.iter().filter(|&&b| b & 0xC0 != 0x80).count()
        };
        let mut obj_end = b.len();
        while obj_end > val_end && b[obj_end - 1] != b'}' { obj_end -= 1; }
        if obj_end <= val_end { return false; }
        buf.extend_from_slice(&b[pos..val_start]);
        push_jq_number_bytes(buf, char_len as f64);
        buf.extend_from_slice(&b[val_end..obj_end]);
        return true;
    }
    false
}

/// Update a field by applying tostring.
/// For strings: identity. For numbers: wrap in quotes. For booleans/null: wrap in quotes.
pub fn json_object_update_field_tostring(
    b: &[u8], pos: usize, field: &str, buf: &mut Vec<u8>,
) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    if let Some((val_start, val_end)) = json_object_get_field_raw(b, pos, field) {
        if val_start >= val_end { return false; }
        let first = b[val_start];
        let mut obj_end = b.len();
        while obj_end > val_end && b[obj_end - 1] != b'}' { obj_end -= 1; }
        if obj_end <= val_end { return false; }
        if first == b'"' {
            // Already a string — tostring is identity
            buf.extend_from_slice(&b[pos..obj_end]);
            return true;
        }
        // Number, boolean, or null — wrap the raw value in quotes
        let val_bytes = &b[val_start..val_end];
        buf.extend_from_slice(&b[pos..val_start]);
        buf.push(b'"');
        // For numbers, use push_jq_number_bytes for correct formatting
        if first == b'-' || first.is_ascii_digit() {
            if let Ok(n) = unsafe { std::str::from_utf8_unchecked(val_bytes) }.parse::<f64>() {
                push_jq_number_bytes(buf, n);
            } else {
                buf.extend_from_slice(val_bytes);
            }
        } else {
            // true, false, null → "true", "false", "null"
            buf.extend_from_slice(val_bytes);
        }
        buf.push(b'"');
        buf.extend_from_slice(&b[val_end..obj_end]);
        return true;
    }
    false
}

/// Update a field by concatenating prefix/suffix strings.
/// `.field |= (. + "suffix")` or `.field |= ("prefix" + .)`.
pub fn json_object_update_field_str_concat(
    b: &[u8], pos: usize, field: &str, prefix: &[u8], suffix: &[u8], buf: &mut Vec<u8>,
) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    if let Some((val_start, val_end)) = json_object_get_field_raw(b, pos, field) {
        if val_start >= val_end || b[val_start] != b'"' { return false; }
        let inner = &b[val_start + 1..val_end - 1];
        let mut obj_end = b.len();
        while obj_end > val_end && b[obj_end - 1] != b'}' { obj_end -= 1; }
        if obj_end <= val_end { return false; }
        buf.extend_from_slice(&b[pos..val_start]);
        buf.push(b'"');
        buf.extend_from_slice(prefix);
        buf.extend_from_slice(inner);
        buf.extend_from_slice(suffix);
        buf.push(b'"');
        buf.extend_from_slice(&b[val_end..obj_end]);
        return true;
    }
    false
}

/// Update a field by mapping string equality: `.field |= if . == "a" then "b" else "c" end`.
pub fn json_object_update_field_str_map(
    b: &[u8], pos: usize, field: &str, cond_str: &[u8], then_json: &[u8], else_json: &[u8], buf: &mut Vec<u8>,
) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    if let Some((val_start, val_end)) = json_object_get_field_raw(b, pos, field) {
        if val_start >= val_end || b[val_start] != b'"' { return false; }
        let inner = &b[val_start + 1..val_end - 1];
        let mut obj_end = b.len();
        while obj_end > val_end && b[obj_end - 1] != b'}' { obj_end -= 1; }
        if obj_end <= val_end { return false; }
        buf.extend_from_slice(&b[pos..val_start]);
        if inner == cond_str {
            buf.extend_from_slice(then_json);
        } else {
            buf.extend_from_slice(else_json);
        }
        buf.extend_from_slice(&b[val_end..obj_end]);
        return true;
    }
    false
}

/// Combined select (numeric cmp) + string concat update.
/// Returns Some(true) = matched & updated, Some(false) = matched but update failed, None = not matched.
pub fn json_object_select_then_update_str_concat(
    b: &[u8], pos: usize,
    sel_field: &str, cmp_op: crate::ir::BinOp, threshold: f64,
    upd_field: &str, prefix: &[u8], suffix: &[u8],
    buf: &mut Vec<u8>,
) -> Option<bool> {
    use crate::ir::BinOp;
    if pos >= b.len() || b[pos] != b'{' { return None; }
    let sel_val = if let Some((vs, ve)) = json_object_get_field_raw(b, pos, sel_field) {
        match parse_json_num(&b[vs..ve]) { Some(v) => v, None => return None }
    } else { return None; };
    let cond = match cmp_op {
        BinOp::Gt => sel_val > threshold,
        BinOp::Lt => sel_val < threshold,
        BinOp::Ge => sel_val >= threshold,
        BinOp::Le => sel_val <= threshold,
        BinOp::Eq => sel_val == threshold,
        BinOp::Ne => sel_val != threshold,
        _ => return None,
    };
    if !cond { return None; }
    if json_object_update_field_str_concat(b, pos, upd_field, prefix, suffix, buf) {
        Some(true)
    } else {
        Some(false)
    }
}

/// Combined compound select (AND/OR of numeric cmps) + numeric field update.
/// Returns Some(true) = matched & updated, Some(false) = matched but update failed, None = not matched.
pub fn json_object_select_compound_then_update_num(
    b: &[u8], pos: usize,
    logic_op: crate::ir::BinOp, conds: &[(String, crate::ir::BinOp, f64)],
    upd_field: &str, arith_op: crate::ir::BinOp, arith_val: f64,
    buf: &mut Vec<u8>,
) -> Option<bool> {
    use crate::ir::BinOp;
    if pos >= b.len() || b[pos] != b'{' { return None; }
    // Evaluate all conditions
    let eval_cmp = |field: &str, cmp_op: BinOp, threshold: f64| -> Option<bool> {
        let val = if let Some((vs, ve)) = json_object_get_field_raw(b, pos, field) {
            parse_json_num(&b[vs..ve])?
        } else { return None; };
        Some(match cmp_op {
            BinOp::Gt => val > threshold,
            BinOp::Lt => val < threshold,
            BinOp::Ge => val >= threshold,
            BinOp::Le => val <= threshold,
            BinOp::Eq => val == threshold,
            BinOp::Ne => val != threshold,
            _ => return None,
        })
    };
    let result = match logic_op {
        BinOp::And => {
            let mut all = true;
            for (f, op, thr) in conds {
                match eval_cmp(f, *op, *thr) {
                    Some(true) => {}
                    Some(false) => { all = false; break; }
                    None => return None,
                }
            }
            all
        }
        BinOp::Or => {
            let mut any = false;
            for (f, op, thr) in conds {
                match eval_cmp(f, *op, *thr) {
                    Some(true) => { any = true; break; }
                    Some(false) => {}
                    None => return None,
                }
            }
            any
        }
        _ => return None,
    };
    if !result { return None; }
    if json_object_update_field_num(b, pos, upd_field, arith_op, arith_val, buf) {
        Some(true)
    } else {
        Some(false)
    }
}

/// Combined string-test select + numeric field update.
/// test_type: "eq", "startswith", "endswith", "contains"
/// Returns Some(true) = matched & updated, Some(false) = matched but update failed, None = not matched.
pub fn json_object_select_str_then_update_num(
    b: &[u8], pos: usize,
    cond_field: &str, test_type: &str, test_arg: &str,
    upd_field: &str, arith_op: crate::ir::BinOp, arith_val: f64,
    buf: &mut Vec<u8>,
) -> Option<bool> {
    if pos >= b.len() || b[pos] != b'{' { return None; }
    // 1. Check string condition
    let (vs, ve) = json_object_get_field_raw(b, pos, cond_field)?;
    let val = &b[vs..ve];
    if val.len() < 2 || val[0] != b'"' || val[ve - vs - 1] != b'"' { return None; }
    let inner = &val[1..ve - vs - 1];
    // Handle escaped strings via serde fallback
    let unescaped: Option<String>;
    let check_bytes: &[u8] = if memchr::memchr(b'\\', inner).is_some() {
        let s: String = match serde_json::from_slice(val) { Ok(s) => s, Err(_) => return None };
        unescaped = Some(s);
        unescaped.as_ref().unwrap().as_bytes()
    } else {
        inner
    };
    let pass = match test_type {
        "eq" => check_bytes == test_arg.as_bytes(),
        "startswith" => check_bytes.starts_with(test_arg.as_bytes()),
        "endswith" => check_bytes.ends_with(test_arg.as_bytes()),
        "contains" => {
            let needle = test_arg.as_bytes();
            check_bytes.windows(needle.len()).any(|w| w == needle)
        }
        _ => return None,
    };
    if !pass { return None; }
    // 2. Apply numeric update
    if json_object_update_field_num(b, pos, upd_field, arith_op, arith_val, buf) {
        Some(true)
    } else {
        Some(false)
    }
}

/// Merge literal key-value pairs into a raw JSON object.
/// `pairs` is a list of (key_name, json_value_bytes).
/// Existing keys are replaced (last-write-wins), new keys are appended.
/// Returns true on success.
pub fn json_object_merge_literal(b: &[u8], pos: usize, merge_pairs: &[(String, Vec<u8>)], buf: &mut Vec<u8>) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' {
        // Empty object — just output the merge pairs
        buf.push(b'{');
        for (idx, (k, v)) in merge_pairs.iter().enumerate() {
            if idx > 0 { buf.push(b','); }
            buf.push(b'"');
            buf.extend_from_slice(k.as_bytes());
            buf.push(b'"');
            buf.push(b':');
            buf.extend_from_slice(v);
        }
        buf.push(b'}');
        return true;
    }
    // Fast path: if no merge key appears in the raw bytes,
    // skip key-by-key scanning and just copy-and-append.
    {
        // Check if any merge key pattern ("key":) appears in the object bytes
        let obj_end = b.len();
        let mut any_key_found = false;
        for (mk, _) in merge_pairs.iter() {
            let kb = mk.as_bytes();
            if kb.len() + 3 <= obj_end {
                let mut search_pos = pos + 1;
                while search_pos + kb.len() + 2 < obj_end {
                    if let Some(p) = memchr::memchr(b'"', &b[search_pos..obj_end]) {
                        let abs = search_pos + p;
                        if abs + 1 + kb.len() + 1 <= obj_end
                            && &b[abs + 1..abs + 1 + kb.len()] == kb
                            && b[abs + 1 + kb.len()] == b'"'
                        {
                            any_key_found = true;
                            break;
                        }
                        search_pos = abs + 1;
                    } else {
                        break;
                    }
                }
            }
            if any_key_found { break; }
        }
        if !any_key_found {
            // No merge key exists — copy object, append merge pairs before closing }
            let compact = is_json_compact(&b[pos..]);
            if compact {
                // Compact input: just find } and copy up to it
                let mut close = obj_end - 1;
                while close > pos && b[close] != b'}' { close -= 1; }
                if b[close] == b'}' {
                    buf.extend_from_slice(&b[pos..close]);
                    for (mk, mv) in merge_pairs.iter() {
                        buf.push(b',');
                        buf.push(b'"');
                        buf.extend_from_slice(mk.as_bytes());
                        buf.extend_from_slice(b"\":");
                        buf.extend_from_slice(mv);
                    }
                    buf.push(b'}');
                    return true;
                }
            } else {
                // Non-compact input: compact the whole object, then splice in merge pairs
                let start = buf.len();
                push_json_compact_raw(buf, &b[pos..]);
                // Find the closing } in the compacted output and insert merge pairs before it
                let end = buf.len();
                if end > start && buf[end - 1] == b'}' {
                    buf.pop(); // remove }
                    for (mk, mv) in merge_pairs.iter() {
                        buf.push(b',');
                        buf.push(b'"');
                        buf.extend_from_slice(mk.as_bytes());
                        buf.extend_from_slice(b"\":");
                        buf.extend_from_slice(mv);
                    }
                    buf.push(b'}');
                    return true;
                }
            }
        }
    }
    // Single-pass: scan keys, write output directly (no Vec allocations)
    let mut merge_emitted: u64 = 0;
    buf.push(b'{');
    let mut first = true;
    loop {
        if i >= b.len() || b[i] != b'"' { return false; }
        i += 1;
        let key_start = i;
        let mut has_escape = false;
        while i < b.len() {
            match b[i] { b'"' => break, b'\\' => { has_escape = true; i += 2; continue }, _ => i += 1 }
        }
        let key_end = i;
        let key_bytes = &b[key_start..key_end];
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        let val_start = i;
        i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return false };
        let val_end = i;
        // Check if this key matches any merge pair
        let mut replaced = false;
        if !has_escape {
            for (mi, (mk, mv)) in merge_pairs.iter().enumerate() {
                if key_bytes == mk.as_bytes() {
                    if !first { buf.push(b','); }
                    first = false;
                    buf.push(b'"');
                    buf.extend_from_slice(key_bytes);
                    buf.extend_from_slice(b"\":");
                    buf.extend_from_slice(mv);
                    merge_emitted |= 1u64 << mi;
                    replaced = true;
                    break;
                }
            }
        }
        if !replaced {
            if !first { buf.push(b','); }
            first = false;
            buf.push(b'"');
            buf.extend_from_slice(key_bytes);
            buf.extend_from_slice(b"\":");
            buf.extend_from_slice(&b[val_start..val_end]);
        }
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return false; }
        if b[i] == b'}' { break; }
        if b[i] != b',' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
    // Append any merge pairs that weren't replacements
    for (mi, (mk, mv)) in merge_pairs.iter().enumerate() {
        if (merge_emitted >> mi) & 1 == 0 {
            if !first { buf.push(b','); }
            first = false;
            buf.push(b'"');
            buf.extend_from_slice(mk.as_bytes());
            buf.extend_from_slice(b"\":");
            buf.extend_from_slice(mv);
        }
    }
    buf.push(b'}');
    true
}

/// Sort an object's keys alphabetically and write the result to `buf`.
/// Extracts (key_start, key_end, val_start, val_end) pairs, sorts by key bytes, reassembles.
/// `pairs` is a reusable scratch buffer to avoid allocations.
pub fn json_object_sort_keys(b: &[u8], pos: usize, buf: &mut Vec<u8>, pairs: &mut Vec<(usize, usize, usize, usize)>) -> bool {
    pairs.clear();
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' {
        buf.extend_from_slice(b"{}");
        return true;
    }
    loop {
        if i >= b.len() || b[i] != b'"' { return false; }
        let key_start = i;
        i += 1;
        while i < b.len() { match b[i] { b'"' => break, b'\\' => { i += 2; continue }, _ => i += 1 } }
        if i >= b.len() { return false; }
        let key_end = i + 1;
        i = key_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        let val_start = i;
        i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return false };
        let val_end = i;
        pairs.push((key_start, key_end, val_start, val_end));
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return false; }
        if b[i] == b'}' { break; }
        if b[i] != b',' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
    // Sort by key content (bytes between quotes)
    pairs.sort_unstable_by(|a, c| b[a.0+1..a.1-1].cmp(&b[c.0+1..c.1-1]));
    // Deduplicate: last occurrence wins (like jq)
    let mut deduped: Vec<usize> = Vec::with_capacity(pairs.len());
    for idx in 0..pairs.len() {
        if idx + 1 < pairs.len() && b[pairs[idx].0+1..pairs[idx].1-1] == b[pairs[idx+1].0+1..pairs[idx+1].1-1] {
            continue; // skip earlier duplicate
        }
        deduped.push(idx);
    }
    buf.push(b'{');
    for (di, &pidx) in deduped.iter().enumerate() {
        if di > 0 { buf.push(b','); }
        let (ks, ke, vs, ve) = pairs[pidx];
        buf.extend_from_slice(&b[ks..ke]);
        buf.push(b':');
        buf.extend_from_slice(&b[vs..ve]);
    }
    buf.push(b'}');
    true
}

/// Filter an object's entries by value type and rebuild.
/// `type_byte_match` is a function that checks if a value's first byte matches the desired type.
/// Returns the filtered object in compact JSON.
pub fn json_object_filter_by_value_type(b: &[u8], pos: usize, type_name: &str, buf: &mut Vec<u8>) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' {
        buf.extend_from_slice(b"{}");
        return true;
    }
    buf.push(b'{');
    let mut first = true;
    loop {
        if i >= b.len() || b[i] != b'"' { return false; }
        let key_start = i;
        i += 1;
        while i < b.len() { match b[i] { b'"' => break, b'\\' => { i += 2; continue }, _ => i += 1 } }
        if i >= b.len() { return false; }
        let key_end = i + 1;
        i = key_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        let val_start = i;
        i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return false };
        let val_end = i;
        // Check if value type matches
        let first_byte = b[val_start];
        let matches_type = match type_name {
            "number" => first_byte == b'-' || (first_byte >= b'0' && first_byte <= b'9'),
            "string" => first_byte == b'"',
            "array" => first_byte == b'[',
            "object" => first_byte == b'{',
            "boolean" => first_byte == b't' || first_byte == b'f',
            "null" => first_byte == b'n',
            _ => false,
        };
        if matches_type {
            if !first { buf.push(b','); }
            first = false;
            buf.extend_from_slice(&b[key_start..key_end]);
            buf.push(b':');
            buf.extend_from_slice(&b[val_start..val_end]);
        }
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return false; }
        if b[i] == b'}' { break; }
        if b[i] != b',' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
    buf.push(b'}');
    true
}

/// Iterate over values of a JSON object or elements of a JSON array,
/// calling `cb` with (value_start, value_end) for each. Returns true on success.
pub fn json_each_value_raw(b: &[u8], pos: usize, buf: &mut Vec<u8>) -> bool {
    if pos >= b.len() { return false; }
    match b[pos] {
        b'{' => {
            let mut i = pos + 1;
            while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
            if i < b.len() && b[i] == b'}' { return true; }
            let buf_start = buf.len();
            let mut seen: [(usize, usize); 16] = [(0, 0); 16];
            let mut seen_count: usize = 0;
            loop {
                if i >= b.len() || b[i] != b'"' { return false; }
                let key_start = i;
                i += 1;
                while i < b.len() { match b[i] { b'"' => break, b'\\' => { i += 2; continue }, _ => i += 1 } }
                if i >= b.len() { return false; }
                let key_end = i + 1;

                let key_bytes = &b[key_start + 1 .. key_end - 1];
                let mut is_dup = false;
                for j in 0..seen_count {
                    let (ks, ke) = seen[j];
                    if &b[ks + 1 .. ke - 1] == key_bytes { is_dup = true; break; }
                }
                if is_dup || seen_count >= seen.len() {
                    // Roll back the streaming emission and fall through
                    // to the allocating dedup helper. Last-wins-first-position
                    // dedup may overwrite earlier emissions with later
                    // values, so we can't keep the partial output.
                    buf.truncate(buf_start);
                    let mut pairs: Vec<(usize, usize, usize, usize)> = Vec::new();
                    if json_object_dedup_pairs(b, pos, &mut pairs).is_none() { return false; }
                    for (_, _, vs, ve) in &pairs {
                        buf.extend_from_slice(&b[*vs..*ve]);
                        buf.push(b'\n');
                    }
                    return true;
                }
                seen[seen_count] = (key_start, key_end);
                seen_count += 1;

                i = key_end;
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
                if i >= b.len() || b[i] != b':' { return false; }
                i += 1;
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
                let vs = i;
                i = match skip_json_value(b, i) { Ok(e) => e, Err(_) => return false };
                buf.extend_from_slice(&b[vs..i]);
                buf.push(b'\n');
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
                if i >= b.len() { return false; }
                if b[i] == b'}' { break; }
                if b[i] != b',' { return false; }
                i += 1;
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
            }
            true
        }
        b'[' => {
            let mut i = pos + 1;
            while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
            if i < b.len() && b[i] == b']' { return true; }
            loop {
                let vs = i;
                i = match skip_json_value(b, i) { Ok(e) => e, Err(_) => return false };
                buf.extend_from_slice(&b[vs..i]);
                buf.push(b'\n');
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
                if i >= b.len() { return false; }
                if b[i] == b']' { break; }
                if b[i] != b',' { return false; }
                i += 1;
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
            }
            true
        }
        _ => false,
    }
}

/// Iterator-style `.[]` that calls a closure for each value span.
/// Returns true if successful. The callback receives (value_start, value_end) byte offsets into `b`.
pub fn json_each_value_cb(b: &[u8], pos: usize, mut cb: impl FnMut(usize, usize)) -> bool {
    if pos >= b.len() { return false; }
    match b[pos] {
        b'{' => {
            let mut i = pos + 1;
            while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
            if i < b.len() && b[i] == b'}' { return true; }
            // Stack buffer of (k_start, k_end, v_start, v_end). Dedup
            // inline; on overflow or first detected duplicate, fall back
            // to `json_object_dedup_pairs`. The callback is *not*
            // invoked until we know all pairs are unique (or we have a
            // deduped list), since callbacks may have non-rollbackable
            // side effects.
            let mut stack: [(usize, usize, usize, usize); 16] = [(0, 0, 0, 0); 16];
            let mut count: usize = 0;
            loop {
                if i >= b.len() || b[i] != b'"' { return false; }
                let key_start = i;
                i += 1;
                while i < b.len() { match b[i] { b'"' => break, b'\\' => { i += 2; continue }, _ => i += 1 } }
                if i >= b.len() { return false; }
                let key_end = i + 1;

                let key_bytes = &b[key_start + 1 .. key_end - 1];
                let mut is_dup = false;
                for j in 0..count {
                    let (ks, ke, _, _) = stack[j];
                    if &b[ks + 1 .. ke - 1] == key_bytes { is_dup = true; break; }
                }
                if is_dup || count >= stack.len() {
                    let mut pairs: Vec<(usize, usize, usize, usize)> = Vec::new();
                    if json_object_dedup_pairs(b, pos, &mut pairs).is_none() { return false; }
                    for (_, _, vs, ve) in &pairs { cb(*vs, *ve); }
                    return true;
                }

                i = key_end;
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
                if i >= b.len() || b[i] != b':' { return false; }
                i += 1;
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
                let vs = i;
                i = match skip_json_value(b, i) { Ok(e) => e, Err(_) => return false };
                stack[count] = (key_start, key_end, vs, i);
                count += 1;
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
                if i >= b.len() { return false; }
                if b[i] == b'}' { break; }
                if b[i] != b',' { return false; }
                i += 1;
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
            }
            for j in 0..count { let (_, _, vs, ve) = stack[j]; cb(vs, ve); }
            true
        }
        b'[' => {
            let mut i = pos + 1;
            while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
            if i < b.len() && b[i] == b']' { return true; }
            loop {
                let vs = i;
                i = match skip_json_value(b, i) { Ok(e) => e, Err(_) => return false };
                cb(vs, i);
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
                if i >= b.len() { return false; }
                if b[i] == b']' { break; }
                if b[i] != b',' { return false; }
                i += 1;
                while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
            }
            true
        }
        _ => false,
    }
}

/// Convert a JSON object to `to_entries` format directly from raw bytes.
/// Produces `[{"key":"k1","value":v1},{"key":"k2","value":v2},...]\n`.
pub fn json_to_entries_raw(b: &[u8], pos: usize, buf: &mut Vec<u8>) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' {
        buf.extend_from_slice(b"[]\n");
        return true;
    }
    let buf_start = buf.len();
    let mut seen: [(usize, usize); 16] = [(0, 0); 16];
    let mut seen_count: usize = 0;
    buf.push(b'[');
    let mut first = true;
    loop {
        if i >= b.len() || b[i] != b'"' { return false; }
        let key_start = i;
        i += 1;
        while i < b.len() { match b[i] { b'"' => break, b'\\' => { i += 2; continue }, _ => i += 1 } }
        if i >= b.len() { return false; }
        let key_end = i + 1;

        let key_bytes = &b[key_start + 1 .. key_end - 1];
        let mut is_dup = false;
        for j in 0..seen_count {
            let (ks, ke) = seen[j];
            if &b[ks + 1 .. ke - 1] == key_bytes { is_dup = true; break; }
        }
        if is_dup || seen_count >= seen.len() {
            buf.truncate(buf_start);
            let mut pairs: Vec<(usize, usize, usize, usize)> = Vec::new();
            if json_object_dedup_pairs(b, pos, &mut pairs).is_none() { return false; }
            if pairs.is_empty() {
                buf.extend_from_slice(b"[]\n");
                return true;
            }
            buf.push(b'[');
            for (idx, (ks, ke, vs, ve)) in pairs.iter().enumerate() {
                if idx > 0 { buf.push(b','); }
                buf.extend_from_slice(b"{\"key\":");
                buf.extend_from_slice(&b[*ks..*ke]);
                buf.extend_from_slice(b",\"value\":");
                buf.extend_from_slice(&b[*vs..*ve]);
                buf.push(b'}');
            }
            buf.extend_from_slice(b"]\n");
            return true;
        }
        seen[seen_count] = (key_start, key_end);
        seen_count += 1;

        i = key_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        let vs = i;
        i = match skip_json_value(b, i) { Ok(e) => e, Err(_) => return false };
        if !first { buf.push(b','); }
        first = false;
        buf.extend_from_slice(b"{\"key\":");
        buf.extend_from_slice(&b[key_start..key_end]);
        buf.extend_from_slice(b",\"value\":");
        buf.extend_from_slice(&b[vs..i]);
        buf.push(b'}');
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return false; }
        if b[i] == b'}' { break; }
        if b[i] != b',' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
    buf.extend_from_slice(b"]\n");
    true
}

/// Filter an object's key-value pairs by a numeric comparison on the value.
/// `with_entries(select(.value CMP threshold))` as raw byte operation.
/// Emits the filtered object into `buf` followed by newline. Returns true on success.
pub fn json_with_entries_select_value_cmp(
    b: &[u8], pos: usize, cmp: u8, threshold: f64, buf: &mut Vec<u8>,
) -> bool {
    // cmp: b'>' = Gt, b'G' = Ge, b'<' = Lt, b'L' = Le, b'=' = Eq, b'!' = Ne
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' {
        buf.extend_from_slice(b"{}\n");
        return true;
    }
    buf.push(b'{');
    let mut first = true;
    loop {
        if i >= b.len() || b[i] != b'"' { return false; }
        let key_start = i; // includes opening quote
        i += 1;
        while i < b.len() { match b[i] { b'"' => break, b'\\' => { i += 2; continue }, _ => i += 1 } }
        if i >= b.len() { return false; }
        let key_end = i + 1; // includes closing quote
        i = key_end;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        let vs = i;
        i = match skip_json_value(b, i) { Ok(e) => e, Err(_) => return false };
        let ve = i;
        // Check value against threshold using jq type ordering:
        // null(0) < false(1) < true(2) < number(3) < string(4) < array(5) < object(6)
        let val_bytes = &b[vs..ve];
        let include = if let Some(num) = parse_json_num(val_bytes) {
            match cmp {
                b'>' => num > threshold,
                b'G' => num >= threshold,
                b'<' => num < threshold,
                b'L' => num <= threshold,
                b'=' => num == threshold,
                b'!' => num != threshold,
                _ => false,
            }
        } else if !val_bytes.is_empty() {
            // Determine type order relative to number(3)
            // type > number: string("), array([), object({)
            // type < number: null(n), false(f), true(t)
            let type_gt_num = matches!(val_bytes[0], b'"' | b'[' | b'{');
            let type_lt_num = matches!(val_bytes[0], b'n' | b'f' | b't');
            match cmp {
                b'>' => type_gt_num,
                b'G' => type_gt_num,
                b'<' => type_lt_num,
                b'L' => type_lt_num,
                b'=' => false, // different type never equal to number
                b'!' => true,  // different type always != number
                _ => false,
            }
        } else {
            false
        };
        if include {
            if !first { buf.push(b','); }
            first = false;
            buf.extend_from_slice(&b[key_start..key_end]);
            buf.push(b':');
            buf.extend_from_slice(val_bytes);
        }
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return false; }
        if b[i] == b'}' { break; }
        if b[i] != b',' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
    buf.extend_from_slice(b"}\n");
    true
}

/// Set a field in a JSON object to a constant value (raw bytes).
/// `.field = value` or `setpath(["field"]; value)` as raw byte operation.
/// `new_val` is the raw JSON bytes of the new value (e.g. b"99").
/// Preserves key ordering: if field exists, replaces value in-place; otherwise appends.
pub fn json_object_set_field_raw(
    b: &[u8], pos: usize, field: &str, new_val: &[u8], buf: &mut Vec<u8>,
) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let field_bytes = field.as_bytes();
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' {
        // Empty object: just create {"field":val}
        buf.push(b'{');
        buf.push(b'"');
        buf.extend_from_slice(field_bytes);
        buf.extend_from_slice(b"\":");
        buf.extend_from_slice(new_val);
        buf.extend_from_slice(b"}\n");
        return true;
    }
    buf.push(b'{');
    let mut first = true;
    let mut found = false;
    loop {
        if i >= b.len() || b[i] != b'"' { return false; }
        let key_start = i + 1;
        i += 1;
        let mut j = i;
        while j < b.len() { match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 } }
        if j >= b.len() { return false; }
        let key_matches = (j - key_start) == field_bytes.len()
            && b[key_start..j] == *field_bytes;
        let key_q_start = key_start - 1; // includes opening quote
        let key_q_end = j + 1; // includes closing quote
        i = j + 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        let vs = i;
        i = match skip_json_value(b, i) { Ok(e) => e, Err(_) => return false };
        if !first { buf.push(b','); }
        first = false;
        buf.extend_from_slice(&b[key_q_start..key_q_end]);
        buf.push(b':');
        if key_matches {
            buf.extend_from_slice(new_val);
            found = true;
        } else {
            buf.extend_from_slice(&b[vs..i]);
        }
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return false; }
        if b[i] == b'}' { break; }
        if b[i] != b',' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
    if !found {
        // Append new field
        buf.push(b',');
        buf.push(b'"');
        for &byte in field_bytes {
            match byte {
                b'"' => buf.extend_from_slice(b"\\\""),
                b'\\' => buf.extend_from_slice(b"\\\\"),
                _ => buf.push(byte),
            }
        }
        buf.extend_from_slice(b"\":");
        buf.extend_from_slice(new_val);
    }
    buf.extend_from_slice(b"}\n");
    true
}

/// Update a string field in a JSON object with ascii case conversion.
/// Scans the object, copies all key-value pairs, and applies case conversion
/// to the target field's string value. Returns false if field not found or not a string.
pub fn json_object_update_field_case(
    b: &[u8], pos: usize, field: &str, is_upcase: bool, buf: &mut Vec<u8>,
) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let field_bytes = field.as_bytes();
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { return false; } // empty object, field not found
    buf.push(b'{');
    let mut first = true;
    let mut found = false;
    loop {
        if i >= b.len() || b[i] != b'"' { return false; }
        let key_start = i + 1;
        i += 1;
        while i < b.len() { match b[i] { b'"' => break, b'\\' => { i += 2; continue }, _ => i += 1 } }
        if i >= b.len() { return false; }
        let key_matches = (i - key_start) == field_bytes.len()
            && b[key_start..i] == *field_bytes;
        let key_q_start = key_start - 1;
        let key_q_end = i + 1;
        i = i + 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        let vs = i;
        i = match skip_json_value(b, i) { Ok(e) => e, Err(_) => return false };
        if !first { buf.push(b','); }
        first = false;
        buf.extend_from_slice(&b[key_q_start..key_q_end]);
        buf.push(b':');
        if key_matches && vs < b.len() && b[vs] == b'"' {
            // Apply case conversion to string value
            buf.push(b'"');
            let mut j = vs + 1;
            while j < i {
                if b[j] == b'"' { break; }
                if b[j] == b'\\' {
                    buf.push(b'\\');
                    j += 1;
                    if j < i { buf.push(b[j]); j += 1; }
                    continue;
                }
                if is_upcase {
                    buf.push(if b[j] >= b'a' && b[j] <= b'z' { b[j] - 32 } else { b[j] });
                } else {
                    buf.push(if b[j] >= b'A' && b[j] <= b'Z' { b[j] + 32 } else { b[j] });
                }
                j += 1;
            }
            buf.push(b'"');
            found = true;
        } else {
            buf.extend_from_slice(&b[vs..i]);
        }
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return false; }
        if b[i] == b'}' { break; }
        if b[i] != b',' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    }
    if !found { return false; }
    buf.extend_from_slice(b"}\n");
    true
}

/// Extract two numeric field values from a JSON object without full parsing.
/// More efficient than calling json_object_get_num twice (single scan).
pub fn json_object_get_two_nums(b: &[u8], pos: usize, field1: &str, field2: &str) -> Option<(f64, f64)> {
    if pos >= b.len() || b[pos] != b'{' { return None; }
    // Same field optimization: just need one lookup
    if field1 == field2 {
        if let Some(v) = json_object_get_num(b, pos, field1) {
            return Some((v, v));
        }
        return None;
    }
    let f1 = field1.as_bytes();
    let f2 = field2.as_bytes();
    // jq dedupes duplicate input keys last-wins (#233 / #325 / #360):
    // walk the entire object and overwrite each `val*` slot whenever a
    // later duplicate arrives. The pre-#371 version exited on the first
    // match for each field, returning a first-wins read that disagreed
    // with jq for inputs like `{"a":5,"a":1,"b":3}` (#371).
    //
    // After both targets are recorded, the in-loop
    // `no_target_first_byte_in_remainder` proof (#410 / #422) lets us
    // early-exit when neither key's first byte still appears in the
    // remaining bytes — first-wins agrees with last-wins exactly when
    // no further occurrence is possible. The check runs at most once
    // per call and is gated on `pos == 0` so nested calls do not
    // scan into unrelated sibling data.
    let allow_early_exit = pos == 0 && !f1.is_empty() && !f2.is_empty();
    let mut early_exit_attempted = false;
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
        let match1 = key_len == f1.len() && b[key_start..j] == *f1;
        let match2 = !match1 && key_len == f2.len() && b[key_start..j] == *f2;
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
        } else {
            i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return None };
        }
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return None; }
        // End of object — `val1` / `val2` are now their last-wins
        // values across the entire scan (#371). Return Some only when
        // both fields appeared at least once.
        if b[i] == b'}' {
            return match (val1, val2) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => None,
            };
        }
        if b[i] != b',' { return None; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        // Both targets recorded and there are more keys ahead → run
        // the cheap remainder proof. If neither key's first byte
        // appears in the remaining bytes, no duplicate is possible
        // and first-wins agrees with last-wins (#410 / #422). The
        // proof is gated on `allow_early_exit` (top-level only) and
        // runs at most once per call.
        if allow_early_exit && !early_exit_attempted {
            if let (Some(a), Some(b_val)) = (val1, val2) {
                early_exit_attempted = true;
                if no_target_first_byte_in_remainder(b, i, &[f1[0], f2[0]]) {
                    return Some((a, b_val));
                }
            }
        }
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

/// Like json_object_get_fields_raw but writes into a pre-allocated `&mut [(usize, usize)]`.
/// Returns true if all fields were found; false otherwise.
/// `out` must have length >= `input_fields.len()`.
pub fn json_object_get_fields_raw_buf(b: &[u8], pos: usize, input_fields: &[&str], out: &mut [(usize, usize)]) -> bool {
    if pos >= b.len() || b[pos] != b'{' { return false; }
    let n = input_fields.len();
    debug_assert!(out.len() >= n);
    // jq dedupes duplicate input keys last-wins (#233 / #325): the
    // helper cannot exit early on the first match — a later duplicate
    // may rebind the same key. Walk to the end of the object,
    // overwriting each `out[idx]` whenever a later match arrives.
    // The early-exit on `found == n` was removed for #325; the cost
    // is a slight regression on object-construct fast paths that read
    // a small subset of fields from larger inputs, but the value-level
    // path was already this expensive — every call site routes
    // through here for value-level coverage parity.
    //
    // After all requested fields are matched, the in-loop
    // `no_target_first_byte_in_remainder` proof (#410 / #422) gates a
    // `found == n` exit. The proof scans only the trailing bytes, so
    // it stays cheap on NDJSON shapes where the last requested key
    // sits at the end of the row and the upfront-pre-check variant
    // would have paid full-row scans for zero savings.
    let mut first_bytes_small: [u8; 16] = [0; 16];
    let allow_early_exit = pos == 0 && n > 0 && n <= first_bytes_small.len() && {
        let mut all_non_empty = true;
        for (i, f) in input_fields.iter().enumerate() {
            let fb = f.as_bytes();
            if fb.is_empty() { all_non_empty = false; break; }
            first_bytes_small[i] = fb[0];
        }
        all_non_empty
    };
    let mut early_exit_attempted = false;
    let mut found_mask: u64 = 0;
    let mut i = pos + 1;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
    if i < b.len() && b[i] == b'}' { return false; }
    loop {
        if i >= b.len() || b[i] != b'"' { return false; }
        let key_start = i + 1;
        let mut j = key_start;
        while j < b.len() {
            match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 }
        }
        let key_len = j - key_start;
        let mut matched_idx = None;
        for (idx, field) in input_fields.iter().enumerate() {
            if key_len == field.len() && b[key_start..j] == *field.as_bytes() {
                matched_idx = Some(idx);
                break;
            }
        }
        i = j + 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() || b[i] != b':' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if let Some(idx) = matched_idx {
            let val_start = i;
            let val_end = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return false };
            out[idx] = (val_start, val_end);
            found_mask |= 1u64 << idx;
            i = val_end;
        } else {
            i = match skip_json_value(b, i) { Ok(end) => end, Err(_) => return false };
        }
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        if i >= b.len() { return false; }
        if b[i] == b'}' {
            return found_mask.count_ones() as usize == n;
        }
        if b[i] != b',' { return false; }
        i += 1;
        while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
        // All requested fields seen and there are more keys ahead →
        // run the cheap remainder proof. If no requested key's first
        // byte appears in the remaining bytes, no duplicate is
        // possible and first-wins agrees with last-wins (#410 /
        // #422). Skipping this when the next byte was `}` keeps the
        // last-key-in-row case (e.g. `[.name, .x, .y] | @csv` on
        // `{"x":…,"y":…,"name":…}`) free of memchr overhead. Run at
        // most once per call.
        if allow_early_exit
            && !early_exit_attempted
            && found_mask.count_ones() as usize == n
        {
            early_exit_attempted = true;
            if no_target_first_byte_in_remainder(b, i, &first_bytes_small[..n]) {
                return true;
            }
        }
    }
}

/// Walk JSON bytes, applying a numeric transformation to all number values.
/// Copies the JSON structure verbatim but replaces each number with op(number, operand).
/// Returns true on success, false if the JSON is malformed.
pub fn walk_json_transform_nums(buf: &mut Vec<u8>, b: &[u8], op: u8, operand: f64) -> bool {
    walk_json_nums_inner(buf, b, &mut 0, op, operand)
}

fn walk_json_nums_inner(buf: &mut Vec<u8>, b: &[u8], pos: &mut usize, op: u8, operand: f64) -> bool {
    // Skip whitespace
    while *pos < b.len() && matches!(b[*pos], b' ' | b'\t' | b'\n' | b'\r') { *pos += 1; }
    if *pos >= b.len() { return false; }
    match b[*pos] {
        b'"' => {
            // String: copy verbatim
            let start = *pos;
            *pos += 1;
            while *pos < b.len() {
                match b[*pos] {
                    b'"' => { *pos += 1; buf.extend_from_slice(&b[start..*pos]); return true; }
                    b'\\' => { *pos += 2; }
                    _ => { *pos += 1; }
                }
            }
            false
        }
        b'{' => {
            buf.push(b'{');
            *pos += 1;
            while *pos < b.len() && matches!(b[*pos], b' ' | b'\t' | b'\n' | b'\r') { *pos += 1; }
            if *pos < b.len() && b[*pos] == b'}' { *pos += 1; buf.push(b'}'); return true; }
            let mut first = true;
            loop {
                if !first { buf.push(b','); }
                first = false;
                // Key
                if !walk_json_nums_inner(buf, b, pos, op, operand) { return false; }
                while *pos < b.len() && matches!(b[*pos], b' ' | b'\t' | b'\n' | b'\r') { *pos += 1; }
                if *pos >= b.len() || b[*pos] != b':' { return false; }
                *pos += 1;
                buf.push(b':');
                // Value
                if !walk_json_nums_inner(buf, b, pos, op, operand) { return false; }
                while *pos < b.len() && matches!(b[*pos], b' ' | b'\t' | b'\n' | b'\r') { *pos += 1; }
                if *pos >= b.len() { return false; }
                if b[*pos] == b'}' { *pos += 1; buf.push(b'}'); return true; }
                if b[*pos] != b',' { return false; }
                *pos += 1;
                while *pos < b.len() && matches!(b[*pos], b' ' | b'\t' | b'\n' | b'\r') { *pos += 1; }
            }
        }
        b'[' => {
            buf.push(b'[');
            *pos += 1;
            while *pos < b.len() && matches!(b[*pos], b' ' | b'\t' | b'\n' | b'\r') { *pos += 1; }
            if *pos < b.len() && b[*pos] == b']' { *pos += 1; buf.push(b']'); return true; }
            let mut first = true;
            loop {
                if !first { buf.push(b','); }
                first = false;
                if !walk_json_nums_inner(buf, b, pos, op, operand) { return false; }
                while *pos < b.len() && matches!(b[*pos], b' ' | b'\t' | b'\n' | b'\r') { *pos += 1; }
                if *pos >= b.len() { return false; }
                if b[*pos] == b']' { *pos += 1; buf.push(b']'); return true; }
                if b[*pos] != b',' { return false; }
                *pos += 1;
                while *pos < b.len() && matches!(b[*pos], b' ' | b'\t' | b'\n' | b'\r') { *pos += 1; }
            }
        }
        b't' => {
            if b.len() - *pos >= 4 && &b[*pos..*pos+4] == b"true" {
                buf.extend_from_slice(b"true"); *pos += 4; true
            } else { false }
        }
        b'f' => {
            if b.len() - *pos >= 5 && &b[*pos..*pos+5] == b"false" {
                buf.extend_from_slice(b"false"); *pos += 5; true
            } else { false }
        }
        b'n' => {
            if b.len() - *pos >= 4 && &b[*pos..*pos+4] == b"null" {
                buf.extend_from_slice(b"null"); *pos += 4; true
            } else { false }
        }
        b'-' | b'0'..=b'9' => {
            // Number: parse, transform, format
            let start = *pos;
            if b[*pos] == b'-' { *pos += 1; }
            while *pos < b.len() && b[*pos].is_ascii_digit() { *pos += 1; }
            if *pos < b.len() && b[*pos] == b'.' {
                *pos += 1;
                while *pos < b.len() && b[*pos].is_ascii_digit() { *pos += 1; }
            }
            if *pos < b.len() && (b[*pos] == b'e' || b[*pos] == b'E') {
                *pos += 1;
                if *pos < b.len() && (b[*pos] == b'+' || b[*pos] == b'-') { *pos += 1; }
                while *pos < b.len() && b[*pos].is_ascii_digit() { *pos += 1; }
            }
            let num_str = unsafe { std::str::from_utf8_unchecked(&b[start..*pos]) };
            if let Ok(n) = fast_float::parse::<f64, _>(num_str) {
                let result = match op {
                    b'+' => n + operand,
                    b'-' => n - operand,
                    b'*' => n * operand,
                    b'/' => n / operand,
                    _ => n,
                };
                push_jq_number_bytes(buf, result);
                true
            } else {
                buf.extend_from_slice(&b[start..*pos]);
                true
            }
        }
        _ => false,
    }
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

/// Single-pass tojson: strip whitespace + escape `"` and `\` into a JSON string.
/// Writes `"<escaped_compact_json>"` to buf (no trailing newline).
///
/// Numeric tokens are renormalised through [`normalize_jq_repr`] so the
/// fast path matches jq 1.8.1's canonical scientific form (uppercase `E`,
/// explicit `+`/`-` sign — issue #190 items 1 and 2).
pub fn push_tojson_raw(buf: &mut Vec<u8>, b: &[u8]) {
    buf.push(b'"');
    let mut i = 0;
    let len = b.len();
    while i < len {
        match b[i] {
            b' ' | b'\t' | b'\n' | b'\r' => { i += 1; }
            b'"' => {
                buf.extend_from_slice(b"\\\"");
                i += 1;
                // Inside JSON string: scan chunks between special chars.
                // Also break at 0x7F so it can be escaped as `` —
                // jq escapes DEL the same way it escapes U+0000..U+001F
                // (#446). Raw 0x00..0x1F never reach this path because the
                // input parser rejects unescaped control chars.
                loop {
                    let chunk_start = i;
                    while i < len && b[i] != b'"' && b[i] != b'\\' && b[i] != 0x7F { i += 1; }
                    if i > chunk_start { buf.extend_from_slice(&b[chunk_start..i]); }
                    if i >= len { break; }
                    if b[i] == b'"' {
                        buf.extend_from_slice(b"\\\"");
                        i += 1;
                        break;
                    }
                    if b[i] == 0x7F {
                        buf.extend_from_slice(b"\\\\u007f");
                        i += 1;
                        continue;
                    }
                    // backslash escape sequence
                    buf.extend_from_slice(b"\\\\");
                    i += 1;
                    if i < len {
                        if b[i] == b'"' { buf.extend_from_slice(b"\\\""); }
                        else if b[i] == b'\\' { buf.extend_from_slice(b"\\\\"); }
                        else { buf.push(b[i]); }
                        i += 1;
                    }
                }
            }
            b'-' | b'0'..=b'9' => {
                let start = i;
                if b[i] == b'-' { i += 1; }
                while i < len && b[i].is_ascii_digit() { i += 1; }
                if i < len && b[i] == b'.' {
                    i += 1;
                    while i < len && b[i].is_ascii_digit() { i += 1; }
                }
                if i < len && (b[i] == b'e' || b[i] == b'E') {
                    i += 1;
                    if i < len && (b[i] == b'+' || b[i] == b'-') { i += 1; }
                    while i < len && b[i].is_ascii_digit() { i += 1; }
                }
                let num_bytes = &b[start..i];
                // SAFETY: scanned bytes are ASCII digits/sign/dot/e/E/+/-.
                let num_str = unsafe { std::str::from_utf8_unchecked(num_bytes) };
                if let Some(canonical) = normalize_jq_repr(num_str) {
                    buf.extend_from_slice(canonical.as_bytes());
                } else {
                    buf.extend_from_slice(num_bytes);
                }
            }
            c => { buf.push(c); i += 1; }
        }
    }
    buf.push(b'"');
}

/// Pretty-print raw JSON bytes without parsing into Value.
/// Produces jq-compatible indented output with trailing newline.
pub fn push_json_pretty_raw(buf: &mut Vec<u8>, b: &[u8], indent_n: usize, use_tab: bool) {
    push_json_pretty_raw_at(buf, b, indent_n, use_tab, 0);
}

/// Pretty-print raw JSON bytes with a base indentation depth.
pub fn push_json_pretty_raw_at(buf: &mut Vec<u8>, b: &[u8], indent_n: usize, use_tab: bool, base_depth: usize) {
    // jq's --tab uses exactly one tab per indent level regardless of --indent.
    let indent_n = if use_tab { 1 } else { indent_n };
    let mut depth: usize = base_depth;
    let mut i = 0;
    let len = b.len();
    let indent_char = if use_tab { b'\t' } else { b' ' };
    while i < len {
        match b[i] {
            b' ' | b'\t' | b'\n' | b'\r' => { i += 1; }
            b'{' => {
                i += 1;
                // Skip whitespace to check for empty object
                while i < len && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
                if i < len && b[i] == b'}' {
                    buf.extend_from_slice(b"{}");
                    i += 1;
                } else {
                    buf.push(b'{');
                    depth += 1;
                    buf.push(b'\n');
                    for _ in 0..depth * indent_n { buf.push(indent_char); }
                }
            }
            b'[' => {
                i += 1;
                while i < len && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') { i += 1; }
                if i < len && b[i] == b']' {
                    buf.extend_from_slice(b"[]");
                    i += 1;
                } else {
                    buf.push(b'[');
                    depth += 1;
                    buf.push(b'\n');
                    for _ in 0..depth * indent_n { buf.push(indent_char); }
                }
            }
            b'}' => {
                if depth > 0 { depth -= 1; }
                buf.push(b'\n');
                for _ in 0..depth * indent_n { buf.push(indent_char); }
                buf.push(b'}');
                i += 1;
            }
            b']' => {
                if depth > 0 { depth -= 1; }
                buf.push(b'\n');
                for _ in 0..depth * indent_n { buf.push(indent_char); }
                buf.push(b']');
                i += 1;
            }
            b',' => {
                buf.push(b',');
                buf.push(b'\n');
                for _ in 0..depth * indent_n { buf.push(indent_char); }
                i += 1;
            }
            b':' => {
                buf.extend_from_slice(b": ");
                i += 1;
            }
            b'"' => {
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
            _ => {
                // Numbers, true, false, null — copy until structural char or whitespace
                let start = i;
                while i < len && !matches!(b[i], b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
                    i += 1;
                }
                buf.extend_from_slice(&b[start..i]);
            }
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

/// Walk a JSON value at `pos` checking that every nested object has unique
/// keys. Returns `Some(end_pos)` on success, `None` if a duplicate key is
/// found anywhere inside the value or if the input is malformed.
///
/// Used by [`json_value_has_duplicate_keys`] to route input objects with
/// duplicate keys (#233) through the Value-level path that dedupes at parse
/// time. The malformed-input case is folded into `None` too: routing through
/// the Value path then surfaces the jq-compatible parse error from
/// [`json_to_value`], which is exactly what the user wants.
#[inline]
fn check_value_dedup_clean(b: &[u8], pos: usize) -> Option<usize> {
    if pos >= b.len() { return None; }
    let p = skip_ws(b, pos);
    if p >= b.len() { return None; }
    match b[p] {
        b'{' => check_object_dedup_clean(b, p),
        b'[' => check_array_dedup_clean(b, p),
        _ => skip_json_value(b, p).ok(),
    }
}

fn check_object_dedup_clean(b: &[u8], pos: usize) -> Option<usize> {
    debug_assert_eq!(b[pos], b'{');
    let mut i = skip_ws(b, pos + 1);
    if i < b.len() && b[i] == b'}' { return Some(i + 1); }
    // Small linear-scan dedup set. Object key counts are typically small;
    // when we cross a threshold we fall back to a HashSet allocation.
    let mut keys_small: [(usize, usize); 8] = [(0, 0); 8];
    let mut keys_small_len: usize = 0;
    let mut keys_big: Option<std::collections::HashSet<Vec<u8>>> = None;
    loop {
        if i >= b.len() || b[i] != b'"' { return None; }
        let key_start = i + 1;
        let mut j = key_start;
        while j < b.len() {
            match b[j] { b'"' => break, b'\\' => { j += 2; continue }, _ => j += 1 }
        }
        if j >= b.len() { return None; }
        let key_bytes = &b[key_start..j];
        if let Some(set) = keys_big.as_mut() {
            if !set.insert(key_bytes.to_vec()) { return None; }
        } else {
            for k in 0..keys_small_len {
                let (ks, ke) = keys_small[k];
                if (ke - ks) == key_bytes.len() && &b[ks..ke] == key_bytes { return None; }
            }
            if keys_small_len < keys_small.len() {
                keys_small[keys_small_len] = (key_start, j);
                keys_small_len += 1;
            } else {
                let mut set: std::collections::HashSet<Vec<u8>> =
                    keys_small.iter().map(|&(ks, ke)| b[ks..ke].to_vec()).collect();
                set.insert(key_bytes.to_vec());
                keys_big = Some(set);
            }
        }
        i = j + 1;
        i = skip_ws(b, i);
        if i >= b.len() || b[i] != b':' { return None; }
        i = skip_ws(b, i + 1);
        // Inline the value-shape dispatch: shallow objects with primitive
        // values dominate input streams in practice, so the recursive
        // `check_value_dedup_clean` indirection is worth eliding on the
        // primitive branch. The compiler rarely inlines across separate fns
        // when one of them has the HashSet allocation in its body.
        if i >= b.len() { return None; }
        i = match b[i] {
            b'{' => check_object_dedup_clean(b, i)?,
            b'[' => check_array_dedup_clean(b, i)?,
            _ => skip_json_value(b, i).ok()?,
        };
        i = skip_ws(b, i);
        if i >= b.len() { return None; }
        if b[i] == b'}' { return Some(i + 1); }
        if b[i] != b',' { return None; }
        i = skip_ws(b, i + 1);
    }
}

fn check_array_dedup_clean(b: &[u8], pos: usize) -> Option<usize> {
    debug_assert_eq!(b[pos], b'[');
    let mut i = skip_ws(b, pos + 1);
    if i < b.len() && b[i] == b']' { return Some(i + 1); }
    loop {
        if i >= b.len() { return None; }
        // Same primitive-inlining trick as in `check_object_dedup_clean`.
        i = match b[i] {
            b'{' => check_object_dedup_clean(b, i)?,
            b'[' => check_array_dedup_clean(b, i)?,
            _ => skip_json_value(b, i).ok()?,
        };
        i = skip_ws(b, i);
        if i >= b.len() { return None; }
        if b[i] == b']' { return Some(i + 1); }
        if b[i] != b',' { return None; }
        i = skip_ws(b, i + 1);
    }
}

/// True if the JSON value in `bytes` contains an object — at any nesting
/// depth — with duplicate keys.
///
/// Intended as a routing check for raw-byte fast paths: jq dedupes input
/// objects last-wins at parse time, so any path that copies input bytes
/// verbatim diverges on duplicate-key inputs (#233). When this returns
/// `true`, callers should route through the Value-level path
/// ([`json_to_value`] / [`parse_json_object`]) which handles dedup.
///
/// Also returns `true` for malformed input so the caller routes through
/// [`json_to_value`], whose error message is jq-compatible — better than
/// having a raw-byte fast path swallow the malformed bytes.
pub fn json_value_has_duplicate_keys(bytes: &[u8]) -> bool {
    let p = skip_ws(bytes, 0);
    if p >= bytes.len() { return false; }
    // Only objects (and containers that may contain objects) need scanning.
    match bytes[p] {
        b'{' => check_object_dedup_clean(bytes, p).is_none(),
        b'[' => check_array_dedup_clean(bytes, p).is_none(),
        _ => false,
    }
}

/// Stream-level variant of [`json_value_has_duplicate_keys`]: walk every
/// JSON value in a (possibly NDJSON-style) byte stream and return `true` if
/// any of them contains an object with duplicate keys.
///
/// Used by the whole-file identity shortcut to gate the byte-copy path —
/// duplicate keys would otherwise survive verbatim and diverge from jq.
pub fn json_stream_has_duplicate_keys(content: &[u8]) -> bool {
    let mut pos = 0;
    if content.len() >= 3 && content[0] == 0xEF && content[1] == 0xBB && content[2] == 0xBF {
        pos = 3;
    }
    pos = skip_ws(content, pos);
    while pos < content.len() {
        match check_value_dedup_clean(content, pos) {
            Some(end) => {
                pos = skip_ws(content, end);
            }
            None => return true,
        }
    }
    false
}

#[inline(always)]
fn skip_ws(b: &[u8], mut pos: usize) -> usize {
    while pos < b.len() && matches!(b[pos], b' ' | b'\t' | b'\n' | b'\r') { pos += 1; }
    pos
}

const MAX_JSON_DEPTH: usize = 10000;

/// Skip past a JSON value without constructing it. Returns the position after the value.
pub fn skip_json_value(b: &[u8], pos: usize) -> Result<usize> {
    if pos >= b.len() { bail!("Unexpected end of JSON"); }
    match b[pos] {
        b'n' => {
            if b.get(pos..pos+4) == Some(b"null") { Ok(pos + 4) }
            else if b.get(pos..pos+3) == Some(b"nan") { Ok(pos + 3) }
            else { bail!("Invalid JSON at position {}", pos) }
        }
        b't' => { if b.get(pos..pos+4) == Some(b"true") { Ok(pos + 4) } else { bail!("Invalid JSON at position {}", pos) } }
        b'f' => { if b.get(pos..pos+5) == Some(b"false") { Ok(pos + 5) } else { bail!("Invalid JSON at position {}", pos) } }
        b'N' => { if b.get(pos..pos+3) == Some(b"NaN") { Ok(pos + 3) } else { bail!("Invalid JSON at position {}", pos) } }
        b'I' => { if b.get(pos..pos+8) == Some(b"Infinity") { Ok(pos + 8) } else { bail!("Invalid JSON at position {}", pos) } }
        b'"' => {
            // Validate the same way parse_json_string does — raw U+0000..U+001F
            // inside a string is illegal JSON (RFC 8259). The fast path uses
            // memchr2 to jump to the next `"`/`\\` and only then sweeps the
            // in-between run for control chars (auto-vectorises to SIMD).
            let mut i = pos + 1;
            loop {
                let rest = &b[i..];
                match memchr::memchr2(b'"', b'\\', rest) {
                    Some(offset) => {
                        if rest[..offset].iter().any(|&c| c < 0x20) {
                            bail!("Invalid string: control characters from U+0000 through U+001F must be escaped");
                        }
                        if rest[offset] == b'"' { return Ok(i + offset + 1); }
                        // '\' : consume escape pair
                        if i + offset + 1 >= b.len() { bail!("Unterminated string escape"); }
                        i += offset + 2;
                    }
                    None => bail!("Unterminated string"),
                }
            }
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
        b'-' | b'+' | b'0'..=b'9' => {
            let mut i = pos;
            if b[i] == b'-' {
                // -Infinity and -NaN are accepted by parse_json_value; match here too.
                if b.get(pos..pos+9) == Some(b"-Infinity") { return Ok(pos + 9); }
                if b.get(pos..pos+4) == Some(b"-NaN") { return Ok(pos + 4); }
                i += 1;
            } else if b[i] == b'+' {
                i += 1;
            }
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
        for &f in fields { map.push_unique(KeyStr::from(f), Value::Null); }
        return Ok((Value::object_from_map(map), i + 1));
    }
    let mut found = 0u64;
    loop {
        if i >= b.len() || b[i] != b'"' { bail!("Expected string key at position {}", i); }
        // Scan key bytes directly without creating KeyStr for non-matching keys
        let key_start = i + 1;
        let mut j = key_start;
        let mut has_escape = false;
        while j < b.len() {
            match b[j] { b'"' => break, b'\\' => { has_escape = true; j += 2; continue }, _ => j += 1 }
        }
        let key_end = j; // exclusive, points at closing quote
        let key_byte_end = j + 1;
        i = skip_ws(b, key_byte_end);
        if i >= b.len() || b[i] != b':' { bail!("Expected ':' at position {}", i); }
        i = skip_ws(b, i + 1);
        // Compare key bytes against projection fields (avoid KeyStr allocation for non-matches).
        // Last-wins on duplicate input keys (#233): always re-match and `insert` to overwrite.
        let mut matched = false;
        if !has_escape {
            let key_len = key_end - key_start;
            let key_bytes = &b[key_start..key_end];
            for (fi, &f) in fields.iter().enumerate() {
                if key_len == f.len() && key_bytes == f.as_bytes() {
                    let key = KeyStr::from(unsafe { std::str::from_utf8_unchecked(key_bytes) });
                    let (val, end) = parse_json_value(b, i, depth + 1)?;
                    map.insert(key, val);
                    found |= 1u64 << fi.min(63);
                    i = end;
                    matched = true;
                    break;
                }
            }
        } else {
            // Escaped key: use full parser for correct handling
            let (key, _) = parse_json_key(b, key_start - 1)?;
            let key_str = key.as_str();
            for (fi, &f) in fields.iter().enumerate() {
                if key_str == f {
                    let (val, end) = parse_json_value(b, i, depth + 1)?;
                    map.insert(key, val);
                    found |= 1u64 << fi.min(63);
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
    }
    for (fi, &f) in fields.iter().enumerate() {
        if fi < 64 && (found & (1u64 << fi)) != 0 { continue; }
        map.push_unique(KeyStr::from(f), Value::Null);
    }
    Ok((Value::object_from_map(map), i + 1))
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
    if depth >= MAX_JSON_DEPTH { bail!("Exceeds depth limit for parsing"); }
    match b[pos] {
        b'n' => {
            if b.get(pos..pos+4) == Some(b"null") { Ok((Value::Null, pos + 4)) }
            else if b.get(pos..pos+3) == Some(b"nan") { Ok((Value::number(f64::NAN), pos + 3)) }
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
            if b.get(pos..pos+3) == Some(b"NaN") { Ok((Value::number(f64::NAN), pos + 3)) }
            else { bail!("Invalid JSON at position {}", pos) }
        }
        b'I' => {
            if b.get(pos..pos+8) == Some(b"Infinity") { Ok((Value::number(f64::INFINITY), pos + 8)) }
            else { bail!("Invalid JSON at position {}", pos) }
        }
        b'"' => parse_json_string(b, pos),
        b'[' => parse_json_array(b, pos, depth),
        b'{' => parse_json_object(b, pos, depth),
        b'+' => {
            // jq extension: leading '+' is ignored and the rest is parsed as a positive number.
            parse_json_number(b, pos)
        }
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
                    return Ok((Value::number(n as f64), k));
                }
                // Has decimal/exponent — fall through to full parser
                parse_json_number(b, pos)
            }
            // -Infinity, -NaN, or other
            else if b.get(pos..pos+9) == Some(b"-Infinity") { Ok((Value::number(f64::NEG_INFINITY), pos + 9)) }
            else if b.get(pos..pos+4) == Some(b"-NaN") { Ok((Value::number(f64::NAN), pos + 4)) }
            else { parse_json_number(b, pos) }
        }
        b'0' => {
            // Fast path for zero or numbers starting with 0 (must be just "0" unless "0." etc)
            let j = pos + 1;
            if j >= b.len() || (b[j] != b'.' && b[j] != b'e' && b[j] != b'E' && !b[j].is_ascii_digit()) {
                Ok((Value::number(0.0), j))
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
                Ok((Value::number(n as f64), j))
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
            c if c < 0x20 => bail!("Invalid string: control characters from U+0000 through U+001F must be escaped"),
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
                        if i + 4 >= b.len() { bail!("Invalid \\uXXXX escape"); }
                        let hex = std::str::from_utf8(&b[i+1..i+5])
                            .map_err(|_| anyhow::anyhow!("Invalid characters in \\uXXXX escape"))?;
                        let cp = u16::from_str_radix(hex, 16)
                            .map_err(|_| anyhow::anyhow!("Invalid characters in \\uXXXX escape"))?;
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
                    _ => bail!("Invalid escape"),
                }
                i += 1;
            }
            c if c < 0x20 => bail!("Invalid string: control characters from U+0000 through U+001F must be escaped"),
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
            c if c < 0x20 => bail!("Invalid string: control characters from U+0000 through U+001F must be escaped"),
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
            c if c < 0x20 => bail!("Invalid string: control characters from U+0000 through U+001F must be escaped"),
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
    // jq extension: tolerate a leading '+' the same way it tolerates '-'.
    else if i < b.len() && b[i] == b'+' { i += 1; }
    let digits_start = i;
    if i < b.len() && b[i] == b'0' { i += 1; }
    else { while i < b.len() && b[i].is_ascii_digit() { i += 1; } }
    let int_len = i - digits_start;
    let has_dot = i < b.len() && b[i] == b'.';
    let mut frac_len = 0usize;
    if has_dot {
        i += 1;
        let frac_start = i;
        while i < b.len() && b[i].is_ascii_digit() { i += 1; }
        frac_len = i - frac_start;
    }
    // JSON requires at least one digit in either the integer or fractional
    // part — bare "-", "-.", ".e5" etc. are invalid.
    if int_len == 0 && frac_len == 0 {
        bail!("Invalid numeric literal");
    }
    let has_exp = i < b.len() && (b[i] == b'e' || b[i] == b'E');
    if has_exp {
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') { i += 1; }
        let exp_start = i;
        while i < b.len() && b[i].is_ascii_digit() { i += 1; }
        if i == exp_start {
            bail!("Invalid numeric literal");
        }
    }
    // Fast path for simple integers: parse directly without f64::from_str overhead
    if !has_dot && !has_exp && (i - digits_start) <= 15 {
        let mut n: i64 = 0;
        for &c in &b[digits_start..i] {
            n = n * 10 + (c - b'0') as i64;
        }
        if is_neg { n = -n; }
        // Preserve the sign of `-0` / `-0000...` (#110): i64 negation loses it,
        // but f64 has a distinct -0.0 and jq keeps the lexical repr around so
        // `tostring` / `tojson` can emit "-0". The test `n == 0 && is_neg`
        // catches the generic leading-zeros case too.
        if n == 0 && is_neg {
            // Safety: digits are ASCII, always valid UTF-8
            let num_str = unsafe { std::str::from_utf8_unchecked(&b[pos..i]) };
            return Ok((Value::number_with_repr(-0.0, Rc::from(num_str)), i));
        }
        return Ok((Value::number(n as f64), i));
    }
    // Safety: number bytes are ASCII digits/signs/dots, always valid UTF-8.
    // jq tolerates a leading `+` but never preserves it on output, so strip it
    // before stashing the repr (issue #143).
    let num_str = unsafe { std::str::from_utf8_unchecked(&b[pos..i]) };
    let canonical_in = if num_str.starts_with('+') { &num_str[1..] } else { num_str };
    let n: f64 = fast_float::parse(num_str).unwrap_or(0.0);
    // Fast path: for simple decimals (no exponent, no trailing zeros after dot, ≤15 sig digits),
    // Rust's Display roundtrips identically — skip format_jq_number to avoid String allocation.
    if !has_exp && has_dot && (i - digits_start) <= 16 {
        let last = b[i - 1];
        if last != b'0' && (b[digits_start] != b'0' || digits_start + 1 == i || b[digits_start + 1] == b'.') {
            return Ok((Value::number(n), i));
        }
        // Integer value with decimal notation (e.g., "1.0", "2.00") — format_jq_number would
        // return just the integer, so repr always differs. Skip format call.
        if n == n.trunc() && n.abs() < 1e16 {
            let iv = n as i64;
            if iv as f64 == n {
                return Ok((Value::number_with_repr(n, Rc::from(canonical_in)), i));
            }
        }
    }
    // Compare against the canonical decnum-style form (uppercase `E+`, decimal
    // expansion when |te| is small) rather than the f64 default. Without this,
    // scientific input like `1.5e-10` would lose its repr because the lowercase
    // f64 default matches the lowercase source, and the no-repr path renders
    // lowercase too — but jq's literal preservation expects uppercase. See #426.
    let canonical = normalize_jq_repr(canonical_in).unwrap_or_else(|| canonical_in.to_string());
    let f64_repr = format_jq_number(n);
    let repr = if canonical == f64_repr { None } else { Some(Rc::from(canonical_in)) };
    Ok((Value::number_opt(n, repr), i))
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
    if i < b.len() && b[i] == b'}' { return Ok((Value::object_from_map(ObjMap::new()), i + 1)); }
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
        // jq dedupes duplicate input keys last-wins at parse time (#233).
        // `insert` updates the value in place when a key reappears, preserving
        // the position of the first occurrence — matching jq's semantics.
        map.insert(key, val);
        i = skip_ws(b, end);
        if i >= b.len() { bail!("Unterminated object"); }
        if b[i] == b'}' { return Ok((Value::Obj(ObjInner(rc)), i + 1)); }
        if b[i] != b',' { bail!("Expected ',' or '}}' at position {}", i); }
        i = skip_ws(b, i + 1);
    }
}

// =============================================================================
// jq-compatible fromjson parser (port of jq 1.8.1 src/jv_parse.c)
// =============================================================================
//
// The `fromjson` builtin needs error messages byte-identical to jq's, because
// many filters use `try fromjson catch .` to inspect the message. This parser
// mirrors jq's state machine so we match observable behaviour without an
// FFI bridge to libjq.
//
// Key invariants preserved from jv_parse.c:
//   - `column` is incremented for every consumed byte; a newline also resets
//     it to 0 (so the next byte is column 1 on the new line).
//   - Pending-literal flush ("check_literal") happens on every transition from
//     LITERAL to a non-LITERAL class, and at EOF.
//   - Error messages are emitted without "at EOF" for mid-stream errors, and
//     with "at EOF" when the scan reached the end of input before erroring.
//   - The wrapping `jv_parse_sized_custom_flags` runs the parser twice: if the
//     first value completes successfully and a second one is also produced, the
//     error is the context-free "Unexpected extra JSON values".

/// Parse `input` as JSON for the `fromjson` builtin.
///
/// Produces jq-1.8.1-compatible error messages such as:
///   `Unfinished JSON term at EOF at line N, column M (while parsing '<input>')`
/// Positions are 1-indexed; `"at EOF"` appears only when the error was reached
/// at end-of-input.
pub fn json_to_value_fromjson(input: &str) -> Result<Value> {
    let mut p = JqFromJsonParser::new(input.as_bytes());
    match p.parse_one() {
        Ok(Some(first)) => match p.parse_one() {
            Ok(None) => Ok(first),
            Ok(Some(_)) => {
                bail!("Unexpected extra JSON values (while parsing '{}')", input)
            }
            Err((msg, at_eof)) => Err(format_jq_error(msg, at_eof, p.line, p.col, input)),
        },
        Ok(None) => bail!("Expected JSON value (while parsing '{}')", input),
        Err((msg, at_eof)) => Err(format_jq_error(msg, at_eof, p.line, p.col, input)),
    }
}

fn format_jq_error(
    msg: &str,
    at_eof: bool,
    line: u32,
    col: u32,
    input: &str,
) -> anyhow::Error {
    if at_eof {
        anyhow::anyhow!(
            "{} at EOF at line {}, column {} (while parsing '{}')",
            msg,
            line,
            col,
            input
        )
    } else {
        anyhow::anyhow!(
            "{} at line {}, column {} (while parsing '{}')",
            msg,
            line,
            col,
            input
        )
    }
}

#[derive(Clone, Copy, PartialEq)]
enum JqPState { Normal, StringBody, StringEscape }

enum JqStack {
    Arr(Vec<Value>),
    Obj(ObjMap),
    Key(KeyStr),
}

#[derive(Clone, Copy, PartialEq)]
enum JqCharClass { Literal, Whitespace, Quote, Structure }

#[inline]
fn jq_classify(c: u8) -> JqCharClass {
    match c {
        b' ' | b'\t' | b'\r' | b'\n' => JqCharClass::Whitespace,
        b'"' => JqCharClass::Quote,
        b'[' | b',' | b']' | b'{' | b':' | b'}' => JqCharClass::Structure,
        _ => JqCharClass::Literal,
    }
}

struct JqFromJsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
    line: u32,
    col: u32,
    state: JqPState,
    token: Vec<u8>,
    stack: Vec<JqStack>,
    next: Option<Value>,
    next_is_string: bool,
}

impl<'a> JqFromJsonParser<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        let mut pos = 0;
        if bytes.len() >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF {
            pos = 3;
        }
        Self {
            bytes,
            pos,
            line: 1,
            col: 0,
            state: JqPState::Normal,
            token: Vec::new(),
            stack: Vec::new(),
            next: None,
            next_is_string: false,
        }
    }

    /// Parse a single top-level value.
    /// Returns `Ok(Some(v))` when one value is complete, `Ok(None)` when EOF
    /// was reached without consuming any token, and `Err((msg, at_eof))`
    /// otherwise. `self.pos` is left just past the last byte consumed.
    fn parse_one(&mut self) -> std::result::Result<Option<Value>, (&'static str, bool)> {
        while self.pos < self.bytes.len() {
            let ch = self.bytes[self.pos];
            self.pos += 1;
            self.col += 1;
            if ch == b'\n' {
                self.line += 1;
                self.col = 0;
            }
            match self.state {
                JqPState::StringBody => {
                    if ch == b'"' {
                        self.found_string().map_err(|m| (m, false))?;
                        self.state = JqPState::Normal;
                        if self.stack.is_empty() && self.next.is_some() {
                            return Ok(self.next.take());
                        }
                    } else {
                        self.token.push(ch);
                        if ch == b'\\' {
                            self.state = JqPState::StringEscape;
                        }
                    }
                }
                JqPState::StringEscape => {
                    self.token.push(ch);
                    self.state = JqPState::StringBody;
                }
                JqPState::Normal => {
                    let cls = jq_classify(ch);
                    if cls != JqCharClass::Literal && !self.token.is_empty() {
                        self.check_literal().map_err(|m| (m, false))?;
                        if self.stack.is_empty() && self.next.is_some() {
                            // First-value flush via whitespace; we still need
                            // to record that we've flushed, but we can't yet
                            // return because the char we just consumed was ws
                            // or structural — and a structural char may error.
                            //
                            // For whitespace we're done with this parse_one.
                            if cls == JqCharClass::Whitespace {
                                return Ok(self.next.take());
                            }
                            // For structure/quote the top-level value is
                            // followed by more content; fall through so the
                            // structural char gets processed (and likely errs
                            // as unmatched `]`/`}` or "Expected separator").
                        }
                    }
                    match cls {
                        JqCharClass::Literal => self.token.push(ch),
                        JqCharClass::Whitespace => {}
                        JqCharClass::Quote => self.state = JqPState::StringBody,
                        JqCharClass::Structure => {
                            self.parse_structure(ch).map_err(|m| (m, false))?;
                            if self.stack.is_empty() && self.next.is_some() {
                                return Ok(self.next.take());
                            }
                        }
                    }
                }
            }
        }
        // EOF
        match self.state {
            JqPState::StringBody | JqPState::StringEscape => {
                return Err(("Unfinished string", true));
            }
            _ => {}
        }
        if !self.token.is_empty() {
            self.check_literal().map_err(|m| (m, true))?;
        }
        if !self.stack.is_empty() {
            return Err(("Unfinished JSON term", true));
        }
        Ok(self.next.take())
    }

    fn push_value(&mut self, v: Value, is_string: bool) -> std::result::Result<(), &'static str> {
        if self.next.is_some() {
            return Err("Expected separator between values");
        }
        self.next = Some(v);
        self.next_is_string = is_string;
        Ok(())
    }

    fn check_literal(&mut self) -> std::result::Result<(), &'static str> {
        if self.token.is_empty() {
            return Ok(());
        }
        let first = self.token[0];
        let keyword: Option<(&'static [u8], Value)> = match first {
            b't' => Some((b"true", Value::True)),
            b'f' => Some((b"false", Value::False)),
            b'\'' => return Err("Invalid string literal; expected \", but got '"),
            b'n' if self.token.len() > 1 && self.token[1] == b'u' => {
                Some((b"null", Value::Null))
            }
            _ => None,
        };
        if let Some((pat, val)) = keyword {
            if self.token.len() != pat.len() {
                return Err("Invalid literal");
            }
            for i in 0..pat.len() {
                if self.token[i] != pat[i] {
                    return Err("Invalid literal");
                }
            }
            self.push_value(val, false)?;
        } else {
            let s = match std::str::from_utf8(&self.token) {
                Ok(s) => s,
                Err(_) => return Err("Invalid numeric literal"),
            };
            match parse_jq_strtod(s) {
                Some(n) => self.push_value(Value::number(n), false)?,
                None => return Err("Invalid numeric literal"),
            }
        }
        self.token.clear();
        Ok(())
    }

    fn found_string(&mut self) -> std::result::Result<(), &'static str> {
        let t = std::mem::take(&mut self.token);
        let mut out: Vec<u8> = Vec::with_capacity(t.len());
        let mut i = 0;
        while i < t.len() {
            let c = t[i];
            i += 1;
            if c == b'\\' {
                if i >= t.len() {
                    return Err("Expected escape character at end of string");
                }
                let e = t[i];
                i += 1;
                match e {
                    b'\\' | b'"' | b'/' => out.push(e),
                    b'b' => out.push(0x08),
                    b'f' => out.push(0x0C),
                    b't' => out.push(b'\t'),
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b'u' => {
                        if i + 4 > t.len() {
                            return Err("Invalid \\uXXXX escape");
                        }
                        let hi = match unhex4(&t[i..i + 4]) {
                            Some(v) => v,
                            None => return Err("Invalid characters in \\uXXXX escape"),
                        };
                        i += 4;
                        let cp = if (0xD800..=0xDBFF).contains(&hi) {
                            if i + 6 > t.len() || t[i] != b'\\' || t[i + 1] != b'u' {
                                return Err("Invalid \\uXXXX\\uXXXX surrogate pair escape");
                            }
                            let lo = match unhex4(&t[i + 2..i + 6]) {
                                Some(v) => v,
                                None => return Err("Invalid \\uXXXX\\uXXXX surrogate pair escape"),
                            };
                            if !(0xDC00..=0xDFFF).contains(&lo) {
                                return Err("Invalid \\uXXXX\\uXXXX surrogate pair escape");
                            }
                            i += 6;
                            0x10000 + (((hi - 0xD800) << 10) | (lo - 0xDC00))
                        } else {
                            hi
                        };
                        let cp = if cp > 0x10FFFF { 0xFFFD } else { cp };
                        let chr = char::from_u32(cp).unwrap_or('\u{FFFD}');
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(chr.encode_utf8(&mut buf).as_bytes());
                    }
                    _ => return Err("Invalid escape"),
                }
            } else {
                if c < 0x20 {
                    return Err("Invalid string: control characters from U+0000 through U+001F must be escaped");
                }
                out.push(c);
            }
        }
        let s = String::from_utf8(out)
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned());
        self.push_value(Value::Str(KeyStr::from(s)), true)
    }

    fn parse_structure(&mut self, ch: u8) -> std::result::Result<(), &'static str> {
        match ch {
            b'[' => {
                if self.next.is_some() {
                    return Err("Expected separator between values");
                }
                if self.stack.len() >= MAX_JSON_DEPTH {
                    return Err("Exceeds depth limit for parsing");
                }
                self.stack.push(JqStack::Arr(Vec::new()));
            }
            b'{' => {
                if self.next.is_some() {
                    return Err("Expected separator between values");
                }
                if self.stack.len() >= MAX_JSON_DEPTH {
                    return Err("Exceeds depth limit for parsing");
                }
                self.stack.push(JqStack::Obj(new_objmap()));
            }
            b':' => {
                if self.next.is_none() {
                    return Err("Expected string key before ':'");
                }
                if !matches!(self.stack.last(), Some(JqStack::Obj(_))) {
                    return Err("':' not as part of an object");
                }
                if !self.next_is_string {
                    return Err("Object keys must be strings");
                }
                let key = match self.next.take() {
                    Some(Value::Str(s)) => s,
                    _ => unreachable!("next_is_string guarded"),
                };
                self.next_is_string = false;
                self.stack.push(JqStack::Key(key));
            }
            b',' => {
                if self.next.is_none() {
                    return Err("Expected value before ','");
                }
                match self.stack.last_mut() {
                    None => return Err("',' not as part of an object or array"),
                    Some(JqStack::Arr(a)) => {
                        a.push(self.next.take().unwrap());
                        self.next_is_string = false;
                    }
                    Some(JqStack::Key(_)) => {
                        let key = match self.stack.pop() {
                            Some(JqStack::Key(k)) => k,
                            _ => unreachable!(),
                        };
                        let val = self.next.take().unwrap();
                        self.next_is_string = false;
                        match self.stack.last_mut() {
                            Some(JqStack::Obj(o)) => {
                                o.insert(key, val);
                            }
                            _ => return Err("Objects must consist of key:value pairs"),
                        }
                    }
                    Some(JqStack::Obj(_)) => {
                        return Err("Objects must consist of key:value pairs");
                    }
                }
            }
            b']' => {
                match self.stack.last_mut() {
                    Some(JqStack::Arr(a)) => {
                        if let Some(v) = self.next.take() {
                            a.push(v);
                            self.next_is_string = false;
                        } else if !a.is_empty() {
                            return Err("Expected another array element");
                        }
                        let arr = match self.stack.pop() {
                            Some(JqStack::Arr(a)) => a,
                            _ => unreachable!(),
                        };
                        self.next = Some(Value::Arr(Rc::new(arr)));
                        self.next_is_string = false;
                    }
                    _ => return Err("Unmatched ']'"),
                }
            }
            b'}' => {
                match self.stack.last() {
                    None | Some(JqStack::Arr(_)) => return Err("Unmatched '}'"),
                    _ => {}
                }
                if self.next.is_some() {
                    if !matches!(self.stack.last(), Some(JqStack::Key(_))) {
                        return Err("Objects must consist of key:value pairs");
                    }
                    let key = match self.stack.pop() {
                        Some(JqStack::Key(k)) => k,
                        _ => unreachable!(),
                    };
                    let val = self.next.take().unwrap();
                    self.next_is_string = false;
                    match self.stack.last_mut() {
                        Some(JqStack::Obj(o)) => {
                            o.insert(key, val);
                        }
                        _ => unreachable!(),
                    }
                } else {
                    match self.stack.last() {
                        Some(JqStack::Obj(o)) if !o.is_empty() => {
                            return Err("Expected another key-value pair");
                        }
                        Some(JqStack::Obj(_)) => {}
                        _ => return Err("Unmatched '}'"),
                    }
                }
                let obj = match self.stack.pop() {
                    Some(JqStack::Obj(o)) => o,
                    _ => unreachable!(),
                };
                self.next = Some(Value::object_from_map(obj));
                self.next_is_string = false;
            }
            _ => unreachable!("non-structural char dispatched to parse_structure"),
        }
        Ok(())
    }
}

fn unhex4(h: &[u8]) -> Option<u32> {
    if h.len() != 4 {
        return None;
    }
    let mut r: u32 = 0;
    for &c in h {
        let n = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => return None,
        };
        r = (r << 4) | n as u32;
    }
    Some(r)
}

/// jq-compatible number parser (mirrors `jvp_strtod`).
///
/// Accepts the same shapes jq's `jv_parse` would hand to `strtod`:
///   [+-]?digits(.digits)?([eE][+-]?digits)?
///   [+-]?.digits([eE][+-]?digits)?
///   [+-]?(nan|inf|infinity)  (case-insensitive)
fn parse_jq_strtod(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    if b.is_empty() {
        return None;
    }
    let (sign_end, neg) = match b[0] {
        b'+' => (1usize, false),
        b'-' => (1usize, true),
        _ => (0usize, false),
    };
    let rest = &b[sign_end..];
    let eq_ci = |a: &[u8], w: &[u8]| -> bool {
        a.len() == w.len() && a.iter().zip(w.iter()).all(|(x, y)| x.eq_ignore_ascii_case(y))
    };
    if eq_ci(rest, b"nan") {
        return Some(f64::NAN);
    }
    if eq_ci(rest, b"inf") || eq_ci(rest, b"infinity") {
        return Some(if neg { f64::NEG_INFINITY } else { f64::INFINITY });
    }
    // Structural validation of the numeric form.
    let mut i = 0;
    let int_start = i;
    while i < rest.len() && rest[i].is_ascii_digit() {
        i += 1;
    }
    let has_int = i > int_start;
    let mut has_frac = false;
    if i < rest.len() && rest[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < rest.len() && rest[i].is_ascii_digit() {
            i += 1;
        }
        has_frac = i > frac_start;
    }
    if !has_int && !has_frac {
        return None;
    }
    if i < rest.len() && (rest[i] == b'e' || rest[i] == b'E') {
        i += 1;
        if i < rest.len() && (rest[i] == b'+' || rest[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < rest.len() && rest[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return None;
        }
    }
    if i != rest.len() {
        return None;
    }
    // Normalize forms Rust's f64 parser rejects: leading `.` and trailing `.`.
    let tail = unsafe { std::str::from_utf8_unchecked(rest) };
    let mut normalized = String::with_capacity(tail.len() + 2);
    if tail.starts_with('.') {
        normalized.push('0');
    }
    let mut it = tail.chars().peekable();
    while let Some(c) = it.next() {
        normalized.push(c);
        if c == '.' && !matches!(it.peek(), Some(d) if d.is_ascii_digit()) {
            normalized.push('0');
        }
    }
    let parsed: f64 = normalized.parse().ok()?;
    Some(if neg { -parsed } else { parsed })
}


/// Format f64 the way jq does — shortest representation with scientific notation for large/small values.
/// Whether a string is a JSON-conformant number (RFC 8259 §6).
/// Used to guard against emitting preserved source forms like `.5` or `+1`
/// that parse in jq but break strict JSON consumers.
pub fn is_valid_json_number(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0;
    if bytes.first() == Some(&b'-') { i += 1; }
    let int_start = i;
    match bytes.get(i) {
        Some(&b'0') => { i += 1; }
        Some(&c) if c.is_ascii_digit() => {
            while bytes.get(i).map_or(false, |b| b.is_ascii_digit()) { i += 1; }
        }
        _ => return false,
    }
    let _ = int_start;
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        let frac_start = i;
        while bytes.get(i).map_or(false, |b| b.is_ascii_digit()) { i += 1; }
        if i == frac_start { return false; }
    }
    if matches!(bytes.get(i), Some(b'e') | Some(b'E')) {
        i += 1;
        if matches!(bytes.get(i), Some(b'+') | Some(b'-')) { i += 1; }
        let exp_start = i;
        while bytes.get(i).map_or(false, |b| b.is_ascii_digit()) { i += 1; }
        if i == exp_start { return false; }
    }
    i == bytes.len()
}

pub fn format_jq_number(n: f64) -> String {
    let mut buf = String::with_capacity(24);
    push_jq_number_str(&mut buf, n);
    buf
}

/// Push a formatted jq number directly to a String buffer (avoids intermediate allocation).
#[inline]
/// Push a number's canonical jq representation, preferring the original
/// repr (so `1.0` stays `1.0`, `0e10` stays `0E+10`) when it round-trips
/// through f64. Falls back to f64 dtoa otherwise. Used by @csv / @tsv
/// formatters where jq preserves the literal shape (#475).
pub fn push_value_num_repr_str(buf: &mut String, n: f64, repr: Option<&Rc<str>>) {
    if let Some(r) = repr.filter(|r| is_valid_json_number(r) && repr_is_exact_for_f64(r, n)) {
        if let Some(canonical) = normalize_jq_repr(r) {
            buf.push_str(&canonical);
        } else {
            buf.push_str(r);
        }
    } else {
        push_jq_number_str(buf, n);
    }
}

/// Same as [`push_value_num_repr_str`] but writes UTF-8 bytes to a `Vec<u8>`.
pub fn push_value_num_repr_bytes(buf: &mut Vec<u8>, n: f64, repr: Option<&Rc<str>>) {
    if let Some(r) = repr.filter(|r| is_valid_json_number(r) && repr_is_exact_for_f64(r, n)) {
        if let Some(canonical) = normalize_jq_repr(r) {
            buf.extend_from_slice(canonical.as_bytes());
        } else {
            buf.extend_from_slice(r.as_bytes());
        }
    } else {
        push_jq_number_bytes(buf, n);
    }
}

pub fn push_jq_number_str(buf: &mut String, n: f64) {
    if n.is_nan() {
        buf.push_str("null");
        return;
    }
    if n.is_infinite() {
        buf.push_str(if n.is_sign_positive() { "1.7976931348623157e+308" } else { "-1.7976931348623157e+308" });
        return;
    }
    if n == 0.0 {
        // Preserve IEEE-754 negative zero: jq emits `-0` for arithmetic
        // results that land on `-0.0` (e.g. `-0 * 1`, `-0 - 0`). Issue #110.
        if n.is_sign_negative() {
            buf.push_str("-0");
        } else {
            buf.push_str("0");
        }
        return;
    }

    // Integer fast path for common values (fits in i64, reasonable length)
    if n == n.trunc() && n.abs() < 1e16 {
        let i = n as i64;
        if i as f64 == n {
            let mut ibuf = itoa::Buffer::new();
            buf.push_str(ibuf.format(i));
            return;
        }
    }

    // Use Rust's shortest-representation (like jq's jvp_dtoa)
    let s = format!("{}", n);
    let abs = n.abs();

    // For very small numbers (abs < 1e-4), always use scientific notation (matching %g)
    if abs != 0.0 && abs < 1e-4 {
        buf.push_str(&format_as_scientific_lowercase(n));
        return;
    }

    // For large numbers: compare fixed vs scientific length, prefer shorter
    if abs >= 1e16 {
        let sci = format_as_scientific_lowercase(n);
        if sci.len() < s.len() {
            buf.push_str(&sci);
            return;
        }
    }

    buf.push_str(&s);
}

/// Pick the bytes to emit for `Value::Num(_, repr)` — either the canonical
/// rewrite of `repr` (when scientific notation needs case/sign normalization
/// or decimal expansion, see [`normalize_jq_repr`]) or the original repr.
#[inline]
pub fn canonical_repr_bytes(repr: &str) -> std::borrow::Cow<'_, str> {
    match normalize_jq_repr(repr) {
        Some(c) => std::borrow::Cow::Owned(c),
        None => std::borrow::Cow::Borrowed(repr),
    }
}

/// Re-render a JSON-shaped number literal `repr` into jq 1.8.1's canonical
/// output form. Returns `None` if the repr is already canonical (or not
/// scientific). Handles:
/// - case + sign: `e` → `E+`/`E-`, no leading `+` on the mantissa,
/// - decimal vs scientific threshold: small-exponent scientific literals
///   expand to decimal (`1e-5` → `0.00001`) while larger magnitudes keep
///   `E+N` (`1e10` → `1E+10`).
///
/// Mirrors jq's libmpdecimal output policy closely enough for the cases
/// flagged in issues #110 and #143; not a full decnum reimplementation.
pub fn normalize_jq_repr(repr: &str) -> Option<String> {
    let bytes = repr.as_bytes();
    let mut idx = 0;
    let mut sign = "";
    match bytes.first() {
        Some(b'-') => { sign = "-"; idx = 1; }
        Some(b'+') => { idx = 1; }
        _ => {}
    }
    let mantissa_start = idx;
    let mut dot_pos: Option<usize> = None;
    let mut e_pos: Option<usize> = None;
    while idx < bytes.len() {
        match bytes[idx] {
            b'.' => { dot_pos.get_or_insert(idx); }
            b'e' | b'E' => { e_pos = Some(idx); break; }
            _ => {}
        }
        idx += 1;
    }
    // No exponent and no leading `+`: nothing to normalize.
    if e_pos.is_none() && sign != "-" && bytes.first() != Some(&b'+') {
        return None;
    }
    let mantissa_end = e_pos.unwrap_or(bytes.len());
    let mant_int_end = dot_pos.unwrap_or(mantissa_end);
    let mant_frac_start = dot_pos.map(|d| d + 1).unwrap_or(mantissa_end);
    let mant_int = &repr[mantissa_start..mant_int_end];
    let mant_frac = &repr[mant_frac_start..mantissa_end];
    let exp: i32 = match e_pos {
        Some(p) => repr[p + 1..].parse().ok()?,
        None => 0,
    };
    let mut combined = String::with_capacity(mant_int.len() + mant_frac.len());
    combined.push_str(mant_int);
    combined.push_str(mant_frac);
    if combined.is_empty() || combined.bytes().any(|b| !b.is_ascii_digit()) {
        return None;
    }
    let leading_zeros = combined.bytes().take_while(|&b| b == b'0').count();
    if leading_zeros == combined.len() {
        // Pure-zero value: jq keeps the explicit-exponent form when present.
        // `0e10`  → `0E+10` (positive exp stays scientific)
        // `0e-1`  → `0.0`   (negative exp expands to decimal — N+frac zeros)
        // `0.0e-1` → `0.00` (frac digits add to the zero count)
        // `0.0e0` → `0.0`   (no exp shift, drop the `e0`)
        // See #452.
        if let Some(_) = e_pos {
            if exp >= 1 {
                return Some(format!("{}0E+{}", sign, exp));
            }
            // exp <= 0: expand to decimal `0.000...0` with `mant_frac.len() - exp` zeros.
            // exp == 0 with no frac → "0"; exp == 0 with frac → keep frac zeros.
            let total_zeros = (mant_frac.len() as i64) - exp as i64;
            if total_zeros <= 0 {
                return Some(format!("{}0", sign));
            }
            let mut s = String::with_capacity(2 + total_zeros as usize);
            s.push_str(sign);
            s.push_str("0.");
            for _ in 0..total_zeros { s.push('0'); }
            return Some(s);
        }
        return None;
    }
    let sig: &str = &combined[leading_zeros..];
    let ndigits = sig.len() as i32;
    let te = ndigits - 1 + exp - (mant_frac.len() as i32);
    // Bail if the input was a plain decimal (no exponent) — those are
    // already canonical, except for the leading-`+` case handled below.
    if e_pos.is_none() {
        return None;
    }
    if te >= ndigits || te < -6 {
        // Scientific form, jq style: <first>.<rest>E[+-]<|te|>.
        let mantissa_str = format_canonical_mantissa(sig);
        let sign_e = if te >= 0 { '+' } else { '-' };
        return Some(format!("{}{}E{}{}", sign, mantissa_str, sign_e, te.abs()));
    }
    if te >= 0 {
        let int_len = (te + 1) as usize;
        if int_len >= sig.len() {
            let pad = int_len - sig.len();
            let mut s = String::with_capacity(int_len);
            s.push_str(sig);
            for _ in 0..pad { s.push('0'); }
            return Some(format!("{}{}", sign, s));
        }
        // jq's decnum keeps the mantissa's trailing zeros through expansion
        // (`1.0e0` → `1.0`, not `1`). Don't strip — preserve the original
        // significand digit count. See #457.
        let frac = &sig[int_len..];
        if frac.is_empty() {
            return Some(format!("{}{}", sign, &sig[..int_len]));
        }
        return Some(format!("{}{}.{}", sign, &sig[..int_len], frac));
    }
    let zeros = (-te - 1) as usize;
    // Same trailing-zero preservation for negative-`te` expansion
    // (`1.0e-5` → `0.000010`, not `0.00001`). See #457.
    let frac = sig;
    let mut buf = String::with_capacity(zeros + frac.len() + 4);
    buf.push_str(sign);
    buf.push_str("0.");
    for _ in 0..zeros { buf.push('0'); }
    buf.push_str(frac);
    Some(buf)
}

fn format_canonical_mantissa(sig: &str) -> String {
    if sig.len() <= 1 {
        return sig.to_string();
    }
    // Preserve the mantissa's trailing zeros (`1.0e10` → `1.0E+10`,
    // `1.00e10` → `1.00E+10`). See #457.
    let frac = &sig[1..];
    format!("{}.{}", &sig[..1], frac)
}

/// Format a number in jq's f64 dtoa scientific style: lowercase `e`, explicit
/// `+`/`-` sign, single-digit negative exponent zero-padded to two digits.
///
/// jq 1.8.1 uses two number renderers depending on whether the value is
/// decnum-resident or f64-resident. Decnum literals print uppercase
/// (`1E+10`) via `canonical_repr_bytes`; values without a preserved repr —
/// arithmetic results, `tonumber`, etc — print lowercase here. See #426.
fn format_as_scientific_lowercase(n: f64) -> String {
    let sci = format!("{:e}", n);
    let (mantissa, exp_str) = match sci.find('e') {
        Some(idx) => (&sci[..idx], &sci[idx + 1..]),
        None => return sci,
    };
    let (sign, digits) = if let Some(rest) = exp_str.strip_prefix('-') {
        ('-', rest)
    } else {
        ('+', exp_str)
    };
    let exp: i32 = digits.parse().unwrap_or(0);
    if exp.abs() < 10 {
        format!("{}e{}{:02}", mantissa, sign, exp.abs())
    } else {
        format!("{}e{}{}", mantissa, sign, exp.abs())
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

/// Convert Value to compact JSON string for `tojson`. Uses the repr when it
/// exactly represents the stored f64, otherwise falls back to the f64 form.
/// jq 1.8.1 decnum preserves all reprs; without decnum we can only honor the
/// ones f64 can hold exactly (so `1.0` stays `"1.0"` but `13911860366432393`
/// renders as the rounded `"13911860366432392"`).
pub fn value_to_json_tojson(v: &Value) -> String {
    let mut out = String::new();
    push_value_tojson(v, &mut out, 0);
    out
}

fn push_value_tojson(v: &Value, out: &mut String, depth: usize) {
    if depth > MAX_JSON_DEPTH {
        out.push_str("\"<skipped: too deep>\"");
        return;
    }
    match v {
        Value::Null => out.push_str("null"),
        Value::False => out.push_str("false"),
        Value::True => out.push_str("true"),
        Value::Num(n, NumRepr(repr)) => {
            if let Some(r) = repr.as_ref().filter(|r| is_valid_json_number(r) && repr_is_exact_for_f64(r, *n)) {
                if let Some(canonical) = normalize_jq_repr(r) {
                    out.push_str(&canonical);
                } else {
                    out.push_str(r);
                }
            } else {
                push_jq_number_str(out, *n);
            }
        }
        Value::Str(s) => push_json_string(out, s),
        Value::Arr(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 { out.push(','); }
                push_value_tojson(item, out, depth + 1);
            }
            out.push(']');
        }
        Value::Obj(ObjInner(o)) => {
            out.push('{');
            for (i, (k, v)) in o.iter().enumerate() {
                if i > 0 { out.push(','); }
                push_json_string(out, k);
                out.push(':');
                push_value_tojson(v, out, depth + 1);
            }
            out.push('}');
        }
        Value::Error(e) => push_json_string(out, e),
    }
}

/// True iff `repr` (a JSON-valid decimal literal) represents a value that f64
/// can hold exactly. Rejects mantissas with more than 15 significant decimal
/// digits and exponents outside f64's dynamic range.
pub(crate) fn repr_is_exact_for_f64(repr: &str, n: f64) -> bool {
    if !n.is_finite() { return false; }
    let bytes = repr.as_bytes();
    let mut i = 0;
    if matches!(bytes.first(), Some(b'-') | Some(b'+')) { i += 1; }
    let mut sig_digits = 0u32;
    let mut started = false;
    while i < bytes.len() && bytes[i] != b'e' && bytes[i] != b'E' {
        let c = bytes[i];
        if c == b'.' { i += 1; continue; }
        if !c.is_ascii_digit() { return false; }
        if c != b'0' { started = true; }
        if started { sig_digits += 1; }
        i += 1;
    }
    let mut exp_val: i32 = 0;
    if i < bytes.len() {
        i += 1;
        let mut exp_neg = false;
        if matches!(bytes.get(i), Some(b'-')) { exp_neg = true; i += 1; }
        else if matches!(bytes.get(i), Some(b'+')) { i += 1; }
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            exp_val = exp_val.saturating_mul(10).saturating_add((bytes[i] - b'0') as i32);
            i += 1;
        }
        if exp_neg { exp_val = -exp_val; }
    }
    if exp_val > 308 || exp_val < -323 { return false; }
    sig_digits <= 15
}

fn value_to_json_depth(v: &Value, depth: usize, precise: bool) -> String {
    if depth > MAX_JSON_DEPTH {
        return "\"<skipped: too deep>\"".to_string();
    }
    match v {
        Value::Null => "null".to_string(),
        Value::False => "false".to_string(),
        Value::True => "true".to_string(),
        Value::Num(n, NumRepr(repr)) => {
            if precise {
                if let Some(r) = repr {
                    if is_valid_json_number(r) {
                        if let Some(canonical) = normalize_jq_repr(r) {
                            return canonical;
                        }
                        return r.to_string();
                    }
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
        Value::Obj(ObjInner(o)) => {
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
    write_pretty_to_string(&mut out, v, depth, step, use_tab, sort_keys, false);
    out
}

pub fn value_to_json_pretty_color(v: &Value, depth: usize, step: usize, use_tab: bool, sort_keys: bool) -> String {
    let mut out = String::with_capacity(128);
    write_pretty_to_string(&mut out, v, depth, step, use_tab, sort_keys, true);
    out
}

// ANSI color constants (matches jq defaults: JQ_COLORS="0;90:0;39:0;39:0;39:0;32:1;39:1;39:1;34")
const COLOR_RESET: &str = "\x1b[0m";
const COLOR_NULL: &str = "\x1b[0;90m";     // bright black (gray)
const COLOR_FALSE: &str = "\x1b[0;39m";    // default
const COLOR_TRUE: &str = "\x1b[0;39m";     // default
const COLOR_NUMBER: &str = "\x1b[0;39m";   // default
const COLOR_STRING: &str = "\x1b[0;32m";   // green
const COLOR_ARRAY: &str = "\x1b[1;39m";    // bold default
const COLOR_OBJECT: &str = "\x1b[1;39m";   // bold default
const COLOR_KEY: &str = "\x1b[1;34m";      // bold blue

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
            0x7f => 0x7f,
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

fn write_pretty_to_string_impl<const COLOR: bool>(out: &mut String, v: &Value, depth: usize, step: usize, use_tab: bool, sort_keys: bool) {
    // jq's --tab uses exactly one tab per indent level regardless of --indent.
    let step = if use_tab { 1 } else { step };
    macro_rules! c {
        ($code:expr) => { if COLOR { out.push_str($code); } };
    }
    match v {
        Value::Null => { c!(COLOR_NULL); out.push_str("null"); c!(COLOR_RESET); }
        Value::False => { c!(COLOR_FALSE); out.push_str("false"); c!(COLOR_RESET); }
        Value::True => { c!(COLOR_TRUE); out.push_str("true"); c!(COLOR_RESET); }
        Value::Num(n, NumRepr(repr)) => {
            c!(COLOR_NUMBER);
            if let Some(r) = repr.as_ref().filter(|r| is_valid_json_number(r)) {
                if let Some(canonical) = normalize_jq_repr(r) {
                    out.push_str(&canonical);
                } else {
                    out.push_str(r);
                }
            } else {
                push_jq_number(out, *n);
            }
            c!(COLOR_RESET);
        }
        Value::Str(s) => { c!(COLOR_STRING); push_json_string(out, s); c!(COLOR_RESET); }
        Value::Error(e) => push_json_string(out, e),
        Value::Arr(a) if a.is_empty() => { c!(COLOR_ARRAY); out.push_str("[]"); c!(COLOR_RESET); }
        Value::Obj(ObjInner(o)) if o.is_empty() => { c!(COLOR_OBJECT); out.push_str("{}"); c!(COLOR_RESET); }
        Value::Arr(a) => {
            let inner_depth = depth + step;
            // Emit `\033[0m` before each newline so terminals that paint
            // the rest of the line (powerline, less -R) do not extend the
            // structural-character color. Matches jq -C byte-for-byte.
            c!(COLOR_ARRAY); out.push('['); c!(COLOR_RESET); out.push('\n');
            for (i, item) in a.iter().enumerate() {
                if i > 0 { c!(COLOR_ARRAY); out.push(','); c!(COLOR_RESET); out.push('\n'); }
                push_indent(out, inner_depth, use_tab);
                write_pretty_to_string_impl::<COLOR>(out, item, inner_depth, step, use_tab, sort_keys);
            }
            out.push('\n');
            push_indent(out, depth, use_tab);
            c!(COLOR_ARRAY); out.push(']'); c!(COLOR_RESET);
        }
        Value::Obj(ObjInner(o)) => {
            let inner_depth = depth + step;
            c!(COLOR_OBJECT); out.push('{'); c!(COLOR_RESET); out.push('\n');
            if sort_keys {
                let mut entries: Vec<_> = o.iter().collect();
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                for (i, (k, val)) in entries.iter().enumerate() {
                    if i > 0 { c!(COLOR_OBJECT); out.push(','); c!(COLOR_RESET); out.push('\n'); }
                    push_indent(out, inner_depth, use_tab);
                    c!(COLOR_KEY); push_json_string(out, k); c!(COLOR_RESET);
                    c!(COLOR_OBJECT); out.push(':'); c!(COLOR_RESET); out.push(' ');
                    write_pretty_to_string_impl::<COLOR>(out, val, inner_depth, step, use_tab, sort_keys);
                }
            } else {
                for (i, (k, val)) in o.iter().enumerate() {
                    if i > 0 { c!(COLOR_OBJECT); out.push(','); c!(COLOR_RESET); out.push('\n'); }
                    push_indent(out, inner_depth, use_tab);
                    c!(COLOR_KEY); push_json_string(out, k); c!(COLOR_RESET);
                    c!(COLOR_OBJECT); out.push(':'); c!(COLOR_RESET); out.push(' ');
                    write_pretty_to_string_impl::<COLOR>(out, val, inner_depth, step, use_tab, sort_keys);
                }
            }
            out.push('\n');
            push_indent(out, depth, use_tab);
            c!(COLOR_OBJECT); out.push('}'); c!(COLOR_RESET);
        }
    }
}

#[inline]
fn write_pretty_to_string(out: &mut String, v: &Value, depth: usize, step: usize, use_tab: bool, sort_keys: bool, color: bool) {
    if color {
        write_pretty_to_string_impl::<true>(out, v, depth, step, use_tab, sort_keys);
    } else {
        write_pretty_to_string_impl::<false>(out, v, depth, step, use_tab, sort_keys);
    }
}

// ============================================================================
// Streaming JSON output — writes directly to io::Write, avoiding intermediate Strings
// ============================================================================

use std::io;

fn write_json_string(w: &mut dyn io::Write, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    // Find first byte needing escape
    let first_esc = bytes.iter().position(|&b| b == b'"' || b == b'\\' || b < 0x20 || b == 0x7f);
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
            c if c < 0x20 || c == 0x7f => {
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
        // jq normalises negative zero to "0" on output; match that.
        return w.write_all(b"0");
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

// Reusable String buffer for pretty-print output. Per-thread to keep
// `cargo test` parallel runs honest — see OBJMAP_POOL.
thread_local! {
    static PRETTY_BUF: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Write a Value as pretty JSON + newline, directly to writer.
/// For scalar values and small containers, uses stack buffer to avoid String allocation.
pub fn write_value_pretty_line(w: &mut dyn io::Write, v: &Value, indent: usize, use_tab: bool, sort_keys: bool) -> io::Result<()> {
    write_value_pretty_line_color(w, v, indent, use_tab, sort_keys, false)
}

pub fn write_value_pretty_line_color(w: &mut dyn io::Write, v: &Value, indent: usize, use_tab: bool, sort_keys: bool, color: bool) -> io::Result<()> {
    if !color {
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
                return w.write_all(b"\n");
            }
            Value::Arr(a) if a.is_empty() => return w.write_all(b"[]\n"),
            Value::Obj(ObjInner(o)) if o.is_empty() => return w.write_all(b"{}\n"),
            _ => {}
        }
    }
    // Reuse a per-thread String buffer to avoid per-value allocation
    PRETTY_BUF.with_borrow_mut(|out| {
        out.clear();
        write_pretty_to_string(out, v, 0, indent, use_tab, sort_keys, color);
        w.write_all(out.as_bytes())?;
        w.write_all(b"\n")
    })
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

pub fn push_compact_line_color(buf: &mut Vec<u8>, v: &Value) {
    push_compact_value_color(buf, v);
    buf.push(b'\n');
}

fn push_compact_value_color(buf: &mut Vec<u8>, v: &Value) {
    macro_rules! c {
        ($code:expr) => { buf.extend_from_slice($code.as_bytes()); };
    }
    match v {
        Value::Null => { c!(COLOR_NULL); buf.extend_from_slice(b"null"); c!(COLOR_RESET); }
        Value::False => { c!(COLOR_FALSE); buf.extend_from_slice(b"false"); c!(COLOR_RESET); }
        Value::True => { c!(COLOR_TRUE); buf.extend_from_slice(b"true"); c!(COLOR_RESET); }
        Value::Num(n, NumRepr(repr)) => {
            c!(COLOR_NUMBER);
            if let Some(r) = repr.as_ref().filter(|r| is_valid_json_number(r)) {
                buf.extend_from_slice(canonical_repr_bytes(r).as_bytes());
            } else {
                push_jq_number_bytes(buf, *n);
            }
            c!(COLOR_RESET);
        }
        Value::Str(s) => { c!(COLOR_STRING); push_json_string_to_vec(buf, s.as_str()); c!(COLOR_RESET); }
        Value::Arr(a) => {
            c!(COLOR_ARRAY); buf.push(b'[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 { buf.push(b','); }
                c!(COLOR_RESET);
                push_compact_value_color(buf, item);
            }
            c!(COLOR_ARRAY); buf.push(b']'); c!(COLOR_RESET);
        }
        Value::Obj(ObjInner(o)) => {
            c!(COLOR_OBJECT); buf.push(b'{');
            for (i, (k, val)) in o.iter().enumerate() {
                if i > 0 { buf.push(b','); }
                c!(COLOR_KEY); push_json_string_to_vec(buf, k.as_str()); c!(COLOR_RESET);
                c!(COLOR_OBJECT); buf.push(b':'); c!(COLOR_RESET);
                push_compact_value_color(buf, val);
            }
            c!(COLOR_OBJECT); buf.push(b'}'); c!(COLOR_RESET);
        }
        Value::Error(e) => push_json_string_to_vec(buf, e.as_str()),
    }
}

/// Push pretty-printed JSON + newline directly into a Vec<u8> buffer.
/// Mirrors push_compact_line but with indentation. Avoids per-value write_all overhead.
pub fn push_pretty_line(buf: &mut Vec<u8>, v: &Value, indent: usize, use_tab: bool) {
    push_pretty_value(buf, v, 0, indent, use_tab, false);
    buf.push(b'\n');
}

pub fn push_pretty_line_color(buf: &mut Vec<u8>, v: &Value, indent: usize, use_tab: bool) {
    push_pretty_value(buf, v, 0, indent, use_tab, true);
    buf.push(b'\n');
}

fn push_pretty_value_impl<const COLOR: bool>(buf: &mut Vec<u8>, v: &Value, depth: usize, step: usize, use_tab: bool) {
    // jq's --tab uses exactly one tab per indent level regardless of --indent.
    let step = if use_tab { 1 } else { step };
    macro_rules! c {
        ($code:expr) => { if COLOR { buf.extend_from_slice($code.as_bytes()); } };
    }
    match v {
        Value::Null => { c!(COLOR_NULL); buf.extend_from_slice(b"null"); c!(COLOR_RESET); }
        Value::False => { c!(COLOR_FALSE); buf.extend_from_slice(b"false"); c!(COLOR_RESET); }
        Value::True => { c!(COLOR_TRUE); buf.extend_from_slice(b"true"); c!(COLOR_RESET); }
        Value::Num(n, NumRepr(repr)) => {
            c!(COLOR_NUMBER);
            if let Some(r) = repr.as_ref().filter(|r| is_valid_json_number(r)) {
                buf.extend_from_slice(canonical_repr_bytes(r).as_bytes());
            } else {
                push_jq_number_bytes(buf, *n);
            }
            c!(COLOR_RESET);
        }
        Value::Str(s) => { c!(COLOR_STRING); push_json_string_to_vec(buf, s.as_str()); c!(COLOR_RESET); }
        Value::Error(e) => push_json_string_to_vec(buf, e.as_str()),
        Value::Arr(a) if a.is_empty() => { c!(COLOR_ARRAY); buf.extend_from_slice(b"[]"); c!(COLOR_RESET); }
        Value::Obj(ObjInner(o)) if o.is_empty() => { c!(COLOR_OBJECT); buf.extend_from_slice(b"{}"); c!(COLOR_RESET); }
        Value::Arr(a) => {
            let inner = depth + step;
            // Reset color before every newline so terminals that paint
            // the rest of the line don't extend the structural-character
            // color. Matches jq -C byte-for-byte.
            c!(COLOR_ARRAY); buf.push(b'['); c!(COLOR_RESET); buf.push(b'\n');
            for (i, item) in a.iter().enumerate() {
                if i > 0 { c!(COLOR_ARRAY); buf.push(b','); c!(COLOR_RESET); buf.push(b'\n'); }
                push_indent_bytes(buf, inner, use_tab);
                push_pretty_value_impl::<COLOR>(buf, item, inner, step, use_tab);
            }
            buf.push(b'\n');
            push_indent_bytes(buf, depth, use_tab);
            c!(COLOR_ARRAY); buf.push(b']'); c!(COLOR_RESET);
        }
        Value::Obj(ObjInner(o)) => {
            let inner = depth + step;
            c!(COLOR_OBJECT); buf.push(b'{'); c!(COLOR_RESET); buf.push(b'\n');
            for (i, (k, val)) in o.iter().enumerate() {
                if i > 0 { c!(COLOR_OBJECT); buf.push(b','); c!(COLOR_RESET); buf.push(b'\n'); }
                push_indent_bytes(buf, inner, use_tab);
                c!(COLOR_KEY); push_json_string_to_vec(buf, k.as_str()); c!(COLOR_RESET);
                c!(COLOR_OBJECT); buf.push(b':'); c!(COLOR_RESET); buf.push(b' ');
                push_pretty_value_impl::<COLOR>(buf, val, inner, step, use_tab);
            }
            buf.push(b'\n');
            push_indent_bytes(buf, depth, use_tab);
            c!(COLOR_OBJECT); buf.push(b'}'); c!(COLOR_RESET);
        }
    }
}

#[inline]
fn push_pretty_value(buf: &mut Vec<u8>, v: &Value, depth: usize, step: usize, use_tab: bool, color: bool) {
    if color {
        push_pretty_value_impl::<true>(buf, v, depth, step, use_tab);
    } else {
        push_pretty_value_impl::<false>(buf, v, depth, step, use_tab);
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
        Value::Num(n, NumRepr(repr)) => {
            if let Some(r) = repr.as_ref().filter(|r| is_valid_json_number(r)) {
                buf.extend_from_slice(canonical_repr_bytes(r).as_bytes());
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
        Value::Obj(ObjInner(o)) => {
            buf.push(b'{');
            for (i, (k, val)) in o.iter().enumerate() {
                let kb = k.as_bytes();
                let klen = kb.len();
                let key_needs_escape = kb.iter().any(|&b| b == b'"' || b == b'\\' || b < 0x20 || b == 0x7f);
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
    // Fast path: exact integer in displayable range (covers vast majority of JSON numbers).
    // The i64 roundtrip check (i as f64 == n) naturally rejects NaN, infinity, and non-integers.
    let i = n as i64;
    if i as f64 == n && i.unsigned_abs() < 10_000_000_000_000_000 {
        // Negative zero round-trips through `as i64` as `0`, which would
        // erase the sign. Emit `-0` directly (issue #110).
        if i == 0 && n.is_sign_negative() {
            buf.extend_from_slice(b"-0");
            return;
        }
        let mut ibuf = itoa::Buffer::new();
        buf.extend_from_slice(ibuf.format(i).as_bytes());
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
    // Common decimal case: use ryu for fast f64-to-decimal conversion
    let mut rbuf = ryu::Buffer::new();
    let s = rbuf.format(n);
    buf.extend_from_slice(s.as_bytes());
}

#[inline]
fn push_json_string_to_vec(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let len = bytes.len();
    // Ultra-fast path for single-byte strings (common object keys like "x", "y")
    if len == 1 {
        let c = bytes[0];
        if c >= 0x20 && c != b'"' && c != b'\\' && c != 0x7f {
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
    let needs_escape = bytes.iter().any(|&b| b == b'"' || b == b'\\' || b < 0x20 || b == 0x7f);
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
            c if c < 0x20 || c == 0x7f => {
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
        Value::Num(n, NumRepr(repr)) => {
            if let Some(r) = repr.as_ref().filter(|r| is_valid_json_number(r)) {
                let canonical = canonical_repr_bytes(r);
                push!(canonical.as_bytes());
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
            let needs_escape = bytes.iter().any(|&b| b == b'"' || b == b'\\' || b < 0x20 || b == 0x7f);
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
        Value::Obj(ObjInner(o)) => {
            push!(b"{");
            for (i, (k, val)) in o.iter().enumerate() {
                if i > 0 { push!(b","); }
                // Write key
                let kb = k.as_bytes();
                let key_escape = kb.iter().any(|&b| b == b'"' || b == b'\\' || b < 0x20 || b == 0x7f);
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
        Value::Num(n, NumRepr(repr)) => {
            if let Some(r) = repr.as_ref().filter(|r| is_valid_json_number(r)) {
                w.write_all(canonical_repr_bytes(r).as_bytes())
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
        Value::Obj(ObjInner(o)) => {
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
