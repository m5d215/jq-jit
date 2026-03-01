//! Filter cache: stores compiled shared libraries and reloads them via dlopen.
//!
//! On first use of a filter, the cache compiles it to a shared library (.dylib/.so)
//! and stores it in ~/.cache/jq-jit/. On subsequent uses, it loads the cached
//! library via dlopen, avoiding JIT compilation overhead.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::codegen::compile_to_shared_object;
use crate::cps_ir::Expr;
use crate::value::Value;

/// A filter loaded from a cached shared library.
pub struct CachedFilter {
    /// Function pointer to the compiled filter.
    fn_ptr: *const u8,
    /// Handle from dlopen (must stay open while fn_ptr is in use).
    _lib_handle: *mut std::ffi::c_void,
    /// Reconstructed literal values (kept alive for the pointer table).
    _literals: Vec<Box<Value>>,
    /// Pointer table passed as the 4th argument to the filter function.
    literal_ptrs: Vec<*const Value>,
}

impl CachedFilter {
    /// Execute the cached filter on the given input.
    pub fn execute(&self, input: &Value) -> Vec<Value> {
        let mut results: Vec<Value> = Vec::new();

        type FilterFn =
            fn(*const Value, extern "C" fn(*const Value, *mut u8), *mut u8, *const *const Value);
        let fn_typed: FilterFn =
            unsafe { std::mem::transmute::<*const u8, FilterFn>(self.fn_ptr) };

        fn_typed(
            input as *const Value,
            collect_callback,
            &mut results as *mut Vec<Value> as *mut u8,
            self.literal_ptrs.as_ptr(),
        );

        // Filter out errors (same as JitFilter::execute)
        results.retain(|v| {
            if let Value::Error(msg) = v {
                eprintln!("jq: error (at <stdin>:0): {}", msg);
                false
            } else {
                true
            }
        });

        results
    }
}

impl Drop for CachedFilter {
    fn drop(&mut self) {
        if !self._lib_handle.is_null() {
            unsafe {
                libc::dlclose(self._lib_handle);
            }
        }
    }
}

// Re-use the collect_callback from codegen (it's the same signature).
// Duplicated here intentionally — cache.rs doesn't depend on the private
// function in codegen.rs.
extern "C" fn collect_callback(value_ptr: *const Value, ctx: *mut u8) {
    assert!(!value_ptr.is_null() && !ctx.is_null());
    let results = unsafe { &mut *(ctx as *mut Vec<Value>) };
    let value = unsafe { (*value_ptr).clone() };
    results.push(value);
}

/// Cache manager for compiled filter shared libraries.
pub struct FilterCache {
    cache_dir: PathBuf,
}

impl FilterCache {
    /// Create a new cache manager. Creates cache directory if needed.
    pub fn new() -> Self {
        let dir = std::env::var("HOME")
            .map(|h| PathBuf::from(h).join(".cache").join("jq-jit"))
            .unwrap_or_else(|_| PathBuf::from("/tmp/jq-jit-cache"));
        std::fs::create_dir_all(&dir).ok();
        Self { cache_dir: dir }
    }

    /// Create a cache manager with a custom cache directory.
    pub fn with_dir(dir: PathBuf) -> Self {
        std::fs::create_dir_all(&dir).ok();
        Self { cache_dir: dir }
    }

    /// Generate a cache key from the filter string.
    /// Includes the package version so cache is invalidated on upgrades.
    fn cache_key(&self, filter: &str) -> String {
        let mut hasher = DefaultHasher::new();
        filter.hash(&mut hasher);
        env!("CARGO_PKG_VERSION").hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Try to load a cached filter. Returns None on cache miss.
    pub fn load(&self, filter: &str) -> Option<CachedFilter> {
        let key = self.cache_key(filter);
        let lib_ext = if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let dylib_path = self.cache_dir.join(format!("{}.{}", key, lib_ext));
        let meta_path = self.cache_dir.join(format!("{}.meta", key));

        if !dylib_path.exists() || !meta_path.exists() {
            return None;
        }

        // dlopen the shared library
        let c_path = std::ffi::CString::new(dylib_path.to_str()?).ok()?;
        let lib = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW) };
        if lib.is_null() {
            return None;
        }

        // dlsym to find the filter function
        let sym_name = std::ffi::CString::new("jit_filter").ok()?;
        let sym = unsafe { libc::dlsym(lib, sym_name.as_ptr()) };
        if sym.is_null() {
            unsafe {
                libc::dlclose(lib);
            }
            return None;
        }

        // Reconstruct literals from metadata
        let meta_bytes = std::fs::read(&meta_path).ok()?;
        let literals = deserialize_literals(&meta_bytes)?;
        let literal_ptrs: Vec<*const Value> = literals.iter().map(|b| &**b as *const Value).collect();

        Some(CachedFilter {
            fn_ptr: sym as *const u8,
            _lib_handle: lib,
            _literals: literals,
            literal_ptrs,
        })
    }

    /// Compile the filter to a shared library and cache it.
    /// Returns the CachedFilter ready to execute.
    pub fn compile_and_store(&self, filter: &str, expr: &Expr) -> Result<CachedFilter> {
        let key = self.cache_key(filter);
        let lib_ext = if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let dylib_path = self.cache_dir.join(format!("{}.{}", key, lib_ext));
        let meta_path = self.cache_dir.join(format!("{}.meta", key));

        // Compile to shared library
        let literals = compile_to_shared_object(expr, &dylib_path)?;

        // Serialize literal metadata
        let meta_bytes = serialize_literals(&literals);
        std::fs::write(&meta_path, &meta_bytes).context("writing cache metadata")?;

        // Load the just-compiled library
        self.load(filter)
            .ok_or_else(|| anyhow::anyhow!("failed to load just-compiled cached filter"))
    }

    /// Clear all cached files.
    pub fn clear(&self) -> Result<()> {
        if self.cache_dir.exists() {
            for entry in std::fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                std::fs::remove_file(entry.path()).ok();
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Literal serialization/deserialization
// ---------------------------------------------------------------------------

// Format: [version: u8] [count: u32-LE] [entries...]
// Each entry: [type: u8] [data...]
// Types: 0=Null, 1=Bool, 2=Num, 3=Str, 4=EmptyArr, 5=Arr, 6=EmptyObj, 7=Obj, 8=Error

pub fn serialize_literals(literals: &[Box<Value>]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(1); // version
    buf.extend_from_slice(&(literals.len() as u32).to_le_bytes());
    for lit in literals {
        serialize_value(lit, &mut buf);
    }
    buf
}

fn serialize_value(val: &Value, buf: &mut Vec<u8>) {
    match val {
        Value::Null => buf.push(0),
        Value::Bool(b) => {
            buf.push(1);
            buf.push(if *b { 1 } else { 0 });
        }
        Value::Num(n) => {
            buf.push(2);
            buf.extend_from_slice(&n.to_le_bytes());
        }
        Value::Str(s) => {
            buf.push(3);
            let bytes = s.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
        Value::Arr(items) => {
            if items.is_empty() {
                buf.push(4);
            } else {
                buf.push(5);
                buf.extend_from_slice(&(items.len() as u32).to_le_bytes());
                for item in items.iter() {
                    serialize_value(item, buf);
                }
            }
        }
        Value::Obj(map) => {
            if map.is_empty() {
                buf.push(6);
            } else {
                buf.push(7);
                buf.extend_from_slice(&(map.len() as u32).to_le_bytes());
                for (k, v) in map.iter() {
                    let key_bytes = k.as_bytes();
                    buf.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
                    buf.extend_from_slice(key_bytes);
                    serialize_value(v, buf);
                }
            }
        }
        Value::Error(msg) => {
            buf.push(8);
            let bytes = msg.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
    }
}

pub fn deserialize_literals(data: &[u8]) -> Option<Vec<Box<Value>>> {
    let mut pos = 0;
    if data.is_empty() || data[pos] != 1 {
        return None;
    } // version check
    pos += 1;
    if pos + 4 > data.len() {
        return None;
    }
    let count = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
    pos += 4;
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let (val, new_pos) = deserialize_value(data, pos)?;
        result.push(Box::new(val));
        pos = new_pos;
    }
    Some(result)
}

fn deserialize_value(data: &[u8], pos: usize) -> Option<(Value, usize)> {
    if pos >= data.len() {
        return None;
    }
    let typ = data[pos];
    let mut p = pos + 1;
    match typ {
        0 => Some((Value::Null, p)),
        1 => {
            if p >= data.len() {
                return None;
            }
            let b = data[p] != 0;
            Some((Value::Bool(b), p + 1))
        }
        2 => {
            if p + 8 > data.len() {
                return None;
            }
            let n = f64::from_le_bytes(data[p..p + 8].try_into().ok()?);
            Some((Value::Num(n), p + 8))
        }
        3 => {
            if p + 4 > data.len() {
                return None;
            }
            let len = u32::from_le_bytes(data[p..p + 4].try_into().ok()?) as usize;
            p += 4;
            if p + len > data.len() {
                return None;
            }
            let s = std::str::from_utf8(&data[p..p + len]).ok()?;
            Some((Value::Str(std::rc::Rc::new(s.to_string())), p + len))
        }
        4 => Some((Value::Arr(std::rc::Rc::new(Vec::new())), p)),
        5 => {
            if p + 4 > data.len() {
                return None;
            }
            let count = u32::from_le_bytes(data[p..p + 4].try_into().ok()?) as usize;
            p += 4;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                let (val, new_p) = deserialize_value(data, p)?;
                items.push(val);
                p = new_p;
            }
            Some((Value::Arr(std::rc::Rc::new(items)), p))
        }
        6 => Some((
            Value::Obj(std::rc::Rc::new(std::collections::BTreeMap::new())),
            p,
        )),
        7 => {
            if p + 4 > data.len() {
                return None;
            }
            let count = u32::from_le_bytes(data[p..p + 4].try_into().ok()?) as usize;
            p += 4;
            let mut map = std::collections::BTreeMap::new();
            for _ in 0..count {
                if p + 4 > data.len() {
                    return None;
                }
                let klen = u32::from_le_bytes(data[p..p + 4].try_into().ok()?) as usize;
                p += 4;
                if p + klen > data.len() {
                    return None;
                }
                let key = std::str::from_utf8(&data[p..p + klen]).ok()?.to_string();
                p += klen;
                let (val, new_p) = deserialize_value(data, p)?;
                map.insert(key, val);
                p = new_p;
            }
            Some((Value::Obj(std::rc::Rc::new(map)), p))
        }
        8 => {
            if p + 4 > data.len() {
                return None;
            }
            let len = u32::from_le_bytes(data[p..p + 4].try_into().ok()?) as usize;
            p += 4;
            if p + len > data.len() {
                return None;
            }
            let s = std::str::from_utf8(&data[p..p + len]).ok()?;
            Some((Value::Error(std::rc::Rc::new(s.to_string())), p + len))
        }
        _ => None,
    }
}
