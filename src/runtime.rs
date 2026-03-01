//! Runtime helper functions for JIT-compiled code.
//!
//! These `extern "C"` functions are registered with Cranelift's `JITBuilder`
//! and called by JIT-generated native code.  Two calling conventions are used:
//!
//! **Binary (3-ptr)**: `fn(out: *mut Value, a: *const Value, b: *const Value)`
//!   - Arithmetic: rt_add, rt_sub, rt_mul, rt_div, rt_mod
//!   - Comparison: rt_eq, rt_ne, rt_lt, rt_gt, rt_le, rt_ge
//!   - Index: rt_index
//!
//! **Unary (2-ptr)**: `fn(out: *mut Value, v: *const Value)`
//!   - rt_length, rt_type, rt_tostring, rt_tonumber, rt_keys
//!
//! # Safety
//!
//! All pointer arguments must be non-null and properly aligned (8-byte).
//! The caller (JIT code) is responsible for allocating output buffers on the
//! stack via Cranelift StackSlots.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::value::{Value, TAG_NUM, value_to_json, json_to_value};

/// Phase 6-2: Fast path for binary numeric operations.
/// Checks tags directly and performs the f64 operation without full match dispatch.
/// Value layout: [tag: u64 at offset 0][payload: f64 at offset 8]
macro_rules! fast_num_bin {
    ($out:expr, $a:expr, $b:expr, $op:tt) => {
        unsafe {
            let a_tag = *($a as *const u64);
            let b_tag = *($b as *const u64);
            if a_tag == TAG_NUM && b_tag == TAG_NUM {
                let a_val = *(($a as *const u8).add(8) as *const f64);
                let b_val = *(($b as *const u8).add(8) as *const f64);
                let result = a_val $op b_val;
                // Write tag
                *($out as *mut u64) = TAG_NUM;
                // Write f64 payload
                *(($out as *mut u8).add(8) as *mut f64) = result;
                return;
            }
        }
    };
}

/// Phase 6-2: Fast path for binary numeric comparison.
/// Similar to fast_num_bin but writes a Bool result instead of Num.
macro_rules! fast_num_cmp {
    ($out:expr, $a:expr, $b:expr, $op:tt) => {
        unsafe {
            let a_tag = *($a as *const u64);
            let b_tag = *($b as *const u64);
            if a_tag == TAG_NUM && b_tag == TAG_NUM {
                let a_val = *(($a as *const u8).add(8) as *const f64);
                let b_val = *(($b as *const u8).add(8) as *const f64);
                let result = a_val $op b_val;
                // Write Bool tag (TAG_BOOL = 1)
                *($out as *mut u64) = crate::value::TAG_BOOL;
                // Write bool payload (true = 1, false = 0)
                *(($out as *mut u8).add(8) as *mut u64) = result as u64;
                return;
            }
        }
    };
}

/// Propagate Error values through binary operations.
/// If either input is an Error, write it to `out` and return early.
macro_rules! propagate_error_bin {
    ($out:expr, $a:expr, $b:expr) => {
        if let Value::Error(_) = $a {
            unsafe { std::ptr::write($out, $a.clone()); }
            return;
        }
        if let Value::Error(_) = $b {
            unsafe { std::ptr::write($out, $b.clone()); }
            return;
        }
    };
}

/// Propagate Error values through unary operations.
/// If the input is an Error, write it to `out` and return early.
macro_rules! propagate_error_unary {
    ($out:expr, $v:expr) => {
        if let Value::Error(_) = $v {
            unsafe { std::ptr::write($out, $v.clone()); }
            return;
        }
    };
}

/// Arithmetic addition: `a + b`.
///
/// Supports: Num+Num, Str+Str (concat), Arr+Arr (concat), Obj+Obj (merge), Null+X, X+Null.
#[unsafe(no_mangle)]
pub extern "C" fn rt_add(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    // Phase 6-2: Fast path for Num+Num (skips Error check and match dispatch)
    fast_num_bin!(out, a, b, +);
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    propagate_error_bin!(out, a, b);
    let result = match (a, b) {
        (Value::Num(x), Value::Num(y)) => Value::Num(x + y),
        (Value::Str(x), Value::Str(y)) => {
            let mut s = String::with_capacity(x.len() + y.len());
            s.push_str(x);
            s.push_str(y);
            Value::Str(Rc::new(s))
        }
        (Value::Arr(x), Value::Arr(y)) => {
            let mut v = Vec::with_capacity(x.len() + y.len());
            v.extend(x.iter().cloned());
            v.extend(y.iter().cloned());
            Value::Arr(Rc::new(v))
        }
        (Value::Obj(x), Value::Obj(y)) => {
            let mut m = (**x).clone();
            for (k, v) in y.iter() {
                m.insert(k.clone(), v.clone());
            }
            Value::Obj(Rc::new(m))
        }
        (Value::Null, other) => other.clone(),
        (other, Value::Null) => other.clone(),
        _ => Value::Error(Rc::new(format!(
            "{} and {} cannot be added",
            type_desc(a),
            type_desc(b)
        ))),
    };
    unsafe {
        std::ptr::write(out, result);
    }
}

/// Arithmetic subtraction: `a - b`.
#[unsafe(no_mangle)]
pub extern "C" fn rt_sub(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    fast_num_bin!(out, a, b, -);
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    propagate_error_bin!(out, a, b);
    let result = match (a, b) {
        (Value::Num(x), Value::Num(y)) => Value::Num(x - y),
        _ => Value::Error(Rc::new(format!(
            "{} and {} cannot be subtracted",
            type_desc(a),
            type_desc(b)
        ))),
    };
    unsafe {
        std::ptr::write(out, result);
    }
}

/// Arithmetic multiplication: `a * b`.
#[unsafe(no_mangle)]
pub extern "C" fn rt_mul(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    fast_num_bin!(out, a, b, *);
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    propagate_error_bin!(out, a, b);
    let result = match (a, b) {
        (Value::Num(x), Value::Num(y)) => Value::Num(x * y),
        // jq: string * number = repeat string n times
        (Value::Str(s), Value::Num(n)) | (Value::Num(n), Value::Str(s)) => {
            let n = *n as usize;
            if s.is_empty() || n == 0 {
                Value::Str(Rc::new(String::new()))
            } else {
                Value::Str(Rc::new(s.repeat(n)))
            }
        }
        // jq: null * anything = null, anything * null = null
        (Value::Null, _) | (_, Value::Null) => Value::Null,
        // jq: object * object = recursive merge
        (Value::Obj(x), Value::Obj(y)) => {
            recursive_obj_merge(x, y)
        }
        _ => Value::Error(Rc::new(format!(
            "{} and {} cannot be multiplied",
            type_name(a),
            type_name(b)
        ))),
    };
    unsafe {
        std::ptr::write(out, result);
    }
}

/// Arithmetic division: `a / b`.
///
/// Supports: Num/Num, Str/Str (split). Division by zero produces an error.
#[unsafe(no_mangle)]
pub extern "C" fn rt_div(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    // No fast_num_bin! here: we need to check for division by zero.
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    propagate_error_bin!(out, a, b);
    let result = match (a, b) {
        (Value::Num(x), Value::Num(y)) => {
            if *y == 0.0 {
                let x_str = if *x == x.trunc() && x.abs() < 1e18 {
                    format!("{}", *x as i64)
                } else {
                    format!("{}", x)
                };
                Value::Error(Rc::new(format!(
                    "number ({}) and number (0) cannot be divided because the divisor is zero",
                    x_str
                )))
            } else {
                Value::Num(x / y)
            }
        }
        (Value::Str(s), Value::Str(d)) => {
            // jq: "a,b,c" / "," => ["a","b","c"]
            if d.is_empty() {
                let parts: Vec<Value> = s.chars()
                    .map(|c| Value::Str(Rc::new(c.to_string())))
                    .collect();
                Value::Arr(Rc::new(parts))
            } else {
                let parts: Vec<Value> = s.split(d.as_str())
                    .map(|p| Value::Str(Rc::new(p.to_string())))
                    .collect();
                Value::Arr(Rc::new(parts))
            }
        }
        _ => Value::Error(Rc::new(format!(
            "{} and {} cannot be divided",
            type_name(a),
            type_name(b)
        ))),
    };
    unsafe {
        std::ptr::write(out, result);
    }
}

/// Arithmetic modulo: `a % b`.
///
/// Modulo by zero produces an error.
#[unsafe(no_mangle)]
pub extern "C" fn rt_mod(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    // No fast_num_bin! here: we need to check for modulo by zero.
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    propagate_error_bin!(out, a, b);
    let result = match (a, b) {
        (Value::Num(x), Value::Num(y)) => {
            if *y == 0.0 {
                let x_str = if *x == x.trunc() && x.abs() < 1e18 {
                    format!("{}", *x as i64)
                } else {
                    format!("{}", x)
                };
                Value::Error(Rc::new(format!(
                    "number ({}) and number (0) cannot be divided (remainder) because the divisor is zero",
                    x_str
                )))
            } else {
                Value::Num(x % y)
            }
        }
        _ => Value::Error(Rc::new(format!(
            "{} and {} cannot be modulo'd",
            type_name(a),
            type_name(b)
        ))),
    };
    unsafe {
        std::ptr::write(out, result);
    }
}

/// Field/index access: `obj[key]`.
///
/// Supports Obj[Str] and Arr[Num] access.
/// For unsupported types (e.g., indexing a string or number), returns Value::Error
/// instead of panicking — this enables try-catch to work.
/// For null input, returns null (jq semantics: null.foo == null).
#[unsafe(no_mangle)]
pub extern "C" fn rt_index(out: *mut Value, obj: *const Value, key: *const Value) {
    assert!(!out.is_null() && !obj.is_null() && !key.is_null());
    let obj = unsafe { &*obj };
    let key = unsafe { &*key };
    propagate_error_bin!(out, obj, key);
    let result = match (obj, key) {
        (Value::Obj(map), Value::Str(k)) => {
            map.get(k.as_str()).cloned().unwrap_or(Value::Null)
        }
        (Value::Arr(arr), Value::Num(idx)) => {
            if idx.is_nan() || idx.is_infinite() {
                Value::Null
            } else {
                let i = *idx as i64;
                if i < 0 {
                    let resolved = arr.len() as i64 + i;
                    if resolved >= 0 {
                        arr.get(resolved as usize).cloned().unwrap_or(Value::Null)
                    } else {
                        Value::Null
                    }
                } else {
                    arr.get(i as usize).cloned().unwrap_or(Value::Null)
                }
            }
        }
        // Slice: key is an object with "start" and/or "end" fields
        (Value::Arr(arr), Value::Obj(slice_obj)) => {
            slice_array(arr, slice_obj)
        }
        (Value::Str(s), Value::Obj(slice_obj)) => {
            slice_string(s, slice_obj)
        }
        (Value::Str(s), Value::Num(idx)) => {
            // String indexing: s[n] returns a single-character string
            if idx.is_nan() || idx.is_infinite() || *idx != idx.trunc() {
                return unsafe { std::ptr::write(out, Value::Error(Rc::new(format!(
                    "Cannot index string with number ({})",
                    crate::value::format_jq_number(*idx)
                )))) };
            }
            let i = *idx as i64;
            let len = s.len() as i64;
            let resolved = if i < 0 { len + i } else { i };
            if resolved >= 0 && (resolved as usize) < s.len() {
                // jq does codepoint-based indexing, but for simplicity use byte-based
                // (which is correct for ASCII)
                let ch = s.as_bytes()[resolved as usize];
                Value::Num(ch as f64)
            } else {
                Value::Null
            }
        }
        (Value::Null, _) => Value::Null,
        _ => Value::Error(Rc::new(format!(
            "Cannot index {} with {}",
            type_name(obj),
            type_desc(key)
        ))),
    };
    unsafe {
        std::ptr::write(out, result);
    }
}

/// Resolve slice bounds for an array/string of given length.
/// Handles negative indices (from end), null (default), float truncation.
/// Returns (start, end) clamped to [0, len].
fn resolve_slice_bounds(slice_obj: &BTreeMap<String, Value>, len: usize) -> (usize, usize) {
    let len_i = len as i64;

    let start = match slice_obj.get("start") {
        Some(Value::Num(n)) if !n.is_nan() => {
            let s = n.floor() as i64; // floor for start index
            if s < 0 {
                (len_i + s).max(0) as usize
            } else {
                s.min(len_i) as usize
            }
        }
        _ => 0, // null, missing, or NaN → 0
    };

    let end = match slice_obj.get("end") {
        Some(Value::Num(n)) if !n.is_nan() => {
            let e = n.ceil() as i64; // ceil for end index (jq behavior)
            if e < 0 {
                (len_i + e).max(0) as usize
            } else {
                e.min(len_i) as usize
            }
        }
        _ => len, // null, missing, or NaN → len
    };

    // If start > end, return empty range
    if start >= end {
        (0, 0)
    } else {
        (start, end)
    }
}

/// Slice an array: arr[start:end]
fn slice_array(arr: &[Value], slice_obj: &BTreeMap<String, Value>) -> Value {
    let (start, end) = resolve_slice_bounds(slice_obj, arr.len());
    if start >= end {
        Value::Arr(Rc::new(Vec::new()))
    } else {
        Value::Arr(Rc::new(arr[start..end].to_vec()))
    }
}

/// Slice a string: str[start:end]
fn slice_string(s: &str, slice_obj: &BTreeMap<String, Value>) -> Value {
    // jq does codepoint-based slicing
    let chars: Vec<char> = s.chars().collect();
    let (start, end) = resolve_slice_bounds(slice_obj, chars.len());
    if start >= end {
        Value::Str(Rc::new(String::new()))
    } else {
        Value::Str(Rc::new(chars[start..end].iter().collect()))
    }
}

// =========================================================================
// Phase 2: Comparison operators (binary, Num-only for now)
// =========================================================================

/// Equality: `a == b`.
#[unsafe(no_mangle)]
pub extern "C" fn rt_eq(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    propagate_error_bin!(out, a, b);
    let result = Value::Bool(values_equal_deep(a, b));
    unsafe { std::ptr::write(out, result); }
}

/// Not-equal: `a != b`.
#[unsafe(no_mangle)]
pub extern "C" fn rt_ne(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    propagate_error_bin!(out, a, b);
    let result = Value::Bool(!values_equal_deep(a, b));
    unsafe { std::ptr::write(out, result); }
}

/// Deep equality comparison for jq values.
fn values_equal_deep(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Arr(x), Value::Arr(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| values_equal_deep(a, b))
        }
        (Value::Obj(x), Value::Obj(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| {
                y.get(k).map_or(false, |yv| values_equal_deep(v, yv))
            })
        }
        _ => false,
    }
}

/// jq type ordering for cross-type comparisons.
/// null(0) < false(1) < true(2) < numbers(3) < strings(4) < arrays(5) < objects(6)
fn type_order(v: &Value) -> u8 {
    match v {
        Value::Null => 0,
        Value::Bool(false) => 1,
        Value::Bool(true) => 2,
        Value::Num(_) => 3,
        Value::Str(_) => 4,
        Value::Arr(_) => 5,
        Value::Obj(_) => 6,
        Value::Error(_) => 7,
    }
}

/// Compare two jq values using jq's ordering rules.
/// Returns Ordering for same-type and cross-type comparisons.
fn value_cmp(a: &Value, b: &Value) -> std::cmp::Ordering {
    let ta = type_order(a);
    let tb = type_order(b);
    if ta != tb {
        return ta.cmp(&tb);
    }
    // Same type: deep comparison
    value_compare(a, b)
}

/// Less-than: `a < b`.
///
/// Uses jq's type ordering for cross-type comparisons.
#[unsafe(no_mangle)]
pub extern "C" fn rt_lt(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    fast_num_cmp!(out, a, b, <);
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    propagate_error_bin!(out, a, b);
    let result = Value::Bool(value_cmp(a, b) == std::cmp::Ordering::Less);
    unsafe { std::ptr::write(out, result); }
}

/// Greater-than: `a > b`.
#[unsafe(no_mangle)]
pub extern "C" fn rt_gt(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    fast_num_cmp!(out, a, b, >);
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    propagate_error_bin!(out, a, b);
    let result = Value::Bool(value_cmp(a, b) == std::cmp::Ordering::Greater);
    unsafe { std::ptr::write(out, result); }
}

/// Less-or-equal: `a <= b`.
#[unsafe(no_mangle)]
pub extern "C" fn rt_le(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    fast_num_cmp!(out, a, b, <=);
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    propagate_error_bin!(out, a, b);
    let result = Value::Bool(value_cmp(a, b) != std::cmp::Ordering::Greater);
    unsafe { std::ptr::write(out, result); }
}

/// Greater-or-equal: `a >= b`.
#[unsafe(no_mangle)]
pub extern "C" fn rt_ge(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    fast_num_cmp!(out, a, b, >=);
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    propagate_error_bin!(out, a, b);
    let result = Value::Bool(value_cmp(a, b) != std::cmp::Ordering::Less);
    unsafe { std::ptr::write(out, result); }
}

// =========================================================================
// Phase 2: Unary builtins (2-ptr signature: out, v)
// =========================================================================

/// `length` builtin.
///
/// Str→char count, Arr→element count, Obj→key count, Null→0.
#[unsafe(no_mangle)]
pub extern "C" fn rt_length(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Str(s) => Value::Num(s.chars().count() as f64),
        Value::Arr(a) => Value::Num(a.len() as f64),
        Value::Obj(o) => Value::Num(o.len() as f64),
        Value::Null => Value::Num(0.0),
        Value::Num(n) => Value::Num(n.abs()),
        Value::Bool(_) => Value::Null,
        _ => Value::Error(Rc::new(format!("{} has no length", type_name(v)))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `type` builtin.
///
/// Returns the type name as a string.
#[unsafe(no_mangle)]
pub extern "C" fn rt_type(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let type_name = match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Num(_) => "number",
        Value::Str(_) => "string",
        Value::Arr(_) => "array",
        Value::Obj(_) => "object",
        Value::Error(_) => "error",
    };
    let result = Value::Str(Rc::new(type_name.to_string()));
    unsafe { std::ptr::write(out, result); }
}

/// `tostring` builtin.
///
/// Num→decimal string, Str→itself, Null→"null", Bool→"true"/"false".
#[unsafe(no_mangle)]
pub extern "C" fn rt_tostring(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Str(s) => Value::Str(Rc::clone(s)),
        Value::Num(n) => {
            Value::Str(Rc::new(crate::value::format_jq_number(*n)))
        }
        Value::Null => Value::Str(Rc::new("null".to_string())),
        Value::Bool(b) => Value::Str(Rc::new(if *b { "true" } else { "false" }.to_string())),
        Value::Arr(_) | Value::Obj(_) => {
            Value::Str(Rc::new(crate::value::value_to_json(v)))
        }
        _ => Value::Error(Rc::new(format!("cannot convert {} to string", type_name(v)))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `tonumber` builtin.
///
/// Str→parse as f64, Num→itself.
#[unsafe(no_mangle)]
pub extern "C" fn rt_tonumber(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Num(n) => Value::Num(*n),
        Value::Str(s) => {
            match s.parse::<f64>() {
                Ok(n) => Value::Num(n),
                Err(_) => Value::Error(Rc::new(format!("cannot parse {:?} as number", s))),
            }
        }
        _ => Value::Error(Rc::new(format!("cannot convert {} to number", type_name(v)))),
    };
    unsafe { std::ptr::write(out, result); }
}

// =========================================================================
// Phase 3: Truthy check for if-then-else
// =========================================================================

/// jq truthy check: `false` and `null` are falsy, everything else is truthy.
///
/// Returns 0 for falsy, 1 for truthy.
#[unsafe(no_mangle)]
pub extern "C" fn rt_is_truthy(v: *const Value) -> i32 {
    assert!(!v.is_null());
    let v = unsafe { &*v };
    match v {
        Value::Null | Value::Bool(false) => 0,
        _ => 1,
    }
}

// =========================================================================
// Phase 3: Error check for try-catch
// =========================================================================

/// Check if a Value is an error.
///
/// Returns 1 if the value is `Value::Error`, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn rt_is_error(v: *const Value) -> i32 {
    assert!(!v.is_null());
    let v = unsafe { &*v };
    match v {
        Value::Error(_) => 1,
        _ => 0,
    }
}

// =========================================================================
// Phase 10-5: Try-catch generator wrapper
// =========================================================================

/// Context structure for `rt_try_callback_wrapper`.
///
/// When `try (generator) catch expr` is used, the generator's callback is
/// replaced with `rt_try_callback_wrapper`. This wrapper checks each output
/// for errors: on the first error, it sets the flag and saves the error value,
/// then skips all subsequent outputs. After the generator completes, the JIT
/// code checks the flag and runs catch if needed.
///
/// Layout (must match the StackSlot allocation in codegen):
///   [0..8]   original_callback: fn(*const Value, *mut u8)
///   [8..16]  original_ctx: *mut u8
///   [16..20] error_flag: i32 (0 = no error, 1 = error occurred)
///   [20..24] _padding
///   [24..40] error_value: Value (16 bytes)
#[repr(C)]
pub struct TryCallbackCtx {
    pub original_callback: extern "C" fn(*const Value, *mut u8),
    pub original_ctx: *mut u8,
    pub error_flag: i32,
    pub _padding: i32,
    pub error_value: Value,
}

/// Wrapper callback for `try (generator) catch expr`.
///
/// Called for each output of the generator inside a try block.
/// On the first error, sets the error flag and saves the error value.
/// All subsequent outputs (error or not) are silently discarded.
/// Non-error outputs before the first error are forwarded to the original callback.
#[unsafe(no_mangle)]
pub extern "C" fn rt_try_callback_wrapper(value: *const Value, ctx: *mut u8) {
    assert!(!value.is_null() && !ctx.is_null());
    let ctx = unsafe { &mut *(ctx as *mut TryCallbackCtx) };

    // Already had an error? Skip all subsequent outputs.
    if ctx.error_flag != 0 {
        return;
    }

    let v = unsafe { &*value };
    if matches!(v, Value::Error(_)) {
        // First error: set flag and save error value.
        // IMPORTANT: Use ptr::write instead of assignment because error_value
        // may be uninitialized (stack memory). Assignment would try to drop
        // the old value, which could be garbage → SIGSEGV.
        ctx.error_flag = 1;
        unsafe {
            std::ptr::write(&mut ctx.error_value, v.clone());
        }
        return;
    }

    // No error: forward to original callback
    (ctx.original_callback)(value, ctx.original_ctx);
}

/// Context structure for `rt_limit_callback_wrapper`.
///
/// Used by `limit(n; gen)` to cap the number of outputs from a generator.
/// The wrapper counts outputs and stops forwarding after the limit is reached.
///
/// Layout (must match the StackSlot allocation in codegen):
///   [0..8]   original_callback: fn(*const Value, *mut u8)
///   [8..16]  original_ctx: *mut u8
///   [16..24] remaining: i64 (counts down from n to 0)
#[repr(C)]
pub struct LimitCallbackCtx {
    pub original_callback: extern "C" fn(*const Value, *mut u8),
    pub original_ctx: *mut u8,
    pub remaining: i64,
}

/// Wrapper callback for `limit(n; gen)`.
///
/// Called for each output of the generator inside a limit expression.
/// Forwards the output to the original callback if remaining > 0,
/// then decrements remaining. Once remaining reaches 0, all outputs are discarded.
#[unsafe(no_mangle)]
pub extern "C" fn rt_limit_callback_wrapper(value: *const Value, ctx: *mut u8) {
    assert!(!value.is_null() && !ctx.is_null());
    let ctx = unsafe { &mut *(ctx as *mut LimitCallbackCtx) };

    if ctx.remaining <= 0 {
        return;
    }

    ctx.remaining -= 1;
    (ctx.original_callback)(value, ctx.original_ctx);
}

/// Context structure for `rt_skip_callback_wrapper`.
///
/// Used by `skip(n; gen)` to skip the first n outputs from a generator.
/// The wrapper counts skipped outputs and starts forwarding after the skip count.
///
/// Layout (must match the StackSlot allocation in codegen):
///   [0..8]   original_callback: fn(*const Value, *mut u8)
///   [8..16]  original_ctx: *mut u8
///   [16..24] remaining_skip: i64 (counts down from n to 0)
#[repr(C)]
pub struct SkipCallbackCtx {
    pub original_callback: extern "C" fn(*const Value, *mut u8),
    pub original_ctx: *mut u8,
    pub remaining_skip: i64,
}

/// Wrapper callback for `skip(n; gen)`.
///
/// Called for each output of the generator inside a skip expression.
/// Skips the first n outputs, then forwards the rest to the original callback.
#[unsafe(no_mangle)]
pub extern "C" fn rt_skip_callback_wrapper(value: *const Value, ctx: *mut u8) {
    assert!(!value.is_null() && !ctx.is_null());
    let ctx = unsafe { &mut *(ctx as *mut SkipCallbackCtx) };

    if ctx.remaining_skip > 0 {
        ctx.remaining_skip -= 1;
        return;
    }

    (ctx.original_callback)(value, ctx.original_ctx);
}

/// Helper to get a human-readable type name for error messages.
fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Num(_) => "number",
        Value::Str(_) => "string",
        Value::Arr(_) => "array",
        Value::Obj(_) => "object",
        Value::Error(_) => "error",
    }
}

/// Format a value description like jq does for error messages.
/// Example: `string ("hello")`, `number (42)`, `array ([1,2,3])`
/// Long values are truncated with "..." (jq truncates at about 30 chars of the preview).
fn type_desc(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => format!("boolean ({})", if *b { "true" } else { "false" }),
        Value::Num(n) => format!("number ({})", crate::value::format_jq_number(*n)),
        Value::Str(_s) => {
            let json = value_to_json(v);
            // jq uses jv_dump_string_trunc(v, errbuf, sizeof(errbuf)) with
            // char errbuf[30]. The full JSON string (with quotes) must fit in
            // 29 bytes (30 - null). If it doesn't, truncate inner content to
            // max 24 bytes (29 - 1 opening quote - 4 closing '..."') at a
            // valid UTF-8 boundary.
            if json.len() > 29 {
                let inner = &json[1..json.len()-1]; // remove quotes
                let max_bytes = 24;
                let mut end = 0;
                for (i, _ch) in inner.char_indices() {
                    if i + _ch.len_utf8() > max_bytes {
                        break;
                    }
                    end = i + _ch.len_utf8();
                }
                format!("string (\"{}...\")", &inner[..end])
            } else {
                format!("string ({})", json)
            }
        }
        Value::Arr(_) => {
            let json = value_to_json(v);
            if json.len() > 30 {
                format!("array ({}...)", &json[..27])
            } else {
                format!("array ({})", json)
            }
        }
        Value::Obj(_) => {
            let json = value_to_json(v);
            if json.len() > 30 {
                format!("object ({}...)", &json[..27])
            } else {
                format!("object ({})", json)
            }
        }
        Value::Error(e) => format!("error ({})", e),
    }
}

/// `_negate` builtin: unary minus.
///
/// Num → negated Num.
#[unsafe(no_mangle)]
pub extern "C" fn rt_negate(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Num(n) => Value::Num(-n),
        _ => Value::Error(Rc::new(format!(
            "{} cannot be negated",
            type_desc(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// Phase 11: Make error value from input.
/// Converts the input value to a Value::Error.
/// The error message is the JSON representation of the input value.
#[unsafe(no_mangle)]
pub extern "C" fn rt_make_error(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    // If already an error, pass through
    if let Value::Error(_) = v {
        unsafe { std::ptr::write(out, v.clone()); }
        return;
    }
    let result = Value::Error(Rc::new(value_to_json(v)));
    unsafe { std::ptr::write(out, result); }
}

/// Phase 11: Extract error message from a Value::Error.
/// Converts Value::Error(json_msg) back to the original value by parsing the JSON.
/// For non-Error values, returns the value as-is.
#[unsafe(no_mangle)]
pub extern "C" fn rt_extract_error(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    let result = match v {
        Value::Error(msg) => {
            // Try to parse the JSON message back to a value
            match json_to_value(msg) {
                Ok(val) => val,
                Err(_) => Value::Str(msg.clone()),
            }
        }
        _ => v.clone(),
    };
    unsafe { std::ptr::write(out, result); }
}

// =========================================================================
// Phase 4-3: Each (.[] iteration) helpers
// =========================================================================

/// Get the length of an array or object (for .[] iteration).
///
/// Returns the number of elements (array) or key-value pairs (object).
/// For non-iterable types, returns 0.
#[unsafe(no_mangle)]
pub extern "C" fn rt_iter_length(v: *const Value) -> i64 {
    assert!(!v.is_null());
    let v = unsafe { &*v };
    match v {
        Value::Arr(arr) => arr.len() as i64,
        Value::Obj(obj) => obj.len() as i64,
        _ => 0,
    }
}

/// Get the i-th element from an array, or the i-th value from an object.
///
/// For arrays: returns arr[idx].
/// For objects: returns the idx-th value in insertion order (BTreeMap = sorted order).
/// Writes the result to `out`.
#[unsafe(no_mangle)]
pub extern "C" fn rt_iter_get(out: *mut Value, v: *const Value, idx: i64) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    let result = match v {
        Value::Arr(arr) => {
            let i = idx as usize;
            if i < arr.len() {
                arr[i].clone()
            } else {
                Value::Null
            }
        }
        Value::Obj(obj) => {
            let i = idx as usize;
            if let Some((_key, val)) = obj.iter().nth(i) {
                val.clone()
            } else {
                Value::Null
            }
        }
        _ => Value::Error(Rc::new(format!(
            "Cannot iterate over {}",
            type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// Prepare a value for O(1) indexed iteration.
///
/// For arrays: returns the array as-is (Rc clone).
/// For objects: extracts the values into a Vec and returns as Value::Arr.
/// This converts O(n) BTreeMap::iter().nth(i) into O(1) array indexing.
///
/// The prepared value should be used as the container for rt_iter_length/rt_iter_get.
#[unsafe(no_mangle)]
pub extern "C" fn rt_iter_prepare(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    let result = match v {
        Value::Arr(_) => v.clone(), // Rc clone only — O(1)
        Value::Obj(obj) => {
            let values: Vec<Value> = obj.values().cloned().collect();
            Value::Arr(Rc::new(values))
        }
        _ => v.clone(), // Non-iterable: pass through, rt_iter_get will handle the error
    };
    unsafe { std::ptr::write(out, result); }
}

/// Check if a value is iterable (array or object).
///
/// Returns 1 if iterable, 0 otherwise.
/// Used to generate errors for `.[]` on non-iterable types.
#[unsafe(no_mangle)]
pub extern "C" fn rt_is_iterable(v: *const Value) -> i32 {
    assert!(!v.is_null());
    let v = unsafe { &*v };
    match v {
        Value::Arr(_) | Value::Obj(_) => 1,
        _ => 0,
    }
}

/// Phase 11: Generate an error value for iterating over a non-iterable type.
///
/// Produces `Value::Error("Cannot iterate over {type} ({value})")`.
/// Used by the `Each` codegen when the input is not an array or object.
#[unsafe(no_mangle)]
pub extern "C" fn rt_iter_error(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    let msg = format!("Cannot iterate over {} ({})", type_name(v), value_to_json(v));
    unsafe { std::ptr::write(out, Value::Error(Rc::new(msg))); }
}

// =========================================================================
// Phase 5-1: Array constructor (Collect) helpers
// =========================================================================

/// Initialize a collect accumulator with an empty array.
///
/// Writes `Value::Arr(Rc::new(Vec::new()))` to `out`.
#[unsafe(no_mangle)]
pub extern "C" fn rt_collect_init(out: *mut Value) {
    assert!(!out.is_null());
    unsafe {
        std::ptr::write(out, Value::Arr(Rc::new(Vec::new())));
    }
}

/// Append a value to a collect accumulator array.
///
/// This function has the same signature as the JIT callback:
///   `fn(elem_ptr: *const Value, ctx: *mut u8)`
/// where `ctx` is actually a `*mut Value` pointing to the accumulator array.
///
/// The element at `elem_ptr` is cloned and appended to the array.
#[unsafe(no_mangle)]
pub extern "C" fn rt_collect_append(elem_ptr: *const Value, ctx: *mut u8) {
    assert!(!elem_ptr.is_null() && !ctx.is_null());
    let arr = unsafe { &mut *(ctx as *mut Value) };
    let elem = unsafe { (*elem_ptr).clone() };

    // Phase 11: If the accumulator is already an Error, skip all further appends.
    // This handles the case where a previous element was an error.
    if matches!(arr, Value::Error(_)) {
        return;
    }

    // If the element is an Error, replace the entire accumulator with the error.
    // In jq, an error during array construction propagates as an error.
    if let Value::Error(_) = &elem {
        unsafe { std::ptr::write(ctx as *mut Value, elem); }
        return;
    }

    if let Value::Arr(rc_vec) = arr {
        Rc::make_mut(rc_vec).push(elem);
    } else {
        panic!("rt_collect_append: expected Value::Arr, got {:?}", arr);
    }
}

/// Append a value to a collect accumulator array, including Error values.
///
/// Unlike `rt_collect_append`, this function does NOT propagate errors.
/// Error values are appended as regular array elements. This is used by
/// try-catch to collect all generator outputs for per-element error handling.
#[unsafe(no_mangle)]
pub extern "C" fn rt_collect_append_raw(elem_ptr: *const Value, ctx: *mut u8) {
    assert!(!elem_ptr.is_null() && !ctx.is_null());
    let arr = unsafe { &mut *(ctx as *mut Value) };
    let elem = unsafe { (*elem_ptr).clone() };

    if let Value::Arr(rc_vec) = arr {
        Rc::make_mut(rc_vec).push(elem);
    } else {
        panic!("rt_collect_append_raw: expected Value::Arr, got {:?}", arr);
    }
}

/// `keys` builtin.
///
/// Obj→sorted key array, Arr→[0,1,...,n-1].
#[unsafe(no_mangle)]
pub extern "C" fn rt_keys(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Obj(o) => {
            // BTreeMap keys are already sorted
            let keys: Vec<Value> = o.keys().map(|k| Value::Str(Rc::new(k.clone()))).collect();
            Value::Arr(Rc::new(keys))
        }
        Value::Arr(a) => {
            let indices: Vec<Value> = (0..a.len()).map(|i| Value::Num(i as f64)).collect();
            Value::Arr(Rc::new(indices))
        }
        _ => Value::Error(Rc::new(format!("{} has no keys", type_name(v)))),
    };
    unsafe { std::ptr::write(out, result); }
}

// =========================================================================
// Phase 5-2: Additional unary builtins (nargs=1)
// =========================================================================

/// `sort` builtin: sort an array.
#[unsafe(no_mangle)]
pub extern "C" fn rt_sort(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Arr(arr) => {
            let mut sorted = (**arr).clone();
            sorted.sort_by(value_compare);
            Value::Arr(Rc::new(sorted))
        }
        _ => Value::Error(Rc::new(format!(
            "{} ({}) cannot be sorted, as it is not an array",
            type_name(v), type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `keys_unsorted` builtin: object keys in insertion order, array indices.
#[unsafe(no_mangle)]
pub extern "C" fn rt_keys_unsorted(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Obj(o) => {
            // BTreeMap iterates in sorted order by key, which matches jq's keys_unsorted
            // for our implementation (BTreeMap = sorted order, not insertion order).
            // Note: jq's keys_unsorted returns keys in the order they were inserted,
            // but since we use BTreeMap (sorted), our "unsorted" is actually sorted.
            let keys: Vec<Value> = o.keys().map(|k| Value::Str(Rc::new(k.clone()))).collect();
            Value::Arr(Rc::new(keys))
        }
        Value::Arr(a) => {
            let indices: Vec<Value> = (0..a.len()).map(|i| Value::Num(i as f64)).collect();
            Value::Arr(Rc::new(indices))
        }
        _ => Value::Error(Rc::new(format!(
            "{} has no keys", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `floor` builtin: floor of a number.
#[unsafe(no_mangle)]
pub extern "C" fn rt_floor(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Num(n) => Value::Num(n.floor()),
        _ => Value::Error(Rc::new(format!(
            "{} number required", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `ceil` builtin: ceiling of a number.
#[unsafe(no_mangle)]
pub extern "C" fn rt_ceil(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Num(n) => Value::Num(n.ceil()),
        _ => Value::Error(Rc::new(format!(
            "{} number required", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `round` builtin: round to nearest integer.
#[unsafe(no_mangle)]
pub extern "C" fn rt_round(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Num(n) => Value::Num(n.round()),
        _ => Value::Error(Rc::new(format!(
            "{} number required", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `fabs` builtin: absolute value.
#[unsafe(no_mangle)]
pub extern "C" fn rt_fabs(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Num(n) => Value::Num(n.abs()),
        // jq: abs on non-number types is identity
        other => other.clone(),
    };
    unsafe { std::ptr::write(out, result); }
}

// ===== Math unary functions =====

/// Emulate C logb(): returns the exponent of x as a float.
/// logb(0) = -inf, logb(inf) = inf, logb(nan) = nan.
fn c_logb(x: f64) -> f64 {
    if x == 0.0 {
        f64::NEG_INFINITY
    } else if x.is_infinite() {
        f64::INFINITY
    } else if x.is_nan() {
        f64::NAN
    } else {
        libm::ilogb(x) as f64
    }
}

/// Helper macro for math unary functions: extract f64, apply function, return Value::Num.
macro_rules! math_unary_fn {
    ($fn_name:ident, $op:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $fn_name(out: *mut Value, v: *const Value) {
            assert!(!out.is_null() && !v.is_null());
            let v = unsafe { &*v };
            propagate_error_unary!(out, v);
            let result = match v {
                Value::Num(n) => Value::Num($op(*n)),
                _ => Value::Error(Rc::new(format!(
                    "{} number required", type_name(v)
                ))),
            };
            unsafe { std::ptr::write(out, result); }
        }
    };
}

math_unary_fn!(rt_sqrt, f64::sqrt);
math_unary_fn!(rt_sin, f64::sin);
math_unary_fn!(rt_cos, f64::cos);
math_unary_fn!(rt_tan, f64::tan);
math_unary_fn!(rt_asin, f64::asin);
math_unary_fn!(rt_acos, f64::acos);
math_unary_fn!(rt_atan, f64::atan);
math_unary_fn!(rt_exp, f64::exp);
math_unary_fn!(rt_exp2, f64::exp2);
math_unary_fn!(rt_log, f64::ln);
math_unary_fn!(rt_log2, f64::log2);
math_unary_fn!(rt_log10, f64::log10);
math_unary_fn!(rt_cbrt, f64::cbrt);
math_unary_fn!(rt_trunc, f64::trunc);

/// `exp10` builtin: 10^x. Not in Rust std, computed as 10_f64.powf(n).
#[unsafe(no_mangle)]
pub extern "C" fn rt_exp10(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Num(n) => Value::Num(10_f64.powf(*n)),
        _ => Value::Error(Rc::new(format!(
            "{} number required", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `significand` builtin: returns the significand (mantissa) of a float.
/// In jq, significand is defined as: x * pow(2; -(logb(x)))
#[unsafe(no_mangle)]
pub extern "C" fn rt_significand(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Num(n) => {
            // jq uses C significand() which returns x * 2^(-ilogb(x))
            // For normalized doubles, significand is in [1, 2)
            let logb_val = c_logb(*n);
            if logb_val.is_finite() {
                Value::Num(*n * 2_f64.powf(-logb_val))
            } else {
                Value::Num(*n) // inf, nan, zero
            }
        }
        _ => Value::Error(Rc::new(format!(
            "{} number required", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `exponent` builtin: returns the exponent (ilogb) of a float.
/// In jq, exponent is the integer exponent, equivalent to C ilogb().
#[unsafe(no_mangle)]
pub extern "C" fn rt_exponent(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Num(n) => Value::Num(c_logb(*n).floor()),
        _ => Value::Error(Rc::new(format!(
            "{} number required", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `logb` builtin: returns the exponent of a float as a float.
#[unsafe(no_mangle)]
pub extern "C" fn rt_logb(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Num(n) => Value::Num(c_logb(*n)),
        _ => Value::Error(Rc::new(format!(
            "{} number required", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `nearbyint` builtin: round to nearest integer using current rounding mode.
#[unsafe(no_mangle)]
pub extern "C" fn rt_nearbyint(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        // nearbyint rounds to nearest integer; use rint as equivalent
        Value::Num(n) => Value::Num(libm::rint(*n)),
        _ => Value::Error(Rc::new(format!(
            "{} number required", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `rint` builtin: round to nearest integer.
#[unsafe(no_mangle)]
pub extern "C" fn rt_rint(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Num(n) => Value::Num(libm::rint(*n)),
        _ => Value::Error(Rc::new(format!(
            "{} number required", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `j0` builtin: Bessel function of the first kind, order 0.
#[unsafe(no_mangle)]
pub extern "C" fn rt_j0(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Num(n) => Value::Num(libm::j0(*n)),
        _ => Value::Error(Rc::new(format!(
            "{} number required", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `j1` builtin: Bessel function of the first kind, order 1.
#[unsafe(no_mangle)]
pub extern "C" fn rt_j1(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Num(n) => Value::Num(libm::j1(*n)),
        _ => Value::Error(Rc::new(format!(
            "{} number required", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

// ===== Math binary functions =====

/// `pow(base; exp)` builtin: raise base to the power of exp.
#[unsafe(no_mangle)]
pub extern "C" fn rt_pow(out: *mut Value, base: *const Value, exp: *const Value) {
    assert!(!out.is_null() && !base.is_null() && !exp.is_null());
    let base = unsafe { &*base };
    let exp = unsafe { &*exp };
    if let Value::Error(_) = base { unsafe { std::ptr::write(out, base.clone()); } return; }
    if let Value::Error(_) = exp { unsafe { std::ptr::write(out, exp.clone()); } return; }
    let result = match (base, exp) {
        (Value::Num(b), Value::Num(e)) => Value::Num(b.powf(*e)),
        _ => Value::Error(Rc::new(format!("pow requires number arguments"))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `atan(y; x)` builtin: two-argument arctangent (atan2).
#[unsafe(no_mangle)]
pub extern "C" fn rt_atan2(out: *mut Value, y: *const Value, x: *const Value) {
    assert!(!out.is_null() && !y.is_null() && !x.is_null());
    let y = unsafe { &*y };
    let x = unsafe { &*x };
    if let Value::Error(_) = y { unsafe { std::ptr::write(out, y.clone()); } return; }
    if let Value::Error(_) = x { unsafe { std::ptr::write(out, x.clone()); } return; }
    let result = match (y, x) {
        (Value::Num(a), Value::Num(b)) => Value::Num(a.atan2(*b)),
        _ => Value::Error(Rc::new(format!("atan requires number arguments"))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `drem(x; y)` builtin: IEEE remainder.
#[unsafe(no_mangle)]
pub extern "C" fn rt_drem(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    if let Value::Error(_) = a { unsafe { std::ptr::write(out, a.clone()); } return; }
    if let Value::Error(_) = b { unsafe { std::ptr::write(out, b.clone()); } return; }
    let result = match (a, b) {
        (Value::Num(x), Value::Num(y)) => Value::Num(libm::remainder(*x, *y)),
        _ => Value::Error(Rc::new(format!("drem requires number arguments"))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `ldexp(x; n)` builtin: x * 2^n.
#[unsafe(no_mangle)]
pub extern "C" fn rt_ldexp(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    if let Value::Error(_) = a { unsafe { std::ptr::write(out, a.clone()); } return; }
    if let Value::Error(_) = b { unsafe { std::ptr::write(out, b.clone()); } return; }
    let result = match (a, b) {
        (Value::Num(x), Value::Num(n)) => Value::Num(libm::ldexp(*x, *n as i32)),
        _ => Value::Error(Rc::new(format!("ldexp requires number arguments"))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `scalb(x; y)` builtin: x * 2^y (using scalbn).
#[unsafe(no_mangle)]
pub extern "C" fn rt_scalb(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    if let Value::Error(_) = a { unsafe { std::ptr::write(out, a.clone()); } return; }
    if let Value::Error(_) = b { unsafe { std::ptr::write(out, b.clone()); } return; }
    let result = match (a, b) {
        (Value::Num(x), Value::Num(y)) => Value::Num(libm::scalbn(*x, *y as i32)),
        _ => Value::Error(Rc::new(format!("scalb requires number arguments"))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `scalbln(x; y)` builtin: x * 2^y (long version).
#[unsafe(no_mangle)]
pub extern "C" fn rt_scalbln(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    if let Value::Error(_) = a { unsafe { std::ptr::write(out, a.clone()); } return; }
    if let Value::Error(_) = b { unsafe { std::ptr::write(out, b.clone()); } return; }
    let result = match (a, b) {
        (Value::Num(x), Value::Num(y)) => Value::Num(libm::scalbn(*x, *y as i32)),
        _ => Value::Error(Rc::new(format!("scalbln requires number arguments"))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `bsearch(x)` builtin: binary search in sorted array.
/// Returns index if found, or -(insertion_point + 1) if not found.
#[unsafe(no_mangle)]
pub extern "C" fn rt_bsearch(out: *mut Value, input: *const Value, target: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !target.is_null());
    let input = unsafe { &*input };
    let target = unsafe { &*target };
    if let Value::Error(_) = input { unsafe { std::ptr::write(out, input.clone()); } return; }
    if let Value::Error(_) = target { unsafe { std::ptr::write(out, target.clone()); } return; }
    let result = match input {
        Value::Arr(arr) => {
            // Binary search using jq's value ordering
            let mut lo: i64 = 0;
            let mut hi: i64 = arr.len() as i64 - 1;
            while lo <= hi {
                let mid = lo + (hi - lo) / 2;
                let cmp = value_compare(&arr[mid as usize], target);
                if cmp == std::cmp::Ordering::Equal {
                    unsafe { std::ptr::write(out, Value::Num(mid as f64)); }
                    return;
                } else if cmp == std::cmp::Ordering::Less {
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            // Not found: return -(insertion_point + 1)
            Value::Num(-(lo + 1) as f64)
        }
        _ => Value::Error(Rc::new(format!(
            "{} ({}) cannot be searched from", type_name(input), crate::value::value_to_json(input)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `explode` builtin: string → array of codepoints.
#[unsafe(no_mangle)]
pub extern "C" fn rt_explode(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Str(s) => {
            let codepoints: Vec<Value> = s.chars()
                .map(|c| Value::Num(c as u32 as f64))
                .collect();
            Value::Arr(Rc::new(codepoints))
        }
        _ => Value::Error(Rc::new(format!(
            "explode input must be a string"
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `implode` builtin: array of codepoints → string.
#[unsafe(no_mangle)]
pub extern "C" fn rt_implode(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Arr(arr) => {
            let mut s = String::with_capacity(arr.len());
            for elem in arr.iter() {
                match elem {
                    Value::Num(n) => {
                        if n.is_nan() || n.is_infinite() {
                            let desc = if n.is_nan() { "null" } else { "number" };
                            return unsafe {
                                std::ptr::write(out, Value::Error(Rc::new(format!(
                                    "number ({}) can't be imploded, unicode codepoint needs to be numeric",
                                    desc
                                ))));
                            };
                        }
                        // Truncate float to integer
                        let code = *n as i64;
                        if code < 0 || code > 0x10FFFF {
                            // Out of range: use replacement character U+FFFD
                            s.push('\u{FFFD}');
                        } else {
                            let u = code as u32;
                            // Surrogate range (0xD800-0xDFFF) and some invalid ranges
                            match char::from_u32(u) {
                                Some(c) => s.push(c),
                                None => s.push('\u{FFFD}'),
                            }
                        }
                    }
                    other => {
                        let desc = type_desc(other);
                        return unsafe {
                            std::ptr::write(out, Value::Error(Rc::new(format!(
                                "{} can't be imploded, unicode codepoint needs to be numeric",
                                desc
                            ))));
                        };
                    }
                }
            }
            Value::Str(Rc::new(s))
        }
        _ => Value::Error(Rc::new(format!(
            "implode input must be an array"
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

// =========================================================================
// Phase 5-2: Binary builtins (nargs=2, 3-ptr signature: out, input, arg)
// =========================================================================

/// `split(s)` builtin: split string by separator.
#[unsafe(no_mangle)]
pub extern "C" fn rt_split(out: *mut Value, input: *const Value, sep: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !sep.is_null());
    let input = unsafe { &*input };
    let sep = unsafe { &*sep };
    propagate_error_bin!(out, input, sep);
    let result = match (input, sep) {
        (Value::Str(s), Value::Str(d)) => {
            if d.is_empty() {
                // jq: split("") returns individual characters (no empty strings at boundaries)
                let parts: Vec<Value> = s.chars()
                    .map(|c| Value::Str(Rc::new(c.to_string())))
                    .collect();
                Value::Arr(Rc::new(parts))
            } else {
                let parts: Vec<Value> = s.split(d.as_str())
                    .map(|p| Value::Str(Rc::new(p.to_string())))
                    .collect();
                Value::Arr(Rc::new(parts))
            }
        }
        _ => Value::Error(Rc::new(format!(
            "split input and separator must be strings"
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `has(key)` builtin: check if object has key or array has index.
#[unsafe(no_mangle)]
pub extern "C" fn rt_has(out: *mut Value, input: *const Value, key: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !key.is_null());
    let input = unsafe { &*input };
    let key = unsafe { &*key };
    propagate_error_bin!(out, input, key);
    let result = match (input, key) {
        (Value::Obj(o), Value::Str(k)) => Value::Bool(o.contains_key(k.as_str())),
        (Value::Arr(a), Value::Num(n)) => {
            if n.is_nan() || n.is_infinite() || *n < 0.0 || *n != n.trunc() {
                Value::Bool(false)
            } else {
                let idx = *n as usize;
                Value::Bool(idx < a.len())
            }
        }
        _ => Value::Error(Rc::new(format!(
            "has() is not defined for {} and {}",
            type_name(input), type_name(key)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `startswith(s)` builtin: check if string starts with prefix.
#[unsafe(no_mangle)]
pub extern "C" fn rt_startswith(out: *mut Value, input: *const Value, prefix: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !prefix.is_null());
    let input = unsafe { &*input };
    let prefix = unsafe { &*prefix };
    propagate_error_bin!(out, input, prefix);
    let result = match (input, prefix) {
        (Value::Str(s), Value::Str(p)) => Value::Bool(s.starts_with(p.as_str())),
        _ => Value::Error(Rc::new(format!(
            "startswith requires string arguments"
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `endswith(s)` builtin: check if string ends with suffix.
#[unsafe(no_mangle)]
pub extern "C" fn rt_endswith(out: *mut Value, input: *const Value, suffix: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !suffix.is_null());
    let input = unsafe { &*input };
    let suffix = unsafe { &*suffix };
    propagate_error_bin!(out, input, suffix);
    let result = match (input, suffix) {
        (Value::Str(s), Value::Str(p)) => Value::Bool(s.ends_with(p.as_str())),
        _ => Value::Error(Rc::new(format!(
            "endswith requires string arguments"
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `join(s)` builtin: join array elements with separator.
#[unsafe(no_mangle)]
pub extern "C" fn rt_join(out: *mut Value, input: *const Value, sep: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !sep.is_null());
    let input = unsafe { &*input };
    let sep = unsafe { &*sep };
    propagate_error_bin!(out, input, sep);
    let result = match (input, sep) {
        (Value::Arr(arr), Value::Str(d)) => {
            let mut parts: Vec<String> = Vec::with_capacity(arr.len());
            for v in arr.iter() {
                match v {
                    Value::Str(s) => parts.push(s.to_string()),
                    Value::Null => parts.push(String::new()),
                    Value::Num(n) => {
                        // jq coerces numbers to strings in join
                        let s = if *n == n.trunc() && n.abs() < 1e15 && !n.is_nan() && !n.is_infinite() && *n as i64 as f64 == *n {
                            format!("{}", *n as i64)
                        } else {
                            format!("{}", n)
                        };
                        parts.push(s);
                    }
                    Value::Bool(b) => {
                        parts.push(if *b { "true" } else { "false" }.to_string());
                    }
                    other => {
                        // Build the accumulated string so far (with trailing separator)
                        let accum = if parts.is_empty() {
                            String::new()
                        } else {
                            format!("{}{}", parts.join(d.as_str()), d.as_str())
                        };
                        let accum_val = Value::Str(Rc::new(accum));
                        unsafe { std::ptr::write(out, Value::Error(Rc::new(format!(
                            "{} and {} cannot be added",
                            type_desc(&accum_val), type_desc(other)
                        )))); }
                        return;
                    }
                }
            }
            Value::Str(Rc::new(parts.join(d.as_str())))
        }
        _ => Value::Error(Rc::new(format!(
            "join requires array input and string separator"
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

// =========================================================================
// Phase 5-3: jq-defined functions implemented as direct runtime calls
// =========================================================================

/// `add` builtin (jq-defined): reduce .[] as $x (null; . + $x).
///
/// Sums all elements of an array. Supports Num (addition), Str (concat),
/// Arr (concat), and Null (identity for +).
#[unsafe(no_mangle)]
pub extern "C" fn rt_jq_add(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Arr(arr) => {
            if arr.is_empty() {
                Value::Null
            } else {
                let mut acc = Value::Null;
                for elem in arr.iter() {
                    acc = match (&acc, elem) {
                        (Value::Null, other) => other.clone(),
                        (Value::Num(x), Value::Num(y)) => Value::Num(x + y),
                        (Value::Str(x), Value::Str(y)) => {
                            let mut s = String::with_capacity(x.len() + y.len());
                            s.push_str(x);
                            s.push_str(y);
                            Value::Str(Rc::new(s))
                        }
                        (Value::Arr(x), Value::Arr(y)) => {
                            let mut v = Vec::with_capacity(x.len() + y.len());
                            v.extend(x.iter().cloned());
                            v.extend(y.iter().cloned());
                            Value::Arr(Rc::new(v))
                        }
                        (Value::Obj(x), Value::Obj(y)) => {
                            let mut m = (**x).clone();
                            for (k, v) in y.iter() {
                                m.insert(k.clone(), v.clone());
                            }
                            Value::Obj(Rc::new(m))
                        }
                        (other, Value::Null) => other.clone(),
                        _ => {
                            return unsafe {
                                std::ptr::write(out, Value::Error(Rc::new(format!(
                                    "{} and {} cannot be added",
                                    type_desc(&acc), type_desc(elem)
                                ))));
                            };
                        }
                    };
                }
                acc
            }
        }
        Value::Null => Value::Null,
        _ => Value::Error(Rc::new(format!(
            "{} is not iterable", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `values` builtin (jq-defined): [.[] | select(. != null)].
///
/// Returns an array of all values (for objects) or elements (for arrays),
/// filtering out null values.
/// Note: jq's `values` actually keeps all non-null values. For arrays, it's
/// equivalent to `[.[] | select(. != null)]`. For objects, `[.[]]`.
/// Actually, jq's `values` is defined as `[.[] | select(. != null)]` but
/// the actual jq behavior is simply `[.[]]` — values keeps ALL values including nulls.
/// Let me check... `echo '{"a":1,"b":null}' | jq 'values'` returns `[1, null]`.
/// So values is just `[.[]]` — all values without filtering.
#[unsafe(no_mangle)]
pub extern "C" fn rt_values(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Obj(o) => {
            let vals: Vec<Value> = o.values().cloned().collect();
            Value::Arr(Rc::new(vals))
        }
        Value::Arr(arr) => {
            // For arrays, values is identity (returns the array itself)
            Value::Arr(Rc::clone(arr))
        }
        _ => Value::Error(Rc::new(format!(
            "{} has no values", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `reverse` builtin (jq-defined): reverses an array.
#[unsafe(no_mangle)]
pub extern "C" fn rt_reverse(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Arr(arr) => {
            let mut reversed = (**arr).clone();
            reversed.reverse();
            Value::Arr(Rc::new(reversed))
        }
        _ => Value::Error(Rc::new(format!(
            "{} is not an array", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `to_entries` builtin (jq-defined): converts object to array of {key, value} pairs.
#[unsafe(no_mangle)]
pub extern "C" fn rt_to_entries(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Obj(o) => {
            let entries: Vec<Value> = o.iter().map(|(k, v)| {
                let mut entry = std::collections::BTreeMap::new();
                entry.insert("key".to_string(), Value::Str(Rc::new(k.clone())));
                entry.insert("value".to_string(), v.clone());
                Value::Obj(Rc::new(entry))
            }).collect();
            Value::Arr(Rc::new(entries))
        }
        _ => Value::Error(Rc::new(format!(
            "{} has no keys", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `from_entries` builtin (jq-defined): converts array of {key/name, value} pairs to object.
#[unsafe(no_mangle)]
pub extern "C" fn rt_from_entries(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Arr(arr) => {
            let mut obj = std::collections::BTreeMap::new();
            for entry in arr.iter() {
                if let Value::Obj(o) = entry {
                    // jq accepts "key", "Key", "name", "Name" for the key field
                    let key = o.get("key")
                        .or_else(|| o.get("Key"))
                        .or_else(|| o.get("name"))
                        .or_else(|| o.get("Name"))
                        .cloned()
                        .unwrap_or(Value::Null);
                    // jq accepts "value" and "Value" for the value field
                    let value = o.get("value")
                        .or_else(|| o.get("Value"))
                        .cloned()
                        .unwrap_or(Value::Null);
                    // Convert key to string
                    let key_str = match &key {
                        Value::Str(s) => s.to_string(),
                        Value::Num(n) => {
                            if *n == n.trunc() && n.abs() < 1e15 && *n as i64 as f64 == *n {
                                format!("{}", *n as i64)
                            } else {
                                format!("{}", n)
                            }
                        }
                        _ => continue,
                    };
                    obj.insert(key_str, value);
                }
            }
            Value::Obj(Rc::new(obj))
        }
        _ => Value::Error(Rc::new(format!(
            "{} is not an array", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `unique` builtin (jq-defined): sort and deduplicate array elements.
#[unsafe(no_mangle)]
pub extern "C" fn rt_unique(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Arr(arr) => {
            let mut sorted = (**arr).clone();
            sorted.sort_by(value_compare);
            sorted.dedup_by(|a, b| value_compare(a, b) == std::cmp::Ordering::Equal);
            Value::Arr(Rc::new(sorted))
        }
        _ => Value::Error(Rc::new(format!(
            "{} is not an array", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `ascii_downcase` builtin (jq-defined): converts ASCII uppercase to lowercase.
#[unsafe(no_mangle)]
pub extern "C" fn rt_ascii_downcase(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Str(s) => {
            // Only convert ASCII characters (A-Z → a-z), leave non-ASCII unchanged
            let lowered: String = s.chars().map(|c| {
                if c.is_ascii_uppercase() {
                    c.to_ascii_lowercase()
                } else {
                    c
                }
            }).collect();
            Value::Str(Rc::new(lowered))
        }
        _ => Value::Error(Rc::new(format!(
            "{} is not a string", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `ascii_upcase` builtin (jq-defined): converts ASCII lowercase to uppercase.
#[unsafe(no_mangle)]
pub extern "C" fn rt_ascii_upcase(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Str(s) => {
            // Only convert ASCII characters (a-z → A-Z), leave non-ASCII unchanged
            let uppered: String = s.chars().map(|c| {
                if c.is_ascii_lowercase() {
                    c.to_ascii_uppercase()
                } else {
                    c
                }
            }).collect();
            Value::Str(Rc::new(uppered))
        }
        _ => Value::Error(Rc::new(format!(
            "{} is not a string", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

// =========================================================================
// Phase 8-1: Missing BinOp runtime functions
// =========================================================================

/// `contains(b)` builtin: check if a contains b.
///
/// String: a contains b as substring.
/// Array: every element of b is contained in a (recursive containment).
/// Object: a contains all keys of b with matching values.
/// Other types: equality check.
#[unsafe(no_mangle)]
pub extern "C" fn rt_contains(out: *mut Value, a: *const Value, b: *const Value) {
    assert!(!out.is_null() && !a.is_null() && !b.is_null());
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    propagate_error_bin!(out, a, b);
    let result = Value::Bool(value_contains(a, b));
    unsafe { std::ptr::write(out, result); }
}

/// Recursive object merge (jq semantics for obj * obj).
/// When both values at a key are objects, merges recursively.
/// Otherwise, the right-hand value wins.
fn recursive_obj_merge(x: &BTreeMap<String, Value>, y: &BTreeMap<String, Value>) -> Value {
    let mut m = x.clone();
    for (k, yv) in y.iter() {
        match (m.get(k), yv) {
            (Some(Value::Obj(xo)), Value::Obj(yo)) => {
                m.insert(k.clone(), recursive_obj_merge(xo, yo));
            }
            _ => {
                m.insert(k.clone(), yv.clone());
            }
        }
    }
    Value::Obj(Rc::new(m))
}

/// Recursive containment check (jq semantics).
fn value_contains(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Str(sa), Value::Str(sb)) => sa.contains(sb.as_str()),
        (Value::Arr(aa), Value::Arr(ab)) => {
            // Every element of b must be contained in some element of a
            ab.iter().all(|belem| {
                aa.iter().any(|aelem| value_contains(aelem, belem))
            })
        }
        (Value::Obj(oa), Value::Obj(ob)) => {
            // Every key in b must exist in a with a contained value
            ob.iter().all(|(k, vb)| {
                oa.get(k).map_or(false, |va| value_contains(va, vb))
            })
        }
        // For scalar types: equality
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Null, Value::Null) => true,
        _ => false,
    }
}

/// `ltrimstr(prefix)` builtin: remove prefix from string.
///
/// If the string starts with `prefix`, returns the string without the prefix.
/// Otherwise returns the string unchanged.
#[unsafe(no_mangle)]
pub extern "C" fn rt_ltrimstr(out: *mut Value, input: *const Value, prefix: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !prefix.is_null());
    let input = unsafe { &*input };
    let prefix = unsafe { &*prefix };
    propagate_error_bin!(out, input, prefix);
    let result = match (input, prefix) {
        (Value::Str(s), Value::Str(p)) => {
            if s.starts_with(p.as_str()) {
                Value::Str(Rc::new(s[p.len()..].to_string()))
            } else {
                Value::Str(Rc::clone(s))
            }
        }
        // jq raises an error when argument types don't match
        (Value::Str(_), other) => {
            Value::Error(Rc::new(format!(
                "{} and {} cannot have their strings trimmed",
                type_desc(input), type_desc(other)
            )))
        }
        (_, Value::Str(_)) => {
            Value::Error(Rc::new(format!(
                "{} and {} cannot have their strings trimmed",
                type_desc(input), type_desc(prefix)
            )))
        }
        _ => input.clone(),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `rtrimstr(suffix)` builtin: remove suffix from string.
///
/// If the string ends with `suffix`, returns the string without the suffix.
/// Otherwise returns the string unchanged.
#[unsafe(no_mangle)]
pub extern "C" fn rt_rtrimstr(out: *mut Value, input: *const Value, suffix: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !suffix.is_null());
    let input = unsafe { &*input };
    let suffix = unsafe { &*suffix };
    propagate_error_bin!(out, input, suffix);
    let result = match (input, suffix) {
        (Value::Str(s), Value::Str(p)) => {
            if p.is_empty() {
                // rtrimstr("") is identity — jq returns the string unchanged
                Value::Str(Rc::clone(s))
            } else if s.ends_with(p.as_str()) {
                Value::Str(Rc::new(s[..s.len() - p.len()].to_string()))
            } else {
                Value::Str(Rc::clone(s))
            }
        }
        // jq raises an error when argument types don't match
        (Value::Str(_), other) => {
            Value::Error(Rc::new(format!(
                "{} and {} cannot have their strings trimmed",
                type_desc(input), type_desc(other)
            )))
        }
        (_, Value::Str(_)) => {
            Value::Error(Rc::new(format!(
                "{} and {} cannot have their strings trimmed",
                type_desc(input), type_desc(suffix)
            )))
        }
        _ => input.clone(),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `in(obj)` builtin: check if key exists in object.
///
/// This is the inverse of `has`: `"foo" | in({"foo": 1})` → true.
/// Input is the key, argument is the object.
#[unsafe(no_mangle)]
pub extern "C" fn rt_in(out: *mut Value, key: *const Value, obj: *const Value) {
    assert!(!out.is_null() && !key.is_null() && !obj.is_null());
    let key = unsafe { &*key };
    let obj = unsafe { &*obj };
    propagate_error_bin!(out, key, obj);
    let result = match (key, obj) {
        (Value::Str(k), Value::Obj(o)) => Value::Bool(o.contains_key(k.as_str())),
        (Value::Num(n), Value::Arr(a)) => {
            let idx = *n as usize;
            Value::Bool(idx < a.len())
        }
        _ => Value::Error(Rc::new(format!(
            "Cannot check if {} is in {}",
            type_name(key), type_name(obj)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

// =========================================================================
// Phase 8-1: Missing UnaryOp runtime functions
// =========================================================================

/// `min` builtin: minimum element of an array.
///
/// Returns the minimum element according to jq value ordering.
/// Returns null for empty arrays.
#[unsafe(no_mangle)]
pub extern "C" fn rt_min(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Arr(arr) => {
            if arr.is_empty() {
                Value::Null
            } else {
                arr.iter().min_by(|a, b| value_compare(a, b)).unwrap().clone()
            }
        }
        _ => Value::Error(Rc::new(format!(
            "{} is not an array", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `max` builtin: maximum element of an array.
///
/// Returns the maximum element according to jq value ordering.
/// Returns null for empty arrays.
#[unsafe(no_mangle)]
pub extern "C" fn rt_max(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Arr(arr) => {
            if arr.is_empty() {
                Value::Null
            } else {
                arr.iter().max_by(|a, b| value_compare(a, b)).unwrap().clone()
            }
        }
        _ => Value::Error(Rc::new(format!(
            "{} is not an array", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `flatten` builtin: recursively flatten all levels of nesting.
///
/// jq's `flatten` (no argument) recursively removes all array nesting.
/// Non-array elements are kept as-is.
#[unsafe(no_mangle)]
pub extern "C" fn rt_flatten(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Arr(arr) => {
            let mut flat = Vec::new();
            flatten_recursive(arr, &mut flat);
            Value::Arr(Rc::new(flat))
        }
        _ => Value::Error(Rc::new(format!(
            "{} is not an array", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// Recursively flatten an array, appending non-array elements to `out`.
fn flatten_recursive(arr: &[Value], out: &mut Vec<Value>) {
    for elem in arr {
        match elem {
            Value::Arr(inner) => flatten_recursive(inner, out),
            other => out.push(other.clone()),
        }
    }
}

/// `flatten(depth)` builtin: flatten with depth limit.
///
/// jq's `flatten(n)` removes up to `n` levels of array nesting.
/// `flatten(0)` is identity. Negative depth is an error.
#[unsafe(no_mangle)]
pub extern "C" fn rt_flatten_depth(out: *mut Value, v: *const Value, depth: *const Value) {
    assert!(!out.is_null() && !v.is_null() && !depth.is_null());
    let v = unsafe { &*v };
    let depth = unsafe { &*depth };
    propagate_error_bin!(out, v, depth);
    let result = match (v, depth) {
        (Value::Arr(arr), Value::Num(d)) => {
            let d = *d as i64;
            if d < 0 {
                Value::Error(Rc::new("flatten depth must not be negative".to_string()))
            } else {
                let mut flat = Vec::new();
                flatten_depth_recursive(arr, d as usize, &mut flat);
                Value::Arr(Rc::new(flat))
            }
        }
        _ => Value::Error(Rc::new(format!(
            "{} is not an array", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// Recursively flatten an array up to `depth` levels.
fn flatten_depth_recursive(arr: &[Value], depth: usize, out: &mut Vec<Value>) {
    for elem in arr {
        match elem {
            Value::Arr(inner) if depth > 0 => flatten_depth_recursive(inner, depth - 1, out),
            other => out.push(other.clone()),
        }
    }
}

/// `rt_index_opt`: optional field/index access `.foo?`.
///
/// Same as rt_index but returns null instead of Error for non-existent keys
/// or incompatible types.
#[unsafe(no_mangle)]
pub extern "C" fn rt_index_opt(out: *mut Value, obj: *const Value, key: *const Value) {
    assert!(!out.is_null() && !obj.is_null() && !key.is_null());
    let obj = unsafe { &*obj };
    let key = unsafe { &*key };
    // For optional access, errors propagate but type mismatches return null
    if let Value::Error(_) = obj {
        unsafe { std::ptr::write(out, Value::Null); }
        return;
    }
    if let Value::Error(_) = key {
        unsafe { std::ptr::write(out, Value::Null); }
        return;
    }
    let result = match (obj, key) {
        (Value::Obj(map), Value::Str(k)) => {
            map.get(k.as_str()).cloned().unwrap_or(Value::Null)
        }
        (Value::Arr(arr), Value::Num(idx)) => {
            let i = *idx as i64;
            if i < 0 {
                let resolved = arr.len() as i64 + i;
                if resolved >= 0 {
                    arr.get(resolved as usize).cloned().unwrap_or(Value::Null)
                } else {
                    Value::Null
                }
            } else {
                arr.get(i as usize).cloned().unwrap_or(Value::Null)
            }
        }
        // Slice: key is an object with "start" and/or "end" fields
        (Value::Arr(arr), Value::Obj(slice_obj)) => {
            slice_array(arr, slice_obj)
        }
        (Value::Str(s), Value::Obj(slice_obj)) => {
            slice_string(s, slice_obj)
        }
        (Value::Str(s), Value::Num(idx)) => {
            let i = *idx as i64;
            let len = s.len() as i64;
            let resolved = if i < 0 { len + i } else { i };
            if resolved >= 0 && (resolved as usize) < s.len() {
                Value::Num(s.as_bytes()[resolved as usize] as f64)
            } else {
                Value::Null
            }
        }
        (Value::Null, _) => Value::Null,
        // Optional: return null instead of Error for type mismatch
        _ => Value::Null,
    };
    unsafe { std::ptr::write(out, result); }
}

/// Compare two Values for sorting (jq ordering).
fn value_compare(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    // jq sort order: null < false < true < numbers < strings < arrays < objects
    fn type_order(v: &Value) -> u8 {
        match v {
            Value::Null => 0,
            Value::Bool(false) => 1,
            Value::Bool(true) => 2,
            Value::Num(_) => 3,
            Value::Str(_) => 4,
            Value::Arr(_) => 5,
            Value::Obj(_) => 6,
            Value::Error(_) => 7,
        }
    }
    let ta = type_order(a);
    let tb = type_order(b);
    if ta != tb {
        return ta.cmp(&tb);
    }
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (Value::Str(x), Value::Str(y)) => x.cmp(y),
        (Value::Arr(x), Value::Arr(y)) => {
            // Lexicographic comparison of array elements
            for (xa, ya) in x.iter().zip(y.iter()) {
                let cmp = value_compare(xa, ya);
                if cmp != Ordering::Equal {
                    return cmp;
                }
            }
            x.len().cmp(&y.len())
        }
        (Value::Obj(x), Value::Obj(y)) => {
            // jq compares objects by comparing sorted key-value pairs.
            // First compare by number of keys, then lexicographic on (key, value) pairs.
            // Actually jq compares: keys sorted, then values by those sorted keys.
            let mut x_keys: Vec<&String> = x.keys().collect();
            let mut y_keys: Vec<&String> = y.keys().collect();
            x_keys.sort();
            y_keys.sort();
            // Compare keys first
            for (xk, yk) in x_keys.iter().zip(y_keys.iter()) {
                let cmp = xk.cmp(yk);
                if cmp != Ordering::Equal {
                    return cmp;
                }
            }
            match x_keys.len().cmp(&y_keys.len()) {
                Ordering::Equal => {
                    // Same keys, compare values
                    for k in &x_keys {
                        let xv = x.get(*k).unwrap();
                        let yv = y.get(*k).unwrap();
                        let cmp = value_compare(xv, yv);
                        if cmp != Ordering::Equal {
                            return cmp;
                        }
                    }
                    Ordering::Equal
                }
                other => other,
            }
        }
        _ => Ordering::Equal,
    }
}

/// Phase 8-4: Object insert — insert a key-value pair into an object.
///
/// Signature: fn(out: *mut Value, obj: *const Value, key: *const Value, val: *const Value)
///
/// If `obj` is an Obj(map), clones the map, inserts `key→val`, and writes the
/// new Obj to `out`.
/// If `obj` is not an Obj or `key` is not a Str, writes Value::Error to `out`.
///
/// # Safety
/// All pointers must be non-null and properly aligned.
#[unsafe(no_mangle)]
pub extern "C" fn rt_obj_insert(
    out: *mut Value,
    obj: *const Value,
    key: *const Value,
    val: *const Value,
) {
    assert!(!out.is_null() && !obj.is_null() && !key.is_null() && !val.is_null());
    let obj_val = unsafe { &*obj };
    let key_val = unsafe { &*key };
    let val_val = unsafe { &*val };

    // Error propagation
    if let Value::Error(_) = obj_val {
        unsafe { std::ptr::write(out, obj_val.clone()); }
        return;
    }
    if let Value::Error(_) = key_val {
        unsafe { std::ptr::write(out, key_val.clone()); }
        return;
    }
    if let Value::Error(_) = val_val {
        unsafe { std::ptr::write(out, val_val.clone()); }
        return;
    }

    let result = match (obj_val, key_val) {
        (Value::Obj(map), Value::Str(key_str)) => {
            let mut new_map = (**map).clone();
            new_map.insert((**key_str).clone(), val_val.clone());
            Value::Obj(Rc::new(new_map))
        }
        (Value::Obj(_), _) => {
            Value::Error(Rc::new(format!(
                "object key must be string, got {}",
                type_name(key_val)
            )))
        }
        _ => {
            Value::Error(Rc::new(format!(
                "cannot insert into {}, expected object",
                type_name(obj_val)
            )))
        }
    };
    unsafe { std::ptr::write(out, result); }
}

// =========================================================================
// Phase 9-3: Format string runtime functions (@base64, @html, @uri, etc.)
// =========================================================================

/// @base64 — Base64 encode a string.
#[unsafe(no_mangle)]
pub extern "C" fn rt_format_base64(out: *mut Value, v: *const Value) {
    let val = unsafe { &*v };
    propagate_error_unary!(out, val);
    let s = match val {
        Value::Str(s) => (**s).clone(),
        _ => fmt_value_to_string(val),
    };
    let encoded = base64_encode_string(&s);
    unsafe { std::ptr::write(out, Value::Str(Rc::new(encoded))); }
}

/// @base64d — Base64 decode a string.
#[unsafe(no_mangle)]
pub extern "C" fn rt_format_base64d(out: *mut Value, v: *const Value) {
    let val = unsafe { &*v };
    propagate_error_unary!(out, val);
    let s = match val {
        Value::Str(s) => (**s).clone(),
        _ => {
            unsafe { std::ptr::write(out, Value::Error(Rc::new("@base64d: input must be a string".to_string()))); }
            return;
        }
    };
    match base64_decode_string(&s) {
        Ok(decoded) => unsafe { std::ptr::write(out, Value::Str(Rc::new(decoded))); },
        Err(e) => unsafe { std::ptr::write(out, Value::Error(Rc::new(format!("@base64d: {}", e)))); },
    }
}

/// @html — HTML escape a string.
#[unsafe(no_mangle)]
pub extern "C" fn rt_format_html(out: *mut Value, v: *const Value) {
    let val = unsafe { &*v };
    propagate_error_unary!(out, val);
    let s = match val {
        Value::Str(s) => (**s).clone(),
        _ => fmt_value_to_string(val),
    };
    let escaped = s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('\'', "&apos;")
        .replace('"', "&quot;");
    unsafe { std::ptr::write(out, Value::Str(Rc::new(escaped))); }
}

/// @sh — Shell-quote a string (POSIX single-quote style).
/// Strings are wrapped in single quotes. Internal single quotes become '\\''
#[unsafe(no_mangle)]
pub extern "C" fn rt_format_sh(out: *mut Value, v: *const Value) {
    let val = unsafe { &*v };
    propagate_error_unary!(out, val);
    match val {
        Value::Str(s) => {
            let mut result = String::new();
            result.push('\'');
            for ch in s.chars() {
                if ch == '\'' {
                    result.push_str("'\\''");
                } else {
                    result.push(ch);
                }
            }
            result.push('\'');
            unsafe { std::ptr::write(out, Value::Str(Rc::new(result))); }
        }
        Value::Arr(arr) => {
            let parts: Vec<String> = arr.iter().map(|elem| {
                match elem {
                    Value::Str(s) => {
                        let mut result = String::new();
                        result.push('\'');
                        for ch in s.chars() {
                            if ch == '\'' {
                                result.push_str("'\\''");
                            } else {
                                result.push(ch);
                            }
                        }
                        result.push('\'');
                        result
                    }
                    _ => fmt_value_to_string(elem),
                }
            }).collect();
            unsafe { std::ptr::write(out, Value::Str(Rc::new(parts.join(" ")))); }
        }
        _ => {
            let s = fmt_value_to_string(val);
            let mut result = String::new();
            result.push('\'');
            for ch in s.chars() {
                if ch == '\'' {
                    result.push_str("'\\''");
                } else {
                    result.push(ch);
                }
            }
            result.push('\'');
            unsafe { std::ptr::write(out, Value::Str(Rc::new(result))); }
        }
    }
}

/// @uri — URI/percent-encode a string.
#[unsafe(no_mangle)]
pub extern "C" fn rt_format_uri(out: *mut Value, v: *const Value) {
    let val = unsafe { &*v };
    propagate_error_unary!(out, val);
    let s = match val {
        Value::Str(s) => (**s).clone(),
        _ => fmt_value_to_string(val),
    };
    let mut encoded = String::new();
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9'
            | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => {
                encoded.push_str(&format!("%{:02X}", byte));
            }
        }
    }
    unsafe { std::ptr::write(out, Value::Str(Rc::new(encoded))); }
}

/// @urid — URI/percent-decode a string.
#[unsafe(no_mangle)]
pub extern "C" fn rt_format_urid(out: *mut Value, v: *const Value) {
    let val = unsafe { &*v };
    propagate_error_unary!(out, val);
    let s = match val {
        Value::Str(s) => (**s).clone(),
        _ => {
            unsafe { std::ptr::write(out, Value::Error(Rc::new("@urid: input must be a string".to_string()))); }
            return;
        }
    };
    match percent_decode(&s) {
        Ok(decoded) => unsafe { std::ptr::write(out, Value::Str(Rc::new(decoded))); },
        Err(e) => unsafe { std::ptr::write(out, Value::Error(Rc::new(format!("@urid: {}", e)))); },
    }
}

/// Decode a percent-encoded string.
fn percent_decode(s: &str) -> Result<String, String> {
    let mut bytes: Vec<u8> = Vec::with_capacity(s.len());
    let src = s.as_bytes();
    let mut i = 0;
    while i < src.len() {
        if src[i] == b'%' {
            if i + 2 >= src.len() {
                return Err("incomplete percent-encoding".to_string());
            }
            let hi = hex_digit(src[i + 1]).ok_or_else(|| "invalid percent-encoding".to_string())?;
            let lo = hex_digit(src[i + 2]).ok_or_else(|| "invalid percent-encoding".to_string())?;
            bytes.push((hi << 4) | lo);
            i += 3;
        } else if src[i] == b'+' {
            bytes.push(b' ');
            i += 1;
        } else {
            bytes.push(src[i]);
            i += 1;
        }
    }
    String::from_utf8(bytes).map_err(|e| format!("invalid UTF-8 in decoded URI: {}", e))
}

fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// @csv — Format an array as a CSV row.
#[unsafe(no_mangle)]
pub extern "C" fn rt_format_csv(out: *mut Value, v: *const Value) {
    let val = unsafe { &*v };
    propagate_error_unary!(out, val);
    match val {
        Value::Arr(arr) => {
            let parts: Vec<String> = arr.iter().map(|elem| {
                match elem {
                    Value::Str(s) => {
                        let escaped = s.replace('"', "\"\"");
                        format!("\"{}\"", escaped)
                    }
                    Value::Num(n) => {
                        if *n == n.trunc() && n.abs() < 1e15 {
                            format!("{}", *n as i64)
                        } else {
                            format!("{}", n)
                        }
                    }
                    Value::Bool(b) => if *b { "true".to_string() } else { "false".to_string() },
                    Value::Null => "".to_string(),
                    _ => crate::value::value_to_json(elem),
                }
            }).collect();
            unsafe { std::ptr::write(out, Value::Str(Rc::new(parts.join(",")))); }
        }
        _ => {
            unsafe { std::ptr::write(out, Value::Error(Rc::new("@csv: input must be an array".to_string()))); }
        }
    }
}

/// @tsv — Format an array as a TSV row.
#[unsafe(no_mangle)]
pub extern "C" fn rt_format_tsv(out: *mut Value, v: *const Value) {
    let val = unsafe { &*v };
    propagate_error_unary!(out, val);
    match val {
        Value::Arr(arr) => {
            let parts: Vec<String> = arr.iter().map(|elem| {
                match elem {
                    Value::Str(s) => {
                        s.replace('\\', "\\\\")
                            .replace('\t', "\\t")
                            .replace('\n', "\\n")
                            .replace('\r', "\\r")
                    }
                    Value::Num(n) => {
                        if *n == n.trunc() && n.abs() < 1e15 {
                            format!("{}", *n as i64)
                        } else {
                            format!("{}", n)
                        }
                    }
                    Value::Bool(b) => if *b { "true".to_string() } else { "false".to_string() },
                    Value::Null => "".to_string(),
                    _ => crate::value::value_to_json(elem),
                }
            }).collect();
            unsafe { std::ptr::write(out, Value::Str(Rc::new(parts.join("\t")))); }
        }
        _ => {
            unsafe { std::ptr::write(out, Value::Error(Rc::new("@tsv: input must be an array".to_string()))); }
        }
    }
}

/// @json — Serialize a value to JSON string.
#[unsafe(no_mangle)]
pub extern "C" fn rt_format_json(out: *mut Value, v: *const Value) {
    let val = unsafe { &*v };
    propagate_error_unary!(out, val);
    let json = crate::value::value_to_json(val);
    unsafe { std::ptr::write(out, Value::Str(Rc::new(json))); }
}

// Base64 encoding/decoding helpers (no external crate dependency).

fn base64_encode_string(input: &str) -> String {
    const CHARSET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = input.as_bytes();
    let mut result = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARSET[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARSET[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARSET[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARSET[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

fn base64_decode_string(input: &str) -> Result<String, String> {
    let input = input.trim_end_matches('=');
    let mut bytes = Vec::new();
    let decode_char = |c: u8| -> Result<u32, String> {
        match c {
            b'A'..=b'Z' => Ok((c - b'A') as u32),
            b'a'..=b'z' => Ok((c - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((c - b'0' + 52) as u32),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("invalid base64 character: '{}'", c as char)),
        }
    };
    let chars: Vec<u8> = input.bytes().collect();
    let mut i = 0;
    while i < chars.len() {
        let b0 = decode_char(chars[i])?;
        let b1 = if i + 1 < chars.len() { decode_char(chars[i + 1])? } else { 0 };
        let b2 = if i + 2 < chars.len() { decode_char(chars[i + 2])? } else { 0 };
        let b3 = if i + 3 < chars.len() { decode_char(chars[i + 3])? } else { 0 };
        let triple = (b0 << 18) | (b1 << 12) | (b2 << 6) | b3;
        bytes.push(((triple >> 16) & 0xFF) as u8);
        if i + 2 < chars.len() {
            bytes.push(((triple >> 8) & 0xFF) as u8);
        }
        if i + 3 < chars.len() {
            bytes.push((triple & 0xFF) as u8);
        }
        i += 4;
    }
    String::from_utf8(bytes).map_err(|e| format!("invalid UTF-8 in decoded base64: {}", e))
}

/// Helper: convert a Value to a string for format functions (non-strings get tostring'd).
fn fmt_value_to_string(v: &Value) -> String {
    match v {
        Value::Str(s) => (**s).clone(),
        _ => crate::value::value_to_json(v),
    }
}

// ---------------------------------------------------------------------------
// Phase 9-2: Range generator
// ---------------------------------------------------------------------------

/// Range generator: yields numeric values from `from` up to (but not including) `to`.
///
/// Signature: `fn(from: *const Value, to: *const Value, callback: fn(*const Value, *mut u8), ctx: *mut u8)`
///
/// Extracts f64 from both from and to, then loops from..to yielding each integer.
#[unsafe(no_mangle)]
pub extern "C" fn rt_range(
    from_ptr: *const Value,
    to_ptr: *const Value,
    callback: extern "C" fn(*const Value, *mut u8),
    ctx: *mut u8,
) {
    assert!(!from_ptr.is_null() && !to_ptr.is_null());
    let from_val = unsafe { &*from_ptr };
    let to_val = unsafe { &*to_ptr };

    let from_f = match from_val {
        Value::Num(n) => *n,
        _ => return, // non-numeric: produce no output (like jq)
    };
    let to_f = match to_val {
        Value::Num(n) => *n,
        _ => return,
    };

    let mut i = from_f;
    while i < to_f {
        let val = Value::Num(i);
        callback(&val as *const Value, ctx);
        i += 1.0;
    }
}

/// Range generator with step: `range(from; to; step)`.
///
/// Generates numeric values from `from`, incrementing by `step`.
/// For positive step: yields while counter < to.
/// For negative step: yields while counter > to.
/// Step of 0 produces no output (avoids infinite loop).
#[unsafe(no_mangle)]
pub extern "C" fn rt_range_step(
    from_ptr: *const Value,
    to_ptr: *const Value,
    step_ptr: *const Value,
    callback: extern "C" fn(*const Value, *mut u8),
    ctx: *mut u8,
) {
    assert!(!from_ptr.is_null() && !to_ptr.is_null() && !step_ptr.is_null());
    let from_val = unsafe { &*from_ptr };
    let to_val = unsafe { &*to_ptr };
    let step_val = unsafe { &*step_ptr };

    let from_f = match from_val {
        Value::Num(n) => *n,
        _ => return,
    };
    let to_f = match to_val {
        Value::Num(n) => *n,
        _ => return,
    };
    let step_f = match step_val {
        Value::Num(n) => *n,
        _ => return,
    };

    if step_f == 0.0 {
        return; // step of 0: no output (avoid infinite loop)
    }

    let mut i = from_f;
    if step_f > 0.0 {
        while i < to_f {
            let val = Value::Num(i);
            callback(&val as *const Value, ctx);
            i += step_f;
        }
    } else {
        while i > to_f {
            let val = Value::Num(i);
            callback(&val as *const Value, ctx);
            i += step_f;
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 9-1: Closure-based array operations (sort_by, group_by, etc.)
// ---------------------------------------------------------------------------

/// `sort_by` builtin: sort an array by keys computed from each element.
///
/// Signature: fn(out: *mut Value, input: *const Value, keys: *const Value)
///
/// `input` is the array to sort.
/// `keys` is an array of key values (same length as input), computed by
/// applying the closure to each element.
/// The result is the input array sorted by the corresponding key values.
#[unsafe(no_mangle)]
pub extern "C" fn rt_sort_by_keys(
    out: *mut Value,
    input: *const Value,
    keys: *const Value,
) {
    assert!(!out.is_null() && !input.is_null() && !keys.is_null());
    let input_val = unsafe { &*input };
    let keys_val = unsafe { &*keys };
    propagate_error_bin!(out, input_val, keys_val);

    let result = match (input_val, keys_val) {
        (Value::Arr(arr), Value::Arr(key_arr)) => {
            if arr.len() != key_arr.len() {
                Value::Error(Rc::new("sort_by: keys array length mismatch".to_string()))
            } else {
                let mut indices: Vec<usize> = (0..arr.len()).collect();
                indices.sort_by(|&a, &b| value_compare(&key_arr[a], &key_arr[b]));
                let sorted: Vec<Value> = indices.iter().map(|&i| arr[i].clone()).collect();
                Value::Arr(Rc::new(sorted))
            }
        }
        _ => Value::Error(Rc::new(format!(
            "sort_by input must be an array, got {}",
            type_name(input_val)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `group_by` builtin: group elements by keys.
///
/// Signature: fn(out: *mut Value, input: *const Value, keys: *const Value)
///
/// Groups consecutive elements with equal keys (after sorting by key).
/// Result is an array of arrays (groups).
#[unsafe(no_mangle)]
pub extern "C" fn rt_group_by_keys(
    out: *mut Value,
    input: *const Value,
    keys: *const Value,
) {
    assert!(!out.is_null() && !input.is_null() && !keys.is_null());
    let input_val = unsafe { &*input };
    let keys_val = unsafe { &*keys };
    propagate_error_bin!(out, input_val, keys_val);

    let result = match (input_val, keys_val) {
        (Value::Arr(arr), Value::Arr(key_arr)) => {
            if arr.len() != key_arr.len() {
                Value::Error(Rc::new("group_by: keys array length mismatch".to_string()))
            } else if arr.is_empty() {
                Value::Arr(Rc::new(Vec::new()))
            } else {
                // Sort indices by key
                let mut indices: Vec<usize> = (0..arr.len()).collect();
                indices.sort_by(|&a, &b| value_compare(&key_arr[a], &key_arr[b]));

                // Group consecutive elements with equal keys
                let mut groups: Vec<Value> = Vec::new();
                let mut current_group: Vec<Value> = vec![arr[indices[0]].clone()];
                for i in 1..indices.len() {
                    if value_compare(&key_arr[indices[i]], &key_arr[indices[i - 1]]) == std::cmp::Ordering::Equal {
                        current_group.push(arr[indices[i]].clone());
                    } else {
                        groups.push(Value::Arr(Rc::new(current_group)));
                        current_group = vec![arr[indices[i]].clone()];
                    }
                }
                groups.push(Value::Arr(Rc::new(current_group)));
                Value::Arr(Rc::new(groups))
            }
        }
        _ => Value::Error(Rc::new(format!(
            "group_by input must be an array, got {}",
            type_name(input_val)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `unique_by` builtin: remove duplicates by key.
///
/// Signature: fn(out: *mut Value, input: *const Value, keys: *const Value)
///
/// Returns elements with unique keys (first occurrence), sorted by key.
#[unsafe(no_mangle)]
pub extern "C" fn rt_unique_by_keys(
    out: *mut Value,
    input: *const Value,
    keys: *const Value,
) {
    assert!(!out.is_null() && !input.is_null() && !keys.is_null());
    let input_val = unsafe { &*input };
    let keys_val = unsafe { &*keys };
    propagate_error_bin!(out, input_val, keys_val);

    let result = match (input_val, keys_val) {
        (Value::Arr(arr), Value::Arr(key_arr)) => {
            if arr.len() != key_arr.len() {
                Value::Error(Rc::new("unique_by: keys array length mismatch".to_string()))
            } else if arr.is_empty() {
                Value::Arr(Rc::new(Vec::new()))
            } else {
                // Sort indices by key
                let mut indices: Vec<usize> = (0..arr.len()).collect();
                indices.sort_by(|&a, &b| value_compare(&key_arr[a], &key_arr[b]));

                // Keep first occurrence of each unique key
                let mut unique_elems: Vec<Value> = vec![arr[indices[0]].clone()];
                for i in 1..indices.len() {
                    if value_compare(&key_arr[indices[i]], &key_arr[indices[i - 1]]) != std::cmp::Ordering::Equal {
                        unique_elems.push(arr[indices[i]].clone());
                    }
                }
                Value::Arr(Rc::new(unique_elems))
            }
        }
        _ => Value::Error(Rc::new(format!(
            "unique_by input must be an array, got {}",
            type_name(input_val)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `min_by` builtin: find element with minimum key.
///
/// Signature: fn(out: *mut Value, input: *const Value, keys: *const Value)
///
/// Returns the element whose key is the minimum.
#[unsafe(no_mangle)]
pub extern "C" fn rt_min_by_keys(
    out: *mut Value,
    input: *const Value,
    keys: *const Value,
) {
    assert!(!out.is_null() && !input.is_null() && !keys.is_null());
    let input_val = unsafe { &*input };
    let keys_val = unsafe { &*keys };
    propagate_error_bin!(out, input_val, keys_val);

    let result = match (input_val, keys_val) {
        (Value::Arr(arr), Value::Arr(key_arr)) => {
            if arr.len() != key_arr.len() {
                Value::Error(Rc::new("min_by: keys array length mismatch".to_string()))
            } else if arr.is_empty() {
                Value::Null
            } else {
                let mut min_idx = 0;
                for i in 1..arr.len() {
                    if value_compare(&key_arr[i], &key_arr[min_idx]) == std::cmp::Ordering::Less {
                        min_idx = i;
                    }
                }
                arr[min_idx].clone()
            }
        }
        _ => Value::Error(Rc::new(format!(
            "min_by input must be an array, got {}",
            type_name(input_val)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `max_by` builtin: find element with maximum key.
///
/// Signature: fn(out: *mut Value, input: *const Value, keys: *const Value)
///
/// Returns the element whose key is the maximum.
#[unsafe(no_mangle)]
pub extern "C" fn rt_max_by_keys(
    out: *mut Value,
    input: *const Value,
    keys: *const Value,
) {
    assert!(!out.is_null() && !input.is_null() && !keys.is_null());
    let input_val = unsafe { &*input };
    let keys_val = unsafe { &*keys };
    propagate_error_bin!(out, input_val, keys_val);

    let result = match (input_val, keys_val) {
        (Value::Arr(arr), Value::Arr(key_arr)) => {
            if arr.len() != key_arr.len() {
                Value::Error(Rc::new("max_by: keys array length mismatch".to_string()))
            } else if arr.is_empty() {
                Value::Null
            } else {
                let mut max_idx = 0;
                for i in 1..arr.len() {
                    // Use >= so that equal keys pick the last element
                    if value_compare(&key_arr[i], &key_arr[max_idx]) != std::cmp::Ordering::Less {
                        max_idx = i;
                    }
                }
                arr[max_idx].clone()
            }
        }
        _ => Value::Error(Rc::new(format!(
            "max_by input must be an array, got {}",
            type_name(input_val)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

// ---------------------------------------------------------------------------
// Phase 9-4: Recursive descent (..)
// ---------------------------------------------------------------------------

/// Recursive descent generator: yields the input value itself, then recurses
/// into all children (array elements or object values) depth-first.
///
/// Signature: `fn(input: *const Value, callback: fn(*const Value, *mut u8), ctx: *mut u8)`
///
/// This is called by JIT-generated code for the `..` / `recurse` filter.
/// The callback has the same signature as the JIT's internal generator callback:
/// `fn(value_ptr: *const Value, ctx: *mut c_void)`.
#[unsafe(no_mangle)]
pub extern "C" fn rt_recurse(
    input: *const Value,
    callback: extern "C" fn(*const Value, *mut u8),
    ctx: *mut u8,
) {
    assert!(!input.is_null());
    let v = unsafe { &*input };
    // Yield the value itself first
    callback(input, ctx);
    // Recurse into children
    match v {
        Value::Arr(arr) => {
            for elem in arr.iter() {
                let elem_ptr = elem as *const Value;
                rt_recurse(elem_ptr, callback, ctx);
            }
        }
        Value::Obj(obj) => {
            for (_key, val) in obj.iter() {
                let val_ptr = val as *const Value;
                rt_recurse(val_ptr, callback, ctx);
            }
        }
        // Scalars (Null, Bool, Num, Str, Error): no children to recurse into
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Phase 9-6: Remaining builtin functions
// ---------------------------------------------------------------------------

/// `any` — returns true if any element in the array is truthy.
/// Unary signature: fn(out: *mut Value, v: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_any(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let val = unsafe { &*v };
    let result = match val {
        Value::Arr(arr) => arr.iter().any(|elem| is_truthy(elem)),
        _ => false,
    };
    unsafe { std::ptr::write(out, Value::Bool(result)) };
}

/// `all` — returns true if all elements in the array are truthy.
/// Unary signature: fn(out: *mut Value, v: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_all(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let val = unsafe { &*v };
    let result = match val {
        Value::Arr(arr) => arr.iter().all(|elem| is_truthy(elem)),
        _ => true,
    };
    unsafe { std::ptr::write(out, Value::Bool(result)) };
}

/// Helper: check if a Value is truthy (not null and not false).
fn is_truthy(v: &Value) -> bool {
    !matches!(v, Value::Null | Value::Bool(false))
}

/// Convert a byte offset in a UTF-8 string to a codepoint (character) offset.
fn byte_offset_to_codepoint(s: &str, byte_offset: usize) -> usize {
    s[..byte_offset].chars().count()
}

/// Find all codepoint-based indices of a substring in a string.
fn string_indices(haystack: &str, needle: &str) -> Vec<usize> {
    if needle.is_empty() {
        return vec![];
    }
    let mut result = vec![];
    let mut byte_start = 0;
    while let Some(byte_pos) = haystack[byte_start..].find(needle) {
        let abs_byte_pos = byte_start + byte_pos;
        result.push(byte_offset_to_codepoint(haystack, abs_byte_pos));
        // Advance by one character (not one byte)
        let next_char_boundary = abs_byte_pos + haystack[abs_byte_pos..].chars().next().map_or(1, |c| c.len_utf8());
        byte_start = next_char_boundary;
    }
    result
}

/// Find all indices where a subsequence appears consecutively in an array.
fn array_subsequence_indices(arr: &[Value], needle: &[Value]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > arr.len() {
        return vec![];
    }
    let mut result = vec![];
    for i in 0..=(arr.len() - needle.len()) {
        if arr[i..i + needle.len()].iter().zip(needle.iter()).all(|(a, b)| values_equal(a, b)) {
            result.push(i);
        }
    }
    result
}

/// `indices(s)` — find all indices of substring s in a string, or subsequence s in an array.
/// Binary signature: fn(out: *mut Value, input: *const Value, s: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_indices(out: *mut Value, input: *const Value, s: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !s.is_null());
    let input_val = unsafe { &*input };
    let s_val = unsafe { &*s };

    let result = match (input_val, s_val) {
        (Value::Str(haystack), Value::Str(needle)) => {
            let indices = string_indices(haystack, needle);
            Value::Arr(Rc::new(indices.into_iter().map(|i| Value::Num(i as f64)).collect()))
        }
        (Value::Arr(arr), Value::Arr(needle)) => {
            // Subsequence matching for arrays
            let indices = array_subsequence_indices(arr, needle);
            Value::Arr(Rc::new(indices.into_iter().map(|i| Value::Num(i as f64)).collect()))
        }
        (Value::Arr(arr), _) => {
            // Single element matching
            let mut indices = vec![];
            for (i, elem) in arr.iter().enumerate() {
                if values_equal(elem, s_val) {
                    indices.push(Value::Num(i as f64));
                }
            }
            Value::Arr(Rc::new(indices))
        }
        _ => Value::Arr(Rc::new(vec![])),
    };
    unsafe { std::ptr::write(out, result) };
}

/// `index(s)` — find first index of s.
/// Binary signature: fn(out: *mut Value, input: *const Value, s: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_str_index(out: *mut Value, input: *const Value, s: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !s.is_null());
    let input_val = unsafe { &*input };
    let s_val = unsafe { &*s };

    let result = match (input_val, s_val) {
        (Value::Str(haystack), Value::Str(needle)) => {
            if needle.is_empty() {
                Value::Null
            } else {
                match haystack.find(needle.as_str()) {
                    Some(byte_pos) => Value::Num(byte_offset_to_codepoint(haystack, byte_pos) as f64),
                    None => Value::Null,
                }
            }
        }
        (Value::Arr(arr), Value::Arr(needle)) => {
            let indices = array_subsequence_indices(arr, needle);
            match indices.first() {
                Some(&idx) => Value::Num(idx as f64),
                None => Value::Null,
            }
        }
        (Value::Arr(arr), _) => {
            match arr.iter().position(|elem| values_equal(elem, s_val)) {
                Some(pos) => Value::Num(pos as f64),
                None => Value::Null,
            }
        }
        _ => Value::Null,
    };
    unsafe { std::ptr::write(out, result) };
}

/// `rindex(s)` — find last index of s.
/// Binary signature: fn(out: *mut Value, input: *const Value, s: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_str_rindex(out: *mut Value, input: *const Value, s: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !s.is_null());
    let input_val = unsafe { &*input };
    let s_val = unsafe { &*s };

    let result = match (input_val, s_val) {
        (Value::Str(haystack), Value::Str(needle)) => {
            if needle.is_empty() {
                Value::Null
            } else {
                match haystack.rfind(needle.as_str()) {
                    Some(byte_pos) => Value::Num(byte_offset_to_codepoint(haystack, byte_pos) as f64),
                    None => Value::Null,
                }
            }
        }
        (Value::Arr(arr), Value::Arr(needle)) => {
            let indices = array_subsequence_indices(arr, needle);
            match indices.last() {
                Some(&idx) => Value::Num(idx as f64),
                None => Value::Null,
            }
        }
        (Value::Arr(arr), _) => {
            match arr.iter().rposition(|elem| values_equal(elem, s_val)) {
                Some(pos) => Value::Num(pos as f64),
                None => Value::Null,
            }
        }
        _ => Value::Null,
    };
    unsafe { std::ptr::write(out, result) };
}

/// Helper: compare two Values for equality (like jq's == operator).
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Num(a), Value::Num(b)) => a == b,
        (Value::Str(a), Value::Str(b)) => a == b,
        (Value::Arr(a), Value::Arr(b)) => a == b,
        (Value::Obj(a), Value::Obj(b)) => a == b,
        _ => false,
    }
}

/// `inside(b)` — returns true if b contains the input.
/// This is the reverse of `contains`: `A | inside(B)` == `B | contains(A)`.
/// Binary signature: fn(out: *mut Value, input: *const Value, b: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_inside(out: *mut Value, input: *const Value, b: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !b.is_null());
    // inside(b) = b | contains(input)
    // Swap arguments and call contains logic
    rt_contains(out, b, input);
}

/// `tojson` — convert value to JSON string.
/// Unary signature: fn(out: *mut Value, v: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_tojson(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let val = unsafe { &*v };
    let json = value_to_json(val);
    unsafe { std::ptr::write(out, Value::Str(Rc::new(json))) };
}

/// `fromjson` — parse JSON string to value.
/// Unary signature: fn(out: *mut Value, v: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_fromjson(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let val = unsafe { &*v };
    let result = match val {
        Value::Str(s) => {
            match json_to_value(s.as_str()) {
                Ok(parsed) => parsed,
                Err(e) => Value::Error(Rc::new(format!("{}", e))),
            }
        }
        _ => Value::Error(Rc::new("fromjson requires a string".to_string())),
    };
    unsafe { std::ptr::write(out, result) };
}

/// `getpath(path)` — get a value at the given path.
/// Binary signature: fn(out: *mut Value, input: *const Value, path: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_getpath(out: *mut Value, input: *const Value, path: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !path.is_null());
    let input_val = unsafe { &*input };
    let path_val = unsafe { &*path };

    let result = match path_val {
        Value::Arr(path_arr) => {
            let mut current = input_val.clone();
            for key in path_arr.iter() {
                current = match (&current, key) {
                    (Value::Obj(obj), Value::Str(k)) => {
                        obj.get(k.as_str()).cloned().unwrap_or(Value::Null)
                    }
                    (Value::Arr(arr), Value::Num(n)) => {
                        let idx = *n as i64;
                        let actual_idx = if idx < 0 {
                            (arr.len() as i64 + idx) as usize
                        } else {
                            idx as usize
                        };
                        arr.get(actual_idx).cloned().unwrap_or(Value::Null)
                    }
                    (Value::Null, _) => Value::Null,
                    _ => Value::Null,
                };
            }
            current
        }
        _ => Value::Null,
    };
    unsafe { std::ptr::write(out, result) };
}

/// `setpath(path; value)` — set a value at the given path.
/// 4-ptr signature: fn(out: *mut Value, input: *const Value, path: *const Value, val: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_setpath(
    out: *mut Value,
    input: *const Value,
    path: *const Value,
    val: *const Value,
) {
    assert!(!out.is_null() && !input.is_null() && !path.is_null() && !val.is_null());
    let input_val = unsafe { &*input };
    let path_val = unsafe { &*path };
    let set_val = unsafe { &*val };

    let result = match path_val {
        Value::Arr(path_arr) => setpath_recursive(input_val, path_arr.as_ref(), 0, set_val),
        _ => input_val.clone(),
    };
    unsafe { std::ptr::write(out, result) };
}

fn setpath_recursive(current: &Value, path: &[Value], idx: usize, val: &Value) -> Value {
    if idx >= path.len() {
        return val.clone();
    }
    let key = &path[idx];
    match key {
        Value::Str(k) => {
            let mut obj = match current {
                Value::Obj(o) => (**o).clone(),
                _ => BTreeMap::new(),
            };
            let existing = obj.get(k.as_str()).cloned().unwrap_or(Value::Null);
            let new_val = setpath_recursive(&existing, path, idx + 1, val);
            // Propagate errors from recursive calls
            if matches!(&new_val, Value::Error(_)) {
                return new_val;
            }
            obj.insert(k.to_string(), new_val);
            Value::Obj(Rc::new(obj))
        }
        Value::Num(n) => {
            if n.is_nan() {
                return Value::Error(Rc::new(
                    "Cannot set array element at NaN index".to_string()
                ));
            }
            // Cannot index an object with a number
            if matches!(current, Value::Obj(_)) {
                return Value::Error(Rc::new(format!(
                    "Cannot index object with number ({})",
                    if *n == n.trunc() && n.abs() < 1e15 { format!("{}", *n as i64) } else { format!("{}", n) }
                )));
            }
            let raw_i = *n as i64;
            let mut arr = match current {
                Value::Arr(a) => (**a).clone(),
                _ => vec![],
            };
            let i = if raw_i < 0 {
                let resolved = arr.len() as i64 + raw_i;
                if resolved < 0 {
                    // Out of bounds negative index
                    return Value::Error(Rc::new(
                        "Out of bounds negative array index".to_string()
                    ));
                }
                resolved as usize
            } else {
                raw_i as usize
            };
            while arr.len() <= i {
                arr.push(Value::Null);
            }
            let existing = arr[i].clone();
            let new_val = setpath_recursive(&existing, path, idx + 1, val);
            // Propagate errors from recursive calls
            if matches!(&new_val, Value::Error(_)) {
                return new_val;
            }
            arr[i] = new_val;
            Value::Arr(Rc::new(arr))
        }
        Value::Arr(_) => {
            // Cannot update field at array index of array
            Value::Error(Rc::new(
                "Cannot update field at array index of array".to_string()
            ))
        }
        _ => current.clone(),
    }
}

/// `delpaths(paths)` — delete values at the given paths.
/// Binary signature: fn(out: *mut Value, input: *const Value, paths: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_delpaths(out: *mut Value, input: *const Value, paths: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !paths.is_null());
    let input_val = unsafe { &*input };
    let paths_val = unsafe { &*paths };

    let result = match paths_val {
        Value::Arr(paths_arr) => {
            let mut current = input_val.clone();
            // Sort paths in reverse order to delete from end first (to preserve indices)
            let mut sorted_paths: Vec<&Value> = paths_arr.iter().collect();
            sorted_paths.sort_by(|a, b| {
                // Compare as path arrays, reverse order
                let a_arr = match a { Value::Arr(a) => a.as_ref(), _ => return std::cmp::Ordering::Equal };
                let b_arr = match b { Value::Arr(b) => b.as_ref(), _ => return std::cmp::Ordering::Equal };
                // Reverse so we delete later indices first
                for (ak, bk) in a_arr.iter().zip(b_arr.iter()) {
                    match (ak, bk) {
                        (Value::Num(an), Value::Num(bn)) => {
                            match bn.partial_cmp(an) {
                                Some(ord) if ord != std::cmp::Ordering::Equal => return ord,
                                _ => continue,
                            }
                        }
                        (Value::Str(a_s), Value::Str(b_s)) => {
                            match b_s.cmp(a_s) {
                                std::cmp::Ordering::Equal => continue,
                                ord => return ord,
                            }
                        }
                        _ => continue,
                    }
                }
                b_arr.len().cmp(&a_arr.len())
            });
            for path in sorted_paths {
                if let Value::Arr(path_arr) = path {
                    current = delpath_single(&current, path_arr.as_ref(), 0);
                }
            }
            current
        }
        _ => Value::Error(Rc::new(
            "Paths must be specified as an array".to_string()
        )),
    };
    unsafe { std::ptr::write(out, result) };
}

/// Delete a single path from a value.
/// Binary signature: fn(out: *mut Value, input: *const Value, path: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_delpath(out: *mut Value, input: *const Value, path: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !path.is_null());
    let input_val = unsafe { &*input };
    let path_val = unsafe { &*path };

    let result = match path_val {
        Value::Arr(path_arr) => delpath_single(input_val, path_arr.as_ref(), 0),
        _ => input_val.clone(),
    };
    unsafe { std::ptr::write(out, result) };
}

fn delpath_single(current: &Value, path: &[Value], idx: usize) -> Value {
    if idx >= path.len() {
        return Value::Null; // Should not normally happen for valid paths
    }
    let key = &path[idx];
    if idx == path.len() - 1 {
        // Last key: delete it
        match (current, key) {
            (Value::Obj(obj), Value::Str(k)) => {
                let mut new_obj = (**obj).clone();
                new_obj.remove(k.as_str());
                Value::Obj(Rc::new(new_obj))
            }
            (Value::Arr(arr), Value::Num(n)) => {
                let i = *n as usize;
                if i < arr.len() {
                    let mut new_arr = (**arr).clone();
                    new_arr.remove(i);
                    Value::Arr(Rc::new(new_arr))
                } else {
                    current.clone()
                }
            }
            _ => current.clone(),
        }
    } else {
        // Intermediate key: recurse
        match (current, key) {
            (Value::Obj(obj), Value::Str(k)) => {
                let mut new_obj = (**obj).clone();
                if let Some(child) = new_obj.get(k.as_str()) {
                    let new_child = delpath_single(child, path, idx + 1);
                    new_obj.insert(k.to_string(), new_child);
                }
                Value::Obj(Rc::new(new_obj))
            }
            (Value::Arr(arr), Value::Num(n)) => {
                let i = *n as usize;
                if i < arr.len() {
                    let mut new_arr = (**arr).clone();
                    new_arr[i] = delpath_single(&arr[i], path, idx + 1);
                    Value::Arr(Rc::new(new_arr))
                } else {
                    current.clone()
                }
            }
            _ => current.clone(),
        }
    }
}

/// `debug` — output value to stderr and pass through.
/// Unary signature: fn(out: *mut Value, v: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_debug(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let val = unsafe { &*v };
    eprintln!("[\"DEBUG:\",{}]", value_to_json(val));
    unsafe { std::ptr::write(out, val.clone()) };
}

/// `env` — returns an object of environment variables.
/// Unary signature: fn(out: *mut Value, v: *const Value)
/// Input is ignored; always returns env object.
#[unsafe(no_mangle)]
pub extern "C" fn rt_env(out: *mut Value, _v: *const Value) {
    assert!(!out.is_null());
    let mut map = BTreeMap::new();
    for (key, value) in std::env::vars() {
        map.insert(key, Value::Str(Rc::new(value)));
    }
    unsafe { std::ptr::write(out, Value::Obj(Rc::new(map))) };
}

/// `builtins` — returns an array of builtin function names.
/// Unary signature: fn(out: *mut Value, v: *const Value)
/// Input is ignored.
#[unsafe(no_mangle)]
pub extern "C" fn rt_builtins(out: *mut Value, _v: *const Value) {
    assert!(!out.is_null());
    // Return a representative list of supported builtins (matching jq's format: "name/arity")
    let builtins = vec![
        "length/0", "type/0", "keys/0", "keys_unsorted/0", "values/0",
        "add/0", "any/0", "all/0", "sort/0", "reverse/0", "unique/0",
        "flatten/0", "min/0", "max/0", "tostring/0", "tonumber/0",
        "ascii_downcase/0", "ascii_upcase/0", "to_entries/0", "from_entries/0",
        "tojson/0", "fromjson/0", "explode/0", "implode/0",
        "floor/0", "ceil/0", "round/0", "fabs/0", "not/0",
        "infinite/0", "nan/0", "isinfinite/0", "isnan/0", "isnormal/0",
        "env/0", "builtins/0", "debug/0", "empty/0",
        "select/1", "map/1", "sort_by/1", "group_by/1", "unique_by/1",
        "min_by/1", "max_by/1", "any/1", "all/1",
        "has/1", "in/1", "contains/1", "inside/1", "split/1",
        "join/1", "startswith/1", "endswith/1", "ltrimstr/1", "rtrimstr/1",
        "indices/1", "index/1", "rindex/1",
        "limit/2", "first/1", "last/1", "nth/2",
        "range/1", "range/2", "range/3",
        "getpath/1", "setpath/2", "delpaths/1",
        "with_entries/1",
        "recurse/0", "recurse/1",
        "input/0", "inputs/0",
        "path/1",
        "reduce/0", "foreach/0",
        "ascii/0",
        "null/0", "true/0", "false/0",
        "if/0", "then/0", "else/0", "elif/0", "end/0",
        "try/0", "catch/0",
        "and/0", "or/0",
    ];
    let arr: Vec<Value> = builtins.iter().map(|s| Value::Str(Rc::new(s.to_string()))).collect();
    unsafe { std::ptr::write(out, Value::Arr(Rc::new(arr))) };
}

/// `infinite` — returns infinity.
/// Unary signature: fn(out: *mut Value, v: *const Value)
/// Input is ignored.
///
/// Returns actual f64::INFINITY. The output formatting (value_to_json)
/// handles the display format to match jq.
#[unsafe(no_mangle)]
pub extern "C" fn rt_infinite(out: *mut Value, _v: *const Value) {
    assert!(!out.is_null());
    unsafe { std::ptr::write(out, Value::Num(f64::INFINITY)) };
}

/// `nan` — returns NaN.
/// Unary signature: fn(out: *mut Value, v: *const Value)
/// Input is ignored.
///
/// Returns actual f64::NAN as Value::Num. The output formatting (value_to_json)
/// will display it as "null" to match jq 1.7.1 behavior, but isnan will
/// correctly identify it as NaN.
#[unsafe(no_mangle)]
pub extern "C" fn rt_nan(out: *mut Value, _v: *const Value) {
    assert!(!out.is_null());
    unsafe { std::ptr::write(out, Value::Num(f64::NAN)) };
}

/// `isinfinite` — returns true if the value is infinite.
/// Unary signature: fn(out: *mut Value, v: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_isinfinite(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let val = unsafe { &*v };
    propagate_error_unary!(out, val);
    let result = match val {
        Value::Num(n) => n.is_infinite(),
        _ => false,
    };
    unsafe { std::ptr::write(out, Value::Bool(result)) };
}

/// `isnan` — returns true if the value is NaN.
/// Unary signature: fn(out: *mut Value, v: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_isnan(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let val = unsafe { &*v };
    propagate_error_unary!(out, val);
    let result = match val {
        Value::Num(n) => n.is_nan(),
        _ => false,
    };
    unsafe { std::ptr::write(out, Value::Bool(result)) };
}

/// `isnormal` — returns true if the value is a normal (finite, non-zero, non-subnormal) number.
/// Unary signature: fn(out: *mut Value, v: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_isnormal(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let val = unsafe { &*v };
    propagate_error_unary!(out, val);
    let result = match val {
        Value::Num(n) => n.is_normal(),
        _ => false,
    };
    unsafe { std::ptr::write(out, Value::Bool(result)) };
}

/// `to_number` — alias for tonumber.
/// Unary signature: fn(out: *mut Value, v: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_to_number(out: *mut Value, v: *const Value) {
    rt_tonumber(out, v);
}

/// `to_string` — alias for tostring.
/// Unary signature: fn(out: *mut Value, v: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_to_string(out: *mut Value, v: *const Value) {
    rt_tostring(out, v);
}

// `with_entries(f)` — equivalent to `to_entries | map(f) | from_entries`.
// This is handled at the compiler level by recognizing the bytecode pattern.
// A runtime fallback is not needed since jq expands it to bytecode.

// =========================================================================
// Phase 10-2: Regex functions
// =========================================================================

/// Build a regex::Regex from a pattern string and flags.
/// Flags: "i" (case insensitive), "x" (extended), "s" (dotall), "m" (multiline).
/// "g" (global) is not a regex flag — it's handled by the caller.
fn build_regex(pattern: &str, flags: &str) -> Result<regex::Regex, String> {
    let mut regex_pattern = String::new();

    // Build inline flags prefix
    let mut inline_flags = String::new();
    for ch in flags.chars() {
        match ch {
            'i' => inline_flags.push('i'),
            'x' => inline_flags.push('x'),
            's' => inline_flags.push('s'),
            'm' => inline_flags.push('m'),
            'g' => {} // handled by caller
            _ => {} // ignore unknown flags
        }
    }

    if !inline_flags.is_empty() {
        regex_pattern.push_str(&format!("(?{})", inline_flags));
    }
    regex_pattern.push_str(pattern);

    regex::Regex::new(&regex_pattern).map_err(|e| format!("{}", e))
}

/// Extract the flags string from a Value. Returns "" if null.
fn extract_flags(flags: &Value) -> &str {
    match flags {
        Value::Str(s) => s.as_str(),
        _ => "",
    }
}

/// Check if the flags string contains "g" (global).
fn has_global_flag(flags: &str) -> bool {
    flags.contains('g')
}

/// Build a match result object matching jq's format:
/// {"offset": N, "length": N, "string": "...", "captures": [...]}
fn build_match_object(
    _input: &str,
    m: &regex::Match,
    captures: &regex::Captures,
    re: &regex::Regex,
) -> Value {
    let mut obj = BTreeMap::new();

    // Main match
    obj.insert(
        "offset".to_string(),
        Value::Num(m.start() as f64),
    );
    obj.insert(
        "length".to_string(),
        Value::Num(m.len() as f64),
    );
    obj.insert(
        "string".to_string(),
        Value::Str(Rc::new(m.as_str().to_string())),
    );

    // Build captures array (skip group 0 = the full match)
    let mut caps_arr = Vec::new();
    let capture_names: Vec<Option<&str>> = re.capture_names().collect();
    for i in 1..captures.len() {
        let cap = captures.get(i);
        let mut cap_obj = BTreeMap::new();
        match cap {
            Some(c) => {
                cap_obj.insert(
                    "offset".to_string(),
                    Value::Num(c.start() as f64),
                );
                cap_obj.insert(
                    "length".to_string(),
                    Value::Num(c.len() as f64),
                );
                cap_obj.insert(
                    "string".to_string(),
                    Value::Str(Rc::new(c.as_str().to_string())),
                );
            }
            None => {
                cap_obj.insert("offset".to_string(), Value::Num(-1.0));
                cap_obj.insert("length".to_string(), Value::Num(0.0));
                cap_obj.insert("string".to_string(), Value::Null);
            }
        }
        // Add capture name (null if unnamed)
        let name = if i < capture_names.len() {
            capture_names[i]
        } else {
            None
        };
        cap_obj.insert(
            "name".to_string(),
            match name {
                Some(n) => Value::Str(Rc::new(n.to_string())),
                None => Value::Null,
            },
        );
        caps_arr.push(Value::Obj(Rc::new(cap_obj)));
    }

    obj.insert(
        "captures".to_string(),
        Value::Arr(Rc::new(caps_arr)),
    );

    Value::Obj(Rc::new(obj))
}

/// `test(re; flags)` — returns true if input matches the regex.
/// Ternary signature: fn(out: *mut Value, input: *const Value, re: *const Value, flags: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_regex_test(
    out: *mut Value,
    input: *const Value,
    re_val: *const Value,
    flags_val: *const Value,
) {
    assert!(!out.is_null() && !input.is_null() && !re_val.is_null() && !flags_val.is_null());
    let input_v = unsafe { &*input };
    let re_v = unsafe { &*re_val };
    let flags_v = unsafe { &*flags_val };

    let result = match (input_v, re_v) {
        (Value::Str(s), Value::Str(pattern)) => {
            let flags = extract_flags(flags_v);
            match build_regex(pattern, flags) {
                Ok(re) => Value::Bool(re.is_match(s)),
                Err(e) => Value::Error(Rc::new(e)),
            }
        }
        _ => Value::Error(Rc::new("test requires string input and pattern".to_string())),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `match(re; flags)` — returns match object(s).
/// When flags contain "g", this is a generator (calls callback for each match).
/// Otherwise, returns a single match object.
///
/// Generator signature: fn(input: ptr, re: ptr, flags: ptr, callback: ptr, ctx: ptr)
#[unsafe(no_mangle)]
pub extern "C" fn rt_regex_match(
    input: *const Value,
    re_val: *const Value,
    flags_val: *const Value,
    callback: extern "C" fn(*const Value, *mut u8),
    ctx: *mut u8,
) {
    assert!(!input.is_null() && !re_val.is_null() && !flags_val.is_null());
    let input_v = unsafe { &*input };
    let re_v = unsafe { &*re_val };
    let flags_v = unsafe { &*flags_val };

    match (input_v, re_v) {
        (Value::Str(s), Value::Str(pattern)) => {
            let flags = extract_flags(flags_v);
            let global = has_global_flag(flags);
            match build_regex(pattern, flags) {
                Ok(re) => {
                    if global {
                        // Generator: yield each match
                        for caps in re.captures_iter(s) {
                            if let Some(m) = caps.get(0) {
                                let match_obj = build_match_object(s, &m, &caps, &re);
                                callback(&match_obj as *const Value, ctx);
                            }
                        }
                    } else {
                        // Scalar: single match
                        if let Some(caps) = re.captures(s) {
                            if let Some(m) = caps.get(0) {
                                let match_obj = build_match_object(s, &m, &caps, &re);
                                callback(&match_obj as *const Value, ctx);
                            }
                        }
                    }
                }
                Err(e) => {
                    let err = Value::Error(Rc::new(e));
                    callback(&err as *const Value, ctx);
                }
            }
        }
        _ => {
            let err = Value::Error(Rc::new("match requires string input and pattern".to_string()));
            callback(&err as *const Value, ctx);
        }
    }
}

/// `capture(re; flags)` — returns object of named captures.
/// Ternary signature: fn(out: *mut Value, input: *const Value, re: *const Value, flags: *const Value)
#[unsafe(no_mangle)]
pub extern "C" fn rt_regex_capture(
    out: *mut Value,
    input: *const Value,
    re_val: *const Value,
    flags_val: *const Value,
) {
    assert!(!out.is_null() && !input.is_null() && !re_val.is_null() && !flags_val.is_null());
    let input_v = unsafe { &*input };
    let re_v = unsafe { &*re_val };
    let flags_v = unsafe { &*flags_val };

    let result = match (input_v, re_v) {
        (Value::Str(s), Value::Str(pattern)) => {
            let flags = extract_flags(flags_v);
            match build_regex(pattern, flags) {
                Ok(re) => {
                    if let Some(caps) = re.captures(s) {
                        let mut obj = BTreeMap::new();
                        for name in re.capture_names().flatten() {
                            let val = match caps.name(name) {
                                Some(m) => Value::Str(Rc::new(m.as_str().to_string())),
                                None => Value::Null,
                            };
                            obj.insert(name.to_string(), val);
                        }
                        Value::Obj(Rc::new(obj))
                    } else {
                        Value::Null
                    }
                }
                Err(e) => Value::Error(Rc::new(e)),
            }
        }
        _ => Value::Error(Rc::new("capture requires string input and pattern".to_string())),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `scan(re; flags)` — generator yielding each match.
/// For patterns without capture groups: yields the matched string.
/// For patterns with capture groups: yields an array of captured strings.
///
/// Generator signature: fn(input: ptr, re: ptr, flags: ptr, callback: ptr, ctx: ptr)
#[unsafe(no_mangle)]
pub extern "C" fn rt_regex_scan(
    input: *const Value,
    re_val: *const Value,
    flags_val: *const Value,
    callback: extern "C" fn(*const Value, *mut u8),
    ctx: *mut u8,
) {
    assert!(!input.is_null() && !re_val.is_null() && !flags_val.is_null());
    let input_v = unsafe { &*input };
    let re_v = unsafe { &*re_val };
    let flags_v = unsafe { &*flags_val };

    match (input_v, re_v) {
        (Value::Str(s), Value::Str(pattern)) => {
            let flags = extract_flags(flags_v);
            match build_regex(pattern, flags) {
                Ok(re) => {
                    let has_groups = re.captures_len() > 1;
                    for caps in re.captures_iter(s) {
                        if has_groups {
                            // Yield array of capture group strings
                            let mut arr = Vec::new();
                            for i in 1..caps.len() {
                                match caps.get(i) {
                                    Some(m) => arr.push(Value::Str(Rc::new(m.as_str().to_string()))),
                                    None => arr.push(Value::Null),
                                }
                            }
                            let result = Value::Arr(Rc::new(arr));
                            callback(&result as *const Value, ctx);
                        } else {
                            // Yield the matched string directly
                            if let Some(m) = caps.get(0) {
                                let result = Value::Str(Rc::new(m.as_str().to_string()));
                                callback(&result as *const Value, ctx);
                            }
                        }
                    }
                }
                Err(e) => {
                    let err = Value::Error(Rc::new(e));
                    callback(&err as *const Value, ctx);
                }
            }
        }
        _ => {
            let err = Value::Error(Rc::new("scan requires string input and pattern".to_string()));
            callback(&err as *const Value, ctx);
        }
    }
}

/// `sub(re; tostr)` / `sub(re; tostr; flags)` — replace first match.
/// 4-arg signature: fn(out: *mut Value, input: ptr, re: ptr, tostr: ptr, flags: ptr)
#[unsafe(no_mangle)]
pub extern "C" fn rt_regex_sub(
    out: *mut Value,
    input: *const Value,
    re_val: *const Value,
    tostr_val: *const Value,
    flags_val: *const Value,
) {
    assert!(!out.is_null() && !input.is_null() && !re_val.is_null() && !tostr_val.is_null() && !flags_val.is_null());
    let input_v = unsafe { &*input };
    let re_v = unsafe { &*re_val };
    let tostr_v = unsafe { &*tostr_val };
    let flags_v = unsafe { &*flags_val };

    let result = match (input_v, re_v, tostr_v) {
        (Value::Str(s), Value::Str(pattern), Value::Str(replacement)) => {
            let flags = extract_flags(flags_v);
            match build_regex(pattern, flags) {
                Ok(re) => {
                    // Replace first match only
                    let replaced = re.replace(s.as_str(), replacement.as_str());
                    Value::Str(Rc::new(replaced.into_owned()))
                }
                Err(e) => Value::Error(Rc::new(e)),
            }
        }
        _ => Value::Error(Rc::new("sub requires string input, pattern, and replacement".to_string())),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `gsub(re; tostr)` / `gsub(re; tostr; flags)` — replace all matches.
/// 4-arg signature: fn(out: *mut Value, input: ptr, re: ptr, tostr: ptr, flags: ptr)
#[unsafe(no_mangle)]
pub extern "C" fn rt_regex_gsub(
    out: *mut Value,
    input: *const Value,
    re_val: *const Value,
    tostr_val: *const Value,
    flags_val: *const Value,
) {
    assert!(!out.is_null() && !input.is_null() && !re_val.is_null() && !tostr_val.is_null() && !flags_val.is_null());
    let input_v = unsafe { &*input };
    let re_v = unsafe { &*re_val };
    let tostr_v = unsafe { &*tostr_val };
    let flags_v = unsafe { &*flags_val };

    let result = match (input_v, re_v, tostr_v) {
        (Value::Str(s), Value::Str(pattern), Value::Str(replacement)) => {
            let flags = extract_flags(flags_v);
            match build_regex(pattern, flags) {
                Ok(re) => {
                    let replaced = re.replace_all(s.as_str(), replacement.as_str());
                    Value::Str(Rc::new(replaced.into_owned()))
                }
                Err(e) => Value::Error(Rc::new(e)),
            }
        }
        _ => Value::Error(Rc::new("gsub requires string input, pattern, and replacement".to_string())),
    };
    unsafe { std::ptr::write(out, result); }
}

// =========================================================================
// Phase 9-5 / 10-3: Path operations and update operators
// =========================================================================

/// Extract all paths from a value that a given expression would access.
///
/// This is a helper for `path(expr)`. The `expr_descriptor` is a pre-analyzed
/// representation of the path expression, stored as a Value:
///
/// - `Value::Arr([Value::Str("a"), Value::Str("b")])` for `.a.b` → single static path
/// - `Value::Str("each")` for `.[]` → enumerate all keys/indices
/// - `Value::Str("each_opt")` for `.[]?` → enumerate all keys/indices (no error on non-iterable)
/// - `Value::Str("recurse")` for `..` → all recursive paths
///
/// Generator signature: fn(input: ptr, descriptor: ptr, callback: ptr, ctx: ptr)
#[unsafe(no_mangle)]
pub extern "C" fn rt_path_of(
    input: *const Value,
    descriptor: *const Value,
    callback: extern "C" fn(*const Value, *mut u8),
    ctx: *mut u8,
) {
    assert!(!input.is_null() && !descriptor.is_null());
    let input_v = unsafe { &*input };
    let desc_v = unsafe { &*descriptor };

    match desc_v {
        Value::Arr(_path_arr) => {
            // Static path: just yield the path array itself
            callback(descriptor, ctx);
        }
        Value::Str(s) if s.as_str() == "each" => {
            // .[] — enumerate all keys/indices
            enumerate_paths(input_v, callback, ctx);
        }
        Value::Str(s) if s.as_str() == "each_opt" => {
            // .[]? — enumerate all keys/indices, silently skip non-iterables
            match input_v {
                Value::Arr(_) | Value::Obj(_) => enumerate_paths(input_v, callback, ctx),
                _ => {} // silently produce nothing
            }
        }
        Value::Str(s) if s.as_str() == "recurse" => {
            // .. — all recursive paths
            let mut path: Vec<Value> = Vec::new();
            recurse_paths(input_v, &mut path, callback, ctx);
        }
        Value::Obj(map) => {
            // Compound descriptor — currently used for "index_then_each" pattern (.a[])
            // {"type": "index_then_each", "keys": ["a"]}
            if let Some(Value::Str(ty)) = map.get("type") {
                if ty.as_str() == "index_then_each" {
                    if let Some(Value::Arr(keys)) = map.get("keys") {
                        // First navigate to the nested object via the keys...
                        let mut current = input_v.clone();
                        for key in keys.iter() {
                            current = match (&current, key) {
                                (Value::Obj(obj), Value::Str(k)) => {
                                    obj.get(k.as_str()).cloned().unwrap_or(Value::Null)
                                }
                                (Value::Arr(arr), Value::Num(n)) => {
                                    let idx = *n as usize;
                                    arr.get(idx).cloned().unwrap_or(Value::Null)
                                }
                                _ => Value::Null,
                            };
                        }
                        // ...then enumerate paths within that nested value
                        match &current {
                            Value::Arr(arr) => {
                                for i in 0..arr.len() {
                                    let mut path: Vec<Value> = keys.iter().cloned().collect();
                                    path.push(Value::Num(i as f64));
                                    let path_val = Value::Arr(Rc::new(path));
                                    callback(&path_val as *const Value, ctx);
                                }
                            }
                            Value::Obj(obj) => {
                                let mut sorted_keys: Vec<&String> = obj.keys().collect();
                                sorted_keys.sort();
                                for k in sorted_keys {
                                    let mut path: Vec<Value> = keys.iter().cloned().collect();
                                    path.push(Value::Str(Rc::new(k.clone())));
                                    let path_val = Value::Arr(Rc::new(path));
                                    callback(&path_val as *const Value, ctx);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        _ => {
            // Unrecognized descriptor — produce nothing
        }
    }
}

/// Enumerate all immediate child paths of a value.
fn enumerate_paths(v: &Value, callback: extern "C" fn(*const Value, *mut u8), ctx: *mut u8) {
    match v {
        Value::Arr(arr) => {
            for i in 0..arr.len() {
                let path = Value::Arr(Rc::new(vec![Value::Num(i as f64)]));
                callback(&path as *const Value, ctx);
            }
        }
        Value::Obj(obj) => {
            // jq iterates object keys in insertion order, but our BTreeMap is sorted
            for key in obj.keys() {
                let path = Value::Arr(Rc::new(vec![Value::Str(Rc::new(key.clone()))]));
                callback(&path as *const Value, ctx);
            }
        }
        _ => {
            // Non-iterable: error
            let err = Value::Error(Rc::new(format!(
                "Cannot iterate over {}",
                match v {
                    Value::Null => "null",
                    Value::Bool(_) => "boolean",
                    Value::Num(_) => "number",
                    Value::Str(_) => "string",
                    _ => "value",
                }
            )));
            callback(&err as *const Value, ctx);
        }
    }
}

/// Recursively enumerate all paths in a value (for `..` / `paths`).
fn recurse_paths(
    v: &Value,
    current_path: &mut Vec<Value>,
    callback: extern "C" fn(*const Value, *mut u8),
    ctx: *mut u8,
) {
    // Yield current path (skip empty path = root)
    if !current_path.is_empty() {
        let path = Value::Arr(Rc::new(current_path.clone()));
        callback(&path as *const Value, ctx);
    }

    match v {
        Value::Arr(arr) => {
            for (i, elem) in arr.iter().enumerate() {
                current_path.push(Value::Num(i as f64));
                recurse_paths(elem, current_path, callback, ctx);
                current_path.pop();
            }
        }
        Value::Obj(obj) => {
            for (key, val) in obj.iter() {
                current_path.push(Value::Str(Rc::new(key.clone())));
                recurse_paths(val, current_path, callback, ctx);
                current_path.pop();
            }
        }
        _ => {} // Scalars have no children
    }
}

/// Update operation: for each path that the path expression accesses,
/// get the current value, apply the update function, and set the result back.
///
/// 5-arg signature: fn(out: ptr, input: ptr, descriptor: ptr, update_fn: ptr, update_ctx_data: ptr)
///
/// update_fn has JIT callback signature:
/// fn(input: *const Value, callback: fn(*const Value, *mut u8), ctx: *mut u8)
#[unsafe(no_mangle)]
pub extern "C" fn rt_update(
    out: *mut Value,
    input: *const Value,
    descriptor: *const Value,
    update_fn: extern "C" fn(*const Value, extern "C" fn(*const Value, *mut u8), *mut u8),
    _update_ctx_data: *mut u8,
) {
    assert!(!out.is_null() && !input.is_null() && !descriptor.is_null());
    let input_v = unsafe { &*input };

    // Collect all paths first
    let mut paths: Vec<Value> = Vec::new();
    let paths_ptr = &mut paths as *mut Vec<Value> as *mut u8;
    rt_path_of(input, descriptor, collect_values_callback, paths_ptr);

    // Two-phase update:
    // Phase 1: Evaluate update_fn for each path, collecting results
    //          (Some(value) = setpath, None = delpath)
    let mut updates: Vec<(&Value, Option<Value>)> = Vec::with_capacity(paths.len());
    let input_snapshot = input_v.clone();
    for path in &paths {
        let current = getpath_value(&input_snapshot, path);

        let mut update_results: Vec<Value> = Vec::new();
        let results_ptr = &mut update_results as *mut Vec<Value> as *mut u8;
        update_fn(&current as *const Value, collect_values_callback, results_ptr);

        if let Some(new_val) = update_results.into_iter().next() {
            if matches!(&new_val, Value::Error(_)) {
                // Error from update_fn means "empty" in try-catch context → delete
                updates.push((path, None));
            } else {
                updates.push((path, Some(new_val)));
            }
        } else {
            // No output → delete this path
            updates.push((path, None));
        }
    }

    // Phase 2: Apply setpath operations forward, collect delpath indices
    let mut result = input_v.clone();
    let mut del_paths: Vec<&Value> = Vec::new();

    for (path, update_val) in &updates {
        if let Some(val) = update_val {
            result = setpath_value(&result, path, val);
        } else {
            del_paths.push(path);
        }
    }

    // Apply delpath in reverse order (to preserve indices)
    for path in del_paths.into_iter().rev() {
        if let Value::Arr(path_arr) = path {
            result = delpath_single(&result, path_arr.as_ref(), 0);
        }
    }

    unsafe { std::ptr::write(out, result) };
}

/// Callback used to collect Values into a Vec<Value>.
extern "C" fn collect_values_callback(value_ptr: *const Value, ctx: *mut u8) {
    assert!(!value_ptr.is_null() && !ctx.is_null());
    let results = unsafe { &mut *(ctx as *mut Vec<Value>) };
    let value = unsafe { (*value_ptr).clone() };
    results.push(value);
}

/// Get a value at a path (helper for rt_update).
fn getpath_value(input: &Value, path: &Value) -> Value {
    match path {
        Value::Arr(path_arr) => {
            let mut current = input.clone();
            for key in path_arr.iter() {
                current = match (&current, key) {
                    (Value::Obj(obj), Value::Str(k)) => {
                        obj.get(k.as_str()).cloned().unwrap_or(Value::Null)
                    }
                    (Value::Arr(arr), Value::Num(n)) => {
                        let idx = *n as i64;
                        let actual_idx = if idx < 0 {
                            (arr.len() as i64 + idx) as usize
                        } else {
                            idx as usize
                        };
                        arr.get(actual_idx).cloned().unwrap_or(Value::Null)
                    }
                    (Value::Null, _) => Value::Null,
                    _ => Value::Null,
                };
            }
            current
        }
        _ => Value::Null,
    }
}

/// Set a value at a path (helper for rt_update). Reuses setpath_recursive.
fn setpath_value(input: &Value, path: &Value, val: &Value) -> Value {
    match path {
        Value::Arr(path_arr) => setpath_recursive(input, path_arr.as_ref(), 0, val),
        _ => input.clone(),
    }
}

/// Slice assignment: `input | .[start:end] = value`
///
/// For strings: always returns Error("Cannot update string slices") (jq behavior).
/// For arrays: replaces arr[start..end] with the contents of value (which must be an array).
/// Float indices are truncated (start→floor, end→ceil) to match jq's slice semantics.
///
/// 4-arg signature: fn(out: ptr, input: ptr, slice_key: ptr, value: ptr)
#[unsafe(no_mangle)]
pub extern "C" fn rt_slice_assign(
    out: *mut Value,
    input: *const Value,
    slice_key: *const Value,
    value: *const Value,
) {
    assert!(!out.is_null() && !input.is_null() && !slice_key.is_null() && !value.is_null());
    let input_v = unsafe { &*input };
    let slice_key_v = unsafe { &*slice_key };
    let value_v = unsafe { &*value };

    let result = match input_v {
        Value::Str(_) => {
            Value::Error(Rc::new("Cannot update string slices".to_string()))
        }
        Value::Arr(arr) => {
            // Extract slice bounds from the key object
            if let Value::Obj(slice_obj) = slice_key_v {
                let len = arr.len();
                let (start, end) = resolve_slice_bounds(slice_obj, len);

                match value_v {
                    Value::Arr(val_arr) => {
                        // Build new array: arr[..start] + val_arr + arr[end..]
                        let mut new_arr = Vec::with_capacity(start + val_arr.len() + (len - end));
                        new_arr.extend_from_slice(&arr[..start]);
                        new_arr.extend_from_slice(val_arr);
                        if end < len {
                            new_arr.extend_from_slice(&arr[end..]);
                        }
                        Value::Arr(Rc::new(new_arr))
                    }
                    _ => {
                        // jq allows assigning non-array to array slice
                        // e.g., [1,2,3] | .[1:2] = "x" → [1,"x",3]
                        // Actually in jq, if value is not array, it errors.
                        // Let's match jq: only arrays can be assigned to slices.
                        Value::Error(Rc::new(format!(
                            "Cannot update array slice with a {}",
                            type_name(value_v)
                        )))
                    }
                }
            } else {
                // Shouldn't happen: key is not a slice object
                input_v.clone()
            }
        }
        Value::Null => {
            // null | .[start:end] = [...] → same as for empty array
            if let Value::Obj(slice_obj) = slice_key_v {
                match value_v {
                    Value::Arr(val_arr) => {
                        let (start, _end) = resolve_slice_bounds(slice_obj, 0);
                        // Fill nulls up to start, then append value
                        let mut new_arr: Vec<Value> = (0..start).map(|_| Value::Null).collect();
                        new_arr.extend_from_slice(val_arr);
                        Value::Arr(Rc::new(new_arr))
                    }
                    _ => Value::Null,
                }
            } else {
                Value::Null
            }
        }
        _ => {
            Value::Error(Rc::new(format!(
                "{} is not an array or string",
                type_name(input_v)
            )))
        }
    };

    unsafe { std::ptr::write(out, result) };
}

// =========================================================================
// Transpose, trim/ltrim/rtrim, date/time builtins
// =========================================================================

/// `transpose` builtin: transposes a 2D array.
/// Pads shorter rows with null.
#[unsafe(no_mangle)]
pub extern "C" fn rt_transpose(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Arr(rows) => {
            if rows.is_empty() {
                Value::Arr(Rc::new(vec![]))
            } else {
                // Find the maximum row length
                let max_len = rows.iter().map(|r| {
                    match r {
                        Value::Arr(a) => a.len(),
                        _ => 0,
                    }
                }).max().unwrap_or(0);
                let mut transposed = Vec::with_capacity(max_len);
                for col in 0..max_len {
                    let mut new_row = Vec::with_capacity(rows.len());
                    for row in rows.iter() {
                        match row {
                            Value::Arr(a) => {
                                new_row.push(a.get(col).cloned().unwrap_or(Value::Null));
                            }
                            _ => {
                                new_row.push(Value::Null);
                            }
                        }
                    }
                    transposed.push(Value::Arr(Rc::new(new_row)));
                }
                Value::Arr(Rc::new(transposed))
            }
        }
        _ => Value::Error(Rc::new(format!(
            "{} is not an array", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `utf8bytelength` builtin: returns the UTF-8 byte length of a string.
#[unsafe(no_mangle)]
pub extern "C" fn rt_utf8bytelength(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Str(s) => Value::Num(s.len() as f64),
        _ => Value::Error(Rc::new(format!(
            "{} ({}) only strings have UTF-8 byte length",
            type_name(v),
            value_desc_short(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// Short value description for error messages (matches jq's format).
fn value_desc_short(v: &Value) -> String {
    crate::value::value_to_json(v)
}

/// `toboolean` builtin: convert string "true"/"false" to boolean.
#[unsafe(no_mangle)]
pub extern "C" fn rt_toboolean(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Bool(b) => Value::Bool(*b),
        Value::Str(s) => {
            match s.as_str() {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                _ => Value::Error(Rc::new(format!(
                    "string ({}) cannot be parsed as a boolean",
                    value_desc_short(v)
                ))),
            }
        }
        _ => Value::Error(Rc::new(format!(
            "{} ({}) cannot be parsed as a boolean",
            type_name(v),
            value_desc_short(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// Check if a character is Unicode whitespace (jq's definition matches Unicode Zs + ASCII ws).
fn is_jq_whitespace(c: char) -> bool {
    matches!(c,
        '\t' | '\n' | '\x0B' | '\x0C' | '\r' | ' ' |
        '\u{0085}' | '\u{00A0}' | '\u{1680}' |
        '\u{2000}'..='\u{200A}' |
        '\u{2028}' | '\u{2029}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
    )
}

/// `trim` builtin: remove leading and trailing whitespace.
#[unsafe(no_mangle)]
pub extern "C" fn rt_trim(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Str(s) => {
            let trimmed = s.trim_matches(is_jq_whitespace);
            Value::Str(Rc::new(trimmed.to_string()))
        }
        _ => Value::Error(Rc::new("trim input must be a string".to_string())),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `ltrim` builtin: remove leading whitespace.
#[unsafe(no_mangle)]
pub extern "C" fn rt_ltrim(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Str(s) => {
            let trimmed = s.trim_start_matches(is_jq_whitespace);
            Value::Str(Rc::new(trimmed.to_string()))
        }
        _ => Value::Error(Rc::new("trim input must be a string".to_string())),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `rtrim` builtin: remove trailing whitespace.
#[unsafe(no_mangle)]
pub extern "C" fn rt_rtrim(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Str(s) => {
            let trimmed = s.trim_end_matches(is_jq_whitespace);
            Value::Str(Rc::new(trimmed.to_string()))
        }
        _ => Value::Error(Rc::new("trim input must be a string".to_string())),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `gmtime` builtin: epoch number → broken-down time array.
/// Returns [year-1900, month-0-indexed, day, hour, min, sec, weekday, yearday].
#[unsafe(no_mangle)]
pub extern "C" fn rt_gmtime(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Num(epoch) => {
            let secs = *epoch as i64;
            let mut tm: libc::tm = unsafe { std::mem::zeroed() };
            let time_val: libc::time_t = secs;
            unsafe { libc::gmtime_r(&time_val, &mut tm); }
            // jq convention: year is full year (tm_year + 1900)
            Value::Arr(Rc::new(vec![
                Value::Num((tm.tm_year + 1900) as f64),  // full year
                Value::Num(tm.tm_mon as f64),             // month 0-11
                Value::Num(tm.tm_mday as f64),            // day 1-31
                Value::Num(tm.tm_hour as f64),            // hour 0-23
                Value::Num(tm.tm_min as f64),             // min 0-59
                Value::Num(tm.tm_sec as f64),             // sec 0-60
                Value::Num(tm.tm_wday as f64),            // weekday 0-6 (Sunday=0)
                Value::Num(tm.tm_yday as f64),            // yearday 0-365
            ]))
        }
        _ => Value::Error(Rc::new(format!(
            "{} is not a number", type_name(v)
        ))),
    };
    unsafe { std::ptr::write(out, result); }
}

/// Parse a broken-down time array into a libc::tm struct.
/// Returns None if the input is not a valid time array.
fn parse_tm_array(arr: &[Value]) -> Option<libc::tm> {
    let get_int = |i: usize| -> Option<i32> {
        if i < arr.len() {
            match &arr[i] {
                Value::Num(n) => Some(*n as i32),
                _ => None,
            }
        } else {
            Some(0) // default to 0 for missing fields
        }
    };

    // Need at least year, month, day (first 3)
    let tm_year = get_int(0)?;
    let tm_mon = get_int(1)?;
    let tm_mday = get_int(2)?;
    let tm_hour = get_int(3).unwrap_or(0);
    let tm_min = get_int(4).unwrap_or(0);
    let tm_sec = get_int(5).unwrap_or(0);
    let tm_wday = get_int(6).unwrap_or(0);
    let tm_yday = get_int(7).unwrap_or(0);

    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // jq convention: year is full year, C needs year-1900
    tm.tm_year = tm_year - 1900;
    tm.tm_mon = tm_mon;
    tm.tm_mday = tm_mday;
    tm.tm_hour = tm_hour;
    tm.tm_min = tm_min;
    tm.tm_sec = tm_sec;
    tm.tm_wday = tm_wday;
    tm.tm_yday = tm_yday;
    tm.tm_isdst = 0;
    // Set timezone to UTC
    tm.tm_gmtoff = 0;

    Some(tm)
}

/// `mktime` builtin: broken-down time array → epoch number.
#[unsafe(no_mangle)]
pub extern "C" fn rt_mktime(out: *mut Value, v: *const Value) {
    assert!(!out.is_null() && !v.is_null());
    let v = unsafe { &*v };
    propagate_error_unary!(out, v);
    let result = match v {
        Value::Arr(arr) => {
            match parse_tm_array(arr) {
                Some(mut tm) => {
                    // Use timegm for UTC (available on macOS and most Unix)
                    let epoch = unsafe { libc::timegm(&mut tm) };
                    Value::Num(epoch as f64)
                }
                None => Value::Error(Rc::new("mktime requires parsed datetime inputs".to_string())),
            }
        }
        _ => Value::Error(Rc::new("mktime requires parsed datetime inputs".to_string())),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `now` builtin: returns current epoch time.
#[unsafe(no_mangle)]
pub extern "C" fn rt_now(out: *mut Value, _v: *const Value) {
    assert!(!out.is_null());
    let mut tv: libc::timeval = unsafe { std::mem::zeroed() };
    unsafe { libc::gettimeofday(&mut tv, std::ptr::null_mut()); }
    let epoch = tv.tv_sec as f64 + tv.tv_usec as f64 / 1_000_000.0;
    unsafe { std::ptr::write(out, Value::Num(epoch)); }
}

/// `strftime(fmt)` builtin: broken-down time array → formatted string.
/// Binary op: input = time array or epoch number, arg = format string.
#[unsafe(no_mangle)]
pub extern "C" fn rt_strftime(out: *mut Value, input: *const Value, fmt: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !fmt.is_null());
    let input = unsafe { &*input };
    let fmt = unsafe { &*fmt };
    propagate_error_bin!(out, input, fmt);

    // Validate format is a string
    let fmt_str = match fmt {
        Value::Str(s) => s.as_str(),
        _ => {
            unsafe { std::ptr::write(out, Value::Error(Rc::new("strftime/1 requires a string format".to_string()))); }
            return;
        }
    };

    // If input is a number (epoch), convert to gmtime first
    let tm_result = match input {
        Value::Num(epoch) => {
            let secs = *epoch as i64;
            let mut tm: libc::tm = unsafe { std::mem::zeroed() };
            let time_val: libc::time_t = secs;
            unsafe { libc::gmtime_r(&time_val, &mut tm); }
            Some(tm)
        }
        Value::Arr(arr) => {
            parse_tm_array(arr)
        }
        _ => None,
    };

    let result = match tm_result {
        Some(tm) => {
            let c_fmt = std::ffi::CString::new(fmt_str).unwrap_or_default();
            let mut buf = vec![0u8; 256];
            let len = unsafe {
                libc::strftime(
                    buf.as_mut_ptr() as *mut libc::c_char,
                    buf.len(),
                    c_fmt.as_ptr(),
                    &tm,
                )
            };
            if len > 0 {
                let s = String::from_utf8_lossy(&buf[..len]).to_string();
                Value::Str(Rc::new(s))
            } else {
                Value::Str(Rc::new(String::new()))
            }
        }
        None => Value::Error(Rc::new("strftime/1 requires parsed datetime inputs".to_string())),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `strflocaltime(fmt)` builtin: like strftime but uses local time.
/// Binary op: input = time array or epoch number, arg = format string.
#[unsafe(no_mangle)]
pub extern "C" fn rt_strflocaltime(out: *mut Value, input: *const Value, fmt: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !fmt.is_null());
    let input = unsafe { &*input };
    let fmt = unsafe { &*fmt };
    propagate_error_bin!(out, input, fmt);

    // Validate format is a string
    let fmt_str = match fmt {
        Value::Str(s) => s.as_str(),
        _ => {
            unsafe { std::ptr::write(out, Value::Error(Rc::new("strflocaltime/1 requires a string format".to_string()))); }
            return;
        }
    };

    let tm_result = match input {
        Value::Num(epoch) => {
            let secs = *epoch as i64;
            let mut tm: libc::tm = unsafe { std::mem::zeroed() };
            let time_val: libc::time_t = secs;
            unsafe { libc::localtime_r(&time_val, &mut tm); }
            Some(tm)
        }
        Value::Arr(arr) => {
            parse_tm_array(arr)
        }
        _ => None,
    };

    let result = match tm_result {
        Some(tm) => {
            let c_fmt = std::ffi::CString::new(fmt_str).unwrap_or_default();
            let mut buf = vec![0u8; 256];
            let len = unsafe {
                libc::strftime(
                    buf.as_mut_ptr() as *mut libc::c_char,
                    buf.len(),
                    c_fmt.as_ptr(),
                    &tm,
                )
            };
            if len > 0 {
                let s = String::from_utf8_lossy(&buf[..len]).to_string();
                Value::Str(Rc::new(s))
            } else {
                Value::Str(Rc::new(String::new()))
            }
        }
        None => Value::Error(Rc::new("strflocaltime/1 requires parsed datetime inputs".to_string())),
    };
    unsafe { std::ptr::write(out, result); }
}

/// `strptime(fmt)` builtin: string → broken-down time array.
/// Binary op: input = date string, arg = format string.
#[unsafe(no_mangle)]
pub extern "C" fn rt_strptime(out: *mut Value, input: *const Value, fmt: *const Value) {
    assert!(!out.is_null() && !input.is_null() && !fmt.is_null());
    let input = unsafe { &*input };
    let fmt = unsafe { &*fmt };
    propagate_error_bin!(out, input, fmt);

    let input_str = match input {
        Value::Str(s) => s.as_str(),
        _ => {
            unsafe { std::ptr::write(out, Value::Error(Rc::new("strptime/1 requires string input".to_string()))); }
            return;
        }
    };

    let fmt_str = match fmt {
        Value::Str(s) => s.as_str(),
        _ => {
            unsafe { std::ptr::write(out, Value::Error(Rc::new("strptime/1 requires a string format".to_string()))); }
            return;
        }
    };

    let c_input = std::ffi::CString::new(input_str).unwrap_or_default();
    let c_fmt = std::ffi::CString::new(fmt_str).unwrap_or_default();
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };

    let ret = unsafe {
        libc::strptime(
            c_input.as_ptr(),
            c_fmt.as_ptr(),
            &mut tm,
        )
    };

    let result = if ret.is_null() {
        Value::Error(Rc::new(format!("date \"{}\" does not match format \"{}\"", input_str, fmt_str)))
    } else {
        // Normalize the tm struct: call timegm then gmtime_r to fill wday/yday
        let epoch = unsafe { libc::timegm(&mut tm) };
        unsafe { libc::gmtime_r(&epoch, &mut tm); }
        // jq convention: year is full year (tm_year + 1900)
        Value::Arr(Rc::new(vec![
            Value::Num((tm.tm_year + 1900) as f64),
            Value::Num(tm.tm_mon as f64),
            Value::Num(tm.tm_mday as f64),
            Value::Num(tm.tm_hour as f64),
            Value::Num(tm.tm_min as f64),
            Value::Num(tm.tm_sec as f64),
            Value::Num(tm.tm_wday as f64),
            Value::Num(tm.tm_yday as f64),
        ]))
    };
    unsafe { std::ptr::write(out, result); }
}
