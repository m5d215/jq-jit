//! Raw FFI bindings for libjq (jq 1.8.1).
//!
//! As of #67 the crate no longer calls libjq — parsing, evaluation, and
//! (optionally) JIT all run on native Rust. This file is kept for one more
//! step so the libjq link can be torn down cleanly in #68; none of these
//! declarations have live consumers, hence the blanket `allow(dead_code)`.

#![allow(dead_code)]

use std::ffi::c_char;
use std::ffi::c_int;

// ---------------------------------------------------------------------------
// jv type (jv.h)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JvKind {
    Invalid = 0,
    Null = 1,
    False = 2,
    True = 3,
    Number = 4,
    String = 5,
    Array = 6,
    Object = 7,
}

#[repr(C)]
pub struct JvRefcnt {
    _opaque: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Jv {
    pub kind_flags: u8,
    pub pad_: u8,
    pub offset: u16,
    pub size: i32,
    pub u: JvPayload,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union JvPayload {
    pub ptr: *mut JvRefcnt,
    pub number: f64,
}

const _: () = assert!(std::mem::size_of::<Jv>() == 16);

// ---------------------------------------------------------------------------
// libjq public API
// ---------------------------------------------------------------------------

unsafe extern "C" {
    pub fn jv_string(s: *const c_char) -> Jv;

    pub fn jv_get_kind(v: Jv) -> JvKind;
    pub fn jv_number_value(v: Jv) -> f64;
    pub fn jv_string_value(v: Jv) -> *const c_char;
    pub fn jv_string_length_bytes(v: Jv) -> c_int;
    pub fn jv_array_length(v: Jv) -> c_int;
    pub fn jv_array_get(arr: Jv, idx: c_int) -> Jv;

    pub fn jv_copy(v: Jv) -> Jv;
    pub fn jv_free(v: Jv);

    pub fn jv_object_iter(v: Jv) -> c_int;
    pub fn jv_object_iter_next(v: Jv, iter: c_int) -> c_int;
    pub fn jv_object_iter_valid(v: Jv, iter: c_int) -> c_int;
    pub fn jv_object_iter_key(v: Jv, iter: c_int) -> Jv;
    pub fn jv_object_iter_value(v: Jv, iter: c_int) -> Jv;

    pub fn jv_dump_string(v: Jv, flags: c_int) -> Jv;
    pub fn jv_object_get(obj: Jv, key: Jv) -> Jv;
}

