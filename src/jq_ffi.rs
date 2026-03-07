//! Raw FFI bindings for libjq (jq 1.8.1).

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
// Opcode enum (opcode_list.h, jq 1.8.1)
// ---------------------------------------------------------------------------

#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum Opcode {
    LOADK = 0,
    DUP = 1,
    DUPN = 2,
    DUP2 = 3,
    PUSHK_UNDER = 4,
    POP = 5,
    LOADV = 6,
    LOADVN = 7,
    STOREV = 8,
    STORE_GLOBAL = 9,
    INDEX = 10,
    INDEX_OPT = 11,
    EACH = 12,
    EACH_OPT = 13,
    FORK = 14,
    TRY_BEGIN = 15,
    TRY_END = 16,
    JUMP = 17,
    JUMP_F = 18,
    BACKTRACK = 19,
    APPEND = 20,
    INSERT = 21,
    RANGE = 22,
    SUBEXP_BEGIN = 23,
    SUBEXP_END = 24,
    PATH_BEGIN = 25,
    PATH_END = 26,
    CALL_BUILTIN = 27,
    CALL_JQ = 28,
    RET = 29,
    TAIL_CALL_JQ = 30,
    CLOSURE_PARAM = 31,
    CLOSURE_REF = 32,
    CLOSURE_CREATE = 33,
    CLOSURE_CREATE_C = 34,
    TOP = 35,
    CLOSURE_PARAM_REGULAR = 36,
    DEPS = 37,
    MODULEMETA = 38,
    GENLABEL = 39,
    DESTRUCTURE_ALT = 40,
    STOREVN = 41,
    ERRORK = 42,
}

impl Opcode {
    pub fn from_u16(v: u16) -> Option<Self> {
        if v <= 42 {
            Some(unsafe { std::mem::transmute::<u16, Opcode>(v) })
        } else {
            None
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::LOADK => "LOADK",
            Self::DUP => "DUP",
            Self::DUPN => "DUPN",
            Self::DUP2 => "DUP2",
            Self::PUSHK_UNDER => "PUSHK_UNDER",
            Self::POP => "POP",
            Self::LOADV => "LOADV",
            Self::LOADVN => "LOADVN",
            Self::STOREV => "STOREV",
            Self::STORE_GLOBAL => "STORE_GLOBAL",
            Self::INDEX => "INDEX",
            Self::INDEX_OPT => "INDEX_OPT",
            Self::EACH => "EACH",
            Self::EACH_OPT => "EACH_OPT",
            Self::FORK => "FORK",
            Self::TRY_BEGIN => "TRY_BEGIN",
            Self::TRY_END => "TRY_END",
            Self::JUMP => "JUMP",
            Self::JUMP_F => "JUMP_F",
            Self::BACKTRACK => "BACKTRACK",
            Self::APPEND => "APPEND",
            Self::INSERT => "INSERT",
            Self::RANGE => "RANGE",
            Self::SUBEXP_BEGIN => "SUBEXP_BEGIN",
            Self::SUBEXP_END => "SUBEXP_END",
            Self::PATH_BEGIN => "PATH_BEGIN",
            Self::PATH_END => "PATH_END",
            Self::CALL_BUILTIN => "CALL_BUILTIN",
            Self::CALL_JQ => "CALL_JQ",
            Self::RET => "RET",
            Self::TAIL_CALL_JQ => "TAIL_CALL_JQ",
            Self::CLOSURE_PARAM => "CLOSURE_PARAM",
            Self::CLOSURE_REF => "CLOSURE_REF",
            Self::CLOSURE_CREATE => "CLOSURE_CREATE",
            Self::CLOSURE_CREATE_C => "CLOSURE_CREATE_C",
            Self::TOP => "TOP",
            Self::CLOSURE_PARAM_REGULAR => "CLOSURE_PARAM_REGULAR",
            Self::DEPS => "DEPS",
            Self::MODULEMETA => "MODULEMETA",
            Self::GENLABEL => "GENLABEL",
            Self::DESTRUCTURE_ALT => "DESTRUCTURE_ALT",
            Self::STOREVN => "STOREVN",
            Self::ERRORK => "ERRORK",
        }
    }
}

// ---------------------------------------------------------------------------
// Opcode flags
// ---------------------------------------------------------------------------

pub const OP_HAS_CONSTANT: u32 = 2;
pub const OP_HAS_VARIABLE: u32 = 4;
pub const OP_HAS_BRANCH: u32 = 8;
pub const OP_HAS_CFUNC: u32 = 32;
pub const OP_HAS_UFUNC: u32 = 64;
pub const OP_IS_CALL_PSEUDO: u32 = 128;
pub const OP_HAS_BINDING: u32 = 1024;

// ---------------------------------------------------------------------------
// struct bytecode (bytecode.h, jq 1.8.1)
// ---------------------------------------------------------------------------

#[repr(C)]
pub union CFunctionPtr {
    pub a1: unsafe extern "C" fn(*mut JqState, Jv) -> Jv,
    pub a2: unsafe extern "C" fn(*mut JqState, Jv, Jv) -> Jv,
    pub a3: unsafe extern "C" fn(*mut JqState, Jv, Jv, Jv) -> Jv,
    pub a4: unsafe extern "C" fn(*mut JqState, Jv, Jv, Jv, Jv) -> Jv,
}

#[repr(C)]
pub struct CFunction {
    pub fptr: CFunctionPtr,
    pub name: *const c_char,
    pub nargs: c_int,
}

#[repr(C)]
pub struct SymbolTable {
    pub cfunctions: *mut CFunction,
    pub ncfunctions: c_int,
    pub cfunc_names: Jv,
}

#[repr(C)]
pub struct Bytecode {
    pub code: *mut u16,
    pub codelen: c_int,
    pub nlocals: c_int,
    pub nclosures: c_int,
    pub constants: Jv,
    pub globals: *mut SymbolTable,
    pub subfunctions: *mut *mut Bytecode,
    pub nsubfunctions: c_int,
    pub parent: *mut Bytecode,
    pub debuginfo: Jv,
}

// ---------------------------------------------------------------------------
// jq_state
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct JqState {
    _opaque: [u8; 0],
}

const JQ_STATE_BC_OFFSET: usize = 16;

/// # Safety
/// `state` must be a valid pointer to a `JqState` obtained from libjq.
pub unsafe fn jq_state_get_bytecode(state: *mut JqState) -> *mut Bytecode {
    unsafe {
        let base = state as *const u8;
        let bc_ptr_location = base.add(JQ_STATE_BC_OFFSET) as *const *mut Bytecode;
        *bc_ptr_location
    }
}

// ---------------------------------------------------------------------------
// libjq public API
// ---------------------------------------------------------------------------

unsafe extern "C" {
    pub fn jq_init() -> *mut JqState;
    pub fn jq_compile(state: *mut JqState, program: *const c_char) -> c_int;
    pub fn jq_teardown(state: *mut *mut JqState);
    pub fn jq_start(state: *mut JqState, value: Jv, flags: c_int);
    pub fn jq_next(state: *mut JqState) -> Jv;

    pub fn jv_null() -> Jv;
    pub fn jv_number(n: f64) -> Jv;
    pub fn jv_string(s: *const c_char) -> Jv;
    pub fn jv_array() -> Jv;

    pub fn jv_get_kind(v: Jv) -> JvKind;
    pub fn jv_number_value(v: Jv) -> f64;
    pub fn jv_string_value(v: Jv) -> *const c_char;
    pub fn jv_string_length_bytes(v: Jv) -> c_int;
    pub fn jv_array_length(v: Jv) -> c_int;
    pub fn jv_array_get(arr: Jv, idx: c_int) -> Jv;

    pub fn jv_copy(v: Jv) -> Jv;
    pub fn jv_free(v: Jv);

    pub fn jv_invalid_get_msg(v: Jv) -> Jv;
    pub fn jv_parse(s: *const c_char) -> Jv;

    pub fn jv_object_iter(v: Jv) -> c_int;
    pub fn jv_object_iter_next(v: Jv, iter: c_int) -> c_int;
    pub fn jv_object_iter_valid(v: Jv, iter: c_int) -> c_int;
    pub fn jv_object_iter_key(v: Jv, iter: c_int) -> Jv;
    pub fn jv_object_iter_value(v: Jv, iter: c_int) -> Jv;

    pub fn jv_dump_string(v: Jv, flags: c_int) -> Jv;
    pub fn jv_object_get(obj: Jv, key: Jv) -> Jv;
}

// ---------------------------------------------------------------------------
// Opcode metadata
// ---------------------------------------------------------------------------

fn opcode_meta(op: Opcode) -> (u32, usize) {
    match op {
        Opcode::DUP | Opcode::DUPN | Opcode::DUP2 | Opcode::POP | Opcode::INDEX
        | Opcode::INDEX_OPT | Opcode::EACH | Opcode::EACH_OPT | Opcode::TRY_END
        | Opcode::BACKTRACK | Opcode::INSERT | Opcode::SUBEXP_BEGIN | Opcode::SUBEXP_END
        | Opcode::PATH_BEGIN | Opcode::PATH_END | Opcode::RET | Opcode::TOP
        | Opcode::GENLABEL => (0, 1),

        Opcode::LOADK | Opcode::PUSHK_UNDER | Opcode::DEPS | Opcode::MODULEMETA
        | Opcode::ERRORK => (OP_HAS_CONSTANT, 2),

        Opcode::LOADV | Opcode::LOADVN | Opcode::STOREV | Opcode::APPEND | Opcode::RANGE
        | Opcode::STOREVN => (OP_HAS_VARIABLE | OP_HAS_BINDING, 3),

        Opcode::STORE_GLOBAL => (
            OP_HAS_CONSTANT | OP_HAS_VARIABLE | OP_HAS_BINDING | OP_IS_CALL_PSEUDO,
            4,
        ),

        Opcode::FORK | Opcode::TRY_BEGIN | Opcode::JUMP | Opcode::JUMP_F
        | Opcode::DESTRUCTURE_ALT => (OP_HAS_BRANCH, 2),

        Opcode::CALL_BUILTIN => (OP_HAS_CFUNC | OP_HAS_BINDING, 3),

        Opcode::CALL_JQ | Opcode::TAIL_CALL_JQ => {
            (OP_HAS_UFUNC | OP_HAS_BINDING | OP_IS_CALL_PSEUDO, 4)
        }

        Opcode::CLOSURE_PARAM | Opcode::CLOSURE_CREATE | Opcode::CLOSURE_CREATE_C
        | Opcode::CLOSURE_PARAM_REGULAR => (OP_IS_CALL_PSEUDO | OP_HAS_BINDING, 0),

        Opcode::CLOSURE_REF => (OP_IS_CALL_PSEUDO | OP_HAS_BINDING, 2),
    }
}

pub fn opcode_flags(op: Opcode) -> u32 {
    opcode_meta(op).0
}

pub fn opcode_base_length(op: Opcode) -> usize {
    opcode_meta(op).1
}

pub fn opcode_length_with_code(op: Opcode, code: &[u16]) -> usize {
    let base = opcode_base_length(op);
    if (op == Opcode::CALL_JQ || op == Opcode::TAIL_CALL_JQ) && code.len() > 1 {
        let nargs = code[1] as usize;
        base + nargs * 2
    } else {
        base
    }
}
