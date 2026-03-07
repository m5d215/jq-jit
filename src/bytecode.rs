//! Safe wrappers around libjq's bytecode structures.

use std::ffi::{CStr, CString};

use anyhow::{Context, Result, bail};

use crate::jq_ffi::{self, Bytecode, Jv, JvKind, Opcode};

// ---------------------------------------------------------------------------
// JqState: RAII wrapper
// ---------------------------------------------------------------------------

pub struct JqState {
    state: *mut jq_ffi::JqState,
}

impl JqState {
    pub fn new() -> Result<Self> {
        let state = unsafe { jq_ffi::jq_init() };
        if state.is_null() {
            bail!("jq_init() returned null");
        }
        Ok(Self { state })
    }

    pub fn compile(&mut self, program: &str) -> Result<BytecodeRef<'_>> {
        let c_program = CString::new(program).context("jq program contains null byte")?;
        let result = unsafe { jq_ffi::jq_compile(self.state, c_program.as_ptr()) };
        if result == 0 {
            bail!("jq_compile({:?}) failed", program);
        }

        let bc = unsafe { jq_ffi::jq_state_get_bytecode(self.state) };
        if bc.is_null() {
            bail!("jq_state->bc is null after successful compile");
        }

        Ok(BytecodeRef {
            ptr: bc,
            _lifetime: std::marker::PhantomData,
        })
    }

    pub fn as_ptr(&self) -> *mut jq_ffi::JqState {
        self.state
    }
}

impl Drop for JqState {
    fn drop(&mut self) {
        if !self.state.is_null() {
            unsafe {
                jq_ffi::jq_teardown(&mut self.state);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BytecodeRef
// ---------------------------------------------------------------------------

pub struct BytecodeRef<'a> {
    ptr: *mut Bytecode,
    _lifetime: std::marker::PhantomData<&'a Bytecode>,
}

impl<'a> BytecodeRef<'a> {
    pub fn code(&self) -> &[u16] {
        unsafe {
            let bc = &*self.ptr;
            if bc.code.is_null() || bc.codelen <= 0 {
                return &[];
            }
            std::slice::from_raw_parts(bc.code, bc.codelen as usize)
        }
    }

    pub fn codelen(&self) -> usize {
        unsafe { (*self.ptr).codelen as usize }
    }

    pub fn nlocals(&self) -> usize {
        unsafe { (*self.ptr).nlocals as usize }
    }

    pub fn nclosures(&self) -> usize {
        unsafe { (*self.ptr).nclosures as usize }
    }

    pub fn nsubfunctions(&self) -> usize {
        unsafe { (*self.ptr).nsubfunctions as usize }
    }

    pub fn constants_raw(&self) -> Jv {
        unsafe { jq_ffi::jv_copy((*self.ptr).constants) }
    }

    pub fn constant_display(&self, index: usize) -> String {
        unsafe {
            let constants = jq_ffi::jv_copy((*self.ptr).constants);
            let kind = jq_ffi::jv_get_kind(constants);
            if kind != JvKind::Array {
                jq_ffi::jv_free(constants);
                return format!("<constants not an array: {:?}>", kind);
            }
            let len = jq_ffi::jv_array_length(jq_ffi::jv_copy(constants));
            if (index as i32) >= len {
                jq_ffi::jv_free(constants);
                return format!("<constant index {} out of range (len={})>", index, len);
            }
            let val = jq_ffi::jv_array_get(jq_ffi::jv_copy(constants), index as i32);
            let s = jv_to_display_string(val);
            jq_ffi::jv_free(constants);
            s
        }
    }

    pub fn cfunc_name(&self, index: usize) -> String {
        unsafe {
            let bc = &*self.ptr;
            if bc.globals.is_null() {
                return "<no globals>".to_string();
            }
            let globals = &*bc.globals;
            let names = jq_ffi::jv_copy(globals.cfunc_names);
            let kind = jq_ffi::jv_get_kind(names);
            if kind != JvKind::Array {
                jq_ffi::jv_free(names);
                return "<cfunc_names not array>".to_string();
            }
            let len = jq_ffi::jv_array_length(jq_ffi::jv_copy(names));
            if (index as i32) >= len {
                jq_ffi::jv_free(names);
                return format!("<cfunc index {} out of range (len={})>", index, len);
            }
            let name_jv = jq_ffi::jv_array_get(jq_ffi::jv_copy(names), index as i32);
            let s = jv_to_display_string(name_jv);
            jq_ffi::jv_free(names);
            s.trim_matches('"').to_string()
        }
    }

    pub fn cfunc_nargs(&self, index: usize) -> i32 {
        unsafe {
            let bc = &*self.ptr;
            if bc.globals.is_null() {
                return -1;
            }
            let globals = &*bc.globals;
            if index >= globals.ncfunctions as usize {
                return -1;
            }
            (*globals.cfunctions.add(index)).nargs
        }
    }

    pub fn subfunctions(&self) -> Vec<BytecodeRef<'a>> {
        unsafe {
            let bc = &*self.ptr;
            let n = bc.nsubfunctions as usize;
            if n == 0 || bc.subfunctions.is_null() {
                return vec![];
            }
            (0..n)
                .map(|i| BytecodeRef {
                    ptr: *bc.subfunctions.add(i),
                    _lifetime: std::marker::PhantomData,
                })
                .collect()
        }
    }

    pub fn as_ptr(&self) -> *const Bytecode {
        self.ptr
    }

    pub fn debugname(&self) -> Option<String> {
        unsafe {
            let bc = &*self.ptr;
            let debuginfo = jq_ffi::jv_copy(bc.debuginfo);
            let kind = jq_ffi::jv_get_kind(debuginfo);
            if kind != JvKind::Object {
                jq_ffi::jv_free(debuginfo);
                return None;
            }
            let name_key = jq_ffi::jv_string(c"name".as_ptr());
            let name_val = jq_ffi::jv_object_get(jq_ffi::jv_copy(debuginfo), name_key);
            jq_ffi::jv_free(debuginfo);
            let name_kind = jq_ffi::jv_get_kind(name_val);
            if name_kind != JvKind::String {
                jq_ffi::jv_free(name_val);
                return None;
            }
            let cstr = jq_ffi::jv_string_value(name_val);
            let s = CStr::from_ptr(cstr).to_string_lossy().into_owned();
            jq_ffi::jv_free(name_val);
            Some(s)
        }
    }
}

unsafe fn jv_to_display_string(val: Jv) -> String {
    unsafe {
        let kind = jq_ffi::jv_get_kind(val);
        match kind {
            JvKind::Number => {
                let n = jq_ffi::jv_number_value(val);
                jq_ffi::jv_free(val);
                if n == n.trunc() && n.abs() < 1e15 {
                    format!("{}", n as i64)
                } else {
                    format!("{}", n)
                }
            }
            JvKind::String => {
                let cstr = jq_ffi::jv_string_value(val);
                let s = CStr::from_ptr(cstr).to_string_lossy().into_owned();
                jq_ffi::jv_free(val);
                format!("\"{}\"", s)
            }
            JvKind::Null => {
                jq_ffi::jv_free(val);
                "null".to_string()
            }
            JvKind::True => {
                jq_ffi::jv_free(val);
                "true".to_string()
            }
            JvKind::False => {
                jq_ffi::jv_free(val);
                "false".to_string()
            }
            JvKind::Array => {
                let len = jq_ffi::jv_array_length(jq_ffi::jv_copy(val));
                jq_ffi::jv_free(val);
                format!("[array of {}]", len)
            }
            JvKind::Object => {
                jq_ffi::jv_free(val);
                "{object}".to_string()
            }
            JvKind::Invalid => {
                jq_ffi::jv_free(val);
                "<invalid>".to_string()
            }
        }
    }
}

pub fn dump_bytecode(bc: &BytecodeRef, indent: usize) -> String {
    let mut out = String::new();
    let prefix = " ".repeat(indent);
    let code = bc.code();
    let mut pc: usize = 0;

    while pc < code.len() {
        let raw_opcode = code[pc];
        match Opcode::from_u16(raw_opcode) {
            Some(op) => {
                let base_len = jq_ffi::opcode_base_length(op);
                if base_len == 0 {
                    out.push_str(&format!("{}{:04} {} (pseudo)\n", prefix, pc, op.name()));
                    pc += 1;
                    continue;
                }

                out.push_str(&format!("{}{:04} {}", prefix, pc, op.name()));

                if base_len > 1 {
                    let imm = code.get(pc + 1).copied().unwrap_or(0);

                    if op == Opcode::CALL_JQ || op == Opcode::TAIL_CALL_JQ {
                        let nargs = imm as usize;
                        let mut sub_pc = pc + 2;
                        for _ in 0..nargs + 1 {
                            let level = code.get(sub_pc).copied().unwrap_or(0);
                            let idx = code.get(sub_pc + 1).copied().unwrap_or(0);
                            out.push_str(&format!(" {}:{}", level, idx));
                            sub_pc += 2;
                        }
                    } else if op == Opcode::CALL_BUILTIN {
                        let cfunc_idx = code.get(pc + 2).copied().unwrap_or(0) as usize;
                        let name = bc.cfunc_name(cfunc_idx);
                        out.push_str(&format!(" {}", name));
                    } else if jq_ffi::opcode_flags(op) & jq_ffi::OP_HAS_BRANCH != 0 {
                        let target = (pc as i32 + 2) + (imm as i16 as i32);
                        out.push_str(&format!(" {:04}", target));
                    } else if jq_ffi::opcode_flags(op) & jq_ffi::OP_HAS_CONSTANT != 0 {
                        let display = bc.constant_display(imm as usize);
                        out.push_str(&format!(" {}", display));
                    } else if jq_ffi::opcode_flags(op) & jq_ffi::OP_HAS_VARIABLE != 0 {
                        let local_idx = code.get(pc + 2).copied().unwrap_or(0);
                        out.push_str(&format!(" v{}:{}", imm, local_idx));
                    } else {
                        out.push_str(&format!(" {}", imm));
                    }
                }

                out.push('\n');
                let instr_len = jq_ffi::opcode_length_with_code(op, &code[pc..]);
                if instr_len == 0 {
                    pc += 1;
                } else {
                    pc += instr_len;
                }
            }
            None => {
                out.push_str(&format!(
                    "{}{:04} <unknown opcode {}>\n",
                    prefix, pc, raw_opcode
                ));
                pc += 1;
            }
        }
    }

    let subs = bc.subfunctions();
    for (i, sub) in subs.iter().enumerate() {
        let name = sub.debugname().unwrap_or_default();
        out.push_str(&format!(
            "\n{}--- subfunction {} \"{}\" (codelen={}, nlocals={}, nclosures={}) ---\n",
            prefix,
            i,
            name,
            sub.codelen(),
            sub.nlocals(),
            sub.nclosures(),
        ));
        out.push_str(&dump_bytecode(sub, indent + 2));
    }

    out
}
