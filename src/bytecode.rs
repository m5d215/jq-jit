//! Safe wrappers around libjq's bytecode structures.
//!
//! This module provides RAII-managed access to jq's compilation pipeline:
//!
//! - [`JqState`]: Wraps `jq_init()` / `jq_compile()` / `jq_teardown()`.
//! - [`BytecodeRef`]: A borrowed view into a `struct bytecode*` with safe accessors.
//! - [`dump_bytecode`]: Pretty-prints a bytecode tree for debugging.

use std::ffi::{CStr, CString};

use anyhow::{Context, Result, bail};

use crate::jq_ffi::{self, Bytecode, Jv, JvKind, Opcode};

// ---------------------------------------------------------------------------
// JqState: RAII wrapper
// ---------------------------------------------------------------------------

/// Owns a `jq_state*` and ensures proper cleanup via `jq_teardown()`.
pub struct JqState {
    state: *mut jq_ffi::JqState,
}

impl JqState {
    /// Initialize a new jq state.
    pub fn new() -> Result<Self> {
        let state = unsafe { jq_ffi::jq_init() };
        if state.is_null() {
            bail!("jq_init() returned null");
        }
        Ok(Self { state })
    }

    /// Compile a jq filter expression.
    ///
    /// Returns a [`BytecodeRef`] that borrows from this state.
    pub fn compile(&mut self, program: &str) -> Result<BytecodeRef<'_>> {
        let c_program =
            CString::new(program).context("jq program contains null byte")?;
        let result = unsafe { jq_ffi::jq_compile(self.state, c_program.as_ptr()) };
        if result == 0 {
            bail!("jq_compile({:?}) failed", program);
        }

        // Extract the bytecode pointer from jq_state's internal field.
        let bc = unsafe { jq_ffi::jq_state_get_bytecode(self.state) };
        if bc.is_null() {
            bail!("jq_state->bc is null after successful compile");
        }

        Ok(BytecodeRef {
            ptr: bc,
            _lifetime: std::marker::PhantomData,
        })
    }

    /// Get the raw state pointer (for FFI calls that need it).
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
            // jq_teardown nulls the pointer
            debug_assert!(self.state.is_null());
        }
    }
}

// ---------------------------------------------------------------------------
// BytecodeRef: safe accessor for struct bytecode
// ---------------------------------------------------------------------------

/// A borrowed, read-only view of a `struct bytecode`.
///
/// The lifetime `'a` ties this to the [`JqState`] that owns the bytecode tree.
pub struct BytecodeRef<'a> {
    ptr: *mut Bytecode,
    _lifetime: std::marker::PhantomData<&'a Bytecode>,
}

impl<'a> BytecodeRef<'a> {
    /// Raw instruction array as a `&[u16]` slice.
    pub fn code(&self) -> &[u16] {
        unsafe {
            let bc = &*self.ptr;
            if bc.code.is_null() || bc.codelen <= 0 {
                return &[];
            }
            std::slice::from_raw_parts(bc.code, bc.codelen as usize)
        }
    }

    /// Number of 16-bit words in the instruction stream.
    pub fn codelen(&self) -> usize {
        unsafe { (*self.ptr).codelen as usize }
    }

    /// Number of local variables.
    pub fn nlocals(&self) -> usize {
        unsafe { (*self.ptr).nlocals as usize }
    }

    /// Number of closure arguments.
    pub fn nclosures(&self) -> usize {
        unsafe { (*self.ptr).nclosures as usize }
    }

    /// Number of subfunctions.
    pub fn nsubfunctions(&self) -> usize {
        unsafe { (*self.ptr).nsubfunctions as usize }
    }

    /// Access the constants pool (a `jv` array).
    pub fn constants_raw(&self) -> Jv {
        unsafe { jq_ffi::jv_copy((*self.ptr).constants) }
    }

    /// Get the i-th constant from the pool as a display string.
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

    /// Get the name of the i-th C builtin function from `globals->cfunc_names`.
    ///
    /// This mirrors how jq's `dump_operation` resolves CALL_BUILTIN names:
    /// `jv_array_get(jv_copy(bc->globals->cfunc_names), func)`
    pub fn cfunc_name(&self, index: usize) -> String {
        unsafe {
            let bc = &*self.ptr;
            if bc.globals.is_null() {
                return format!("<no globals>");
            }
            let globals = &*bc.globals;
            let names = jq_ffi::jv_copy(globals.cfunc_names);
            let kind = jq_ffi::jv_get_kind(names);
            if kind != JvKind::Array {
                jq_ffi::jv_free(names);
                return format!("<cfunc_names not array>");
            }
            let len = jq_ffi::jv_array_length(jq_ffi::jv_copy(names));
            if (index as i32) >= len {
                jq_ffi::jv_free(names);
                return format!("<cfunc index {} out of range (len={})>", index, len);
            }
            let name_jv = jq_ffi::jv_array_get(jq_ffi::jv_copy(names), index as i32);
            let s = jv_to_display_string(name_jv);
            jq_ffi::jv_free(names);
            // Strip quotes from string display
            s.trim_matches('"').to_string()
        }
    }

    /// Get the nargs of the i-th C builtin function.
    #[allow(dead_code)]
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

    /// Iterator over subfunctions.
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

    /// Raw pointer (for advanced use).
    pub fn as_ptr(&self) -> *const Bytecode {
        self.ptr
    }

    /// Get the debug name of this bytecode function.
    ///
    /// Reads `debuginfo.name` from the bytecode's debuginfo jv object.
    /// Returns `None` if debuginfo is not an object or doesn't contain "name".
    pub fn debugname(&self) -> Option<String> {
        unsafe {
            let bc = &*self.ptr;
            let debuginfo = jq_ffi::jv_copy(bc.debuginfo);
            let kind = jq_ffi::jv_get_kind(debuginfo);
            if kind != JvKind::Object {
                jq_ffi::jv_free(debuginfo);
                return None;
            }
            let name_key = jq_ffi::jv_string(b"name\0".as_ptr() as *const std::ffi::c_char);
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

// ---------------------------------------------------------------------------
// jv display helper
// ---------------------------------------------------------------------------

/// Convert a jv value to a human-readable string (consumes the jv).
unsafe fn jv_to_display_string(val: Jv) -> String {
    unsafe {
        let kind = jq_ffi::jv_get_kind(val);
        match kind {
            JvKind::Number => {
                let n = jq_ffi::jv_number_value(val);
                jq_ffi::jv_free(val);
                // Format integer-valued doubles without decimal point
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

// ---------------------------------------------------------------------------
// Bytecode disassembler
// ---------------------------------------------------------------------------

/// Pretty-print a bytecode's instruction stream.
///
/// Output format matches jq's `--debug-dump-disasm` as closely as possible:
/// ```text
/// 0000 TOP
/// 0001 PUSHK_UNDER 1
/// 0003 DUP
/// 0004 CALL_BUILTIN _plus
/// 0007 RET
/// ```
///
/// The decoding logic mirrors jq's `dump_operation()` in bytecode.c.
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

                // Skip opcodes with length 0 (pseudo-instructions like CLOSURE_CREATE)
                if base_len == 0 {
                    // These shouldn't appear independently in the bytecode stream,
                    // but if they do, just skip one word.
                    out.push_str(&format!("{}{:04} {} (pseudo)\n", prefix, pc, op.name()));
                    pc += 1;
                    continue;
                }

                out.push_str(&format!("{}{:04} {}", prefix, pc, op.name()));

                // Decode immediates following jq's dump_operation logic:
                //
                // if (op->length > 1) {
                //   uint16_t imm = bc->code[pc++];   // first immediate
                //   if CALL_JQ/TAIL_CALL_JQ: special variable-length decoding
                //   else if CALL_BUILTIN: imm=nargs, next word=cfunc_idx
                //   else if BRANCH: imm is offset, print pc+imm as target
                //   else if CONSTANT: imm is constant pool index
                //   else if VARIABLE: imm=level, next word=local_idx
                //   else: print imm as raw number
                // }
                if base_len > 1 {
                    let imm = code.get(pc + 1).copied().unwrap_or(0);

                    if op == Opcode::CALL_JQ || op == Opcode::TAIL_CALL_JQ {
                        // imm = nargs (number of closure args)
                        // Followed by (imm+1) pairs of (level, idx)
                        let nargs = imm as usize;
                        let mut sub_pc = pc + 2;
                        for _i in 0..nargs + 1 {
                            let level = code.get(sub_pc).copied().unwrap_or(0);
                            let idx = code.get(sub_pc + 1).copied().unwrap_or(0);
                            out.push_str(&format!(" {}:{}", level, idx));
                            sub_pc += 2;
                        }
                    } else if op == Opcode::CALL_BUILTIN {
                        // imm = nargs
                        // next word = cfunc index into globals->cfunc_names
                        let cfunc_idx = code.get(pc + 2).copied().unwrap_or(0) as usize;
                        let name = bc.cfunc_name(cfunc_idx);
                        out.push_str(&format!(" {}", name));
                    } else if jq_ffi::opcode_flags(op) & jq_ffi::OP_HAS_BRANCH != 0 {
                        // imm is a relative offset; target = pc+2+imm
                        // In jq's dump_operation: after reading opcode, pc points to imm.
                        // After `*pc++`, pc = imm_addr+1 = opcode_addr+2.
                        // Then target = pc + imm = opcode_addr + 2 + imm.
                        let target = (pc as i32 + 2) + (imm as i16 as i32);
                        out.push_str(&format!(" {:04}", target));
                    } else if jq_ffi::opcode_flags(op) & jq_ffi::OP_HAS_CONSTANT != 0 {
                        let display = bc.constant_display(imm as usize);
                        out.push_str(&format!(" {}", display));
                    } else if jq_ffi::opcode_flags(op) & jq_ffi::OP_HAS_VARIABLE != 0 {
                        // imm = level, next word = local variable index
                        let local_idx = code.get(pc + 2).copied().unwrap_or(0);
                        out.push_str(&format!(" v{}:{}", imm, local_idx));
                    } else {
                        out.push_str(&format!(" {}", imm));
                    }
                }

                out.push('\n');

                // Advance PC using the proper length calculation
                let instr_len =
                    jq_ffi::opcode_length_with_code(op, &code[pc..]);
                if instr_len == 0 {
                    // Safety fallback: avoid infinite loop
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
                pc += 1; // best effort: skip one word
            }
        }
    }

    // Recurse into subfunctions
    let subs = bc.subfunctions();
    for (i, sub) in subs.iter().enumerate() {
        out.push_str(&format!(
            "\n{}--- subfunction {} (codelen={}, nlocals={}, nclosures={}) ---\n",
            prefix,
            i,
            sub.codelen(),
            sub.nlocals(),
            sub.nclosures(),
        ));
        out.push_str(&dump_bytecode(sub, indent + 2));
    }

    out
}

/// Dump bytecode summary (metadata only, no instruction decode).
pub fn dump_bytecode_summary(bc: &BytecodeRef) -> String {
    format!(
        "codelen={}, nlocals={}, nclosures={}, nsubfunctions={}",
        bc.codelen(),
        bc.nlocals(),
        bc.nclosures(),
        bc.nsubfunctions(),
    )
}
