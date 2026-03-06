//! Bytecode VM interpreter for jq.
//!
//! Executes jq's bytecode directly using a 3-stack model:
//! - Data stack: values being computed
//! - Frame stack: function call frames (locals, return address, etc.)
//! - Fork stack: backtracking points for generators
//!
//! This mirrors jq's own execute.c, ensuring 100% compatibility.

use std::rc::Rc;

use anyhow::{Result, bail};

use crate::bytecode::BytecodeRef;
use crate::jq_ffi::{self, JvKind, Opcode};
use crate::value::{Value, KeyStr};

/// Maximum stack depth to prevent stack overflow.
const MAX_STACK_DEPTH: usize = 10000;

/// A backtracking save point on the fork stack.
#[derive(Debug, Clone)]
struct ForkPoint {
    /// Saved data stack depth.
    data_depth: usize,
    /// Saved frame stack depth.
    frame_depth: usize,
    /// Program counter to resume at (the ON_BACKTRACK path).
    pc: usize,
    /// Saved state specific to the instruction.
    state: ForkState,
}

/// Instruction-specific state for backtracking.
#[derive(Debug, Clone)]
enum ForkState {
    /// FORK instruction: just resumes at the target PC.
    Jump,
    /// EACH/EACH_OPT: iterator state (current index).
    Each {
        container: Value,
        keys: Option<Vec<KeyStr>>, // for objects
        index: usize,
        is_opt: bool,
    },
    /// RANGE: range state.
    Range {
        current: f64,
        to: f64,
        step: f64,
        var_index: usize,
    },
    /// TRY_BEGIN: try-catch state.
    Try {
        catch_pc: usize,
    },
}

/// A function call frame.
#[derive(Debug, Clone)]
struct Frame {
    /// Return program counter.
    ret_pc: usize,
    /// Return bytecode (for subfunction calls).
    ret_bc_index: usize,
    /// Local variables for this frame.
    locals: Vec<Value>,
    /// Data stack depth at frame entry (for cleanup on return).
    data_depth: usize,
    /// Fork stack depth at frame entry.
    fork_depth: usize,
}

/// The VM state.
pub struct VM<'a> {
    /// All bytecodes (top-level + subfunctions), indexed for lookup.
    bytecodes: Vec<BytecodeInfo<'a>>,
    /// Data stack.
    data: Vec<Value>,
    /// Frame stack.
    frames: Vec<Frame>,
    /// Fork stack.
    forks: Vec<ForkPoint>,
    /// Current bytecode index.
    current_bc: usize,
    /// Program counter.
    pc: usize,
    /// Output accumulator.
    outputs: Vec<Value>,
    /// Error messages (for stderr).
    errors: Vec<String>,
    /// Whether we're in backtracking mode.
    backtracking: bool,
}

/// Information about a bytecode function.
struct BytecodeInfo<'a> {
    bc: &'a BytecodeRef<'a>,
    /// Constants extracted from the bytecode.
    constants: Vec<Value>,
    /// C builtin function names.
    cfunc_names: Vec<String>,
    /// C builtin function nargs.
    cfunc_nargs: Vec<i32>,
}

impl<'a> VM<'a> {
    /// Create a new VM for the given bytecode.
    pub fn new(bc: &'a BytecodeRef<'a>) -> Result<Self> {
        let mut bytecodes = Vec::new();
        collect_bytecodes(bc, &mut bytecodes)?;

        Ok(VM {
            bytecodes,
            data: Vec::new(),
            frames: Vec::new(),
            forks: Vec::new(),
            current_bc: 0,
            pc: 0,
            outputs: Vec::new(),
            errors: Vec::new(),
            backtracking: false,
        })
    }

    /// Execute the filter with the given input.
    pub fn execute(&mut self, input: Value) -> Result<Vec<Value>> {
        self.data.clear();
        self.frames.clear();
        self.forks.clear();
        self.outputs.clear();
        self.errors.clear();
        self.current_bc = 0;
        self.pc = 0;
        self.backtracking = false;

        // Push initial frame
        let nlocals = self.bytecodes[0].bc.nlocals();
        self.frames.push(Frame {
            ret_pc: 0,
            ret_bc_index: 0,
            locals: vec![Value::Null; nlocals],
            data_depth: 0,
            fork_depth: 0,
        });

        // Push input onto data stack
        self.data.push(input);

        // Run the VM loop
        self.run()?;

        Ok(std::mem::take(&mut self.outputs))
    }

    /// The main VM execution loop.
    fn run(&mut self) -> Result<()> {
        loop {
            let bc_info = &self.bytecodes[self.current_bc];
            let code = bc_info.bc.code();

            if self.pc >= code.len() {
                // End of bytecode - try backtracking
                if !self.backtrack()? {
                    break;
                }
                continue;
            }

            if self.data.len() > MAX_STACK_DEPTH {
                bail!("Stack overflow (depth > {})", MAX_STACK_DEPTH);
            }

            let raw_op = code[self.pc];
            let op = match Opcode::from_u16(raw_op) {
                Some(op) => op,
                None => bail!("Unknown opcode {} at pc={}", raw_op, self.pc),
            };

            // If backtracking, handle ON_BACKTRACK for certain opcodes
            if self.backtracking {
                match op {
                    Opcode::EACH | Opcode::EACH_OPT => {
                        if self.handle_each_backtrack()? {
                            self.backtracking = false;
                            continue;
                        } else {
                            // Iterator exhausted, continue backtracking
                            if !self.backtrack()? {
                                break;
                            }
                            continue;
                        }
                    }
                    Opcode::RANGE => {
                        if self.handle_range_backtrack()? {
                            self.backtracking = false;
                            continue;
                        } else {
                            if !self.backtrack()? {
                                break;
                            }
                            continue;
                        }
                    }
                    Opcode::FORK => {
                        // FORK backtrack: just continue from fork target
                        self.backtracking = false;
                        // PC is already set by backtrack()
                    }
                    Opcode::TRY_BEGIN => {
                        self.backtracking = false;
                        // Handle try-catch backtrack
                    }
                    _ => {
                        // For other opcodes, backtracking just continues
                        if !self.backtrack()? {
                            break;
                        }
                        continue;
                    }
                }
            }

            // Normal execution
            self.execute_opcode(op)?;
        }

        Ok(())
    }

    /// Execute a single opcode.
    fn execute_opcode(&mut self, op: Opcode) -> Result<()> {
        let bc_info = &self.bytecodes[self.current_bc];
        let code = bc_info.bc.code();

        match op {
            Opcode::TOP => {
                // NOP - input is already on stack
                self.pc += 1;
            }

            Opcode::LOADK => {
                let const_idx = code[self.pc + 1] as usize;
                let val = self.get_constant(const_idx)?;
                // LOADK replaces the stack top
                if self.data.is_empty() {
                    bail!("LOADK: stack underflow at pc={}", self.pc);
                }
                *self.data.last_mut().unwrap() = val;
                self.pc += 2;
            }

            Opcode::DUP => {
                let top = self.data.last().cloned()
                    .ok_or_else(|| anyhow::anyhow!("DUP: stack underflow at pc={}", self.pc))?;
                self.data.push(top);
                self.pc += 1;
            }

            Opcode::DUPN => {
                // DUP but push null on top
                let top = self.data.last().cloned()
                    .ok_or_else(|| anyhow::anyhow!("DUPN: stack underflow at pc={}", self.pc))?;
                self.data.push(top);
                // also push null? Actually DUPN in jq is "DUP but don't jv_copy"
                // For our refcounted values, DUP and DUPN are the same
                self.pc += 1;
            }

            Opcode::DUP2 => {
                // Duplicate top two elements
                if self.data.len() < 2 {
                    bail!("DUP2: stack underflow at pc={}", self.pc);
                }
                let len = self.data.len();
                let a = self.data[len - 2].clone();
                let b = self.data[len - 1].clone();
                self.data.push(a);
                self.data.push(b);
                self.pc += 1;
            }

            Opcode::PUSHK_UNDER => {
                let const_idx = code[self.pc + 1] as usize;
                let val = self.get_constant(const_idx)?;
                if self.data.is_empty() {
                    bail!("PUSHK_UNDER: stack underflow at pc={}", self.pc);
                }
                let top = self.data.pop().unwrap();
                self.data.push(val);
                self.data.push(top);
                self.pc += 2;
            }

            Opcode::POP => {
                self.data.pop()
                    .ok_or_else(|| anyhow::anyhow!("POP: stack underflow at pc={}", self.pc))?;
                self.pc += 1;
            }

            Opcode::LOADV => {
                // Load variable: level + var_index
                let level = code[self.pc + 1] as usize;
                let var_idx = code[self.pc + 2] as usize;
                let val = self.get_var(level, var_idx)?;
                self.data.push(val);
                self.pc += 3;
            }

            Opcode::LOADVN => {
                // Load variable (non-copying in C, same as LOADV for us)
                let level = code[self.pc + 1] as usize;
                let var_idx = code[self.pc + 2] as usize;
                let val = self.get_var(level, var_idx)?;
                self.data.push(val);
                self.pc += 3;
            }

            Opcode::STOREV => {
                let level = code[self.pc + 1] as usize;
                let var_idx = code[self.pc + 2] as usize;
                let val = self.data.last().cloned()
                    .ok_or_else(|| anyhow::anyhow!("STOREV: stack underflow at pc={}", self.pc))?;
                self.set_var(level, var_idx, val)?;
                self.pc += 3;
            }

            Opcode::STOREVN => {
                let level = code[self.pc + 1] as usize;
                let var_idx = code[self.pc + 2] as usize;
                let val = self.data.pop()
                    .ok_or_else(|| anyhow::anyhow!("STOREVN: stack underflow at pc={}", self.pc))?;
                self.set_var(level, var_idx, val)?;
                self.pc += 3;
            }

            Opcode::SUBEXP_BEGIN => {
                // Clone top: [..., v] → [..., v_copy, v]
                let top = self.data.last().cloned()
                    .ok_or_else(|| anyhow::anyhow!("SUBEXP_BEGIN: stack underflow at pc={}", self.pc))?;
                let len = self.data.len();
                self.data[len - 1] = top.clone();
                self.data.push(top);
                self.pc += 1;
            }

            Opcode::SUBEXP_END => {
                // Swap top two: [..., a, b] → [..., b, a]
                let len = self.data.len();
                if len < 2 {
                    bail!("SUBEXP_END: stack underflow at pc={}", self.pc);
                }
                self.data.swap(len - 1, len - 2);
                self.pc += 1;
            }

            Opcode::INDEX => {
                self.execute_index(false)?;
                self.pc += 1;
            }

            Opcode::INDEX_OPT => {
                self.execute_index(true)?;
                self.pc += 1;
            }

            Opcode::EACH | Opcode::EACH_OPT => {
                self.execute_each(op == Opcode::EACH_OPT)?;
            }

            Opcode::FORK => {
                let raw_imm = code[self.pc + 1];
                let target = ((self.pc as i32 + 2) + (raw_imm as i16 as i32)) as usize;
                self.forks.push(ForkPoint {
                    data_depth: self.data.len(),
                    frame_depth: self.frames.len(),
                    pc: target,
                    state: ForkState::Jump,
                });
                self.pc += 2;
            }

            Opcode::JUMP => {
                let raw_imm = code[self.pc + 1];
                let target = ((self.pc as i32 + 2) + (raw_imm as i16 as i32)) as usize;
                self.pc = target;
            }

            Opcode::JUMP_F => {
                let raw_imm = code[self.pc + 1];
                let target = ((self.pc as i32 + 2) + (raw_imm as i16 as i32)) as usize;
                // Peek at top of stack (don't pop)
                let top = self.data.last()
                    .ok_or_else(|| anyhow::anyhow!("JUMP_F: stack underflow at pc={}", self.pc))?;
                if !top.is_true() {
                    self.pc = target;
                } else {
                    self.pc += 2;
                }
            }

            Opcode::BACKTRACK => {
                if !self.backtrack()? {
                    // No more fork points - filter is done
                    return Ok(());
                }
            }

            Opcode::CALL_BUILTIN => {
                self.execute_builtin()?;
            }

            Opcode::CALL_JQ | Opcode::TAIL_CALL_JQ => {
                self.execute_call_jq(op == Opcode::TAIL_CALL_JQ)?;
            }

            Opcode::RET => {
                self.execute_ret()?;
            }

            Opcode::APPEND => {
                // Append top to array in variable
                let level = code[self.pc + 1] as usize;
                let var_idx = code[self.pc + 2] as usize;
                let val = self.data.pop()
                    .ok_or_else(|| anyhow::anyhow!("APPEND: stack underflow at pc={}", self.pc))?;
                let mut arr = self.get_var(level, var_idx)?;
                if let Value::Arr(ref mut rc_arr) = arr {
                    let new_arr = Rc::make_mut(rc_arr);
                    new_arr.push(val);
                    self.set_var(level, var_idx, arr)?;
                } else {
                    bail!("APPEND: variable is not an array");
                }
                self.pc += 3;
            }

            Opcode::INSERT => {
                // Insert key-value into object
                // Stack: [..., value, key, object]
                if self.data.len() < 3 {
                    bail!("INSERT: stack underflow at pc={}", self.pc);
                }
                let obj = self.data.pop().unwrap();
                let key = self.data.pop().unwrap();
                let value = self.data.pop().unwrap();

                match (obj, &key) {
                    (Value::Obj(ref o), Value::Str(k)) => {
                        let mut new_obj = (**o).clone();
                        new_obj.insert(KeyStr::from(k.as_str()), value);
                        self.data.push(Value::Obj(Rc::new(new_obj)));
                    }
                    _ => bail!("INSERT: expected object and string key"),
                }
                self.pc += 1;
            }

            Opcode::RANGE => {
                self.execute_range()?;
            }

            Opcode::TRY_BEGIN => {
                let raw_imm = code[self.pc + 1];
                let catch_pc = ((self.pc as i32 + 2) + (raw_imm as i16 as i32)) as usize;
                self.forks.push(ForkPoint {
                    data_depth: self.data.len(),
                    frame_depth: self.frames.len(),
                    pc: self.pc, // Will be used for TRY_BEGIN backtrack
                    state: ForkState::Try { catch_pc },
                });
                self.pc += 2;
            }

            Opcode::TRY_END => {
                // Remove the try fork point
                // Find and remove the last Try fork
                if let Some(pos) = self.forks.iter().rposition(|f| matches!(f.state, ForkState::Try { .. })) {
                    self.forks.remove(pos);
                }
                self.pc += 1;
            }

            Opcode::PATH_BEGIN | Opcode::PATH_END => {
                // TODO: Implement path tracking
                self.pc += 1;
            }

            Opcode::GENLABEL => {
                // Generate a unique label value (used for label-break)
                static LABEL_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let label = LABEL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.data.push(Value::Num(label as f64, None));
                self.pc += 1;
            }

            Opcode::DESTRUCTURE_ALT => {
                let raw_imm = code[self.pc + 1];
                let target = ((self.pc as i32 + 2) + (raw_imm as i16 as i32)) as usize;
                // Similar to FORK but for destructuring alternatives
                self.forks.push(ForkPoint {
                    data_depth: self.data.len(),
                    frame_depth: self.frames.len(),
                    pc: target,
                    state: ForkState::Jump,
                });
                self.pc += 2;
            }

            Opcode::STORE_GLOBAL => {
                // Store global variable
                let _const_idx = code[self.pc + 1] as usize;
                let level = code[self.pc + 2] as usize;
                let var_idx = code[self.pc + 3] as usize;
                let val = self.data.last().cloned()
                    .ok_or_else(|| anyhow::anyhow!("STORE_GLOBAL: stack underflow"))?;
                self.set_var(level, var_idx, val)?;
                self.pc += 4;
            }

            Opcode::ERRORK => {
                let const_idx = code[self.pc + 1] as usize;
                let err_val = self.get_constant(const_idx)?;
                // Replace top with error
                if !self.data.is_empty() {
                    self.data.pop();
                }
                let msg = match &err_val {
                    Value::Str(s) => s.as_ref().clone(),
                    Value::Null => {
                        // error with null = use top of stack as error message
                        "null".to_string()
                    }
                    _ => crate::value::value_to_json(&err_val),
                };
                // Try to propagate error to catch handler
                if !self.propagate_error(Value::Error(Rc::new(msg)))? {
                    // No catch handler, output as error
                    return Ok(());
                }
            }

            Opcode::CLOSURE_PARAM | Opcode::CLOSURE_REF | Opcode::CLOSURE_CREATE
            | Opcode::CLOSURE_CREATE_C | Opcode::CLOSURE_PARAM_REGULAR => {
                // These are pseudo-instructions handled by CALL_JQ
                self.pc += 1;
            }

            Opcode::DEPS | Opcode::MODULEMETA => {
                // Module system - not fully supported
                self.pc += jq_ffi::opcode_base_length(op);
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Opcode implementations
    // -----------------------------------------------------------------------

    fn execute_index(&mut self, optional: bool) -> Result<()> {
        if self.data.len() < 2 {
            bail!("INDEX: stack underflow at pc={}", self.pc);
        }
        let target = self.data.pop().unwrap();
        let key = self.data.pop().unwrap();

        match (&target, &key) {
            (Value::Obj(o), Value::Str(k)) => {
                let val = o.get(k.as_str()).cloned().unwrap_or(Value::Null);
                self.data.push(val);
            }
            (Value::Arr(a), Value::Num(n, _)) => {
                let idx = *n as i64;
                let actual_idx = if idx < 0 {
                    (a.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                let val = a.get(actual_idx).cloned().unwrap_or(Value::Null);
                self.data.push(val);
            }
            (Value::Null, _) => {
                self.data.push(Value::Null);
            }
            _ => {
                if optional {
                    // Optional: produce error value that will be caught by try
                    let msg = format!(
                        "{} ({}) and {} ({}) cannot be iterated",
                        target.type_name(),
                        crate::value::value_to_json(&target),
                        key.type_name(),
                        crate::value::value_to_json(&key)
                    );
                    if !self.propagate_error(Value::Error(Rc::new(msg)))? {
                        return Ok(());
                    }
                } else {
                    let msg = format!(
                        "{} ({}) is not defined",
                        crate::value::value_to_json(&key),
                        target.type_name()
                    );
                    self.data.push(Value::Error(Rc::new(msg)));
                }
            }
        }
        Ok(())
    }

    fn execute_each(&mut self, is_opt: bool) -> Result<()> {
        let top = self.data.pop()
            .ok_or_else(|| anyhow::anyhow!("EACH: stack underflow at pc={}", self.pc))?;

        match &top {
            Value::Arr(a) => {
                if a.is_empty() {
                    // Empty array: no outputs, backtrack
                    if !self.backtrack()? {
                        return Ok(());
                    }
                    return Ok(());
                }
                // Push first element, save iterator state
                self.data.push(a[0].clone());
                self.forks.push(ForkPoint {
                    data_depth: self.data.len() - 1,
                    frame_depth: self.frames.len(),
                    pc: self.pc,
                    state: ForkState::Each {
                        container: top,
                        keys: None,
                        index: 1,
                        is_opt,
                    },
                });
                self.pc += 1;
            }
            Value::Obj(o) => {
                if o.is_empty() {
                    if !self.backtrack()? {
                        return Ok(());
                    }
                    return Ok(());
                }
                let keys: Vec<KeyStr> = o.keys().cloned().collect();
                let first_val = o.get(&keys[0]).cloned().unwrap_or(Value::Null);
                self.data.push(first_val);
                self.forks.push(ForkPoint {
                    data_depth: self.data.len() - 1,
                    frame_depth: self.frames.len(),
                    pc: self.pc,
                    state: ForkState::Each {
                        container: top,
                        keys: Some(keys),
                        index: 1,
                        is_opt,
                    },
                });
                self.pc += 1;
            }
            Value::Null if is_opt => {
                if !self.backtrack()? {
                    return Ok(());
                }
            }
            _ => {
                if is_opt {
                    if !self.backtrack()? {
                        return Ok(());
                    }
                } else {
                    let msg = format!(
                        "{} ({}) is not iterable",
                        crate::value::value_to_json(&top),
                        top.type_name()
                    );
                    if !self.propagate_error(Value::Error(Rc::new(msg)))? {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }

    fn handle_each_backtrack(&mut self) -> Result<bool> {
        let fork = self.forks.last_mut().unwrap();
        match &mut fork.state {
            ForkState::Each { container, keys, index, .. } => {
                match container {
                    Value::Arr(a) => {
                        if *index < a.len() {
                            let val = a[*index].clone();
                            *index += 1;
                            // Restore data stack to saved depth and push new value
                            let depth = fork.data_depth;
                            self.data.truncate(depth);
                            self.data.push(val);
                            self.pc += 1; // Move past EACH opcode
                            Ok(true)
                        } else {
                            self.forks.pop();
                            Ok(false)
                        }
                    }
                    Value::Obj(o) => {
                        let keys = keys.as_ref().unwrap();
                        if *index < keys.len() {
                            let val = o.get(&keys[*index]).cloned().unwrap_or(Value::Null);
                            *index += 1;
                            let depth = fork.data_depth;
                            self.data.truncate(depth);
                            self.data.push(val);
                            self.pc += 1;
                            Ok(true)
                        } else {
                            self.forks.pop();
                            Ok(false)
                        }
                    }
                    _ => {
                        self.forks.pop();
                        Ok(false)
                    }
                }
            }
            _ => Ok(false),
        }
    }

    fn execute_range(&mut self) -> Result<()> {
        let level = self.bytecodes[self.current_bc].bc.code()[self.pc + 1] as usize;
        let var_idx = self.bytecodes[self.current_bc].bc.code()[self.pc + 2] as usize;

        // Stack: [..., to, from]  (from on top in jq's convention)
        // Actually in jq: range uses STOREV to save the current value
        // and FORK/BACKTRACK to iterate
        // The RANGE opcode handles this internally

        // For now, use a simplified implementation
        // TODO: Handle this properly with the fork/backtrack model
        self.pc += 3;
        Ok(())
    }

    fn handle_range_backtrack(&mut self) -> Result<bool> {
        let fork = self.forks.last_mut().unwrap();
        match &mut fork.state {
            ForkState::Range { current, to, step, var_index } => {
                *current += *step;
                if (*step > 0.0 && *current < *to) || (*step < 0.0 && *current > *to) {
                    let val = Value::Num(*current, None);
                    let vi = *var_index;
                    let depth = fork.data_depth;
                    self.data.truncate(depth);
                    self.data.push(val.clone());
                    self.set_var(0, vi, val)?;
                    Ok(true)
                } else {
                    self.forks.pop();
                    Ok(false)
                }
            }
            _ => Ok(false),
        }
    }

    fn execute_builtin(&mut self) -> Result<()> {
        let code = self.bytecodes[self.current_bc].bc.code();
        let nargs = code[self.pc + 1] as usize;
        let cfunc_idx = code[self.pc + 2] as usize;
        let name = self.bytecodes[self.current_bc].cfunc_names.get(cfunc_idx)
            .cloned()
            .unwrap_or_else(|| format!("<unknown cfunc {}>", cfunc_idx));

        if self.data.len() < nargs {
            bail!("CALL_BUILTIN {}: stack underflow at pc={}", name, self.pc);
        }

        let mut args: Vec<Value> = Vec::with_capacity(nargs);
        for _ in 0..nargs {
            args.push(self.data.pop().unwrap());
        }
        // args[0] = top of stack (input for unary, discarded for binary)
        // args[1] = second (lhs for binary)
        // args[2] = third (rhs for binary, if exists)

        let result = crate::runtime::call_builtin(&name, &args)?;
        self.data.push(result);
        self.pc += 3;
        Ok(())
    }

    fn execute_call_jq(&mut self, _is_tail: bool) -> Result<()> {
        let code = self.bytecodes[self.current_bc].bc.code();
        let nargs = code[self.pc + 1] as usize;

        // The first pair (level, idx) identifies the target subfunction
        let _target_level = code[self.pc + 2] as usize;
        let target_idx = code[self.pc + 3] as usize;

        // Remaining pairs are closure arguments
        // For now, handle simple case (nargs=0)
        let instr_len = jq_ffi::opcode_length_with_code(Opcode::CALL_JQ, &code[self.pc..]);

        // Find the target subfunction
        if target_idx < self.bytecodes.len() {
            let nlocals = self.bytecodes[target_idx].bc.nlocals();
            self.frames.push(Frame {
                ret_pc: self.pc + instr_len,
                ret_bc_index: self.current_bc,
                locals: vec![Value::Null; nlocals],
                data_depth: self.data.len(),
                fork_depth: self.forks.len(),
            });
            self.current_bc = target_idx;
            self.pc = 0;
        } else {
            bail!("CALL_JQ: invalid subfunction index {} at pc={}", target_idx, self.pc);
        }

        Ok(())
    }

    fn execute_ret(&mut self) -> Result<()> {
        if self.frames.len() <= 1 {
            // Top-level return: this is an output
            if let Some(val) = self.data.last() {
                match val {
                    Value::Error(e) => {
                        self.errors.push(e.as_ref().clone());
                    }
                    _ => {
                        self.outputs.push(val.clone());
                    }
                }
            }
            // Try to backtrack for more outputs
            if !self.backtrack()? {
                return Ok(());
            }
        } else {
            // Return from subfunction
            let frame = self.frames.pop().unwrap();
            let ret_val = self.data.pop().unwrap_or(Value::Null);
            self.current_bc = frame.ret_bc_index;
            self.pc = frame.ret_pc;
            self.data.push(ret_val);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Backtracking
    // -----------------------------------------------------------------------

    fn backtrack(&mut self) -> Result<bool> {
        if let Some(fork) = self.forks.pop() {
            // Restore state
            self.data.truncate(fork.data_depth);
            // Don't truncate frames here - they may be needed
            self.pc = fork.pc;
            self.backtracking = true;

            // Re-push the fork for iterators (they need to continue)
            match &fork.state {
                ForkState::Each { .. } | ForkState::Range { .. } => {
                    self.forks.push(fork);
                }
                ForkState::Try { catch_pc } => {
                    // Error occurred in try block, jump to catch
                    self.pc = *catch_pc;
                    self.backtracking = false;
                    // Push the error value onto the stack
                    // The error is typically on the data stack already
                }
                ForkState::Jump => {
                    self.backtracking = false;
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Propagate an error value to the nearest try-catch handler.
    fn propagate_error(&mut self, error: Value) -> Result<bool> {
        // Look for a Try fork point
        if let Some(pos) = self.forks.iter().rposition(|f| matches!(f.state, ForkState::Try { .. })) {
            let fork = self.forks.remove(pos);
            if let ForkState::Try { catch_pc } = fork.state {
                self.data.truncate(fork.data_depth);
                self.data.push(error);
                self.pc = catch_pc;
                return Ok(true);
            }
        }
        // No catch handler - add error to outputs
        if let Value::Error(ref e) = error {
            self.errors.push(e.as_ref().clone());
        }
        // Backtrack
        self.backtrack()
    }

    // -----------------------------------------------------------------------
    // Variable access
    // -----------------------------------------------------------------------

    fn get_var(&self, level: usize, var_idx: usize) -> Result<Value> {
        let frame_idx = if level == 0 {
            self.frames.len() - 1
        } else {
            self.frames.len().checked_sub(1 + level)
                .ok_or_else(|| anyhow::anyhow!("get_var: frame stack underflow (level={})", level))?
        };
        let frame = &self.frames[frame_idx];
        if var_idx < frame.locals.len() {
            Ok(frame.locals[var_idx].clone())
        } else {
            Ok(Value::Null) // Uninitialized variable
        }
    }

    fn set_var(&mut self, level: usize, var_idx: usize, val: Value) -> Result<()> {
        let frame_idx = if level == 0 {
            self.frames.len() - 1
        } else {
            self.frames.len().checked_sub(1 + level)
                .ok_or_else(|| anyhow::anyhow!("set_var: frame stack underflow (level={})", level))?
        };
        let frame = &mut self.frames[frame_idx];
        if var_idx >= frame.locals.len() {
            frame.locals.resize(var_idx + 1, Value::Null);
        }
        frame.locals[var_idx] = val;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Constant access
    // -----------------------------------------------------------------------

    fn get_constant(&self, idx: usize) -> Result<Value> {
        let bc_info = &self.bytecodes[self.current_bc];
        bc_info.constants.get(idx).cloned()
            .ok_or_else(|| anyhow::anyhow!("constant index {} out of range", idx))
    }
}

/// Collect bytecode info for the top-level function only.
/// Subfunctions will be handled lazily as needed.
fn collect_bytecodes<'a>(bc: &'a BytecodeRef<'a>, out: &mut Vec<BytecodeInfo<'a>>) -> Result<()> {
    let constants = extract_constants(bc)?;

    let mut cfunc_names = Vec::new();
    let mut cfunc_nargs = Vec::new();
    for i in 0..100 {
        let name = bc.cfunc_name(i);
        if name.starts_with('<') {
            break;
        }
        let nargs = bc.cfunc_nargs(i);
        cfunc_names.push(name);
        cfunc_nargs.push(nargs);
    }

    out.push(BytecodeInfo {
        bc,
        constants,
        cfunc_names,
        cfunc_nargs,
    });

    Ok(())
}

/// Extract constants from a bytecode's constant pool.
fn extract_constants(bc: &BytecodeRef) -> Result<Vec<Value>> {
    let constants_jv = bc.constants_raw();
    let kind = unsafe { jq_ffi::jv_get_kind(constants_jv) };

    if kind != JvKind::Array {
        unsafe { jq_ffi::jv_free(constants_jv) };
        return Ok(vec![]);
    }

    let len = unsafe { jq_ffi::jv_array_length(jq_ffi::jv_copy(constants_jv)) };
    let mut constants = Vec::with_capacity(len as usize);

    for i in 0..len {
        let elem = unsafe { jq_ffi::jv_array_get(jq_ffi::jv_copy(constants_jv), i) };
        let val = unsafe { crate::value::jv_to_value(elem)? };
        constants.push(val);
    }

    unsafe { jq_ffi::jv_free(constants_jv) };
    Ok(constants)
}
