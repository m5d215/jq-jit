//! Bytecode → CPS IR translation.
//!
//! This module translates jq's stack-based bytecode into the tree-structured
//! [`Expr`] representation defined in [`crate::cps_ir`].
//!
//! The translation works by simulating the bytecode's stack machine, but instead
//! of pushing actual `jv` values, we push `Expr` nodes.  At the end of execution
//! (RET), the stack top is the resulting expression tree.
//!
//! # Supported opcodes
//!
//! - `TOP`: push `Expr::Input`
//! - `PUSHK_UNDER`: insert constant below stack top
//! - `LOADK`: replace stack top with constant
//! - `DUP`: duplicate stack top
//! - `POP`: discard stack top
//! - `SUBEXP_BEGIN`: duplicate top (copy below, original on top)
//! - `SUBEXP_END`: swap top two elements
//! - `CALL_BUILTIN` (nargs=3): binary operations (arithmetic + comparison)
//! - `CALL_BUILTIN` (nargs=1): unary builtins (length, type, tostring, tonumber, keys)
//! - `INDEX`: field/index access
//! - `JUMP_F` + `JUMP`: if-then-else control flow (Phase 3)
//! - `CALL_JQ` (nargs=0): subfunction inline expansion (Phase 3)
//! - `RET`: return stack top as result

use anyhow::{Context, Result, bail};

use crate::bytecode::BytecodeRef;
use crate::cps_ir::{BinOp, ClosureOp, Expr, Literal, UnaryOp};
use crate::jq_ffi::{self, Jv, JvKind, Opcode};

/// Offset multiplier for variable scoping.
///
/// jq bytecode uses (level, index) to identify variables across closure scopes.
/// `level=0` means current scope, `level=1` means parent scope, etc.
/// We flatten this 2D address into a single `var_index` by:
///   var_index = scope_depth * VAR_SCOPE_OFFSET + raw_index
/// where `scope_depth = parent_bcs.len()` (0 for top-level, 1 for first closure, etc.)
/// and `raw_index` is the local variable index from the bytecode.
///
/// For LOADV with level>0, the target scope_depth = current_depth - level, so:
///   var_index = (current_depth - level) * VAR_SCOPE_OFFSET + raw_index
const VAR_SCOPE_OFFSET: u16 = 1000;

/// Compute a scoped var_index from scope_depth and raw_index.
fn scoped_var(scope_depth: usize, raw_index: u16) -> u16 {
    (scope_depth as u16) * VAR_SCOPE_OFFSET + raw_index
}

/// Translate a bytecode function into a CPS IR expression.
///
/// Returns the `Expr` tree representing the filter's computation.
pub fn bytecode_to_ir(bc: &BytecodeRef) -> Result<Expr> {
    let code = bc.code();
    let mut stack: Vec<Expr> = Vec::new();
    let mut pc: usize = 0;

    compile_range(bc, code, &mut pc, code.len(), &mut stack, true, &[], &[])
}

/// The result of processing a bytecode range.
enum RangeResult {
    /// A RET instruction was hit — the expression is the final result.
    Ret(Expr),
    /// A JUMP instruction was hit — the expression is the branch result,
    /// and the usize is the JUMP target offset.
    Jump(Expr, usize),
    /// Reached the end of the specified range — the expression is the stack top.
    End(Expr),
}

/// Compile a range of bytecode [pc .. end_pc) into an IR expression.
///
/// This is the core recursive compiler.  For the top-level call, `end_pc` is the
/// full bytecode length.  For if-then-else branches, it's the branch boundary.
///
/// `is_toplevel` indicates whether RET should be expected (true for main, false
/// for branches where JUMP or range-end terminates).
fn compile_range(
    bc: &BytecodeRef,
    code: &[u16],
    pc: &mut usize,
    end_pc: usize,
    stack: &mut Vec<Expr>,
    is_toplevel: bool,
    closures: &[Expr],
    parent_bcs: &[&BytecodeRef],
) -> Result<Expr> {
    match compile_range_inner(bc, code, pc, end_pc, stack, closures, parent_bcs)? {
        RangeResult::Ret(expr) => Ok(expr),
        RangeResult::Jump(expr, _target) => {
            if is_toplevel {
                bail!(
                    "unexpected JUMP at pc={} in top-level bytecode",
                    *pc
                );
            }
            Ok(expr)
        }
        RangeResult::End(expr) => {
            if is_toplevel {
                bail!(
                    "bytecode ended without RET (pc={}, stack depth={})",
                    *pc,
                    stack.len()
                );
            }
            Ok(expr)
        }
    }
}

/// When `caller_input` is a generator (currently Recurse), and we want to apply
/// an expression `continuation` to each output, we fold the continuation into the
/// generator's body instead of embedding the generator inside a scalar expression.
///
/// For Recurse { input_expr, body: Input } piped through continuation f:
///   → Recurse { input_expr, body: f }
///
/// For Recurse { input_expr, body: existing_body } piped through continuation f:
///   → Recurse { input_expr, body: substitute_input(f, existing_body) }
///
/// For non-generators, falls back to the standard substitute_input.
fn pipe_through(continuation: &Expr, caller_input: &Expr) -> Expr {
    match caller_input {
        Expr::Recurse { input_expr, body } => {
            // Apply continuation to each output of the recursion.
            // The continuation has Input references that should bind to each recursed value.
            let new_body = if matches!(body.as_ref(), Expr::Input) {
                // body is identity: continuation directly becomes the new body
                continuation.clone()
            } else {
                // body already has content: pipe body through continuation
                crate::cps_ir::substitute_input(continuation, body)
            };
            Expr::Recurse {
                input_expr: input_expr.clone(),
                body: Box::new(new_body),
            }
        }
        // Comma generator: distribute continuation to each arm.
        // This ensures (1, 2) | "ok" becomes Comma("ok", "ok") → two outputs.
        Expr::Comma { left, right } => {
            Expr::Comma {
                left: Box::new(pipe_through(continuation, left)),
                right: Box::new(pipe_through(continuation, right)),
            }
        }
        // Each generator: fold continuation into the body.
        Expr::Each { input_expr, body } => {
            let new_body = Box::new(pipe_through(continuation, body));
            Expr::Each {
                input_expr: input_expr.clone(),
                body: new_body,
            }
        }
        // EachOpt generator: fold continuation into the body.
        Expr::EachOpt { input_expr, body } => {
            let new_body = Box::new(pipe_through(continuation, body));
            Expr::EachOpt {
                input_expr: input_expr.clone(),
                body: new_body,
            }
        }
        // IfThenElse generator: fold continuation into both branches.
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            Expr::IfThenElse {
                cond: cond.clone(),
                then_branch: Box::new(pipe_through(continuation, then_branch)),
                else_branch: Box::new(pipe_through(continuation, else_branch)),
            }
        }
        // TryCatch generator: fold continuation into both expressions.
        Expr::TryCatch { try_expr, catch_expr } => {
            Expr::TryCatch {
                try_expr: Box::new(pipe_through(continuation, try_expr)),
                catch_expr: Box::new(pipe_through(continuation, catch_expr)),
            }
        }
        // Empty generator: stays empty (0 outputs piped through anything = 0 outputs)
        Expr::Empty => Expr::Empty,
        // Non-generator: standard substitute_input
        _ => crate::cps_ir::substitute_input(continuation, caller_input),
    }
}

/// Inner compilation loop.  Returns how the range was terminated.
fn compile_range_inner(
    bc: &BytecodeRef,
    code: &[u16],
    pc: &mut usize,
    end_pc: usize,
    stack: &mut Vec<Expr>,
    closures: &[Expr],
    parent_bcs: &[&BytecodeRef],
) -> Result<RangeResult> {
    while *pc < end_pc {
        let raw_op = code[*pc];
        let op = Opcode::from_u16(raw_op)
            .ok_or_else(|| anyhow::anyhow!("unknown opcode {} at pc={}", raw_op, *pc))?;

        match op {
            Opcode::TOP => {
                // NOP in jq's VM, but for top-level bytecode it means
                // "the implicit input is on the stack".
                // We push Input to represent the filter's input value.
                stack.push(Expr::Input);
                *pc += 1;
            }

            Opcode::PUSHK_UNDER => {
                // Insert constant below stack top.
                // Encoding: PUSHK_UNDER + const_idx (2 words)
                let const_idx = code.get(*pc + 1).copied().ok_or_else(|| {
                    anyhow::anyhow!("PUSHK_UNDER at pc={}: missing constant index", *pc)
                })? as usize;

                let lit = jv_constant_to_literal(bc, const_idx)?;
                let expr = Expr::Literal(lit);

                // Insert below top: [..., top] → [..., expr, top]
                if stack.is_empty() {
                    bail!(
                        "PUSHK_UNDER at pc={}: stack underflow (need at least 1 element)",
                        *pc
                    );
                }
                let top = stack.pop().unwrap();
                stack.push(expr);
                stack.push(top);

                *pc += 2; // opcode + const_idx
            }

            Opcode::LOADK => {
                // Replace stack top with constant.
                // Encoding: LOADK + const_idx (2 words)
                let const_idx = code.get(*pc + 1).copied().ok_or_else(|| {
                    anyhow::anyhow!("LOADK at pc={}: missing constant index", *pc)
                })? as usize;

                let lit = jv_constant_to_literal(bc, const_idx)?;

                if stack.is_empty() {
                    bail!(
                        "LOADK at pc={}: stack underflow (need at least 1 element)",
                        *pc
                    );
                }
                let old_top = stack.pop().unwrap();
                // When the stack top is a generator (e.g., Comma from FORK),
                // replacing it with a constant must preserve generator semantics:
                // each generator output should be replaced with the constant.
                // Use pipe_through to distribute the constant through the generator.
                let new_expr = pipe_through(&Expr::Literal(lit), &old_top);
                stack.push(new_expr);

                *pc += 2;
            }

            Opcode::DUP => {
                // Duplicate stack top.
                if stack.is_empty() {
                    bail!("DUP at pc={}: stack underflow", *pc);
                }
                let top = stack.last().unwrap().clone();
                stack.push(top);
                *pc += 1;
            }

            Opcode::POP => {
                // Discard stack top.
                if stack.is_empty() {
                    bail!("POP at pc={}: stack underflow", *pc);
                }
                stack.pop();
                *pc += 1;
            }

            Opcode::SUBEXP_BEGIN => {
                // From execute.c:
                //   pop(v), push(copy(v)), push(v)
                // Effect: [..., v] → [..., v_copy, v]
                // For IR purposes: clone the top expr node.
                if stack.is_empty() {
                    bail!("SUBEXP_BEGIN at pc={}: stack underflow", *pc);
                }
                let top = stack.pop().unwrap();
                stack.push(top.clone());
                stack.push(top);
                *pc += 1;
            }

            Opcode::SUBEXP_END => {
                // From execute.c:
                //   pop(a), pop(b), push(a), push(b)
                // Effect: swap top two elements.
                if stack.len() < 2 {
                    bail!(
                        "SUBEXP_END at pc={}: stack underflow (need 2, have {})",
                        *pc,
                        stack.len()
                    );
                }
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(a);
                stack.push(b);
                *pc += 1;
            }

            Opcode::CALL_BUILTIN => {
                // Encoding: CALL_BUILTIN + nargs + cfunc_idx (3 words)
                let nargs = code.get(*pc + 1).copied().ok_or_else(|| {
                    anyhow::anyhow!("CALL_BUILTIN at pc={}: missing nargs", *pc)
                })? as usize;
                let cfunc_idx = code.get(*pc + 2).copied().ok_or_else(|| {
                    anyhow::anyhow!("CALL_BUILTIN at pc={}: missing cfunc_idx", *pc)
                })? as usize;

                let name = bc.cfunc_name(cfunc_idx);

                if stack.len() < nargs {
                    bail!(
                        "CALL_BUILTIN {} at pc={}: stack underflow (need {}, have {})",
                        name,
                        *pc,
                        nargs,
                        stack.len()
                    );
                }

                // Pop nargs items from stack (top first).
                // For binary arithmetic builtins with nargs=3:
                //   in[0] = top (input, discarded by the C function)
                //   in[1] = second (lhs operand)
                //   in[2] = third (rhs operand)
                let mut args: Vec<Expr> = Vec::with_capacity(nargs);
                for _ in 0..nargs {
                    args.push(stack.pop().unwrap());
                }

                let result = translate_builtin(&name, &args, *pc)?;
                stack.push(result);

                *pc += 3; // opcode + nargs + cfunc_idx
            }

            Opcode::INDEX => {
                // From execute.c:
                //   t = stack_pop()  → top = target object
                //   k = stack_pop()  → second = key
                //   push(jv_get(t, k))  → target[key]
                if stack.len() < 2 {
                    bail!(
                        "INDEX at pc={}: stack underflow (need 2, have {})",
                        *pc,
                        stack.len()
                    );
                }
                let target = stack.pop().unwrap(); // top = object/target
                let key = stack.pop().unwrap(); // second = key
                stack.push(Expr::Index {
                    expr: Box::new(target),
                    key: Box::new(key),
                });
                *pc += 1;
            }

            Opcode::INDEX_OPT => {
                // Optional index access `.foo?` (Phase 8-2, Phase 11 fix).
                //
                // `.foo?` is semantically `try .foo catch empty`:
                // - On success: produce the value normally
                // - On error: suppress (empty), not null
                //
                // We desugar INDEX_OPT into TryCatch { Index, Empty }.
                if stack.len() < 2 {
                    bail!(
                        "INDEX_OPT at pc={}: stack underflow (need 2, have {})",
                        *pc,
                        stack.len()
                    );
                }
                let target = stack.pop().unwrap();
                let key = stack.pop().unwrap();
                stack.push(Expr::TryCatch {
                    try_expr: Box::new(Expr::Index {
                        expr: Box::new(target),
                        key: Box::new(key),
                    }),
                    catch_expr: Box::new(Expr::Empty),
                });
                *pc += 1;
            }

            Opcode::JUMP_F => {
                // if-then-else pattern.
                //
                // JUMP_F peeks the condition value (pops then pushes back),
                // so the stack is unchanged after JUMP_F.
                //
                // Two patterns exist:
                //
                // Pattern A (main filter, gen_cond):
                //   DUP                     ; save input
                //   gen_subexp(cond) + POP  ; evaluate condition
                //   JUMP_F else_offset      ; branch (value stays on stack)
                //   POP                     ; then: discard cond_result
                //   iftrue code
                //   JUMP end_offset
                //   [else_offset:]
                //   POP                     ; else: discard cond_result
                //   iffalse code
                //   [end_offset:]
                //   → stack at JUMP_F: [..., input_expr, cond_expr]
                //
                // Pattern B (subfunction, e.g. `not`):
                //   JUMP_F else_offset      ; input is the condition
                //   LOADK false
                //   JUMP end_offset
                //   LOADK true
                //   RET
                //   → stack at JUMP_F: [cond_expr] (cond = input)
                //
                let raw_imm = code.get(*pc + 1).copied().ok_or_else(|| {
                    anyhow::anyhow!("JUMP_F at pc={}: missing branch offset", *pc)
                })?;
                // Branch offset is relative: target = (pc + 2) + imm (signed 16-bit)
                let else_offset = ((*pc as i32 + 2) + (raw_imm as i16 as i32)) as usize;

                if stack.is_empty() {
                    bail!(
                        "JUMP_F at pc={}: stack underflow (empty stack)",
                        *pc
                    );
                }

                // Determine pattern based on stack depth.
                // JUMP_F peeks (doesn't pop), so the stack passed to branches
                // matches what we have now.
                let (cond_expr, branch_stack) = if stack.len() >= 2 {
                    // Pattern A: [..., input_expr, cond_expr]
                    let cond_expr = stack.pop().unwrap();
                    let input_expr = stack.pop().unwrap();
                    (cond_expr.clone(), vec![input_expr, cond_expr])
                } else {
                    // Pattern B: [cond_expr] where cond = input
                    let cond_expr = stack.pop().unwrap();
                    (cond_expr.clone(), vec![cond_expr])
                };

                let then_start = *pc + 2; // instruction after JUMP_F

                // Compile then branch: [then_start .. else_offset)
                {
                    let mut then_stack = branch_stack.clone();
                    let mut then_pc = then_start;
                    match compile_range_inner(bc, code, &mut then_pc, else_offset, &mut then_stack, closures, parent_bcs)? {
                        RangeResult::Jump(then_ir, end_offset) => {
                            // JUMP end found — record the end offset
                            // Compile else branch: [else_offset .. end_offset)
                            let else_ir = {
                                let mut else_stack = branch_stack;
                                let mut else_pc = else_offset;
                                compile_range(
                                    bc, code, &mut else_pc, end_offset,
                                    &mut else_stack, false, closures, parent_bcs,
                                )?
                            };

                            // Build IfThenElse and push onto the main stack
                            let if_expr = Expr::IfThenElse {
                                cond: Box::new(cond_expr),
                                then_branch: Box::new(then_ir),
                                else_branch: Box::new(else_ir),
                            };
                            stack.push(if_expr);

                            // Advance main PC past the entire if-then-else
                            *pc = end_offset;
                            continue;
                        }
                        other => {
                            bail!(
                                "JUMP_F at pc={}: then branch did not end with JUMP (got {})",
                                *pc,
                                match other {
                                    RangeResult::Ret(_) => "RET",
                                    RangeResult::End(_) => "End",
                                    RangeResult::Jump(_, _) => unreachable!(),
                                }
                            );
                        }
                    }
                }
            }

            Opcode::JUMP => {
                // Unconditional jump.  In the context of if-then-else, this marks
                // the end of the then branch.  Return the current stack top and
                // the jump target.
                let raw_imm = code.get(*pc + 1).copied().ok_or_else(|| {
                    anyhow::anyhow!("JUMP at pc={}: missing branch offset", *pc)
                })?;
                // Branch offset is relative: target = (pc + 2) + imm (signed 16-bit)
                // Same logic as JUMP_F: opcode_addr + 2 + imm.
                let target = ((*pc as i32 + 2) + (raw_imm as i16 as i32)) as usize;

                if stack.is_empty() {
                    bail!("JUMP at pc={}: stack is empty (expected branch result)", *pc);
                }
                let result = stack.pop().unwrap();
                *pc += 2;
                return Ok(RangeResult::Jump(result, target));
            }

            Opcode::RET => {
                // Return stack top as the result.
                if stack.is_empty() {
                    bail!("RET at pc={}: stack is empty", *pc);
                }
                let result = stack.pop().unwrap();
                *pc += 1;
                return Ok(RangeResult::Ret(result));
            }

            Opcode::TRY_BEGIN => {
                // try-catch pattern (Phase 3):
                //
                //   TRY_BEGIN catch_offset    ; error → catch
                //   <try body>
                //   TRY_END
                //   JUMP end_offset           ; success → skip catch
                //   [catch label:]
                //   POP                       ; discard error message (or use it)
                //   <catch body>
                //   [end label:]
                //   RET
                //
                let raw_imm = code.get(*pc + 1).copied().ok_or_else(|| {
                    anyhow::anyhow!("TRY_BEGIN at pc={}: missing branch offset", *pc)
                })?;
                // Branch offset: target = (pc + 2) + imm (signed 16-bit)
                let catch_offset = ((*pc as i32 + 2) + (raw_imm as i16 as i32)) as usize;

                // Compile try body: from after TRY_BEGIN to catch_offset.
                // The try body may contain TRY_END + JUMP end — we scan for TRY_END
                // and JUMP to find the end offset.
                //
                // Start by compiling [pc+2 .. catch_offset) which covers the try body
                // including TRY_END and JUMP.
                let try_start = *pc + 2;
                let mut try_stack = stack.clone();
                let mut try_pc = try_start;

                match compile_range_inner(bc, code, &mut try_pc, catch_offset, &mut try_stack, closures, parent_bcs)? {
                    RangeResult::Jump(try_ir, end_offset) => {
                        // JUMP found at end of try body → end_offset is where catch ends.
                        // Compile catch body: [catch_offset .. end_offset)
                        // The catch body starts with the error message on the stack.
                        // We use Expr::Input as a placeholder — in the common case (POP + LOADK)
                        // it gets discarded immediately.
                        let catch_ir = {
                            let mut catch_stack = vec![Expr::Input];
                            let mut catch_pc = catch_offset;
                            compile_range(
                                bc, code, &mut catch_pc, end_offset,
                                &mut catch_stack, false, closures, parent_bcs,
                            )?
                        };

                        let try_catch_expr = Expr::TryCatch {
                            try_expr: Box::new(try_ir),
                            catch_expr: Box::new(catch_ir),
                        };
                        stack.push(try_catch_expr);
                        *pc = end_offset;
                        continue;
                    }
                    RangeResult::Ret(try_ir) => {
                        // Try body ends with RET without JUMP — try-catch wrapping
                        // a complete expression. This means there might be no separate
                        // end_offset. The catch body runs until the next RET.
                        // For now, compile catch body up to the full code end.
                        let catch_ir = {
                            let mut catch_stack = vec![Expr::Input];
                            let mut catch_pc = catch_offset;
                            compile_range(
                                bc, code, &mut catch_pc, end_pc,
                                &mut catch_stack, true, closures, parent_bcs,
                            )?
                        };

                        let try_catch_expr = Expr::TryCatch {
                            try_expr: Box::new(try_ir),
                            catch_expr: Box::new(catch_ir),
                        };
                        stack.push(try_catch_expr);
                        // Advance PC past everything
                        *pc = end_pc;
                        return Ok(RangeResult::Ret(stack.pop().unwrap()));
                    }
                    RangeResult::End(_try_ir) => {
                        // Try body reached end of range without JUMP or RET.
                        // This shouldn't normally happen in well-formed try-catch,
                        // but handle gracefully.
                        bail!(
                            "TRY_BEGIN at pc={}: try body ended without JUMP or RET",
                            *pc
                        );
                    }
                }
            }

            Opcode::TRY_END => {
                // TRY_END is a no-op in our compilation model.
                // The actual error checking happens at the codegen level.
                *pc += 1;
            }

            Opcode::FORK => {
                let raw_imm = code.get(*pc + 1).copied().ok_or_else(|| {
                    anyhow::anyhow!("FORK at pc={}: missing branch offset", *pc)
                })?;
                // Branch offset: target = (pc + 2) + imm (signed 16-bit)
                let done_pc = ((*pc as i32 + 2) + (raw_imm as i16 as i32)) as usize;
                let left_start = *pc + 2;

                // Check for array constructor pattern (Phase 5-1):
                //   FORK done → <generator> → APPEND $acc → BACKTRACK
                //   done: LOADVN $acc → RET
                //
                // The generator can be EACH (.[] | expr), nested FORK (.,expr),
                // CALL_JQ (empty, select, etc.) — any generator pattern.
                //
                // Detection: APPEND (3 words) + BACKTRACK (1 word) at done_pc - 4.
                //
                // Also check for reduce pattern (Phase 4-5):
                //   FORK done → DUPN → EACH → STOREV $x → LOADVN $acc → <update> → STOREV $acc → BACKTRACK
                //   done: LOADVN $acc → RET
                let next_op = code.get(left_start).copied()
                    .and_then(Opcode::from_u16);

                // Check for APPEND pattern first (array constructor)
                let append_pc = if done_pc >= 4 { done_pc - 4 } else { 0 };
                let has_append = done_pc >= 4
                    && code.get(append_pc).copied().and_then(Opcode::from_u16) == Some(Opcode::APPEND)
                    && code.get(done_pc - 1).copied().and_then(Opcode::from_u16) == Some(Opcode::BACKTRACK);

                if has_append && next_op != Some(Opcode::DUPN) {
                    // Array constructor pattern: [expr]
                    //   FORK done → <generator> → APPEND $acc → BACKTRACK
                    //   done: LOADVN $acc → RET
                    //
                    // Extract acc_index from APPEND operand (with scope offset).
                    let acc_index = scoped_var(parent_bcs.len(), code[append_pc + 2]);

                    // Compile the generator: left_start .. append_pc
                    // The generator may contain EACH, nested FORK (comma), CALL_JQ, etc.
                    let mut gen_stack = stack.clone();
                    let mut gen_pc = left_start;
                    let generator_ir = compile_range(
                        bc, code, &mut gen_pc, append_pc,
                        &mut gen_stack, false, closures, parent_bcs,
                    ).with_context(|| format!(
                        "FORK at pc={}: failed to compile array constructor generator",
                        *pc
                    ))?;

                    // Build Collect node
                    let collect_expr = Expr::Collect {
                        generator: Box::new(generator_ir),
                        acc_index,
                    };

                    stack.clear();
                    stack.push(collect_expr);

                    // Advance past done: LOADVN $acc (3 words)
                    *pc = done_pc + 3;

                    // Check what follows: if RET, return immediately.
                    // If more instructions follow (e.g., `[expr] | f`),
                    // continue the loop to process them.
                    let after_loadvn_op = code.get(*pc).copied()
                        .and_then(Opcode::from_u16);
                    if after_loadvn_op == Some(Opcode::RET) {
                        *pc += 1;
                        return Ok(RangeResult::Ret(stack.pop().unwrap()));
                    }
                    // Otherwise, fall through to continue processing

                } else if next_op == Some(Opcode::DUPN) {
                    // Reduce pattern: FORK → DUPN → EACH → ...
                    let each_pc = left_start + 1;
                    let each_op = code.get(each_pc).copied()
                        .and_then(Opcode::from_u16);

                    if each_op != Some(Opcode::EACH) {
                        bail!(
                            "FORK at pc={}: DUPN not followed by EACH (got {:?})",
                            *pc, each_op
                        );
                    }

                    // STOREV $x: 3 words at each_pc + 1
                    let storev_x_pc = each_pc + 1;
                    let storev_x_op = code.get(storev_x_pc).copied()
                        .and_then(Opcode::from_u16);
                    if storev_x_op != Some(Opcode::STOREV) {
                        bail!(
                            "FORK at pc={}: reduce pattern expected STOREV after EACH (got {:?})",
                            *pc, storev_x_op
                        );
                    }
                    let var_index = scoped_var(parent_bcs.len(), code[storev_x_pc + 2]); // $x variable index

                    // LOADVN $acc: 3 words at storev_x_pc + 3
                    let loadvn_acc_pc = storev_x_pc + 3;
                    let loadvn_acc_op = code.get(loadvn_acc_pc).copied()
                        .and_then(Opcode::from_u16);
                    if loadvn_acc_op != Some(Opcode::LOADVN) {
                        bail!(
                            "FORK at pc={}: reduce pattern expected LOADVN after STOREV (got {:?})",
                            *pc, loadvn_acc_op
                        );
                    }
                    let raw_acc_index = code[loadvn_acc_pc + 2]; // $acc raw variable index
                    let acc_index = scoped_var(parent_bcs.len(), raw_acc_index); // $acc scoped variable index

                    // Update expression: from loadvn_acc_pc + 3 to (done_pc - 4)
                    // The end of the update is: STOREV $acc (3 words) + BACKTRACK (1 word) = 4 words before done_pc
                    let storev_acc_pc = done_pc - 4;
                    let backtrack_pc = done_pc - 1;

                    // Verify STOREV $acc at storev_acc_pc (compare against raw bytecode index)
                    let storev_acc_op = code.get(storev_acc_pc).copied()
                        .and_then(Opcode::from_u16);
                    if storev_acc_op != Some(Opcode::STOREV) || code[storev_acc_pc + 2] != raw_acc_index {
                        bail!(
                            "FORK at pc={}: reduce pattern expected STOREV $acc at pc={} (got {:?})",
                            *pc, storev_acc_pc, storev_acc_op
                        );
                    }

                    // Verify BACKTRACK at backtrack_pc
                    let bt_op = code.get(backtrack_pc).copied()
                        .and_then(Opcode::from_u16);
                    if bt_op != Some(Opcode::BACKTRACK) {
                        bail!(
                            "FORK at pc={}: reduce pattern expected BACKTRACK at pc={} (got {:?})",
                            *pc, backtrack_pc, bt_op
                        );
                    }

                    // Compile the update expression.
                    // The update takes acc as input (.) and can reference $x via LoadVar.
                    let update_start = loadvn_acc_pc + 3;
                    let mut update_stack: Vec<Expr> = vec![Expr::Input]; // acc value as input
                    let mut update_pc = update_start;
                    let update_ir = compile_range(
                        bc, code, &mut update_pc, storev_acc_pc,
                        &mut update_stack, false, closures, parent_bcs,
                    ).with_context(|| format!(
                        "FORK at pc={}: failed to compile reduce update expression",
                        *pc
                    ))?;

                    // Get the source (container) from the stack.
                    // Before FORK, the stack has the container (from DUP before STOREV acc).
                    if stack.is_empty() {
                        bail!("FORK at pc={}: reduce pattern needs container on stack", *pc);
                    }
                    let source_expr = stack.pop().unwrap();

                    // The init value is provided by the enclosing LetBinding (STOREV $acc).
                    // We need to extract it — but at this point it's already been consumed
                    // by the STOREV handler above us.  Instead, we use LoadVar to reference
                    // the accumulator's current value as the init.
                    //
                    // Actually, the Reduce node needs init to be compiled from the
                    // LOADK instruction that preceded STOREV $acc.  Since STOREV $acc
                    // wraps the rest of the bytecode as a LetBinding, the init value
                    // is the LetBinding's `value` field.  We don't have access to it here.
                    //
                    // Solution: Use Expr::LoadVar { var_index: acc_index } as init.
                    // The LetBinding above us has already set the initial value.
                    // Reduce codegen will read the initial acc from the var_slot.
                    let init_expr = Expr::LoadVar { var_index: acc_index };

                    // Build Reduce node
                    let reduce_expr = Expr::Reduce {
                        source: Box::new(source_expr),
                        init: Box::new(init_expr),
                        var_index,
                        acc_index,
                        update: Box::new(update_ir),
                    };

                    // After reduce, the done label loads the final acc value.
                    // done: LOADVN $acc (3 words)
                    // The Reduce node itself represents the final value, so we push it
                    // and advance PC past the LOADVN.
                    stack.push(reduce_expr);

                    // Advance past done: LOADVN (3 words)
                    *pc = done_pc + 3;

                    // Check what follows: if RET, return immediately.
                    // If more instructions follow (e.g., `reduce ... | . * 2`),
                    // continue the loop to process them.
                    let after_loadvn_op = code.get(*pc).copied()
                        .and_then(Opcode::from_u16);
                    if after_loadvn_op == Some(Opcode::RET) {
                        *pc += 1;
                        return Ok(RangeResult::Ret(stack.pop().unwrap()));
                    }
                    // Otherwise, fall through to continue processing remaining instructions
                    // (the reduce result is on the stack as input to the next operation)

                } else if code.get(done_pc).copied().and_then(Opcode::from_u16) == Some(Opcode::DUP)
                    && code.get(done_pc + 1).copied().and_then(Opcode::from_u16) == Some(Opcode::LOADV)
                {
                    // Alternative operator `//` pattern (Phase 5-2):
                    //
                    // Before FORK (already processed by STOREV handler):
                    //   DUP → LOADK false → STOREV $found → (we are inside LetBinding body)
                    //
                    // FORK body (left_start to done_pc):
                    //   [optional primary computation] → JUMP_F (done_pc-1) →
                    //   DUP → LOADK true → STOREV $found → JUMP end → BACKTRACK
                    //
                    // After FORK done target (done_pc):
                    //   DUP → LOADV $found → JUMP_F fallback_start →
                    //   BACKTRACK → POP → <fallback expr> → end: RET
                    //
                    // Fixed layout:
                    //   JUMP_F at done_pc - 11
                    //   JUMP end at done_pc - 3
                    //   fallback starts at done_pc + 8

                    let jump_f_pc = done_pc - 11;
                    let jump_end_pc = done_pc - 3;

                    // Calculate end_offset from the JUMP instruction
                    let jump_end_imm = code.get(jump_end_pc + 1).copied().ok_or_else(|| {
                        anyhow::anyhow!("FORK at pc={}: alternative pattern missing JUMP imm at pc={}", *pc, jump_end_pc)
                    })?;
                    let end_offset = ((jump_end_pc as i32 + 2) + (jump_end_imm as i16 as i32)) as usize;

                    // Compile primary expression: from left_start to jump_f_pc
                    let primary_ir = if left_start < jump_f_pc {
                        let mut primary_stack = stack.clone();
                        let mut primary_pc = left_start;
                        compile_range(
                            bc, code, &mut primary_pc, jump_f_pc,
                            &mut primary_stack, false, closures, parent_bcs,
                        ).with_context(|| format!(
                            "FORK at pc={}: failed to compile alternative primary expression",
                            *pc
                        ))?
                    } else {
                        // Primary is identity (.) — stack top
                        if stack.is_empty() {
                            bail!("FORK at pc={}: alternative pattern needs value on stack", *pc);
                        }
                        stack.last().unwrap().clone()
                    };

                    // Compile fallback expression: from done_pc + 8 to end_offset
                    let fallback_start = done_pc + 8;
                    let mut fallback_stack: Vec<Expr> = vec![stack.last().cloned().unwrap_or(Expr::Input)];
                    let mut fallback_pc = fallback_start;
                    let fallback_ir = compile_range(
                        bc, code, &mut fallback_pc, end_offset,
                        &mut fallback_stack, false, closures, parent_bcs,
                    ).with_context(|| format!(
                        "FORK at pc={}: failed to compile alternative fallback expression",
                        *pc
                    ))?;

                    let alt_expr = Expr::Alternative {
                        primary: Box::new(primary_ir),
                        fallback: Box::new(fallback_ir),
                    };

                    stack.clear();
                    stack.push(alt_expr);

                    *pc = end_offset;

                    // Check if RET follows
                    let after_op = code.get(*pc).copied().and_then(Opcode::from_u16);
                    if after_op == Some(Opcode::RET) {
                        *pc += 1;
                        return Ok(RangeResult::Ret(stack.pop().unwrap()));
                    }
                    // Otherwise continue processing

                } else {
                    // Comma pattern (Phase 4-2):
                    //
                    //   FORK right_offset     ; fork point (backtrack → right)
                    //   <left body>
                    //   JUMP end_offset       ; left done → skip right
                    //   [right_offset:]
                    //   <right body>
                    //   [end_offset:]
                    //   (continues or RET)
                    //
                    let mut left_stack = stack.clone();
                    let mut left_pc = left_start;

                    match compile_range_inner(bc, code, &mut left_pc, done_pc, &mut left_stack, closures, parent_bcs)? {
                        RangeResult::Jump(left_ir, end_offset) => {
                            let mut right_stack = stack.clone();
                            let mut right_pc = done_pc;
                            let right_ir = compile_range(
                                bc, code, &mut right_pc, end_offset,
                                &mut right_stack, false, closures, parent_bcs,
                            )?;

                            let comma_expr = Expr::Comma {
                                left: Box::new(left_ir),
                                right: Box::new(right_ir),
                            };

                            stack.clear();
                            stack.push(comma_expr);

                            *pc = end_offset;
                            continue;
                        }
                        RangeResult::Ret(_left_ir) => {
                            bail!(
                                "FORK at pc={}: left body ended with RET (expected JUMP for comma pattern)",
                                *pc
                            );
                        }
                        RangeResult::End(_left_ir) => {
                            bail!(
                                "FORK at pc={}: left body reached end of range without JUMP",
                                *pc
                            );
                        }
                    }
                }
            }

            Opcode::EACH => {
                // .[] iterator (Phase 4-3).
                //
                // EACH pops the array/object from the stack and generates one
                // output per element.  In the jq VM this uses backtracking;
                // in our CPS model we compile the rest of the bytecode (after
                // EACH) as the "body" that runs for each element.
                //
                // Bytecode patterns:
                //   .[]              → TOP EACH RET
                //   .[] | f          → TOP EACH <f opcodes> RET
                //   .[], .[]         → TOP FORK EACH JUMP EACH RET  (FORK handles comma)
                //   foreach          → ... DUP EACH STOREV $x LOADVN $acc <update> DUP STOREV $acc RET
                //
                // Algorithm:
                //   1. Pop input_expr from stack (the container to iterate)
                //   2. Check for foreach pattern (Phase 4-5)
                //   3. Otherwise compile the remaining instructions as "body"
                //      with initial stack = [Expr::Input] (each element = Input)
                //   4. The body's RangeResult determines how we terminate
                //
                if stack.is_empty() {
                    bail!("EACH at pc={}: stack underflow", *pc);
                }
                let input_expr = stack.pop().unwrap();
                let each_pc = *pc;
                *pc += 1; // advance past EACH opcode

                // Check for foreach pattern (Phase 4-5):
                //   EACH → STOREV $x → LOADVN $acc → <update> → DUP → STOREV $acc → RET
                //
                // foreach has no FORK.  The pattern is:
                //   DUP → EACH → foreach body
                //
                // We detect it by checking if the instructions after EACH match
                // STOREV + LOADVN and the body ends with DUP + STOREV + RET.
                let maybe_foreach = {
                    let after_each = *pc;
                    let op1 = code.get(after_each).copied().and_then(Opcode::from_u16);
                    let op2 = code.get(after_each + 3).copied().and_then(Opcode::from_u16);
                    if op1 == Some(Opcode::STOREV) && op2 == Some(Opcode::LOADVN) {
                        let raw_var_index = code[after_each + 2]; // $x (raw)
                        let raw_acc_index = code[after_each + 5]; // $acc (raw)

                        // Check if the body ends with DUP + STOREV $acc + RET
                        // Scan backwards from end_pc:
                        //   end_pc - 1: RET (1 word)
                        //   end_pc - 4: STOREV [lv] [$acc] (3 words)
                        //   end_pc - 5: DUP (1 word)
                        let ret_pc = end_pc - 1;
                        let storev_acc_pc = end_pc - 4;
                        let dup_pc = end_pc - 5;

                        let is_ret = code.get(ret_pc).copied().and_then(Opcode::from_u16) == Some(Opcode::RET);
                        let is_storev_acc = code.get(storev_acc_pc).copied().and_then(Opcode::from_u16) == Some(Opcode::STOREV)
                            && code.get(storev_acc_pc + 2).copied() == Some(raw_acc_index);
                        let is_dup = code.get(dup_pc).copied().and_then(Opcode::from_u16) == Some(Opcode::DUP);

                        if is_ret && is_storev_acc && is_dup {
                            // Apply scope offset to var indices
                            let var_index = scoped_var(parent_bcs.len(), raw_var_index);
                            let acc_index = scoped_var(parent_bcs.len(), raw_acc_index);
                            Some((var_index, acc_index, dup_pc))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                if let Some((var_index, acc_index, dup_pc)) = maybe_foreach {
                    // Foreach pattern detected.
                    //
                    // Compile the update expression: from LOADVN $acc + 3 to dup_pc
                    // The update takes acc as input (.) and can reference $x via LoadVar.
                    let update_start = *pc + 6; // after STOREV $x (3) + LOADVN $acc (3)
                    let mut update_stack: Vec<Expr> = vec![Expr::Input]; // acc value as input
                    let mut update_pc = update_start;
                    let update_ir = compile_range(
                        bc, code, &mut update_pc, dup_pc,
                        &mut update_stack, false, closures, parent_bcs,
                    ).with_context(|| format!(
                        "EACH at pc={}: failed to compile foreach update expression",
                        each_pc
                    ))?;

                    // Init is LoadVar (the value set by enclosing LetBinding STOREV $acc)
                    let init_expr = Expr::LoadVar { var_index: acc_index };

                    let foreach_expr = Expr::Foreach {
                        source: Box::new(input_expr),
                        init: Box::new(init_expr),
                        var_index,
                        acc_index,
                        update: Box::new(update_ir),
                    };

                    // Foreach is a generator — push and return as Ret
                    // (end_pc covers the entire bytecode including the final RET)
                    stack.push(foreach_expr);
                    *pc = end_pc;
                    return Ok(RangeResult::Ret(stack.pop().unwrap()));
                }

                // Normal EACH: compile the rest of the bytecode as the body.
                // Each element will be passed as Expr::Input.
                let mut body_stack: Vec<Expr> = vec![Expr::Input];
                let body_result = compile_range_inner(
                    bc, code, pc, end_pc, &mut body_stack, closures, parent_bcs,
                )?;

                let (body_expr, range_result_kind) = match body_result {
                    RangeResult::Ret(expr) => (expr, "ret"),
                    RangeResult::Jump(expr, target) => {
                        // JUMP found — this happens when EACH is inside a FORK
                        // (e.g., .[], .[] pattern).  Push the Each node and
                        // propagate the Jump result.
                        let each_node = Expr::Each {
                            input_expr: Box::new(input_expr),
                            body: Box::new(expr),
                        };
                        stack.push(each_node);
                        return Ok(RangeResult::Jump(stack.pop().unwrap(), target));
                    }
                    RangeResult::End(expr) => (expr, "end"),
                };

                let each_node = Expr::Each {
                    input_expr: Box::new(input_expr),
                    body: Box::new(body_expr),
                };
                stack.push(each_node);

                if range_result_kind == "ret" {
                    return Ok(RangeResult::Ret(stack.pop().unwrap()));
                } else {
                    return Ok(RangeResult::End(stack.pop().unwrap()));
                }
            }

            Opcode::EACH_OPT => {
                // .[]? optional iterator (Phase 8-3).
                //
                // Same as EACH but produces zero outputs for non-iterable inputs
                // instead of erroring. We reuse the EACH compilation logic but
                // produce EachOpt nodes instead.
                if stack.is_empty() {
                    bail!("EACH_OPT at pc={}: stack underflow", *pc);
                }
                let input_expr = stack.pop().unwrap();
                *pc += 1; // advance past EACH_OPT opcode

                // Compile the rest of the bytecode as the body.
                let mut body_stack: Vec<Expr> = vec![Expr::Input];
                let body_result = compile_range_inner(
                    bc, code, pc, end_pc, &mut body_stack, closures, parent_bcs,
                )?;

                let (body_expr, range_result_kind) = match body_result {
                    RangeResult::Ret(expr) => (expr, "ret"),
                    RangeResult::Jump(expr, target) => {
                        let each_node = Expr::EachOpt {
                            input_expr: Box::new(input_expr),
                            body: Box::new(expr),
                        };
                        stack.push(each_node);
                        return Ok(RangeResult::Jump(stack.pop().unwrap(), target));
                    }
                    RangeResult::End(expr) => (expr, "end"),
                };

                let each_node = Expr::EachOpt {
                    input_expr: Box::new(input_expr),
                    body: Box::new(body_expr),
                };
                stack.push(each_node);

                if range_result_kind == "ret" {
                    return Ok(RangeResult::Ret(stack.pop().unwrap()));
                } else {
                    return Ok(RangeResult::End(stack.pop().unwrap()));
                }
            }

            Opcode::STOREV | Opcode::STOREVN => {
                // Variable binding (Phase 4-4, Phase 9-8: STOREVN is move-semantics store,
                // same as STOREV for our CPS model since we don't track slot nullification).
                //
                // STOREV [level] [index] (3 words)
                //
                // Pops the stack top and stores it in local variable slot [index].
                // The remainder of the bytecode is the body where the variable is in scope.
                //
                // Pattern:  ... DUP <expr> POP STOREV[l,i] <body using LOADV[l,i]> ...
                //
                // We compile the rest of the bytecode as the body, wrapping it in
                // LetBinding { var_index, value, body }.
                //
                let _level = code.get(*pc + 1).copied().unwrap_or(0);
                let raw_index = code.get(*pc + 2).copied().ok_or_else(|| {
                    anyhow::anyhow!("STOREV at pc={}: missing variable index", *pc)
                })?;
                // Apply scope offset: STOREV is always level=0 (current scope)
                let var_index = scoped_var(parent_bcs.len(), raw_index);

                if stack.is_empty() {
                    bail!("STOREV at pc={}: stack underflow", *pc);
                }
                let value_expr = stack.pop().unwrap();

                *pc += 3; // opcode + level + index

                // Compile the rest of the bytecode as the body.
                let body_result = compile_range_inner(bc, code, pc, end_pc, stack, closures, parent_bcs)?;

                let (body_expr, result_kind) = match body_result {
                    RangeResult::Ret(expr) => (expr, "ret"),
                    RangeResult::Jump(expr, target) => {
                        // Body ended with JUMP — wrap in LetBinding and propagate Jump
                        let let_expr = Expr::LetBinding {
                            var_index,
                            value: Box::new(value_expr),
                            body: Box::new(expr),
                        };
                        stack.push(let_expr);
                        return Ok(RangeResult::Jump(stack.pop().unwrap(), target));
                    }
                    RangeResult::End(expr) => (expr, "end"),
                };

                let let_expr = Expr::LetBinding {
                    var_index,
                    value: Box::new(value_expr),
                    body: Box::new(body_expr),
                };
                stack.push(let_expr);

                if result_kind == "ret" {
                    return Ok(RangeResult::Ret(stack.pop().unwrap()));
                } else {
                    return Ok(RangeResult::End(stack.pop().unwrap()));
                }
            }

            Opcode::LOADV => {
                // Variable reference (Phase 4-4):
                //
                // LOADV [level] [index] (3 words)
                //
                // Loads the value of local variable slot [index] onto the stack.
                // In the CPS IR, this becomes a LoadVar node.
                //
                // `level` indicates how many closure scopes up to look:
                //   level=0 → current scope, level=1 → parent scope, etc.
                //
                let level = code.get(*pc + 1).copied().unwrap_or(0) as usize;
                let raw_index = code.get(*pc + 2).copied().ok_or_else(|| {
                    anyhow::anyhow!("LOADV at pc={}: missing variable index", *pc)
                })?;
                // Compute target scope depth: current_depth - level
                let target_depth = parent_bcs.len().saturating_sub(level);
                let var_index = scoped_var(target_depth, raw_index);

                if stack.is_empty() {
                    bail!("LOADV at pc={}: stack underflow", *pc);
                }
                // LOADV replaces stack top (stack_in=1, stack_out=1)
                stack.pop();
                stack.push(Expr::LoadVar { var_index });

                *pc += 3;
            }

            Opcode::LOADVN => {
                // Move-semantics variable reference (Phase 4-4):
                //
                // LOADVN [level] [index] (3 words)
                //
                // Same as LOADV but nullifies the variable slot after reading.
                // For our CPS IR purposes, this is identical to LOADV since we
                // don't model the variable slot's post-read state.
                //
                let level = code.get(*pc + 1).copied().unwrap_or(0) as usize;
                let raw_index = code.get(*pc + 2).copied().ok_or_else(|| {
                    anyhow::anyhow!("LOADVN at pc={}: missing variable index", *pc)
                })?;
                let target_depth = parent_bcs.len().saturating_sub(level);
                let var_index = scoped_var(target_depth, raw_index);

                if stack.is_empty() {
                    bail!("LOADVN at pc={}: stack underflow", *pc);
                }
                stack.pop();
                stack.push(Expr::LoadVar { var_index });

                *pc += 3;
            }

            Opcode::INSERT => {
                // Object insert (Phase 8-4):
                //
                // INSERT has no operands (1 word).
                // Stack In=4, Stack Out=2:
                //   [..., objv, k, v, stktop] -> [..., updated_obj, stktop]
                //
                // From jq's execute.c:
                //   stktop = stack_pop()   // saved pipeline input
                //   v      = stack_pop()   // value to insert
                //   k      = stack_pop()   // key string
                //   objv   = stack_pop()   // object to insert into
                //   stack_push(object_set(objv, k, v))
                //   stack_push(stktop)     // restore saved input
                //
                if stack.len() < 4 {
                    bail!(
                        "INSERT at pc={}: stack underflow (need at least 4, have {})",
                        *pc,
                        stack.len()
                    );
                }
                let stktop = stack.pop().unwrap();
                let value_expr = stack.pop().unwrap();
                let key_expr = stack.pop().unwrap();
                let obj_expr = stack.pop().unwrap();

                stack.push(Expr::ObjectInsert {
                    obj: Box::new(obj_expr),
                    key: Box::new(key_expr),
                    value: Box::new(value_expr),
                });
                stack.push(stktop); // restore saved pipeline input

                *pc += 1; // INSERT is 1 word (no operands)
            }

            Opcode::BACKTRACK => {
                // BACKTRACK = `empty` in CPS model.
                // The expression produces zero outputs (callback is never called).
                //
                // In the jq VM, BACKTRACK pops back to the previous fork point.
                // In CPS, it simply means "don't call callback" = Empty.
                //
                // Push Empty onto the stack.  In some contexts (e.g., else branch
                // of select), BACKTRACK replaces the current output.
                if !stack.is_empty() {
                    stack.pop();
                }
                stack.push(Expr::Empty);
                *pc += 1;
            }

            Opcode::CALL_JQ | Opcode::TAIL_CALL_JQ => {
                // Subfunction call.
                //
                // Encoding: CALL_JQ + nargs + (nargs+1) pairs of (level, idx)
                //   - For nargs=0: CALL_JQ + 0 + level0 + idx0 → 4 words
                //   - For nargs=1: CALL_JQ + 1 + level0 + idx0 + level1 + idx1 → 6 words
                //   - (level0, idx0) = the subfunction to call
                //   - (level1, idx1) = first closure argument
                //
                // nargs=0: Inline-expand the subfunction (Phase 3: not, etc.)
                // nargs=1: select(f) pattern — inline as IfThenElse { cond: f(input), then: Input, else: Empty }
                //
                let nargs = code.get(*pc + 1).copied().ok_or_else(|| {
                    anyhow::anyhow!("CALL_JQ at pc={}: missing nargs", *pc)
                })? as usize;

                const ARG_NEWCLOSURE: usize = 0x1000;

                // Pop the input expression from the stack (CALL_JQ consumes the stack top)
                if stack.is_empty() {
                    bail!("CALL_JQ at pc={}: stack underflow", *pc);
                }
                let caller_input = stack.pop().unwrap();

                // Advance PC past the CALL_JQ instruction first
                let instr_len = jq_ffi::opcode_length_with_code(op, &code[*pc..]);

                if nargs == 0 {
                    // nargs=0: Simple subfunction inline expansion (Phase 3).
                    //
                    // If we have closure bindings and the index is within the closure
                    // range, resolve from closures instead of subfunctions.
                    // This supports CALL_JQ nargs=0 inside closure-bearing subfunctions
                    // (e.g., subfunction 0 of map calls its closure argument).
                    let level = code.get(*pc + 2).copied().unwrap_or(0) as usize;
                    let raw_idx = code.get(*pc + 3).copied().ok_or_else(|| {
                        anyhow::anyhow!("CALL_JQ at pc={}: missing function index", *pc)
                    })? as usize;
                    let fidx = raw_idx & !ARG_NEWCLOSURE;

                    if fidx < closures.len() {
                        // Closure reference: use the pre-compiled closure IR.
                        // substitute_input replaces Input with the caller's input.
                        let inlined = crate::cps_ir::substitute_input(&closures[fidx], &caller_input);
                        stack.push(inlined);
                    } else {
                        // Phase 8-5: Resolve target bytecode using level.
                        // level=0 → current bc's subfunctions
                        // level>0 → parent_bcs[len - level]'s subfunctions
                        let target_bc: &BytecodeRef = if level == 0 {
                            bc
                        } else if level <= parent_bcs.len() {
                            parent_bcs[parent_bcs.len() - level]
                        } else {
                            bail!(
                                "CALL_JQ at pc={}: level {} exceeds parent depth {} (nargs=0)",
                                *pc, level, parent_bcs.len()
                            );
                        };
                        let subfunctions = target_bc.subfunctions();
                        if fidx >= subfunctions.len() {
                            bail!(
                                "CALL_JQ at pc={}: subfunction index {} out of range (have {}, closures={}, level={})",
                                *pc, fidx, subfunctions.len(), closures.len(), level
                            );
                        }

                        // Check if this is a known jq-defined function that we can
                        // implement as a direct runtime call (Phase 5-3).
                        let sub_bc = &subfunctions[fidx];

                        // Phase 9-4: Recognize recurse/.. pattern
                        if sub_bc.debugname().as_deref() == Some("recurse") {
                            stack.push(Expr::Recurse {
                                input_expr: Box::new(caller_input),
                                body: Box::new(Expr::Input),
                            });
                            *pc += instr_len;
                            continue;
                        }

                        if let Some(unary_op) = recognize_jq_defined_function(sub_bc) {
                            let expr = Expr::UnaryOp {
                                op: unary_op,
                                operand: Box::new(caller_input),
                            };
                            stack.push(expr);
                        } else {
                        // Compile the subfunction to IR.
                        // When compiling a subfunction, current bc becomes a parent.
                        let mut new_parent_bcs: Vec<&BytecodeRef> = parent_bcs.to_vec();
                        new_parent_bcs.push(bc);
                        let sub_code = sub_bc.code();
                        let mut sub_pc: usize = 0;
                        let mut sub_stack: Vec<Expr> = vec![Expr::Input];
                        let sub_ir = compile_range(
                            sub_bc, sub_code, &mut sub_pc, sub_code.len(),
                            &mut sub_stack, true, &[], &new_parent_bcs,
                        ).with_context(|| format!(
                            "CALL_JQ at pc={}: failed to compile subfunction {}",
                            *pc, fidx
                        ))?;

                        // Inline-expand: replace Input with caller's input
                        // Use pipe_through to properly handle generator inputs (e.g., Recurse)
                        let inlined = pipe_through(&sub_ir, &caller_input);
                        stack.push(inlined);
                        }
                    }
                } else if nargs >= 1 {
                    // nargs>=1: CALL_JQ with one or more closure arguments.
                    //
                    // Phase 8-5: Generalized to handle nargs>=1 with level>0 support.
                    //
                    // Encoding: CALL_JQ + nargs + (nargs+1) pairs of (level, idx)
                    //   - (level0, idx0) = the subfunction to call (fn)
                    //   - (level1, idx1) = first closure argument
                    //   - (level2, idx2) = second closure argument (if nargs>=2)
                    //
                    // Extract the function index and level.
                    let fn_level = code.get(*pc + 2).copied().unwrap_or(0) as usize;
                    let fn_raw_idx = code.get(*pc + 3).copied().ok_or_else(|| {
                        anyhow::anyhow!("CALL_JQ at pc={}: missing function index", *pc)
                    })? as usize;
                    let fn_fidx = fn_raw_idx & !ARG_NEWCLOSURE;

                    // Resolve fn bytecode using level
                    let fn_target_bc: &BytecodeRef = if fn_level == 0 {
                        bc
                    } else if fn_level <= parent_bcs.len() {
                        parent_bcs[parent_bcs.len() - fn_level]
                    } else {
                        bail!(
                            "CALL_JQ at pc={}: fn level {} exceeds parent depth {} (nargs={})",
                            *pc, fn_level, parent_bcs.len(), nargs
                        );
                    };
                    let fn_subfunctions = fn_target_bc.subfunctions();
                    if fn_fidx >= fn_subfunctions.len() {
                        bail!(
                            "CALL_JQ at pc={}: function subfunction index {} out of range (have {}, level={})",
                            *pc, fn_fidx, fn_subfunctions.len(), fn_level
                        );
                    }
                    let fn_bc = &fn_subfunctions[fn_fidx];

                    // Extract and compile all closure arguments
                    let mut closure_irs: Vec<Expr> = Vec::with_capacity(nargs);
                    for arg_i in 0..nargs {
                        let offset = 4 + arg_i * 2;
                        let arg_level = code.get(*pc + offset).copied().unwrap_or(0) as usize;
                        let arg_raw_idx = code.get(*pc + offset + 1).copied().ok_or_else(|| {
                            anyhow::anyhow!("CALL_JQ at pc={}: missing argument {} index", *pc, arg_i)
                        })? as usize;
                        let arg_fidx = arg_raw_idx & !ARG_NEWCLOSURE;

                        // Resolve arg bytecode using arg_level
                        let arg_target_bc: &BytecodeRef = if arg_level == 0 {
                            bc
                        } else if arg_level <= parent_bcs.len() {
                            parent_bcs[parent_bcs.len() - arg_level]
                        } else {
                            bail!(
                                "CALL_JQ at pc={}: arg {} level {} exceeds parent depth {}",
                                *pc, arg_i, arg_level, parent_bcs.len()
                            );
                        };
                        let arg_subfunctions = arg_target_bc.subfunctions();
                        if arg_fidx >= arg_subfunctions.len() {
                            bail!(
                                "CALL_JQ at pc={}: argument {} subfunction index {} out of range (have {}, level={})",
                                *pc, arg_i, arg_fidx, arg_subfunctions.len(), arg_level
                            );
                        }

                        let arg_bc = &arg_subfunctions[arg_fidx];
                        let arg_code = arg_bc.code();
                        let mut arg_pc: usize = 0;
                        let mut arg_stack: Vec<Expr> = vec![Expr::Input];
                        let mut new_parent_bcs: Vec<&BytecodeRef> = parent_bcs.to_vec();
                        new_parent_bcs.push(bc);
                        let arg_ir = compile_range(
                            arg_bc, arg_code, &mut arg_pc, arg_code.len(),
                            &mut arg_stack, true, &[], &new_parent_bcs,
                        ).with_context(|| format!(
                            "CALL_JQ at pc={}: failed to compile argument {} closure (fidx={}, level={})",
                            *pc, arg_i, arg_fidx, arg_level
                        ))?;
                        closure_irs.push(arg_ir);
                    }

                    let fn_code = fn_bc.code();

                    // Phase 8-1: Recognize jq-defined binary functions (rtrimstr, ltrimstr, etc.)
                    // IMPORTANT: This must come BEFORE PathOf detection, because some jq builtins
                    // (e.g., rtrimstr) have PATH_BEGIN in their bytecode but should be handled
                    // as BinOp, not PathOf.
                    if nargs == 1 {
                        if let Some(binop) = recognize_jq_defined_binop(fn_bc) {
                            // Build the BinOp with Input references, then pipe through generator if needed
                            let arg_with_input = crate::cps_ir::substitute_input(&closure_irs[0], &Expr::Input);
                            let binop_expr = Expr::BinOp {
                                op: binop,
                                lhs: Box::new(Expr::Input),
                                rhs: Box::new(arg_with_input),
                            };
                            let expr = pipe_through(&binop_expr, &caller_input);
                            stack.push(expr);
                            *pc += instr_len;
                            continue;
                        }
                    }

                    // Phase 10-2: Recognize regex functions (test, match, capture, scan, sub, gsub)
                    // IMPORTANT: This must come BEFORE PathOf/Update detection, because
                    // some regex functions (e.g., sub, gsub) may have PATH_BEGIN in
                    // sibling subfunctions and would be misidentified as Update.
                    if let Some(regex_expr) = recognize_regex_function(fn_bc, nargs, &closure_irs, &caller_input) {
                        stack.push(regex_expr);
                        *pc += instr_len;
                        continue;
                    }

                    // Phase 9-1: Recognize closure-based builtins (sort_by, group_by, etc.)
                    // Also before PathOf/Update detection for the same reason.
                    if nargs == 1 {
                        if let Some(closure_op) = recognize_closure_builtin(fn_bc) {
                            // closure_irs[0] is the key function (e.g., .a for sort_by(.a))
                            let closure_apply = Expr::ClosureApply {
                                op: closure_op,
                                input_expr: Box::new(caller_input.clone()),
                                key_expr: Box::new(closure_irs[0].clone()),
                            };
                            stack.push(closure_apply);
                            *pc += instr_len;
                            continue;
                        }
                    }

                    // Phase 11: Detect while/until/range(3-arg) by debugname.
                    if let Some(loop_expr) = recognize_loop_function(fn_bc, nargs, &closure_irs, &caller_input) {
                        // For while/until, pipe_through handles wiring caller_input
                        // as the initial value. For range(3-arg), caller_input is already
                        // substituted into the closures by recognize_loop_function.
                        let inlined = pipe_through(&loop_expr, &caller_input);
                        stack.push(inlined);
                        *pc += instr_len;
                        continue;
                    }

                    // Phase 9-2: Detect range function (contains RANGE opcode)
                    let has_range_opcode = fn_code.iter().any(|&w| {
                        Opcode::from_u16(w) == Some(Opcode::RANGE)
                    });

                    if has_range_opcode && nargs == 2 {
                        // range(from; to) pattern: 2 closures provide from and to.
                        // closure_irs[0] = from expression
                        // closure_irs[1] = to expression
                        let from_expr = crate::cps_ir::substitute_input(&closure_irs[0], &caller_input);
                        let to_expr = crate::cps_ir::substitute_input(&closure_irs[1], &caller_input);
                        let range_expr = Expr::Range {
                            from: Box::new(from_expr),
                            to: Box::new(to_expr),
                            step: None,
                        };
                        stack.push(range_expr);
                        *pc += instr_len;
                        continue;
                    }

                    // Phase 9-5 / 10-3: Detect path/update patterns.
                    //
                    // path(expr): nargs=1, fn itself has PATH_BEGIN/PATH_END
                    //   - fn = PATH_BEGIN; CALL_JQ 0:0; PATH_END; RET
                    //   - closure[0] = the expression whose paths we extract
                    //
                    // |= (update): nargs=2, fn is _modify (225 lines).
                    //   The _modify function is called with 2 closures:
                    //   - closure[0] = path expression (e.g., .a, .[], .a.b)
                    //   - closure[1] = update expression (e.g., . + 1, . * 2)
                    //   Detection: nargs==2 AND a sibling subfunction (same parent level)
                    //   contains PATH_BEGIN. The _modify fn itself uses TAIL_CALL_JQ
                    //   to call the sibling path subfunction, so PATH_BEGIN won't appear
                    //   in _modify's own bytecode or its direct children.
                    let fn_has_path_opcode = fn_code.iter().any(|&w| {
                        Opcode::from_u16(w) == Some(Opcode::PATH_BEGIN)
                    });

                    if fn_has_path_opcode && nargs == 1 {
                        // path(expr) pattern: closure[0] is the expression whose
                        // access paths are extracted.
                        let path_expr = closure_irs[0].clone();
                        stack.push(Expr::PathOf {
                            input_expr: Box::new(caller_input),
                            path_expr: Box::new(path_expr),
                        });
                        *pc += instr_len;
                        continue;
                    }

                    if nargs == 2 {
                        // Check if any sibling subfunction has PATH_BEGIN.
                        // This detects the |= pattern where _modify (fn_fidx)
                        // references a sibling via TAIL_CALL_JQ.
                        let sibling_has_path = fn_subfunctions.iter().enumerate().any(|(i, sub)| {
                            i != fn_fidx && sub.code().iter().any(|&w| {
                                Opcode::from_u16(w) == Some(Opcode::PATH_BEGIN)
                            })
                        });
                        if sibling_has_path {
                            // |= (update) pattern:
                            // closure[0] is the path expression (e.g., .a, .[], .a.b)
                            // closure[1] is the update expression (e.g., . + 1, . * 2)
                            let path_expr = closure_irs[0].clone();
                            let update_expr = closure_irs[1].clone();

                            // Detect _assign vs _modify:
                            // _assign (= operator): update_expr runs against the ORIGINAL input
                            // _modify (|= operator): update_expr runs against the getpath result
                            let is_assign = fn_bc.debugname().map_or(false, |n| n.contains("_assign"));

                            stack.push(Expr::Update {
                                input_expr: Box::new(caller_input),
                                path_expr: Box::new(path_expr),
                                update_expr: Box::new(update_expr),
                                is_plain_assign: is_assign,
                            });
                            *pc += instr_len;
                            continue;
                        }
                    }

                    // Detect select pattern vs generic.
                    let is_select = fn_code.iter().any(|&w| {
                        Opcode::from_u16(w) == Some(Opcode::JUMP_F)
                    });

                    if is_select && nargs == 1 {
                        // select(f) pattern (Phase 4-4).
                        //
                        // When caller_input is a generator (e.g., Recurse), we build the
                        // select with Input references first, then pipe_through at the end.
                        let is_gen_input = matches!(&caller_input, Expr::Recurse { .. });
                        let effective_input = if is_gen_input {
                            Expr::Input
                        } else {
                            caller_input.clone()
                        };

                        let cond_expr = crate::cps_ir::substitute_input(&closure_irs[0], &effective_input);

                        *pc += instr_len;

                        stack.push(effective_input.clone());
                        let body_result = compile_range_inner(bc, code, pc, end_pc, stack, closures, parent_bcs)?;

                        let (body_expr, result_kind) = match body_result {
                            RangeResult::Ret(expr) => (expr, "ret"),
                            RangeResult::Jump(expr, target) => {
                                let mut select_expr = Expr::IfThenElse {
                                    cond: Box::new(cond_expr),
                                    then_branch: Box::new(expr),
                                    else_branch: Box::new(Expr::Empty),
                                };
                                if is_gen_input {
                                    select_expr = pipe_through(&select_expr, &caller_input);
                                }
                                stack.push(select_expr);
                                return Ok(RangeResult::Jump(stack.pop().unwrap(), target));
                            }
                            RangeResult::End(expr) => (expr, "end"),
                        };

                        let mut select_expr = Expr::IfThenElse {
                            cond: Box::new(cond_expr),
                            then_branch: Box::new(body_expr),
                            else_branch: Box::new(Expr::Empty),
                        };
                        if is_gen_input {
                            select_expr = pipe_through(&select_expr, &caller_input);
                        }
                        stack.push(select_expr);

                        if result_kind == "ret" {
                            return Ok(RangeResult::Ret(stack.pop().unwrap()));
                        } else {
                            return Ok(RangeResult::End(stack.pop().unwrap()));
                        }
                    } else {
                        // Generic nargs>=1 pattern (e.g., map(f), sort_by(f), def f(a;b):...).
                        //
                        // Compile fn with all closure arguments as closure bindings.
                        // When fn's bytecode contains CALL_JQ(0, i), it will resolve
                        // to closure_irs[i] via the closures parameter.
                        let fn_code_owned = fn_code.to_vec();
                        let mut new_parent_bcs: Vec<&BytecodeRef> = parent_bcs.to_vec();
                        new_parent_bcs.push(bc);
                        let mut fn_pc: usize = 0;
                        let mut fn_stack: Vec<Expr> = vec![Expr::Input];
                        let fn_ir = compile_range(
                            fn_bc, &fn_code_owned, &mut fn_pc, fn_code_owned.len(),
                            &mut fn_stack, true, &closure_irs, &new_parent_bcs,
                        ).with_context(|| format!(
                            "CALL_JQ at pc={}: failed to compile function subfunction {} with {} closures",
                            *pc, fn_fidx, nargs
                        ))?;

                        // Use pipe_through to properly handle generator inputs (e.g., Recurse)
                        let inlined = pipe_through(&fn_ir, &caller_input);
                        stack.push(inlined);
                    }
                } else {
                    bail!(
                        "CALL_JQ at pc={}: nargs={} not supported",
                        *pc, nargs
                    );
                }

                *pc += instr_len;
            }

            Opcode::RANGE => {
                // Phase 9-2: RANGE generator opcode.
                //
                // RANGE [level] [index] (3 words)
                //
                // In the jq VM, RANGE reads a counter from variable (level, index) and
                // a limit from the stack top. If counter < limit, it pushes the counter
                // value and increments the counter variable. When counter >= limit, it
                // backtracks.
                //
                // In our CPS model, we emit a Range { from, to } generator node.
                // The `from` is the initial value of the counter variable, and `to`
                // is the limit from the stack.
                //
                // Typical bytecode pattern (for range(from; to)):
                //   ... DUP LOADV $from STOREV $counter RANGE $counter RET
                //
                // At this point, the stack has: [..., to_copy, to_limit]
                // (the to was DUPed: one copy for RANGE's loop, one for backtrack restore)

                let _level = code.get(*pc + 1).copied().unwrap_or(0);
                let raw_index = code.get(*pc + 2).copied().ok_or_else(|| {
                    anyhow::anyhow!("RANGE at pc={}: missing variable index", *pc)
                })?;
                let var_index = scoped_var(parent_bcs.len(), raw_index);

                if stack.is_empty() {
                    bail!("RANGE at pc={}: stack underflow", *pc);
                }

                // Read the limit from stack top. In the jq VM, RANGE reads but doesn't
                // pop the limit (it stays for subsequent iterations). When done, RANGE
                // pops it and backtracks. In our CPS model, we pop it since the generator
                // handles all iterations.
                let to_expr = stack.pop().unwrap();

                // The counter variable was initialized with `from` via STOREV before RANGE.
                // We reference it via LoadVar.
                let from_expr = Expr::LoadVar { var_index };

                *pc += 3; // opcode + level + index

                // Like EACH, RANGE is a generator: it yields values, and the remaining
                // bytecodes form the "body" applied to each value. We compile the body
                // with a fresh stack containing Expr::Input (each yielded value).
                let mut body_stack: Vec<Expr> = vec![Expr::Input];
                let body_result = compile_range_inner(
                    bc, code, pc, end_pc, &mut body_stack, closures, parent_bcs,
                )?;

                let (_body_expr, result_kind) = match body_result {
                    RangeResult::Ret(expr) => (expr, "ret"),
                    RangeResult::Jump(_expr, target) => {
                        let range_expr = Expr::Range {
                            from: Box::new(from_expr),
                            to: Box::new(to_expr),
                            step: None,
                        };
                        stack.push(range_expr);
                        return Ok(RangeResult::Jump(stack.pop().unwrap(), target));
                    }
                    RangeResult::End(expr) => (expr, "end"),
                };

                let range_expr = Expr::Range {
                    from: Box::new(from_expr),
                    to: Box::new(to_expr),
                    step: None,
                };
                stack.push(range_expr);

                if result_kind == "ret" {
                    return Ok(RangeResult::Ret(stack.pop().unwrap()));
                } else {
                    return Ok(RangeResult::End(stack.pop().unwrap()));
                }
            }

            Opcode::DUPN => {
                // DUPN: like DUP, but replace the original with null (move semantics).
                // Stack effect: [..., v] → [..., null, v]
                // The original is consumed (moved to top), and null is left in its place.
                if stack.is_empty() {
                    bail!("DUPN at pc={}: stack underflow", *pc);
                }
                let top = stack.pop().unwrap();
                stack.push(Expr::Literal(Literal::Null));
                stack.push(top);
                *pc += 1;
            }

            Opcode::DUP2 => {
                // DUP2: duplicate the second element and push on top.
                // Stack effect: [..., a, b] → [..., a, b, a]
                if stack.len() < 2 {
                    bail!(
                        "DUP2 at pc={}: stack underflow (need 2, have {})",
                        *pc,
                        stack.len()
                    );
                }
                let second = stack[stack.len() - 2].clone();
                stack.push(second);
                *pc += 1;
            }

            Opcode::ERRORK => {
                // ERRORK: generate an error with a constant message.
                // Encoding: ERRORK + const_idx (2 words)
                //
                // In jq VM, this creates an error value and backtracks.
                // In CPS model, we produce a Value::Error literal which
                // TryCatch can catch via rt_is_error.
                let const_idx = code.get(*pc + 1).copied().ok_or_else(|| {
                    anyhow::anyhow!("ERRORK at pc={}: missing constant index", *pc)
                })? as usize;

                let lit = jv_constant_to_literal(bc, const_idx)?;
                let msg = match &lit {
                    Literal::Str(s) => s.clone(),
                    _ => "error".to_string(),
                };

                // Replace stack top (if any) with Error literal
                if !stack.is_empty() {
                    stack.pop();
                }
                stack.push(Expr::Literal(Literal::Error(msg)));
                *pc += 2;
            }

            Opcode::DESTRUCTURE_ALT => {
                // DESTRUCTURE_ALT: destructuring alternative pattern (?//).
                // Not yet implemented — requires mutable variable slots across
                // TryCatch boundaries, which the CPS LetBinding model doesn't support.
                bail!(
                    "unsupported opcode DESTRUCTURE_ALT at pc={} (destructuring alternatives not yet implemented)",
                    *pc
                );
            }

            Opcode::GENLABEL => {
                // GENLABEL: generate a unique label ID and push onto stack.
                // In our CPS model, label-break patterns are complex and involve
                // TRY_BEGIN/BACKTRACK with label matching. For now, push a sentinel
                // value that will be stored in a variable and used for error-based
                // control flow (break $label triggers error($label_id)).
                //
                // We push a unique integer as a literal. The actual label-break
                // mechanism works through TRY_BEGIN/error/catch in jq 1.8.
                stack.push(Expr::Literal(Literal::Str("__label__".to_string())));
                *pc += 1;
            }

            Opcode::DEPS => {
                // DEPS: module dependency metadata. NOP for JIT.
                *pc += 2; // opcode + const_idx
            }

            Opcode::MODULEMETA => {
                // MODULEMETA: module metadata. NOP for JIT.
                *pc += 2; // opcode + const_idx
            }

            Opcode::STORE_GLOBAL => {
                // STORE_GLOBAL: store a constant to a global variable.
                // Used for $ENV initialization etc. NOP for JIT (handled at init time).
                *pc += 4; // opcode + const_idx + level + var_idx
            }

            Opcode::CLOSURE_CREATE | Opcode::CLOSURE_CREATE_C
            | Opcode::CLOSURE_PARAM | Opcode::CLOSURE_PARAM_REGULAR => {
                // These are pseudo-instructions that are part of CALL_JQ's
                // variable-length encoding. They should not appear independently
                // in the bytecode stream. If we encounter them, skip (length 0
                // in the description table — they don't consume code words).
                bail!(
                    "unexpected pseudo-opcode {} at pc={} (should be part of CALL_JQ)",
                    op.name(),
                    *pc
                );
            }

            Opcode::CLOSURE_REF => {
                // CLOSURE_REF: another pseudo-instruction (2 words).
                bail!(
                    "unexpected CLOSURE_REF at pc={} (should be part of CALL_JQ)",
                    *pc
                );
            }

            Opcode::PATH_BEGIN | Opcode::PATH_END => {
                // PATH_BEGIN/PATH_END: path tracking opcodes.
                // Not yet implemented (Phase 9-5/10-3).
                bail!(
                    "unsupported opcode {} at pc={} (path operations not yet implemented)",
                    op.name(),
                    *pc
                );
            }

            _ => {
                bail!(
                    "unsupported opcode {} ({}) at pc={}",
                    op.name(),
                    raw_op,
                    *pc
                );
            }
        }
    }

    // Reached end of range without RET or JUMP.
    if stack.is_empty() {
        bail!(
            "bytecode range ended with empty stack (pc={})",
            *pc
        );
    }
    Ok(RangeResult::End(stack.pop().unwrap()))
}

/// Translate a CALL_BUILTIN invocation into an IR expression.
///
/// `args` is ordered as popped from the stack: args[0] = top, args[1] = second, etc.
///
/// For binary builtins (nargs=3):
/// - args[0] = input (discarded by the C function via jv_free)
/// - args[1] = lhs operand
/// - args[2] = rhs operand
///
/// For unary builtins (nargs=1):
/// - args[0] = the input value
fn translate_builtin(name: &str, args: &[Expr], pc: usize) -> Result<Expr> {
    // Try binary operators (nargs=3): arithmetic + comparison + math 2-arg
    let binop = match name {
        "_plus" => Some(BinOp::Add),
        "_minus" => Some(BinOp::Sub),
        "_multiply" => Some(BinOp::Mul),
        "_divide" => Some(BinOp::Div),
        "_modulo" | "_mod" => Some(BinOp::Mod),
        "_equal" => Some(BinOp::Eq),
        "_notequal" => Some(BinOp::Ne),
        "_less" => Some(BinOp::Lt),
        "_greater" => Some(BinOp::Gt),
        "_lesseq" => Some(BinOp::Le),
        "_greatereq" => Some(BinOp::Ge),
        // Math binary functions (nargs=3: input, arg1, arg2)
        "pow" => Some(BinOp::Pow),
        "atan" if args.len() == 3 => Some(BinOp::Atan2),
        "drem" | "remainder" => Some(BinOp::Drem),
        "ldexp" => Some(BinOp::Ldexp),
        "scalb" => Some(BinOp::Scalb),
        "scalbln" => Some(BinOp::Scalbln),
        _ => None,
    };

    if let Some(op) = binop {
        if args.len() != 3 {
            bail!(
                "CALL_BUILTIN {} at pc={}: expected nargs=3, got {}",
                name,
                pc,
                args.len()
            );
        }
        // args[0] = input (discarded), args[1] = lhs, args[2] = rhs
        return Ok(Expr::BinOp {
            op,
            lhs: Box::new(args[1].clone()),
            rhs: Box::new(args[2].clone()),
        });
    }

    // Phase 9-6: setpath (nargs=3, ternary operation)
    if name == "setpath" && args.len() == 3 {
        // args[0] = input, args[1] = path, args[2] = value
        return Ok(Expr::SetPath {
            input_expr: Box::new(args[0].clone()),
            path: Box::new(args[1].clone()),
            value: Box::new(args[2].clone()),
        });
    }

    // Phase 8-7 / Phase 9-3: format builtin for string interpolation and @-formats.
    // `format("text")` is nargs=2 and maps to tostring (UnaryOp::ToString).
    // `@base64` etc. compile to: CALL_BUILTIN format/2 with args[1] = format name.
    // In jq's bytecode, string interpolation `\(expr)` compiles to:
    //   SUBEXP_BEGIN → expr → PUSHK_UNDER "text" → CALL_BUILTIN format/2 → ...
    if name == "format" && args.len() == 2 {
        // args[0] = value to format, args[1] = format name string
        // Determine the format name from the literal string argument.
        let format_name = if let Expr::Literal(Literal::Str(s)) = &args[1] {
            s.as_str()
        } else {
            "text" // fallback
        };
        let op = match format_name {
            "text" => UnaryOp::ToString,
            "base64" => UnaryOp::FormatBase64,
            "base64d" => UnaryOp::FormatBase64d,
            "html" => UnaryOp::FormatHtml,
            "uri" => UnaryOp::FormatUri,
            "urid" => UnaryOp::FormatUrid,
            "csv" => UnaryOp::FormatCsv,
            "tsv" => UnaryOp::FormatTsv,
            "json" => UnaryOp::FormatJson,
            "sh" => UnaryOp::FormatSh,
            _ => UnaryOp::ToString, // unknown format → tostring
        };
        return Ok(Expr::UnaryOp {
            op,
            operand: Box::new(args[0].clone()),
        });
    }

    // Try nargs=2 builtins (input + 1 arg = binary operation)
    // For nargs=2: args[0] = top (input), args[1] = second (the argument)
    let binop2 = match name {
        "split" => Some(BinOp::Split),
        "has" => Some(BinOp::Has),
        "startswith" => Some(BinOp::StartsWith),
        "endswith" => Some(BinOp::EndsWith),
        "join" => Some(BinOp::Join),
        "contains" => Some(BinOp::Contains),
        "ltrimstr" => Some(BinOp::Ltrimstr),
        "rtrimstr" => Some(BinOp::Rtrimstr),
        "in" => Some(BinOp::In),
        // Phase 9-6: indices, index, rindex, inside, getpath, delpaths
        "indices" => Some(BinOp::Indices),
        "index" => Some(BinOp::StrIndex),
        "rindex" => Some(BinOp::StrRindex),
        "inside" => Some(BinOp::Inside),
        "getpath" => Some(BinOp::GetPath),
        "delpaths" => Some(BinOp::DelPaths),
        "flatten" => Some(BinOp::FlattenDepth),
        "bsearch" => Some(BinOp::Bsearch),
        "strftime" => Some(BinOp::Strftime),
        "strptime" => Some(BinOp::Strptime),
        "strflocaltime" => Some(BinOp::Strflocaltime),
        _ => None,
    };

    if let Some(op) = binop2 {
        if args.len() != 2 {
            bail!(
                "CALL_BUILTIN {} at pc={}: expected nargs=2, got {}",
                name,
                pc,
                args.len()
            );
        }
        // args[0] = input (the value), args[1] = the argument
        return Ok(Expr::BinOp {
            op,
            lhs: Box::new(args[0].clone()),
            rhs: Box::new(args[1].clone()),
        });
    }

    // Try unary operators (nargs=1)
    let unaryop = match name {
        "length" => Some(UnaryOp::Length),
        "type" => Some(UnaryOp::Type),
        "tostring" => Some(UnaryOp::ToString),
        "tonumber" => Some(UnaryOp::ToNumber),
        "keys" => Some(UnaryOp::Keys),
        "_negate" => Some(UnaryOp::Negate),
        "sort" => Some(UnaryOp::Sort),
        "keys_unsorted" => Some(UnaryOp::KeysUnsorted),
        "floor" => Some(UnaryOp::Floor),
        "ceil" => Some(UnaryOp::Ceil),
        "round" => Some(UnaryOp::Round),
        "fabs" => Some(UnaryOp::Fabs),
        "explode" => Some(UnaryOp::Explode),
        "implode" => Some(UnaryOp::Implode),
        // Phase 5-3: In jq 1.8.1, some previously jq-defined functions became CALL_BUILTIN
        "unique" => Some(UnaryOp::Unique),
        "reverse" => Some(UnaryOp::Reverse),
        "add" => Some(UnaryOp::Add),
        "ascii_downcase" => Some(UnaryOp::AsciiDowncase),
        "ascii_upcase" => Some(UnaryOp::AsciiUpcase),
        "to_entries" => Some(UnaryOp::ToEntries),
        "from_entries" => Some(UnaryOp::FromEntries),
        // Phase 8-1: Missing unary builtins
        "min" => Some(UnaryOp::Min),
        "max" => Some(UnaryOp::Max),
        "flatten" => Some(UnaryOp::Flatten),
        // Phase 9-6: Remaining unary builtins
        "tojson" => Some(UnaryOp::ToJson),
        "fromjson" => Some(UnaryOp::FromJson),
        "debug" => Some(UnaryOp::Debug),
        "infinite" => Some(UnaryOp::Infinite),
        "nan" => Some(UnaryOp::Nan),
        "isinfinite" => Some(UnaryOp::IsInfinite),
        "isnan" => Some(UnaryOp::IsNan),
        "isnormal" => Some(UnaryOp::IsNormal),
        "any" => Some(UnaryOp::Any),
        "all" => Some(UnaryOp::All),
        "env" => Some(UnaryOp::Env),
        "builtins" => Some(UnaryOp::Builtins),
        // Math unary functions
        "sqrt" => Some(UnaryOp::Sqrt),
        "sin" => Some(UnaryOp::Sin),
        "cos" => Some(UnaryOp::Cos),
        "tan" => Some(UnaryOp::Tan),
        "asin" => Some(UnaryOp::Asin),
        "acos" => Some(UnaryOp::Acos),
        "atan" => Some(UnaryOp::Atan),
        "exp" => Some(UnaryOp::Exp),
        "exp2" => Some(UnaryOp::Exp2),
        "exp10" => Some(UnaryOp::Exp10),
        "log" => Some(UnaryOp::Log),
        "log2" => Some(UnaryOp::Log2),
        "log10" => Some(UnaryOp::Log10),
        "cbrt" => Some(UnaryOp::Cbrt),
        "significand" => Some(UnaryOp::Significand),
        "exponent" => Some(UnaryOp::Exponent),
        "logb" => Some(UnaryOp::Logb),
        "nearbyint" => Some(UnaryOp::NearbyInt),
        "trunc" => Some(UnaryOp::Trunc),
        "rint" => Some(UnaryOp::Rint),
        "j0" => Some(UnaryOp::J0),
        "j1" => Some(UnaryOp::J1),
        "transpose" => Some(UnaryOp::Transpose),
        "utf8bytelength" => Some(UnaryOp::Utf8ByteLength),
        "toboolean" => Some(UnaryOp::ToBoolean),
        "trim" => Some(UnaryOp::Trim),
        "ltrim" => Some(UnaryOp::Ltrim),
        "rtrim" => Some(UnaryOp::Rtrim),
        "gmtime" => Some(UnaryOp::Gmtime),
        "mktime" => Some(UnaryOp::Mktime),
        "now" => Some(UnaryOp::Now),
        _ => None,
    };

    if let Some(op) = unaryop {
        if args.len() != 1 {
            bail!(
                "CALL_BUILTIN {} at pc={}: expected nargs=1, got {}",
                name,
                pc,
                args.len()
            );
        }
        return Ok(Expr::UnaryOp {
            op,
            operand: Box::new(args[0].clone()),
        });
    }

    // Phase 10-4: error builtin (nargs=1).
    // `error` takes the stack top as the error message and produces an error value.
    // In CPS model, this produces a Value::Error which TryCatch can catch.
    // The error message comes from the argument (the input value).
    if name == "error" && args.len() == 1 {
        // Use the input value as the error message.
        // MakeError converts the input to Value::Error(input_as_string).
        return Ok(Expr::UnaryOp {
            op: UnaryOp::MakeError,
            operand: Box::new(args[0].clone()),
        });
    }

    bail!(
        "CALL_BUILTIN {} at pc={}: unsupported builtin (nargs={})",
        name,
        pc,
        args.len()
    );
}

/// Convert the i-th constant from the bytecode's constant pool to a [`Literal`].
fn jv_constant_to_literal(bc: &BytecodeRef, index: usize) -> Result<Literal> {
    unsafe {
        let constants = bc.constants_raw(); // jv_copy'd array
        let kind = jq_ffi::jv_get_kind(constants);

        if kind != JvKind::Array {
            jq_ffi::jv_free(constants);
            bail!("constants is not an array (kind={:?})", kind);
        }

        let len = jq_ffi::jv_array_length(jq_ffi::jv_copy(constants));
        if (index as i32) >= len {
            jq_ffi::jv_free(constants);
            bail!(
                "constant index {} out of range (len={})",
                index,
                len
            );
        }

        let val = jq_ffi::jv_array_get(jq_ffi::jv_copy(constants), index as i32);
        let lit = jv_to_literal(val)?;
        jq_ffi::jv_free(constants);
        Ok(lit)
    }
}

/// Convert a jv value to a [`Literal`].  Consumes the jv.
unsafe fn jv_to_literal(val: Jv) -> Result<Literal> {
    unsafe {
        let kind = jq_ffi::jv_get_kind(val);
        match kind {
            JvKind::Null => {
                jq_ffi::jv_free(val);
                Ok(Literal::Null)
            }
            JvKind::True => {
                jq_ffi::jv_free(val);
                Ok(Literal::Bool(true))
            }
            JvKind::False => {
                jq_ffi::jv_free(val);
                Ok(Literal::Bool(false))
            }
            JvKind::Number => {
                let n = jq_ffi::jv_number_value(val);
                jq_ffi::jv_free(val);
                Ok(Literal::Num(n))
            }
            JvKind::String => {
                let cstr = jq_ffi::jv_string_value(jq_ffi::jv_copy(val));
                let len = jq_ffi::jv_string_length_bytes(jq_ffi::jv_copy(val)) as usize;
                // Use raw pointer + length to handle embedded null bytes
                let bytes = std::slice::from_raw_parts(cstr as *const u8, len);
                let s = String::from_utf8_lossy(bytes).into_owned();
                jq_ffi::jv_free(val);
                Ok(Literal::Str(s))
            }
            JvKind::Array => {
                let len = jq_ffi::jv_array_length(jq_ffi::jv_copy(val));
                if len == 0 {
                    jq_ffi::jv_free(val);
                    Ok(Literal::EmptyArr)
                } else {
                    // Phase 8-8: Non-empty array literal.
                    // Convert each element to a Value and store as Literal::Arr.
                    let mut items = Vec::with_capacity(len as usize);
                    for i in 0..len {
                        let elem = jq_ffi::jv_array_get(jq_ffi::jv_copy(val), i);
                        items.push(crate::value::jv_to_value(elem)?);
                    }
                    jq_ffi::jv_free(val);
                    Ok(Literal::Arr(items))
                }
            }
            JvKind::Object => {
                // Phase 8-8: Object literal.
                let val_copy = jq_ffi::jv_copy(val);
                let iter = jq_ffi::jv_object_iter(val_copy);
                let is_empty = jq_ffi::jv_object_iter_valid(val_copy, iter) == 0;
                jq_ffi::jv_free(val_copy);
                if is_empty {
                    jq_ffi::jv_free(val);
                    Ok(Literal::EmptyObj)
                } else {
                    // Convert non-empty object to Literal::Obj
                    let value = crate::value::jv_to_value(val)?;
                    match value {
                        crate::value::Value::Obj(map) => {
                            Ok(Literal::Obj((*map).clone()))
                        }
                        _ => bail!("unexpected value type from jv_to_value for object"),
                    }
                }
            }
            _ => {
                jq_ffi::jv_free(val);
                bail!("unsupported constant kind {:?} for IR literal", kind);
            }
        }
    }
}

/// Recognize jq-defined functions (from builtin.jq) by their debuginfo name.
///
/// These functions are compiled to complex bytecode by libjq, but we can
/// implement them more efficiently as direct runtime calls.
/// Returns `Some(UnaryOp)` if the function is recognized, `None` otherwise.
///
/// Phase 5-3: add, values, reverse, to_entries, from_entries, unique,
/// ascii_downcase, ascii_upcase.
fn recognize_jq_defined_function(bc: &BytecodeRef) -> Option<UnaryOp> {
    let name = bc.debugname()?;
    match name.as_str() {
        "add" => Some(UnaryOp::Add),
        "reverse" => Some(UnaryOp::Reverse),
        "to_entries" => Some(UnaryOp::ToEntries),
        "from_entries" => Some(UnaryOp::FromEntries),
        "unique" => Some(UnaryOp::Unique),
        "ascii_downcase" => Some(UnaryOp::AsciiDowncase),
        "ascii_upcase" => Some(UnaryOp::AsciiUpcase),
        // Phase 8-1: Missing unary builtins
        "min" => Some(UnaryOp::Min),
        "max" => Some(UnaryOp::Max),
        "flatten" => Some(UnaryOp::Flatten),
        // Phase 9-6: Remaining unary builtins
        "tojson" => Some(UnaryOp::ToJson),
        "fromjson" => Some(UnaryOp::FromJson),
        "any" => Some(UnaryOp::Any),
        "all" => Some(UnaryOp::All),
        "debug" => Some(UnaryOp::Debug),
        "env" => Some(UnaryOp::Env),
        "builtins" => Some(UnaryOp::Builtins),
        "isinfinite" => Some(UnaryOp::IsInfinite),
        "isnan" => Some(UnaryOp::IsNan),
        "isnormal" => Some(UnaryOp::IsNormal),
        "infinite" => Some(UnaryOp::Infinite),
        "nan" => Some(UnaryOp::Nan),
        // Math unary functions (may be defined as jq wrappers around builtins)
        "sqrt" => Some(UnaryOp::Sqrt),
        "sin" => Some(UnaryOp::Sin),
        "cos" => Some(UnaryOp::Cos),
        "tan" => Some(UnaryOp::Tan),
        "asin" => Some(UnaryOp::Asin),
        "acos" => Some(UnaryOp::Acos),
        "atan" => Some(UnaryOp::Atan),
        "exp" => Some(UnaryOp::Exp),
        "exp2" => Some(UnaryOp::Exp2),
        "exp10" => Some(UnaryOp::Exp10),
        "log" => Some(UnaryOp::Log),
        "log2" => Some(UnaryOp::Log2),
        "log10" => Some(UnaryOp::Log10),
        "cbrt" => Some(UnaryOp::Cbrt),
        "significand" => Some(UnaryOp::Significand),
        "exponent" => Some(UnaryOp::Exponent),
        "logb" => Some(UnaryOp::Logb),
        "nearbyint" => Some(UnaryOp::NearbyInt),
        "trunc" => Some(UnaryOp::Trunc),
        "rint" => Some(UnaryOp::Rint),
        "j0" => Some(UnaryOp::J0),
        "j1" => Some(UnaryOp::J1),
        "transpose" => Some(UnaryOp::Transpose),
        "utf8bytelength" => Some(UnaryOp::Utf8ByteLength),
        "toboolean" => Some(UnaryOp::ToBoolean),
        "trim" => Some(UnaryOp::Trim),
        "ltrim" => Some(UnaryOp::Ltrim),
        "rtrim" => Some(UnaryOp::Rtrim),
        "gmtime" => Some(UnaryOp::Gmtime),
        "mktime" => Some(UnaryOp::Mktime),
        "now" => Some(UnaryOp::Now),
        // Note: "values" is a generator (select(. != null)), not a 1→1 function.
        // It cannot be implemented as a simple UnaryOp.
        _ => None,
    }
}

/// Recognize jq-defined binary functions (nargs=1 CALL_JQ) that we can
/// implement as direct BinOp runtime calls.
///
/// These functions take one closure argument and operate as binary operations
/// on (input, argument). They are defined in jq source but we have optimized
/// native runtime implementations.
fn recognize_jq_defined_binop(bc: &BytecodeRef) -> Option<BinOp> {
    let name = bc.debugname()?;
    match name.as_str() {
        "ltrimstr" => Some(BinOp::Ltrimstr),
        "rtrimstr" => Some(BinOp::Rtrimstr),
        "contains" => Some(BinOp::Contains),
        "inside" => Some(BinOp::Inside),
        "in" => Some(BinOp::In),
        "join" => Some(BinOp::Join),
        // Phase 9-6: indices, index, rindex
        "indices" => Some(BinOp::Indices),
        "index" => Some(BinOp::StrIndex),
        "rindex" => Some(BinOp::StrRindex),
        "getpath" => Some(BinOp::GetPath),
        "delpaths" => Some(BinOp::DelPaths),
        "flatten" => Some(BinOp::FlattenDepth),
        "bsearch" => Some(BinOp::Bsearch),
        "pow" => Some(BinOp::Pow),
        "atan" => Some(BinOp::Atan2),
        "drem" | "remainder" => Some(BinOp::Drem),
        "ldexp" => Some(BinOp::Ldexp),
        "scalb" => Some(BinOp::Scalb),
        "scalbln" => Some(BinOp::Scalbln),
        "strftime" => Some(BinOp::Strftime),
        "strptime" => Some(BinOp::Strptime),
        "strflocaltime" => Some(BinOp::Strflocaltime),
        _ => None,
    }
}

/// Recognize closure-based builtins (nargs=1 CALL_JQ) that take a key function.
///
/// These functions take an array input and a closure argument that computes
/// a key for each element. The operation (sort/group/unique/min/max) is
/// performed based on the computed keys.
///
/// Phase 11: Recognize loop functions (while, until, range/3) by debugname.
///
/// These functions use label-break-out patterns internally which are complex
/// to handle through generic function inlining. Instead, we detect them by name
/// and emit dedicated IR nodes.
fn recognize_loop_function(
    fn_bc: &BytecodeRef,
    nargs: usize,
    closure_irs: &[Expr],
    caller_input: &Expr,
) -> Option<Expr> {
    let name = fn_bc.debugname()?;

    match name.as_str() {
        "while" if nargs == 2 => {
            // while(cond; update): closure[0]=cond, closure[1]=update
            // While is a generator: yields current value as long as cond is true,
            // then applies update for next iteration.
            // input_expr is Input (will be substituted by pipe_through with caller_input).
            let cond_expr = closure_irs[0].clone();
            let update_expr = closure_irs[1].clone();
            Some(Expr::While {
                input_expr: Box::new(Expr::Input),
                cond: Box::new(cond_expr),
                update: Box::new(update_expr),
            })
        }
        "until" if nargs == 2 => {
            // until(cond; update): closure[0]=cond, closure[1]=update
            // Until is a scalar: loops applying update until cond is true, returns final value.
            // input_expr is Input (will be substituted by pipe_through with caller_input).
            let cond_expr = closure_irs[0].clone();
            let update_expr = closure_irs[1].clone();
            Some(Expr::Until {
                input_expr: Box::new(Expr::Input),
                cond: Box::new(cond_expr),
                update: Box::new(update_expr),
            })
        }
        "range" if nargs == 3 => {
            // range(from; to; step): closure[0]=from, closure[1]=to, closure[2]=step
            let from_expr = crate::cps_ir::substitute_input(&closure_irs[0], caller_input);
            let to_expr = crate::cps_ir::substitute_input(&closure_irs[1], caller_input);
            let step_expr = crate::cps_ir::substitute_input(&closure_irs[2], caller_input);
            Some(Expr::Range {
                from: Box::new(from_expr),
                to: Box::new(to_expr),
                step: Some(Box::new(step_expr)),
            })
        }
        // Phase 12: limit(n; gen) — yields at most n outputs from gen
        "limit" if nargs == 2 => {
            // limit(n; gen): closure[0]=n, closure[1]=gen
            let count_expr = crate::cps_ir::substitute_input(&closure_irs[0], caller_input);
            let gen_expr = closure_irs[1].clone();
            // gen_expr uses Input to refer to its source. pipe_through will
            // substitute caller_input later, so we build with Input references.
            Some(Expr::Limit {
                count: Box::new(count_expr),
                generator: Box::new(gen_expr),
            })
        }
        // Phase 12: first(gen) — yields only the first output of gen
        // first is defined as limit(1; gen) in jq
        "first" if nargs == 1 => {
            let gen_expr = closure_irs[0].clone();
            Some(Expr::Limit {
                count: Box::new(Expr::Literal(Literal::Num(1.0))),
                generator: Box::new(gen_expr),
            })
        }
        // Phase 12: last(gen) — yields only the last output of gen
        // last is defined as reduce gen as $x (null; $x) in jq
        // We can't easily create a Reduce here because we don't have var indices,
        // so we'll skip this for now.

        // Phase 12: skip(n; gen) — skips first n outputs from gen
        "skip" if nargs == 2 => {
            let count_expr = crate::cps_ir::substitute_input(&closure_irs[0], caller_input);
            let gen_expr = closure_irs[1].clone();
            Some(Expr::Skip {
                count: Box::new(count_expr),
                generator: Box::new(gen_expr),
            })
        }
        // Phase 12: map_values(f) — equivalent to .[] |= f
        "map_values" if nargs == 1 => {
            let update_expr = closure_irs[0].clone();
            Some(Expr::Update {
                input_expr: Box::new(Expr::Input),
                path_expr: Box::new(Expr::Each {
                    input_expr: Box::new(Expr::Input),
                    body: Box::new(Expr::Input),
                }),
                update_expr: Box::new(update_expr),
                is_plain_assign: false,
            })
        }
        _ => None,
    }
}

/// Phase 9-1: sort_by, group_by, unique_by, min_by, max_by.
fn recognize_closure_builtin(bc: &BytecodeRef) -> Option<ClosureOp> {
    let name = bc.debugname()?;
    match name.as_str() {
        "sort_by" => Some(ClosureOp::SortBy),
        "group_by" => Some(ClosureOp::GroupBy),
        "unique_by" => Some(ClosureOp::UniqueBy),
        "min_by" => Some(ClosureOp::MinBy),
        "max_by" => Some(ClosureOp::MaxBy),
        _ => None,
    }
}

/// Phase 10-2: Recognize regex functions by debugname and build IR nodes.
///
/// jq compiles regex functions (test, match, capture, scan, sub, gsub) as
/// CALL_JQ with subfunctions. The function bytecode is very complex (wrapping
/// `_match_impl` builtin), so we detect by debugname and implement directly.
///
/// Patterns:
/// - test(re): nargs=1, fn debugname="test", closure[0]=re
/// - test(re; flags): nargs=2, fn debugname="test", closure[0]=re, closure[1]=flags
/// - match(re): nargs=1, fn debugname="match", closure[0]=re
/// - match(re; flags): nargs=2, fn debugname="match", closure[0]=re, closure[1]=flags
/// - capture(re): nargs=1, fn debugname via scan subfunctions
/// - scan(re): nargs=1, fn debugname="scan", closure[0]=re
/// - sub(re; tostr): nargs=2, fn debugname="sub", closure[0]=re, closure[1]=tostr
/// - gsub(re; tostr): nargs=2, fn debugname="gsub", closure[0]=re, closure[1]=tostr
fn recognize_regex_function(
    fn_bc: &BytecodeRef,
    nargs: usize,
    closure_irs: &[Expr],
    caller_input: &Expr,
) -> Option<Expr> {
    let name = fn_bc.debugname()?;

    match name.as_str() {
        "test" => {
            // test(re) nargs=1 or test(re; flags) nargs=2
            let re_expr = crate::cps_ir::substitute_input(&closure_irs[0], caller_input);
            let flags_expr = if nargs >= 2 {
                crate::cps_ir::substitute_input(&closure_irs[1], caller_input)
            } else {
                Expr::Literal(Literal::Null)
            };
            Some(Expr::RegexTest {
                input_expr: Box::new(caller_input.clone()),
                re: Box::new(re_expr),
                flags: Box::new(flags_expr),
            })
        }
        "match" => {
            // match(re) nargs=1 or match(re; flags) nargs=2
            let re_expr = crate::cps_ir::substitute_input(&closure_irs[0], caller_input);
            let flags_expr = if nargs >= 2 {
                crate::cps_ir::substitute_input(&closure_irs[1], caller_input)
            } else {
                Expr::Literal(Literal::Null)
            };
            Some(Expr::RegexMatch {
                input_expr: Box::new(caller_input.clone()),
                re: Box::new(re_expr),
                flags: Box::new(flags_expr),
            })
        }
        "capture" => {
            // capture(re) nargs=1 or capture(re; flags) nargs=2
            let re_expr = crate::cps_ir::substitute_input(&closure_irs[0], caller_input);
            let flags_expr = if nargs >= 2 {
                crate::cps_ir::substitute_input(&closure_irs[1], caller_input)
            } else {
                Expr::Literal(Literal::Null)
            };
            Some(Expr::RegexCapture {
                input_expr: Box::new(caller_input.clone()),
                re: Box::new(re_expr),
                flags: Box::new(flags_expr),
            })
        }
        "scan" => {
            // scan(re) nargs=1 or scan(re; flags) nargs=2
            let re_expr = crate::cps_ir::substitute_input(&closure_irs[0], caller_input);
            let flags_expr = if nargs >= 2 {
                crate::cps_ir::substitute_input(&closure_irs[1], caller_input)
            } else {
                Expr::Literal(Literal::Null)
            };
            Some(Expr::RegexScan {
                input_expr: Box::new(caller_input.clone()),
                re: Box::new(re_expr),
                flags: Box::new(flags_expr),
            })
        }
        "sub" => {
            // sub(re; tostr) nargs=2 or sub(re; tostr; flags) nargs=3
            if nargs >= 2 {
                let re_expr = crate::cps_ir::substitute_input(&closure_irs[0], caller_input);
                let tostr_expr = crate::cps_ir::substitute_input(&closure_irs[1], caller_input);
                let flags_expr = if nargs >= 3 {
                    crate::cps_ir::substitute_input(&closure_irs[2], caller_input)
                } else {
                    Expr::Literal(Literal::Null)
                };
                Some(Expr::RegexSub {
                    input_expr: Box::new(caller_input.clone()),
                    re: Box::new(re_expr),
                    tostr: Box::new(tostr_expr),
                    flags: Box::new(flags_expr),
                })
            } else {
                None
            }
        }
        "gsub" => {
            // gsub(re; tostr) nargs=2 or gsub(re; tostr; flags) nargs=3
            if nargs >= 2 {
                let re_expr = crate::cps_ir::substitute_input(&closure_irs[0], caller_input);
                let tostr_expr = crate::cps_ir::substitute_input(&closure_irs[1], caller_input);
                let flags_expr = if nargs >= 3 {
                    crate::cps_ir::substitute_input(&closure_irs[2], caller_input)
                } else {
                    Expr::Literal(Literal::Null)
                };
                Some(Expr::RegexGsub {
                    input_expr: Box::new(caller_input.clone()),
                    re: Box::new(re_expr),
                    tostr: Box::new(tostr_expr),
                    flags: Box::new(flags_expr),
                })
            } else {
                None
            }
        }
        _ => None,
    }
}
