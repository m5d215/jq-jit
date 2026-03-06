//! Cranelift JIT compiler for jq filters.
//!
//! Two-phase approach:
//! 1. Flatten IR expression tree into linear JitOps
//! 2. Generate Cranelift IR from JitOps
//!
//! Key design: expressions are classified as "scalar" (exactly one output)
//! or "generator" (zero or more outputs). Scalar expressions compile to
//! direct computation; generators compile to loops with Yield operations.

use std::collections::HashMap;
use std::rc::Rc;

use anyhow::{Result, bail};
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, StackSlotData, StackSlotKind};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, FuncId};

use crate::ir::*;
use crate::value::Value;

const GEN_CONTINUE: i64 = 1;
const GEN_STOP: i64 = 0;
const GEN_ERROR: i64 = -1;

// ============================================================================
// Expression classification
// ============================================================================

fn is_jit_supported_unaryop(op: UnaryOp) -> bool {
    matches!(op,
        UnaryOp::Length | UnaryOp::Type | UnaryOp::Keys | UnaryOp::Values |
        UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Fabs |
        UnaryOp::Sort | UnaryOp::Reverse | UnaryOp::Unique | UnaryOp::Flatten |
        UnaryOp::Min | UnaryOp::Max | UnaryOp::Add | UnaryOp::ToString |
        UnaryOp::ToNumber | UnaryOp::ToJson | UnaryOp::FromJson | UnaryOp::Ascii |
        UnaryOp::Explode | UnaryOp::Implode | UnaryOp::AsciiDowncase |
        UnaryOp::AsciiUpcase | UnaryOp::Any | UnaryOp::All | UnaryOp::Transpose |
        UnaryOp::ToEntries | UnaryOp::FromEntries | UnaryOp::KeysUnsorted |
        UnaryOp::Sqrt | UnaryOp::Abs | UnaryOp::Not | UnaryOp::TypeOf |
        UnaryOp::Infinite | UnaryOp::Nan | UnaryOp::IsInfinite | UnaryOp::IsNan |
        UnaryOp::IsNormal | UnaryOp::IsFinite
    )
}

/// Returns true if expr always produces exactly one output value AND the JIT can handle it inline.
/// Conservative: only includes expressions we know the JIT handles correctly as scalars.
fn is_scalar(expr: &Expr) -> bool {
    match expr {
        Expr::Input | Expr::Not => true,
        Expr::Literal(_) | Expr::LoadVar { .. } => true,
        Expr::BinOp { op, lhs, rhs } => match op {
            BinOp::And | BinOp::Or => is_scalar(lhs) && is_scalar(rhs),
            _ => is_scalar(lhs) && is_scalar(rhs),
        },
        Expr::UnaryOp { op, operand } => is_jit_supported_unaryop(*op) && is_scalar(operand),
        Expr::Negate { operand } => is_scalar(operand),
        Expr::Index { expr, key } => is_scalar(expr) && is_scalar(key),
        // IndexOpt is NOT scalar - it produces 0 or 1 outputs (suppresses errors)
        // Expr::IndexOpt { .. } => false (default)
        Expr::Pipe { left, right } => is_scalar(left) && is_scalar(right),
        Expr::IfThenElse { cond, then_branch, else_branch } =>
            is_scalar(cond) && is_scalar(then_branch) && is_scalar(else_branch),
        Expr::LetBinding { value, body, .. } => is_scalar(value) && is_scalar(body),
        Expr::Alternative { primary, fallback } => is_scalar(primary) && is_scalar(fallback),
        _ => false,
    }
}

// ============================================================================
// Flattened IR
// ============================================================================

type SlotId = u32;
type LabelId = u32;

#[derive(Debug, Clone)]
enum JitOp {
    // Value construction
    Clone { dst: SlotId, src: SlotId },
    Drop { slot: SlotId },
    Null { dst: SlotId },
    True { dst: SlotId },
    False { dst: SlotId },
    Num { dst: SlotId, val: f64, repr: Option<Rc<str>> },
    Str { dst: SlotId, val: String },

    // Operations (all write result to dst)
    Index { dst: SlotId, base: SlotId, key: SlotId },
    BinOp { dst: SlotId, op: BinOp, lhs: SlotId, rhs: SlotId },
    UnaryOp { dst: SlotId, op: UnaryOp, src: SlotId },
    Negate { dst: SlotId, src: SlotId },
    Not { dst: SlotId, src: SlotId },

    // Generator output
    Yield { output: SlotId },

    // Control flow
    IfTruthy { src: SlotId, then_label: LabelId, else_label: LabelId },
    Jump { label: LabelId },
    Label { id: LabelId },

    // Array/object iteration
    GetKind { dst_var: u32, src: SlotId },
    GetLen { dst_var: u32, src: SlotId },
    BranchKind { kind_var: u32, arr_label: LabelId, obj_label: LabelId, other_label: LabelId },
    LoopCheck { idx_var: u32, len_var: u32, body_label: LabelId, done_label: LabelId },
    ArrayGet { dst: SlotId, arr: SlotId, idx_var: u32 },
    ObjGetByIdx { dst: SlotId, obj: SlotId, idx_var: u32 },
    IncVar { var: u32 },

    // jq variables
    GetVar { dst: SlotId, var_index: u16 },
    SetVar { var_index: u16, src: SlotId },

    // Collect (array construction)
    CollectBegin,
    CollectPush { src: SlotId },
    CollectFinish { dst: SlotId },

    // Object construction
    ObjNew { dst: SlotId },
    ObjInsert { obj: SlotId, key: SlotId, val: SlotId },

    // Alternative (//)
    IsNullOrFalse { dst_var: u32, src: SlotId },
    BranchOnVar { var: u32, nonzero_label: LabelId, zero_label: LabelId },

    // Generic runtime alternative check
    Alternative { dst: SlotId, primary: SlotId },

    // Range loop
    ToF64Var { dst_var: u32, src: SlotId },
    F64Less { dst_var: u32, a_var: u32, b_var: u32 },
    F64Greater { dst_var: u32, a_var: u32, b_var: u32 },
    F64Add { dst_var: u32, a_var: u32, b_var: u32 },
    F64Num { dst: SlotId, src_var: u32 },
    /// Range check: (step>0 && cur<to) || (step<=0 && cur>to)
    RangeCheck { dst_var: u32, cur_var: u32, to_var: u32, step_var: u32 },

    // String interpolation
    StrBufNew,
    StrBufAppendLit { val: String },
    StrBufAppendVal { src: SlotId },
    StrBufFinish { dst: SlotId },

    // TryCatch support
    TryCatchBegin,
    TryCatchEnd,

    // Termination
    ReturnContinue,
    ReturnError,
}

struct Flattener {
    ops: Vec<JitOp>,
    next_slot: SlotId,
    next_label: LabelId,
    next_var: u32,
    collect_depth: u32,
    try_depth: u32,
}

impl Flattener {
    fn new() -> Self {
        Flattener { ops: Vec::new(), next_slot: 0, next_label: 0, next_var: 0,
                     collect_depth: 0, try_depth: 0 }
    }

    fn alloc_slot(&mut self) -> SlotId { let s = self.next_slot; self.next_slot += 1; s }
    fn alloc_label(&mut self) -> LabelId { let l = self.next_label; self.next_label += 1; l }
    fn alloc_var(&mut self) -> u32 { let v = self.next_var; self.next_var += 1; v }
    fn emit(&mut self, op: JitOp) { self.ops.push(op); }

    fn emit_yield(&mut self, output: SlotId) {
        if self.collect_depth > 0 {
            self.emit(JitOp::CollectPush { src: output });
        } else {
            self.emit(JitOp::Yield { output });
        }
    }

    fn emit_literal(&mut self, dst: SlotId, lit: &Literal) {
        match lit {
            Literal::Null => self.emit(JitOp::Null { dst }),
            Literal::True => self.emit(JitOp::True { dst }),
            Literal::False => self.emit(JitOp::False { dst }),
            Literal::Num(n, repr) => self.emit(JitOp::Num { dst, val: *n, repr: repr.clone() }),
            Literal::Str(s) => self.emit(JitOp::Str { dst, val: s.clone() }),
        }
    }

    /// Compile a scalar expression. Returns the SlotId containing the result (owned).
    /// The caller must Drop the slot when done.
    fn flatten_scalar(&mut self, expr: &Expr, input_slot: SlotId) -> SlotId {
        match expr {
            Expr::Input => {
                let out = self.alloc_slot();
                self.emit(JitOp::Clone { dst: out, src: input_slot });
                out
            }
            Expr::Literal(lit) => {
                let out = self.alloc_slot();
                self.emit_literal(out, lit);
                out
            }
            Expr::Not => {
                let out = self.alloc_slot();
                self.emit(JitOp::Not { dst: out, src: input_slot });
                out
            }
            Expr::Negate { operand } => {
                let val = self.flatten_scalar(operand, input_slot);
                let out = self.alloc_slot();
                self.emit(JitOp::Negate { dst: out, src: val });
                self.emit(JitOp::Drop { slot: val });
                out
            }
            Expr::BinOp { op, lhs, rhs } => {
                match op {
                    BinOp::And => {
                        let lhs_val = self.flatten_scalar(lhs, input_slot);
                        let out = self.alloc_slot();
                        let then_lbl = self.alloc_label();
                        let else_lbl = self.alloc_label();
                        let done_lbl = self.alloc_label();
                        self.emit(JitOp::IfTruthy { src: lhs_val, then_label: then_lbl, else_label: else_lbl });
                        self.emit(JitOp::Label { id: then_lbl });
                        self.emit(JitOp::Drop { slot: lhs_val });
                        let rhs_val = self.flatten_scalar(rhs, input_slot);
                        // out = is_truthy(rhs)
                        let t_lbl = self.alloc_label();
                        let f_lbl = self.alloc_label();
                        let end_lbl = self.alloc_label();
                        self.emit(JitOp::IfTruthy { src: rhs_val, then_label: t_lbl, else_label: f_lbl });
                        self.emit(JitOp::Label { id: t_lbl });
                        self.emit(JitOp::True { dst: out });
                        self.emit(JitOp::Jump { label: end_lbl });
                        self.emit(JitOp::Label { id: f_lbl });
                        self.emit(JitOp::False { dst: out });
                        self.emit(JitOp::Label { id: end_lbl });
                        self.emit(JitOp::Drop { slot: rhs_val });
                        self.emit(JitOp::Jump { label: done_lbl });

                        self.emit(JitOp::Label { id: else_lbl });
                        self.emit(JitOp::Drop { slot: lhs_val });
                        self.emit(JitOp::False { dst: out });
                        self.emit(JitOp::Label { id: done_lbl });
                        out
                    }
                    BinOp::Or => {
                        let lhs_val = self.flatten_scalar(lhs, input_slot);
                        let out = self.alloc_slot();
                        let then_lbl = self.alloc_label();
                        let else_lbl = self.alloc_label();
                        let done_lbl = self.alloc_label();
                        self.emit(JitOp::IfTruthy { src: lhs_val, then_label: then_lbl, else_label: else_lbl });
                        self.emit(JitOp::Label { id: then_lbl });
                        self.emit(JitOp::Drop { slot: lhs_val });
                        self.emit(JitOp::True { dst: out });
                        self.emit(JitOp::Jump { label: done_lbl });

                        self.emit(JitOp::Label { id: else_lbl });
                        self.emit(JitOp::Drop { slot: lhs_val });
                        let rhs_val = self.flatten_scalar(rhs, input_slot);
                        let t_lbl = self.alloc_label();
                        let f_lbl = self.alloc_label();
                        let end_lbl = self.alloc_label();
                        self.emit(JitOp::IfTruthy { src: rhs_val, then_label: t_lbl, else_label: f_lbl });
                        self.emit(JitOp::Label { id: t_lbl });
                        self.emit(JitOp::True { dst: out });
                        self.emit(JitOp::Jump { label: end_lbl });
                        self.emit(JitOp::Label { id: f_lbl });
                        self.emit(JitOp::False { dst: out });
                        self.emit(JitOp::Label { id: end_lbl });
                        self.emit(JitOp::Drop { slot: rhs_val });
                        self.emit(JitOp::Label { id: done_lbl });
                        out
                    }
                    _ => {
                        let l = self.flatten_scalar(lhs, input_slot);
                        let r = self.flatten_scalar(rhs, input_slot);
                        let out = self.alloc_slot();
                        self.emit(JitOp::BinOp { dst: out, op: *op, lhs: l, rhs: r });
                        self.emit(JitOp::Drop { slot: l });
                        self.emit(JitOp::Drop { slot: r });
                        out
                    }
                }
            }
            Expr::UnaryOp { op, operand } => {
                let val = self.flatten_scalar(operand, input_slot);
                let out = self.alloc_slot();
                self.emit(JitOp::UnaryOp { dst: out, op: *op, src: val });
                self.emit(JitOp::Drop { slot: val });
                out
            }
            Expr::Index { expr: base_expr, key: key_expr } => {
                let base = self.flatten_scalar(base_expr, input_slot);
                let key = self.flatten_scalar(key_expr, input_slot);
                let out = self.alloc_slot();
                self.emit(JitOp::Index { dst: out, base, key });
                self.emit(JitOp::Drop { slot: base });
                self.emit(JitOp::Drop { slot: key });
                out
            }
            Expr::IndexOpt { expr: base_expr, key: key_expr } => {
                // For scalar context, IndexOpt is same as Index (always produces one output)
                let base = self.flatten_scalar(base_expr, input_slot);
                let key = self.flatten_scalar(key_expr, input_slot);
                let out = self.alloc_slot();
                self.emit(JitOp::Index { dst: out, base, key });
                self.emit(JitOp::Drop { slot: base });
                self.emit(JitOp::Drop { slot: key });
                out
            }
            Expr::Pipe { left, right } => {
                let mid = self.flatten_scalar(left, input_slot);
                let out = self.flatten_scalar(right, mid);
                self.emit(JitOp::Drop { slot: mid });
                out
            }
            Expr::IfThenElse { cond, then_branch, else_branch } => {
                let cond_val = self.flatten_scalar(cond, input_slot);
                let out = self.alloc_slot();
                let then_lbl = self.alloc_label();
                let else_lbl = self.alloc_label();
                let done_lbl = self.alloc_label();
                self.emit(JitOp::IfTruthy { src: cond_val, then_label: then_lbl, else_label: else_lbl });
                self.emit(JitOp::Drop { slot: cond_val });

                self.emit(JitOp::Label { id: then_lbl });
                let then_val = self.flatten_scalar(then_branch, input_slot);
                self.emit(JitOp::Clone { dst: out, src: then_val });
                self.emit(JitOp::Drop { slot: then_val });
                self.emit(JitOp::Jump { label: done_lbl });

                self.emit(JitOp::Label { id: else_lbl });
                let else_val = self.flatten_scalar(else_branch, input_slot);
                self.emit(JitOp::Clone { dst: out, src: else_val });
                self.emit(JitOp::Drop { slot: else_val });
                self.emit(JitOp::Label { id: done_lbl });
                out
            }
            Expr::LetBinding { var_index, value, body } => {
                let val = self.flatten_scalar(value, input_slot);
                let old = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: old, var_index: *var_index });
                self.emit(JitOp::SetVar { var_index: *var_index, src: val });
                self.emit(JitOp::Drop { slot: val });
                let result = self.flatten_scalar(body, input_slot);
                self.emit(JitOp::SetVar { var_index: *var_index, src: old });
                self.emit(JitOp::Drop { slot: old });
                result
            }
            Expr::LoadVar { var_index } => {
                let out = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: out, var_index: *var_index });
                out
            }
            Expr::Alternative { primary, fallback } => {
                let pval = self.flatten_scalar(primary, input_slot);
                let is_nf = self.alloc_var();
                self.emit(JitOp::IsNullOrFalse { dst_var: is_nf, src: pval });
                let fb_lbl = self.alloc_label();
                let use_primary_lbl = self.alloc_label();
                let done_lbl = self.alloc_label();
                let out = self.alloc_slot();
                self.emit(JitOp::BranchOnVar { var: is_nf, nonzero_label: fb_lbl, zero_label: use_primary_lbl });
                // Use primary
                self.emit(JitOp::Label { id: use_primary_lbl });
                self.emit(JitOp::Clone { dst: out, src: pval });
                self.emit(JitOp::Drop { slot: pval });
                self.emit(JitOp::Jump { label: done_lbl });
                // Use fallback
                self.emit(JitOp::Label { id: fb_lbl });
                self.emit(JitOp::Drop { slot: pval });
                let fval = self.flatten_scalar(fallback, input_slot);
                self.emit(JitOp::Clone { dst: out, src: fval });
                self.emit(JitOp::Drop { slot: fval });
                self.emit(JitOp::Label { id: done_lbl });
                out
            }
            _ => {
                // Should not be called for non-scalar expressions
                let out = self.alloc_slot();
                self.emit(JitOp::Null { dst: out });
                out
            }
        }
    }

    /// Compile a generator expression. Emits Yield ops for each output.
    fn flatten_gen(&mut self, expr: &Expr, input_slot: SlotId) -> bool {
        // If scalar, just compute and yield once
        if is_scalar(expr) {
            let out = self.flatten_scalar(expr, input_slot);
            self.emit_yield(out);
            self.emit(JitOp::Drop { slot: out });
            return true;
        }

        match expr {
            Expr::Empty => true,

            Expr::Comma { left, right } => {
                if !self.flatten_gen(left, input_slot) { return false; }
                self.flatten_gen(right, input_slot)
            }

            Expr::Pipe { left, right } => {
                if is_scalar(left) {
                    // Left is scalar: compute it, use as input to right
                    let mid = self.flatten_scalar(left, input_slot);
                    let ok = self.flatten_gen(right, mid);
                    self.emit(JitOp::Drop { slot: mid });
                    ok
                } else {
                    // Left is generator: emit left's loop, apply right to each output
                    self.flatten_gen_pipe(left, right, input_slot)
                }
            }

            Expr::Each { input_expr } => {
                if is_scalar(input_expr) {
                    let container = self.flatten_scalar(input_expr, input_slot);
                    self.flatten_each_yield(container, false);
                    self.emit(JitOp::Drop { slot: container });
                    true
                } else {
                    false
                }
            }
            // EachOpt has correctness issues with input representation; fall back to interpreter
            Expr::EachOpt { .. } => false,

            Expr::IfThenElse { cond, then_branch, else_branch } => {
                if !is_scalar(cond) { return false; }
                // Pre-check: ensure both branches can be JIT-compiled
                {
                    let mut test = Flattener::new();
                    test.collect_depth = self.collect_depth;
                    test.try_depth = self.try_depth;
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen(then_branch, t_in) { return false; }
                    if !test.flatten_gen(else_branch, t_in) { return false; }
                }
                let cond_val = self.flatten_scalar(cond, input_slot);
                let then_lbl = self.alloc_label();
                let else_lbl = self.alloc_label();
                let done_lbl = self.alloc_label();
                self.emit(JitOp::IfTruthy { src: cond_val, then_label: then_lbl, else_label: else_lbl });
                self.emit(JitOp::Drop { slot: cond_val });

                self.emit(JitOp::Label { id: then_lbl });
                self.flatten_gen(then_branch, input_slot);
                self.emit(JitOp::Jump { label: done_lbl });

                self.emit(JitOp::Label { id: else_lbl });
                self.flatten_gen(else_branch, input_slot);
                self.emit(JitOp::Label { id: done_lbl });
                true
            }

            Expr::LetBinding { var_index, value, body } => {
                if is_scalar(value) {
                    let val = self.flatten_scalar(value, input_slot);
                    let old = self.alloc_slot();
                    self.emit(JitOp::GetVar { dst: old, var_index: *var_index });
                    self.emit(JitOp::SetVar { var_index: *var_index, src: val });
                    self.emit(JitOp::Drop { slot: val });
                    let ok = self.flatten_gen(body, input_slot);
                    self.emit(JitOp::SetVar { var_index: *var_index, src: old });
                    self.emit(JitOp::Drop { slot: old });
                    ok
                } else {
                    false
                }
            }

            Expr::Collect { generator } => {
                self.emit(JitOp::CollectBegin);
                self.collect_depth += 1;
                let ok = self.flatten_gen(generator, input_slot);
                self.collect_depth -= 1;
                if !ok { return false; }
                let out = self.alloc_slot();
                self.emit(JitOp::CollectFinish { dst: out });
                self.emit_yield(out);
                self.emit(JitOp::Drop { slot: out });
                true
            }

            Expr::ObjectConstruct { pairs } => {
                if !pairs.iter().all(|(k, v)| is_scalar(k) && is_scalar(v)) { return false; }
                let out = self.alloc_slot();
                self.emit(JitOp::ObjNew { dst: out });
                for (k, v) in pairs {
                    let kv = self.flatten_scalar(k, input_slot);
                    let vv = self.flatten_scalar(v, input_slot);
                    self.emit(JitOp::ObjInsert { obj: out, key: kv, val: vv });
                    self.emit(JitOp::Drop { slot: kv });
                    self.emit(JitOp::Drop { slot: vv });
                }
                self.emit_yield(out);
                self.emit(JitOp::Drop { slot: out });
                true
            }

            Expr::StringInterpolation { parts } => {
                if !parts.iter().all(|p| match p { StringPart::Literal(_) => true, StringPart::Expr(e) => is_scalar(e) }) {
                    return false;
                }
                self.emit(JitOp::StrBufNew);
                for part in parts {
                    match part {
                        StringPart::Literal(s) => {
                            self.emit(JitOp::StrBufAppendLit { val: s.clone() });
                        }
                        StringPart::Expr(e) => {
                            let val = self.flatten_scalar(e, input_slot);
                            self.emit(JitOp::StrBufAppendVal { src: val });
                            self.emit(JitOp::Drop { slot: val });
                        }
                    }
                }
                let out = self.alloc_slot();
                self.emit(JitOp::StrBufFinish { dst: out });
                self.emit_yield(out);
                self.emit(JitOp::Drop { slot: out });
                true
            }

            Expr::Reduce { source, init, var_index, acc_index, update } => {
                if !is_scalar(init) || !is_scalar(update) { return false; }
                let acc_val = self.flatten_scalar(init, input_slot);
                let old_acc = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: old_acc, var_index: *acc_index });
                self.emit(JitOp::SetVar { var_index: *acc_index, src: acc_val });
                self.emit(JitOp::Drop { slot: acc_val });
                let old_var = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: old_var, var_index: *var_index });
                if !self.flatten_gen_with_reduce(source, *var_index, *acc_index, update, input_slot) {
                    return false;
                }
                let out = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: out, var_index: *acc_index });
                self.emit(JitOp::SetVar { var_index: *acc_index, src: old_acc });
                self.emit(JitOp::Drop { slot: old_acc });
                self.emit(JitOp::SetVar { var_index: *var_index, src: old_var });
                self.emit(JitOp::Drop { slot: old_var });
                self.emit_yield(out);
                self.emit(JitOp::Drop { slot: out });
                true
            }

            Expr::Range { from, to, step } => {
                if !is_scalar(from) || !is_scalar(to) { return false; }
                if let Some(s) = step {
                    if !is_scalar(s) { return false; }
                }
                let from_val = self.flatten_scalar(from, input_slot);
                let to_val = self.flatten_scalar(to, input_slot);
                let step_val = if let Some(s) = step {
                    self.flatten_scalar(s, input_slot)
                } else {
                    let s = self.alloc_slot();
                    self.emit(JitOp::Num { dst: s, val: 1.0, repr: None });
                    s
                };
                let cur_var = self.alloc_var();
                let to_var = self.alloc_var();
                let step_var = self.alloc_var();
                let cmp_var = self.alloc_var();
                self.emit(JitOp::ToF64Var { dst_var: cur_var, src: from_val });
                self.emit(JitOp::ToF64Var { dst_var: to_var, src: to_val });
                self.emit(JitOp::ToF64Var { dst_var: step_var, src: step_val });
                self.emit(JitOp::Drop { slot: from_val });
                self.emit(JitOp::Drop { slot: to_val });
                self.emit(JitOp::Drop { slot: step_val });
                let head = self.alloc_label();
                let body = self.alloc_label();
                let done = self.alloc_label();
                self.emit(JitOp::Label { id: head });
                self.emit(JitOp::RangeCheck { dst_var: cmp_var, cur_var, to_var, step_var });
                self.emit(JitOp::BranchOnVar { var: cmp_var, nonzero_label: body, zero_label: done });
                self.emit(JitOp::Label { id: body });
                let num_slot = self.alloc_slot();
                self.emit(JitOp::F64Num { dst: num_slot, src_var: cur_var });
                self.emit_yield(num_slot);
                self.emit(JitOp::Drop { slot: num_slot });
                self.emit(JitOp::F64Add { dst_var: cur_var, a_var: cur_var, b_var: step_var });
                self.emit(JitOp::Jump { label: head });
                self.emit(JitOp::Label { id: done });
                true
            }

            // TryCatch disabled in JIT — error propagation not yet correct
            Expr::TryCatch { .. } => false,

            // Foreach disabled in JIT — needs accumulator-as-input for update, complex to get right
            Expr::Foreach { .. } => false,

            _ => false,
        }
    }

    /// Handle generator | anything: emit left generator's loop with right inline.
    fn flatten_gen_pipe(&mut self, left: &Expr, right: &Expr, input_slot: SlotId) -> bool {
        match left {
            Expr::Each { input_expr } | Expr::EachOpt { input_expr } => {
                let is_opt = matches!(left, Expr::EachOpt { .. });
                if !is_scalar(input_expr) { return false; }
                let container = self.flatten_scalar(input_expr, input_slot);
                // Emit loop, applying right to each element
                if !self.flatten_each_with_body(container, is_opt, right, input_slot) {
                    return false;
                }
                self.emit(JitOp::Drop { slot: container });
                true
            }
            Expr::Comma { left, right: r } => {
                // (A, B) | C = (A | C), (B | C)
                if !self.flatten_gen_pipe(left, right, input_slot) { return false; }
                self.flatten_gen_pipe(r, right, input_slot)
            }
            Expr::Pipe { left, right: r } => {
                if is_scalar(left) {
                    let mid = self.flatten_scalar(left, input_slot);
                    let ok = self.flatten_gen_pipe(r, right, mid);
                    self.emit(JitOp::Drop { slot: mid });
                    ok
                } else {
                    let composed = Expr::Pipe {
                        left: Box::new((**r).clone()),
                        right: Box::new(right.clone()),
                    };
                    self.flatten_gen_pipe(left, &composed, input_slot)
                }
            }
            Expr::IfThenElse { cond, then_branch, else_branch } => {
                if !is_scalar(cond) { return false; }
                let cond_val = self.flatten_scalar(cond, input_slot);
                let then_lbl = self.alloc_label();
                let else_lbl = self.alloc_label();
                let done_lbl = self.alloc_label();
                self.emit(JitOp::IfTruthy { src: cond_val, then_label: then_lbl, else_label: else_lbl });
                self.emit(JitOp::Drop { slot: cond_val });

                self.emit(JitOp::Label { id: then_lbl });
                let then_pipe = Expr::Pipe {
                    left: Box::new((**then_branch).clone()),
                    right: Box::new(right.clone()),
                };
                self.flatten_gen(&then_pipe, input_slot);
                self.emit(JitOp::Jump { label: done_lbl });

                self.emit(JitOp::Label { id: else_lbl });
                let else_pipe = Expr::Pipe {
                    left: Box::new((**else_branch).clone()),
                    right: Box::new(right.clone()),
                };
                self.flatten_gen(&else_pipe, input_slot);
                self.emit(JitOp::Label { id: done_lbl });
                true
            }
            Expr::Range { from, to, step } => {
                if !is_scalar(from) || !is_scalar(to) { return false; }
                if let Some(s) = step { if !is_scalar(s) { return false; } }
                let from_val = self.flatten_scalar(from, input_slot);
                let to_val = self.flatten_scalar(to, input_slot);
                let step_val = if let Some(s) = step {
                    self.flatten_scalar(s, input_slot)
                } else {
                    let s = self.alloc_slot();
                    self.emit(JitOp::Num { dst: s, val: 1.0, repr: None });
                    s
                };
                let cur = self.alloc_var();
                let to_v = self.alloc_var();
                let step_v = self.alloc_var();
                let cmp = self.alloc_var();
                self.emit(JitOp::ToF64Var { dst_var: cur, src: from_val });
                self.emit(JitOp::ToF64Var { dst_var: to_v, src: to_val });
                self.emit(JitOp::ToF64Var { dst_var: step_v, src: step_val });
                self.emit(JitOp::Drop { slot: from_val });
                self.emit(JitOp::Drop { slot: to_val });
                self.emit(JitOp::Drop { slot: step_val });
                let head = self.alloc_label();
                let body = self.alloc_label();
                let done = self.alloc_label();
                self.emit(JitOp::Label { id: head });
                self.emit(JitOp::RangeCheck { dst_var: cmp, cur_var: cur, to_var: to_v, step_var: step_v });
                self.emit(JitOp::BranchOnVar { var: cmp, nonzero_label: body, zero_label: done });
                self.emit(JitOp::Label { id: body });
                let num = self.alloc_slot();
                self.emit(JitOp::F64Num { dst: num, src_var: cur });
                // Apply right to each number
                self.flatten_gen(right, num);
                self.emit(JitOp::Drop { slot: num });
                self.emit(JitOp::F64Add { dst_var: cur, a_var: cur, b_var: step_v });
                self.emit(JitOp::Jump { label: head });
                self.emit(JitOp::Label { id: done });
                true
            }
            _ => false,
        }
    }

    /// Emit .[] iteration that yields each element directly.
    fn flatten_each_yield(&mut self, container: SlotId, is_opt: bool) {
        let kind_var = self.alloc_var();
        let arr_lbl = self.alloc_label();
        let obj_lbl = self.alloc_label();
        let other_lbl = self.alloc_label();
        let done_lbl = self.alloc_label();

        self.emit(JitOp::GetKind { dst_var: kind_var, src: container });
        self.emit(JitOp::BranchKind { kind_var, arr_label: arr_lbl, obj_label: obj_lbl, other_label: other_lbl });

        // Array loop
        self.emit(JitOp::Label { id: arr_lbl });
        {
            let idx = self.alloc_var();
            let len = self.alloc_var();
            let head = self.alloc_label();
            let body = self.alloc_label();
            let end = self.alloc_label();
            self.emit(JitOp::GetLen { dst_var: len, src: container });
            self.emit(JitOp::Label { id: head });
            self.emit(JitOp::LoopCheck { idx_var: idx, len_var: len, body_label: body, done_label: end });
            self.emit(JitOp::Label { id: body });
            let elem = self.alloc_slot();
            self.emit(JitOp::ArrayGet { dst: elem, arr: container, idx_var: idx });
            self.emit_yield(elem);
            self.emit(JitOp::Drop { slot: elem });
            self.emit(JitOp::IncVar { var: idx });
            self.emit(JitOp::Jump { label: head });
            self.emit(JitOp::Label { id: end });
        }
        self.emit(JitOp::Jump { label: done_lbl });

        // Object loop
        self.emit(JitOp::Label { id: obj_lbl });
        {
            let idx = self.alloc_var();
            let len = self.alloc_var();
            let head = self.alloc_label();
            let body = self.alloc_label();
            let end = self.alloc_label();
            self.emit(JitOp::GetLen { dst_var: len, src: container });
            self.emit(JitOp::Label { id: head });
            self.emit(JitOp::LoopCheck { idx_var: idx, len_var: len, body_label: body, done_label: end });
            self.emit(JitOp::Label { id: body });
            let elem = self.alloc_slot();
            self.emit(JitOp::ObjGetByIdx { dst: elem, obj: container, idx_var: idx });
            self.emit_yield(elem);
            self.emit(JitOp::Drop { slot: elem });
            self.emit(JitOp::IncVar { var: idx });
            self.emit(JitOp::Jump { label: head });
            self.emit(JitOp::Label { id: end });
        }
        self.emit(JitOp::Jump { label: done_lbl });

        // Other
        self.emit(JitOp::Label { id: other_lbl });
        if !is_opt { self.emit(JitOp::ReturnError); }
        self.emit(JitOp::Label { id: done_lbl });
    }

    /// Emit .[] iteration, applying `body_expr` to each element. Returns false if body can't be compiled.
    fn flatten_each_with_body(&mut self, container: SlotId, is_opt: bool, body_expr: &Expr, _original_input: SlotId) -> bool {
        // Pre-check: can we compile the body?
        {
            let mut test = Flattener::new();
            let test_input = test.alloc_slot();
            if !test.flatten_gen(body_expr, test_input) {
                return false;
            }
        }
        let kind_var = self.alloc_var();
        let arr_lbl = self.alloc_label();
        let obj_lbl = self.alloc_label();
        let other_lbl = self.alloc_label();
        let done_lbl = self.alloc_label();

        self.emit(JitOp::GetKind { dst_var: kind_var, src: container });
        self.emit(JitOp::BranchKind { kind_var, arr_label: arr_lbl, obj_label: obj_lbl, other_label: other_lbl });

        // Array loop with body
        self.emit(JitOp::Label { id: arr_lbl });
        {
            let idx = self.alloc_var();
            let len = self.alloc_var();
            let head = self.alloc_label();
            let body = self.alloc_label();
            let end = self.alloc_label();
            self.emit(JitOp::GetLen { dst_var: len, src: container });
            self.emit(JitOp::Label { id: head });
            self.emit(JitOp::LoopCheck { idx_var: idx, len_var: len, body_label: body, done_label: end });
            self.emit(JitOp::Label { id: body });
            let elem = self.alloc_slot();
            self.emit(JitOp::ArrayGet { dst: elem, arr: container, idx_var: idx });
            // Apply body_expr to elem
            self.flatten_gen(body_expr, elem);
            self.emit(JitOp::Drop { slot: elem });
            self.emit(JitOp::IncVar { var: idx });
            self.emit(JitOp::Jump { label: head });
            self.emit(JitOp::Label { id: end });
        }
        self.emit(JitOp::Jump { label: done_lbl });

        // Object loop with body
        self.emit(JitOp::Label { id: obj_lbl });
        {
            let idx = self.alloc_var();
            let len = self.alloc_var();
            let head = self.alloc_label();
            let body = self.alloc_label();
            let end = self.alloc_label();
            self.emit(JitOp::GetLen { dst_var: len, src: container });
            self.emit(JitOp::Label { id: head });
            self.emit(JitOp::LoopCheck { idx_var: idx, len_var: len, body_label: body, done_label: end });
            self.emit(JitOp::Label { id: body });
            let elem = self.alloc_slot();
            self.emit(JitOp::ObjGetByIdx { dst: elem, obj: container, idx_var: idx });
            self.flatten_gen(body_expr, elem);
            self.emit(JitOp::Drop { slot: elem });
            self.emit(JitOp::IncVar { var: idx });
            self.emit(JitOp::Jump { label: head });
            self.emit(JitOp::Label { id: end });
        }
        self.emit(JitOp::Jump { label: done_lbl });

        self.emit(JitOp::Label { id: other_lbl });
        if !is_opt { self.emit(JitOp::ReturnError); }
        self.emit(JitOp::Label { id: done_lbl });
        true
    }

    /// Emit reduce loop inline. Returns false if the source generator can't be compiled.
    fn flatten_gen_with_reduce(&mut self, source: &Expr, var_index: u16, acc_index: u16, update: &Expr, input_slot: SlotId) -> bool {
        // Helper: emit the update step (set $var, load acc as ., evaluate update, store back)
        let emit_update = |s: &mut Flattener, elem: SlotId| {
            s.emit(JitOp::SetVar { var_index, src: elem });
            // Load current accumulator as the input to update (. refers to acc in reduce)
            let acc = s.alloc_slot();
            s.emit(JitOp::GetVar { dst: acc, var_index: acc_index });
            let new_acc = s.flatten_scalar(update, acc);
            s.emit(JitOp::Drop { slot: acc });
            s.emit(JitOp::SetVar { var_index: acc_index, src: new_acc });
            s.emit(JitOp::Drop { slot: new_acc });
        };

        if is_scalar(source) {
            let val = self.flatten_scalar(source, input_slot);
            emit_update(self, val);
            self.emit(JitOp::Drop { slot: val });
            return true;
        }
        match source {
            Expr::Each { input_expr } | Expr::EachOpt { input_expr } => {
                if !is_scalar(input_expr) { return false; }
                let is_opt = matches!(source, Expr::EachOpt { .. });
                let container = self.flatten_scalar(input_expr, input_slot);
                self.flatten_each_with_action(container, is_opt, &|s, elem| {
                    emit_update(s, elem);
                });
                self.emit(JitOp::Drop { slot: container });
                true
            }
            Expr::Range { from, to, step } => {
                if !is_scalar(from) || !is_scalar(to) { return false; }
                if let Some(s) = step { if !is_scalar(s) { return false; } }
                let from_val = self.flatten_scalar(from, input_slot);
                let to_val = self.flatten_scalar(to, input_slot);
                let step_val = if let Some(s) = step {
                    self.flatten_scalar(s, input_slot)
                } else {
                    let s = self.alloc_slot();
                    self.emit(JitOp::Num { dst: s, val: 1.0, repr: None });
                    s
                };
                let cur = self.alloc_var();
                let to_v = self.alloc_var();
                let step_v = self.alloc_var();
                let cmp = self.alloc_var();
                self.emit(JitOp::ToF64Var { dst_var: cur, src: from_val });
                self.emit(JitOp::ToF64Var { dst_var: to_v, src: to_val });
                self.emit(JitOp::ToF64Var { dst_var: step_v, src: step_val });
                self.emit(JitOp::Drop { slot: from_val });
                self.emit(JitOp::Drop { slot: to_val });
                self.emit(JitOp::Drop { slot: step_val });
                let head = self.alloc_label();
                let body = self.alloc_label();
                let done = self.alloc_label();
                self.emit(JitOp::Label { id: head });
                self.emit(JitOp::RangeCheck { dst_var: cmp, cur_var: cur, to_var: to_v, step_var: step_v });
                self.emit(JitOp::BranchOnVar { var: cmp, nonzero_label: body, zero_label: done });
                self.emit(JitOp::Label { id: body });
                let num = self.alloc_slot();
                self.emit(JitOp::F64Num { dst: num, src_var: cur });
                emit_update(self, num);
                self.emit(JitOp::Drop { slot: num });
                self.emit(JitOp::F64Add { dst_var: cur, a_var: cur, b_var: step_v });
                self.emit(JitOp::Jump { label: head });
                self.emit(JitOp::Label { id: done });
                true
            }
            _ => false,
        }
    }

    /// Emit foreach loop: for each output of source, set $var, update acc, yield extract.
    fn flatten_foreach_loop(&mut self, source: &Expr, var_index: u16, acc_index: u16,
                            update: &Expr, extract: Option<&Expr>, input_slot: SlotId) {
        self.flatten_gen_with_action(source, input_slot, &|s, elem_slot| {
            s.emit(JitOp::SetVar { var_index, src: elem_slot });
            let new_acc = s.flatten_scalar(update, input_slot);
            s.emit(JitOp::SetVar { var_index: acc_index, src: new_acc });
            s.emit(JitOp::Drop { slot: new_acc });
            if let Some(ext) = extract {
                s.flatten_gen(ext, input_slot);
            } else {
                // Default: yield the accumulator
                let acc = s.alloc_slot();
                s.emit(JitOp::GetVar { dst: acc, var_index: acc_index });
                s.emit_yield(acc);
                s.emit(JitOp::Drop { slot: acc });
            }
        });
    }

    /// Generic: run a generator, but instead of yielding, execute an action for each output.
    fn flatten_gen_with_action(&mut self, expr: &Expr, input_slot: SlotId,
                               action: &dyn Fn(&mut Flattener, SlotId)) {
        if is_scalar(expr) {
            let out = self.flatten_scalar(expr, input_slot);
            action(self, out);
            self.emit(JitOp::Drop { slot: out });
            return;
        }
        match expr {
            Expr::Empty => {}
            Expr::Comma { left, right } => {
                self.flatten_gen_with_action(left, input_slot, action);
                self.flatten_gen_with_action(right, input_slot, action);
            }
            Expr::Pipe { left, right } => {
                if is_scalar(left) {
                    let mid = self.flatten_scalar(left, input_slot);
                    self.flatten_gen_with_action(right, mid, action);
                    self.emit(JitOp::Drop { slot: mid });
                } else {
                    // For gen | expr, handle each output of left
                    self.flatten_gen_with_action(left, input_slot, &|s, mid| {
                        if is_scalar(right) {
                            let out = s.flatten_scalar(right, mid);
                            action(s, out);
                            s.emit(JitOp::Drop { slot: out });
                        } else {
                            s.flatten_gen_with_action(right, mid, action);
                        }
                    });
                }
            }
            Expr::Each { input_expr } | Expr::EachOpt { input_expr } => {
                let is_opt = matches!(expr, Expr::EachOpt { .. });
                if is_scalar(input_expr) {
                    let container = self.flatten_scalar(input_expr, input_slot);
                    self.flatten_each_with_action(container, is_opt, action);
                    self.emit(JitOp::Drop { slot: container });
                }
            }
            Expr::Range { from, to, step } => {
                if is_scalar(from) && is_scalar(to) && step.as_ref().map_or(true, |s| is_scalar(s)) {
                    let from_val = self.flatten_scalar(from, input_slot);
                    let to_val = self.flatten_scalar(to, input_slot);
                    let step_val = if let Some(s) = step {
                        self.flatten_scalar(s, input_slot)
                    } else {
                        let s = self.alloc_slot();
                        self.emit(JitOp::Num { dst: s, val: 1.0, repr: None });
                        s
                    };
                    let cur = self.alloc_var();
                    let to_v = self.alloc_var();
                    let step_v = self.alloc_var();
                    let cmp = self.alloc_var();
                    self.emit(JitOp::ToF64Var { dst_var: cur, src: from_val });
                    self.emit(JitOp::ToF64Var { dst_var: to_v, src: to_val });
                    self.emit(JitOp::ToF64Var { dst_var: step_v, src: step_val });
                    self.emit(JitOp::Drop { slot: from_val });
                    self.emit(JitOp::Drop { slot: to_val });
                    self.emit(JitOp::Drop { slot: step_val });
                    let head = self.alloc_label();
                    let body = self.alloc_label();
                    let done = self.alloc_label();
                    self.emit(JitOp::Label { id: head });
                    self.emit(JitOp::RangeCheck { dst_var: cmp, cur_var: cur, to_var: to_v, step_var: step_v });
                    self.emit(JitOp::BranchOnVar { var: cmp, nonzero_label: body, zero_label: done });
                    self.emit(JitOp::Label { id: body });
                    let num = self.alloc_slot();
                    self.emit(JitOp::F64Num { dst: num, src_var: cur });
                    action(self, num);
                    self.emit(JitOp::Drop { slot: num });
                    self.emit(JitOp::F64Add { dst_var: cur, a_var: cur, b_var: step_v });
                    self.emit(JitOp::Jump { label: head });
                    self.emit(JitOp::Label { id: done });
                }
            }
            _ => {
                // Fallback: not supported
            }
        }
    }

    /// Emit .[] iteration, executing action for each element instead of yielding.
    fn flatten_each_with_action(&mut self, container: SlotId, is_opt: bool,
                                action: &dyn Fn(&mut Flattener, SlotId)) {
        let kind_var = self.alloc_var();
        let arr_lbl = self.alloc_label();
        let obj_lbl = self.alloc_label();
        let other_lbl = self.alloc_label();
        let done_lbl = self.alloc_label();

        self.emit(JitOp::GetKind { dst_var: kind_var, src: container });
        self.emit(JitOp::BranchKind { kind_var, arr_label: arr_lbl, obj_label: obj_lbl, other_label: other_lbl });

        // Array loop
        self.emit(JitOp::Label { id: arr_lbl });
        {
            let idx = self.alloc_var();
            let len = self.alloc_var();
            let head = self.alloc_label();
            let body = self.alloc_label();
            let end = self.alloc_label();
            self.emit(JitOp::GetLen { dst_var: len, src: container });
            self.emit(JitOp::Label { id: head });
            self.emit(JitOp::LoopCheck { idx_var: idx, len_var: len, body_label: body, done_label: end });
            self.emit(JitOp::Label { id: body });
            let elem = self.alloc_slot();
            self.emit(JitOp::ArrayGet { dst: elem, arr: container, idx_var: idx });
            action(self, elem);
            self.emit(JitOp::Drop { slot: elem });
            self.emit(JitOp::IncVar { var: idx });
            self.emit(JitOp::Jump { label: head });
            self.emit(JitOp::Label { id: end });
        }
        self.emit(JitOp::Jump { label: done_lbl });

        // Object loop
        self.emit(JitOp::Label { id: obj_lbl });
        {
            let idx = self.alloc_var();
            let len = self.alloc_var();
            let head = self.alloc_label();
            let body = self.alloc_label();
            let end = self.alloc_label();
            self.emit(JitOp::GetLen { dst_var: len, src: container });
            self.emit(JitOp::Label { id: head });
            self.emit(JitOp::LoopCheck { idx_var: idx, len_var: len, body_label: body, done_label: end });
            self.emit(JitOp::Label { id: body });
            let elem = self.alloc_slot();
            self.emit(JitOp::ObjGetByIdx { dst: elem, obj: container, idx_var: idx });
            action(self, elem);
            self.emit(JitOp::Drop { slot: elem });
            self.emit(JitOp::IncVar { var: idx });
            self.emit(JitOp::Jump { label: head });
            self.emit(JitOp::Label { id: end });
        }
        self.emit(JitOp::Jump { label: done_lbl });

        self.emit(JitOp::Label { id: other_lbl });
        if !is_opt { self.emit(JitOp::ReturnError); }
        self.emit(JitOp::Label { id: done_lbl });
    }
}

// ============================================================================
// Runtime helpers
// ============================================================================

extern "C" fn jit_rt_clone(dst: *mut Value, src: *const Value) {
    unsafe { std::ptr::write(dst, (*src).clone()); }
}
extern "C" fn jit_rt_drop(v: *mut Value) {
    unsafe { std::ptr::drop_in_place(v); }
}
extern "C" fn jit_rt_null(dst: *mut Value) {
    unsafe { std::ptr::write(dst, Value::Null); }
}
extern "C" fn jit_rt_true(dst: *mut Value) {
    unsafe { std::ptr::write(dst, Value::True); }
}
extern "C" fn jit_rt_false(dst: *mut Value) {
    unsafe { std::ptr::write(dst, Value::False); }
}
extern "C" fn jit_rt_num(dst: *mut Value, n: f64) {
    unsafe { std::ptr::write(dst, Value::Num(n, None)); }
}
extern "C" fn jit_rt_num_repr(dst: *mut Value, n: f64, repr_ptr: *const Rc<str>) {
    unsafe { std::ptr::write(dst, Value::Num(n, Some((*repr_ptr).clone()))); }
}
extern "C" fn jit_rt_str(dst: *mut Value, ptr: *const u8, len: usize) {
    unsafe {
        let s = std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len));
        std::ptr::write(dst, Value::Str(Rc::new(s.to_string())));
    }
}
extern "C" fn jit_rt_is_truthy(v: *const Value) -> i64 {
    unsafe { if (*v).is_truthy() { 1 } else { 0 } }
}
extern "C" fn jit_rt_index(dst: *mut Value, base: *const Value, key: *const Value) -> i64 {
    unsafe {
        match crate::eval::eval_index(&*base, &*key, false) {
            Ok(v) => { std::ptr::write(dst, v); 0 }
            Err(_) => { std::ptr::write(dst, Value::Null); GEN_ERROR }
        }
    }
}
extern "C" fn jit_rt_binop(dst: *mut Value, op: i32, lhs: *const Value, rhs: *const Value) -> i64 {
    unsafe {
        let binop = match op {
            0 => BinOp::Add, 1 => BinOp::Sub, 2 => BinOp::Mul, 3 => BinOp::Div,
            4 => BinOp::Mod, 5 => BinOp::Eq, 6 => BinOp::Ne, 7 => BinOp::Lt,
            8 => BinOp::Gt, 9 => BinOp::Le, 10 => BinOp::Ge,
            _ => { std::ptr::write(dst, Value::Null); return GEN_ERROR; }
        };
        match crate::eval::eval_binop(binop, &*lhs, &*rhs) {
            Ok(v) => { std::ptr::write(dst, v); 0 }
            Err(_) => { std::ptr::write(dst, Value::Null); GEN_ERROR }
        }
    }
}
extern "C" fn jit_rt_unaryop(dst: *mut Value, op: i32, input: *const Value) -> i64 {
    unsafe {
        match unaryop_from_i32(op) {
            Some(u) => match crate::eval::eval_unaryop(u, &*input) {
                Ok(v) => { std::ptr::write(dst, v); 0 }
                Err(_) => { std::ptr::write(dst, Value::Null); GEN_ERROR }
            },
            None => { std::ptr::write(dst, Value::Null); GEN_ERROR }
        }
    }
}
extern "C" fn jit_rt_negate(dst: *mut Value, input: *const Value) -> i64 {
    unsafe {
        match &*input {
            Value::Num(n, _) => { std::ptr::write(dst, Value::Num(-n, None)); 0 }
            _ => { std::ptr::write(dst, Value::Null); GEN_ERROR }
        }
    }
}
extern "C" fn jit_rt_not(dst: *mut Value, input: *const Value) -> i64 {
    unsafe {
        std::ptr::write(dst, if (*input).is_truthy() { Value::False } else { Value::True });
        0
    }
}
extern "C" fn jit_rt_kind(v: *const Value) -> i64 {
    unsafe {
        match &*v { Value::Null => 0, Value::False => 1, Value::True => 2, Value::Num(..) => 3,
            Value::Str(_) => 4, Value::Arr(_) => 5, Value::Obj(_) => 6, Value::Error(_) => 7 }
    }
}
extern "C" fn jit_rt_len(v: *const Value) -> i64 {
    unsafe {
        match &*v { Value::Arr(a) => a.len() as i64, Value::Obj(o) => o.len() as i64, _ => 0 }
    }
}
extern "C" fn jit_rt_array_get(dst: *mut Value, arr: *const Value, idx: i64) {
    unsafe {
        match &*arr {
            Value::Arr(a) if idx >= 0 && (idx as usize) < a.len() =>
                std::ptr::write(dst, a[idx as usize].clone()),
            _ => std::ptr::write(dst, Value::Null),
        }
    }
}
extern "C" fn jit_rt_obj_get_idx(dst: *mut Value, obj: *const Value, idx: i64) {
    unsafe {
        match &*obj {
            Value::Obj(o) => match o.get_index(idx as usize) {
                Some((_, v)) => std::ptr::write(dst, v.clone()),
                None => std::ptr::write(dst, Value::Null),
            },
            _ => std::ptr::write(dst, Value::Null),
        }
    }
}
extern "C" fn jit_rt_get_var(dst: *mut Value, env: *mut JitEnv, idx: u32) {
    unsafe {
        let env_ref = &*env;
        let v = env_ref.vars.get(idx as usize).cloned().unwrap_or(Value::Null);
        std::ptr::write(dst, v);
    }
}
extern "C" fn jit_rt_set_var(env: *mut JitEnv, idx: u32, val: *const Value) {
    unsafe {
        let i = idx as usize;
        let env = &mut *env;
        if i >= env.vars.len() { env.vars.resize(i + 1, Value::Null); }
        env.vars[i] = (*val).clone();
    }
}

// Collect (array construction) helpers
extern "C" fn jit_rt_collect_begin(env: *mut JitEnv) {
    unsafe { (*env).collect_stacks.push(Vec::new()); }
}
extern "C" fn jit_rt_collect_push(env: *mut JitEnv, val: *const Value) {
    unsafe { (*env).collect_stacks.last_mut().unwrap().push((*val).clone()); }
}
extern "C" fn jit_rt_collect_finish(dst: *mut Value, env: *mut JitEnv) {
    unsafe {
        let vec = (*env).collect_stacks.pop().unwrap();
        std::ptr::write(dst, Value::Arr(Rc::new(vec)));
    }
}

// Object construction helpers
extern "C" fn jit_rt_obj_new(dst: *mut Value) {
    unsafe { std::ptr::write(dst, Value::Obj(Rc::new(indexmap::IndexMap::new()))); }
}
extern "C" fn jit_rt_obj_insert(obj: *mut Value, key: *const Value, val: *const Value) {
    unsafe {
        if let Value::Obj(o) = &mut *obj {
            let k = match &*key {
                Value::Str(s) => s.as_str().to_string(),
                _ => crate::value::value_to_json(&*key),
            };
            Rc::make_mut(o).insert(k, (*val).clone());
        }
    }
}

// Alternative helper
extern "C" fn jit_rt_is_null_or_false(v: *const Value) -> i64 {
    unsafe { match &*v { Value::Null | Value::False => 1, _ => 0 } }
}

// Float helpers for Range
extern "C" fn jit_rt_to_f64(v: *const Value) -> f64 {
    unsafe { match &*v { Value::Num(n, _) => *n, _ => 0.0 } }
}
extern "C" fn jit_rt_f64_to_num(dst: *mut Value, n: f64) {
    unsafe { std::ptr::write(dst, Value::Num(n, None)); }
}
extern "C" fn jit_rt_range_check(cur: f64, to: f64, step: f64) -> i64 {
    if step > 0.0 { if cur < to { 1 } else { 0 } }
    else if step < 0.0 { if cur > to { 1 } else { 0 } }
    else { 0 }
}

// String interpolation helpers
extern "C" fn jit_rt_strbuf_new(env: *mut JitEnv) {
    unsafe { (*env).str_bufs.push(String::new()); }
}
extern "C" fn jit_rt_strbuf_append_lit(env: *mut JitEnv, ptr: *const u8, len: usize) {
    unsafe {
        let s = std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len));
        (*env).str_bufs.last_mut().unwrap().push_str(s);
    }
}
extern "C" fn jit_rt_strbuf_append_val(env: *mut JitEnv, val: *const Value) {
    unsafe {
        let s = match &*val {
            Value::Str(s) => s.as_str().to_string(),
            v => crate::value::value_to_json(v),
        };
        (*env).str_bufs.last_mut().unwrap().push_str(&s);
    }
}
extern "C" fn jit_rt_strbuf_finish(dst: *mut Value, env: *mut JitEnv) {
    unsafe {
        let s = (*env).str_bufs.pop().unwrap();
        std::ptr::write(dst, Value::Str(Rc::new(s)));
    }
}

// TryCatch helpers
extern "C" fn jit_rt_try_begin(env: *mut JitEnv) {
    unsafe { (*env).try_depth += 1; }
}
extern "C" fn jit_rt_try_end(env: *mut JitEnv) {
    unsafe { (*env).try_depth -= 1; }
}

fn binop_to_i32(op: BinOp) -> i32 {
    match op {
        BinOp::Add => 0, BinOp::Sub => 1, BinOp::Mul => 2, BinOp::Div => 3,
        BinOp::Mod => 4, BinOp::Eq => 5, BinOp::Ne => 6, BinOp::Lt => 7,
        BinOp::Gt => 8, BinOp::Le => 9, BinOp::Ge => 10, BinOp::And => 11, BinOp::Or => 12,
    }
}

fn unaryop_from_i32(op: i32) -> Option<UnaryOp> {
    Some(match op {
        0 => UnaryOp::Length, 1 => UnaryOp::Type, 2 => UnaryOp::Keys, 3 => UnaryOp::Values,
        4 => UnaryOp::Floor, 5 => UnaryOp::Ceil, 6 => UnaryOp::Round, 7 => UnaryOp::Fabs,
        8 => UnaryOp::Sort, 9 => UnaryOp::Reverse, 10 => UnaryOp::Unique, 11 => UnaryOp::Flatten,
        12 => UnaryOp::Min, 13 => UnaryOp::Max, 14 => UnaryOp::Add, 15 => UnaryOp::ToString,
        16 => UnaryOp::ToNumber, 17 => UnaryOp::ToJson, 18 => UnaryOp::FromJson, 19 => UnaryOp::Ascii,
        20 => UnaryOp::Explode, 21 => UnaryOp::Implode, 22 => UnaryOp::AsciiDowncase,
        23 => UnaryOp::AsciiUpcase, 24 => UnaryOp::Any, 25 => UnaryOp::All, 26 => UnaryOp::Transpose,
        27 => UnaryOp::ToEntries, 28 => UnaryOp::FromEntries, 29 => UnaryOp::KeysUnsorted,
        30 => UnaryOp::Sqrt, 31 => UnaryOp::Abs, 32 => UnaryOp::Not, 33 => UnaryOp::TypeOf,
        34 => UnaryOp::Infinite, 35 => UnaryOp::Nan, 36 => UnaryOp::IsInfinite, 37 => UnaryOp::IsNan,
        38 => UnaryOp::IsNormal, 39 => UnaryOp::IsFinite,
        _ => return None,
    })
}

fn unaryop_to_i32(op: UnaryOp) -> i32 {
    match op {
        UnaryOp::Length => 0, UnaryOp::Type => 1, UnaryOp::Keys => 2, UnaryOp::Values => 3,
        UnaryOp::Floor => 4, UnaryOp::Ceil => 5, UnaryOp::Round => 6, UnaryOp::Fabs => 7,
        UnaryOp::Sort => 8, UnaryOp::Reverse => 9, UnaryOp::Unique => 10, UnaryOp::Flatten => 11,
        UnaryOp::Min => 12, UnaryOp::Max => 13, UnaryOp::Add => 14, UnaryOp::ToString => 15,
        UnaryOp::ToNumber => 16, UnaryOp::ToJson => 17, UnaryOp::FromJson => 18, UnaryOp::Ascii => 19,
        UnaryOp::Explode => 20, UnaryOp::Implode => 21, UnaryOp::AsciiDowncase => 22,
        UnaryOp::AsciiUpcase => 23, UnaryOp::Any => 24, UnaryOp::All => 25, UnaryOp::Transpose => 26,
        UnaryOp::ToEntries => 27, UnaryOp::FromEntries => 28, UnaryOp::KeysUnsorted => 29,
        UnaryOp::Sqrt => 30, UnaryOp::Abs => 31, UnaryOp::Not => 32, UnaryOp::TypeOf => 33,
        UnaryOp::Infinite => 34, UnaryOp::Nan => 35, UnaryOp::IsInfinite => 36, UnaryOp::IsNan => 37,
        UnaryOp::IsNormal => 38, UnaryOp::IsFinite => 39,
        _ => 99, // catch-all for less common ops
    }
}

// ============================================================================
// JIT Environment & Compiler
// ============================================================================

pub struct JitEnv {
    pub vars: Vec<Value>,
    pub collect_stacks: Vec<Vec<Value>>,
    pub str_bufs: Vec<String>,
    pub try_depth: u32,
}

impl JitEnv {
    pub fn new() -> Self {
        JitEnv {
            vars: vec![Value::Null; 4096],
            collect_stacks: Vec::new(),
            str_bufs: Vec::new(),
            try_depth: 0,
        }
    }
}

pub type JitFilterFn = unsafe extern "C" fn(*const Value, *mut JitEnv,
    unsafe extern "C" fn(*const Value, *mut u8) -> i64, *mut u8) -> i64;

pub struct JitCompiler {
    module: JITModule,
    ctx: cranelift_codegen::Context,
    func_ctx: FunctionBuilderContext,
    rt_funcs: HashMap<&'static str, FuncId>,
    _string_constants: Vec<&'static str>,
    _repr_constants: Vec<Rc<str>>,
}

impl JitCompiler {
    pub fn new() -> Result<Self> {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed")?;
        let isa_builder = cranelift_native::builder().map_err(|e| anyhow::anyhow!("{}", e))?;
        let isa = isa_builder.finish(settings::Flags::new(flag_builder))?;

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        let symbols: &[(&str, *const u8)] = &[
            ("jit_rt_clone", jit_rt_clone as *const u8),
            ("jit_rt_drop", jit_rt_drop as *const u8),
            ("jit_rt_null", jit_rt_null as *const u8),
            ("jit_rt_true", jit_rt_true as *const u8),
            ("jit_rt_false", jit_rt_false as *const u8),
            ("jit_rt_num", jit_rt_num as *const u8),
            ("jit_rt_num_repr", jit_rt_num_repr as *const u8),
            ("jit_rt_str", jit_rt_str as *const u8),
            ("jit_rt_is_truthy", jit_rt_is_truthy as *const u8),
            ("jit_rt_index", jit_rt_index as *const u8),
            ("jit_rt_binop", jit_rt_binop as *const u8),
            ("jit_rt_unaryop", jit_rt_unaryop as *const u8),
            ("jit_rt_negate", jit_rt_negate as *const u8),
            ("jit_rt_not", jit_rt_not as *const u8),
            ("jit_rt_kind", jit_rt_kind as *const u8),
            ("jit_rt_len", jit_rt_len as *const u8),
            ("jit_rt_array_get", jit_rt_array_get as *const u8),
            ("jit_rt_obj_get_idx", jit_rt_obj_get_idx as *const u8),
            ("jit_rt_get_var", jit_rt_get_var as *const u8),
            ("jit_rt_set_var", jit_rt_set_var as *const u8),
            ("jit_rt_collect_begin", jit_rt_collect_begin as *const u8),
            ("jit_rt_collect_push", jit_rt_collect_push as *const u8),
            ("jit_rt_collect_finish", jit_rt_collect_finish as *const u8),
            ("jit_rt_obj_new", jit_rt_obj_new as *const u8),
            ("jit_rt_obj_insert", jit_rt_obj_insert as *const u8),
            ("jit_rt_is_null_or_false", jit_rt_is_null_or_false as *const u8),
            ("jit_rt_to_f64", jit_rt_to_f64 as *const u8),
            ("jit_rt_f64_to_num", jit_rt_f64_to_num as *const u8),
            ("jit_rt_range_check", jit_rt_range_check as *const u8),
            ("jit_rt_strbuf_new", jit_rt_strbuf_new as *const u8),
            ("jit_rt_strbuf_append_lit", jit_rt_strbuf_append_lit as *const u8),
            ("jit_rt_strbuf_append_val", jit_rt_strbuf_append_val as *const u8),
            ("jit_rt_strbuf_finish", jit_rt_strbuf_finish as *const u8),
            ("jit_rt_try_begin", jit_rt_try_begin as *const u8),
            ("jit_rt_try_end", jit_rt_try_end as *const u8),
        ];
        for (name, ptr) in symbols {
            jit_builder.symbol(*name, *ptr);
        }

        let mut module = JITModule::new(jit_builder);
        let mut rt_funcs = HashMap::new();
        declare_rt_funcs(&mut module, &mut rt_funcs)?;

        Ok(JitCompiler {
            module, ctx: cranelift_codegen::Context::new(),
            func_ctx: FunctionBuilderContext::new(), rt_funcs,
            _string_constants: Vec::new(), _repr_constants: Vec::new(),
        })
    }

    pub fn compile(&mut self, expr: &Expr) -> Result<JitFilterFn> {
        // Phase 1: Flatten
        let mut fl = Flattener::new();
        let input_slot = fl.alloc_slot(); // slot 0 = input ptr (special)
        let clone_slot = fl.alloc_slot(); // slot 1 = cloned input
        fl.emit(JitOp::Clone { dst: clone_slot, src: input_slot });
        if !fl.flatten_gen(expr, clone_slot) {
            bail!("Expression not JIT-compilable");
        }
        fl.emit(JitOp::Drop { slot: clone_slot });
        fl.emit(JitOp::ReturnContinue);

        let ops = fl.ops;
        let num_slots = fl.next_slot;
        let num_vars = fl.next_var;
        let num_labels = fl.next_label;

        // Phase 2: Cranelift codegen
        let ptr_ty = types::I64;
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty)); // input
        sig.params.push(AbiParam::new(ptr_ty)); // env
        sig.params.push(AbiParam::new(ptr_ty)); // cb
        sig.params.push(AbiParam::new(ptr_ty)); // ctx
        sig.returns.push(AbiParam::new(ptr_ty));

        let func_id = self.module.declare_anonymous_function(&sig)?;
        self.ctx.func.signature = sig;
        self.ctx.func.name = cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32());

        {
            let mut b = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            b.seal_block(entry);

            let params = b.block_params(entry).to_vec();
            let input_ptr = params[0];
            let env_ptr = params[1];
            let cb_fn = params[2];
            let cb_ctx = params[3];

            let val_size = std::mem::size_of::<Value>() as u32;
            let val_align = std::mem::align_of::<Value>() as u8;

            // Slot 0 = input_ptr (not a stack slot), slots 1..n are stack slots
            let mut slots = vec![None]; // slot 0
            for _ in 1..num_slots {
                slots.push(Some(b.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, val_size, val_align,
                ))));
            }

            let mut vars = Vec::new();
            for _ in 0..num_vars {
                let v = b.declare_var(ptr_ty);
                let zero = b.ins().iconst(ptr_ty, 0);
                b.def_var(v, zero);
                vars.push(v);
            }

            let mut label_blocks = Vec::new();
            for _ in 0..num_labels {
                label_blocks.push(b.create_block());
            }

            let cb_sig = {
                let mut s = self.module.make_signature();
                s.params.push(AbiParam::new(ptr_ty));
                s.params.push(AbiParam::new(ptr_ty));
                s.returns.push(AbiParam::new(ptr_ty));
                b.import_signature(s)
            };

            let mut rt: HashMap<&str, cranelift_codegen::ir::FuncRef> = HashMap::new();
            for (name, fid) in &self.rt_funcs {
                rt.insert(name, self.module.declare_func_in_func(*fid, b.func));
            }

            let slot_addr = |b: &mut FunctionBuilder, s: SlotId| -> cranelift_codegen::ir::Value {
                if s == 0 { input_ptr } else { b.ins().stack_addr(ptr_ty, slots[s as usize].unwrap(), 0) }
            };

            let mut terminated = false;

            for op in &ops {
                if terminated {
                    // After a return/jump, we need to be in a valid block
                    // Create a dead block to absorb any following ops until a Label
                    if !matches!(op, JitOp::Label { .. }) {
                        continue;
                    }
                }

                match op {
                    JitOp::Clone { dst, src } => {
                        let d = slot_addr(&mut b, *dst);
                        let s = slot_addr(&mut b, *src);
                        b.ins().call(rt["clone"], &[d, s]);
                        terminated = false;
                    }
                    JitOp::Drop { slot } if *slot == 0 => { /* don't drop input */ }
                    JitOp::Drop { slot } => {
                        let a = slot_addr(&mut b, *slot);
                        b.ins().call(rt["drop"], &[a]);
                        terminated = false;
                    }
                    JitOp::Null { dst } => {
                        let a = slot_addr(&mut b, *dst);
                        b.ins().call(rt["null"], &[a]);
                    }
                    JitOp::True { dst } => {
                        let a = slot_addr(&mut b, *dst);
                        b.ins().call(rt["true"], &[a]);
                    }
                    JitOp::False { dst } => {
                        let a = slot_addr(&mut b, *dst);
                        b.ins().call(rt["false"], &[a]);
                    }
                    JitOp::Num { dst, val, repr } => {
                        let a = slot_addr(&mut b, *dst);
                        let n = b.ins().f64const(*val);
                        if let Some(r) = repr {
                            self._repr_constants.push(r.clone());
                            let rp = b.ins().iconst(ptr_ty, self._repr_constants.last().unwrap() as *const Rc<str> as i64);
                            b.ins().call(rt["num_repr"], &[a, n, rp]);
                        } else {
                            b.ins().call(rt["num"], &[a, n]);
                        }
                    }
                    JitOp::Str { dst, val } => {
                        let a = slot_addr(&mut b, *dst);
                        let leaked = Box::leak(val.clone().into_boxed_str());
                        self._string_constants.push(leaked);
                        let sp = b.ins().iconst(ptr_ty, leaked.as_ptr() as i64);
                        let sl = b.ins().iconst(ptr_ty, leaked.len() as i64);
                        b.ins().call(rt["str"], &[a, sp, sl]);
                    }
                    JitOp::Index { dst, base, key } => {
                        let d = slot_addr(&mut b, *dst);
                        let ba = slot_addr(&mut b, *base);
                        let ka = slot_addr(&mut b, *key);
                        b.ins().call(rt["index"], &[d, ba, ka]);
                    }
                    JitOp::BinOp { dst, op, lhs, rhs } => {
                        let d = slot_addr(&mut b, *dst);
                        let l = slot_addr(&mut b, *lhs);
                        let r = slot_addr(&mut b, *rhs);
                        let o = b.ins().iconst(types::I32, binop_to_i32(*op) as i64);
                        b.ins().call(rt["binop"], &[d, o, l, r]);
                    }
                    JitOp::UnaryOp { dst, op, src } => {
                        let d = slot_addr(&mut b, *dst);
                        let s = slot_addr(&mut b, *src);
                        let o = b.ins().iconst(types::I32, unaryop_to_i32(*op) as i64);
                        b.ins().call(rt["unaryop"], &[d, o, s]);
                    }
                    JitOp::Negate { dst, src } => {
                        let d = slot_addr(&mut b, *dst);
                        let s = slot_addr(&mut b, *src);
                        b.ins().call(rt["negate"], &[d, s]);
                    }
                    JitOp::Not { dst, src } => {
                        let d = slot_addr(&mut b, *dst);
                        let s = slot_addr(&mut b, *src);
                        b.ins().call(rt["not"], &[d, s]);
                    }
                    JitOp::Yield { output } => {
                        let oa = slot_addr(&mut b, *output);
                        let call = b.ins().call_indirect(cb_sig, cb_fn, &[oa, cb_ctx]);
                        let result = b.inst_results(call)[0];
                        let zero = b.ins().iconst(ptr_ty, 0);
                        let stop = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::SignedLessThanOrEqual, result, zero);
                        let stop_blk = b.create_block();
                        let cont_blk = b.create_block();
                        b.ins().brif(stop, stop_blk, &[], cont_blk, &[]);
                        b.switch_to_block(stop_blk);
                        b.seal_block(stop_blk);
                        b.ins().return_(&[result]);
                        b.switch_to_block(cont_blk);
                        b.seal_block(cont_blk);
                    }
                    JitOp::IfTruthy { src, then_label, else_label } => {
                        let sa = slot_addr(&mut b, *src);
                        let call = b.ins().call(rt["is_truthy"], &[sa]);
                        let truthy = b.inst_results(call)[0];
                        let zero = b.ins().iconst(ptr_ty, 0);
                        let is_t = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, truthy, zero);
                        b.ins().brif(is_t, label_blocks[*then_label as usize], &[], label_blocks[*else_label as usize], &[]);
                        terminated = true;
                    }
                    JitOp::Jump { label } => {
                        b.ins().jump(label_blocks[*label as usize], &[]);
                        terminated = true;
                    }
                    JitOp::Label { id } => {
                        let block = label_blocks[*id as usize];
                        if !terminated {
                            b.ins().jump(block, &[]);
                        }
                        b.switch_to_block(block);
                        terminated = false;
                    }
                    JitOp::GetKind { dst_var, src } => {
                        let sa = slot_addr(&mut b, *src);
                        let call = b.ins().call(rt["kind"], &[sa]);
                        let kind = b.inst_results(call)[0];
                        b.def_var(vars[*dst_var as usize], kind);
                    }
                    JitOp::GetLen { dst_var, src } => {
                        let sa = slot_addr(&mut b, *src);
                        let call = b.ins().call(rt["len"], &[sa]);
                        let len = b.inst_results(call)[0];
                        b.def_var(vars[*dst_var as usize], len);
                    }
                    JitOp::BranchKind { kind_var, arr_label, obj_label, other_label } => {
                        let kind = b.use_var(vars[*kind_var as usize]);
                        let five = b.ins().iconst(ptr_ty, 5);
                        let is_arr = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, kind, five);
                        let not_arr = b.create_block();
                        b.ins().brif(is_arr, label_blocks[*arr_label as usize], &[], not_arr, &[]);
                        b.switch_to_block(not_arr);
                        b.seal_block(not_arr);
                        let six = b.ins().iconst(ptr_ty, 6);
                        let is_obj = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, kind, six);
                        b.ins().brif(is_obj, label_blocks[*obj_label as usize], &[], label_blocks[*other_label as usize], &[]);
                        terminated = true;
                    }
                    JitOp::LoopCheck { idx_var, len_var, body_label, done_label } => {
                        let i = b.use_var(vars[*idx_var as usize]);
                        let len = b.use_var(vars[*len_var as usize]);
                        let cmp = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, i, len);
                        b.ins().brif(cmp, label_blocks[*body_label as usize], &[], label_blocks[*done_label as usize], &[]);
                        terminated = true;
                    }
                    JitOp::ArrayGet { dst, arr, idx_var } => {
                        let d = slot_addr(&mut b, *dst);
                        let a = slot_addr(&mut b, *arr);
                        let i = b.use_var(vars[*idx_var as usize]);
                        b.ins().call(rt["array_get"], &[d, a, i]);
                    }
                    JitOp::ObjGetByIdx { dst, obj, idx_var } => {
                        let d = slot_addr(&mut b, *dst);
                        let o = slot_addr(&mut b, *obj);
                        let i = b.use_var(vars[*idx_var as usize]);
                        b.ins().call(rt["obj_get_idx"], &[d, o, i]);
                    }
                    JitOp::IncVar { var } => {
                        let v = b.use_var(vars[*var as usize]);
                        let one = b.ins().iconst(ptr_ty, 1);
                        let next = b.ins().iadd(v, one);
                        b.def_var(vars[*var as usize], next);
                    }
                    JitOp::GetVar { dst, var_index } => {
                        let d = slot_addr(&mut b, *dst);
                        let vi = b.ins().iconst(types::I32, *var_index as i64);
                        b.ins().call(rt["get_var"], &[d, env_ptr, vi]);
                    }
                    JitOp::SetVar { var_index, src } => {
                        let s = slot_addr(&mut b, *src);
                        let vi = b.ins().iconst(types::I32, *var_index as i64);
                        b.ins().call(rt["set_var"], &[env_ptr, vi, s]);
                    }
                    // Collect ops
                    JitOp::CollectBegin => {
                        b.ins().call(rt["collect_begin"], &[env_ptr]);
                    }
                    JitOp::CollectPush { src } => {
                        let s = slot_addr(&mut b, *src);
                        b.ins().call(rt["collect_push"], &[env_ptr, s]);
                    }
                    JitOp::CollectFinish { dst } => {
                        let d = slot_addr(&mut b, *dst);
                        b.ins().call(rt["collect_finish"], &[d, env_ptr]);
                    }

                    // Object ops
                    JitOp::ObjNew { dst } => {
                        let d = slot_addr(&mut b, *dst);
                        b.ins().call(rt["obj_new"], &[d]);
                    }
                    JitOp::ObjInsert { obj, key, val } => {
                        let o = slot_addr(&mut b, *obj);
                        let k = slot_addr(&mut b, *key);
                        let v = slot_addr(&mut b, *val);
                        b.ins().call(rt["obj_insert"], &[o, k, v]);
                    }

                    // Alternative
                    JitOp::IsNullOrFalse { dst_var, src } => {
                        let s = slot_addr(&mut b, *src);
                        let call = b.ins().call(rt["is_null_or_false"], &[s]);
                        let result = b.inst_results(call)[0];
                        b.def_var(vars[*dst_var as usize], result);
                    }
                    JitOp::BranchOnVar { var, nonzero_label, zero_label } => {
                        let v = b.use_var(vars[*var as usize]);
                        let zero = b.ins().iconst(ptr_ty, 0);
                        let cmp = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, v, zero);
                        b.ins().brif(cmp, label_blocks[*nonzero_label as usize], &[], label_blocks[*zero_label as usize], &[]);
                        terminated = true;
                    }
                    JitOp::Alternative { .. } => {
                        // Not used directly in codegen (handled by IsNullOrFalse + BranchOnVar)
                    }

                    // Float/Range ops
                    JitOp::ToF64Var { dst_var, src } => {
                        let s = slot_addr(&mut b, *src);
                        let call = b.ins().call(rt["to_f64"], &[s]);
                        let f_val = b.inst_results(call)[0];
                        // Store f64 as raw bits in i64 var
                        let bits = b.ins().bitcast(ptr_ty, cranelift_codegen::ir::MemFlags::new(), f_val);
                        b.def_var(vars[*dst_var as usize], bits);
                    }
                    JitOp::F64Less { dst_var, a_var, b_var: bv } => {
                        let a_bits = b.use_var(vars[*a_var as usize]);
                        let b_bits = b.use_var(vars[*bv as usize]);
                        let a_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), a_bits);
                        let b_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), b_bits);
                        let cmp = b.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::LessThan, a_f, b_f);
                        let result = b.ins().uextend(ptr_ty, cmp);
                        b.def_var(vars[*dst_var as usize], result);
                    }
                    JitOp::F64Greater { dst_var, a_var, b_var: bv } => {
                        let a_bits = b.use_var(vars[*a_var as usize]);
                        let b_bits = b.use_var(vars[*bv as usize]);
                        let a_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), a_bits);
                        let b_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), b_bits);
                        let cmp = b.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::GreaterThan, a_f, b_f);
                        let result = b.ins().uextend(ptr_ty, cmp);
                        b.def_var(vars[*dst_var as usize], result);
                    }
                    JitOp::F64Add { dst_var, a_var, b_var: bv } => {
                        let a_bits = b.use_var(vars[*a_var as usize]);
                        let b_bits = b.use_var(vars[*bv as usize]);
                        let a_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), a_bits);
                        let b_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), b_bits);
                        let sum = b.ins().fadd(a_f, b_f);
                        let bits = b.ins().bitcast(ptr_ty, cranelift_codegen::ir::MemFlags::new(), sum);
                        b.def_var(vars[*dst_var as usize], bits);
                    }
                    JitOp::RangeCheck { dst_var, cur_var, to_var, step_var } => {
                        let cur_bits = b.use_var(vars[*cur_var as usize]);
                        let to_bits = b.use_var(vars[*to_var as usize]);
                        let step_bits = b.use_var(vars[*step_var as usize]);
                        let cur_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), cur_bits);
                        let to_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), to_bits);
                        let step_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), step_bits);
                        let call = b.ins().call(rt["range_check"], &[cur_f, to_f, step_f]);
                        let result = b.inst_results(call)[0];
                        b.def_var(vars[*dst_var as usize], result);
                    }
                    JitOp::F64Num { dst, src_var } => {
                        let d = slot_addr(&mut b, *dst);
                        let bits = b.use_var(vars[*src_var as usize]);
                        let f_val = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), bits);
                        b.ins().call(rt["f64_to_num"], &[d, f_val]);
                    }

                    // String interpolation ops
                    JitOp::StrBufNew => {
                        b.ins().call(rt["strbuf_new"], &[env_ptr]);
                    }
                    JitOp::StrBufAppendLit { val } => {
                        let leaked = Box::leak(val.clone().into_boxed_str());
                        self._string_constants.push(leaked);
                        let sp = b.ins().iconst(ptr_ty, leaked.as_ptr() as i64);
                        let sl = b.ins().iconst(ptr_ty, leaked.len() as i64);
                        b.ins().call(rt["strbuf_append_lit"], &[env_ptr, sp, sl]);
                    }
                    JitOp::StrBufAppendVal { src } => {
                        let s = slot_addr(&mut b, *src);
                        b.ins().call(rt["strbuf_append_val"], &[env_ptr, s]);
                    }
                    JitOp::StrBufFinish { dst } => {
                        let d = slot_addr(&mut b, *dst);
                        b.ins().call(rt["strbuf_finish"], &[d, env_ptr]);
                    }

                    // TryCatch ops
                    JitOp::TryCatchBegin => {
                        b.ins().call(rt["try_begin"], &[env_ptr]);
                    }
                    JitOp::TryCatchEnd => {
                        b.ins().call(rt["try_end"], &[env_ptr]);
                    }

                    JitOp::ReturnContinue => {
                        let v = b.ins().iconst(ptr_ty, GEN_CONTINUE);
                        b.ins().return_(&[v]);
                        terminated = true;
                    }
                    JitOp::ReturnError => {
                        let v = b.ins().iconst(ptr_ty, GEN_ERROR);
                        b.ins().return_(&[v]);
                        terminated = true;
                    }
                }
            }

            // Seal all label blocks
            for block in &label_blocks {
                b.seal_block(*block);
            }

            if !terminated {
                let v = b.ins().iconst(ptr_ty, GEN_CONTINUE);
                b.ins().return_(&[v]);
            }

            b.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        let code_ptr = self.module.get_finalized_function(func_id);
        Ok(unsafe { std::mem::transmute(code_ptr) })
    }
}

fn declare_rt_funcs(module: &mut JITModule, map: &mut HashMap<&'static str, FuncId>) -> Result<()> {
    let p = types::I64;
    let i = types::I32;
    let f = types::F64;
    macro_rules! decl {
        ($name:expr, [$($pt:expr),*], [$($rt:expr),*]) => {{
            let mut sig = module.make_signature();
            $(sig.params.push(AbiParam::new($pt));)*
            $(sig.returns.push(AbiParam::new($rt));)*
            map.insert($name, module.declare_function(
                &format!("jit_rt_{}", $name), Linkage::Import, &sig)?);
        }};
    }
    decl!("clone", [p, p], []);
    decl!("drop", [p], []);
    decl!("null", [p], []);
    decl!("true", [p], []);
    decl!("false", [p], []);
    decl!("num", [p, f], []);
    decl!("num_repr", [p, f, p], []);
    decl!("str", [p, p, p], []);
    decl!("is_truthy", [p], [p]);
    decl!("index", [p, p, p], [p]);
    decl!("binop", [p, i, p, p], [p]);
    decl!("unaryop", [p, i, p], [p]);
    decl!("negate", [p, p], [p]);
    decl!("not", [p, p], [p]);
    decl!("kind", [p], [p]);
    decl!("len", [p], [p]);
    decl!("array_get", [p, p, p], []);
    decl!("obj_get_idx", [p, p, p], []);
    decl!("get_var", [p, p, i], []);
    decl!("set_var", [p, i, p], []);
    decl!("collect_begin", [p], []);
    decl!("collect_push", [p, p], []);
    decl!("collect_finish", [p, p], []);
    decl!("obj_new", [p], []);
    decl!("obj_insert", [p, p, p], []);
    decl!("is_null_or_false", [p], [p]);
    decl!("to_f64", [p], [f]);
    decl!("f64_to_num", [p, f], []);
    decl!("range_check", [f, f, f], [p]);
    decl!("strbuf_new", [p], []);
    decl!("strbuf_append_lit", [p, p, p], []);
    decl!("strbuf_append_val", [p, p], []);
    decl!("strbuf_finish", [p, p], []);
    decl!("try_begin", [p], []);
    decl!("try_end", [p], []);
    Ok(())
}

// ============================================================================
// Public API
// ============================================================================

pub fn is_jit_compilable(expr: &Expr) -> bool {
    let mut fl = Flattener::new();
    let input = fl.alloc_slot();
    fl.flatten_gen(expr, input)
}

unsafe extern "C" fn collect_callback(value: *const Value, ctx: *mut u8) -> i64 {
    let results = &mut *(ctx as *mut Vec<Value>);
    results.push((*value).clone());
    GEN_CONTINUE
}

pub fn execute_jit(func: JitFilterFn, input: &Value) -> Result<Vec<Value>> {
    let mut results: Vec<Value> = Vec::new();
    let mut env = JitEnv::new();
    let result = unsafe {
        func(input as *const Value, &mut env, collect_callback,
             &mut results as *mut Vec<Value> as *mut u8)
    };
    if result == GEN_ERROR { bail!("JIT execution error"); }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_identity() {
        let mut c = JitCompiler::new().unwrap();
        let f = c.compile(&Expr::Input).unwrap();
        let r = execute_jit(f, &Value::Num(42.0, None)).unwrap();
        assert_eq!(r.len(), 1);
        assert!(matches!(&r[0], Value::Num(n, _) if *n == 42.0));
    }

    #[test]
    fn test_jit_literal() {
        let mut c = JitCompiler::new().unwrap();
        let f = c.compile(&Expr::Literal(Literal::Str("hello".into()))).unwrap();
        let r = execute_jit(f, &Value::Null).unwrap();
        assert_eq!(r.len(), 1);
        assert!(matches!(&r[0], Value::Str(s) if s.as_str() == "hello"));
    }

    #[test]
    fn test_jit_pipe() {
        let mut c = JitCompiler::new().unwrap();
        let expr = Expr::Pipe {
            left: Box::new(Expr::Input),
            right: Box::new(Expr::Input),
        };
        let f = c.compile(&expr).unwrap();
        let r = execute_jit(f, &Value::Num(7.0, None)).unwrap();
        assert_eq!(r.len(), 1);
        assert!(matches!(&r[0], Value::Num(n, _) if *n == 7.0));
    }
}
