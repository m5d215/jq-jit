//! Cranelift JIT compiler for jq filters.
//!
//! Two-phase approach:
//! 1. Flatten IR expression tree into linear JitOps
//! 2. Generate Cranelift IR from JitOps
//!
//! Key design: expressions are classified as "scalar" (exactly one output)
//! or "generator" (zero or more outputs). Scalar expressions compile to
//! direct computation; generators compile to loops with Yield operations.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use anyhow::{Result, bail};

// Thread-local error storage for JIT runtime functions.
// When a fallible runtime function encounters an error, it stores the error
// message here. The CheckError JitOp checks this and branches to catch handlers.
// Global statics — safe because jq-jit is single-threaded (uses Rc, not Arc).
struct GlobalCell<T>(std::cell::UnsafeCell<T>);
unsafe impl<T> Sync for GlobalCell<T> {}

static JIT_LAST_ERROR: GlobalCell<Option<String>> = GlobalCell(std::cell::UnsafeCell::new(None));
/// Direct Value storage for `error` builtin — avoids Value→JSON→Value round-trip in try-catch.
static JIT_ERROR_VALUE: GlobalCell<Option<Value>> = GlobalCell(std::cell::UnsafeCell::new(None));
static JIT_CLOSURE_OPS: GlobalCell<Vec<Expr>> = GlobalCell(std::cell::UnsafeCell::new(Vec::new()));

fn set_jit_error(msg: String) {
    unsafe { *JIT_LAST_ERROR.0.get() = Some(msg); }
}

fn take_jit_error() -> Option<String> {
    unsafe { (*JIT_LAST_ERROR.0.get()).take() }
}

fn take_jit_error_value() -> Option<Value> {
    unsafe { (*JIT_ERROR_VALUE.0.get()).take() }
}

fn clear_jit_error() {
    unsafe {
        *JIT_LAST_ERROR.0.get() = None;
        *JIT_ERROR_VALUE.0.get() = None;
    }
}
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, StackSlotData, StackSlotKind};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, FuncId};

use crate::ir::*;
use crate::value::Value;

const GEN_CONTINUE: i64 = 1;
const GEN_ERROR: i64 = -1;

// ============================================================================
// Expression classification
// ============================================================================

fn is_jit_supported_unaryop(_op: UnaryOp) -> bool {
    // All unary ops are supported via runtime dispatch
    true
}

/// Returns true if expr always produces exactly one output value AND the JIT can handle it inline.
/// Conservative: only includes expressions we know the JIT handles correctly as scalars.
/// Check if a generator expression can be safely collected into an array
/// as a scalar operation. This is conservative: we only allow generators
/// that we know flatten_gen can handle.
fn can_scalar_collect(expr: &Expr) -> bool {
    if is_scalar(expr) { return true; }
    match expr {
        Expr::Comma { left, right } => can_scalar_collect(left) && can_scalar_collect(right),
        Expr::Each { input_expr } => is_scalar(input_expr),
        Expr::EachOpt { input_expr } => is_scalar(input_expr),
        Expr::Pipe { left, right } => {
            if is_scalar(left) { can_scalar_collect(right) }
            else { can_scalar_collect(left) && can_scalar_collect(right) }
        }
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            is_scalar(cond) && can_scalar_collect(then_branch) && can_scalar_collect(else_branch)
        }
        Expr::Empty => true,
        Expr::Range { from, to, step } => {
            is_scalar(from) && is_scalar(to) && step.as_ref().is_none_or(|s| is_scalar(s))
        }
        Expr::Collect { generator } => can_scalar_collect(generator),
        Expr::LetBinding { value, body, .. } => {
            (is_scalar(value) || can_scalar_collect(value)) && can_scalar_collect(body)
        }
        Expr::TryCatch { try_expr, catch_expr } => {
            can_scalar_collect(try_expr) && can_scalar_collect(catch_expr)
        }
        Expr::Error { msg } => msg.as_ref().is_none_or(|m| is_scalar(m)),
        _ => false,
    }
}

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
        Expr::Until { cond, update } => is_scalar(cond) && is_scalar(update),
        Expr::CallBuiltin { name, args } => {
            // Filter-argument builtins can't be treated as scalar
            match (name.as_str(), args.len()) {
                ("walk", _) | ("pick", _) | ("skip", _) | ("del", _) => false,
                ("add", 1) => false,
                _ => args.iter().all(is_scalar),
            }
        }
        Expr::ObjectConstruct { pairs } => pairs.iter().all(|(k, v)| is_scalar(k) && is_scalar(v)),
        Expr::Slice { expr, from, to } => is_scalar(expr) && from.as_ref().is_none_or(|f| is_scalar(f)) && to.as_ref().is_none_or(|t| is_scalar(t)),
        Expr::Format { expr, .. } => is_scalar(expr),
        Expr::SetPath { path, value } => is_scalar(path) && is_scalar(value),
        Expr::GetPath { path } => is_scalar(path),
        Expr::DelPaths { paths } => is_scalar(paths),
        // Collect is scalar: it produces exactly one value (an array).
        // flatten_scalar handles it via flatten_gen for the inner generator.
        Expr::Collect { generator } => can_scalar_collect(generator),
        Expr::Loc { .. } | Expr::Env | Expr::Builtins => true,
        Expr::Debug { expr } => is_scalar(expr),
        Expr::Stderr { expr } => is_scalar(expr),
        // TryCatch is scalar when both try and catch are scalar:
        // try produces one value or fails, catch produces one value on failure
        Expr::TryCatch { try_expr, catch_expr } => is_scalar(try_expr) && is_scalar(catch_expr),
        Expr::StringInterpolation { parts } => parts.iter().all(|p| match p {
            StringPart::Literal(_) => true,
            StringPart::Expr(e) => is_scalar(e),
        }),
        Expr::Reduce { source: _source, init, update, .. } => {
            // Reduce is scalar if init and update are scalar (source is the generator)
            is_scalar(init) && is_scalar(update)
        }
        // AllShort/AnyShort produce exactly one bool
        Expr::AllShort { predicate, .. } | Expr::AnyShort { predicate, .. } => is_scalar(predicate),
        // Assign with simple path and scalar value produces exactly one output
        Expr::Assign { path_expr, value_expr } => {
            extract_simple_path(path_expr).is_some() && is_scalar(value_expr)
        }
        // Update with simple path and scalar update produces exactly one output
        Expr::Update { path_expr, update_expr } => {
            extract_simple_path(path_expr).is_some() && is_scalar(update_expr)
        }
        _ => false,
    }
}

// ============================================================================
// Flattened IR
// ============================================================================

type SlotId = u32;
type LabelId = u32;

#[derive(Debug, Clone, Copy)]
enum MutateFn { Reverse, Sort }

#[derive(Debug, Clone)]
#[allow(dead_code)]
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
    IndexField { dst: SlotId, base: SlotId, field: String },
    BinOp { dst: SlotId, op: BinOp, lhs: SlotId, rhs: SlotId },
    /// Add with move: moves lhs out of slot (writes Null), enabling in-place mutation.
    AddMove { dst: SlotId, lhs: SlotId, rhs: SlotId },
    /// Combined two-field lookup + binop: .field_a OP .field_b without intermediate Values.
    FieldBinopField { dst: SlotId, base: SlotId, field_a: String, field_b: String, op: i32 },
    /// Combined field lookup + binop with pre-allocated constant: .field OP const (or const OP .field).
    /// field_is_lhs: true → field OP const, false → const OP field.
    /// const_ptr points to a Value that lives as long as JitCompiler (no per-call create/drop).
    FieldBinopConst { dst: SlotId, base: SlotId, field: String, const_ptr: *const Value, op: i32, field_is_lhs: bool },
    UnaryOp { dst: SlotId, op: UnaryOp, src: SlotId },
    Negate { dst: SlotId, src: SlotId },
    Not { dst: SlotId, src: SlotId },

    // Generator output
    Yield { output: SlotId },
    /// Fused field lookup + yield: borrows the field value from base without cloning.
    /// Replaces IndexField(dst) + Yield(dst) + Drop(dst).
    YieldFieldRef { base: SlotId, field: String },

    // Control flow
    IfTruthy { src: SlotId, then_label: LabelId, else_label: LabelId },
    /// Combined field access + truthiness check: avoids clone+drop of the field value.
    FieldIsTruthy { base: SlotId, field: String, then_label: LabelId, else_label: LabelId },
    /// Combined field access + numeric comparison + branch: avoids creating intermediate Values.
    FieldCmpNum { base: SlotId, field: String, value: f64, op: i32, then_label: LabelId, else_label: LabelId },
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
    /// Reset a variable to 0 (used to reinitialize loop counters)
    InitVar { var: u32 },

    // jq variables
    GetVar { dst: SlotId, var_index: u16 },
    SetVar { var_index: u16, src: SlotId },
    /// Take var value (move out, leave Null) — avoids clone, gives refcount = 1.
    TakeVar { dst: SlotId, var_index: u16 },

    /// In-place path extract: swap container[key] with Null, return old element.
    /// Container is modified in-place via Rc::make_mut (no clone if refcount == 1).
    PathExtract { element: SlotId, container: SlotId, key: SlotId },
    /// In-place path insert: set container[key] = val, consuming val (slot becomes Null).
    /// Container is modified in-place via Rc::make_mut (no clone if refcount == 1).
    PathInsert { container: SlotId, key: SlotId, val: SlotId },

    // Collect (array construction)
    CollectBegin,
    CollectPush { src: SlotId },
    CollectFinish { dst: SlotId },

    // Object construction
    ObjNew { dst: SlotId, cap: u16 },
    ObjInsert { obj: SlotId, key: SlotId, val: SlotId },
    ObjInsertStrKey { obj: SlotId, key: String, val: SlotId },
    ObjPushStrKey { obj: SlotId, key: String, val: SlotId },
    /// Fused field extraction + push: copies a field from src object directly into dst object.
    /// Eliminates intermediate slot allocation and one function call vs IndexField + ObjPushStrKey.
    ObjCopyField { obj: SlotId, src: SlotId, obj_key: String, src_field: String },
    /// Batch field extraction: build an object from multiple fields of src in a single pass.
    /// Replaces ObjNew + N×ObjCopyField when all fields are extracted from the same source.
    /// Each pair is (output_key, source_field).
    ObjFromFields { dst: SlotId, src: SlotId, pairs: Vec<(String, String)> },

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
    /// Set a Cranelift variable to f64 constant (stored as i64 bits)
    F64Const { dst_var: u32, val: f64 },

    // String interpolation
    StrBufNew,
    StrBufAppendLit { val: String },
    StrBufAppendVal { src: SlotId },
    StrBufFinish { dst: SlotId },

    // TryCatch support
    TryCatchBegin,
    TryCatchEnd,
    /// Check if the last fallible operation produced an error.
    /// If so, write the error value to error_dst and jump to catch_label.
    CheckError { error_dst: SlotId, catch_label: LabelId },

    // Error throwing
    ThrowError { msg: SlotId },

    // CallBuiltin: call a runtime builtin function
    // args[0] is input, args[1..] are the evaluated arguments
    CallBuiltin { dst: SlotId, name: String, args: Vec<SlotId> },

    /// In-place unary mutate: call runtime fn(v: *mut Value) -> i64
    /// Consumes input_slot and mutates the clone in-place.
    MutateInplace { slot: SlotId, func: MutateFn },

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
    /// When inside a try block, this holds (catch_label, error_slot).
    /// After each fallible op, CheckError is emitted to branch to catch_label on error.
    try_catch_target: Option<(LabelId, SlotId)>,
    /// Label targets for break: var_index → done_label
    label_targets: HashMap<u16, LabelId>,
    /// Available compiled functions for FuncCall
    funcs: Vec<CompiledFunc>,
    /// Limit state: when inside a `limit(n; gen)`, emit_yield increments counter and breaks.
    /// (counter_var, limit_var, one_var, done_label, collect_depth_at_install)
    /// All vars store f64 bits.
    limit_state: Option<(u32, u32, u32, LabelId, u32)>,
    /// Closure ops: key expressions for sort_by/group_by/etc.
    closure_ops: Vec<Expr>,
    /// Track which functions are currently being expanded (cycle detection)
    expanding_funcs: HashSet<usize>,
    /// Pre-allocated Value constants for fused ops (hoisted out of per-call code).
    #[allow(clippy::vec_box)]
    value_constants: Vec<Box<Value>>,
}

impl Flattener {
    fn new() -> Self {
        Flattener { ops: Vec::new(), next_slot: 0, next_label: 0, next_var: 0,
                     collect_depth: 0, try_depth: 0, try_catch_target: None,
                     label_targets: HashMap::new(), funcs: Vec::new(),
                     limit_state: None, closure_ops: Vec::new(),
                     expanding_funcs: HashSet::new(), value_constants: Vec::new() }
    }

    /// Materialize a Literal into a heap-allocated Value and return a stable pointer.
    /// The Value is stored in value_constants and lives as long as JitCompiler.
    fn hoist_literal(&mut self, lit: &Literal) -> *const Value {
        let val = match lit {
            Literal::Null => Value::Null,
            Literal::True => Value::True,
            Literal::False => Value::False,
            Literal::Num(n, repr) => Value::Num(*n, repr.clone()),
            Literal::Str(s) => Value::Str(crate::value::KeyStr::from(s.as_str())),
        };
        let boxed = Box::new(val);
        let ptr = &*boxed as *const Value;
        self.value_constants.push(boxed);
        ptr
    }

    /// Create a test Flattener inheriting compile-time context from self.
    fn test_flattener(&self) -> Self {
        let mut t = Flattener::new();
        t.collect_depth = self.collect_depth;
        t.try_depth = self.try_depth;
        t.try_catch_target = self.try_catch_target;
        t.label_targets = self.label_targets.clone();
        t.funcs = self.funcs.clone();
        t.limit_state = self.limit_state;
        t.closure_ops = self.closure_ops.clone();
        t.expanding_funcs = self.expanding_funcs.clone();
        t
    }

    /// Inline FuncCall nodes by substituting function bodies.
    /// Returns the expression with FuncCall replaced by the function body.
    /// If a recursive call is detected, returns the original expression unchanged.
    fn inline_func_calls(&self, expr: &Expr) -> Expr {
        match expr {
            Expr::FuncCall { func_id, args } => {
                if *func_id >= self.funcs.len() { return expr.clone(); }
                if self.expanding_funcs.contains(func_id) { return expr.clone(); }
                let func = &self.funcs[*func_id];
                let body = if !func.param_vars.is_empty() && !args.is_empty() {
                    crate::eval::substitute_params(&func.body, &func.param_vars, args)
                } else {
                    func.body.clone()
                };
                // Recursively inline any FuncCalls in the resulting body
                self.inline_func_calls(&body)
            }
            Expr::Pipe { left, right } => Expr::Pipe {
                left: Box::new(self.inline_func_calls(left)),
                right: Box::new(self.inline_func_calls(right)),
            },
            Expr::Comma { left, right } => Expr::Comma {
                left: Box::new(self.inline_func_calls(left)),
                right: Box::new(self.inline_func_calls(right)),
            },
            Expr::Index { expr: e, key } => Expr::Index {
                expr: Box::new(self.inline_func_calls(e)),
                key: Box::new(self.inline_func_calls(key)),
            },
            Expr::IndexOpt { expr: e, key } => Expr::IndexOpt {
                expr: Box::new(self.inline_func_calls(e)),
                key: Box::new(self.inline_func_calls(key)),
            },
            Expr::Each { input_expr } => Expr::Each {
                input_expr: Box::new(self.inline_func_calls(input_expr)),
            },
            Expr::EachOpt { input_expr } => Expr::EachOpt {
                input_expr: Box::new(self.inline_func_calls(input_expr)),
            },
            Expr::IfThenElse { cond, then_branch, else_branch } => Expr::IfThenElse {
                cond: Box::new(self.inline_func_calls(cond)),
                then_branch: Box::new(self.inline_func_calls(then_branch)),
                else_branch: Box::new(self.inline_func_calls(else_branch)),
            },
            Expr::LetBinding { var_index, value, body } => Expr::LetBinding {
                var_index: *var_index,
                value: Box::new(self.inline_func_calls(value)),
                body: Box::new(self.inline_func_calls(body)),
            },
            Expr::BinOp { op, lhs, rhs } => Expr::BinOp {
                op: *op,
                lhs: Box::new(self.inline_func_calls(lhs)),
                rhs: Box::new(self.inline_func_calls(rhs)),
            },
            Expr::Collect { generator } => Expr::Collect {
                generator: Box::new(self.inline_func_calls(generator)),
            },
            Expr::Assign { path_expr, value_expr } => Expr::Assign {
                path_expr: Box::new(self.inline_func_calls(path_expr)),
                value_expr: Box::new(self.inline_func_calls(value_expr)),
            },
            Expr::Update { path_expr, update_expr } => Expr::Update {
                path_expr: Box::new(self.inline_func_calls(path_expr)),
                update_expr: Box::new(self.inline_func_calls(update_expr)),
            },
            Expr::TryCatch { try_expr, catch_expr } => Expr::TryCatch {
                try_expr: Box::new(self.inline_func_calls(try_expr)),
                catch_expr: Box::new(self.inline_func_calls(catch_expr)),
            },
            Expr::Alternative { primary, fallback } => Expr::Alternative {
                primary: Box::new(self.inline_func_calls(primary)),
                fallback: Box::new(self.inline_func_calls(fallback)),
            },
            Expr::UnaryOp { op, operand } => Expr::UnaryOp {
                op: *op,
                operand: Box::new(self.inline_func_calls(operand)),
            },
            Expr::Negate { operand } => Expr::Negate {
                operand: Box::new(self.inline_func_calls(operand)),
            },
            Expr::While { cond, update } => Expr::While {
                cond: Box::new(self.inline_func_calls(cond)),
                update: Box::new(self.inline_func_calls(update)),
            },
            Expr::Until { cond, update } => Expr::Until {
                cond: Box::new(self.inline_func_calls(cond)),
                update: Box::new(self.inline_func_calls(update)),
            },
            Expr::Repeat { update } => Expr::Repeat {
                update: Box::new(self.inline_func_calls(update)),
            },
            Expr::Reduce { source, init, var_index, acc_index, update } => Expr::Reduce {
                source: Box::new(self.inline_func_calls(source)),
                init: Box::new(self.inline_func_calls(init)),
                var_index: *var_index,
                acc_index: *acc_index,
                update: Box::new(self.inline_func_calls(update)),
            },
            Expr::Foreach { source, init, var_index, acc_index, update, extract } => Expr::Foreach {
                source: Box::new(self.inline_func_calls(source)),
                init: Box::new(self.inline_func_calls(init)),
                var_index: *var_index,
                acc_index: *acc_index,
                update: Box::new(self.inline_func_calls(update)),
                extract: extract.as_ref().map(|e| Box::new(self.inline_func_calls(e))),
            },
            Expr::Range { from, to, step } => Expr::Range {
                from: Box::new(self.inline_func_calls(from)),
                to: Box::new(self.inline_func_calls(to)),
                step: step.as_ref().map(|s| Box::new(self.inline_func_calls(s))),
            },
            Expr::Limit { count, generator } => Expr::Limit {
                count: Box::new(self.inline_func_calls(count)),
                generator: Box::new(self.inline_func_calls(generator)),
            },
            Expr::AllShort { generator, predicate } => Expr::AllShort {
                generator: Box::new(self.inline_func_calls(generator)),
                predicate: Box::new(self.inline_func_calls(predicate)),
            },
            Expr::AnyShort { generator, predicate } => Expr::AnyShort {
                generator: Box::new(self.inline_func_calls(generator)),
                predicate: Box::new(self.inline_func_calls(predicate)),
            },
            Expr::Label { var_index, body } => Expr::Label {
                var_index: *var_index,
                body: Box::new(self.inline_func_calls(body)),
            },
            Expr::Break { var_index, value } => Expr::Break {
                var_index: *var_index,
                value: Box::new(self.inline_func_calls(value)),
            },
            Expr::CallBuiltin { name, args } => Expr::CallBuiltin {
                name: name.clone(),
                args: args.iter().map(|a| self.inline_func_calls(a)).collect(),
            },
            Expr::ObjectConstruct { pairs } => Expr::ObjectConstruct {
                pairs: pairs.iter().map(|(k, v)| (self.inline_func_calls(k), self.inline_func_calls(v))).collect(),
            },
            Expr::Slice { expr, from, to } => Expr::Slice {
                expr: Box::new(self.inline_func_calls(expr)),
                from: from.as_ref().map(|f| Box::new(self.inline_func_calls(f))),
                to: to.as_ref().map(|t| Box::new(self.inline_func_calls(t))),
            },
            Expr::StringInterpolation { parts } => Expr::StringInterpolation {
                parts: parts.iter().map(|p| match p {
                    StringPart::Literal(s) => StringPart::Literal(s.clone()),
                    StringPart::Expr(e) => StringPart::Expr(self.inline_func_calls(e)),
                }).collect(),
            },
            Expr::Format { name, expr } => Expr::Format {
                name: name.clone(),
                expr: Box::new(self.inline_func_calls(expr)),
            },
            Expr::SetPath { path, value } => Expr::SetPath {
                path: Box::new(self.inline_func_calls(path)),
                value: Box::new(self.inline_func_calls(value)),
            },
            Expr::GetPath { path } => Expr::GetPath {
                path: Box::new(self.inline_func_calls(path)),
            },
            Expr::DelPaths { paths } => Expr::DelPaths {
                paths: Box::new(self.inline_func_calls(paths)),
            },
            Expr::Debug { expr } => Expr::Debug {
                expr: Box::new(self.inline_func_calls(expr)),
            },
            Expr::Stderr { expr } => Expr::Stderr {
                expr: Box::new(self.inline_func_calls(expr)),
            },
            Expr::Error { msg } => Expr::Error {
                msg: msg.as_ref().map(|m| Box::new(self.inline_func_calls(m))),
            },
            Expr::Recurse { input_expr } => Expr::Recurse {
                input_expr: Box::new(self.inline_func_calls(input_expr)),
            },
            Expr::PathExpr { expr } => Expr::PathExpr {
                expr: Box::new(self.inline_func_calls(expr)),
            },
            Expr::ClosureOp { op, input_expr, key_expr } => Expr::ClosureOp {
                op: *op,
                input_expr: Box::new(self.inline_func_calls(input_expr)),
                key_expr: Box::new(self.inline_func_calls(key_expr)),
            },
            Expr::RegexTest { input_expr, re, flags } => Expr::RegexTest {
                input_expr: Box::new(self.inline_func_calls(input_expr)),
                re: Box::new(self.inline_func_calls(re)),
                flags: Box::new(self.inline_func_calls(flags)),
            },
            Expr::RegexMatch { input_expr, re, flags } => Expr::RegexMatch {
                input_expr: Box::new(self.inline_func_calls(input_expr)),
                re: Box::new(self.inline_func_calls(re)),
                flags: Box::new(self.inline_func_calls(flags)),
            },
            Expr::RegexCapture { input_expr, re, flags } => Expr::RegexCapture {
                input_expr: Box::new(self.inline_func_calls(input_expr)),
                re: Box::new(self.inline_func_calls(re)),
                flags: Box::new(self.inline_func_calls(flags)),
            },
            Expr::RegexScan { input_expr, re, flags } => Expr::RegexScan {
                input_expr: Box::new(self.inline_func_calls(input_expr)),
                re: Box::new(self.inline_func_calls(re)),
                flags: Box::new(self.inline_func_calls(flags)),
            },
            Expr::RegexSub { input_expr, re, tostr, flags } => Expr::RegexSub {
                input_expr: Box::new(self.inline_func_calls(input_expr)),
                re: Box::new(self.inline_func_calls(re)),
                tostr: Box::new(self.inline_func_calls(tostr)),
                flags: Box::new(self.inline_func_calls(flags)),
            },
            Expr::RegexGsub { input_expr, re, tostr, flags } => Expr::RegexGsub {
                input_expr: Box::new(self.inline_func_calls(input_expr)),
                re: Box::new(self.inline_func_calls(re)),
                tostr: Box::new(self.inline_func_calls(tostr)),
                flags: Box::new(self.inline_func_calls(flags)),
            },
            Expr::AlternativeDestructure { alternatives } => Expr::AlternativeDestructure {
                alternatives: alternatives.iter().map(|a| self.inline_func_calls(a)).collect(),
            },
            // Leaf nodes: Input, Literal, LoadVar, Empty, Not, Loc, Env, Builtins,
            // ReadInput, ReadInputs, ModuleMeta, GenLabel
            _ => expr.clone(),
        }
    }

    /// Extract field name if expr is `.field` on Input (for fused op detection).
    fn extract_input_field(expr: &Expr) -> Option<String> {
        if let Expr::Index { expr: base, key } | Expr::IndexOpt { expr: base, key } = expr {
            if matches!(base.as_ref(), Expr::Input) {
                if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                    return Some(field.clone());
                }
            }
        }
        None
    }

    fn alloc_slot(&mut self) -> SlotId { let s = self.next_slot; self.next_slot += 1; s }
    fn alloc_label(&mut self) -> LabelId { let l = self.next_label; self.next_label += 1; l }
    fn alloc_var(&mut self) -> u32 { let v = self.next_var; self.next_var += 1; v }

    fn emit(&mut self, op: JitOp) {
        // Check if this op is fallible (can produce errors)
        let is_fallible = matches!(&op,
            JitOp::Index { .. } | JitOp::IndexField { .. } |
            JitOp::BinOp { .. } | JitOp::AddMove { .. } | JitOp::UnaryOp { .. } |
            JitOp::Negate { .. } | JitOp::CallBuiltin { .. } |
            JitOp::MutateInplace { .. } | JitOp::ThrowError { .. }
        );
        self.ops.push(op);
        if is_fallible {
            if let Some((catch_label, error_slot)) = self.try_catch_target {
                self.ops.push(JitOp::CheckError { error_dst: error_slot, catch_label });
            }
        }
    }

    /// Emit a fallible op with explicit error propagation even outside try blocks.
    fn emit_propagating(&mut self, op: JitOp) {
        self.ops.push(op);
        if let Some((catch_label, error_slot)) = self.try_catch_target {
            self.ops.push(JitOp::CheckError { error_dst: error_slot, catch_label });
        } else {
            let error_slot = self.alloc_slot();
            let error_label = self.alloc_label();
            let ok_label = self.alloc_label();
            self.ops.push(JitOp::CheckError { error_dst: error_slot, catch_label: error_label });
            self.ops.push(JitOp::Jump { label: ok_label });
            self.ops.push(JitOp::Label { id: error_label });
            self.ops.push(JitOp::Drop { slot: error_slot });
            self.ops.push(JitOp::ReturnError);
            self.ops.push(JitOp::Label { id: ok_label });
        }
    }

    fn emit_yield(&mut self, output: SlotId) {
        if self.collect_depth > 0 {
            self.emit(JitOp::CollectPush { src: output });
        } else {
            self.emit(JitOp::Yield { output });
        }
        // If inside a limit, check counter after yield (all vars are f64-encoded)
        // Only check at the same collect_depth where the limit was installed.
        // This avoids incorrectly triggering limit checks for inner Collect operations.
        if let Some((counter_var, limit_var, one_var, done_label, install_depth)) = self.limit_state {
            if self.collect_depth == install_depth {
                self.ops.push(JitOp::F64Add { dst_var: counter_var, a_var: counter_var, b_var: one_var });
                let cmp = self.alloc_var();
                self.ops.push(JitOp::F64Less { dst_var: cmp, a_var: counter_var, b_var: limit_var });
                let cont_label = self.alloc_label();
                self.ops.push(JitOp::BranchOnVar { var: cmp, nonzero_label: cont_label, zero_label: done_label });
                self.ops.push(JitOp::Label { id: cont_label });
            }
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
                        // Optimization: .field_a OP .field_b on same input → FieldBinopField
                        let fused = Self::extract_input_field(lhs).and_then(|fa| {
                            Self::extract_input_field(rhs).map(|fb| (fa, fb))
                        });
                        if let Some((field_a, field_b)) = fused {
                            let out = self.alloc_slot();
                            self.emit(JitOp::FieldBinopField { dst: out, base: input_slot, field_a, field_b, op: binop_to_i32(*op) });
                            out
                        }
                        // Optimization: .field OP literal → FieldBinopConst
                        else if let Some(field) = Self::extract_input_field(lhs) {
                            if let Expr::Literal(lit) = rhs.as_ref() {
                                let const_ptr = self.hoist_literal(lit);
                                let out = self.alloc_slot();
                                self.emit(JitOp::FieldBinopConst { dst: out, base: input_slot, field, const_ptr, op: binop_to_i32(*op), field_is_lhs: true });
                                out
                            } else {
                                let l = self.flatten_scalar(lhs, input_slot);
                                let r = self.flatten_scalar(rhs, input_slot);
                                let out = self.alloc_slot();
                                if matches!(op, BinOp::Add) {
                                    self.emit(JitOp::AddMove { dst: out, lhs: l, rhs: r });
                                } else {
                                    self.emit(JitOp::BinOp { dst: out, op: *op, lhs: l, rhs: r });
                                }
                                self.emit(JitOp::Drop { slot: l });
                                self.emit(JitOp::Drop { slot: r });
                                out
                            }
                        }
                        // Optimization: literal OP .field → FieldBinopConst (reversed)
                        else if let Some(field) = Self::extract_input_field(rhs) {
                            if let Expr::Literal(lit) = lhs.as_ref() {
                                let const_ptr = self.hoist_literal(lit);
                                let out = self.alloc_slot();
                                self.emit(JitOp::FieldBinopConst { dst: out, base: input_slot, field, const_ptr, op: binop_to_i32(*op), field_is_lhs: false });
                                out
                            } else {
                                let l = self.flatten_scalar(lhs, input_slot);
                                let r = self.flatten_scalar(rhs, input_slot);
                                let out = self.alloc_slot();
                                if matches!(op, BinOp::Add) {
                                    self.emit(JitOp::AddMove { dst: out, lhs: l, rhs: r });
                                } else {
                                    self.emit(JitOp::BinOp { dst: out, op: *op, lhs: l, rhs: r });
                                }
                                self.emit(JitOp::Drop { slot: l });
                                self.emit(JitOp::Drop { slot: r });
                                out
                            }
                        }
                        else {
                            let l = self.flatten_scalar(lhs, input_slot);
                            let r = self.flatten_scalar(rhs, input_slot);
                            let out = self.alloc_slot();
                            if matches!(op, BinOp::Add) {
                                self.emit(JitOp::AddMove { dst: out, lhs: l, rhs: r });
                            } else {
                                self.emit(JitOp::BinOp { dst: out, op: *op, lhs: l, rhs: r });
                            }
                            self.emit(JitOp::Drop { slot: l });
                            self.emit(JitOp::Drop { slot: r });
                            out
                        }
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
            Expr::Index { expr: base_expr, key: key_expr } | Expr::IndexOpt { expr: base_expr, key: key_expr } => {
                // Optimization: if base is Input, borrow the input slot directly (skip clone+drop)
                let (base, base_owned) = if matches!(base_expr.as_ref(), Expr::Input) {
                    (input_slot, false)
                } else {
                    (self.flatten_scalar(base_expr, input_slot), true)
                };
                // Optimization: if key is a string literal, use IndexField
                if let Expr::Literal(Literal::Str(s)) = key_expr.as_ref() {
                    let out = self.alloc_slot();
                    self.emit(JitOp::IndexField { dst: out, base, field: s.clone() });
                    if base_owned { self.emit(JitOp::Drop { slot: base }); }
                    out
                } else {
                    let key = self.flatten_scalar(key_expr, input_slot);
                    let out = self.alloc_slot();
                    self.emit(JitOp::Index { dst: out, base, key });
                    if base_owned { self.emit(JitOp::Drop { slot: base }); }
                    self.emit(JitOp::Drop { slot: key });
                    out
                }
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
            Expr::Until { cond, update } => {
                // until(cond; update): loop { if cond then break; update }; yield current
                let current = self.alloc_slot();
                self.emit(JitOp::Clone { dst: current, src: input_slot });
                let head = self.alloc_label();
                let body = self.alloc_label();
                let done = self.alloc_label();
                self.emit(JitOp::Label { id: head });
                let cond_val = self.flatten_scalar(cond, current);
                self.emit(JitOp::IfTruthy { src: cond_val, then_label: done, else_label: body });
                self.emit(JitOp::Drop { slot: cond_val });
                self.emit(JitOp::Label { id: body });
                let new_val = self.flatten_scalar(update, current);
                self.emit(JitOp::Drop { slot: current });
                self.emit(JitOp::Clone { dst: current, src: new_val });
                self.emit(JitOp::Drop { slot: new_val });
                self.emit(JitOp::Jump { label: head });
                self.emit(JitOp::Label { id: done });
                self.emit(JitOp::Drop { slot: cond_val });
                current
            }
            Expr::ObjectConstruct { pairs } => {
                let out = self.alloc_slot();
                self.emit(JitOp::ObjNew { dst: out, cap: pairs.len() as u16 });
                // Check if all keys are distinct literal strings — if so, skip dedup
                let all_unique_str_keys = {
                    let str_keys: Vec<&str> = pairs.iter().filter_map(|(k, _)| {
                        if let Expr::Literal(Literal::Str(s)) = k { Some(s.as_str()) } else { None }
                    }).collect();
                    str_keys.len() == pairs.len() && {
                        let mut uniq = str_keys.clone();
                        uniq.sort_unstable();
                        uniq.dedup();
                        uniq.len() == str_keys.len()
                    }
                };
                for (k, v) in pairs {
                    // Fast path: literal string key avoids creating/dropping a Value
                    if let Expr::Literal(Literal::Str(s)) = k {
                        // Fused path: if value is .field from input, use ObjCopyField
                        // to combine field extraction + push into a single runtime call
                        let used_copy_field = if all_unique_str_keys {
                            if let Expr::Index { expr: base, key } | Expr::IndexOpt { expr: base, key } = v {
                                if matches!(base.as_ref(), Expr::Input) {
                                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                        self.emit(JitOp::ObjCopyField {
                                            obj: out, src: input_slot,
                                            obj_key: s.clone(), src_field: field.clone(),
                                        });
                                        true
                                    } else { false }
                                } else { false }
                            } else { false }
                        } else { false };
                        if !used_copy_field {
                            let vv = self.flatten_scalar(v, input_slot);
                            if all_unique_str_keys {
                                self.emit(JitOp::ObjPushStrKey { obj: out, key: s.clone(), val: vv });
                            } else {
                                self.emit(JitOp::ObjInsertStrKey { obj: out, key: s.clone(), val: vv });
                            }
                        }
                    } else {
                        let kv = self.flatten_scalar(k, input_slot);
                        let vv = self.flatten_scalar(v, input_slot);
                        self.emit(JitOp::ObjInsert { obj: out, key: kv, val: vv });
                    }
                    // ObjInsert/ObjInsertStrKey/ObjCopyField takes ownership of val slot (ptr::read), no Drop needed
                }
                out
            }
            Expr::CallBuiltin { name, args } => {
                // In-place mutation for unary builtins: reverse, sort
                // Clone input → out, drop input_slot (so Rc refcount → 1), mutate in-place
                let mutate_fn = if args.is_empty() {
                    match name.as_str() {
                        "reverse" => Some(MutateFn::Reverse),
                        "sort" => Some(MutateFn::Sort),
                        _ => None,
                    }
                } else {
                    None
                };
                if let Some(func) = mutate_fn {
                    let out = self.alloc_slot();
                    self.emit(JitOp::Clone { dst: out, src: input_slot });
                    // Drop input_slot so the clone becomes sole owner (refcount 1)
                    // → Rc::make_mut can mutate in-place without cloning inner data
                    self.emit(JitOp::Drop { slot: input_slot });
                    self.emit(JitOp::Null { dst: input_slot });
                    self.emit_propagating(JitOp::MutateInplace { slot: out, func });
                    return out;
                }

                let mut arg_slots = Vec::new();
                // First arg is always the input (.)
                let inp = self.alloc_slot();
                self.emit(JitOp::Clone { dst: inp, src: input_slot });
                arg_slots.push(inp);
                for arg in args {
                    arg_slots.push(self.flatten_scalar(arg, input_slot));
                }
                let out = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: out, name: name.clone(), args: arg_slots.clone() });
                for s in &arg_slots {
                    self.emit(JitOp::Drop { slot: *s });
                }
                out
            }
            Expr::Slice { expr: base_expr, from, to } => {
                let base = self.flatten_scalar(base_expr, input_slot);
                let from_slot = if let Some(f) = from {
                    self.flatten_scalar(f, input_slot)
                } else {
                    let s = self.alloc_slot();
                    self.emit(JitOp::Null { dst: s });
                    s
                };
                let to_slot = if let Some(t) = to {
                    self.flatten_scalar(t, input_slot)
                } else {
                    let s = self.alloc_slot();
                    self.emit(JitOp::Null { dst: s });
                    s
                };
                let out = self.alloc_slot();
                // Use CallBuiltin("_slice", [input, from, to]) - we define this in runtime
                self.emit(JitOp::CallBuiltin { dst: out, name: "_slice".to_string(), args: vec![base, from_slot, to_slot] });
                self.emit(JitOp::Drop { slot: base });
                self.emit(JitOp::Drop { slot: from_slot });
                self.emit(JitOp::Drop { slot: to_slot });
                out
            }
            Expr::Format { name, expr: format_expr } => {
                let val = self.flatten_scalar(format_expr, input_slot);
                let out = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: out, name: format!("@{}", name), args: vec![val] });
                self.emit(JitOp::Drop { slot: val });
                out
            }
            Expr::SetPath { path, value } => {
                let path_val = self.flatten_scalar(path, input_slot);
                let val = self.flatten_scalar(value, input_slot);
                let inp = self.alloc_slot();
                self.emit(JitOp::Clone { dst: inp, src: input_slot });
                let out = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: out, name: "setpath".to_string(), args: vec![inp, path_val, val] });
                self.emit(JitOp::Drop { slot: inp });
                self.emit(JitOp::Drop { slot: path_val });
                self.emit(JitOp::Drop { slot: val });
                out
            }
            Expr::GetPath { path } => {
                let path_val = self.flatten_scalar(path, input_slot);
                let inp = self.alloc_slot();
                self.emit(JitOp::Clone { dst: inp, src: input_slot });
                let out = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: out, name: "getpath".to_string(), args: vec![inp, path_val] });
                self.emit(JitOp::Drop { slot: inp });
                self.emit(JitOp::Drop { slot: path_val });
                out
            }
            Expr::DelPaths { paths } => {
                let paths_val = self.flatten_scalar(paths, input_slot);
                let inp = self.alloc_slot();
                self.emit(JitOp::Clone { dst: inp, src: input_slot });
                let out = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: out, name: "delpaths".to_string(), args: vec![inp, paths_val] });
                self.emit(JitOp::Drop { slot: inp });
                self.emit(JitOp::Drop { slot: paths_val });
                out
            }
            Expr::Loc { file, line } => {
                let out = self.alloc_slot();
                // Build {"file":"...","line":N} as a CallBuiltin
                self.emit(JitOp::CallBuiltin { dst: out, name: format!("__loc__:{}:{}", file, line), args: vec![] });
                out
            }
            Expr::Env => {
                let out = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: out, name: "__env__".to_string(), args: vec![] });
                out
            }
            Expr::Builtins => {
                let out = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: out, name: "__builtins__".to_string(), args: vec![] });
                out
            }
            Expr::Debug { expr } => {
                let val = self.flatten_scalar(expr, input_slot);
                let inp = self.alloc_slot();
                self.emit(JitOp::Clone { dst: inp, src: input_slot });
                let out = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: out, name: "debug".to_string(), args: vec![inp, val] });
                self.emit(JitOp::Drop { slot: inp });
                self.emit(JitOp::Drop { slot: val });
                out
            }
            Expr::Stderr { expr } => {
                let val = self.flatten_scalar(expr, input_slot);
                let inp = self.alloc_slot();
                self.emit(JitOp::Clone { dst: inp, src: input_slot });
                let out = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: out, name: "stderr".to_string(), args: vec![inp, val] });
                self.emit(JitOp::Drop { slot: inp });
                self.emit(JitOp::Drop { slot: val });
                out
            }
            Expr::TryCatch { try_expr, catch_expr } => {
                // Scalar try-catch: try produces one value, or catch produces one value
                let catch_label = self.alloc_label();
                let done_label = self.alloc_label();
                let error_slot = self.alloc_slot();
                let out = self.alloc_slot();

                let old_target = self.try_catch_target;
                self.try_catch_target = Some((catch_label, error_slot));
                self.try_depth += 1;
                self.emit(JitOp::TryCatchBegin);

                let try_val = self.flatten_scalar(try_expr, input_slot);
                self.emit(JitOp::Clone { dst: out, src: try_val });
                self.emit(JitOp::Drop { slot: try_val });
                self.emit(JitOp::TryCatchEnd);
                self.try_depth -= 1;
                self.try_catch_target = old_target;
                self.emit(JitOp::Jump { label: done_label });

                // Catch handler
                self.emit(JitOp::Label { id: catch_label });
                self.emit(JitOp::TryCatchEnd);
                let catch_val = self.flatten_scalar(catch_expr, error_slot);
                self.emit(JitOp::Clone { dst: out, src: catch_val });
                self.emit(JitOp::Drop { slot: catch_val });
                self.emit(JitOp::Drop { slot: error_slot });
                self.emit(JitOp::Label { id: done_label });
                out
            }
            Expr::StringInterpolation { parts } => {
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
                out
            }
            Expr::Reduce { source, init, var_index, acc_index, update } => {
                // Scalar reduce: init is scalar, update is scalar, source is a generator
                let acc_val = self.flatten_scalar(init, input_slot);
                let old_acc = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: old_acc, var_index: *acc_index });
                self.emit(JitOp::SetVar { var_index: *acc_index, src: acc_val });
                self.emit(JitOp::Drop { slot: acc_val });
                let old_var = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: old_var, var_index: *var_index });
                // Detect in-place update pattern: reduce ... (init; path |= f) or path += f
                let inplace_info = detect_inplace_update(update);

                // Detect last(g) pattern: reduce g as $x ([]; [$x]) — avoid per-iteration allocation
                let is_last_pattern = matches!(update.as_ref(),
                    Expr::Collect { generator } if matches!(generator.as_ref(),
                        Expr::LoadVar { var_index: vi } if *vi == *var_index
                    )
                );

                // Detect accumulator-add pattern: reduce ... (init; . + rhs) — in-place via TakeVar + AddMove
                let acc_add_rhs = if let Expr::BinOp { op, lhs, rhs } = update.as_ref() {
                    if matches!(op, BinOp::Add) && matches!(lhs.as_ref(), Expr::Input) && is_scalar(rhs) {
                        Some(rhs.as_ref())
                    } else { None }
                } else { None };

                if is_last_pattern {
                    // Optimized: reuse array buffer via TakeVar + PathInsert
                    self.flatten_gen_with_each_output(source, input_slot, &|this, elem| {
                        this.emit(JitOp::SetVar { var_index: *var_index, src: elem });
                        let acc = this.alloc_slot();
                        this.emit(JitOp::TakeVar { dst: acc, var_index: *acc_index });
                        let key = this.alloc_slot();
                        this.emit(JitOp::Num { dst: key, val: 0.0, repr: None });
                        let val = this.alloc_slot();
                        this.emit(JitOp::GetVar { dst: val, var_index: *var_index });
                        this.emit(JitOp::PathInsert { container: acc, key, val });
                        this.emit(JitOp::Drop { slot: val });
                        this.emit(JitOp::Drop { slot: key });
                        this.emit(JitOp::SetVar { var_index: *acc_index, src: acc });
                        this.emit(JitOp::Drop { slot: acc });
                    });
                } else if let Some(rhs_expr) = acc_add_rhs {
                    // Optimized: TakeVar + AddMove for `. + rhs` — in-place accumulator append
                    self.flatten_gen_with_each_output(source, input_slot, &|this, elem| {
                        this.emit(JitOp::SetVar { var_index: *var_index, src: elem });
                        let acc = this.alloc_slot();
                        this.emit(JitOp::TakeVar { dst: acc, var_index: *acc_index });
                        let rhs_val = this.flatten_scalar(rhs_expr, acc);
                        let result = this.alloc_slot();
                        this.emit(JitOp::AddMove { dst: result, lhs: acc, rhs: rhs_val });
                        this.emit(JitOp::Drop { slot: rhs_val });
                        this.emit(JitOp::Drop { slot: acc });
                        this.emit(JitOp::SetVar { var_index: *acc_index, src: result });
                        this.emit(JitOp::Drop { slot: result });
                    });
                } else if let Some(info) = inplace_info {
                    // Optimized: TakeVar + PathExtract/PathInsert avoids container cloning
                    self.flatten_gen_with_each_output(source, input_slot, &|this, elem| {
                        this.emit(JitOp::SetVar { var_index: *var_index, src: elem });
                        this.emit_reduce_update_with_lets(*acc_index, &info);
                    });
                } else {
                    // For each output of source, set $var and update acc
                    self.flatten_gen_with_each_output(source, input_slot, &|this, elem| {
                        this.emit(JitOp::SetVar { var_index: *var_index, src: elem });
                        let acc = this.alloc_slot();
                        this.emit(JitOp::GetVar { dst: acc, var_index: *acc_index });
                        let new_acc = this.flatten_scalar(update, acc);
                        this.emit(JitOp::SetVar { var_index: *acc_index, src: new_acc });
                        this.emit(JitOp::Drop { slot: new_acc });
                        this.emit(JitOp::Drop { slot: acc });
                    });
                }
                let out = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: out, var_index: *acc_index });
                self.emit(JitOp::SetVar { var_index: *acc_index, src: old_acc });
                self.emit(JitOp::Drop { slot: old_acc });
                self.emit(JitOp::SetVar { var_index: *var_index, src: old_var });
                self.emit(JitOp::Drop { slot: old_var });
                out
            }
            Expr::Collect { generator } => {
                // Collect is scalar: produce one array value.
                // Use CollectBegin/flatten_gen/CollectFinish.
                self.emit(JitOp::CollectBegin);
                self.collect_depth += 1;
                self.flatten_gen(generator, input_slot);
                self.collect_depth -= 1;
                let out = self.alloc_slot();
                self.emit(JitOp::CollectFinish { dst: out });
                out
            }
            Expr::AllShort { generator, predicate } => {
                // all(pred): iterate generator, short-circuit to false on first falsey predicate
                let result_var = self.alloc_var();
                // result = 1 (true)
                self.emit(JitOp::InitVar { var: result_var });
                self.emit(JitOp::IncVar { var: result_var });
                let done_label = self.alloc_label();
                self.flatten_gen_with_each_output(generator, input_slot, &|this, elem| {
                    let pred_result = this.flatten_scalar(predicate, elem);
                    let cont = this.alloc_label();
                    let fail = this.alloc_label();
                    this.emit(JitOp::IfTruthy { src: pred_result, then_label: cont, else_label: fail });
                    this.emit(JitOp::Label { id: fail });
                    this.emit(JitOp::Drop { slot: pred_result });
                    this.emit(JitOp::InitVar { var: result_var }); // result = 0 (false)
                    this.emit(JitOp::Jump { label: done_label });
                    this.emit(JitOp::Label { id: cont });
                    this.emit(JitOp::Drop { slot: pred_result });
                });
                self.emit(JitOp::Label { id: done_label });
                let out = self.alloc_slot();
                let true_lbl = self.alloc_label();
                let false_lbl = self.alloc_label();
                let end_lbl = self.alloc_label();
                self.emit(JitOp::BranchOnVar { var: result_var, nonzero_label: true_lbl, zero_label: false_lbl });
                self.emit(JitOp::Label { id: true_lbl });
                self.emit(JitOp::True { dst: out });
                self.emit(JitOp::Jump { label: end_lbl });
                self.emit(JitOp::Label { id: false_lbl });
                self.emit(JitOp::False { dst: out });
                self.emit(JitOp::Label { id: end_lbl });
                out
            }
            Expr::AnyShort { generator, predicate } => {
                // any(pred): iterate generator, short-circuit to true on first truthy predicate
                let result_var = self.alloc_var();
                self.emit(JitOp::InitVar { var: result_var }); // result = 0 (false)
                let done_label = self.alloc_label();
                self.flatten_gen_with_each_output(generator, input_slot, &|this, elem| {
                    let pred_result = this.flatten_scalar(predicate, elem);
                    let cont = this.alloc_label();
                    let success = this.alloc_label();
                    this.emit(JitOp::IfTruthy { src: pred_result, then_label: success, else_label: cont });
                    this.emit(JitOp::Label { id: success });
                    this.emit(JitOp::Drop { slot: pred_result });
                    this.emit(JitOp::IncVar { var: result_var }); // result = 1 (true)
                    this.emit(JitOp::Jump { label: done_label });
                    this.emit(JitOp::Label { id: cont });
                    this.emit(JitOp::Drop { slot: pred_result });
                });
                self.emit(JitOp::Label { id: done_label });
                let out = self.alloc_slot();
                let true_lbl = self.alloc_label();
                let false_lbl = self.alloc_label();
                let end_lbl = self.alloc_label();
                self.emit(JitOp::BranchOnVar { var: result_var, nonzero_label: true_lbl, zero_label: false_lbl });
                self.emit(JitOp::Label { id: true_lbl });
                self.emit(JitOp::True { dst: out });
                self.emit(JitOp::Jump { label: end_lbl });
                self.emit(JitOp::Label { id: false_lbl });
                self.emit(JitOp::False { dst: out });
                self.emit(JitOp::Label { id: end_lbl });
                out
            }
            Expr::FuncCall { func_id, args } if *func_id < self.funcs.len() && !self.expanding_funcs.contains(func_id) => {
                self.expanding_funcs.insert(*func_id);
                let func = &self.funcs[*func_id].clone();
                let body = if !func.param_vars.is_empty() && !args.is_empty() {
                    crate::eval::substitute_params(&func.body, &func.param_vars, args)
                } else {
                    func.body.clone()
                };
                let result = self.flatten_scalar(&body, input_slot);
                self.expanding_funcs.remove(func_id);
                result
            }
            // Assign with simple path: setpath(path, value)
            Expr::Assign { path_expr, value_expr } => {
                if let Some(path_components) = extract_simple_path(path_expr) {
                    let val = self.flatten_scalar(value_expr, input_slot);
                    let path_arr = self.build_path_array(&path_components, input_slot);
                    let inp = self.alloc_slot();
                    self.emit(JitOp::Clone { dst: inp, src: input_slot });
                    let out = self.alloc_slot();
                    self.emit_propagating(JitOp::CallBuiltin { dst: out, name: "setpath".to_string(), args: vec![inp, path_arr, val] });
                    self.emit(JitOp::Drop { slot: inp });
                    self.emit(JitOp::Drop { slot: path_arr });
                    self.emit(JitOp::Drop { slot: val });
                    out
                } else {
                    let out = self.alloc_slot();
                    self.emit(JitOp::Null { dst: out });
                    out
                }
            }
            // Update with simple path: getpath, apply, setpath
            Expr::Update { path_expr, update_expr } => {
                if let Some(path_components) = extract_simple_path(path_expr) {
                    let path_arr = self.build_path_array(&path_components, input_slot);
                    let inp = self.alloc_slot();
                    self.emit(JitOp::Clone { dst: inp, src: input_slot });
                    let path_clone = self.alloc_slot();
                    self.emit(JitOp::Clone { dst: path_clone, src: path_arr });
                    let old_val = self.alloc_slot();
                    self.emit_propagating(JitOp::CallBuiltin { dst: old_val, name: "getpath".to_string(), args: vec![inp, path_clone] });
                    self.emit(JitOp::Drop { slot: inp });
                    self.emit(JitOp::Drop { slot: path_clone });
                    let new_val = self.flatten_scalar(update_expr, old_val);
                    self.emit(JitOp::Drop { slot: old_val });
                    let inp2 = self.alloc_slot();
                    self.emit(JitOp::Clone { dst: inp2, src: input_slot });
                    let out = self.alloc_slot();
                    self.emit_propagating(JitOp::CallBuiltin { dst: out, name: "setpath".to_string(), args: vec![inp2, path_arr, new_val] });
                    self.emit(JitOp::Drop { slot: inp2 });
                    self.emit(JitOp::Drop { slot: new_val });
                    out
                } else {
                    let out = self.alloc_slot();
                    self.emit(JitOp::Null { dst: out });
                    out
                }
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

            Expr::Error { msg } => {
                if let Some(msg_expr) = msg {
                    if !is_scalar(msg_expr) { return false; }
                    let msg_val = self.flatten_scalar(msg_expr, input_slot);
                    self.emit(JitOp::ThrowError { msg: msg_val });
                    // Note: ThrowError is fallible, so if in try block,
                    // CheckError is auto-emitted and will jump to catch.
                    // If NOT in try block, add explicit ReturnError.
                    if self.try_catch_target.is_none() {
                        self.ops.push(JitOp::ReturnError);
                    }
                    self.emit(JitOp::Drop { slot: msg_val });
                } else {
                    let msg_val = self.alloc_slot();
                    self.emit(JitOp::Clone { dst: msg_val, src: input_slot });
                    self.emit(JitOp::ThrowError { msg: msg_val });
                    if self.try_catch_target.is_none() {
                        self.ops.push(JitOp::ReturnError);
                    }
                    self.emit(JitOp::Drop { slot: msg_val });
                }
                true
            }

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
                if matches!(input_expr.as_ref(), Expr::Input) {
                    // Borrow input directly — skip clone+drop
                    self.flatten_each_yield(input_slot, false);
                    true
                } else if is_scalar(input_expr) {
                    let container = self.flatten_scalar(input_expr, input_slot);
                    self.flatten_each_yield(container, false);
                    self.emit(JitOp::Drop { slot: container });
                    true
                } else {
                    false
                }
            }
            Expr::EachOpt { input_expr } => {
                if matches!(input_expr.as_ref(), Expr::Input) {
                    self.flatten_each_yield(input_slot, true);
                    true
                } else if is_scalar(input_expr) {
                    let container = self.flatten_scalar(input_expr, input_slot);
                    self.flatten_each_yield(container, true);
                    self.emit(JitOp::Drop { slot: container });
                    true
                } else {
                    false
                }
            }
            // IndexOpt: produces 0 or 1 outputs (suppresses errors)
            Expr::IndexOpt { expr: base_expr, key: key_expr } => {
                if is_scalar(base_expr) && is_scalar(key_expr) {
                    // Compile as: try (base[key]) catch empty
                    let try_catch = Expr::TryCatch {
                        try_expr: Box::new(Expr::Index {
                            expr: Box::new((**base_expr).clone()),
                            key: Box::new((**key_expr).clone()),
                        }),
                        catch_expr: Box::new(Expr::Empty),
                    };
                    return self.flatten_gen(&try_catch, input_slot);
                }
                // Non-scalar base: pipe base into try(.key) catch empty
                let inner = Expr::Pipe {
                    left: Box::new((**base_expr).clone()),
                    right: Box::new(Expr::TryCatch {
                        try_expr: Box::new(Expr::Index {
                            expr: Box::new(Expr::Input),
                            key: Box::new((**key_expr).clone()),
                        }),
                        catch_expr: Box::new(Expr::Empty),
                    }),
                };
                self.flatten_gen(&inner, input_slot)
            }

            Expr::IfThenElse { cond, then_branch, else_branch } => {
                // Pre-check: ensure both branches can be JIT-compiled
                {
                    let mut test = self.test_flattener();
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen(then_branch, t_in) { return false; }
                    if !test.flatten_gen(else_branch, t_in) { return false; }
                }
                if is_scalar(cond) {
                    let then_lbl = self.alloc_label();
                    let else_lbl = self.alloc_label();
                    let done_lbl = self.alloc_label();
                    // Optimization: if cond is .field OP num on input, use combined FieldCmpNum
                    let used_fused = if let Expr::BinOp { op, lhs, rhs } = cond.as_ref() {
                        if matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge) {
                            // .field OP num
                            let field_num = if let Expr::Index { expr: base, key } | Expr::IndexOpt { expr: base, key } = lhs.as_ref() {
                                if matches!(base.as_ref(), Expr::Input) {
                                    if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                        if let Expr::Literal(Literal::Num(n, _)) = rhs.as_ref() {
                                            Some((field.clone(), *n, binop_to_i32(*op)))
                                        } else { None }
                                    } else { None }
                                } else { None }
                            } else { None };
                            // num OP .field → flip the comparison
                            let field_num = field_num.or_else(|| {
                                if let Expr::Index { expr: base, key } | Expr::IndexOpt { expr: base, key } = rhs.as_ref() {
                                    if matches!(base.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                            if let Expr::Literal(Literal::Num(n, _)) = lhs.as_ref() {
                                                let flipped = match op {
                                                    BinOp::Lt => BinOp::Gt, BinOp::Gt => BinOp::Lt,
                                                    BinOp::Le => BinOp::Ge, BinOp::Ge => BinOp::Le,
                                                    _ => *op, // Eq, Ne are symmetric
                                                };
                                                Some((field.clone(), *n, binop_to_i32(flipped)))
                                            } else { None }
                                        } else { None }
                                    } else { None }
                                } else { None }
                            });
                            if let Some((field, value, op_code)) = field_num {
                                self.emit(JitOp::FieldCmpNum { base: input_slot, field, value, op: op_code, then_label: then_lbl, else_label: else_lbl });
                                true
                            } else { false }
                        } else { false }
                    } else { false };
                    // Optimization: if cond is .field on input, use combined FieldIsTruthy
                    let used_fused = used_fused || if let Expr::Index { expr: base, key } | Expr::IndexOpt { expr: base, key } = cond.as_ref() {
                        if matches!(base.as_ref(), Expr::Input) {
                            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                self.emit(JitOp::FieldIsTruthy { base: input_slot, field: field.clone(), then_label: then_lbl, else_label: else_lbl });
                                true
                            } else { false }
                        } else { false }
                    } else { false };
                    if !used_fused {
                        let cond_val = self.flatten_scalar(cond, input_slot);
                        self.emit(JitOp::IfTruthy { src: cond_val, then_label: then_lbl, else_label: else_lbl });
                        self.emit(JitOp::Drop { slot: cond_val });
                    }

                    self.emit(JitOp::Label { id: then_lbl });
                    self.flatten_gen(then_branch, input_slot);
                    self.emit(JitOp::Jump { label: done_lbl });

                    self.emit(JitOp::Label { id: else_lbl });
                    self.flatten_gen(else_branch, input_slot);
                    self.emit(JitOp::Label { id: done_lbl });
                    true
                } else {
                    // Generator condition: for each output of cond, evaluate then/else
                    // Emit: flatten cond as generator, for each output check truthy and run branch
                    self.flatten_gen_cond_if(cond, then_branch, else_branch, input_slot)
                }
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
                    // Generator value: for each output of value, set $var and run body
                    // body uses original input as ., not the value output
                    self.flatten_gen_let_binding(*var_index, value, body, input_slot)
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
                if pairs.iter().all(|(k, v)| is_scalar(k) && is_scalar(v)) {
                    let out = self.alloc_slot();
                    self.emit(JitOp::ObjNew { dst: out, cap: pairs.len() as u16 });
                    let all_unique_str_keys = {
                        let str_keys: Vec<&str> = pairs.iter().filter_map(|(k, _)| {
                            if let Expr::Literal(Literal::Str(s)) = k { Some(s.as_str()) } else { None }
                        }).collect();
                        str_keys.len() == pairs.len() && {
                            let mut uniq = str_keys.clone();
                            uniq.sort_unstable();
                            uniq.dedup();
                            uniq.len() == str_keys.len()
                        }
                    };
                    for (k, v) in pairs {
                        if let Expr::Literal(Literal::Str(s)) = k {
                            let used_copy_field = if all_unique_str_keys {
                                if let Expr::Index { expr: base, key } | Expr::IndexOpt { expr: base, key } = v {
                                    if matches!(base.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                                            self.emit(JitOp::ObjCopyField {
                                                obj: out, src: input_slot,
                                                obj_key: s.clone(), src_field: field.clone(),
                                            });
                                            true
                                        } else { false }
                                    } else { false }
                                } else { false }
                            } else { false };
                            if !used_copy_field {
                                let vv = self.flatten_scalar(v, input_slot);
                                if all_unique_str_keys {
                                    self.emit(JitOp::ObjPushStrKey { obj: out, key: s.clone(), val: vv });
                                } else {
                                    self.emit(JitOp::ObjInsertStrKey { obj: out, key: s.clone(), val: vv });
                                }
                            }
                        } else {
                            let kv = self.flatten_scalar(k, input_slot);
                            let vv = self.flatten_scalar(v, input_slot);
                            self.emit(JitOp::ObjInsert { obj: out, key: kv, val: vv });
                        }
                    }
                    self.emit_yield(out);
                    self.emit(JitOp::Drop { slot: out });
                    true
                } else {
                    // Has generator key/value: expand via recursive flattening
                    // For each pair with a generator, iterate its outputs
                    self.flatten_obj_construct_gen(pairs, 0, input_slot)
                }
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
                let step_scalar = step.as_ref().is_none_or(|s| is_scalar(s));
                if !is_scalar(from) || !is_scalar(to) || !step_scalar {
                    // Convert to nested evaluation:
                    // range(A; B; C) where some are generators =
                    //   A as from_val | B as to_val | C as step_val | range_scalar(from_val, to_val, step_val)
                    // We handle this by converting each generator arg into a
                    // Comma generator and evaluating as nested flatten_gen_with_each_output.
                    // But closures are tricky, so just delegate to a helper.
                    return self.flatten_range_gen(from, to, step.as_deref(), input_slot);
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
                self.emit_range_loop(from_val, to_val, step_val);
                self.emit(JitOp::Drop { slot: from_val });
                self.emit(JitOp::Drop { slot: to_val });
                self.emit(JitOp::Drop { slot: step_val });
                true
            }

            Expr::While { cond, update } => {
                if !is_scalar(cond) || !is_scalar(update) { return false; }
                let current = self.alloc_slot();
                self.emit(JitOp::Clone { dst: current, src: input_slot });
                let head = self.alloc_label();
                let body = self.alloc_label();
                let done = self.alloc_label();
                self.emit(JitOp::Label { id: head });
                let cond_val = self.flatten_scalar(cond, current);
                self.emit(JitOp::IfTruthy { src: cond_val, then_label: body, else_label: done });
                self.emit(JitOp::Drop { slot: cond_val });
                self.emit(JitOp::Label { id: body });
                self.emit_yield(current);
                let new_val = self.flatten_scalar(update, current);
                self.emit(JitOp::Drop { slot: current });
                self.emit(JitOp::Clone { dst: current, src: new_val });
                self.emit(JitOp::Drop { slot: new_val });
                self.emit(JitOp::Jump { label: head });
                self.emit(JitOp::Label { id: done });
                self.emit(JitOp::Drop { slot: cond_val });
                self.emit(JitOp::Drop { slot: current });
                true
            }

            Expr::Repeat { update } => {
                if !is_scalar(update) { return false; }
                let current = self.alloc_slot();
                self.emit(JitOp::Clone { dst: current, src: input_slot });
                let head = self.alloc_label();
                self.emit(JitOp::Label { id: head });
                self.emit_yield(current);
                let new_val = self.flatten_scalar(update, current);
                self.emit(JitOp::Drop { slot: current });
                self.emit(JitOp::Clone { dst: current, src: new_val });
                self.emit(JitOp::Drop { slot: new_val });
                self.emit(JitOp::Jump { label: head });
                // Note: no "done" label — this is an infinite loop.
                // Early termination happens via Yield returning stop.
                true
            }

            Expr::TryCatch { try_expr, catch_expr } => {
                self.flatten_gen_try_catch(try_expr, catch_expr, input_slot)
            }

            // BinOp with generator operands: iterate generators and compute BinOp for each combination
            Expr::BinOp { op, lhs, rhs } if !matches!(op, BinOp::And | BinOp::Or) => {
                // Convert to Pipe(generator, BinOp(Input, scalar)) patterns
                if is_scalar(rhs) {
                    // lhs is generator, rhs is scalar: for each output of lhs, yield lhs_out op rhs
                    let rhs_val = self.flatten_scalar(rhs, input_slot);
                    let _gen_body = Expr::BinOp {
                        op: *op,
                        lhs: Box::new(Expr::Input),
                        rhs: Box::new(Expr::Input), // placeholder
                    };
                    // Instead of the generic approach, manually emit:
                    // for each output x of lhs: yield x op rhs
                    let ok = self.flatten_gen_with_each_output(lhs, input_slot, &|s, elem| {
                        let out = s.alloc_slot();
                        if matches!(op, BinOp::Add) {
                            s.emit(JitOp::AddMove { dst: out, lhs: elem, rhs: rhs_val });
                        } else {
                            s.emit(JitOp::BinOp { dst: out, op: *op, lhs: elem, rhs: rhs_val });
                        }
                        s.emit_yield(out);
                        s.emit(JitOp::Drop { slot: out });
                    });
                    self.emit(JitOp::Drop { slot: rhs_val });
                    ok
                } else if is_scalar(lhs) {
                    // rhs is generator, lhs is scalar: for each output of rhs, yield lhs op rhs_out
                    let lhs_val = self.flatten_scalar(lhs, input_slot);
                    let ok = self.flatten_gen_with_each_output(rhs, input_slot, &|s, elem| {
                        let out = s.alloc_slot();
                        if matches!(op, BinOp::Add) {
                            s.emit(JitOp::AddMove { dst: out, lhs: lhs_val, rhs: elem });
                        } else {
                            s.emit(JitOp::BinOp { dst: out, op: *op, lhs: lhs_val, rhs: elem });
                        }
                        s.emit_yield(out);
                        s.emit(JitOp::Drop { slot: out });
                    });
                    self.emit(JitOp::Drop { slot: lhs_val });
                    ok
                } else {
                    // Both generators: rewrite as nested LetBinding
                    // jq evaluates rhs as outer, lhs as inner:
                    // (rhs_gen) as $__r | (lhs_gen) as $__l | $__l op $__r
                    let lhs_var: u16 = 10100;
                    let rhs_var: u16 = 10101;
                    let rewritten = Expr::LetBinding {
                        var_index: rhs_var,
                        value: Box::new((**rhs).clone()),
                        body: Box::new(Expr::LetBinding {
                            var_index: lhs_var,
                            value: Box::new((**lhs).clone()),
                            body: Box::new(Expr::BinOp {
                                op: *op,
                                lhs: Box::new(Expr::LoadVar { var_index: lhs_var }),
                                rhs: Box::new(Expr::LoadVar { var_index: rhs_var }),
                            }),
                        }),
                    };
                    self.flatten_gen(&rewritten, input_slot)
                }
            }

            // Negate with generator operand
            Expr::Negate { operand } => {
                self.flatten_gen_with_each_output(operand, input_slot, &|s, elem| {
                    let out = s.alloc_slot();
                    s.emit(JitOp::Negate { dst: out, src: elem });
                    s.emit_yield(out);
                    s.emit(JitOp::Drop { slot: out });
                })
            }

            // Index with generator operands
            Expr::Index { expr: base_expr, key: key_expr } if is_scalar(base_expr) && !is_scalar(key_expr) => {
                // base is scalar, key is generator
                let base = self.flatten_scalar(base_expr, input_slot);
                let ok = self.flatten_gen_with_each_output(key_expr, input_slot, &|s, key| {
                    let out = s.alloc_slot();
                    s.emit(JitOp::Index { dst: out, base, key });
                    s.emit_yield(out);
                    s.emit(JitOp::Drop { slot: out });
                });
                self.emit(JitOp::Drop { slot: base });
                ok
            }

            Expr::Foreach { source, init, var_index, acc_index, update, extract } if !is_scalar(init) => {
                // Generator init: rewrite as init as $tmp | foreach source as $var ($tmp; update; extract)
                if !is_scalar(update) { return false; }
                if let Some(ext) = extract.as_ref() {
                    let mut test = self.test_flattener();
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen(ext, t_in) { return false; }
                }
                let init_var: u16 = 10500;
                let rewritten = Expr::LetBinding {
                    var_index: init_var,
                    value: Box::new((**init).clone()),
                    body: Box::new(Expr::Foreach {
                        source: source.clone(),
                        init: Box::new(Expr::LoadVar { var_index: init_var }),
                        var_index: *var_index,
                        acc_index: *acc_index,
                        update: update.clone(),
                        extract: extract.clone(),
                    }),
                };
                self.flatten_gen(&rewritten, input_slot)
            }

            Expr::Foreach { source, init, var_index, acc_index, update, extract } => {
                if !is_scalar(update) { return false; }
                // Pre-check: if extract exists, it must be compilable as a generator
                if let Some(ext) = extract.as_ref() {
                    let mut test = self.test_flattener();
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen(ext, t_in) { return false; }
                }

                // Evaluate init → accumulator
                let acc_val = self.flatten_scalar(init, input_slot);
                let old_acc = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: old_acc, var_index: *acc_index });
                self.emit(JitOp::SetVar { var_index: *acc_index, src: acc_val });
                self.emit(JitOp::Drop { slot: acc_val });
                let old_var = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: old_var, var_index: *var_index });

                // The action for each source element: update acc, yield extract
                let vi = *var_index;
                let ai = *acc_index;
                let ok = self.flatten_gen_with_foreach(source, vi, ai, update, extract.as_deref(), input_slot);

                // Restore vars
                self.emit(JitOp::SetVar { var_index: *var_index, src: old_var });
                self.emit(JitOp::Drop { slot: old_var });
                self.emit(JitOp::SetVar { var_index: *acc_index, src: old_acc });
                self.emit(JitOp::Drop { slot: old_acc });
                ok
            }

            Expr::Assign { path_expr, value_expr } => {
                // Inline FuncCalls to make expressions suitable for runtime delegation
                let inlined_path = self.inline_func_calls(path_expr);
                let inlined_value = self.inline_func_calls(value_expr);
                let path_expr = &inlined_path;
                let value_expr = &inlined_value;
                if let Some(path_components) = extract_simple_path(path_expr) {
                    if is_scalar(value_expr) {
                        // Simple static path with scalar value
                        let val = self.flatten_scalar(value_expr, input_slot);
                        let path_arr = self.build_path_array(&path_components, input_slot);
                        let inp = self.alloc_slot();
                        self.emit(JitOp::Clone { dst: inp, src: input_slot });
                        let out = self.alloc_slot();
                        self.emit_propagating(JitOp::CallBuiltin { dst: out, name: "setpath".to_string(), args: vec![inp, path_arr, val] });
                        self.emit(JitOp::Drop { slot: inp });
                        self.emit(JitOp::Drop { slot: path_arr });
                        self.emit(JitOp::Drop { slot: val });
                        self.emit_yield(out);
                        self.emit(JitOp::Drop { slot: out });
                        return true;
                    }
                    // Generator value: for each value output, assign
                    let ok = self.flatten_gen_with_each_output(value_expr, input_slot, &|s, val| {
                        let path_arr = s.build_path_array(&path_components, input_slot);
                        let inp = s.alloc_slot();
                        s.emit(JitOp::Clone { dst: inp, src: input_slot });
                        let out = s.alloc_slot();
                        s.emit_propagating(JitOp::CallBuiltin { dst: out, name: "setpath".to_string(), args: vec![inp, path_arr, val] });
                        s.emit(JitOp::Drop { slot: inp });
                        s.emit(JitOp::Drop { slot: path_arr });
                        s.emit_yield(out);
                        s.emit(JitOp::Drop { slot: out });
                    });
                    return ok;
                }
                // Complex path: delegate to runtime (only safe without FuncCall refs)
                if contains_func_call(path_expr) || contains_func_call(value_expr) {
                    return false;
                }
                let idx = self.closure_ops.len();
                self.closure_ops.push(path_expr.clone());
                let idx2 = self.closure_ops.len();
                self.closure_ops.push(value_expr.clone());
                let inp = self.alloc_slot();
                self.emit(JitOp::Clone { dst: inp, src: input_slot });
                let arr = self.alloc_slot();
                self.emit_propagating(JitOp::CallBuiltin {
                    dst: arr,
                    name: format!("__assign__:{}:{}", idx, idx2),
                    args: vec![inp],
                });
                self.emit(JitOp::Drop { slot: inp });
                // Result is an array of outputs; iterate and yield each
                self.flatten_each_with_action(arr, false, &|s, elem| {
                    s.emit_yield(elem);
                });
                self.emit(JitOp::Drop { slot: arr });
                true
            }

            Expr::Update { path_expr, update_expr } => {
                // Inline FuncCalls to make expressions suitable for runtime delegation
                let inlined_path = self.inline_func_calls(path_expr);
                let inlined_update = self.inline_func_calls(update_expr);
                let path_expr = &inlined_path;
                let update_expr = &inlined_update;
                // Handle .[] and .[]? paths with scalar update_expr:
                // iterate all paths from eval_path, apply update to each
                let is_each = matches!(path_expr, Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input));
                let is_each_opt = matches!(path_expr, Expr::EachOpt { input_expr } if matches!(input_expr.as_ref(), Expr::Input));
                if (is_each || is_each_opt) && is_scalar(update_expr) {
                    // Emit: get all keys/indices, iterate them, for each: getpath([k]), apply f, setpath([k])
                    // Use CallBuiltin "keys" to get all keys, then iterate
                    let inp = self.alloc_slot();
                    self.emit(JitOp::Clone { dst: inp, src: input_slot });
                    let keys = self.alloc_slot();
                    self.emit_propagating(JitOp::CallBuiltin {
                        dst: keys, name: "keys".to_string(), args: vec![inp],
                    });
                    self.emit(JitOp::Drop { slot: inp });
                    // Accumulate result starting from input
                    let result = self.alloc_slot();
                    self.emit(JitOp::Clone { dst: result, src: input_slot });
                    // Iterate keys
                    self.flatten_each_with_action(keys, false, &|s, key_elem| {
                        // Build path array [key]
                        let path_arr = s.alloc_slot();
                        s.emit(JitOp::CollectBegin);
                        s.emit(JitOp::CollectPush { src: key_elem });
                        s.emit(JitOp::CollectFinish { dst: path_arr });
                        // getpath
                        let cur = s.alloc_slot();
                        s.emit(JitOp::Clone { dst: cur, src: result });
                        let path_c = s.alloc_slot();
                        s.emit(JitOp::Clone { dst: path_c, src: path_arr });
                        let old_val = s.alloc_slot();
                        s.emit_propagating(JitOp::CallBuiltin {
                            dst: old_val, name: "getpath".to_string(), args: vec![cur, path_c],
                        });
                        s.emit(JitOp::Drop { slot: cur });
                        s.emit(JitOp::Drop { slot: path_c });
                        // apply update
                        let new_val = s.flatten_scalar(update_expr, old_val);
                        s.emit(JitOp::Drop { slot: old_val });
                        // setpath
                        let cur2 = s.alloc_slot();
                        s.emit(JitOp::Clone { dst: cur2, src: result });
                        let updated = s.alloc_slot();
                        s.emit_propagating(JitOp::CallBuiltin {
                            dst: updated, name: "setpath".to_string(), args: vec![cur2, path_arr, new_val],
                        });
                        s.emit(JitOp::Drop { slot: cur2 });
                        s.emit(JitOp::Drop { slot: new_val });
                        // Update result in place (drop old, clone new)
                        s.emit(JitOp::Drop { slot: result });
                        s.emit(JitOp::Clone { dst: result, src: updated });
                        s.emit(JitOp::Drop { slot: updated });
                    });
                    self.emit(JitOp::Drop { slot: keys });
                    self.emit_yield(result);
                    self.emit(JitOp::Drop { slot: result });
                    return true;
                }
                if let Some(path_components) = extract_simple_path(path_expr) {
                    // Simple static path: getpath, apply update, setpath
                    if is_scalar(update_expr) {
                        let path_arr = self.build_path_array(&path_components, input_slot);
                        let inp = self.alloc_slot();
                        self.emit(JitOp::Clone { dst: inp, src: input_slot });
                        let path_clone = self.alloc_slot();
                        self.emit(JitOp::Clone { dst: path_clone, src: path_arr });
                        let old_val = self.alloc_slot();
                        self.emit_propagating(JitOp::CallBuiltin { dst: old_val, name: "getpath".to_string(), args: vec![inp, path_clone] });
                        self.emit(JitOp::Drop { slot: inp });
                        self.emit(JitOp::Drop { slot: path_clone });
                        let new_val = self.flatten_scalar(update_expr, old_val);
                        self.emit(JitOp::Drop { slot: old_val });
                        let inp2 = self.alloc_slot();
                        self.emit(JitOp::Clone { dst: inp2, src: input_slot });
                        let out = self.alloc_slot();
                        self.emit_propagating(JitOp::CallBuiltin { dst: out, name: "setpath".to_string(), args: vec![inp2, path_arr, new_val] });
                        self.emit(JitOp::Drop { slot: inp2 });
                        self.emit(JitOp::Drop { slot: new_val });
                        self.emit_yield(out);
                        self.emit(JitOp::Drop { slot: out });
                        return true;
                    }
                    // Generator update_expr with simple path
                    let path_arr = self.build_path_array(&path_components, input_slot);
                    let inp = self.alloc_slot();
                    self.emit(JitOp::Clone { dst: inp, src: input_slot });
                    let path_clone = self.alloc_slot();
                    self.emit(JitOp::Clone { dst: path_clone, src: path_arr });
                    let old_val = self.alloc_slot();
                    self.emit_propagating(JitOp::CallBuiltin { dst: old_val, name: "getpath".to_string(), args: vec![inp, path_clone] });
                    self.emit(JitOp::Drop { slot: inp });
                    self.emit(JitOp::Drop { slot: path_clone });
                    // For each output of update_expr applied to old_val
                    let ok = self.flatten_gen_with_each_output(update_expr, old_val, &|s, new_val| {
                        let inp3 = s.alloc_slot();
                        s.emit(JitOp::Clone { dst: inp3, src: input_slot });
                        let path_c2 = s.alloc_slot();
                        s.emit(JitOp::Clone { dst: path_c2, src: path_arr });
                        let out = s.alloc_slot();
                        s.emit_propagating(JitOp::CallBuiltin { dst: out, name: "setpath".to_string(), args: vec![inp3, path_c2, new_val] });
                        s.emit(JitOp::Drop { slot: inp3 });
                        s.emit(JitOp::Drop { slot: path_c2 });
                        s.emit_yield(out);
                        s.emit(JitOp::Drop { slot: out });
                    });
                    self.emit(JitOp::Drop { slot: old_val });
                    self.emit(JitOp::Drop { slot: path_arr });
                    return ok;
                }
                // Complex path: delegate to runtime (only safe without FuncCall refs)
                if contains_func_call(path_expr) || contains_func_call(update_expr) {
                    return false;
                }
                let idx = self.closure_ops.len();
                self.closure_ops.push(path_expr.clone());
                let idx2 = self.closure_ops.len();
                self.closure_ops.push(update_expr.clone());
                let inp = self.alloc_slot();
                self.emit(JitOp::Clone { dst: inp, src: input_slot });
                let arr = self.alloc_slot();
                self.emit_propagating(JitOp::CallBuiltin {
                    dst: arr,
                    name: format!("__update__:{}:{}", idx, idx2),
                    args: vec![inp],
                });
                self.emit(JitOp::Drop { slot: inp });
                self.flatten_each_with_action(arr, false, &|s, elem| {
                    s.emit_yield(elem);
                });
                self.emit(JitOp::Drop { slot: arr });
                true
            }

            Expr::Label { var_index, body } => {
                let done_label = self.alloc_label();
                self.label_targets.insert(*var_index, done_label);
                let ok = self.flatten_gen(body, input_slot);
                self.label_targets.remove(var_index);
                self.emit(JitOp::Label { id: done_label });
                ok
            }

            Expr::Break { var_index, .. } => {
                if let Some(&done_label) = self.label_targets.get(var_index) {
                    self.emit(JitOp::Jump { label: done_label });
                    true
                } else {
                    false
                }
            }

            Expr::Limit { count, generator } if !is_scalar(count) => {
                // Generator count: rewrite as count as $n | limit($n; body)
                let count_var: u16 = 10300;
                let rewritten = Expr::LetBinding {
                    var_index: count_var,
                    value: Box::new((**count).clone()),
                    body: Box::new(Expr::Limit {
                        count: Box::new(Expr::LoadVar { var_index: count_var }),
                        generator: generator.clone(),
                    }),
                };
                self.flatten_gen(&rewritten, input_slot)
            }

            Expr::Limit { count, generator } => {
                // Pre-check: generator must be compilable
                {
                    let mut test = self.test_flattener();
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen(generator, t_in) { return false; }
                }
                let count_val = self.flatten_scalar(count, input_slot);
                let counter_var = self.alloc_var();
                let limit_var = self.alloc_var();
                let one_var = self.alloc_var();
                let done_label = self.alloc_label();
                // Initialize counter to f64 0.0
                self.emit(JitOp::F64Const { dst_var: counter_var, val: 0.0 });
                self.emit(JitOp::F64Const { dst_var: one_var, val: 1.0 });
                self.emit(JitOp::ToF64Var { dst_var: limit_var, src: count_val });
                self.emit(JitOp::Drop { slot: count_val });

                // Check limit: if limit == 0, skip; if limit < 0, error; else start
                let start_label = self.alloc_label();
                let zero_label = self.alloc_label();
                let zero_var = self.alloc_var();
                self.emit(JitOp::F64Const { dst_var: zero_var, val: 0.0 });
                // Check if limit > 0
                let cmp = self.alloc_var();
                self.emit(JitOp::F64Less { dst_var: cmp, a_var: zero_var, b_var: limit_var });
                self.emit(JitOp::BranchOnVar { var: cmp, nonzero_label: start_label, zero_label });
                self.emit(JitOp::Label { id: zero_label });
                // limit <= 0: check if negative (error) or zero (skip)
                let neg_cmp = self.alloc_var();
                self.emit(JitOp::F64Less { dst_var: neg_cmp, a_var: limit_var, b_var: zero_var });
                let error_label = self.alloc_label();
                self.emit(JitOp::BranchOnVar { var: neg_cmp, nonzero_label: error_label, zero_label: done_label });
                self.emit(JitOp::Label { id: error_label });
                // Negative limit: throw error
                let err_msg = self.alloc_slot();
                self.emit(JitOp::Str { dst: err_msg, val: "limit doesn't support negative count".to_string() });
                self.emit(JitOp::ThrowError { msg: err_msg });
                if self.try_catch_target.is_none() {
                    self.ops.push(JitOp::ReturnError);
                }
                self.emit(JitOp::Drop { slot: err_msg });
                self.emit(JitOp::Jump { label: done_label });
                self.emit(JitOp::Label { id: start_label });

                // Set limit_state so emit_yield checks counter after each yield
                let old_limit = self.limit_state;
                self.limit_state = Some((counter_var, limit_var, one_var, done_label, self.collect_depth));
                let ok = self.flatten_gen(generator, input_slot);
                self.limit_state = old_limit;
                self.emit(JitOp::Label { id: done_label });
                ok
            }

            Expr::Recurse { input_expr } => {
                if !is_scalar(input_expr) { return false; }
                // Recurse: yield input, then recurse into each child
                // Collect all recursive outputs via runtime, then iterate
                let val = self.flatten_scalar(input_expr, input_slot);
                let arr = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: arr, name: "recurse_collect".to_string(), args: vec![val] });
                self.emit(JitOp::Drop { slot: val });
                // Iterate the collected array, yielding each element
                self.flatten_each_with_action(arr, false, &|s, elem| {
                    s.emit_yield(elem);
                });
                self.emit(JitOp::Drop { slot: arr });
                true
            }

            Expr::FuncCall { func_id, args } => {
                if *func_id >= self.funcs.len() { return false; }
                // Detect recursive function calls — not JIT-compilable
                if self.expanding_funcs.contains(func_id) { return false; }
                self.expanding_funcs.insert(*func_id);
                let func = &self.funcs[*func_id].clone();
                let body = if !func.param_vars.is_empty() && !args.is_empty() {
                    crate::eval::substitute_params(&func.body, &func.param_vars, args)
                } else {
                    func.body.clone()
                };
                let result = self.flatten_gen(&body, input_slot);
                self.expanding_funcs.remove(func_id);
                result
            }

            Expr::AlternativeDestructure { alternatives } => {
                // Try each alternative until one succeeds
                // Pre-check: all alternatives must be compilable
                for alt in alternatives {
                    let mut test = self.test_flattener();
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen(alt, t_in) { return false; }
                }

                let done_label = self.alloc_label();
                for (i, alt) in alternatives.iter().enumerate() {
                    let is_last = i == alternatives.len() - 1;
                    if is_last {
                        // Last alternative: just run it (errors propagate)
                        self.flatten_gen(alt, input_slot);
                    } else {
                        // Try this alternative; on error, try next
                        let catch_label = self.alloc_label();
                        let error_slot = self.alloc_slot();
                        let old_target = self.try_catch_target;
                        self.try_catch_target = Some((catch_label, error_slot));
                        self.try_depth += 1;
                        self.emit(JitOp::TryCatchBegin);
                        self.flatten_gen(alt, input_slot);
                        self.emit(JitOp::TryCatchEnd);
                        self.try_depth -= 1;
                        self.try_catch_target = old_target;
                        self.emit(JitOp::Jump { label: done_label });
                        self.emit(JitOp::Label { id: catch_label });
                        self.emit(JitOp::TryCatchEnd);
                        self.emit(JitOp::Drop { slot: error_slot });
                        // Continue to next alternative
                    }
                }
                self.emit(JitOp::Label { id: done_label });
                true
            }

            Expr::PathExpr { expr: path_expr } => {
                // path(expr): compute the paths that expr would access
                if is_scalar(path_expr) {
                    if let Some(path_components) = extract_simple_path(path_expr) {
                        let path_arr = self.build_path_array(&path_components, input_slot);
                        self.emit_yield(path_arr);
                        self.emit(JitOp::Drop { slot: path_arr });
                        return true;
                    }
                }
                // Delegate complex path expressions to runtime
                let idx = self.closure_ops.len();
                self.closure_ops.push((**path_expr).clone());
                let inp = self.alloc_slot();
                self.emit(JitOp::Clone { dst: inp, src: input_slot });
                // path() can produce multiple outputs (generator paths)
                let arr = self.alloc_slot();
                self.emit_propagating(JitOp::CallBuiltin {
                    dst: arr,
                    name: format!("__path__:{}", idx),
                    args: vec![inp],
                });
                self.emit(JitOp::Drop { slot: inp });
                // arr is an array of paths; iterate and yield each
                self.flatten_each_with_action(arr, false, &|s, elem| {
                    s.emit_yield(elem);
                });
                self.emit(JitOp::Drop { slot: arr });
                true
            }

            // `not` as a generator (already handled as scalar but needs generator wrapper)
            Expr::Not => {
                let out = self.flatten_scalar(&Expr::Not, input_slot);
                self.emit_yield(out);
                self.emit(JitOp::Drop { slot: out });
                true
            }

            // Filter-argument builtins: delegate to eval (not JIT-compilable)
            Expr::CallBuiltin { name, args } if matches!(
                (name.as_str(), args.len()),
                ("walk", _) | ("pick", _) | ("skip", _) | ("add", 1) | ("del", _)
            ) => {
                false
            }

            // CallBuiltin with generator args: rewrite as nested LetBinding
            // e.g., join(",","/") → (",","/") as $__arg | join($__arg)
            Expr::CallBuiltin { name, args } if !args.is_empty() && args.iter().any(|a| !is_scalar(a)) => {
                // Each generator arg gets bound to a temp var
                let base_var: u16 = 10200;
                let mut var_args = Vec::new();
                let mut all_compilable = true;
                for (i, arg) in args.iter().enumerate() {
                    if !is_scalar(arg) {
                        // Check if the generator arg is compilable
                        let mut test = self.test_flattener();
                        let t_in = test.alloc_slot();
                        if !test.flatten_gen(arg, t_in) {
                            all_compilable = false;
                            break;
                        }
                    }
                    var_args.push((base_var + i as u16, arg));
                }
                if !all_compilable { return false; }

                // Build nested LetBinding: arg0 as $v0 | arg1 as $v1 | ... | CallBuiltin(name, [$v0, $v1, ...])
                let inner = Expr::CallBuiltin {
                    name: name.clone(),
                    args: var_args.iter().map(|(vi, _)| Expr::LoadVar { var_index: *vi }).collect(),
                };
                let mut rewritten = inner;
                for (vi, arg) in var_args.iter().rev() {
                    if is_scalar(arg) {
                        // Scalar arg: still use LetBinding for uniformity
                        rewritten = Expr::LetBinding {
                            var_index: *vi,
                            value: Box::new((*arg).clone()),
                            body: Box::new(rewritten),
                        };
                    } else {
                        // Generator arg: wrap in LetBinding
                        rewritten = Expr::LetBinding {
                            var_index: *vi,
                            value: Box::new((*arg).clone()),
                            body: Box::new(rewritten),
                        };
                    }
                }
                self.flatten_gen(&rewritten, input_slot)
            }

            // Alternative with generator primary:
            // Collect non-null/false outputs. If any, yield them. If none, yield fallback.
            Expr::Alternative { primary, fallback } if !is_scalar(primary) => {
                // Collect all outputs of primary into an array
                self.emit(JitOp::CollectBegin);
                self.collect_depth += 1;
                let ok = self.flatten_gen(primary, input_slot);
                self.collect_depth -= 1;
                if !ok { return false; }
                let all_outputs = self.alloc_slot();
                self.emit(JitOp::CollectFinish { dst: all_outputs });

                // Filter: keep only non-null/non-false values
                let has_match = self.alloc_var();
                self.emit(JitOp::F64Const { dst_var: has_match, val: 0.0 });
                let one_var = self.alloc_var();
                self.emit(JitOp::F64Const { dst_var: one_var, val: 1.0 });

                // Iterate and yield non-null/false values
                self.flatten_each_with_action(all_outputs, false, &|s, elem| {
                    let is_nf = s.alloc_var();
                    s.emit(JitOp::IsNullOrFalse { dst_var: is_nf, src: elem });
                    let skip_lbl = s.alloc_label();
                    let emit_lbl = s.alloc_label();
                    s.emit(JitOp::BranchOnVar { var: is_nf, nonzero_label: skip_lbl, zero_label: emit_lbl });
                    s.emit(JitOp::Label { id: emit_lbl });
                    s.emit(JitOp::F64Const { dst_var: has_match, val: 1.0 });
                    s.emit_yield(elem);
                    s.emit(JitOp::Label { id: skip_lbl });
                });

                // If no non-null/false values, yield fallback
                let fb_lbl = self.alloc_label();
                let done_lbl = self.alloc_label();
                self.emit(JitOp::BranchOnVar { var: has_match, nonzero_label: done_lbl, zero_label: fb_lbl });
                self.emit(JitOp::Label { id: fb_lbl });
                if is_scalar(fallback) {
                    let fval = self.flatten_scalar(fallback, input_slot);
                    self.emit_yield(fval);
                    self.emit(JitOp::Drop { slot: fval });
                } else {
                    if !self.flatten_gen(fallback, input_slot) {
                        self.emit(JitOp::Drop { slot: all_outputs });
                        return false;
                    }
                }
                self.emit(JitOp::Label { id: done_lbl });
                self.emit(JitOp::Drop { slot: all_outputs });
                true
            }

            // (dead code removed — foreach gen-init moved earlier, Assign/Update handled above)

            // ClosureOp: sort_by, group_by, min_by, max_by, unique_by
            Expr::ClosureOp { op, input_expr, key_expr } => {
                if !is_scalar(input_expr) { return false; }
                let container = self.flatten_scalar(input_expr, input_slot);
                let out = self.alloc_slot();
                let op_name = match op {
                    ClosureOpKind::SortBy => "sort_by",
                    ClosureOpKind::GroupBy => "group_by",
                    ClosureOpKind::UniqueBy => "unique_by",
                    ClosureOpKind::MinBy => "min_by",
                    ClosureOpKind::MaxBy => "max_by",
                };
                // Delegate to runtime via CallBuiltin with the key expression
                // We need to evaluate key_expr for each element at runtime
                // Create a special closure-op call
                self.emit(JitOp::CallBuiltin {
                    dst: out,
                    name: format!("__closure_op__:{}:{}", op_name, self.closure_ops.len()),
                    args: vec![container],
                });
                self.closure_ops.push((**key_expr).clone());
                self.emit(JitOp::Drop { slot: container });
                self.emit_yield(out);
                self.emit(JitOp::Drop { slot: out });
                true
            }

            // Slice with generator args
            Expr::Slice { expr: base_expr, from, to } if is_scalar(base_expr) => {
                let from_scalar = from.as_ref().is_none_or(|f| is_scalar(f));
                let to_scalar = to.as_ref().is_none_or(|t| is_scalar(t));
                if from_scalar && to_scalar {
                    // Already scalar, should be handled by is_scalar. Yield once.
                    let out = self.flatten_scalar(expr, input_slot);
                    self.emit_yield(out);
                    self.emit(JitOp::Drop { slot: out });
                    true
                } else {
                    false
                }
            }

            _ => false,
        }
    }

    /// Generic helper: iterate a generator expression and call action for each output.
    /// Returns true if the generator can be compiled.
    fn flatten_gen_with_each_output(&mut self, gen_expr: &Expr, input_slot: SlotId,
                                     action: &dyn Fn(&mut Flattener, SlotId)) -> bool {
        if is_scalar(gen_expr) {
            let out = self.flatten_scalar(gen_expr, input_slot);
            action(self, out);
            self.emit(JitOp::Drop { slot: out });
            return true;
        }
        match gen_expr {
            Expr::Comma { left, right } => {
                if !self.flatten_gen_with_each_output(left, input_slot, action) { return false; }
                self.flatten_gen_with_each_output(right, input_slot, action)
            }
            Expr::Empty => true,
            Expr::Each { input_expr } if is_scalar(input_expr) => {
                let container = self.flatten_scalar(input_expr, input_slot);
                self.flatten_each_with_action(container, false, action);
                self.emit(JitOp::Drop { slot: container });
                true
            }
            Expr::Pipe { left, right } if is_scalar(left) => {
                let mid = self.flatten_scalar(left, input_slot);
                let ok = self.flatten_gen_with_each_output(right, mid, action);
                self.emit(JitOp::Drop { slot: mid });
                ok
            }
            Expr::Range { from, to, step } if is_scalar(from) && is_scalar(to) && step.as_ref().is_none_or(|s| is_scalar(s)) => {
                let from_val = self.flatten_scalar(from, input_slot);
                let to_val = self.flatten_scalar(to, input_slot);
                let step_val = if let Some(s) = step {
                    self.flatten_scalar(s, input_slot)
                } else {
                    let one = self.alloc_slot();
                    self.emit(JitOp::Num { dst: one, val: 1.0, repr: None });
                    one
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
                true
            }
            Expr::IfThenElse { cond, then_branch, else_branch } if is_scalar(cond) => {
                // Pre-check both branches
                {
                    let mut test = self.test_flattener();
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen_with_each_output(then_branch, t_in, action) { return false; }
                    if !test.flatten_gen_with_each_output(else_branch, t_in, action) { return false; }
                }
                let cond_val = self.flatten_scalar(cond, input_slot);
                let then_lbl = self.alloc_label();
                let else_lbl = self.alloc_label();
                let done_lbl = self.alloc_label();
                self.emit(JitOp::IfTruthy { src: cond_val, then_label: then_lbl, else_label: else_lbl });
                self.emit(JitOp::Drop { slot: cond_val });

                self.emit(JitOp::Label { id: then_lbl });
                self.flatten_gen_with_each_output(then_branch, input_slot, action);
                self.emit(JitOp::Jump { label: done_lbl });

                self.emit(JitOp::Label { id: else_lbl });
                self.flatten_gen_with_each_output(else_branch, input_slot, action);
                self.emit(JitOp::Label { id: done_lbl });
                true
            }
            Expr::LetBinding { var_index, value, body } if is_scalar(value) => {
                let val = self.flatten_scalar(value, input_slot);
                let old = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: old, var_index: *var_index });
                self.emit(JitOp::SetVar { var_index: *var_index, src: val });
                self.emit(JitOp::Drop { slot: val });
                let ok = self.flatten_gen_with_each_output(body, input_slot, action);
                self.emit(JitOp::SetVar { var_index: *var_index, src: old });
                self.emit(JitOp::Drop { slot: old });
                ok
            }
            Expr::EachOpt { input_expr } if is_scalar(input_expr) => {
                let container = self.flatten_scalar(input_expr, input_slot);
                self.flatten_each_with_action(container, true, action);
                self.emit(JitOp::Drop { slot: container });
                true
            }
            _ => {
                // Generic fallback: collect all outputs, then iterate
                // Pre-check: can we compile gen_expr?
                {
                    let mut test = self.test_flattener();
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen(gen_expr, t_in) { return false; }
                }
                // Collect all outputs from gen_expr into an array, then iterate
                self.emit(JitOp::CollectBegin);
                self.collect_depth += 1;
                let ok = self.flatten_gen(gen_expr, input_slot);
                self.collect_depth -= 1;
                if !ok { return false; }
                let arr = self.alloc_slot();
                self.emit(JitOp::CollectFinish { dst: arr });
                // Iterate the collected array
                self.flatten_each_with_action(arr, false, action);
                self.emit(JitOp::Drop { slot: arr });
                true
            }
        }
    }

    /// Handle try-catch: try try_expr catch catch_expr
    fn flatten_gen_try_catch(&mut self, try_expr: &Expr, catch_expr: &Expr, input_slot: SlotId) -> bool {
        // Pre-check: the try body must be compilable
        {
            let mut test = self.test_flattener();
                    test.try_depth = self.try_depth + 1;
                    test.try_catch_target = Some((0, 0)); // dummy target for pre-check
            let t_in = test.alloc_slot();
            if !test.flatten_gen(try_expr, t_in) { return false; }
        }
        // Pre-check: the catch body must be compilable
        {
            let mut test = self.test_flattener();
            let t_in = test.alloc_slot();
            if !test.flatten_gen(catch_expr, t_in) { return false; }
        }

        let catch_label = self.alloc_label();
        let done_label = self.alloc_label();
        let error_slot = self.alloc_slot();

        // Save the old try_catch_target and set the new one
        let old_target = self.try_catch_target;
        self.try_catch_target = Some((catch_label, error_slot));
        self.try_depth += 1;
        self.emit(JitOp::TryCatchBegin);

        // Compile the try body
        let ok = self.flatten_gen(try_expr, input_slot);

        self.emit(JitOp::TryCatchEnd);
        self.try_depth -= 1;
        self.try_catch_target = old_target;

        if !ok { return false; }

        // Jump over catch handler
        self.emit(JitOp::Jump { label: done_label });

        // Catch handler: error_slot contains the error value
        self.emit(JitOp::Label { id: catch_label });
        self.emit(JitOp::TryCatchEnd);

        // Execute catch_expr with error value as input
        self.flatten_gen(catch_expr, error_slot);
        self.emit(JitOp::Drop { slot: error_slot });

        self.emit(JitOp::Label { id: done_label });
        true
    }

    /// Handle generator | anything: emit left generator's loop with right inline.
    fn flatten_gen_pipe(&mut self, left: &Expr, right: &Expr, input_slot: SlotId) -> bool {
        match left {
            // Empty | anything = empty (no outputs)
            Expr::Empty => true,

            Expr::Each { input_expr } | Expr::EachOpt { input_expr } => {
                let is_opt = matches!(left, Expr::EachOpt { .. });
                // Borrow input directly when iterating over it
                if matches!(input_expr.as_ref(), Expr::Input) {
                    if !self.flatten_each_with_body(input_slot, is_opt, right, input_slot) {
                        return false;
                    }
                    true
                } else if is_scalar(input_expr) {
                    let container = self.flatten_scalar(input_expr, input_slot);
                    if !self.flatten_each_with_body(container, is_opt, right, input_slot) {
                        return false;
                    }
                    self.emit(JitOp::Drop { slot: container });
                    true
                } else {
                    false
                }
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
                let then_pipe = Expr::Pipe {
                    left: Box::new((**then_branch).clone()),
                    right: Box::new(right.clone()),
                };
                let else_pipe = Expr::Pipe {
                    left: Box::new((**else_branch).clone()),
                    right: Box::new(right.clone()),
                };
                // Pre-check both branches
                {
                    let mut test = self.test_flattener();
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen(&then_pipe, t_in) { return false; }
                    if !test.flatten_gen(&else_pipe, t_in) { return false; }
                }
                let cond_val = self.flatten_scalar(cond, input_slot);
                let then_lbl = self.alloc_label();
                let else_lbl = self.alloc_label();
                let done_lbl = self.alloc_label();
                self.emit(JitOp::IfTruthy { src: cond_val, then_label: then_lbl, else_label: else_lbl });
                self.emit(JitOp::Drop { slot: cond_val });

                self.emit(JitOp::Label { id: then_lbl });
                self.flatten_gen(&then_pipe, input_slot);
                self.emit(JitOp::Jump { label: done_lbl });

                self.emit(JitOp::Label { id: else_lbl });
                self.flatten_gen(&else_pipe, input_slot);
                self.emit(JitOp::Label { id: done_lbl });
                true
            }
            Expr::Range { from, to, step } => {
                if !is_scalar(from) || !is_scalar(to) { return false; }
                if let Some(s) = step { if !is_scalar(s) { return false; } }
                // Pre-check: can we compile the body (right)?
                {
                    let mut test = self.test_flattener();
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen(right, t_in) { return false; }
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
            // TryCatch | right: compile try-catch body inline, each output piped to right
            Expr::TryCatch { try_expr, catch_expr } => {
                // Pre-check: try body and right must be compilable
                {
                    let mut test = self.test_flattener();
                    test.try_depth = self.try_depth + 1;
                    test.try_catch_target = Some((0, 0));
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen(try_expr, t_in) { return false; }
                }
                {
                    let mut test = self.test_flattener();
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen(catch_expr, t_in) { return false; }
                }

                let catch_label = self.alloc_label();
                let done_label = self.alloc_label();
                let error_slot = self.alloc_slot();

                let old_target = self.try_catch_target;
                self.try_catch_target = Some((catch_label, error_slot));
                self.try_depth += 1;
                self.emit(JitOp::TryCatchBegin);

                // Compile try body, but pipe results to right OUTSIDE the inner try scope.
                // Semantics of (try X catch Y) | Z:
                //   - errors in X → caught by Y
                //   - errors in Z → propagate to outer try (not caught by Y)
                if is_scalar(try_expr) {
                    let mid = self.flatten_scalar(try_expr, input_slot);
                    // End inner try scope before piping to right
                    self.emit(JitOp::TryCatchEnd);
                    self.try_depth -= 1;
                    self.try_catch_target = old_target;
                    self.flatten_gen(right, mid);
                    self.emit(JitOp::Drop { slot: mid });
                } else {
                    // try_expr is a generator: collect outputs inside try, then pipe outside
                    self.emit(JitOp::CollectBegin);
                    self.collect_depth += 1;
                    let ok = self.flatten_gen(try_expr, input_slot);
                    self.collect_depth -= 1;
                    if !ok {
                        self.emit(JitOp::TryCatchEnd);
                        self.try_depth -= 1;
                        self.try_catch_target = old_target;
                        return false;
                    }
                    let collected = self.alloc_slot();
                    self.emit(JitOp::CollectFinish { dst: collected });
                    // End try scope
                    self.emit(JitOp::TryCatchEnd);
                    self.try_depth -= 1;
                    self.try_catch_target = old_target;
                    // Iterate collected outputs and pipe to right (outside try)
                    self.flatten_each_with_action(collected, false, &|s, elem| {
                        s.flatten_gen(right, elem);
                    });
                    self.emit(JitOp::Drop { slot: collected });
                }

                self.emit(JitOp::Jump { label: done_label });

                // Catch handler
                self.emit(JitOp::Label { id: catch_label });
                self.emit(JitOp::TryCatchEnd);
                // Pipe catch output through right (outside inner try scope)
                if is_scalar(catch_expr) {
                    let catch_val = self.flatten_scalar(catch_expr, error_slot);
                    self.flatten_gen(right, catch_val);
                    self.emit(JitOp::Drop { slot: catch_val });
                } else {
                    self.flatten_gen_pipe(catch_expr, right, error_slot);
                }
                self.emit(JitOp::Drop { slot: error_slot });

                self.emit(JitOp::Label { id: done_label });
                true
            }

            // Collect | right: collect produces exactly one array, pipe to right
            Expr::Collect { generator } => {
                // Pre-check: can we compile the generator?
                {
                    let mut test = self.test_flattener();
                    test.collect_depth = self.collect_depth + 1;
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen(generator, t_in) { return false; }
                }
                // Pre-check: can we compile right?
                {
                    let mut test = self.test_flattener();
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen(right, t_in) { return false; }
                }
                self.emit(JitOp::CollectBegin);
                self.collect_depth += 1;
                let ok = self.flatten_gen(generator, input_slot);
                self.collect_depth -= 1;
                if !ok { return false; }
                let arr = self.alloc_slot();
                self.emit(JitOp::CollectFinish { dst: arr });
                let r = self.flatten_gen(right, arr);
                self.emit(JitOp::Drop { slot: arr });
                r
            }

            // LetBinding | right: handle binding inline
            Expr::LetBinding { var_index, value, body } if is_scalar(value) => {
                let val = self.flatten_scalar(value, input_slot);
                let old = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: old, var_index: *var_index });
                self.emit(JitOp::SetVar { var_index: *var_index, src: val });
                self.emit(JitOp::Drop { slot: val });
                // Now compile body | right
                let ok = if is_scalar(body) {
                    let mid = self.flatten_scalar(body, input_slot);
                    let r = self.flatten_gen(right, mid);
                    self.emit(JitOp::Drop { slot: mid });
                    r
                } else {
                    self.flatten_gen_pipe(body, right, input_slot)
                };
                self.emit(JitOp::SetVar { var_index: *var_index, src: old });
                self.emit(JitOp::Drop { slot: old });
                ok
            }

            // Generic fallback: for any generator left, iterate its outputs and apply right
            _ => {
                // Pre-check: can we compile both left and right?
                {
                    let mut test = self.test_flattener();
                    let t_in = test.alloc_slot();
                    if !test.flatten_gen(left, t_in) { return false; }
                    if !test.flatten_gen(right, t_in) { return false; }
                }
                let right_clone = right.clone();
                self.flatten_gen_with_each_output(left, input_slot, &|this, elem| {
                    this.flatten_gen(&right_clone, elem);
                })
            }
        }
    }

    /// Flatten ObjectConstruct with generator pairs by converting to LetBinding chain.
    fn flatten_obj_construct_gen(&mut self, pairs: &[(Expr, Expr)], _pair_idx: usize, input_slot: SlotId) -> bool {
        // Only handle generator values (keys must be scalar)
        if !pairs.iter().all(|(k, _)| is_scalar(k)) { return false; }

        // Convert: {k1: gen1, k2: gen2} =>
        //   gen1 as $__v0 | gen2 as $__v1 | {k1: $__v0, k2: $__v1}
        // Use high var indices to avoid conflicts (starting at 10000)
        let base_var: u16 = 10000;
        let mut scalar_pairs = Vec::new();
        let mut bindings = Vec::new();

        for (i, (k, v)) in pairs.iter().enumerate() {
            if is_scalar(v) {
                scalar_pairs.push((k.clone(), v.clone()));
            } else {
                let var_idx = base_var + i as u16;
                bindings.push((var_idx, v.clone()));
                scalar_pairs.push((k.clone(), Expr::LoadVar { var_index: var_idx }));
            }
        }

        // Build the scalar object construct
        let mut result: Expr = Expr::ObjectConstruct { pairs: scalar_pairs };

        // Wrap with LetBindings (innermost first)
        for (var_idx, value) in bindings.into_iter().rev() {
            result = Expr::LetBinding {
                var_index: var_idx,
                value: Box::new(value),
                body: Box::new(result),
            };
        }

        self.flatten_gen(&result, input_slot)
    }

    /// Emit a range loop from f_slot to t_slot with step s_slot. Yields numbers.
    fn emit_range_loop(&mut self, f_slot: SlotId, t_slot: SlotId, s_slot: SlotId) {
        let cur_var = self.alloc_var();
        let to_var = self.alloc_var();
        let step_var = self.alloc_var();
        let cmp_var = self.alloc_var();
        self.emit(JitOp::ToF64Var { dst_var: cur_var, src: f_slot });
        self.emit(JitOp::ToF64Var { dst_var: to_var, src: t_slot });
        self.emit(JitOp::ToF64Var { dst_var: step_var, src: s_slot });
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
    }

    /// Handle range(from; to; step) where some args are generators.
    fn flatten_range_gen(&mut self, from: &Expr, to: &Expr, step: Option<&Expr>, input_slot: SlotId) -> bool {
        // Build a fully scalar range call wrapped in appropriate generator nesting
        // For each combination of (from_val, to_val, step_val), emit range loop
        if is_scalar(from) && is_scalar(to) && step.is_none_or(is_scalar) {
            // All scalar - just emit directly
            let fv = self.flatten_scalar(from, input_slot);
            let tv = self.flatten_scalar(to, input_slot);
            let sv = if let Some(s) = step {
                self.flatten_scalar(s, input_slot)
            } else {
                let v = self.alloc_slot();
                self.emit(JitOp::Num { dst: v, val: 1.0, repr: None });
                v
            };
            self.emit_range_loop(fv, tv, sv);
            self.emit(JitOp::Drop { slot: fv });
            self.emit(JitOp::Drop { slot: tv });
            self.emit(JitOp::Drop { slot: sv });
            return true;
        }

        // Strategy: iterate generators as needed, calling emit_range_loop for each combo
        // Use Comma { ... } rewriting or flatten_gen_with_each_output
        if !is_scalar(from) {
            // from is generator
            let to_clone = to.clone();
            let step_clone = step.cloned();
            return self.flatten_gen_with_each_output(from, input_slot, &|this, fv| {
                let step_ref = step_clone.as_ref();
                this.flatten_range_gen_inner(fv, &to_clone, step_ref, input_slot);
            });
        }

        // from is scalar, to is generator
        let fv = self.flatten_scalar(from, input_slot);
        if !is_scalar(to) {
            let step_clone = step.cloned();
            let ok = self.flatten_gen_with_each_output(to, input_slot, &|this, tv| {
                let step_ref = step_clone.as_ref();
                if let Some(s) = step_ref {
                    if is_scalar(s) {
                        let sv = this.flatten_scalar(s, input_slot);
                        this.emit_range_loop(fv, tv, sv);
                        this.emit(JitOp::Drop { slot: sv });
                    } else {
                        this.flatten_gen_with_each_output(s, input_slot, &|this2, sv| {
                            this2.emit_range_loop(fv, tv, sv);
                        });
                    }
                } else {
                    let sv = this.alloc_slot();
                    this.emit(JitOp::Num { dst: sv, val: 1.0, repr: None });
                    this.emit_range_loop(fv, tv, sv);
                    this.emit(JitOp::Drop { slot: sv });
                }
            });
            self.emit(JitOp::Drop { slot: fv });
            return ok;
        }

        // from and to are scalar, step is generator
        let tv = self.flatten_scalar(to, input_slot);
        if let Some(s) = step {
            let ok = self.flatten_gen_with_each_output(s, input_slot, &|this, sv| {
                this.emit_range_loop(fv, tv, sv);
            });
            self.emit(JitOp::Drop { slot: fv });
            self.emit(JitOp::Drop { slot: tv });
            return ok;
        }

        // Should not reach here
        false
    }

    /// Inner helper for flatten_range_gen when from is already resolved.
    fn flatten_range_gen_inner(&mut self, fv: SlotId, to: &Expr, step: Option<&Expr>, input_slot: SlotId) {
        if is_scalar(to) {
            let tv = self.flatten_scalar(to, input_slot);
            if let Some(s) = step {
                if is_scalar(s) {
                    let sv = self.flatten_scalar(s, input_slot);
                    self.emit_range_loop(fv, tv, sv);
                    self.emit(JitOp::Drop { slot: sv });
                } else {
                    self.flatten_gen_with_each_output(s, input_slot, &|this, sv| {
                        this.emit_range_loop(fv, tv, sv);
                    });
                }
            } else {
                let sv = self.alloc_slot();
                self.emit(JitOp::Num { dst: sv, val: 1.0, repr: None });
                self.emit_range_loop(fv, tv, sv);
                self.emit(JitOp::Drop { slot: sv });
            }
            self.emit(JitOp::Drop { slot: tv });
        } else {
            let step_clone = step.cloned();
            self.flatten_gen_with_each_output(to, input_slot, &|this, tv| {
                let step_ref = step_clone.as_ref();
                if let Some(s) = step_ref {
                    if is_scalar(s) {
                        let sv = this.flatten_scalar(s, input_slot);
                        this.emit_range_loop(fv, tv, sv);
                        this.emit(JitOp::Drop { slot: sv });
                    } else {
                        this.flatten_gen_with_each_output(s, input_slot, &|this2, sv| {
                            this2.emit_range_loop(fv, tv, sv);
                        });
                    }
                } else {
                    let sv = this.alloc_slot();
                    this.emit(JitOp::Num { dst: sv, val: 1.0, repr: None });
                    this.emit_range_loop(fv, tv, sv);
                    this.emit(JitOp::Drop { slot: sv });
                }
            });
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
            self.emit(JitOp::InitVar { var: idx });
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
            self.emit(JitOp::InitVar { var: idx });
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

        // Other (non-iterable)
        self.emit(JitOp::Label { id: other_lbl });
        if !is_opt {
            // Set error: "Cannot iterate over TYPE (VALUE)"
            let err_slot = self.alloc_slot();
            self.emit(JitOp::CallBuiltin {
                dst: err_slot,
                name: "__each_error__".to_string(),
                args: vec![container],
            });
            self.emit(JitOp::Drop { slot: err_slot });
            if let Some((catch_label, error_slot)) = self.try_catch_target {
                self.emit(JitOp::CheckError { error_dst: error_slot, catch_label });
            } else {
                self.emit(JitOp::ReturnError);
            }
        }
        self.emit(JitOp::Label { id: done_lbl });
    }

    /// Emit .[] iteration, applying `body_expr` to each element. Returns false if body can't be compiled.
    fn flatten_each_with_body(&mut self, container: SlotId, is_opt: bool, body_expr: &Expr, _original_input: SlotId) -> bool {
        // Pre-check: can we compile the body?
        {
            let mut test = self.test_flattener();
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
            self.emit(JitOp::InitVar { var: idx });
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
            self.emit(JitOp::InitVar { var: idx });
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
        // Detect in-place update pattern: path |= f or path += f (wrapped in LetBinding)
        let inplace_info = detect_inplace_update(update);

        // Detect last(g) pattern: reduce g as $x ([]; [$x])
        let is_last_pattern = matches!(update,
            Expr::Collect { generator } if matches!(generator.as_ref(),
                Expr::LoadVar { var_index: vi } if *vi == var_index
            )
        );

        // Detect accumulator-add pattern: reduce ... (init; . + rhs)
        let acc_add_rhs = if let Expr::BinOp { op, lhs, rhs } = update {
            if matches!(op, BinOp::Add) && matches!(lhs.as_ref(), Expr::Input) && is_scalar(rhs) {
                Some(rhs.as_ref())
            } else { None }
        } else { None };

        // Helper: emit the update step (set $var, load acc as ., evaluate update, store back)
        let emit_update = |s: &mut Flattener, elem: SlotId| {
            s.emit(JitOp::SetVar { var_index, src: elem });
            if is_last_pattern {
                // Optimized: reuse array buffer via TakeVar + PathInsert
                let acc = s.alloc_slot();
                s.emit(JitOp::TakeVar { dst: acc, var_index: acc_index });
                let key = s.alloc_slot();
                s.emit(JitOp::Num { dst: key, val: 0.0, repr: None });
                let val = s.alloc_slot();
                s.emit(JitOp::GetVar { dst: val, var_index: var_index });
                s.emit(JitOp::PathInsert { container: acc, key, val });
                s.emit(JitOp::Drop { slot: val });
                s.emit(JitOp::Drop { slot: key });
                s.emit(JitOp::SetVar { var_index: acc_index, src: acc });
                s.emit(JitOp::Drop { slot: acc });
            } else if let Some(rhs_expr) = acc_add_rhs {
                // Optimized: TakeVar + AddMove for `. + rhs`
                let acc = s.alloc_slot();
                s.emit(JitOp::TakeVar { dst: acc, var_index: acc_index });
                let rhs_val = s.flatten_scalar(rhs_expr, acc);
                let result = s.alloc_slot();
                s.emit(JitOp::AddMove { dst: result, lhs: acc, rhs: rhs_val });
                s.emit(JitOp::Drop { slot: rhs_val });
                s.emit(JitOp::Drop { slot: acc });
                s.emit(JitOp::SetVar { var_index: acc_index, src: result });
                s.emit(JitOp::Drop { slot: result });
            } else if let Some(ref info) = inplace_info {
                s.emit_reduce_update_with_lets(acc_index, info);
            } else {
                // Load current accumulator as the input to update (. refers to acc in reduce)
                let acc = s.alloc_slot();
                s.emit(JitOp::GetVar { dst: acc, var_index: acc_index });
                let new_acc = s.flatten_scalar(update, acc);
                s.emit(JitOp::Drop { slot: acc });
                s.emit(JitOp::SetVar { var_index: acc_index, src: new_acc });
                s.emit(JitOp::Drop { slot: new_acc });
            }
        };

        if is_scalar(source) {
            let val = self.flatten_scalar(source, input_slot);
            emit_update(self, val);
            self.emit(JitOp::Drop { slot: val });
            return true;
        }
        match source {
            Expr::Comma { left, right } => {
                if !self.flatten_gen_with_reduce(left, var_index, acc_index, update, input_slot) { return false; }
                self.flatten_gen_with_reduce(right, var_index, acc_index, update, input_slot)
            }
            Expr::Pipe { left, right } if is_scalar(left) => {
                let mid = self.flatten_scalar(left, input_slot);
                let ok = self.flatten_gen_with_reduce(right, var_index, acc_index, update, mid);
                self.emit(JitOp::Drop { slot: mid });
                ok
            }
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

    /// Iterate source for foreach: update acc and yield extract for each element.
    fn flatten_gen_with_foreach(&mut self, source: &Expr, var_index: u16, acc_index: u16,
                                update: &Expr, extract: Option<&Expr>, input_slot: SlotId) -> bool {
        // The update+extract step for each source element
        let emit_step = |s: &mut Flattener, elem: SlotId| {
            s.emit(JitOp::SetVar { var_index, src: elem });
            // Load current accumulator as input to update (. refers to acc in foreach)
            let acc = s.alloc_slot();
            s.emit(JitOp::GetVar { dst: acc, var_index: acc_index });
            let new_acc = s.flatten_scalar(update, acc);
            s.emit(JitOp::Drop { slot: acc });
            s.emit(JitOp::SetVar { var_index: acc_index, src: new_acc });
            s.emit(JitOp::Drop { slot: new_acc });
            // Now yield extract (evaluated with new accumulator as input)
            if let Some(ext) = extract {
                let ext_input = s.alloc_slot();
                s.emit(JitOp::GetVar { dst: ext_input, var_index: acc_index });
                s.flatten_gen(ext, ext_input);
                s.emit(JitOp::Drop { slot: ext_input });
            } else {
                let acc_out = s.alloc_slot();
                s.emit(JitOp::GetVar { dst: acc_out, var_index: acc_index });
                s.emit_yield(acc_out);
                s.emit(JitOp::Drop { slot: acc_out });
            }
        };

        if is_scalar(source) {
            let val = self.flatten_scalar(source, input_slot);
            emit_step(self, val);
            self.emit(JitOp::Drop { slot: val });
            return true;
        }
        match source {
            Expr::Each { input_expr } | Expr::EachOpt { input_expr } => {
                if !is_scalar(input_expr) { return false; }
                let is_opt = matches!(source, Expr::EachOpt { .. });
                let container = self.flatten_scalar(input_expr, input_slot);
                self.flatten_each_with_action(container, is_opt, &|s, elem| {
                    emit_step(s, elem);
                });
                self.emit(JitOp::Drop { slot: container });
                true
            }
            Expr::Comma { left, right } => {
                if !self.flatten_gen_with_foreach(left, var_index, acc_index, update, extract, input_slot) {
                    return false;
                }
                self.flatten_gen_with_foreach(right, var_index, acc_index, update, extract, input_slot)
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
                emit_step(self, num);
                self.emit(JitOp::Drop { slot: num });
                self.emit(JitOp::F64Add { dst_var: cur, a_var: cur, b_var: step_v });
                self.emit(JitOp::Jump { label: head });
                self.emit(JitOp::Label { id: done });
                true
            }
            _ => {
                // Generic source: use flatten_gen_with_each_output as a fallback
                self.flatten_gen_with_each_output(source, input_slot, &|s, elem| {
                    emit_step(s, elem);
                })
            }
        }
    }

    /// Handle LetBinding with generator value.
    /// For each output of value, set $var and run body with original input.
    fn flatten_gen_let_binding(&mut self, var_index: u16, value: &Expr, body: &Expr, input_slot: SlotId) -> bool {
        // Pre-check: can we compile the body?
        {
            let mut test = self.test_flattener();
            let t_in = test.alloc_slot();
            if !test.flatten_gen(body, t_in) { return false; }
        }

        let old = self.alloc_slot();
        self.emit(JitOp::GetVar { dst: old, var_index });

        match value {
            Expr::Comma { left, right } => {
                if !self.flatten_gen_let_binding(var_index, left, body, input_slot) { return false; }
                if !self.flatten_gen_let_binding(var_index, right, body, input_slot) { return false; }
            }
            _ if is_scalar(value) => {
                let val = self.flatten_scalar(value, input_slot);
                self.emit(JitOp::SetVar { var_index, src: val });
                self.emit(JitOp::Drop { slot: val });
                self.flatten_gen(body, input_slot);
            }
            Expr::Each { input_expr } if is_scalar(input_expr) => {
                let container = self.flatten_scalar(input_expr, input_slot);
                self.flatten_each_with_action(container, false, &|s, elem| {
                    s.emit(JitOp::SetVar { var_index, src: elem });
                    s.flatten_gen(body, input_slot);
                });
                self.emit(JitOp::Drop { slot: container });
            }
            _ => {
                // Generic fallback: collect value outputs, iterate each
                self.emit(JitOp::CollectBegin);
                self.collect_depth += 1;
                let ok = self.flatten_gen(value, input_slot);
                self.collect_depth -= 1;
                if !ok {
                    self.emit(JitOp::SetVar { var_index, src: old });
                    self.emit(JitOp::Drop { slot: old });
                    return false;
                }
                let collected = self.alloc_slot();
                self.emit(JitOp::CollectFinish { dst: collected });
                self.flatten_each_with_action(collected, false, &|s, elem| {
                    s.emit(JitOp::SetVar { var_index, src: elem });
                    s.flatten_gen(body, input_slot);
                });
                self.emit(JitOp::Drop { slot: collected });
            }
        }

        self.emit(JitOp::SetVar { var_index, src: old });
        self.emit(JitOp::Drop { slot: old });
        true
    }

    /// Handle if-then-else with generator condition.
    /// For each output of cond, check truthiness, run then/else with original input.
    fn flatten_gen_cond_if(&mut self, cond: &Expr, then_branch: &Expr, else_branch: &Expr, input_slot: SlotId) -> bool {
        // We handle this by wrapping the if-then-else body into a
        // "for each cond output" loop using the existing flatten_gen pipe mechanism.
        // Semantics: for each cond_val, if truthy run then_branch(input), else run else_branch(input)

        // The approach: emit the cond as a generator that yields to an inline handler
        // For Comma { left, right }, we can directly handle it
        match cond {
            Expr::Comma { left, right } => {
                // (A, B) as cond: handle A first, then B
                let then_clone = then_branch.clone();
                let else_clone = else_branch.clone();
                if !self.flatten_gen_cond_if(left, &then_clone, &else_clone, input_slot) { return false; }
                self.flatten_gen_cond_if(right, then_branch, else_branch, input_slot)
            }
            Expr::Empty => {
                // empty condition = no outputs = nothing happens
                true
            }
            _ if is_scalar(cond) => {
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
            _ => {
                // Generic fallback: collect cond outputs, iterate each
                self.emit(JitOp::CollectBegin);
                self.collect_depth += 1;
                let ok = self.flatten_gen(cond, input_slot);
                self.collect_depth -= 1;
                if !ok { return false; }
                let collected = self.alloc_slot();
                self.emit(JitOp::CollectFinish { dst: collected });
                self.flatten_each_with_action(collected, false, &|s, elem| {
                    let then_lbl = s.alloc_label();
                    let else_lbl = s.alloc_label();
                    let done_lbl = s.alloc_label();
                    s.emit(JitOp::IfTruthy { src: elem, then_label: then_lbl, else_label: else_lbl });
                    s.emit(JitOp::Label { id: then_lbl });
                    s.flatten_gen(then_branch, input_slot);
                    s.emit(JitOp::Jump { label: done_lbl });
                    s.emit(JitOp::Label { id: else_lbl });
                    s.flatten_gen(else_branch, input_slot);
                    s.emit(JitOp::Label { id: done_lbl });
                });
                self.emit(JitOp::Drop { slot: collected });
                true
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
            self.emit(JitOp::InitVar { var: idx });
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
            self.emit(JitOp::InitVar { var: idx });
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

    /// Build a path array Value from extracted path components.
    /// Returns a slot containing the path array.
    fn build_path_array(&mut self, components: &[PathComponent], input_slot: SlotId) -> SlotId {
        // Build array literal: [comp1, comp2, ...]
        self.emit(JitOp::CollectBegin);
        self.collect_depth += 1;
        for comp in components {
            match comp {
                PathComponent::Field(name) => {
                    let s = self.alloc_slot();
                    self.emit(JitOp::Str { dst: s, val: name.clone() });
                    self.ops.push(JitOp::CollectPush { src: s });
                    self.emit(JitOp::Drop { slot: s });
                }
                PathComponent::Expr(e) => {
                    let s = self.flatten_scalar(e, input_slot);
                    self.ops.push(JitOp::CollectPush { src: s });
                    self.emit(JitOp::Drop { slot: s });
                }
            }
        }
        self.collect_depth -= 1;
        let arr = self.alloc_slot();
        self.emit(JitOp::CollectFinish { dst: arr });
        arr
    }

    /// Emit optimized in-place reduce update with LetBinding wrapping.
    /// Handles `path |= f` and `path += f` (which is LetBinding { body: Update }).
    fn emit_reduce_update_with_lets(&mut self, acc_index: u16, info: &InplaceUpdateInfo) {
        // Save and set let-binding vars (for += desugaring: rhs evaluated first)
        let mut saved_vars = Vec::new();
        let acc_for_lets = self.alloc_slot();
        self.emit(JitOp::GetVar { dst: acc_for_lets, var_index: acc_index });
        for &(var_idx, ref value_expr) in &info.let_bindings {
            let old = self.alloc_slot();
            self.emit(JitOp::GetVar { dst: old, var_index: var_idx });
            let val = self.flatten_scalar(value_expr, acc_for_lets);
            self.emit(JitOp::SetVar { var_index: var_idx, src: val });
            self.emit(JitOp::Drop { slot: val });
            saved_vars.push((var_idx, old));
        }
        self.emit(JitOp::Drop { slot: acc_for_lets });

        // TakeVar: move acc out (refcount = 1)
        let acc = self.alloc_slot();
        self.emit(JitOp::TakeVar { dst: acc, var_index: acc_index });

        // PathExtract chain (multi-level)
        let mut containers = Vec::new();
        let mut keys = Vec::new();
        let mut current = acc;

        for component in &info.path_components {
            let key = match component {
                PathComponent::Field(name) => {
                    let s = self.alloc_slot();
                    self.emit(JitOp::Str { dst: s, val: name.clone() });
                    s
                }
                PathComponent::Expr(e) => {
                    self.flatten_scalar(e, current)
                }
            };
            let element = self.alloc_slot();
            self.emit(JitOp::PathExtract { element, container: current, key });
            containers.push(current);
            keys.push(key);
            current = element;
        }

        let old_val = current;

        // Detect fused += pattern: update_expr is `. + rhs` where rhs doesn't use Input
        let new_val = if let Expr::BinOp { op, lhs, rhs } = info.update_expr {
            if matches!(op, BinOp::Add) && matches!(lhs.as_ref(), Expr::Input) && !uses_input(rhs) {
                let rhs_val = self.flatten_scalar(rhs, acc);
                let out = self.alloc_slot();
                self.emit(JitOp::AddMove { dst: out, lhs: old_val, rhs: rhs_val });
                self.emit(JitOp::Drop { slot: rhs_val });
                out
            } else {
                self.flatten_scalar(info.update_expr, old_val)
            }
        } else {
            self.flatten_scalar(info.update_expr, old_val)
        };
        self.emit(JitOp::Drop { slot: old_val });

        // PathInsert chain (reverse)
        let mut current_val = new_val;
        for (container, key) in containers.iter().rev().zip(keys.iter().rev()) {
            self.emit(JitOp::PathInsert { container: *container, key: *key, val: current_val });
            self.emit(JitOp::Drop { slot: current_val });
            self.emit(JitOp::Drop { slot: *key });
            current_val = *container;
        }

        self.emit(JitOp::SetVar { var_index: acc_index, src: current_val });
        self.emit(JitOp::Drop { slot: current_val });

        // Restore let-binding vars
        for (var_idx, old) in saved_vars.into_iter().rev() {
            self.emit(JitOp::SetVar { var_index: var_idx, src: old });
            self.emit(JitOp::Drop { slot: old });
        }
    }
}

/// Path component for simple path extraction.
#[derive(Debug, Clone)]
enum PathComponent {
    Field(String),
    Expr(Expr),
}

/// Extract simple path components from a path expression.
/// Returns None if the path is too complex (contains generators like .[], etc.)
fn extract_simple_path(expr: &Expr) -> Option<Vec<PathComponent>> {
    match expr {
        Expr::Input => Some(vec![]),
        Expr::Index { expr: base, key } => {
            let mut components = extract_simple_path(base)?;
            match key.as_ref() {
                Expr::Literal(Literal::Str(s)) => {
                    components.push(PathComponent::Field(s.clone()));
                    Some(components)
                }
                Expr::Literal(Literal::Num(n, repr)) => {
                    components.push(PathComponent::Expr(Expr::Literal(Literal::Num(*n, repr.clone()))));
                    Some(components)
                }
                other if is_scalar(other) => {
                    components.push(PathComponent::Expr(other.clone()));
                    Some(components)
                }
                _ => None, // Generator key — too complex
            }
        }
        Expr::Pipe { left, right } => {
            let mut components = extract_simple_path(left)?;
            let right_components = extract_simple_path(right)?;
            components.extend(right_components);
            Some(components)
        }
        _ => None,
    }
}

/// Info for in-place reduce update optimization.
struct InplaceUpdateInfo<'a> {
    /// LetBindings wrapping the Update (from += desugaring).
    /// Each entry is (var_index, value_expr).
    let_bindings: Vec<(u16, &'a Expr)>,
    /// Path components for the update target.
    path_components: Vec<PathComponent>,
    /// The update expression (applied to the old value at the path).
    update_expr: &'a Expr,
}

/// Detect an in-place update pattern in a reduce update expression.
/// Handles direct `path |= f` and `path += f` (which is `LetBinding { body: Update }`).
fn detect_inplace_update(expr: &Expr) -> Option<InplaceUpdateInfo<'_>> {
    match expr {
        Expr::Update { path_expr, update_expr } => {
            let pc = extract_simple_path(path_expr)?;
            if !pc.is_empty() && is_scalar(update_expr) {
                Some(InplaceUpdateInfo {
                    let_bindings: vec![],
                    path_components: pc,
                    update_expr,
                })
            } else {
                None
            }
        }
        Expr::LetBinding { var_index, value, body } if is_scalar(value) => {
            let mut info = detect_inplace_update(body)?;
            // Prepend this let-binding (outermost first)
            info.let_bindings.insert(0, (*var_index, value.as_ref()));
            Some(info)
        }
        _ => None,
    }
}

/// Check if an expression tree contains any FuncCall references.
/// Used to guard runtime delegation (which loses function context).
fn contains_func_call(expr: &Expr) -> bool {
    match expr {
        Expr::FuncCall { .. } => true,
        Expr::Pipe { left, right } => contains_func_call(left) || contains_func_call(right),
        Expr::Comma { left, right } => contains_func_call(left) || contains_func_call(right),
        Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => contains_func_call(expr) || contains_func_call(key),
        Expr::Each { input_expr } | Expr::EachOpt { input_expr } => contains_func_call(input_expr),
        Expr::IfThenElse { cond, then_branch, else_branch } => contains_func_call(cond) || contains_func_call(then_branch) || contains_func_call(else_branch),
        Expr::LetBinding { value, body, .. } => contains_func_call(value) || contains_func_call(body),
        Expr::TryCatch { try_expr, catch_expr } => contains_func_call(try_expr) || contains_func_call(catch_expr),
        Expr::Collect { generator } => contains_func_call(generator),
        Expr::BinOp { lhs, rhs, .. } => contains_func_call(lhs) || contains_func_call(rhs),
        Expr::UnaryOp { operand, .. } | Expr::Negate { operand } => contains_func_call(operand),
        Expr::CallBuiltin { args, .. } => args.iter().any(contains_func_call),
        Expr::Update { path_expr, update_expr } => contains_func_call(path_expr) || contains_func_call(update_expr),
        Expr::Assign { path_expr, value_expr } => contains_func_call(path_expr) || contains_func_call(value_expr),
        Expr::Alternative { primary, fallback } => contains_func_call(primary) || contains_func_call(fallback),
        Expr::Slice { expr, from, to } => contains_func_call(expr) || from.as_ref().is_some_and(|f| contains_func_call(f)) || to.as_ref().is_some_and(|t| contains_func_call(t)),
        _ => false, // Literals, Input, LoadVar, etc. are safe
    }
}

/// Check if an expression uses Input (`.`). Used to determine if the rhs of
/// a += can be evaluated independently of the path target.
fn uses_input(expr: &Expr) -> bool {
    match expr {
        Expr::Input => true,
        Expr::Literal(_) | Expr::LoadVar { .. } | Expr::Empty => false,
        Expr::BinOp { lhs, rhs, .. } => uses_input(lhs) || uses_input(rhs),
        Expr::UnaryOp { operand, .. } | Expr::Negate { operand } => uses_input(operand),
        Expr::Pipe { left, right } | Expr::Comma { left, right } => uses_input(left) || uses_input(right),
        Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => uses_input(expr) || uses_input(key),
        Expr::Collect { generator } => uses_input(generator),
        Expr::IfThenElse { cond, then_branch, else_branch } => uses_input(cond) || uses_input(then_branch) || uses_input(else_branch),
        Expr::LetBinding { value, body, .. } => uses_input(value) || uses_input(body),
        Expr::CallBuiltin { args, .. } => args.iter().any(|a| uses_input(a)),
        _ => true, // Conservative: assume uses input
    }
}

// ============================================================================
// Runtime helpers
// ============================================================================

extern "C" fn jit_rt_clone(dst: *mut Value, src: *const Value) {
    unsafe { std::ptr::write(dst, (*src).clone()); }
}
extern "C" fn jit_rt_drop(v: *mut Value) {
    unsafe {
        // Fast path: try to pool Rc<ObjMap> instead of deallocating
        let val = std::ptr::read(v);
        if let Value::Obj(rc) = val {
            if !crate::value::rc_objmap_pool_return(rc) {
                // Pool full or refcount > 1, drop normally (rc is consumed above)
            }
        } else {
            drop(val);
        }
    }
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
        std::ptr::write(dst, Value::Str(crate::value::KeyStr::from(s)));
    }
}
/// Copy a pre-allocated CompactString constant (24-byte inline copy).
extern "C" fn jit_rt_str_rc(dst: *mut Value, cs_ptr: *const crate::value::KeyStr) {
    unsafe {
        std::ptr::write(dst, Value::Str((*cs_ptr).clone()));
    }
}
extern "C" fn jit_rt_is_truthy(v: *const Value) -> i64 {
    unsafe { if (*v).is_truthy() { 1 } else { 0 } }
}
/// Combined field access + truthiness check — avoids cloning and dropping the field value.
extern "C" fn jit_rt_field_is_truthy(base: *const Value, key_ptr: *const u8, key_len: usize) -> i64 {
    unsafe {
        match &*base {
            Value::Obj(o) => {
                let key = std::str::from_utf8_unchecked(std::slice::from_raw_parts(key_ptr, key_len));
                match o.get(key) {
                    Some(v) => if v.is_truthy() { 1 } else { 0 },
                    None => 0,
                }
            }
            _ => 0,
        }
    }
}
/// Combined field access + numeric comparison — avoids creating intermediate Value objects.
/// Returns 1 if the comparison is true, 0 otherwise.
/// op encoding: 5=Eq, 6=Ne, 7=Lt, 8=Gt, 9=Le, 10=Ge
extern "C" fn jit_rt_field_cmp_num(base: *const Value, key_ptr: *const u8, key_len: usize, rhs: f64, op: i32) -> i64 {
    unsafe {
        let field_val = match &*base {
            Value::Obj(o) => {
                let key = std::str::from_utf8_unchecked(std::slice::from_raw_parts(key_ptr, key_len));
                match o.get(key) {
                    Some(v) => v,
                    None => return 0,
                }
            }
            _ => return 0,
        };
        if let Value::Num(lhs, _) = field_val {
            let result = match op {
                5 => lhs == &rhs,
                6 => lhs != &rhs,
                7 => lhs < &rhs,
                8 => lhs > &rhs,
                9 => lhs <= &rhs,
                10 => lhs >= &rhs,
                _ => return 0,
            };
            if result { 1 } else { 0 }
        } else {
            0
        }
    }
}
/// Combined two-field lookup + binop: base.field_a OP base.field_b
/// Avoids creating/dropping intermediate Value objects for the field lookups.
extern "C" fn jit_rt_field_binop_field(
    dst: *mut Value, base: *const Value,
    ka_ptr: *const u8, ka_len: usize,
    kb_ptr: *const u8, kb_len: usize,
    op: i32,
) -> i64 {
    unsafe {
        let obj = match &*base {
            Value::Obj(o) => o,
            Value::Null => {
                // null.field = null, so this is null OP null
                match crate::eval::eval_binop(binop_from_i32(op), &Value::Null, &Value::Null) {
                    Ok(v) => { std::ptr::write(dst, v); return 0; }
                    Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); return GEN_ERROR; }
                }
            }
            _ => {
                set_jit_error(format!("Cannot index {} with string", (*base).type_name()));
                std::ptr::write(dst, Value::Null);
                return GEN_ERROR;
            }
        };
        let ka = std::slice::from_raw_parts(ka_ptr, ka_len);
        let kb = std::slice::from_raw_parts(kb_ptr, kb_len);
        // Single-pass scan for both fields
        let mut va: Option<&Value> = None;
        let mut vb: Option<&Value> = None;
        for (k, v) in obj.iter() {
            let kb_entry = k.as_bytes();
            if va.is_none() && kb_entry == ka { va = Some(v); if vb.is_some() { break; } }
            else if vb.is_none() && kb_entry == kb { vb = Some(v); if va.is_some() { break; } }
        }
        // Fast path: both fields are Num (most common case for arithmetic)
        if let (Some(Value::Num(na, _)), Some(Value::Num(nb, _))) = (va, vb) {
            match op {
                0 => { std::ptr::write(dst, Value::Num(na + nb, None)); return 0; }
                1 => { std::ptr::write(dst, Value::Num(na - nb, None)); return 0; }
                2 => { std::ptr::write(dst, Value::Num(na * nb, None)); return 0; }
                5 => { std::ptr::write(dst, Value::from_bool(na == nb)); return 0; }
                6 => { std::ptr::write(dst, Value::from_bool(na != nb)); return 0; }
                7 => { std::ptr::write(dst, Value::from_bool(na < nb)); return 0; }
                8 => { std::ptr::write(dst, Value::from_bool(na > nb)); return 0; }
                9 => { std::ptr::write(dst, Value::from_bool(na <= nb)); return 0; }
                10 => { std::ptr::write(dst, Value::from_bool(na >= nb)); return 0; }
                _ => {} // div, mod: fall through to general path
            }
        }
        // General path: clone field values and use eval_binop
        let a = va.cloned().unwrap_or(Value::Null);
        let b = vb.cloned().unwrap_or(Value::Null);
        match crate::eval::eval_binop(binop_from_i32(op), &a, &b) {
            Ok(v) => { std::ptr::write(dst, v); 0 }
            Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); GEN_ERROR }
        }
    }
}
/// Combined field lookup + binop with constant: field OP const (or const OP field).
/// field_is_lhs: 1 → field OP const, 0 → const OP field.
extern "C" fn jit_rt_field_binop_const(
    dst: *mut Value, base: *const Value,
    key_ptr: *const u8, key_len: usize,
    rhs: *const Value, op: i32, field_is_lhs: i32,
) -> i64 {
    unsafe {
        let obj = match &*base {
            Value::Obj(o) => o,
            Value::Null => {
                let a = Value::Null;
                let (l, r) = if field_is_lhs != 0 { (&a, &*rhs) } else { (&*rhs, &a) };
                match crate::eval::eval_binop(binop_from_i32(op), l, r) {
                    Ok(v) => { std::ptr::write(dst, v); return 0; }
                    Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); return GEN_ERROR; }
                }
            }
            _ => {
                set_jit_error(format!("Cannot index {} with string", (*base).type_name()));
                std::ptr::write(dst, Value::Null);
                return GEN_ERROR;
            }
        };
        let key = std::str::from_utf8_unchecked(std::slice::from_raw_parts(key_ptr, key_len));
        let field_val = obj.get(key).unwrap_or(&Value::Null);
        let rhs_val = &*rhs;
        let (lv, rv) = if field_is_lhs != 0 { (field_val, rhs_val) } else { (rhs_val, field_val) };
        // Fast path: Num OP Num
        if let (Value::Num(na, _), Value::Num(nb, _)) = (lv, rv) {
            match op {
                0 => { std::ptr::write(dst, Value::Num(na + nb, None)); return 0; }
                1 => { std::ptr::write(dst, Value::Num(na - nb, None)); return 0; }
                2 => { std::ptr::write(dst, Value::Num(na * nb, None)); return 0; }
                5 => { std::ptr::write(dst, Value::from_bool(na == nb)); return 0; }
                6 => { std::ptr::write(dst, Value::from_bool(na != nb)); return 0; }
                7 => { std::ptr::write(dst, Value::from_bool(na < nb)); return 0; }
                8 => { std::ptr::write(dst, Value::from_bool(na > nb)); return 0; }
                9 => { std::ptr::write(dst, Value::from_bool(na <= nb)); return 0; }
                10 => { std::ptr::write(dst, Value::from_bool(na >= nb)); return 0; }
                _ => {}
            }
        }
        // Fast path: Str + Str (concatenation)
        if op == 0 {
            if let (Value::Str(a), Value::Str(b_str)) = (lv, rv) {
                let mut cs = crate::value::KeyStr::new(a.as_str());
                cs.push_str(b_str.as_str());
                std::ptr::write(dst, Value::Str(cs));
                return 0;
            }
        }
        // General path
        let a = lv.clone();
        let b = rv.clone();
        match crate::eval::eval_binop(binop_from_i32(op), &a, &b) {
            Ok(v) => { std::ptr::write(dst, v); 0 }
            Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); GEN_ERROR }
        }
    }
}
extern "C" fn jit_rt_index_field(dst: *mut Value, base: *const Value, key_ptr: *const u8, key_len: usize) -> i64 {
    unsafe {
        match &*base {
            Value::Obj(o) => {
                let key = std::str::from_utf8_unchecked(std::slice::from_raw_parts(key_ptr, key_len));
                match o.get(key) {
                    Some(v) => { std::ptr::write(dst, v.clone()); 0 }
                    None => { std::ptr::write(dst, Value::Null); 0 }
                }
            }
            Value::Null => { std::ptr::write(dst, Value::Null); 0 }
            _ => {
                let key = std::str::from_utf8_unchecked(std::slice::from_raw_parts(key_ptr, key_len));
                let key_val = Value::from_str(key);
                match crate::eval::eval_index(&*base, &key_val, false) {
                    Ok(v) => { std::ptr::write(dst, v); 0 }
                    Err(e) => {
                        set_jit_error(e.to_string());
                        std::ptr::write(dst, Value::Null); GEN_ERROR
                    }
                }
            }
        }
    }
}
/// Fused field lookup + yield: borrows field from base, calls callback, no clone/drop.
/// For non-object bases or missing fields, creates a temporary Null value.
/// Returns the callback result (GEN_CONTINUE or 0).
extern "C" fn jit_rt_yield_field_ref(
    base: *const Value, key_ptr: *const u8, key_len: usize,
    cb: unsafe extern "C" fn(*const Value, *mut u8) -> i64, ctx: *mut u8,
) -> i64 {
    unsafe {
        match &*base {
            Value::Obj(o) => {
                let key = std::str::from_utf8_unchecked(std::slice::from_raw_parts(key_ptr, key_len));
                match o.get(key) {
                    Some(v) => cb(v as *const Value, ctx),
                    None => {
                        let null = Value::Null;
                        cb(&null as *const Value, ctx)
                    }
                }
            }
            Value::Null => {
                let null = Value::Null;
                cb(&null as *const Value, ctx)
            }
            _ => {
                // Non-object: clone + yield + drop (rare fallback)
                let key = std::str::from_utf8_unchecked(std::slice::from_raw_parts(key_ptr, key_len));
                let key_val = Value::from_str(key);
                match crate::eval::eval_index(&*base, &key_val, false) {
                    Ok(v) => {
                        let result = cb(&v as *const Value, ctx);
                        result
                    }
                    Err(e) => {
                        set_jit_error(e.to_string());
                        GEN_ERROR
                    }
                }
            }
        }
    }
}
extern "C" fn jit_rt_index(dst: *mut Value, base: *const Value, key: *const Value) -> i64 {
    unsafe {
        // Fast path: Array[Num] — most common in loops
        if let (Value::Arr(a), Value::Num(n, _)) = (&*base, &*key) {
            if n.is_nan() {
                std::ptr::write(dst, Value::Null);
                return 0;
            }
            let idx = *n as i64;
            let i = if idx < 0 { (a.len() as i64 + idx) as usize } else { idx as usize };
            std::ptr::write(dst, a.get(i).cloned().unwrap_or(Value::Null));
            return 0;
        }
        // Fast path: Object[String]
        if let (Value::Obj(o), Value::Str(k)) = (&*base, &*key) {
            std::ptr::write(dst, o.get(k.as_str()).cloned().unwrap_or(Value::Null));
            return 0;
        }
        // Fast path: null[anything] = null
        if matches!(&*base, Value::Null) {
            std::ptr::write(dst, Value::Null);
            return 0;
        }
        match crate::eval::eval_index(&*base, &*key, false) {
            Ok(v) => { std::ptr::write(dst, v); 0 }
            Err(e) => {
                set_jit_error(e.to_string());
                std::ptr::write(dst, Value::Null); GEN_ERROR
            }
        }
    }
}
extern "C" fn jit_rt_binop(dst: *mut Value, op: i32, lhs: *const Value, rhs: *const Value) -> i64 {
    unsafe {
        // Fast path for Num op Num (most common case in filters)
        if let (Value::Num(a, _), Value::Num(b, _)) = (&*lhs, &*rhs) {
            match op {
                0 => { std::ptr::write(dst, Value::Num(a + b, None)); return 0; }
                1 => { std::ptr::write(dst, Value::Num(a - b, None)); return 0; }
                2 => { std::ptr::write(dst, Value::Num(a * b, None)); return 0; }
                3 => { // Div: fast path for non-zero, non-NaN
                    if *b != 0.0 && !b.is_nan() && !a.is_nan() {
                        std::ptr::write(dst, Value::Num(a / b, None)); return 0;
                    }
                }
                4 => { // Mod: fast path for non-zero integer modulo
                    if !b.is_nan() && !a.is_nan() {
                        let yi = *b as i64;
                        if yi != 0 {
                            std::ptr::write(dst, Value::Num((*a as i64 % yi) as f64, None)); return 0;
                        }
                    }
                }
                5 => { std::ptr::write(dst, Value::from_bool(a == b)); return 0; }
                6 => { std::ptr::write(dst, Value::from_bool(a != b)); return 0; }
                7 => { std::ptr::write(dst, Value::from_bool(a < b)); return 0; }
                8 => { std::ptr::write(dst, Value::from_bool(a > b)); return 0; }
                9 => { std::ptr::write(dst, Value::from_bool(a <= b)); return 0; }
                10 => { std::ptr::write(dst, Value::from_bool(a >= b)); return 0; }
                _ => {}
            }
        }
        // Fast path for Str + Str (string concatenation)
        // Use CompactString directly to avoid intermediate String heap allocation for short results.
        if op == 0 {
            if let (Value::Str(a), Value::Str(b)) = (&*lhs, &*rhs) {
                let mut cs = crate::value::KeyStr::new(a.as_str());
                cs.push_str(b.as_str());
                std::ptr::write(dst, Value::Str(cs));
                return 0;
            }
        }
        let binop = match op {
            0 => BinOp::Add, 1 => BinOp::Sub, 2 => BinOp::Mul, 3 => BinOp::Div,
            4 => BinOp::Mod, 5 => BinOp::Eq, 6 => BinOp::Ne, 7 => BinOp::Lt,
            8 => BinOp::Gt, 9 => BinOp::Le, 10 => BinOp::Ge,
            _ => { set_jit_error("invalid binop".to_string()); std::ptr::write(dst, Value::Null); return GEN_ERROR; }
        };
        match crate::eval::eval_binop(binop, &*lhs, &*rhs) {
            Ok(v) => { std::ptr::write(dst, v); 0 }
            Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); GEN_ERROR }
        }
    }
}
/// Add with move semantics: takes ownership of lhs (writes Null in its place).
/// This enables in-place string/array/object append when lhs has refcount 1.
extern "C" fn jit_rt_add_move(dst: *mut Value, lhs: *mut Value, rhs: *const Value) -> i64 {
    unsafe {
        // Fast path: Num + Num
        if let (Value::Num(a, _), Value::Num(b, _)) = (&*lhs, &*rhs) {
            let result = Value::Num(a + b, None);
            std::ptr::write(dst, result);
            return 0;
        }
        // Move lhs out, enabling in-place mutation
        let lhs_val = std::ptr::read(lhs);
        std::ptr::write(lhs, Value::Null);
        match crate::runtime::rt_add_owned(lhs_val, &*rhs) {
            Ok(v) => { std::ptr::write(dst, v); 0 }
            Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); GEN_ERROR }
        }
    }
}
extern "C" fn jit_rt_unaryop(dst: *mut Value, op: i32, input: *const Value) -> i64 {
    unsafe {
        match unaryop_from_i32(op) {
            Some(u) => match crate::eval::eval_unaryop(u, &*input) {
                Ok(v) => { std::ptr::write(dst, v); 0 }
                Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); GEN_ERROR }
            },
            None => { set_jit_error("invalid unaryop".to_string()); std::ptr::write(dst, Value::Null); GEN_ERROR }
        }
    }
}
extern "C" fn jit_rt_throw_error(msg: *const Value, env: *mut JitEnv) -> i64 {
    unsafe {
        let val = (*msg).clone();
        // Store value directly — defer serialization until propagation (if not caught)
        *JIT_ERROR_VALUE.0.get() = Some(val);
        let _ = env;
        GEN_ERROR
    }
}
/// In-place reverse: Rc::make_mut avoids inner Vec clone when refcount == 1.
extern "C" fn jit_rt_reverse_inplace(v: *mut Value) -> i64 {
    unsafe {
        match &mut *v {
            Value::Arr(a) => { Rc::make_mut(a).reverse(); 0 }
            Value::Str(s) => {
                *v = Value::from_string(s.chars().rev().collect());
                0
            }
            Value::Null => { *v = Value::Arr(Rc::new(vec![])); 0 }
            _ => { set_jit_error(format!("{} cannot be reversed", (*v).type_name())); GEN_ERROR }
        }
    }
}
/// In-place sort.
extern "C" fn jit_rt_sort_inplace(v: *mut Value) -> i64 {
    unsafe {
        match &mut *v {
            Value::Arr(a) => {
                let arr = Rc::make_mut(a);
                arr.sort_by(crate::runtime::compare_values);
                0
            }
            Value::Null => { *v = Value::Arr(Rc::new(vec![])); 0 }
            _ => { set_jit_error(format!("{} is not an array", (*v).type_name())); GEN_ERROR }
        }
    }
}
extern "C" fn jit_rt_negate(dst: *mut Value, input: *const Value) -> i64 {
    unsafe {
        match &*input {
            Value::Num(n, _) => { std::ptr::write(dst, Value::Num(-n, None)); 0 }
            _ => {
                set_jit_error(format!("{} cannot be negated", crate::runtime::errdesc_pub(&*input)));
                std::ptr::write(dst, Value::Null); GEN_ERROR
            }
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

/// Take var value (move out, leave Null). Gives the caller sole ownership (refcount unincremented).
extern "C" fn jit_rt_take_var(dst: *mut Value, env: *mut JitEnv, idx: u32) {
    unsafe {
        let env = &mut *env;
        let i = idx as usize;
        let v = if i < env.vars.len() {
            std::mem::replace(&mut env.vars[i], Value::Null)
        } else {
            Value::Null
        };
        std::ptr::write(dst, v);
    }
}

/// In-place path extract: swap container[key] with Null, writing old element to `element`.
/// Uses Rc::make_mut so no clone occurs when refcount == 1.
extern "C" fn jit_rt_path_extract(element: *mut Value, container: *mut Value, key: *const Value) -> i64 {
    unsafe {
        let cont = &mut *container;
        let key_ref = &*key;
        match (cont, key_ref) {
            (Value::Arr(rc_arr), Value::Num(n, _)) => {
                let idx = if *n < 0.0 {
                    let len = rc_arr.len() as i64;
                    (len + *n as i64).max(0) as usize
                } else {
                    *n as usize
                };
                let arr = Rc::make_mut(rc_arr);
                if idx < arr.len() {
                    std::ptr::write(element, std::mem::replace(&mut arr[idx], Value::Null));
                } else {
                    std::ptr::write(element, Value::Null);
                }
                0
            }
            (Value::Obj(rc_obj), Value::Str(k)) => {
                let obj = Rc::make_mut(rc_obj);
                let found = obj.iter_mut().find(|(ek, _)| ek.as_str() == k.as_str());
                if let Some((_, v)) = found {
                    std::ptr::write(element, std::mem::replace(v, Value::Null));
                } else {
                    std::ptr::write(element, Value::Null);
                }
                0
            }
            _ => {
                std::ptr::write(element, Value::Null);
                0
            }
        }
    }
}

/// In-place path insert: set container[key] = val, consuming val (slot becomes Null).
/// Uses Rc::make_mut so no clone occurs when refcount == 1.
extern "C" fn jit_rt_path_insert(container: *mut Value, key: *const Value, val: *mut Value) -> i64 {
    unsafe {
        let new_val = std::ptr::read(val);
        std::ptr::write(val, Value::Null);
        let cont = &mut *container;
        let key_ref = &*key;
        match (cont, key_ref) {
            (Value::Arr(rc_arr), Value::Num(n, _)) => {
                let idx = if *n < 0.0 {
                    let len = rc_arr.len() as i64;
                    (len + *n as i64).max(0) as usize
                } else {
                    *n as usize
                };
                let arr = Rc::make_mut(rc_arr);
                while arr.len() <= idx {
                    arr.push(Value::Null);
                }
                arr[idx] = new_val;
                0
            }
            (Value::Obj(rc_obj), Value::Str(k)) => {
                let obj = Rc::make_mut(rc_obj);
                let found = obj.iter_mut().find(|(ek, _)| ek.as_str() == k.as_str());
                if let Some((_, v)) = found {
                    *v = new_val;
                } else {
                    obj.insert(crate::value::KeyStr::from(k.as_str()), new_val);
                }
                0
            }
            (Value::Null, Value::Num(n, _)) => {
                let idx = (*n).max(0.0) as usize;
                let mut arr = vec![Value::Null; idx + 1];
                arr[idx] = new_val;
                *container = Value::Arr(Rc::new(arr));
                0
            }
            (Value::Null, Value::Str(k)) => {
                let mut obj = crate::value::new_objmap();
                obj.insert(crate::value::KeyStr::from(k.as_str()), new_val);
                *container = Value::Obj(Rc::new(obj));
                0
            }
            _ => {
                drop(new_val);
                0
            }
        }
    }
}

// Collect (array construction) helpers
extern "C" fn jit_rt_collect_begin(env: *mut JitEnv) {
    unsafe { (*env).collect_stacks.push(Vec::with_capacity(16)); }
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
extern "C" fn jit_rt_obj_new(dst: *mut Value, cap: usize) {
    unsafe { std::ptr::write(dst, Value::Obj(crate::value::rc_objmap_pool_get(cap))); }
}
/// Insert key-value pair into object. Takes ownership of both key and val (ptr::read, no clone).
extern "C" fn jit_rt_obj_insert(obj: *mut Value, key: *mut Value, val: *mut Value) {
    unsafe {
        let key_val = std::ptr::read(key);
        let val_val = std::ptr::read(val);
        if let Value::Obj(o) = &mut *obj {
            // Move KeyStr out of Value::Str to avoid clone+drop
            let k: crate::value::KeyStr = match key_val {
                Value::Str(s) => s,
                other => {
                    let k = crate::value::KeyStr::from(crate::value::value_to_json(&other));
                    drop(other);
                    k
                }
            };
            // Safety: obj was just created by jit_rt_obj_new, refcount is always 1
            Rc::get_mut(o).unwrap_unchecked().insert(k, val_val);
        }
    }
}

/// Insert into object with raw string key — avoids creating/dropping a Value for the key.
extern "C" fn jit_rt_obj_insert_str_key(obj: *mut Value, key_ptr: *const u8, key_len: usize, val: *mut Value) {
    unsafe {
        let val_val = std::ptr::read(val);
        if let Value::Obj(o) = &mut *obj {
            let key = std::str::from_utf8_unchecked(std::slice::from_raw_parts(key_ptr, key_len));
            let k = crate::value::KeyStr::from(key);
            Rc::get_mut(o).unwrap_unchecked().insert(k, val_val);
        }
    }
}

/// Push into object with raw string key — skips duplicate check (caller guarantees unique keys).
extern "C" fn jit_rt_obj_push_str_key(obj: *mut Value, key_ptr: *const u8, key_len: usize, val: *mut Value) {
    unsafe {
        let val_val = std::ptr::read(val);
        if let Value::Obj(o) = &mut *obj {
            let key = std::str::from_utf8_unchecked(std::slice::from_raw_parts(key_ptr, key_len));
            let k = crate::value::KeyStr::from(key);
            Rc::get_mut(o).unwrap_unchecked().push_unique(k, val_val);
        }
    }
}

/// Fused field extraction + push: look up src_field in src object, push clone into dst object with obj_key.
/// Combines jit_rt_index_field + jit_rt_obj_push_str_key into one call, saving function call
/// overhead and intermediate slot read/write.
extern "C" fn jit_rt_obj_copy_field(
    obj: *mut Value,
    src: *const Value,
    obj_key_ptr: *const u8, obj_key_len: usize,
    src_field_ptr: *const u8, src_field_len: usize,
) {
    unsafe {
        let src_field = std::str::from_utf8_unchecked(std::slice::from_raw_parts(src_field_ptr, src_field_len));
        let val = match &*src {
            Value::Obj(o) => o.get(src_field).cloned().unwrap_or(Value::Null),
            Value::Null => Value::Null,
            _ => Value::Null,
        };
        if let Value::Obj(o) = &mut *obj {
            let obj_key = std::str::from_utf8_unchecked(std::slice::from_raw_parts(obj_key_ptr, obj_key_len));
            let k = crate::value::KeyStr::from(obj_key);
            Rc::get_mut(o).unwrap_unchecked().push_unique(k, val);
        }
    }
}

/// Batch field extraction: build a new object from multiple fields of src in a single pass.
/// pair_table is an array of (out_key_ptr, out_key_len, src_field_ptr, src_field_len) tuples.
/// Replaces ObjNew + N×ObjCopyField with a single function call, scanning the source once.
/// Output fields are always in filter order (pair_table order), not source order.
extern "C" fn jit_rt_obj_from_fields(
    dst: *mut Value,
    src: *const Value,
    pair_table: *const (*const u8, usize, *const u8, usize),
    count: usize,
) {
    unsafe {
        let pairs = std::slice::from_raw_parts(pair_table, count);
        if let Value::Obj(src_obj) = &*src {
            // General path: single-pass extraction with field-order output
            let mut vals: [std::mem::MaybeUninit<Value>; 16] = std::mem::MaybeUninit::uninit().assume_init();
            let use_heap = count > 16;
            let mut heap_vals: Vec<std::mem::MaybeUninit<Value>> = if use_heap {
                let mut v = Vec::with_capacity(count);
                v.resize_with(count, std::mem::MaybeUninit::uninit);
                v
            } else { Vec::new() };
            let val_buf: &mut [std::mem::MaybeUninit<Value>] = if use_heap { &mut heap_vals } else { &mut vals[..count] };
            let mut found = 0u64;
            let all_found: u64 = if count < 64 { (1u64 << count) - 1 } else { u64::MAX };
            for (k, v) in src_obj.iter() {
                if found == all_found { break; }
                let k_bytes = k.as_bytes();
                for (fi, &(_, _, sfp, sfl)) in pairs.iter().enumerate() {
                    if fi < 64 && (found & (1u64 << fi)) != 0 { continue; }
                    if k_bytes.len() == sfl && k_bytes == std::slice::from_raw_parts(sfp, sfl) {
                        val_buf[fi].write(v.clone());
                        found |= 1u64 << fi;
                        break;
                    }
                }
            }
            let mut obj = crate::value::rc_objmap_pool_get(count);
            let map = Rc::get_mut(&mut obj).unwrap_unchecked();
            for (fi, &(okp, okl, _, _)) in pairs.iter().enumerate() {
                let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(okp, okl));
                let val = if fi < 64 && (found & (1u64 << fi)) != 0 {
                    val_buf[fi].assume_init_read()
                } else {
                    Value::Null
                };
                map.push_unique(crate::value::KeyStr::from(name), val);
            }
            std::ptr::write(dst, Value::Obj(obj));
        } else {
            // Non-object source: all fields null
            let mut obj = crate::value::rc_objmap_pool_get(count);
            let map = Rc::get_mut(&mut obj).unwrap_unchecked();
            for &(okp, okl, _, _) in pairs {
                let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(okp, okl));
                map.push_unique(crate::value::KeyStr::from(name), Value::Null);
            }
            std::ptr::write(dst, Value::Obj(obj));
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
        std::ptr::write(dst, Value::from_string(s));
    }
}

// CallBuiltin helper: call runtime::call_builtin with name and args
extern "C" fn jit_rt_call_builtin(dst: *mut Value, name_ptr: *const u8, name_len: usize,
                                   args_ptr: *const Value, nargs: usize) -> i64 {
    unsafe {
        let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(name_ptr, name_len));
        let args = std::slice::from_raw_parts(args_ptr, nargs);
        let args_vec: Vec<Value> = args.to_vec();

        // Handle special names
        if let Some(rest) = name.strip_prefix("__loc__:") {
            // Parse file:line
            let parts: Vec<&str> = rest.rsplitn(2, ':').collect();
            if parts.len() == 2 {
                let line_n: i64 = parts[0].parse().unwrap_or(0);
                let file = parts[1];
                let mut obj = crate::value::new_objmap();
                obj.insert(crate::value::KeyStr::from("file"), Value::from_str(file));
                obj.insert(crate::value::KeyStr::from("line"), Value::Num(line_n as f64, None));
                std::ptr::write(dst, Value::Obj(Rc::new(obj)));
                return 0;
            }
        }
        if name == "__env__" {
            let mut obj = crate::value::new_objmap();
            for (k, v) in std::env::vars() {
                obj.insert(crate::value::KeyStr::from(k), Value::from_string(v));
            }
            std::ptr::write(dst, Value::Obj(Rc::new(obj)));
            return 0;
        }
        if name == "__each_error__" {
            // Generate "Cannot iterate over TYPE (VALUE)" error
            if !args_vec.is_empty() {
                let v = &args_vec[0];
                let ty = v.type_name();
                let json = crate::value::value_to_json(v);
                set_jit_error(format!("Cannot iterate over {} ({})", ty, json));
                std::ptr::write(dst, Value::Null);
                return GEN_ERROR;
            }
        }
        if name == "__builtins__" {
            std::ptr::write(dst, crate::runtime::rt_builtins());
            return 0;
        }
        if name == "recurse_collect" {
            // Collect all recursive descendants: ., .[]?, (.[]? | .[]?), ...
            if !args_vec.is_empty() {
                let mut results = Vec::new();
                let mut stack = vec![args_vec[0].clone()];
                while let Some(v) = stack.pop() {
                    results.push(v.clone());
                    match &v {
                        Value::Arr(a) => {
                            for item in a.iter().rev() {
                                stack.push(item.clone());
                            }
                        }
                        Value::Obj(o) => {
                            for val in o.values().rev() {
                                stack.push(val.clone());
                            }
                        }
                        _ => {}
                    }
                }
                std::ptr::write(dst, Value::Arr(Rc::new(results)));
            } else {
                std::ptr::write(dst, Value::Arr(Rc::new(vec![])));
            }
            return 0;
        }
        if name == "debug" {
            // debug: print to stderr, return input
            if args_vec.len() >= 2 {
                let input = &args_vec[0];
                let label = &args_vec[1];
                match label {
                    Value::Str(s) if !s.is_empty() => {
                        eprintln!("[\"DEBUG:\",{}]", crate::value::value_to_json(input));
                    }
                    _ => {
                        eprintln!("[\"DEBUG:\",{}]", crate::value::value_to_json(input));
                    }
                }
                std::ptr::write(dst, input.clone());
            } else if !args_vec.is_empty() {
                eprintln!("[\"DEBUG:\",{}]", crate::value::value_to_json(&args_vec[0]));
                std::ptr::write(dst, args_vec[0].clone());
            } else {
                std::ptr::write(dst, Value::Null);
            }
            return 0;
        }
        if name == "stderr" {
            if !args_vec.is_empty() {
                eprint!("{}", crate::value::value_to_json(&args_vec[0]));
                std::ptr::write(dst, args_vec[0].clone());
            } else {
                std::ptr::write(dst, Value::Null);
            }
            return 0;
        }
        if name == "_slice" {
            // _slice(base, from, to)
            if args_vec.len() >= 3 {
                match crate::eval::eval_slice(&args_vec[0], &args_vec[1], &args_vec[2]) {
                    Ok(v) => { std::ptr::write(dst, v); return 0; }
                    Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); return GEN_ERROR; }
                }
            }
        }
        if let Some(format_name) = name.strip_prefix('@') {
            // Format: @base64, @uri, etc.
            if args_vec.is_empty() {
                set_jit_error("format: no input".to_string());
                std::ptr::write(dst, Value::Null);
                return GEN_ERROR;
            }
            match crate::eval::eval_format(format_name, &args_vec[0]) {
                Ok(s) => { std::ptr::write(dst, Value::from_str(&s)); return 0; }
                Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); return GEN_ERROR; }
            }
        }

        // __assign__:path_idx:value_idx — runtime path assignment
        if let Some(rest) = name.strip_prefix("__assign__:") {
            let parts: Vec<&str> = rest.splitn(2, ':').collect();
            if parts.len() == 2 {
                let path_idx: usize = parts[0].parse().unwrap_or(0);
                let value_idx: usize = parts[1].parse().unwrap_or(0);
                let (path_expr, value_expr) = {
                    let ops = &*JIT_CLOSURE_OPS.0.get();
                    (ops.get(path_idx).cloned(), ops.get(value_idx).cloned())
                };
                if let (Some(path_expr), Some(value_expr)) = (path_expr, value_expr) {
                    let input = if !args_vec.is_empty() { args_vec[0].clone() } else { Value::Null };
                    let env = Rc::new(RefCell::new(crate::eval::Env::new(vec![])));
                    match crate::eval::eval_assign_standalone(&path_expr, &value_expr, input, &env) {
                        Ok(v) => { std::ptr::write(dst, v); return 0; }
                        Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); return GEN_ERROR; }
                    }
                }
            }
        }

        // __update__:path_idx:update_idx — runtime path update
        if let Some(rest) = name.strip_prefix("__update__:") {
            let parts: Vec<&str> = rest.splitn(2, ':').collect();
            if parts.len() == 2 {
                let path_idx: usize = parts[0].parse().unwrap_or(0);
                let update_idx: usize = parts[1].parse().unwrap_or(0);
                let (path_expr, update_expr) = {
                    let ops = &*JIT_CLOSURE_OPS.0.get();
                    (ops.get(path_idx).cloned(), ops.get(update_idx).cloned())
                };
                if let (Some(path_expr), Some(update_expr)) = (path_expr, update_expr) {
                    let input = if !args_vec.is_empty() { args_vec[0].clone() } else { Value::Null };
                    let env = Rc::new(RefCell::new(crate::eval::Env::new(vec![])));
                    match crate::eval::eval_update_standalone(&path_expr, &update_expr, input, &env) {
                        Ok(v) => { std::ptr::write(dst, v); return 0; }
                        Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); return GEN_ERROR; }
                    }
                }
            }
        }

        // __path__:idx — runtime path expression evaluation
        if let Some(rest) = name.strip_prefix("__path__:") {
            let idx: usize = rest.parse().unwrap_or(0);
            let path_expr = (&*JIT_CLOSURE_OPS.0.get()).get(idx).cloned();
            if let Some(path_expr) = path_expr {
                let input = if !args_vec.is_empty() { args_vec[0].clone() } else { Value::Null };
                let env = Rc::new(RefCell::new(crate::eval::Env::new(vec![])));
                match crate::eval::eval_path_standalone(&path_expr, input, &env) {
                    Ok(v) => { std::ptr::write(dst, v); return 0; }
                    Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); return GEN_ERROR; }
                }
            }
        }

        // __closure_op__:op_name:idx — delegate to eval-based closure operation
        if let Some(rest) = name.strip_prefix("__closure_op__:") {
            let parts: Vec<&str> = rest.splitn(2, ':').collect();
            if parts.len() == 2 {
                let op_name = parts[0];
                let idx: usize = parts[1].parse().unwrap_or(0);
                if !args_vec.is_empty() {
                    let container = &args_vec[0];
                    let key_expr = (&*JIT_CLOSURE_OPS.0.get()).get(idx).cloned();
                    if let Some(key_expr) = key_expr {
                        let op_kind = match op_name {
                            "sort_by" => Some(ClosureOpKind::SortBy),
                            "group_by" => Some(ClosureOpKind::GroupBy),
                            "unique_by" => Some(ClosureOpKind::UniqueBy),
                            "min_by" => Some(ClosureOpKind::MinBy),
                            "max_by" => Some(ClosureOpKind::MaxBy),
                            _ => None,
                        };
                        if let Some(op_kind) = op_kind {
                            // Use eval infrastructure to perform the closure op
                            let env = Rc::new(RefCell::new(crate::eval::Env::new(vec![])));
                            let eval_result = crate::eval::eval_closure_op_standalone(
                                op_kind, container, &key_expr, &env,
                            );
                            match eval_result {
                                Ok(v) => { std::ptr::write(dst, v); return 0; }
                                Err(e) => {
                                    set_jit_error(format!("{}", e));
                                    std::ptr::write(dst, Value::Null);
                                    return GEN_ERROR;
                                }
                            }
                        }
                    }
                }
                set_jit_error(format!("closure_op error: {}", rest));
                std::ptr::write(dst, Value::Null);
                return GEN_ERROR;
            }
        }

        match crate::runtime::call_builtin(name, &args_vec) {
            Ok(v) => { std::ptr::write(dst, v); 0 }
            Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); GEN_ERROR }
        }
    }
}

// TryCatch helpers
extern "C" fn jit_rt_try_begin(env: *mut JitEnv) {
    unsafe { (*env).try_depth += 1; }
    clear_jit_error();
}
extern "C" fn jit_rt_try_end(env: *mut JitEnv) {
    unsafe { (*env).try_depth -= 1; }
}

/// Transfer error from JIT_LAST_ERROR to env.error_msg for propagation.
extern "C" fn jit_rt_propagate_error(env: *mut JitEnv) {
    // Check for direct Value from `error` builtin (deferred serialization)
    if let Some(val) = take_jit_error_value() {
        let _ = take_jit_error(); // clear the marker
        let msg_json = crate::value::value_to_json(&val);
        unsafe { (*env).error_msg = Some(format!("__jqerror__:{}", msg_json)); }
        return;
    }
    if let Some(msg) = take_jit_error() {
        unsafe { (*env).error_msg = Some(msg); }
    }
}

/// Check if the last operation produced an error.
/// Returns 1 if error, 0 if ok.
extern "C" fn jit_rt_has_error() -> i64 {
    unsafe {
        if (*JIT_ERROR_VALUE.0.get()).is_some() || (*JIT_LAST_ERROR.0.get()).is_some() { 1 } else { 0 }
    }
}

/// Get the last error as a Value and write it to dst. Clears the error.
extern "C" fn jit_rt_get_error(dst: *mut Value) {
    // Fast path: use directly stored Value from `error` builtin
    if let Some(v) = take_jit_error_value() {
        let _ = take_jit_error(); // clear the string too
        unsafe { std::ptr::write(dst, v); }
        return;
    }
    let err = take_jit_error();
    unsafe {
        if let Some(msg) = err {
            // Parse error: if it starts with __jqerror__, extract the JSON value
            if let Some(json) = msg.strip_prefix("__jqerror__:") {
                if let Ok(v) = crate::value::json_to_value(json) {
                    std::ptr::write(dst, v);
                } else {
                    std::ptr::write(dst, Value::from_str(&msg));
                }
            } else {
                std::ptr::write(dst, Value::from_str(&msg));
            }
        } else {
            std::ptr::write(dst, Value::Null);
        }
    }
}

fn binop_to_i32(op: BinOp) -> i32 {
    match op {
        BinOp::Add => 0, BinOp::Sub => 1, BinOp::Mul => 2, BinOp::Div => 3,
        BinOp::Mod => 4, BinOp::Eq => 5, BinOp::Ne => 6, BinOp::Lt => 7,
        BinOp::Gt => 8, BinOp::Le => 9, BinOp::Ge => 10, BinOp::And => 11, BinOp::Or => 12,
    }
}

fn binop_from_i32(op: i32) -> BinOp {
    match op {
        0 => BinOp::Add, 1 => BinOp::Sub, 2 => BinOp::Mul, 3 => BinOp::Div,
        4 => BinOp::Mod, 5 => BinOp::Eq, 6 => BinOp::Ne, 7 => BinOp::Lt,
        8 => BinOp::Gt, 9 => BinOp::Le, 10 => BinOp::Ge, 11 => BinOp::And, 12 => BinOp::Or,
        _ => BinOp::Add,
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
        40 => UnaryOp::Trim, 41 => UnaryOp::Ltrim, 42 => UnaryOp::Rtrim,
        43 => UnaryOp::Utf8ByteLength,
        44 => UnaryOp::Sin, 45 => UnaryOp::Cos, 46 => UnaryOp::Tan,
        47 => UnaryOp::Asin, 48 => UnaryOp::Acos, 49 => UnaryOp::Atan,
        50 => UnaryOp::Exp, 51 => UnaryOp::Exp2, 52 => UnaryOp::Exp10,
        53 => UnaryOp::Log, 54 => UnaryOp::Log2, 55 => UnaryOp::Log10,
        56 => UnaryOp::Cbrt, 57 => UnaryOp::Significand, 58 => UnaryOp::Exponent,
        59 => UnaryOp::Logb, 60 => UnaryOp::NearbyInt, 61 => UnaryOp::Trunc,
        62 => UnaryOp::Rint, 63 => UnaryOp::J0, 64 => UnaryOp::J1,
        65 => UnaryOp::Gmtime, 66 => UnaryOp::Mktime, 67 => UnaryOp::Now,
        68 => UnaryOp::GetModuleMeta,
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
        UnaryOp::Trim => 40, UnaryOp::Ltrim => 41, UnaryOp::Rtrim => 42,
        UnaryOp::Utf8ByteLength => 43,
        UnaryOp::Sin => 44, UnaryOp::Cos => 45, UnaryOp::Tan => 46,
        UnaryOp::Asin => 47, UnaryOp::Acos => 48, UnaryOp::Atan => 49,
        UnaryOp::Exp => 50, UnaryOp::Exp2 => 51, UnaryOp::Exp10 => 52,
        UnaryOp::Log => 53, UnaryOp::Log2 => 54, UnaryOp::Log10 => 55,
        UnaryOp::Cbrt => 56, UnaryOp::Significand => 57, UnaryOp::Exponent => 58,
        UnaryOp::Logb => 59, UnaryOp::NearbyInt => 60, UnaryOp::Trunc => 61,
        UnaryOp::Rint => 62, UnaryOp::J0 => 63, UnaryOp::J1 => 64,
        UnaryOp::Gmtime => 65, UnaryOp::Mktime => 66, UnaryOp::Now => 67,
        UnaryOp::GetModuleMeta => 68,
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
    pub error_msg: Option<String>,
}

impl Default for JitEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl JitEnv {
    pub fn new() -> Self {
        JitEnv {
            vars: vec![Value::Null; 4096],
            collect_stacks: Vec::new(),
            str_bufs: Vec::new(),
            try_depth: 0,
            error_msg: None,
        }
    }
}

pub type JitFilterFn = unsafe extern "C" fn(*const Value, *mut JitEnv,
    unsafe extern "C" fn(*const Value, *mut u8) -> i64, *mut u8) -> i64;

/// Peephole: eliminate redundant Clone+Yield+Drop sequences.
/// Pattern: Clone{dst=A, src=B}, Yield{A}, Drop{A} where A is used nowhere else.
/// Replaced with: Yield{B} (skipping clone and drop entirely).
fn optimize_clone_yield(mut ops: Vec<JitOp>) -> Vec<JitOp> {
    // Build use counts for each slot (how many times each slot appears as a source/operand)
    let mut use_count: HashMap<SlotId, u32> = HashMap::new();
    for op in &ops {
        match op {
            JitOp::Clone { src, .. } => { *use_count.entry(*src).or_insert(0) += 1; }
            JitOp::Yield { output } => { *use_count.entry(*output).or_insert(0) += 1; }
            JitOp::Drop { slot } => { *use_count.entry(*slot).or_insert(0) += 1; }
            JitOp::Index { base, key, .. } => {
                *use_count.entry(*base).or_insert(0) += 1;
                *use_count.entry(*key).or_insert(0) += 1;
            }
            JitOp::IndexField { base, .. } | JitOp::FieldBinopField { base, .. } | JitOp::YieldFieldRef { base, .. } => { *use_count.entry(*base).or_insert(0) += 1; }
            JitOp::FieldBinopConst { base, .. } => {
                *use_count.entry(*base).or_insert(0) += 1;
            }
            JitOp::BinOp { lhs, rhs, .. } | JitOp::AddMove { lhs, rhs, .. } => {
                *use_count.entry(*lhs).or_insert(0) += 1;
                *use_count.entry(*rhs).or_insert(0) += 1;
            }
            JitOp::UnaryOp { src, .. } | JitOp::Negate { src, .. } | JitOp::Not { src, .. } => {
                *use_count.entry(*src).or_insert(0) += 1;
            }
            JitOp::CollectPush { src } => { *use_count.entry(*src).or_insert(0) += 1; }
            JitOp::ObjInsert { obj, key, val } => {
                *use_count.entry(*obj).or_insert(0) += 1;
                *use_count.entry(*key).or_insert(0) += 1;
                *use_count.entry(*val).or_insert(0) += 1;
            }
            JitOp::ObjInsertStrKey { obj, val, .. } | JitOp::ObjPushStrKey { obj, val, .. } => {
                *use_count.entry(*obj).or_insert(0) += 1;
                *use_count.entry(*val).or_insert(0) += 1;
            }
            JitOp::ObjCopyField { obj, src, .. } => {
                *use_count.entry(*obj).or_insert(0) += 1;
                *use_count.entry(*src).or_insert(0) += 1;
            }
            JitOp::ObjFromFields { src, .. } => {
                *use_count.entry(*src).or_insert(0) += 1;
            }
            JitOp::IfTruthy { src, .. } => { *use_count.entry(*src).or_insert(0) += 1; }
            JitOp::FieldIsTruthy { base, .. } | JitOp::FieldCmpNum { base, .. } => { *use_count.entry(*base).or_insert(0) += 1; }
            JitOp::IsNullOrFalse { src, .. } => { *use_count.entry(*src).or_insert(0) += 1; }
            JitOp::Alternative { primary, .. } => { *use_count.entry(*primary).or_insert(0) += 1; }
            JitOp::SetVar { src, .. } => { *use_count.entry(*src).or_insert(0) += 1; }
            JitOp::PathExtract { container, key, .. } => {
                *use_count.entry(*container).or_insert(0) += 1;
                *use_count.entry(*key).or_insert(0) += 1;
            }
            JitOp::PathInsert { container, key, val } => {
                *use_count.entry(*container).or_insert(0) += 1;
                *use_count.entry(*key).or_insert(0) += 1;
                *use_count.entry(*val).or_insert(0) += 1;
            }
            JitOp::StrBufAppendVal { src } => { *use_count.entry(*src).or_insert(0) += 1; }
            JitOp::ThrowError { msg } => { *use_count.entry(*msg).or_insert(0) += 1; }
            JitOp::MutateInplace { slot, .. } => { *use_count.entry(*slot).or_insert(0) += 1; }
            JitOp::CallBuiltin { args, .. } => {
                for a in args { *use_count.entry(*a).or_insert(0) += 1; }
            }
            JitOp::ArrayGet { arr, .. } => { *use_count.entry(*arr).or_insert(0) += 1; }
            JitOp::ObjGetByIdx { obj, .. } => { *use_count.entry(*obj).or_insert(0) += 1; }
            JitOp::GetKind { src, .. } | JitOp::GetLen { src, .. } => {
                *use_count.entry(*src).or_insert(0) += 1;
            }
            JitOp::ToF64Var { src, .. } => { *use_count.entry(*src).or_insert(0) += 1; }
            _ => {}
        }
    }

    // Iteratively find Clone{A,B} + Yield{A} + Drop{A} where A is used exactly twice
    loop {
        let mut changed = false;
        // Rebuild use counts after each pass
        use_count.clear();
        for op in &ops {
            match op {
                JitOp::Clone { src, .. } => { *use_count.entry(*src).or_insert(0) += 1; }
                JitOp::Yield { output } => { *use_count.entry(*output).or_insert(0) += 1; }
                JitOp::Drop { slot } => { *use_count.entry(*slot).or_insert(0) += 1; }
                JitOp::Index { base, key, .. } => {
                    *use_count.entry(*base).or_insert(0) += 1;
                    *use_count.entry(*key).or_insert(0) += 1;
                }
                JitOp::IndexField { base, .. } | JitOp::FieldBinopField { base, .. } | JitOp::YieldFieldRef { base, .. } => { *use_count.entry(*base).or_insert(0) += 1; }
            JitOp::FieldBinopConst { base, .. } => {
                *use_count.entry(*base).or_insert(0) += 1;
            }
                JitOp::BinOp { lhs, rhs, .. } | JitOp::AddMove { lhs, rhs, .. } => {
                    *use_count.entry(*lhs).or_insert(0) += 1;
                    *use_count.entry(*rhs).or_insert(0) += 1;
                }
                JitOp::UnaryOp { src, .. } | JitOp::Negate { src, .. } | JitOp::Not { src, .. } => {
                    *use_count.entry(*src).or_insert(0) += 1;
                }
                JitOp::CollectPush { src } => { *use_count.entry(*src).or_insert(0) += 1; }
                JitOp::ObjInsert { obj, key, val } => {
                    *use_count.entry(*obj).or_insert(0) += 1;
                    *use_count.entry(*key).or_insert(0) += 1;
                    *use_count.entry(*val).or_insert(0) += 1;
                }
                JitOp::ObjInsertStrKey { obj, val, .. } | JitOp::ObjPushStrKey { obj, val, .. } => {
                    *use_count.entry(*obj).or_insert(0) += 1;
                    *use_count.entry(*val).or_insert(0) += 1;
                }
                JitOp::ObjCopyField { obj, src, .. } => {
                    *use_count.entry(*obj).or_insert(0) += 1;
                    *use_count.entry(*src).or_insert(0) += 1;
                }
                JitOp::ObjFromFields { src, .. } => {
                    *use_count.entry(*src).or_insert(0) += 1;
                }
                JitOp::IfTruthy { src, .. } => { *use_count.entry(*src).or_insert(0) += 1; }
                JitOp::FieldIsTruthy { base, .. } | JitOp::FieldCmpNum { base, .. } => { *use_count.entry(*base).or_insert(0) += 1; }
                JitOp::IsNullOrFalse { src, .. } => { *use_count.entry(*src).or_insert(0) += 1; }
                JitOp::Alternative { primary, .. } => { *use_count.entry(*primary).or_insert(0) += 1; }
                JitOp::SetVar { src, .. } => { *use_count.entry(*src).or_insert(0) += 1; }
                JitOp::PathExtract { container, key, .. } => {
                    *use_count.entry(*container).or_insert(0) += 1;
                    *use_count.entry(*key).or_insert(0) += 1;
                }
                JitOp::PathInsert { container, key, val } => {
                    *use_count.entry(*container).or_insert(0) += 1;
                    *use_count.entry(*key).or_insert(0) += 1;
                    *use_count.entry(*val).or_insert(0) += 1;
                }
                JitOp::StrBufAppendVal { src } => { *use_count.entry(*src).or_insert(0) += 1; }
                JitOp::ThrowError { msg } => { *use_count.entry(*msg).or_insert(0) += 1; }
                JitOp::MutateInplace { slot, .. } => { *use_count.entry(*slot).or_insert(0) += 1; }
                JitOp::CallBuiltin { args, .. } => {
                    for a in args { *use_count.entry(*a).or_insert(0) += 1; }
                }
                JitOp::ArrayGet { arr, .. } => { *use_count.entry(*arr).or_insert(0) += 1; }
                JitOp::ObjGetByIdx { obj, .. } => { *use_count.entry(*obj).or_insert(0) += 1; }
                JitOp::GetKind { src, .. } | JitOp::GetLen { src, .. } => {
                    *use_count.entry(*src).or_insert(0) += 1;
                }
                JitOp::ToF64Var { src, .. } => { *use_count.entry(*src).or_insert(0) += 1; }
                _ => {}
            }
        }
        let mut i = 0;
        while i + 2 < ops.len() {
            if let JitOp::Clone { dst, src } = ops[i] {
                if let JitOp::Yield { output } = ops[i + 1] {
                    if let JitOp::Drop { slot } = ops[i + 2] {
                        if output == dst && slot == dst && use_count.get(&dst) == Some(&2) {
                            ops[i] = JitOp::Yield { output: src };
                            ops.remove(i + 2);
                            ops.remove(i + 1);
                            changed = true;
                            continue;
                        }
                    }
                }
            }
            i += 1;
        }
        if !changed { break; }
    }

    // Fuse IndexField(dst) + Yield(dst) + Drop(dst) → YieldFieldRef(base, field)
    // Avoids cloning the field value — yields a borrowed reference directly.
    let mut i = 0;
    while i + 2 < ops.len() {
        if let JitOp::IndexField { dst, base, ref field } = ops[i] {
            if let JitOp::Yield { output } = ops[i + 1] {
                if let JitOp::Drop { slot } = ops[i + 2] {
                    if output == dst && slot == dst {
                        let field = field.clone();
                        ops[i] = JitOp::YieldFieldRef { base, field };
                        ops.remove(i + 2);
                        ops.remove(i + 1);
                        continue;
                    }
                }
            }
        }
        i += 1;
    }

    // Fuse ObjNew + consecutive ObjCopyField (same obj & src) → ObjFromFields
    let mut i = 0;
    while i < ops.len() {
        if let JitOp::ObjNew { dst: obj_slot, cap } = ops[i] {
            let n = cap as usize;
            if n >= 2 && i + n < ops.len() {
                // Check that the next N ops are all ObjCopyField with same obj/src
                let mut all_match = true;
                let mut src_slot: SlotId = 0;
                let mut pairs = Vec::with_capacity(n);
                for j in 0..n {
                    if let JitOp::ObjCopyField { obj, src, ref obj_key, ref src_field } = ops[i + 1 + j] {
                        if obj != obj_slot { all_match = false; break; }
                        if j == 0 { src_slot = src; } else if src != src_slot { all_match = false; break; }
                        pairs.push((obj_key.clone(), src_field.clone()));
                    } else {
                        all_match = false;
                        break;
                    }
                }
                if all_match {
                    // Replace ObjNew + N×ObjCopyField with single ObjFromFields
                    ops[i] = JitOp::ObjFromFields { dst: obj_slot, src: src_slot, pairs };
                    for _ in 0..n {
                        ops.remove(i + 1);
                    }
                }
            }
        }
        i += 1;
    }

    ops
}

pub struct JitCompiler {
    module: JITModule,
    ctx: cranelift_codegen::Context,
    func_ctx: FunctionBuilderContext,
    rt_funcs: HashMap<&'static str, FuncId>,
    _string_constants: Vec<&'static str>,
    #[allow(clippy::vec_box)]
    _repr_constants: Vec<Box<Rc<str>>>,
    /// Pre-allocated CompactString constants for string literals in JIT code.
    #[allow(clippy::vec_box)]
    _rc_str_constants: Vec<Box<crate::value::KeyStr>>,
    /// Pre-allocated Value constants for fused ops (hoisted out of per-call code).
    #[allow(clippy::vec_box)]
    _value_constants: Vec<Box<Value>>,
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
            ("jit_rt_str_rc", jit_rt_str_rc as *const u8),
            ("jit_rt_is_truthy", jit_rt_is_truthy as *const u8),
            ("jit_rt_field_is_truthy", jit_rt_field_is_truthy as *const u8),
            ("jit_rt_field_cmp_num", jit_rt_field_cmp_num as *const u8),
            ("jit_rt_field_binop_field", jit_rt_field_binop_field as *const u8),
            ("jit_rt_field_binop_const", jit_rt_field_binop_const as *const u8),
            ("jit_rt_index", jit_rt_index as *const u8),
            ("jit_rt_index_field", jit_rt_index_field as *const u8),
            ("jit_rt_yield_field_ref", jit_rt_yield_field_ref as *const u8),
            ("jit_rt_binop", jit_rt_binop as *const u8),
            ("jit_rt_add_move", jit_rt_add_move as *const u8),
            ("jit_rt_unaryop", jit_rt_unaryop as *const u8),
            ("jit_rt_negate", jit_rt_negate as *const u8),
            ("jit_rt_not", jit_rt_not as *const u8),
            ("jit_rt_kind", jit_rt_kind as *const u8),
            ("jit_rt_len", jit_rt_len as *const u8),
            ("jit_rt_array_get", jit_rt_array_get as *const u8),
            ("jit_rt_obj_get_idx", jit_rt_obj_get_idx as *const u8),
            ("jit_rt_get_var", jit_rt_get_var as *const u8),
            ("jit_rt_set_var", jit_rt_set_var as *const u8),
            ("jit_rt_take_var", jit_rt_take_var as *const u8),
            ("jit_rt_path_extract", jit_rt_path_extract as *const u8),
            ("jit_rt_path_insert", jit_rt_path_insert as *const u8),
            ("jit_rt_collect_begin", jit_rt_collect_begin as *const u8),
            ("jit_rt_collect_push", jit_rt_collect_push as *const u8),
            ("jit_rt_collect_finish", jit_rt_collect_finish as *const u8),
            ("jit_rt_obj_new", jit_rt_obj_new as *const u8),
            ("jit_rt_obj_insert", jit_rt_obj_insert as *const u8),
            ("jit_rt_obj_insert_str_key", jit_rt_obj_insert_str_key as *const u8),
            ("jit_rt_obj_push_str_key", jit_rt_obj_push_str_key as *const u8),
            ("jit_rt_obj_copy_field", jit_rt_obj_copy_field as *const u8),
            ("jit_rt_obj_from_fields", jit_rt_obj_from_fields as *const u8),
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
            ("jit_rt_propagate_error", jit_rt_propagate_error as *const u8),
            ("jit_rt_has_error", jit_rt_has_error as *const u8),
            ("jit_rt_get_error", jit_rt_get_error as *const u8),
            ("jit_rt_throw_error", jit_rt_throw_error as *const u8),
            ("jit_rt_call_builtin", jit_rt_call_builtin as *const u8),
            ("jit_rt_reverse_inplace", jit_rt_reverse_inplace as *const u8),
            ("jit_rt_sort_inplace", jit_rt_sort_inplace as *const u8),
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
            _rc_str_constants: Vec::new(),
            _value_constants: Vec::new(),
        })
    }

    pub fn compile(&mut self, expr: &Expr) -> Result<JitFilterFn> {
        self.compile_with_funcs(expr, &[])
    }

    pub fn compile_with_funcs(&mut self, expr: &Expr, funcs: &[CompiledFunc]) -> Result<JitFilterFn> {
        // Phase 1: Flatten
        let mut fl = Flattener::new();
        fl.funcs = funcs.to_vec();
        // Pre-inline function calls so is_scalar sees no FuncCall nodes,
        // enabling JIT for expressions that use user-defined functions.
        // Skip if no functions defined (avoids unnecessary tree clone for simple filters).
        let inlined;
        let compile_expr = if !funcs.is_empty() {
            inlined = fl.inline_func_calls(expr);
            &inlined
        } else {
            expr
        };
        let input_slot = fl.alloc_slot(); // slot 0 = input ptr (read-only, owned by caller)
        // Use input_slot directly — flatten_gen only reads it, never writes/drops it.
        // The codegen handles Drop{slot:0} as a no-op.
        if !fl.flatten_gen(compile_expr, input_slot) {
            bail!("Expression not JIT-compilable");
        }
        fl.emit(JitOp::ReturnContinue);

        // Store closure ops for runtime access
        if !fl.closure_ops.is_empty() {
            unsafe { *JIT_CLOSURE_OPS.0.get() = fl.closure_ops.clone(); }
        }

        let ops = optimize_clone_yield(fl.ops);
        let num_slots = fl.next_slot;
        let num_vars = fl.next_var;
        let num_labels = fl.next_label;
        // Transfer hoisted value constants to compiler lifetime
        self._value_constants.extend(fl.value_constants);

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
                            let boxed = Box::new(r.clone());
                            let rp = b.ins().iconst(ptr_ty, &*boxed as *const Rc<str> as i64);
                            self._repr_constants.push(boxed);
                            b.ins().call(rt["num_repr"], &[a, n, rp]);
                        } else {
                            b.ins().call(rt["num"], &[a, n]);
                        }
                    }
                    JitOp::Str { dst, val } => {
                        let a = slot_addr(&mut b, *dst);
                        // Pre-allocate CompactString constant — copy 24 bytes at runtime
                        let cs = crate::value::KeyStr::from(val.as_str());
                        let boxed = Box::new(cs);
                        let cs_ptr = &*boxed as *const crate::value::KeyStr;
                        self._rc_str_constants.push(boxed);
                        let rp = b.ins().iconst(ptr_ty, cs_ptr as i64);
                        b.ins().call(rt["str_rc"], &[a, rp]);
                    }
                    JitOp::Index { dst, base, key } => {
                        let d = slot_addr(&mut b, *dst);
                        let ba = slot_addr(&mut b, *base);
                        let ka = slot_addr(&mut b, *key);
                        b.ins().call(rt["index"], &[d, ba, ka]);
                    }
                    JitOp::IndexField { dst, base, field } => {
                        let d = slot_addr(&mut b, *dst);
                        let ba = slot_addr(&mut b, *base);
                        let leaked = Box::leak(field.clone().into_boxed_str());
                        self._string_constants.push(leaked);
                        let fp = b.ins().iconst(ptr_ty, leaked.as_ptr() as i64);
                        let fl = b.ins().iconst(ptr_ty, leaked.len() as i64);
                        b.ins().call(rt["index_field"], &[d, ba, fp, fl]);
                    }
                    JitOp::BinOp { dst, op, lhs, rhs } => {
                        let d = slot_addr(&mut b, *dst);
                        let l = slot_addr(&mut b, *lhs);
                        let r = slot_addr(&mut b, *rhs);
                        let o = b.ins().iconst(types::I32, binop_to_i32(*op) as i64);
                        b.ins().call(rt["binop"], &[d, o, l, r]);
                    }
                    JitOp::AddMove { dst, lhs, rhs } => {
                        let d = slot_addr(&mut b, *dst);
                        let l = slot_addr(&mut b, *lhs);
                        let r = slot_addr(&mut b, *rhs);
                        b.ins().call(rt["add_move"], &[d, l, r]);
                    }
                    JitOp::FieldBinopField { dst, base, field_a, field_b, op } => {
                        let d = slot_addr(&mut b, *dst);
                        let ba = slot_addr(&mut b, *base);
                        let la = Box::leak(field_a.clone().into_boxed_str());
                        self._string_constants.push(la);
                        let lb = Box::leak(field_b.clone().into_boxed_str());
                        self._string_constants.push(lb);
                        let kap = b.ins().iconst(ptr_ty, la.as_ptr() as i64);
                        let kal = b.ins().iconst(ptr_ty, la.len() as i64);
                        let kbp = b.ins().iconst(ptr_ty, lb.as_ptr() as i64);
                        let kbl = b.ins().iconst(ptr_ty, lb.len() as i64);
                        let op_i32 = b.ins().iconst(types::I32, *op as i64);
                        b.ins().call(rt["field_binop_field"], &[d, ba, kap, kal, kbp, kbl, op_i32]);
                    }
                    JitOp::FieldBinopConst { dst, base, field, const_ptr, op, field_is_lhs } => {
                        let d = slot_addr(&mut b, *dst);
                        let ba = slot_addr(&mut b, *base);
                        let leaked = Box::leak(field.clone().into_boxed_str());
                        self._string_constants.push(leaked);
                        let kp = b.ins().iconst(ptr_ty, leaked.as_ptr() as i64);
                        let kl = b.ins().iconst(ptr_ty, leaked.len() as i64);
                        let cv = b.ins().iconst(ptr_ty, *const_ptr as i64);
                        let op_i32 = b.ins().iconst(types::I32, *op as i64);
                        let lhs_flag = b.ins().iconst(types::I32, if *field_is_lhs { 1 } else { 0 });
                        b.ins().call(rt["field_binop_const"], &[d, ba, kp, kl, cv, op_i32, lhs_flag]);
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
                    JitOp::YieldFieldRef { base, field } => {
                        let ba = slot_addr(&mut b, *base);
                        let leaked = Box::leak(field.clone().into_boxed_str());
                        self._string_constants.push(leaked);
                        let kp = b.ins().iconst(ptr_ty, leaked.as_ptr() as i64);
                        let kl = b.ins().iconst(ptr_ty, leaked.len() as i64);
                        let call = b.ins().call(rt["yield_field_ref"], &[ba, kp, kl, cb_fn, cb_ctx]);
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
                    JitOp::FieldIsTruthy { base, field, then_label, else_label } => {
                        let ba = slot_addr(&mut b, *base);
                        let leaked = Box::leak(field.clone().into_boxed_str());
                        self._string_constants.push(leaked);
                        let kp = b.ins().iconst(ptr_ty, leaked.as_ptr() as i64);
                        let kl = b.ins().iconst(ptr_ty, leaked.len() as i64);
                        let call = b.ins().call(rt["field_is_truthy"], &[ba, kp, kl]);
                        let truthy = b.inst_results(call)[0];
                        let zero = b.ins().iconst(ptr_ty, 0);
                        let is_t = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, truthy, zero);
                        b.ins().brif(is_t, label_blocks[*then_label as usize], &[], label_blocks[*else_label as usize], &[]);
                        terminated = true;
                    }
                    JitOp::FieldCmpNum { base, field, value, op, then_label, else_label } => {
                        let ba = slot_addr(&mut b, *base);
                        let leaked = Box::leak(field.clone().into_boxed_str());
                        self._string_constants.push(leaked);
                        let kp = b.ins().iconst(ptr_ty, leaked.as_ptr() as i64);
                        let kl = b.ins().iconst(ptr_ty, leaked.len() as i64);
                        let rhs_f64 = b.ins().f64const(*value);
                        let op_i32 = b.ins().iconst(types::I32, *op as i64);
                        let call = b.ins().call(rt["field_cmp_num"], &[ba, kp, kl, rhs_f64, op_i32]);
                        let result = b.inst_results(call)[0];
                        let zero = b.ins().iconst(ptr_ty, 0);
                        let is_t = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, result, zero);
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
                    JitOp::InitVar { var } => {
                        let zero = b.ins().iconst(ptr_ty, 0);
                        b.def_var(vars[*var as usize], zero);
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
                    JitOp::TakeVar { dst, var_index } => {
                        let d = slot_addr(&mut b, *dst);
                        let vi = b.ins().iconst(types::I32, *var_index as i64);
                        b.ins().call(rt["take_var"], &[d, env_ptr, vi]);
                    }
                    JitOp::PathExtract { element, container, key } => {
                        let e = slot_addr(&mut b, *element);
                        let c = slot_addr(&mut b, *container);
                        let k = slot_addr(&mut b, *key);
                        b.ins().call(rt["path_extract"], &[e, c, k]);
                    }
                    JitOp::PathInsert { container, key, val } => {
                        let c = slot_addr(&mut b, *container);
                        let k = slot_addr(&mut b, *key);
                        let v = slot_addr(&mut b, *val);
                        b.ins().call(rt["path_insert"], &[c, k, v]);
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
                    JitOp::ObjNew { dst, cap } => {
                        let d = slot_addr(&mut b, *dst);
                        let c = b.ins().iconst(ptr_ty, *cap as i64);
                        b.ins().call(rt["obj_new"], &[d, c]);
                    }
                    JitOp::ObjInsert { obj, key, val } => {
                        let o = slot_addr(&mut b, *obj);
                        let k = slot_addr(&mut b, *key);
                        let v = slot_addr(&mut b, *val);
                        b.ins().call(rt["obj_insert"], &[o, k, v]);
                    }
                    JitOp::ObjInsertStrKey { obj, key, val } => {
                        let o = slot_addr(&mut b, *obj);
                        let leaked = Box::leak(key.clone().into_boxed_str());
                        self._string_constants.push(leaked);
                        let kp = b.ins().iconst(ptr_ty, leaked.as_ptr() as i64);
                        let kl = b.ins().iconst(ptr_ty, leaked.len() as i64);
                        let v = slot_addr(&mut b, *val);
                        b.ins().call(rt["obj_insert_str_key"], &[o, kp, kl, v]);
                    }
                    JitOp::ObjPushStrKey { obj, key, val } => {
                        let o = slot_addr(&mut b, *obj);
                        let leaked = Box::leak(key.clone().into_boxed_str());
                        self._string_constants.push(leaked);
                        let kp = b.ins().iconst(ptr_ty, leaked.as_ptr() as i64);
                        let kl = b.ins().iconst(ptr_ty, leaked.len() as i64);
                        let v = slot_addr(&mut b, *val);
                        b.ins().call(rt["obj_push_str_key"], &[o, kp, kl, v]);
                    }
                    JitOp::ObjCopyField { obj, src, obj_key, src_field } => {
                        let o = slot_addr(&mut b, *obj);
                        let s = slot_addr(&mut b, *src);
                        let ok_leaked = Box::leak(obj_key.clone().into_boxed_str());
                        self._string_constants.push(ok_leaked);
                        let ok_p = b.ins().iconst(ptr_ty, ok_leaked.as_ptr() as i64);
                        let ok_l = b.ins().iconst(ptr_ty, ok_leaked.len() as i64);
                        let sf_leaked = Box::leak(src_field.clone().into_boxed_str());
                        self._string_constants.push(sf_leaked);
                        let sf_p = b.ins().iconst(ptr_ty, sf_leaked.as_ptr() as i64);
                        let sf_l = b.ins().iconst(ptr_ty, sf_leaked.len() as i64);
                        b.ins().call(rt["obj_copy_field"], &[o, s, ok_p, ok_l, sf_p, sf_l]);
                    }
                    JitOp::ObjFromFields { dst, src, pairs } => {
                        let d = slot_addr(&mut b, *dst);
                        let s = slot_addr(&mut b, *src);
                        // Build pair table: (out_key_ptr, out_key_len, src_field_ptr, src_field_len)
                        let table: Vec<(*const u8, usize, *const u8, usize)> = pairs.iter().map(|(ok, sf)| {
                            let ok_leaked = Box::leak(ok.clone().into_boxed_str());
                            self._string_constants.push(ok_leaked);
                            let sf_leaked = Box::leak(sf.clone().into_boxed_str());
                            self._string_constants.push(sf_leaked);
                            (ok_leaked.as_ptr(), ok_leaked.len(), sf_leaked.as_ptr(), sf_leaked.len())
                        }).collect();
                        let table_leaked = Box::leak(table.into_boxed_slice());
                        let table_ptr = table_leaked.as_ptr();
                        let tp = b.ins().iconst(ptr_ty, table_ptr as i64);
                        let cnt = b.ins().iconst(ptr_ty, pairs.len() as i64);
                        b.ins().call(rt["obj_from_fields"], &[d, s, tp, cnt]);
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
                    JitOp::F64Const { dst_var, val } => {
                        let bits = val.to_bits() as i64;
                        let c = b.ins().iconst(ptr_ty, bits);
                        b.def_var(vars[*dst_var as usize], c);
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
                    JitOp::CheckError { error_dst, catch_label } => {
                        // Check if the last operation produced an error
                        let call = b.ins().call(rt["has_error"], &[]);
                        let has_err = b.inst_results(call)[0];
                        let zero = b.ins().iconst(ptr_ty, 0);
                        let is_err = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, has_err, zero);
                        let err_blk = b.create_block();
                        let ok_blk = b.create_block();
                        b.ins().brif(is_err, err_blk, &[], ok_blk, &[]);
                        // Error path: get the error value and jump to catch
                        b.switch_to_block(err_blk);
                        b.seal_block(err_blk);
                        let d = slot_addr(&mut b, *error_dst);
                        b.ins().call(rt["get_error"], &[d]);
                        b.ins().jump(label_blocks[*catch_label as usize], &[]);
                        // OK path: continue
                        b.switch_to_block(ok_blk);
                        b.seal_block(ok_blk);
                    }

                    JitOp::ReturnContinue => {
                        let v = b.ins().iconst(ptr_ty, GEN_CONTINUE);
                        b.ins().return_(&[v]);
                        terminated = true;
                    }
                    JitOp::ReturnError => {
                        // Transfer error from JIT_LAST_ERROR to env.error_msg
                        b.ins().call(rt["propagate_error"], &[env_ptr]);
                        let v = b.ins().iconst(ptr_ty, GEN_ERROR);
                        b.ins().return_(&[v]);
                        terminated = true;
                    }
                    JitOp::CallBuiltin { dst, name, args } => {
                        let d = slot_addr(&mut b, *dst);
                        // Leak the name string so it lives as long as the JIT code
                        let leaked_name = Box::leak(name.clone().into_boxed_str());
                        self._string_constants.push(leaked_name);
                        let name_p = b.ins().iconst(ptr_ty, leaked_name.as_ptr() as i64);
                        let name_l = b.ins().iconst(ptr_ty, leaked_name.len() as i64);
                        // Build args array on stack
                        if args.is_empty() {
                            let null_p = b.ins().iconst(ptr_ty, 0);
                            let zero = b.ins().iconst(ptr_ty, 0);
                            b.ins().call(rt["call_builtin"], &[d, name_p, name_l, null_p, zero]);
                        } else {
                            // Allocate stack space for args array
                            let arg_count = args.len() as u32;
                            let arg_array_slot = b.create_sized_stack_slot(StackSlotData::new(
                                StackSlotKind::ExplicitSlot, val_size * arg_count, val_align,
                            ));
                            // Clone each arg value into the array
                            for (i, arg_slot) in args.iter().enumerate() {
                                let arg_dst = b.ins().stack_addr(ptr_ty, arg_array_slot, (i as i32) * (val_size as i32));
                                let arg_src = slot_addr(&mut b, *arg_slot);
                                b.ins().call(rt["clone"], &[arg_dst, arg_src]);
                            }
                            let arr_ptr = b.ins().stack_addr(ptr_ty, arg_array_slot, 0);
                            let n = b.ins().iconst(ptr_ty, args.len() as i64);
                            b.ins().call(rt["call_builtin"], &[d, name_p, name_l, arr_ptr, n]);
                            // Drop the cloned args
                            for i in 0..args.len() {
                                let arg_p = b.ins().stack_addr(ptr_ty, arg_array_slot, (i as i32) * (val_size as i32));
                                b.ins().call(rt["drop"], &[arg_p]);
                            }
                        }
                        terminated = false;
                    }
                    JitOp::ThrowError { msg } => {
                        let msg_addr = slot_addr(&mut b, *msg);
                        b.ins().call(rt["throw_error"], &[msg_addr, env_ptr]);
                        // Don't return here — CheckError or ReturnError will handle it
                    }
                    JitOp::MutateInplace { slot, func } => {
                        let s = slot_addr(&mut b, *slot);
                        let rt_name = match func {
                            MutateFn::Reverse => "reverse_inplace",
                            MutateFn::Sort => "sort_inplace",
                        };
                        b.ins().call(rt[rt_name], &[s]);
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
        Ok(unsafe { std::mem::transmute::<*const u8, JitFilterFn>(code_ptr) })
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
    decl!("str_rc", [p, p], []);
    decl!("is_truthy", [p], [p]);
    decl!("field_is_truthy", [p, p, p], [p]);
    decl!("field_cmp_num", [p, p, p, f, i], [p]);
    decl!("field_binop_field", [p, p, p, p, p, p, i], [p]);
    decl!("field_binop_const", [p, p, p, p, p, i, i], [p]);
    decl!("index", [p, p, p], [p]);
    decl!("index_field", [p, p, p, p], [p]);
    decl!("yield_field_ref", [p, p, p, p, p], [p]);
    decl!("binop", [p, i, p, p], [p]);
    decl!("add_move", [p, p, p], [p]);
    decl!("unaryop", [p, i, p], [p]);
    decl!("negate", [p, p], [p]);
    decl!("not", [p, p], [p]);
    decl!("kind", [p], [p]);
    decl!("len", [p], [p]);
    decl!("array_get", [p, p, p], []);
    decl!("obj_get_idx", [p, p, p], []);
    decl!("get_var", [p, p, i], []);
    decl!("set_var", [p, i, p], []);
    decl!("take_var", [p, p, i], []);
    decl!("path_extract", [p, p, p], [p]);
    decl!("path_insert", [p, p, p], [p]);
    decl!("collect_begin", [p], []);
    decl!("collect_push", [p, p], []);
    decl!("collect_finish", [p, p], []);
    decl!("obj_new", [p, p], []);
    decl!("obj_insert", [p, p, p], []);
    decl!("obj_insert_str_key", [p, p, p, p], []);
    decl!("obj_push_str_key", [p, p, p, p], []);
    decl!("obj_copy_field", [p, p, p, p, p, p], []);
    decl!("obj_from_fields", [p, p, p, p], []);
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
    decl!("propagate_error", [p], []);
    decl!("has_error", [], [p]);  // -> 0/1
    decl!("get_error", [p], []);  // dst
    decl!("throw_error", [p, p], [p]);
    decl!("call_builtin", [p, p, p, p, p], [p]);  // dst, name_ptr, name_len, args_ptr, nargs -> status
    decl!("reverse_inplace", [p], [p]);  // v: *mut Value -> status
    decl!("sort_inplace", [p], [p]);     // v: *mut Value -> status
    Ok(())
}

// ============================================================================
// Public API
// ============================================================================

pub fn is_jit_compilable(expr: &Expr) -> bool {
    is_jit_compilable_with_funcs(expr, &[])
}

pub fn is_jit_compilable_with_funcs(expr: &Expr, funcs: &[CompiledFunc]) -> bool {
    let mut fl = Flattener::new();
    fl.funcs = funcs.to_vec();
    let input = fl.alloc_slot();
    fl.flatten_gen(expr, input)
}

unsafe extern "C" fn collect_callback(value: *const Value, ctx: *mut u8) -> i64 {
    let results = &mut *(ctx as *mut Vec<Value>);
    results.push((*value).clone());
    GEN_CONTINUE
}

// Global reusable JIT env — safe because jq-jit is single-threaded (uses Rc, not Arc).
struct GlobalJitEnv(std::cell::UnsafeCell<Option<JitEnv>>);
unsafe impl Sync for GlobalJitEnv {}
static REUSABLE_ENV: GlobalJitEnv = GlobalJitEnv(std::cell::UnsafeCell::new(None));

fn with_jit_env<R>(f: impl FnOnce(&mut JitEnv) -> R) -> R {
    let env_opt = unsafe { &mut *REUSABLE_ENV.0.get() };
    let env = env_opt.get_or_insert_with(JitEnv::new);
    env.collect_stacks.clear();
    env.str_bufs.clear();
    env.try_depth = 0;
    env.error_msg = None;
    f(env)
}


pub fn execute_jit(func: JitFilterFn, input: &Value) -> Result<Vec<Value>> {
    with_jit_env(|env| {
        let mut results: Vec<Value> = Vec::new();
        let result = unsafe {
            func(input as *const Value, env, collect_callback,
                 &mut results as *mut Vec<Value> as *mut u8)
        };
        if result == GEN_ERROR {
            let err_msg = env.error_msg.take()
                .unwrap_or_else(|| "JIT execution error".to_string());
            results.push(Value::Error(Rc::new(err_msg)));
        }
        Ok(results)
    })
}

/// Streaming callback context for execute_jit_cb.
/// cb_fat stores the fat pointer (data+vtable) as two usize values.
struct StreamCtx {
    cb_fat: [usize; 2],
    had_error: bool,
    error_msg: Option<String>,
}

unsafe extern "C" fn stream_callback(value: *const Value, ctx: *mut u8) -> i64 {
    let sctx = &mut *(ctx as *mut StreamCtx);
    let cb: &mut dyn FnMut(&Value) -> Result<bool> =
        std::mem::transmute::<[usize; 2], &mut dyn FnMut(&Value) -> Result<bool>>(sctx.cb_fat);
    match cb(&*value) {
        Ok(true) => GEN_CONTINUE,
        Ok(false) => 0, // stop
        Err(e) => {
            sctx.had_error = true;
            sctx.error_msg = Some(format!("{}", e));
            0 // stop
        }
    }
}

/// Execute JIT filter with streaming callback (avoids Vec allocation).
pub fn execute_jit_cb(func: JitFilterFn, input: &Value, cb: &mut dyn FnMut(&Value) -> Result<bool>) -> Result<bool> {
    with_jit_env(|env| {
        // Store the fat pointer as raw bytes to avoid lifetime issues.
        // Safety: sctx lives only within this scope, same as cb.
        let cb_fat: [usize; 2] = unsafe {
            std::mem::transmute::<&mut dyn FnMut(&Value) -> Result<bool>, [usize; 2]>(cb)
        };
        let mut sctx = StreamCtx {
            cb_fat,
            had_error: false,
            error_msg: None,
        };
        let result = unsafe {
            func(input as *const Value, env, stream_callback,
                 &mut sctx as *mut StreamCtx as *mut u8)
        };
        if sctx.had_error {
            if let Some(msg) = sctx.error_msg {
                return Err(anyhow::anyhow!("{}", msg));
            }
        }
        if result == GEN_ERROR {
            let err_msg = env.error_msg.take()
                .unwrap_or_else(|| "JIT execution error".to_string());
            return Err(anyhow::anyhow!("{}", err_msg));
        }
        Ok(true)
    })
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
