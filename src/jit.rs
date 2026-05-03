//! Cranelift JIT compiler for jq filters.
//!
//! Two-phase approach:
//! 1. Flatten IR expression tree into linear JitOps
//! 2. Generate Cranelift IR from JitOps
//!
//! Key design: expressions are classified as "scalar" (exactly one output)
//! or "generator" (zero or more outputs). Scalar expressions compile to
//! direct computation; generators compile to loops with Yield operations.

use std::cell::{Cell, RefCell, UnsafeCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use anyhow::{Result, bail};

// JIT runtime state used by trampolines and codegen.
//
// jq-jit is single-threaded per execution but `cargo test` runs many JIT'd
// queries in parallel. We keep transient runtime state thread-local, and the
// per-execution error flag lives on `JitEnv` itself — codegen loads it via
// `env_ptr + offset_of!(JitEnv, error_flag)`, giving every thread its own
// flag without runtime address resolution and without a process-wide lock.
thread_local! {
    /// Last error message produced by a runtime trampoline.
    static JIT_LAST_ERROR: UnsafeCell<Option<String>> = const { UnsafeCell::new(None) };
    /// Direct `Value` storage for the `error` builtin — avoids Value→JSON→Value
    /// round-trip in try-catch.
    static JIT_ERROR_VALUE: UnsafeCell<Option<Value>> = const { UnsafeCell::new(None) };
    /// Closure ops captured at compile time; consumed by runtime trampolines
    /// during execution.
    static JIT_CLOSURE_OPS: UnsafeCell<Vec<Expr>> = const { UnsafeCell::new(Vec::new()) };
    /// Pointer to the `JitEnv` currently being driven by `execute_jit*`.
    /// `set_jit_error` consults this to flip the env's `error_flag` without
    /// plumbing an env pointer through every fallible trampoline signature.
    static CURRENT_ENV: Cell<*mut JitEnv> = const { Cell::new(std::ptr::null_mut()) };
}

fn current_env() -> *mut JitEnv {
    CURRENT_ENV.with(|c| c.get())
}

fn set_error_flag() {
    let env = current_env();
    if !env.is_null() {
        unsafe { (*env).error_flag = 1; }
    }
}

fn clear_error_flag() {
    let env = current_env();
    if !env.is_null() {
        unsafe { (*env).error_flag = 0; }
    }
}

fn set_jit_error(msg: String) {
    JIT_LAST_ERROR.with(|cell| unsafe { *cell.get() = Some(msg); });
    set_error_flag();
}

fn set_jit_error_value(val: Value) {
    JIT_ERROR_VALUE.with(|cell| unsafe { *cell.get() = Some(val); });
    set_error_flag();
}

fn take_jit_error() -> Option<String> {
    JIT_LAST_ERROR.with(|cell| unsafe { (*cell.get()).take() })
}

fn take_jit_error_value() -> Option<Value> {
    JIT_ERROR_VALUE.with(|cell| unsafe { (*cell.get()).take() })
}

fn clear_jit_error() {
    JIT_LAST_ERROR.with(|cell| unsafe { *cell.get() = None; });
    JIT_ERROR_VALUE.with(|cell| unsafe { *cell.get() = None; });
    clear_error_flag();
}
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, StackSlotData, StackSlotKind};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, FuncId};

use crate::ir::*;
use crate::value::{Value, ObjInner, NumRepr};

const GEN_CONTINUE: i64 = 1;
const GEN_ERROR: i64 = -1;

// ============================================================================
// Compile-time constant evaluation
// ============================================================================

/// Try to evaluate an expression at compile time. Returns Some(Value) if the
/// expression contains only literals and structural constructors (no Input/LoadVar).
/// Used to constant-fold large literal arrays/objects to avoid generating
/// hundreds of JitOps for their construction.
fn try_const_eval(expr: &Expr) -> Option<Value> {
    match expr {
        Expr::Literal(lit) => Some(match lit {
            Literal::Null => Value::Null,
            Literal::True => Value::True,
            Literal::False => Value::False,
            Literal::Num(n, repr) => Value::number_opt(*n, repr.clone()),
            Literal::Str(s) => Value::from_str(s),
        }),
        Expr::Negate { operand } => {
            if let Value::Num(n, _) = try_const_eval(operand)? {
                Some(Value::number(-n))
            } else { None }
        }
        Expr::Collect { generator } => {
            let mut arr = Vec::new();
            const_eval_gen(generator, &mut arr)?;
            Some(Value::Arr(Rc::new(arr)))
        }
        Expr::ObjectConstruct { pairs } => {
            let mut obj = crate::value::ObjMap::new();
            for (k, v) in pairs {
                let key = match try_const_eval(k)? {
                    Value::Str(s) => s,
                    _ => return None,
                };
                let val = try_const_eval(v)?;
                obj.insert(key, val);
            }
            Some(Value::object_from_map(obj))
        }
        Expr::BinOp { op, lhs, rhs } => {
            let l = try_const_eval(lhs)?;
            let r = try_const_eval(rhs)?;
            crate::eval::eval_binop(*op, &l, &r).ok()
        }
        Expr::StringInterpolation { parts } => {
            let mut s = String::new();
            for p in parts {
                match p {
                    StringPart::Literal(lit) => s.push_str(lit),
                    StringPart::Expr(e) => {
                        let v = try_const_eval(e)?;
                        match &v {
                            Value::Str(vs) => s.push_str(vs.as_str()),
                            _ => return None,
                        }
                    }
                }
            }
            Some(Value::from_str(&s))
        }
        _ => None,
    }
}

/// Collect constant generator outputs into a Vec.
fn const_eval_gen(expr: &Expr, out: &mut Vec<Value>) -> Option<()> {
    match expr {
        Expr::Comma { left, right } => {
            const_eval_gen(left, out)?;
            const_eval_gen(right, out)?;
            Some(())
        }
        Expr::Range { from, to, step } => {
            let f = if let Value::Num(n, _) = try_const_eval(from)? { n } else { return None; };
            let t = if let Value::Num(n, _) = try_const_eval(to)? { n } else { return None; };
            let s = if let Some(step_expr) = step {
                if let Value::Num(n, _) = try_const_eval(step_expr)? { n } else { return None; }
            } else { 1.0 };
            if s == 0.0 || !s.is_finite() || !f.is_finite() || !t.is_finite() { return None; }
            // Limit constant range to avoid huge arrays
            let count = ((t - f) / s).ceil() as i64;
            if count < 0 || count > 10000 { return None; }
            let mut i = f;
            if s > 0.0 {
                while i < t { out.push(Value::number(i)); i += s; }
            } else {
                while i > t { out.push(Value::number(i)); i += s; }
            }
            Some(())
        }
        _ => {
            out.push(try_const_eval(expr)?);
            Some(())
        }
    }
}

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
        // match/capture are 0-or-1 generators: non-match throws error → empty
        Expr::RegexMatch { input_expr, re, flags } | Expr::RegexCapture { input_expr, re, flags } => {
            is_scalar(input_expr) && is_scalar(re) && is_scalar(flags)
        }
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
                ("walk", _) | ("pick", _) | ("skip", _) | ("del", _) | ("exec", 2)
                | ("fromcsv", _) | ("fromtsv", _) | ("fromcsvh", _) | ("fromtsvh", _) => false,
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
        // Regex operations produce exactly one value
        Expr::RegexTest { input_expr, re, flags } => is_scalar(input_expr) && is_scalar(re) && is_scalar(flags),
        // match/capture are generators (0 or 1 outputs) — non-match produces empty
        Expr::RegexMatch { .. } => false,
        Expr::RegexCapture { .. } => false,
        Expr::RegexSub { input_expr, re, tostr, flags } => {
            // Only JIT when tostr is a literal string (no capture group references)
            matches!(tostr.as_ref(), Expr::Literal(Literal::Str(_)))
                && is_scalar(input_expr) && is_scalar(re) && is_scalar(flags)
        }
        Expr::RegexGsub { input_expr, re, tostr, flags } => {
            // Only JIT when tostr is a literal string (no capture group references)
            matches!(tostr.as_ref(), Expr::Literal(Literal::Str(_)))
                && is_scalar(input_expr) && is_scalar(re) && is_scalar(flags)
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
enum JitOp {
    // Value construction
    Clone { dst: SlotId, src: SlotId },
    /// Clone a value from a pre-computed constant pointer (lives as long as JitCompiler).
    LoadConst { dst: SlotId, const_ptr: *const Value },
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
    /// Branch on type tag comparison: loads tag byte from src and compares against expected.
    /// tags is a bitmask of matching tag values (bit 0 = Null, 1 = False, 2 = True, 3 = Num, etc.)
    TypeCmpBranch { src: SlotId, tags: u8, then_label: LabelId, else_label: LabelId },
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
    /// Move src into var (drop old, leave src as Null) — avoids clone for accumulator stores.
    MoveToVar { var_index: u16, src: SlotId },

    /// In-place path extract: swap container[key] with Null, return old element.
    /// Container is modified in-place via Rc::make_mut (no clone if refcount == 1).
    PathExtract { element: SlotId, container: SlotId, key: SlotId },
    /// In-place path insert: set container[key] = val, consuming val (slot becomes Null).
    /// Container is modified in-place via Rc::make_mut (no clone if refcount == 1).
    PathInsert { container: SlotId, key: SlotId, val: SlotId },
    /// Take value at integer index from container (arr[i] or obj.entries[i].value).
    /// O(1) direct index access, no hash lookup. Container must have refcount == 1.
    TakeByIdx { dst: SlotId, container: SlotId, idx_var: u32 },
    /// Put value at integer index into container (arr[i] or obj.entries[i].value).
    /// O(1) direct index access, no hash lookup. Container must have refcount == 1.
    PutByIdx { container: SlotId, idx_var: u32, val: SlotId },
    /// In-place setpath: container[path...] = val. Container must have refcount == 1.
    /// Both path and val slots become Null after the call.
    SetPathMut { container: SlotId, path: SlotId, val: SlotId },

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

    // Range loop
    ToF64Var { dst_var: u32, src: SlotId },
    F64Less { dst_var: u32, a_var: u32, b_var: u32 },
    F64Add { dst_var: u32, a_var: u32, b_var: u32 },
    F64Sub { dst_var: u32, a_var: u32, b_var: u32 },
    F64Mul { dst_var: u32, a_var: u32, b_var: u32 },
    F64Div { dst_var: u32, a_var: u32, b_var: u32 },
    F64Rem { dst_var: u32, a_var: u32, b_var: u32 },
    F64Neg { dst_var: u32, src_var: u32 },
    /// Native unary math: 0=floor, 1=ceil, 2=sqrt, 3=fabs, 4=trunc, 5=nearest
    F64Math { dst_var: u32, src_var: u32, kind: u8 },
    /// Libm unary call: 0=sin, 1=cos, 2=tan, 3=asin, 4=acos, 5=atan, 6=exp, 7=exp2, 8=log, 9=log2, 10=log10, 11=cbrt
    F64Libm { dst_var: u32, src_var: u32, func: u8 },
    /// Float comparison: dst = 1 if a `cc` b, else 0. cc: 0=Ge, 1=Gt, 2=Le, 3=Lt, 4=Eq, 5=Ne
    F64Cmp { dst_var: u32, a_var: u32, b_var: u32, cc: u8 },
    F64Num { dst: SlotId, src_var: u32 },
    /// Range check: (step>0 && cur<to) || (step<=0 && cur>to)
    RangeCheck { dst_var: u32, cur_var: u32, to_var: u32, step_var: u32 },
    /// Set a Cranelift variable to f64 constant (stored as i64 bits)
    F64Const { dst_var: u32, val: f64 },
    /// Copy f64 variable value: dst = src (no arithmetic)
    F64Move { dst_var: u32, src_var: u32 },

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

    /// Like `CheckError` but does *not* drain the pending error. Used on the
    /// outside-try-catch propagation path so `ReturnError` / `propagate_error`
    /// can still read `JIT_LAST_ERROR` and transfer it to `env.error_msg`.
    JumpIfError { label: LabelId },

    // Error throwing
    ThrowError { msg: SlotId },

    // CallBuiltin: call a runtime builtin function
    // args[0] is input, args[1..] are the evaluated arguments
    CallBuiltin { dst: SlotId, name: String, args: Vec<SlotId> },

    /// In-place unary mutate: call runtime fn(v: *mut Value) -> i64
    /// Consumes input_slot and mutates the clone in-place.
    MutateInplace { slot: SlotId, func: MutateFn },

    /// Fused [range(n)]: pre-allocate and fill array in one call.
    CollectRange { dst: SlotId, n: SlotId },

    /// In-place array push: arr.push(val). Container must have refcount == 1.
    /// Handles null + push → [val] (jq null identity for addition).
    ArrPush { arr: SlotId, val: SlotId },

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
            Literal::Num(n, repr) => Value::number_opt(*n, repr.clone()),
            Literal::Str(s) => Value::Str(crate::value::KeyStr::from(s.as_str())),
        };
        self.hoist_value(val)
    }

    /// Store a pre-computed Value and return a stable pointer.
    fn hoist_value(&mut self, val: Value) -> *const Value {
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
            Expr::Pipe { left, right } => {
                let new_left = Box::new(self.inline_func_calls(left));
                let new_right = Box::new(self.inline_func_calls(right));
                // Pipe identity simplification: . | X → X, X | . → X
                if matches!(new_left.as_ref(), Expr::Input) { return *new_right; }
                if matches!(new_right.as_ref(), Expr::Input) { return *new_left; }
                // Semantic: cmp_expr | not → inverted cmp_expr (must run before beta-reduction)
                if matches!(new_right.as_ref(), Expr::Not) {
                    if let Expr::BinOp { op, lhs, rhs } = new_left.as_ref() {
                        if let Some(inv) = op.invert_cmp() {
                            return Expr::BinOp { op: inv, lhs: lhs.clone(), rhs: rhs.clone() };
                        }
                    }
                }
                // Semantic optimizations for to_entries pipelines (must run before beta-reduction)
                if let Expr::UnaryOp { op: UnaryOp::ToEntries, .. } = new_left.as_ref() {
                    // NOTE: `to_entries | from_entries` is NOT universal identity — arrays
                    // must surface the numeric-key type error and `[]` round-trips to `{}`
                    // (issue #73). Do not rewrite.
                    // to_entries | map(.value) → [.[]]
                    // to_entries | map(.key) → keys_unsorted
                    if let Expr::Collect { generator } = new_right.as_ref() {
                        if let Expr::Pipe { left: map_l, right: map_r } = generator.as_ref() {
                            if matches!(map_l.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                                if let Expr::Index { expr: idx_e, key } = map_r.as_ref() {
                                    if matches!(idx_e.as_ref(), Expr::Input) {
                                        if let Expr::Literal(Literal::Str(s)) = key.as_ref() {
                                            if s == "value" {
                                                return Expr::Collect {
                                                    generator: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                                                };
                                            }
                                            if s == "key" {
                                                return Expr::UnaryOp { op: UnaryOp::KeysUnsorted, operand: Box::new(Expr::Input) };
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Composed: to_entries | Pipe(map(.key/.value), tail) → rewrite | tail
                if let Expr::UnaryOp { op: UnaryOp::ToEntries, .. } = new_left.as_ref() {
                    if let Expr::Pipe { left: pl, right: pr } = new_right.as_ref() {
                        if let Expr::Collect { generator } = pl.as_ref() {
                            if let Expr::Pipe { left: map_l, right: map_r } = generator.as_ref() {
                                if matches!(map_l.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) {
                                    if let Expr::Index { expr: idx_e, key } = map_r.as_ref() {
                                        if matches!(idx_e.as_ref(), Expr::Input) {
                                            if let Expr::Literal(Literal::Str(s)) = key.as_ref() {
                                                if s == "key" {
                                                    let rewritten = Box::new(Expr::UnaryOp { op: UnaryOp::KeysUnsorted, operand: Box::new(Expr::Input) });
                                                    return self.inline_func_calls(&Expr::Pipe { left: rewritten, right: pr.clone() });
                                                }
                                                if s == "value" {
                                                    let rewritten = Box::new(Expr::Collect {
                                                        generator: Box::new(Expr::Each { input_expr: Box::new(Expr::Input) }),
                                                    });
                                                    return self.inline_func_calls(&Expr::Pipe { left: rewritten, right: pr.clone() });
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Beta-reduction: Pipe(E, F) → F[E/Input] when E is scalar and F has free Input
                if new_left.is_simple_scalar() && new_right.is_input_free() {
                    return new_right.substitute_input(&new_left);
                }
                Expr::Pipe { left: new_left, right: new_right }
            }
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
            } else {
                // Outside a try-catch: branch to ReturnError without draining the
                // pending error, so `propagate_error` can still hand it off to
                // `env.error_msg` (fixes generic "JIT execution error" messages).
                let error_label = self.alloc_label();
                let ok_label = self.alloc_label();
                self.ops.push(JitOp::JumpIfError { label: error_label });
                self.ops.push(JitOp::Jump { label: ok_label });
                self.ops.push(JitOp::Label { id: error_label });
                self.ops.push(JitOp::ReturnError);
                self.ops.push(JitOp::Label { id: ok_label });
            }
        }
    }

    /// Emit a fallible op with explicit error propagation even outside try blocks.
    fn emit_propagating(&mut self, op: JitOp) {
        self.ops.push(op);
        if let Some((catch_label, error_slot)) = self.try_catch_target {
            self.ops.push(JitOp::CheckError { error_dst: error_slot, catch_label });
        } else {
            let error_label = self.alloc_label();
            let ok_label = self.alloc_label();
            self.ops.push(JitOp::JumpIfError { label: error_label });
            self.ops.push(JitOp::Jump { label: ok_label });
            self.ops.push(JitOp::Label { id: error_label });
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
                self.emit(JitOp::MoveToVar { var_index: *var_index, src: val });
                let result = self.flatten_scalar(body, input_slot);
                self.emit(JitOp::MoveToVar { var_index: *var_index, src: old });
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
                // Try narrow fused path: until(. cmp K; . op S)
                let narrow_fused = Self::classify_f64_until_narrow(cond, update);
                // Collect all variable references in cond and update
                let mut ext_vars = Vec::new();
                if narrow_fused.is_none() {
                    Self::collect_loadvar_indices(cond, &mut ext_vars);
                    Self::collect_loadvar_indices(update, &mut ext_vars);
                }
                // Try general fused f64 until: both cond and update are pure f64 on . and vars
                let general_f64 = narrow_fused.is_none()
                    && Self::is_pure_f64_expr_multi(cond, &ext_vars)
                    && Self::is_pure_f64_expr_multi(update, &ext_vars);

                // Both fused paths coerce input to f64 via `to_f64`, which
                // silently turns a non-numeric input into 0.0 and runs the
                // loop from there. jq instead compares the raw input value
                // via cross-type ordering and e.g. exits immediately from
                // `"hello" | until(. > 5; . + 1)` (issue #57). Guard the
                // fused paths with a runtime Num check on the input; fall
                // back to the generic semantic path otherwise.
                let use_type_guard = narrow_fused.is_some() || general_f64;
                let (generic_entry, merge_label, result_slot) = if use_type_guard {
                    let gen = self.alloc_label();
                    let merge = self.alloc_label();
                    let result = self.alloc_slot();
                    let fused_label = self.alloc_label();
                    self.emit(JitOp::TypeCmpBranch {
                        src: input_slot,
                        tags: 1u8 << 3, // TAG_NUM
                        then_label: fused_label,
                        else_label: gen,
                    });
                    self.emit(JitOp::Label { id: fused_label });
                    (Some(gen), Some(merge), Some(result))
                } else {
                    (None, None, None)
                };

                let emit_fused_tail = |s: &mut Self, fused_out: SlotId| {
                    if let (Some(merge), Some(gen_lbl)) = (merge_label, generic_entry) {
                        s.emit(JitOp::Jump { label: merge });
                        s.emit(JitOp::Label { id: gen_lbl });
                        // Inline generic path writing result into fused_out.
                        let current = s.alloc_slot();
                        s.emit(JitOp::Clone { dst: current, src: input_slot });
                        let g_head = s.alloc_label();
                        let g_body = s.alloc_label();
                        let g_done = s.alloc_label();
                        s.emit(JitOp::Label { id: g_head });
                        let cond_val = s.flatten_scalar(cond, current);
                        s.emit(JitOp::IfTruthy { src: cond_val, then_label: g_done, else_label: g_body });
                        s.emit(JitOp::Drop { slot: cond_val });
                        s.emit(JitOp::Label { id: g_body });
                        let new_val = s.flatten_scalar(update, current);
                        s.emit(JitOp::Drop { slot: current });
                        s.emit(JitOp::Clone { dst: current, src: new_val });
                        s.emit(JitOp::Drop { slot: new_val });
                        s.emit(JitOp::Jump { label: g_head });
                        s.emit(JitOp::Label { id: g_done });
                        s.emit(JitOp::Clone { dst: fused_out, src: current });
                        s.emit(JitOp::Drop { slot: current });
                        s.emit(JitOp::Label { id: merge });
                    }
                };
                if let Some((cond_cc, cond_const, update_op, update_const)) = narrow_fused {
                    // Fast narrow path: preload constants outside loop
                    let acc = self.alloc_var();
                    self.emit(JitOp::ToF64Var { dst_var: acc, src: input_slot });
                    let k_var = self.alloc_var();
                    self.emit(JitOp::F64Const { dst_var: k_var, val: cond_const });
                    let s_var = self.alloc_var();
                    self.emit(JitOp::F64Const { dst_var: s_var, val: update_const });
                    let cmp = self.alloc_var();
                    let head = self.alloc_label();
                    let body = self.alloc_label();
                    let done = self.alloc_label();
                    self.emit(JitOp::Label { id: head });
                    self.emit(JitOp::F64Cmp { dst_var: cmp, a_var: acc, b_var: k_var, cc: cond_cc });
                    self.emit(JitOp::BranchOnVar { var: cmp, nonzero_label: done, zero_label: body });
                    self.emit(JitOp::Label { id: body });
                    match update_op {
                        0 => self.emit(JitOp::F64Add { dst_var: acc, a_var: acc, b_var: s_var }),
                        1 => self.emit(JitOp::F64Sub { dst_var: acc, a_var: acc, b_var: s_var }),
                        2 => self.emit(JitOp::F64Mul { dst_var: acc, a_var: acc, b_var: s_var }),
                        _ => unreachable!(),
                    }
                    self.emit(JitOp::Jump { label: head });
                    self.emit(JitOp::Label { id: done });
                    let fused_out = result_slot.unwrap_or_else(|| self.alloc_slot());
                    self.emit(JitOp::F64Num { dst: fused_out, src_var: acc });
                    emit_fused_tail(self, fused_out);
                    fused_out
                } else if general_f64 {
                    // General fused path: compile_f64_expr inside loop
                    // Pre-load external variables as f64 vars
                    let acc = self.alloc_var();
                    self.emit(JitOp::ToF64Var { dst_var: acc, src: input_slot });
                    let mut var_map: Vec<(u16, u32)> = Vec::new();
                    for &vi in &ext_vars {
                        let slot = self.alloc_slot();
                        self.emit(JitOp::GetVar { dst: slot, var_index: vi });
                        let fvar = self.alloc_var();
                        self.emit(JitOp::ToF64Var { dst_var: fvar, src: slot });
                        self.emit(JitOp::Drop { slot });
                        var_map.push((vi, fvar));
                    }
                    let head = self.alloc_label();
                    let body = self.alloc_label();
                    let done = self.alloc_label();
                    self.emit(JitOp::Label { id: head });
                    let cond_v = self.compile_f64_expr_multi(cond, acc, &var_map);
                    self.emit(JitOp::BranchOnVar { var: cond_v, nonzero_label: done, zero_label: body });
                    self.emit(JitOp::Label { id: body });
                    let new_acc = self.compile_f64_expr_multi(update, acc, &var_map);
                    if new_acc != acc {
                        self.emit(JitOp::F64Move { dst_var: acc, src_var: new_acc });
                    }
                    self.emit(JitOp::Jump { label: head });
                    self.emit(JitOp::Label { id: done });
                    let fused_out = result_slot.unwrap_or_else(|| self.alloc_slot());
                    self.emit(JitOp::F64Num { dst: fused_out, src_var: acc });
                    emit_fused_tail(self, fused_out);
                    fused_out
                } else {
                    // Generic path: until(cond; update): loop { if cond then break; update }; yield current
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
            // Regex operations — delegate to runtime builtins
            Expr::RegexTest { input_expr, re, flags } => {
                let inp = self.flatten_scalar(input_expr, input_slot);
                let re_val = self.flatten_scalar(re, input_slot);
                let flags_val = self.flatten_scalar(flags, input_slot);
                let out = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: out, name: "test".to_string(), args: vec![inp, re_val, flags_val] });
                self.emit(JitOp::Drop { slot: inp });
                self.emit(JitOp::Drop { slot: re_val });
                self.emit(JitOp::Drop { slot: flags_val });
                out
            }
            Expr::RegexScan { input_expr, re, flags } => {
                let inp = self.flatten_scalar(input_expr, input_slot);
                let re_val = self.flatten_scalar(re, input_slot);
                let flags_val = self.flatten_scalar(flags, input_slot);
                let out = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: out, name: "scan".to_string(), args: vec![inp, re_val, flags_val] });
                self.emit(JitOp::Drop { slot: inp });
                self.emit(JitOp::Drop { slot: re_val });
                self.emit(JitOp::Drop { slot: flags_val });
                out
            }
            // RegexMatch/RegexCapture are handled as generators (via eval fallback)
            // since they can produce 0 or 1 outputs (non-match → empty)
            Expr::RegexMatch { .. } | Expr::RegexCapture { .. } => {
                unreachable!("RegexMatch/RegexCapture should not reach flatten_scalar")
            }
            Expr::RegexSub { input_expr, re, tostr, flags } => {
                let inp = self.flatten_scalar(input_expr, input_slot);
                let re_val = self.flatten_scalar(re, input_slot);
                let tostr_val = self.flatten_scalar(tostr, input_slot);
                let flags_val = self.flatten_scalar(flags, input_slot);
                let out = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: out, name: "sub".to_string(), args: vec![inp, re_val, tostr_val, flags_val] });
                self.emit(JitOp::Drop { slot: inp });
                self.emit(JitOp::Drop { slot: re_val });
                self.emit(JitOp::Drop { slot: tostr_val });
                self.emit(JitOp::Drop { slot: flags_val });
                out
            }
            Expr::RegexGsub { input_expr, re, tostr, flags } => {
                let inp = self.flatten_scalar(input_expr, input_slot);
                let re_val = self.flatten_scalar(re, input_slot);
                let tostr_val = self.flatten_scalar(tostr, input_slot);
                let flags_val = self.flatten_scalar(flags, input_slot);
                let out = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: out, name: "gsub".to_string(), args: vec![inp, re_val, tostr_val, flags_val] });
                self.emit(JitOp::Drop { slot: inp });
                self.emit(JitOp::Drop { slot: re_val });
                self.emit(JitOp::Drop { slot: tostr_val });
                self.emit(JitOp::Drop { slot: flags_val });
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
                // Detect reduce gen as $x ([]; . + [f($x)]) → [gen | f(.)]
                // where init is [] and update is . + [inner] with inner not using accumulator
                if matches!(init.as_ref(), Expr::Collect { generator } if matches!(generator.as_ref(), Expr::Empty)) {
                    if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = update.as_ref() {
                        if matches!(lhs.as_ref(), Expr::Input) {
                            if let Expr::Collect { generator: inner_gen } = rhs.as_ref() {
                                if !uses_input(inner_gen) {
                                    // Build equivalent expression: [source | f(.)]
                                    // where f replaces $x with .
                                    let f = inner_gen.substitute_var(*var_index, &Expr::Input);
                                    let collect_gen = if matches!(&f, Expr::Input) {
                                        source.as_ref().clone()
                                    } else {
                                        Expr::Pipe { left: source.clone(), right: Box::new(f) }
                                    };
                                    let equiv = Expr::Collect { generator: Box::new(collect_gen) };
                                    return self.flatten_scalar(&equiv, input_slot);
                                }
                            }
                        }
                    }
                }
                // Scalar reduce: init is scalar, update is scalar, source is a generator
                let acc_val = self.flatten_scalar(init, input_slot);
                let old_acc = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: old_acc, var_index: *acc_index });
                self.emit(JitOp::MoveToVar { var_index: *acc_index, src: acc_val });
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
                // Also detect `+= rhs` which is: LetBinding { var, value: rhs, body: Update { path: ., update: . + LoadVar(var) } }
                let acc_add_rhs = if let Expr::BinOp { op, lhs, rhs } = update.as_ref() {
                    if matches!(op, BinOp::Add) && matches!(lhs.as_ref(), Expr::Input) && is_scalar(rhs) {
                        Some(rhs.as_ref())
                    } else { None }
                } else if let Expr::LetBinding { var_index: rhs_var, value: rhs_value, body } = update.as_ref() {
                    if let Expr::Update { path_expr, update_expr } = body.as_ref() {
                        if matches!(path_expr.as_ref(), Expr::Input) {
                            if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = update_expr.as_ref() {
                                if matches!(lhs.as_ref(), Expr::Input)
                                    && matches!(rhs.as_ref(), Expr::LoadVar { var_index: v } if *v == *rhs_var)
                                    && is_scalar(rhs_value)
                                {
                                    Some(rhs_value.as_ref())
                                } else { None }
                            } else { None }
                        } else { None }
                    } else { None }
                } else { None };

                // Detect accumulator-push pattern: reduce ... (init; . + [gen]) — in-place push
                let acc_push_collect = if acc_add_rhs.is_none() {
                    if let Expr::BinOp { op, lhs, rhs } = update.as_ref() {
                        if matches!(op, BinOp::Add) && matches!(lhs.as_ref(), Expr::Input) {
                            if let Expr::Collect { generator } = rhs.as_ref() {
                                Some(generator.as_ref())
                            } else { None }
                        } else { None }
                    } else { None }
                } else { None };

                // Detect fused numeric reduce: reduce range(from;to) as $x (num_init; f64_body)
                // Keeps everything in f64 variables — zero Value boxing in the loop body.
                //
                // init=Input was previously allowed but `to_f64` silently coerces
                // non-number Values to 0.0, so a string `.` input slipped through
                // the f64 loop and returned the bare range sum instead of erroring
                // (#223 type-violation case). Only allow numeric literals — any
                // dynamic init falls back to the boxed reduce path that surfaces
                // jq's "X and Y cannot be added" error.
                let is_fused_range_f64 = if let Expr::Range { from, to, step } = source.as_ref() {
                    is_scalar(from) && is_scalar(to) && step.as_ref().is_none_or(|s| is_scalar(s))
                        && matches!(init.as_ref(), Expr::Literal(Literal::Num(..)))
                        && Self::is_pure_f64_expr(update, *var_index)
                } else { false };

                if is_fused_range_f64 {
                    if let Expr::Range { from, to, step } = source.as_ref() {
                        // Set up range variables
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
                        let acc_f64 = self.alloc_var();
                        self.emit(JitOp::ToF64Var { dst_var: cur, src: from_val });
                        self.emit(JitOp::ToF64Var { dst_var: to_v, src: to_val });
                        self.emit(JitOp::ToF64Var { dst_var: step_v, src: step_val });
                        // The MoveToVar at the start of this Reduce arm consumed
                        // `acc_val` and replaced the slot with Null, so reading
                        // from it directly would seed the f64 accumulator with 0
                        // instead of the init expression's value (#223). Re-fetch
                        // from the accumulator variable we just wrote.
                        let acc_init = self.alloc_slot();
                        self.emit(JitOp::GetVar { dst: acc_init, var_index: *acc_index });
                        self.emit(JitOp::ToF64Var { dst_var: acc_f64, src: acc_init });
                        self.emit(JitOp::Drop { slot: acc_init });
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
                        // Compile update body as pure f64 expression
                        let new_acc = self.compile_f64_expr(update, *var_index, acc_f64, cur);
                        // Copy result to accumulator var (may be same if update = . + $x)
                        if new_acc != acc_f64 {
                            self.emit(JitOp::F64Move { dst_var: acc_f64, src_var: new_acc });
                        }
                        self.emit(JitOp::F64Add { dst_var: cur, a_var: cur, b_var: step_v });
                        self.emit(JitOp::Jump { label: head });
                        self.emit(JitOp::Label { id: done });
                        // Convert f64 accumulator back to Value::Num
                        let result = self.alloc_slot();
                        self.emit(JitOp::F64Num { dst: result, src_var: acc_f64 });
                        self.emit(JitOp::MoveToVar { var_index: *acc_index, src: result });
                    }
                } else if is_last_pattern {
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
                        this.emit(JitOp::MoveToVar { var_index: *acc_index, src: acc });
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
                        this.emit(JitOp::MoveToVar { var_index: *acc_index, src: result });
                    });
                } else if let Some(gen) = acc_push_collect {
                    // Optimized: TakeVar + ArrPush for `. + [gen]` — in-place push, no temp array
                    let gen = gen.clone();
                    self.flatten_gen_with_each_output(source, input_slot, &|this, elem| {
                        this.emit(JitOp::SetVar { var_index: *var_index, src: elem });
                        let acc = this.alloc_slot();
                        this.emit(JitOp::TakeVar { dst: acc, var_index: *acc_index });
                        // Evaluate generator with input = accumulator, push each output
                        this.flatten_gen_with_each_output(&gen, acc, &|this2, val| {
                            this2.emit(JitOp::ArrPush { arr: acc, val });
                            this2.emit(JitOp::Drop { slot: val });
                        });
                        this.emit(JitOp::MoveToVar { var_index: *acc_index, src: acc });
                    });
                } else if let Some(info) = inplace_info {
                    // Optimized: TakeVar + PathExtract/PathInsert avoids container cloning
                    self.flatten_gen_with_each_output(source, input_slot, &|this, elem| {
                        this.emit(JitOp::SetVar { var_index: *var_index, src: elem });
                        this.emit_reduce_update_with_lets(*acc_index, &info);
                    });
                } else if let Some(info) = detect_inplace_assign(update) {
                    // Optimized: TakeVar + PathInsert for .[path] = val
                    self.flatten_gen_with_each_output(source, input_slot, &|this, elem| {
                        this.emit(JitOp::SetVar { var_index: *var_index, src: elem });
                        this.emit_reduce_assign(*acc_index, &info);
                    });
                } else if let Expr::SetPath { path, value } = update.as_ref() {
                    // Optimized: TakeVar + SetPathMut for setpath(path; val) in reduce
                    if is_scalar(path) && is_scalar(value) {
                        self.flatten_gen_with_each_output(source, input_slot, &|this, elem| {
                            this.emit(JitOp::SetVar { var_index: *var_index, src: elem });
                            let acc = this.alloc_slot();
                            this.emit(JitOp::TakeVar { dst: acc, var_index: *acc_index });
                            let path_val = this.flatten_scalar(path, acc);
                            let val = this.flatten_scalar(value, acc);
                            this.emit(JitOp::SetPathMut { container: acc, path: path_val, val });
                            this.emit(JitOp::Drop { slot: val });
                            this.emit(JitOp::Drop { slot: path_val });
                            this.emit(JitOp::MoveToVar { var_index: *acc_index, src: acc });
                        });
                    } else {
                        // Fall through to generic reduce
                        self.flatten_gen_with_each_output(source, input_slot, &|this, elem| {
                            this.emit(JitOp::SetVar { var_index: *var_index, src: elem });
                            let acc = this.alloc_slot();
                            this.emit(JitOp::GetVar { dst: acc, var_index: *acc_index });
                            let new_acc = this.flatten_scalar(update, acc);
                            this.emit(JitOp::MoveToVar { var_index: *acc_index, src: new_acc });
                            this.emit(JitOp::Drop { slot: acc });
                        });
                    }
                } else {
                    // For each output of source, set $var and update acc
                    self.flatten_gen_with_each_output(source, input_slot, &|this, elem| {
                        this.emit(JitOp::SetVar { var_index: *var_index, src: elem });
                        let acc = this.alloc_slot();
                        this.emit(JitOp::GetVar { dst: acc, var_index: *acc_index });
                        let new_acc = this.flatten_scalar(update, acc);
                        this.emit(JitOp::MoveToVar { var_index: *acc_index, src: new_acc });
                        this.emit(JitOp::Drop { slot: acc });
                    });
                }
                let out = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: out, var_index: *acc_index });
                self.emit(JitOp::MoveToVar { var_index: *acc_index, src: old_acc });
                self.emit(JitOp::MoveToVar { var_index: *var_index, src: old_var });
                out
            }
            Expr::Collect { generator } => {
                // Constant fold: if the entire Collect is pure literals, eval at compile time
                if let Some(val) = try_const_eval(expr) {
                    let ptr = self.hoist_value(val);
                    let out = self.alloc_slot();
                    self.emit(JitOp::LoadConst { dst: out, const_ptr: ptr });
                    return out;
                }
                // Fused [range(n)]: single call creates the array directly
                if let Expr::Range { from, to, step } = generator.as_ref() {
                    let is_from_zero = matches!(from.as_ref(),
                        Expr::Literal(Literal::Num(n, _)) if *n == 0.0);
                    if is_from_zero && step.is_none() {
                        let n_slot = self.flatten_scalar(to, input_slot);
                        let out = self.alloc_slot();
                        self.emit(JitOp::CollectRange { dst: out, n: n_slot });
                        self.emit(JitOp::Drop { slot: n_slot });
                        return out;
                    }
                }
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
                    // Optimization: type == "TYPE" → direct tag comparison
                    let used_fused = used_fused || if let Expr::BinOp { op: bop @ (BinOp::Eq | BinOp::Ne), lhs, rhs } = cond.as_ref() {
                        let type_name = extract_type_cmp(lhs, rhs).or_else(|| extract_type_cmp(rhs, lhs));
                        if let Some(tname) = type_name {
                            let tags = match tname {
                                "null" => Some(1u8 << 0),
                                "boolean" => Some((1u8 << 1) | (1u8 << 2)), // False | True
                                "number" => Some(1u8 << 3),
                                "string" => Some(1u8 << 4),
                                "array" => Some(1u8 << 5),
                                "object" => Some(1u8 << 6),
                                _ => None,
                            };
                            if let Some(tags) = tags {
                                let (tl, el) = if matches!(bop, BinOp::Ne) {
                                    (else_lbl, then_lbl) // swap for !=
                                } else {
                                    (then_lbl, else_lbl)
                                };
                                self.emit(JitOp::TypeCmpBranch { src: input_slot, tags, then_label: tl, else_label: el });
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
                    self.emit(JitOp::MoveToVar { var_index: *var_index, src: val });
                    let ok = self.flatten_gen(body, input_slot);
                    self.emit(JitOp::MoveToVar { var_index: *var_index, src: old });
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
                self.emit(JitOp::MoveToVar { var_index: *acc_index, src: acc_val });
                let old_var = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: old_var, var_index: *var_index });
                if !self.flatten_gen_with_reduce(source, *var_index, *acc_index, update, input_slot) {
                    return false;
                }
                let out = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: out, var_index: *acc_index });
                self.emit(JitOp::MoveToVar { var_index: *acc_index, src: old_acc });
                self.emit(JitOp::MoveToVar { var_index: *var_index, src: old_var });
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
                // repeat(f) = def _repeat: f, _repeat; _repeat;
                // Each iteration applies f to the SAME input (comma semantics),
                // not chaining outputs.
                let head = self.alloc_label();
                self.emit(JitOp::Label { id: head });
                let result = self.flatten_scalar(update, input_slot);
                self.emit_yield(result);
                self.emit(JitOp::Drop { slot: result });
                self.emit(JitOp::Jump { label: head });
                // Note: no "done" label — this is an infinite loop.
                // Early termination happens via Yield returning stop or limit.
                true
            }

            Expr::TryCatch { try_expr, catch_expr } => {
                self.flatten_gen_try_catch(try_expr, catch_expr, input_slot)
            }

            // RegexMatch/RegexCapture: 0-or-1 output generator
            // match("re") yields one object or empty (non-match throws error → empty)
            Expr::RegexMatch { input_expr, re, flags } | Expr::RegexCapture { input_expr, re, flags } => {
                let is_capture = matches!(expr, Expr::RegexCapture { .. });
                if !is_scalar(input_expr) || !is_scalar(re) || !is_scalar(flags) { return false; }
                let inp = self.flatten_scalar(input_expr, input_slot);
                let re_val = self.flatten_scalar(re, input_slot);
                let flags_val = self.flatten_scalar(flags, input_slot);
                let out = self.alloc_slot();
                let builtin_name = if is_capture { "capture" } else { "match" };
                // Wrap in try-catch: non-match throws error → skip yield
                let catch_label = self.alloc_label();
                let done_label = self.alloc_label();
                let error_slot = self.alloc_slot();
                let old_target = self.try_catch_target;
                self.try_catch_target = Some((catch_label, error_slot));
                self.try_depth += 1;
                self.emit(JitOp::TryCatchBegin);
                self.emit(JitOp::CallBuiltin { dst: out, name: builtin_name.to_string(), args: vec![inp, re_val, flags_val] });
                self.emit_yield(out);
                self.emit(JitOp::Drop { slot: out });
                self.emit(JitOp::TryCatchEnd);
                self.try_depth -= 1;
                self.try_catch_target = old_target;
                self.emit(JitOp::Jump { label: done_label });
                // Catch: non-match → empty (just drop error)
                self.emit(JitOp::Label { id: catch_label });
                self.emit(JitOp::TryCatchEnd);
                self.emit(JitOp::Drop { slot: error_slot });
                self.emit(JitOp::Label { id: done_label });
                self.emit(JitOp::Drop { slot: inp });
                self.emit(JitOp::Drop { slot: re_val });
                self.emit(JitOp::Drop { slot: flags_val });
                true
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
                self.emit(JitOp::MoveToVar { var_index: *acc_index, src: acc_val });
                let old_var = self.alloc_slot();
                self.emit(JitOp::GetVar { dst: old_var, var_index: *var_index });

                // The action for each source element: update acc, yield extract
                let vi = *var_index;
                let ai = *acc_index;
                let ok = self.flatten_gen_with_foreach(source, vi, ai, update, extract.as_deref(), input_slot);

                // Restore vars
                self.emit(JitOp::MoveToVar { var_index: *var_index, src: old_var });
                self.emit(JitOp::MoveToVar { var_index: *acc_index, src: old_acc });
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
                    // O(n) in-place update: iterate by integer index, TakeByIdx/PutByIdx
                    // No hash lookups — direct Vec index access for O(1) per element
                    let result = self.alloc_slot();
                    self.emit(JitOp::Clone { dst: result, src: input_slot });

                    // map_values / .[]= over a non-iterable used to fall through
                    // the loop silently and return the original value (#194).
                    // Branch on the kind first; the "other" arm raises jq's
                    // "Cannot iterate over X (Y)" error (skipped for `.[]?`).
                    let kind_var = self.alloc_var();
                    let iter_lbl = self.alloc_label();
                    let other_lbl = self.alloc_label();
                    self.emit(JitOp::GetKind { dst_var: kind_var, src: result });
                    self.emit(JitOp::BranchKind {
                        kind_var,
                        arr_label: iter_lbl,
                        obj_label: iter_lbl,
                        other_label: other_lbl,
                    });
                    let final_lbl = self.alloc_label();
                    self.emit(JitOp::Label { id: other_lbl });
                    if !is_each_opt {
                        let err_slot = self.alloc_slot();
                        self.emit(JitOp::CallBuiltin {
                            dst: err_slot,
                            name: "__each_error__".to_string(),
                            args: vec![result],
                        });
                        self.emit(JitOp::Drop { slot: err_slot });
                        if let Some((catch_label, error_slot)) = self.try_catch_target {
                            self.emit(JitOp::CheckError { error_dst: error_slot, catch_label });
                        } else {
                            self.emit(JitOp::ReturnError);
                        }
                    } else {
                        // `(.[]?) |= ...` on a non-iterable yields the input
                        // once. Skip the iteration and the post-loop yield.
                        self.emit_yield(result);
                        self.emit(JitOp::Jump { label: final_lbl });
                    }
                    self.emit(JitOp::Label { id: iter_lbl });

                    let idx = self.alloc_var();
                    let len = self.alloc_var();
                    let head = self.alloc_label();
                    let body = self.alloc_label();
                    let done = self.alloc_label();

                    self.emit(JitOp::GetLen { dst_var: len, src: result });
                    self.emit(JitOp::InitVar { var: idx });
                    self.emit(JitOp::Label { id: head });
                    self.emit(JitOp::LoopCheck { idx_var: idx, len_var: len, body_label: body, done_label: done });
                    self.emit(JitOp::Label { id: body });
                    // Take value at index (O(1), replaces with Null)
                    let old_val = self.alloc_slot();
                    self.emit(JitOp::TakeByIdx { dst: old_val, container: result, idx_var: idx });
                    // Apply update function
                    let new_val = self.flatten_scalar(update_expr, old_val);
                    self.emit(JitOp::Drop { slot: old_val });
                    // Put value back at index (O(1))
                    self.emit(JitOp::PutByIdx { container: result, idx_var: idx, val: new_val });
                    self.emit(JitOp::Drop { slot: new_val });
                    self.emit(JitOp::IncVar { var: idx });
                    self.emit(JitOp::Jump { label: head });
                    self.emit(JitOp::Label { id: done });

                    self.emit_yield(result);
                    self.emit(JitOp::Label { id: final_lbl });
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
                    // For the first output of update_expr applied to old_val.
                    // jq's `path |= F` takes ONLY the first generator value;
                    // broadcasting was issue #208. Use a flag var to drop
                    // subsequent yields silently (we can't break out of the
                    // closure cleanly mid-iteration, but skipping is enough —
                    // `gen` is bounded for typical updates and the cost of
                    // generating extra values is just discarded clones).
                    let first_var = self.alloc_var();
                    self.emit(JitOp::InitVar { var: first_var });
                    let ok = self.flatten_gen_with_each_output(update_expr, old_val, &|s, new_val| {
                        let skip_lbl = s.alloc_label();
                        let take_lbl = s.alloc_label();
                        let after_lbl = s.alloc_label();
                        s.emit(JitOp::BranchOnVar { var: first_var, nonzero_label: skip_lbl, zero_label: take_lbl });
                        s.emit(JitOp::Label { id: skip_lbl });
                        s.emit(JitOp::Drop { slot: new_val });
                        s.emit(JitOp::Jump { label: after_lbl });
                        s.emit(JitOp::Label { id: take_lbl });
                        s.emit(JitOp::IncVar { var: first_var });
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
                        s.emit(JitOp::Label { id: after_lbl });
                    });
                    self.emit(JitOp::Drop { slot: old_val });
                    if ok {
                        // Empty closure → del(path) per jq's `path |= F`
                        // semantics. Without this, `.a |= (.[]?)` on a scalar
                        // `.a` value silently produced zero outputs because
                        // the loop body above never ran. The literal-empty
                        // case is rewritten to `del(path)` upstream by
                        // simplify_expr (#155), but any runtime-empty
                        // generator hits this path. See #552.
                        let del_lbl = self.alloc_label();
                        let end_lbl = self.alloc_label();
                        self.emit(JitOp::BranchOnVar { var: first_var, nonzero_label: end_lbl, zero_label: del_lbl });
                        self.emit(JitOp::Label { id: del_lbl });
                        let path_for_del = self.alloc_slot();
                        self.emit(JitOp::Clone { dst: path_for_del, src: path_arr });
                        self.emit(JitOp::CollectBegin);
                        self.emit(JitOp::CollectPush { src: path_for_del });
                        let paths_arr = self.alloc_slot();
                        self.emit(JitOp::CollectFinish { dst: paths_arr });
                        let inp_for_del = self.alloc_slot();
                        self.emit(JitOp::Clone { dst: inp_for_del, src: input_slot });
                        let del_out = self.alloc_slot();
                        self.emit_propagating(JitOp::CallBuiltin { dst: del_out, name: "delpaths".to_string(), args: vec![inp_for_del, paths_arr] });
                        self.emit(JitOp::Drop { slot: inp_for_del });
                        self.emit(JitOp::Drop { slot: paths_arr });
                        self.emit_yield(del_out);
                        self.emit(JitOp::Drop { slot: del_out });
                        self.emit(JitOp::Label { id: end_lbl });
                    }
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
                // Only the descent-only 0-arg form (`recurse` / `..`) is
                // safe for the recurse_collect fast path — the parser
                // shapes that as `Recurse { EachOpt(Input) }`. Custom
                // step expressions like `recurse(.)` (infinite) or
                // `recurse(f)` need the eval-side generic loop. See #497.
                let is_default_descent = matches!(
                    input_expr.as_ref(),
                    Expr::EachOpt { input_expr: inner } if matches!(**inner, Expr::Input)
                );
                if !is_default_descent { return false; }
                // Seed is the pipeline input (jq's `def recurse: ., (.[]? | recurse);`).
                let val = self.flatten_scalar(&Expr::Input, input_slot);
                let arr = self.alloc_slot();
                self.emit(JitOp::CallBuiltin { dst: arr, name: "recurse_collect".to_string(), args: vec![val] });
                self.emit(JitOp::Drop { slot: val });
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
                // path(a, b, ...) — handle Comma by recursively emitting each branch as PathExpr
                if let Expr::Comma { left, right } = path_expr.as_ref() {
                    let l_ok = self.flatten_gen(&Expr::PathExpr { expr: left.clone() }, input_slot);
                    if !l_ok { return false; }
                    return self.flatten_gen(&Expr::PathExpr { expr: right.clone() }, input_slot);
                }
                // Native path(recurse) — bypass eval engine, generate paths
                // directly. Only the descent-only 0-arg form qualifies; the
                // parser shapes that as `Recurse { EachOpt(Input) }`. See #497.
                if let Expr::Recurse { input_expr } = path_expr.as_ref() {
                    let is_default_descent = matches!(
                        input_expr.as_ref(),
                        Expr::EachOpt { input_expr: inner } if matches!(**inner, Expr::Input)
                    );
                    if is_default_descent {
                        let inp = self.alloc_slot();
                        self.emit(JitOp::Clone { dst: inp, src: input_slot });
                        let arr = self.alloc_slot();
                        self.emit(JitOp::CallBuiltin { dst: arr, name: "paths_collect_all".to_string(), args: vec![inp] });
                        self.emit(JitOp::Drop { slot: inp });
                        self.flatten_each_with_action(arr, false, &|s, elem| {
                            s.emit_yield(elem);
                        });
                        self.emit(JitOp::Drop { slot: arr });
                        return true;
                    }
                }
                // Native paths(f) — path(recurse | if f then . else empty end)
                // DFS with filter applied at each node. The descent shape
                // is `Recurse { EachOpt(Input) }` after #497.
                if let Expr::Pipe { left, right } = path_expr.as_ref() {
                    if let Expr::Recurse { input_expr } = left.as_ref() {
                        if matches!(
                            input_expr.as_ref(),
                            Expr::EachOpt { input_expr: inner } if matches!(**inner, Expr::Input)
                        ) {
                            if let Expr::IfThenElse { cond, then_branch, else_branch } = right.as_ref() {
                                if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                                    // This is paths(f) where f = cond
                                    let idx = self.closure_ops.len();
                                    self.closure_ops.push((**cond).clone());
                                    let inp = self.alloc_slot();
                                    self.emit(JitOp::Clone { dst: inp, src: input_slot });
                                    let arr = self.alloc_slot();
                                    self.emit(JitOp::CallBuiltin { dst: arr, name: format!("__paths_filtered__:{}", idx), args: vec![inp] });
                                    self.emit(JitOp::Drop { slot: inp });
                                    self.flatten_each_with_action(arr, false, &|s, elem| {
                                        s.emit_yield(elem);
                                    });
                                    self.emit(JitOp::Drop { slot: arr });
                                    return true;
                                }
                            }
                        }
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
                self.emit(JitOp::MoveToVar { var_index: *var_index, src: val });
                let ok = self.flatten_gen_with_each_output(body, input_slot, action);
                self.emit(JitOp::MoveToVar { var_index: *var_index, src: old });
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
                self.emit(JitOp::MoveToVar { var_index: *var_index, src: val });
                // Now compile body | right
                let ok = if is_scalar(body) {
                    let mid = self.flatten_scalar(body, input_slot);
                    let r = self.flatten_gen(right, mid);
                    self.emit(JitOp::Drop { slot: mid });
                    r
                } else {
                    self.flatten_gen_pipe(body, right, input_slot)
                };
                self.emit(JitOp::MoveToVar { var_index: *var_index, src: old });
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
        if !is_opt {
            // Set "Cannot iterate over TYPE (VALUE)" before returning the error;
            // bare ReturnError leaves the error message blank (issue #107).
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
        true
    }

    /// Emit reduce loop inline. Returns false if the source generator can't be compiled.
    fn flatten_gen_with_reduce(&mut self, source: &Expr, var_index: u16, acc_index: u16, update: &Expr, input_slot: SlotId) -> bool {
        // Detect in-place update pattern: path |= f or path += f (wrapped in LetBinding)
        let inplace_info = detect_inplace_update(update);

        // Detect in-place assign pattern: path = val (e.g. .[$key] = $x in reduce)
        let assign_info = detect_inplace_assign(update);

        // Detect last(g) pattern: reduce g as $x ([]; [$x])
        let is_last_pattern = matches!(update,
            Expr::Collect { generator } if matches!(generator.as_ref(),
                Expr::LoadVar { var_index: vi } if *vi == var_index
            )
        );

        // Detect accumulator-add pattern: reduce ... (init; . + rhs)
        // Also detect `+= rhs` which is: LetBinding { var, value: rhs, body: Update { path: ., update: . + LoadVar(var) } }
        let acc_add_rhs = if let Expr::BinOp { op, lhs, rhs } = update {
            if matches!(op, BinOp::Add) && matches!(lhs.as_ref(), Expr::Input) && is_scalar(rhs) {
                Some(rhs.as_ref())
            } else { None }
        } else if let Expr::LetBinding { var_index: rhs_var, value: rhs_value, body } = update {
            if let Expr::Update { path_expr, update_expr } = body.as_ref() {
                if matches!(path_expr.as_ref(), Expr::Input) {
                    if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = update_expr.as_ref() {
                        if matches!(lhs.as_ref(), Expr::Input)
                            && matches!(rhs.as_ref(), Expr::LoadVar { var_index: v } if *v == *rhs_var)
                            && is_scalar(rhs_value)
                        {
                            Some(rhs_value.as_ref())
                        } else { None }
                    } else { None }
                } else { None }
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
                s.emit(JitOp::MoveToVar { var_index: acc_index, src: acc });
            } else if let Some(rhs_expr) = acc_add_rhs {
                // Optimized: TakeVar + AddMove for `. + rhs`
                let acc = s.alloc_slot();
                s.emit(JitOp::TakeVar { dst: acc, var_index: acc_index });
                let rhs_val = s.flatten_scalar(rhs_expr, acc);
                let result = s.alloc_slot();
                s.emit(JitOp::AddMove { dst: result, lhs: acc, rhs: rhs_val });
                s.emit(JitOp::Drop { slot: rhs_val });
                s.emit(JitOp::Drop { slot: acc });
                s.emit(JitOp::MoveToVar { var_index: acc_index, src: result });
            } else if let Some(ref info) = inplace_info {
                s.emit_reduce_update_with_lets(acc_index, info);
            } else if let Some(ref info) = assign_info {
                s.emit_reduce_assign(acc_index, info);
            } else {
                // Load current accumulator as the input to update (. refers to acc in reduce)
                let acc = s.alloc_slot();
                s.emit(JitOp::GetVar { dst: acc, var_index: acc_index });
                let new_acc = s.flatten_scalar(update, acc);
                s.emit(JitOp::Drop { slot: acc });
                s.emit(JitOp::MoveToVar { var_index: acc_index, src: new_acc });
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

    /// Check if an expression is a pure f64 function of acc (Input) and vars.
    /// Single-var convenience wrapper.
    fn is_pure_f64_expr(expr: &Expr, var_index: u16) -> bool {
        Self::is_pure_f64_expr_multi(expr, &[var_index])
    }

    /// Check with multiple allowed variable indices (for LetBinding support).
    fn is_pure_f64_expr_multi(expr: &Expr, allowed: &[u16]) -> bool {
        match expr {
            Expr::Input => true,
            Expr::LoadVar { var_index } if allowed.contains(var_index) => true,
            Expr::Literal(Literal::Num(..)) => true,
            Expr::BinOp { op, lhs, rhs } => {
                matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod
                    | BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge
                    | BinOp::And | BinOp::Or)
                && Self::is_pure_f64_expr_multi(lhs, allowed)
                && Self::is_pure_f64_expr_multi(rhs, allowed)
            }
            Expr::Negate { operand } => Self::is_pure_f64_expr_multi(operand, allowed),
            Expr::Not => true, // `not` on f64: 0.0 → 1.0, nonzero → 0.0
            Expr::Pipe { left, right } => {
                Self::is_pure_f64_expr_multi(left, allowed)
                && Self::is_pure_f64_expr_multi(right, allowed)
            }
            Expr::UnaryOp { op, operand } => {
                matches!(op, UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Sqrt
                    | UnaryOp::Fabs | UnaryOp::Round | UnaryOp::Trunc | UnaryOp::Abs
                    | UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Tan
                    | UnaryOp::Asin | UnaryOp::Acos | UnaryOp::Atan
                    | UnaryOp::Exp | UnaryOp::Exp2 | UnaryOp::Log | UnaryOp::Log2 | UnaryOp::Log10
                    | UnaryOp::Cbrt)
                && Self::is_pure_f64_expr_multi(operand, allowed)
            }
            Expr::IfThenElse { cond, then_branch, else_branch } => {
                Self::is_pure_f64_expr_multi(cond, allowed)
                && Self::is_pure_f64_expr_multi(then_branch, allowed)
                && Self::is_pure_f64_expr_multi(else_branch, allowed)
            }
            Expr::LetBinding { var_index, value, body } => {
                Self::is_pure_f64_expr_multi(value, allowed) && {
                    let mut ext = allowed.to_vec();
                    ext.push(*var_index);
                    Self::is_pure_f64_expr_multi(body, &ext)
                }
            }
            _ => false,
        }
    }

    /// Compile a pure f64 expression into Cranelift f64 variables.
    /// acc_var = accumulator (Input). var_map maps IR var indices to f64 vars.
    /// Returns the f64 variable holding the result.
    fn compile_f64_expr(&mut self, expr: &Expr, _unused: u16, acc_var: u32, x_var: u32) -> u32 {
        self.compile_f64_expr_multi(expr, acc_var, &[(_unused, x_var)])
    }

    fn compile_f64_expr_multi(&mut self, expr: &Expr, acc_var: u32, var_map: &[(u16, u32)]) -> u32 {
        match expr {
            Expr::Input => acc_var,
            Expr::LoadVar { var_index } => {
                for &(vi, fv) in var_map {
                    if vi == *var_index { return fv; }
                }
                unreachable!("is_pure_f64_expr should have rejected unknown var");
            }
            Expr::Literal(Literal::Num(n, _)) => {
                let v = self.alloc_var();
                self.emit(JitOp::F64Const { dst_var: v, val: *n });
                v
            }
            Expr::BinOp { op, lhs, rhs } => {
                let a = self.compile_f64_expr_multi(lhs, acc_var, var_map);
                let b = self.compile_f64_expr_multi(rhs, acc_var, var_map);
                let dst = self.alloc_var();
                match op {
                    BinOp::Add => self.emit(JitOp::F64Add { dst_var: dst, a_var: a, b_var: b }),
                    BinOp::Sub => self.emit(JitOp::F64Sub { dst_var: dst, a_var: a, b_var: b }),
                    BinOp::Mul => self.emit(JitOp::F64Mul { dst_var: dst, a_var: a, b_var: b }),
                    BinOp::Div => self.emit(JitOp::F64Div { dst_var: dst, a_var: a, b_var: b }),
                    BinOp::Mod => self.emit(JitOp::F64Rem { dst_var: dst, a_var: a, b_var: b }),
                    BinOp::Eq => self.emit(JitOp::F64Cmp { dst_var: dst, a_var: a, b_var: b, cc: 4 }),
                    BinOp::Ne => self.emit(JitOp::F64Cmp { dst_var: dst, a_var: a, b_var: b, cc: 5 }),
                    BinOp::Lt => self.emit(JitOp::F64Cmp { dst_var: dst, a_var: a, b_var: b, cc: 3 }),
                    BinOp::Gt => self.emit(JitOp::F64Cmp { dst_var: dst, a_var: a, b_var: b, cc: 1 }),
                    BinOp::Le => self.emit(JitOp::F64Cmp { dst_var: dst, a_var: a, b_var: b, cc: 2 }),
                    BinOp::Ge => self.emit(JitOp::F64Cmp { dst_var: dst, a_var: a, b_var: b, cc: 0 }),
                    BinOp::And => {
                        // and: both nonzero (truthiness in f64 context: 0.0 and false=0 are falsy)
                        // For f64: a != 0 && b != 0 → emit as: if a == 0 then 0 else b
                        let zero = self.alloc_var();
                        self.emit(JitOp::F64Const { dst_var: zero, val: 0.0 });
                        let a_is_zero = self.alloc_var();
                        self.emit(JitOp::F64Cmp { dst_var: a_is_zero, a_var: a, b_var: zero, cc: 4 });
                        // Use BranchOnVar to select
                        let then_l = self.alloc_label();
                        let else_l = self.alloc_label();
                        let done_l = self.alloc_label();
                        self.emit(JitOp::BranchOnVar { var: a_is_zero, nonzero_label: then_l, zero_label: else_l });
                        self.emit(JitOp::Label { id: then_l });
                        // a is zero (falsy) → result is false
                        self.emit(JitOp::F64Const { dst_var: dst, val: 0.0 });
                        self.emit(JitOp::Jump { label: done_l });
                        self.emit(JitOp::Label { id: else_l });
                        // a is truthy → result is b's truthiness
                        // jq 'and' actually returns the rhs value if lhs is truthy
                        // But we just need truthiness: b
                        let b_val = self.alloc_var();
                        self.emit(JitOp::F64Cmp { dst_var: b_val, a_var: b, b_var: zero, cc: 5 }); // b != 0 → 1, else 0
                        self.emit(JitOp::F64Const { dst_var: dst, val: 0.0 }); // placeholder, will be overwritten
                        // `a and b` = if a then b else false end → b_val (0 or 1)
                        self.emit(JitOp::F64Move { dst_var: dst, src_var: b_val });
                        self.emit(JitOp::Label { id: done_l });
                    }
                    BinOp::Or => {
                        let zero = self.alloc_var();
                        self.emit(JitOp::F64Const { dst_var: zero, val: 0.0 });
                        let a_nz = self.alloc_var();
                        self.emit(JitOp::F64Cmp { dst_var: a_nz, a_var: a, b_var: zero, cc: 5 }); // a != 0
                        let then_l = self.alloc_label();
                        let else_l = self.alloc_label();
                        let done_l = self.alloc_label();
                        self.emit(JitOp::BranchOnVar { var: a_nz, nonzero_label: then_l, zero_label: else_l });
                        self.emit(JitOp::Label { id: then_l });
                        // a is truthy → result is true (1.0)
                        self.emit(JitOp::F64Const { dst_var: dst, val: 1.0 });
                        self.emit(JitOp::Jump { label: done_l });
                        self.emit(JitOp::Label { id: else_l });
                        // a is falsy → result is b's truthiness
                        let b_nz = self.alloc_var();
                        self.emit(JitOp::F64Cmp { dst_var: b_nz, a_var: b, b_var: zero, cc: 5 });
                        self.emit(JitOp::F64Move { dst_var: dst, src_var: b_nz });
                        self.emit(JitOp::Label { id: done_l });
                    }
                };
                dst
            }
            Expr::IfThenElse { cond, then_branch, else_branch } => {
                let c = self.compile_f64_expr_multi(cond, acc_var, var_map);
                let dst = self.alloc_var();
                let then_l = self.alloc_label();
                let else_l = self.alloc_label();
                let done_l = self.alloc_label();
                self.emit(JitOp::BranchOnVar { var: c, nonzero_label: then_l, zero_label: else_l });
                self.emit(JitOp::Label { id: then_l });
                let t = self.compile_f64_expr_multi(then_branch, acc_var, var_map);
                self.emit(JitOp::F64Move { dst_var: dst, src_var: t });
                self.emit(JitOp::Jump { label: done_l });
                self.emit(JitOp::Label { id: else_l });
                let e = self.compile_f64_expr_multi(else_branch, acc_var, var_map);
                self.emit(JitOp::F64Move { dst_var: dst, src_var: e });
                self.emit(JitOp::Label { id: done_l });
                dst
            }
            Expr::Negate { operand } => {
                let src = self.compile_f64_expr_multi(operand, acc_var, var_map);
                let dst = self.alloc_var();
                self.emit(JitOp::F64Neg { dst_var: dst, src_var: src });
                dst
            }
            Expr::Not => {
                // `not` on f64: 0.0 → 1.0, nonzero → 0.0
                let zero = self.alloc_var();
                self.emit(JitOp::F64Const { dst_var: zero, val: 0.0 });
                let dst = self.alloc_var();
                self.emit(JitOp::F64Cmp { dst_var: dst, a_var: acc_var, b_var: zero, cc: 4 }); // acc == 0 → 1, else 0
                dst
            }
            Expr::Pipe { left, right } => {
                // Compile left, then use its result as Input for right
                let left_result = self.compile_f64_expr_multi(left, acc_var, var_map);
                self.compile_f64_expr_multi(right, left_result, var_map)
            }
            Expr::LetBinding { var_index, value, body } => {
                let val = self.compile_f64_expr_multi(value, acc_var, var_map);
                let mut ext = var_map.to_vec();
                ext.push((*var_index, val));
                self.compile_f64_expr_multi(body, acc_var, &ext)
            }
            Expr::UnaryOp { op, operand } => {
                let src = self.compile_f64_expr_multi(operand, acc_var, var_map);
                let dst = self.alloc_var();
                match op {
                    UnaryOp::Floor => self.emit(JitOp::F64Math { dst_var: dst, src_var: src, kind: 0 }),
                    UnaryOp::Ceil => self.emit(JitOp::F64Math { dst_var: dst, src_var: src, kind: 1 }),
                    UnaryOp::Sqrt => self.emit(JitOp::F64Math { dst_var: dst, src_var: src, kind: 2 }),
                    UnaryOp::Fabs | UnaryOp::Abs => self.emit(JitOp::F64Math { dst_var: dst, src_var: src, kind: 3 }),
                    UnaryOp::Trunc => self.emit(JitOp::F64Math { dst_var: dst, src_var: src, kind: 4 }),
                    UnaryOp::Round => self.emit(JitOp::F64Math { dst_var: dst, src_var: src, kind: 5 }),
                    UnaryOp::Sin => self.emit(JitOp::F64Libm { dst_var: dst, src_var: src, func: 0 }),
                    UnaryOp::Cos => self.emit(JitOp::F64Libm { dst_var: dst, src_var: src, func: 1 }),
                    UnaryOp::Tan => self.emit(JitOp::F64Libm { dst_var: dst, src_var: src, func: 2 }),
                    UnaryOp::Asin => self.emit(JitOp::F64Libm { dst_var: dst, src_var: src, func: 3 }),
                    UnaryOp::Acos => self.emit(JitOp::F64Libm { dst_var: dst, src_var: src, func: 4 }),
                    UnaryOp::Atan => self.emit(JitOp::F64Libm { dst_var: dst, src_var: src, func: 5 }),
                    UnaryOp::Exp => self.emit(JitOp::F64Libm { dst_var: dst, src_var: src, func: 6 }),
                    UnaryOp::Exp2 => self.emit(JitOp::F64Libm { dst_var: dst, src_var: src, func: 7 }),
                    UnaryOp::Log => self.emit(JitOp::F64Libm { dst_var: dst, src_var: src, func: 8 }),
                    UnaryOp::Log2 => self.emit(JitOp::F64Libm { dst_var: dst, src_var: src, func: 9 }),
                    UnaryOp::Log10 => self.emit(JitOp::F64Libm { dst_var: dst, src_var: src, func: 10 }),
                    UnaryOp::Cbrt => self.emit(JitOp::F64Libm { dst_var: dst, src_var: src, func: 11 }),
                    _ => unreachable!(),
                };
                dst
            }
            _ => unreachable!("is_pure_f64_expr should have rejected this"),
        }
    }


    /// Collect all LoadVar var_indices referenced anywhere in an expression.
    ///
    /// Must be exhaustive across every `Expr` variant that can contain sub-expressions:
    /// the JIT→eval closure-op dispatchers rely on this to seed delegated `eval::Env`
    /// from the JIT env (see `new_delegated_env` / `reset_delegated_env`). A missing
    /// variant here silently loses any `$var` buried inside and degrades the
    /// delegated call to `null`.
    fn collect_loadvar_indices(expr: &Expr, out: &mut Vec<u16>) {
        use crate::ir::StringPart;
        match expr {
            Expr::LoadVar { var_index } => {
                if !out.contains(var_index) { out.push(*var_index); }
            }
            Expr::Input | Expr::Empty | Expr::Literal(_) | Expr::Not
            | Expr::Loc { .. } | Expr::Env | Expr::Builtins
            | Expr::ReadInput | Expr::ReadInputs
            | Expr::ModuleMeta | Expr::GenLabel => {}
            Expr::BinOp { lhs, rhs, .. } => {
                Self::collect_loadvar_indices(lhs, out);
                Self::collect_loadvar_indices(rhs, out);
            }
            Expr::UnaryOp { operand, .. } | Expr::Negate { operand } => {
                Self::collect_loadvar_indices(operand, out);
            }
            Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => {
                Self::collect_loadvar_indices(expr, out);
                Self::collect_loadvar_indices(key, out);
            }
            Expr::Pipe { left, right } | Expr::Comma { left, right } => {
                Self::collect_loadvar_indices(left, out);
                Self::collect_loadvar_indices(right, out);
            }
            Expr::IfThenElse { cond, then_branch, else_branch } => {
                Self::collect_loadvar_indices(cond, out);
                Self::collect_loadvar_indices(then_branch, out);
                Self::collect_loadvar_indices(else_branch, out);
            }
            Expr::TryCatch { try_expr, catch_expr } => {
                Self::collect_loadvar_indices(try_expr, out);
                Self::collect_loadvar_indices(catch_expr, out);
            }
            Expr::Each { input_expr } | Expr::EachOpt { input_expr }
            | Expr::Recurse { input_expr } => {
                Self::collect_loadvar_indices(input_expr, out);
            }
            Expr::LetBinding { value, body, .. } => {
                Self::collect_loadvar_indices(value, out);
                Self::collect_loadvar_indices(body, out);
            }
            Expr::Reduce { source, init, update, .. } => {
                Self::collect_loadvar_indices(source, out);
                Self::collect_loadvar_indices(init, out);
                Self::collect_loadvar_indices(update, out);
            }
            Expr::Foreach { source, init, update, extract, .. } => {
                Self::collect_loadvar_indices(source, out);
                Self::collect_loadvar_indices(init, out);
                Self::collect_loadvar_indices(update, out);
                if let Some(e) = extract { Self::collect_loadvar_indices(e, out); }
            }
            Expr::Collect { generator } => Self::collect_loadvar_indices(generator, out),
            Expr::ObjectConstruct { pairs } => {
                for (k, v) in pairs {
                    Self::collect_loadvar_indices(k, out);
                    Self::collect_loadvar_indices(v, out);
                }
            }
            Expr::Alternative { primary, fallback } => {
                Self::collect_loadvar_indices(primary, out);
                Self::collect_loadvar_indices(fallback, out);
            }
            Expr::Range { from, to, step } => {
                Self::collect_loadvar_indices(from, out);
                Self::collect_loadvar_indices(to, out);
                if let Some(s) = step { Self::collect_loadvar_indices(s, out); }
            }
            Expr::Label { body, .. } => Self::collect_loadvar_indices(body, out),
            Expr::Break { value, .. } => Self::collect_loadvar_indices(value, out),
            Expr::Update { path_expr, update_expr } => {
                Self::collect_loadvar_indices(path_expr, out);
                Self::collect_loadvar_indices(update_expr, out);
            }
            Expr::Assign { path_expr, value_expr } => {
                Self::collect_loadvar_indices(path_expr, out);
                Self::collect_loadvar_indices(value_expr, out);
            }
            Expr::PathExpr { expr } => Self::collect_loadvar_indices(expr, out),
            Expr::SetPath { path, value } => {
                Self::collect_loadvar_indices(path, out);
                Self::collect_loadvar_indices(value, out);
            }
            Expr::GetPath { path } => Self::collect_loadvar_indices(path, out),
            Expr::DelPaths { paths } => Self::collect_loadvar_indices(paths, out),
            Expr::FuncCall { args, .. } => {
                for a in args { Self::collect_loadvar_indices(a, out); }
            }
            Expr::StringInterpolation { parts } => {
                for p in parts {
                    if let StringPart::Expr(e) = p { Self::collect_loadvar_indices(e, out); }
                }
            }
            Expr::Limit { count, generator } => {
                Self::collect_loadvar_indices(count, out);
                Self::collect_loadvar_indices(generator, out);
            }
            Expr::While { cond, update } | Expr::Until { cond, update } => {
                Self::collect_loadvar_indices(cond, out);
                Self::collect_loadvar_indices(update, out);
            }
            Expr::Repeat { update } => Self::collect_loadvar_indices(update, out),
            Expr::AllShort { generator, predicate }
            | Expr::AnyShort { generator, predicate } => {
                Self::collect_loadvar_indices(generator, out);
                Self::collect_loadvar_indices(predicate, out);
            }
            Expr::Error { msg } => {
                if let Some(m) = msg { Self::collect_loadvar_indices(m, out); }
            }
            Expr::Format { expr, .. } => Self::collect_loadvar_indices(expr, out),
            Expr::ClosureOp { input_expr, key_expr, .. } => {
                Self::collect_loadvar_indices(input_expr, out);
                Self::collect_loadvar_indices(key_expr, out);
            }
            Expr::RegexTest { input_expr, re, flags }
            | Expr::RegexMatch { input_expr, re, flags }
            | Expr::RegexCapture { input_expr, re, flags }
            | Expr::RegexScan { input_expr, re, flags } => {
                Self::collect_loadvar_indices(input_expr, out);
                Self::collect_loadvar_indices(re, out);
                Self::collect_loadvar_indices(flags, out);
            }
            Expr::RegexSub { input_expr, re, tostr, flags }
            | Expr::RegexGsub { input_expr, re, tostr, flags } => {
                Self::collect_loadvar_indices(input_expr, out);
                Self::collect_loadvar_indices(re, out);
                Self::collect_loadvar_indices(tostr, out);
                Self::collect_loadvar_indices(flags, out);
            }
            Expr::AlternativeDestructure { alternatives } => {
                for a in alternatives { Self::collect_loadvar_indices(a, out); }
            }
            Expr::Slice { expr, from, to } => {
                Self::collect_loadvar_indices(expr, out);
                if let Some(f) = from { Self::collect_loadvar_indices(f, out); }
                if let Some(t) = to { Self::collect_loadvar_indices(t, out); }
            }
            Expr::Debug { expr } | Expr::Stderr { expr } => {
                Self::collect_loadvar_indices(expr, out);
            }
            Expr::CallBuiltin { args, .. } => {
                for a in args { Self::collect_loadvar_indices(a, out); }
            }
        }
    }

    /// Classify simple `until(. cmp K; . op S)` for narrow fused path.
    fn classify_f64_until_narrow(cond: &Expr, update: &Expr) -> Option<(u8, f64, u8, f64)> {
        let (cond_cc, cond_const) = match cond {
            Expr::BinOp { op, lhs, rhs } if matches!(lhs.as_ref(), Expr::Input) => {
                if let Expr::Literal(Literal::Num(k, _)) = rhs.as_ref() {
                    let cc = match op {
                        BinOp::Ge => 0, BinOp::Gt => 1, BinOp::Le => 2, BinOp::Lt => 3,
                        BinOp::Eq => 4, BinOp::Ne => 5,
                        _ => return None,
                    };
                    (cc, *k)
                } else { return None; }
            }
            _ => return None,
        };
        let (update_op, update_const) = match update {
            Expr::BinOp { op, lhs, rhs } if matches!(lhs.as_ref(), Expr::Input) => {
                if let Expr::Literal(Literal::Num(s, _)) = rhs.as_ref() {
                    let uop = match op {
                        BinOp::Add => 0, BinOp::Sub => 1, BinOp::Mul => 2,
                        _ => return None,
                    };
                    (uop, *s)
                } else { return None; }
            }
            _ => return None,
        };
        Some((cond_cc, cond_const, update_op, update_const))
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
            s.emit(JitOp::MoveToVar { var_index: acc_index, src: new_acc });
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

                // Check if update is a pure f64 expression of . (acc) and $x (loop var)
                let is_f64_update = Self::is_pure_f64_expr(update, var_index);

                // Check if extract is fuseable as f64
                // None = yield acc, scalar f64 = yield result, Comma of f64 = yield both
                let is_f64_extract = is_f64_update && match extract {
                    None => true,
                    Some(ext) => {
                        Self::is_pure_f64_expr(ext, var_index)
                        || matches!(ext, Expr::Comma { left, right }
                            if Self::is_pure_f64_expr(left, var_index)
                                && Self::is_pure_f64_expr(right, var_index))
                    }
                };

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

                if is_f64_extract {
                    // Fused path: compile update as pure f64, yield extract result
                    let acc_f64 = self.alloc_var();
                    let init_slot = self.alloc_slot();
                    self.emit(JitOp::GetVar { dst: init_slot, var_index: acc_index });
                    self.emit(JitOp::ToF64Var { dst_var: acc_f64, src: init_slot });
                    self.emit(JitOp::Drop { slot: init_slot });
                    let head = self.alloc_label();
                    let body = self.alloc_label();
                    let done = self.alloc_label();
                    self.emit(JitOp::Label { id: head });
                    self.emit(JitOp::RangeCheck { dst_var: cmp, cur_var: cur, to_var: to_v, step_var: step_v });
                    self.emit(JitOp::BranchOnVar { var: cmp, nonzero_label: body, zero_label: done });
                    self.emit(JitOp::Label { id: body });
                    // Compile update as f64
                    let new_acc = self.compile_f64_expr(update, var_index, acc_f64, cur);
                    if new_acc != acc_f64 {
                        self.emit(JitOp::F64Move { dst_var: acc_f64, src_var: new_acc });
                    }
                    // Yield extract result(s)
                    match extract {
                        None => {
                            let val = self.alloc_slot();
                            self.emit(JitOp::F64Num { dst: val, src_var: acc_f64 });
                            self.emit_yield(val);
                            self.emit(JitOp::Drop { slot: val });
                        }
                        Some(Expr::Comma { left, right }) => {
                            let lv = self.compile_f64_expr(left, var_index, acc_f64, cur);
                            let l_val = self.alloc_slot();
                            self.emit(JitOp::F64Num { dst: l_val, src_var: lv });
                            self.emit_yield(l_val);
                            self.emit(JitOp::Drop { slot: l_val });
                            let rv = self.compile_f64_expr(right, var_index, acc_f64, cur);
                            let r_val = self.alloc_slot();
                            self.emit(JitOp::F64Num { dst: r_val, src_var: rv });
                            self.emit_yield(r_val);
                            self.emit(JitOp::Drop { slot: r_val });
                        }
                        Some(ext) => {
                            let ev = self.compile_f64_expr(ext, var_index, acc_f64, cur);
                            let val = self.alloc_slot();
                            self.emit(JitOp::F64Num { dst: val, src_var: ev });
                            self.emit_yield(val);
                            self.emit(JitOp::Drop { slot: val });
                        }
                    }
                    self.emit(JitOp::F64Add { dst_var: cur, a_var: cur, b_var: step_v });
                    self.emit(JitOp::Jump { label: head });
                    self.emit(JitOp::Label { id: done });
                    // Store final acc back to var store
                    let final_val = self.alloc_slot();
                    self.emit(JitOp::F64Num { dst: final_val, src_var: acc_f64 });
                    self.emit(JitOp::MoveToVar { var_index: acc_index, src: final_val });
                } else {
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
                }
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
                    self.emit(JitOp::MoveToVar { var_index, src: old });
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

        self.emit(JitOp::MoveToVar { var_index, src: old });
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
        if !is_opt {
            // Set "Cannot iterate over TYPE (VALUE)" before returning the error;
            // bare ReturnError leaves the error message blank (issue #107).
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
            self.emit(JitOp::MoveToVar { var_index: var_idx, src: val });
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

        self.emit(JitOp::MoveToVar { var_index: acc_index, src: current_val });

        // Restore let-binding vars
        for (var_idx, old) in saved_vars.into_iter().rev() {
            self.emit(JitOp::MoveToVar { var_index: var_idx, src: old });
        }
    }

    /// Emit in-place assign for reduce: .[path] = val with TakeVar for zero-copy.
    fn emit_reduce_assign(&mut self, acc_index: u16, info: &InplaceAssignInfo) {
        // Save and set let-binding vars
        let mut saved_vars = Vec::new();
        let acc_for_lets = self.alloc_slot();
        self.emit(JitOp::GetVar { dst: acc_for_lets, var_index: acc_index });
        for &(var_idx, ref value_expr) in &info.let_bindings {
            let old = self.alloc_slot();
            self.emit(JitOp::GetVar { dst: old, var_index: var_idx });
            let val = self.flatten_scalar(value_expr, acc_for_lets);
            self.emit(JitOp::MoveToVar { var_index: var_idx, src: val });
            saved_vars.push((var_idx, old));
        }

        // Evaluate value_expr while we still have acc in the var
        // (value_expr may reference . which is the accumulator)
        let assign_val = self.flatten_scalar(info.value_expr, acc_for_lets);
        self.emit(JitOp::Drop { slot: acc_for_lets });

        // TakeVar: move acc out (refcount = 1)
        let acc = self.alloc_slot();
        self.emit(JitOp::TakeVar { dst: acc, var_index: acc_index });

        // Build path keys and PathInsert chain
        let mut containers = Vec::new();
        let mut keys_vec = Vec::new();
        let mut current = acc;

        for component in &info.path_components[..info.path_components.len() - 1] {
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
            keys_vec.push(key);
            current = element;
        }

        // Last path component: direct PathInsert with the value
        let last_key = match info.path_components.last().unwrap() {
            PathComponent::Field(name) => {
                let s = self.alloc_slot();
                self.emit(JitOp::Str { dst: s, val: name.clone() });
                s
            }
            PathComponent::Expr(e) => {
                self.flatten_scalar(e, current)
            }
        };
        self.emit(JitOp::PathInsert { container: current, key: last_key, val: assign_val });
        self.emit(JitOp::Drop { slot: assign_val });
        self.emit(JitOp::Drop { slot: last_key });

        // PathInsert chain back up (reverse)
        for (container, key) in containers.iter().rev().zip(keys_vec.iter().rev()) {
            self.emit(JitOp::PathInsert { container: *container, key: *key, val: current });
            self.emit(JitOp::Drop { slot: current });
            self.emit(JitOp::Drop { slot: *key });
            current = *container;
        }

        self.emit(JitOp::MoveToVar { var_index: acc_index, src: current });

        // Restore let-binding vars
        for (var_idx, old) in saved_vars.into_iter().rev() {
            self.emit(JitOp::MoveToVar { var_index: var_idx, src: old });
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

/// Info for in-place reduce assign optimization (.[path] = val).
struct InplaceAssignInfo<'a> {
    /// LetBindings wrapping the Assign.
    let_bindings: Vec<(u16, &'a Expr)>,
    /// Path components for the assign target.
    path_components: Vec<PathComponent>,
    /// The value expression to assign.
    value_expr: &'a Expr,
}

/// Compile a filter expression into a fast `fn(&Value) -> bool` check for paths(f).
/// Returns None if the filter can't be compiled to a direct check (falls back to eval).
fn compile_type_filter(expr: &Expr) -> Option<fn(&Value) -> bool> {
    // type == "T"
    if let Expr::BinOp { op: crate::ir::BinOp::Eq, lhs, rhs } = expr {
        if let Some(type_name) = extract_type_cmp(lhs, rhs).or_else(|| extract_type_cmp(rhs, lhs)) {
            return match type_name {
                "number" => Some(|v| matches!(v, Value::Num(..))),
                "string" => Some(|v| matches!(v, Value::Str(..))),
                "array" => Some(|v| matches!(v, Value::Arr(..))),
                "object" => Some(|v| matches!(v, Value::Obj(..))),
                "null" => Some(|v| matches!(v, Value::Null)),
                "boolean" => Some(|v| matches!(v, Value::True | Value::False)),
                _ => None,
            };
        }
    }
    // type != "T"
    if let Expr::BinOp { op: crate::ir::BinOp::Ne, lhs, rhs } = expr {
        if let Some(type_name) = extract_type_cmp(lhs, rhs).or_else(|| extract_type_cmp(rhs, lhs)) {
            return match type_name {
                "number" => Some(|v| !matches!(v, Value::Num(..))),
                "string" => Some(|v| !matches!(v, Value::Str(..))),
                "array" => Some(|v| !matches!(v, Value::Arr(..))),
                "object" => Some(|v| !matches!(v, Value::Obj(..))),
                "null" => Some(|v| !matches!(v, Value::Null)),
                "boolean" => Some(|v| !matches!(v, Value::True | Value::False)),
                _ => None,
            };
        }
    }
    // scalars: type != "array" and type != "object"
    if let Expr::BinOp { op: crate::ir::BinOp::And, lhs, rhs } = expr {
        let lf = compile_type_filter(lhs)?;
        let rf = compile_type_filter(rhs)?;
        // Combine two function pointers. We need a static dispatch.
        // Use known combinations:
        return match (lf as usize, rf as usize) {
            _ => {
                // Can't combine arbitrary fn pointers into a single fn pointer easily.
                // Instead detect specific patterns.
                None
            }
        };
    }
    None
}

fn extract_type_cmp<'a>(lhs: &'a Expr, rhs: &'a Expr) -> Option<&'a str> {
    if let Expr::UnaryOp { op: crate::ir::UnaryOp::Type, operand } = lhs {
        if matches!(operand.as_ref(), Expr::Input) {
            if let Expr::Literal(crate::ir::Literal::Str(s)) = rhs {
                return Some(s.as_str());
            }
        }
    }
    None
}

/// Detect scalars filter: type != "array" and type != "object"
fn is_scalars_filter(expr: &Expr) -> bool {
    if let Expr::BinOp { op: crate::ir::BinOp::And, lhs, rhs } = expr {
        let l_type = extract_ne_type(lhs);
        let r_type = extract_ne_type(rhs);
        if let (Some(a), Some(b)) = (l_type, r_type) {
            return (a == "array" && b == "object") || (a == "object" && b == "array");
        }
    }
    false
}

fn extract_ne_type<'a>(expr: &'a Expr) -> Option<&'a str> {
    if let Expr::BinOp { op: crate::ir::BinOp::Ne, lhs, rhs } = expr {
        extract_type_cmp(lhs, rhs).or_else(|| extract_type_cmp(rhs, lhs))
    } else {
        None
    }
}

/// Detect an in-place assign pattern: .[path] = val
fn detect_inplace_assign(expr: &Expr) -> Option<InplaceAssignInfo<'_>> {
    match expr {
        Expr::Assign { path_expr, value_expr } => {
            let pc = extract_simple_path(path_expr)?;
            if !pc.is_empty() && is_scalar(value_expr) {
                Some(InplaceAssignInfo {
                    let_bindings: vec![],
                    path_components: pc,
                    value_expr,
                })
            } else {
                None
            }
        }
        Expr::LetBinding { var_index, value, body } if is_scalar(value) => {
            let mut info = detect_inplace_assign(body)?;
            info.let_bindings.insert(0, (*var_index, value.as_ref()));
            Some(info)
        }
        _ => None,
    }
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
        if let Value::Obj(ObjInner(rc)) = val {
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
    unsafe { std::ptr::write(dst, Value::number(n)); }
}
extern "C" fn jit_rt_num_repr(dst: *mut Value, n: f64, repr_ptr: *const Rc<str>) {
    unsafe { std::ptr::write(dst, Value::number_with_repr(n, (*repr_ptr).clone())); }
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
            Value::Obj(ObjInner(o)) => {
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
            Value::Obj(ObjInner(o)) => {
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
            Value::Obj(ObjInner(o)) => o,
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
                0 => { std::ptr::write(dst, Value::number(na + nb)); return 0; }
                1 => { std::ptr::write(dst, Value::number(na - nb)); return 0; }
                2 => { std::ptr::write(dst, Value::number(na * nb)); return 0; }
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
            Value::Obj(ObjInner(o)) => o,
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
                0 => { std::ptr::write(dst, Value::number(na + nb)); return 0; }
                1 => { std::ptr::write(dst, Value::number(na - nb)); return 0; }
                2 => { std::ptr::write(dst, Value::number(na * nb)); return 0; }
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
            Value::Obj(ObjInner(o)) => {
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
            Value::Obj(ObjInner(o)) => {
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
        if let (Value::Obj(ObjInner(o)), Value::Str(k)) = (&*base, &*key) {
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
                0 => { std::ptr::write(dst, Value::number(a + b)); return 0; }
                1 => { std::ptr::write(dst, Value::number(a - b)); return 0; }
                2 => { std::ptr::write(dst, Value::number(a * b)); return 0; }
                3 => { // Div: fast path for non-zero, non-NaN
                    if *b != 0.0 && !b.is_nan() && !a.is_nan() {
                        std::ptr::write(dst, Value::number(a / b)); return 0;
                    }
                }
                4 => { // Mod: fast path for non-zero integer modulo
                    if a.is_finite() && b.is_finite() {
                        let yi = *b as i64;
                        if yi != 0 {
                            let xi = *a as i64;
                            // Avoid overflow for i64::MIN % -1
                            let r = if xi == i64::MIN && yi == -1 { 0 } else { xi % yi };
                            std::ptr::write(dst, Value::number(r as f64)); return 0;
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
            let result = Value::number(a + b);
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
        // Fast path: tostring (op 15) — avoid full eval chain
        if op == 15 {
            match &*input {
                Value::Str(_) => { std::ptr::write(dst, (*input).clone()); return 0; }
                Value::Num(n, _) => {
                    // Delegate to value_to_json_tojson so the repr-preservation
                    // policy matches the slow path (#110): keep the original
                    // lexical form when f64 can round-trip it, otherwise fall
                    // back to the canonical f64 form.
                    let s = crate::value::value_to_json_tojson(&*input);
                    std::ptr::write(dst, Value::from_string(s));
                    let _ = n;
                    return 0;
                }
                Value::Null => { std::ptr::write(dst, Value::from_str("null")); return 0; }
                Value::True => { std::ptr::write(dst, Value::from_str("true")); return 0; }
                Value::False => { std::ptr::write(dst, Value::from_str("false")); return 0; }
                _ => { /* arrays/objects → fall through to value_to_json */ }
            }
        }
        // Fast path: tonumber (op 16) — avoid full eval chain
        if op == 16 {
            match &*input {
                Value::Num(_, _) => { std::ptr::write(dst, (*input).clone()); return 0; }
                Value::Str(s) => {
                    match s.as_str().trim().parse::<f64>() {
                        Ok(n) => {
                            std::ptr::write(dst, Value::number(n));
                            return 0;
                        }
                        Err(_) => {
                            set_jit_error(format!("invalid number: {:?}", s.as_str()));
                            std::ptr::write(dst, Value::Null);
                            return GEN_ERROR;
                        }
                    }
                }
                _ => {}
            }
        }
        // Fast path: length (op 0) — very common, avoid full dispatch
        if op == 0 {
            let v = match &*input {
                Value::Null => Value::number(0.0),
                Value::True | Value::False => {
                    set_jit_error(format!(
                        "{} ({}) has no length",
                        (*input).type_name(),
                        crate::value::value_to_json(&*input)
                    ));
                    std::ptr::write(dst, Value::Null);
                    return GEN_ERROR;
                }
                Value::Num(n, _) => Value::number(n.abs()),
                Value::Str(s) => {
                    let len = if s.is_ascii() { s.len() } else { s.chars().count() };
                    Value::number(len as f64)
                }
                Value::Arr(a) => Value::number(a.len() as f64),
                Value::Obj(ObjInner(o)) => Value::number(o.len() as f64),
                Value::Error(_) => Value::number(0.0),
            };
            std::ptr::write(dst, v);
            return 0;
        }
        // Fast path: type (op 1)
        if op == 1 {
            let s = match &*input {
                Value::Null => "null",
                Value::False | Value::True => "boolean",
                Value::Num(_, _) => "number",
                Value::Str(_) => "string",
                Value::Arr(_) => "array",
                Value::Obj(_) => "object",
                Value::Error(_) => "error",
            };
            std::ptr::write(dst, Value::from_str(s));
            return 0;
        }
        // Fast path: keys (op 2) and keys_unsorted (op 29)
        if op == 2 || op == 29 {
            match &*input {
                Value::Obj(ObjInner(o)) => {
                    let mut keys: Vec<Value> = o.keys().map(|k| Value::from_str(k)).collect();
                    if op == 2 { keys.sort_by(crate::runtime::compare_values); }
                    std::ptr::write(dst, Value::Arr(Rc::new(keys)));
                    return 0;
                }
                Value::Arr(a) => {
                    let keys: Vec<Value> = (0..a.len()).map(|i| Value::number(i as f64)).collect();
                    std::ptr::write(dst, Value::Arr(Rc::new(keys)));
                    return 0;
                }
                _ => {}
            }
        }
        // Fast path: values (op 3)
        if op == 3 {
            match &*input {
                Value::Obj(ObjInner(o)) => {
                    let vals: Vec<Value> = o.values().cloned().collect();
                    std::ptr::write(dst, Value::Arr(Rc::new(vals)));
                    return 0;
                }
                Value::Arr(_) => {
                    std::ptr::write(dst, (*input).clone());
                    return 0;
                }
                _ => {}
            }
        }
        // Fast path: to_entries (op 27)
        if op == 27 {
            if let Value::Obj(ObjInner(o)) = &*input {
                let mut entries = Vec::with_capacity(o.len());
                for (k, v) in o.iter() {
                    let mut entry = crate::value::new_objmap();
                    entry.insert("key".into(), Value::from_str(k));
                    entry.insert("value".into(), v.clone());
                    entries.push(Value::object_from_map(entry));
                }
                std::ptr::write(dst, Value::Arr(Rc::new(entries)));
                return 0;
            }
        }
        // Fast path: tojson (op 17)
        if op == 17 {
            std::ptr::write(dst, Value::from_string(crate::value::value_to_json(&*input)));
            return 0;
        }
        // Fast path: fromjson (op 18)
        if op == 18 {
            if let Value::Str(s) = &*input {
                // Trivial value fast paths — avoid the state-machine parser for
                // common literals and simple numbers/strings with no escapes.
                let trimmed = s.trim();
                let result = if trimmed == "null" {
                    Some(Value::Null)
                } else if trimmed == "true" {
                    Some(Value::True)
                } else if trimmed == "false" {
                    Some(Value::False)
                } else if !trimmed.is_empty() {
                    let b = trimmed.as_bytes()[0];
                    if b == b'-' || b.is_ascii_digit() {
                        trimmed.parse::<f64>().ok().map(Value::number)
                    } else if b == b'"' {
                        if trimmed.len() >= 2 && trimmed.as_bytes()[trimmed.len()-1] == b'"' && !trimmed[1..trimmed.len()-1].contains('\\') {
                            Some(Value::from_string(trimmed[1..trimmed.len()-1].to_string()))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(v) = result {
                    std::ptr::write(dst, v);
                    return 0;
                }
                match crate::value::json_to_value(trimmed) {
                    Ok(v) => { std::ptr::write(dst, v); return 0; }
                    Err(_) => {
                        // Strict parser rejected it; route through the
                        // jq-compatible parser so the surfaced error message
                        // stays byte-identical to jq 1.8.1.
                        match crate::value::json_to_value_fromjson(s) {
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
        }
        // Fast path: utf8bytelength (op 43)
        if op == 43 {
            if let Value::Str(s) = &*input {
                std::ptr::write(dst, Value::number(s.len() as f64));
                return 0;
            }
        }
        // Fast path: ascii_downcase (op 22) and ascii_upcase (op 23)
        if op == 22 || op == 23 {
            if let Value::Str(s) = &*input {
                if s.is_ascii() {
                    let mut bytes = s.as_bytes().to_vec();
                    if op == 22 { bytes.make_ascii_lowercase(); } else { bytes.make_ascii_uppercase(); }
                    std::ptr::write(dst, Value::from_string(String::from_utf8_unchecked(bytes)));
                } else {
                    std::ptr::write(dst, Value::from_string(
                        if op == 22 {
                            s.chars().map(|c| if c.is_ascii() { c.to_ascii_lowercase() } else { c }).collect()
                        } else {
                            s.chars().map(|c| if c.is_ascii() { c.to_ascii_uppercase() } else { c }).collect()
                        }
                    ));
                }
                return 0;
            }
        }
        // Fast path: from_entries (op 28). Bail to the generic builtin path when the
        // key resolves to a non-string so the eval path emits jq's type error (issue #73).
        if op == 28 {
            if let Value::Arr(a) = &*input {
                let mut obj = crate::value::new_objmap();
                let mut ok = true;
                for entry in a.iter() {
                    match entry {
                        Value::Obj(ObjInner(o)) => {
                            let pick_truthy = |name: &str| -> Option<&Value> {
                                match o.get(name) {
                                    Some(v) if !matches!(v, Value::Null | Value::False) => Some(v),
                                    _ => None,
                                }
                            };
                            let key = pick_truthy("key")
                                .or_else(|| pick_truthy("key_"))
                                .or_else(|| pick_truthy("Key"))
                                .or_else(|| pick_truthy("name"))
                                .or_else(|| pick_truthy("Name"))
                                .cloned()
                                .unwrap_or(Value::Null);
                            let val = o.get("value").or_else(|| o.get("Value"))
                                .cloned().unwrap_or(Value::Null);
                            let key_str = match &key {
                                Value::Str(s) => crate::value::KeyStr::from(s.as_str()),
                                _ => { ok = false; break; }
                            };
                            obj.insert(key_str, val);
                        }
                        _ => { ok = false; break; }
                    }
                }
                if ok {
                    std::ptr::write(dst, Value::object_from_map(obj));
                    return 0;
                }
            }
        }
        // Fast path: flatten (op 11) — recursive flatten
        if op == 11 {
            if let Value::Arr(a) = &*input {
                fn flatten_into(arr: &[Value], result: &mut Vec<Value>) {
                    for item in arr {
                        match item {
                            Value::Arr(inner) => flatten_into(inner, result),
                            _ => result.push(item.clone()),
                        }
                    }
                }
                let mut result = Vec::new();
                flatten_into(a, &mut result);
                std::ptr::write(dst, Value::Arr(Rc::new(result)));
                return 0;
            }
        }
        // Fast path: explode (op 20) — string → array of codepoints
        if op == 20 {
            if let Value::Str(s) = &*input {
                let str = s.as_str();
                if str.is_ascii() {
                    let mut codepoints = Vec::with_capacity(str.len());
                    for &b in str.as_bytes() {
                        codepoints.push(Value::number(b as f64));
                    }
                    std::ptr::write(dst, Value::Arr(Rc::new(codepoints)));
                } else {
                    let codepoints: Vec<Value> = str.chars()
                        .map(|c| Value::number(c as u32 as f64))
                        .collect();
                    std::ptr::write(dst, Value::Arr(Rc::new(codepoints)));
                }
                return 0;
            }
        }
        // Fast path: implode (op 21) — array of codepoints → string
        if op == 21 {
            if let Value::Arr(a) = &*input {
                let mut s = String::with_capacity(a.len());
                let mut ok = true;
                for item in a.iter() {
                    if let Value::Num(n, _) = item {
                        let cp = *n as u32;
                        if let Some(c) = char::from_u32(cp) {
                            s.push(c);
                        } else {
                            s.push('\u{FFFD}');
                        }
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    std::ptr::write(dst, Value::from_string(s));
                    return 0;
                }
            }
        }
        // Fast path: not (op 32)
        if op == 32 {
            let truthy = match &*input {
                Value::Null | Value::False => false,
                _ => true,
            };
            std::ptr::write(dst, if truthy { Value::False } else { Value::True });
            return 0;
        }
        // Fast path: isnan/isinfinite/isnormal/isfinite (ops 36-39)
        if op >= 36 && op <= 39 {
            let result = if let Value::Num(n, _) = &*input {
                match op {
                    36 => n.is_infinite(),
                    37 => n.is_nan(),
                    38 => n.is_normal(),
                    // isfinite: jq's def is `type == "number" and (isinfinite | not)`,
                    // so NaN counts as finite (issue #108).
                    39 => !n.is_infinite(),
                    _ => unreachable!(),
                }
            } else {
                // Non-numeric: isnan/isinfinite/isnormal→false, isfinite→false per jq semantics
                false
            };
            std::ptr::write(dst, if result { Value::True } else { Value::False });
            return 0;
        }
        // Fast path: inline math ops on numbers — avoids eval_unaryop → call_builtin chain
        if let Value::Num(n, NumRepr(repr)) = &*input {
            let n = *n;
            // abs/fabs: preserve repr for non-negative numbers
            if (op == 7 || op == 31) && n >= 0.0 {
                std::ptr::write(dst, (*input).clone());
                return 0;
            }
            let result = match op {
                4 => Some(n.floor()),   // Floor
                5 => Some(n.ceil()),    // Ceil
                6 => Some(n.round()),   // Round
                7 | 31 => Some(n.abs()), // Fabs / Abs (negative case)
                30 => Some(n.sqrt()),   // Sqrt
                44 => Some(n.sin()),    // Sin
                45 => Some(n.cos()),    // Cos
                46 => Some(n.tan()),    // Tan
                47 => Some(n.asin()),   // Asin
                48 => Some(n.acos()),   // Acos
                49 => Some(n.atan()),   // Atan
                50 => Some(n.exp()),    // Exp
                51 => Some(n.exp2()),   // Exp2
                53 => Some(n.ln()),     // Log
                54 => Some(n.log2()),   // Log2
                55 => Some(n.log10()),  // Log10
                56 => Some(n.cbrt()),   // Cbrt
                61 => Some(n.trunc()),  // Trunc
                _ => None,
            };
            if let Some(r) = result {
                // Preserve repr if result equals the original value
                let keep_repr = repr.is_some() && r == n;
                std::ptr::write(dst, Value::number_opt(r, if keep_repr { repr.clone() } else { None }));
                return 0;
            }
        }
        // Fast path: any (op 24) — array any truthy
        if op == 24 {
            if let Value::Arr(a) = &*input {
                let r = a.iter().any(|v| !matches!(v, Value::Null | Value::False));
                std::ptr::write(dst, if r { Value::True } else { Value::False });
                return 0;
            }
        }
        // Fast path: all (op 25) — array all truthy
        if op == 25 {
            if let Value::Arr(a) = &*input {
                let r = a.iter().all(|v| !matches!(v, Value::Null | Value::False));
                std::ptr::write(dst, if r { Value::True } else { Value::False });
                return 0;
            }
        }
        // Fast path: min (op 12)
        if op == 12 {
            if let Value::Arr(a) = &*input {
                if a.is_empty() {
                    std::ptr::write(dst, Value::Null);
                    return 0;
                }
                let mut m = &a[0];
                for v in &a[1..] {
                    if crate::runtime::compare_values(v, m) == std::cmp::Ordering::Less { m = v; }
                }
                std::ptr::write(dst, m.clone());
                return 0;
            }
        }
        // Fast path: max (op 13)
        if op == 13 {
            if let Value::Arr(a) = &*input {
                if a.is_empty() {
                    std::ptr::write(dst, Value::Null);
                    return 0;
                }
                let mut m = &a[0];
                for v in &a[1..] {
                    if crate::runtime::compare_values(v, m) == std::cmp::Ordering::Greater { m = v; }
                }
                std::ptr::write(dst, m.clone());
                return 0;
            }
        }
        // Fast path: sort (op 8)
        if op == 8 {
            if let Value::Arr(a) = &*input {
                let mut sorted = a.as_ref().clone();
                sorted.sort_by(crate::runtime::compare_values);
                std::ptr::write(dst, Value::Arr(Rc::new(sorted)));
                return 0;
            }
        }
        // Fast path: unique (op 10)
        if op == 10 {
            if let Value::Arr(a) = &*input {
                let mut sorted = a.as_ref().clone();
                sorted.sort_by(crate::runtime::compare_values);
                sorted.dedup_by(|a, b| crate::runtime::compare_values(a, b) == std::cmp::Ordering::Equal);
                std::ptr::write(dst, Value::Arr(Rc::new(sorted)));
                return 0;
            }
        }
        // Fast path: reverse (op 9)
        if op == 9 {
            match &*input {
                Value::Arr(a) => {
                    let mut r = a.as_ref().clone();
                    r.reverse();
                    std::ptr::write(dst, Value::Arr(Rc::new(r)));
                    return 0;
                }
                Value::Str(s) => {
                    std::ptr::write(dst, Value::from_string(s.chars().rev().collect()));
                    return 0;
                }
                Value::Null => {
                    std::ptr::write(dst, Value::Arr(Rc::new(vec![])));
                    return 0;
                }
                _ => {}
            }
        }
        // Fast path: add (op 14)
        if op == 14 {
            match &*input {
                Value::Arr(a) if a.is_empty() => {
                    std::ptr::write(dst, Value::Null);
                    return 0;
                }
                Value::Arr(a) => {
                    // Numeric sum fast path
                    if let Value::Num(_, _) = &a[0] {
                        let mut sum = 0.0f64;
                        let mut all_num = true;
                        for item in a.iter() {
                            match item {
                                Value::Num(n, _) => sum += n,
                                Value::Null => {}
                                _ => { all_num = false; break; }
                            }
                        }
                        if all_num {
                            std::ptr::write(dst, Value::number(sum));
                            return 0;
                        }
                    }
                    // Fall through to eval_unaryop for strings/arrays/objects
                }
                _ => {}
            }
        }
        // Fast path: transpose (op 26)
        if op == 26 {
            if let Value::Arr(a) = &*input {
                let max_len = a.iter().filter_map(|v| if let Value::Arr(sub) = v { Some(sub.len()) } else { None }).max().unwrap_or(0);
                let mut result = Vec::with_capacity(max_len);
                for i in 0..max_len {
                    let mut row = Vec::with_capacity(a.len());
                    for item in a.iter() {
                        if let Value::Arr(sub) = item {
                            row.push(sub.get(i).cloned().unwrap_or(Value::Null));
                        } else {
                            row.push(Value::Null);
                        }
                    }
                    result.push(Value::Arr(Rc::new(row)));
                }
                std::ptr::write(dst, Value::Arr(Rc::new(result)));
                return 0;
            }
        }
        // Fast path: infinite (op 34), nan (op 35)
        if op == 34 {
            std::ptr::write(dst, Value::number(f64::INFINITY));
            return 0;
        }
        if op == 35 {
            std::ptr::write(dst, Value::number(f64::NAN));
            return 0;
        }
        // Fast path: now (op 67)
        if op == 67 {
            std::ptr::write(dst, Value::number(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64()));
            return 0;
        }
        // Fast path: ltrimstr/rtrimstr/trim (op 40/41/42) — null-arg form trims whitespace
        if op >= 40 && op <= 42 {
            if let Value::Str(s) = &*input {
                let r = match op {
                    40 => s.trim_start(),
                    41 => s.trim_end(),
                    42 => s.trim(),
                    _ => unreachable!(),
                };
                std::ptr::write(dst, Value::from_str(r));
                return 0;
            }
        }
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
        // Store value directly — defer serialization until propagation (if not caught)
        set_jit_error_value((*msg).clone());
        // `set_jit_error_value` flips the flag via CURRENT_ENV; we also have
        // env here explicitly, so write through it to keep the contract local.
        (*env).error_flag = 1;
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
            Value::Num(n, _) => { std::ptr::write(dst, Value::number(-n)); 0 }
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
        match &*v { Value::Arr(a) => a.len() as i64, Value::Obj(ObjInner(o)) => o.len() as i64, _ => 0 }
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
/// Take value at index from container (O(1) direct Vec access, no hash).
extern "C" fn jit_rt_take_by_idx(dst: *mut Value, container: *mut Value, idx: i64) {
    unsafe {
        let i = idx as usize;
        match &mut *container {
            Value::Arr(rc) => {
                let arr = Rc::make_mut(rc);
                if i < arr.len() {
                    std::ptr::write(dst, std::mem::replace(&mut arr[i], Value::Null));
                } else {
                    std::ptr::write(dst, Value::Null);
                }
            }
            Value::Obj(ObjInner(rc)) => {
                let obj = Rc::make_mut(rc);
                if let Some(v) = obj.get_value_mut_by_index(i) {
                    std::ptr::write(dst, std::mem::replace(v, Value::Null));
                } else {
                    std::ptr::write(dst, Value::Null);
                }
            }
            _ => std::ptr::write(dst, Value::Null),
        }
    }
}

/// Put value at index into container (O(1) direct Vec access, no hash).
/// In-place setpath for reduce contexts. Takes ownership of the container (via TakeVar).
/// Void return — errors are silently swallowed (setpath in reduce rarely errors).
extern "C" fn jit_rt_setpath_mut(container: *mut Value, path: *const Value, val: *mut Value) {
    unsafe {
        let path_val = &*path;
        let new_val = std::ptr::read(val);
        std::ptr::write(val, Value::Null);

        if let Value::Arr(p) = path_val {
            let _ = crate::runtime::rt_setpath_mut(&mut *container, p.as_slice(), new_val);
        }
    }
}

extern "C" fn jit_rt_put_by_idx(container: *mut Value, idx: i64, val: *mut Value) {
    unsafe {
        let new_val = std::ptr::read(val);
        std::ptr::write(val, Value::Null);
        let i = idx as usize;
        match &mut *container {
            Value::Arr(rc) => {
                let arr = Rc::make_mut(rc);
                if i < arr.len() {
                    arr[i] = new_val;
                }
            }
            Value::Obj(ObjInner(rc)) => {
                let obj = Rc::make_mut(rc);
                if let Some(v) = obj.get_value_mut_by_index(i) {
                    *v = new_val;
                }
            }
            _ => {}
        }
    }
}

extern "C" fn jit_rt_obj_get_idx(dst: *mut Value, obj: *const Value, idx: i64) {
    unsafe {
        match &*obj {
            Value::Obj(ObjInner(o)) => match o.get_index(idx as usize) {
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

/// Move value from slot to var (drop old var, leave slot as Null). Avoids clone.
extern "C" fn jit_rt_move_to_var(env: *mut JitEnv, idx: u32, val: *mut Value) {
    unsafe {
        let env = &mut *env;
        let i = idx as usize;
        if i >= env.vars.len() { env.vars.resize(i + 1, Value::Null); }
        // Drop old value, move new value in, leave source as Null
        let new_val = std::mem::replace(&mut *val, Value::Null);
        let _old = std::mem::replace(&mut env.vars[i], new_val);
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
            (Value::Obj(ObjInner(rc_obj)), Value::Str(k)) => {
                let obj = Rc::make_mut(rc_obj);
                if let Some(v) = obj.get_mut(k.as_str()) {
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
            (Value::Obj(ObjInner(rc_obj)), Value::Str(k)) => {
                let obj = Rc::make_mut(rc_obj);
                obj.insert(crate::value::KeyStr::from(k.as_str()), new_val);
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
                *container = Value::object_from_map(obj);
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
/// Fused [range(n)]: pre-allocate and fill array in a single call.
extern "C" fn jit_rt_collect_range(dst: *mut Value, n: f64) {
    let count = n as i64;
    if count <= 0 {
        unsafe { std::ptr::write(dst, Value::Arr(Rc::new(Vec::new()))); }
        return;
    }
    let count = count as usize;
    let mut arr = Vec::with_capacity(count);
    for i in 0..count {
        arr.push(Value::number(i as f64));
    }
    unsafe { std::ptr::write(dst, Value::Arr(Rc::new(arr))); }
}
/// In-place array push: arr.push(val). Handles null → [val] (jq null + array identity).
extern "C" fn jit_rt_arr_push(arr: *mut Value, val: *const Value) {
    unsafe {
        match &mut *arr {
            Value::Arr(rc) => {
                Rc::make_mut(rc).push((*val).clone());
            }
            Value::Null => {
                // null + [x] = [x] in jq
                std::ptr::write(arr, Value::Arr(Rc::new(vec![(*val).clone()])));
            }
            _ => {
                // Type error: silently ignore (should not happen in well-typed reduce)
            }
        }
    }
}
// System math wrappers — use platform-optimized implementations (not libm pure-Rust)
extern "C" fn sys_sin(x: f64) -> f64 { x.sin() }
extern "C" fn sys_cos(x: f64) -> f64 { x.cos() }
extern "C" fn sys_tan(x: f64) -> f64 { x.tan() }
extern "C" fn sys_asin(x: f64) -> f64 { x.asin() }
extern "C" fn sys_acos(x: f64) -> f64 { x.acos() }
extern "C" fn sys_atan(x: f64) -> f64 { x.atan() }
extern "C" fn sys_exp(x: f64) -> f64 { x.exp() }
extern "C" fn sys_exp2(x: f64) -> f64 { x.exp2() }
extern "C" fn sys_log(x: f64) -> f64 { x.ln() }
extern "C" fn sys_log2(x: f64) -> f64 { x.log2() }
extern "C" fn sys_log10(x: f64) -> f64 { x.log10() }
extern "C" fn sys_cbrt(x: f64) -> f64 { x.cbrt() }

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
    unsafe { std::ptr::write(dst, Value::Obj(ObjInner(crate::value::rc_objmap_pool_get(cap)))); }
}
/// Insert key-value pair into object. Takes ownership of both key and val (ptr::read, no clone).
/// Returns 0 on success, -1 on error (jq errors when the key is not a string).
extern "C" fn jit_rt_obj_insert(obj: *mut Value, key: *mut Value, val: *mut Value) -> i64 {
    unsafe {
        let key_val = std::ptr::read(key);
        let val_val = std::ptr::read(val);
        if let Value::Obj(ObjInner(o)) = &mut *obj {
            let k: crate::value::KeyStr = match key_val {
                Value::Str(s) => s,
                other => {
                    set_jit_error(format!(
                        "Cannot use {} as object key",
                        crate::runtime::errdesc_pub(&other)
                    ));
                    drop(other);
                    drop(val_val);
                    return -1;
                }
            };
            // Safety: obj was just created by jit_rt_obj_new, refcount is always 1
            Rc::get_mut(o).unwrap_unchecked().insert(k, val_val);
        }
        0
    }
}

/// Insert into object with raw string key — avoids creating/dropping a Value for the key.
extern "C" fn jit_rt_obj_insert_str_key(obj: *mut Value, key_ptr: *const u8, key_len: usize, val: *mut Value) {
    unsafe {
        let val_val = std::ptr::read(val);
        if let Value::Obj(ObjInner(o)) = &mut *obj {
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
        if let Value::Obj(ObjInner(o)) = &mut *obj {
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
            Value::Obj(ObjInner(o)) => o.get(src_field).cloned().unwrap_or(Value::Null),
            Value::Null => Value::Null,
            _ => Value::Null,
        };
        if let Value::Obj(ObjInner(o)) = &mut *obj {
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
        if let Value::Obj(ObjInner(src_obj)) = &*src {
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
            std::ptr::write(dst, Value::Obj(ObjInner(obj)));
        } else {
            // Non-object source: all fields null
            let mut obj = crate::value::rc_objmap_pool_get(count);
            let map = Rc::get_mut(&mut obj).unwrap_unchecked();
            for &(okp, okl, _, _) in pairs {
                let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(okp, okl));
                map.push_unique(crate::value::KeyStr::from(name), Value::Null);
            }
            std::ptr::write(dst, Value::Obj(ObjInner(obj)));
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
    unsafe { std::ptr::write(dst, Value::number(n)); }
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
        let buf = (*env).str_bufs.last_mut().unwrap();
        match &*val {
            Value::Str(s) => buf.push_str(s.as_str()),
            Value::Null => buf.push_str("null"),
            Value::False => buf.push_str("false"),
            Value::True => buf.push_str("true"),
            Value::Num(n, NumRepr(repr)) => {
                if let Some(r) = repr {
                    buf.push_str(r);
                } else {
                    // Write directly to the string's byte buffer — avoids intermediate String
                    let bytes = buf.as_mut_vec();
                    crate::value::push_jq_number_bytes(bytes, *n);
                }
            }
            v => {
                let s = crate::value::value_to_json(v);
                buf.push_str(&s);
            }
        }
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

        // Handle special names
        if let Some(rest) = name.strip_prefix("__loc__:") {
            // Parse file:line
            let parts: Vec<&str> = rest.rsplitn(2, ':').collect();
            if parts.len() == 2 {
                let line_n: i64 = parts[0].parse().unwrap_or(0);
                let file = parts[1];
                let mut obj = crate::value::new_objmap();
                obj.insert(crate::value::KeyStr::from("file"), Value::from_str(file));
                obj.insert(crate::value::KeyStr::from("line"), Value::number(line_n as f64));
                std::ptr::write(dst, Value::object_from_map(obj));
                return 0;
            }
        }
        if name == "input_line_number" {
            std::ptr::write(dst, Value::number(crate::eval::get_input_line_number() as f64));
            return 0;
        }
        if name == "__env__" {
            // Cache env object — environment is constant during execution.
            thread_local! {
                static ENV_CACHE: RefCell<Option<Value>> = const { RefCell::new(None) };
            }
            let env_value = ENV_CACHE.with_borrow_mut(|cached| {
                if cached.is_none() {
                    let mut obj = crate::value::new_objmap();
                    for (k, v) in std::env::vars() {
                        obj.insert(crate::value::KeyStr::from(k), Value::from_string(v));
                    }
                    *cached = Some(Value::object_from_map(obj));
                }
                cached.as_ref().unwrap().clone()
            });
            std::ptr::write(dst, env_value);
            return 0;
        }
        if name == "__each_error__" {
            // Generate "Cannot iterate over TYPE (VALUE)" error. Use
            // errdesc so number reprs survive (`0.0` stays `0.0`) and long
            // values get jq's `...` truncation. See #574.
            if !args.is_empty() {
                let v = &args[0];
                set_jit_error(format!("Cannot iterate over {}", crate::runtime::errdesc_pub(v)));
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
            if !args.is_empty() {
                let mut results = Vec::new();
                let mut stack = vec![args[0].clone()];
                while let Some(v) = stack.pop() {
                    results.push(v.clone());
                    match &v {
                        Value::Arr(a) => {
                            for item in a.iter().rev() {
                                stack.push(item.clone());
                            }
                        }
                        Value::Obj(ObjInner(o)) => {
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
        if name == "paths_collect_all" {
            // Native path(recurse): generate all paths (including root []) via DFS
            if !args.is_empty() {
                let mut results = Vec::new();
                // Root path []
                results.push(Value::Arr(Rc::new(vec![])));
                // DFS stack: (child_value, path_to_child)
                let mut stack: Vec<(Value, Vec<Value>)> = Vec::new();
                // Push root's children in reverse order for correct DFS ordering
                match &args[0] {
                    Value::Obj(ObjInner(o)) => {
                        for (key, val) in o.iter().rev() {
                            stack.push((val.clone(), vec![Value::from_str(key.as_str())]));
                        }
                    }
                    Value::Arr(a) => {
                        for (i, val) in a.iter().enumerate().rev() {
                            stack.push((val.clone(), vec![Value::number(i as f64)]));
                        }
                    }
                    _ => {}
                }
                while let Some((val, path)) = stack.pop() {
                    results.push(Value::Arr(Rc::new(path.clone())));
                    match &val {
                        Value::Obj(ObjInner(o)) => {
                            for (key, child) in o.iter().rev() {
                                let mut p = path.clone();
                                p.push(Value::from_str(key.as_str()));
                                stack.push((child.clone(), p));
                            }
                        }
                        Value::Arr(a) => {
                            for (i, child) in a.iter().enumerate().rev() {
                                let mut p = path.clone();
                                p.push(Value::number(i as f64));
                                stack.push((child.clone(), p));
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
            if args.len() >= 2 {
                let input = &args[0];
                let label = &args[1];
                match label {
                    Value::Str(s) if !s.is_empty() => {
                        eprintln!("[\"DEBUG:\",{}]", crate::value::value_to_json_tojson(input));
                    }
                    _ => {
                        eprintln!("[\"DEBUG:\",{}]", crate::value::value_to_json_tojson(input));
                    }
                }
                std::ptr::write(dst, input.clone());
            } else if !args.is_empty() {
                eprintln!("[\"DEBUG:\",{}]", crate::value::value_to_json_tojson(&args[0]));
                std::ptr::write(dst, args[0].clone());
            } else {
                std::ptr::write(dst, Value::Null);
            }
            return 0;
        }
        if name == "stderr" {
            if !args.is_empty() {
                eprint!("{}", crate::value::value_to_json_tojson(&args[0]));
                std::ptr::write(dst, args[0].clone());
            } else {
                std::ptr::write(dst, Value::Null);
            }
            return 0;
        }
        // Fast path: split(sep) — avoid call_builtin dispatch
        if name == "split" && args.len() == 2 {
            if let (Value::Str(s), Value::Str(p)) = (&args[0], &args[1]) {
                let parts: Vec<Value> = if p.is_empty() {
                    if s.is_ascii() {
                        (0..s.len()).map(|i| Value::from_str(&s.as_str()[i..i+1])).collect()
                    } else {
                        let mut buf = [0u8; 4];
                        s.chars().map(|c| Value::from_str(c.encode_utf8(&mut buf))).collect()
                    }
                } else {
                    s.split(p.as_str()).map(Value::from_str).collect()
                };
                std::ptr::write(dst, Value::Arr(Rc::new(parts)));
                return 0;
            }
        }
        // Fast path: join(sep) — avoid call_builtin dispatch
        if name == "join" && args.len() == 2 {
            if let (Value::Arr(a), Value::Str(sep)) = (&args[0], &args[1]) {
                // Check all elements are scalar (not arr/obj) — otherwise fall through to runtime for error
                let all_scalar = a.iter().all(|v| !matches!(v, Value::Arr(_) | Value::Obj(_) | Value::Error(_)));
                if all_scalar {
                    let cap = a.len() * (8 + sep.len());
                    let mut buf: Vec<u8> = Vec::with_capacity(cap);
                    for (i, item) in a.iter().enumerate() {
                        if i > 0 { buf.extend_from_slice(sep.as_bytes()); }
                        match item {
                            Value::Str(sv) => buf.extend_from_slice(sv.as_bytes()),
                            Value::Null => {},
                            Value::Num(n, NumRepr(repr)) => {
                                if let Some(r) = repr.as_ref().filter(|r| crate::value::is_valid_json_number(r)) {
                                    buf.extend_from_slice(r.as_bytes());
                                } else {
                                    crate::value::push_jq_number_bytes(&mut buf, *n);
                                }
                            }
                            Value::True => buf.extend_from_slice(b"true"),
                            Value::False => buf.extend_from_slice(b"false"),
                            _ => {}
                        }
                    }
                    std::ptr::write(dst, Value::from_string(String::from_utf8_unchecked(buf)));
                    return 0;
                }
            }
        }
        // Fast path: has(key) — avoid dispatch overhead
        if name == "has" && args.len() == 2 {
            let result = match (&args[0], &args[1]) {
                (Value::Obj(ObjInner(o)), Value::Str(k)) => Some(Value::from_bool(o.contains_key(k.as_str()))),
                (Value::Arr(a), Value::Num(n, _)) => {
                    if n.is_nan() || n.is_infinite() { Some(Value::False) }
                    else {
                        let idx = *n as i64;
                        Some(Value::from_bool(idx >= 0 && (idx as usize) < a.len()))
                    }
                }
                (Value::Null, _) => Some(Value::False),
                _ => None,
            };
            if let Some(v) = result { std::ptr::write(dst, v); return 0; }
        }
        // Fast path: in(container) — avoid dispatch overhead.
        // Mirrors `has` with swapped operands: args are (key, container) here.
        // Error cases (type mismatch) fall through to runtime for a proper jq error.
        if name == "in" && args.len() == 2 {
            let result = match (&args[1], &args[0]) {
                (Value::Obj(ObjInner(o)), Value::Str(k)) => Some(Value::from_bool(o.contains_key(k.as_str()))),
                (Value::Arr(a), Value::Num(n, _)) => {
                    if n.is_nan() || n.is_infinite() { Some(Value::False) }
                    else {
                        let idx = *n as i64;
                        Some(Value::from_bool(idx >= 0 && (idx as usize) < a.len()))
                    }
                }
                (Value::Null, _) => Some(Value::False),
                _ => None,
            };
            if let Some(v) = result { std::ptr::write(dst, v); return 0; }
        }
        // Fast path: indices(str) — avoid call_builtin dispatch
        if (name == "indices" || name == "rindices") && args.len() == 2 {
            if let (Value::Str(s), Value::Str(t)) = (&args[0], &args[1]) {
                if !t.is_empty() && s.is_ascii() && t.is_ascii() {
                    let sb = s.as_bytes();
                    let tb = t.as_bytes();
                    let mut indices: Vec<Value> = Vec::new();
                    if tb.len() == 1 {
                        let needle = tb[0];
                        let mut pos = 0;
                        while pos < sb.len() {
                            if let Some(found) = memchr::memchr(needle, &sb[pos..]) {
                                indices.push(Value::number((pos + found) as f64));
                                pos += found + 1;
                            } else {
                                break;
                            }
                        }
                    } else if tb.len() <= sb.len() {
                        for i in 0..=sb.len() - tb.len() {
                            if &sb[i..i+tb.len()] == tb {
                                indices.push(Value::number(i as f64));
                            }
                        }
                    }
                    std::ptr::write(dst, Value::Arr(Rc::new(indices)));
                    return 0;
                }
            }
        }
        // Fast path: contains(str) — direct string containment check
        if name == "contains" && args.len() == 2 {
            if let (Value::Str(s), Value::Str(t)) = (&args[0], &args[1]) {
                std::ptr::write(dst, Value::from_bool(s.contains(t.as_str())));
                return 0;
            }
        }
        // Fast path: inside(str) — reverse containment
        if name == "inside" && args.len() == 2 {
            if let (Value::Str(s), Value::Str(t)) = (&args[0], &args[1]) {
                std::ptr::write(dst, Value::from_bool(t.contains(s.as_str())));
                return 0;
            }
        }
        // Fast path: startswith/endswith/ltrimstr/rtrimstr — string operations
        if args.len() == 2 {
            if let (Value::Str(s), Value::Str(t)) = (&args[0], &args[1]) {
                match name {
                    "startswith" => { std::ptr::write(dst, Value::from_bool(s.starts_with(t.as_str()))); return 0; }
                    "endswith" => { std::ptr::write(dst, Value::from_bool(s.ends_with(t.as_str()))); return 0; }
                    "ltrimstr" => {
                        std::ptr::write(dst, if let Some(rest) = s.strip_prefix(t.as_str()) {
                            Value::from_str(rest)
                        } else {
                            args[0].clone()
                        });
                        return 0;
                    }
                    "rtrimstr" => {
                        std::ptr::write(dst, if t.is_empty() {
                            Value::from_str("")
                        } else if let Some(rest) = s.strip_suffix(t.as_str()) {
                            Value::from_str(rest)
                        } else {
                            args[0].clone()
                        });
                        return 0;
                    }
                    _ => {}
                }
            }
        }
        // Fast path: getpath([path]) — inline path traversal.
        //
        // Only the type-matched arms (Obj+Str, Arr+Num, Null+Str/Num/Obj) are
        // safe to short-circuit here; for any other (current, key) combination
        // jq raises "Cannot index <type> with <kind>", which the slow path
        // (`rt_getpath`) reproduces. The earlier version returned Null for
        // unmatched arms (treating "ok=false but current==Null" as success),
        // which made callers like the JIT Update branch silently see Null at
        // a path that should have errored — e.g. `.a |= error` on a number
        // ran the closure with `null` instead of erroring at the path step.
        // See #554.
        if name == "getpath" && args.len() == 2 {
            if let Value::Arr(path) = &args[1] {
                let mut current = args[0].clone();
                let mut ok = true;
                for key in path.iter() {
                    match (&current, key) {
                        (Value::Obj(ObjInner(o)), Value::Str(k)) => {
                            current = o.get(k.as_str()).cloned().unwrap_or(Value::Null);
                        }
                        (Value::Arr(a), Value::Num(n, _)) => {
                            let idx = *n as i64;
                            let actual = if idx < 0 { (a.len() as i64 + idx) as usize } else { idx as usize };
                            current = a.get(actual).cloned().unwrap_or(Value::Null);
                        }
                        (Value::Null, Value::Str(_)) | (Value::Null, Value::Num(_, _)) | (Value::Null, Value::Obj(_)) => {
                            current = Value::Null;
                        }
                        _ => { ok = false; break; }
                    }
                }
                if ok {
                    std::ptr::write(dst, current);
                    return 0;
                }
            }
        }
        if name == "_slice" {
            // _slice(base, from, to)
            if args.len() >= 3 {
                match crate::eval::eval_slice(&args[0], &args[1], &args[2]) {
                    Ok(v) => { std::ptr::write(dst, v); return 0; }
                    Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); return GEN_ERROR; }
                }
            }
        }
        if let Some(format_name) = name.strip_prefix('@') {
            // Format: @base64, @uri, etc.
            if args.is_empty() {
                set_jit_error("format: no input".to_string());
                std::ptr::write(dst, Value::Null);
                return GEN_ERROR;
            }
            // Inline @csv fast path — avoid String→CompactString conversion
            if format_name == "csv" {
                if let Value::Arr(arr) = &args[0] {
                    let mut buf: Vec<u8> = Vec::with_capacity(arr.len() * 16);
                    for (i, v) in arr.iter().enumerate() {
                        if i > 0 { buf.push(b','); }
                        match v {
                            Value::Str(s) => {
                                buf.push(b'"');
                                if s.as_bytes().contains(&b'"') {
                                    for &b in s.as_bytes() {
                                        if b == b'"' { buf.push(b'"'); }
                                        buf.push(b);
                                    }
                                } else {
                                    buf.extend_from_slice(s.as_bytes());
                                }
                                buf.push(b'"');
                            }
                            Value::Null => {}
                            Value::True => buf.extend_from_slice(b"true"),
                            Value::False => buf.extend_from_slice(b"false"),
                            Value::Num(n, crate::value::NumRepr(repr)) => crate::value::push_value_num_repr_bytes(&mut buf, *n, repr.as_ref()),
                            _ => buf.extend_from_slice(crate::value::value_to_json(v).as_bytes()),
                        }
                    }
                    std::ptr::write(dst, Value::from_string(String::from_utf8_unchecked(buf)));
                    return 0;
                } else {
                    set_jit_error(format!(
                        "{} cannot be csv-formatted, only array",
                        crate::runtime::errdesc_pub(&args[0])
                    ));
                    std::ptr::write(dst, Value::Null);
                    return GEN_ERROR;
                }
            }
            // Inline @tsv fast path
            if format_name == "tsv" {
                if let Value::Arr(arr) = &args[0] {
                    let mut buf: Vec<u8> = Vec::with_capacity(arr.len() * 16);
                    for (i, v) in arr.iter().enumerate() {
                        if i > 0 { buf.push(b'\t'); }
                        match v {
                            Value::Str(s) => {
                                for &b in s.as_bytes() {
                                    match b {
                                        b'\\' => buf.extend_from_slice(b"\\\\"),
                                        b'\t' => buf.extend_from_slice(b"\\t"),
                                        b'\n' => buf.extend_from_slice(b"\\n"),
                                        b'\r' => buf.extend_from_slice(b"\\r"),
                                        _ => buf.push(b),
                                    }
                                }
                            }
                            Value::Null => {}
                            Value::True => buf.extend_from_slice(b"true"),
                            Value::False => buf.extend_from_slice(b"false"),
                            Value::Num(n, crate::value::NumRepr(repr)) => crate::value::push_value_num_repr_bytes(&mut buf, *n, repr.as_ref()),
                            _ => buf.extend_from_slice(crate::value::value_to_json(v).as_bytes()),
                        }
                    }
                    std::ptr::write(dst, Value::from_string(String::from_utf8_unchecked(buf)));
                    return 0;
                } else {
                    set_jit_error(format!(
                        "{} cannot be tsv-formatted, only array",
                        crate::runtime::errdesc_pub(&args[0])
                    ));
                    std::ptr::write(dst, Value::Null);
                    return GEN_ERROR;
                }
            }
            match crate::eval::eval_format(format_name, &args[0]) {
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
                let (path_expr, value_expr) = JIT_CLOSURE_OPS.with(|cell| {
                    let ops = &*cell.get();
                    (ops.get(path_idx).cloned(), ops.get(value_idx).cloned())
                });
                if let (Some(path_expr), Some(value_expr)) = (path_expr, value_expr) {
                    let input = if !args.is_empty() { args[0].clone() } else { Value::Null };
                    let env = new_delegated_env(&[&path_expr, &value_expr]);
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
                let (path_expr, update_expr) = JIT_CLOSURE_OPS.with(|cell| {
                    let ops = &*cell.get();
                    (ops.get(path_idx).cloned(), ops.get(update_idx).cloned())
                });
                if let (Some(path_expr), Some(update_expr)) = (path_expr, update_expr) {
                    let input = if !args.is_empty() { args[0].clone() } else { Value::Null };
                    let env = new_delegated_env(&[&path_expr, &update_expr]);
                    match crate::eval::eval_update_standalone(&path_expr, &update_expr, input, &env) {
                        Ok(v) => { std::ptr::write(dst, v); return 0; }
                        Err(e) => { set_jit_error(format!("{}", e)); std::ptr::write(dst, Value::Null); return GEN_ERROR; }
                    }
                }
            }
        }

        // __paths_filtered__:idx — native paths(f) DFS with filter
        if let Some(rest) = name.strip_prefix("__paths_filtered__:") {
            let idx: usize = rest.parse().unwrap_or(0);
            let filter_expr = JIT_CLOSURE_OPS.with(|cell| (&*cell.get()).get(idx).cloned());
            if let Some(filter_expr) = filter_expr {
                let input = if !args.is_empty() { args[0].clone() } else { Value::Null };

                // Try to compile filter to a direct fn(&Value)->bool check
                let compiled_filter: Option<fn(&Value) -> bool> = compile_type_filter(&filter_expr)
                    .or_else(|| if is_scalars_filter(&filter_expr) {
                        Some(|v: &Value| !matches!(v, Value::Arr(..) | Value::Obj(..)))
                    } else {
                        None
                    });

                // For general filters, use cached Env (auto-seeded from JIT env).
                let env = if compiled_filter.is_none() {
                    thread_local! {
                        static PATHS_FILTER_ENV: RefCell<Option<Rc<RefCell<crate::eval::Env>>>> = const { RefCell::new(None) };
                    }
                    Some(PATHS_FILTER_ENV.with(|cell| {
                        let mut opt = cell.borrow_mut();
                        let env = opt.get_or_insert_with(|| {
                            Rc::new(RefCell::new(crate::eval::Env::new(vec![])))
                        }).clone();
                        reset_delegated_env(&env, &[&filter_expr]);
                        env
                    }))
                } else {
                    None
                };

                let mut results = Vec::new();
                // DFS using mutable path stack
                fn paths_dfs_filtered(
                    val: &Value,
                    path: &mut Vec<Value>,
                    results: &mut Vec<Value>,
                    compiled_filter: Option<fn(&Value) -> bool>,
                    filter_expr: &crate::ir::Expr,
                    env: &Option<Rc<RefCell<crate::eval::Env>>>,
                ) {
                    // Check filter
                    let matched = if let Some(f) = compiled_filter {
                        f(val)
                    } else {
                        let mut m = false;
                        let _ = crate::eval::eval(filter_expr, val.clone(), env.as_ref().unwrap(), &mut |result| {
                            if result.is_truthy() { m = true; }
                            Ok(true)
                        });
                        m
                    };
                    if matched {
                        results.push(Value::Arr(Rc::new(path.clone())));
                    }
                    // Recurse into children
                    match val {
                        Value::Obj(ObjInner(o)) => {
                            for (key, child) in o.iter() {
                                path.push(Value::from_str(key.as_str()));
                                paths_dfs_filtered(child, path, results, compiled_filter, filter_expr, env);
                                path.pop();
                            }
                        }
                        Value::Arr(a) => {
                            for (i, child) in a.iter().enumerate() {
                                path.push(Value::number(i as f64));
                                paths_dfs_filtered(child, path, results, compiled_filter, filter_expr, env);
                                path.pop();
                            }
                        }
                        _ => {}
                    }
                }

                let mut path_stack = Vec::new();
                paths_dfs_filtered(&input, &mut path_stack, &mut results, compiled_filter, &filter_expr, &env);
                std::ptr::write(dst, Value::Arr(Rc::new(results)));
            } else {
                std::ptr::write(dst, Value::Arr(Rc::new(vec![])));
            }
            return 0;
        }

        // __path__:idx — runtime path expression evaluation
        if let Some(rest) = name.strip_prefix("__path__:") {
            let idx: usize = rest.parse().unwrap_or(0);
            let path_expr = JIT_CLOSURE_OPS.with(|cell| (&*cell.get()).get(idx).cloned());
            if let Some(path_expr) = path_expr {
                let input = if !args.is_empty() { args[0].clone() } else { Value::Null };
                let env = new_delegated_env(&[&path_expr]);
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
                if !args.is_empty() {
                    let container = &args[0];
                    let key_expr = JIT_CLOSURE_OPS.with(|cell| (&*cell.get()).get(idx).cloned());
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
                            // Use eval infrastructure to perform the closure op.
                            // Cache the Env to avoid 2MB allocation per call, and
                            // auto-seed any $vars referenced by `key_expr`.
                            thread_local! {
                                static CLOSURE_OP_ENV: RefCell<Option<Rc<RefCell<crate::eval::Env>>>> = const { RefCell::new(None) };
                            }
                            let env = CLOSURE_OP_ENV.with(|cell| {
                                let mut opt = cell.borrow_mut();
                                let env = opt.get_or_insert_with(|| {
                                    Rc::new(RefCell::new(crate::eval::Env::new(vec![])))
                                }).clone();
                                reset_delegated_env(&env, &[&key_expr]);
                                env
                            });
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

        // __shift_codepoints__(shift): explode | map(. + shift) | implode fused
        if name == "__shift_codepoints__" && args.len() == 2 {
            let input = &args[0];
            let shift = &args[1];
            if let (Value::Str(s), Value::Num(n, _)) = (input, shift) {
                let shift_i32 = *n as i32;
                let mut result = String::with_capacity(s.len());
                for c in s.chars() {
                    let cp = (c as i32).wrapping_add(shift_i32);
                    if cp >= 0 {
                        if let Some(ch) = char::from_u32(cp as u32) {
                            result.push(ch);
                        } else {
                            result.push('\u{FFFD}');
                        }
                    } else {
                        result.push('\u{FFFD}');
                    }
                }
                std::ptr::write(dst, Value::from_string(result));
                return 0;
            }
        }

        match crate::runtime::call_builtin(name, args) {
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
    // Always clear the flag — the error is either consumed here or already
    // drained by CheckError::get_error.
    unsafe { (*env).error_flag = 0; }
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
    let has_value = JIT_ERROR_VALUE.with(|cell| unsafe { (*cell.get()).is_some() });
    let has_msg = JIT_LAST_ERROR.with(|cell| unsafe { (*cell.get()).is_some() });
    if has_value || has_msg { 1 } else { 0 }
}

/// Get the last error as a Value and write it to dst. Clears the error.
extern "C" fn jit_rt_get_error(dst: *mut Value, env: *mut JitEnv) {
    unsafe { (*env).error_flag = 0; }
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
        68 => UnaryOp::GetModuleMeta, 69 => UnaryOp::Localtime,
        70 => UnaryOp::Sinh, 71 => UnaryOp::Cosh, 72 => UnaryOp::Tanh,
        73 => UnaryOp::Asinh, 74 => UnaryOp::Acosh, 75 => UnaryOp::Atanh,
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
        UnaryOp::GetModuleMeta => 68, UnaryOp::Localtime => 69,
        UnaryOp::Sinh => 70, UnaryOp::Cosh => 71, UnaryOp::Tanh => 72,
        UnaryOp::Asinh => 73, UnaryOp::Acosh => 74, UnaryOp::Atanh => 75,
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
    /// Non-zero means a runtime error is pending. Codegen loads this directly
    /// via `env_ptr + offset_of!(JitEnv, error_flag)` in `CheckError` /
    /// `JumpIfError`, avoiding an embedded static address.
    pub error_flag: i64,
}

impl Default for JitEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl JitEnv {
    pub fn new() -> Self {
        JitEnv {
            vars: vec![Value::Null; 65536],
            collect_stacks: Vec::new(),
            str_bufs: Vec::new(),
            try_depth: 0,
            error_msg: None,
            error_flag: 0,
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
            JitOp::FieldIsTruthy { base, .. } | JitOp::FieldCmpNum { base, .. } | JitOp::TypeCmpBranch { src: base, .. } => { *use_count.entry(*base).or_insert(0) += 1; }
            JitOp::IsNullOrFalse { src, .. } => { *use_count.entry(*src).or_insert(0) += 1; }
            JitOp::SetVar { src, .. } | JitOp::MoveToVar { src, .. } => { *use_count.entry(*src).or_insert(0) += 1; }
            JitOp::PathExtract { container, key, .. } => {
                *use_count.entry(*container).or_insert(0) += 1;
                *use_count.entry(*key).or_insert(0) += 1;
            }
            JitOp::PathInsert { container, key, val } => {
                *use_count.entry(*container).or_insert(0) += 1;
                *use_count.entry(*key).or_insert(0) += 1;
                *use_count.entry(*val).or_insert(0) += 1;
            }
            JitOp::TakeByIdx { container, .. } => { *use_count.entry(*container).or_insert(0) += 1; }
            JitOp::PutByIdx { container, val, .. } => {
                *use_count.entry(*container).or_insert(0) += 1;
                *use_count.entry(*val).or_insert(0) += 1;
            }
            JitOp::SetPathMut { container, path, val } => {
                *use_count.entry(*container).or_insert(0) += 1;
                *use_count.entry(*path).or_insert(0) += 1;
                *use_count.entry(*val).or_insert(0) += 1;
            }
            JitOp::StrBufAppendVal { src } => { *use_count.entry(*src).or_insert(0) += 1; }
            JitOp::ThrowError { msg } => { *use_count.entry(*msg).or_insert(0) += 1; }
            JitOp::MutateInplace { slot, .. } => { *use_count.entry(*slot).or_insert(0) += 1; }
            JitOp::CollectRange { n, .. } => { *use_count.entry(*n).or_insert(0) += 1; }
            JitOp::ArrPush { arr, val } => {
                *use_count.entry(*arr).or_insert(0) += 1;
                *use_count.entry(*val).or_insert(0) += 1;
            }
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
                JitOp::FieldIsTruthy { base, .. } | JitOp::FieldCmpNum { base, .. } | JitOp::TypeCmpBranch { src: base, .. } => { *use_count.entry(*base).or_insert(0) += 1; }
                JitOp::IsNullOrFalse { src, .. } => { *use_count.entry(*src).or_insert(0) += 1; }
                JitOp::SetVar { src, .. } | JitOp::MoveToVar { src, .. } => { *use_count.entry(*src).or_insert(0) += 1; }
                JitOp::PathExtract { container, key, .. } => {
                    *use_count.entry(*container).or_insert(0) += 1;
                    *use_count.entry(*key).or_insert(0) += 1;
                }
                JitOp::PathInsert { container, key, val } => {
                    *use_count.entry(*container).or_insert(0) += 1;
                    *use_count.entry(*key).or_insert(0) += 1;
                    *use_count.entry(*val).or_insert(0) += 1;
                }
                JitOp::TakeByIdx { container, .. } => { *use_count.entry(*container).or_insert(0) += 1; }
                JitOp::PutByIdx { container, val, .. } => {
                    *use_count.entry(*container).or_insert(0) += 1;
                    *use_count.entry(*val).or_insert(0) += 1;
                }
                JitOp::SetPathMut { container, path, val } => {
                    *use_count.entry(*container).or_insert(0) += 1;
                    *use_count.entry(*path).or_insert(0) += 1;
                    *use_count.entry(*val).or_insert(0) += 1;
                }
                JitOp::StrBufAppendVal { src } => { *use_count.entry(*src).or_insert(0) += 1; }
                JitOp::ThrowError { msg } => { *use_count.entry(*msg).or_insert(0) += 1; }
                JitOp::MutateInplace { slot, .. } => { *use_count.entry(*slot).or_insert(0) += 1; }
                JitOp::CollectRange { n, .. } => { *use_count.entry(*n).or_insert(0) += 1; }
                JitOp::ArrPush { arr, val } => {
                    *use_count.entry(*arr).or_insert(0) += 1;
                    *use_count.entry(*val).or_insert(0) += 1;
                }
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
            ("jit_rt_move_to_var", jit_rt_move_to_var as *const u8),
            ("jit_rt_path_extract", jit_rt_path_extract as *const u8),
            ("jit_rt_path_insert", jit_rt_path_insert as *const u8),
            ("jit_rt_take_by_idx", jit_rt_take_by_idx as *const u8),
            ("jit_rt_put_by_idx", jit_rt_put_by_idx as *const u8),
            ("jit_rt_setpath_mut", jit_rt_setpath_mut as *const u8),
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
            ("jit_rt_collect_range", jit_rt_collect_range as *const u8),
            ("jit_rt_arr_push", jit_rt_arr_push as *const u8),
            ("jit_rt_libm_sin", sys_sin as *const u8),
            ("jit_rt_libm_cos", sys_cos as *const u8),
            ("jit_rt_libm_tan", sys_tan as *const u8),
            ("jit_rt_libm_asin", sys_asin as *const u8),
            ("jit_rt_libm_acos", sys_acos as *const u8),
            ("jit_rt_libm_atan", sys_atan as *const u8),
            ("jit_rt_libm_exp", sys_exp as *const u8),
            ("jit_rt_libm_exp2", sys_exp2 as *const u8),
            ("jit_rt_libm_log", sys_log as *const u8),
            ("jit_rt_libm_log2", sys_log2 as *const u8),
            ("jit_rt_libm_log10", sys_log10 as *const u8),
            ("jit_rt_libm_cbrt", sys_cbrt as *const u8),
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
        // Pre-inline function calls and apply semantic optimizations.
        // Always run to catch to_entries|from_entries and similar rewrites.
        let inlined = fl.inline_func_calls(expr);
        let compile_expr = &inlined;
        let input_slot = fl.alloc_slot(); // slot 0 = input ptr (read-only, owned by caller)
        // Use input_slot directly — flatten_gen only reads it, never writes/drops it.
        // The codegen handles Drop{slot:0} as a no-op.
        if !fl.flatten_gen(compile_expr, input_slot) {
            bail!("Expression not JIT-compilable");
        }
        fl.emit(JitOp::ReturnContinue);

        // Store closure ops for runtime access
        if !fl.closure_ops.is_empty() {
            JIT_CLOSURE_OPS.with(|cell| unsafe { *cell.get() = fl.closure_ops.clone(); });
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
                        // Inline trivial clone: check tag to avoid function call
                        let s = slot_addr(&mut b, *src);
                        let tag = b.ins().load(types::I8, cranelift_codegen::ir::MemFlags::new(), s, 0);
                        let tag_i32 = b.ins().uextend(types::I32, tag);
                        let three = b.ins().iconst(types::I32, 3);
                        let is_trivial = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan, tag_i32, three);

                        let copy_tag_blk = b.create_block();
                        let check_num_blk = b.create_block();
                        let call_clone_blk = b.create_block();
                        let done_blk = b.create_block();

                        // tag < 3 (Null/False/True): copy tag byte only
                        b.ins().brif(is_trivial, copy_tag_blk, &[], check_num_blk, &[]);

                        // copy_tag_blk: copy tag word (i64 at offset 0) for Null/False/True
                        b.switch_to_block(copy_tag_blk);
                        b.seal_block(copy_tag_blk);
                        let d1 = slot_addr(&mut b, *dst);
                        let s1 = slot_addr(&mut b, *src);
                        let tag_word = b.ins().load(ptr_ty, cranelift_codegen::ir::MemFlags::new(), s1, 0);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), tag_word, d1, 0);
                        b.ins().jump(done_blk, &[]);

                        // check tag == 3 (Num)
                        b.switch_to_block(check_num_blk);
                        b.seal_block(check_num_blk);
                        let is_num = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, tag_i32, three);
                        let check_repr_blk = b.create_block();
                        let check_str_clone_blk = b.create_block();
                        b.ins().brif(is_num, check_repr_blk, &[], check_str_clone_blk, &[]);

                        // Num: check repr at offset 16
                        b.switch_to_block(check_repr_blk);
                        b.seal_block(check_repr_blk);
                        let s_r = slot_addr(&mut b, *src);
                        let repr = b.ins().load(ptr_ty, cranelift_codegen::ir::MemFlags::new(), s_r, 16);
                        let zero = b.ins().iconst(ptr_ty, 0);
                        let repr_null = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, repr, zero);
                        let copy_num_blk = b.create_block();
                        b.ins().brif(repr_null, copy_num_blk, &[], call_clone_blk, &[]);

                        // check_str_clone_blk: tag >= 4, check if Str with inline CompactString
                        b.switch_to_block(check_str_clone_blk);
                        b.seal_block(check_str_clone_blk);
                        let four = b.ins().iconst(types::I32, 4);
                        let is_str = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, tag_i32, four);
                        let str_check_inline_blk = b.create_block();
                        b.ins().brif(is_str, str_check_inline_blk, &[], call_clone_blk, &[]);

                        // str_check_inline_blk: tag == 4, check CompactString last byte at offset 31
                        b.switch_to_block(str_check_inline_blk);
                        b.seal_block(str_check_inline_blk);
                        let s_str = slot_addr(&mut b, *src);
                        let last_byte = b.ins().load(types::I8, cranelift_codegen::ir::MemFlags::new(), s_str, 31);
                        let last_u32 = b.ins().uextend(types::I32, last_byte);
                        let heap_mask = b.ins().iconst(types::I32, 216);
                        let is_inline_str = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan, last_u32, heap_mask);
                        let copy_str_blk = b.create_block();
                        b.ins().brif(is_inline_str, copy_str_blk, &[], call_clone_blk, &[]);

                        // copy_str_blk: inline Str clone — copy all 32 bytes (tag + CompactString)
                        b.switch_to_block(copy_str_blk);
                        b.seal_block(copy_str_blk);
                        let d_str = slot_addr(&mut b, *dst);
                        let s_str2 = slot_addr(&mut b, *src);
                        let sw0 = b.ins().load(ptr_ty, cranelift_codegen::ir::MemFlags::new(), s_str2, 0);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), sw0, d_str, 0);
                        let sw1 = b.ins().load(ptr_ty, cranelift_codegen::ir::MemFlags::new(), s_str2, 8);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), sw1, d_str, 8);
                        let sw2 = b.ins().load(ptr_ty, cranelift_codegen::ir::MemFlags::new(), s_str2, 16);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), sw2, d_str, 16);
                        let sw3 = b.ins().load(ptr_ty, cranelift_codegen::ir::MemFlags::new(), s_str2, 24);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), sw3, d_str, 24);
                        b.ins().jump(done_blk, &[]);

                        // copy_num_blk: inline Num(f64, None) clone — copy all 32 bytes
                        b.switch_to_block(copy_num_blk);
                        b.seal_block(copy_num_blk);
                        let d2 = slot_addr(&mut b, *dst);
                        let s2 = slot_addr(&mut b, *src);
                        let w0 = b.ins().load(ptr_ty, cranelift_codegen::ir::MemFlags::new(), s2, 0);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), w0, d2, 0);
                        let w1 = b.ins().load(ptr_ty, cranelift_codegen::ir::MemFlags::new(), s2, 8);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), w1, d2, 8);
                        let z = b.ins().iconst(ptr_ty, 0);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), z, d2, 16);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), z, d2, 24);
                        b.ins().jump(done_blk, &[]);

                        // call_clone_blk: non-trivial, call runtime
                        b.switch_to_block(call_clone_blk);
                        b.seal_block(call_clone_blk);
                        let d3 = slot_addr(&mut b, *dst);
                        let s3 = slot_addr(&mut b, *src);
                        b.ins().call(rt["clone"], &[d3, s3]);
                        b.ins().jump(done_blk, &[]);

                        // done_blk: continue
                        b.switch_to_block(done_blk);
                        b.seal_block(done_blk);
                        terminated = false;
                    }
                    JitOp::LoadConst { dst, const_ptr } => {
                        // Clone a pre-computed constant Value into dst
                        let d = slot_addr(&mut b, *dst);
                        let s = b.ins().iconst(ptr_ty, *const_ptr as i64);
                        b.ins().call(rt["clone"], &[d, s]);
                    }
                    JitOp::Drop { slot } if *slot == 0 => { /* don't drop input */ }
                    JitOp::Drop { slot } => {
                        // Inline trivial drop: check discriminant tag to skip function call
                        // Value layout: byte 0 = tag (0=Null,1=False,2=True,3=Num,4+=heap)
                        // Tags 0-2: no-op drop. Tag 3: no-op if repr (offset 16) is null.
                        let a = slot_addr(&mut b, *slot);
                        let tag = b.ins().load(types::I8, cranelift_codegen::ir::MemFlags::new(), a, 0);
                        let tag_i32 = b.ins().uextend(types::I32, tag);
                        let three = b.ins().iconst(types::I32, 3);
                        let is_le_2 = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan, tag_i32, three);

                        let check_num_blk = b.create_block();
                        let do_drop_blk = b.create_block();
                        let skip_blk = b.create_block();

                        // tag < 3 (Null/False/True) → skip
                        b.ins().brif(is_le_2, skip_blk, &[], check_num_blk, &[]);

                        // check_num_blk: tag >= 3
                        b.switch_to_block(check_num_blk);
                        b.seal_block(check_num_blk);
                        let is_eq_3 = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, tag_i32, three);
                        let repr_ptr_blk = b.create_block();
                        let check_str_blk = b.create_block();
                        b.ins().brif(is_eq_3, repr_ptr_blk, &[], check_str_blk, &[]);

                        // repr_ptr_blk: tag == 3 (Num), check repr at offset 16
                        b.switch_to_block(repr_ptr_blk);
                        b.seal_block(repr_ptr_blk);
                        let repr = b.ins().load(ptr_ty, cranelift_codegen::ir::MemFlags::new(), a, 16);
                        let zero = b.ins().iconst(ptr_ty, 0);
                        let repr_is_null = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, repr, zero);
                        b.ins().brif(repr_is_null, skip_blk, &[], do_drop_blk, &[]);

                        // check_str_blk: tag >= 4, check if Str with inline CompactString
                        b.switch_to_block(check_str_blk);
                        b.seal_block(check_str_blk);
                        let four = b.ins().iconst(types::I32, 4);
                        let is_eq_4 = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, tag_i32, four);
                        let str_inline_blk = b.create_block();
                        b.ins().brif(is_eq_4, str_inline_blk, &[], do_drop_blk, &[]);

                        // str_inline_blk: tag == 4 (Str), check CompactString last byte
                        // CompactString occupies bytes 8-31. Last byte at offset 31.
                        // If last_byte < 216 (HEAP_MASK), string is inline → no-op drop.
                        b.switch_to_block(str_inline_blk);
                        b.seal_block(str_inline_blk);
                        let a_str = slot_addr(&mut b, *slot);
                        let last_byte = b.ins().load(types::I8, cranelift_codegen::ir::MemFlags::new(), a_str, 31);
                        let last_byte_u32 = b.ins().uextend(types::I32, last_byte);
                        let heap_mask = b.ins().iconst(types::I32, 216); // HEAP_MASK
                        let is_inline = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan, last_byte_u32, heap_mask);
                        b.ins().brif(is_inline, skip_blk, &[], do_drop_blk, &[]);

                        // do_drop_blk: call jit_rt_drop
                        b.switch_to_block(do_drop_blk);
                        b.seal_block(do_drop_blk);
                        let a2 = slot_addr(&mut b, *slot);
                        b.ins().call(rt["drop"], &[a2]);
                        b.ins().jump(skip_blk, &[]);

                        // skip_blk: continue
                        b.switch_to_block(skip_blk);
                        b.seal_block(skip_blk);
                        terminated = false;
                    }
                    JitOp::Null { dst } => {
                        // Inline: store tag word 0 (Null) — full i64 includes padding
                        let a = slot_addr(&mut b, *dst);
                        let tag_word = b.ins().iconst(ptr_ty, 0);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), tag_word, a, 0);
                    }
                    JitOp::True { dst } => {
                        // Inline: store tag word 2 (True)
                        let a = slot_addr(&mut b, *dst);
                        let tag_word = b.ins().iconst(ptr_ty, 2);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), tag_word, a, 0);
                    }
                    JitOp::False { dst } => {
                        // Inline: store tag word 1 (False)
                        let a = slot_addr(&mut b, *dst);
                        let tag_word = b.ins().iconst(ptr_ty, 1);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), tag_word, a, 0);
                    }
                    JitOp::Num { dst, val, repr } => {
                        let a = slot_addr(&mut b, *dst);
                        if let Some(r) = repr {
                            let n = b.ins().f64const(*val);
                            let boxed = Box::new(r.clone());
                            let rp = b.ins().iconst(ptr_ty, &*boxed as *const Rc<str> as i64);
                            self._repr_constants.push(boxed);
                            b.ins().call(rt["num_repr"], &[a, n, rp]);
                        } else {
                            // Inline: tag_word=3, f64 at +8, repr=0 at +16, len=0 at +24
                            let tag_word = b.ins().iconst(ptr_ty, 3);
                            b.ins().store(cranelift_codegen::ir::MemFlags::new(), tag_word, a, 0);
                            let n = b.ins().f64const(*val);
                            b.ins().store(cranelift_codegen::ir::MemFlags::new(), n, a, 8);
                            let zero = b.ins().iconst(ptr_ty, 0);
                            b.ins().store(cranelift_codegen::ir::MemFlags::new(), zero, a, 16);
                            b.ins().store(cranelift_codegen::ir::MemFlags::new(), zero, a, 24);
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
                        // Inline null fast path: null.field → null
                        let ba = slot_addr(&mut b, *base);
                        let tag = b.ins().load(types::I8, cranelift_codegen::ir::MemFlags::new(), ba, 0);
                        let tag_i32 = b.ins().uextend(types::I32, tag);
                        let zero_i32 = b.ins().iconst(types::I32, 0);
                        let is_null = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, tag_i32, zero_i32);
                        let fast_blk = b.create_block();
                        let slow_blk = b.create_block();
                        let done_blk = b.create_block();
                        b.ins().brif(is_null, fast_blk, &[], slow_blk, &[]);

                        b.switch_to_block(fast_blk);
                        b.seal_block(fast_blk);
                        let d2 = slot_addr(&mut b, *dst);
                        let null_word = b.ins().iconst(ptr_ty, 0);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), null_word, d2, 0);
                        b.ins().jump(done_blk, &[]);

                        b.switch_to_block(slow_blk);
                        b.seal_block(slow_blk);
                        let d3 = slot_addr(&mut b, *dst);
                        let ba3 = slot_addr(&mut b, *base);
                        let leaked = Box::leak(field.clone().into_boxed_str());
                        self._string_constants.push(leaked);
                        let fp = b.ins().iconst(ptr_ty, leaked.as_ptr() as i64);
                        let fl = b.ins().iconst(ptr_ty, leaked.len() as i64);
                        b.ins().call(rt["index_field"], &[d3, ba3, fp, fl]);
                        b.ins().jump(done_blk, &[]);

                        b.switch_to_block(done_blk);
                        b.seal_block(done_blk);
                    }
                    JitOp::BinOp { dst, op, lhs, rhs } => {
                        // Inline fast path for Num+Num arithmetic and comparison
                        let inline_kind: Option<u8> = match op {
                            BinOp::Add => Some(0), BinOp::Sub => Some(1), BinOp::Mul => Some(2),
                            BinOp::Div => Some(3),
                            BinOp::Eq => Some(10), BinOp::Ne => Some(11),
                            BinOp::Lt => Some(12), BinOp::Gt => Some(13),
                            BinOp::Le => Some(14), BinOp::Ge => Some(15),
                            _ => None,
                        };
                        if let Some(kind) = inline_kind {
                            let l = slot_addr(&mut b, *lhs);
                            let r = slot_addr(&mut b, *rhs);
                            let lt = b.ins().load(types::I8, cranelift_codegen::ir::MemFlags::new(), l, 0);
                            let rt_tag = b.ins().load(types::I8, cranelift_codegen::ir::MemFlags::new(), r, 0);
                            let lt32 = b.ins().uextend(types::I32, lt);
                            let rt32 = b.ins().uextend(types::I32, rt_tag);
                            let three = b.ins().iconst(types::I32, 3);
                            let l_is_num = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, lt32, three);
                            let r_is_num = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, rt32, three);
                            let both_num = b.ins().band(l_is_num, r_is_num);

                            let fast_blk = b.create_block();
                            let slow_blk = b.create_block();
                            let done_blk = b.create_block();
                            b.ins().brif(both_num, fast_blk, &[], slow_blk, &[]);

                            b.switch_to_block(fast_blk);
                            b.seal_block(fast_blk);
                            let l2 = slot_addr(&mut b, *lhs);
                            let r2 = slot_addr(&mut b, *rhs);
                            let lf = b.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), l2, 8);
                            let rf = b.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), r2, 8);
                            let d2 = slot_addr(&mut b, *dst);
                            if kind == 3 {
                                // Div: need to check for zero/NaN divisor
                                let fzero = b.ins().f64const(0.0);
                                let rhs_is_zero = b.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::Equal, rf, fzero);
                                let rhs_is_nan = b.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::Unordered, rf, rf);
                                let lhs_is_nan = b.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::Unordered, lf, lf);
                                let bad = b.ins().bor(rhs_is_zero, rhs_is_nan);
                                let bad = b.ins().bor(bad, lhs_is_nan);
                                let div_ok_blk = b.create_block();
                                b.ins().brif(bad, slow_blk, &[], div_ok_blk, &[]);
                                b.switch_to_block(div_ok_blk);
                                b.seal_block(div_ok_blk);
                                let l3 = slot_addr(&mut b, *lhs);
                                let r3 = slot_addr(&mut b, *rhs);
                                let lf2 = b.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), l3, 8);
                                let rf2 = b.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), r3, 8);
                                let result = b.ins().fdiv(lf2, rf2);
                                let d3 = slot_addr(&mut b, *dst);
                                let tag_word = b.ins().iconst(ptr_ty, 3);
                                b.ins().store(cranelift_codegen::ir::MemFlags::new(), tag_word, d3, 0);
                                b.ins().store(cranelift_codegen::ir::MemFlags::new(), result, d3, 8);
                                let zero = b.ins().iconst(ptr_ty, 0);
                                b.ins().store(cranelift_codegen::ir::MemFlags::new(), zero, d3, 16);
                                b.ins().store(cranelift_codegen::ir::MemFlags::new(), zero, d3, 24);
                            } else if kind < 10 {
                                // Arithmetic: result is f64
                                let result = match kind {
                                    0 => b.ins().fadd(lf, rf),
                                    1 => b.ins().fsub(lf, rf),
                                    2 => b.ins().fmul(lf, rf),
                                    _ => unreachable!(),
                                };
                                let tag_word = b.ins().iconst(ptr_ty, 3);
                                b.ins().store(cranelift_codegen::ir::MemFlags::new(), tag_word, d2, 0);
                                b.ins().store(cranelift_codegen::ir::MemFlags::new(), result, d2, 8);
                                let zero = b.ins().iconst(ptr_ty, 0);
                                b.ins().store(cranelift_codegen::ir::MemFlags::new(), zero, d2, 16);
                                b.ins().store(cranelift_codegen::ir::MemFlags::new(), zero, d2, 24);
                            } else {
                                // Comparison: result is True/False
                                use cranelift_codegen::ir::condcodes::FloatCC;
                                let cc = match kind {
                                    10 => FloatCC::Equal, 11 => FloatCC::NotEqual,
                                    12 => FloatCC::LessThan, 13 => FloatCC::GreaterThan,
                                    14 => FloatCC::LessThanOrEqual, 15 => FloatCC::GreaterThanOrEqual,
                                    _ => unreachable!(),
                                };
                                let cmp = b.ins().fcmp(cc, lf, rf);
                                // True=2, False=1
                                let one = b.ins().iconst(ptr_ty, 1);
                                let two = b.ins().iconst(ptr_ty, 2);
                                let tag_word = b.ins().select(cmp, two, one);
                                b.ins().store(cranelift_codegen::ir::MemFlags::new(), tag_word, d2, 0);
                            }
                            b.ins().jump(done_blk, &[]);

                            b.switch_to_block(slow_blk);
                            b.seal_block(slow_blk);
                            let d3 = slot_addr(&mut b, *dst);
                            let l3 = slot_addr(&mut b, *lhs);
                            let r3 = slot_addr(&mut b, *rhs);
                            let o = b.ins().iconst(types::I32, binop_to_i32(*op) as i64);
                            b.ins().call(rt["binop"], &[d3, o, l3, r3]);
                            b.ins().jump(done_blk, &[]);

                            b.switch_to_block(done_blk);
                            b.seal_block(done_blk);
                        } else {
                            let d = slot_addr(&mut b, *dst);
                            let l = slot_addr(&mut b, *lhs);
                            let r = slot_addr(&mut b, *rhs);
                            let o = b.ins().iconst(types::I32, binop_to_i32(*op) as i64);
                            b.ins().call(rt["binop"], &[d, o, l, r]);
                        }
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
                        // Inline for Num inputs: floor, ceil, sqrt, fabs, abs, trunc
                        let inlineable = matches!(op,
                            UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Sqrt |
                            UnaryOp::Fabs | UnaryOp::Abs | UnaryOp::Trunc);
                        if inlineable {
                            let s = slot_addr(&mut b, *src);
                            let tag = b.ins().load(types::I8, cranelift_codegen::ir::MemFlags::new(), s, 0);
                            let tag_i32 = b.ins().uextend(types::I32, tag);
                            let three = b.ins().iconst(types::I32, 3);
                            let is_num = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, tag_i32, three);
                            // Also check repr ptr at offset 16 is null (no string repr to preserve)
                            let repr_ptr = b.ins().load(ptr_ty, cranelift_codegen::ir::MemFlags::new(), s, 16);
                            let zero_ptr = b.ins().iconst(ptr_ty, 0);
                            let no_repr = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, repr_ptr, zero_ptr);
                            let can_inline = b.ins().band(is_num, no_repr);
                            let fast_blk = b.create_block();
                            let slow_blk = b.create_block();
                            let done_blk = b.create_block();
                            b.ins().brif(can_inline, fast_blk, &[], slow_blk, &[]);

                            b.switch_to_block(fast_blk);
                            b.seal_block(fast_blk);
                            let s2 = slot_addr(&mut b, *src);
                            let d2 = slot_addr(&mut b, *dst);
                            let f_val = b.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), s2, 8);
                            let result = match op {
                                UnaryOp::Floor => b.ins().floor(f_val),
                                UnaryOp::Ceil => b.ins().ceil(f_val),
                                UnaryOp::Sqrt => b.ins().sqrt(f_val),
                                UnaryOp::Fabs | UnaryOp::Abs => b.ins().fabs(f_val),
                                UnaryOp::Trunc => b.ins().trunc(f_val),
                                _ => unreachable!(),
                            };
                            let tag_word = b.ins().iconst(ptr_ty, 3);
                            b.ins().store(cranelift_codegen::ir::MemFlags::new(), tag_word, d2, 0);
                            b.ins().store(cranelift_codegen::ir::MemFlags::new(), result, d2, 8);
                            let zero = b.ins().iconst(ptr_ty, 0);
                            b.ins().store(cranelift_codegen::ir::MemFlags::new(), zero, d2, 16);
                            b.ins().store(cranelift_codegen::ir::MemFlags::new(), zero, d2, 24);
                            b.ins().jump(done_blk, &[]);

                            b.switch_to_block(slow_blk);
                            b.seal_block(slow_blk);
                            let d3 = slot_addr(&mut b, *dst);
                            let s3 = slot_addr(&mut b, *src);
                            let o = b.ins().iconst(types::I32, unaryop_to_i32(*op) as i64);
                            b.ins().call(rt["unaryop"], &[d3, o, s3]);
                            b.ins().jump(done_blk, &[]);

                            b.switch_to_block(done_blk);
                            b.seal_block(done_blk);
                        } else {
                            let d = slot_addr(&mut b, *dst);
                            let s = slot_addr(&mut b, *src);
                            let o = b.ins().iconst(types::I32, unaryop_to_i32(*op) as i64);
                            b.ins().call(rt["unaryop"], &[d, o, s]);
                        }
                    }
                    JitOp::Negate { dst, src } => {
                        // Inline for Num: fneg
                        let s = slot_addr(&mut b, *src);
                        let tag = b.ins().load(types::I8, cranelift_codegen::ir::MemFlags::new(), s, 0);
                        let tag_i32 = b.ins().uextend(types::I32, tag);
                        let three = b.ins().iconst(types::I32, 3);
                        let is_num = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, tag_i32, three);
                        let fast_blk = b.create_block();
                        let slow_blk = b.create_block();
                        let done_blk = b.create_block();
                        b.ins().brif(is_num, fast_blk, &[], slow_blk, &[]);

                        b.switch_to_block(fast_blk);
                        b.seal_block(fast_blk);
                        let s2 = slot_addr(&mut b, *src);
                        let d2 = slot_addr(&mut b, *dst);
                        let f_val = b.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), s2, 8);
                        let neg = b.ins().fneg(f_val);
                        let tag_word = b.ins().iconst(ptr_ty, 3);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), tag_word, d2, 0);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), neg, d2, 8);
                        let zero = b.ins().iconst(ptr_ty, 0);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), zero, d2, 16);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), zero, d2, 24);
                        b.ins().jump(done_blk, &[]);

                        b.switch_to_block(slow_blk);
                        b.seal_block(slow_blk);
                        let d3 = slot_addr(&mut b, *dst);
                        let s3 = slot_addr(&mut b, *src);
                        b.ins().call(rt["negate"], &[d3, s3]);
                        b.ins().jump(done_blk, &[]);

                        b.switch_to_block(done_blk);
                        b.seal_block(done_blk);
                    }
                    JitOp::Not { dst, src } => {
                        // Inline: null(0)/false(1) → True(2), else → False(1)
                        let s = slot_addr(&mut b, *src);
                        let d = slot_addr(&mut b, *dst);
                        let tag = b.ins().load(types::I8, cranelift_codegen::ir::MemFlags::new(), s, 0);
                        let tag_i32 = b.ins().uextend(types::I32, tag);
                        let two = b.ins().iconst(types::I32, 2);
                        let is_truthy = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::UnsignedGreaterThanOrEqual, tag_i32, two);
                        // truthy → write False(1), falsy → write True(2)
                        let one = b.ins().iconst(ptr_ty, 1);
                        let two_64 = b.ins().iconst(ptr_ty, 2);
                        let result = b.ins().select(is_truthy, one, two_64);
                        b.ins().store(cranelift_codegen::ir::MemFlags::new(), result, d, 0);
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
                        // Inline: null(0) and false(1) are falsy, everything else truthy
                        let sa = slot_addr(&mut b, *src);
                        let tag = b.ins().load(types::I8, cranelift_codegen::ir::MemFlags::new(), sa, 0);
                        let tag_i32 = b.ins().uextend(types::I32, tag);
                        let two = b.ins().iconst(types::I32, 2);
                        let is_truthy = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::UnsignedGreaterThanOrEqual, tag_i32, two);
                        b.ins().brif(is_truthy, label_blocks[*then_label as usize], &[], label_blocks[*else_label as usize], &[]);
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
                    JitOp::TypeCmpBranch { src, tags, then_label, else_label } => {
                        // Load tag byte (offset 0) from Value at slot
                        let sa = slot_addr(&mut b, *src);
                        let tag_val = b.ins().load(types::I8, cranelift_codegen::ir::MemFlags::new(), sa, 0);
                        let tag_i64 = b.ins().uextend(ptr_ty, tag_val);
                        // Build bitmask check: (1 << tag) & tags != 0
                        let one = b.ins().iconst(ptr_ty, 1);
                        let bit = b.ins().ishl(one, tag_i64);
                        let mask = b.ins().iconst(ptr_ty, *tags as i64);
                        let result = b.ins().band(bit, mask);
                        let zero = b.ins().iconst(ptr_ty, 0);
                        let is_match = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, result, zero);
                        b.ins().brif(is_match, label_blocks[*then_label as usize], &[], label_blocks[*else_label as usize], &[]);
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
                    JitOp::MoveToVar { var_index, src } => {
                        let s = slot_addr(&mut b, *src);
                        let vi = b.ins().iconst(types::I32, *var_index as i64);
                        b.ins().call(rt["move_to_var"], &[env_ptr, vi, s]);
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
                    JitOp::TakeByIdx { dst, container, idx_var } => {
                        let d = slot_addr(&mut b, *dst);
                        let c = slot_addr(&mut b, *container);
                        let i = b.use_var(vars[*idx_var as usize]);
                        b.ins().call(rt["take_by_idx"], &[d, c, i]);
                    }
                    JitOp::PutByIdx { container, idx_var, val } => {
                        let c = slot_addr(&mut b, *container);
                        let i = b.use_var(vars[*idx_var as usize]);
                        let v = slot_addr(&mut b, *val);
                        b.ins().call(rt["put_by_idx"], &[c, i, v]);
                    }
                    JitOp::SetPathMut { container, path, val } => {
                        let c = slot_addr(&mut b, *container);
                        let p = slot_addr(&mut b, *path);
                        let v = slot_addr(&mut b, *val);
                        b.ins().call(rt["setpath_mut"], &[c, p, v]);
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
                        let call = b.ins().call(rt["obj_insert"], &[o, k, v]);
                        let status = b.inst_results(call)[0];
                        let zero = b.ins().iconst(ptr_ty, 0);
                        let is_err = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, status, zero);
                        let err_blk = b.create_block();
                        let ok_blk = b.create_block();
                        b.ins().brif(is_err, err_blk, &[], ok_blk, &[]);
                        b.switch_to_block(err_blk);
                        b.seal_block(err_blk);
                        b.ins().call(rt["propagate_error"], &[env_ptr]);
                        let gerr = b.ins().iconst(ptr_ty, GEN_ERROR);
                        b.ins().return_(&[gerr]);
                        b.switch_to_block(ok_blk);
                        b.seal_block(ok_blk);
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
                        // Inline: null(0) or false(1) → 1, else → 0
                        let s = slot_addr(&mut b, *src);
                        let tag = b.ins().load(types::I8, cranelift_codegen::ir::MemFlags::new(), s, 0);
                        let tag_i32 = b.ins().uextend(types::I32, tag);
                        let two = b.ins().iconst(types::I32, 2);
                        let is_nf = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan, tag_i32, two);
                        let result = b.ins().uextend(ptr_ty, is_nf);
                        b.def_var(vars[*dst_var as usize], result);
                    }
                    JitOp::BranchOnVar { var, nonzero_label, zero_label } => {
                        let v = b.use_var(vars[*var as usize]);
                        let zero = b.ins().iconst(ptr_ty, 0);
                        let cmp = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, v, zero);
                        b.ins().brif(cmp, label_blocks[*nonzero_label as usize], &[], label_blocks[*zero_label as usize], &[]);
                        terminated = true;
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
                    JitOp::F64Add { dst_var, a_var, b_var: bv } => {
                        let a_bits = b.use_var(vars[*a_var as usize]);
                        let b_bits = b.use_var(vars[*bv as usize]);
                        let a_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), a_bits);
                        let b_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), b_bits);
                        let sum = b.ins().fadd(a_f, b_f);
                        let bits = b.ins().bitcast(ptr_ty, cranelift_codegen::ir::MemFlags::new(), sum);
                        b.def_var(vars[*dst_var as usize], bits);
                    }
                    JitOp::F64Mul { dst_var, a_var, b_var: bv } => {
                        let a_bits = b.use_var(vars[*a_var as usize]);
                        let b_bits = b.use_var(vars[*bv as usize]);
                        let a_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), a_bits);
                        let b_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), b_bits);
                        let prod = b.ins().fmul(a_f, b_f);
                        let bits = b.ins().bitcast(ptr_ty, cranelift_codegen::ir::MemFlags::new(), prod);
                        b.def_var(vars[*dst_var as usize], bits);
                    }
                    JitOp::F64Sub { dst_var, a_var, b_var: bv } => {
                        let a_bits = b.use_var(vars[*a_var as usize]);
                        let b_bits = b.use_var(vars[*bv as usize]);
                        let a_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), a_bits);
                        let b_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), b_bits);
                        let diff = b.ins().fsub(a_f, b_f);
                        let bits = b.ins().bitcast(ptr_ty, cranelift_codegen::ir::MemFlags::new(), diff);
                        b.def_var(vars[*dst_var as usize], bits);
                    }
                    JitOp::F64Rem { dst_var, a_var, b_var: bv } => {
                        // jq's `%` truncates both operands toward zero before the
                        // remainder, so 3.7 % 2 = 1 and 1 % 1.5 = 0 (#183). We
                        // compute trunc(a) - trunc(trunc(a)/trunc(b)) * trunc(b)
                        // to keep the result an integer with the dividend's sign.
                        let a_bits = b.use_var(vars[*a_var as usize]);
                        let b_bits = b.use_var(vars[*bv as usize]);
                        let a_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), a_bits);
                        let b_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), b_bits);
                        let a_t = b.ins().trunc(a_f);
                        let b_t = b.ins().trunc(b_f);
                        let div = b.ins().fdiv(a_t, b_t);
                        let q = b.ins().trunc(div);
                        let prod = b.ins().fmul(q, b_t);
                        let rem_f = b.ins().fsub(a_t, prod);
                        let bits = b.ins().bitcast(ptr_ty, cranelift_codegen::ir::MemFlags::new(), rem_f);
                        b.def_var(vars[*dst_var as usize], bits);
                    }
                    JitOp::F64Div { dst_var, a_var, b_var: bv } => {
                        let a_bits = b.use_var(vars[*a_var as usize]);
                        let b_bits = b.use_var(vars[*bv as usize]);
                        let a_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), a_bits);
                        let b_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), b_bits);
                        let quot = b.ins().fdiv(a_f, b_f);
                        let bits = b.ins().bitcast(ptr_ty, cranelift_codegen::ir::MemFlags::new(), quot);
                        b.def_var(vars[*dst_var as usize], bits);
                    }
                    JitOp::F64Neg { dst_var, src_var } => {
                        let src_bits = b.use_var(vars[*src_var as usize]);
                        let src_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), src_bits);
                        let neg = b.ins().fneg(src_f);
                        let bits = b.ins().bitcast(ptr_ty, cranelift_codegen::ir::MemFlags::new(), neg);
                        b.def_var(vars[*dst_var as usize], bits);
                    }
                    JitOp::F64Math { dst_var, src_var, kind } => {
                        let src_bits = b.use_var(vars[*src_var as usize]);
                        let src_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), src_bits);
                        let result = match kind {
                            0 => b.ins().floor(src_f),
                            1 => b.ins().ceil(src_f),
                            2 => b.ins().sqrt(src_f),
                            3 => b.ins().fabs(src_f),
                            4 => b.ins().trunc(src_f),
                            5 => b.ins().nearest(src_f),
                            _ => unreachable!(),
                        };
                        let bits = b.ins().bitcast(ptr_ty, cranelift_codegen::ir::MemFlags::new(), result);
                        b.def_var(vars[*dst_var as usize], bits);
                    }
                    JitOp::F64Libm { dst_var, src_var, func } => {
                        let src_bits = b.use_var(vars[*src_var as usize]);
                        let src_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), src_bits);
                        let fname = match func {
                            0 => "libm_sin", 1 => "libm_cos", 2 => "libm_tan",
                            3 => "libm_asin", 4 => "libm_acos", 5 => "libm_atan",
                            6 => "libm_exp", 7 => "libm_exp2",
                            8 => "libm_log", 9 => "libm_log2", 10 => "libm_log10",
                            11 => "libm_cbrt",
                            _ => unreachable!(),
                        };
                        let call = b.ins().call(rt[fname], &[src_f]);
                        let result = b.inst_results(call)[0];
                        let bits = b.ins().bitcast(ptr_ty, cranelift_codegen::ir::MemFlags::new(), result);
                        b.def_var(vars[*dst_var as usize], bits);
                    }
                    JitOp::F64Cmp { dst_var, a_var, b_var: bv, cc } => {
                        use cranelift_codegen::ir::condcodes::FloatCC;
                        let a_bits = b.use_var(vars[*a_var as usize]);
                        let b_bits = b.use_var(vars[*bv as usize]);
                        let a_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), a_bits);
                        let b_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), b_bits);
                        let fcc = match cc {
                            0 => FloatCC::GreaterThanOrEqual,
                            1 => FloatCC::GreaterThan,
                            2 => FloatCC::LessThanOrEqual,
                            3 => FloatCC::LessThan,
                            4 => FloatCC::Equal,
                            _ => FloatCC::NotEqual,
                        };
                        let cmp = b.ins().fcmp(fcc, a_f, b_f);
                        let result = b.ins().uextend(ptr_ty, cmp);
                        b.def_var(vars[*dst_var as usize], result);
                    }
                    JitOp::RangeCheck { dst_var, cur_var, to_var, step_var } => {
                        // Inline range check: step>0 → cur<to, step<0 → cur>to, else 0
                        use cranelift_codegen::ir::condcodes::FloatCC;
                        let cur_bits = b.use_var(vars[*cur_var as usize]);
                        let to_bits = b.use_var(vars[*to_var as usize]);
                        let step_bits = b.use_var(vars[*step_var as usize]);
                        let cur_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), cur_bits);
                        let to_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), to_bits);
                        let step_f = b.ins().bitcast(types::F64, cranelift_codegen::ir::MemFlags::new(), step_bits);
                        let zero_f = b.ins().f64const(0.0);
                        // Check step > 0
                        let step_pos = b.ins().fcmp(FloatCC::GreaterThan, step_f, zero_f);
                        // cur < to (for positive step)
                        let lt = b.ins().fcmp(FloatCC::LessThan, cur_f, to_f);
                        // cur > to (for negative step)
                        let gt = b.ins().fcmp(FloatCC::GreaterThan, cur_f, to_f);
                        // Select: step > 0 ? (cur < to) : (cur > to)
                        let selected = b.ins().select(step_pos, lt, gt);
                        // If step == 0, force false (0)
                        let step_nz = b.ins().fcmp(FloatCC::NotEqual, step_f, zero_f);
                        let result = b.ins().band(selected, step_nz);
                        let ext = b.ins().uextend(ptr_ty, result);
                        b.def_var(vars[*dst_var as usize], ext);
                    }
                    JitOp::F64Const { dst_var, val } => {
                        let bits = val.to_bits() as i64;
                        let c = b.ins().iconst(ptr_ty, bits);
                        b.def_var(vars[*dst_var as usize], c);
                    }
                    JitOp::F64Move { dst_var, src_var } => {
                        let v = b.use_var(vars[*src_var as usize]);
                        b.def_var(vars[*dst_var as usize], v);
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
                        // Load env.error_flag via env_ptr + offset_of!(JitEnv, error_flag).
                        // Per-env so threads don't race; codegen carries no static address.
                        let flag_off = std::mem::offset_of!(JitEnv, error_flag) as i32;
                        let flag_val = b.ins().load(types::I64, cranelift_codegen::ir::MemFlags::new(), env_ptr, flag_off);
                        let zero = b.ins().iconst(types::I64, 0);
                        let is_err = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, flag_val, zero);
                        let err_blk = b.create_block();
                        let ok_blk = b.create_block();
                        b.ins().brif(is_err, err_blk, &[], ok_blk, &[]);
                        // Error path: get the error value and jump to catch
                        b.switch_to_block(err_blk);
                        b.seal_block(err_blk);
                        let d = slot_addr(&mut b, *error_dst);
                        b.ins().call(rt["get_error"], &[d, env_ptr]);
                        b.ins().jump(label_blocks[*catch_label as usize], &[]);
                        // OK path: continue
                        b.switch_to_block(ok_blk);
                        b.seal_block(ok_blk);
                    }
                    JitOp::JumpIfError { label } => {
                        // Branch on env.error_flag without draining the pending error.
                        let flag_off = std::mem::offset_of!(JitEnv, error_flag) as i32;
                        let flag_val = b.ins().load(types::I64, cranelift_codegen::ir::MemFlags::new(), env_ptr, flag_off);
                        let zero = b.ins().iconst(types::I64, 0);
                        let is_err = b.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, flag_val, zero);
                        let ok_blk = b.create_block();
                        b.ins().brif(is_err, label_blocks[*label as usize], &[], ok_blk, &[]);
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
                    JitOp::CollectRange { dst, n } => {
                        let d = slot_addr(&mut b, *dst);
                        let n_addr = slot_addr(&mut b, *n);
                        let n_f64 = b.ins().call(rt["to_f64"], &[n_addr]);
                        let n_val = b.inst_results(n_f64)[0];
                        b.ins().call(rt["collect_range"], &[d, n_val]);
                    }
                    JitOp::ArrPush { arr, val } => {
                        let a = slot_addr(&mut b, *arr);
                        let v = slot_addr(&mut b, *val);
                        b.ins().call(rt["arr_push"], &[a, v]);
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
    decl!("move_to_var", [p, i, p], []);
    decl!("path_extract", [p, p, p], [p]);
    decl!("path_insert", [p, p, p], [p]);
    decl!("take_by_idx", [p, p, p], []);
    decl!("put_by_idx", [p, p, p], []);
    decl!("setpath_mut", [p, p, p], []);
    decl!("collect_begin", [p], []);
    decl!("collect_push", [p, p], []);
    decl!("collect_finish", [p, p], []);
    decl!("obj_new", [p, p], []);
    decl!("obj_insert", [p, p, p], [p]);
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
    decl!("get_error", [p, p], []);  // dst, env
    decl!("throw_error", [p, p], [p]);
    decl!("call_builtin", [p, p, p, p, p], [p]);  // dst, name_ptr, name_len, args_ptr, nargs -> status
    decl!("reverse_inplace", [p], [p]);  // v: *mut Value -> status
    decl!("sort_inplace", [p], [p]);     // v: *mut Value -> status
    decl!("collect_range", [p, f], []);  // dst, n (f64)
    decl!("arr_push", [p, p], []);       // arr: *mut Value, val: *const Value
    // Libm transcendental functions (f64 -> f64)
    decl!("libm_sin", [f], [f]);
    decl!("libm_cos", [f], [f]);
    decl!("libm_tan", [f], [f]);
    decl!("libm_asin", [f], [f]);
    decl!("libm_acos", [f], [f]);
    decl!("libm_atan", [f], [f]);
    decl!("libm_exp", [f], [f]);
    decl!("libm_exp2", [f], [f]);
    decl!("libm_log", [f], [f]);
    decl!("libm_log2", [f], [f]);
    decl!("libm_log10", [f], [f]);
    decl!("libm_cbrt", [f], [f]);
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

// Per-thread reusable JIT env. Each thread that runs JIT'd code keeps its
// own pre-allocated `JitEnv` to avoid re-allocating the 65536-slot vars
// vector on every `execute_jit` call.
thread_local! {
    static REUSABLE_ENV: UnsafeCell<Option<JitEnv>> = const { UnsafeCell::new(None) };
}

/// Copy live LoadVar bindings from the JIT env into the eval Env before we delegate
/// a complex closure-op expression back to eval. Without this, constructs like
/// `let $r = 100 in (.a, .b) |= . + $r` (emitted as `+=` desugaring) lose `$r`
/// when the Update falls off the JIT's fast path and is handed to eval with a
/// fresh Env.
///
/// Every `__xxx__:` runtime dispatcher must construct its delegated `eval::Env`
/// through [`new_delegated_env`] or [`reset_delegated_env`] rather than calling
/// `Env::new` directly — they guarantee the JIT-set let-bindings are seeded so
/// new closure ops can't silently regress.
fn seed_eval_env_from_jit(env: &Rc<RefCell<crate::eval::Env>>, exprs: &[&Expr]) {
    let mut indices: Vec<u16> = Vec::new();
    for e in exprs {
        Flattener::collect_loadvar_indices(e, &mut indices);
    }
    if indices.is_empty() { return; }
    REUSABLE_ENV.with(|cell| {
        let jit_env_opt = unsafe { &*cell.get() };
        let jit_env = match jit_env_opt.as_ref() { Some(e) => e, None => return };
        let mut env_mut = env.borrow_mut();
        for idx in indices {
            let i = idx as usize;
            if i < jit_env.vars.len() {
                env_mut.seed_var(idx, jit_env.vars[i].clone());
            }
        }
    });
}

/// Build a fresh `eval::Env` for a JIT→eval closure-op delegation, auto-seeded
/// with any `$var` referenced by `exprs`. Use this instead of
/// `Rc::new(RefCell::new(Env::new(vec![])))` from inside an `__xxx__:` dispatcher.
fn new_delegated_env(exprs: &[&Expr]) -> Rc<RefCell<crate::eval::Env>> {
    let env = Rc::new(RefCell::new(crate::eval::Env::new(vec![])));
    seed_eval_env_from_jit(&env, exprs);
    env
}

/// Reset a cached `eval::Env` for a JIT→eval closure-op delegation and re-seed
/// any `$var` referenced by `exprs` from the JIT env. Use this instead of a
/// bare `env.borrow_mut().reset()` when the dispatcher caches its Env in a
/// thread-local to amortize allocation.
fn reset_delegated_env(env: &Rc<RefCell<crate::eval::Env>>, exprs: &[&Expr]) {
    env.borrow_mut().reset();
    seed_eval_env_from_jit(env, exprs);
}

fn with_jit_env<R>(f: impl FnOnce(&mut JitEnv) -> R) -> R {
    struct EnvGuard(*mut JitEnv);
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            CURRENT_ENV.with(|c| c.set(self.0));
        }
    }
    REUSABLE_ENV.with(|cell| {
        let env_opt = unsafe { &mut *cell.get() };
        let env = env_opt.get_or_insert_with(JitEnv::new);
        env.collect_stacks.clear();
        env.str_bufs.clear();
        env.try_depth = 0;
        env.error_msg = None;
        env.error_flag = 0;
        // Publish the env pointer so runtime trampolines that don't take env
        // (e.g. via `set_jit_error`) can flip `error_flag` on the right env.
        let _guard = EnvGuard(CURRENT_ENV.with(|c| c.replace(env as *mut JitEnv)));
        f(env)
    })
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
        let r = execute_jit(f, &Value::number(42.0)).unwrap();
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
        let r = execute_jit(f, &Value::number(7.0)).unwrap();
        assert_eq!(r.len(), 1);
        assert!(matches!(&r[0], Value::Num(n, _) if *n == 7.0));
    }
}
