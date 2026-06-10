//! Tree-walking interpreter for the IR.
//!
//! Every expression is a generator: it takes an input Value and produces
//! zero or more output Values via a callback.
//!
//! We use Rc<RefCell<Env>> to allow nested closures to share the environment.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use anyhow::{Result, bail};

use crate::ir::*;
use crate::value::{Value, ObjInner, NumRepr, KeyStr, ValueKey};

use std::sync::atomic::{AtomicBool, Ordering};

static MUTATE_TRACE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable or disable `--trace-mutate` emission. Set from the CLI startup.
pub fn set_mutate_trace_enabled(on: bool) {
    MUTATE_TRACE_ENABLED.store(on, Ordering::Relaxed);
}

/// Emit one trace line per `mutate(...)` invocation. Reports whether the
/// in-place fast path will engage (input container refcount = 1) or copy-
/// on-write will kick in (refcount > 1). Scalars are reported as `n/a`.
///
/// Called from both the AST evaluator (`Expr::Mutate` arm) and the JIT
/// runtime helper (`jit_rt_trace_mutate`) so the two paths emit identical
/// trace lines. The `MUTATE_TRACE_ENABLED` short-circuit at the top makes
/// this cheap to leave inline at every `mutate(...)` call site.
pub(crate) fn trace_mutate_event(kind: MutateKind, input: &Value) {
    if !MUTATE_TRACE_ENABLED.load(Ordering::Relaxed) { return; }
    let (container, refc) = match input {
        Value::Arr(rc) => ("array", Some(Rc::strong_count(rc))),
        Value::Obj(ObjInner(rc)) => ("object", Some(Rc::strong_count(rc))),
        _ => (input.type_name(), None),
    };
    let kind_str = match kind { MutateKind::Update => "update", MutateKind::Assign => "assign" };
    match refc {
        Some(n) => {
            let mode = if n <= 1 { "in_place" } else { "copy_fallback" };
            eprintln!("[trace:mutate] kind={} container={} refcount={} mode={}", kind_str, container, n, mode);
        }
        None => {
            eprintln!("[trace:mutate] kind={} container={} refcount=n/a mode=in_place", kind_str, container);
        }
    }
}

type GenResult = Result<bool>;
pub type EnvRef = Rc<RefCell<Env>>;

// Per-thread inputs queue for `input`/`inputs` builtins.
// Pre-populated by CLI before eval/JIT execution. Per-thread to keep
// `cargo test` parallel runs honest — see `value::OBJMAP_POOL`.
thread_local! {
    static INPUTS_STATE: RefCell<(Vec<InputEntry>, usize)> = const { RefCell::new((Vec::new(), 0)) };
}

/// A queued document for `input`/`inputs`: the value, the `input_line_number`
/// jq reports after consuming it (#855), and the source filename for
/// `input_filename` (#926; `None` => stdin-less null, but the CLI always
/// supplies "<stdin>" or a file path so this is `Some` in practice).
pub type InputEntry = (Value, u64, Option<std::rc::Rc<str>>);

// Per-thread 1-indexed line number for `input_line_number`. The CLI updates
// it before executing the filter on each input; jq defines it as the count
// of `\n` bytes consumed at the point the value was emitted (not where the
// value started), which for multi-value lines means every value on that
// line sees the same number.
thread_local! {
    static INPUT_LINE_STATE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

// Per-thread filename reported by `input_filename` (#926). `None` means no
// input source has been consumed yet — jq returns `null` in that case (e.g.
// `-n` mode before the first `input`). The CLI sets it to the file path for a
// named file argument, or "<stdin>" for the stdin stream, as each source is
// read. `read_next_input` advances it alongside the line counter so a filter
// that pulls across a file boundary via `input` sees the right source.
thread_local! {
    static INPUT_FILENAME_STATE: RefCell<Option<std::rc::Rc<str>>> = const { RefCell::new(None) };
}

/// Set the line number reported by `input_line_number` for the current input.
pub fn set_input_line_number(n: u64) {
    INPUT_LINE_STATE.with(|c| c.set(n));
}

/// Read the current line number reported by `input_line_number`.
pub fn get_input_line_number() -> u64 {
    INPUT_LINE_STATE.with(|c| c.get())
}

/// Set the filename reported by `input_filename` for the current input source
/// (`None` => jq's `null`, before any input is consumed). See #926.
pub fn set_input_filename(name: Option<std::rc::Rc<str>>) {
    INPUT_FILENAME_STATE.with_borrow_mut(|c| *c = name);
}

/// Read the current `input_filename` value as a jq `Value` (`null` until an
/// input source has been consumed, otherwise the source path string).
pub fn get_input_filename() -> Value {
    INPUT_FILENAME_STATE.with_borrow(|c| match c {
        Some(name) => Value::from_str(name),
        None => Value::Null,
    })
}

/// Set the inputs queue for `input`/`inputs` builtins. Each entry pairs the
/// document value with the `input_line_number` jq reports after consuming it
/// (#855): reading a document via `input`/`inputs` advances the counter, just
/// like the main per-document loop.
pub fn set_inputs_queue(values: Vec<InputEntry>) {
    INPUTS_STATE.with_borrow_mut(|state| {
        state.0 = values;
        state.1 = 0;
    });
}

/// Clear the inputs queue.
pub fn clear_inputs_queue() {
    INPUTS_STATE.with_borrow_mut(|state| {
        state.0.clear();
        state.1 = 0;
    });
}

/// Read the next input value, advancing `input_line_number` to the line on
/// which that document ended (#855). Returns None if exhausted.
pub fn read_next_input() -> Option<Value> {
    let next = INPUTS_STATE.with_borrow_mut(|state| {
        if state.1 < state.0.len() {
            let idx = state.1;
            state.1 += 1;
            Some(state.0[idx].clone())
        } else {
            None
        }
    });
    match next {
        Some((v, line, filename)) => {
            set_input_line_number(line);
            set_input_filename(filename);
            Some(v)
        }
        None => None,
    }
}

use crate::signal::{BreakError, ErrorValue, take_error_payload};

/// The value a `try ... catch` / `?` receives when it catches a `break`
/// signal. jq surfaces the break as `{"__jq": <label id>}`; matching the
/// shape lets `catch`/`?` handle it like any other error. Only the shape is
/// guaranteed — the id is an internal counter and need not match jq's. #715.
fn break_catch_value(label_id: u64) -> Value {
    let mut obj = crate::value::new_objmap();
    obj.insert("__jq".into(), Value::number(label_id as f64));
    Value::object_from_map(obj)
}

/// Cached output sequence for a single memoize call site.
///
/// The vast majority of memoize use cases — fib, Collatz, totient, etc. —
/// have single-output bodies, so we special-case that to skip an
/// `Rc<Vec<Value>>` allocation and a 1-element iteration on every cache hit.
#[derive(Clone)]
pub enum MemoEntry {
    Single(Value),
    Many(Rc<Vec<Value>>),
}

/// Per-slot cache state: the entries themselves plus diagnostic counters.
/// `hits` / `misses` are populated unconditionally — the branch is cheap and
/// reading them is the whole point of `--debug-memo`.
pub struct SlotState {
    pub entries: HashMap<ValueKey, MemoEntry>,
    pub hits: u64,
    pub misses: u64,
}

impl SlotState {
    fn new() -> Self {
        SlotState { entries: HashMap::new(), hits: 0, misses: 0 }
    }
}

pub type MemoSlot = Rc<RefCell<SlotState>>;

/// Memoize state: per-program slots plus the per-slot cap.
/// Boxed onto `Env` so programs without `memoize(...)` pay zero per-Env
/// footprint and the cold path stays out of the central `eval()` match's
/// hot cache lines (#673).
pub struct MemoState {
    pub slots: Vec<MemoSlot>,
    pub max_entries: usize,
}

pub struct Env {
    vars: Vec<Value>,
    funcs: Vec<Rc<CompiledFunc>>,
    next_label: u64,
    pub next_var: u16,
    pub lib_dirs: Vec<String>,
    /// Closure bindings: (param_var_index, arg_expression).
    /// Used to avoid deep-cloning function bodies via substitute_params.
    closures: Vec<(VarIdx, Expr)>,
    /// Cache for is_recursive check per func_id.
    recursive_cache: Vec<(FuncId, bool)>,
    /// Cache for substituted function bodies: (func_id, arg_var_indices) → substituted body.
    /// Only used when all args are LoadVar (the common case).
    subst_cache: Vec<((FuncId, Vec<VarIdx>), Rc<Expr>)>,
    /// Pointer-based substitution cache: func_id → (args_ptr, substituted_body).
    /// For non-LoadVar args from stable (cached) call sites.
    subst_ptr_cache: Vec<(FuncId, usize, Rc<Expr>)>,
    /// One cache map per lexical `memoize(...)` occurrence in the program,
    /// plus the per-slot entry cap. `None` for programs that don't use
    /// `memoize(...)` so the Env stays small. Lifetime is the whole program
    /// execution (persists across NDJSON inputs).
    pub memo: Option<Box<MemoState>>,
}

const DEFAULT_MEMO_MAX_ENTRIES: usize = 1_000_000;

pub fn default_memo_max_entries() -> usize {
    DEFAULT_MEMO_MAX_ENTRIES
}

fn fresh_memo_slots(n: u32) -> Vec<MemoSlot> {
    (0..n).map(|_| Rc::new(RefCell::new(SlotState::new()))).collect()
}

impl Env {
    pub fn new(funcs: Vec<CompiledFunc>) -> Self {
        Env { vars: vec![Value::Null; 65536], funcs: funcs.into_iter().map(Rc::new).collect(), next_label: 0, next_var: 256, lib_dirs: Vec::new(), closures: Vec::new(), recursive_cache: Vec::new(), subst_cache: Vec::new(), subst_ptr_cache: Vec::new(), memo: None }
    }
    pub fn with_lib_dirs(funcs: Vec<CompiledFunc>, lib_dirs: Vec<String>) -> Self {
        Env { vars: vec![Value::Null; 65536], funcs: funcs.into_iter().map(Rc::new).collect(), next_label: 0, next_var: 256, lib_dirs, closures: Vec::new(), recursive_cache: Vec::new(), subst_cache: Vec::new(), subst_ptr_cache: Vec::new(), memo: None }
    }
    /// Allocate `n` memo cache slots. Called once per program after parsing,
    /// using `ParseResult::memo_slots`. No-op when `n == 0` so non-memoize
    /// programs never allocate `MemoState`.
    pub fn set_memo_slots(&mut self, n: u32) {
        if n == 0 { return; }
        let max_entries = self.memo.as_ref().map_or(DEFAULT_MEMO_MAX_ENTRIES, |m| m.max_entries);
        self.memo = Some(Box::new(MemoState { slots: fresh_memo_slots(n), max_entries }));
    }
    pub fn set_memo_max_entries(&mut self, n: usize) {
        match &mut self.memo {
            Some(m) => m.max_entries = n,
            None => self.memo = Some(Box::new(MemoState { slots: Vec::new(), max_entries: n })),
        }
    }
    /// Reset env state for reuse across multiple inputs.
    /// Keeps allocated buffers (vars, caches) but resets mutable state.
    pub fn reset(&mut self) {
        // Only reset vars that were actually used (0..next_var), not all 65536
        let used = self.next_var as usize;
        for v in self.vars[..used].iter_mut() {
            *v = Value::Null;
        }
        self.next_label = 0;
        self.next_var = 256;
        self.closures.clear();
        // Keep recursive_cache, subst_cache, subst_ptr_cache, memo —
        // they stay valid across inputs (memo cache spans the whole program
        // lifetime per #671 design).
    }
    #[inline(always)]
    fn get_var(&self, idx: VarIdx) -> Value {
        let i = idx.idx();
        if i < self.vars.len() {
            // SAFETY: bounds checked above
            unsafe { self.vars.get_unchecked(i) }.clone()
        } else {
            Value::Null
        }
    }
    #[inline(always)]
    fn set_var(&mut self, idx: VarIdx, val: Value) {
        let idx = idx.idx();
        if idx >= self.vars.len() { self.vars.resize(idx + 1, Value::Null); }
        // SAFETY: bounds ensured above
        unsafe { *self.vars.get_unchecked_mut(idx) = val; }
    }
    /// Public setter used by the JIT runtime when it delegates complex paths
    /// back to eval — the JIT has its own var storage, so we copy the live
    /// bindings into the eval Env before dispatch.
    pub fn seed_var(&mut self, idx: VarIdx, val: Value) {
        self.set_var(idx, val);
    }
    fn ensure_var(&mut self, idx: VarIdx) {
        let idx = idx.idx();
        if idx >= self.vars.len() { self.vars.resize(idx + 1, Value::Null); }
    }
}

/// Substitute param var references with arg expressions in a function body.
/// This implements jq's closure semantics: each time a param is referenced,
/// the arg filter is re-evaluated with the current input.
/// Uses COW: unchanged subtrees are not cloned.
pub fn substitute_params(expr: &Expr, param_vars: &[VarIdx], args: &[Expr]) -> Expr {
    subst_cow(expr, param_vars, args).unwrap_or_else(|| expr.clone())
}

/// Substitute params AND rename all local variable bindings (LetBinding, Reduce,
/// Foreach, Label) to fresh indices from `next_var`. This prevents recursive calls
/// from clobbering each other's local variables in the callback-based eval model.
pub fn substitute_and_rename(expr: &Expr, param_vars: &[VarIdx], args: &[Expr], next_var: &mut u16) -> Expr {
    subst_inner(expr, param_vars, args, true, next_var, &mut HashMap::new())
}

fn subst_inner(
    expr: &Expr, pv: &[VarIdx], args: &[Expr],
    rename: bool, nv: &mut u16, rn: &mut HashMap<VarIdx, VarIdx>,
) -> Expr {
    macro_rules! s { ($e:expr) => { subst_inner($e, pv, args, rename, nv, rn) } }
    macro_rules! sb { ($e:expr) => { Box::new(s!($e)) } }
    macro_rules! alloc {
        ($old:expr) => { if rename { let n = VarIdx(*nv); *nv += 1; rn.insert($old, n); n } else { $old } }
    }
    match expr {
        Expr::LoadVar { var_index } => {
            for (i, p) in pv.iter().enumerate() {
                if *var_index == *p {
                    if let Some(arg) = args.get(i) { return arg.clone(); }
                }
            }
            if let Some(&new_idx) = rn.get(var_index) {
                Expr::LoadVar { var_index: new_idx }
            } else {
                expr.clone()
            }
        }
        Expr::Pipe { left, right } => Expr::Pipe { left: sb!(left), right: sb!(right) },
        Expr::Comma { left, right } => Expr::Comma { left: sb!(left), right: sb!(right) },
        Expr::BinOp { op, lhs, rhs } => Expr::BinOp { op: *op, lhs: sb!(lhs), rhs: sb!(rhs) },
        Expr::UnaryOp { op, operand } => Expr::UnaryOp { op: *op, operand: sb!(operand) },
        Expr::Index { expr: e, key } => Expr::Index { expr: sb!(e), key: sb!(key) },
        Expr::IndexOpt { expr: e, key } => Expr::IndexOpt { expr: sb!(e), key: sb!(key) },
        Expr::Each { input_expr } => Expr::Each { input_expr: sb!(input_expr) },
        Expr::EachOpt { input_expr } => Expr::EachOpt { input_expr: sb!(input_expr) },
        Expr::IfThenElse { cond, then_branch, else_branch } => Expr::IfThenElse {
            cond: sb!(cond), then_branch: sb!(then_branch), else_branch: sb!(else_branch),
        },
        Expr::LetBinding { var_index, value, body } => {
            let value = sb!(value); // process value before allocating (value doesn't see new binding)
            let new_idx = alloc!(*var_index);
            Expr::LetBinding { var_index: new_idx, value, body: sb!(body) }
        }
        Expr::TryCatch { try_expr, catch_expr, restore_dot } => Expr::TryCatch { try_expr: sb!(try_expr), catch_expr: sb!(catch_expr), restore_dot: *restore_dot },
        Expr::Collect { generator } => Expr::Collect { generator: sb!(generator) },
        Expr::Negate { operand } => Expr::Negate { operand: sb!(operand) },
        Expr::Alternative { primary, fallback } => Expr::Alternative { primary: sb!(primary), fallback: sb!(fallback) },
        Expr::Reduce { source, init, var_index, acc_index, update } => {
            let source = sb!(source);
            let init = sb!(init);
            let vi = alloc!(*var_index);
            let ai = alloc!(*acc_index);
            Expr::Reduce { source, init, var_index: vi, acc_index: ai, update: sb!(update) }
        }
        Expr::Foreach { source, init, var_index, acc_index, update, extract } => {
            let source = sb!(source);
            let init = sb!(init);
            let vi = alloc!(*var_index);
            let ai = alloc!(*acc_index);
            Expr::Foreach { source, init, var_index: vi, acc_index: ai, update: sb!(update), extract: extract.as_ref().map(|e| sb!(e)) }
        }
        Expr::ObjectConstruct { pairs } => Expr::ObjectConstruct {
            pairs: pairs.iter().map(|(k, v)| (s!(k), s!(v))).collect(),
        },
        Expr::Recurse { input_expr } => Expr::Recurse { input_expr: sb!(input_expr) },
        Expr::Range { from, to, step } => Expr::Range {
            from: sb!(from), to: sb!(to), step: step.as_ref().map(|s| sb!(s)),
        },
        Expr::Update { path_expr, update_expr } => Expr::Update { path_expr: sb!(path_expr), update_expr: sb!(update_expr) },
        Expr::Assign { path_expr, value_expr } => Expr::Assign { path_expr: sb!(path_expr), value_expr: sb!(value_expr) },
        Expr::Mutate { path_expr, value_expr, kind } => Expr::Mutate {
            path_expr: sb!(path_expr), value_expr: sb!(value_expr), kind: *kind,
        },
        Expr::PathExpr { expr: e } => Expr::PathExpr { expr: sb!(e) },
        Expr::SetPath { path, value } => Expr::SetPath { path: sb!(path), value: sb!(value) },
        Expr::GetPath { path } => Expr::GetPath { path: sb!(path) },
        Expr::DelPaths { paths } => Expr::DelPaths { paths: sb!(paths) },
        Expr::FuncCall { func_id, args: fargs } => Expr::FuncCall {
            func_id: *func_id, args: fargs.iter().map(|a| s!(a)).collect(),
        },
        Expr::StringInterpolation { parts } => Expr::StringInterpolation {
            parts: parts.iter().map(|p| match p {
                StringPart::Literal(s) => StringPart::Literal(s.clone()),
                StringPart::Expr(e) => StringPart::Expr(s!(e)),
            }).collect(),
        },
        Expr::Limit { count, generator } => Expr::Limit { count: sb!(count), generator: sb!(generator) },
        Expr::While { cond, update } => Expr::While { cond: sb!(cond), update: sb!(update) },
        Expr::Until { cond, update } => Expr::Until { cond: sb!(cond), update: sb!(update) },
        Expr::Repeat { update } => Expr::Repeat { update: sb!(update) },
        Expr::AllShort { generator, predicate } => Expr::AllShort { generator: sb!(generator), predicate: sb!(predicate) },
        Expr::AnyShort { generator, predicate } => Expr::AnyShort { generator: sb!(generator), predicate: sb!(predicate) },
        Expr::Label { var_index, body } => {
            let new_idx = alloc!(*var_index);
            Expr::Label { var_index: new_idx, body: sb!(body) }
        }
        Expr::Break { var_index, value } => {
            let idx = rn.get(var_index).copied().unwrap_or(*var_index);
            Expr::Break { var_index: idx, value: sb!(value) }
        }
        Expr::Error { msg } => Expr::Error { msg: msg.as_ref().map(|m| sb!(m)) },
        Expr::Format { kind, expr: e } => Expr::Format { kind: kind.clone(), expr: sb!(e) },
        Expr::ClosureOp { op, input_expr, key_expr } => Expr::ClosureOp {
            op: *op, input_expr: sb!(input_expr), key_expr: sb!(key_expr),
        },
        Expr::CallBuiltin { name, args: bargs } => Expr::CallBuiltin {
            name: name.clone(), args: bargs.iter().map(|a| s!(a)).collect(),
        },
        Expr::Slice { expr: e, from, to } => Expr::Slice {
            expr: sb!(e), from: from.as_ref().map(|f| sb!(f)), to: to.as_ref().map(|t| sb!(t)),
        },
        Expr::Debug { expr: e } => Expr::Debug { expr: sb!(e) },
        Expr::Stderr { expr: e } => Expr::Stderr { expr: sb!(e) },
        Expr::RegexTest { input_expr, re, flags } => Expr::RegexTest { input_expr: sb!(input_expr), re: sb!(re), flags: sb!(flags) },
        Expr::RegexMatch { input_expr, re, flags } => Expr::RegexMatch { input_expr: sb!(input_expr), re: sb!(re), flags: sb!(flags) },
        Expr::RegexCapture { input_expr, re, flags } => Expr::RegexCapture { input_expr: sb!(input_expr), re: sb!(re), flags: sb!(flags) },
        Expr::RegexScan { input_expr, re, flags } => Expr::RegexScan { input_expr: sb!(input_expr), re: sb!(re), flags: sb!(flags) },
        Expr::RegexSub { input_expr, re, tostr, flags } => Expr::RegexSub { input_expr: sb!(input_expr), re: sb!(re), tostr: sb!(tostr), flags: sb!(flags) },
        Expr::RegexGsub { input_expr, re, tostr, flags } => Expr::RegexGsub { input_expr: sb!(input_expr), re: sb!(re), tostr: sb!(tostr), flags: sb!(flags) },
        Expr::AlternativeDestructure { alternatives } => Expr::AlternativeDestructure {
            alternatives: alternatives.iter().map(|a| s!(a)).collect(),
        },
        Expr::Memoize { slot_id, key, body } => Expr::Memoize {
            slot_id: *slot_id,
            key: key.as_ref().map(|k| sb!(k)),
            body: sb!(body),
        },
        Expr::Input | Expr::Empty | Expr::Not | Expr::Env | Expr::Builtins
        | Expr::ReadInput | Expr::ReadInputs | Expr::ModuleMeta | Expr::GenLabel
        | Expr::Literal(_) | Expr::Loc { .. } => expr.clone(),
    }
}

/// Append `extra` arguments to every `FuncCall` that targets `target` within
/// `expr`, recursively. Used by the parser to forward a lambda-lifted nested
/// def's hidden capture parameters through the def's own recursive self-calls,
/// which were emitted (as zero-arg calls) before the captures were known.
/// Exhaustive (no catch-all) so a new `Expr` variant forces review. See #714.
pub(crate) fn append_call_args(expr: &Expr, target: FuncId, extra: &[Expr]) -> Expr {
    macro_rules! r { ($e:expr) => { append_call_args($e, target, extra) } }
    macro_rules! rb { ($e:expr) => { Box::new(r!($e)) } }
    match expr {
        Expr::FuncCall { func_id, args } => {
            let mut new_args: Vec<Expr> = args.iter().map(|a| r!(a)).collect();
            if *func_id == target {
                new_args.extend(extra.iter().cloned());
            }
            Expr::FuncCall { func_id: *func_id, args: new_args }
        }
        Expr::Pipe { left, right } => Expr::Pipe { left: rb!(left), right: rb!(right) },
        Expr::Comma { left, right } => Expr::Comma { left: rb!(left), right: rb!(right) },
        Expr::BinOp { op, lhs, rhs } => Expr::BinOp { op: *op, lhs: rb!(lhs), rhs: rb!(rhs) },
        Expr::UnaryOp { op, operand } => Expr::UnaryOp { op: *op, operand: rb!(operand) },
        Expr::Index { expr: e, key } => Expr::Index { expr: rb!(e), key: rb!(key) },
        Expr::IndexOpt { expr: e, key } => Expr::IndexOpt { expr: rb!(e), key: rb!(key) },
        Expr::Each { input_expr } => Expr::Each { input_expr: rb!(input_expr) },
        Expr::EachOpt { input_expr } => Expr::EachOpt { input_expr: rb!(input_expr) },
        Expr::IfThenElse { cond, then_branch, else_branch } => Expr::IfThenElse {
            cond: rb!(cond), then_branch: rb!(then_branch), else_branch: rb!(else_branch),
        },
        Expr::LetBinding { var_index, value, body } => Expr::LetBinding {
            var_index: *var_index, value: rb!(value), body: rb!(body),
        },
        Expr::TryCatch { try_expr, catch_expr, restore_dot } => Expr::TryCatch { try_expr: rb!(try_expr), catch_expr: rb!(catch_expr), restore_dot: *restore_dot },
        Expr::Collect { generator } => Expr::Collect { generator: rb!(generator) },
        Expr::Negate { operand } => Expr::Negate { operand: rb!(operand) },
        Expr::Alternative { primary, fallback } => Expr::Alternative { primary: rb!(primary), fallback: rb!(fallback) },
        Expr::Reduce { source, init, var_index, acc_index, update } => Expr::Reduce {
            source: rb!(source), init: rb!(init), var_index: *var_index, acc_index: *acc_index, update: rb!(update),
        },
        Expr::Foreach { source, init, var_index, acc_index, update, extract } => Expr::Foreach {
            source: rb!(source), init: rb!(init), var_index: *var_index, acc_index: *acc_index,
            update: rb!(update), extract: extract.as_ref().map(|e| rb!(e)),
        },
        Expr::ObjectConstruct { pairs } => Expr::ObjectConstruct {
            pairs: pairs.iter().map(|(k, v)| (r!(k), r!(v))).collect(),
        },
        Expr::Recurse { input_expr } => Expr::Recurse { input_expr: rb!(input_expr) },
        Expr::Range { from, to, step } => Expr::Range {
            from: rb!(from), to: rb!(to), step: step.as_ref().map(|s| rb!(s)),
        },
        Expr::Update { path_expr, update_expr } => Expr::Update { path_expr: rb!(path_expr), update_expr: rb!(update_expr) },
        Expr::Assign { path_expr, value_expr } => Expr::Assign { path_expr: rb!(path_expr), value_expr: rb!(value_expr) },
        Expr::Mutate { path_expr, value_expr, kind } => Expr::Mutate {
            path_expr: rb!(path_expr), value_expr: rb!(value_expr), kind: *kind,
        },
        Expr::PathExpr { expr: e } => Expr::PathExpr { expr: rb!(e) },
        Expr::SetPath { path, value } => Expr::SetPath { path: rb!(path), value: rb!(value) },
        Expr::GetPath { path } => Expr::GetPath { path: rb!(path) },
        Expr::DelPaths { paths } => Expr::DelPaths { paths: rb!(paths) },
        Expr::StringInterpolation { parts } => Expr::StringInterpolation {
            parts: parts.iter().map(|p| match p {
                StringPart::Literal(s) => StringPart::Literal(s.clone()),
                StringPart::Expr(e) => StringPart::Expr(r!(e)),
            }).collect(),
        },
        Expr::Limit { count, generator } => Expr::Limit { count: rb!(count), generator: rb!(generator) },
        Expr::While { cond, update } => Expr::While { cond: rb!(cond), update: rb!(update) },
        Expr::Until { cond, update } => Expr::Until { cond: rb!(cond), update: rb!(update) },
        Expr::Repeat { update } => Expr::Repeat { update: rb!(update) },
        Expr::AllShort { generator, predicate } => Expr::AllShort { generator: rb!(generator), predicate: rb!(predicate) },
        Expr::AnyShort { generator, predicate } => Expr::AnyShort { generator: rb!(generator), predicate: rb!(predicate) },
        Expr::Label { var_index, body } => Expr::Label { var_index: *var_index, body: rb!(body) },
        Expr::Break { var_index, value } => Expr::Break { var_index: *var_index, value: rb!(value) },
        Expr::Error { msg } => Expr::Error { msg: msg.as_ref().map(|m| rb!(m)) },
        Expr::Format { kind, expr: e } => Expr::Format { kind: kind.clone(), expr: rb!(e) },
        Expr::ClosureOp { op, input_expr, key_expr } => Expr::ClosureOp {
            op: *op, input_expr: rb!(input_expr), key_expr: rb!(key_expr),
        },
        Expr::CallBuiltin { name, args } => Expr::CallBuiltin {
            name: name.clone(), args: args.iter().map(|a| r!(a)).collect(),
        },
        Expr::Slice { expr: e, from, to } => Expr::Slice {
            expr: rb!(e), from: from.as_ref().map(|f| rb!(f)), to: to.as_ref().map(|t| rb!(t)),
        },
        Expr::Debug { expr: e } => Expr::Debug { expr: rb!(e) },
        Expr::Stderr { expr: e } => Expr::Stderr { expr: rb!(e) },
        Expr::RegexTest { input_expr, re, flags } => Expr::RegexTest { input_expr: rb!(input_expr), re: rb!(re), flags: rb!(flags) },
        Expr::RegexMatch { input_expr, re, flags } => Expr::RegexMatch { input_expr: rb!(input_expr), re: rb!(re), flags: rb!(flags) },
        Expr::RegexCapture { input_expr, re, flags } => Expr::RegexCapture { input_expr: rb!(input_expr), re: rb!(re), flags: rb!(flags) },
        Expr::RegexScan { input_expr, re, flags } => Expr::RegexScan { input_expr: rb!(input_expr), re: rb!(re), flags: rb!(flags) },
        Expr::RegexSub { input_expr, re, tostr, flags } => Expr::RegexSub { input_expr: rb!(input_expr), re: rb!(re), tostr: rb!(tostr), flags: rb!(flags) },
        Expr::RegexGsub { input_expr, re, tostr, flags } => Expr::RegexGsub { input_expr: rb!(input_expr), re: rb!(re), tostr: rb!(tostr), flags: rb!(flags) },
        Expr::AlternativeDestructure { alternatives } => Expr::AlternativeDestructure {
            alternatives: alternatives.iter().map(|a| r!(a)).collect(),
        },
        Expr::Memoize { slot_id, key, body } => Expr::Memoize {
            slot_id: *slot_id, key: key.as_ref().map(|k| rb!(k)), body: rb!(body),
        },
        Expr::LoadVar { .. } | Expr::Input | Expr::Empty | Expr::Not | Expr::Env
        | Expr::Builtins | Expr::ReadInput | Expr::ReadInputs | Expr::ModuleMeta
        | Expr::GenLabel | Expr::Literal(_) | Expr::Loc { .. } => expr.clone(),
    }
}

/// COW substitution: returns None if no param vars were found (no changes needed).
/// Avoids deep-cloning unchanged subtrees.
fn subst_cow(expr: &Expr, pv: &[VarIdx], args: &[Expr]) -> Option<Expr> {
    // Helpers: s returns Option, sb returns Option<Box>
    macro_rules! s { ($e:expr) => { subst_cow($e, pv, args) } }
    match expr {
        Expr::LoadVar { var_index } => {
            for (i, p) in pv.iter().enumerate() {
                if *var_index == *p {
                    if let Some(arg) = args.get(i) { return Some(arg.clone()); }
                }
            }
            None
        }
        // Two-child nodes
        Expr::Pipe { left, right } => {
            let l = s!(left); let r = s!(right);
            if l.is_none() && r.is_none() { return None; }
            Some(Expr::Pipe {
                left: Box::new(l.unwrap_or_else(|| left.as_ref().clone())),
                right: Box::new(r.unwrap_or_else(|| right.as_ref().clone())),
            })
        }
        Expr::Comma { left, right } => {
            let l = s!(left); let r = s!(right);
            if l.is_none() && r.is_none() { return None; }
            Some(Expr::Comma {
                left: Box::new(l.unwrap_or_else(|| left.as_ref().clone())),
                right: Box::new(r.unwrap_or_else(|| right.as_ref().clone())),
            })
        }
        Expr::BinOp { op, lhs, rhs } => {
            let l = s!(lhs); let r = s!(rhs);
            if l.is_none() && r.is_none() { return None; }
            Some(Expr::BinOp {
                op: *op,
                lhs: Box::new(l.unwrap_or_else(|| lhs.as_ref().clone())),
                rhs: Box::new(r.unwrap_or_else(|| rhs.as_ref().clone())),
            })
        }
        Expr::UnaryOp { op, operand } => s!(operand).map(|o| Expr::UnaryOp { op: *op, operand: Box::new(o) }),
        Expr::Index { expr: e, key } => {
            let ev = s!(e); let kv = s!(key);
            if ev.is_none() && kv.is_none() { return None; }
            Some(Expr::Index {
                expr: Box::new(ev.unwrap_or_else(|| e.as_ref().clone())),
                key: Box::new(kv.unwrap_or_else(|| key.as_ref().clone())),
            })
        }
        Expr::IndexOpt { expr: e, key } => {
            let ev = s!(e); let kv = s!(key);
            if ev.is_none() && kv.is_none() { return None; }
            Some(Expr::IndexOpt {
                expr: Box::new(ev.unwrap_or_else(|| e.as_ref().clone())),
                key: Box::new(kv.unwrap_or_else(|| key.as_ref().clone())),
            })
        }
        Expr::Each { input_expr } => s!(input_expr).map(|e| Expr::Each { input_expr: Box::new(e) }),
        Expr::EachOpt { input_expr } => s!(input_expr).map(|e| Expr::EachOpt { input_expr: Box::new(e) }),
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            let c = s!(cond); let t = s!(then_branch); let e = s!(else_branch);
            if c.is_none() && t.is_none() && e.is_none() { return None; }
            Some(Expr::IfThenElse {
                cond: Box::new(c.unwrap_or_else(|| cond.as_ref().clone())),
                then_branch: Box::new(t.unwrap_or_else(|| then_branch.as_ref().clone())),
                else_branch: Box::new(e.unwrap_or_else(|| else_branch.as_ref().clone())),
            })
        }
        Expr::LetBinding { var_index, value, body } => {
            let v = s!(value); let b = s!(body);
            if v.is_none() && b.is_none() { return None; }
            Some(Expr::LetBinding {
                var_index: *var_index,
                value: Box::new(v.unwrap_or_else(|| value.as_ref().clone())),
                body: Box::new(b.unwrap_or_else(|| body.as_ref().clone())),
            })
        }
        Expr::TryCatch { try_expr, catch_expr, restore_dot } => {
            let t = s!(try_expr); let c = s!(catch_expr);
            if t.is_none() && c.is_none() { return None; }
            Some(Expr::TryCatch {
                try_expr: Box::new(t.unwrap_or_else(|| try_expr.as_ref().clone())),
                catch_expr: Box::new(c.unwrap_or_else(|| catch_expr.as_ref().clone())),
                restore_dot: *restore_dot,
            })
        }
        Expr::Collect { generator } => s!(generator).map(|g| Expr::Collect { generator: Box::new(g) }),
        Expr::Negate { operand } => s!(operand).map(|o| Expr::Negate { operand: Box::new(o) }),
        Expr::Alternative { primary, fallback } => {
            let p = s!(primary); let f = s!(fallback);
            if p.is_none() && f.is_none() { return None; }
            Some(Expr::Alternative {
                primary: Box::new(p.unwrap_or_else(|| primary.as_ref().clone())),
                fallback: Box::new(f.unwrap_or_else(|| fallback.as_ref().clone())),
            })
        }
        Expr::Reduce { source, init, var_index, acc_index, update } => {
            let sv = s!(source); let iv = s!(init); let uv = s!(update);
            if sv.is_none() && iv.is_none() && uv.is_none() { return None; }
            Some(Expr::Reduce {
                source: Box::new(sv.unwrap_or_else(|| source.as_ref().clone())),
                init: Box::new(iv.unwrap_or_else(|| init.as_ref().clone())),
                var_index: *var_index, acc_index: *acc_index,
                update: Box::new(uv.unwrap_or_else(|| update.as_ref().clone())),
            })
        }
        Expr::Foreach { source, init, var_index, acc_index, update, extract } => {
            let sv = s!(source); let iv = s!(init); let uv = s!(update);
            let ev = extract.as_ref().and_then(|e| s!(e));
            if sv.is_none() && iv.is_none() && uv.is_none() && ev.is_none() { return None; }
            Some(Expr::Foreach {
                source: Box::new(sv.unwrap_or_else(|| source.as_ref().clone())),
                init: Box::new(iv.unwrap_or_else(|| init.as_ref().clone())),
                var_index: *var_index, acc_index: *acc_index,
                update: Box::new(uv.unwrap_or_else(|| update.as_ref().clone())),
                extract: if ev.is_some() { ev.map(Box::new) } else { extract.clone() },
            })
        }
        Expr::ObjectConstruct { pairs } => {
            let results: Vec<_> = pairs.iter().map(|(k, v)| (s!(k), s!(v))).collect();
            if results.iter().all(|(k, v)| k.is_none() && v.is_none()) { return None; }
            Some(Expr::ObjectConstruct {
                pairs: pairs.iter().zip(results).map(|((k, v), (kn, vn))| {
                    (kn.unwrap_or_else(|| k.clone()), vn.unwrap_or_else(|| v.clone()))
                }).collect(),
            })
        }
        Expr::Recurse { input_expr } => s!(input_expr).map(|e| Expr::Recurse { input_expr: Box::new(e) }),
        Expr::Range { from, to, step } => {
            let fv = s!(from); let tv = s!(to); let sv = step.as_ref().and_then(|s2| s!(s2));
            if fv.is_none() && tv.is_none() && sv.is_none() { return None; }
            Some(Expr::Range {
                from: Box::new(fv.unwrap_or_else(|| from.as_ref().clone())),
                to: Box::new(tv.unwrap_or_else(|| to.as_ref().clone())),
                step: if sv.is_some() { sv.map(Box::new) } else { step.clone() },
            })
        }
        Expr::Update { path_expr, update_expr } => {
            let p = s!(path_expr); let u = s!(update_expr);
            if p.is_none() && u.is_none() { return None; }
            Some(Expr::Update {
                path_expr: Box::new(p.unwrap_or_else(|| path_expr.as_ref().clone())),
                update_expr: Box::new(u.unwrap_or_else(|| update_expr.as_ref().clone())),
            })
        }
        Expr::Assign { path_expr, value_expr } => {
            let p = s!(path_expr); let v = s!(value_expr);
            if p.is_none() && v.is_none() { return None; }
            Some(Expr::Assign {
                path_expr: Box::new(p.unwrap_or_else(|| path_expr.as_ref().clone())),
                value_expr: Box::new(v.unwrap_or_else(|| value_expr.as_ref().clone())),
            })
        }
        Expr::Mutate { path_expr, value_expr, kind } => {
            let p = s!(path_expr); let v = s!(value_expr);
            if p.is_none() && v.is_none() { return None; }
            Some(Expr::Mutate {
                path_expr: Box::new(p.unwrap_or_else(|| path_expr.as_ref().clone())),
                value_expr: Box::new(v.unwrap_or_else(|| value_expr.as_ref().clone())),
                kind: *kind,
            })
        }
        Expr::PathExpr { expr: e } => s!(e).map(|v| Expr::PathExpr { expr: Box::new(v) }),
        Expr::SetPath { path, value } => {
            let p = s!(path); let v = s!(value);
            if p.is_none() && v.is_none() { return None; }
            Some(Expr::SetPath {
                path: Box::new(p.unwrap_or_else(|| path.as_ref().clone())),
                value: Box::new(v.unwrap_or_else(|| value.as_ref().clone())),
            })
        }
        Expr::GetPath { path } => s!(path).map(|p| Expr::GetPath { path: Box::new(p) }),
        Expr::DelPaths { paths } => s!(paths).map(|p| Expr::DelPaths { paths: Box::new(p) }),
        Expr::FuncCall { func_id, args: fargs } => {
            let results: Vec<_> = fargs.iter().map(|a| s!(a)).collect();
            if results.iter().all(|r| r.is_none()) { return None; }
            Some(Expr::FuncCall {
                func_id: *func_id,
                args: fargs.iter().zip(results).map(|(a, r)| r.unwrap_or_else(|| a.clone())).collect(),
            })
        }
        Expr::StringInterpolation { parts } => {
            let results: Vec<_> = parts.iter().map(|p| match p {
                StringPart::Literal(_) => None,
                StringPart::Expr(e) => s!(e),
            }).collect();
            if results.iter().all(|r| r.is_none()) { return None; }
            Some(Expr::StringInterpolation {
                parts: parts.iter().zip(results).map(|(p, r)| match (p, r) {
                    (_, Some(new_e)) => StringPart::Expr(new_e),
                    (orig, None) => orig.clone(),
                }).collect(),
            })
        }
        Expr::Limit { count, generator } => {
            let c = s!(count); let g = s!(generator);
            if c.is_none() && g.is_none() { return None; }
            Some(Expr::Limit {
                count: Box::new(c.unwrap_or_else(|| count.as_ref().clone())),
                generator: Box::new(g.unwrap_or_else(|| generator.as_ref().clone())),
            })
        }
        Expr::While { cond, update } => {
            let c = s!(cond); let u = s!(update);
            if c.is_none() && u.is_none() { return None; }
            Some(Expr::While {
                cond: Box::new(c.unwrap_or_else(|| cond.as_ref().clone())),
                update: Box::new(u.unwrap_or_else(|| update.as_ref().clone())),
            })
        }
        Expr::Until { cond, update } => {
            let c = s!(cond); let u = s!(update);
            if c.is_none() && u.is_none() { return None; }
            Some(Expr::Until {
                cond: Box::new(c.unwrap_or_else(|| cond.as_ref().clone())),
                update: Box::new(u.unwrap_or_else(|| update.as_ref().clone())),
            })
        }
        Expr::Repeat { update } => s!(update).map(|u| Expr::Repeat { update: Box::new(u) }),
        Expr::AllShort { generator, predicate } => {
            let g = s!(generator); let p = s!(predicate);
            if g.is_none() && p.is_none() { return None; }
            Some(Expr::AllShort {
                generator: Box::new(g.unwrap_or_else(|| generator.as_ref().clone())),
                predicate: Box::new(p.unwrap_or_else(|| predicate.as_ref().clone())),
            })
        }
        Expr::AnyShort { generator, predicate } => {
            let g = s!(generator); let p = s!(predicate);
            if g.is_none() && p.is_none() { return None; }
            Some(Expr::AnyShort {
                generator: Box::new(g.unwrap_or_else(|| generator.as_ref().clone())),
                predicate: Box::new(p.unwrap_or_else(|| predicate.as_ref().clone())),
            })
        }
        Expr::Label { var_index, body } => s!(body).map(|b| Expr::Label { var_index: *var_index, body: Box::new(b) }),
        Expr::Break { var_index, value } => s!(value).map(|v| Expr::Break { var_index: *var_index, value: Box::new(v) }),
        Expr::Error { msg } => {
            let m = msg.as_ref().and_then(|m2| s!(m2));
            m.map(|v| Expr::Error { msg: Some(Box::new(v)) })
        }
        Expr::Format { kind, expr: e } => s!(e).map(|v| Expr::Format { kind: kind.clone(), expr: Box::new(v) }),
        Expr::ClosureOp { op, input_expr, key_expr } => {
            let i = s!(input_expr); let k = s!(key_expr);
            if i.is_none() && k.is_none() { return None; }
            Some(Expr::ClosureOp {
                op: *op,
                input_expr: Box::new(i.unwrap_or_else(|| input_expr.as_ref().clone())),
                key_expr: Box::new(k.unwrap_or_else(|| key_expr.as_ref().clone())),
            })
        }
        Expr::CallBuiltin { name, args: bargs } => {
            let results: Vec<_> = bargs.iter().map(|a| s!(a)).collect();
            if results.iter().all(|r| r.is_none()) { return None; }
            Some(Expr::CallBuiltin {
                name: name.clone(),
                args: bargs.iter().zip(results).map(|(a, r)| r.unwrap_or_else(|| a.clone())).collect(),
            })
        }
        Expr::Slice { expr: e, from, to } => {
            let ev = s!(e);
            let fv = from.as_ref().and_then(|f2| s!(f2));
            let tv = to.as_ref().and_then(|t2| s!(t2));
            if ev.is_none() && fv.is_none() && tv.is_none() { return None; }
            Some(Expr::Slice {
                expr: Box::new(ev.unwrap_or_else(|| e.as_ref().clone())),
                from: if fv.is_some() { fv.map(Box::new) } else { from.clone() },
                to: if tv.is_some() { tv.map(Box::new) } else { to.clone() },
            })
        }
        Expr::Debug { expr: e } => s!(e).map(|v| Expr::Debug { expr: Box::new(v) }),
        Expr::Stderr { expr: e } => s!(e).map(|v| Expr::Stderr { expr: Box::new(v) }),
        Expr::RegexTest { input_expr, re, flags } => {
            let i = s!(input_expr); let r = s!(re); let f = s!(flags);
            if i.is_none() && r.is_none() && f.is_none() { return None; }
            Some(Expr::RegexTest {
                input_expr: Box::new(i.unwrap_or_else(|| input_expr.as_ref().clone())),
                re: Box::new(r.unwrap_or_else(|| re.as_ref().clone())),
                flags: Box::new(f.unwrap_or_else(|| flags.as_ref().clone())),
            })
        }
        Expr::RegexMatch { input_expr, re, flags } => {
            let i = s!(input_expr); let r = s!(re); let f = s!(flags);
            if i.is_none() && r.is_none() && f.is_none() { return None; }
            Some(Expr::RegexMatch {
                input_expr: Box::new(i.unwrap_or_else(|| input_expr.as_ref().clone())),
                re: Box::new(r.unwrap_or_else(|| re.as_ref().clone())),
                flags: Box::new(f.unwrap_or_else(|| flags.as_ref().clone())),
            })
        }
        Expr::RegexCapture { input_expr, re, flags } => {
            let i = s!(input_expr); let r = s!(re); let f = s!(flags);
            if i.is_none() && r.is_none() && f.is_none() { return None; }
            Some(Expr::RegexCapture {
                input_expr: Box::new(i.unwrap_or_else(|| input_expr.as_ref().clone())),
                re: Box::new(r.unwrap_or_else(|| re.as_ref().clone())),
                flags: Box::new(f.unwrap_or_else(|| flags.as_ref().clone())),
            })
        }
        Expr::RegexScan { input_expr, re, flags } => {
            let i = s!(input_expr); let r = s!(re); let f = s!(flags);
            if i.is_none() && r.is_none() && f.is_none() { return None; }
            Some(Expr::RegexScan {
                input_expr: Box::new(i.unwrap_or_else(|| input_expr.as_ref().clone())),
                re: Box::new(r.unwrap_or_else(|| re.as_ref().clone())),
                flags: Box::new(f.unwrap_or_else(|| flags.as_ref().clone())),
            })
        }
        Expr::RegexSub { input_expr, re, tostr, flags } => {
            let i = s!(input_expr); let r = s!(re); let t = s!(tostr); let f = s!(flags);
            if i.is_none() && r.is_none() && t.is_none() && f.is_none() { return None; }
            Some(Expr::RegexSub {
                input_expr: Box::new(i.unwrap_or_else(|| input_expr.as_ref().clone())),
                re: Box::new(r.unwrap_or_else(|| re.as_ref().clone())),
                tostr: Box::new(t.unwrap_or_else(|| tostr.as_ref().clone())),
                flags: Box::new(f.unwrap_or_else(|| flags.as_ref().clone())),
            })
        }
        Expr::RegexGsub { input_expr, re, tostr, flags } => {
            let i = s!(input_expr); let r = s!(re); let t = s!(tostr); let f = s!(flags);
            if i.is_none() && r.is_none() && t.is_none() && f.is_none() { return None; }
            Some(Expr::RegexGsub {
                input_expr: Box::new(i.unwrap_or_else(|| input_expr.as_ref().clone())),
                re: Box::new(r.unwrap_or_else(|| re.as_ref().clone())),
                tostr: Box::new(t.unwrap_or_else(|| tostr.as_ref().clone())),
                flags: Box::new(f.unwrap_or_else(|| flags.as_ref().clone())),
            })
        }
        Expr::AlternativeDestructure { alternatives } => {
            let results: Vec<_> = alternatives.iter().map(|a| s!(a)).collect();
            if results.iter().all(|r| r.is_none()) { return None; }
            Some(Expr::AlternativeDestructure {
                alternatives: alternatives.iter().zip(results).map(|(a, r)| r.unwrap_or_else(|| a.clone())).collect(),
            })
        }
        Expr::Memoize { slot_id, key, body } => {
            let k = key.as_ref().and_then(|k2| s!(k2));
            let b = s!(body);
            if k.is_none() && b.is_none() { return None; }
            Some(Expr::Memoize {
                slot_id: *slot_id,
                key: if k.is_some() { k.map(Box::new) } else { key.clone() },
                body: Box::new(b.unwrap_or_else(|| body.as_ref().clone())),
            })
        }
        // Leaf nodes never contain param var references
        Expr::Input | Expr::Empty | Expr::Not | Expr::Env | Expr::Builtins
        | Expr::ReadInput | Expr::ReadInputs | Expr::ModuleMeta | Expr::GenLabel
        | Expr::Literal(_) | Expr::Loc { .. } => None,
    }
}

/// Check if an expression contains a FuncCall to the given func_id (direct recursion check).
fn contains_func_call(expr: &Expr, target: FuncId) -> bool {
    macro_rules! c { ($e:expr) => { contains_func_call($e, target) } }
    match expr {
        Expr::FuncCall { func_id, args } => *func_id == target || args.iter().any(|a| c!(a)),
        Expr::Pipe { left, right } | Expr::Comma { left, right } => c!(left) || c!(right),
        Expr::BinOp { lhs, rhs, .. } => c!(lhs) || c!(rhs),
        Expr::UnaryOp { operand, .. } | Expr::Negate { operand } => c!(operand),
        Expr::Index { expr: e, key } | Expr::IndexOpt { expr: e, key } => c!(e) || c!(key),
        Expr::Each { input_expr } | Expr::EachOpt { input_expr } | Expr::Recurse { input_expr } => c!(input_expr),
        Expr::IfThenElse { cond, then_branch, else_branch } => c!(cond) || c!(then_branch) || c!(else_branch),
        Expr::LetBinding { value, body, .. } => c!(value) || c!(body),
        Expr::TryCatch { try_expr, catch_expr, .. } => c!(try_expr) || c!(catch_expr),
        Expr::Collect { generator } => c!(generator),
        Expr::Alternative { primary, fallback } => c!(primary) || c!(fallback),
        Expr::Reduce { source, init, update, .. } => c!(source) || c!(init) || c!(update),
        Expr::Foreach { source, init, update, extract, .. } => {
            c!(source) || c!(init) || c!(update) || extract.as_ref().is_some_and(|e| c!(e))
        }
        Expr::ObjectConstruct { pairs } => pairs.iter().any(|(k, v)| c!(k) || c!(v)),
        Expr::Range { from, to, step } => c!(from) || c!(to) || step.as_ref().is_some_and(|s| c!(s)),
        Expr::Update { path_expr, update_expr } | Expr::Assign { path_expr, value_expr: update_expr } => c!(path_expr) || c!(update_expr),
        Expr::Mutate { path_expr, value_expr, .. } => c!(path_expr) || c!(value_expr),
        Expr::PathExpr { expr: e } | Expr::GetPath { path: e } | Expr::DelPaths { paths: e }
        | Expr::Debug { expr: e } | Expr::Stderr { expr: e } | Expr::Format { expr: e, .. } => c!(e),
        Expr::SetPath { path, value } => c!(path) || c!(value),
        Expr::Label { body, .. } => c!(body),
        Expr::Break { value, .. } | Expr::Error { msg: Some(value) } => c!(value),
        Expr::StringInterpolation { parts } => parts.iter().any(|p| matches!(p, StringPart::Expr(e) if c!(e))),
        Expr::Limit { count, generator } => c!(count) || c!(generator),
        Expr::While { cond, update } | Expr::Until { cond, update } => c!(cond) || c!(update),
        Expr::Repeat { update } => c!(update),
        Expr::AllShort { generator, predicate } | Expr::AnyShort { generator, predicate } => c!(generator) || c!(predicate),
        Expr::ClosureOp { input_expr, key_expr, .. } => c!(input_expr) || c!(key_expr),
        Expr::CallBuiltin { args, .. } => args.iter().any(|a| c!(a)),
        Expr::Slice { expr: e, from, to } => c!(e) || from.as_ref().is_some_and(|f| c!(f)) || to.as_ref().is_some_and(|t| c!(t)),
        Expr::RegexTest { input_expr, re, flags } | Expr::RegexMatch { input_expr, re, flags }
        | Expr::RegexCapture { input_expr, re, flags } | Expr::RegexScan { input_expr, re, flags } => {
            c!(input_expr) || c!(re) || c!(flags)
        }
        Expr::RegexSub { input_expr, re, tostr, flags } | Expr::RegexGsub { input_expr, re, tostr, flags } => {
            c!(input_expr) || c!(re) || c!(tostr) || c!(flags)
        }
        Expr::AlternativeDestructure { alternatives } => alternatives.iter().any(|a| c!(a)),
        Expr::Memoize { key, body, .. } => {
            key.as_ref().is_some_and(|k| c!(k)) || c!(body)
        }
        Expr::Input | Expr::Empty | Expr::Not | Expr::Env | Expr::Builtins
        | Expr::ReadInput | Expr::ReadInputs | Expr::ModuleMeta | Expr::GenLabel
        | Expr::Literal(_) | Expr::Loc { .. } | Expr::LoadVar { .. } | Expr::Error { msg: None } => false,
    }
}

/// Check if an expression uses the input (`.`) passed to it.
/// This tracks input flow: Pipe's right side gets new input from left's output.
fn expr_uses_outer_input(expr: &Expr) -> bool {
    match expr {
        // `Expr::Not` evaluates `!(input.is_truthy())` — it reads `.`.
        // Misclassifying it as input-free caused the Reduce fast path
        // to feed `Value::Null` to the source when the source contained
        // `not` (e.g. `reduce ({a: not}) as $x (null; . + $x)` on input
        // `[]` returning `{a:true}` instead of `{a:false}`). See #683.
        Expr::Input | Expr::Not => true,
        Expr::LoadVar { .. } | Expr::Literal(_) | Expr::Empty
        | Expr::Env | Expr::Builtins | Expr::ReadInput | Expr::ReadInputs
        | Expr::ModuleMeta | Expr::GenLabel | Expr::Loc { .. } => false,
        // Pipe: only left receives our input; right gets left's output
        Expr::Pipe { left, .. } => expr_uses_outer_input(left),
        Expr::Comma { left, right }
        | Expr::BinOp { lhs: left, rhs: right, .. }
        | Expr::Alternative { primary: left, fallback: right }
        | Expr::Index { expr: left, key: right }
        | Expr::IndexOpt { expr: left, key: right }
        | Expr::TryCatch { try_expr: left, catch_expr: right, .. } => {
            expr_uses_outer_input(left) || expr_uses_outer_input(right)
        }
        // GetPath/SetPath/DelPaths/Update/Assign always read `.` to produce
        // their result, regardless of whether the path/value sub-expressions
        // mention Input. The LetBinding fast path uses `expr_uses_outer_input`
        // to decide whether the body still needs the bound input or whether it
        // can run on `Value::Null` — saying "no" here for these forms made
        // `. as $x | getpath(["a"])` quietly return `null` instead of the
        // proper `Cannot index <type> with string "a"`. See #556.
        Expr::GetPath { .. } | Expr::SetPath { .. } | Expr::DelPaths { .. }
        | Expr::PathExpr { .. } | Expr::Update { .. } | Expr::Assign { .. }
        | Expr::Mutate { .. } => true,
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            expr_uses_outer_input(cond) || expr_uses_outer_input(then_branch) || expr_uses_outer_input(else_branch)
        }
        Expr::LetBinding { value, body, .. } => {
            expr_uses_outer_input(value) || expr_uses_outer_input(body)
        }
        Expr::Each { input_expr } | Expr::EachOpt { input_expr }
        | Expr::Recurse { input_expr }
        | Expr::Negate { operand: input_expr } | Expr::UnaryOp { operand: input_expr, .. }
        | Expr::Collect { generator: input_expr }
        | Expr::Debug { expr: input_expr }
        | Expr::Stderr { expr: input_expr } | Expr::Format { expr: input_expr, .. } => {
            expr_uses_outer_input(input_expr)
        }
        // While/Until/Repeat: cond/update get the loop value, not our input
        Expr::While { .. } | Expr::Until { .. } | Expr::Repeat { .. } => false,
        // Reduce/Foreach: source and init get our input, update gets accumulator
        Expr::Reduce { source, init, .. } | Expr::Foreach { source, init, .. } => {
            expr_uses_outer_input(source) || expr_uses_outer_input(init)
        }
        Expr::Limit { count, generator } => {
            expr_uses_outer_input(count) || expr_uses_outer_input(generator)
        }
        Expr::Range { from, to, step } => {
            expr_uses_outer_input(from) || expr_uses_outer_input(to)
                || step.as_ref().is_some_and(|s| expr_uses_outer_input(s))
        }
        Expr::AllShort { generator, .. } | Expr::AnyShort { generator, .. } => {
            expr_uses_outer_input(generator)
        }
        Expr::ObjectConstruct { pairs } => {
            pairs.iter().any(|(k, v)| expr_uses_outer_input(k) || expr_uses_outer_input(v))
        }
        Expr::StringInterpolation { parts } => {
            parts.iter().any(|p| matches!(p, StringPart::Expr(e) if expr_uses_outer_input(e)))
        }
        Expr::Slice { expr: e, from, to } => {
            expr_uses_outer_input(e)
                || from.as_ref().is_some_and(|f| expr_uses_outer_input(f))
                || to.as_ref().is_some_and(|t| expr_uses_outer_input(t))
        }
        // Conservative: assume these use input
        Expr::FuncCall { .. } | Expr::CallBuiltin { .. }
        | Expr::Label { .. } | Expr::Break { .. }
        | Expr::Error { .. } | Expr::ClosureOp { .. }
        | Expr::RegexTest { .. } | Expr::RegexMatch { .. } | Expr::RegexCapture { .. }
        | Expr::RegexScan { .. } | Expr::RegexSub { .. } | Expr::RegexGsub { .. }
        | Expr::AlternativeDestructure { .. } => true,
        // Memoize keys cache by current input → always uses input
        Expr::Memoize { .. } => true,
    }
}

/// Check if an expression references a specific variable (for reduce optimization).
pub(crate) fn expr_uses_var(expr: &Expr, target: VarIdx) -> bool {
    match expr {
        Expr::LoadVar { var_index } => *var_index == target,
        Expr::Input | Expr::Empty | Expr::Not | Expr::Env | Expr::Builtins
        | Expr::ReadInput | Expr::ReadInputs | Expr::ModuleMeta | Expr::GenLabel
        | Expr::Literal(_) | Expr::Loc { .. } => false,
        Expr::Pipe { left, right } | Expr::Comma { left, right }
        | Expr::BinOp { lhs: left, rhs: right, .. }
        | Expr::Alternative { primary: left, fallback: right }
        | Expr::While { cond: left, update: right }
        | Expr::Until { cond: left, update: right }
        | Expr::Limit { count: left, generator: right }
        | Expr::Index { expr: left, key: right }
        | Expr::IndexOpt { expr: left, key: right }
        | Expr::Update { path_expr: left, update_expr: right }
        | Expr::Assign { path_expr: left, value_expr: right }
        | Expr::SetPath { path: left, value: right }
        | Expr::TryCatch { try_expr: left, catch_expr: right, .. } => {
            expr_uses_var(left, target) || expr_uses_var(right, target)
        }
        Expr::Mutate { path_expr, value_expr, .. } => {
            expr_uses_var(path_expr, target) || expr_uses_var(value_expr, target)
        }
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            expr_uses_var(cond, target) || expr_uses_var(then_branch, target) || expr_uses_var(else_branch, target)
        }
        Expr::LetBinding { value, body, .. } => {
            expr_uses_var(value, target) || expr_uses_var(body, target)
        }
        Expr::Each { input_expr } | Expr::EachOpt { input_expr }
        | Expr::Recurse { input_expr } | Expr::Repeat { update: input_expr }
        | Expr::Negate { operand: input_expr } | Expr::UnaryOp { operand: input_expr, .. }
        | Expr::Collect { generator: input_expr }
        | Expr::PathExpr { expr: input_expr } | Expr::GetPath { path: input_expr }
        | Expr::DelPaths { paths: input_expr } | Expr::Debug { expr: input_expr }
        | Expr::Stderr { expr: input_expr } | Expr::Format { expr: input_expr, .. } => {
            expr_uses_var(input_expr, target)
        }
        Expr::Reduce { source, init, update, .. }
        | Expr::Foreach { source, init, update, .. } => {
            expr_uses_var(source, target) || expr_uses_var(init, target) || expr_uses_var(update, target)
        }
        Expr::Range { from, to, step } => {
            expr_uses_var(from, target) || expr_uses_var(to, target) || step.as_ref().is_some_and(|s| expr_uses_var(s, target))
        }
        Expr::FuncCall { args, .. } => args.iter().any(|a| expr_uses_var(a, target)),
        Expr::CallBuiltin { args, .. } => args.iter().any(|a| expr_uses_var(a, target)),
        Expr::ObjectConstruct { pairs } => pairs.iter().any(|(k, v)| expr_uses_var(k, target) || expr_uses_var(v, target)),
        Expr::StringInterpolation { parts } => parts.iter().any(|p| matches!(p, StringPart::Expr(e) if expr_uses_var(e, target))),
        Expr::AllShort { generator, predicate } | Expr::AnyShort { generator, predicate } => {
            expr_uses_var(generator, target) || expr_uses_var(predicate, target)
        }
        Expr::Label { body, .. } | Expr::Break { value: body, .. } => expr_uses_var(body, target),
        Expr::Error { msg } => msg.as_ref().is_some_and(|m| expr_uses_var(m, target)),
        Expr::ClosureOp { input_expr, key_expr, .. } => {
            expr_uses_var(input_expr, target) || expr_uses_var(key_expr, target)
        }
        Expr::Slice { expr, from, to } => {
            expr_uses_var(expr, target) || from.as_ref().is_some_and(|f| expr_uses_var(f, target)) || to.as_ref().is_some_and(|t| expr_uses_var(t, target))
        }
        Expr::RegexTest { input_expr, re, flags } | Expr::RegexMatch { input_expr, re, flags }
        | Expr::RegexCapture { input_expr, re, flags } | Expr::RegexScan { input_expr, re, flags } => {
            expr_uses_var(input_expr, target) || expr_uses_var(re, target) || expr_uses_var(flags, target)
        }
        Expr::RegexSub { input_expr, re, tostr, flags } | Expr::RegexGsub { input_expr, re, tostr, flags } => {
            expr_uses_var(input_expr, target) || expr_uses_var(re, target) || expr_uses_var(tostr, target) || expr_uses_var(flags, target)
        }
        Expr::AlternativeDestructure { alternatives } => alternatives.iter().any(|a| expr_uses_var(a, target)),
        Expr::Memoize { key, body, .. } => {
            key.as_ref().is_some_and(|k| expr_uses_var(k, target))
                || expr_uses_var(body, target)
        }
    }
}

/// Push the `func_id` of every `FuncCall` that appears anywhere in `expr` onto
/// `out` (one level deep — the caller follows callees transitively). Exhaustive
/// over `Expr` so a call buried in any sub-expression is found, which is what
/// makes the #765 reachability check sound. Mirrors `expr_uses_var`'s structure.
pub(crate) fn collect_func_calls(expr: &Expr, out: &mut Vec<FuncId>) {
    match expr {
        Expr::Input | Expr::Empty | Expr::Not | Expr::Env | Expr::Builtins
        | Expr::ReadInput | Expr::ReadInputs | Expr::ModuleMeta | Expr::GenLabel
        | Expr::Literal(_) | Expr::Loc { .. } | Expr::LoadVar { .. } => {}
        Expr::Pipe { left, right } | Expr::Comma { left, right }
        | Expr::BinOp { lhs: left, rhs: right, .. }
        | Expr::Alternative { primary: left, fallback: right }
        | Expr::While { cond: left, update: right }
        | Expr::Until { cond: left, update: right }
        | Expr::Limit { count: left, generator: right }
        | Expr::Index { expr: left, key: right }
        | Expr::IndexOpt { expr: left, key: right }
        | Expr::Update { path_expr: left, update_expr: right }
        | Expr::Assign { path_expr: left, value_expr: right }
        | Expr::SetPath { path: left, value: right }
        | Expr::TryCatch { try_expr: left, catch_expr: right, .. } => {
            collect_func_calls(left, out);
            collect_func_calls(right, out);
        }
        Expr::Mutate { path_expr, value_expr, .. } => {
            collect_func_calls(path_expr, out);
            collect_func_calls(value_expr, out);
        }
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            collect_func_calls(cond, out);
            collect_func_calls(then_branch, out);
            collect_func_calls(else_branch, out);
        }
        Expr::LetBinding { value, body, .. } => {
            collect_func_calls(value, out);
            collect_func_calls(body, out);
        }
        Expr::Each { input_expr } | Expr::EachOpt { input_expr }
        | Expr::Recurse { input_expr } | Expr::Repeat { update: input_expr }
        | Expr::Negate { operand: input_expr } | Expr::UnaryOp { operand: input_expr, .. }
        | Expr::Collect { generator: input_expr }
        | Expr::PathExpr { expr: input_expr } | Expr::GetPath { path: input_expr }
        | Expr::DelPaths { paths: input_expr } | Expr::Debug { expr: input_expr }
        | Expr::Stderr { expr: input_expr } | Expr::Format { expr: input_expr, .. } => {
            collect_func_calls(input_expr, out);
        }
        Expr::Reduce { source, init, update, .. }
        | Expr::Foreach { source, init, update, .. } => {
            collect_func_calls(source, out);
            collect_func_calls(init, out);
            collect_func_calls(update, out);
        }
        Expr::Range { from, to, step } => {
            collect_func_calls(from, out);
            collect_func_calls(to, out);
            if let Some(s) = step { collect_func_calls(s, out); }
        }
        Expr::FuncCall { func_id, args } => {
            out.push(*func_id);
            for a in args { collect_func_calls(a, out); }
        }
        Expr::CallBuiltin { args, .. } => { for a in args { collect_func_calls(a, out); } }
        Expr::ObjectConstruct { pairs } => {
            for (k, v) in pairs { collect_func_calls(k, out); collect_func_calls(v, out); }
        }
        Expr::StringInterpolation { parts } => {
            for p in parts { if let StringPart::Expr(e) = p { collect_func_calls(e, out); } }
        }
        Expr::AllShort { generator, predicate } | Expr::AnyShort { generator, predicate } => {
            collect_func_calls(generator, out);
            collect_func_calls(predicate, out);
        }
        Expr::Label { body, .. } | Expr::Break { value: body, .. } => collect_func_calls(body, out),
        Expr::Error { msg } => { if let Some(m) = msg { collect_func_calls(m, out); } }
        Expr::ClosureOp { input_expr, key_expr, .. } => {
            collect_func_calls(input_expr, out);
            collect_func_calls(key_expr, out);
        }
        Expr::Slice { expr, from, to } => {
            collect_func_calls(expr, out);
            if let Some(f) = from { collect_func_calls(f, out); }
            if let Some(t) = to { collect_func_calls(t, out); }
        }
        Expr::RegexTest { input_expr, re, flags } | Expr::RegexMatch { input_expr, re, flags }
        | Expr::RegexCapture { input_expr, re, flags } | Expr::RegexScan { input_expr, re, flags } => {
            collect_func_calls(input_expr, out);
            collect_func_calls(re, out);
            collect_func_calls(flags, out);
        }
        Expr::RegexSub { input_expr, re, tostr, flags } | Expr::RegexGsub { input_expr, re, tostr, flags } => {
            collect_func_calls(input_expr, out);
            collect_func_calls(re, out);
            collect_func_calls(tostr, out);
            collect_func_calls(flags, out);
        }
        Expr::AlternativeDestructure { alternatives } => {
            for a in alternatives { collect_func_calls(a, out); }
        }
        Expr::Memoize { key, body, .. } => {
            if let Some(k) = key { collect_func_calls(k, out); }
            collect_func_calls(body, out);
        }
    }
}

/// Extract f64 from a leaf expression (LoadVar, Input, Literal) or simple
/// numeric BinOp without cloning. Used by the zero-clone BinOp fast path.
#[inline(always)]
fn get_num_leaf(expr: &Expr, input: &Value, vars: &[Value]) -> Option<f64> {
    match expr {
        Expr::LoadVar { var_index } => {
            if let Some(Value::Num(n, _)) = vars.get(var_index.idx()) {
                Some(*n)
            } else { None }
        }
        Expr::Input => {
            if let Value::Num(n, _) = input { Some(*n) } else { None }
        }
        Expr::Literal(Literal::Num(n, _)) => Some(*n),
        Expr::BinOp { op, lhs, rhs } => {
            let ln = get_num_leaf(lhs, input, vars)?;
            let rn = get_num_leaf(rhs, input, vars)?;
            Some(match op {
                BinOp::Add => ln + rn,
                BinOp::Sub => ln - rn,
                BinOp::Mul => ln * rn,
                BinOp::Div => { if rn == 0.0 { return None; } ln / rn }
                BinOp::Mod => { if !ln.is_finite() || !rn.is_finite() { return None; } let yi = rn as i64; if yi == 0 { return None; } crate::runtime::jq_mod_i64(ln as i64, yi) as f64 }
                _ => return None,
            })
        }
        _ => None,
    }
}

/// Evaluate a compound boolean expression with nested And/Or/comparisons
/// using only numeric leaf values. Handles patterns like:
///   `a % 2 != 0 and (a == 5 or a % 5 != 0)`
/// without creating intermediate Value objects or borrowing env per sub-expression.
#[inline(always)]
fn eval_bool_compound(expr: &Expr, input: &Value, vars: &[Value]) -> Option<bool> {
    match expr {
        Expr::BinOp { op, lhs, rhs } => {
            match op {
                BinOp::And => {
                    if !eval_bool_compound(lhs, input, vars)? { return Some(false); }
                    eval_bool_compound(rhs, input, vars)
                }
                BinOp::Or => {
                    if eval_bool_compound(lhs, input, vars)? { return Some(true); }
                    eval_bool_compound(rhs, input, vars)
                }
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                    let ln = get_num_leaf(lhs, input, vars)?;
                    let rn = get_num_leaf(rhs, input, vars)?;
                    Some(match op {
                        BinOp::Eq => ln == rn,
                        BinOp::Ne => ln != rn,
                        BinOp::Lt => ln < rn,
                        BinOp::Gt => ln > rn,
                        BinOp::Le => ln <= rn,
                        BinOp::Ge => ln >= rn,
                        _ => unreachable!(),
                    })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Evaluate a boolean expression entirely via f64 arithmetic, with one variable
/// override (avoids env borrow/store). Returns Some(true/false) or None if not applicable.
#[inline(always)]
fn eval_bool_numeric(expr: &Expr, vars: &[Value], override_vi: VarIdx, override_val: f64) -> Option<bool> {
    match expr {
        Expr::BinOp { op, lhs, rhs } => {
            let ln = get_num_leaf_override(lhs, vars, override_vi, override_val)?;
            let rn = get_num_leaf_override(rhs, vars, override_vi, override_val)?;
            match op {
                BinOp::Eq => Some(ln == rn),
                BinOp::Ne => Some(ln != rn),
                BinOp::Lt => Some(ln < rn),
                BinOp::Gt => Some(ln > rn),
                BinOp::Le => Some(ln <= rn),
                BinOp::Ge => Some(ln >= rn),
                BinOp::And => {
                    if ln == 0.0 { return Some(false); }
                    Some(rn != 0.0)
                }
                BinOp::Or => {
                    if ln != 0.0 { return Some(true); }
                    Some(rn != 0.0)
                }
                _ => None,
            }
        }
        Expr::Not => None, // Not operates on input, complex
        _ => None,
    }
}

/// Like get_num_leaf but with a variable override for one var_index.
#[inline(always)]
fn get_num_leaf_override(expr: &Expr, vars: &[Value], override_vi: VarIdx, override_val: f64) -> Option<f64> {
    match expr {
        Expr::LoadVar { var_index } => {
            if *var_index == override_vi {
                Some(override_val)
            } else if let Some(Value::Num(n, _)) = vars.get(var_index.idx()) {
                Some(*n)
            } else { None }
        }
        Expr::Literal(Literal::Num(n, _)) => Some(*n),
        Expr::BinOp { op, lhs, rhs } => {
            let ln = get_num_leaf_override(lhs, vars, override_vi, override_val)?;
            let rn = get_num_leaf_override(rhs, vars, override_vi, override_val)?;
            Some(match op {
                BinOp::Add => ln + rn,
                BinOp::Sub => ln - rn,
                BinOp::Mul => ln * rn,
                BinOp::Div => { if rn == 0.0 { return None; } ln / rn }
                BinOp::Mod => { if !ln.is_finite() || !rn.is_finite() { return None; } let yi = rn as i64; if yi == 0 { return None; } crate::runtime::jq_mod_i64(ln as i64, yi) as f64 }
                _ => return None,
            })
        }
        _ => None,
    }
}

/// Fast path for scalar expressions: evaluate without callback overhead.
/// Returns Ok(value) for simple expressions, Err for generators/complex expressions.
#[inline]
fn eval_one(expr: &Expr, input: &Value, env: &EnvRef) -> std::result::Result<Value, ()> {
    match expr {
        Expr::Input => Ok(input.clone()),
        Expr::Literal(lit) => Ok(match lit {
            Literal::Null => Value::Null,
            Literal::True => Value::True,
            Literal::False => Value::False,
            Literal::Num(n, repr) => Value::number_opt(*n, repr.clone()),
            Literal::Str(s) => Value::from_str(s),
        }),
        Expr::LoadVar { var_index } => {
            let e = env.borrow();
            if e.closures.is_empty() {
                return Ok(e.get_var(*var_index));
            }
            // Resolve closure chains iteratively for LoadVar→LoadVar→...→env
            let mut idx = *var_index;
            drop(e);
            loop {
                let e = env.borrow();
                if let Some(c) = e.closures.iter().rev().find(|c| c.0 == idx) {
                    if let Expr::LoadVar { var_index: next_idx } = &c.1 {
                        idx = *next_idx;
                        continue;
                    }
                    let arg = c.1.clone();
                    drop(e);
                    return eval_one(&arg, input, env);
                } else {
                    return Ok(e.get_var(idx));
                }
            }
        }
        Expr::Not => Ok(Value::from_bool(!input.is_truthy())),
        Expr::BinOp { op, lhs, rhs } => {
            // Skip eval_one for Add+Collect to use array-push fusion in full eval
            if matches!(op, BinOp::Add) && matches!(rhs.as_ref(), Expr::Collect { .. }) {
                return Err(());
            }
            match *op {
                BinOp::And => {
                    let l = eval_one(lhs, input, env)?;
                    if !l.is_truthy() { return Ok(Value::False); }
                    let r = eval_one(rhs, input, env)?;
                    Ok(Value::from_bool(r.is_truthy()))
                }
                BinOp::Or => {
                    let l = eval_one(lhs, input, env)?;
                    if l.is_truthy() { return Ok(Value::True); }
                    let r = eval_one(rhs, input, env)?;
                    Ok(Value::from_bool(r.is_truthy()))
                }
                _ => {
                    // Zero-clone numeric path: extract f64 directly from env/input
                    // without creating intermediate Value objects.
                    {
                        let e = env.borrow();
                        if e.closures.is_empty() {
                            if let (Some(ln), Some(rn)) = (
                                get_num_leaf(lhs, input, &e.vars),
                                get_num_leaf(rhs, input, &e.vars),
                            ) {
                                return Ok(match *op {
                                    BinOp::Add => Value::number(ln + rn),
                                    BinOp::Sub => Value::number(ln - rn),
                                    BinOp::Mul => Value::number(ln * rn),
                                    BinOp::Div => {
                                        if rn == 0.0 { drop(e); return Err(()); }
                                        Value::number(ln / rn)
                                    }
                                    BinOp::Mod => {
                                        if !ln.is_finite() || !rn.is_finite() { drop(e); return Err(()); }
                                        let yi = rn as i64;
                                        if yi == 0 { drop(e); return Err(()); }
                                        Value::number(crate::runtime::jq_mod_i64(ln as i64, yi) as f64)
                                    }
                                    BinOp::Eq => if ln == rn { Value::True } else { Value::False },
                                    BinOp::Ne => if ln != rn { Value::True } else { Value::False },
                                    BinOp::Lt => if jq_num_lt(ln, rn) { Value::True } else { Value::False },
                                    BinOp::Gt => if jq_num_gt(ln, rn) { Value::True } else { Value::False },
                                    BinOp::Le => if jq_num_le(ln, rn) { Value::True } else { Value::False },
                                    BinOp::Ge => if jq_num_ge(ln, rn) { Value::True } else { Value::False },
                                    _ => { drop(e); return Err(()); }
                                });
                            }
                        }
                    }
                    let r = eval_one(rhs, input, env)?;
                    let l = eval_one(lhs, input, env)?;
                    // Fast path: both numeric, avoid function call dispatch
                    if let (Value::Num(ln, _), Value::Num(rn, _)) = (&l, &r) {
                        return Ok(match *op {
                            BinOp::Add => Value::number(ln + rn),
                            BinOp::Sub => Value::number(ln - rn),
                            BinOp::Mul => Value::number(ln * rn),
                            BinOp::Div => {
                                if *rn == 0.0 { return eval_binop(*op, &l, &r).map_err(|_| ()); }
                                Value::number(ln / rn)
                            }
                            BinOp::Mod => {
                                if !ln.is_finite() || !rn.is_finite() { return eval_binop(*op, &l, &r).map_err(|_| ()); }
                                let yi = *rn as i64;
                                if yi == 0 { return eval_binop(*op, &l, &r).map_err(|_| ()); }
                                Value::number(crate::runtime::jq_mod_i64(*ln as i64, yi) as f64)
                            }
                            BinOp::Eq => if ln == rn { Value::True } else { Value::False },
                            BinOp::Ne => if ln != rn { Value::True } else { Value::False },
                            BinOp::Lt => if jq_num_lt(*ln, *rn) { Value::True } else { Value::False },
                            BinOp::Gt => if jq_num_gt(*ln, *rn) { Value::True } else { Value::False },
                            BinOp::Le => if jq_num_le(*ln, *rn) { Value::True } else { Value::False },
                            BinOp::Ge => if jq_num_ge(*ln, *rn) { Value::True } else { Value::False },
                            _ => return eval_binop(*op, &l, &r).map_err(|_| ()),
                        });
                    }
                    eval_binop(*op, &l, &r).map_err(|_| ())
                }
            }
        }
        Expr::UnaryOp { op, operand } => {
            let val = eval_one(operand, input, env)?;
            // Fast path for numeric unary ops
            if let Value::Num(n, _) = &val {
                return Ok(match *op {
                    UnaryOp::Floor => Value::number(n.floor()),
                    UnaryOp::Ceil => Value::number(n.ceil()),
                    UnaryOp::Round => Value::number(n.round()),
                    // jq's `abs` keeps the literal repr while `fabs`
                    // returns the canonical f64 form (#578). The two
                    // shared the same fast-path branch and dropped the
                    // repr for both. Delegate to the runtime entry points
                    // so the fast path matches `rt_abs` / `rt_fabs`.
                    UnaryOp::Fabs => Value::number(n.abs()),
                    UnaryOp::Abs => return eval_unaryop(*op, &val).map_err(|_| ()),
                    // jq's `length` on a number is `abs` but preserves the
                    // literal repr (`-1.0 | length` → `1.0`,
                    // `-0.0 | length` → `0.0`). Delegate to `Value::length`
                    // so this fast path matches the runtime `rt_length`
                    // path. See #576.
                    UnaryOp::Length => val.length().map_err(|_| ())?,
                    UnaryOp::Sqrt => Value::number(n.sqrt()),
                    _ => return eval_unaryop(*op, &val).map_err(|_| ()),
                });
            }
            eval_unaryop(*op, &val).map_err(|_| ())
        }
        Expr::Index { expr: base_expr, key: key_expr } => {
            let base = eval_one(base_expr, input, env)?;
            let key = eval_one(key_expr, input, env)?;
            eval_index(&base, &key, false).map_err(|_| ())
        }
        Expr::IndexOpt { expr: base_expr, key: key_expr } => {
            // `?` yields an *empty* stream on type error, not null. eval_one is
            // single-value only, so the type-error branch returns Err(()) and
            // the caller's generator fallback is responsible for producing zero
            // outputs (#200). Only the success path stays on the scalar route.
            let base = eval_one(base_expr, input, env)?;
            let key = eval_one(key_expr, input, env)?;
            eval_index(&base, &key, true).map_err(|_| ())
        }
        Expr::Negate { operand } => {
            let val = eval_one(operand, input, env)?;
            match val {
                Value::Num(n, NumRepr(repr)) => {
                    let neg = if n == 0.0 { 0.0 } else { -n };
                    Ok(Value::number_opt(neg, crate::value::Value::negate_repr(repr)))
                }
                _ => Err(()),
            }
        }
        Expr::Pipe { left, right } => {
            let mid = eval_one(left, input, env)?;
            eval_one(right, &mid, env)
        }
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            let c = eval_one(cond, input, env)?;
            if c.is_truthy() {
                eval_one(then_branch, input, env)
            } else {
                eval_one(else_branch, input, env)
            }
        }
        Expr::FuncCall { func_id, args } => {
            if !args.is_empty() { return Err(()); }
            let func = env.borrow().funcs.get(func_id.idx()).cloned();
            if let Some(f) = func {
                eval_one(&f.body, input, env)
            } else {
                Err(())
            }
        }
        Expr::LetBinding { var_index, value, body } => {
            let vi = var_index.idx();
            // Fast path: `. as $var` avoids eval_one dispatch for Input
            let val = if matches!(value.as_ref(), Expr::Input) {
                input.clone()
            } else {
                eval_one(value, input, env)?
            };
            let old = std::mem::replace(&mut env.borrow_mut().vars[vi], val);
            let result = eval_one(body, input, env);
            env.borrow_mut().vars[vi] = old;
            result
        }
        _ => Err(()),
    }
}

/// Like eval_one but returns Ok(None) for Empty/select(false) instead of Err.
/// This lets callers distinguish "no output" from "can't handle".
#[inline]
fn eval_one_filter(expr: &Expr, input: &Value, env: &EnvRef) -> std::result::Result<Option<Value>, ()> {
    match expr {
        Expr::Empty => Ok(None),
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            let c = eval_one(cond, input, env)?;
            if c.is_truthy() {
                eval_one_filter(then_branch, input, env)
            } else {
                eval_one_filter(else_branch, input, env)
            }
        }
        Expr::Pipe { left, right } => {
            match eval_one_filter(left, input, env)? {
                Some(mid) => eval_one_filter(right, &mid, env),
                None => Ok(None),
            }
        }
        _ => eval_one(expr, input, env).map(Some),
    }
}

/// Components of a linear recursive generator pattern:
/// `if cond then pre, (transform | self), post else else_branch end`
struct LinearRecursiveGen<'a> {
    cond: &'a Expr,
    pre: &'a Expr,
    transform: &'a Expr,
    post: &'a Expr,
    else_branch: &'a Expr,
}

/// Detect if a function body is a linear recursive generator.
/// Pattern: `if cond then pre, (transform | self), post else else_branch end`
/// where cond, pre, transform, post, else_branch are all scalar (no generators).
/// Handles both left-associated and right-associated Comma nesting.
fn detect_linear_recursive_gen(body: &Expr, func_id: FuncId) -> Option<LinearRecursiveGen<'_>> {
    let (cond, then_branch, else_branch) = match body {
        Expr::IfThenElse { cond, then_branch, else_branch } => (cond.as_ref(), then_branch.as_ref(), else_branch.as_ref()),
        _ => return None,
    };
    // Find the recursive Pipe { transform, FuncCall(func_id, []) } within the then_branch.
    // Accept two patterns:
    //   Left-assoc:  Comma { Comma { pre, Pipe { transform, self } }, post }
    //   Right-assoc: Comma { pre, Comma { Pipe { transform, self }, post } }
    let (pre, transform, post) = match then_branch {
        Expr::Comma { left, right } => {
            // Try left-associated: Comma(Comma(pre, Pipe), post)
            if let Expr::Comma { left: pre, right: pipe_part } = left.as_ref() {
                if let Some(transform) = extract_recursive_pipe(pipe_part, func_id) {
                    (pre.as_ref(), transform, right.as_ref())
                } else { return None; }
            }
            // Try right-associated: Comma(pre, Comma(Pipe, post))
            else if let Expr::Comma { left: pipe_part, right: post } = right.as_ref() {
                if let Some(transform) = extract_recursive_pipe(pipe_part, func_id) {
                    (left.as_ref(), transform, post.as_ref())
                } else { return None; }
            } else { return None; }
        }
        _ => return None,
    };
    // All parts must be scalar (no generators)
    if !is_eval_scalar(cond) || !is_eval_scalar(pre) || !is_eval_scalar(transform)
        || !is_eval_scalar(post) || !is_eval_scalar(else_branch) { return None; }
    Some(LinearRecursiveGen { cond, pre, transform, post, else_branch })
}

/// Extract the transform expression from Pipe { transform, FuncCall(func_id, []) }.
fn extract_recursive_pipe(expr: &Expr, func_id: FuncId) -> Option<&Expr> {
    if let Expr::Pipe { left, right } = expr {
        if let Expr::FuncCall { func_id: fid, args } = right.as_ref() {
            if *fid == func_id && args.is_empty() {
                return Some(left.as_ref());
            }
        }
    }
    None
}

/// Check if an expression produces exactly one output (no generators).
fn is_eval_scalar(expr: &Expr) -> bool {
    match expr {
        Expr::Input | Expr::Literal(_) | Expr::LoadVar { .. } | Expr::Not => true,
        Expr::BinOp { lhs, rhs, .. } => is_eval_scalar(lhs) && is_eval_scalar(rhs),
        Expr::UnaryOp { operand, .. } | Expr::Negate { operand } => is_eval_scalar(operand),
        Expr::Pipe { left, right } => is_eval_scalar(left) && is_eval_scalar(right),
        Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => is_eval_scalar(expr) && is_eval_scalar(key),
        Expr::IfThenElse { cond, then_branch, else_branch } =>
            is_eval_scalar(cond) && is_eval_scalar(then_branch) && is_eval_scalar(else_branch),
        Expr::LetBinding { value, body, .. } => is_eval_scalar(value) && is_eval_scalar(body),
        Expr::Alternative { primary, fallback } => is_eval_scalar(primary) && is_eval_scalar(fallback),
        Expr::StringInterpolation { parts } => parts.iter().all(|p| match p {
            StringPart::Literal(_) => true,
            StringPart::Expr(e) => is_eval_scalar(e),
        }),
        _ => false,
    }
}

/// Evaluate a linear recursive generator iteratively.
/// Converts `if cond then pre, (transform | self), post else else_branch end`
/// into a loop: emit pre values on the way down, emit post values on the way up.
fn eval_linear_recursive_gen(
    parts: LinearRecursiveGen<'_>,
    input: Value,
    env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    let LinearRecursiveGen { cond, pre, transform, post, else_branch } = parts;
    // Try pure numeric fast path (no env borrow needed per iteration)
    let numeric = {
        let e = env.borrow();
        if e.closures.is_empty() {
            // Check if condition is a numeric comparison and transform is numeric
            eval_bool_compound(cond, &input, &e.vars).is_some()
                && get_num_leaf(transform, &input, &e.vars).is_some()
        } else { false }
    };

    if numeric {
        // Pure numeric path: avoid eval_one overhead per iteration
        let mut current = input;
        let mut post_stack: Vec<Value> = Vec::new();
        loop {
            let cond_true = {
                let e = env.borrow();
                eval_bool_compound(cond, &current, &e.vars).unwrap_or(false)
            };
            if !cond_true { break; }
            // Emit pre
            let pre_val = eval_one(pre, &current, env).map_err(|_| anyhow::anyhow!("linear recursive gen: pre eval failed"))?;
            if !cb(pre_val)? { return Ok(false); }
            // Save post for later
            let post_val = eval_one(post, &current, env).map_err(|_| anyhow::anyhow!("linear recursive gen: post eval failed"))?;
            post_stack.push(post_val);
            // Transform
            let next = eval_one(transform, &current, env).map_err(|_| anyhow::anyhow!("linear recursive gen: transform eval failed"))?;
            current = next;
        }
        // Base case (else branch)
        let else_val = eval_one(else_branch, &current, env).map_err(|_| anyhow::anyhow!("linear recursive gen: else eval failed"))?;
        if !cb(else_val)? { return Ok(false); }
        // Unwind post values
        while let Some(v) = post_stack.pop() {
            if !cb(v)? { return Ok(false); }
        }
        Ok(true)
    } else {
        // General path with full eval dispatch
        let mut current = input;
        let mut post_stack: Vec<Value> = Vec::new();
        loop {
            let cond_true = {
                match eval_one(cond, &current, env) {
                    Ok(v) => v.is_truthy(),
                    Err(()) => {
                        let mut t = false;
                        eval(cond, current.clone(), env, &mut |v| { t = v.is_truthy(); Ok(true) })?;
                        t
                    }
                }
            };
            if !cond_true { break; }
            match eval_one(pre, &current, env) {
                Ok(v) => { if !cb(v)? { return Ok(false); } }
                Err(()) => { if !eval(pre, current.clone(), env, cb)? { return Ok(false); } }
            }
            match eval_one(post, &current, env) {
                Ok(v) => post_stack.push(v),
                Err(()) => {
                    let mut pv = Value::Null;
                    eval(post, current.clone(), env, &mut |v| { pv = v; Ok(true) })?;
                    post_stack.push(pv);
                }
            }
            let next = match eval_one(transform, &current, env) {
                Ok(v) => v,
                Err(()) => {
                    let mut nv = Value::Null;
                    eval(transform, current.clone(), env, &mut |v| { nv = v; Ok(true) })?;
                    nv
                }
            };
            current = next;
        }
        match eval_one(else_branch, &current, env) {
            Ok(v) => { if !cb(v)? { return Ok(false); } }
            Err(()) => { if !eval(else_branch, current.clone(), env, cb)? { return Ok(false); } }
        }
        while let Some(v) = post_stack.pop() {
            if !cb(v)? { return Ok(false); }
        }
        Ok(true)
    }
}

/// Body of `Expr::Update` extracted so `Expr::Mutate { kind: Update }` can
/// dispatch into the same path-update logic without cloning the path/value
/// sub-trees per invocation. See the comment on the Mutate arm in eval().
fn eval_update_body(
    path_expr: &Expr, update_expr: &Expr, input: Value, env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    // Single-output LHS (`.a`, `.a.b`, …) yields exactly one path and can't
    // hang, so keep the in-place fast path: collect the lone path (dropping
    // eval_path's input clone), move `input` into `result` so it is uniquely
    // held, and `rt_setpath_mut` mutates without a per-call deep clone — the
    // #652 `reduce gen as $x (acc; .[$x] |= …)` shape and per-record `.x += 1`
    // depend on this.
    if path_expr.is_single_output() {
        let mut paths = Vec::new();
        let path_result = eval_path(path_expr, input.clone(), env, &mut |p| { paths.push(p); Ok(true) });
        if let Err(e) = path_result {
            return Err(invalid_path_expr_err(e));
        }
        let mut result = input;
        let mut del_paths = Vec::new();
        for path in &paths {
            let old_val = crate::runtime::rt_getpath(&result, path).unwrap_or(Value::Null);
            let mut has_output = false;
            let mut new_val = Value::Null;
            // jq `|=` takes the FIRST value the RHS emits (`.a |= (1,2)` keeps
            // 1); `Ok(false)` stops after the first (#323).
            eval(update_expr, old_val, env, &mut |v| { has_output = true; new_val = v; Ok(false) })?;
            if has_output {
                let path_slice = match path {
                    Value::Arr(a) => a.as_slice(),
                    _ => bail!("Path must be specified as an array"),
                };
                crate::runtime::rt_setpath_mut(&mut result, path_slice, new_val)?;
            } else {
                del_paths.push(path.clone());
            }
        }
        if !del_paths.is_empty() {
            let dp = Value::Arr(Rc::new(del_paths));
            result = crate::runtime::rt_delpaths(&result, &dp)?;
        }
        return cb(result);
    }
    // Multi-output / generator LHS: apply each update as its path is produced,
    // matching jq's `_modify` (`reduce path(paths) as $p (.; …)`), instead of
    // materialising the whole path stream first. An infinite path generator
    // whose update invalidates a later step (`(recurse(.a)) |= 9`: setpath([];9)
    // makes the doc `9`, then the next `["a"]` setpath errors) now aborts at the
    // first bad setpath rather than looping forever collecting paths. After the
    // first touch `result` is uniquely held, so the remaining setpaths stay
    // in-place (only the first amortised clone is paid per call). #995
    let mut result = input.clone();
    let mut del_paths = Vec::new();
    let path_result = eval_path(path_expr, input, env, &mut |path| {
        let path_slice = match &path {
            Value::Arr(a) => a.as_slice(),
            _ => bail!("Path must be specified as an array"),
        };
        let old_val = crate::runtime::rt_getpath(&result, &path).unwrap_or(Value::Null);
        let mut has_output = false;
        let mut new_val = Value::Null;
        eval(update_expr, old_val, env, &mut |v| {
            has_output = true;
            new_val = v;
            Ok(false)
        })?;
        if has_output {
            crate::runtime::rt_setpath_mut(&mut result, path_slice, new_val)?;
        } else {
            del_paths.push(path.clone());
        }
        Ok(true)
    });
    if let Err(e) = path_result {
        return Err(invalid_path_expr_err(e));
    }
    if !del_paths.is_empty() {
        let dp = Value::Arr(Rc::new(del_paths));
        result = crate::runtime::rt_delpaths(&result, &dp)?;
    }
    cb(result)
}

/// Body of `Expr::Assign` extracted so `Expr::Mutate { kind: Assign }` can
/// dispatch into the same path-set logic without cloning the path/value
/// sub-trees per invocation.
fn eval_assign_body(
    path_expr: &Expr, value_expr: &Expr, input: Value, env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    // Single-output LHS: at most one path, can't hang. Keep the in-place fast
    // path — collect the lone path (dropping eval_path's clone), move `input`
    // into `result` (uniquely held), and rt_setpath_mut mutates without a
    // per-call deep clone (#659). Multi-output / generator LHS applies lazily so
    // `(recurse(.a)) = 9` aborts at the first invalid setpath instead of
    // materialising an unbounded path stream. #995
    let single = path_expr.is_single_output();

    // Scalar-value fast path: compute new_val via eval_one so the generator
    // form's `input.clone()` never sits alive in an outer callback frame.
    if let Ok(new_val) = eval_one(value_expr, &input, env) {
        if single {
            let mut paths = Vec::new();
            let path_result = eval_path(path_expr, input.clone(), env, &mut |p| { paths.push(p); Ok(true) });
            if let Err(e) = path_result {
                return Err(invalid_path_expr_err(e));
            }
            let mut result = input;
            for path in &paths {
                let path_slice = match path {
                    Value::Arr(a) => a.as_slice(),
                    _ => bail!("Path must be specified as an array"),
                };
                crate::runtime::rt_setpath_mut(&mut result, path_slice, new_val.clone())?;
            }
            return cb(result);
        }
        let mut result = input.clone();
        let path_result = eval_path(path_expr, input, env, &mut |path| {
            let path_slice = match &path {
                Value::Arr(a) => a.as_slice(),
                _ => bail!("Path must be specified as an array"),
            };
            crate::runtime::rt_setpath_mut(&mut result, path_slice, new_val.clone())?;
            Ok(true)
        });
        if let Err(e) = path_result {
            return Err(invalid_path_expr_err(e));
        }
        return cb(result);
    }
    eval(value_expr, input.clone(), env, &mut |new_val| {
        if single {
            let mut paths = Vec::new();
            let path_result = eval_path(path_expr, input.clone(), env, &mut |p| { paths.push(p); Ok(true) });
            if let Err(e) = path_result {
                return Err(invalid_path_expr_err(e));
            }
            let mut result = input.clone();
            for path in &paths {
                let path_slice = match path {
                    Value::Arr(a) => a.as_slice(),
                    _ => bail!("Path must be specified as an array"),
                };
                crate::runtime::rt_setpath_mut(&mut result, path_slice, new_val.clone())?;
            }
            return cb(result);
        }
        let mut result = input.clone();
        let path_result = eval_path(path_expr, input.clone(), env, &mut |path| {
            let path_slice = match &path {
                Value::Arr(a) => a.as_slice(),
                _ => bail!("Path must be specified as an array"),
            };
            crate::runtime::rt_setpath_mut(&mut result, path_slice, new_val.clone())?;
            Ok(true)
        });
        if let Err(e) = path_result {
            return Err(invalid_path_expr_err(e));
        }
        cb(result)
    })
}

/// `while(cond; update)` with full generator semantics.
///
/// jq desugars it to `def _while: if cond then ., (update | _while) else empty
/// end; _while;`. The update is therefore a *generator*: each value it yields is
/// an independent successor that re-enters the loop. The common case (update
/// yields exactly one value) stays a tight tail-iteration; an empty update
/// terminates the chain (#760), and a multi-valued update fans out (#767).
fn eval_while_gen(
    cond: &Expr, update: &Expr, always_true: bool, mut current: Value,
    env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    loop {
        if !always_true {
            // jq desugars `while` to `if cond then ., (update|_while) else empty
            // end`. A generator `cond` forks the `if` per output: each truthy
            // output emits the current value and recurses through `update`, each
            // falsy output contributes nothing. eval_one only succeeds for a
            // single-output cond (the hot path); a multi-valued (or erroring)
            // cond falls to the fork below. Folding it to the last output —
            // as the previous code did — dropped the multiplicity and could
            // even stop the loop on a trailing falsy value. #906
            match eval_one(cond, &current, env) {
                Ok(v) => {
                    if !v.is_truthy() { return Ok(true); }
                    // single truthy → emit + advance via the tail path below
                }
                Err(_) => {
                    let mut conds: Vec<bool> = Vec::new();
                    eval(cond, current.clone(), env, &mut |v| { conds.push(v.is_truthy()); Ok(true) })?;
                    for is_true in conds {
                        if !is_true { continue; }
                        if !cb(current.clone())? { return Ok(false); }
                        let mut succ: Vec<Value> = Vec::new();
                        eval(update, current.clone(), env, &mut |v| { succ.push(v); Ok(true) })?;
                        for u in succ {
                            if !stacker::maybe_grow(128 * 1024, 32 * 1024 * 1024,
                                || eval_while_gen(cond, update, always_true, u, env, cb))?
                            {
                                return Ok(false);
                            }
                        }
                    }
                    return Ok(true);
                }
            }
        }
        if !cb(current.clone())? { return Ok(false); }
        // Advance through `update`, honouring its generator semantics.
        if let Ok(next) = eval_one(update, &current, env) {
            current = next; // single value → tail-iterate (hot path)
            continue;
        }
        let mut succ: Vec<Value> = Vec::new();
        eval(update, current.clone(), env, &mut |v| { succ.push(v); Ok(true) })?;
        match succ.len() {
            0 => return Ok(true),                  // empty update → no successor (#760)
            1 => current = succ.pop().unwrap(),    // single value → tail-iterate
            _ => {                                 // multi-valued → fan out (#767)
                for u in succ {
                    if !stacker::maybe_grow(128 * 1024, 32 * 1024 * 1024,
                        || eval_while_gen(cond, update, always_true, u, env, cb))?
                    {
                        return Ok(false);
                    }
                }
                return Ok(true);
            }
        }
    }
}

/// `until(cond; update)` with full generator semantics.
///
/// jq desugars it to `def _until: if cond then . else (update | _until) end;
/// _until;`. As with `while`, the update is a generator: an empty update yields
/// no successor (so the loop terminates emitting nothing) and a multi-valued
/// update fans out, each value re-entering the loop.
fn eval_until_gen(
    cond: &Expr, update: &Expr, mut current: Value,
    env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    loop {
        // jq desugars `until` to `if cond then . else (update|_until) end`. A
        // generator `cond` forks the `if` per output: each truthy output emits
        // the current value, each falsy output recurses through `update`.
        // eval_one only succeeds for a single-output cond (the hot path); a
        // multi-valued (or erroring) cond falls to the fork below. Folding it to
        // the last output — as the previous code did — dropped the multiplicity
        // and, on a trailing falsy value, looped forever emitting nothing. #906
        match eval_one(cond, &current, env) {
            Ok(v) => {
                if v.is_truthy() { return cb(current); }
                // single falsy → advance via the tail path below
            }
            Err(_) => {
                let mut conds: Vec<bool> = Vec::new();
                eval(cond, current.clone(), env, &mut |v| { conds.push(v.is_truthy()); Ok(true) })?;
                for is_true in conds {
                    if is_true {
                        if !cb(current.clone())? { return Ok(false); }
                    } else {
                        let mut succ: Vec<Value> = Vec::new();
                        eval(update, current.clone(), env, &mut |v| { succ.push(v); Ok(true) })?;
                        for u in succ {
                            if !stacker::maybe_grow(128 * 1024, 32 * 1024 * 1024,
                                || eval_until_gen(cond, update, u, env, cb))?
                            {
                                return Ok(false);
                            }
                        }
                    }
                }
                return Ok(true);
            }
        }
        if let Ok(next) = eval_one(update, &current, env) {
            current = next; // single value → tail-iterate (hot path)
            continue;
        }
        let mut succ: Vec<Value> = Vec::new();
        eval(update, current.clone(), env, &mut |v| { succ.push(v); Ok(true) })?;
        match succ.len() {
            0 => return Ok(true),                  // empty update → no successor
            1 => current = succ.pop().unwrap(),    // single value → tail-iterate
            _ => {                                 // multi-valued → fan out
                for u in succ {
                    if !stacker::maybe_grow(128 * 1024, 32 * 1024 * 1024,
                        || eval_until_gen(cond, update, u, env, cb))?
                    {
                        return Ok(false);
                    }
                }
                return Ok(true);
            }
        }
    }
}

pub fn eval(
    expr: &Expr, input: Value, env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    match expr {
        Expr::Input => cb(input),
        Expr::Literal(lit) => cb(match lit {
            Literal::Null => Value::Null,
            Literal::True => Value::True,
            Literal::False => Value::False,
            Literal::Num(n, repr) => Value::number_opt(*n, repr.clone()),
            Literal::Str(s) => Value::from_str(s),
        }),

        Expr::BinOp { op, lhs, rhs } => {
            // Try scalar fast path first
            if let Ok(v) = eval_one(expr, &input, env) {
                return cb(v);
            }
            let op = *op;
            match op {
                BinOp::And => {
                    eval(lhs, input.clone(), env, &mut |lval| {
                        if !lval.is_truthy() {
                            cb(Value::False)
                        } else {
                            eval(rhs, input.clone(), env, &mut |rval| {
                                cb(Value::from_bool(rval.is_truthy()))
                            })
                        }
                    })
                }
                BinOp::Or => {
                    eval(lhs, input.clone(), env, &mut |lval| {
                        if lval.is_truthy() {
                            cb(Value::True)
                        } else {
                            eval(rhs, input.clone(), env, &mut |rval| {
                                cb(Value::from_bool(rval.is_truthy()))
                            })
                        }
                    })
                }
                BinOp::Add if matches!(rhs.as_ref(), Expr::Collect { .. }) => {
                    // Optimize `arr + [elems]`: push directly instead of creating intermediate array
                    let gen = match rhs.as_ref() { Expr::Collect { generator } => generator, _ => unreachable!() };
                    eval(lhs, input.clone(), env, &mut |lval| {
                        match lval {
                            Value::Arr(arr_rc) => {
                                // Try direct try_unwrap; on failure, deep-clone the Vec.
                                // We previously had a "drop env's copy" fast path for
                                // `LoadVar(vi) + [gen]`, but that mutated `vars[vi]` to
                                // Null and left a hole that subsequent generator
                                // iterations / sibling reads observed (#642).
                                match Rc::try_unwrap(arr_rc) {
                                    Ok(mut vec) => {
                                        eval(gen, input.clone(), env, &mut |elem| { vec.push(elem); Ok(true) })?;
                                        cb(Value::Arr(Rc::new(vec)))
                                    }
                                    Err(arr_rc) => {
                                        let mut vec = (*arr_rc).clone();
                                        eval(gen, input.clone(), env, &mut |elem| { vec.push(elem); Ok(true) })?;
                                        cb(Value::Arr(Rc::new(vec)))
                                    }
                                }
                            }
                            _ => {
                                // Not an array - fall back to normal add
                                let mut rhs_arr = Vec::new();
                                eval(gen, input.clone(), env, &mut |elem| { rhs_arr.push(elem); Ok(true) })?;
                                let rval = Value::Arr(Rc::new(rhs_arr));
                                cb(crate::runtime::rt_add_owned(lval, &rval)?)
                            }
                        }
                    })
                }
                _ => {
                    // jq evaluates rhs as outer generator, lhs as inner
                    eval(rhs, input.clone(), env, &mut |rval| {
                        eval(lhs, input.clone(), env, &mut |lval| {
                            cb(eval_binop_owned(op, lval, &rval)?)
                        })
                    })
                }
            }
        }

        Expr::UnaryOp { op, operand } => {
            eval(operand, input, env, &mut |val| cb(eval_unaryop(*op, &val)?))
        }

        Expr::Index { expr: base_expr, key: key_expr } => {
            // Try scalar fast path
            if let Ok(v) = eval_one(expr, &input, env) {
                return cb(v);
            }
            // jq iterates the subscript generator in the OUTER loop and the
            // base generator in the inner loop, so the leftmost generator
            // varies fastest: `.[0,1][0,1]` yields .[0][0], .[1][0], .[0][1],
            // .[1][1]. Nesting base outer reversed that order (#817).
            eval(key_expr, input.clone(), env, &mut |key| {
                eval(base_expr, input.clone(), env, &mut |base| {
                    match eval_index(&base, &key, false) {
                        Ok(v) => cb(v),
                        Err(msg) => bail!("{}", msg),
                    }
                })
            })
        }

        Expr::IndexOpt { expr: base_expr, key: key_expr } => {
            // Subscript generator outer, base inner — see Expr::Index (#817).
            eval(key_expr, input.clone(), env, &mut |key| {
                eval(base_expr, input.clone(), env, &mut |base| {
                    match eval_index(&base, &key, true) {
                        Ok(v) => cb(v),
                        Err(_) => Ok(true),
                    }
                })
            })
        }

        Expr::Pipe { left, right } => {
            // Fuse While/Until | scalar_expr to avoid cloning intermediate values
            match left.as_ref() {
                Expr::While { cond, update } => {
                    // Detect `. as $V | $V + [gen]` pattern for in-place array append
                    let append_info = if let Expr::LetBinding { var_index, value, body } = update.as_ref() {
                        if matches!(value.as_ref(), Expr::Input) {
                            if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = body.as_ref() {
                                if let (Expr::LoadVar { var_index: lv }, Expr::Collect { generator: gen }) = (lhs.as_ref(), rhs.as_ref()) {
                                    if *lv == *var_index { Some((*var_index, gen.as_ref())) } else { None }
                                } else { None }
                            } else { None }
                        } else { None }
                    } else { None };

                    let always_true = matches!(cond.as_ref(), Expr::Literal(Literal::True));
                    // Detect leading select in right side: `select(cond) | rest`
                    // Avoids redundant re-evaluation of the select condition on fallback.
                    let select_prefix = if let Expr::Pipe { left: sel, right: rest } = right.as_ref() {
                        if let Expr::IfThenElse { cond: sc, then_branch, else_branch } = sel.as_ref() {
                            if matches!(then_branch.as_ref(), Expr::Input) && matches!(else_branch.as_ref(), Expr::Empty) {
                                Some((sc.as_ref(), rest.as_ref()))
                            } else { None }
                        } else { None }
                    } else { None };
                    let mut current = input;
                    loop {
                        if !always_true {
                            let is_true = if let Ok(v) = eval_one(cond, &current, env) {
                                v.is_truthy()
                            } else {
                                let mut t = false;
                                eval(cond, current.clone(), env, &mut |v| { t = v.is_truthy(); Ok(true) })?;
                                t
                            };
                            if !is_true { break; }
                        }
                        // Try eval_one_filter on right to handle select/Empty without cloning
                        if let Some((sel_cond, rest)) = select_prefix {
                            // Try compound boolean evaluation first (no env borrow needed for pure Input+Literal)
                            let cond_true = if let Some(result) = eval_bool_compound(sel_cond, &current, &env.borrow().vars) {
                                result
                            } else {
                                match eval_one(sel_cond, &current, env) {
                                    Ok(v) => v.is_truthy(),
                                    Err(()) => {
                                        let mut t = false;
                                        eval(sel_cond, current.clone(), env, &mut |v| { t = v.is_truthy(); Ok(true) })?;
                                        t
                                    }
                                }
                            };
                            if cond_true {
                                // Select passed: evaluate rest directly (skip redundant cond re-check)
                                match eval_one_filter(rest, &current, env) {
                                    Ok(Some(v)) => { if !cb(v)? { return Ok(false); } }
                                    Ok(None) => {}
                                    Err(()) => { if !eval(rest, current.clone(), env, cb)? { return Ok(false); } }
                                }
                            }
                        } else {
                            match eval_one_filter(right, &current, env) {
                                Ok(Some(result)) => { if !cb(result)? { return Ok(false); } }
                                Ok(None) => { /* filtered out (select/empty), skip */ }
                                Err(()) => {
                                    if !eval(right, current.clone(), env, cb)? { return Ok(false); }
                                }
                            }
                        }
                        if let Some((v_idx, gen)) = append_info {
                            // In-place array append: move current into env, eval gen, take back, append
                            let old = env.borrow().get_var(v_idx);
                            env.borrow_mut().set_var(v_idx, current);
                            let mut elems = Vec::new();
                            let gen_result = eval(gen, Value::Null, env, &mut |elem| { elems.push(elem); Ok(true) });
                            // Take array back from env (refcount should be 1 after gen drops its refs)
                            let arr_val = std::mem::replace(&mut env.borrow_mut().vars[v_idx.idx()], old);
                            gen_result?;
                            current = match arr_val {
                                Value::Arr(rc) => {
                                    match Rc::try_unwrap(rc) {
                                        Ok(mut vec) => {
                                            vec.extend(elems);
                                            Value::Arr(Rc::new(vec))
                                        }
                                        Err(rc) => {
                                            let mut vec = (*rc).clone();
                                            vec.extend(elems);
                                            Value::Arr(Rc::new(vec))
                                        }
                                    }
                                }
                                other => {
                                    let rhs_val = Value::Arr(Rc::new(elems));
                                    crate::runtime::rt_add_owned(other, &rhs_val)?
                                }
                            };
                        } else if let Ok(next) = eval_one(update, &current, env) {
                            current = next;
                        } else {
                            // `update` is a generator here: 0, 1, or many successors.
                            let mut succ: Vec<Value> = Vec::new();
                            eval(update, current.clone(), env, &mut |v| { succ.push(v); Ok(true) })?;
                            match succ.len() {
                                0 => break,                        // empty update terminates (#760)
                                1 => current = succ.pop().unwrap(),
                                _ => {
                                    // Multi-valued update (#767): each successor spawns an
                                    // independent continuation of `while(cond; update)`, whose
                                    // values flow through the piped `right`. Delegate to the
                                    // generator helper so the linear fast path stays untouched.
                                    for u in succ {
                                        let go = eval_while_gen(cond, update, always_true, u, env,
                                            &mut |wv| eval(right, wv, env, cb))?;
                                        if !go { return Ok(false); }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    Ok(true)
                }
                // Fuse [generator] | all(predicate) → single-pass short-circuit
                Expr::Collect { generator } => {
                    match right.as_ref() {
                        Expr::AllShort { generator: all_gen, predicate }
                            if matches!(all_gen.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) =>
                        {
                            let mut all_true = true;
                            // Detect `. as $var | numeric_body` to bypass env entirely
                            let no_closures = env.borrow().closures.is_empty();
                            let numeric_bind = if no_closures {
                                if let Expr::LetBinding { var_index, value, body } = predicate.as_ref() {
                                    if matches!(value.as_ref(), Expr::Input) && !expr_uses_outer_input(body) {
                                        // Test if body is a pure numeric boolean expr
                                        if eval_bool_numeric(body, &env.borrow().vars, *var_index, 0.0).is_some() {
                                            Some((*var_index, body.as_ref()))
                                        } else { None }
                                    } else { None }
                                } else { None }
                            } else { None };
                            let let_bind = if numeric_bind.is_none() {
                                if let Expr::LetBinding { var_index, value, body } = predicate.as_ref() {
                                    if matches!(value.as_ref(), Expr::Input) && !expr_uses_outer_input(body) {
                                        Some((*var_index, body.as_ref()))
                                    } else { None }
                                } else { None }
                            } else { None };
                            // Pre-check: can we evaluate the predicate as a compound boolean
                            // with pre-cached vars? (one borrow before the loop, none during)
                            let pred_compound = if numeric_bind.is_none() && let_bind.is_none() {
                                // Test if predicate works with eval_bool_compound.
                                // Use 1.0 as dummy (not 0.0) to avoid false negatives from
                                // fmod/div-by-zero returning None in get_num_leaf.
                                let dummy = Value::number(1.0);
                                eval_bool_compound(predicate, &dummy, &env.borrow().vars).is_some()
                            } else { false };
                            eval(generator, input, env, &mut |elem| {
                                if let Some((vi, body)) = numeric_bind {
                                    // Ultra-fast path: pure f64 predicate, no env writes
                                    if let Value::Num(n, _) = &elem {
                                        let result = eval_bool_numeric(body, &env.borrow().vars, vi, *n).unwrap();
                                        return if result { Ok(true) } else { all_true = false; Ok(false) };
                                    }
                                }
                                if let Some((vi, body)) = let_bind {
                                    let old = std::mem::replace(&mut env.borrow_mut().vars[vi.idx()], elem);
                                    // jq's `all` is vacuously true when the
                                    // predicate yields no values for an
                                    // element, and false on the first falsy
                                    // value (#519).
                                    let is_true = match eval_one(body, &Value::Null, env) {
                                        Ok(v) => v.is_truthy(),
                                        Err(()) => {
                                            let mut found_falsy = false;
                                            eval(body, Value::Null, env, &mut |v| {
                                                if !v.is_truthy() { found_falsy = true; Ok(false) }
                                                else { Ok(true) }
                                            })?;
                                            !found_falsy
                                        }
                                    };
                                    env.borrow_mut().vars[vi.idx()] = old;
                                    if is_true { Ok(true) } else { all_true = false; Ok(false) }
                                } else {
                                    // Compound bool fast path: dummy-probe at line ~1609 may
                                    // succeed while the actual elem (null, mixed types, …) trips
                                    // the evaluator and yields None. Fall through to the generic
                                    // path instead of unwrapping.
                                    if pred_compound {
                                        if let Some(result) = eval_bool_compound(predicate, &elem, &env.borrow().vars) {
                                            return if result { Ok(true) } else { all_true = false; Ok(false) };
                                        }
                                    }
                                    let pred_result = eval_one(predicate, &elem, env);
                                    match pred_result {
                                        Ok(v) => {
                                            if v.is_truthy() { Ok(true) } else { all_true = false; Ok(false) }
                                        }
                                        Err(()) => {
                                            let mut found_falsy = false;
                                            eval(predicate, elem, env, &mut |v| {
                                                if !v.is_truthy() { found_falsy = true; Ok(false) }
                                                else { Ok(true) }
                                            })?;
                                            if found_falsy { all_true = false; Ok(false) } else { Ok(true) }
                                        }
                                    }
                                }
                            })?;
                            cb(Value::from_bool(all_true))
                        }
                        Expr::AnyShort { generator: any_gen, predicate }
                            if matches!(any_gen.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) =>
                        {
                            let mut any_true = false;
                            let let_bind = if let Expr::LetBinding { var_index, value, body } = predicate.as_ref() {
                                if matches!(value.as_ref(), Expr::Input) && !expr_uses_outer_input(body) {
                                    Some((*var_index, body.as_ref()))
                                } else { None }
                            } else { None };
                            eval(generator, input, env, &mut |elem| {
                                if let Some((vi, body)) = let_bind {
                                    if let Value::Num(n, _) = &elem {
                                        let e = env.borrow();
                                        if e.closures.is_empty() {
                                            if let Some(result) = eval_bool_numeric(body, &e.vars, vi, *n) {
                                                drop(e);
                                                return if result { any_true = true; Ok(false) } else { Ok(true) };
                                            }
                                        }
                                        drop(e);
                                    }
                                    let old = std::mem::replace(&mut env.borrow_mut().vars[vi.idx()], elem);
                                    // jq's `any` is vacuously false when the
                                    // predicate yields no values for an
                                    // element, and true on the first truthy
                                    // value (#519).
                                    let is_true = match eval_one(body, &Value::Null, env) {
                                        Ok(v) => v.is_truthy(),
                                        Err(()) => {
                                            let mut found_truthy = false;
                                            eval(body, Value::Null, env, &mut |v| {
                                                if v.is_truthy() { found_truthy = true; Ok(false) }
                                                else { Ok(true) }
                                            })?;
                                            found_truthy
                                        }
                                    };
                                    env.borrow_mut().vars[vi.idx()] = old;
                                    if is_true { any_true = true; Ok(false) } else { Ok(true) }
                                } else {
                                    let pred_result = eval_one(predicate, &elem, env);
                                    match pred_result {
                                        Ok(v) => {
                                            if v.is_truthy() { any_true = true; Ok(false) } else { Ok(true) }
                                        }
                                        Err(()) => {
                                            let mut found_truthy = false;
                                            eval(predicate, elem, env, &mut |v| {
                                                if v.is_truthy() { found_truthy = true; Ok(false) }
                                                else { Ok(true) }
                                            })?;
                                            if found_truthy { any_true = true; Ok(false) } else { Ok(true) }
                                        }
                                    }
                                }
                            })?;
                            cb(Value::from_bool(any_true))
                        }
                        _ => eval(left, input, env, &mut |mid| eval(right, mid, env, cb)),
                    }
                }
                _ => {
                    // Scalar fast path: avoid closure overhead when left produces one value
                    if let Ok(mid) = eval_one(left, &input, env) {
                        return eval(right, mid, env, cb);
                    }
                    eval(left, input, env, &mut |mid| eval(right, mid, env, cb))
                }
            }
        }

        Expr::Comma { left, right } => {
            let cont = eval(left, input.clone(), env, cb)?;
            if !cont { return Ok(false); }
            eval(right, input, env, cb)
        }

        Expr::Empty => Ok(true),

        Expr::IfThenElse { cond, then_branch, else_branch } => {
            // Try scalar fast path for the condition
            if let Ok(cond_val) = eval_one(cond, &input, env) {
                return if cond_val.is_truthy() {
                    eval(then_branch, input, env, cb)
                } else {
                    eval(else_branch, input, env, cb)
                };
            }
            eval(cond, input.clone(), env, &mut |cond_val| {
                if cond_val.is_truthy() {
                    eval(then_branch, input.clone(), env, cb)
                } else {
                    eval(else_branch, input.clone(), env, cb)
                }
            })
        }

        Expr::TryCatch { try_expr, catch_expr, .. } => {
            let cb_error = std::cell::Cell::new(false);
            let result = eval(try_expr, input.clone(), env, &mut |val| {
                match &val {
                    Value::Error(msg) => {
                        eval(catch_expr, Value::from_str(msg.as_str()), env, cb)
                    }
                    _ => {
                        let r = cb(val);
                        if r.is_err() {
                            cb_error.set(true);
                        }
                        r
                    }
                }
            });
            match result {
                Ok(cont) => Ok(cont),
                Err(e) if cb_error.get() => Err(e),
                Err(e) => {
                    // jq makes `break` a catchable signal: a `try`/`?` in the
                    // unwind path catches it (surfacing `{"__jq": id}`) and
                    // execution continues — only an uncaught break reaches its
                    // label. A break that came from the downstream callback is
                    // excluded above by `cb_error`. See #715.
                    if let Some(be) = e.downcast_ref::<BreakError>() {
                        return eval(catch_expr, break_catch_value(be.0), env, cb);
                    }
                    // A typed `error(value)` payload is recovered losslessly,
                    // dodging the lossy JSON round-trip (#844).
                    if let Some(ev) = e.downcast_ref::<ErrorValue>() {
                        let v = take_error_payload(ev);
                        return eval(catch_expr, v, env, cb);
                    }
                    let msg = format!("{}", e);
                    // halt / halt_error are non-recoverable: jq lets them
                    // propagate past `try ... catch` so the process exits with
                    // the requested code (#182).
                    if e.downcast_ref::<crate::signal::HaltSignal>().is_some() { return Err(e); }
                    let catch_val = if let Some(json) = msg.strip_prefix("__jqerror__:") {
                        crate::value::json_to_value(json).unwrap_or(Value::from_str(&msg))
                    } else {
                        Value::from_str(&msg)
                    };
                    eval(catch_expr, catch_val, env, cb)
                }
            }
        }

        Expr::Each { input_expr } => {
            eval(input_expr, input, env, &mut |container| {
                match &container {
                    Value::Arr(a) => {
                        for v in a.iter() {
                            if !cb(v.clone())? { return Ok(false); }
                        }
                        Ok(true)
                    }
                    Value::Obj(ObjInner(o)) => {
                        for v in o.values() {
                            if !cb(v.clone())? { return Ok(false); }
                        }
                        Ok(true)
                    }
                    // Use errdesc so number reprs survive (`0.0` stays
                    // `0.0`) and long values get jq's truncation. See #574.
                    _ => bail!("Cannot iterate over {}", crate::runtime::errdesc_pub(&container)),
                }
            })
        }

        Expr::EachOpt { input_expr } => {
            eval(input_expr, input, env, &mut |container| {
                match &container {
                    Value::Arr(a) => { for v in a.iter() { if !cb(v.clone())? { return Ok(false); } } Ok(true) }
                    Value::Obj(ObjInner(o)) => { for v in o.values() { if !cb(v.clone())? { return Ok(false); } } Ok(true) }
                    _ => Ok(true),
                }
            })
        }

        Expr::LetBinding { var_index, value, body } => {
            // Register `. as $var` so a path-mode `path($var)` in the body sees
            // the identity-path provenance (#837). Cheap push/pop; only fires
            // for the identity binding form.
            let _id_guard = if matches!(value.as_ref(), Expr::Input) {
                Some(push_identity_path_var(*var_index))
            } else {
                None
            };
            if matches!(value.as_ref(), Expr::Input) {
                // Fast path: `. as $var | body`
                let tmp_vi = *var_index;
                if !expr_uses_outer_input(body) {
                    // Detect destructuring: body is chain of LetBinding { vi, Index(LoadVar(tmp), i) }
                    let mut bindings: Vec<(VarIdx, usize)> = Vec::new();
                    let mut inner = body.as_ref();
                    while let Expr::LetBinding { var_index: vi, value: val, body: b } = inner {
                        if let Expr::Index { expr: base, key } = val.as_ref() {
                            if let Expr::LoadVar { var_index: lv } = base.as_ref() {
                                if *lv == tmp_vi {
                                    if let Expr::Literal(Literal::Num(n, _)) = key.as_ref() {
                                        bindings.push((*vi, *n as usize));
                                        inner = b;
                                        continue;
                                    }
                                }
                            }
                        }
                        break;
                    }
                    if !bindings.is_empty() && !expr_uses_var(inner, tmp_vi) {
                        // Optimized array destructuring: batch env operations into 2 borrow_muts.
                        // Setup: store input in tmp, extract elements, null tmp (1 borrow_mut).
                        // Teardown: restore all vars (1 borrow_mut).
                        let n_bind = bindings.len();
                        // Use stack storage for common 2-element case to avoid Vec allocation
                        let mut olds_buf: [(VarIdx, Value); 3] = [(VarIdx(0), Value::Null), (VarIdx(0), Value::Null), (VarIdx(0), Value::Null)];
                        let mut olds_vec: Vec<(VarIdx, Value)> = Vec::new();
                        let use_buf = n_bind <= 2;
                        let destructured = {
                            let mut e = env.borrow_mut();
                            let old_tmp = std::mem::replace(&mut e.vars[tmp_vi.idx()], input);
                            if use_buf { olds_buf[0] = (tmp_vi, old_tmp); }
                            else { olds_vec.push((tmp_vi, old_tmp)); }
                            let rc_opt = match &e.vars[tmp_vi.idx()] {
                                Value::Arr(rc) => Some(rc.clone()),
                                _ => None,
                            };
                            if let Some(rc) = rc_opt {
                                for (i, &(vi, idx)) in bindings.iter().enumerate() {
                                    let elem = rc.get(idx).cloned().unwrap_or(Value::Null);
                                    let old = std::mem::replace(&mut e.vars[vi.idx()], elem);
                                    if use_buf { olds_buf[i + 1] = (vi, old); }
                                    else { olds_vec.push((vi, old)); }
                                }
                                drop(rc);
                                e.vars[tmp_vi.idx()] = Value::Null;
                                true
                            } else {
                                false
                            }
                        };
                        let result = if destructured {
                            eval(inner, Value::Null, env, cb)
                        } else {
                            eval(body, Value::Null, env, cb)
                        };
                        // Restore all in one borrow_mut
                        {
                            let mut e = env.borrow_mut();
                            if use_buf {
                                if destructured {
                                    for i in (1..=n_bind).rev() {
                                        let (vi, old) = std::mem::replace(&mut olds_buf[i], (VarIdx(0), Value::Null));
                                        e.vars[vi.idx()] = old;
                                    }
                                }
                                e.vars[tmp_vi.idx()] = std::mem::replace(&mut olds_buf[0], (VarIdx(0), Value::Null)).1;
                            } else {
                                if destructured {
                                    while olds_vec.len() > 1 {
                                        let (vi, old) = olds_vec.pop().unwrap();
                                        e.vars[vi.idx()] = old;
                                    }
                                }
                                e.vars[tmp_vi.idx()] = olds_vec.pop().unwrap().1;
                            }
                        }
                        result
                    } else {
                        // No destructuring, normal path
                        let old = std::mem::replace(&mut env.borrow_mut().vars[tmp_vi.idx()], input);
                        let result = eval(body, Value::Null, env, cb);
                        env.borrow_mut().vars[tmp_vi.idx()] = old;
                        result
                    }
                } else {
                    let old = std::mem::replace(&mut env.borrow_mut().vars[tmp_vi.idx()], input.clone());
                    let result = eval(body, input, env, cb);
                    env.borrow_mut().vars[tmp_vi.idx()] = old;
                    result
                }
            } else if let Ok(val) = eval_one(value, &input, env) {
                // Scalar fast path: avoid closure + input clone
                let vi = var_index.idx();
                let old = std::mem::replace(&mut env.borrow_mut().vars[vi], val);
                let result = eval(body, input, env, cb);
                env.borrow_mut().vars[vi] = old;
                result
            } else {
                eval(value, input.clone(), env, &mut |val| {
                    let vi = var_index.idx();
                    let old = std::mem::replace(&mut env.borrow_mut().vars[vi], val);
                    let result = eval(body, input.clone(), env, cb);
                    env.borrow_mut().vars[vi] = old;
                    result
                })
            }
        }

        Expr::LoadVar { var_index } => {
            let e = env.borrow();
            if e.closures.is_empty() {
                let val = e.get_var(*var_index);
                drop(e);
                return cb(val);
            }
            // Resolve closure chains iteratively for LoadVar→LoadVar→...→env
            let mut idx = *var_index;
            drop(e);
            loop {
                let e = env.borrow();
                if let Some(c) = e.closures.iter().rev().find(|c| c.0 == idx) {
                    if let Expr::LoadVar { var_index: next_idx } = &c.1 {
                        idx = *next_idx;
                        continue;
                    }
                    let arg = c.1.clone();
                    drop(e);
                    return eval(&arg, input, env, cb);
                } else {
                    let val = e.get_var(idx);
                    drop(e);
                    return cb(val);
                }
            }
        }

        Expr::Collect { generator } => {
            let mut arr = Vec::new();
            eval(generator, input, env, &mut |val| { arr.push(val); Ok(true) })?;
            cb(Value::Arr(Rc::new(arr)))
        }

        Expr::ObjectConstruct { pairs } => {
            eval_object_construct(pairs, input, env, cb)
        }

        Expr::Reduce { source, init, var_index, acc_index, update } => {
            // Unwrap `. |= f` and `. = f` to just `f` — path `.` is identity.
            // `mutate(. |= f)` / `mutate(. = f)` is the same shape under the
            // marker; peel through Mutate so the marker doesn't suppress this
            // peephole (#666 follow-up).
            let update = match update.as_ref() {
                Expr::Update { path_expr, update_expr }
                    if matches!(path_expr.as_ref(), Expr::Input) => {
                        update_expr.as_ref()
                    },
                Expr::Assign { path_expr, value_expr }
                    if matches!(path_expr.as_ref(), Expr::Input) => value_expr.as_ref(),
                Expr::Mutate { path_expr, value_expr, .. }
                    if matches!(path_expr.as_ref(), Expr::Input) => value_expr.as_ref(),
                _ => update.as_ref(),
            };
            // For init = `.`, move input directly into the accumulator (no clone).
            // For sources that don't reference outer input (e.g. range), pass
            // Value::Null and drop our own input ref — without this the outer
            // reduce's input.clone() pins the Rc at refcount ≥ 2 across every
            // iteration, so nested `.[$i] = .[$i] + 1` bodies fall off the
            // in-place fast path and deep-copy the array on each step (#664).
            let init_is_input = matches!(init.as_ref(), Expr::Input);
            let source_needs_input = expr_uses_outer_input(source);
            // INIT is a generator: jq 1.8.1 runs the reduce once per INIT value
            // and emits one accumulator each — nothing for an `empty` INIT. The
            // single-accumulator fast paths below assume exactly one INIT value;
            // previously a multi-output INIT collapsed to its last value and an
            // `empty` INIT silently became `null`. Peek with the leaf evaluator
            // first (no allocation, no generator side effects) so the common
            // single-value case is byte-for-byte the old fast path; only a true
            // generator INIT (0 or ≥2 outputs) takes the per-INIT loop. #718
            let vi = *var_index;
            let ai = *acc_index;
            { let mut e = env.borrow_mut(); e.ensure_var(vi); e.ensure_var(ai); }
            let init_single: Option<Value> = if init_is_input {
                None
            } else if let Ok(v) = eval_one(init, &input, env) {
                Some(v)
            } else {
                let mut vs = Vec::new();
                eval(init, input.clone(), env, &mut |v| { vs.push(v); Ok(true) })?;
                if vs.len() == 1 {
                    Some(vs.pop().unwrap())
                } else {
                    // 0 or ≥2 INIT values: run the reduce once per value. jq
                    // threads the reduce's input through a single register, so it
                    // is clobbered after the first INIT value: the source sees the
                    // real input only on the first iteration and `null` thereafter.
                    // Hence `reduce .[] as $x ((0,100); .+$x)` on `[1,2,3]` emits
                    // the first result (6) then errors "Cannot iterate over null",
                    // while an input-independent source like `(1,2)` is unaffected
                    // (`reduce (1,2) as $x ((10,20); .+$x)` → 13, 23). #718.
                    let mut first = true;
                    for init_val in vs {
                        let src_input = if first { input.clone() } else { Value::Null };
                        first = false;
                        let mut acc = init_val;
                        eval(source, src_input, env, &mut |sv| {
                            let acc_val = std::mem::replace(&mut acc, Value::Null);
                            let (old_var, old_acc) = {
                                let mut e = env.borrow_mut();
                                let ov = std::mem::replace(&mut e.vars[vi.idx()], sv);
                                let oa = std::mem::replace(&mut e.vars[ai.idx()], acc_val.clone());
                                (ov, oa)
                            };
                            let r = eval(update, acc_val, env, &mut |new_acc| { acc = new_acc; Ok(true) });
                            {
                                let mut e = env.borrow_mut();
                                e.vars[ai.idx()] = old_acc;
                                e.vars[vi.idx()] = old_var;
                            }
                            r?;
                            Ok(true)
                        })?;
                        if !cb(acc)? { return Ok(false); }
                    }
                    return Ok(true);
                }
            };
            let (mut acc, source_input) = if init_is_input {
                if source_needs_input {
                    (input.clone(), input)
                } else {
                    (input, Value::Null)
                }
            } else {
                let acc_val = init_single.unwrap();
                if source_needs_input {
                    (acc_val, input)
                } else {
                    drop(input);
                    (acc_val, Value::Null)
                }
            };
            let acc_used_in_update = expr_uses_var(update, ai);
            // Detect fused Reduce+destructure pattern:
            // update = LetBinding(tmp, Input, LetBinding(a, Index(tmp, 0), LetBinding(b, Index(tmp, 1), inner)))
            // where inner doesn't use tmp and body doesn't use outer input.
            let fused = if !acc_used_in_update {
                if let Expr::LetBinding { var_index: tmp_vi, value, body } = update {
                    if matches!(value.as_ref(), Expr::Input) && !expr_uses_outer_input(body) {
                        let mut bindings: Vec<(VarIdx, usize)> = Vec::new();
                        let mut inner = body.as_ref();
                        while let Expr::LetBinding { var_index: bvi, value: bval, body: bb } = inner {
                            if let Expr::Index { expr: base, key } = bval.as_ref() {
                                if let Expr::LoadVar { var_index: lv } = base.as_ref() {
                                    if *lv == *tmp_vi {
                                        if let Expr::Literal(Literal::Num(n, _)) = key.as_ref() {
                                            bindings.push((*bvi, *n as usize));
                                            inner = bb;
                                            continue;
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        if !bindings.is_empty() && !expr_uses_var(inner, *tmp_vi) {
                            Some((*tmp_vi, bindings, inner))
                        } else { None }
                    } else { None }
                } else { None }
            } else { None };
            if let Some((tmp_vi, ref bindings, inner_body)) = fused {
                // Fused Reduce+destructure: batch $var store + array destructure into 2 borrow_muts
                eval(source, source_input, env, &mut |val| {
                    let acc_val = std::mem::replace(&mut acc, Value::Null);
                    // Setup: store $var, destructure acc, null tmp (1 borrow_mut)
                    let (old_var, old_tmp, destructured) = {
                        let mut e = env.borrow_mut();
                        let old_var = std::mem::replace(&mut e.vars[vi.idx()], val);
                        let old_tmp = std::mem::replace(&mut e.vars[tmp_vi.idx()], acc_val);
                        let rc_opt = match &e.vars[tmp_vi.idx()] {
                            Value::Arr(rc) => Some(rc.clone()),
                            _ => None,
                        };
                        if let Some(rc) = rc_opt {
                            for &(bvi, idx) in bindings {
                                let elem = rc.get(idx).cloned().unwrap_or(Value::Null);
                                e.vars[bvi.idx()] = elem;
                            }
                            drop(rc);
                            e.vars[tmp_vi.idx()] = Value::Null;
                            (old_var, old_tmp, true)
                        } else {
                            (old_var, old_tmp, false)
                        }
                    };
                    if destructured {
                        eval(inner_body, Value::Null, env, &mut |new_acc| { acc = new_acc; Ok(true) })?;
                    } else {
                        // Not an array; restore tmp and run update normally
                        env.borrow_mut().vars[tmp_vi.idx()] = old_tmp;
                        let acc_val_for_update = env.borrow().get_var(tmp_vi);
                        eval(update, acc_val_for_update, env, &mut |new_acc| { acc = new_acc; Ok(true) })?;
                    }
                    // Teardown: restore $var (1 borrow_mut)
                    // Note: destructured bindings' old values were whatever was in env before;
                    // since we're in a reduce loop and these are scratch vars, we just restore $var.
                    env.borrow_mut().vars[vi.idx()] = old_var;
                    Ok(true)
                })?;
                cb(acc)
            } else if !acc_used_in_update {
                // Detect `. + rhs` pattern where rhs doesn't use accumulator — in-place merge
                // Also detect `+= rhs` which is: LetBinding { var, value: rhs, body: Update { path: ., update: . + LoadVar(var) } }
                let add_inplace = if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = update {
                    if matches!(lhs.as_ref(), Expr::Input) && !expr_uses_outer_input(rhs) {
                        Some((rhs.as_ref(), None::<VarIdx>))
                    } else { None }
                } else if let Expr::LetBinding { var_index: rhs_var, value: rhs_value, body } = update {
                    // `. += rhs` pattern
                    if let Expr::Update { path_expr, update_expr } = body.as_ref() {
                        if matches!(path_expr.as_ref(), Expr::Input) {
                            if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = update_expr.as_ref() {
                                if matches!(lhs.as_ref(), Expr::Input)
                                    && matches!(rhs.as_ref(), Expr::LoadVar { var_index: v } if *v == *rhs_var)
                                    && !expr_uses_outer_input(rhs_value)
                                {
                                    Some((rhs_value.as_ref(), Some(*rhs_var)))
                                } else { None }
                            } else { None }
                        } else { None }
                    } else { None }
                } else { None };
                if let Some((add_rhs, _temp_var)) = add_inplace {
                    eval(source, source_input, env, &mut |val| {
                        let old_var = std::mem::replace(&mut env.borrow_mut().vars[vi.idx()], val);
                        let rhs_val = {
                            let mut r = Value::Null;
                            eval(add_rhs, Value::Null, env, &mut |v| { r = v; Ok(true) })?;
                            r
                        };
                        match (&mut acc, rhs_val) {
                            (Value::Obj(ObjInner(o)), Value::Obj(ObjInner(rhs_obj))) => {
                                let obj = Rc::make_mut(o);
                                for (k, v) in rhs_obj.iter() {
                                    obj.insert(k.clone(), v.clone());
                                }
                            }
                            (Value::Arr(a), Value::Arr(rhs_arr)) => {
                                let arr = Rc::make_mut(a);
                                arr.extend(rhs_arr.iter().cloned());
                            }
                            (Value::Str(s), Value::Str(rhs_s)) => {
                                s.push_str(rhs_s.as_str());
                            }
                            (acc_ref, rhs_val) => {
                                let acc_val = std::mem::replace(acc_ref, Value::Null);
                                *acc_ref = crate::runtime::rt_add(&acc_val, &rhs_val)?;
                            }
                        }
                        env.borrow_mut().vars[vi.idx()] = old_var;
                        Ok(true)
                    })?;
                    cb(acc)
                } else {
                    eval(source, source_input, env, &mut |val| {
                        let acc_val = std::mem::replace(&mut acc, Value::Null);
                        let old_var = std::mem::replace(&mut env.borrow_mut().vars[vi.idx()], val);
                        eval(update, acc_val, env, &mut |new_acc| { acc = new_acc; Ok(true) })?;
                        env.borrow_mut().vars[vi.idx()] = old_var;
                        Ok(true)
                    })?;
                    cb(acc)
                }
            } else {
                eval(source, source_input, env, &mut |val| {
                    let acc_val = std::mem::replace(&mut acc, Value::Null);
                    let (old_var, old_acc) = {
                        let mut e = env.borrow_mut();
                        let ov = std::mem::replace(&mut e.vars[vi.idx()], val);
                        let oa = std::mem::replace(&mut e.vars[ai.idx()], acc_val.clone());
                        (ov, oa)
                    };
                    eval(update, acc_val, env, &mut |new_acc| { acc = new_acc; Ok(true) })?;
                    {
                        let mut e = env.borrow_mut();
                        e.vars[ai.idx()] = old_acc;
                        e.vars[vi.idx()] = old_var;
                    }
                    Ok(true)
                })?;
                cb(acc)
            }
        }

        Expr::Foreach { source, init, var_index, acc_index, update, extract } => {
            let vi = *var_index;
            let ai = *acc_index;
            { let mut e = env.borrow_mut(); e.ensure_var(vi); e.ensure_var(ai); }
            // Fast path: foreach with null init and null update (pure transform/filter pattern, e.g. takeWhile)
            let trivial_acc = matches!(init.as_ref(), Expr::Literal(Literal::Null))
                && matches!(update.as_ref(), Expr::Literal(Literal::Null));
            if trivial_acc {
                if let Some(extract_expr) = extract {
                    // Detect takeWhile pattern: if $item | cond then $item else break/empty
                    // Avoids redundant LoadVar reads by reusing val directly.
                    if let Expr::IfThenElse { cond, then_branch, else_branch } = extract_expr.as_ref() {
                        if let Expr::Pipe { left, right: cond_body } = cond.as_ref() {
                            if matches!(left.as_ref(), Expr::LoadVar { var_index: lvi } if *lvi == vi)
                                && matches!(then_branch.as_ref(), Expr::LoadVar { var_index: tvi } if *tvi == vi)
                            {
                                let else_is_break = matches!(else_branch.as_ref(), Expr::Break { .. });
                                let else_is_empty = matches!(else_branch.as_ref(), Expr::Empty);
                                if else_is_break || else_is_empty {
                                    // If cond_body doesn't reference $item, we can evaluate it
                                    // before storing val in env, avoiding clone + borrow_mut on
                                    // the common (true) path.
                                    let cond_needs_var = expr_uses_var(cond_body, vi);
                                    return eval(source, input, env, &mut |val| {
                                        if cond_needs_var {
                                            // Slow path: cond_body references $item
                                            let old_var = std::mem::replace(
                                                &mut env.borrow_mut().vars[vi.idx()], val.clone());
                                            let is_true = match eval_one(cond_body, &val, env) {
                                                Ok(v) => v.is_truthy(),
                                                Err(()) => {
                                                    let mut t = false;
                                                    eval(cond_body, val.clone(), env, &mut |v| {
                                                        t = v.is_truthy(); Ok(true)
                                                    })?;
                                                    t
                                                }
                                            };
                                            let cont = if is_true {
                                                cb(val)?
                                            } else if else_is_break {
                                                env.borrow_mut().vars[vi.idx()] = old_var;
                                                return eval(else_branch, Value::Null, env, cb);
                                            } else {
                                                true
                                            };
                                            env.borrow_mut().vars[vi.idx()] = old_var;
                                            Ok(cont)
                                        } else {
                                            // Fast path: evaluate cond before touching env
                                            // Try compound boolean first (single borrow for nested And/Or)
                                            let is_true = if let Some(result) = eval_bool_compound(cond_body, &val, &env.borrow().vars) {
                                                result
                                            } else {
                                                match eval_one(cond_body, &val, env) {
                                                    Ok(v) => v.is_truthy(),
                                                    Err(()) => {
                                                        let mut t = false;
                                                        eval(cond_body, val.clone(), env, &mut |v| {
                                                            t = v.is_truthy(); Ok(true)
                                                        })?;
                                                        t
                                                    }
                                                }
                                            };
                                            if is_true {
                                                Ok(cb(val)?)
                                            } else if else_is_break {
                                                // Store val in env only for break path
                                                let old_var = std::mem::replace(
                                                    &mut env.borrow_mut().vars[vi.idx()], val);
                                                let r = eval(else_branch, Value::Null, env, cb);
                                                env.borrow_mut().vars[vi.idx()] = old_var;
                                                r
                                            } else {
                                                Ok(true)
                                            }
                                        }
                                    });
                                }
                            }
                        }
                    }
                    return eval(source, input, env, &mut |val| {
                        let old_var = std::mem::replace(&mut env.borrow_mut().vars[vi.idx()], val);
                        let cont = match eval_one_filter(extract_expr, &Value::Null, env) {
                            Ok(Some(v)) => cb(v)?,
                            Ok(None) => true,
                            Err(()) => eval(extract_expr, Value::Null, env, cb)?,
                        };
                        env.borrow_mut().vars[vi.idx()] = old_var;
                        Ok(cont)
                    });
                }
            }
            let acc_used = expr_uses_var(update, ai)
                || extract.as_ref().is_some_and(|e| expr_uses_var(e, ai));
            eval(init, input.clone(), env, &mut |init_val| {
                let mut acc = init_val;
                eval(source, input.clone(), env, &mut |val| {
                    let acc_val = std::mem::replace(&mut acc, Value::Null);
                    let (old_var, old_acc) = {
                        let mut e = env.borrow_mut();
                        let ov = std::mem::replace(&mut e.vars[vi.idx()], val);
                        let oa = if acc_used {
                            std::mem::replace(&mut e.vars[ai.idx()], acc_val.clone())
                        } else {
                            Value::Null
                        };
                        (ov, oa)
                    };
                    let mut stopped = false;
                    // jq semantics: for each value yielded by update, update the
                    // accumulator and emit extract(acc) (or acc itself when no extract).
                    let update_result = eval(update, acc_val, env, &mut |new_acc| {
                        acc = new_acc.clone();
                        if acc_used {
                            env.borrow_mut().vars[ai.idx()] = new_acc.clone();
                        }
                        let cont = if let Some(extract_expr) = extract {
                            match eval_one_filter(extract_expr, &new_acc, env) {
                                Ok(Some(v)) => cb(v)?,
                                Ok(None) => true,
                                Err(()) => eval(extract_expr, new_acc, env, cb)?,
                            }
                        } else {
                            cb(new_acc)?
                        };
                        if !cont { stopped = true; }
                        Ok(cont)
                    });
                    {
                        let mut e = env.borrow_mut();
                        if acc_used {
                            e.vars[ai.idx()] = old_acc;
                        }
                        e.vars[vi.idx()] = old_var;
                    }
                    update_result?;
                    Ok(!stopped)
                })
            })
        }

        Expr::Alternative { primary, fallback } => {
            // `A // B`: yield each truthy value from A; fall back to B only when
            // A emits nothing non-false/non-null. Errors must propagate — use
            // `f?` or `try f catch g` to suppress them, per jq semantics.
            let mut has_output = false;
            let result = eval(primary, input.clone(), env, &mut |val| {
                if val.is_truthy() { has_output = true; cb(val) } else { Ok(true) }
            });
            match result {
                Ok(_) if !has_output => eval(fallback, input, env, cb),
                Ok(cont) => Ok(cont),
                Err(e) => Err(e),
            }
        }

        Expr::Negate { operand } => {
            eval(operand, input, env, &mut |val| {
                match &val {
                    Value::Num(n, NumRepr(repr)) => {
                        // jq normalises `-(0)` back to `+0` (the literal `-0`
                        // and the `Negate` expr never produce a signed zero —
                        // only IEEE arithmetic like `0 * -1` does). Issue #110.
                        let neg = if *n == 0.0 { 0.0 } else { -*n };
                        cb(Value::number_opt(neg, crate::value::Value::negate_repr(repr.clone())))
                    }
                    _ => {
                        bail!("{} cannot be negated", crate::runtime::errdesc_pub(&val))
                    }
                }
            })
        }

        Expr::Not => cb(if input.is_truthy() { Value::False } else { Value::True }),

        Expr::Recurse { input_expr } => eval_recurse_expr(input_expr, &input, env, cb),

        Expr::Range { from, to, step } => {
            // Fast path: from/to/step are almost always scalar
            if let (Ok(from_val), Ok(to_val)) = (eval_one(from, &input, env), eval_one(to, &input, env)) {
                if let Some(step_expr) = step.as_ref() {
                    if let Ok(step_val) = eval_one(step_expr, &input, env) {
                        return eval_range(&from_val, &to_val, Some(&step_val), cb);
                    }
                } else {
                    return eval_range(&from_val, &to_val, None, cb);
                }
            }
            eval(from, input.clone(), env, &mut |from_val| {
                eval(to, input.clone(), env, &mut |to_val| {
                    if let Some(step_expr) = step.as_ref() {
                        eval(step_expr, input.clone(), env, &mut |step_val| {
                            eval_range(&from_val, &to_val, Some(&step_val), cb)
                        })
                    } else {
                        eval_range(&from_val, &to_val, None, cb)
                    }
                })
            })
        }

        Expr::Label { var_index, body } => {
            // Allocate a fresh runtime label id and bind it to the label var,
            // saving the slot's prior value. The save/restore matters when the
            // same lexical `label $x` is dynamically re-entered — e.g. a
            // `recurse(f)` whose `f` contains `label $x | … break $x`, which
            // nests because the recursion is lazy/depth-first. Without the
            // restore, a `break $x` evaluated after an inner instance exited
            // read the inner (stale) id and unwound to the wrong label. #916
            let (label_id, old) = {
                let mut e = env.borrow_mut();
                let id = e.next_label;
                e.next_label = id + 1;
                let old = e.get_var(*var_index);
                e.set_var(*var_index, Value::number(id as f64));
                (id, old)
            };
            let result = eval(body, input, env, cb);
            env.borrow_mut().set_var(*var_index, old);
            match result {
                Err(e) => {
                    if let Some(be) = e.downcast_ref::<BreakError>() {
                        if be.0 == label_id { return Ok(true); }
                    }
                    Err(e)
                }
                other => other,
            }
        }

        Expr::Break { var_index, .. } => {
            let label = env.borrow().get_var(*var_index);
            if let Value::Num(n, NumRepr(None)) = &label {
                return Err(BreakError(*n as u64).into());
            }
            bail!("break: invalid label")
        }

        Expr::Update { path_expr, update_expr } => {
            return eval_update_body(path_expr, update_expr, input, env, cb);
        }

        Expr::Assign { path_expr, value_expr } => {
            return eval_assign_body(path_expr, value_expr, input, env, cb);
        }

        Expr::Mutate { path_expr, value_expr, kind } => {
            // mutate(...) is semantically identical to its wrapped Update/
            // Assign at the eval layer — the in-place fast path is already
            // engaged because the Update/Assign handlers move `input` into
            // `result` rather than cloning. The marker's payoff is in the
            // JIT, where it suppresses the input Clone before setpath calls
            // so refcount stays at 1 in nested `reduce` contexts. See #666.
            //
            // Dispatch directly into the eval Update/Assign helpers — the
            // previous implementation built a fresh `Expr::Update` /
            // `Expr::Assign` per invocation, cloning the path and value
            // sub-trees every reduce iteration. That clone was the
            // observable mutate(...) regression vs. the unmarked form
            // (#666 follow-up).
            trace_mutate_event(*kind, &input);
            return match kind {
                MutateKind::Update => eval_update_body(path_expr, value_expr, input, env, cb),
                MutateKind::Assign => eval_assign_body(path_expr, value_expr, input, env, cb),
            };
        }

        Expr::PathExpr { expr: path_expr } => {
            let result = eval_path(path_expr, input, env, cb);
            match result {
                Err(e) => Err(invalid_path_expr_err(e)),
                other => other,
            }
        }

        Expr::SetPath { path, value } => {
            if !expr_uses_outer_input(path) && !expr_uses_outer_input(value)
                && path.is_single_output() && value.is_single_output() {
                // path and value don't reference `.` and each yields exactly one
                // output — avoid cloning input so Rc refcount stays 1 and
                // rt_setpath_mut can mutate in-place. Multi-output or `empty`
                // generators must take the iterating branch below: `setpath(p; g)`
                // emits one result per `g` value (and nothing for `empty`); the
                // last-value-wins shortcut here collapsed `(1,2)` to `2` and
                // turned `empty` into `null` (#717).
                let pv = {
                    let mut r = Value::Null;
                    eval(path, Value::Null, env, &mut |v| { r = v; Ok(true) })?;
                    r
                };
                let val = {
                    let mut r = Value::Null;
                    eval(value, Value::Null, env, &mut |v| { r = v; Ok(true) })?;
                    r
                };
                let mut base = input;
                if let Value::Arr(ref p) = pv {
                    crate::runtime::rt_setpath_mut(&mut base, p, val)?;
                    cb(base)
                } else {
                    cb(crate::runtime::rt_setpath(&base, &pv, &val)?)
                }
            } else {
                // jq iterates the value generator in the outer loop and the
                // path generator in the inner loop: `setpath((["a"],["b"]); (1,2))`
                // yields a/1, b/1, a/2, b/2. Match that nesting order.
                eval(value, input.clone(), env, &mut |v| {
                    eval(path, input.clone(), env, &mut |pv| {
                        cb(crate::runtime::rt_setpath(&input, &pv, &v)?)
                    })
                })
            }
        }

        Expr::GetPath { path } => {
            eval(path, input.clone(), env, &mut |pv| {
                cb(crate::runtime::rt_getpath(&input, &pv)?)
            })
        }

        Expr::DelPaths { paths } => {
            eval(paths, input.clone(), env, &mut |pv| {
                cb(crate::runtime::rt_delpaths(&input, &pv)?)
            })
        }

        Expr::FuncCall { func_id, args } => {
            // Consolidated function call: single env borrow for func lookup + recursive check + cache hit
            enum FuncAction {
                Direct(Rc<CompiledFunc>, FuncId),
                Recursive(Rc<CompiledFunc>),
                CacheHit(Rc<Expr>),
                CacheMiss(Rc<CompiledFunc>),
            }
            // An arg that itself calls the same def re-enters with the
            // same parser-allocated parameter slots. In the CPS eval model
            // the inner frame's writes leak through the outer's
            // continuation: the outer LetBinding for $y fires its cb while
            // $x is still bound to the inner's value, so the outer body
            // reads the inner's $x. Route through the rename path so the
            // outer frame gets its own slots — same fix the recursive-body
            // arms already apply (#679).
            let arg_recursive = args.iter().any(|a| contains_func_call(a, *func_id));
            let action = {
                let e = env.borrow();
                let func = match e.funcs.get(func_id.idx()) {
                    Some(f) => f.clone(),
                    None => bail!("undefined function id {}", func_id),
                };
                if func.param_vars.is_empty() || args.is_empty() {
                    FuncAction::Direct(func, *func_id)
                } else {
                    let is_recursive = e.recursive_cache.iter()
                        .find(|(k, _)| *k == *func_id)
                        .map(|&(_, r)| r);
                    if is_recursive == Some(true) || arg_recursive {
                        FuncAction::Recursive(func)
                    } else if is_recursive == Some(false) {
                        // Known non-recursive: try cache lookup in same borrow
                        // Single-arg LoadVar fast path: avoid Vec allocation
                        let cached = if args.len() == 1 {
                            if let Expr::LoadVar { var_index: vi0 } = &args[0] {
                                e.subst_cache.iter()
                                    .find(|((fid, vis), _)| *fid == *func_id && vis.len() == 1 && vis[0] == *vi0)
                                    .map(|(_, v)| v.clone())
                            } else {
                                let args_ptr = args.as_ptr() as usize;
                                e.subst_ptr_cache.iter()
                                    .find(|(fid, ptr, _)| *fid == *func_id && *ptr == args_ptr)
                                    .map(|(_, _, body)| body.clone())
                            }
                        } else {
                            // Multi-arg: check all LoadVar
                            let all_loadvar: Option<Vec<VarIdx>> = args.iter().map(|a| {
                                if let Expr::LoadVar { var_index } = a { Some(*var_index) } else { None }
                            }).collect();
                            if let Some(ref vis) = all_loadvar {
                                e.subst_cache.iter()
                                    .find(|((fid, v), _)| *fid == *func_id && v == vis)
                                    .map(|(_, v)| v.clone())
                            } else {
                                let args_ptr = args.as_ptr() as usize;
                                e.subst_ptr_cache.iter()
                                    .find(|(fid, ptr, _)| *fid == *func_id && *ptr == args_ptr)
                                    .map(|(_, _, body)| body.clone())
                            }
                        };
                        match cached {
                            Some(body) => FuncAction::CacheHit(body),
                            None => FuncAction::CacheMiss(func),
                        }
                    } else {
                        // Unknown recursive status: need to check
                        FuncAction::CacheMiss(func)
                    }
                }
            };
            match action {
                FuncAction::Direct(func, fid) => {
                    // Detect linear recursive generator: if cond then A, (transform | f), B else E end
                    // Convert to iterative loop to avoid deep recursion overhead.
                    if contains_func_call(&func.body, fid) {
                        if let Some(parts) = detect_linear_recursive_gen(&func.body, fid) {
                            return eval_linear_recursive_gen(parts, input, env, cb);
                        }
                        // Recursive zero-arg (or arity-mismatched) call: rename
                        // the body's local bindings to fresh indices so each
                        // frame's `as $x` / Reduce / Foreach / Label gets its
                        // own slot. Without this, the callback-driven eval
                        // model lets an outer frame's body run while an inner
                        // frame still owns the shared slot, and the outer
                        // read sees the inner value (#635). The non-zero-arg
                        // recursive path already routes through `Recursive` /
                        // `CacheMiss` for the same reason.
                        //
                        // Roll `next_var` back when the frame returns: indices
                        // only need to be unique among CONCURRENTLY-active
                        // frames, not across an entire run's total calls.
                        // Without the rollback, every recursive call (including
                        // backtracking branches that already returned) leaves
                        // its rename slots claimed forever, and a deep solver
                        // (e.g. Sudoku) overflows the u16 counter back into
                        // parse-time index territory — #653.
                        let nv_before = env.borrow().next_var;
                        let mut nv = nv_before;
                        let body = substitute_and_rename(&func.body, &[], &[], &mut nv);
                        env.borrow_mut().next_var = nv;
                        let result = stacker::maybe_grow(128 * 1024, 32 * 1024 * 1024, || eval(&body, input, env, cb));
                        env.borrow_mut().next_var = nv_before;
                        return result;
                    }
                    stacker::maybe_grow(128 * 1024, 32 * 1024 * 1024, || eval(&func.body, input, env, cb))
                }
                FuncAction::CacheHit(body) => eval(&body, input, env, cb),
                FuncAction::Recursive(func) => {
                    let nv_before = env.borrow().next_var;
                    let mut nv = nv_before;
                    let body = substitute_and_rename(&func.body, &func.param_vars, args, &mut nv);
                    env.borrow_mut().next_var = nv;
                    let result = stacker::maybe_grow(128 * 1024, 32 * 1024 * 1024, || eval(&body, input, env, cb));
                    env.borrow_mut().next_var = nv_before;
                    result
                }
                FuncAction::CacheMiss(func) => {
                    // Check if recursive (first call or unknown)
                    let is_recursive = {
                        let e = env.borrow();
                        match e.recursive_cache.iter().find(|(k, _)| *k == *func_id) {
                            Some(&(_, r)) => r,
                            None => {
                                drop(e);
                                let r = contains_func_call(&func.body, *func_id);
                                env.borrow_mut().recursive_cache.push((*func_id, r));
                                r
                            }
                        }
                    };
                    if is_recursive {
                        let nv_before = env.borrow().next_var;
                        let mut nv = nv_before;
                        let body = substitute_and_rename(&func.body, &func.param_vars, args, &mut nv);
                        env.borrow_mut().next_var = nv;
                        let result = stacker::maybe_grow(128 * 1024, 32 * 1024 * 1024, || eval(&body, input, env, cb));
                        env.borrow_mut().next_var = nv_before;
                        result
                    } else {
                        let body = substitute_params(&func.body, &func.param_vars, args);
                        let body_rc = Rc::new(body);
                        // Single-arg LoadVar fast path for caching
                        if args.len() == 1 {
                            if let Expr::LoadVar { var_index: vi0 } = &args[0] {
                                env.borrow_mut().subst_cache.push(((*func_id, vec![*vi0]), body_rc.clone()));
                                return eval(&body_rc, input, env, cb);
                            }
                        }
                        let all_loadvar: Option<Vec<VarIdx>> = args.iter().map(|a| {
                            if let Expr::LoadVar { var_index } = a { Some(*var_index) } else { None }
                        }).collect();
                        if let Some(var_indices) = all_loadvar {
                            env.borrow_mut().subst_cache.push(((*func_id, var_indices), body_rc.clone()));
                        } else {
                            let args_ptr = args.as_ptr() as usize;
                            env.borrow_mut().subst_ptr_cache.push((*func_id, args_ptr, body_rc.clone()));
                        }
                        eval(&body_rc, input, env, cb)
                    }
                }
            }
        }

        Expr::StringInterpolation { parts } => {
            eval_interp_parts(parts, parts.len() as isize - 1, String::new(), input, env, cb)
        }

        Expr::Limit { count, generator } => {
            eval(count, input.clone(), env, &mut |cv| {
                // Match `Num(n, _)` (any repr) so `limit(2.0; ...)` works
                // — the prior `NumRepr(None)` constraint silently rejected
                // float-formatted counts (#539).
                match &cv {
                    Value::Num(n, _) => {
                        let n = *n;
                        // jq emits while a 1-based counter stays below `n` and
                        // breaks once it reaches `n` — i.e. ceil(n) items for
                        // positive n. Comparing the integer counter against the
                        // raw f64 (instead of truncating `n` to an integer)
                        // matches jq for fractional counts: `*n as i64` floored,
                        // dropping the ceil item (#719). The JIT path already
                        // does this float compare (jit.rs emit_yield); this
                        // aligns the eval/interpreter path with it and with jq.
                        // jq orders NaN below every number, so `count < 0` is
                        // true for NaN and it takes the negative-count error
                        // path rather than acting as an unbounded count (#813).
                        if n < 0.0 || n.is_nan() {
                            bail!("__jqerror__:\"limit doesn't support negative count\"");
                        }
                        if n == 0.0 { return Ok(true); } // 0.0 and -0.0 → empty
                        let mut emitted: i64 = 0;
                        let mut stopped_by_outer = false;
                        let result = eval(generator, input.clone(), env, &mut |val| {
                            emitted += 1;
                            let cont = cb(val)?;
                            if !cont {
                                stopped_by_outer = true;
                                Ok(false)
                            } else if emitted as f64 >= n {
                                Ok(false)
                            } else {
                                Ok(true)
                            }
                        });
                        match result {
                            Ok(_) if stopped_by_outer => Ok(false),
                            Ok(_) => Ok(true),
                            Err(e) => Err(e),
                        }
                    }
                    // jq's value ordering puts null/false/true below numbers,
                    // so `$n < 0` is true for these and they take jq's
                    // "negative count" branch (#539).
                    Value::Null | Value::True | Value::False => {
                        bail!("__jqerror__:\"limit doesn't support negative count\"");
                    }
                    // String / array / object surface jq's `$n - 1`
                    // arithmetic error from the limit reduce. jq computes that
                    // update lazily — only once the generator yields its first
                    // item — so an empty generator produces no error at all
                    // (#806). Defer the error until `generator` yields.
                    other => {
                        let msg = format!(
                            "{} and number (1) cannot be subtracted",
                            crate::runtime::errdesc_pub(other),
                        );
                        let err = format!(
                            "__jqerror__:{}",
                            crate::value::value_to_json_precise(&Value::from_string(msg)),
                        );
                        let mut yielded = false;
                        eval(generator, input.clone(), env, &mut |_val| {
                            yielded = true;
                            Ok(false)
                        })?;
                        if yielded {
                            bail!("{}", err);
                        }
                        Ok(true)
                    }
                }
            })
        }

        Expr::While { cond, update } => {
            let always_true = matches!(cond.as_ref(), Expr::Literal(Literal::True));
            eval_while_gen(cond, update, always_true, input, env, cb)
        }

        Expr::Until { cond, update } => {
            eval_until_gen(cond, update, input, env, cb)
        }

        Expr::Repeat { update } => {
            // repeat(f) = def _repeat: f, _repeat; _repeat;
            // Comma semantics: apply f to the SAME input each time, not chaining outputs.
            loop {
                if !eval(update, input.clone(), env, cb)? { return Ok(false); }
            }
        }

        Expr::AllShort { generator, predicate } => {
            let mut all_true = true;
            let let_bind = if let Expr::LetBinding { var_index, value, body } = predicate.as_ref() {
                if matches!(value.as_ref(), Expr::Input) && !expr_uses_outer_input(body) {
                    Some((*var_index, body.as_ref()))
                } else { None }
            } else { None };
            eval(generator, input.clone(), env, &mut |elem| {
                if let Some((vi, body)) = let_bind {
                    if let Value::Num(n, _) = &elem {
                        let e = env.borrow();
                        if e.closures.is_empty() {
                            if let Some(result) = eval_bool_numeric(body, &e.vars, vi, *n) {
                                drop(e);
                                return if result { Ok(true) } else { all_true = false; Ok(false) };
                            }
                        }
                        drop(e);
                    }
                    let old = std::mem::replace(&mut env.borrow_mut().vars[vi.idx()], elem);
                    // jq's `all` returns true vacuously when the predicate
                    // emits no values for an element, and false on the first
                    // falsy value (other values are short-circuited away).
                    let is_true = match eval_one(body, &Value::Null, env) {
                        Ok(v) => v.is_truthy(),
                        Err(()) => {
                            let mut found_falsy = false;
                            eval(body, Value::Null, env, &mut |v| {
                                if !v.is_truthy() { found_falsy = true; Ok(false) }
                                else { Ok(true) }
                            })?;
                            !found_falsy
                        }
                    };
                    env.borrow_mut().vars[vi.idx()] = old;
                    if is_true { Ok(true) } else { all_true = false; Ok(false) }
                } else {
                    let pred_result = eval_one(predicate, &elem, env);
                    match pred_result {
                        Ok(v) => {
                            if v.is_truthy() { Ok(true) } else { all_true = false; Ok(false) }
                        }
                        Err(()) => {
                            let mut found_falsy = false;
                            eval(predicate, elem, env, &mut |v| {
                                if !v.is_truthy() { found_falsy = true; Ok(false) }
                                else { Ok(true) }
                            })?;
                            if found_falsy { all_true = false; Ok(false) } else { Ok(true) }
                        }
                    }
                }
            })?;
            cb(Value::from_bool(all_true))
        }

        Expr::AnyShort { generator, predicate } => {
            let mut any_true = false;
            let let_bind = if let Expr::LetBinding { var_index, value, body } = predicate.as_ref() {
                if matches!(value.as_ref(), Expr::Input) && !expr_uses_outer_input(body) {
                    Some((*var_index, body.as_ref()))
                } else { None }
            } else { None };
            eval(generator, input.clone(), env, &mut |elem| {
                if let Some((vi, body)) = let_bind {
                    if let Value::Num(n, _) = &elem {
                        let e = env.borrow();
                        if e.closures.is_empty() {
                            if let Some(result) = eval_bool_numeric(body, &e.vars, vi, *n) {
                                drop(e);
                                return if result { any_true = true; Ok(false) } else { Ok(true) };
                            }
                        }
                        drop(e);
                    }
                    let old = std::mem::replace(&mut env.borrow_mut().vars[vi.idx()], elem);
                    // jq's `any` returns false vacuously when the predicate
                    // emits no values, and true on the first truthy value
                    // (other values are short-circuited away).
                    let is_true = match eval_one(body, &Value::Null, env) {
                        Ok(v) => v.is_truthy(),
                        Err(()) => {
                            let mut found_truthy = false;
                            eval(body, Value::Null, env, &mut |v| {
                                if v.is_truthy() { found_truthy = true; Ok(false) }
                                else { Ok(true) }
                            })?;
                            found_truthy
                        }
                    };
                    env.borrow_mut().vars[vi.idx()] = old;
                    if is_true { any_true = true; Ok(false) } else { Ok(true) }
                } else {
                    let pred_result = eval_one(predicate, &elem, env);
                    match pred_result {
                        Ok(v) => {
                            if v.is_truthy() { any_true = true; Ok(false) } else { Ok(true) }
                        }
                        Err(()) => {
                            let mut found_truthy = false;
                            eval(predicate, elem, env, &mut |v| {
                                if v.is_truthy() { found_truthy = true; Ok(false) }
                                else { Ok(true) }
                            })?;
                            if found_truthy { any_true = true; Ok(false) } else { Ok(true) }
                        }
                    }
                }
            })?;
            cb(Value::from_bool(any_true))
        }

        Expr::Error { msg } => {
            // Carry the payload as a typed `ErrorValue` so a downstream
            // `catch` recovers the exact value, including non-finite numbers
            // that the JSON channel would corrupt (#844).
            if let Some(msg_expr) = msg {
                eval(msg_expr, input, env, &mut |val| {
                    Err(ErrorValue::raise(val))
                })
            } else {
                Err(ErrorValue::raise(input))
            }
        }

        Expr::Format { kind, expr: fmt_expr } => {
            eval(fmt_expr, input, env, &mut |val| {
                cb(Value::from_str(&eval_format(kind, &val)?))
            })
        }

        Expr::ClosureOp { op, input_expr, key_expr } => {
            eval(input_expr, input.clone(), env, &mut |container| {
                eval_closure_op(*op, &container, key_expr, &input, env, cb)
            })
        }

        Expr::RegexTest { input_expr, re, flags } => {
            // jq nests the 2-arg `test(re; flags)` generator args rightmost-outer:
            // `flags` is the outer loop, `re` the inner one (#983).
            eval(input_expr, input.clone(), env, &mut |s| {
                eval(flags, input.clone(), env, &mut |fv| {
                    eval(re, input.clone(), env, &mut |re_val| {
                        cb(crate::runtime::call_builtin("test", &[s.clone(), re_val.clone(), fv.clone()])?)
                    })
                })
            })
        }

        Expr::RegexMatch { input_expr, re, flags } => {
            // Rightmost-outer nesting: `flags` outer, `re` inner (#983).
            eval(input_expr, input.clone(), env, &mut |s| {
                eval(flags, input.clone(), env, &mut |fv| {
                    eval(re, input.clone(), env, &mut |re_val| {
                        match crate::runtime::call_builtin("match", &[s.clone(), re_val.clone(), fv.clone()]) {
                            Ok(v) => {
                                // "g" flag: match returns array of all matches
                                if let Value::Arr(a) = &v {
                                    for item in a.iter() {
                                        if !cb(item.clone())? { return Ok(false); }
                                    }
                                    Ok(true)
                                } else {
                                    cb(v)
                                }
                            }
                            Err(e) => {
                                // No match → empty stream. Type errors and
                                // anything else must propagate so jq's
                                // `<type> cannot be matched, as it is not a
                                // string` error fires (#160).
                                let msg = e.to_string();
                                if msg.contains("match failed") {
                                    Ok(true)
                                } else {
                                    Err(e)
                                }
                            }
                        }
                    })
                })
            })
        }

        Expr::RegexCapture { input_expr, re, flags } => {
            // Rightmost-outer nesting: `flags` outer, `re` inner (#983).
            eval(input_expr, input.clone(), env, &mut |s| {
                eval(flags, input.clone(), env, &mut |fv| {
                    eval(re, input.clone(), env, &mut |re_val| {
                        let global = matches!(&fv, Value::Str(f) if f.as_str().contains('g'));
                        match crate::runtime::call_builtin("capture", &[s.clone(), re_val.clone(), fv.clone()]) {
                            Ok(v) => {
                                if global {
                                    if let Value::Arr(a) = &v {
                                        for item in a.iter() {
                                            if !cb(item.clone())? { return Ok(false); }
                                        }
                                        return Ok(true);
                                    }
                                }
                                cb(v)
                            }
                            Err(e) => {
                                let msg = e.to_string();
                                if msg.contains("capture failed") {
                                    Ok(true)
                                } else {
                                    Err(e)
                                }
                            }
                        }
                    })
                })
            })
        }

        Expr::RegexScan { input_expr, re, flags } => {
            eval(input_expr, input.clone(), env, &mut |s| {
                eval(re, input.clone(), env, &mut |re_val| {
                    eval(flags, input.clone(), env, &mut |fv| {
                        let result = crate::runtime::call_builtin("scan", &[s.clone(), re_val.clone(), fv.clone()])?;
                        if let Value::Arr(a) = &result {
                            for v in a.iter() { if !cb(v.clone())? { return Ok(false); } }
                            Ok(true)
                        } else { cb(result) }
                    })
                })
            })
        }

        Expr::RegexSub { input_expr, re, tostr, flags } |
        Expr::RegexGsub { input_expr, re, tostr, flags } => {
            let is_global = matches!(expr, Expr::RegexGsub { .. });
            eval(input_expr, input.clone(), env, &mut |s| {
                eval(re, input.clone(), env, &mut |rv| {
                    eval(flags, input.clone(), env, &mut |fv| {
                        let input_str = s.as_str().ok_or_else(|| anyhow::anyhow!(
                            "{} cannot be matched, as it is not a string",
                            crate::runtime::errdesc_pub(&s),
                        ))?;
                        let re_str = rv.as_str().ok_or_else(|| anyhow::anyhow!(
                            "{} is not a string",
                            crate::runtime::errdesc_pub(&rv),
                        ))?;
                        let segments = crate::runtime::sub_gsub_segments(input_str, re_str, &fv, is_global)?;
                        // jq treats the replacement as a generator and applies the
                        // i-th value of every match in lockstep — NOT a Cartesian
                        // product (#768). So the number of outputs is the largest
                        // number of values any single match yields; a match whose
                        // generator runs out at index i contributes nothing (drops
                        // the match) for that output. When *every* match yields
                        // nothing (or there are no matches at all), jq emits the
                        // original string unchanged.
                        let mut reps: Vec<Vec<Value>> = Vec::new();
                        for seg in &segments {
                            if let Some(ref cap_obj) = seg.captures {
                                let mut vals = Vec::new();
                                eval(tostr, cap_obj.clone(), env, &mut |tv| { vals.push(tv); Ok(true) })?;
                                reps.push(vals);
                            }
                        }
                        let n = reps.iter().map(|v| v.len()).max().unwrap_or(0);
                        if n == 0 {
                            // No matches, or every replacement generator was empty:
                            // the original string is preserved as a single output.
                            return cb(s.clone());
                        }
                        for i in 0..n {
                            let mut result = String::new();
                            let mut cap_idx = 0;
                            for seg in &segments {
                                result.push_str(&seg.literal);
                                if seg.captures.is_some() {
                                    let rep = reps[cap_idx].get(i);
                                    cap_idx += 1;
                                    // jq concatenates the replacement via `+`: a
                                    // string appends, null is the additive identity
                                    // (drops the match), other types surface jq's
                                    // standard addition error referencing the partial
                                    // result built so far (#545). A missing value at
                                    // this index likewise drops the match.
                                    match rep {
                                        Some(Value::Str(rs)) => result.push_str(rs),
                                        Some(Value::Null) | None => {},
                                        Some(other) => {
                                            let partial = Value::from_string(result);
                                            bail!(
                                                "{} and {} cannot be added",
                                                crate::runtime::errdesc_pub(&partial),
                                                crate::runtime::errdesc_pub(other),
                                            );
                                        }
                                    }
                                }
                            }
                            if !cb(Value::from_string(result))? {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    })
                })
            })
        }

        Expr::AlternativeDestructure { alternatives } => {
            for (i, alt) in alternatives.iter().enumerate() {
                match eval(alt, input.clone(), env, cb) {
                    Ok(cont) => return Ok(cont),
                    Err(_) if i < alternatives.len() - 1 => continue,
                    Err(e) => return Err(e),
                }
            }
            Ok(true)
        }

        Expr::Slice { expr: base_expr, from, to } => {
            // jq yields the Cartesian product of the bound generators, nested
            // from (outer) → to (middle) → base (inner). A missing bound is a
            // single Null (slice-from-start / slice-to-end); an *empty* bound
            // generator contributes no values, so the whole slice yields nothing. #761
            //
            // Fast path: both bounds resolve to a single value (the common
            // `.[a:b]` shape) — no product, no intermediate Vec.
            let from_one = match from { None => Some(Value::Null), Some(f) => eval_one(f, &input, env).ok() };
            let to_one = match to { None => Some(Value::Null), Some(t) => eval_one(t, &input, env).ok() };
            if let (Some(fv), Some(tv)) = (&from_one, &to_one) {
                return eval(base_expr, input.clone(), env, &mut |base| {
                    cb(eval_slice(&base, fv, tv)?)
                });
            }
            // Slow path: at least one bound is a generator (or empty).
            let from_vals: Vec<Value> = match (&from_one, from) {
                (Some(v), _) => vec![v.clone()],
                (None, Some(f)) => {
                    let mut vs = Vec::new();
                    eval(f, input.clone(), env, &mut |v| { vs.push(v); Ok(true) })?;
                    vs
                }
                (None, None) => vec![Value::Null],
            };
            if from_vals.is_empty() { return Ok(true); }
            let to_vals: Vec<Value> = match (&to_one, to) {
                (Some(v), _) => vec![v.clone()],
                (None, Some(t)) => {
                    let mut vs = Vec::new();
                    eval(t, input.clone(), env, &mut |v| { vs.push(v); Ok(true) })?;
                    vs
                }
                (None, None) => vec![Value::Null],
            };
            if to_vals.is_empty() { return Ok(true); }
            for fv in &from_vals {
                for tv in &to_vals {
                    if !eval(base_expr, input.clone(), env, &mut |base| {
                        cb(eval_slice(&base, fv, tv)?)
                    })? {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        }

        Expr::Loc { file, line } => {
            let mut obj = crate::value::new_objmap();
            obj.insert("file".into(), Value::from_str(file));
            obj.insert("line".into(), Value::number(*line as f64));
            cb(Value::object_from_map(obj))
        }

        Expr::Env => {
            thread_local! {
                static ENV_CACHE: RefCell<Option<Value>> = const { RefCell::new(None) };
            }
            let env_value = ENV_CACHE.with_borrow_mut(|cached| {
                if cached.is_none() {
                    let mut obj = crate::value::new_objmap();
                    for (k, v) in std::env::vars() { obj.insert(KeyStr::from(k), Value::from_str(&v)); }
                    *cached = Some(Value::object_from_map(obj));
                }
                cached.as_ref().unwrap().clone()
            });
            cb(env_value)
        }

        Expr::Builtins => cb(crate::runtime::rt_builtins()),

        Expr::ReadInput => {
            // `input` — read one value
            if let Some(v) = read_next_input() {
                cb(v)
            } else {
                bail!("break")
            }
        }
        Expr::ReadInputs => {
            // `inputs` — yield all remaining values
            while let Some(v) = read_next_input() {
                if !cb(v)? { return Ok(false); }
            }
            Ok(true)
        }

        Expr::Debug { expr: de } => {
            eval(de, input.clone(), env, &mut |val| {
                eprintln!("[\"DEBUG:\",{}]", crate::value::value_to_json_tojson(&val));
                cb(input.clone())
            })
        }

        Expr::Stderr { expr: se } => {
            eval(se, input.clone(), env, &mut |val| {
                // jq prints strings raw (no surrounding quotes) and other
                // values as compact JSON (#189). The filter passes the value
                // through to `cb` unchanged.
                match &val {
                    Value::Str(s) => eprint!("{}", s.as_str()),
                    _ => eprint!("{}", crate::value::value_to_json_tojson(&val)),
                }
                cb(input.clone())
            })
        }

        Expr::ModuleMeta => {
            let lib_dirs = env.borrow().lib_dirs.clone();
            let result = crate::module::get_modulemeta(&input, &lib_dirs)?;
            cb(result)
        }

        Expr::GenLabel => {
            let id = env.borrow().next_label;
            env.borrow_mut().next_label = id + 1;
            cb(Value::number(id as f64))
        }

        Expr::CallBuiltin { name, args } => {
            eval_call_builtin(name, args, input, env, cb)
        }

        Expr::Memoize { slot_id, key, body } => {
            let slot = {
                let env_ref = env.borrow();
                match env_ref.memo.as_ref().and_then(|m| m.slots.get(*slot_id as usize)) {
                    Some(s) => s.clone(),
                    None => bail!("memoize slot {} not allocated", slot_id),
                }
            };
            match key {
                None => {
                    // 1-arg form: cache key IS the current input.
                    memo_lookup_or_run(&slot, ValueKey(input.clone()), body, input, env, cb)
                }
                Some(key_expr) => {
                    // 2-arg form: each value yielded by `key` drives a separate
                    // cache lookup; the body runs against the original input.
                    eval(key_expr, input.clone(), env, &mut |k| {
                        memo_lookup_or_run(&slot, ValueKey(k), body, input.clone(), env, cb)
                    })
                }
            }
        }
    }
}

/// Shared cache-lookup-or-compute path for both arities of `memoize`.
///
/// On hit, re-yields the cached output sequence. On miss, runs `body`
/// against `body_input` to completion (collecting every output), inserts
/// the result subject to the per-slot cap, and yields it.
///
/// Body errors are propagated without poisoning the cache — the next call
/// re-evaluates.
fn memo_lookup_or_run(
    slot: &MemoSlot,
    cache_key: ValueKey,
    body: &Expr,
    body_input: Value,
    env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    let cached = {
        let mut state = slot.borrow_mut();
        let hit = state.entries.get(&cache_key).cloned();
        if hit.is_some() { state.hits += 1; } else { state.misses += 1; }
        hit
    };
    if let Some(entry) = cached {
        return match entry {
            MemoEntry::Single(v) => cb(v),
            MemoEntry::Many(rc) => {
                for v in rc.iter() {
                    if !cb(v.clone())? { return Ok(false); }
                }
                Ok(true)
            }
        };
    }
    // Miss: run the body to completion. We do not pass the consumer's `cb`
    // directly — `memoize` must observe the full output sequence to cache
    // it, even if the consumer wants to stop after the first value. This
    // means `memoize` forces the body to fully evaluate; for the typical
    // single-output pure body that is the natural case, but it is a
    // documented difference for generator/side-effecting bodies.
    let mut outputs: Vec<Value> = Vec::new();
    eval(body, body_input, env, &mut |v| { outputs.push(v); Ok(true) })?;
    let entry = if outputs.len() == 1 {
        MemoEntry::Single(outputs[0].clone())
    } else {
        MemoEntry::Many(Rc::new(outputs))
    };
    {
        let mut state = slot.borrow_mut();
        let cap = env.borrow().memo.as_ref().map_or(DEFAULT_MEMO_MAX_ENTRIES, |m| m.max_entries);
        if state.entries.len() < cap {
            state.entries.insert(cache_key, entry.clone());
        }
    }
    match entry {
        MemoEntry::Single(v) => cb(v),
        MemoEntry::Many(rc) => {
            for v in rc.iter() {
                if !cb(v.clone())? { return Ok(false); }
            }
            Ok(true)
        }
    }
}

// jq treats NaN as less than every number (including itself, reflexively
// via `<`) so the ordering operators stay total over numeric inputs and
// `sort` is stable in the presence of NaN. IEEE 754's "all comparisons
// false" leaves NaNs scattered, so each numeric fast path routes through
// these helpers instead of `<`/`>` directly. (`==` / `!=` keep IEEE 754
// inequality semantics — `nan == nan` is still false.)
#[inline]
pub fn jq_num_lt(ln: f64, rn: f64) -> bool {
    if ln.is_nan() { true }
    else if rn.is_nan() { false }
    else { ln < rn }
}
#[inline]
pub fn jq_num_gt(ln: f64, rn: f64) -> bool {
    if rn.is_nan() && !ln.is_nan() { true }
    else if ln.is_nan() { false }
    else { ln > rn }
}
#[inline]
pub fn jq_num_le(ln: f64, rn: f64) -> bool { !jq_num_gt(ln, rn) }
#[inline]
pub fn jq_num_ge(ln: f64, rn: f64) -> bool { !jq_num_lt(ln, rn) }

// ---------------------------------------------------------------------------
#[inline]
pub fn eval_binop(op: BinOp, lhs: &Value, rhs: &Value) -> Result<Value> {
    // Numeric fast path: avoid runtime function dispatch for common numeric ops
    if let (Value::Num(ln, _), Value::Num(rn, _)) = (lhs, rhs) {
        return Ok(match op {
            BinOp::Add => Value::number(ln + rn),
            BinOp::Sub => Value::number(ln - rn),
            BinOp::Mul => Value::number(ln * rn),
            BinOp::Div => {
                if *rn == 0.0 { return crate::runtime::rt_div(lhs, rhs); }
                Value::number(ln / rn)
            }
            BinOp::Mod => {
                if !ln.is_finite() || !rn.is_finite() { return crate::runtime::rt_mod(lhs, rhs); }
                let yi = *rn as i64;
                if yi == 0 { return crate::runtime::rt_mod(lhs, rhs); }
                Value::number(crate::runtime::jq_mod_i64(*ln as i64, yi) as f64)
            }
            BinOp::Eq => if ln == rn { Value::True } else { Value::False },
            BinOp::Ne => if ln != rn { Value::True } else { Value::False },
            BinOp::Lt => if jq_num_lt(*ln, *rn) { Value::True } else { Value::False },
            BinOp::Gt => if jq_num_gt(*ln, *rn) { Value::True } else { Value::False },
            BinOp::Le => if jq_num_le(*ln, *rn) { Value::True } else { Value::False },
            BinOp::Ge => if jq_num_ge(*ln, *rn) { Value::True } else { Value::False },
            BinOp::And => if lhs.is_truthy() && rhs.is_truthy() { Value::True } else { Value::False },
            BinOp::Or => if lhs.is_truthy() || rhs.is_truthy() { Value::True } else { Value::False },
        });
    }
    match op {
        BinOp::Add => crate::runtime::rt_add(lhs, rhs),
        BinOp::Sub => crate::runtime::rt_sub(lhs, rhs),
        BinOp::Mul => crate::runtime::rt_mul(lhs, rhs),
        BinOp::Div => crate::runtime::rt_div(lhs, rhs),
        BinOp::Mod => crate::runtime::rt_mod(lhs, rhs),
        BinOp::Eq => Ok(if crate::runtime::values_equal(lhs, rhs) { Value::True } else { Value::False }),
        BinOp::Ne => Ok(if crate::runtime::values_equal(lhs, rhs) { Value::False } else { Value::True }),
        BinOp::Lt => Ok(if crate::runtime::compare_values(lhs, rhs) == std::cmp::Ordering::Less { Value::True } else { Value::False }),
        BinOp::Gt => Ok(if crate::runtime::compare_values(lhs, rhs) == std::cmp::Ordering::Greater { Value::True } else { Value::False }),
        BinOp::Le => Ok(if crate::runtime::compare_values(lhs, rhs) != std::cmp::Ordering::Greater { Value::True } else { Value::False }),
        BinOp::Ge => Ok(if crate::runtime::compare_values(lhs, rhs) != std::cmp::Ordering::Less { Value::True } else { Value::False }),
        BinOp::And => Ok(if lhs.is_truthy() && rhs.is_truthy() { Value::True } else { Value::False }),
        BinOp::Or => Ok(if lhs.is_truthy() || rhs.is_truthy() { Value::True } else { Value::False }),
    }
}

/// Like eval_binop but takes ownership of lhs for in-place mutation (array/object append).
#[inline]
fn eval_binop_owned(op: BinOp, lhs: Value, rhs: &Value) -> Result<Value> {
    // Numeric fast path: avoid function dispatch overhead
    if let (Value::Num(ln, _), Value::Num(rn, _)) = (&lhs, rhs) {
        return Ok(match op {
            BinOp::Add => Value::number(ln + rn),
            BinOp::Sub => Value::number(ln - rn),
            BinOp::Mul => Value::number(ln * rn),
            BinOp::Div => {
                if *rn == 0.0 { return crate::runtime::rt_div(&lhs, rhs); }
                Value::number(ln / rn)
            }
            BinOp::Mod => {
                if !ln.is_finite() || !rn.is_finite() { return crate::runtime::rt_mod(&lhs, rhs); }
                let yi = *rn as i64;
                if yi == 0 { return crate::runtime::rt_mod(&lhs, rhs); }
                Value::number(crate::runtime::jq_mod_i64(*ln as i64, yi) as f64)
            }
            BinOp::Eq => if ln == rn { Value::True } else { Value::False },
            BinOp::Ne => if ln != rn { Value::True } else { Value::False },
            BinOp::Lt => if jq_num_lt(*ln, *rn) { Value::True } else { Value::False },
            BinOp::Gt => if jq_num_gt(*ln, *rn) { Value::True } else { Value::False },
            BinOp::Le => if jq_num_le(*ln, *rn) { Value::True } else { Value::False },
            BinOp::Ge => if jq_num_ge(*ln, *rn) { Value::True } else { Value::False },
            BinOp::And | BinOp::Or => return eval_binop(op, &lhs, rhs),
        });
    }
    match op {
        BinOp::Add => crate::runtime::rt_add_owned(lhs, rhs),
        _ => eval_binop(op, &lhs, rhs),
    }
}

pub fn eval_unaryop(op: UnaryOp, val: &Value) -> Result<Value> {
    match op {
        UnaryOp::Not => return Ok(if val.is_truthy() { Value::False } else { Value::True }),
        UnaryOp::Infinite => return Ok(Value::number(f64::INFINITY)),
        UnaryOp::Nan => return Ok(Value::number(f64::NAN)),
        _ => {}
    }
    let name = match op {
        UnaryOp::Length => "length", UnaryOp::Type | UnaryOp::TypeOf => "type",
        UnaryOp::IsInfinite => "isinfinite", UnaryOp::IsNan => "isnan",
        UnaryOp::IsNormal => "isnormal", UnaryOp::IsFinite => "isfinite",
        UnaryOp::ToString => "tostring", UnaryOp::ToNumber => "tonumber",
        UnaryOp::ToJson => "tojson", UnaryOp::FromJson => "fromjson",
        UnaryOp::Ascii => "ascii", UnaryOp::Explode => "explode", UnaryOp::Implode => "implode",
        UnaryOp::AsciiDowncase => "ascii_downcase", UnaryOp::AsciiUpcase => "ascii_upcase",
        UnaryOp::Trim => "trim", UnaryOp::Ltrim => "ltrim", UnaryOp::Rtrim => "rtrim",
        UnaryOp::Utf8ByteLength => "utf8bytelength",
        UnaryOp::Floor => "floor", UnaryOp::Ceil => "ceil", UnaryOp::Round => "round",
        UnaryOp::Fabs => "fabs", UnaryOp::Sqrt => "sqrt",
        UnaryOp::Sin => "sin", UnaryOp::Cos => "cos", UnaryOp::Tan => "tan",
        UnaryOp::Asin => "asin", UnaryOp::Acos => "acos", UnaryOp::Atan => "atan",
        UnaryOp::Sinh => "sinh", UnaryOp::Cosh => "cosh", UnaryOp::Tanh => "tanh",
        UnaryOp::Asinh => "asinh", UnaryOp::Acosh => "acosh", UnaryOp::Atanh => "atanh",
        UnaryOp::Exp => "exp", UnaryOp::Exp2 => "exp2", UnaryOp::Exp10 => "exp10",
        UnaryOp::Log => "log", UnaryOp::Log2 => "log2", UnaryOp::Log10 => "log10",
        UnaryOp::Cbrt => "cbrt", UnaryOp::Significand => "significand",
        UnaryOp::Logb => "logb",
        UnaryOp::NearbyInt => "nearbyint", UnaryOp::Trunc => "trunc",
        UnaryOp::Rint => "rint", UnaryOp::J0 => "j0", UnaryOp::J1 => "j1",
        UnaryOp::Keys => "keys", UnaryOp::KeysUnsorted => "keys_unsorted",
        UnaryOp::Values => "values", UnaryOp::Sort => "sort", UnaryOp::Reverse => "reverse",
        UnaryOp::Unique => "unique", UnaryOp::Flatten => "flatten",
        UnaryOp::Min => "min", UnaryOp::Max => "max", UnaryOp::Add => "add",
        UnaryOp::Any => "any", UnaryOp::All => "all", UnaryOp::Transpose => "transpose",
        UnaryOp::ToEntries => "to_entries", UnaryOp::FromEntries => "from_entries",
        UnaryOp::Gmtime => "gmtime", UnaryOp::Localtime => "localtime", UnaryOp::Mktime => "mktime", UnaryOp::Now => "now",
        UnaryOp::Abs => "abs", UnaryOp::GetModuleMeta => "modulemeta",
        _ => unreachable!(),
    };
    crate::runtime::call_builtin(name, std::slice::from_ref(val))
}

pub fn eval_index(base: &Value, key: &Value, optional: bool) -> std::result::Result<Value, String> {
    match (base, key) {
        (Value::Obj(ObjInner(o)), Value::Str(k)) => Ok(o.get(k.as_str()).cloned().unwrap_or(Value::Null)),
        (Value::Arr(a), Value::Num(n, _)) => {
            Ok(crate::value::resolve_array_index(*n, a.len())
                .map(|i| a[i].clone())
                .unwrap_or(Value::Null))
        }
        (Value::Str(_), Value::Num(_, _)) => {
            // jq's "Cannot index string with number" omits the value (#440).
            if optional {
                Err("type error".into())
            } else {
                Err("Cannot index string with number".to_string())
            }
        }
        // jq aliases `.[arr]` on an array to `indices(arr)` — returns the
        // offsets where the subsequence appears. String receivers still
        // error (`Cannot index string with array`). See #467.
        (Value::Arr(_), Value::Arr(_)) => {
            crate::runtime::call_builtin("indices", &[base.clone(), key.clone()])
                .map_err(|e| e.to_string())
        }
        // jq dispatches `.[obj]` on an array or string as a slice when the
        // object has both `start` and `end` keys (each being a number or
        // null). Otherwise it errors with the slice-indices wording — even
        // when the base is null. See #463.
        (Value::Arr(_), Value::Obj(ObjInner(spec)))
        | (Value::Str(_), Value::Obj(ObjInner(spec))) => {
            let start = spec.get("start");
            let end = spec.get("end");
            match (start, end) {
                (Some(s), Some(e)) => {
                    let valid = matches!(s, Value::Num(_, _) | Value::Null)
                        && matches!(e, Value::Num(_, _) | Value::Null);
                    if !valid {
                        if optional { return Err("type error".into()); }
                        return Err("Array/string slice indices must be integers".to_string());
                    }
                    eval_slice(base, s, e).map_err(|e| e.to_string())
                }
                _ => {
                    if optional { Err("type error".into()) }
                    else { Err("Array/string slice indices must be integers".to_string()) }
                }
            }
        }
        // Null receiver: only string/number/object keys short-circuit to null;
        // null/bool/array keys still raise the same type error jq emits on a
        // non-null base (#193). The keys here mirror what jq's `.[$k]` accepts
        // before the null short-circuit kicks in.
        (Value::Null, Value::Str(_)) | (Value::Null, Value::Num(_, _)) | (Value::Null, Value::Obj(_)) => {
            Ok(Value::Null)
        }
        _ => {
            if optional { Err("type error".into()) }
            else {
                // jq's "Cannot index X with Y" wording: string keys are
                // quoted without parens (`with string "k"`), number keys
                // omit the value entirely (`with number`). See #440 and
                // `runtime::index_err_desc`.
                let key_desc = match key {
                    Value::Str(s) => format!("string \"{}\"", s),
                    Value::Num(_, _) => "number".to_string(),
                    _ => key.type_name().to_string(),
                };
                Err(format!("Cannot index {} with {}", base.type_name(), key_desc))
            }
        }
    }
}

fn eval_recurse_expr(step: &Expr, val: &Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    // The 0-arg `recurse` and `..` are desugared by the parser to
    // `recurse(.[]?)` — i.e. `Expr::EachOpt { Expr::Input }`. Take the
    // descent fast path only for that exact shape; user-written
    // `recurse(.)` carries `Expr::Input` and should fall through to the
    // generic loop (which never terminates, matching jq). See #497.
    if matches!(step, Expr::EachOpt { input_expr } if matches!(**input_expr, Expr::Input)) {
        eval_recurse_default(val, cb)
    } else {
        // Custom step `recurse(f)` = `def r: ., (f | r); r`. The recursion is
        // lazy and depth-first: emit the current value, then for EACH value
        // `f` produces (in order) immediately recurse into it before pulling
        // the next one. Buffering all of `f`'s outputs first (the old
        // explicit-stack approach) evaluated `f` breadth-first, so a
        // `break $label` / error raised by a *later* generator alternative
        // inside `f` fired before an *earlier* alternative's subtree ran —
        // truncating output (#916). Errors raised by `step` still propagate
        // (#195). `stacker::maybe_grow` guards deep recursion.
        if !cb(val.clone())? { return Ok(false); }
        eval(step, val.clone(), env, &mut |next| {
            stacker::maybe_grow(64 * 1024, 1024 * 1024, || eval_recurse_expr(step, &next, env, cb))
        })
    }
}

fn eval_recurse_default(val: &Value, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    if !cb(val.clone())? { return Ok(false); }
    match val {
        Value::Arr(a) => { for item in a.iter() { if !stacker::maybe_grow(64 * 1024, 1024 * 1024, || eval_recurse_default(item, cb))? { return Ok(false); } } }
        Value::Obj(ObjInner(o)) => { for v in o.values() { if !stacker::maybe_grow(64 * 1024, 1024 * 1024, || eval_recurse_default(v, cb))? { return Ok(false); } } }
        _ => {}
    }
    Ok(true)
}

fn eval_range(from: &Value, to: &Value, step: Option<&Value>, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    // jq emits a single "Range bounds must be numeric" error for both
    // out-of-type from and to (#527).
    let f = match from { Value::Num(n, _) => *n, _ => bail!("Range bounds must be numeric") };
    let t = match to { Value::Num(n, _) => *n, _ => bail!("Range bounds must be numeric") };
    // jq's `range/3` desugars to `from | while(. < upto; . + by)`. The
    // step is consumed lazily, so `range(0; 10; "a")` emits `0` first and
    // then surfaces jq's `+`-error on the next iteration. `null`/`true`/
    // `false` steps silently emit nothing (jq's range short-circuits
    // before yielding anything when the addition would be an identity or
    // a type error it would have already filtered). See #582.
    let s = match step {
        Some(Value::Num(n, _)) => *n,
        Some(Value::Null) | Some(Value::True) | Some(Value::False) => return Ok(true),
        Some(non_num) => {
            if !cb(from.clone())? { return Ok(false); }
            let _ = crate::runtime::rt_add(from, non_num)?;
            return Ok(true);
        }
        None => 1.0,
    };
    if s == 0.0 { return Ok(true); }
    let mut c = f;
    let mut first = true;
    if s > 0.0 {
        while c < t {
            let v = if first { from.clone() } else { Value::number(c) };
            first = false;
            if !cb(v)? { return Ok(false); }
            c += s;
        }
    } else {
        while c > t {
            let v = if first { from.clone() } else { Value::number(c) };
            first = false;
            if !cb(v)? { return Ok(false); }
            c += s;
        }
    }
    Ok(true)
}

fn object_key_from_value(kv: &Value) -> Result<KeyStr> {
    match kv {
        Value::Str(s) => Ok(KeyStr::from(s.as_str())),
        _ => bail!(
            "Cannot use {} as object key",
            crate::runtime::errdesc_pub(kv)
        ),
    }
}

fn eval_object_construct(pairs: &[(Expr, Expr)], input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    // Fast path: if all keys and values are scalar expressions, build directly without cloning
    let mut obj = crate::value::new_objmap_with_capacity(pairs.len());
    for (ke, ve) in pairs {
        let kv = match eval_one(ke, &input, env) {
            Ok(v) => v,
            Err(()) => return eval_obj_pairs(pairs, 0, crate::value::new_objmap_with_capacity(pairs.len()), input, env, cb),
        };
        // Defer validation: if the value generator turns out to be empty, jq
        // short-circuits without complaining about the key (#201). Pull the
        // value first; only then check the key type.
        let vv = match eval_one(ve, &input, env) {
            Ok(v) => v,
            Err(()) => return eval_obj_pairs(pairs, 0, crate::value::new_objmap_with_capacity(pairs.len()), input, env, cb),
        };
        let ks = object_key_from_value(&kv)?;
        obj.insert(ks, vv);
    }
    cb(Value::object_from_map(obj))
}

fn eval_obj_pairs(pairs: &[(Expr, Expr)], idx: usize, cur: crate::value::ObjMap, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    if idx >= pairs.len() { return cb(Value::object_from_map(cur)); }
    let (ke, ve) = &pairs[idx];
    eval(ke, input.clone(), env, &mut |kv| {
        // Defer the key-type check until V yields at least one value: jq lets
        // `{(non_string_key): empty}` short-circuit silently because no
        // (key, value) pair actually materializes (#201).
        eval(ve, input.clone(), env, &mut |vv| {
            let ks = object_key_from_value(&kv)?;
            let mut next = cur.clone();
            next.insert(ks, vv);
            eval_obj_pairs(pairs, idx + 1, next, input.clone(), env, cb)
        })
    })
}

fn format_sh_scalar(val: &Value) -> Result<String> {
    match val {
        // jq escapes an embedded NUL to `\0` inside the quoted shell word (#849).
        Value::Str(s) => Ok(format!("'{}'", s.replace('\'', "'\\''").replace('\0', "\\0"))),
        Value::Null => Ok("null".to_string()),
        Value::True => Ok("true".to_string()),
        Value::False => Ok("false".to_string()),
        // `value_to_json_tojson` keeps the carried number repr (`0.0` stays
        // `"0.0"`) while gracefully falling back to f64-formatted form when
        // the literal can't round-trip — same logic the `tostring` builtin
        // uses. The bare `format_jq_number` arm dropped the repr so
        // `0.0 | @sh` produced `"0"` instead of jq's `"0.0"`. See #564.
        Value::Num(_, _) => Ok(crate::value::value_to_json_tojson(val)),
        _ => bail!(
            "{} ({}) can not be escaped for shell",
            val.type_name(),
            crate::value::value_to_json(val),
        ),
    }
}

fn format_sh(val: &Value) -> Result<String> {
    match val {
        Value::Arr(a) => {
            let mut parts: Vec<String> = Vec::with_capacity(a.len());
            for v in a.iter() {
                parts.push(format_sh_scalar(v)?);
            }
            Ok(parts.join(" "))
        }
        _ => format_sh_scalar(val),
    }
}

pub fn eval_format(kind: &FormatKind, val: &Value) -> Result<String> {
    // For csv/tsv, the input must be an array
    match kind {
        FormatKind::Csv => {
            let arr = match val { Value::Arr(a) => a, _ => bail!("{} cannot be csv-formatted, only array", crate::runtime::errdesc_pub(val)) };
            let mut buf = String::with_capacity(arr.len() * 16);
            for (i, v) in arr.iter().enumerate() {
                if i > 0 { buf.push(','); }
                match v {
                    Value::Str(s) => {
                        buf.push('"');
                        // jq doubles `"` and escapes an embedded NUL to the
                        // two-char `\0` (the #849 NUL fix missed @csv/@tsv).
                        // Single pass over the cell bytes. #929
                        if s.as_bytes().iter().any(|&b| b == b'"' || b == 0) {
                            for c in s.chars() {
                                match c {
                                    '"' => { buf.push('"'); buf.push('"'); }
                                    '\0' => buf.push_str("\\0"),
                                    _ => buf.push(c),
                                }
                            }
                        } else {
                            buf.push_str(s);
                        }
                        buf.push('"');
                    }
                    Value::Null => {}
                    Value::True => buf.push_str("true"),
                    Value::False => buf.push_str("false"),
                    Value::Num(n, crate::value::NumRepr(repr)) => {
                        crate::value::push_value_num_repr_str(&mut buf, *n, repr.as_ref());
                    }
                    // jq rejects arrays/objects as row elements (issue #79).
                    Value::Arr(_) | Value::Obj(_) => bail!(
                        "{} ({}) is not valid in a csv row",
                        v.type_name(),
                        crate::value::value_to_json(v),
                    ),
                    _ => buf.push_str(&crate::value::value_to_json(v)),
                }
            }
            return Ok(buf);
        }
        FormatKind::Tsv => {
            let arr = match val { Value::Arr(a) => a, _ => bail!("{} cannot be tsv-formatted, only array", crate::runtime::errdesc_pub(val)) };
            let mut buf = String::with_capacity(arr.len() * 16);
            for (i, v) in arr.iter().enumerate() {
                if i > 0 { buf.push('\t'); }
                match v {
                    Value::Str(s) => {
                        for c in s.chars() {
                            match c {
                                '\\' => buf.push_str("\\\\"),
                                '\t' => buf.push_str("\\t"),
                                '\n' => buf.push_str("\\n"),
                                '\r' => buf.push_str("\\r"),
                                // jq escapes an embedded NUL to `\0` (#849/#929).
                                '\0' => buf.push_str("\\0"),
                                _ => buf.push(c),
                            }
                        }
                    }
                    Value::Null => {}
                    Value::True => buf.push_str("true"),
                    Value::False => buf.push_str("false"),
                    Value::Num(n, crate::value::NumRepr(repr)) => crate::value::push_value_num_repr_str(&mut buf, *n, repr.as_ref()),
                    // jq uses the same "csv row" wording for @tsv (issue #79).
                    Value::Arr(_) | Value::Obj(_) => bail!(
                        "{} ({}) is not valid in a csv row",
                        v.type_name(),
                        crate::value::value_to_json(v),
                    ),
                    _ => buf.push_str(&crate::value::value_to_json(v)),
                }
            }
            return Ok(buf);
        }
        _ => {}
    }

    // For other formats, stringify the value first.
    //
    // jq's `@text`, `@uri`, `@html`, `@sh`, `@base64`, `@base64d` all run
    // their input through the equivalent of `tostring` first. `value_to_json`
    // discarded the carried number repr, so `0.0 | @text` produced `"0"`
    // instead of `"0.0"` (and `@base64` encoded `MA==` instead of jq's
    // `MC4w`). `value_to_json_tojson` keeps the literal form when f64 can
    // round-trip it, matching `rt_tostring`. See #564.
    let s = match val { Value::Str(s) => s.to_string(), _ => crate::value::value_to_json_tojson(val) };
    match kind {
        // Returned from the array-format match above.
        FormatKind::Csv | FormatKind::Tsv => unreachable!(),
        FormatKind::Text => Ok(s),
        // `@json` mirrors the `tojson` builtin, so it must preserve the
        // carried number repr exactly the way `rt_tojson` does — otherwise
        // `0.0 | @json | length` returned 1 (the value was the string
        // `"0"`, even though the raw-byte CLI fast path printed `"0.0"` for
        // the standalone `@json` form). See #562.
        FormatKind::Json => Ok(crate::value::value_to_json_tojson(val)),
        FormatKind::Html => {
            let mut r = String::with_capacity(s.len());
            for c in s.chars() {
                match c {
                    '&' => r.push_str("&amp;"),
                    '<' => r.push_str("&lt;"),
                    '>' => r.push_str("&gt;"),
                    '\'' => r.push_str("&apos;"),
                    '"' => r.push_str("&quot;"),
                    // jq escapes an embedded NUL to the two-character sequence
                    // `\0` in @html (and @sh); other control bytes pass through
                    // unchanged. See #849.
                    '\0' => r.push_str("\\0"),
                    _ => r.push(c),
                }
            }
            Ok(r)
        }
        FormatKind::Uri => {
            const HEX: &[u8; 16] = b"0123456789ABCDEF";
            let mut r = String::with_capacity(s.len());
            for b in s.bytes() {
                match b {
                    b'A'..=b'Z'|b'a'..=b'z'|b'0'..=b'9'|b'-'|b'_'|b'.'|b'~' => r.push(b as char),
                    _ => { r.push('%'); r.push(HEX[(b >> 4) as usize] as char); r.push(HEX[(b & 0xf) as usize] as char); }
                }
            }
            Ok(r)
        }
        FormatKind::Urid => {
            // jq validates `@urid` input (#961): a `%` must be followed by two
            // hex digits, and a maximal run of consecutive `%XX` escapes must
            // decode to valid UTF-8 — a malformed escape (`%`, `%4`, `%GG`) or
            // an ill-formed escaped byte sequence (`%80`, `%C3`, `%C0%80`) is a
            // hard `"string (X) is not a valid uri encoding"` error. A *raw*
            // (non-escaped) input byte ≥ 0x80 is instead replaced per-byte with
            // U+FFFD (so `"é"` → two U+FFFD), and raw ASCII passes through.
            let bytes = s.as_bytes();
            let uri_err = || anyhow::anyhow!(
                "string ({}) is not a valid uri encoding",
                crate::value::value_to_json(&Value::from_str(&s))
            );
            // Validate + flush the pending escaped-byte run as UTF-8.
            let flush = |esc: &mut Vec<u8>, out: &mut String| -> Result<()> {
                if !esc.is_empty() {
                    match std::str::from_utf8(esc) {
                        Ok(decoded) => out.push_str(decoded),
                        Err(_) => return Err(uri_err()),
                    }
                    esc.clear();
                }
                Ok(())
            };
            let mut out = String::with_capacity(s.len());
            let mut esc: Vec<u8> = Vec::new();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'%' {
                    match (bytes.get(i + 1).copied().and_then(hex_val),
                           bytes.get(i + 2).copied().and_then(hex_val)) {
                        (Some(h), Some(l)) => { esc.push(h * 16 + l); i += 3; }
                        _ => return Err(uri_err()),
                    }
                } else {
                    flush(&mut esc, &mut out)?;
                    let b = bytes[i];
                    if b < 0x80 { out.push(b as char); } else { out.push('\u{FFFD}'); }
                    i += 1;
                }
            }
            flush(&mut esc, &mut out)?;
            Ok(out)
        }
        FormatKind::Sh => format_sh(val),
        FormatKind::Base64 => {
            const C: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let d = s.as_bytes(); let mut r = String::new();
            for ch in d.chunks(3) { let (b0,b1,b2) = (ch[0] as u32, ch.get(1).copied().unwrap_or(0) as u32, ch.get(2).copied().unwrap_or(0) as u32); let n = (b0<<16)|(b1<<8)|b2;
                r.push(C[((n>>18)&0x3f) as usize] as char); r.push(C[((n>>12)&0x3f) as usize] as char);
                r.push(if ch.len()>1 { C[((n>>6)&0x3f) as usize] as char } else { '=' }); r.push(if ch.len()>2 { C[(n&0x3f) as usize] as char } else { '=' }); }
            Ok(r)
        }
        FormatKind::Base64d => {
            const D: [i8;128] = { let mut t = [-1i8;128]; let c = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; let mut i=0; while i<c.len() { t[c[i] as usize]=i as i8; i+=1; } t };
            // jq's `@base64d` validates the data in this order (#557, #605):
            //
            // 1. Truncate at the *first* `=` (anywhere). So `"=A=="` decodes
            //    as `""`, and `"a=b"` truncates to `"a"` before length check.
            // 2. Reject any non-`A-Za-z0-9+/` byte — `"is not valid base64
            //    data"` fires before any length check, so `"_"` and `" "`
            //    take this path instead of the trailing-byte path.
            // 3. Then check length mod 4 != 1 → "trailing base64 byte found".
            //
            // Whitespace is *not* stripped — `" a"` and `"a "` both fail
            // step 2 with "is not valid base64 data" (#557).
            let raw = s.as_bytes();
            let bs: &[u8] = match raw.iter().position(|&b| b == b'=') {
                Some(i) => &raw[..i],
                None => raw,
            };
            // jq's error message wraps the value in `string (X)`, and X is
            // truncated exactly like every other type-error site: long previews
            // are cut to the first 11 bytes (UTF-8-boundary safe) + `...`.
            // Route through the shared errdesc dumper so this site matches the
            // others (jq truncates here too; #981).
            let err_desc = || crate::runtime::errdesc_pub(&Value::from_str(&s));
            for &b in bs {
                let v = D.get(b as usize).copied().unwrap_or(-1);
                if v < 0 {
                    bail!("{} is not valid base64 data", err_desc());
                }
            }
            if bs.len() % 4 == 1 {
                bail!("{} trailing base64 byte found", err_desc());
            }
            let mut r = Vec::new();
            for ch in bs.chunks(4) {
                let a = D[ch[0] as usize] as u8;
                let b = D[ch[1] as usize] as u8;
                r.push((a << 2) | (b >> 4));
                if ch.len() > 2 {
                    let c = D[ch[2] as usize] as u8;
                    r.push((b << 4) | (c >> 2));
                    if ch.len() > 3 {
                        let d = D[ch[3] as usize] as u8;
                        r.push((c << 6) | d);
                    }
                }
            }
            Ok(jq_utf8_lossy(&r))
        }
        FormatKind::Invalid(name) => bail!("{} is not a valid format", name),
    }
}

/// UTF-8 lossy decode that matches jq 1.8.1's `@base64d` substitution policy
/// (Unicode 6.1+ "maximal subpart of an ill-formed subsequence", #607).
///
/// Differs from `String::from_utf8_lossy`, which emits one U+FFFD per invalid
/// byte even for partial / overlong multi-byte sequences. jq emits one U+FFFD
/// per ill-formed *sequence*: a leader plus any valid continuations that
/// follow (up to the expected count), or all remaining bytes if the buffer
/// ends short.
fn jq_utf8_lossy(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x80 {
            out.push(b as char);
            i += 1;
            continue;
        }
        // Leader byte width (excluding C0/C1 which are always overlong, and
        // F5..FF which exceed U+10FFFF).
        let needed = if (0xC2..0xE0).contains(&b) { 2 }
                     else if (0xE0..0xF0).contains(&b) { 3 }
                     else if (0xF0..0xF5).contains(&b) { 4 }
                     else { 0 };
        if needed == 0 {
            out.push('\u{FFFD}');
            i += 1;
            continue;
        }
        // Not enough bytes: consume all remaining as one U+FFFD.
        if bytes.len() - i < needed {
            out.push('\u{FFFD}');
            i = bytes.len();
            continue;
        }
        // Validate continuations.
        let c1 = bytes[i + 1];
        let c1_cont = c1 & 0xC0 == 0x80;
        if !c1_cont {
            out.push('\u{FFFD}');
            i += 1;
            continue;
        }
        // Special leader-byte ranges that further restrict the first
        // continuation (avoid overlongs and surrogates). When violated, jq
        // still consumes the full nominal sequence as one U+FFFD.
        let bad_first = match b {
            0xE0 => c1 < 0xA0,
            0xED => c1 >= 0xA0,
            0xF0 => c1 < 0x90,
            0xF4 => c1 >= 0x90,
            _ => false,
        };
        if needed >= 3 {
            let c2 = bytes[i + 2];
            if c2 & 0xC0 != 0x80 {
                // Maximal subpart: leader + one valid continuation.
                out.push('\u{FFFD}');
                i += 2;
                continue;
            }
            if needed == 4 {
                let c3 = bytes[i + 3];
                if c3 & 0xC0 != 0x80 {
                    out.push('\u{FFFD}');
                    i += 3;
                    continue;
                }
            }
        }
        if bad_first {
            // All `needed` bytes look like a complete sequence at the byte
            // level, but the codepoint is overlong/surrogate. jq consumes
            // the whole sequence as one U+FFFD.
            out.push('\u{FFFD}');
            i += needed;
            continue;
        }
        // All continuations are valid — decode the codepoint.
        let cp: u32 = match needed {
            2 => ((b as u32 & 0x1F) << 6) | (c1 as u32 & 0x3F),
            3 => ((b as u32 & 0x0F) << 12)
                | ((c1 as u32 & 0x3F) << 6)
                | (bytes[i + 2] as u32 & 0x3F),
            4 => ((b as u32 & 0x07) << 18)
                | ((c1 as u32 & 0x3F) << 12)
                | ((bytes[i + 2] as u32 & 0x3F) << 6)
                | (bytes[i + 3] as u32 & 0x3F),
            _ => unreachable!(),
        };
        match char::from_u32(cp) {
            Some(c) => out.push(c),
            None => out.push('\u{FFFD}'),
        }
        i += needed;
    }
    out
}

// jq normalizes a negative slice bound (`n + len`) *before* converting it to an
// integer; rounding the raw float first mis-placed fractional negatives like
// `-0.5`, because e.g. `(-0.5).ceil() == 0` discards the negativity and `len` is
// never added. Normalize, then floor (start) / ceil (end), then clamp. #722.
fn slice_index_start(n: f64, len: i64) -> usize {
    if n.is_nan() { return 0; }
    let norm = if n < 0.0 { n + len as f64 } else { n };
    (norm.floor() as i64).clamp(0, len) as usize
}

fn slice_index_end(n: f64, len: i64) -> usize {
    if n.is_nan() { return len as usize; }
    let norm = if n < 0.0 { n + len as f64 } else { n };
    (norm.ceil() as i64).clamp(0, len) as usize
}

pub fn eval_slice(base: &Value, from: &Value, to: &Value) -> Result<Value> {
    match base {
        Value::Arr(a) => {
            let len = a.len() as i64;
            let fi = match from { Value::Num(n, _) => slice_index_start(*n, len), Value::Null => 0, _ => bail!("Array/string slice indices must be integers") };
            let ti = match to { Value::Num(n, _) => slice_index_end(*n, len), Value::Null => len as usize, _ => bail!("Array/string slice indices must be integers") };
            Ok(if fi>=ti { Value::Arr(Rc::new(vec![])) } else { Value::Arr(Rc::new(a[fi..ti].to_vec())) })
        }
        Value::Str(s) => {
            let s_str = s.as_str();
            // ASCII fast path: byte index == char index, no allocation needed
            if s_str.is_ascii() {
                let len = s_str.len() as i64;
                let fi = match from { Value::Num(n, _) => slice_index_start(*n, len), Value::Null => 0, _ => bail!("Array/string slice indices must be integers") };
                let ti = match to { Value::Num(n, _) => slice_index_end(*n, len), Value::Null => len as usize, _ => bail!("Array/string slice indices must be integers") };
                Ok(if fi>=ti { Value::from_str("") } else { Value::from_str(&s_str[fi..ti]) })
            } else {
                // Unicode: count chars without allocation, use char_indices for byte offsets
                let char_count = s_str.chars().count() as i64;
                let fi = match from { Value::Num(n, _) => slice_index_start(*n, char_count), Value::Null => 0, _ => bail!("Array/string slice indices must be integers") };
                let ti = match to { Value::Num(n, _) => slice_index_end(*n, char_count), Value::Null => char_count as usize, _ => bail!("Array/string slice indices must be integers") };
                Ok(if fi>=ti { Value::from_str("") } else {
                    let mut ci = s_str.char_indices();
                    let start_byte = ci.nth(fi).map(|(pos, _)| pos).unwrap_or(s_str.len());
                    let end_byte = ci.nth(ti - fi - 1).map(|(pos, _)| pos).unwrap_or(s_str.len());
                    Value::from_str(&s_str[start_byte..end_byte])
                })
            }
        }
        Value::Null => Ok(Value::Null),
        // jq treats slice as a path access whose key is the {start, end}
        // object, so type errors share the "Cannot index X with object"
        // wording rather than a slice-specific message. See #442.
        _ => bail!("Cannot index {} with object", base.type_name()),
    }
}

/// Specialized closure ops with pre-computed f64 keys — avoids eval overhead entirely.
fn eval_closure_op_f64(op: ClosureOpKind, a: &[Value], keys: &[f64], cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    let cmp_f64 = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    match op {
        ClosureOpKind::SortBy => {
            let mut indices: Vec<usize> = (0..a.len()).collect();
            indices.sort_by(|&i, &j| cmp_f64(&keys[i], &keys[j]));
            cb(Value::Arr(Rc::new(indices.iter().map(|&i| a[i].clone()).collect())))
        }
        ClosureOpKind::GroupBy => {
            let mut indices: Vec<usize> = (0..a.len()).collect();
            indices.sort_by(|&i, &j| cmp_f64(&keys[i], &keys[j]));
            let mut groups: Vec<Value> = Vec::new();
            let mut cg: Vec<Value> = Vec::new();
            let mut cur_key: Option<f64> = None;
            for &idx in &indices {
                let k = keys[idx];
                if let Some(ck) = cur_key {
                    if k == ck {
                        cg.push(a[idx].clone());
                    } else {
                        groups.push(Value::Arr(Rc::new(std::mem::take(&mut cg))));
                        cg.push(a[idx].clone());
                        cur_key = Some(k);
                    }
                } else {
                    cg.push(a[idx].clone());
                    cur_key = Some(k);
                }
            }
            if !cg.is_empty() { groups.push(Value::Arr(Rc::new(cg))); }
            cb(Value::Arr(Rc::new(groups)))
        }
        ClosureOpKind::UniqueBy => {
            // jq: unique_by(f) = group_by(f) | map(.[0]) — sorted by key, deduped.
            let mut indices: Vec<usize> = (0..a.len()).collect();
            indices.sort_by(|&i, &j| cmp_f64(&keys[i], &keys[j]));
            let mut result: Vec<Value> = Vec::new();
            let mut prev: Option<f64> = None;
            for &idx in &indices {
                let k = keys[idx];
                if prev.map_or(true, |pk| cmp_f64(&pk, &k) != std::cmp::Ordering::Equal) {
                    result.push(a[idx].clone());
                    prev = Some(k);
                }
            }
            cb(Value::Arr(Rc::new(result)))
        }
        ClosureOpKind::MinBy => {
            if a.is_empty() { return cb(Value::Null); }
            let mut mi = 0;
            for i in 1..a.len() {
                if cmp_f64(&keys[i], &keys[mi]) == std::cmp::Ordering::Less { mi = i; }
            }
            cb(a[mi].clone())
        }
        ClosureOpKind::MaxBy => {
            if a.is_empty() { return cb(Value::Null); }
            let mut mi = 0;
            for i in 1..a.len() {
                let c = cmp_f64(&keys[i], &keys[mi]);
                if c == std::cmp::Ordering::Greater || c == std::cmp::Ordering::Equal { mi = i; }
            }
            cb(a[mi].clone())
        }
    }
}

/// Specialized closure ops with pre-extracted Value references — avoids eval and clone overhead.
/// Choose a specialized comparator based on the first key's type.
fn sort_indexed_by_key(indexed: &mut [(usize, &Value)]) {
    if indexed.is_empty() { return; }
    match indexed[0].1 {
        Value::Str(_) => {
            indexed.sort_by(|(_, ka), (_, kb)| {
                if let (Value::Str(a), Value::Str(b)) = (ka, kb) {
                    a.cmp(b)
                } else {
                    crate::runtime::compare_values(ka, kb)
                }
            });
        }
        Value::Num(..) => {
            // `partial_cmp` returns None for NaN and would collapse to Equal,
            // leaving NaN keys misordered (and dropped by group_by/unique_by).
            // jq's total order (#115) places NaN below every number, which
            // `compare_values` implements. #770
            indexed.sort_by(|(_, ka), (_, kb)| crate::runtime::compare_values(ka, kb));
        }
        _ => {
            indexed.sort_by(|(_, ka), (_, kb)| crate::runtime::compare_values(ka, kb));
        }
    }
}

fn eval_closure_op_value_ref(op: ClosureOpKind, a: &[Value], keyed: Vec<(&Value, &Value)>, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    match op {
        ClosureOpKind::SortBy => {
            let mut indexed: Vec<(usize, &Value)> = keyed.iter().enumerate().map(|(i, (k, _))| (i, *k)).collect();
            sort_indexed_by_key(&mut indexed);
            cb(Value::Arr(Rc::new(indexed.iter().map(|&(i, _)| a[i].clone()).collect())))
        }
        ClosureOpKind::GroupBy => {
            let mut indexed: Vec<(usize, &Value)> = keyed.iter().enumerate().map(|(i, (k, _))| (i, *k)).collect();
            sort_indexed_by_key(&mut indexed);
            let mut groups: Vec<Value> = Vec::new();
            let mut cg: Vec<Value> = Vec::new();
            let mut cur_key: Option<&Value> = None;
            for &(idx, key) in &indexed {
                if let Some(ck) = cur_key {
                    if crate::runtime::values_equal(key, ck) {
                        cg.push(a[idx].clone());
                    } else {
                        groups.push(Value::Arr(Rc::new(std::mem::take(&mut cg))));
                        cg.push(a[idx].clone());
                        cur_key = Some(key);
                    }
                } else {
                    cg.push(a[idx].clone());
                    cur_key = Some(key);
                }
            }
            if !cg.is_empty() { groups.push(Value::Arr(Rc::new(cg))); }
            cb(Value::Arr(Rc::new(groups)))
        }
        ClosureOpKind::UniqueBy => {
            // jq: unique_by(f) = group_by(f) | map(.[0]) — sorted by key, deduped.
            let mut indexed: Vec<(usize, &Value)> = keyed.iter().enumerate().map(|(i, (k, _))| (i, *k)).collect();
            sort_indexed_by_key(&mut indexed);
            let mut result: Vec<Value> = Vec::new();
            let mut prev: Option<&Value> = None;
            for &(idx, key) in &indexed {
                if prev.map_or(true, |pk| !crate::runtime::values_equal(pk, key)) {
                    result.push(a[idx].clone());
                    prev = Some(key);
                }
            }
            cb(Value::Arr(Rc::new(result)))
        }
        ClosureOpKind::MinBy => {
            if keyed.is_empty() { return cb(Value::Null); }
            let mut mi = 0;
            for i in 1..keyed.len() {
                if crate::runtime::compare_values(keyed[i].0, keyed[mi].0) == std::cmp::Ordering::Less { mi = i; }
            }
            cb(keyed[mi].1.clone())
        }
        ClosureOpKind::MaxBy => {
            if keyed.is_empty() { return cb(Value::Null); }
            let mut mi = 0;
            for i in 1..keyed.len() {
                let c = crate::runtime::compare_values(keyed[i].0, keyed[mi].0);
                if c == std::cmp::Ordering::Greater || c == std::cmp::Ordering::Equal { mi = i; }
            }
            cb(keyed[mi].1.clone())
        }
    }
}

/// Try to evaluate a key expression as a single f64 without full eval overhead.
fn try_eval_key_f64(expr: &Expr, input: &Value) -> Option<f64> {
    match expr {
        Expr::Input => match input { Value::Num(n, _) => Some(*n), _ => None },
        Expr::Literal(Literal::Num(n, _)) => Some(*n),
        Expr::BinOp { op, lhs, rhs } => {
            let l = try_eval_key_f64(lhs, input)?;
            let r = try_eval_key_f64(rhs, input)?;
            match op {
                BinOp::Add => Some(l + r),
                BinOp::Sub => Some(l - r),
                BinOp::Mul => Some(l * r),
                BinOp::Div => if r != 0.0 { Some(l / r) } else { None },
                BinOp::Mod => Some(l % r),
                _ => None,
            }
        }
        Expr::Negate { operand } => try_eval_key_f64(operand, input).map(|v| if v == 0.0 { 0.0 } else { -v }),
        Expr::UnaryOp { op, operand } => {
            // Length can work on any type (string→charcount, array→len, object→len, number→fabs)
            if matches!(op, UnaryOp::Length) {
                // Try to get the original Value for length calculation
                if let Some(v) = try_eval_key_value(operand, input) {
                    return match v {
                        Value::Str(s) => Some(s.chars().count() as f64),
                        Value::Arr(a) => Some(a.len() as f64),
                        Value::Obj(ObjInner(o)) => Some(o.len() as f64),
                        Value::Num(n, _) => Some(n.abs()),
                        Value::Null => Some(0.0),
                        _ => None,
                    };
                }
                // Fallback: try as f64 (for piped numeric expressions)
                return try_eval_key_f64(operand, input).map(|v| v.abs());
            }
            let v = try_eval_key_f64(operand, input)?;
            match op {
                UnaryOp::Floor => Some(v.floor()),
                UnaryOp::Ceil => Some(v.ceil()),
                UnaryOp::Sqrt => Some(v.sqrt()),
                UnaryOp::Fabs | UnaryOp::Abs => Some(v.abs()),
                UnaryOp::Round => Some(v.round()),
                _ => None,
            }
        }
        Expr::Index { expr: base, key } => {
            if !matches!(base.as_ref(), Expr::Input) { return None; }
            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                match input {
                    Value::Obj(ObjInner(o)) => match o.get(field.as_str()) {
                        Some(Value::Num(n, _)) => Some(*n),
                        _ => None,
                    },
                    _ => None,
                }
            } else { None }
        }
        Expr::Pipe { left, right } => {
            // Try f64 pipe first
            if let Some(mid_val) = try_eval_key_f64(left, input) {
                let mid = Value::number(mid_val);
                return try_eval_key_f64(right, &mid);
            }
            // Try Value pipe (e.g., .name | length)
            if let Some(mid_val) = try_eval_key_value(left, input) {
                return try_eval_key_f64(right, mid_val);
            }
            None
        }
        _ => None,
    }
}

fn try_eval_key_value<'a>(expr: &Expr, input: &'a Value) -> Option<&'a Value> {
    match expr {
        Expr::Input => Some(input),
        Expr::Index { expr: base, key } => {
            if !matches!(base.as_ref(), Expr::Input) { return None; }
            if let Expr::Literal(Literal::Str(field)) = key.as_ref() {
                match input {
                    Value::Obj(ObjInner(o)) => o.get(field.as_str()),
                    _ => None,
                }
            } else { None }
        }
        Expr::Pipe { left, right } => {
            let mid = try_eval_key_value(left, input)?;
            try_eval_key_value(right, mid)
        }
        _ => None,
    }
}

fn eval_closure_op(op: ClosureOpKind, container: &Value, key_expr: &Expr, _input: &Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    let a = match container {
        Value::Arr(a) => a,
        // jq evaluates the projection over object values first (so any
        // projection error propagates), then bails with a wording that
        // names both the input and the projected key array. See #456.
        // sort_by/group_by/unique_by say "cannot be sorted, as they are
        // not both arrays"; min_by/max_by say "cannot be iterated over".
        Value::Obj(crate::value::ObjInner(o)) => {
            let mut projections: Vec<Value> = Vec::with_capacity(o.len());
            for v in o.values() {
                let mut keys: Vec<Value> = Vec::new();
                eval(key_expr, v.clone(), env, &mut |k| { keys.push(k); Ok(true) })?;
                projections.push(Value::Arr(Rc::new(keys)));
            }
            let proj = Value::Arr(Rc::new(projections));
            let suffix = match op {
                ClosureOpKind::SortBy
                | ClosureOpKind::GroupBy
                | ClosureOpKind::UniqueBy => "cannot be sorted, as they are not both arrays",
                ClosureOpKind::MinBy | ClosureOpKind::MaxBy => "cannot be iterated over",
            };
            bail!(
                "{} and {} {}",
                crate::runtime::errdesc_pub(container),
                crate::runtime::errdesc_pub(&proj),
                suffix
            );
        }
        _ => bail!("Cannot iterate over {}", crate::runtime::errdesc_pub(container)),
    };

    // Fast path: f64 key extraction — avoids eval overhead and Vec<Value> allocations.
    // A NaN key disqualifies this path: its raw IEEE comparator neither orders
    // NaN (jq sorts it smallest) nor keeps NaN keys distinct under group_by /
    // unique_by (NaN is never equal to itself). Such inputs fall through to the
    // Value-based paths, which route ordering through `compare_values` and
    // grouping through `values_equal`. #770
    if !a.is_empty() {
        if let Some(first_key) = try_eval_key_f64(key_expr, &a[0]).filter(|k| !k.is_nan()) {
            let mut f64_keys: Vec<f64> = Vec::with_capacity(a.len());
            f64_keys.push(first_key);
            let mut all_f64 = true;
            for item in &a[1..] {
                match try_eval_key_f64(key_expr, item) {
                    Some(k) if !k.is_nan() => f64_keys.push(k),
                    _ => {
                        all_f64 = false;
                        break;
                    }
                }
            }
            if all_f64 {
                return eval_closure_op_f64(op, a, &f64_keys, cb);
            }
        }
    }

    // Fast path: direct Value key extraction (handles .field with any type)
    if !a.is_empty() && try_eval_key_value(key_expr, &a[0]).is_some() {
        let mut all_ok = true;
        let mut keyed: Vec<(&Value, &Value)> = Vec::with_capacity(a.len());
        for item in a.iter() {
            if let Some(k) = try_eval_key_value(key_expr, item) {
                keyed.push((k, item));
            } else {
                all_ok = false;
                break;
            }
        }
        if all_ok {
            return eval_closure_op_value_ref(op, a, keyed, cb);
        }
    }

    // Multi-valued key path: jq collects every output of the key expression
    // into an array and orders those arrays with its standard value comparison —
    // lexicographically, with a shorter prefix sorting before a longer one
    // (`[] < [1] < [1,2]`). The previous hand-rolled comparator zipped to the
    // shorter length and treated a prefix as equal, and never applied jq's NaN
    // ordering. Wrapping the collected keys in `Value::Arr` lets `compare_values`
    // (ordering) and `values_equal` (grouping) handle both correctly. #770
    let mut keyed: Vec<(Value, Value)> = Vec::new();
    for item in a.iter() {
        let mut keys = Vec::new();
        eval(key_expr, item.clone(), env, &mut |k| { keys.push(k); Ok(true) })?;
        keyed.push((Value::Arr(Rc::new(keys)), item.clone()));
    }
    match op {
        ClosureOpKind::SortBy => {
            keyed.sort_by(|(ka, _), (kb, _)| crate::runtime::compare_values(ka, kb));
            cb(Value::Arr(Rc::new(keyed.into_iter().map(|(_, v)| v).collect())))
        }
        ClosureOpKind::GroupBy => {
            keyed.sort_by(|(ka, _), (kb, _)| crate::runtime::compare_values(ka, kb));
            let mut groups: Vec<Value> = Vec::new(); let mut cg: Vec<Value> = Vec::new(); let mut ck: Option<Value> = None;
            for (key, val) in keyed {
                if let Some(ref pk) = ck {
                    if crate::runtime::values_equal(&key, pk) { cg.push(val); }
                    else { groups.push(Value::Arr(Rc::new(std::mem::take(&mut cg)))); cg.push(val); ck = Some(key); }
                } else { cg.push(val); ck = Some(key); }
            }
            if !cg.is_empty() { groups.push(Value::Arr(Rc::new(cg))); }
            cb(Value::Arr(Rc::new(groups)))
        }
        ClosureOpKind::UniqueBy => {
            // jq: unique_by(f) = group_by(f) | map(.[0]) — sorted by key, deduped.
            keyed.sort_by(|(ka, _), (kb, _)| crate::runtime::compare_values(ka, kb));
            let mut result: Vec<Value> = Vec::new();
            let mut prev: Option<Value> = None;
            for (key, val) in keyed {
                let is_dup = prev.as_ref().is_some_and(|pk| crate::runtime::values_equal(pk, &key));
                if !is_dup {
                    result.push(val);
                    prev = Some(key);
                }
            }
            cb(Value::Arr(Rc::new(result)))
        }
        ClosureOpKind::MinBy => {
            if keyed.is_empty() { cb(Value::Null) } else {
                let mut mi = 0;
                for i in 1..keyed.len() {
                    if crate::runtime::compare_values(&keyed[i].0, &keyed[mi].0) == std::cmp::Ordering::Less { mi = i; }
                }
                cb(keyed[mi].1.clone())
            }
        }
        ClosureOpKind::MaxBy => {
            if keyed.is_empty() { cb(Value::Null) } else {
                let mut mi = 0;
                for i in 1..keyed.len() {
                    let cmp = crate::runtime::compare_values(&keyed[i].0, &keyed[mi].0);
                    if cmp == std::cmp::Ordering::Greater || cmp == std::cmp::Ordering::Equal { mi = i; }
                }
                cb(keyed[mi].1.clone())
            }
        }
    }
}

/// Standalone assign for JIT: collect all results into an array.
pub fn eval_assign_standalone(path_expr: &Expr, value_expr: &Expr, input: Value, env_ref: &Rc<RefCell<Env>>) -> Result<Value> {
    let mut results = Vec::new();
    let assign_expr = Expr::Assign {
        path_expr: Box::new(path_expr.clone()),
        value_expr: Box::new(value_expr.clone()),
    };
    eval(&assign_expr, input, env_ref, &mut |v| { results.push(v); Ok(true) })?;
    Ok(Value::Arr(Rc::new(results)))
}

/// Standalone update for JIT: collect all results into an array.
pub fn eval_update_standalone(path_expr: &Expr, update_expr: &Expr, input: Value, env_ref: &Rc<RefCell<Env>>) -> Result<Value> {
    let mut results = Vec::new();
    let update_expr_ir = Expr::Update {
        path_expr: Box::new(path_expr.clone()),
        update_expr: Box::new(update_expr.clone()),
    };
    eval(&update_expr_ir, input, env_ref, &mut |v| { results.push(v); Ok(true) })?;
    Ok(Value::Arr(Rc::new(results)))
}

/// Standalone path evaluation for JIT: collect all path results into an array.
pub fn eval_path_standalone(path_expr: &Expr, input: Value, env_ref: &Rc<RefCell<Env>>) -> Result<Value> {
    let mut results = Vec::new();
    let result = eval_path(path_expr, input, env_ref, &mut |v| {
        results.push(v);
        Ok(true)
    });
    match result {
        Err(e) => Err(invalid_path_expr_err(e)),
        Ok(_) => Ok(Value::Arr(Rc::new(results))),
    }
}

/// Standalone closure op for JIT: evaluate closure operation with a fresh env.
pub fn eval_closure_op_standalone(op: ClosureOpKind, container: &Value, key_expr: &Expr, env_ref: &Rc<RefCell<Env>>) -> Result<Value> {
    let mut result = Value::Null;
    eval_closure_op(op, container, key_expr, container, env_ref, &mut |v| {
        result = v;
        Ok(true)
    })?;
    Ok(result)
}

// Build the interpolated string by recursing from the LAST part to the first,
// prepending each piece to the `suffix` accumulated from the parts to its
// right. This makes the rightmost generator hole the outermost loop, so the
// leftmost hole varies fastest — matching jq: `"\(1,2)\(3,4)"` yields
// "13","23","14","24". Iterating left-to-right reversed that order (#817).
fn eval_interp_parts(parts: &[StringPart], idx: isize, suffix: String, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    if idx < 0 { return cb(Value::from_str(&suffix)); }
    match &parts[idx as usize] {
        StringPart::Literal(s) => { let mut n = s.clone(); n.push_str(&suffix); eval_interp_parts(parts, idx-1, n, input, env, cb) }
        StringPart::Expr(e) => {
            eval(e, input.clone(), env, &mut |val| {
                // String interpolation runs `tostring` semantics on the
                // interpolated value (see jq's manual). `value_to_json`
                // discards the carried number repr, so `"\(0.0)"` would
                // render as `"0"` instead of jq's `"0.0"`. Use
                // `value_to_json_tojson` to keep the literal form when the
                // f64 round-trips it exactly. See #560.
                let s = match &val { Value::Str(s) => s.to_string(), _ => crate::value::value_to_json_tojson(&val) };
                let mut n = s; n.push_str(&suffix);
                eval_interp_parts(parts, idx-1, n, input.clone(), env, cb)
            })
        }
    }
}

/// jq treats `.[OBJ]` on array/string as a slice when OBJ has both
/// `start` and `end` keys whose values are Num or Null. Extra keys are
/// allowed; floats are accepted (truncated downstream by rt_slice).
/// See #596.
fn is_valid_slice_object(v: &Value) -> bool {
    let Value::Obj(crate::value::ObjInner(o)) = v else { return false; };
    let valid = |k: &str| matches!(o.get(k), Some(Value::Num(_, _) | Value::Null));
    valid("start") && valid("end")
}

thread_local! {
    // Active path-mode `foreach` element bindings: (element-var index, source
    // path). When a bare `$x` reference to a registered element var is
    // evaluated in path mode — as the `nth(n; g)` desugar's extract does — it
    // forwards the element's source path instead of being rejected as a value.
    // Outside a path-mode `foreach`, the stack is empty and variables behave
    // as ordinary (non-path) values. See #711.
    static FOREACH_PATH_BIND: RefCell<Vec<(VarIdx, Value)>> = const { RefCell::new(Vec::new()) };
}

thread_local! {
    // Depth>0 means the `eval_path` INPUT is a rootless `foreach` accumulator:
    // a navigating source (`.[]`) owns the path register, so the accumulator
    // carries no provenance. In that state the bare identity `.` surfaces the
    // PathResultSignal (→ "Invalid path expression with result
    // <acc>") and a navigation off it (`.a`) becomes "near attempt to access …"
    // via the existing Index/Each recovery, while a `$x` reference still
    // forwards its source path (`LoadVar` never touches the identity arm). Only
    // the EXTRACT (3-arg) / UPDATE (2-arg) evaluation of a navigating-source
    // foreach raises the depth, so ordinary path expressions are unaffected.
    // #915
    static FOREACH_ROOTLESS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };

    // Document root for the active rootless navigating-source `foreach`
    // extract/update, pushed in lockstep with `FOREACH_ROOTLESS`. A
    // `$x`-anchored navigation (`$x.k`, `$x[i]`) forwards the element's source
    // path, which is *document*-relative — so its key must be validated against
    // the document the element lives in, not the rootless accumulator. Resolving
    // it against the accumulator lenient-nulled a non-indexable element and let
    // an invalid navigation slip through. #953
    static FOREACH_DOC_ROOT: RefCell<Vec<Value>> = const { RefCell::new(Vec::new()) };
}

/// RAII guard that marks the current `eval_path` input as a rootless `foreach`
/// accumulator for its lifetime (see [`FOREACH_ROOTLESS`]) and records the
/// document root for `$x`-anchored key validation (see [`FOREACH_DOC_ROOT`]).
/// Restores the prior state on drop, so early returns / error propagation can't
/// leak it.
struct RootlessAccGuard;
impl RootlessAccGuard {
    fn enter(doc_root: Value) -> Self {
        FOREACH_ROOTLESS.with(|c| c.set(c.get() + 1));
        FOREACH_DOC_ROOT.with(|s| s.borrow_mut().push(doc_root));
        RootlessAccGuard
    }
}
impl Drop for RootlessAccGuard {
    fn drop(&mut self) {
        FOREACH_ROOTLESS.with(|c| c.set(c.get().saturating_sub(1)));
        FOREACH_DOC_ROOT.with(|s| { s.borrow_mut().pop(); });
    }
}

/// The document root of the innermost active rootless navigating-source
/// `foreach` extract/update, or `None` when not inside one. Used to validate a
/// `$x`-anchored navigation's key against the document rather than the rootless
/// accumulator (#953).
fn rootless_doc_root() -> Option<Value> {
    if FOREACH_ROOTLESS.with(|c| c.get()) == 0 {
        return None;
    }
    FOREACH_DOC_ROOT.with(|s| s.borrow().last().cloned())
}

thread_local! {
    // Variables bound to the identity path via `. as $x`. jq tracks the empty
    // path provenance of an identity binding, so `path($x)` is the identity
    // path `[]` as long as the current input still equals the captured value
    // (it becomes invalid once `.` navigates away). A literal binding such as
    // `5 as $x` carries no path and is always rejected. Each binding site has a
    // globally unique var index (the parser never reuses one), so a non-identity
    // rebind of the same name gets a fresh index absent from this set and
    // correctly shadows. See #837.
    static IDENTITY_PATH_VARS: RefCell<Vec<VarIdx>> = const { RefCell::new(Vec::new()) };
}

/// RAII guard registering `var_index` as identity-path-bound for its lifetime.
struct IdentityVarGuard;
impl Drop for IdentityVarGuard {
    fn drop(&mut self) {
        IDENTITY_PATH_VARS.with(|s| { s.borrow_mut().pop(); });
    }
}
fn push_identity_path_var(var_index: VarIdx) -> IdentityVarGuard {
    IDENTITY_PATH_VARS.with(|s| s.borrow_mut().push(var_index));
    IdentityVarGuard
}
fn is_identity_path_var(var_index: VarIdx) -> bool {
    IDENTITY_PATH_VARS.with(|s| s.borrow().contains(&var_index))
}

thread_local! {
    // Variables that are valid *navigation sources* for path-tracked
    // destructuring: their value carries a path provenance AND they branch into
    // at most one navigated sub-binding (a single spine). A child binding
    // `$tmp[k] as $a` only inherits a path when `$tmp` is registered here. jq
    // tracks a single path register through destructuring, so two or more
    // sibling navigations off the same source corrupt it — we mirror that by
    // refusing to register a source that is referenced more than once. See #880.
    static NAV_PATH_SOURCES: RefCell<Vec<VarIdx>> = const { RefCell::new(Vec::new()) };
}

/// RAII guard registering `var_index` as a single-spine navigation source.
struct NavSourceGuard;
impl Drop for NavSourceGuard {
    fn drop(&mut self) {
        NAV_PATH_SOURCES.with(|s| { s.borrow_mut().pop(); });
    }
}
fn push_nav_path_source(var_index: VarIdx) -> NavSourceGuard {
    NAV_PATH_SOURCES.with(|s| s.borrow_mut().push(var_index));
    NavSourceGuard
}
fn is_nav_path_source(var_index: VarIdx) -> bool {
    NAV_PATH_SOURCES.with(|s| s.borrow().contains(&var_index))
}

/// Emit one slice path component `{start, end}` appended to base path `bp`,
/// type-checking the receiver the same way jq does. Shared by the single-bound
/// fast path and the Cartesian-product slow path of path-context slicing (#761).
fn emit_slice_path(
    input: &Value, bp: &Value, from_val: &Value, to_val: &Value,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    // Type-check the receiver: jq errors on path slicing of non-array/string/null.
    let base = crate::runtime::rt_getpath(input, bp).unwrap_or(Value::Null);
    match &base {
        Value::Arr(_) | Value::Str(_) | Value::Null => {}
        other => bail!("Cannot index {} with object", other.type_name()),
    }
    // jq preserves the literal slice expressions in path output without
    // clamping to the receiver's actual length. Omitted bounds → null.
    let mut p = match bp { Value::Arr(a) => a.as_ref().clone(), _ => vec![] };
    p.push(Value::object_from_map({
        let mut m = crate::value::new_objmap();
        m.insert("start".into(), from_val.clone());
        m.insert("end".into(), to_val.clone());
        m
    }));
    cb(Value::Arr(Rc::new(p)))
}

/// Truncate a rendered JSON dump for path-expression error messages the way
/// jq's `jv_dump_string_trunc` does (buffer size 30): a dump longer than 29
/// bytes keeps its first 26 bytes followed by "...". jq does a raw byte
/// `strncpy`; we cut at the largest char boundary at or below byte 26 so the
/// message stays valid UTF-8 (identical to jq for ASCII, which covers the
/// "result <X>" and "near attempt to access <K> of <V>" sinks). #870 follow-up.
/// Rewrite a [`PathResultSignal`](crate::signal::PathResultSignal) into the
/// user-facing "Invalid path expression with result ..." error; any other
/// error passes through unchanged.
fn invalid_path_expr_err(e: anyhow::Error) -> anyhow::Error {
    match crate::signal::take_path_result(&e) {
        Some(v) => anyhow::anyhow!(
            "Invalid path expression with result {}",
            trunc_path_dump(&crate::value::value_to_json(&v))
        ),
        None => e,
    }
}

fn trunc_path_dump(s: &str) -> String {
    if s.len() <= 29 {
        return s.to_string();
    }
    let mut cut = 26;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}...", &s[..cut])
}

/// jq's error for a `reduce`/`foreach` SOURCE that navigates the tracked input
/// while the accumulator path is already non-empty (INIT navigated). It names
/// the source's leftmost navigation hop: an iteration (`.[]`) reports "iterate
/// through <input>"; a field/index access (`.a`, `.k[]`) reports "access
/// element <key> of <input>". The reported container is always the root input.
/// #915
fn source_nav_error(source: &Expr, input: &Value, env: &EnvRef) -> anyhow::Error {
    enum Hop { Iterate, Access(Value) }
    fn leftmost(e: &Expr, input: &Value, env: &EnvRef) -> Option<Hop> {
        match e {
            Expr::Each { input_expr } | Expr::EachOpt { input_expr } => {
                leftmost(input_expr, input, env).or(Some(Hop::Iterate))
            }
            Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => {
                if let Some(h) = leftmost(expr, input, env) { return Some(h); }
                let mut kv = Value::Null;
                let _ = eval(key, input.clone(), env, &mut |k| { kv = k; Ok(false) });
                Some(Hop::Access(kv))
            }
            Expr::Pipe { left, right } => {
                leftmost(left, input, env).or_else(|| leftmost(right, input, env))
            }
            _ => None,
        }
    }
    let tail = match leftmost(source, input, env) {
        Some(Hop::Access(k)) => {
            let kd = match &k {
                Value::Str(s) => format!("\"{}\"", s),
                Value::Num(n, _) => crate::value::format_jq_number(*n),
                other => crate::value::value_to_json(other),
            };
            format!("attempt to access element {} of {}", kd, crate::value::value_to_json(input))
        }
        _ => format!("attempt to iterate through {}", crate::value::value_to_json(input)),
    };
    anyhow::anyhow!("Invalid path expression near {}", tail)
}

/// Lower the path-transparent `*str` trim builtins to their jq slice
/// definitions so the existing slice-path machinery produces jq's paths:
///   `ltrimstr($x)` → `if startswith($x) then .[($x|length):] else . end`
///   `rtrimstr($x)` → `if endswith($x)   then .[:-($x|length)] else . end`
///   `trimstr($x)`  → `ltrimstr($x) | rtrimstr($x)`
/// A no-match yields the identity path `[]`; a match yields the slice path
/// (`[{start,end}]`), which `|=`/`=` reject with "Cannot update string slices"
/// and `del` removes — exactly as jq does (#962). The startswith/endswith
/// condition raises jq's "… requires string inputs" on a non-string input or
/// argument. `$x` (`arg`) is evaluated against the same input as the builtin,
/// matching `def`-bound semantics; it is referenced twice, so a generator
/// argument (vanishingly rare in a path expression) is re-evaluated.
fn lower_trimstr_path(name: &str, arg: &Expr) -> Expr {
    let len_of_arg = || Expr::UnaryOp { op: crate::ir::UnaryOp::Length, operand: Box::new(arg.clone()) };
    match name {
        "ltrimstr" => Expr::IfThenElse {
            cond: Box::new(Expr::CallBuiltin { name: "startswith".to_string(), args: vec![arg.clone()] }),
            then_branch: Box::new(Expr::Slice { expr: Box::new(Expr::Input), from: Some(Box::new(len_of_arg())), to: None }),
            else_branch: Box::new(Expr::Input),
        },
        "rtrimstr" => Expr::IfThenElse {
            cond: Box::new(Expr::CallBuiltin { name: "endswith".to_string(), args: vec![arg.clone()] }),
            then_branch: Box::new(Expr::Slice { expr: Box::new(Expr::Input), from: None, to: Some(Box::new(Expr::Negate { operand: Box::new(len_of_arg()) })) }),
            else_branch: Box::new(Expr::Input),
        },
        // `trimstr` is `ltrimstr | rtrimstr`: a left match then a right match
        // compose into a two-component slice path, exactly as jq emits.
        _ => Expr::Pipe {
            left: Box::new(lower_trimstr_path("ltrimstr", arg)),
            right: Box::new(lower_trimstr_path("rtrimstr", arg)),
        },
    }
}

fn eval_path(expr: &Expr, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    match expr {
        Expr::Input => {
            // A rootless `foreach` accumulator has no path: the identity `.`
            // surfaces as the sink's "Invalid path expression with result
            // <acc>" rather than the root path `[]`. Navigation off it (`.a`)
            // raises the same signal from the base, which the enclosing
            // Index/Each arm rewrites into "near attempt to access …". #915
            if FOREACH_ROOTLESS.with(|c| c.get()) > 0 {
                return Err(crate::signal::PathResultSignal::raise(&input));
            }
            cb(Value::Arr(Rc::new(vec![])))
        }
        Expr::Index { expr: be, key: ke } => {
            let cb_called = std::cell::Cell::new(false);
            let input_for_check = input.clone();
            // Collect the base paths first (preserving the path-validity error
            // recovery below), then iterate subscript-outer / base-inner so the
            // leftmost generator varies fastest, matching jq's order for
            // `path(.[0,1][0,1])` (#817). Validating per (key, base) is
            // equivalent to the old (base, key) loop — only the iteration
            // order differs.
            let mut base_paths: Vec<Value> = Vec::new();
            let result = eval_path(be, input.clone(), env, &mut |bp| {
                cb_called.set(true);
                base_paths.push(bp);
                Ok(true)
            })
            .and_then(|_| {
                eval(ke, input.clone(), env, &mut |key| {
                    for bp in &base_paths {
                        // jq errors `path(.field)` when the base value at the
                        // current path can't accept the key type (issue #46).
                        // Only objects (with string keys), arrays (with number
                        // keys), and null (a no-op) are valid bases.
                        //
                        // In a rootless navigating-source `foreach` extract/
                        // update, any base path reaching here is `$x`-anchored:
                        // a `.`-anchored navigation bails the rootless sentinel
                        // via the `Input` arm before producing a path. That path
                        // is document-relative, so validate the key against the
                        // document the element lives in — resolving it against
                        // the rootless accumulator lenient-nulled a non-indexable
                        // element and accepted an invalid navigation (#953).
                        let base_val = rootless_doc_root()
                            .map(|doc| crate::runtime::rt_getpath(&doc, bp).unwrap_or(Value::Null))
                            .unwrap_or_else(|| crate::runtime::rt_getpath(&input_for_check, bp).unwrap_or(Value::Null));
                        match (&base_val, &key) {
                            (Value::Obj(_), Value::Str(_)) => {}
                            (Value::Arr(_), Value::Num(_, _)) => {}
                            // jq accepts `path(.[arr])` on an array — the array
                            // key becomes a single path component. Updates via
                            // this path still fail in rt_setpath with `Cannot
                            // update field at array index of array`. See #467.
                            (Value::Arr(_), Value::Arr(_)) => {}
                            // jq treats `.[OBJ]` on array/string as a slice when
                            // OBJ has both `start` and `end` keys with Num/Null
                            // values. Otherwise it errors with `Array/string
                            // slice indices must be integers`. See #596.
                            (Value::Arr(_) | Value::Str(_), Value::Obj(_)) => {
                                if !is_valid_slice_object(&key) {
                                    bail!("Array/string slice indices must be integers");
                                }
                            }
                            // null accepts string/number/object keys (the slicing
                            // form), but jq errors on bool/null/array keys with
                            // `Cannot index null with <type>` (#594).
                            (Value::Null, Value::Str(_) | Value::Num(_, _) | Value::Obj(_)) => {}
                            _ => {
                                // jq's wording: string keys keep the quoted
                                // value (`string "x"`), other key types use
                                // the bare type name (`number`, `boolean`,
                                // `null`). Aligns with the read-side fix from
                                // #440. See #500.
                                let key_desc = match &key {
                                    Value::Str(s) => format!("string \"{}\"", s),
                                    other => other.type_name().to_string(),
                                };
                                bail!("Cannot index {} with {}", base_val.type_name(), key_desc);
                            }
                        }
                        let mut p = match bp { Value::Arr(a) => a.as_ref().clone(), _ => vec![] };
                        p.push(key.clone());
                        if !cb(Value::Arr(Rc::new(p)))? { return Ok(false); }
                    }
                    Ok(true)
                })
            });
            match result {
                Err(e) if !cb_called.get() => {
                    if let Some(pv) = crate::signal::take_path_result(&e) {
                        let mut key_val = Value::Null;
                        let _ = eval(ke, input, env, &mut |k| { key_val = k; Ok(true) });
                        let key_desc = match &key_val {
                            Value::Num(n, _) => format!("element {} of", crate::value::format_jq_number(*n)),
                            Value::Str(s) => format!("element \"{}\" of", s),
                            _ => format!("element {} of", crate::value::value_to_json(&key_val)),
                        };
                        bail!("Invalid path expression near attempt to access {} {}", key_desc, trunc_path_dump(&crate::value::value_to_json(&pv)));
                    }
                    Err(e)
                }
                other => other,
            }
        }
        Expr::Each { input_expr } => {
            let cb_called = std::cell::Cell::new(false);
            let result = eval_path(input_expr, input.clone(), env, &mut |bp| {
                cb_called.set(true);
                let base = crate::runtime::rt_getpath(&input, &bp).unwrap_or(Value::Null);
                match &base {
                    Value::Arr(a) => { for i in 0..a.len() { let mut p = match &bp { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] }; p.push(Value::number(i as f64)); if !cb(Value::Arr(Rc::new(p)))? { return Ok(false); } } Ok(true) }
                    Value::Obj(ObjInner(o)) => { for k in o.keys() { let mut p = match &bp { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] }; p.push(Value::from_str(k)); if !cb(Value::Arr(Rc::new(p)))? { return Ok(false); } } Ok(true) }
                    _ => {
                        // jq errors `del(.[])` etc. when the current path
                        // points at a non-iterable (issue #54). Silent
                        // "no paths" turned type errors into no-ops. Use
                        // errdesc so number reprs survive (`0.0` stays
                        // `0.0`) and long values get truncated. See #574.
                        bail!("Cannot iterate over {}", crate::runtime::errdesc_pub(&base))
                    }
                }
            });
            match result {
                Err(e) if !cb_called.get() => {
                    if let Some(pv) = crate::signal::take_path_result(&e) {
                        bail!("Invalid path expression near attempt to iterate through {}", crate::value::value_to_json(&pv));
                    }
                    Err(e)
                }
                other => other,
            }
        }
        Expr::Pipe { left, right } => {
            // Track whether LEFT produced a path: if so, a PathResultSignal
            // below came from RIGHT applied to a real `mid` value, so its
            // payload IS the offending value already — reporting it directly
            // avoids re-running RIGHT on it (which double-applies a non-path
            // transform: `path(.a | . + 2)` reported 9 instead of 7). #1005
            let left_pathed = std::cell::Cell::new(false);
            let result = eval_path(left, input.clone(), env, &mut |lp| {
                left_pathed.set(true);
                let mid = crate::runtime::rt_getpath(&input, &lp).unwrap_or(Value::Null);
                eval_path(right, mid.clone(), env, &mut |rp| {
                    let mut p = match &lp { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] };
                    if let Value::Arr(rpa) = &rp { p.extend(rpa.iter().cloned()); }
                    cb(Value::Arr(Rc::new(p)))
                })
            });
            match result {
                Err(e) => {
                    if let Some(pv) = crate::signal::take_path_result(&e) {
                        if left_pathed.get() {
                            // The signal originates from RIGHT on a navigated value;
                            // its payload is the already-computed result. Don't re-eval.
                            bail!("Invalid path expression with result {}", trunc_path_dump(&crate::value::value_to_json(&pv)));
                        }
                        match right.as_ref() {
                            Expr::Index { key, .. } | Expr::IndexOpt { key, .. } => {
                                let mut key_val = Value::Null;
                                let _ = eval(key, input, env, &mut |k| { key_val = k; Ok(true) });
                                let key_desc = match &key_val {
                                    Value::Num(n, _) => format!("element {} of", crate::value::format_jq_number(*n)),
                                    Value::Str(s) => format!("element \"{}\" of", s),
                                    _ => format!("element {} of", crate::value::value_to_json(&key_val)),
                                };
                                bail!("Invalid path expression near attempt to access {} {}", key_desc, trunc_path_dump(&crate::value::value_to_json(&pv)));
                            }
                            Expr::Each { .. } | Expr::EachOpt { .. } => {
                                bail!("Invalid path expression near attempt to iterate through {}", crate::value::value_to_json(&pv));
                            }
                            _ => {
                                // jq evaluates `A | B` in path context by running B
                                // on the VALUE A produced, even when A is a rootless
                                // (non-path) value. A discarding B (`empty`,
                                // `select(false)`, an `else empty` branch) yields
                                // nothing, so no invalid-path error reaches the sink:
                                // `path(last(empty))` = `… reduce empty … | if
                                // length>0 then .[0] else empty end` → []. A
                                // transforming B surfaces its own result/error. When
                                // B navigates the rootless value, jq reports the
                                // first hop (`near attempt to access element 0 of
                                // [1]`); an identity B leaves the value at the sink
                                // (`result <A>`). #839
                                {
                                    let v = pv;
                                    {
                                        let mut nav: Option<Value> = None;
                                        match eval_path(right, v.clone(), env, &mut |rp| {
                                            nav = Some(rp);
                                            Ok(false)
                                        }) {
                                            Err(re) => Err(re),
                                            Ok(_) => match &nav {
                                                None => Ok(true),
                                                Some(Value::Arr(comps)) if !comps.is_empty() => {
                                                    let key_desc = match &comps[0] {
                                                        Value::Num(n, _) => format!("element {} of", crate::value::format_jq_number(*n)),
                                                        Value::Str(s) => format!("element \"{}\" of", s),
                                                        other => format!("element {} of", crate::value::value_to_json(other)),
                                                    };
                                                    bail!("Invalid path expression near attempt to access {} {}", key_desc, trunc_path_dump(&crate::value::value_to_json(&v)))
                                                }
                                                _ => Err(e),
                                            },
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        Err(e)
                    }
                }
                other => other,
            }
        }
        Expr::Comma { left, right } => {
            let cont = eval_path(left, input.clone(), env, cb)?;
            if !cont { return Ok(false); }
            eval_path(right, input, env, cb)
        }
        Expr::Recurse { input_expr } => {
            // The 0-arg `recurse` / `..` desugar to `recurse(.[]?)` —
            // `EachOpt { Input }` — whose path closure is the full descent.
            // A custom step `recurse(f)` must instead follow `f` in path
            // mode at every level: jq navigates via `f` (not the default
            // descent) and propagates a per-step type error when `f` cannot
            // apply to a leaf (`1 | .a`). Ignoring the step both produced the
            // wrong paths and masked that error (#917).
            if matches!(input_expr.as_ref(), Expr::EachOpt { input_expr: ie } if matches!(ie.as_ref(), Expr::Input)) {
                eval_recurse_paths(&input, &Value::Arr(Rc::new(vec![])), cb)
            } else {
                let mut path: Vec<Value> = Vec::new();
                eval_recurse_step_paths(input_expr, &input, &mut path, env, cb)
            }
        }
        Expr::LetBinding { var_index, value, body } => {
            // Identity (`. as $x`, #837) and tracked-navigation destructuring
            // sub-bindings (`$src[k] as $a`, #880) both forward a path and are
            // cold relative to the common value-mode binding. Handle them in
            // never-inlined helpers so this hot arm stays compact — an inline
            // cold path here regressed a path-heavy bench (#839 / #880 note).
            if matches!(value.as_ref(), Expr::Input) {
                return eval_path_letbinding_identity(*var_index, body, input, env, cb);
            }
            if nav_source_var(value).is_some_and(is_nav_path_source) {
                if let Some(r) = eval_path_navbind(*var_index, value, body, &input, env, cb) {
                    return r;
                }
            }
            eval(value, input.clone(), env, &mut |val| {
                let old = env.borrow().get_var(*var_index);
                env.borrow_mut().set_var(*var_index, val);
                let result = eval_path(body, input.clone(), env, cb);
                env.borrow_mut().set_var(*var_index, old);
                result
            })
        }
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            eval(cond, input.clone(), env, &mut |cond_val| {
                if cond_val.is_truthy() {
                    eval_path(then_branch, input.clone(), env, cb)
                } else {
                    eval_path(else_branch, input.clone(), env, cb)
                }
            })
        }
        // `until(cond; update)` / `while(cond; update)` in path context. jq
        // desugars them to recursive if/pipe forms that all preserve paths:
        //   _until: if cond then . else (update | _until) end
        //   _while: if cond then ., (update | _while) else empty end
        // so a loop that survives by identity yields the path of the input
        // (e.g. `path(until(true; .a))` = []) rather than an "Invalid path
        // expression" error. Mirror the value-mode desugar, tracking the path
        // through `update` exactly as the Pipe arm composes sub-paths. #882.
        Expr::Until { cond, update } => {
            eval_path_until(cond, update, input.clone(), Vec::new(), env, cb)
        }
        Expr::While { cond, update } => {
            eval_path_while(cond, update, input.clone(), Vec::new(), env, cb)
        }
        Expr::Slice { expr: base_expr, from, to } => {
            // Slice bounds are generators: produce the Cartesian product of
            // path components, nested from (outer) → to (middle) → base (inner),
            // mirroring the value-context handler. #761
            let from_one = match from { None => Some(Value::Null), Some(f) => eval_one(f, &input, env).ok() };
            let to_one = match to { None => Some(Value::Null), Some(t) => eval_one(t, &input, env).ok() };
            if let (Some(fv), Some(tv)) = (&from_one, &to_one) {
                return eval_path(base_expr, input.clone(), env, &mut |bp| {
                    emit_slice_path(&input, &bp, fv, tv, cb)
                });
            }
            let from_vals: Vec<Value> = match (&from_one, from) {
                (Some(v), _) => vec![v.clone()],
                (None, Some(f)) => {
                    let mut vs = Vec::new();
                    eval(f, input.clone(), env, &mut |v| { vs.push(v); Ok(true) })?;
                    vs
                }
                (None, None) => vec![Value::Null],
            };
            if from_vals.is_empty() { return Ok(true); }
            let to_vals: Vec<Value> = match (&to_one, to) {
                (Some(v), _) => vec![v.clone()],
                (None, Some(t)) => {
                    let mut vs = Vec::new();
                    eval(t, input.clone(), env, &mut |v| { vs.push(v); Ok(true) })?;
                    vs
                }
                (None, None) => vec![Value::Null],
            };
            if to_vals.is_empty() { return Ok(true); }
            for fv in &from_vals {
                for tv in &to_vals {
                    if !eval_path(base_expr, input.clone(), env, &mut |bp| {
                        emit_slice_path(&input, &bp, fv, tv, cb)
                    })? {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        }
        Expr::CallBuiltin { name, args } if name == "getpath" && args.len() == 1 => {
            // In path context, getpath(p) yields the path p — but jq still
            // traverses the input along p and raises a type error if an
            // intermediate value cannot be indexed. `rt_getpath` performs that
            // exact check (lenient on missing keys, strict on type mismatch), so
            // validate before emitting the path. #775
            eval(&args[0], input.clone(), env, &mut |pv| {
                crate::runtime::rt_getpath(&input, &pv)?;
                cb(pv)
            })
        }
        Expr::GetPath { path } => {
            // In path context, getpath(p) = the path p itself, validated against
            // the input the same way as the CallBuiltin form above. #775
            eval(path, input.clone(), env, &mut |pv| {
                crate::runtime::rt_getpath(&input, &pv)?;
                cb(pv)
            })
        }
        Expr::FuncCall { func_id, args } => {
            let func = env.borrow().funcs.get(func_id.idx()).cloned();
            let f = match func {
                Some(f) => f,
                None => bail!("undefined function id {}", func_id),
            };
            // A path forwarded through a filter parameter (`def f(g): g; path(f(.a))`)
            // must keep its provenance: jq treats the closure argument as
            // path-transparent. Mirror the value-mode call by substituting the
            // argument expressions into the body before collecting paths, so a
            // parameter reference resolves to the path of the forwarded filter
            // rather than evaluating to a value (which bailed "Invalid path
            // expression with result null"). See #982.
            if f.param_vars.is_empty() || args.is_empty() {
                eval_path(&f.body, input, env, cb)
            } else if contains_func_call(&f.body, *func_id) {
                // Recursive body: rename its local bindings to fresh slots so
                // concurrently-active frames don't share var indices (matching
                // the value-mode recursive call path).
                let nv_before = env.borrow().next_var;
                let mut nv = nv_before;
                let body = substitute_and_rename(&f.body, &f.param_vars, args, &mut nv);
                env.borrow_mut().next_var = nv;
                let result = eval_path(&body, input, env, cb);
                env.borrow_mut().next_var = nv_before;
                result
            } else {
                let body = substitute_params(&f.body, &f.param_vars, args);
                eval_path(&body, input, env, cb)
            }
        }
        Expr::TryCatch { try_expr, catch_expr, restore_dot } => {
            let result = eval_path(try_expr, input.clone(), env, cb);
            match result {
                Ok(cont) => Ok(cont),
                Err(e) => {
                    let msg = format!("{}", e);
                    // halt / halt_error are non-recoverable: jq lets them
                    // propagate past `try ... catch` so the process exits with
                    // the requested code (#182).
                    if e.downcast_ref::<crate::signal::HaltSignal>().is_some() { return Err(e); }
                    // The `?//` desugar (`restore_dot`) keeps `.` set to the
                    // original input across fallbacks rather than binding the
                    // caught destructuring error: jq never exposes that error to
                    // the body, so a path expression in the body stays
                    // path-transparent — `path(. as [$a] ?// $a | .x)` is
                    // `["x"]` and `del(. as [$a] ?// $a | .x)` drops `.x` (#840).
                    if *restore_dot {
                        return eval_path(catch_expr, input.clone(), env, cb);
                    }
                    // Plain `try/catch`: the catch branch is itself a path
                    // expression, evaluated against the caught value (the error
                    // payload, or the rethrown input for a bare `error`). jq
                    // tracks `path(try error catch .b)` as `["b"]` and raises a
                    // path-mode type error when the caught value can't accept
                    // the navigation. Evaluating it in VALUE mode dropped the
                    // tracking (#836).
                    if let Some(be) = e.downcast_ref::<BreakError>() {
                        return eval_path_catch(catch_expr, break_catch_value(be.0), &input, env, cb);
                    }
                    // Recover a typed `error(value)` payload losslessly (#844).
                    if let Some(ev) = e.downcast_ref::<ErrorValue>() {
                        let v = take_error_payload(ev);
                        return eval_path_catch(catch_expr, v, &input, env, cb);
                    }
                    let catch_val = if let Some(json) = msg.strip_prefix("__jqerror__:") {
                        crate::value::json_to_value(json).unwrap_or(Value::from_str(&msg))
                    } else {
                        Value::from_str(&msg)
                    };
                    eval_path_catch(catch_expr, catch_val, &input, env, cb)
                }
            }
        }
        Expr::IndexOpt { expr: be, key: ke } => {
            // jq suppresses paths whose access would error: `path(.a?)` on
            // a non-object/non-null base emits no path. Mirror the type
            // check from the non-`?` Index path but skip silently on
            // mismatch instead of bailing. See #590, #594, #596.
            let input_for_check = input.clone();
            eval_path(be, input.clone(), env, &mut |bp| {
                eval(ke, input.clone(), env, &mut |key| {
                    let base_val = crate::runtime::rt_getpath(&input_for_check, &bp).unwrap_or(Value::Null);
                    match (&base_val, &key) {
                        (Value::Obj(_), Value::Str(_)) => {}
                        (Value::Arr(_), Value::Num(_, _)) => {}
                        (Value::Arr(_), Value::Arr(_)) => {}
                        (Value::Arr(_) | Value::Str(_), Value::Obj(_)) => {
                            if !is_valid_slice_object(&key) { return Ok(true); }
                        }
                        (Value::Null, Value::Str(_) | Value::Num(_, _) | Value::Obj(_)) => {}
                        _ => return Ok(true),
                    }
                    let mut p = match &bp { Value::Arr(a) => a.as_ref().clone(), _ => vec![] };
                    p.push(key); cb(Value::Arr(Rc::new(p)))
                })
            })
        }
        Expr::EachOpt { input_expr } => {
            eval_path(input_expr, input.clone(), env, &mut |bp| {
                let base = crate::runtime::rt_getpath(&input, &bp).unwrap_or(Value::Null);
                match &base {
                    Value::Arr(a) => { for i in 0..a.len() { let mut p = match &bp { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] }; p.push(Value::number(i as f64)); if !cb(Value::Arr(Rc::new(p)))? { return Ok(false); } } Ok(true) }
                    Value::Obj(ObjInner(o)) => { for k in o.keys() { let mut p = match &bp { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] }; p.push(Value::from_str(k)); if !cb(Value::Arr(Rc::new(p)))? { return Ok(false); } } Ok(true) }
                    _ => Ok(true),
                }
            })
        }
        Expr::Alternative { primary, fallback } => {
            // `path(A // B)` — emit paths from A whose values are truthy;
            // if none, fall through to paths from B.
            let mut any_truthy = false;
            let cont = eval_path(primary, input.clone(), env, &mut |bp| {
                let v = crate::runtime::rt_getpath(&input, &bp).unwrap_or(Value::Null);
                if v.is_truthy() {
                    any_truthy = true;
                    cb(bp)
                } else {
                    Ok(true)
                }
            })?;
            if !cont { return Ok(false); }
            if any_truthy { return Ok(true); }
            eval_path(fallback, input, env, cb)
        }
        Expr::LoadVar { var_index } => {
            // Inside a path-mode `foreach` (the `nth(n; g)` desugar), a bare
            // reference to the element variable forwards that element's source
            // path. Outside that context a variable is an ordinary value, so a
            // `null`/`true`/`false` that equals the input is the identity path
            // and anything else is an invalid path expression (mirrors the
            // catch-all). See #711.
            if let Some(p) = FOREACH_PATH_BIND.with(|s| {
                s.borrow().iter().rev().find(|(idx, _)| idx == var_index).map(|(_, p)| p.clone())
            }) {
                return cb(p);
            }
            let v = env.borrow().get_var(*var_index);
            // An identity-bound variable (`. as $x`) is the identity path `[]`
            // while the current input still equals its captured value (#837).
            // Literal `null`/`true`/`false` equal to the input are likewise
            // identity paths (#434). Any other value is an invalid path expr.
            if v == input && (is_identity_path_var(*var_index)
                || matches!(&v, Value::Null | Value::True | Value::False))
            {
                return cb(Value::Arr(Rc::new(vec![])));
            }
            return Err(crate::signal::PathResultSignal::raise(&v));
        }
        Expr::Label { var_index, body } => {
            // Forward the body's paths; a matching `break` ends the label
            // cleanly (mirrors the value-mode handler). #711.
            let label_id = {
                let mut e = env.borrow_mut();
                let id = e.next_label;
                e.next_label = id + 1;
                e.set_var(*var_index, Value::number(id as f64));
                id
            };
            match eval_path(body, input, env, cb) {
                Err(e) => {
                    if let Some(be) = e.downcast_ref::<BreakError>() {
                        if be.0 == label_id { return Ok(true); }
                    }
                    Err(e)
                }
                other => other,
            }
        }
        Expr::Break { var_index, .. } => {
            let label = env.borrow().get_var(*var_index);
            if let Value::Num(n, NumRepr(None)) = &label {
                return Err(BreakError(*n as u64).into());
            }
            bail!("break: invalid label")
        }
        Expr::Limit { count, generator } => {
            // Forward the generator's paths, stopping after the same count as
            // value context (ceil(n) for positive n). Covers `first(p)` =
            // `limit(1; p)` and the `nth` desugar's `limit(1; foreach …)`. #711.
            let mut stopped_by_outer = false;
            let result = eval(count, input.clone(), env, &mut |cv| {
                let n = match &cv {
                    Value::Num(n, _) => *n,
                    Value::Null | Value::True | Value::False => {
                        bail!("__jqerror__:\"limit doesn't support negative count\"");
                    }
                    // A string/array/object count surfaces jq's `$n - 1`
                    // arithmetic error lazily — only when the generator yields
                    // its first path — so an empty generator produces nothing
                    // rather than erroring (#806).
                    other => {
                        let msg = format!(
                            "{} and number (1) cannot be subtracted",
                            crate::runtime::errdesc_pub(other),
                        );
                        let err = format!(
                            "__jqerror__:{}",
                            crate::value::value_to_json_precise(&Value::from_string(msg)),
                        );
                        let mut yielded = false;
                        eval_path(generator, input.clone(), env, &mut |_p| {
                            yielded = true;
                            Ok(false)
                        })?;
                        if yielded {
                            bail!("{}", err);
                        }
                        return Ok(true);
                    }
                };
                if n < 0.0 || n.is_nan() {
                    bail!("__jqerror__:\"limit doesn't support negative count\"");
                }
                if n == 0.0 { return Ok(true); }
                let mut emitted: i64 = 0;
                let inner = eval_path(generator, input.clone(), env, &mut |p| {
                    emitted += 1;
                    let cont = cb(p)?;
                    if !cont {
                        stopped_by_outer = true;
                        Ok(false)
                    } else if emitted as f64 >= n {
                        Ok(false)
                    } else {
                        Ok(true)
                    }
                })?;
                Ok(inner && !stopped_by_outer)
            });
            match result {
                Ok(_) if stopped_by_outer => Ok(false),
                other => other,
            }
        }
        Expr::Foreach { source, init, var_index, acc_index, update, extract } => {
            // Forward the path of selected source elements. The source is
            // evaluated in PATH mode so each element's path is available; the
            // element var is bound to the element VALUE (for use in
            // update/cond) and registered in FOREACH_PATH_BIND so a bare `$x`
            // in the extract forwards the element path — this is what makes the
            // `nth(n; .[])` desugar path-transparent. #711.
            let vi = *var_index;
            let ai = *acc_index;
            { let mut e = env.borrow_mut(); e.ensure_var(vi); e.ensure_var(ai); }

            // Decide whether SOURCE navigates the tracked input (a path
            // generator like `.[]`) or is a plain value generator (`range`,
            // literals, `empty` — the `nth(n; range)` / #839 shape). A
            // navigating source forwards the element path through `$x` (handled
            // by the block below); a value generator instead threads the
            // ACCUMULATOR as the path carrier. Only probe when the source reads
            // `.` (otherwise it cannot navigate), mirroring the reduce probe so
            // a stream-consuming `input` is not evaluated twice. #839
            let source_navigates = if expr_uses_outer_input(source) {
                let mut nav = false;
                match eval_path(source, input.clone(), env, &mut |_p| { nav = true; Ok(false) }) {
                    Ok(_) => {}
                    Err(e) => {
                        if !crate::signal::is_path_result(&e) { return Err(e); }
                    }
                }
                nav
            } else {
                false
            };

            if !source_navigates {
                // Value-generator source: the accumulator carries the path.
                // Kept out-of-line so the (cold) path-mode foreach machinery
                // does not bloat eval_path's hot index/reduce arms. #839
                return eval_foreach_valuegen_path(source, init, vi, ai, update, extract, &input, env, cb);
            }

            // ---- navigating source: forward the element path through `$x` ----
            // When INIT itself navigates the input (a non-empty accumulator
            // path) AND the source also navigates, jq cannot reconcile the two
            // and reports the source's first hop — `del(foreach .k[] as $x
            // (.a; .; .))` errors instead of deleting the whole document (the
            // old code forwarded the INIT path and corrupted/deleted). Mirrors
            // the reduce rule. #915
            if expr_uses_outer_input(init) {
                let mut init_navigated = false;
                match eval_path(init, input.clone(), env, &mut |p| {
                    if matches!(&p, Value::Arr(a) if !a.is_empty()) { init_navigated = true; }
                    Ok(true)
                }) {
                    Ok(_) => {}
                    Err(e) => { if !crate::signal::is_path_result(&e) { return Err(e); } }
                }
                if init_navigated {
                    return Err(source_nav_error(source, &input, env));
                }
            }
            if extract.is_none() {
                // 2-arg `foreach .[] as $x (init; update)` yields UPDATE each
                // step, so UPDATE's path is the output (a bare `$x`/`$x.k`
                // forwards the element spine). Cold path kept out-of-line. #880
                return eval_foreach_nav_noextract_path(source, init, vi, ai, update, &input, env, cb);
            }
            eval(init, input.clone(), env, &mut |init_val| {
                let mut acc = init_val;
                eval_path(source, input.clone(), env, &mut |src_path| {
                    let elem_val = crate::runtime::rt_getpath(&input, &src_path).unwrap_or(Value::Null);
                    let acc_val = std::mem::replace(&mut acc, Value::Null);
                    let (old_var, old_acc) = {
                        let mut e = env.borrow_mut();
                        let ov = std::mem::replace(&mut e.vars[vi.idx()], elem_val);
                        let oa = std::mem::replace(&mut e.vars[ai.idx()], acc_val.clone());
                        (ov, oa)
                    };
                    let mut stopped = false;
                    let update_result = eval(update, acc_val, env, &mut |new_acc| {
                        acc = new_acc.clone();
                        env.borrow_mut().vars[ai.idx()] = new_acc.clone();
                        let extract_expr = extract.as_ref().unwrap();
                        FOREACH_PATH_BIND.with(|s| s.borrow_mut().push((vi, src_path.clone())));
                        // The navigating source owns the path register, so the
                        // accumulator EXTRACT runs against is rootless: a bare
                        // `.`/`.k` surfaces "result <acc>"/"near attempt …", a
                        // `$x` still forwards the element path. #915
                        let r = {
                            let _rootless = RootlessAccGuard::enter(input.clone());
                            eval_path(extract_expr, new_acc.clone(), env, cb)
                        };
                        FOREACH_PATH_BIND.with(|s| { s.borrow_mut().pop(); });
                        let cont = r?;
                        if !cont { stopped = true; }
                        Ok(cont)
                    });
                    {
                        let mut e = env.borrow_mut();
                        e.vars[ai.idx()] = old_acc;
                        e.vars[vi.idx()] = old_var;
                    }
                    update_result?;
                    Ok(!stopped)
                })
            })
        }
        Expr::Reduce { source, init, var_index, acc_index, update } => {
            // The accumulator threads as a PATH: INIT seeds the path(s) and
            // each source element runs UPDATE in path mode relative to the
            // current accumulator path. With an empty source the INIT path is
            // forwarded unchanged: `path(reduce empty as $x (.a; .))` → ["a"].
            // #711.
            let vi = *var_index;
            let ai = *acc_index;
            { let mut e = env.borrow_mut(); e.ensure_var(vi); e.ensure_var(ai); }
            let mut acc_paths: Vec<Value> = Vec::new();
            if let Err(e) = eval_path(init, input.clone(), env, &mut |p| { acc_paths.push(p); Ok(true) }) {
                if crate::signal::is_path_result(&e) {
                    // A non-path INIT means the reduce isn't path-trackable: jq
                    // treats it as a value computation whose (rootless) result
                    // flows to the sink. Compute that value and defer it as the
                    // sentinel so a downstream discard (`| if … else empty`,
                    // `path(last(empty))`) swallows it and a navigation/assignment
                    // surfaces the real result rather than the INIT value. #839
                    let reduced = Expr::Reduce {
                        source: source.clone(),
                        init: init.clone(),
                        var_index: *var_index,
                        acc_index: *acc_index,
                        update: update.clone(),
                    };
                    let mut last_val = Value::Null;
                    let mut has = false;
                    eval(&reduced, input, env, &mut |v| { last_val = v; has = true; Ok(true) })?;
                    if has {
                        return Err(crate::signal::PathResultSignal::raise(&last_val));
                    }
                    return Ok(true);
                }
                return Err(e);
            }
            // A reduce SOURCE that navigates the tracked input is rejected only
            // when the accumulator path is already non-empty (INIT navigated):
            // jq cannot reconcile the source navigation with the seeded path, so
            // it reports the source's first hop (`reduce .[] as $x (.a; .)` ->
            // "iterate through …"). When INIT is identity (path []), the source
            // is a pure iteration generator and contributes no provenance — the
            // UPDATE/body is what jq path-tracks, so `path(reduce .[] as $x
            // (.; .))` is `[]`, not an error. The previous code rejected *all*
            // navigating sources, inverting the rule (#915, regressed the #838
            // guard direction). Probe in path mode only when the source reads
            // `.`: a value generator surfaces as a PathResultSignal
            // sentinel (→ swallow), other errors propagate, and a source that
            // doesn't read `.` is never probed (so a stream `input` is not
            // evaluated twice).
            let init_navigated = acc_paths.iter().any(|p| matches!(p, Value::Arr(a) if !a.is_empty()));
            if init_navigated && expr_uses_outer_input(source) {
                let mut navigated = false;
                match eval_path(source, input.clone(), env, &mut |_p| { navigated = true; Ok(false) }) {
                    Ok(_) => {}
                    Err(e) => {
                        if !crate::signal::is_path_result(&e) { return Err(e); }
                    }
                }
                if navigated {
                    return Err(source_nav_error(source, &input, env));
                }
            }
            let mut source_vals: Vec<Value> = Vec::new();
            eval(source, input.clone(), env, &mut |v| { source_vals.push(v); Ok(true) })?;
            for sv in source_vals {
                let (old_var, old_acc) = {
                    let mut e = env.borrow_mut();
                    let ov = std::mem::replace(&mut e.vars[vi.idx()], sv);
                    (ov, e.vars[ai.idx()].clone())
                };
                let mut next: Vec<Value> = Vec::new();
                for ap in &acc_paths {
                    let base_val = crate::runtime::rt_getpath(&input, ap).unwrap_or(Value::Null);
                    env.borrow_mut().vars[ai.idx()] = base_val.clone();
                    let ap_vec: Vec<Value> = match ap { Value::Arr(a) => a.as_ref().clone(), _ => vec![] };
                    eval_path(update, base_val.clone(), env, &mut |rp| {
                        // jq only path-tracks a reduce whose UPDATE preserves the
                        // accumulator path — an identity-equivalent body (`.`,
                        // `select(true)`, `if true then . else . end`) yields the
                        // base unchanged (relative path `[]`). A body that
                        // navigates (`.a`, `.[$x]`, …) makes the reduce a value
                        // computation, which jq rejects as a path expression
                        // rather than tracking through it — otherwise `reduce 1
                        // as $x (.; .a) |= 99` silently mutated `.a` (#816). The
                        // reported result is the value at the navigated location.
                        let nav_len = match &rp { Value::Arr(a) => a.len(), _ => 0 };
                        if nav_len != 0 {
                            let nav_val = crate::runtime::rt_getpath(&base_val, &rp).unwrap_or(Value::Null);
                            bail!(
                                "Invalid path expression with result {}",
                                trunc_path_dump(&crate::value::value_to_json(&nav_val))
                            );
                        }
                        next.push(Value::Arr(Rc::new(ap_vec.clone())));
                        Ok(true)
                    })?;
                }
                {
                    let mut e = env.borrow_mut();
                    e.vars[ai.idx()] = old_acc;
                    e.vars[vi.idx()] = old_var;
                }
                acc_paths = next;
            }
            for p in acc_paths {
                if !cb(p)? { return Ok(false); }
            }
            Ok(true)
        }
        // `any`/`all` carry the path of the deciding element. The 1-/2-arg
        // forms parse to AnyShort/AllShort; the 0-arg `any`/`all` to
        // `UnaryOp::{Any,All}` over an operand (≡ `any(operand[]; .)`). #927
        Expr::AnyShort { generator, predicate } => {
            eval_any_all_path(generator, predicate, true, input, env, cb)
        }
        Expr::AllShort { generator, predicate } => {
            eval_any_all_path(generator, predicate, false, input, env, cb)
        }
        Expr::UnaryOp { op: crate::ir::UnaryOp::Any, operand } => {
            let gen = Expr::Each { input_expr: operand.clone() };
            eval_any_all_path(&gen, &Expr::Input, true, input, env, cb)
        }
        Expr::UnaryOp { op: crate::ir::UnaryOp::All, operand } => {
            let gen = Expr::Each { input_expr: operand.clone() };
            eval_any_all_path(&gen, &Expr::Input, false, input, env, cb)
        }
        // The `*str` trim builtins are jq-defined slice expressions, so they are
        // path-transparent: lower to that definition and reuse the slice-path
        // machinery (no-match → identity `[]`, match → slice path). #962
        Expr::CallBuiltin { name, args }
            if args.len() == 1 && matches!(name.as_str(), "ltrimstr" | "rtrimstr" | "trimstr") =>
        {
            let lowered = lower_trimstr_path(name, &args[0]);
            eval_path(&lowered, input, env, cb)
        }
        // The whitespace trim builtins (`trim`/`ltrim`/`rtrim`) are value-
        // producing C builtins: jq keeps the path only when the result is the
        // input unchanged (identity path `[]`); any actual trim has no path and
        // surfaces "Invalid path expression with result <trimmed>". #962
        Expr::UnaryOp { op, operand }
            if matches!(op, crate::ir::UnaryOp::Trim | crate::ir::UnaryOp::Ltrim | crate::ir::UnaryOp::Rtrim) =>
        {
            let root = input.clone();
            eval_path(operand, input, env, &mut |base_path| {
                let base_val = crate::runtime::rt_getpath(&root, &base_path).unwrap_or(Value::Null);
                let trimmed = eval_unaryop(*op, &base_val)?;
                if trimmed == base_val {
                    cb(base_path)
                } else {
                    Err(crate::signal::PathResultSignal::raise(&trimmed))
                }
            })
        }
        // `debug`/`stderr` are path-transparent: they emit their side-effect
        // line on stderr but forward the value and the incoming path unchanged
        // (identity path `[]`), so `path()`/`=`/`|=`/`del` through a debugging
        // insertion keep working. Mirror the value arms above but yield the
        // identity path instead of the value. #997
        Expr::Debug { expr: de } => {
            eval(de, input.clone(), env, &mut |val| {
                eprintln!("[\"DEBUG:\",{}]", crate::value::value_to_json_tojson(&val));
                cb(Value::Arr(Rc::new(vec![])))
            })
        }
        Expr::Stderr { expr: se } => {
            eval(se, input.clone(), env, &mut |val| {
                match &val {
                    Value::Str(s) => eprint!("{}", s.as_str()),
                    _ => eprint!("{}", crate::value::value_to_json_tojson(&val)),
                }
                cb(Value::Arr(Rc::new(vec![])))
            })
        }
        _ => {
            // Non-path-safe expression: evaluate, then accept the value as
            // the empty path `[]` if it is one of `null`/`true`/`false` and
            // equals the input. jq treats those three literals as identity
            // path expressions when their result matches the current input
            // (so `path(.a // null)` and `path(if .x then .y else null end)`
            // work on falsy branches that produce a literal value). All
            // other shapes still report the original "Invalid path
            // expression" error. See #434.
            let input_for_check = input.clone();
            let mut result_val = Value::Null;
            let mut has_result = false;
            eval(expr, input, env, &mut |val| {
                result_val = val;
                has_result = true;
                Ok(true)
            })?;
            if has_result {
                let is_id_value = matches!(&result_val, Value::Null | Value::True | Value::False);
                if is_id_value && result_val == input_for_check {
                    return cb(Value::Arr(Rc::new(vec![])));
                }
                return Err(crate::signal::PathResultSignal::raise(&result_val));
            }
            Ok(true)
        }
    }
}

/// Path-mode `any`/`all`. jq's `any(g; c)` / `all(g; c)` carry the path of the
/// *deciding* element: `path(any)` yields the path of the first element whose
/// condition is truthy (`all`: first falsy). The result *value* is always the
/// boolean `true` (any) / `false` (all); jq's `or`/`and` only keep that
/// boolean's path provenance when the condition output is *exactly* that
/// boolean reached by navigation (`.` on a `true` element). A truthy-but-
/// non-boolean (`.` on `5`), a computed boolean (`.>0`), or a value-generator
/// source (`range`) all produce a rootless boolean, so jq reports "Invalid
/// path expression with result <bool>". A fall-through (`any` finds nothing /
/// `all` no counterexample) yields the init boolean (`false` / `true`),
/// likewise rootless unless it equals the input. #927
fn eval_any_all_path(
    generator: &Expr,
    predicate: &Expr,
    is_any: bool,
    input: Value,
    env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    let decide = |v: &Value| -> bool { if is_any { v.is_truthy() } else { !v.is_truthy() } };
    let decision_bool = if is_any { Value::True } else { Value::False };
    let fallthrough_bool = if is_any { Value::False } else { Value::True };

    // A rootless boolean B is an identity path only when it equals the input
    // (mirrors the `_ =>` arm's null/true/false rule); otherwise it is the
    // classic "Invalid path expression with result B".
    let emit_bool = |b: Value, cb: &mut dyn FnMut(Value) -> GenResult| -> GenResult {
        if b == input {
            cb(Value::Arr(Rc::new(vec![])))
        } else {
            return Err(crate::signal::PathResultSignal::raise(&b));
        }
    };

    // Does the generator navigate the tracked input (path-trackable), or is it
    // a plain value generator (`range`, literals)? Probe in path mode only when
    // it reads `.`, mirroring the reduce/foreach source probes.
    let gen_navigates = if expr_uses_outer_input(generator) {
        let mut nav = false;
        match eval_path(generator, input.clone(), env, &mut |_p| { nav = true; Ok(false) }) {
            Ok(_) => {}
            Err(e) => { if !crate::signal::is_path_result(&e) { return Err(e); } }
        }
        nav
    } else {
        false
    };

    if gen_navigates {
        let mut emit: Option<Vec<Value>> = None;
        let mut rootless = false;
        let gen_result = eval_path(generator, input.clone(), env, &mut |gp| {
            let elem = crate::runtime::rt_getpath(&input, &gp).unwrap_or(Value::Null);
            // Decide via the predicate's *value*; remember which output decided.
            let mut decided: Option<(usize, Value)> = None;
            let mut j = 0usize;
            eval(predicate, elem.clone(), env, &mut |v| {
                if decide(&v) { decided = Some((j, v)); Ok(false) } else { j += 1; Ok(true) }
            })?;
            let Some((target, dv)) = decided else { return Ok(true); };
            // A deciding output that is not exactly the boolean is rootless.
            if dv != decision_bool {
                rootless = true;
                return Ok(false);
            }
            // Emit (generator ++ predicate) path of the deciding output. The
            // predicate path-eval surfaces a rootless value as a sentinel.
            let gp_vec: Vec<Value> = match &gp { Value::Arr(a) => a.as_ref().clone(), _ => vec![] };
            let mut cur = 0usize;
            let pr = eval_path(predicate, elem.clone(), env, &mut |sp| {
                if cur == target {
                    let mut full = gp_vec.clone();
                    if let Value::Arr(a) = &sp { full.extend(a.iter().cloned()); }
                    emit = Some(full);
                    Ok(false)
                } else {
                    cur += 1;
                    Ok(true)
                }
            });
            if let Err(e) = pr {
                if crate::signal::is_path_result(&e) { rootless = true; }
                else { return Err(e); }
            }
            Ok(false)
        });
        gen_result?;
        if rootless {
            return emit_bool(decision_bool, cb);
        }
        if let Some(p) = emit {
            return cb(Value::Arr(Rc::new(p)));
        }
        return emit_bool(fallthrough_bool, cb);
    }

    // Value-generator source: the deciding element has no input path, so the
    // boolean result is rootless.
    let mut decided = false;
    let gr = eval(generator, input.clone(), env, &mut |gv| {
        let mut d = false;
        eval(predicate, gv, env, &mut |v| {
            if decide(&v) { d = true; Ok(false) } else { Ok(true) }
        })?;
        if d { decided = true; Ok(false) } else { Ok(true) }
    });
    gr?;
    let result_bool = if decided { decision_bool } else { fallthrough_bool };
    emit_bool(result_bool, cb)
}

/// Evaluate a `try … catch` body in path context against the caught value.
/// jq gives the caught value a path only when it is the try's own input (a
/// bare `error` / `error(.)` re-raising `.`); an `error(LITERAL)` payload is
/// rootless, so a body that navigates it reports the first hop, an identity
/// body leaves the value-shaped "result <payload>" at the sink, and a
/// discarding body (`catch empty`) yields nothing. `path(try error("E")
/// catch .)` errors with `result "E"` where jq-jit used to emit `[]`. The
/// value-provenance distinction jq draws between `error(5)` and `error(.)`
/// on an identical input is approximated here by value equality. Extends the
/// try/catch path tracking of #836.
fn eval_path_catch(
    catch_expr: &Expr,
    payload: Value,
    input: &Value,
    env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    if &payload == input {
        // The caught value sits at the current path position (re-raised `.`),
        // so the body tracks normally.
        return eval_path(catch_expr, payload, env, cb);
    }
    let mut nav: Option<Value> = None;
    match eval_path(catch_expr, payload.clone(), env, &mut |rp| { nav = Some(rp); Ok(false) }) {
        Err(e) => Err(e),
        Ok(_) => match &nav {
            None => Ok(true),
            Some(Value::Arr(comps)) if !comps.is_empty() => {
                let key_desc = match &comps[0] {
                    Value::Num(n, _) => format!("element {} of", crate::value::format_jq_number(*n)),
                    Value::Str(s) => format!("element \"{}\" of", s),
                    other => format!("element {} of", crate::value::value_to_json(other)),
                };
                bail!("Invalid path expression near attempt to access {} {}", key_desc, trunc_path_dump(&crate::value::value_to_json(&payload)))
            }
            _ => Err(crate::signal::PathResultSignal::raise(&payload)),
        },
    }
}

/// The root variable of a pure navigation chain (`$v`, `$v[k]`, `$v.k`, the
/// `$v | .[k]` desugar of a computed object-pattern key). Returns `None` for any
/// expression that does not bottom out in a single `LoadVar`. Used to decide
/// whether a destructuring sub-binding navigates a path-tracked source. #880
fn nav_source_var(expr: &Expr) -> Option<VarIdx> {
    match expr {
        Expr::LoadVar { var_index } => Some(*var_index),
        Expr::Index { expr, .. } | Expr::IndexOpt { expr, .. } | Expr::Slice { expr, .. } => {
            nav_source_var(expr)
        }
        Expr::Pipe { left, .. } => nav_source_var(left),
        _ => None,
    }
}

/// Whether `target` is referenced two or more times in `expr` (early-exit
/// counter). A single-spine destructuring source is referenced exactly once
/// (in its sole navigated child); two or more references mean sibling
/// navigations, which jq cannot path-track. Mirrors `expr_uses_var`'s shape but
/// counts. #880
fn var_referenced_twice(expr: &Expr, target: VarIdx) -> bool {
    fn go(expr: &Expr, target: VarIdx, count: &mut u8) {
        if *count >= 2 { return; }
        match expr {
            Expr::LoadVar { var_index } => { if *var_index == target { *count += 1; } }
            Expr::Pipe { left, right } | Expr::Comma { left, right }
            | Expr::BinOp { lhs: left, rhs: right, .. }
            | Expr::Alternative { primary: left, fallback: right }
            | Expr::While { cond: left, update: right }
            | Expr::Until { cond: left, update: right }
            | Expr::Limit { count: left, generator: right }
            | Expr::Index { expr: left, key: right }
            | Expr::IndexOpt { expr: left, key: right }
            | Expr::Update { path_expr: left, update_expr: right }
            | Expr::Assign { path_expr: left, value_expr: right }
            | Expr::SetPath { path: left, value: right }
            | Expr::TryCatch { try_expr: left, catch_expr: right, .. } => {
                go(left, target, count); go(right, target, count);
            }
            Expr::Mutate { path_expr, value_expr, .. } => {
                go(path_expr, target, count); go(value_expr, target, count);
            }
            Expr::IfThenElse { cond, then_branch, else_branch } => {
                go(cond, target, count); go(then_branch, target, count); go(else_branch, target, count);
            }
            Expr::LetBinding { value, body, .. } => {
                go(value, target, count); go(body, target, count);
            }
            Expr::Each { input_expr } | Expr::EachOpt { input_expr }
            | Expr::Recurse { input_expr } | Expr::Repeat { update: input_expr }
            | Expr::Negate { operand: input_expr } | Expr::UnaryOp { operand: input_expr, .. }
            | Expr::Collect { generator: input_expr }
            | Expr::PathExpr { expr: input_expr } | Expr::GetPath { path: input_expr }
            | Expr::DelPaths { paths: input_expr } | Expr::Debug { expr: input_expr }
            | Expr::Stderr { expr: input_expr } | Expr::Format { expr: input_expr, .. } => {
                go(input_expr, target, count);
            }
            Expr::Reduce { source, init, update, .. }
            | Expr::Foreach { source, init, update, .. } => {
                go(source, target, count); go(init, target, count); go(update, target, count);
            }
            Expr::Range { from, to, step } => {
                go(from, target, count); go(to, target, count);
                if let Some(s) = step { go(s, target, count); }
            }
            Expr::FuncCall { args, .. } | Expr::CallBuiltin { args, .. } => {
                for a in args { go(a, target, count); }
            }
            Expr::ObjectConstruct { pairs } => {
                for (k, v) in pairs { go(k, target, count); go(v, target, count); }
            }
            Expr::StringInterpolation { parts } => {
                for p in parts { if let StringPart::Expr(e) = p { go(e, target, count); } }
            }
            Expr::AllShort { generator, predicate } | Expr::AnyShort { generator, predicate } => {
                go(generator, target, count); go(predicate, target, count);
            }
            Expr::Label { body, .. } | Expr::Break { value: body, .. } => go(body, target, count),
            Expr::Error { msg } => { if let Some(m) = msg { go(m, target, count); } }
            Expr::ClosureOp { input_expr, key_expr, .. } => {
                go(input_expr, target, count); go(key_expr, target, count);
            }
            Expr::Slice { expr, from, to } => {
                go(expr, target, count);
                if let Some(f) = from { go(f, target, count); }
                if let Some(t) = to { go(t, target, count); }
            }
            Expr::RegexTest { input_expr, re, flags } | Expr::RegexMatch { input_expr, re, flags }
            | Expr::RegexCapture { input_expr, re, flags } | Expr::RegexScan { input_expr, re, flags } => {
                go(input_expr, target, count); go(re, target, count); go(flags, target, count);
            }
            Expr::RegexSub { input_expr, re, tostr, flags } | Expr::RegexGsub { input_expr, re, tostr, flags } => {
                go(input_expr, target, count); go(re, target, count); go(tostr, target, count); go(flags, target, count);
            }
            Expr::AlternativeDestructure { alternatives } => {
                for a in alternatives { go(a, target, count); }
            }
            Expr::Memoize { key, body, .. } => {
                if let Some(k) = key { go(k, target, count); }
                go(body, target, count);
            }
            _ => {}
        }
    }
    let mut count = 0u8;
    go(expr, target, &mut count);
    count >= 2
}

/// Path-mode `. as $x` binding: registers the identity-path provenance (#837)
/// and — when `$x` is referenced just once, a single navigation spine — also
/// registers it as a navigation source so a destructuring child `$x[k] as $a`
/// can inherit a path (#880). Out-of-line so eval_path's hot arms stay compact.
#[inline(never)]
fn eval_path_letbinding_identity(
    var_index: VarIdx,
    body: &Expr,
    input: Value,
    env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    let nav_source = !var_referenced_twice(body, var_index);
    let old = env.borrow().get_var(var_index);
    env.borrow_mut().set_var(var_index, input.clone());
    let _id_guard = push_identity_path_var(var_index);
    let _nav_guard = if nav_source { Some(push_nav_path_source(var_index)) } else { None };
    let result = eval_path(body, input, env, cb);
    drop(_nav_guard);
    env.borrow_mut().set_var(var_index, old);
    result
}

/// Path-mode binding of a destructuring sub-variable to a navigated sub-value
/// (`$src[k] as $new`, the desugar of `. as [$new]` / `. as {k:$new}`). The
/// navigation is evaluated in PATH mode so `$new` inherits the spine path; that
/// path is registered in `FOREACH_PATH_BIND` (consulted by `LoadVar`) for the
/// body. Returns `None` when the navigation is not a path (rootless source) so
/// the caller falls back to the ordinary value-mode binding. Kept `#[inline(never)]`
/// to keep eval_path's hot arms compact (see #839 perf note on #880). #880
#[inline(never)]
fn eval_path_navbind(
    var_index: VarIdx,
    value: &Expr,
    body: &Expr,
    input: &Value,
    env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> Option<GenResult> {
    let mut paths: Vec<Value> = Vec::new();
    match eval_path(value, input.clone(), env, &mut |p| { paths.push(p); Ok(true) }) {
        Ok(_) => {}
        Err(e) => {
            // A rootless source (non-path navigation) bails with the sentinel —
            // not a real error; let the caller bind it the ordinary way.
            if crate::signal::is_path_result(&e) {
                return None;
            }
            return Some(Err(e));
        }
    }
    // The newly bound var is itself a navigation source only if it branches into
    // a single sub-spine in the remaining body (the same single-spine rule that
    // gates the identity source). #880
    let new_is_source = !var_referenced_twice(body, var_index);
    for p in paths {
        let val = crate::runtime::rt_getpath(input, &p).unwrap_or(Value::Null);
        let old = env.borrow().get_var(var_index);
        env.borrow_mut().set_var(var_index, val);
        FOREACH_PATH_BIND.with(|s| s.borrow_mut().push((var_index, p)));
        let _nav_guard = if new_is_source { Some(push_nav_path_source(var_index)) } else { None };
        let result = eval_path(body, input.clone(), env, cb);
        drop(_nav_guard);
        FOREACH_PATH_BIND.with(|s| { s.borrow_mut().pop(); });
        env.borrow_mut().set_var(var_index, old);
        match result {
            Ok(true) => {}
            other => return Some(other),
        }
    }
    Some(Ok(true))
}

/// Path-context 2-arg `foreach` over a *navigating* source (`foreach .[] as $x
/// (init; update)`). The form yields UPDATE each step, so UPDATE — evaluated in
/// PATH mode with `$x` carrying the element spine — is the output path, and its
/// value threads to the next element. Kept `#[inline(never)]` so eval_path's hot
/// arms stay compact (#880 perf note). #880
#[inline(never)]
fn eval_foreach_nav_noextract_path(
    source: &Expr,
    init: &Expr,
    vi: VarIdx,
    ai: VarIdx,
    update: &Expr,
    input: &Value,
    env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    eval(init, input.clone(), env, &mut |init_val| {
        let mut acc = init_val;
        eval_path(source, input.clone(), env, &mut |src_path| {
            let elem_val = crate::runtime::rt_getpath(input, &src_path).unwrap_or(Value::Null);
            let acc_val = std::mem::replace(&mut acc, Value::Null);
            let (old_var, old_acc) = {
                let mut e = env.borrow_mut();
                let ov = std::mem::replace(&mut e.vars[vi.idx()], elem_val);
                let oa = std::mem::replace(&mut e.vars[ai.idx()], acc_val.clone());
                (ov, oa)
            };
            FOREACH_PATH_BIND.with(|s| s.borrow_mut().push((vi, src_path.clone())));
            // A 2-arg `foreach .[] as $x (init; update)` yields UPDATE each
            // step; with a navigating source the accumulator is rootless, so an
            // accumulator-anchored UPDATE (`.`, `.k`) is an invalid path
            // expression while a `$x`-anchored one forwards the element path. #915
            let r = {
                let _rootless = RootlessAccGuard::enter(input.clone());
                eval_path(update, acc_val, env, &mut |upd_path| {
                    // The update result threads as the next accumulator value.
                    acc = crate::runtime::rt_getpath(input, &upd_path).unwrap_or(Value::Null);
                    env.borrow_mut().vars[ai.idx()] = acc.clone();
                    cb(upd_path)
                })
            };
            FOREACH_PATH_BIND.with(|s| { s.borrow_mut().pop(); });
            {
                let mut e = env.borrow_mut();
                e.vars[ai.idx()] = old_acc;
                e.vars[vi.idx()] = old_var;
            }
            r
        })
    })
}

/// Path-context `foreach` over a value-generator source (`range`, literals,
/// `empty`): the accumulator — not the source — carries the path. Held in a
/// separate, never-inlined function so the cold machinery does not bloat
/// `eval_path`'s hot index/reduce arms. #839
#[inline(never)]
fn eval_foreach_valuegen_path(
    source: &Expr,
    init: &Expr,
    vi: VarIdx,
    ai: VarIdx,
    update: &Expr,
    extract: &Option<Box<Expr>>,
    input: &Value,
    env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    let source_vals = {
        let mut v = Vec::new();
        eval(source, input.clone(), env, &mut |x| { v.push(x); Ok(true) })?;
        v
    };
    // Classify INIT: a path-valued seed threads the accumulator as PATH(s) so
    // a path-preserving body (`.`) or a navigating body (`.b`) extends it
    // (`path(foreach range(2) as $i (.a;.;.))` → ["a"],["a"]); a rootless seed
    // threads it as a VALUE, the `nth(n; range)` counter shape
    // (`path(nth(1; range(5)))` → result 1).
    let mut init_paths: Vec<Value> = Vec::new();
    let init_res = eval_path(init, input.clone(), env, &mut |p| { init_paths.push(p); Ok(true) });
    match init_res {
        Ok(_) => {
            // PATH-threaded accumulator.
            for seed in init_paths {
                let mut acc_paths: Vec<Value> = vec![seed];
                for sv in &source_vals {
                    let old_var = { let mut e = env.borrow_mut(); std::mem::replace(&mut e.vars[vi.idx()], sv.clone()) };
                    let mut next: Vec<Value> = Vec::new();
                    let mut stop = false;
                    for ap in &acc_paths {
                        let base = crate::runtime::rt_getpath(input, ap).unwrap_or(Value::Null);
                        let ap_vec: Vec<Value> = match ap { Value::Arr(a) => a.as_ref().clone(), _ => vec![] };
                        let old_acc = { let mut e = env.borrow_mut(); std::mem::replace(&mut e.vars[ai.idx()], base.clone()) };
                        // UPDATE in path mode: each output is relative to the
                        // accumulator value and extends its path.
                        let r = eval_path(update, base.clone(), env, &mut |rp| {
                            let mut np_vec = ap_vec.clone();
                            if let Value::Arr(a) = &rp { np_vec.extend(a.iter().cloned()); }
                            let np = Value::Arr(Rc::new(np_vec.clone()));
                            next.push(np.clone());
                            // EXTRACT emits per updated accumulator, with the
                            // accumulator value bound for navigation.
                            let nbase = crate::runtime::rt_getpath(input, &np).unwrap_or(Value::Null);
                            let old_acc2 = { let mut e = env.borrow_mut(); std::mem::replace(&mut e.vars[ai.idx()], nbase.clone()) };
                            let ec = if let Some(ex) = extract {
                                eval_path(ex, nbase, env, &mut |ep| {
                                    let mut comb = np_vec.clone();
                                    if let Value::Arr(a) = &ep { comb.extend(a.iter().cloned()); }
                                    cb(Value::Arr(Rc::new(comb)))
                                })
                            } else {
                                // 2-arg `foreach gen as $x (init; update)` over a
                                // value-generator source yields UPDATE each step,
                                // so with a PATH-threaded accumulator the output is
                                // the UPDATE path (`path(foreach (1,2) as $x (.;.))`
                                // → `[],[]`; `path(foreach range(1) as $x (.;.a))`
                                // → `["a"]`). The old code bailed as if the
                                // accumulator were rootless, which only holds for
                                // the value-threaded (rootless-INIT) branch below.
                                // #915
                                cb(np.clone())
                            };
                            env.borrow_mut().vars[ai.idx()] = old_acc2;
                            let c = ec?;
                            if !c { stop = true; }
                            Ok(c)
                        });
                        env.borrow_mut().vars[ai.idx()] = old_acc;
                        // Keep $i bound across all accumulator branches of this
                        // source element; restore it only after the loop (or on
                        // error) so a multi-valued UPDATE doesn't reset it
                        // mid-iteration.
                        if let Err(e) = r {
                            env.borrow_mut().vars[vi.idx()] = old_var;
                            return Err(e);
                        }
                        if stop { break; }
                    }
                    env.borrow_mut().vars[vi.idx()] = old_var;
                    if stop { return Ok(false); }
                    acc_paths = next;
                }
            }
            Ok(true)
        }
        Err(e) => {
            if !crate::signal::is_path_result(&e) { return Err(e); }
            // VALUE-threaded accumulator (the `nth` counter): a bare `$x` in
            // EXTRACT is the source value, not a path, so it surfaces as the
            // sink's "result <x>" — no FOREACH_PATH_BIND is registered.
            let init_vals = {
                let mut v = Vec::new();
                eval(init, input.clone(), env, &mut |x| { v.push(x); Ok(true) })?;
                v
            };
            for seed in init_vals {
                let mut acc = seed;
                for sv in &source_vals {
                    let (old_var, old_acc) = {
                        let mut e = env.borrow_mut();
                        let ov = std::mem::replace(&mut e.vars[vi.idx()], sv.clone());
                        let oa = std::mem::replace(&mut e.vars[ai.idx()], acc.clone());
                        (ov, oa)
                    };
                    let acc_in = acc.clone();
                    let mut stop = false;
                    let r = eval(update, acc_in, env, &mut |new_acc| {
                        acc = new_acc.clone();
                        env.borrow_mut().vars[ai.idx()] = new_acc.clone();
                        let c = if let Some(ex) = extract {
                            eval_path(ex, new_acc.clone(), env, cb)?
                        } else {
                            return Err(crate::signal::PathResultSignal::raise(&new_acc));
                        };
                        if !c { stop = true; }
                        Ok(c)
                    });
                    {
                        let mut e = env.borrow_mut();
                        e.vars[ai.idx()] = old_acc;
                        e.vars[vi.idx()] = old_var;
                    }
                    r?;
                    if stop { return Ok(false); }
                }
            }
            Ok(true)
        }
    }
}

/// Path-context evaluation of `until(cond; update)`. Faithful to jq's
/// `def _until: if cond then . else (update | _until) end;`: when `cond` is
/// truthy on the current value the current path is emitted; otherwise the path
/// is extended through `update` (a path expression) and the loop recurses.
/// `cur_val` is the value at `prefix`; `prefix` is the absolute path from the
/// document root. See #882.
fn eval_path_until(
    cond: &Expr, update: &Expr, cur_val: Value, prefix: Vec<Value>,
    env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    stacker::maybe_grow(128 * 1024, 32 * 1024 * 1024, || {
        eval(cond, cur_val.clone(), env, &mut |cv| {
            if cv.is_truthy() {
                cb(Value::Arr(Rc::new(prefix.clone())))
            } else {
                eval_path(update, cur_val.clone(), env, &mut |up| {
                    let sub = crate::runtime::rt_getpath(&cur_val, &up).unwrap_or(Value::Null);
                    let mut next = prefix.clone();
                    if let Value::Arr(a) = &up { next.extend(a.iter().cloned()); }
                    eval_path_until(cond, update, sub, next, env, cb)
                })
            }
        })
    })
}

/// Path-context evaluation of `while(cond; update)`. Faithful to jq's
/// `def _while: if cond then ., (update | _while) else empty end;`: when `cond`
/// is truthy the current path is emitted AND the loop continues through
/// `update`; otherwise the branch terminates. See #882.
fn eval_path_while(
    cond: &Expr, update: &Expr, cur_val: Value, prefix: Vec<Value>,
    env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    stacker::maybe_grow(128 * 1024, 32 * 1024 * 1024, || {
        eval(cond, cur_val.clone(), env, &mut |cv| {
            if cv.is_truthy() {
                if !cb(Value::Arr(Rc::new(prefix.clone())))? { return Ok(false); }
                eval_path(update, cur_val.clone(), env, &mut |up| {
                    let sub = crate::runtime::rt_getpath(&cur_val, &up).unwrap_or(Value::Null);
                    let mut next = prefix.clone();
                    if let Value::Arr(a) = &up { next.extend(a.iter().cloned()); }
                    eval_path_while(cond, update, sub, next, env, cb)
                })
            } else {
                Ok(true)
            }
        })
    })
}

fn eval_recurse_paths(val: &Value, prefix: &Value, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    // Use mutable path stack to avoid O(depth) clones per path
    let mut path_stack: Vec<Value> = match prefix {
        Value::Arr(a) => a.as_ref().clone(),
        _ => vec![],
    };
    eval_recurse_paths_inner(val, &mut path_stack, cb)
}

fn eval_recurse_paths_inner(val: &Value, path: &mut Vec<Value>, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    if !cb(Value::Arr(Rc::new(path.clone())))? { return Ok(false); }
    match val {
        Value::Arr(a) => {
            for (i, item) in a.iter().enumerate() {
                path.push(Value::number(i as f64));
                if !eval_recurse_paths_inner(item, path, cb)? { return Ok(false); }
                path.pop();
            }
        }
        Value::Obj(ObjInner(o)) => {
            for (k, v) in o.iter() {
                path.push(Value::from_str(k));
                if !eval_recurse_paths_inner(v, path, cb)? { return Ok(false); }
                path.pop();
            }
        }
        _ => {}
    }
    Ok(true)
}

/// Path-mode `recurse(f)`: yield the current path, then follow the step `f`
/// in path mode at each level (`def r: ., (f | r)`). Paths are relative to the
/// value passed in (Pipe concatenates any prefix). A step that cannot apply to
/// a leaf surfaces jq's per-step type error rather than stopping silently
/// (`path(recurse(.a))` on `{"a":{"a":1}}` → "Cannot index number"). The
/// recursion is lazy: a `cb` stop request (`limit`/`first`) halts the descent,
/// and `stacker::maybe_grow` guards deep navigation. #917
fn eval_recurse_step_paths(
    step: &Expr,
    val: &Value,
    path: &mut Vec<Value>,
    env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    if !cb(Value::Arr(Rc::new(path.clone())))? { return Ok(false); }
    eval_path(step, val.clone(), env, &mut |sp| {
        let child = crate::runtime::rt_getpath(val, &sp).unwrap_or(Value::Null);
        let saved = path.len();
        if let Value::Arr(a) = &sp { path.extend(a.iter().cloned()); }
        let r = stacker::maybe_grow(64 * 1024, 1024 * 1024, || {
            eval_recurse_step_paths(step, &child, path, env, cb)
        });
        path.truncate(saved);
        r
    })
}

fn eval_call_builtin(name: &str, args: &[Expr], input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    // Special handling for builtins that take filter/closure arguments
    match (name, args.len()) {
        ("input_line_number", 0) => {
            return cb(Value::number(get_input_line_number() as f64));
        }
        ("input_filename", 0) => {
            return cb(get_input_filename());
        }
        ("get_search_list", 0) => {
            // jq reports the *effective* search list: the `-L` dirs (canonical-
            // ised via realpath when they resolve, else kept as given) when any
            // are present, otherwise the compile-time defaults. The static
            // builtin ignored `-L` entirely. #1003
            let dirs = env.borrow().lib_dirs.clone();
            let list: Vec<Value> = if dirs.is_empty() {
                vec![
                    Value::from_str("~/.jq"),
                    Value::from_str("$ORIGIN/../lib/jq"),
                    Value::from_str("$ORIGIN/../lib"),
                ]
            } else {
                dirs.iter()
                    .map(|d| {
                        let canon = std::fs::canonicalize(d)
                            .ok()
                            .map(|p| p.to_string_lossy().into_owned())
                            .unwrap_or_else(|| d.clone());
                        Value::from_str(&canon)
                    })
                    .collect()
            };
            return cb(Value::Arr(Rc::new(list)));
        }
        ("toboolean", 0) => {
            return cb(rt_toboolean(&input)?);
        }
        ("halt", 0) => {
            // halt: terminate with status 0 after emitting any values the
            // preceding generator already yielded. Raising a signal error
            // lets the CLI flush its buffered stdout before exiting (see
            // the HaltSignal handling in bin/jq-jit.rs).
            return Err(crate::signal::HaltSignal::raise(0));
        }
        ("halt_error", 0) => {
            halt_error_write(&input);
            return Err(crate::signal::HaltSignal::raise(5));
        }
        ("halt_error", 1) => {
            return eval(&args[0], input.clone(), env, &mut |code_val| {
                let code = match &code_val {
                    // jq clamps any negative halt_error code to 0 (still emitting
                    // the stderr payload); without this, `n as i32` -> the OS
                    // truncates to `n & 0xff`, so halt_error(-1) wrongly exits 255
                    // instead of 0 (#979).
                    Value::Num(n, _) if *n < 0.0 => 0,
                    Value::Num(n, _) => *n as i32,
                    _ => bail!(
                        "{} halt_error/1: number required",
                        crate::runtime::errdesc_pub(&input)
                    ),
                };
                halt_error_write(&input);
                Err(crate::signal::HaltSignal::raise(code))
            });
        }
        ("add", 1) => {
            // add(f) = reduce .[] as $x (null; . + ($x | f))
            return eval_add_filter(&args[0], input, env, cb);
        }
        ("skip", 2) => {
            // skip(n; exp): evaluate n, then skip that many from exp applied to input
            return eval(&args[0], input.clone(), env, &mut |nval| {
                eval_skip(&args[1], &nval, input.clone(), env, cb)
            });
        }
        ("pick", 1) => {
            // pick(f): extract paths generated by f from input
            return eval_pick(&args[0], input, env, cb);
        }
        ("walk", 1) => {
            // walk(f): recursively apply f bottom-up
            return eval_walk(&args[0], input, env, cb);
        }
        ("del", 1) => {
            // del(f) = delpaths([path(f)])
            return eval_del(&args[0], input, env, cb);
        }
        ("exec", 2) => {
            // exec(generator; "cmd"): spawn cmd once, pipe generator outputs to stdin, yield stdout lines
            return eval_exec_pipe(&args[0], &args[1], input, env, cb);
        }
        ("fromcsv", 0) | ("fromtsv", 0) => {
            return eval_fromcsv(&input, name == "fromtsv", cb);
        }
        ("tostream", 0) => {
            return eval_tostream(&input, cb);
        }
        ("fromstream", 1) => {
            return eval_fromstream(&args[0], input, env, cb);
        }
        ("truncate_stream", 1) => {
            return eval_truncate_stream(&args[0], input, env, cb);
        }
        ("fromcsvh", _) | ("fromtsvh", _) => {
            let is_tsv = name == "fromtsvh";
            if args.is_empty() {
                return eval_fromcsvh_auto(&input, is_tsv, cb);
            } else {
                return eval(&args[0], input.clone(), env, &mut |headers_val| {
                    eval_fromcsvh_with_headers(&input, &headers_val, is_tsv, cb)
                });
            }
        }
        ("bsearch", 1) => {
            // bsearch(target): binary search - evaluate target then call runtime
            return eval(&args[0], input.clone(), env, &mut |target| {
                cb(rt_bsearch(&input, &target)?)
            });
        }
        ("strflocaltime", 1) => {
            // strflocaltime(fmt): evaluate fmt then call runtime
            return eval(&args[0], input.clone(), env, &mut |fmt| {
                cb(rt_strflocaltime(&input, &fmt)?)
            });
        }
        ("format", 1) => {
            // format(f): evaluate f to get the format directive name, then
            // apply it to the current input (same result as `@<fmt>`).
            return eval(&args[0], input.clone(), env, &mut |fmt_val| {
                let kind = match &fmt_val {
                    Value::Str(s) => FormatKind::from_name(s.as_str()),
                    _ => bail!("{} is not a valid format", crate::value::value_to_json(&fmt_val)),
                };
                cb(Value::from_str(&eval_format(&kind, &input)?))
            });
        }
        ("combinations", 0) => {
            return eval_combinations(&input, cb);
        }
        ("combinations", 1) => {
            // combinations(n) = . as $dot | [range(n) | $dot] | combinations
            return eval(&args[0], input.clone(), env, &mut |n_val| {
                let n = match &n_val {
                    Value::Num(x, _) if x.is_finite() && *x >= 0.0 => *x as usize,
                    _ => bail!("combinations/1 requires a non-negative integer"),
                };
                let arrays = Value::Arr(Rc::new(vec![input.clone(); n]));
                eval_combinations(&arrays, cb)
            });
        }
        ("modf", 0) => {
            // modf returns [fractional_part, integer_part]. Use libm::modf
            // so the fractional part keeps the sign of the input (e.g.
            // -1.0 → (-0.0, -1.0)) — naive subtraction loses that.
            let n = match &input {
                Value::Num(n, _) => *n,
                _ => bail!("modf requires number input"),
            };
            let (frac_part, int_part) = libm::modf(n);
            return cb(Value::Arr(Rc::new(vec![
                Value::number(frac_part),
                Value::number(int_part),
            ])));
        }
        _ => {}
    }
    // Default: evaluate args as generators and call runtime with input + args.
    // `collected[0]` is the input; `collected[1+i]` is the value of `args[i]`.
    // jq nests multi-argument builtin generators rightmost-outer (the last
    // argument is the outer loop), so bind the arguments right-to-left. The
    // values land in the same positions — only the stream order changes (#978).
    let mut collected = vec![Value::Null; args.len() + 1];
    collected[0] = input.clone();
    eval_call_builtin_args(name, args, args.len(), collected, input, env, cb)
}

/// Yield each Cartesian-product combination of an array of arrays, in
/// lexicographic order. Empty input array yields a single empty
/// combination (matching jq).
/// jq's `length` of a value is 0 for: null, an empty array/object/string, and
/// the number 0. (Booleans have no length and error.) Used so `combinations`
/// can mirror jq's `if length == 0 then [] else ...` short-circuit (#805).
fn input_length_is_zero(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Arr(a) => a.is_empty(),
        Value::Obj(o) => o.is_empty(),
        Value::Str(s) => s.as_str().is_empty(),
        Value::Num(n, _) => *n == 0.0,
        Value::True | Value::False | Value::Error(_) => false,
    }
}

fn eval_combinations(input: &Value, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    let arrays = match input {
        Value::Arr(a) => a.clone(),
        // jq's `combinations` is `if length == 0 then [] else .[0][] as $x | ... end`,
        // so any length-0 input (null, {}, "", 0) yields a single empty combination
        // before the `.[0]` indexing in the else branch runs (#805). A non-array
        // input with a non-zero length falls through to the array-indexing error.
        _ if input_length_is_zero(input) => return cb(Value::Arr(Rc::new(vec![]))),
        _ => bail!("combinations requires array of arrays input"),
    };
    let mut current: Vec<Value> = Vec::with_capacity(arrays.len());
    fn rec(
        arrays: &[Value],
        idx: usize,
        current: &mut Vec<Value>,
        cb: &mut dyn FnMut(Value) -> GenResult,
    ) -> GenResult {
        if idx == arrays.len() {
            // Propagate the callback's continue/stop signal so an outer
            // first/limit/isempty can truncate the Cartesian product (#815).
            return cb(Value::Arr(Rc::new(current.clone())));
        }
        let inner = match &arrays[idx] {
            Value::Arr(a) => a.clone(),
            _ => bail!("combinations: each element must be an array"),
        };
        for v in inner.iter() {
            current.push(v.clone());
            let cont = rec(arrays, idx + 1, current, cb)?;
            current.pop();
            if !cont {
                return Ok(false);
            }
        }
        Ok(true)
    }
    rec(&arrays, 0, &mut current, cb)
}

/// `tostream` (jq 1.8.1): emit `[path, value]` for every leaf and
/// `[path]` close markers for non-empty containers, depth-first.
/// Empty containers are leaves (`[path, []]` / `[path, {}]` with no
/// close marker). See #89.
fn eval_tostream(input: &Value, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    fn walk(
        v: &Value,
        path: &mut Vec<Value>,
        cb: &mut dyn FnMut(Value) -> GenResult,
    ) -> GenResult {
        match v {
            Value::Arr(a) if !a.is_empty() => {
                let mut last_key = Value::Null;
                for (i, item) in a.iter().enumerate() {
                    let key = Value::number(i as f64);
                    path.push(key.clone());
                    if !walk(item, path, cb)? { return Ok(false); }
                    path.pop();
                    last_key = key;
                }
                let mut close_path = path.clone();
                close_path.push(last_key);
                cb(Value::Arr(Rc::new(vec![Value::Arr(Rc::new(close_path))])))
            }
            Value::Obj(ObjInner(o)) if !o.is_empty() => {
                let mut last_key = Value::Null;
                for (k, item) in o.iter() {
                    let key = Value::from_str(k.as_str());
                    path.push(key.clone());
                    if !walk(item, path, cb)? { return Ok(false); }
                    path.pop();
                    last_key = key;
                }
                let mut close_path = path.clone();
                close_path.push(last_key);
                cb(Value::Arr(Rc::new(vec![Value::Arr(Rc::new(close_path))])))
            }
            _ => {
                let path_val = Value::Arr(Rc::new(path.clone()));
                cb(Value::Arr(Rc::new(vec![path_val, v.clone()])))
            }
        }
    }
    let mut path = Vec::new();
    walk(input, &mut path, cb)
}

/// `fromstream(f)`: reassemble events produced by `f` back into JSON
/// trees. Mirrors jq's `def fromstream(f): foreach f as $i ...` —
/// emit a tree once a top-level close marker (path length == 1) or a
/// root-leaf event (path length == 0) lands. See #89.
/// `length` of a stream event's leading `.[0]` element, used purely to
/// decide close-depth, exactly as jq's `fromstream`/`truncate_stream`
/// builtins do (`$i[0] | length`). jq tolerates any first-element type:
/// a missing/`null` first element has length 0, a number its magnitude,
/// a string its codepoint count, an array/object its size; only a boolean
/// has no length. This leniency is why a degenerate event like `[]` or
/// `[5]` is a no-op in jq rather than an error. See #885.
fn stream_first_length(v: &Value) -> Result<i64> {
    Ok(match v {
        Value::Null => 0,
        Value::Num(n, _) => n.abs() as i64,
        Value::Str(s) => s.chars().count() as i64,
        Value::Arr(a) => a.len() as i64,
        Value::Obj(ObjInner(o)) => o.len() as i64,
        Value::True | Value::False => bail!("boolean ({}) has no length", crate::value::value_to_json(v)),
        Value::Error(_) => bail!("error has no length"),
    })
}

fn eval_fromstream(
    f: &Expr,
    input: Value,
    env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    // Mirror jq's builtin exactly:
    //   { x: null, e: false } as $init
    //   | foreach f as $i ($init;
    //       (if .e then $init else . end)
    //       | if $i|length == 2
    //         then setpath(["e"]; $i[0]|length==0) | setpath(["x"]+$i[0]; $i[1])
    //         else setpath(["e"]; $i[0]|length==1) end;
    //       if .e then .x else empty end)
    // The close-depth is `$i[0]|length`, so a length-1 close event flushes the
    // accumulator (emitting it, possibly `null`) even with no preceding value
    // event, and degenerate events (`[]`, `[5]`) are no-ops rather than errors.
    let mut x: Value = Value::Null;
    let mut e = false;
    let result = eval(f, input.clone(), env, &mut |event| {
        let arr = match &event {
            Value::Arr(a) => a.clone(),
            _ => bail!("fromstream: expected stream event, got {}", event.type_name()),
        };
        // `if .e then $init else . end`: reset after a flush.
        if e {
            x = Value::Null;
            e = false;
        }
        let first = arr.first().cloned().unwrap_or(Value::Null);
        if arr.len() == 2 {
            e = stream_first_length(&first)? == 0;
            // setpath(["x"] + $i[0]; $i[1]) — $i[0] is the leaf path.
            let path_arr = match &first {
                Value::Arr(p) => p.clone(),
                other => bail!(
                    "fromstream: leaf path must be an array, not {}",
                    other.type_name()
                ),
            };
            x = setpath_in_place(x.clone(), &path_arr, arr[1].clone())?;
        } else {
            e = stream_first_length(&first)? == 1;
        }
        if e {
            if !cb(x.clone())? { return Ok(false); }
        }
        Ok(true)
    });
    result?;
    Ok(true)
}

/// `setpath` over a `Vec<Value>` path, autovivifying intermediate
/// arrays/objects as jq does — kept local to fromstream so we don't
/// have to reach into the public setpath path-error wording.
fn setpath_in_place(target: Value, path: &[Value], value: Value) -> Result<Value> {
    if path.is_empty() {
        return Ok(value);
    }
    let head = &path[0];
    let rest = &path[1..];
    match head {
        Value::Num(n, _) => {
            let idx = *n as i64;
            let mut arr = match target {
                Value::Arr(a) => Rc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
                Value::Null => Vec::new(),
                _ => bail!("fromstream: cannot index {} with number", target.type_name()),
            };
            if idx < 0 {
                bail!("fromstream: negative array index in stream path");
            }
            let i = idx as usize;
            while arr.len() <= i { arr.push(Value::Null); }
            let inner = std::mem::replace(&mut arr[i], Value::Null);
            arr[i] = setpath_in_place(inner, rest, value)?;
            Ok(Value::Arr(Rc::new(arr)))
        }
        Value::Str(s) => {
            let mut obj = match target {
                Value::Obj(ObjInner(o)) => Rc::try_unwrap(o).unwrap_or_else(|o| (*o).clone()),
                Value::Null => crate::value::new_objmap(),
                _ => bail!("fromstream: cannot index {} with string", target.type_name()),
            };
            let key = KeyStr::from(s.as_str());
            let inner = obj.shift_remove(&key).unwrap_or(Value::Null);
            obj.insert(key, setpath_in_place(inner, rest, value)?);
            Ok(Value::object_from_map(obj))
        }
        _ => bail!("fromstream: path segments must be strings or numbers"),
    }
}

/// `truncate_stream(f)`: input is the depth `$n` to drop; for each
/// event from `f`, emit the event with the first `$n` path components
/// chopped off (skipping events whose path is too short). See #89.
fn eval_truncate_stream(
    f: &Expr,
    input: Value,
    env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> GenResult,
) -> GenResult {
    // jq's `truncate_stream` is `. as $n | null | stream | ... if (.[0]|length) > $n
    // then setpath([0]; .[0][$n:]) else empty end`, so the depth `$n` (taken from
    // `.`) is used both in a value comparison and as a slice bound. A non-number
    // depth is therefore handled leniently rather than rejected: `null` passes the
    // events through unchanged, while a string/array/object depth makes the
    // `length > $n` comparison false and drops every event. We mirror those
    // semantics instead of hard-requiring a numeric depth (#804).
    // jq's def is `. as $n | null | stream | …`: the depth is captured in `$n`
    // but the stream argument runs with `.` == null, NOT the depth. Passing the
    // depth as the stream's input made any `stream` that reads `.` see the
    // number (value drift, spurious type errors, over-strict errors) — #956.
    let depth = input;
    eval(f, Value::Null, env, &mut |event| {
        // jq's body indexes/sets the event with `.[0]` and `setpath([0]; …)`.
        // A null event behaves exactly like an empty array (`.[0]` is null,
        // length 0; `setpath([0]; …)` extends it to a one-element array), so
        // route it through the same array path. Any other scalar/object event
        // is a `Cannot index <type> with number` error, matching jq — now
        // reachable because the stream runs on null (e.g. `truncate_stream(.)`
        // yields a null event, `truncate_stream(5)` a number event). #956
        let arr = match &event {
            Value::Arr(a) => a.clone(),
            Value::Null => Rc::new(Vec::new()),
            other => bail!("Cannot index {} with number", other.type_name()),
        };
        // jq reads `.[0]|length` leniently: a missing/non-array first element
        // just yields a length (0 for an absent/`null` `.[0]`), so a degenerate
        // event like `[]` interspersed between valid events is skipped by the
        // `(.[0]|length) > $n` test rather than raising an error. See #885.
        let first = arr.first().cloned().unwrap_or(Value::Null);
        let path_len = stream_first_length(&first)? as usize;
        // `(.[0]|length) > $n` in jq's type ordering (null < bool < number < ...).
        // A number is always greater than null/bool and always less than
        // string/array/object, so only a Num depth participates numerically.
        let keep = match &depth {
            Value::Null | Value::True | Value::False => true,
            Value::Num(n, _) => (path_len as f64) > *n,
            _ => false,
        };
        if !keep {
            return Ok(true);
        }
        // jq's kept branch is `setpath([0]; .[0][$n:])`: slice `.[0]` (== `first`)
        // by `$n` and write it back to element 0. A null `.[0]` (null/empty
        // event, or an event whose 0th element is already null) slices to null
        // — and, like jq's `null[$n:]`, ignores the slice-index type, so a bool
        // depth that survived the keep test does NOT error here. An array `.[0]`
        // validates and clamps `$n` like any jq slice. Any other `.[0]` (an
        // exotic hand-built event) keeps the lenient passthrough.
        let sliced = match &first {
            Value::Null => Value::Null,
            Value::Arr(path_arr) => {
                // `.[0][$n:]` slice bound. null means "from the start"; a number
                // is floored and clamped like any jq array slice; anything else
                // errors ("must be integers"), matching jq when a bool depth
                // survives the comparison above.
                let start: usize = match &depth {
                    Value::Null => 0,
                    Value::Num(n, _) => {
                        let mut i = n.floor();
                        if i < 0.0 {
                            i += path_len as f64;
                        }
                        if i < 0.0 {
                            0
                        } else if i > path_len as f64 {
                            path_len
                        } else {
                            i as usize
                        }
                    }
                    _ => bail!("Array/string slice indices must be integers"),
                };
                Value::Arr(Rc::new(path_arr.iter().skip(start).cloned().collect()))
            }
            _ => return cb(event.clone()),
        };
        // setpath([0]; sliced): replace element 0, extending a null/empty event
        // into the one-element array `[sliced]` (jq's setpath creates the slot).
        let mut new_event: Vec<Value> = Vec::with_capacity(arr.len().max(1));
        new_event.push(sliced);
        for v in arr.iter().skip(1) { new_event.push(v.clone()); }
        cb(Value::Arr(Rc::new(new_event)))
    })
}

/// Emit the `halt_error` message to stderr using jq 1.8.1's rules:
/// string inputs are written raw (no quotes, no newline); null inputs
/// produce no output at all; everything else is JSON-encoded (no
/// trailing newline).
fn halt_error_write(input: &Value) {
    use std::io::Write;
    let stderr = std::io::stderr();
    let mut stderr = stderr.lock();
    match input {
        Value::Null => {}
        Value::Str(s) => { let _ = stderr.write_all(s.as_str().as_bytes()); }
        _ => {
            // jq prints a non-string payload as JSON followed by a trailing
            // newline (a string payload is written verbatim, null writes
            // nothing). See #845.
            let json = crate::value::value_to_json_precise(input);
            let _ = stderr.write_all(json.as_bytes());
            let _ = stderr.write_all(b"\n");
        }
    }
}

fn eval_exec_pipe(gen_expr: &Expr, cmd_expr: &Expr, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    use std::io::Write;
    // Evaluate the command string first
    let mut cmd_str = None;
    eval(cmd_expr, input.clone(), env, &mut |cmd_val| {
        match &cmd_val {
            Value::Str(s) => { cmd_str = Some(s.as_str().to_string()); }
            _ => { return Err(anyhow::anyhow!("exec: command must be a string")); }
        }
        Ok(true)
    })?;
    let cmd_str = cmd_str.ok_or_else(|| anyhow::anyhow!("exec: command produced no value"))?;

    // Spawn the process once
    let mut child = std::process::Command::new("sh")
        .args(["-c", &cmd_str])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("exec: failed to spawn: {}", e))?;

    // Pipe generator outputs to stdin
    {
        let mut stdin = child.stdin.take().unwrap();
        eval(gen_expr, input, env, &mut |val| {
            let line = match &val {
                Value::Str(s) => s.as_str().to_string(),
                other => crate::value::value_to_json(other),
            };
            writeln!(stdin, "{}", line)
                .map_err(|e| anyhow::anyhow!("exec: write to stdin failed: {}", e))?;
            Ok(true)
        })?;
        // stdin is dropped here, signaling EOF
    }

    let output = child.wait_with_output()
        .map_err(|e| anyhow::anyhow!("exec: failed to wait: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output.status.code().unwrap_or(-1);
        bail!("exec: command exited with code {}: {}", code, stderr.trim_end());
    }

    // Yield each line of stdout as a separate value
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.as_ref().lines() {
        cb(Value::from_str(line))?;
    }
    Ok(true)
}

fn eval_fromcsv(input: &Value, is_tsv: bool, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    let s = match input {
        Value::Str(s) => s.as_str().to_string(),
        _ => bail!("fromcsv input must be a string"),
    };
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(if is_tsv { b'\t' } else { b',' })
        .from_reader(s.as_bytes());
    for result in rdr.records() {
        let record = result.map_err(|e| anyhow::anyhow!("CSV parse error: {}", e))?;
        let arr: Vec<Value> = record.iter().map(Value::from_str).collect();
        cb(Value::Arr(Rc::new(arr)))?;
    }
    Ok(true)
}

fn eval_fromcsvh_auto(input: &Value, is_tsv: bool, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    let s = match input {
        Value::Str(s) => s.as_str().to_string(),
        _ => bail!("fromcsvh input must be a string"),
    };
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(if is_tsv { b'\t' } else { b',' })
        .from_reader(s.as_bytes());
    let headers: Vec<String> = rdr.headers()
        .map_err(|e| anyhow::anyhow!("CSV parse error: {}", e))?
        .iter()
        .map(|h| h.to_string())
        .collect();
    for result in rdr.records() {
        let record = result.map_err(|e| anyhow::anyhow!("CSV parse error: {}", e))?;
        let mut obj = crate::value::new_objmap();
        for (i, field) in record.iter().enumerate() {
            if let Some(key) = headers.get(i) {
                obj.insert(KeyStr::from(key.as_str()), Value::from_str(field));
            }
        }
        cb(Value::object_from_map(obj))?;
    }
    Ok(true)
}

fn eval_fromcsvh_with_headers(input: &Value, headers_val: &Value, is_tsv: bool, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    let s = match input {
        Value::Str(s) => s.as_str().to_string(),
        _ => bail!("fromcsvh input must be a string"),
    };
    let headers: Vec<String> = match headers_val {
        Value::Arr(arr) => {
            arr.iter().map(|v| match v {
                Value::Str(s) => Ok(s.as_str().to_string()),
                _ => Err(anyhow::anyhow!("fromcsvh headers must be strings")),
            }).collect::<Result<Vec<_>, _>>()?
        }
        _ => bail!("fromcsvh argument must be an array of strings"),
    };
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(if is_tsv { b'\t' } else { b',' })
        .from_reader(s.as_bytes());
    for result in rdr.records() {
        let record = result.map_err(|e| anyhow::anyhow!("CSV parse error: {}", e))?;
        let mut obj = crate::value::new_objmap();
        for (i, field) in record.iter().enumerate() {
            if let Some(key) = headers.get(i) {
                obj.insert(KeyStr::from(key.as_str()), Value::from_str(field));
            }
        }
        cb(Value::object_from_map(obj))?;
    }
    Ok(true)
}

// `remaining` counts arguments not yet bound, from the right: the first call
// binds `args[remaining-1]` (the rightmost argument, the outer loop), and each
// nested level binds the next-earlier argument. `collected[0]` holds the input
// and `collected[1+i]` the value of `args[i]`, so the values stay in argument
// order regardless of the (rightmost-outer) nesting direction. See #978.
fn eval_call_builtin_args(name: &str, args: &[Expr], remaining: usize, collected: Vec<Value>, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    if remaining == 0 {
        return cb(crate::runtime::call_builtin(name, &collected)?);
    }
    let idx = remaining - 1;
    eval(&args[idx], input.clone(), env, &mut |val| {
        let mut next = collected.clone();
        next[idx + 1] = val;
        eval_call_builtin_args(name, args, remaining - 1, next, input.clone(), env, cb)
    })
}

// toboolean: "true" -> true, "false" -> false, bool -> bool, else error
fn rt_toboolean(v: &Value) -> Result<Value> {
    match v {
        Value::True => Ok(Value::True),
        Value::False => Ok(Value::False),
        Value::Str(s) => match s.as_str() {
            "true" => Ok(Value::True),
            "false" => Ok(Value::False),
            _ => bail!("string ({:?}) cannot be parsed as a boolean", s.as_str()),
        },
        _ => {
            let ty = v.type_name();
            let json = crate::value::value_to_json(v);
            bail!("{} ({}) cannot be parsed as a boolean", ty, json);
        }
    }
}

// add(f): reduce f as $x (null; . + $x)
fn eval_add_filter(f: &Expr, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    // Collect all outputs of f applied to input
    let mut acc: Option<Value> = None;
    eval(f, input, env, &mut |val| {
        acc = Some(match acc.take() {
            None => val,
            Some(a) => crate::runtime::rt_add(&a, &val)?,
        });
        Ok(true)
    })?;
    cb(acc.unwrap_or(Value::Null))
}

// skip(n; exp): skip first n outputs of exp generator
fn eval_skip(exp: &Expr, nval: &Value, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    let n = match nval {
        Value::Num(n, _) => {
            let n = *n as i64;
            if n < 0 {
                return Err(anyhow::anyhow!("__jqerror__:\"skip doesn't support negative count\""));
            }
            n
        }
        _ => return Err(anyhow::anyhow!("__jqerror__:\"skip count must be a number\"")),
    };
    let mut count = 0i64;
    eval(exp, input, env, &mut |val| {
        count += 1;
        if count > n {
            cb(val)
        } else {
            Ok(true)
        }
    })
}

// pick(f): For each path generated by f, set that path in the output
fn eval_pick(f: &Expr, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    // Collect all paths generated by f (as path arrays). A non-path argument
    // bails with the internal PathResultSignal; rewrite it
    // to jq's user-facing message (catchable via try/catch), like the other
    // path-eval entry points. #848
    let mut paths: Vec<Value> = Vec::new();
    eval_path(f, input.clone(), env, &mut |path| {
        paths.push(path);
        Ok(true)
    })
    .map_err(invalid_path_expr_err)?;
    // Build result by setting each path
    let mut result = Value::Null;
    for path in &paths {
        let val = crate::runtime::rt_getpath(&input, path)?;
        let path_slice = match path {
            Value::Arr(a) => a.as_slice(),
            _ => bail!("Path must be specified as an array"),
        };
        crate::runtime::rt_setpath_mut(&mut result, path_slice, val)?;
    }
    cb(result)
}

/// Detect `if type == "T" then F else . end` pattern.
/// Returns the type string and the then-branch if matched.
fn detect_walk_type_guard(f: &Expr) -> Option<(&str, &Expr)> {
    if let Expr::IfThenElse { cond, then_branch, else_branch } = f {
        // else branch must be identity
        if !matches!(else_branch.as_ref(), Expr::Input) {
            return None;
        }
        // cond must be `type == "T"`
        if let Expr::BinOp { op: crate::ir::BinOp::Eq, lhs, rhs } = cond.as_ref() {
            // type == "T"
            if let Expr::UnaryOp { op: crate::ir::UnaryOp::Type, operand } = lhs.as_ref() {
                if matches!(operand.as_ref(), Expr::Input) {
                    if let Expr::Literal(crate::ir::Literal::Str(s)) = rhs.as_ref() {
                        return Some((s.as_str(), then_branch.as_ref()));
                    }
                }
            }
            // "T" == type
            if let Expr::UnaryOp { op: crate::ir::UnaryOp::Type, operand } = rhs.as_ref() {
                if matches!(operand.as_ref(), Expr::Input) {
                    if let Expr::Literal(crate::ir::Literal::Str(s)) = lhs.as_ref() {
                        return Some((s.as_str(), then_branch.as_ref()));
                    }
                }
            }
        }
    }
    None
}

fn value_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::False | Value::True => "boolean",
        Value::Num(..) => "number",
        Value::Str(..) => "string",
        Value::Arr(..) => "array",
        Value::Obj(..) => "object",
        Value::Error(..) => "error",
    }
}

// walk(f): Recursively apply f bottom-up
fn eval_walk(f: &Expr, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    // Fast path: walk(.) is identity
    if matches!(f, Expr::Input) {
        return cb(input);
    }
    // Also check if it's just a pipe with identity
    if let Expr::Pipe { left, right } = f {
        if matches!(left.as_ref(), Expr::Input) && matches!(right.as_ref(), Expr::Input) {
            return cb(input);
        }
    }
    // Optimization: walk(if type == "T" then F else . end)
    // For values whose type != T, f is identity, so skip eval entirely.
    // Only call eval(F, ...) on matching-type leaf values.
    // The in-place fast path folds the then-branch to a single value, so it is
    // only valid when that branch yields exactly one output. A multi-valued
    // branch (`.,.+1`, `.+(1,2)`, …) must backtrack: each leaf forks the
    // surrounding reconstruction, which the generic `walk_value_cb` handles
    // correctly (arrays collect every fork via `map`, objects keep the first
    // via `map_values`, and the trailing `f` forks at every level). #769
    if let Some((type_name, then_body)) = detect_walk_type_guard(f) {
        if then_body.is_single_output() {
            return walk_type_guarded(type_name, then_body, f, input, env, cb);
        }
    }
    walk_value_cb(f, input, env, cb)
}

fn walk_type_guarded(type_name: &str, then_body: &Expr, _full_f: &Expr, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    let result = walk_type_guarded_inplace(type_name, then_body, input, env)?;
    cb(result)
}

/// Walk with type guard, mutating in-place when possible.
/// Returns a single walked value.
fn walk_type_guarded_inplace(type_name: &str, then_body: &Expr, mut input: Value, env: &EnvRef) -> Result<Value> {
    match input {
        Value::Arr(ref mut rc_arr) => {
            let arr = Rc::make_mut(rc_arr);
            for item in arr.iter_mut() {
                let taken = std::mem::replace(item, Value::Null);
                *item = walk_type_guarded_inplace(type_name, then_body, taken, env)?;
            }
            if type_name == "array" {
                let mut result = Value::Null;
                eval(then_body, input, env, &mut |val| { result = val; Ok(true) })?;
                Ok(result)
            } else {
                Ok(input)
            }
        }
        Value::Obj(ObjInner(ref mut rc_obj)) => {
            let obj = Rc::make_mut(rc_obj);
            for (_k, v) in obj.iter_mut() {
                let taken = std::mem::replace(v, Value::Null);
                *v = walk_type_guarded_inplace(type_name, then_body, taken, env)?;
            }
            if type_name == "object" {
                let mut result = Value::Null;
                eval(then_body, input, env, &mut |val| { result = val; Ok(true) })?;
                Ok(result)
            } else {
                Ok(input)
            }
        }
        _ => {
            if value_type_name(&input) == type_name {
                let mut result = Value::Null;
                eval(then_body, input, env, &mut |val| { result = val; Ok(true) })?;
                Ok(result)
            } else {
                Ok(input)
            }
        }
    }
}

fn walk_value_cb(f: &Expr, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    match input {
        Value::Arr(ref a) => {
            let mut new_arr = Vec::with_capacity(a.len());
            for item in a.iter() {
                walk_value_single(f, item.clone(), env, &mut new_arr)?;
            }
            let rebuilt = Value::Arr(Rc::new(new_arr));
            eval(f, rebuilt, env, cb)
        }
        Value::Obj(ObjInner(ref o)) => {
            let mut new_obj = crate::value::new_objmap();
            for (k, v) in o.iter() {
                let mut walked = Vec::new();
                walk_value_single(f, v.clone(), env, &mut walked)?;
                if let Some(val) = walked.into_iter().next() {
                    new_obj.insert(k.clone(), val);
                }
            }
            let rebuilt = Value::object_from_map(new_obj);
            eval(f, rebuilt, env, cb)
        }
        _ => {
            eval(f, input, env, cb)
        }
    }
}

/// Walk a single value, pushing results into `out`.
fn walk_value_single(f: &Expr, input: Value, env: &EnvRef, out: &mut Vec<Value>) -> Result<()> {
    match input {
        Value::Arr(ref a) => {
            let mut new_arr = Vec::with_capacity(a.len());
            for item in a.iter() {
                walk_value_single(f, item.clone(), env, &mut new_arr)?;
            }
            let rebuilt = Value::Arr(Rc::new(new_arr));
            eval(f, rebuilt, env, &mut |val| {
                out.push(val);
                Ok(true)
            })?;
        }
        Value::Obj(ObjInner(ref o)) => {
            let mut new_obj = crate::value::new_objmap();
            for (k, v) in o.iter() {
                let mut walked = Vec::new();
                walk_value_single(f, v.clone(), env, &mut walked)?;
                if let Some(val) = walked.into_iter().next() {
                    new_obj.insert(k.clone(), val);
                }
            }
            let rebuilt = Value::object_from_map(new_obj);
            eval(f, rebuilt, env, &mut |val| {
                out.push(val);
                Ok(true)
            })?;
        }
        _ => {
            eval(f, input, env, &mut |val| {
                out.push(val);
                Ok(true)
            })?;
        }
    }
    Ok(())
}

// del(f): delete paths generated by f, including slices
// jq semantics: all paths are computed against the original input, then deleted in sorted order
fn eval_del(f: &Expr, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    // Collect all deletion targets as sets of indices relative to original
    let mut del_ops: Vec<DelOp> = Vec::new();
    collect_del_ops(f, &mut del_ops);

    // For top-level array deletions, collect all indices to remove
    // For nested paths and slices, apply sequentially
    let mut indices_to_del: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
    let mut non_index_ops: Vec<&DelOp> = Vec::new();

    if let Value::Arr(a) = &input {
        let len = a.len() as i64;
        for op in &del_ops {
            match op {
                DelOp::Path(expr) => {
                    // Try to get paths — if they're single-element (top-level index), collect as index
                    let mut paths: Vec<Value> = Vec::new();
                    let r = eval_path(expr, input.clone(), env, &mut |path| {
                        paths.push(path);
                        Ok(true)
                    });
                    if r.is_ok() {
                        let mut all_top_level = true;
                        for p in &paths {
                            if let Value::Arr(pa) = p {
                                if pa.len() == 1 {
                                    if let Value::Num(n, _) = &pa[0] {
                                        // A NaN index casts to 0 and would silently
                                        // delete element 0; reject it, mirroring the
                                        // set path and rt_delpaths. (Upstream jq hangs
                                        // here — see #921 — so we do not match it.)
                                        if n.is_nan() {
                                            bail!("Cannot delete array element at NaN index");
                                        }
                                        // Decide the negative-index offset from the
                                        // ORIGINAL float, not the truncated int:
                                        // `*n as i64` truncates toward zero, so a
                                        // fractional index in (-1, 0) (e.g. -0.9)
                                        // would become 0 and wrongly delete element
                                        // 0. jq treats such indices as out of range
                                        // (no-op). Mirror rt_delpaths (#884): add len
                                        // when `*n < 0.0`, then range-check. #904
                                        let idx = if *n < 0.0 {
                                            *n as i64 + len
                                        } else {
                                            *n as i64
                                        };
                                        if idx >= 0 && idx < len {
                                            indices_to_del.insert(idx);
                                        }
                                        continue;
                                    }
                                }
                            }
                            all_top_level = false;
                        }
                        if !all_top_level {
                            non_index_ops.push(op);
                        }
                    } else {
                        non_index_ops.push(op);
                    }
                }
                DelOp::Slice { base, from, to } => {
                    if matches!(base, Expr::Input) {
                        // Generator bounds: delete the union over the Cartesian
                        // product of (from, to) endpoints. #761
                        let from_idxs = eval_slice_idx_vals(from, len, 0, false, &input, env)?;
                        let to_idxs = eval_slice_idx_vals(to, len, len, true, &input, env)?;
                        for &fi in &from_idxs {
                            for &ti in &to_idxs {
                                for i in fi..ti {
                                    if i >= 0 && i < len {
                                        indices_to_del.insert(i);
                                    }
                                }
                            }
                        }
                    } else {
                        non_index_ops.push(op);
                    }
                }
            }
        }

        if non_index_ops.is_empty() {
            // All ops are top-level index deletions — build result skipping deleted indices
            let mut result = Vec::new();
            for i in 0..len {
                if !indices_to_del.contains(&i) {
                    result.push(a[i as usize].clone());
                }
            }
            return cb(Value::Arr(Rc::new(result)));
        }
    }

    // Fallback (nested paths, slices, non-array input): collect every path
    // `f` generates and delete them in one `delpaths` pass. jq defines
    // `del(f)` as `delpaths([path(f)])`, which sorts the paths descending and
    // type-checks each container. Applying the comma-separated ops sequentially
    // instead shifted later array indices (#841), skipped the slice container
    // type check (#842), and autovivified a missing parent to null (#843).
    let mut paths: Vec<Value> = Vec::new();
    eval_path(f, input.clone(), env, &mut |p| { paths.push(p); Ok(true) })
        .map_err(invalid_path_expr_err)?;
    let result = crate::runtime::rt_delpaths(&input, &Value::Arr(Rc::new(paths)))?;
    cb(result)
}

enum DelOp<'a> {
    Path(&'a Expr),
    Slice { base: &'a Expr, from: Option<&'a Expr>, to: Option<&'a Expr> },
}

fn collect_del_ops<'a>(f: &'a Expr, ops: &mut Vec<DelOp<'a>>) {
    match f {
        Expr::Comma { left, right } => {
            collect_del_ops(left, ops);
            collect_del_ops(right, ops);
        }
        Expr::Slice { expr, from, to } => {
            ops.push(DelOp::Slice {
                base: expr,
                from: from.as_deref(),
                to: to.as_deref(),
            });
        }
        _ => {
            ops.push(DelOp::Path(f));
        }
    }
}

/// Resolve a `del(.[a:b])` slice bound to its normalized array indices.
///
/// The bound is a *generator* (#761): every value it yields is an independent
/// endpoint, so the result is a list of indices (the caller takes the Cartesian
/// product of the two bounds). An absent bound contributes the single default;
/// an empty generator contributes nothing (so the slice deletes nothing).
fn eval_slice_idx_vals(expr: &Option<&Expr>, len: i64, default: i64, is_end: bool, input: &Value, env: &EnvRef) -> Result<Vec<i64>> {
    let Some(e) = expr else { return Ok(vec![default]); };
    let mut out = Vec::new();
    eval(e, input.clone(), env, &mut |v| {
        let i = match &v {
            // jq normalizes a negative bound (`n + len`) before converting it to
            // an integer; the start floors and the end ceils. Truncating `n as
            // i64` before adding `len` mis-placed fractional bounds for
            // `del(.[a:b])` (e.g. `del(.[1.5:3.5])` deleted [1,3) not [1,4)). #722.
            Value::Num(n, _) if !n.is_nan() => {
                let norm = if *n < 0.0 { n + len as f64 } else { *n };
                let i = if is_end { norm.ceil() as i64 } else { norm.floor() as i64 };
                i.clamp(0, len)
            }
            _ => default,
        };
        out.push(i);
        Ok(true)
    })?;
    Ok(out)
}

// bsearch(target): binary search on sorted array
fn rt_bsearch(input: &Value, target: &Value) -> Result<Value> {
    match input {
        Value::Arr(a) => {
            let mut lo: i64 = 0;
            let mut hi: i64 = a.len() as i64 - 1;
            while lo <= hi {
                // jq's `bsearch` (builtin.jq) rounds the midpoint UP —
                // `((.[0]+.[1]+1)/2)|floor` — so for a run of equal elements it
                // converges on a higher index than a floor-midpoint search. The
                // returned index is an observable, deterministic value, so the
                // midpoint must match jq's exactly. See #887.
                let mid = (lo + hi + 1) / 2;
                let cmp = crate::runtime::compare_values(&a[mid as usize], target);
                match cmp {
                    std::cmp::Ordering::Equal => return Ok(Value::number(mid as f64)),
                    std::cmp::Ordering::Less => lo = mid + 1,
                    std::cmp::Ordering::Greater => hi = mid - 1,
                }
            }
            // Not found: return -(insertion_point) - 1
            Ok(Value::number(-(lo as f64) - 1.0))
        }
        _ => {
            let ty = input.type_name();
            let json = crate::value::value_to_json(input);
            bail!("{} ({}) cannot be searched from", ty, json);
        }
    }
}

// strflocaltime(fmt): delegates to runtime
fn rt_strflocaltime(input: &Value, fmt: &Value) -> Result<Value> {
    crate::runtime::call_builtin("strflocaltime", &[input.clone(), fmt.clone()])
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

pub fn execute_ir(expr: &Expr, input: Value, funcs: Vec<CompiledFunc>) -> Result<Vec<Value>> {
    execute_ir_with_libs(expr, input, funcs, vec![])
}

pub fn execute_ir_with_libs(expr: &Expr, input: Value, funcs: Vec<CompiledFunc>, lib_dirs: Vec<String>) -> Result<Vec<Value>> {
    let env = Rc::new(RefCell::new(Env::with_lib_dirs(funcs, lib_dirs)));
    let mut outputs = Vec::new();
    let result = eval(expr, input, &env, &mut |val| {
        match &val { Value::Error(e) => { eprintln!("jq: error: {}", e); }, _ => { outputs.push(val); } }
        Ok(true)
    });
    match result {
        Ok(_) => Ok(outputs),
        Err(e) => {
            if e.downcast_ref::<BreakError>().is_some() {
                Ok(outputs)
            } else {
                let msg = format!("{}", e);
                // Report error to stderr but still return collected outputs
                if let Some(json) = msg.strip_prefix("__jqerror__:") {
                    eprintln!("jq: error: {}", json);
                } else {
                    eprintln!("jq: error: {}", msg);
                }
                Ok(outputs)
            }
        }
    }
}

/// Streaming variant: call cb for each result without collecting into Vec.
pub fn execute_ir_with_libs_cb(
    expr: &Expr, input: Value, funcs: Vec<CompiledFunc>, lib_dirs: Vec<String>,
    cb: &mut dyn FnMut(Value) -> Result<bool>,
) -> Result<bool> {
    let env = Rc::new(RefCell::new(Env::with_lib_dirs(funcs, lib_dirs)));
    let result = eval(expr, input, &env, &mut |val| {
        cb(val)
    });
    match result {
        Ok(v) => Ok(v),
        Err(e) => {
            if e.downcast_ref::<BreakError>().is_some() {
                Ok(true)
            } else {
                Err(e)
            }
        }
    }
}

/// Streaming variant that reuses an existing Env (avoids re-allocation).
pub fn execute_ir_with_env_cb(
    expr: &Expr, input: Value, env: &EnvRef,
    cb: &mut dyn FnMut(Value) -> Result<bool>,
) -> Result<bool> {
    env.borrow_mut().reset();
    let result = eval(expr, input, env, &mut |val| {
        cb(val)
    });
    match result {
        Ok(v) => Ok(v),
        Err(e) => {
            if e.downcast_ref::<BreakError>().is_some() {
                Ok(true)
            } else {
                Err(e)
            }
        }
    }
}
