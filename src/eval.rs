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
use crate::value::{Value, ObjInner, NumRepr, KeyStr};

type GenResult = Result<bool>;
pub type EnvRef = Rc<RefCell<Env>>;

// Per-thread inputs queue for `input`/`inputs` builtins.
// Pre-populated by CLI before eval/JIT execution. Per-thread to keep
// `cargo test` parallel runs honest — see `value::OBJMAP_POOL`.
thread_local! {
    static INPUTS_STATE: RefCell<(Vec<Value>, usize)> = const { RefCell::new((Vec::new(), 0)) };
}

// Per-thread 1-indexed line number for `input_line_number`. The CLI updates
// it before executing the filter on each input; jq defines it as the count
// of `\n` bytes consumed at the point the value was emitted (not where the
// value started), which for multi-value lines means every value on that
// line sees the same number.
thread_local! {
    static INPUT_LINE_STATE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Set the line number reported by `input_line_number` for the current input.
pub fn set_input_line_number(n: u64) {
    INPUT_LINE_STATE.with(|c| c.set(n));
}

/// Read the current line number reported by `input_line_number`.
pub fn get_input_line_number() -> u64 {
    INPUT_LINE_STATE.with(|c| c.get())
}

/// Set the inputs queue for `input`/`inputs` builtins.
pub fn set_inputs_queue(values: Vec<Value>) {
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

/// Read the next input value. Returns None if exhausted.
pub fn read_next_input() -> Option<Value> {
    INPUTS_STATE.with_borrow_mut(|state| {
        if state.1 < state.0.len() {
            let idx = state.1;
            state.1 += 1;
            Some(state.0[idx].clone())
        } else {
            None
        }
    })
}

/// Typed error for label/break to avoid string formatting/parsing overhead.
#[derive(Debug)]
struct BreakError(u64);
impl std::fmt::Display for BreakError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "__break__:{}:", self.0)
    }
}
impl std::error::Error for BreakError {}

pub struct Env {
    vars: Vec<Value>,
    funcs: Vec<Rc<CompiledFunc>>,
    next_label: u64,
    pub next_var: u16,
    pub lib_dirs: Vec<String>,
    /// Closure bindings: (param_var_index, arg_expression).
    /// Used to avoid deep-cloning function bodies via substitute_params.
    closures: Vec<(u16, Expr)>,
    /// Cache for is_recursive check per func_id.
    recursive_cache: Vec<(usize, bool)>,
    /// Cache for substituted function bodies: (func_id, arg_var_indices) → substituted body.
    /// Only used when all args are LoadVar (the common case).
    subst_cache: Vec<((usize, Vec<u16>), Rc<Expr>)>,
    /// Pointer-based substitution cache: func_id → (args_ptr, substituted_body).
    /// For non-LoadVar args from stable (cached) call sites.
    subst_ptr_cache: Vec<(usize, usize, Rc<Expr>)>,
}

impl Env {
    pub fn new(funcs: Vec<CompiledFunc>) -> Self {
        Env { vars: vec![Value::Null; 65536], funcs: funcs.into_iter().map(Rc::new).collect(), next_label: 0, next_var: 256, lib_dirs: Vec::new(), closures: Vec::new(), recursive_cache: Vec::new(), subst_cache: Vec::new(), subst_ptr_cache: Vec::new() }
    }
    pub fn with_lib_dirs(funcs: Vec<CompiledFunc>, lib_dirs: Vec<String>) -> Self {
        Env { vars: vec![Value::Null; 65536], funcs: funcs.into_iter().map(Rc::new).collect(), next_label: 0, next_var: 256, lib_dirs, closures: Vec::new(), recursive_cache: Vec::new(), subst_cache: Vec::new(), subst_ptr_cache: Vec::new() }
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
        // Keep recursive_cache, subst_cache, subst_ptr_cache — they stay valid across inputs
    }
    #[inline(always)]
    fn get_var(&self, idx: u16) -> Value {
        let i = idx as usize;
        if i < self.vars.len() {
            // SAFETY: bounds checked above
            unsafe { self.vars.get_unchecked(i) }.clone()
        } else {
            Value::Null
        }
    }
    #[inline(always)]
    fn set_var(&mut self, idx: u16, val: Value) {
        let idx = idx as usize;
        if idx >= self.vars.len() { self.vars.resize(idx + 1, Value::Null); }
        // SAFETY: bounds ensured above
        unsafe { *self.vars.get_unchecked_mut(idx) = val; }
    }
    /// Public setter used by the JIT runtime when it delegates complex paths
    /// back to eval — the JIT has its own var storage, so we copy the live
    /// bindings into the eval Env before dispatch.
    pub fn seed_var(&mut self, idx: u16, val: Value) {
        self.set_var(idx, val);
    }
    fn ensure_var(&mut self, idx: u16) {
        let idx = idx as usize;
        if idx >= self.vars.len() { self.vars.resize(idx + 1, Value::Null); }
    }
}

/// Substitute param var references with arg expressions in a function body.
/// This implements jq's closure semantics: each time a param is referenced,
/// the arg filter is re-evaluated with the current input.
/// Uses COW: unchanged subtrees are not cloned.
pub fn substitute_params(expr: &Expr, param_vars: &[u16], args: &[Expr]) -> Expr {
    subst_cow(expr, param_vars, args).unwrap_or_else(|| expr.clone())
}

/// Substitute params AND rename all local variable bindings (LetBinding, Reduce,
/// Foreach, Label) to fresh indices from `next_var`. This prevents recursive calls
/// from clobbering each other's local variables in the callback-based eval model.
pub fn substitute_and_rename(expr: &Expr, param_vars: &[u16], args: &[Expr], next_var: &mut u16) -> Expr {
    subst_inner(expr, param_vars, args, true, next_var, &mut HashMap::new())
}

fn subst_inner(
    expr: &Expr, pv: &[u16], args: &[Expr],
    rename: bool, nv: &mut u16, rn: &mut HashMap<u16, u16>,
) -> Expr {
    macro_rules! s { ($e:expr) => { subst_inner($e, pv, args, rename, nv, rn) } }
    macro_rules! sb { ($e:expr) => { Box::new(s!($e)) } }
    macro_rules! alloc {
        ($old:expr) => { if rename { let n = *nv; *nv += 1; rn.insert($old, n); n } else { $old } }
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
        Expr::TryCatch { try_expr, catch_expr } => Expr::TryCatch { try_expr: sb!(try_expr), catch_expr: sb!(catch_expr) },
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
        Expr::Format { name, expr: e } => Expr::Format { name: name.clone(), expr: sb!(e) },
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
        Expr::Input | Expr::Empty | Expr::Not | Expr::Env | Expr::Builtins
        | Expr::ReadInput | Expr::ReadInputs | Expr::ModuleMeta | Expr::GenLabel
        | Expr::Literal(_) | Expr::Loc { .. } => expr.clone(),
    }
}

/// COW substitution: returns None if no param vars were found (no changes needed).
/// Avoids deep-cloning unchanged subtrees.
fn subst_cow(expr: &Expr, pv: &[u16], args: &[Expr]) -> Option<Expr> {
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
        Expr::TryCatch { try_expr, catch_expr } => {
            let t = s!(try_expr); let c = s!(catch_expr);
            if t.is_none() && c.is_none() { return None; }
            Some(Expr::TryCatch {
                try_expr: Box::new(t.unwrap_or_else(|| try_expr.as_ref().clone())),
                catch_expr: Box::new(c.unwrap_or_else(|| catch_expr.as_ref().clone())),
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
        Expr::Format { name, expr: e } => s!(e).map(|v| Expr::Format { name: name.clone(), expr: Box::new(v) }),
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
        // Leaf nodes never contain param var references
        Expr::Input | Expr::Empty | Expr::Not | Expr::Env | Expr::Builtins
        | Expr::ReadInput | Expr::ReadInputs | Expr::ModuleMeta | Expr::GenLabel
        | Expr::Literal(_) | Expr::Loc { .. } => None,
    }
}

/// Check if an expression contains a FuncCall to the given func_id (direct recursion check).
fn contains_func_call(expr: &Expr, target: usize) -> bool {
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
        Expr::TryCatch { try_expr, catch_expr } => c!(try_expr) || c!(catch_expr),
        Expr::Collect { generator } => c!(generator),
        Expr::Alternative { primary, fallback } => c!(primary) || c!(fallback),
        Expr::Reduce { source, init, update, .. } => c!(source) || c!(init) || c!(update),
        Expr::Foreach { source, init, update, extract, .. } => {
            c!(source) || c!(init) || c!(update) || extract.as_ref().is_some_and(|e| c!(e))
        }
        Expr::ObjectConstruct { pairs } => pairs.iter().any(|(k, v)| c!(k) || c!(v)),
        Expr::Range { from, to, step } => c!(from) || c!(to) || step.as_ref().is_some_and(|s| c!(s)),
        Expr::Update { path_expr, update_expr } | Expr::Assign { path_expr, value_expr: update_expr } => c!(path_expr) || c!(update_expr),
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
        Expr::Input | Expr::Empty | Expr::Not | Expr::Env | Expr::Builtins
        | Expr::ReadInput | Expr::ReadInputs | Expr::ModuleMeta | Expr::GenLabel
        | Expr::Literal(_) | Expr::Loc { .. } | Expr::LoadVar { .. } | Expr::Error { msg: None } => false,
    }
}

/// Check if an expression uses the input (`.`) passed to it.
/// This tracks input flow: Pipe's right side gets new input from left's output.
fn expr_uses_outer_input(expr: &Expr) -> bool {
    match expr {
        Expr::Input => true,
        Expr::LoadVar { .. } | Expr::Literal(_) | Expr::Empty | Expr::Not
        | Expr::Env | Expr::Builtins | Expr::ReadInput | Expr::ReadInputs
        | Expr::ModuleMeta | Expr::GenLabel | Expr::Loc { .. } => false,
        // Pipe: only left receives our input; right gets left's output
        Expr::Pipe { left, .. } => expr_uses_outer_input(left),
        Expr::Comma { left, right }
        | Expr::BinOp { lhs: left, rhs: right, .. }
        | Expr::Alternative { primary: left, fallback: right }
        | Expr::Index { expr: left, key: right }
        | Expr::IndexOpt { expr: left, key: right }
        | Expr::SetPath { path: left, value: right }
        | Expr::TryCatch { try_expr: left, catch_expr: right } => {
            expr_uses_outer_input(left) || expr_uses_outer_input(right)
        }
        // Update/Assign: path_expr gets our input, update_expr gets path value
        Expr::Update { path_expr, .. } | Expr::Assign { path_expr, .. } => {
            expr_uses_outer_input(path_expr)
        }
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
        | Expr::PathExpr { expr: input_expr } | Expr::GetPath { path: input_expr }
        | Expr::DelPaths { paths: input_expr } | Expr::Debug { expr: input_expr }
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
    }
}

/// Check if an expression references a specific variable (for reduce optimization).
fn expr_uses_var(expr: &Expr, target: u16) -> bool {
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
        | Expr::TryCatch { try_expr: left, catch_expr: right } => {
            expr_uses_var(left, target) || expr_uses_var(right, target)
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
    }
}

/// Extract f64 from a leaf expression (LoadVar, Input, Literal) or simple
/// numeric BinOp without cloning. Used by the zero-clone BinOp fast path.
#[inline(always)]
fn get_num_leaf(expr: &Expr, input: &Value, vars: &[Value]) -> Option<f64> {
    match expr {
        Expr::LoadVar { var_index } => {
            if let Some(Value::Num(n, _)) = vars.get(*var_index as usize) {
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
                BinOp::Mod => { if !ln.is_finite() || !rn.is_finite() { return None; } let yi = rn as i64; if yi == 0 { return None; } (ln as i64 % yi) as f64 }
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
fn eval_bool_numeric(expr: &Expr, vars: &[Value], override_vi: u16, override_val: f64) -> Option<bool> {
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
fn get_num_leaf_override(expr: &Expr, vars: &[Value], override_vi: u16, override_val: f64) -> Option<f64> {
    match expr {
        Expr::LoadVar { var_index } => {
            if *var_index == override_vi {
                Some(override_val)
            } else if let Some(Value::Num(n, _)) = vars.get(*var_index as usize) {
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
                BinOp::Mod => { if !ln.is_finite() || !rn.is_finite() { return None; } let yi = rn as i64; if yi == 0 { return None; } (ln as i64 % yi) as f64 }
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
                                        Value::number((ln as i64 % yi) as f64)
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
                                Value::number((*ln as i64 % yi) as f64)
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
                    UnaryOp::Fabs | UnaryOp::Abs => Value::number(n.abs()),
                    UnaryOp::Length => Value::number(n.abs()),
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
            let func = env.borrow().funcs.get(*func_id).cloned();
            if let Some(f) = func {
                eval_one(&f.body, input, env)
            } else {
                Err(())
            }
        }
        Expr::LetBinding { var_index, value, body } => {
            let vi = *var_index as usize;
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
fn detect_linear_recursive_gen(body: &Expr, func_id: usize) -> Option<LinearRecursiveGen<'_>> {
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
fn extract_recursive_pipe(expr: &Expr, func_id: usize) -> Option<&Expr> {
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
                    // When lhs is LoadVar and gen doesn't use that var, temporarily clear env
                    // to reduce refcount and enable Rc::try_unwrap for in-place push.
                    let loadvar_idx = if let Expr::LoadVar { var_index } = lhs.as_ref() {
                        if !expr_uses_var(gen, *var_index) { Some(*var_index) } else { None }
                    } else { None };
                    eval(lhs, input.clone(), env, &mut |lval| {
                        match lval {
                            Value::Arr(arr_rc) => {
                                // First try direct try_unwrap
                                match Rc::try_unwrap(arr_rc) {
                                    Ok(mut vec) => {
                                        eval(gen, input.clone(), env, &mut |elem| { vec.push(elem); Ok(true) })?;
                                        cb(Value::Arr(Rc::new(vec)))
                                    }
                                    Err(arr_rc) => {
                                        // If lhs was LoadVar, DROP env's copy to reduce refcount.
                                        // Safe: the surrounding LetBinding will restore env[vi] anyway.
                                        if let Some(vi) = loadvar_idx {
                                            drop(std::mem::replace(&mut env.borrow_mut().vars[vi as usize], Value::Null));
                                            match Rc::try_unwrap(arr_rc) {
                                                Ok(mut vec) => {
                                                    eval(gen, input.clone(), env, &mut |elem| { vec.push(elem); Ok(true) })?;
                                                    cb(Value::Arr(Rc::new(vec)))
                                                }
                                                Err(arr_rc) => {
                                                    // Other references exist, fall back to clone
                                                    let mut vec = (*arr_rc).clone();
                                                    eval(gen, input.clone(), env, &mut |elem| { vec.push(elem); Ok(true) })?;
                                                    cb(Value::Arr(Rc::new(vec)))
                                                }
                                            }
                                        } else {
                                            let mut vec = (*arr_rc).clone();
                                            eval(gen, input.clone(), env, &mut |elem| { vec.push(elem); Ok(true) })?;
                                            cb(Value::Arr(Rc::new(vec)))
                                        }
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
            eval(base_expr, input.clone(), env, &mut |base| {
                eval(key_expr, input.clone(), env, &mut |key| {
                    match eval_index(&base, &key, false) {
                        Ok(v) => cb(v),
                        Err(msg) => bail!("{}", msg),
                    }
                })
            })
        }

        Expr::IndexOpt { expr: base_expr, key: key_expr } => {
            eval(base_expr, input.clone(), env, &mut |base| {
                eval(key_expr, input.clone(), env, &mut |key| {
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
                            let arr_val = std::mem::replace(&mut env.borrow_mut().vars[v_idx as usize], old);
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
                            let mut next = Value::Null;
                            eval(update, current, env, &mut |v| { next = v; Ok(true) })?;
                            current = next;
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
                                    let old = std::mem::replace(&mut env.borrow_mut().vars[vi as usize], elem);
                                    let is_true = match eval_one(body, &Value::Null, env) {
                                        Ok(v) => v.is_truthy(),
                                        Err(()) => {
                                            let mut t = false;
                                            eval(body, Value::Null, env, &mut |v| { t = v.is_truthy(); Ok(true) })?;
                                            t
                                        }
                                    };
                                    env.borrow_mut().vars[vi as usize] = old;
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
                                            let mut truthy = false;
                                            eval(predicate, elem, env, &mut |v| { truthy = v.is_truthy(); Ok(true) })?;
                                            if truthy { Ok(true) } else { all_true = false; Ok(false) }
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
                                    let old = std::mem::replace(&mut env.borrow_mut().vars[vi as usize], elem);
                                    let is_true = match eval_one(body, &Value::Null, env) {
                                        Ok(v) => v.is_truthy(),
                                        Err(()) => {
                                            let mut t = false;
                                            eval(body, Value::Null, env, &mut |v| { t = v.is_truthy(); Ok(true) })?;
                                            t
                                        }
                                    };
                                    env.borrow_mut().vars[vi as usize] = old;
                                    if is_true { any_true = true; Ok(false) } else { Ok(true) }
                                } else {
                                    let pred_result = eval_one(predicate, &elem, env);
                                    match pred_result {
                                        Ok(v) => {
                                            if v.is_truthy() { any_true = true; Ok(false) } else { Ok(true) }
                                        }
                                        Err(()) => {
                                            let mut truthy = false;
                                            eval(predicate, elem, env, &mut |v| { truthy = v.is_truthy(); Ok(true) })?;
                                            if truthy { any_true = true; Ok(false) } else { Ok(true) }
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

        Expr::TryCatch { try_expr, catch_expr } => {
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
                    if e.downcast_ref::<BreakError>().is_some() { return Err(e); }
                    let msg = format!("{}", e);
                    // halt / halt_error are non-recoverable: jq lets them
                    // propagate past `try ... catch` so the process exits with
                    // the requested code (#182).
                    if msg.starts_with("__halt__:") { return Err(e); }
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
                    _ => bail!("Cannot iterate over {} ({})",
                        container.type_name(), crate::value::value_to_json(&container)),
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
            if matches!(value.as_ref(), Expr::Input) {
                // Fast path: `. as $var | body`
                let tmp_vi = *var_index;
                if !expr_uses_outer_input(body) {
                    // Detect destructuring: body is chain of LetBinding { vi, Index(LoadVar(tmp), i) }
                    let mut bindings: Vec<(u16, usize)> = Vec::new();
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
                        let mut olds_buf: [(u16, Value); 3] = [(0, Value::Null), (0, Value::Null), (0, Value::Null)];
                        let mut olds_vec: Vec<(u16, Value)> = Vec::new();
                        let use_buf = n_bind <= 2;
                        let destructured = {
                            let mut e = env.borrow_mut();
                            let old_tmp = std::mem::replace(&mut e.vars[tmp_vi as usize], input);
                            if use_buf { olds_buf[0] = (tmp_vi, old_tmp); }
                            else { olds_vec.push((tmp_vi, old_tmp)); }
                            let rc_opt = match &e.vars[tmp_vi as usize] {
                                Value::Arr(rc) => Some(rc.clone()),
                                _ => None,
                            };
                            if let Some(rc) = rc_opt {
                                for (i, &(vi, idx)) in bindings.iter().enumerate() {
                                    let elem = rc.get(idx).cloned().unwrap_or(Value::Null);
                                    let old = std::mem::replace(&mut e.vars[vi as usize], elem);
                                    if use_buf { olds_buf[i + 1] = (vi, old); }
                                    else { olds_vec.push((vi, old)); }
                                }
                                drop(rc);
                                e.vars[tmp_vi as usize] = Value::Null;
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
                                        let (vi, old) = std::mem::replace(&mut olds_buf[i], (0, Value::Null));
                                        e.vars[vi as usize] = old;
                                    }
                                }
                                e.vars[tmp_vi as usize] = std::mem::replace(&mut olds_buf[0], (0, Value::Null)).1;
                            } else {
                                if destructured {
                                    while olds_vec.len() > 1 {
                                        let (vi, old) = olds_vec.pop().unwrap();
                                        e.vars[vi as usize] = old;
                                    }
                                }
                                e.vars[tmp_vi as usize] = olds_vec.pop().unwrap().1;
                            }
                        }
                        result
                    } else {
                        // No destructuring, normal path
                        let old = std::mem::replace(&mut env.borrow_mut().vars[tmp_vi as usize], input);
                        let result = eval(body, Value::Null, env, cb);
                        env.borrow_mut().vars[tmp_vi as usize] = old;
                        result
                    }
                } else {
                    let old = std::mem::replace(&mut env.borrow_mut().vars[tmp_vi as usize], input.clone());
                    let result = eval(body, input, env, cb);
                    env.borrow_mut().vars[tmp_vi as usize] = old;
                    result
                }
            } else if let Ok(val) = eval_one(value, &input, env) {
                // Scalar fast path: avoid closure + input clone
                let vi = *var_index as usize;
                let old = std::mem::replace(&mut env.borrow_mut().vars[vi], val);
                let result = eval(body, input, env, cb);
                env.borrow_mut().vars[vi] = old;
                result
            } else {
                eval(value, input.clone(), env, &mut |val| {
                    let vi = *var_index as usize;
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
            let update = match update.as_ref() {
                Expr::Update { path_expr, update_expr }
                    if matches!(path_expr.as_ref(), Expr::Input) => {
                        update_expr.as_ref()
                    },
                Expr::Assign { path_expr, value_expr }
                    if matches!(path_expr.as_ref(), Expr::Input) => value_expr.as_ref(),
                _ => update.as_ref(),
            };
            let mut acc = if let Ok(v) = eval_one(init, &input, env) {
                v
            } else {
                let mut a = Value::Null;
                eval(init, input.clone(), env, &mut |v| { a = v; Ok(true) })?;
                a
            };
            let vi = *var_index;
            let ai = *acc_index;
            { let mut e = env.borrow_mut(); e.ensure_var(vi); e.ensure_var(ai); }
            let acc_used_in_update = expr_uses_var(update, ai);
            // Detect fused Reduce+destructure pattern:
            // update = LetBinding(tmp, Input, LetBinding(a, Index(tmp, 0), LetBinding(b, Index(tmp, 1), inner)))
            // where inner doesn't use tmp and body doesn't use outer input.
            let fused = if !acc_used_in_update {
                if let Expr::LetBinding { var_index: tmp_vi, value, body } = update {
                    if matches!(value.as_ref(), Expr::Input) && !expr_uses_outer_input(body) {
                        let mut bindings: Vec<(u16, usize)> = Vec::new();
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
                eval(source, input.clone(), env, &mut |val| {
                    let acc_val = std::mem::replace(&mut acc, Value::Null);
                    // Setup: store $var, destructure acc, null tmp (1 borrow_mut)
                    let (old_var, old_tmp, destructured) = {
                        let mut e = env.borrow_mut();
                        let old_var = std::mem::replace(&mut e.vars[vi as usize], val);
                        let old_tmp = std::mem::replace(&mut e.vars[tmp_vi as usize], acc_val);
                        let rc_opt = match &e.vars[tmp_vi as usize] {
                            Value::Arr(rc) => Some(rc.clone()),
                            _ => None,
                        };
                        if let Some(rc) = rc_opt {
                            for &(bvi, idx) in bindings {
                                let elem = rc.get(idx).cloned().unwrap_or(Value::Null);
                                e.vars[bvi as usize] = elem;
                            }
                            drop(rc);
                            e.vars[tmp_vi as usize] = Value::Null;
                            (old_var, old_tmp, true)
                        } else {
                            (old_var, old_tmp, false)
                        }
                    };
                    if destructured {
                        eval(inner_body, Value::Null, env, &mut |new_acc| { acc = new_acc; Ok(true) })?;
                    } else {
                        // Not an array; restore tmp and run update normally
                        env.borrow_mut().vars[tmp_vi as usize] = old_tmp;
                        let acc_val_for_update = env.borrow().get_var(tmp_vi);
                        eval(update, acc_val_for_update, env, &mut |new_acc| { acc = new_acc; Ok(true) })?;
                    }
                    // Teardown: restore $var (1 borrow_mut)
                    // Note: destructured bindings' old values were whatever was in env before;
                    // since we're in a reduce loop and these are scratch vars, we just restore $var.
                    env.borrow_mut().vars[vi as usize] = old_var;
                    Ok(true)
                })?;
                cb(acc)
            } else if !acc_used_in_update {
                // Detect `. + rhs` pattern where rhs doesn't use accumulator — in-place merge
                // Also detect `+= rhs` which is: LetBinding { var, value: rhs, body: Update { path: ., update: . + LoadVar(var) } }
                let add_inplace = if let Expr::BinOp { op: BinOp::Add, lhs, rhs } = update {
                    if matches!(lhs.as_ref(), Expr::Input) && !expr_uses_outer_input(rhs) {
                        Some((rhs.as_ref(), None::<u16>))
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
                    eval(source, input.clone(), env, &mut |val| {
                        let old_var = std::mem::replace(&mut env.borrow_mut().vars[vi as usize], val);
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
                        env.borrow_mut().vars[vi as usize] = old_var;
                        Ok(true)
                    })?;
                    cb(acc)
                } else {
                    eval(source, input.clone(), env, &mut |val| {
                        let acc_val = std::mem::replace(&mut acc, Value::Null);
                        let old_var = std::mem::replace(&mut env.borrow_mut().vars[vi as usize], val);
                        eval(update, acc_val, env, &mut |new_acc| { acc = new_acc; Ok(true) })?;
                        env.borrow_mut().vars[vi as usize] = old_var;
                        Ok(true)
                    })?;
                    cb(acc)
                }
            } else {
                eval(source, input.clone(), env, &mut |val| {
                    let acc_val = std::mem::replace(&mut acc, Value::Null);
                    let (old_var, old_acc) = {
                        let mut e = env.borrow_mut();
                        let ov = std::mem::replace(&mut e.vars[vi as usize], val);
                        let oa = std::mem::replace(&mut e.vars[ai as usize], acc_val.clone());
                        (ov, oa)
                    };
                    eval(update, acc_val, env, &mut |new_acc| { acc = new_acc; Ok(true) })?;
                    {
                        let mut e = env.borrow_mut();
                        e.vars[ai as usize] = old_acc;
                        e.vars[vi as usize] = old_var;
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
                                                &mut env.borrow_mut().vars[vi as usize], val.clone());
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
                                                env.borrow_mut().vars[vi as usize] = old_var;
                                                return eval(else_branch, Value::Null, env, cb);
                                            } else {
                                                true
                                            };
                                            env.borrow_mut().vars[vi as usize] = old_var;
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
                                                    &mut env.borrow_mut().vars[vi as usize], val);
                                                let r = eval(else_branch, Value::Null, env, cb);
                                                env.borrow_mut().vars[vi as usize] = old_var;
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
                        let old_var = std::mem::replace(&mut env.borrow_mut().vars[vi as usize], val);
                        let cont = match eval_one_filter(extract_expr, &Value::Null, env) {
                            Ok(Some(v)) => cb(v)?,
                            Ok(None) => true,
                            Err(()) => eval(extract_expr, Value::Null, env, cb)?,
                        };
                        env.borrow_mut().vars[vi as usize] = old_var;
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
                        let ov = std::mem::replace(&mut e.vars[vi as usize], val);
                        let oa = if acc_used {
                            std::mem::replace(&mut e.vars[ai as usize], acc_val.clone())
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
                            env.borrow_mut().vars[ai as usize] = new_acc.clone();
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
                            e.vars[ai as usize] = old_acc;
                        }
                        e.vars[vi as usize] = old_var;
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
            let label_id = {
                let mut e = env.borrow_mut();
                let id = e.next_label;
                e.next_label = id + 1;
                e.set_var(*var_index, Value::number(id as f64));
                id
            };
            match eval(body, input, env, cb) {
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
            let mut paths = Vec::new();
            let path_result = eval_path(path_expr, input.clone(), env, &mut |p| { paths.push(p); Ok(true) });
            if let Err(e) = path_result {
                let msg = format!("{}", e);
                if let Some(json) = msg.strip_prefix("__pathexpr_result__:") {
                    bail!("Invalid path expression with result {}", json);
                }
                return Err(e);
            }
            let mut result = input.clone();
            let mut del_paths = Vec::new();
            for path in &paths {
                let old_val = crate::runtime::rt_getpath(&result, path).unwrap_or(Value::Null);
                let mut has_output = false;
                let mut new_val = old_val.clone();
                // jq `|=` takes the FIRST value the RHS emits and discards the
                // rest (`{a:1} | .a |= (1,2)` is `{a:1}`, not `{a:2}`). Returning
                // `Ok(false)` from the callback stops the generator after the
                // first hit; without it the interpreter ended up with the last
                // emitted value and diverged from JIT / fast-path on the same
                // filter (#323 self-diff caught this).
                eval(update_expr, old_val, env, &mut |v| {
                    has_output = true;
                    new_val = v;
                    Ok(false)
                })?;
                if has_output {
                    result = crate::runtime::rt_setpath(&result, path, &new_val)?;
                } else {
                    // No output = delete this path
                    del_paths.push(path.clone());
                }
            }
            if !del_paths.is_empty() {
                let dp = Value::Arr(Rc::new(del_paths));
                result = crate::runtime::rt_delpaths(&result, &dp)?;
            }
            cb(result)
        }

        Expr::Assign { path_expr, value_expr } => {
            eval(value_expr, input.clone(), env, &mut |new_val| {
                let mut paths = Vec::new();
                let path_result = eval_path(path_expr, input.clone(), env, &mut |p| { paths.push(p); Ok(true) });
                if let Err(e) = path_result {
                    let msg = format!("{}", e);
                    if let Some(json) = msg.strip_prefix("__pathexpr_result__:") {
                        bail!("Invalid path expression with result {}", json);
                    }
                    return Err(e);
                }
                let mut result = input.clone();
                for path in &paths {
                    result = crate::runtime::rt_setpath(&result, path, &new_val)?;
                }
                cb(result)
            })
        }

        Expr::PathExpr { expr: path_expr } => {
            let result = eval_path(path_expr, input, env, cb);
            match result {
                Err(e) => {
                    let msg = format!("{}", e);
                    if let Some(json) = msg.strip_prefix("__pathexpr_result__:") {
                        bail!("Invalid path expression with result {}", json);
                    }
                    Err(e)
                }
                other => other,
            }
        }

        Expr::SetPath { path, value } => {
            if !expr_uses_outer_input(path) && !expr_uses_outer_input(value) {
                // path and value don't reference `.` — avoid cloning input so
                // Rc refcount stays 1 and rt_setpath_mut can mutate in-place.
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
                eval(path, input.clone(), env, &mut |pv| {
                    eval(value, input.clone(), env, &mut |v| {
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
                Direct(Rc<CompiledFunc>, usize),
                Recursive(Rc<CompiledFunc>),
                CacheHit(Rc<Expr>),
                CacheMiss(Rc<CompiledFunc>),
            }
            let action = {
                let e = env.borrow();
                let func = match e.funcs.get(*func_id) {
                    Some(f) => f.clone(),
                    None => bail!("undefined function id {}", func_id),
                };
                if func.param_vars.is_empty() || args.is_empty() {
                    FuncAction::Direct(func, *func_id)
                } else {
                    let is_recursive = e.recursive_cache.iter()
                        .find(|(k, _)| *k == *func_id)
                        .map(|&(_, r)| r);
                    if is_recursive == Some(true) {
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
                            let all_loadvar: Option<Vec<u16>> = args.iter().map(|a| {
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
                    }
                    stacker::maybe_grow(128 * 1024, 32 * 1024 * 1024, || eval(&func.body, input, env, cb))
                }
                FuncAction::CacheHit(body) => eval(&body, input, env, cb),
                FuncAction::Recursive(func) => {
                    let mut nv = env.borrow().next_var;
                    let body = substitute_and_rename(&func.body, &func.param_vars, args, &mut nv);
                    env.borrow_mut().next_var = nv;
                    stacker::maybe_grow(128 * 1024, 32 * 1024 * 1024, || eval(&body, input, env, cb))
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
                        let mut nv = env.borrow().next_var;
                        let body = substitute_and_rename(&func.body, &func.param_vars, args, &mut nv);
                        env.borrow_mut().next_var = nv;
                        stacker::maybe_grow(128 * 1024, 32 * 1024 * 1024, || eval(&body, input, env, cb))
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
                        let all_loadvar: Option<Vec<u16>> = args.iter().map(|a| {
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
            eval_interp_parts(parts, 0, String::new(), input, env, cb)
        }

        Expr::Limit { count, generator } => {
            eval(count, input.clone(), env, &mut |cv| {
                if let Value::Num(n, NumRepr(None)) = &cv {
                    let limit = *n as i64;
                    if limit == 0 { return Ok(true); }
                    if limit < 0 {
                        bail!("__jqerror__:\"limit doesn't support negative count\"");
                    }
                    let limit = limit as usize;
                    let mut emitted = 0;
                    let mut stopped_by_outer = false;
                    let result = eval(generator, input.clone(), env, &mut |val| {
                        emitted += 1;
                        let cont = cb(val)?;
                        if !cont {
                            stopped_by_outer = true;
                            Ok(false)
                        } else if emitted >= limit {
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
                } else { bail!("limit: count must be a number") }
            })
        }

        Expr::While { cond, update } => {
            let always_true = matches!(cond.as_ref(), Expr::Literal(Literal::True));
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
                if !cb(current.clone())? { return Ok(false); }
                if let Ok(next) = eval_one(update, &current, env) {
                    current = next;
                } else {
                    let mut next = Value::Null;
                    eval(update, current, env, &mut |v| { next = v; Ok(true) })?;
                    current = next;
                }
            }
            Ok(true)
        }

        Expr::Until { cond, update } => {
            let mut current = input;
            loop {
                let is_true = if let Ok(v) = eval_one(cond, &current, env) {
                    v.is_truthy()
                } else {
                    let mut t = false;
                    eval(cond, current.clone(), env, &mut |v| { t = v.is_truthy(); Ok(true) })?;
                    t
                };
                if is_true { break; }
                if let Ok(next) = eval_one(update, &current, env) {
                    current = next;
                } else {
                    let mut next = Value::Null;
                    eval(update, current, env, &mut |v| { next = v; Ok(true) })?;
                    current = next;
                }
            }
            cb(current)
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
                    let old = std::mem::replace(&mut env.borrow_mut().vars[vi as usize], elem);
                    let is_true = match eval_one(body, &Value::Null, env) {
                        Ok(v) => v.is_truthy(),
                        Err(()) => {
                            let mut t = false;
                            eval(body, Value::Null, env, &mut |v| { t = v.is_truthy(); Ok(true) })?;
                            t
                        }
                    };
                    env.borrow_mut().vars[vi as usize] = old;
                    if is_true { Ok(true) } else { all_true = false; Ok(false) }
                } else {
                    let pred_result = eval_one(predicate, &elem, env);
                    match pred_result {
                        Ok(v) => {
                            if v.is_truthy() { Ok(true) } else { all_true = false; Ok(false) }
                        }
                        Err(()) => {
                            let mut truthy = false;
                            eval(predicate, elem, env, &mut |v| { truthy = v.is_truthy(); Ok(true) })?;
                            if truthy { Ok(true) } else { all_true = false; Ok(false) }
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
                    let old = std::mem::replace(&mut env.borrow_mut().vars[vi as usize], elem);
                    let is_true = match eval_one(body, &Value::Null, env) {
                        Ok(v) => v.is_truthy(),
                        Err(()) => {
                            let mut t = false;
                            eval(body, Value::Null, env, &mut |v| { t = v.is_truthy(); Ok(true) })?;
                            t
                        }
                    };
                    env.borrow_mut().vars[vi as usize] = old;
                    if is_true { any_true = true; Ok(false) } else { Ok(true) }
                } else {
                    let pred_result = eval_one(predicate, &elem, env);
                    match pred_result {
                        Ok(v) => {
                            if v.is_truthy() { any_true = true; Ok(false) } else { Ok(true) }
                        }
                        Err(()) => {
                            let mut truthy = false;
                            eval(predicate, elem, env, &mut |v| { truthy = v.is_truthy(); Ok(true) })?;
                            if truthy { any_true = true; Ok(false) } else { Ok(true) }
                        }
                    }
                }
            })?;
            cb(Value::from_bool(any_true))
        }

        Expr::Error { msg } => {
            if let Some(msg_expr) = msg {
                eval(msg_expr, input, env, &mut |val| {
                    bail!("__jqerror__:{}", crate::value::value_to_json_precise(&val))
                })
            } else {
                bail!("__jqerror__:{}", crate::value::value_to_json_precise(&input))
            }
        }

        Expr::Format { name, expr: fmt_expr } => {
            eval(fmt_expr, input, env, &mut |val| {
                cb(Value::from_str(&eval_format(name, &val)?))
            })
        }

        Expr::ClosureOp { op, input_expr, key_expr } => {
            eval(input_expr, input.clone(), env, &mut |container| {
                eval_closure_op(*op, &container, key_expr, &input, env, cb)
            })
        }

        Expr::RegexTest { input_expr, re, flags } => {
            eval(input_expr, input.clone(), env, &mut |s| {
                eval(re, input.clone(), env, &mut |re_val| {
                    eval(flags, input.clone(), env, &mut |fv| {
                        cb(crate::runtime::call_builtin("test", &[s.clone(), re_val.clone(), fv.clone()])?)
                    })
                })
            })
        }

        Expr::RegexMatch { input_expr, re, flags } => {
            eval(input_expr, input.clone(), env, &mut |s| {
                eval(re, input.clone(), env, &mut |re_val| {
                    eval(flags, input.clone(), env, &mut |fv| {
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
            eval(input_expr, input.clone(), env, &mut |s| {
                eval(re, input.clone(), env, &mut |re_val| {
                    eval(flags, input.clone(), env, &mut |fv| {
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
                        let mut result = String::new();
                        for seg in &segments {
                            result.push_str(&seg.literal);
                            if let Some(ref cap_obj) = seg.captures {
                                let mut replacement = Value::Null;
                                eval(tostr, cap_obj.clone(), env, &mut |tv| {
                                    replacement = tv;
                                    Ok(true)
                                })?;
                                match &replacement {
                                    Value::Str(s) => result.push_str(s),
                                    other => result.push_str(&other.to_string()),
                                }
                            }
                        }
                        cb(Value::from_string(result))
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
            eval(base_expr, input.clone(), env, &mut |base| {
                let from_val = if let Some(f) = from {
                    let mut v = Value::Null;
                    eval(f, input.clone(), env, &mut |fv| { v = fv; Ok(true) })?; v
                } else { Value::Null };
                let to_val = if let Some(t) = to {
                    let mut v = Value::Null;
                    eval(t, input.clone(), env, &mut |tv| { v = tv; Ok(true) })?; v
                } else { Value::Null };
                cb(eval_slice(&base, &from_val, &to_val)?)
            })
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
                eprintln!("[\"DEBUG:\",{}]", crate::value::value_to_json(&val));
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
                    _ => eprint!("{}", crate::value::value_to_json(&val)),
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
                Value::number((*ln as i64 % yi) as f64)
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
                Value::number((*ln as i64 % yi) as f64)
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
        UnaryOp::Exponent => "exponent", UnaryOp::Logb => "logb",
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
            if n.is_nan() { return Ok(Value::Null); }
            let idx = *n as i64;
            let i = if idx < 0 { (a.len() as i64 + idx) as usize } else { idx as usize };
            Ok(a.get(i).cloned().unwrap_or(Value::Null))
        }
        (Value::Str(_), Value::Num(_, _)) => {
            // jq's "Cannot index string with number" omits the value (#440).
            if optional {
                Err("type error".into())
            } else {
                Err("Cannot index string with number".to_string())
            }
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
    if matches!(step, Expr::Input) {
        // Default recurse (.[]): recursive descent into arrays/objects
        eval_recurse_default(val, cb)
    } else {
        // Custom step: use explicit stack to avoid stack overflow.
        // Errors raised by `step` must propagate (#195) — jq emits the
        // already-yielded values AND the type-error to stderr (exit 5).
        // The previous `let _ = eval(...)` silently dropped them.
        let mut work = vec![val.clone()];
        while let Some(current) = work.pop() {
            if !cb(current.clone())? { return Ok(false); }
            let mut next_vals = Vec::new();
            eval(step, current, env, &mut |next| {
                next_vals.push(next);
                Ok(true)
            })?;
            // Push in reverse so first output is processed first
            for v in next_vals.into_iter().rev() {
                work.push(v);
            }
        }
        Ok(true)
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
    let f = match from { Value::Num(n, _) => *n, _ => bail!("range: from must be number") };
    let t = match to { Value::Num(n, _) => *n, _ => bail!("range: to must be number") };
    let s = match step { Some(Value::Num(n, _)) => *n, Some(_) => bail!("range: step must be number"), None => 1.0 };
    if s == 0.0 { return Ok(true); }
    let mut c = f;
    if s > 0.0 { while c < t { if !cb(Value::number(c))? { return Ok(false); } c += s; } }
    else { while c > t { if !cb(Value::number(c))? { return Ok(false); } c += s; } }
    Ok(true)
}

fn object_key_from_value(kv: &Value) -> Result<KeyStr> {
    match kv {
        Value::Str(s) => Ok(KeyStr::from(s.as_str())),
        _ => bail!(
            "Cannot use {} ({}) as object key",
            kv.type_name(),
            crate::value::value_to_json(kv)
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
        Value::Str(s) => Ok(format!("'{}'", s.replace('\'', "'\\''"))),
        Value::Null => Ok("null".to_string()),
        Value::True => Ok("true".to_string()),
        Value::False => Ok("false".to_string()),
        Value::Num(n, _) => Ok(crate::value::format_jq_number(*n)),
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

pub fn eval_format(name: &str, val: &Value) -> Result<String> {
    // For csv/tsv, the input must be an array
    match name {
        "csv" => {
            let arr = match val { Value::Arr(a) => a, _ => bail!("{} cannot be csv-formatted, only array", crate::runtime::errdesc_pub(val)) };
            let mut buf = String::with_capacity(arr.len() * 16);
            for (i, v) in arr.iter().enumerate() {
                if i > 0 { buf.push(','); }
                match v {
                    Value::Str(s) => {
                        buf.push('"');
                        if s.contains('"') {
                            for c in s.chars() {
                                if c == '"' { buf.push('"'); }
                                buf.push(c);
                            }
                        } else {
                            buf.push_str(s);
                        }
                        buf.push('"');
                    }
                    Value::Null => {}
                    Value::True => buf.push_str("true"),
                    Value::False => buf.push_str("false"),
                    Value::Num(n, _) => {
                        crate::value::push_jq_number_str(&mut buf, *n);
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
        "tsv" => {
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
                                _ => buf.push(c),
                            }
                        }
                    }
                    Value::Null => {}
                    Value::True => buf.push_str("true"),
                    Value::False => buf.push_str("false"),
                    Value::Num(n, _) => crate::value::push_jq_number_str(&mut buf, *n),
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

    // For other formats, stringify the value first
    let s = match val { Value::Str(s) => s.to_string(), _ => crate::value::value_to_json(val) };
    match name {
        "text" => Ok(s),
        "json" => Ok(crate::value::value_to_json(val)),
        "html" => {
            let mut r = String::with_capacity(s.len());
            for c in s.chars() {
                match c {
                    '&' => r.push_str("&amp;"),
                    '<' => r.push_str("&lt;"),
                    '>' => r.push_str("&gt;"),
                    '\'' => r.push_str("&apos;"),
                    '"' => r.push_str("&quot;"),
                    _ => r.push(c),
                }
            }
            Ok(r)
        }
        "uri" => {
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
        "urid" => {
            let bytes = s.as_bytes();
            let mut decoded = Vec::new();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'%' && i + 2 < bytes.len() {
                    let hi = hex_val(bytes[i+1]);
                    let lo = hex_val(bytes[i+2]);
                    if let (Some(h), Some(l)) = (hi, lo) {
                        decoded.push(h * 16 + l);
                        i += 3;
                        continue;
                    }
                }
                decoded.push(bytes[i]);
                i += 1;
            }
            Ok(String::from_utf8_lossy(&decoded).into_owned())
        }
        "sh" => format_sh(val),
        "base64" => {
            const C: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let d = s.as_bytes(); let mut r = String::new();
            for ch in d.chunks(3) { let (b0,b1,b2) = (ch[0] as u32, ch.get(1).copied().unwrap_or(0) as u32, ch.get(2).copied().unwrap_or(0) as u32); let n = (b0<<16)|(b1<<8)|b2;
                r.push(C[((n>>18)&0x3f) as usize] as char); r.push(C[((n>>12)&0x3f) as usize] as char);
                r.push(if ch.len()>1 { C[((n>>6)&0x3f) as usize] as char } else { '=' }); r.push(if ch.len()>2 { C[(n&0x3f) as usize] as char } else { '=' }); }
            Ok(r)
        }
        "base64d" => {
            const D: [i8;128] = { let mut t = [-1i8;128]; let c = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; let mut i=0; while i<c.len() { t[c[i] as usize]=i as i8; i+=1; } t };
            let bs: Vec<u8> = s.bytes().filter(|&b| b!=b'\n'&&b!=b'\r'&&b!=b' ').collect();
            // jq accepts unpadded base64 of length 2 or 3 (decodes to 1 or 2
            // bytes) but rejects `len % 4 == 1` with the message
            // "string (...) trailing base64 byte found" (#186). The previous
            // loop silently truncated `len % 4 == 1` cases via `if ch.len()<2 { break; }`.
            if bs.len() % 4 == 1 {
                bail!("string ({}) trailing base64 byte found", crate::value::value_to_json(val));
            }
            let mut r = Vec::new();
            for ch in bs.chunks(4) {
                let a=D.get(ch[0] as usize).copied().unwrap_or(-1);
                let b=D.get(ch[1] as usize).copied().unwrap_or(-1);
                if a<0||b<0 { bail!("string ({}) is not valid base64 data", crate::value::value_to_json(val)); }
                r.push(((a as u8)<<2)|((b as u8)>>4));
                if ch.len()>2 && ch[2]!=b'=' {
                    let c=D.get(ch[2] as usize).copied().unwrap_or(-1);
                    if c<0 { bail!("string ({}) is not valid base64 data", crate::value::value_to_json(val)); }
                    r.push(((b as u8)<<4)|((c as u8)>>2));
                    if ch.len()>3 && ch[3]!=b'=' {
                        let d=D.get(ch[3] as usize).copied().unwrap_or(-1);
                        if d<0 { bail!("string ({}) is not valid base64 data", crate::value::value_to_json(val)); }
                        r.push(((c as u8)<<6)|(d as u8));
                    }
                }
            }
            Ok(String::from_utf8_lossy(&r).into_owned())
        }
        _ => bail!("unknown format: @{}", name),
    }
}

fn slice_index_start(n: f64, len: i64) -> usize {
    if n.is_nan() { return 0; }
    let i = n.floor() as i64;
    if i < 0 { (len + i).max(0) as usize } else { i.min(len) as usize }
}

fn slice_index_end(n: f64, len: i64) -> usize {
    if n.is_nan() { return len as usize; }
    let i = n.ceil() as i64;
    if i < 0 { (len + i).max(0) as usize } else { i.min(len) as usize }
}

pub fn eval_slice(base: &Value, from: &Value, to: &Value) -> Result<Value> {
    match base {
        Value::Arr(a) => {
            let len = a.len() as i64;
            let fi = match from { Value::Num(n, _) => slice_index_start(*n, len), Value::Null => 0, _ => bail!("slice: need number") };
            let ti = match to { Value::Num(n, _) => slice_index_end(*n, len), Value::Null => len as usize, _ => bail!("slice: need number") };
            Ok(if fi>=ti { Value::Arr(Rc::new(vec![])) } else { Value::Arr(Rc::new(a[fi..ti].to_vec())) })
        }
        Value::Str(s) => {
            let s_str = s.as_str();
            // ASCII fast path: byte index == char index, no allocation needed
            if s_str.is_ascii() {
                let len = s_str.len() as i64;
                let fi = match from { Value::Num(n, _) => slice_index_start(*n, len), Value::Null => 0, _ => bail!("slice: need number") };
                let ti = match to { Value::Num(n, _) => slice_index_end(*n, len), Value::Null => len as usize, _ => bail!("slice: need number") };
                Ok(if fi>=ti { Value::from_str("") } else { Value::from_str(&s_str[fi..ti]) })
            } else {
                // Unicode: count chars without allocation, use char_indices for byte offsets
                let char_count = s_str.chars().count() as i64;
                let fi = match from { Value::Num(n, _) => slice_index_start(*n, char_count), Value::Null => 0, _ => bail!("slice: need number") };
                let ti = match to { Value::Num(n, _) => slice_index_end(*n, char_count), Value::Null => char_count as usize, _ => bail!("slice: need number") };
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
            indexed.sort_by(|(_, ka), (_, kb)| {
                if let (Value::Num(a, _), Value::Num(b, _)) = (ka, kb) {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    crate::runtime::compare_values(ka, kb)
                }
            });
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
    let a = match container { Value::Arr(a) => a, _ => bail!("Cannot iterate over {}", crate::runtime::errdesc_pub(container)) };

    // Fast path: f64 key extraction — avoids eval overhead and Vec<Value> allocations
    if !a.is_empty() {
        if let Some(first_key) = try_eval_key_f64(key_expr, &a[0]) {
            let mut f64_keys: Vec<f64> = Vec::with_capacity(a.len());
            f64_keys.push(first_key);
            let mut all_f64 = true;
            for item in &a[1..] {
                if let Some(k) = try_eval_key_f64(key_expr, item) {
                    f64_keys.push(k);
                } else {
                    all_f64 = false;
                    break;
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

    let mut keyed: Vec<(Vec<Value>, Value)> = Vec::new();
    for item in a.iter() {
        let mut keys = Vec::new();
        eval(key_expr, item.clone(), env, &mut |k| { keys.push(k); Ok(true) })?;
        keyed.push((keys, item.clone()));
    }
    match op {
        ClosureOpKind::SortBy => {
            keyed.sort_by(|(ka, _), (kb, _)| { ka.iter().zip(kb.iter()).map(|(a, b)| crate::runtime::compare_values(a, b)).find(|o| *o != std::cmp::Ordering::Equal).unwrap_or(std::cmp::Ordering::Equal) });
            cb(Value::Arr(Rc::new(keyed.into_iter().map(|(_, v)| v).collect())))
        }
        ClosureOpKind::GroupBy => {
            keyed.sort_by(|(ka, _), (kb, _)| { ka.iter().zip(kb.iter()).map(|(a, b)| crate::runtime::compare_values(a, b)).find(|o| *o != std::cmp::Ordering::Equal).unwrap_or(std::cmp::Ordering::Equal) });
            let mut groups: Vec<Value> = Vec::new(); let mut cg: Vec<Value> = Vec::new(); let mut ck: Option<Vec<Value>> = None;
            for (keys, val) in keyed {
                if let Some(ref pk) = ck { if keys.len()==pk.len()&&keys.iter().zip(pk.iter()).all(|(a,b)| crate::runtime::values_equal(a,b)) { cg.push(val); } else { groups.push(Value::Arr(Rc::new(std::mem::take(&mut cg)))); cg.push(val); ck=Some(keys); } } else { cg.push(val); ck=Some(keys); }
            }
            if !cg.is_empty() { groups.push(Value::Arr(Rc::new(cg))); }
            cb(Value::Arr(Rc::new(groups)))
        }
        ClosureOpKind::UniqueBy => {
            // jq: unique_by(f) = group_by(f) | map(.[0]) — sorted by key, deduped.
            keyed.sort_by(|(ka, _), (kb, _)| { ka.iter().zip(kb.iter()).map(|(a, b)| crate::runtime::compare_values(a, b)).find(|o| *o != std::cmp::Ordering::Equal).unwrap_or(std::cmp::Ordering::Equal) });
            let mut result: Vec<Value> = Vec::new();
            let mut prev: Option<Vec<Value>> = None;
            for (keys, val) in keyed {
                let is_dup = match &prev {
                    Some(pk) => pk.len() == keys.len() && pk.iter().zip(keys.iter()).all(|(a, b)| crate::runtime::values_equal(a, b)),
                    None => false,
                };
                if !is_dup {
                    result.push(val);
                    prev = Some(keys);
                }
            }
            cb(Value::Arr(Rc::new(result)))
        }
        ClosureOpKind::MinBy => {
            if keyed.is_empty() { cb(Value::Null) } else {
                let mut mi = 0; for i in 1..keyed.len() { if keyed[i].0.iter().zip(keyed[mi].0.iter()).map(|(a,b)| crate::runtime::compare_values(a,b)).find(|o|*o!=std::cmp::Ordering::Equal).unwrap_or(std::cmp::Ordering::Equal) == std::cmp::Ordering::Less { mi=i; } }
                cb(keyed[mi].1.clone())
            }
        }
        ClosureOpKind::MaxBy => {
            if keyed.is_empty() { cb(Value::Null) } else {
                let mut mi = 0; for i in 1..keyed.len() {
                    let cmp = keyed[i].0.iter().zip(keyed[mi].0.iter()).map(|(a,b)| crate::runtime::compare_values(a,b)).find(|o|*o!=std::cmp::Ordering::Equal).unwrap_or(std::cmp::Ordering::Equal);
                    if cmp == std::cmp::Ordering::Greater || cmp == std::cmp::Ordering::Equal { mi=i; }
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
        Err(e) => {
            let msg = format!("{}", e);
            if let Some(json) = msg.strip_prefix("__pathexpr_result__:") {
                bail!("Invalid path expression with result {}", json);
            }
            Err(e)
        }
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

fn eval_interp_parts(parts: &[StringPart], idx: usize, cur: String, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    if idx >= parts.len() { return cb(Value::from_str(&cur)); }
    match &parts[idx] {
        StringPart::Literal(s) => { let mut n = cur; n.push_str(s); eval_interp_parts(parts, idx+1, n, input, env, cb) }
        StringPart::Expr(e) => {
            eval(e, input.clone(), env, &mut |val| {
                let s = match &val { Value::Str(s) => s.to_string(), _ => crate::value::value_to_json(&val) };
                let mut n = cur.clone(); n.push_str(&s);
                eval_interp_parts(parts, idx+1, n, input.clone(), env, cb)
            })
        }
    }
}

fn eval_path(expr: &Expr, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    match expr {
        Expr::Input => cb(Value::Arr(Rc::new(vec![]))),
        Expr::Index { expr: be, key: ke } => {
            let cb_called = std::cell::Cell::new(false);
            let input_for_check = input.clone();
            let result = eval_path(be, input.clone(), env, &mut |bp| {
                cb_called.set(true);
                eval(ke, input.clone(), env, &mut |key| {
                    // jq errors `path(.field)` when the base value at the
                    // current path can't accept the key type (issue #46).
                    // Only objects (with string keys), arrays (with number
                    // keys), and null (a no-op) are valid bases.
                    let base_val = crate::runtime::rt_getpath(&input_for_check, &bp).unwrap_or(Value::Null);
                    match (&base_val, &key) {
                        (Value::Obj(_), Value::Str(_)) => {}
                        (Value::Arr(_), Value::Num(_, _)) => {}
                        (Value::Null, _) => {}
                        _ => {
                            let key_desc = match &key {
                                Value::Str(s) => format!("string \"{}\"", s),
                                Value::Num(n, _) => format!("number ({})", crate::value::format_jq_number(*n)),
                                other => format!("{} ({})", other.type_name(), crate::value::value_to_json(other)),
                            };
                            bail!("Cannot index {} with {}", base_val.type_name(), key_desc);
                        }
                    }
                    let mut p = match &bp { Value::Arr(a) => a.as_ref().clone(), _ => vec![] };
                    p.push(key); cb(Value::Arr(Rc::new(p)))
                })
            });
            match result {
                Err(e) if !cb_called.get() => {
                    let msg = format!("{}", e);
                    if let Some(json) = msg.strip_prefix("__pathexpr_result__:") {
                        let mut key_val = Value::Null;
                        let _ = eval(ke, input, env, &mut |k| { key_val = k; Ok(true) });
                        let key_desc = match &key_val {
                            Value::Num(n, _) => format!("element {} of", crate::value::format_jq_number(*n)),
                            Value::Str(s) => format!("element \"{}\" of", s),
                            _ => format!("element {} of", crate::value::value_to_json(&key_val)),
                        };
                        bail!("Invalid path expression near attempt to access {} {}", key_desc, json);
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
                        // "no paths" turned type errors into no-ops.
                        bail!("Cannot iterate over {} ({})", base.type_name(), crate::value::value_to_json(&base))
                    }
                }
            });
            match result {
                Err(e) if !cb_called.get() => {
                    let msg = format!("{}", e);
                    if let Some(json) = msg.strip_prefix("__pathexpr_result__:") {
                        bail!("Invalid path expression near attempt to iterate through {}", json);
                    }
                    Err(e)
                }
                other => other,
            }
        }
        Expr::Pipe { left, right } => {
            let result = eval_path(left, input.clone(), env, &mut |lp| {
                let mid = crate::runtime::rt_getpath(&input, &lp).unwrap_or(Value::Null);
                eval_path(right, mid.clone(), env, &mut |rp| {
                    let mut p = match &lp { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] };
                    if let Value::Arr(rpa) = &rp { p.extend(rpa.iter().cloned()); }
                    cb(Value::Arr(Rc::new(p)))
                })
            });
            match result {
                Err(e) => {
                    let msg = format!("{}", e);
                    if let Some(json) = msg.strip_prefix("__pathexpr_result__:") {
                        match right.as_ref() {
                            Expr::Index { key, .. } | Expr::IndexOpt { key, .. } => {
                                let mut key_val = Value::Null;
                                let _ = eval(key, input, env, &mut |k| { key_val = k; Ok(true) });
                                let key_desc = match &key_val {
                                    Value::Num(n, _) => format!("element {} of", crate::value::format_jq_number(*n)),
                                    Value::Str(s) => format!("element \"{}\" of", s),
                                    _ => format!("element {} of", crate::value::value_to_json(&key_val)),
                                };
                                bail!("Invalid path expression near attempt to access {} {}", key_desc, json);
                            }
                            Expr::Each { .. } | Expr::EachOpt { .. } => {
                                bail!("Invalid path expression near attempt to iterate through {}", json);
                            }
                            _ => {
                                // Pass __pathexpr_result__ through for higher-level handlers
                                Err(e)
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
        Expr::Recurse { .. } => eval_recurse_paths(&input, &Value::Arr(Rc::new(vec![])), cb),
        Expr::LetBinding { var_index, value, body } => {
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
        Expr::Slice { expr: base_expr, from, to } => {
            eval_path(base_expr, input.clone(), env, &mut |bp| {
                // Type-check the receiver: jq errors on path slicing of non-array/string/null.
                let base = crate::runtime::rt_getpath(&input, &bp).unwrap_or(Value::Null);
                match &base {
                    Value::Arr(_) | Value::Str(_) | Value::Null => {}
                    other => bail!("Cannot index {} with object", other.type_name()),
                }
                let from_val = if let Some(f) = from {
                    let mut v = Value::Null;
                    eval(f, input.clone(), env, &mut |fv| { v = fv; Ok(true) })?; v
                } else { Value::Null };
                let to_val = if let Some(t) = to {
                    let mut v = Value::Null;
                    eval(t, input.clone(), env, &mut |tv| { v = tv; Ok(true) })?; v
                } else { Value::Null };
                // jq preserves the literal slice expressions in path output without
                // clamping to the receiver's actual length. Omitted bounds → null.
                let mut p = match &bp { Value::Arr(a) => a.as_ref().clone(), _ => vec![] };
                p.push(Value::object_from_map({
                    let mut m = crate::value::new_objmap();
                    m.insert("start".into(), from_val);
                    m.insert("end".into(), to_val);
                    m
                }));
                cb(Value::Arr(Rc::new(p)))
            })
        }
        Expr::CallBuiltin { name, args } if name == "getpath" && args.len() == 1 => {
            // In path context, getpath(p) = the path p itself
            eval(&args[0], input, env, cb)
        }
        Expr::GetPath { path } => {
            // In path context, getpath(p) = the path p itself
            eval(path, input, env, cb)
        }
        Expr::FuncCall { func_id, .. } => {
            let func = env.borrow().funcs.get(*func_id).cloned();
            if let Some(f) = func {
                eval_path(&f.body, input, env, cb)
            } else {
                bail!("undefined function id {}", func_id)
            }
        }
        Expr::TryCatch { try_expr, catch_expr } => {
            let result = eval_path(try_expr, input.clone(), env, cb);
            match result {
                Ok(cont) => Ok(cont),
                Err(e) => {
                    if e.downcast_ref::<BreakError>().is_some() { return Err(e); }
                    let msg = format!("{}", e);
                    // halt / halt_error are non-recoverable: jq lets them
                    // propagate past `try ... catch` so the process exits with
                    // the requested code (#182).
                    if msg.starts_with("__halt__:") { return Err(e); }
                    let catch_val = if let Some(json) = msg.strip_prefix("__jqerror__:") {
                        crate::value::json_to_value(json).unwrap_or(Value::from_str(&msg))
                    } else {
                        Value::from_str(&msg)
                    };
                    eval(catch_expr, catch_val, env, cb)
                }
            }
        }
        Expr::IndexOpt { expr: be, key: ke } => {
            eval_path(be, input.clone(), env, &mut |bp| {
                eval(ke, input.clone(), env, &mut |key| {
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
                bail!("__pathexpr_result__:{}", crate::value::value_to_json(&result_val));
            }
            Ok(true)
        }
    }
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

fn eval_call_builtin(name: &str, args: &[Expr], input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    // Special handling for builtins that take filter/closure arguments
    match (name, args.len()) {
        ("input_line_number", 0) => {
            return cb(Value::number(get_input_line_number() as f64));
        }
        ("toboolean", 0) => {
            return cb(rt_toboolean(&input)?);
        }
        ("halt", 0) => {
            // halt: terminate with status 0 after emitting any values the
            // preceding generator already yielded. Raising a sentinel
            // error lets the CLI flush its buffered stdout before exiting
            // (see `__halt__:` handling in bin/jq-jit.rs).
            bail!("__halt__:0");
        }
        ("halt_error", 0) => {
            halt_error_write(&input);
            bail!("__halt__:5");
        }
        ("halt_error", 1) => {
            return eval(&args[0], input.clone(), env, &mut |code_val| {
                let code = match &code_val {
                    Value::Num(n, _) => *n as i32,
                    _ => bail!(
                        "{} halt_error/1: number required",
                        crate::runtime::errdesc_pub(&input)
                    ),
                };
                halt_error_write(&input);
                bail!("__halt__:{}", code);
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
                let fmt_name = match &fmt_val {
                    Value::Str(s) => s.as_str().to_string(),
                    _ => bail!("{} is not a valid format", crate::value::value_to_json(&fmt_val)),
                };
                cb(Value::from_str(&eval_format(&fmt_name, &input)?))
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
    // Default: evaluate args as generators and call runtime with input + args
    eval_call_builtin_args(name, args, 0, vec![input.clone()], input, env, cb)
}

/// Yield each Cartesian-product combination of an array of arrays, in
/// lexicographic order. Empty input array yields a single empty
/// combination (matching jq).
fn eval_combinations(input: &Value, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    let arrays = match input {
        Value::Arr(a) => a.clone(),
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
            cb(Value::Arr(Rc::new(current.clone())))?;
            return Ok(true);
        }
        let inner = match &arrays[idx] {
            Value::Arr(a) => a.clone(),
            _ => bail!("combinations: each element must be an array"),
        };
        for v in inner.iter() {
            current.push(v.clone());
            rec(arrays, idx + 1, current, cb)?;
            current.pop();
        }
        Ok(true)
    }
    rec(&arrays, 0, &mut current, cb)
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
            let json = crate::value::value_to_json_precise(input);
            let _ = stderr.write_all(json.as_bytes());
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

fn eval_call_builtin_args(name: &str, args: &[Expr], idx: usize, collected: Vec<Value>, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    if idx >= args.len() {
        return cb(crate::runtime::call_builtin(name, &collected)?);
    }
    eval(&args[idx], input.clone(), env, &mut |val| {
        let mut next = collected.clone();
        next.push(val);
        eval_call_builtin_args(name, args, idx + 1, next, input.clone(), env, cb)
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
    // Collect all paths generated by f (as path arrays)
    let mut paths: Vec<Value> = Vec::new();
    eval_path(f, input.clone(), env, &mut |path| {
        paths.push(path);
        Ok(true)
    })?;
    // Build result by setting each path
    let mut result = Value::Null;
    for path in &paths {
        let val = crate::runtime::rt_getpath(&input, path)?;
        result = crate::runtime::rt_setpath(&result, path, &val)?;
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
    if let Some((type_name, then_body)) = detect_walk_type_guard(f) {
        return walk_type_guarded(type_name, then_body, f, input, env, cb);
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
                                        let mut idx = *n as i64;
                                        if idx < 0 { idx += len; }
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
                        let from_idx = eval_slice_idx_val(from, len, 0, &input, env)?;
                        let to_idx = eval_slice_idx_val(to, len, len, &input, env)?;
                        for i in from_idx..to_idx {
                            if i >= 0 && i < len {
                                indices_to_del.insert(i);
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

    // Fallback: apply operations sequentially
    let mut result = input.clone();
    for op in &del_ops {
        result = apply_del_op(op, result, env)?;
    }
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

fn apply_del_op(op: &DelOp, current: Value, env: &EnvRef) -> Result<Value> {
    match op {
        DelOp::Path(expr) => {
            let mut paths: Vec<Value> = Vec::new();
            eval_path(expr, current.clone(), env, &mut |path| {
                paths.push(path);
                Ok(true)
            })?;
            let path_arr = Value::Arr(Rc::new(paths));
            crate::runtime::rt_delpaths(&current, &path_arr)
        }
        DelOp::Slice { base, from, to } => {
            let mut container = Value::Null;
            eval(base, current.clone(), env, &mut |v| { container = v; Ok(true) })?;
            let new_val = match &container {
                Value::Arr(a) => {
                    let len = a.len() as i64;
                    let from_idx = eval_slice_idx_val(from, len, 0, &current, env)?;
                    let to_idx = eval_slice_idx_val(to, len, len, &current, env)?;
                    let mut result = Vec::new();
                    for i in 0..from_idx.min(len) { result.push(a[i as usize].clone()); }
                    for i in to_idx.max(0)..len { result.push(a[i as usize].clone()); }
                    Value::Arr(Rc::new(result))
                }
                _ => container,
            };
            if matches!(base, Expr::Input) {
                Ok(new_val)
            } else {
                let mut base_paths: Vec<Value> = Vec::new();
                let _ = eval_path(base, current.clone(), env, &mut |p| {
                    base_paths.push(p);
                    Ok(false)
                });
                if let Some(bp) = base_paths.first() {
                    crate::runtime::rt_setpath(&current, bp, &new_val)
                } else {
                    Ok(new_val)
                }
            }
        }
    }
}

fn eval_slice_idx_val(expr: &Option<&Expr>, len: i64, default: i64, input: &Value, env: &EnvRef) -> Result<i64> {
    if let Some(e) = expr {
        let mut val = default;
        eval(e, input.clone(), env, &mut |v| {
            if let Value::Num(n, _) = &v { val = *n as i64; }
            Ok(true)
        })?;
        if val < 0 { Ok((len + val).max(0)) } else { Ok(val.min(len)) }
    } else {
        Ok(default)
    }
}

// bsearch(target): binary search on sorted array
fn rt_bsearch(input: &Value, target: &Value) -> Result<Value> {
    match input {
        Value::Arr(a) => {
            let mut lo: i64 = 0;
            let mut hi: i64 = a.len() as i64 - 1;
            while lo <= hi {
                let mid = (lo + hi) / 2;
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
