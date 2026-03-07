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
use crate::value::{Value, KeyStr};

type GenResult = Result<bool>;
type EnvRef = Rc<RefCell<Env>>;

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
    recursive_cache: HashMap<usize, bool>,
    /// Cache for substituted function bodies: (func_id, arg_var_indices) → substituted body.
    /// Only used when all args are LoadVar (the common case).
    subst_cache: HashMap<(usize, Vec<u16>), Rc<Expr>>,
    /// Pointer-based substitution cache: func_id → (args_ptr, substituted_body).
    /// For non-LoadVar args from stable (cached) call sites.
    subst_ptr_cache: HashMap<usize, (usize, Rc<Expr>)>,
}

impl Env {
    pub fn new(funcs: Vec<CompiledFunc>) -> Self {
        Env { vars: vec![Value::Null; 4096], funcs: funcs.into_iter().map(Rc::new).collect(), next_label: 0, next_var: 256, lib_dirs: Vec::new(), closures: Vec::new(), recursive_cache: HashMap::new(), subst_cache: HashMap::new(), subst_ptr_cache: HashMap::new() }
    }
    pub fn with_lib_dirs(funcs: Vec<CompiledFunc>, lib_dirs: Vec<String>) -> Self {
        Env { vars: vec![Value::Null; 4096], funcs: funcs.into_iter().map(Rc::new).collect(), next_label: 0, next_var: 256, lib_dirs, closures: Vec::new(), recursive_cache: HashMap::new(), subst_cache: HashMap::new(), subst_ptr_cache: HashMap::new() }
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
        // Conservative: assume these use input
        Expr::FuncCall { .. } | Expr::CallBuiltin { .. } | Expr::ObjectConstruct { .. }
        | Expr::StringInterpolation { .. } | Expr::Label { .. } | Expr::Break { .. }
        | Expr::Error { .. } | Expr::ClosureOp { .. } | Expr::Slice { .. }
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

/// Extract f64 from a leaf expression (LoadVar, Input, Literal) without cloning.
/// Used by the zero-clone BinOp fast path.
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
            Literal::Num(n, repr) => Value::Num(*n, repr.clone()),
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
                                    BinOp::Add => Value::Num(ln + rn, None),
                                    BinOp::Sub => Value::Num(ln - rn, None),
                                    BinOp::Mul => Value::Num(ln * rn, None),
                                    BinOp::Div => {
                                        if rn == 0.0 { drop(e); return Err(()); }
                                        Value::Num(ln / rn, None)
                                    }
                                    BinOp::Mod => {
                                        let yi = rn as i64;
                                        if yi == 0 { drop(e); return Err(()); }
                                        Value::Num((ln as i64 % yi) as f64, None)
                                    }
                                    BinOp::Eq => if ln == rn { Value::True } else { Value::False },
                                    BinOp::Ne => if ln != rn { Value::True } else { Value::False },
                                    BinOp::Lt => if ln < rn { Value::True } else { Value::False },
                                    BinOp::Gt => if ln > rn { Value::True } else { Value::False },
                                    BinOp::Le => if ln <= rn { Value::True } else { Value::False },
                                    BinOp::Ge => if ln >= rn { Value::True } else { Value::False },
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
                            BinOp::Add => Value::Num(ln + rn, None),
                            BinOp::Sub => Value::Num(ln - rn, None),
                            BinOp::Mul => Value::Num(ln * rn, None),
                            BinOp::Div => {
                                if *rn == 0.0 { return eval_binop(*op, &l, &r).map_err(|_| ()); }
                                Value::Num(ln / rn, None)
                            }
                            BinOp::Mod => {
                                let yi = *rn as i64;
                                if yi == 0 { return eval_binop(*op, &l, &r).map_err(|_| ()); }
                                Value::Num((*ln as i64 % yi) as f64, None)
                            }
                            BinOp::Eq => if ln == rn { Value::True } else { Value::False },
                            BinOp::Ne => if ln != rn { Value::True } else { Value::False },
                            BinOp::Lt => if ln < rn { Value::True } else { Value::False },
                            BinOp::Gt => if ln > rn { Value::True } else { Value::False },
                            BinOp::Le => if ln <= rn { Value::True } else { Value::False },
                            BinOp::Ge => if ln >= rn { Value::True } else { Value::False },
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
                    UnaryOp::Floor => Value::Num(n.floor(), None),
                    UnaryOp::Ceil => Value::Num(n.ceil(), None),
                    UnaryOp::Round => Value::Num(n.round(), None),
                    UnaryOp::Fabs | UnaryOp::Abs => Value::Num(n.abs(), None),
                    UnaryOp::Length => Value::Num(n.abs(), None),
                    UnaryOp::Sqrt => Value::Num(n.sqrt(), None),
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
            let base = eval_one(base_expr, input, env)?;
            let key = eval_one(key_expr, input, env)?;
            Ok(eval_index(&base, &key, true).unwrap_or(Value::Null))
        }
        Expr::Negate { operand } => {
            let val = eval_one(operand, input, env)?;
            match val {
                Value::Num(n, _) => Ok(Value::Num(-n, None)),
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
            let val = eval_one(value, input, env)?;
            let old = env.borrow().get_var(*var_index);
            env.borrow_mut().set_var(*var_index, val);
            let result = eval_one(body, input, env);
            env.borrow_mut().set_var(*var_index, old);
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
            Literal::Num(n, repr) => Value::Num(*n, repr.clone()),
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
                        match eval_one_filter(right, &current, env) {
                            Ok(Some(result)) => { if !cb(result)? { return Ok(false); } }
                            Ok(None) => { /* filtered out (select/empty), skip */ }
                            Err(()) => {
                                if !eval(right, current.clone(), env, cb)? { return Ok(false); }
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
                            eval(generator, input, env, &mut |elem| {
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
                            })?;
                            cb(Value::from_bool(all_true))
                        }
                        Expr::AnyShort { generator: any_gen, predicate }
                            if matches!(any_gen.as_ref(), Expr::Each { input_expr } if matches!(input_expr.as_ref(), Expr::Input)) =>
                        {
                            let mut any_true = false;
                            eval(generator, input, env, &mut |elem| {
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
                    Value::Obj(o) => {
                        for v in o.values() {
                            if !cb(v.clone())? { return Ok(false); }
                        }
                        Ok(true)
                    }
                    Value::Null => Ok(true),
                    _ => bail!("{} ({}) is not iterable",
                        container.type_name(), crate::value::value_to_json(&container)),
                }
            })
        }

        Expr::EachOpt { input_expr } => {
            eval(input_expr, input, env, &mut |container| {
                match &container {
                    Value::Arr(a) => { for v in a.iter() { if !cb(v.clone())? { return Ok(false); } } Ok(true) }
                    Value::Obj(o) => { for v in o.values() { if !cb(v.clone())? { return Ok(false); } } Ok(true) }
                    _ => Ok(true),
                }
            })
        }

        Expr::LetBinding { var_index, value, body } => {
            if matches!(value.as_ref(), Expr::Input) {
                // Fast path: `. as $var | body`
                let old = env.borrow().get_var(*var_index);
                let tmp_vi = *var_index;
                if !expr_uses_outer_input(body) {
                    // Body doesn't use `.` — move input into env (sole owner, lower refcount)
                    env.borrow_mut().set_var(tmp_vi, input);
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
                    let result = if !bindings.is_empty() && !expr_uses_var(inner, tmp_vi) {
                        // Destructuring detected. Extract elements, then clear tmp to release tuple.
                        let arr_val = env.borrow().get_var(tmp_vi);
                        if let Value::Arr(ref arr_rc) = arr_val {
                            let mut olds: Vec<(u16, Value)> = Vec::with_capacity(bindings.len());
                            for &(vi, idx) in &bindings {
                                let elem_old = env.borrow().get_var(vi);
                                let elem = arr_rc.get(idx).cloned().unwrap_or(Value::Null);
                                env.borrow_mut().set_var(vi, elem);
                                olds.push((vi, elem_old));
                            }
                            // Clear tmp to release tuple references → element refcounts drop by 1
                            drop(arr_val);
                            env.borrow_mut().vars[tmp_vi as usize] = Value::Null;
                            let result = eval(inner, Value::Null, env, cb);
                            for (vi, elem_old) in olds.into_iter().rev() {
                                env.borrow_mut().set_var(vi, elem_old);
                            }
                            result
                        } else {
                            drop(arr_val);
                            eval(body, Value::Null, env, cb)
                        }
                    } else {
                        eval(body, Value::Null, env, cb)
                    };
                    env.borrow_mut().set_var(tmp_vi, old);
                    result
                } else {
                    env.borrow_mut().set_var(tmp_vi, input.clone());
                    let result = eval(body, input, env, cb);
                    env.borrow_mut().set_var(tmp_vi, old);
                    result
                }
            } else if let Ok(val) = eval_one(value, &input, env) {
                // Scalar fast path: avoid closure + input clone
                let old = env.borrow().get_var(*var_index);
                env.borrow_mut().set_var(*var_index, val);
                let result = eval(body, input, env, cb);
                env.borrow_mut().set_var(*var_index, old);
                result
            } else {
                eval(value, input.clone(), env, &mut |val| {
                    let old = env.borrow().get_var(*var_index);
                    env.borrow_mut().set_var(*var_index, val);
                    let result = eval(body, input.clone(), env, cb);
                    env.borrow_mut().set_var(*var_index, old);
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
            eval(source, input.clone(), env, &mut |val| {
                let acc_val = std::mem::replace(&mut acc, Value::Null);
                if acc_used_in_update {
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
                } else {
                    let old_var = std::mem::replace(&mut env.borrow_mut().vars[vi as usize], val);
                    eval(update, acc_val, env, &mut |new_acc| { acc = new_acc; Ok(true) })?;
                    env.borrow_mut().vars[vi as usize] = old_var;
                }
                Ok(true)
            })?;
            cb(acc)
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
                                    return eval(source, input, env, &mut |val| {
                                        // Store val in env for any nested $item access in cond_body
                                        let old_var = std::mem::replace(
                                            &mut env.borrow_mut().vars[vi as usize], val.clone());
                                        // Evaluate cond with val as input (replaces Pipe(LoadVar($item), cond_body))
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
                                            cb(val)?  // Output val directly, skip LoadVar re-read
                                        } else if else_is_break {
                                            env.borrow_mut().vars[vi as usize] = old_var;
                                            // Evaluate break to trigger label unwinding
                                            return eval(else_branch, Value::Null, env, cb);
                                        } else {
                                            true // Empty: no output
                                        };
                                        env.borrow_mut().vars[vi as usize] = old_var;
                                        Ok(cont)
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
                    if acc_used {
                        let (old_var, old_acc) = {
                            let mut e = env.borrow_mut();
                            let ov = std::mem::replace(&mut e.vars[vi as usize], val);
                            let oa = std::mem::replace(&mut e.vars[ai as usize], acc_val.clone());
                            (ov, oa)
                        };
                        eval(update, acc_val, env, &mut |new_acc| { acc = new_acc; Ok(true) })?;
                        let cont = if let Some(extract_expr) = extract {
                            match eval_one_filter(extract_expr, &acc, env) {
                                Ok(Some(v)) => cb(v)?,
                                Ok(None) => true,
                                Err(()) => eval(extract_expr, acc.clone(), env, cb)?,
                            }
                        } else {
                            cb(acc.clone())?
                        };
                        {
                            let mut e = env.borrow_mut();
                            e.vars[ai as usize] = old_acc;
                            e.vars[vi as usize] = old_var;
                        }
                        Ok(cont)
                    } else {
                        let old_var = std::mem::replace(&mut env.borrow_mut().vars[vi as usize], val);
                        eval(update, acc_val, env, &mut |new_acc| { acc = new_acc; Ok(true) })?;
                        let cont = if let Some(extract_expr) = extract {
                            match eval_one_filter(extract_expr, &acc, env) {
                                Ok(Some(v)) => cb(v)?,
                                Ok(None) => true,
                                Err(()) => eval(extract_expr, acc.clone(), env, cb)?,
                            }
                        } else {
                            cb(acc.clone())?
                        };
                        env.borrow_mut().vars[vi as usize] = old_var;
                        Ok(cont)
                    }
                })
            })
        }

        Expr::Alternative { primary, fallback } => {
            let mut has_output = false;
            let result = eval(primary, input.clone(), env, &mut |val| {
                if val.is_truthy() { has_output = true; cb(val) } else { Ok(true) }
            });
            match result {
                Ok(_) if !has_output => eval(fallback, input, env, cb),
                Ok(cont) => Ok(cont),
                Err(_) => eval(fallback, input, env, cb),
            }
        }

        Expr::Negate { operand } => {
            eval(operand, input, env, &mut |val| {
                match &val {
                    Value::Num(n, _) => cb(Value::Num(-n, None)),
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
                e.set_var(*var_index, Value::Num(id as f64, None));
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
            if let Value::Num(n, None) = &label {
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
                eval(update_expr, old_val, env, &mut |v| { has_output = true; new_val = v; Ok(true) })?;
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
            eval(path, input.clone(), env, &mut |pv| {
                eval(value, input.clone(), env, &mut |v| {
                    cb(crate::runtime::rt_setpath(&input, &pv, &v)?)
                })
            })
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
            let func = env.borrow().funcs.get(*func_id).cloned();
            if let Some(f) = func {
                if f.param_vars.is_empty() || args.is_empty() {
                    eval(&f.body, input, env, cb)
                } else {
                    let is_recursive = {
                        let e = env.borrow();
                        match e.recursive_cache.get(func_id) {
                            Some(&r) => r,
                            None => {
                                drop(e);
                                let r = contains_func_call(&f.body, *func_id);
                                env.borrow_mut().recursive_cache.insert(*func_id, r);
                                r
                            }
                        }
                    };
                    if is_recursive {
                        let mut nv = env.borrow().next_var;
                        let body = substitute_and_rename(&f.body, &f.param_vars, args, &mut nv);
                        env.borrow_mut().next_var = nv;
                        eval(&body, input, env, cb)
                    } else {
                        // Try to use cached substituted body when all args are LoadVar
                        let all_loadvar: Option<Vec<u16>> = args.iter().map(|a| {
                            if let Expr::LoadVar { var_index } = a { Some(*var_index) } else { None }
                        }).collect();
                        if let Some(var_indices) = all_loadvar {
                            let cache_key = (*func_id, var_indices);
                            let cached = env.borrow().subst_cache.get(&cache_key).cloned();
                            if let Some(body) = cached {
                                eval(&body, input, env, cb)
                            } else {
                                let body = substitute_params(&f.body, &f.param_vars, args);
                                let body_rc = Rc::new(body);
                                env.borrow_mut().subst_cache.insert(cache_key, body_rc.clone());
                                eval(&body_rc, input, env, cb)
                            }
                        } else {
                            // Pointer-based cache: if args come from a stable (cached) body,
                            // the pointer is the same across calls.
                            let args_ptr = args.as_ptr() as usize;
                            let cached = env.borrow().subst_ptr_cache.get(func_id)
                                .filter(|(ptr, _)| *ptr == args_ptr)
                                .map(|(_, body)| body.clone());
                            if let Some(body) = cached {
                                eval(&body, input, env, cb)
                            } else {
                                let body = substitute_params(&f.body, &f.param_vars, args);
                                let body_rc = Rc::new(body);
                                env.borrow_mut().subst_ptr_cache.insert(*func_id, (args_ptr, body_rc.clone()));
                                eval(&body_rc, input, env, cb)
                            }
                        }
                    }
                }
            } else {
                bail!("undefined function id {}", func_id)
            }
        }

        Expr::StringInterpolation { parts } => {
            eval_interp_parts(parts, 0, String::new(), input, env, cb)
        }

        Expr::Limit { count, generator } => {
            eval(count, input.clone(), env, &mut |cv| {
                if let Value::Num(n, None) = &cv {
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
            let mut current = input;
            loop {
                if !cb(current.clone())? { return Ok(false); }
                if let Ok(next) = eval_one(update, &current, env) {
                    current = next;
                } else {
                    let mut next = Value::Null;
                    eval(update, current, env, &mut |v| { next = v; Ok(true) })?;
                    current = next;
                }
            }
        }

        Expr::AllShort { generator, predicate } => {
            let mut all_true = true;
            eval(generator, input.clone(), env, &mut |elem| {
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
            })?;
            cb(Value::from_bool(all_true))
        }

        Expr::AnyShort { generator, predicate } => {
            let mut any_true = false;
            eval(generator, input.clone(), env, &mut |elem| {
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
            })?;
            cb(Value::from_bool(any_true))
        }

        Expr::Error { msg } => {
            if let Some(msg_expr) = msg {
                eval(msg_expr, input, env, &mut |val| {
                    bail!("__jqerror__:{}", crate::value::value_to_json(&val))
                })
            } else {
                bail!("__jqerror__:{}", crate::value::value_to_json(&input))
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
                    eval(flags, input.clone(), env, &mut |_fv| {
                        cb(crate::runtime::call_builtin("test", &[s.clone(), re_val.clone()])?)
                    })
                })
            })
        }

        Expr::RegexMatch { input_expr, re, flags } => {
            eval(input_expr, input.clone(), env, &mut |s| {
                eval(re, input.clone(), env, &mut |re_val| {
                    eval(flags, input.clone(), env, &mut |_fv| {
                        cb(crate::runtime::call_builtin("match_", &[s.clone(), re_val.clone()])?)
                    })
                })
            })
        }

        Expr::RegexCapture { input_expr, re, flags } => {
            eval(input_expr, input.clone(), env, &mut |s| {
                eval(re, input.clone(), env, &mut |re_val| {
                    eval(flags, input.clone(), env, &mut |_fv| {
                        cb(crate::runtime::call_builtin("capture", &[s.clone(), re_val.clone()])?)
                    })
                })
            })
        }

        Expr::RegexScan { input_expr, re, flags } => {
            eval(input_expr, input.clone(), env, &mut |s| {
                eval(re, input.clone(), env, &mut |re_val| {
                    eval(flags, input.clone(), env, &mut |_fv| {
                        let result = crate::runtime::call_builtin("scan", &[s.clone(), re_val.clone()])?;
                        if let Value::Arr(a) = &result {
                            for v in a.iter() { if !cb(v.clone())? { return Ok(false); } }
                            Ok(true)
                        } else { cb(result) }
                    })
                })
            })
        }

        Expr::RegexSub { input_expr, re, tostr, flags } => {
            eval(input_expr, input.clone(), env, &mut |s| {
                eval(re, input.clone(), env, &mut |rv| {
                    eval(tostr, input.clone(), env, &mut |tv| {
                        eval(flags, input.clone(), env, &mut |_fv| {
                            cb(crate::runtime::call_builtin("sub", &[s.clone(), rv.clone(), tv.clone()])?)
                        })
                    })
                })
            })
        }

        Expr::RegexGsub { input_expr, re, tostr, flags } => {
            eval(input_expr, input.clone(), env, &mut |s| {
                eval(re, input.clone(), env, &mut |rv| {
                    eval(tostr, input.clone(), env, &mut |tv| {
                        eval(flags, input.clone(), env, &mut |_fv| {
                            cb(crate::runtime::call_builtin("gsub", &[s.clone(), rv.clone(), tv.clone()])?)
                        })
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
            obj.insert("line".into(), Value::Num(*line as f64, None));
            cb(Value::Obj(Rc::new(obj)))
        }

        Expr::Env => {
            let mut obj = crate::value::new_objmap();
            for (k, v) in std::env::vars() { obj.insert(KeyStr::from(k), Value::from_str(&v)); }
            cb(Value::Obj(Rc::new(obj)))
        }

        Expr::Builtins => cb(crate::runtime::rt_builtins()),

        Expr::ReadInput | Expr::ReadInputs => {
            if matches!(expr, Expr::ReadInput) { bail!("break") }
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
                eprint!("{}", crate::value::value_to_json(&val));
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
            cb(Value::Num(id as f64, None))
        }

        Expr::CallBuiltin { name, args } => {
            eval_call_builtin(name, args, input, env, cb)
        }
    }
}

// ---------------------------------------------------------------------------
#[inline]
pub fn eval_binop(op: BinOp, lhs: &Value, rhs: &Value) -> Result<Value> {
    // Numeric fast path: avoid runtime function dispatch for common numeric ops
    if let (Value::Num(ln, _), Value::Num(rn, _)) = (lhs, rhs) {
        return Ok(match op {
            BinOp::Add => Value::Num(ln + rn, None),
            BinOp::Sub => Value::Num(ln - rn, None),
            BinOp::Mul => Value::Num(ln * rn, None),
            BinOp::Div => {
                if *rn == 0.0 { return crate::runtime::rt_div(lhs, rhs); }
                Value::Num(ln / rn, None)
            }
            BinOp::Mod => {
                if *rn == 0.0 || rn.is_nan() || ln.is_nan() { return crate::runtime::rt_mod(lhs, rhs); }
                let yi = *rn as i64;
                if yi == 0 { return crate::runtime::rt_mod(lhs, rhs); }
                Value::Num((*ln as i64 % yi) as f64, None)
            }
            BinOp::Eq => if ln == rn { Value::True } else { Value::False },
            BinOp::Ne => if ln != rn { Value::True } else { Value::False },
            BinOp::Lt => if ln < rn { Value::True } else { Value::False },
            BinOp::Gt => if ln > rn { Value::True } else { Value::False },
            BinOp::Le => if ln <= rn { Value::True } else { Value::False },
            BinOp::Ge => if ln >= rn { Value::True } else { Value::False },
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
            BinOp::Add => Value::Num(ln + rn, None),
            BinOp::Sub => Value::Num(ln - rn, None),
            BinOp::Mul => Value::Num(ln * rn, None),
            BinOp::Div => {
                if *rn == 0.0 { return crate::runtime::rt_div(&lhs, rhs); }
                Value::Num(ln / rn, None)
            }
            BinOp::Mod => {
                if *rn == 0.0 || rn.is_nan() || ln.is_nan() { return crate::runtime::rt_mod(&lhs, rhs); }
                let yi = *rn as i64;
                if yi == 0 { return crate::runtime::rt_mod(&lhs, rhs); }
                Value::Num((*ln as i64 % yi) as f64, None)
            }
            BinOp::Eq => if ln == rn { Value::True } else { Value::False },
            BinOp::Ne => if ln != rn { Value::True } else { Value::False },
            BinOp::Lt => if ln < rn { Value::True } else { Value::False },
            BinOp::Gt => if ln > rn { Value::True } else { Value::False },
            BinOp::Le => if ln <= rn { Value::True } else { Value::False },
            BinOp::Ge => if ln >= rn { Value::True } else { Value::False },
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
        UnaryOp::Infinite => return Ok(Value::Num(f64::INFINITY, None)),
        UnaryOp::Nan => return Ok(Value::Num(f64::NAN, None)),
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
        UnaryOp::Gmtime => "gmtime", UnaryOp::Mktime => "mktime", UnaryOp::Now => "now",
        UnaryOp::Abs => "abs", UnaryOp::GetModuleMeta => "modulemeta",
        _ => unreachable!(),
    };
    crate::runtime::call_builtin(name, std::slice::from_ref(val))
}

pub fn eval_index(base: &Value, key: &Value, optional: bool) -> std::result::Result<Value, String> {
    match (base, key) {
        (Value::Obj(o), Value::Str(k)) => Ok(o.get(k.as_str()).cloned().unwrap_or(Value::Null)),
        (Value::Arr(a), Value::Num(n, _)) => {
            if n.is_nan() { return Ok(Value::Null); }
            let idx = *n as i64;
            let i = if idx < 0 { (a.len() as i64 + idx) as usize } else { idx as usize };
            Ok(a.get(i).cloned().unwrap_or(Value::Null))
        }
        (Value::Str(_), Value::Num(n, _)) => {
            if n.fract() != 0.0 || n.is_nan() || n.is_infinite() {
                Err(format!("Cannot index string with number ({})", crate::value::format_jq_number(*n)))
            } else if optional {
                Err("type error".into())
            } else {
                Err(format!("Cannot index string with number ({})", crate::value::format_jq_number(*n)))
            }
        }
        (Value::Null, _) => Ok(Value::Null),
        _ => {
            if optional { Err("type error".into()) }
            else {
                let key_desc = match key {
                    Value::Str(s) => format!("string (\"{}\")", s),
                    Value::Num(n, _) => format!("number ({})", crate::value::format_jq_number(*n)),
                    _ => format!("{} ({})", key.type_name(), crate::value::value_to_json(key)),
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
        // Custom step: use explicit stack to avoid stack overflow
        let mut work = vec![val.clone()];
        while let Some(current) = work.pop() {
            if !cb(current.clone())? { return Ok(false); }
            let mut next_vals = Vec::new();
            let _ = eval(step, current, env, &mut |next| {
                next_vals.push(next);
                Ok(true)
            });
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
        Value::Arr(a) => { for item in a.iter() { if !eval_recurse_default(item, cb)? { return Ok(false); } } }
        Value::Obj(o) => { for v in o.values() { if !eval_recurse_default(v, cb)? { return Ok(false); } } }
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
    if s > 0.0 { while c < t { if !cb(Value::Num(c, None))? { return Ok(false); } c += s; } }
    else { while c > t { if !cb(Value::Num(c, None))? { return Ok(false); } c += s; } }
    Ok(true)
}

fn eval_object_construct(pairs: &[(Expr, Expr)], input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    // Fast path: if all keys and values are scalar expressions, build directly without cloning
    let mut obj = crate::value::new_objmap_with_capacity(pairs.len());
    for (ke, ve) in pairs {
        let kv = match eval_one(ke, &input, env) {
            Ok(v) => v,
            Err(()) => return eval_obj_pairs(pairs, 0, crate::value::new_objmap_with_capacity(pairs.len()), input, env, cb),
        };
        let ks: KeyStr = match &kv { Value::Str(s) => KeyStr::from(s.as_str()), _ => KeyStr::from(crate::value::value_to_json(&kv)) };
        let vv = match eval_one(ve, &input, env) {
            Ok(v) => v,
            Err(()) => return eval_obj_pairs(pairs, 0, crate::value::new_objmap_with_capacity(pairs.len()), input, env, cb),
        };
        obj.insert(ks, vv);
    }
    cb(Value::Obj(Rc::new(obj)))
}

fn eval_obj_pairs(pairs: &[(Expr, Expr)], idx: usize, cur: crate::value::ObjMap, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    if idx >= pairs.len() { return cb(Value::Obj(Rc::new(cur))); }
    let (ke, ve) = &pairs[idx];
    eval(ke, input.clone(), env, &mut |kv| {
        let ks: KeyStr = match &kv { Value::Str(s) => KeyStr::from(s.as_str()), _ => KeyStr::from(crate::value::value_to_json(&kv)) };
        eval(ve, input.clone(), env, &mut |vv| {
            let mut next = cur.clone();
            next.insert(ks.clone(), vv);
            eval_obj_pairs(pairs, idx + 1, next, input.clone(), env, cb)
        })
    })
}

pub fn eval_format(name: &str, val: &Value) -> Result<String> {
    // For csv/tsv, the input must be an array
    match name {
        "csv" => {
            let arr = match val { Value::Arr(a) => a, _ => bail!("@csv requires array input") };
            let parts: Vec<String> = arr.iter().map(|v| {
                match v {
                    Value::Str(s) => {
                        if s.contains('"') || s.contains(',') || s.contains('\n') {
                            format!("\"{}\"", s.replace('"', "\"\""))
                        } else {
                            format!("\"{}\"", s)
                        }
                    }
                    Value::Null => "".to_string(),
                    Value::True => "true".to_string(),
                    Value::False => "false".to_string(),
                    _ => crate::value::value_to_json(v),
                }
            }).collect();
            return Ok(parts.join(","));
        }
        "tsv" => {
            let arr = match val { Value::Arr(a) => a, _ => bail!("@tsv requires array input") };
            let parts: Vec<String> = arr.iter().map(|v| {
                match v {
                    Value::Str(s) => s.replace('\\', "\\\\").replace('\t', "\\t").replace('\n', "\\n").replace('\r', "\\r"),
                    Value::Null => "".to_string(),
                    Value::True => "true".to_string(),
                    Value::False => "false".to_string(),
                    _ => crate::value::value_to_json(v),
                }
            }).collect();
            return Ok(parts.join("\t"));
        }
        _ => {}
    }

    // For other formats, stringify the value first
    let s = match val { Value::Str(s) => s.to_string(), _ => crate::value::value_to_json(val) };
    match name {
        "text" => Ok(s),
        "json" => Ok(serde_json::to_string(&s).unwrap_or_else(|_| format!("{:?}", s))),
        "html" => Ok(s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('\'', "&apos;").replace('"', "&quot;")),
        "uri" => { let mut r = String::new(); for b in s.bytes() { match b { b'A'..=b'Z'|b'a'..=b'z'|b'0'..=b'9'|b'-'|b'_'|b'.'|b'~' => r.push(b as char), _ => r.push_str(&format!("%{:02X}", b)) } } Ok(r) }
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
        "sh" => Ok(format!("'{}'", s.replace('\'', "'\\''"))),
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
            let mut r = Vec::new();
            for ch in bs.chunks(4) { if ch.len()<2{break;} let a=D.get(ch[0] as usize).copied().unwrap_or(-1); let b=D.get(ch[1] as usize).copied().unwrap_or(-1); if a<0||b<0{bail!("invalid base64");}
                r.push(((a as u8)<<2)|((b as u8)>>4)); if ch.len()>2&&ch[2]!=b'=' { let c=D.get(ch[2] as usize).copied().unwrap_or(-1); if c<0{bail!("invalid base64");} r.push(((b as u8)<<4)|((c as u8)>>2));
                if ch.len()>3&&ch[3]!=b'=' { let d=D.get(ch[3] as usize).copied().unwrap_or(-1); if d<0{bail!("invalid base64");} r.push(((c as u8)<<6)|(d as u8)); } } }
            String::from_utf8(r).map_err(|e| anyhow::anyhow!("invalid utf8: {}", e))
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
            let chars: Vec<char> = s.chars().collect(); let len = chars.len() as i64;
            let fi = match from { Value::Num(n, _) => slice_index_start(*n, len), Value::Null => 0, _ => bail!("slice: need number") };
            let ti = match to { Value::Num(n, _) => slice_index_end(*n, len), Value::Null => len as usize, _ => bail!("slice: need number") };
            Ok(if fi>=ti { Value::from_str("") } else { Value::from_str(&chars[fi..ti].iter().collect::<String>()) })
        }
        Value::Null => Ok(Value::Null),
        _ => bail!("cannot slice {}", base.type_name()),
    }
}

fn eval_closure_op(op: ClosureOpKind, container: &Value, key_expr: &Expr, _input: &Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    let a = match container { Value::Arr(a) => a, _ => bail!("{} is not an array", container.type_name()) };
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
            let mut seen: Vec<Vec<Value>> = Vec::new(); let mut result: Vec<Value> = Vec::new();
            for (keys, val) in keyed { if !seen.iter().any(|k| k.len()==keys.len()&&k.iter().zip(keys.iter()).all(|(a,b)| crate::runtime::values_equal(a,b))) { seen.push(keys); result.push(val); } }
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
            let result = eval_path(be, input.clone(), env, &mut |bp| {
                cb_called.set(true);
                eval(ke, input.clone(), env, &mut |key| {
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
                    Value::Arr(a) => { for i in 0..a.len() { let mut p = match &bp { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] }; p.push(Value::Num(i as f64, None)); if !cb(Value::Arr(Rc::new(p)))? { return Ok(false); } } Ok(true) }
                    Value::Obj(o) => { for k in o.keys() { let mut p = match &bp { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] }; p.push(Value::from_str(k)); if !cb(Value::Arr(Rc::new(p)))? { return Ok(false); } } Ok(true) }
                    _ => Ok(true),
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
                let base = crate::runtime::rt_getpath(&input, &bp).unwrap_or(Value::Null);
                let from_val = if let Some(f) = from {
                    let mut v = Value::Null;
                    eval(f, input.clone(), env, &mut |fv| { v = fv; Ok(true) })?; v
                } else { Value::Null };
                let to_val = if let Some(t) = to {
                    let mut v = Value::Null;
                    eval(t, input.clone(), env, &mut |tv| { v = tv; Ok(true) })?; v
                } else { Value::Null };
                let len = match &base {
                    Value::Arr(a) => a.len() as i64,
                    Value::Str(s) => s.chars().count() as i64,
                    _ => 0,
                };
                let fi = match &from_val { Value::Num(n, _) => { let i = n.floor() as i64; if i < 0 { (len + i).max(0) } else { i.min(len) } }, Value::Null => 0, _ => 0 };
                let ti = match &to_val { Value::Num(n, _) => { let i = n.ceil() as i64; if i < 0 { (len + i).max(0) } else { i.min(len) } }, Value::Null => len, _ => len };
                // Return a special path that indicates slicing
                let mut p = match &bp { Value::Arr(a) => a.as_ref().clone(), _ => vec![] };
                p.push(Value::Obj(Rc::new({
                    let mut m = crate::value::new_objmap();
                    m.insert("start".into(), Value::Num(fi as f64, None));
                    m.insert("end".into(), Value::Num(ti as f64, None));
                    m
                })));
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
                    Value::Arr(a) => { for i in 0..a.len() { let mut p = match &bp { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] }; p.push(Value::Num(i as f64, None)); if !cb(Value::Arr(Rc::new(p)))? { return Ok(false); } } Ok(true) }
                    Value::Obj(o) => { for k in o.keys() { let mut p = match &bp { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] }; p.push(Value::from_str(k)); if !cb(Value::Arr(Rc::new(p)))? { return Ok(false); } } Ok(true) }
                    _ => Ok(true),
                }
            })
        }
        _ => {
            // Non-path-safe expression: evaluate and report error
            let mut result_val = Value::Null;
            let mut has_result = false;
            eval(expr, input, env, &mut |val| { result_val = val; has_result = true; Ok(true) })?;
            if has_result {
                bail!("__pathexpr_result__:{}", crate::value::value_to_json(&result_val));
            }
            Ok(true)
        }
    }
}

fn eval_recurse_paths(val: &Value, prefix: &Value, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    if !cb(prefix.clone())? { return Ok(false); }
    match val {
        Value::Arr(a) => { for (i, item) in a.iter().enumerate() { let mut p = match prefix { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] }; p.push(Value::Num(i as f64, None)); if !eval_recurse_paths(item, &Value::Arr(Rc::new(p)), cb)? { return Ok(false); } } }
        Value::Obj(o) => { for (k, v) in o.iter() { let mut p = match prefix { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] }; p.push(Value::from_str(k)); if !eval_recurse_paths(v, &Value::Arr(Rc::new(p)), cb)? { return Ok(false); } } }
        _ => {}
    }
    Ok(true)
}

fn eval_call_builtin(name: &str, args: &[Expr], input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    // Special handling for builtins that take filter/closure arguments
    match (name, args.len()) {
        ("toboolean", 0) => {
            return cb(rt_toboolean(&input)?);
        }
        ("halt", 0) => {
            // halt: exit with code 0, print input to stderr if not null
            if !matches!(input, Value::Null) {
                let json = crate::value::value_to_json_precise(&input);
                eprintln!("{}", json);
            }
            std::process::exit(0);
        }
        ("halt_error", 0) => {
            // halt_error: exit with code from input (default 5)
            let code = match &input {
                Value::Num(n, _) => *n as i32,
                _ => {
                    let json = crate::value::value_to_json_precise(&input);
                    eprintln!("{}", json);
                    5
                }
            };
            std::process::exit(code);
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
        _ => {}
    }
    // Default: evaluate args as generators and call runtime with input + args
    eval_call_builtin_args(name, args, 0, vec![input.clone()], input, env, cb)
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

// walk(f): Recursively apply f bottom-up
fn eval_walk(f: &Expr, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    let walked = walk_value(f, input, env)?;
    // walked may be multiple values if f is a generator
    for v in walked {
        if !cb(v)? { return Ok(false); }
    }
    Ok(true)
}

fn walk_value(f: &Expr, input: Value, env: &EnvRef) -> Result<Vec<Value>> {
    match input {
        Value::Arr(ref a) => {
            // Walk each element, collect results
            let mut new_arr = Vec::new();
            for item in a.iter() {
                let walked = walk_value(f, item.clone(), env)?;
                // For arrays, each element should produce exactly one value from walk
                // But if walk is a generator, we need to handle it
                if walked.len() == 1 {
                    new_arr.push(walked.into_iter().next().unwrap());
                } else {
                    // Multiple results from walking a child - extend
                    new_arr.extend(walked);
                }
            }
            let rebuilt = Value::Arr(Rc::new(new_arr));
            // Apply f to the rebuilt array
            let mut results = Vec::new();
            eval(f, rebuilt, env, &mut |val| {
                results.push(val);
                Ok(true)
            })?;
            Ok(results)
        }
        Value::Obj(ref o) => {
            // Walk each value
            let mut new_obj = crate::value::new_objmap();
            for (k, v) in o.iter() {
                let walked = walk_value(f, v.clone(), env)?;
                if !walked.is_empty() {
                    new_obj.insert(k.clone(), walked.into_iter().next().unwrap());
                }
            }
            let rebuilt = Value::Obj(Rc::new(new_obj));
            let mut results = Vec::new();
            eval(f, rebuilt, env, &mut |val| {
                results.push(val);
                Ok(true)
            })?;
            Ok(results)
        }
        _ => {
            // Scalar - just apply f
            let mut results = Vec::new();
            eval(f, input, env, &mut |val| {
                results.push(val);
                Ok(true)
            })?;
            Ok(results)
        }
    }
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
                    std::cmp::Ordering::Equal => return Ok(Value::Num(mid as f64, None)),
                    std::cmp::Ordering::Less => lo = mid + 1,
                    std::cmp::Ordering::Greater => hi = mid - 1,
                }
            }
            // Not found: return -(insertion_point) - 1
            Ok(Value::Num(-(lo as f64) - 1.0, None))
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
