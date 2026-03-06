//! Tree-walking interpreter for the IR.
//!
//! Every expression is a generator: it takes an input Value and produces
//! zero or more output Values via a callback.
//!
//! We use Rc<RefCell<Env>> to allow nested closures to share the environment.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::{Result, bail};

use crate::ir::*;
use crate::value::Value;

type GenResult = Result<bool>;
type EnvRef = Rc<RefCell<Env>>;

pub struct Env {
    vars: Vec<Value>,
    funcs: Vec<CompiledFunc>,
    next_label: u64,
    pub lib_dirs: Vec<String>,
}

impl Env {
    pub fn new(funcs: Vec<CompiledFunc>) -> Self {
        Env { vars: vec![Value::Null; 4096], funcs, next_label: 0, lib_dirs: Vec::new() }
    }
    pub fn with_lib_dirs(funcs: Vec<CompiledFunc>, lib_dirs: Vec<String>) -> Self {
        Env { vars: vec![Value::Null; 4096], funcs, next_label: 0, lib_dirs }
    }
    fn get_var(&self, idx: u16) -> Value {
        self.vars.get(idx as usize).cloned().unwrap_or(Value::Null)
    }
    fn set_var(&mut self, idx: u16, val: Value) {
        let idx = idx as usize;
        if idx >= self.vars.len() { self.vars.resize(idx + 1, Value::Null); }
        self.vars[idx] = val;
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
            Literal::Num(n) => Value::Num(*n),
            Literal::Str(s) => Value::from_str(s),
        }),

        Expr::BinOp { op, lhs, rhs } => {
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
                _ => {
                    eval(lhs, input.clone(), env, &mut |lval| {
                        eval(rhs, input.clone(), env, &mut |rval| {
                            cb(eval_binop(op, &lval, &rval)?)
                        })
                    })
                }
            }
        }

        Expr::UnaryOp { op, operand } => {
            eval(operand, input, env, &mut |val| cb(eval_unaryop(*op, &val)?))
        }

        Expr::Index { expr: base_expr, key: key_expr } => {
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
            eval(left, input, env, &mut |mid| eval(right, mid, env, cb))
        }

        Expr::Comma { left, right } => {
            let cont = eval(left, input.clone(), env, cb)?;
            if !cont { return Ok(false); }
            eval(right, input, env, cb)
        }

        Expr::Empty => Ok(true),

        Expr::IfThenElse { cond, then_branch, else_branch } => {
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
                    let msg = format!("{}", e);
                    if msg.starts_with("__break__:") { return Err(e); }
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
            eval(value, input.clone(), env, &mut |val| {
                let old = env.borrow().get_var(*var_index);
                env.borrow_mut().set_var(*var_index, val);
                let result = eval(body, input.clone(), env, cb);
                env.borrow_mut().set_var(*var_index, old);
                result
            })
        }

        Expr::LoadVar { var_index } => {
            let val = env.borrow().get_var(*var_index);
            cb(val)
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
            let mut acc = Value::Null;
            eval(init, input.clone(), env, &mut |v| { acc = v; Ok(true) })?;
            let vi = *var_index;
            let ai = *acc_index;
            eval(source, input.clone(), env, &mut |val| {
                let old_var = env.borrow().get_var(vi);
                let old_acc = env.borrow().get_var(ai);
                env.borrow_mut().set_var(vi, val);
                env.borrow_mut().set_var(ai, acc.clone());
                eval(update, acc.clone(), env, &mut |new_acc| { acc = new_acc; Ok(true) })?;
                env.borrow_mut().set_var(vi, old_var);
                env.borrow_mut().set_var(ai, old_acc);
                Ok(true)
            })?;
            cb(acc)
        }

        Expr::Foreach { source, init, var_index, acc_index, update, extract } => {
            let vi = *var_index;
            let ai = *acc_index;
            eval(init, input.clone(), env, &mut |init_val| {
                let mut acc = init_val;
                eval(source, input.clone(), env, &mut |val| {
                    let old_var = env.borrow().get_var(vi);
                    let old_acc = env.borrow().get_var(ai);
                    env.borrow_mut().set_var(vi, val);
                    env.borrow_mut().set_var(ai, acc.clone());
                    eval(update, acc.clone(), env, &mut |new_acc| { acc = new_acc; Ok(true) })?;
                    let cont = if let Some(extract_expr) = extract {
                        eval(extract_expr, acc.clone(), env, cb)?
                    } else {
                        cb(acc.clone())?
                    };
                    env.borrow_mut().set_var(vi, old_var);
                    env.borrow_mut().set_var(ai, old_acc);
                    Ok(cont)
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
                    Value::Num(n) => cb(Value::Num(-n)),
                    _ => {
                        bail!("{} cannot be negated", crate::runtime::errdesc_pub(&val))
                    }
                }
            })
        }

        Expr::Not => cb(if input.is_truthy() { Value::False } else { Value::True }),

        Expr::Recurse { input_expr } => eval_recurse_expr(input_expr, &input, env, cb),

        Expr::Range { from, to, step } => {
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
            let label_id = env.borrow().next_label;
            env.borrow_mut().next_label = label_id + 1;
            env.borrow_mut().set_var(*var_index, Value::Num(label_id as f64));
            match eval(body, input, env, cb) {
                Err(e) => {
                    let msg = format!("{}", e);
                    if let Some(rest) = msg.strip_prefix("__break__:") {
                        if let Ok(bid) = rest.trim_end_matches(':').parse::<u64>() {
                            if bid == label_id { return Ok(true); }
                        }
                    }
                    Err(e)
                }
                other => other,
            }
        }

        Expr::Break { var_index, .. } => {
            let label = env.borrow().get_var(*var_index);
            if let Value::Num(n) = &label { bail!("__break__:{}:", *n as u64) }
            else { bail!("break: invalid label") }
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

        Expr::FuncCall { func_id, .. } => {
            let func = env.borrow().funcs.get(*func_id).cloned();
            if let Some(f) = func {
                eval(&f.body, input, env, cb)
            } else {
                bail!("undefined function id {}", func_id)
            }
        }

        Expr::StringInterpolation { parts } => {
            eval_interp_parts(parts, 0, String::new(), input, env, cb)
        }

        Expr::Limit { count, generator } => {
            eval(count, input.clone(), env, &mut |cv| {
                if let Value::Num(n) = &cv {
                    let limit = *n as i64;
                    if limit == 0 { return Ok(true); }
                    if limit < 0 {
                        bail!("__jqerror__:\"limit doesn't support negative count\"");
                    }
                    let limit = limit as usize;
                    let mut emitted = 0;
                    let result = eval(generator, input.clone(), env, &mut |val| {
                        emitted += 1;
                        let cont = cb(val)?;
                        if !cont || emitted >= limit { Ok(false) } else { Ok(true) }
                    });
                    match result {
                        Ok(_) => Ok(true),
                        Err(e) => Err(e),
                    }
                } else { bail!("limit: count must be a number") }
            })
        }

        Expr::While { cond, update } => {
            let mut current = input;
            loop {
                let mut is_true = false;
                eval(cond, current.clone(), env, &mut |v| { is_true = v.is_truthy(); Ok(true) })?;
                if !is_true { break; }
                if !cb(current.clone())? { return Ok(false); }
                let mut next = current.clone();
                eval(update, current, env, &mut |v| { next = v; Ok(true) })?;
                current = next;
            }
            Ok(true)
        }

        Expr::Until { cond, update } => {
            let mut current = input;
            loop {
                let mut is_true = false;
                eval(cond, current.clone(), env, &mut |v| { is_true = v.is_truthy(); Ok(true) })?;
                if is_true { break; }
                let mut next = current.clone();
                eval(update, current, env, &mut |v| { next = v; Ok(true) })?;
                current = next;
            }
            cb(current)
        }

        Expr::Repeat { update } => {
            let mut current = input;
            loop {
                if !cb(current.clone())? { return Ok(false); }
                let mut next = current.clone();
                eval(update, current, env, &mut |v| { next = v; Ok(true) })?;
                current = next;
            }
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
            let mut obj = indexmap::IndexMap::new();
            obj.insert("file".to_string(), Value::from_str(file));
            obj.insert("line".to_string(), Value::Num(*line as f64));
            cb(Value::Obj(Rc::new(obj)))
        }

        Expr::Env => {
            let mut obj = indexmap::IndexMap::new();
            for (k, v) in std::env::vars() { obj.insert(k, Value::from_str(&v)); }
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
            cb(Value::Num(id as f64))
        }

        Expr::CallBuiltin { name, args } => {
            eval_call_builtin(name, args, input, env, cb)
        }
    }
}

// ---------------------------------------------------------------------------
fn eval_binop(op: BinOp, lhs: &Value, rhs: &Value) -> Result<Value> {
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

fn eval_unaryop(op: UnaryOp, val: &Value) -> Result<Value> {
    match op {
        UnaryOp::Not => return Ok(if val.is_truthy() { Value::False } else { Value::True }),
        UnaryOp::Infinite => return Ok(Value::Num(f64::INFINITY)),
        UnaryOp::Nan => return Ok(Value::Num(f64::NAN)),
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
    crate::runtime::call_builtin(name, &[val.clone()])
}

fn eval_index(base: &Value, key: &Value, optional: bool) -> std::result::Result<Value, String> {
    match (base, key) {
        (Value::Obj(o), Value::Str(k)) => Ok(o.get(k.as_str()).cloned().unwrap_or(Value::Null)),
        (Value::Arr(a), Value::Num(n)) => {
            if n.is_nan() { return Ok(Value::Null); }
            let idx = *n as i64;
            let i = if idx < 0 { (a.len() as i64 + idx) as usize } else { idx as usize };
            Ok(a.get(i).cloned().unwrap_or(Value::Null))
        }
        (Value::Str(_), Value::Num(n)) => {
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
                    Value::Num(n) => format!("number ({})", crate::value::format_jq_number(*n)),
                    _ => format!("{} ({})", key.type_name(), crate::value::value_to_json(key)),
                };
                Err(format!("Cannot index {} with {}", base.type_name(), key_desc))
            }
        }
    }
}

fn eval_recurse_expr(step: &Expr, val: &Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    if !cb(val.clone())? { return Ok(false); }
    if matches!(step, Expr::Input) {
        match val {
            Value::Arr(a) => { for item in a.iter() { if !eval_recurse_expr(step, item, env, cb)? { return Ok(false); } } }
            Value::Obj(o) => { for v in o.values() { if !eval_recurse_expr(step, v, env, cb)? { return Ok(false); } } }
            _ => {}
        }
    } else {
        let _ = eval(step, val.clone(), env, &mut |next| eval_recurse_expr(step, &next, env, cb));
    }
    Ok(true)
}

fn eval_range(from: &Value, to: &Value, step: Option<&Value>, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    let f = match from { Value::Num(n) => *n, _ => bail!("range: from must be number") };
    let t = match to { Value::Num(n) => *n, _ => bail!("range: to must be number") };
    let s = match step { Some(Value::Num(n)) => *n, Some(_) => bail!("range: step must be number"), None => 1.0 };
    if s == 0.0 { return Ok(true); }
    let mut c = f;
    if s > 0.0 { while c < t { if !cb(Value::Num(c))? { return Ok(false); } c += s; } }
    else { while c > t { if !cb(Value::Num(c))? { return Ok(false); } c += s; } }
    Ok(true)
}

fn eval_object_construct(pairs: &[(Expr, Expr)], input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    eval_obj_pairs(pairs, 0, indexmap::IndexMap::new(), input, env, cb)
}

fn eval_obj_pairs(pairs: &[(Expr, Expr)], idx: usize, cur: indexmap::IndexMap<String, Value>, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    if idx >= pairs.len() { return cb(Value::Obj(Rc::new(cur))); }
    let (ke, ve) = &pairs[idx];
    eval(ke, input.clone(), env, &mut |kv| {
        let ks = match &kv { Value::Str(s) => s.as_ref().clone(), _ => crate::value::value_to_json(&kv) };
        eval(ve, input.clone(), env, &mut |vv| {
            let mut next = cur.clone();
            next.insert(ks.clone(), vv);
            eval_obj_pairs(pairs, idx + 1, next, input.clone(), env, cb)
        })
    })
}

fn eval_format(name: &str, val: &Value) -> Result<String> {
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
    let s = match val { Value::Str(s) => s.as_ref().clone(), _ => crate::value::value_to_json(val) };
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

fn eval_slice(base: &Value, from: &Value, to: &Value) -> Result<Value> {
    match base {
        Value::Arr(a) => {
            let len = a.len() as i64;
            let fi = match from { Value::Num(n) => slice_index_start(*n, len), Value::Null => 0, _ => bail!("slice: need number") };
            let ti = match to { Value::Num(n) => slice_index_end(*n, len), Value::Null => len as usize, _ => bail!("slice: need number") };
            Ok(if fi>=ti { Value::Arr(Rc::new(vec![])) } else { Value::Arr(Rc::new(a[fi..ti].to_vec())) })
        }
        Value::Str(s) => {
            let chars: Vec<char> = s.chars().collect(); let len = chars.len() as i64;
            let fi = match from { Value::Num(n) => slice_index_start(*n, len), Value::Null => 0, _ => bail!("slice: need number") };
            let ti = match to { Value::Num(n) => slice_index_end(*n, len), Value::Null => len as usize, _ => bail!("slice: need number") };
            Ok(if fi>=ti { Value::from_str("") } else { Value::from_str(&chars[fi..ti].iter().collect::<String>()) })
        }
        Value::Null => Ok(Value::Null),
        _ => bail!("cannot slice {}", base.type_name()),
    }
}

fn eval_closure_op(op: ClosureOpKind, container: &Value, key_expr: &Expr, input: &Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
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

fn eval_interp_parts(parts: &[StringPart], idx: usize, cur: String, input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    if idx >= parts.len() { return cb(Value::from_str(&cur)); }
    match &parts[idx] {
        StringPart::Literal(s) => { let mut n = cur; n.push_str(s); eval_interp_parts(parts, idx+1, n, input, env, cb) }
        StringPart::Expr(e) => {
            eval(e, input.clone(), env, &mut |val| {
                let s = match &val { Value::Str(s) => s.as_ref().clone(), _ => crate::value::value_to_json(&val) };
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
                            Value::Num(n) => format!("element {} of", crate::value::format_jq_number(*n)),
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
                    Value::Arr(a) => { for i in 0..a.len() { let mut p = match &bp { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] }; p.push(Value::Num(i as f64)); if !cb(Value::Arr(Rc::new(p)))? { return Ok(false); } } Ok(true) }
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
                                    Value::Num(n) => format!("element {} of", crate::value::format_jq_number(*n)),
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
                let fi = match &from_val { Value::Num(n) => { let i = n.floor() as i64; if i < 0 { (len + i).max(0) } else { i.min(len) } }, Value::Null => 0, _ => 0 };
                let ti = match &to_val { Value::Num(n) => { let i = n.ceil() as i64; if i < 0 { (len + i).max(0) } else { i.min(len) } }, Value::Null => len, _ => len };
                // Return a special path that indicates slicing
                let mut p = match &bp { Value::Arr(a) => a.as_ref().clone(), _ => vec![] };
                p.push(Value::Obj(Rc::new({
                    let mut m = indexmap::IndexMap::new();
                    m.insert("start".to_string(), Value::Num(fi as f64));
                    m.insert("end".to_string(), Value::Num(ti as f64));
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
                    let msg = format!("{}", e);
                    if msg.starts_with("__break__:") { return Err(e); }
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
                    Value::Arr(a) => { for i in 0..a.len() { let mut p = match &bp { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] }; p.push(Value::Num(i as f64)); if !cb(Value::Arr(Rc::new(p)))? { return Ok(false); } } Ok(true) }
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
        Value::Arr(a) => { for (i, item) in a.iter().enumerate() { let mut p = match prefix { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] }; p.push(Value::Num(i as f64)); if !eval_recurse_paths(item, &Value::Arr(Rc::new(p)), cb)? { return Ok(false); } } }
        Value::Obj(o) => { for (k, v) in o.iter() { let mut p = match prefix { Value::Arr(a)=>a.as_ref().clone(), _=>vec![] }; p.push(Value::from_str(k)); if !eval_recurse_paths(v, &Value::Arr(Rc::new(p)), cb)? { return Ok(false); } } }
        _ => {}
    }
    Ok(true)
}

fn eval_call_builtin(name: &str, args: &[Expr], input: Value, env: &EnvRef, cb: &mut dyn FnMut(Value) -> GenResult) -> GenResult {
    // Evaluate args as generators and call runtime with input + args
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
            let msg = format!("{}", e);
            if msg.starts_with("__break__:") {
                Ok(outputs)
            } else {
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
