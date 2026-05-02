//! Runtime helpers for jq operations.
//!
//! These functions implement jq's built-in operations using our Value type.
//! They're called by both the interpreter and (eventually) JIT-compiled code.

use std::rc::Rc;

use anyhow::{Result, bail};
use crate::value::{Value, ObjMap, ObjInner, NumRepr, KeyStr, new_objmap};

// System libm bindings used to match jq's last-digit precision on
// gamma-family functions, and to obtain logb's f64 return (libm crate
// only exposes ilogb, which collapses zero/NaN to i32::MIN).
unsafe extern "C" {
    fn tgamma(x: f64) -> f64;
    fn lgamma(x: f64) -> f64;
    fn lgamma_r(x: f64, sign: *mut i32) -> f64;
    fn logb(x: f64) -> f64;
    // Use libc-backed erf/erfc/expm1/log1p for byte-identical agreement
    // with jq across platforms; libm-Rust differs from Apple/glibc by
    // 1 ULP on some inputs (e.g. `erf(1)`). See #473.
    fn erf(x: f64) -> f64;
    fn erfc(x: f64) -> f64;
    fn expm1(x: f64) -> f64;
    fn log1p(x: f64) -> f64;
    fn y0(x: f64) -> f64;
    fn y1(x: f64) -> f64;
    fn yn(n: i32, x: f64) -> f64;
    fn jn(n: i32, x: f64) -> f64;
    fn copysign(x: f64, y: f64) -> f64;
    fn fdim(x: f64, y: f64) -> f64;
    fn fmax(x: f64, y: f64) -> f64;
    fn fmin(x: f64, y: f64) -> f64;
    fn fmod(x: f64, y: f64) -> f64;
    fn nextafter(x: f64, y: f64) -> f64;
}

/// Dispatch a builtin call by name.
pub fn call_builtin(name: &str, args: &[Value]) -> Result<Value> {
    match name {
        // Binary arithmetic (nargs=3: input, lhs, rhs)
        "_plus" => binary_op(args, rt_add),
        "_minus" => binary_op(args, rt_sub),
        "_multiply" => binary_op(args, rt_mul),
        "_divide" => binary_op(args, rt_div),
        "_modulo" => binary_op(args, rt_mod),
        "_equal" => binary_op(args, rt_eq),
        "_notequal" => binary_op(args, rt_ne),
        "_less" => binary_op(args, rt_lt),
        "_greater" => binary_op(args, rt_gt),
        "_lesseq" => binary_op(args, rt_le),
        "_greatereq" => binary_op(args, rt_ge),

        // Unary (nargs=1: input)
        "length" => unary_op(args, rt_length),
        "utf8bytelength" => unary_op(args, rt_utf8bytelength),
        "type" => unary_op(args, rt_type),
        "keys" | "keys_unsorted" => unary_op(args, |v| rt_keys(v, name == "keys")),
        "values" => unary_op(args, rt_values),
        "sort" => unary_op(args, rt_sort),
        "reverse" => unary_op(args, rt_reverse),
        "flatten" if args.len() < 2 => unary_op(args, |v| rt_flatten(v, None)),
        "flatten" => {
            match &args[1] {
                Value::Num(n, _) => {
                    if *n < 0.0 { bail!("flatten depth must not be negative"); }
                    rt_flatten(&args[0], Some(*n as usize))
                }
                _ => rt_flatten(&args[0], None)
            }
        }
        "unique" => unary_op(args, rt_unique),
        "min" => unary_op(args, rt_min),
        "max" => unary_op(args, rt_max),
        "add" => unary_op(args, rt_add_all),
        "any" => unary_op(args, rt_any),
        "all" => unary_op(args, rt_all),
        "floor" => unary_op(args, rt_floor),
        "ceil" => unary_op(args, rt_ceil),
        "round" => unary_op(args, rt_round),
        "fabs" => unary_op(args, rt_fabs),
        "abs" => unary_op(args, rt_abs),
        "sqrt" => unary_op(args, rt_sqrt),
        "tostring" => unary_op(args, rt_tostring),
        "tonumber" => unary_op(args, rt_tonumber),
        "ascii_downcase" => unary_op(args, rt_ascii_downcase),
        "ascii_upcase" => unary_op(args, rt_ascii_upcase),
        "ltrim" => unary_op(args, rt_ltrim),
        "rtrim" => unary_op(args, rt_rtrim),
        "trim" => unary_op(args, rt_trim),
        "explode" => unary_op(args, rt_explode),
        "implode" => unary_op(args, rt_implode),
        "tojson" => unary_op(args, rt_tojson),
        "fromjson" => unary_op(args, rt_fromjson),
        "to_entries" => unary_op(args, rt_to_entries),
        "from_entries" => unary_op(args, rt_from_entries),
        "transpose" => unary_op(args, rt_transpose),
        "not" => unary_op(args, rt_not),
        "null" | "true" | "false" => unary_op(args, |_| Ok(match name {
            "null" => Value::Null,
            "true" => Value::True,
            "false" => Value::False,
            _ => unreachable!(),
        })),
        "empty" => bail!("empty"),
        "error" => {
            let input = &args[0];
            let msg = crate::value::value_to_json(input);
            bail!("{}", msg);
        }
        "nan" => Ok(Value::number(f64::NAN)),
        "infinite" => Ok(Value::number(f64::INFINITY)),
        "isinfinite" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::from_bool(n.is_infinite())),
            _ => Ok(Value::False),
        }),
        "isnan" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::from_bool(n.is_nan())),
            _ => Ok(Value::False),
        }),
        "isnormal" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::from_bool(n.is_normal())),
            _ => Ok(Value::False),
        }),
        "isfinite" => unary_op(args, |v| match v {
            // jq's isfinite is `type == "number" and (isinfinite | not)`,
            // so NaN counts as finite (issue #108).
            Value::Num(n, _) => Ok(Value::from_bool(!n.is_infinite())),
            _ => Ok(Value::False),
        }),
        "env" | "$ENV" => Ok(rt_env()),
        "builtins" => Ok(rt_builtins()),
        "debug" => unary_op(args, |v| {
            eprintln!("[\"DEBUG:\",{}]", crate::value::value_to_json(v));
            Ok(v.clone())
        }),
        "stderr" => unary_op(args, |v| {
            eprint!("{}", crate::value::value_to_json(v));
            Ok(v.clone())
        }),
        "input" | "inputs" => {
            // These need special handling with the input source
            Ok(Value::Null)
        }
        "now" => Ok(Value::number(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64())),
        "path" => {
            // path() needs special handling
            Ok(Value::Arr(Rc::new(vec![])))
        }

        // Binary (nargs=2: input, arg)
        "has" => binary_arg(args, rt_has),
        "in" => binary_arg(args, rt_in),
        "contains" => binary_arg(args, rt_contains),
        "inside" => binary_arg(args, |a, b| rt_contains(b, a)),
        "startswith" => binary_arg(args, rt_startswith),
        "endswith" => binary_arg(args, rt_endswith),
        "exec" => binary_arg(args, rt_exec),
        "execv" => binary_arg(args, rt_execv),
        "ltrimstr" => binary_arg(args, rt_ltrimstr),
        "rtrimstr" => binary_arg(args, rt_rtrimstr),
        "split" if args.len() <= 2 => binary_arg(args, rt_split),
        "split" => {
            // split(re; flags) — regex split
            let pat_str = coerce_regex_pat_str(&args[1])?;
            let (pat, _) = apply_regex_flags(&pat_str, &args[2])?;
            rt_regex_split(&args[0], &Value::from_string(pat), &Value::Null)
        }
        "join" => binary_arg(args, rt_join),
        "index" | "rindex" => binary_arg(args, |a, b| rt_str_index(a, b, name == "rindex")),
        "indices" | "rindices" => binary_arg(args, rt_indices),
        "test" => {
            if args.len() >= 3 {
                // 1-arg form is the only one that accepts array patterns;
                // we infer it from a Null flags arg (parser default). See
                // #461 for the heuristic.
                let allow_array = matches!(&args[2], Value::Null);
                let (pat_str, arr_flags) = coerce_regex_pat_or_array(&args[1], allow_array)?;
                let combined = if matches!(&args[2], Value::Null) {
                    Value::from_string(arr_flags)
                } else {
                    args[2].clone()
                };
                let (pat, _) = apply_regex_flags(&pat_str, &combined)?;
                rt_test(&args[0], &Value::from_string(pat))
            } else {
                binary_arg(args, rt_test)
            }
        }
        "match" => {
            if args.len() >= 3 {
                let allow_array = matches!(&args[2], Value::Null);
                let (pat_str, arr_flags) = coerce_regex_pat_or_array(&args[1], allow_array)?;
                let combined = if matches!(&args[2], Value::Null) {
                    Value::from_string(arr_flags)
                } else {
                    args[2].clone()
                };
                let (pat, global) = apply_regex_flags(&pat_str, &combined)?;
                let re = Value::from_string(pat);
                if global {
                    rt_match_global(&args[0], &re)
                } else {
                    rt_match(&args[0], &re)
                }
            } else {
                binary_arg(args, rt_match)
            }
        }
        "capture" => {
            if args.len() >= 3 {
                let allow_array = matches!(&args[2], Value::Null);
                let (pat_str, arr_flags) = coerce_regex_pat_or_array(&args[1], allow_array)?;
                let combined = if matches!(&args[2], Value::Null) {
                    Value::from_string(arr_flags)
                } else {
                    args[2].clone()
                };
                let (pat, global) = apply_regex_flags(&pat_str, &combined)?;
                let re = Value::from_string(pat);
                if global {
                    rt_capture_global(&args[0], &re)
                } else {
                    rt_capture(&args[0], &re)
                }
            } else {
                binary_arg(args, rt_capture)
            }
        }
        "scan" => {
            if args.len() >= 3 {
                let pat_str = coerce_regex_pat_str(&args[1])?;
                let (pat, _) = apply_regex_flags(&pat_str, &args[2])?;
                rt_scan(&args[0], &Value::from_string(pat))
            } else {
                binary_arg(args, rt_scan)
            }
        }
        "sub" | "gsub" => {
            if args.len() >= 4 {
                // sub/gsub with flags: input, regex, replacement, flags
                let pat_str = coerce_regex_pat_str(&args[1])?;
                let (pat, _) = apply_regex_flags(&pat_str, &args[3])?;
                rt_sub_gsub(&args[0], &Value::from_string(pat), &args[2], name == "gsub")
            } else if args.len() >= 3 {
                rt_sub_gsub(&args[0], &args[1], &args[2], name == "gsub")
            } else {
                bail!("{} requires 3 arguments", name);
            }
        }
        "limit" => binary_arg(args, |_a, _b| {
            // limit needs special handling as a generator
            Ok(Value::Null)
        }),
        "first" | "last" | "nth" | "range" | "while" | "until" | "repeat" | "recurse" | "recurse_down"
        | "getpath" | "setpath" | "delpaths" => {
            // These need special handling
            match name {
                "getpath" => binary_arg(args, rt_getpath),
                "setpath" => {
                    if args.len() >= 3 {
                        rt_setpath(&args[0], &args[1], &args[2])
                    } else {
                        bail!("setpath requires 3 arguments");
                    }
                }
                "delpaths" => binary_arg(args, rt_delpaths),
                _ => Ok(Value::Null),
            }
        }
        "sort_by" | "group_by" | "unique_by" | "min_by" | "max_by" => {
            // Closure-based operations need special handling
            Ok(args.first().cloned().unwrap_or(Value::Null))
        }
        "map" | "select" | "map_values" | "with_entries" => {
            Ok(args.first().cloned().unwrap_or(Value::Null))
        }
        // flatten with depth already handled above
        "pow" => ternary_arg(args, rt_pow),
        "atan2" => ternary_arg(args, |a, b| match (a, b) {
            (Value::Num(y, _), Value::Num(x, _)) => Ok(Value::number(y.atan2(*x))),
            _ => bail!("{} number required", errdesc(if !matches!(a, Value::Num(..)) { a } else { b })),
        }),
        "log" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.ln())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "log2" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.log2())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "log10" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.log10())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "exp" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.exp())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "exp2" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(2f64.powf(*n))),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "sin" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.sin())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "cos" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.cos())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "tan" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.tan())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "asin" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.asin())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "acos" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.acos())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "atan" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.atan())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "sinh" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.sinh())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "cosh" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.cosh())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "tanh" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.tanh())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "asinh" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.asinh())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "acosh" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.acosh())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "atanh" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.atanh())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "cbrt" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(n.cbrt())),
            _ => bail!("{} number required", errdesc(v)),
        }),
        // Additional libc unary wrappers (#473). libc-backed for ULP
        // parity with jq's libm-direct calls.
        "erf" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(unsafe { erf(*n) })),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "erfc" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(unsafe { erfc(*n) })),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "expm1" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(unsafe { expm1(*n) })),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "log1p" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(unsafe { log1p(*n) })),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "y0" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(unsafe { y0(*n) })),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "y1" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(unsafe { y1(*n) })),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "exp10" => unary_op(args, |v| match v {
            // Use `10.powf(n)` rather than `libm::exp10`. jq's libc-backed
            // exp10 reuses pow internally on Apple/Linux, so they agree;
            // libm::exp10 takes a different polynomial path that drops the
            // last decimal digit on values like `1.5`.
            Value::Num(n, _) => Ok(Value::number(10f64.powf(*n))),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "gamma" | "tgamma" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(unsafe { tgamma(*n) })),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "lgamma" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::number(unsafe { lgamma(*n) })),
            _ => bail!("{} number required", errdesc(v)),
        }),
        "lgamma_r" => unary_op(args, |v| match v {
            Value::Num(n, _) => {
                let mut sign: i32 = 0;
                let r = unsafe { lgamma_r(*n, &mut sign) };
                // Apple's libm leaves `sign` at 0 for `lgamma_r(0)` (a pole
                // where the sign of gamma is undefined). jq's bundled libm
                // (glibc) treats the limit as +inf from positive arguments
                // and returns sign=1; align with that. Negative integers
                // already get sign=1 from Apple's libm. (#529)
                if *n == 0.0 { sign = 1; }
                Ok(Value::Arr(Rc::new(vec![Value::number(r), Value::number(sign as f64)])))
            }
            _ => bail!("{} number required", errdesc(v)),
        }),
        "frexp" => unary_op(args, |v| match v {
            Value::Num(n, _) => {
                let (m, e) = libm::frexp(*n);
                Ok(Value::Arr(Rc::new(vec![Value::number(m), Value::number(e as f64)])))
            }
            _ => bail!("{} number required", errdesc(v)),
        }),
        "hypot" => ternary_arg(args, |a, b| match (a, b) {
            (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::number(libm::hypot(*x, *y))),
            _ => bail!("{} number required", errdesc(if !matches!(a, Value::Num(..)) { a } else { b })),
        }),
        "remainder" | "drem" => ternary_arg(args, |a, b| match (a, b) {
            (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::number(libm::remainder(*x, *y))),
            _ => bail!("{} number required", errdesc(if !matches!(a, Value::Num(..)) { a } else { b })),
        }),
        "ldexp" | "scalbn" | "scalbln" => ternary_arg(args, |a, b| match (a, b) {
            (Value::Num(x, _), Value::Num(y, _)) => {
                Ok(Value::number(libm::scalbn(*x, *y as i32)))
            }
            _ => bail!("{} number required", errdesc(if !matches!(a, Value::Num(..)) { a } else { b })),
        }),
        "scalb" => ternary_arg(args, |a, b| match (a, b) {
            (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::number(*x * 2f64.powf(*y))),
            _ => bail!("{} number required", errdesc(if !matches!(a, Value::Num(..)) { a } else { b })),
        }),
        // Additional libc binary wrappers (#473).
        "copysign" => ternary_arg(args, |a, b| match (a, b) {
            (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::number(unsafe { copysign(*x, *y) })),
            _ => bail!("{} number required", errdesc(if !matches!(a, Value::Num(..)) { a } else { b })),
        }),
        "fdim" => ternary_arg(args, |a, b| match (a, b) {
            (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::number(unsafe { fdim(*x, *y) })),
            _ => bail!("{} number required", errdesc(if !matches!(a, Value::Num(..)) { a } else { b })),
        }),
        "fmax" => ternary_arg(args, |a, b| match (a, b) {
            (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::number(unsafe { fmax(*x, *y) })),
            _ => bail!("{} number required", errdesc(if !matches!(a, Value::Num(..)) { a } else { b })),
        }),
        "fmin" => ternary_arg(args, |a, b| match (a, b) {
            (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::number(unsafe { fmin(*x, *y) })),
            _ => bail!("{} number required", errdesc(if !matches!(a, Value::Num(..)) { a } else { b })),
        }),
        "fmod" => ternary_arg(args, |a, b| match (a, b) {
            (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::number(unsafe { fmod(*x, *y) })),
            _ => bail!("{} number required", errdesc(if !matches!(a, Value::Num(..)) { a } else { b })),
        }),
        "nextafter" | "nexttoward" => ternary_arg(args, |a, b| match (a, b) {
            (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::number(unsafe { nextafter(*x, *y) })),
            _ => bail!("{} number required", errdesc(if !matches!(a, Value::Num(..)) { a } else { b })),
        }),
        "jn" => ternary_arg(args, |a, b| match (a, b) {
            (Value::Num(n, _), Value::Num(x, _)) => Ok(Value::number(unsafe { jn(*n as i32, *x) })),
            _ => bail!("{} number required", errdesc(if !matches!(a, Value::Num(..)) { a } else { b })),
        }),
        "yn" => ternary_arg(args, |a, b| match (a, b) {
            (Value::Num(n, _), Value::Num(x, _)) => Ok(Value::number(unsafe { yn(*n as i32, *x) })),
            _ => bail!("{} number required", errdesc(if !matches!(a, Value::Num(..)) { a } else { b })),
        }),
        "fma" => {
            // jq registers fma as fma/3: `fma(a; b; c) = a * b + c` with the
            // pipeline input ignored. Args here = [input, a, b, c].
            if args.len() != 4 { bail!("fma: wrong number of args ({})", args.len()); }
            match (&args[1], &args[2], &args[3]) {
                (Value::Num(a, _), Value::Num(b, _), Value::Num(c, _)) => {
                    Ok(Value::number(libm::fma(*a, *b, *c)))
                }
                _ => {
                    let bad = if !matches!(args[1], Value::Num(..)) { &args[1] }
                              else if !matches!(args[2], Value::Num(..)) { &args[2] }
                              else { &args[3] };
                    bail!("{} number required", errdesc(bad));
                }
            }
        },
        "significand" | "exponent" | "logb" | "nearbyint" | "trunc" | "rint" | "j0" | "j1" => {
            unary_op(args, |v| match v {
                Value::Num(n, _) => {
                    let r = match name {
                        "significand" => {
                            // significand(x) = x * 2^(-ilogb(x))
                            let e = libm::ilogb(*n);
                            *n * 2f64.powi(-e)
                        }
                        "exponent" => libm::ilogb(*n) as f64,
                        "logb" => unsafe { logb(*n) },
                        "nearbyint" => libm::rint(*n),
                        "trunc" => n.trunc(),
                        "rint" => libm::rint(*n),
                        "j0" => libm::j0(*n),
                        "j1" => libm::j1(*n),
                        _ => unreachable!(),
                    };
                    Ok(Value::number(r))
                }
                _ => bail!("{} number required", errdesc(v)),
            })
        }
        "gmtime" => unary_op(args, rt_gmtime),
        "localtime" => unary_op(args, rt_localtime),
        "mktime" => unary_op(args, rt_mktime),
        "strftime" => binary_arg(args, rt_strftime),
        "strptime" => binary_arg(args, rt_strptime),
        // jq defines `todate` / `fromdate` as aliases for the ISO-8601
        // variants: `todate := todateiso8601`, `fromdate := fromdateiso8601`.
        // `date` is a long-standing jq-jit extension kept for compatibility.
        "todate" | "date" => unary_op(args, rt_toisodate),
        "fromdate" => unary_op(args, rt_fromisodate),
        // Canonical jq names per the docs at
        // https://jqlang.github.io/jq/manual/#Dates. Keep the
        // non-standard `fromisodate` / `toisodate` aliases for backward
        // compatibility with callers that adopted them before #116.
        "fromdateiso8601" | "fromisodate" => unary_op(args, rt_fromisodate),
        "todateiso8601"   | "toisodate"   => unary_op(args, rt_toisodate),
        "trimstr" => binary_arg(args, |a, b| {
            let v = rt_ltrimstr(a, b)?;
            rt_rtrimstr(&v, b)
        }),
        "modulemeta" => {
            // modulemeta takes a module name string as input and returns metadata
            let input = &args[0];
            let _name = match input {
                Value::Str(s) => s.as_str().to_string(),
                _ => bail!("modulemeta requires string input"),
            };
            // Need lib_dirs from env - they're passed through args[1] if available
            // For now, try to find module in common paths
            Ok(Value::object_from_map({
                let mut m = new_objmap();
                m.insert("version".into(), Value::number(0.1));
                m.insert("deps".into(), Value::Arr(Rc::new(vec![])));
                m.insert("defs".into(), Value::Arr(Rc::new(vec![])));
                m
            }))
        }
        "have_decnum" => {
            // We don't have arbitrary precision, return false
            Ok(Value::False)
        }
        "have_literal_numbers" => {
            // jq 1.8.1 returns true (decnum literals are supported in
            // mainline). jq-jit reads literals through f64 so number
            // identity isn't preserved across all magnitudes; the flag
            // still makes more sense as `true` since we DO carry the
            // original repr around for output (`1.0` stays `1.0`). See
            // #473.
            Ok(Value::True)
        }
        "have_sql" => Ok(Value::False),
        "have_bom" => Ok(Value::True),
        "toboolean" => unary_op(args, |v| {
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
        }),
        "bsearch" => {
            // bsearch(target): args[0] = input array, args[1] = target
            if args.len() < 2 { bail!("bsearch requires 2 arguments"); }
            let input = &args[0];
            let target = &args[1];
            match input {
                Value::Arr(a) => {
                    let mut lo: i64 = 0;
                    let mut hi: i64 = a.len() as i64 - 1;
                    while lo <= hi {
                        let mid = (lo + hi) / 2;
                        let cmp = compare_values(&a[mid as usize], target);
                        match cmp {
                            std::cmp::Ordering::Equal => return Ok(Value::number(mid as f64)),
                            std::cmp::Ordering::Less => lo = mid + 1,
                            std::cmp::Ordering::Greater => hi = mid - 1,
                        }
                    }
                    Ok(Value::number(-(lo as f64) - 1.0))
                }
                _ => {
                    let ty = input.type_name();
                    let json = crate::value::value_to_json(input);
                    bail!("{} ({}) cannot be searched from", ty, json);
                }
            }
        }
        "strflocaltime" => {
            // strflocaltime(fmt): args[0] = input, args[1] = format string
            if args.len() < 2 { bail!("strflocaltime requires 2 arguments"); }
            rt_strflocaltime_impl(&args[0], &args[1])
        }
        // Fused: explode | map(. + N) | implode
        "__shift_codepoints__" => {
            if args.len() >= 2 {
                if let (Value::Str(s), Value::Num(n, _)) = (&args[0], &args[1]) {
                    let shift = *n as i32;
                    let mut result = String::with_capacity(s.len());
                    for c in s.chars() {
                        let cp = (c as i32).wrapping_add(shift);
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
                    return Ok(Value::from_string(result));
                }
            }
            bail!("__shift_codepoints__ requires string input and numeric shift");
        }
        _ => {
            // Unknown builtin - try to handle common patterns
            bail!("unknown builtin: {} (nargs={})", name, args.len());
        }
    }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Format a value for error messages the way jq does: "type (value_repr)"
/// Long values are truncated with "..." appended.
pub fn errdesc_pub(v: &Value) -> String { errdesc(v) }

fn errdesc(v: &Value) -> String {
    // For numbers, prefer the canonical form of the original repr (so
    // `1.0` stays `"1.0"`, `1e30` becomes `"1E+30"` matching jq's decnum
    // output) — but only when the repr round-trips exactly through f64.
    // For repr that can't round-trip (e.g. 25+ digit literals) and for
    // computed values without a repr, fall back to Rust's decimal Display
    // so huge magnitudes show "12345678901234568000000000..." like jq's
    // non-decnum form does.
    let json = match v {
        Value::Num(n, NumRepr(repr)) => {
            if n.is_nan() { "null".to_string() }
            else if n.is_infinite() {
                if n.is_sign_positive() { "1.7976931348623157e+308".to_string() }
                else { "-1.7976931348623157e+308".to_string() }
            } else if let Some(canon) = repr.as_ref()
                .filter(|r| crate::value::is_valid_json_number(r) && crate::value::repr_is_exact_for_f64(r, *n))
                .map(|r| crate::value::canonical_repr_bytes(r).into_owned())
            {
                canon
            } else if *n == 0.0 {
                if n.is_sign_negative() { "-0".to_string() } else { "0".to_string() }
            } else if *n == n.trunc() && n.abs() < 1e16 {
                let i = *n as i64;
                if i as f64 == *n { format!("{}", i) } else { format!("{}", n) }
            } else {
                format!("{}", n)
            }
        }
        _ => crate::value::value_to_json(v),
    };
    let tn = v.type_name();
    let jlen = json.len();
    // jq 1.8.1 truncates errdesc previews when the JSON repr is > 14 bytes:
    // it takes the first 11 bytes (backed up to a UTF-8 char boundary) and
    // appends `...`. This matches both strings (no closing `"`) and other
    // types (`array (...`, `object (...`, `number (...`). See #465.
    if jlen > 14 {
        let mut end = 11.min(jlen);
        while end > 0 && !json.is_char_boundary(end) { end -= 1; }
        format!("{} ({}...)", tn, &json[..end])
    } else {
        format!("{} ({})", tn, json)
    }
}

fn unary_op(args: &[Value], f: impl FnOnce(&Value) -> Result<Value>) -> Result<Value> {
    if args.is_empty() {
        bail!("unary op: no arguments");
    }
    f(&args[0])
}

fn binary_op(args: &[Value], f: impl FnOnce(&Value, &Value) -> Result<Value>) -> Result<Value> {
    if args.len() < 3 {
        bail!("binary op: need 3 args, got {}", args.len());
    }
    // args[0] = input (discarded), args[1] = rhs, args[2] = lhs
    // Note: stack order means args[2] is actually lhs, args[1] is rhs
    f(&args[2], &args[1])
}

fn binary_arg(args: &[Value], f: impl FnOnce(&Value, &Value) -> Result<Value>) -> Result<Value> {
    if args.len() < 2 {
        bail!("binary arg: need 2 args, got {}", args.len());
    }
    // args[0] = input, args[1] = argument
    f(&args[0], &args[1])
}

fn ternary_arg(args: &[Value], f: impl FnOnce(&Value, &Value) -> Result<Value>) -> Result<Value> {
    if args.len() < 3 {
        bail!("ternary arg: need 2 explicit args, got {}", args.len().saturating_sub(1));
    }
    // args[0] = pipeline input (unused), args[1] = first explicit arg, args[2] = second explicit arg
    f(&args[1], &args[2])
}

// -----------------------------------------------------------------------
// Arithmetic
// -----------------------------------------------------------------------

pub fn rt_add(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::number(x + y)),
        (Value::Str(x), Value::Str(y)) => {
            let mut cs = x.clone();
            cs.push_str(y.as_str());
            Ok(Value::Str(cs))
        }
        (Value::Arr(x), Value::Arr(y)) => {
            let mut result = (**x).clone();
            result.extend((**y).iter().cloned());
            Ok(Value::Arr(Rc::new(result)))
        }
        (Value::Obj(ObjInner(x)), Value::Obj(ObjInner(y))) => {
            let mut result = (**x).clone();
            for (k, v) in y.iter() {
                result.insert(k.clone(), v.clone());
            }
            Ok(Value::object_from_map(result))
        }
        (Value::Null, x) | (x, Value::Null) => Ok(x.clone()),
        _ => bail!(
            "{} and {} cannot be added",
            errdesc(a),
            errdesc(b)
        ),
    }
}

/// Like rt_add but takes ownership of the left operand for in-place mutation.
pub fn rt_add_owned(a: Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::number(x + y)),
        (Value::Str(mut x), Value::Str(y)) => {
            x.push_str(y.as_str());
            Ok(Value::Str(x))
        }
        (Value::Arr(x), Value::Arr(y)) => {
            match Rc::try_unwrap(x) {
                Ok(mut vec) => {
                    vec.extend(y.iter().cloned());
                    Ok(Value::Arr(Rc::new(vec)))
                }
                Err(x) => {
                    let mut result = (*x).clone();
                    result.extend(y.iter().cloned());
                    Ok(Value::Arr(Rc::new(result)))
                }
            }
        }
        (Value::Obj(ObjInner(x)), Value::Obj(ObjInner(y))) => {
            match Rc::try_unwrap(x) {
                Ok(mut map) => {
                    for (k, v) in y.iter() {
                        map.insert(k.clone(), v.clone());
                    }
                    Ok(Value::object_from_map(map))
                }
                Err(x) => {
                    let mut result = (*x).clone();
                    for (k, v) in y.iter() {
                        result.insert(k.clone(), v.clone());
                    }
                    Ok(Value::object_from_map(result))
                }
            }
        }
        (Value::Null, x) => Ok(x.clone()),
        (a, Value::Null) => Ok(a),
        (a, b) => bail!(
            "{} and {} cannot be added",
            errdesc(&a),
            errdesc(b)
        ),
    }
}

pub fn rt_sub(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::number(x - y)),
        (Value::Arr(x), Value::Arr(y)) => {
            let result: Vec<Value> = x.iter()
                .filter(|v| !y.contains(v))
                .cloned()
                .collect();
            Ok(Value::Arr(Rc::new(result)))
        }
        _ => bail!(
            "{} and {} cannot be subtracted",
            errdesc(a),
            errdesc(b)
        ),
    }
}

pub fn rt_mul(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::number(x * y)),
        (Value::Str(s), Value::Num(n, _)) | (Value::Num(n, _), Value::Str(s)) => {
            if n.is_nan() || *n < 0.0 {
                Ok(Value::Null)
            } else {
                let count = *n as usize;
                let result_len = s.len().saturating_mul(count);
                if result_len > 536870911 {
                    bail!("Repeat string result too long");
                }
                Ok(Value::from_string(s.as_str().repeat(count)))
            }
        }
        (Value::Obj(ObjInner(x)), Value::Obj(ObjInner(y))) => {
            // Object multiplication = recursive merge
            Ok(Value::object_from_map(merge_objects(x, y)))
        }
        _ => bail!(
            "{} and {} cannot be multiplied",
            errdesc(a),
            errdesc(b)
        ),
    }
}

fn merge_objects(a: &ObjMap, b: &ObjMap) -> ObjMap {
    let mut result = a.clone();
    for (k, v) in b.iter() {
        if let Some(existing) = result.get(k) {
            if let (Value::Obj(ObjInner(ea)), Value::Obj(ObjInner(eb))) = (existing, v) {
                result.insert(k.clone(), Value::object_from_map(merge_objects(ea, eb)));
                continue;
            }
        }
        result.insert(k.clone(), v.clone());
    }
    result
}

pub fn rt_div(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x, _), Value::Num(y, _)) => {
            if *y == 0.0 {
                bail!("{} and {} cannot be divided because the divisor is zero", errdesc(a), errdesc(b));
            }
            Ok(Value::number(x / y))
        }
        (Value::Str(s), Value::Str(sep)) => {
            // String division = split. Mirror jq's `split(s; sep)`: an empty
            // separator splits per codepoint without leading/trailing empty
            // bookends. Rust's `&str::split("")` would insert those bookends,
            // diverging from jq.
            if s.is_empty() {
                return Ok(Value::Arr(Rc::new(Vec::new())));
            }
            if sep.is_empty() {
                let mut parts = Vec::with_capacity(s.len());
                if s.is_ascii() {
                    for i in 0..s.len() {
                        parts.push(Value::from_str(&s.as_str()[i..i + 1]));
                    }
                } else {
                    let mut buf = [0u8; 4];
                    for c in s.chars() {
                        parts.push(Value::from_str(c.encode_utf8(&mut buf)));
                    }
                }
                return Ok(Value::Arr(Rc::new(parts)));
            }
            let parts: Vec<Value> = s.split(sep.as_str())
                .map(Value::from_str)
                .collect();
            Ok(Value::Arr(Rc::new(parts)))
        }
        _ => bail!(
            "{} and {} cannot be divided",
            errdesc(a),
            errdesc(b)
        ),
    }
}

/// jq's `%` semantics for f64 inputs: truncate both operands toward zero,
/// then take the integer remainder. Returns None when either operand is
/// non-finite or the divisor truncates to zero (caller decides the fallback).
/// Generic over `f64` and `&f64` so call sites can pass through whatever
/// shape they hold without per-site dereferencing.
#[inline]
pub fn jq_mod_f64<A: std::borrow::Borrow<f64>, B: std::borrow::Borrow<f64>>(a: A, b: B) -> Option<f64> {
    let a = *a.borrow();
    let b = *b.borrow();
    if !a.is_finite() || !b.is_finite() { return None; }
    let yi = b as i64;
    if yi == 0 { return None; }
    let xi = a as i64;
    let r = if xi == i64::MIN && yi == -1 { 0 } else { xi % yi };
    Some(r as f64)
}

pub fn rt_mod(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x, _), Value::Num(y, _)) => {
            // NaN inputs → NaN output
            if x.is_nan() || y.is_nan() {
                return Ok(Value::number(f64::NAN));
            }
            if *y == 0.0 {
                bail!("{} and {} cannot be divided (remainder) because the divisor is zero", errdesc(a), errdesc(b));
            }
            // jq uses integer modulo (casting to long long first)
            let xi = *x as i64;
            let yi = *y as i64;
            if yi == 0 {
                bail!("{} and {} cannot be divided (remainder) because the divisor is zero", errdesc(a), errdesc(b));
            }
            // Avoid overflow for i64::MIN % -1
            if xi == i64::MIN && yi == -1 {
                Ok(Value::number(0.0))
            } else {
                Ok(Value::number((xi % yi) as f64))
            }
        }
        _ => bail!(
            "{} and {} cannot be divided (remainder)",
            errdesc(a),
            errdesc(b)
        ),
    }
}

// -----------------------------------------------------------------------
// Comparison
// -----------------------------------------------------------------------

fn rt_eq(a: &Value, b: &Value) -> Result<Value> {
    Ok(Value::from_bool(values_equal(a, b)))
}

fn rt_ne(a: &Value, b: &Value) -> Result<Value> {
    Ok(Value::from_bool(!values_equal(a, b)))
}

fn rt_lt(a: &Value, b: &Value) -> Result<Value> {
    Ok(Value::from_bool(compare_values(a, b) == std::cmp::Ordering::Less))
}

fn rt_gt(a: &Value, b: &Value) -> Result<Value> {
    Ok(Value::from_bool(compare_values(a, b) == std::cmp::Ordering::Greater))
}

fn rt_le(a: &Value, b: &Value) -> Result<Value> {
    Ok(Value::from_bool(compare_values(a, b) != std::cmp::Ordering::Greater))
}

fn rt_ge(a: &Value, b: &Value) -> Result<Value> {
    Ok(Value::from_bool(compare_values(a, b) != std::cmp::Ordering::Less))
}

#[inline]
pub fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::True, Value::True) => true,
        (Value::False, Value::False) => true,
        (Value::Num(x, _), Value::Num(y, _)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Arr(x), Value::Arr(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| values_equal(a, b))
        }
        (Value::Obj(ObjInner(x)), Value::Obj(ObjInner(y))) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|yv| values_equal(v, yv)))
        }
        _ => false,
    }
}

/// jq 1.8.1 puts NaN below every number (and reflexively below itself)
/// so `sort`, `unique`, `min`/`max`, and the `<`/`>` operators stay
/// total over numeric inputs. IEEE 754's "all comparisons false" would
/// leave NaNs scattered. (`==` keeps IEEE 754 inequality so
/// `nan == nan` is still false.)
#[inline]
pub fn cmp_num_jq(x: f64, y: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (x.is_nan(), y.is_nan()) {
        (true, true) => Ordering::Less,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
    }
}

#[inline]
pub fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let type_order = |v: &Value| -> u8 {
        match v {
            Value::Null => 0,
            Value::False => 1,
            Value::True => 2,
            Value::Num(_, _) => 3,
            Value::Str(_) => 4,
            Value::Arr(_) => 5,
            Value::Obj(_) => 6,
            Value::Error(_) => 7,
        }
    };

    let ta = type_order(a);
    let tb = type_order(b);
    if ta != tb {
        return ta.cmp(&tb);
    }

    match (a, b) {
        (Value::Num(x, _), Value::Num(y, _)) => cmp_num_jq(*x, *y),
        (Value::Str(x), Value::Str(y)) => x.cmp(y),
        (Value::Arr(x), Value::Arr(y)) => {
            for (a, b) in x.iter().zip(y.iter()) {
                let ord = compare_values(a, b);
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            x.len().cmp(&y.len())
        }
        (Value::Obj(ObjInner(x)), Value::Obj(ObjInner(y))) => {
            // jq's object ordering: compare the *full* sorted-keys
            // arrays first, then — only if they're equal — compare
            // values key by key. The previous implementation
            // interleaved key and value compares per-position, which
            // got the simple `{"c":null}` vs `{"c":false}` shape right
            // but flipped `{"c":null,"y":null}` vs `{"c":false}` (the
            // shorter object should come first when its keys are a
            // strict prefix).
            let mut xkeys: Vec<&KeyStr> = x.keys().collect();
            let mut ykeys: Vec<&KeyStr> = y.keys().collect();
            xkeys.sort();
            ykeys.sort();
            for (xk, yk) in xkeys.iter().zip(ykeys.iter()) {
                let kord = xk.cmp(yk);
                if kord != Ordering::Equal {
                    return kord;
                }
            }
            let len_ord = xkeys.len().cmp(&ykeys.len());
            if len_ord != Ordering::Equal {
                return len_ord;
            }
            for xk in &xkeys {
                let xv = x.get(xk.as_str()).unwrap();
                let yv = y.get(xk.as_str()).unwrap();
                let vord = compare_values(xv, yv);
                if vord != Ordering::Equal {
                    return vord;
                }
            }
            Ordering::Equal
        }
        _ => Ordering::Equal,
    }
}

// -----------------------------------------------------------------------
// Unary builtins
// -----------------------------------------------------------------------

fn rt_length(v: &Value) -> Result<Value> {
    v.length()
}

fn rt_utf8bytelength(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => Ok(Value::number(s.len() as f64)),
        _ => bail!("{} ({}) only strings have UTF-8 byte length", v.type_name(), crate::value::value_to_json(v)),
    }
}

fn rt_type(v: &Value) -> Result<Value> {
    Ok(Value::from_str(v.type_name()))
}

fn rt_keys(v: &Value, sorted: bool) -> Result<Value> {
    match v {
        Value::Obj(ObjInner(o)) => {
            let mut keys: Vec<Value> = o.keys().map(|k| Value::from_str(k)).collect();
            if sorted {
                keys.sort_by(compare_values);
            }
            Ok(Value::Arr(Rc::new(keys)))
        }
        Value::Arr(a) => {
            let keys: Vec<Value> = (0..a.len()).map(|i| Value::number(i as f64)).collect();
            Ok(Value::Arr(Rc::new(keys)))
        }
        _ => bail!("{} has no keys", errdesc(v)),
    }
}

fn rt_values(v: &Value) -> Result<Value> {
    match v {
        Value::Obj(ObjInner(o)) => {
            let vals: Vec<Value> = o.values().cloned().collect();
            Ok(Value::Arr(Rc::new(vals)))
        }
        Value::Arr(_) => Ok(v.clone()),
        _ => bail!("{} has no values", v.type_name()),
    }
}

fn rt_sort(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) => {
            let mut sorted = (**a).clone();
            sort_values(&mut sorted);
            Ok(Value::Arr(Rc::new(sorted)))
        }
        // jq's wording: `<type> (<value>) cannot be sorted, as it is not
        // an array`. Same for unique. See #450.
        _ => bail!("{} cannot be sorted, as it is not an array", errdesc(v)),
    }
}

/// Sort a Value slice using specialized comparators for homogeneous types.
fn sort_values(sorted: &mut [Value]) {
    if sorted.len() <= 1 { return; }
    // Check first element type to select fast comparator
    match &sorted[0] {
        Value::Num(..) => {
            // Optimistic: try numeric sort, fall back if mixed types.
            // NaN must rank below every number (jq's total order, #115);
            // partial_cmp would scatter NaNs as Equal-ish.
            sorted.sort_by(|a, b| {
                if let (Value::Num(x, _), Value::Num(y, _)) = (a, b) {
                    cmp_num_jq(*x, *y)
                } else {
                    compare_values(a, b)
                }
            });
        }
        Value::Str(..) => {
            sorted.sort_by(|a, b| {
                if let (Value::Str(x), Value::Str(y)) = (a, b) {
                    x.cmp(y)
                } else {
                    compare_values(a, b)
                }
            });
        }
        _ => sorted.sort_by(compare_values),
    }
}

fn rt_reverse(v: &Value) -> Result<Value> {
    // jq 1.8.1 defines `reverse` as `[.[length - 1 - range(0; length)]]`. The
    // shape that falls out:
    //   - non-empty arrays reverse normally;
    //   - empty containers (null/""/{}/[]) yield `[]` (range empty -> no
    //     index fires);
    //   - everything else errors via the `.[idx]` step (#197).
    match v {
        Value::Arr(a) => {
            let mut reversed = (**a).clone();
            reversed.reverse();
            Ok(Value::Arr(Rc::new(reversed)))
        }
        Value::Null => Ok(Value::Arr(Rc::new(vec![]))),
        Value::Str(s) if s.is_empty() => Ok(Value::Arr(Rc::new(vec![]))),
        Value::Obj(ObjInner(o)) if o.is_empty() => Ok(Value::Arr(Rc::new(vec![]))),
        // n == 0.0 covers both +0 and -0; range(0; 0) is empty so the
        // `[.[length-1, length-2 ..0]]` desugar yields []. Non-zero
        // (including NaN) errors via the `.[idx]` step (#328).
        Value::Num(n, _) if *n == 0.0 => Ok(Value::Arr(Rc::new(vec![]))),
        Value::Str(_) => bail!("Cannot index string with number"),
        Value::Obj(_) => bail!("Cannot index object with number"),
        Value::Num(_, _) => bail!("Cannot index number with number"),
        Value::True | Value::False => bail!("{} ({}) has no length", v.type_name(), crate::value::value_to_json(v)),
        _ => bail!("{} cannot be reversed", v.type_name()),
    }
}

fn rt_flatten(v: &Value, depth: Option<usize>) -> Result<Value> {
    match v {
        Value::Arr(a) => {
            let mut result = Vec::new();
            flatten_inner(a, depth.unwrap_or(usize::MAX), &mut result);
            Ok(Value::Arr(Rc::new(result)))
        }
        // jq treats an object's values as the array to flatten, then runs
        // the same recursive logic on it (#184). `flatten` on `{"a":[1,2]}`
        // becomes `[[1,2]] | flatten` = `[1,2]`.
        Value::Obj(ObjInner(o)) => {
            let values: Vec<Value> = o.values().cloned().collect();
            let mut result = Vec::new();
            flatten_inner(&values, depth.unwrap_or(usize::MAX), &mut result);
            Ok(Value::Arr(Rc::new(result)))
        }
        _ => bail!("Cannot iterate over {}", errdesc(v)),
    }
}

fn flatten_inner(arr: &[Value], depth: usize, out: &mut Vec<Value>) {
    for v in arr {
        if depth > 0 {
            if let Value::Arr(inner) = v {
                flatten_inner(inner, depth - 1, out);
                continue;
            }
        }
        out.push(v.clone());
    }
}

fn rt_unique(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) => {
            let mut sorted = (**a).clone();
            sort_values(&mut sorted);
            sorted.dedup_by(|a, b| values_equal(a, b));
            Ok(Value::Arr(Rc::new(sorted)))
        }
        // Shares wording with `sort` (#450).
        _ => bail!("{} cannot be sorted, as it is not an array", errdesc(v)),
    }
}

fn rt_min(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) if a.is_empty() => Ok(Value::Null),
        Value::Arr(a) => {
            let mut min = &a[0];
            for v in &a[1..] {
                if compare_values(v, min) == std::cmp::Ordering::Less {
                    min = v;
                }
            }
            Ok(min.clone())
        }
        // jq's wording for min/max — implemented as a fold that compares
        // pairs, so the value appears twice. See #450.
        _ => bail!("{} and {} cannot be iterated over", errdesc(v), errdesc(v)),
    }
}

fn rt_max(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) if a.is_empty() => Ok(Value::Null),
        Value::Arr(a) => {
            let mut max = &a[0];
            for v in &a[1..] {
                if compare_values(v, max) == std::cmp::Ordering::Greater {
                    max = v;
                }
            }
            Ok(max.clone())
        }
        // Shares wording with min (#450).
        _ => bail!("{} and {} cannot be iterated over", errdesc(v), errdesc(v)),
    }
}

fn rt_add_all(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) if a.is_empty() => Ok(Value::Null),
        Value::Arr(a) => {
            // Fast path: detect homogeneous types for O(n) pre-allocated merge
            match &a[0] {
                Value::Num(_, _) => {
                    // Check if all elements are numbers
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
                        return Ok(Value::number(sum));
                    }
                }
                Value::Arr(_) => {
                    // Check if all elements are arrays (or null)
                    let mut total = 0usize;
                    let mut all_arr = true;
                    for item in a.iter() {
                        match item {
                            Value::Arr(sub) => total += sub.len(),
                            Value::Null => {}
                            _ => { all_arr = false; break; }
                        }
                    }
                    if all_arr {
                        let mut result = Vec::with_capacity(total);
                        for item in a.iter() {
                            if let Value::Arr(sub) = item {
                                result.extend(sub.iter().cloned());
                            }
                        }
                        return Ok(Value::Arr(Rc::new(result)));
                    }
                }
                Value::Str(_) => {
                    // Check if all elements are strings (or null)
                    let mut total = 0usize;
                    let mut all_str = true;
                    for item in a.iter() {
                        match item {
                            Value::Str(s) => total += s.len(),
                            Value::Null => {}
                            _ => { all_str = false; break; }
                        }
                    }
                    if all_str {
                        let mut result = String::with_capacity(total);
                        for item in a.iter() {
                            if let Value::Str(s) = item {
                                result.push_str(s.as_str());
                            }
                        }
                        return Ok(Value::from_string(result));
                    }
                }
                Value::Obj(_) => {
                    // Check if all elements are objects (or null)
                    let mut total = 0usize;
                    let mut all_obj = true;
                    for item in a.iter() {
                        match item {
                            Value::Obj(ObjInner(o)) => total += o.len(),
                            Value::Null => {}
                            _ => { all_obj = false; break; }
                        }
                    }
                    if all_obj {
                        // O(n) merge using HashMap index for deduplication
                        use std::collections::HashMap;
                        let mut key_index: HashMap<crate::value::KeyStr, usize> = HashMap::with_capacity(total);
                        let mut entries: Vec<(crate::value::KeyStr, Value)> = Vec::with_capacity(total);
                        for item in a.iter() {
                            if let Value::Obj(ObjInner(o)) = item {
                                for (k, v) in o.iter() {
                                    if let Some(&idx) = key_index.get(k) {
                                        entries[idx].1 = v.clone();
                                    } else {
                                        key_index.insert(k.clone(), entries.len());
                                        entries.push((k.clone(), v.clone()));
                                    }
                                }
                            }
                        }
                        let mut result = crate::value::ObjMap::with_capacity(entries.len());
                        for (k, v) in entries {
                            result.push_unique(k, v);
                        }
                        return Ok(Value::object_from_map(result));
                    }
                }
                _ => {}
            }
            // General fallback: sequential add
            let mut result = a[0].clone();
            for item in &a[1..] {
                result = rt_add(&result, item)?;
            }
            Ok(result)
        }
        Value::Obj(ObjInner(o)) => {
            let mut iter = o.values();
            let Some(first) = iter.next() else { return Ok(Value::Null); };
            let mut result = first.clone();
            for item in iter {
                result = rt_add(&result, item)?;
            }
            Ok(result)
        }
        _ => bail!("Cannot iterate over {} ({})", v.type_name(), crate::value::value_to_json(v)),
    }
}

fn rt_any(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) => Ok(Value::from_bool(a.iter().any(|v| v.is_true()))),
        Value::Obj(ObjInner(o)) => Ok(Value::from_bool(o.values().any(|v| v.is_true()))),
        _ => bail!("Cannot iterate over {}", errdesc(v)),
    }
}

fn rt_all(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) => Ok(Value::from_bool(a.iter().all(|v| v.is_true()))),
        Value::Obj(ObjInner(o)) => Ok(Value::from_bool(o.values().all(|v| v.is_true()))),
        _ => bail!("Cannot iterate over {}", errdesc(v)),
    }
}

fn rt_floor(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n, _) => Ok(Value::number(n.floor())),
        _ => bail!("{} number required", errdesc(v)),
    }
}

fn rt_ceil(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n, _) => Ok(Value::number(n.ceil())),
        _ => bail!("{} number required", errdesc(v)),
    }
}

fn rt_round(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n, _) => Ok(Value::number(n.round())),
        _ => bail!("{} number required", errdesc(v)),
    }
}

fn rt_fabs(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n, NumRepr(repr)) => {
            if *n >= 0.0 { Ok(Value::number_opt(*n, repr.clone())) }
            else { Ok(Value::number(n.abs())) }
        }
        _ => bail!("{} number required", errdesc(v)),
    }
}

fn rt_abs(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n, NumRepr(repr)) => {
            if *n >= 0.0 { Ok(Value::number_opt(*n, repr.clone())) }
            else { Ok(Value::number(n.abs())) }
        }
        Value::Str(_) | Value::Arr(_) | Value::Obj(_) => Ok(v.clone()),
        _ => {
            bail!("{} ({}) cannot be negated", v.type_name(), crate::value::value_to_json(v))
        }
    }
}

fn rt_sqrt(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n, _) => Ok(Value::number(n.sqrt())),
        _ => bail!("{} number required", errdesc(v)),
    }
}

fn rt_tostring(v: &Value) -> Result<Value> {
    match v {
        Value::Str(_) => Ok(v.clone()),
        // Preserve the lexical form of numbers (#75 / #110) — `-0` stays `"-0"`,
        // `0.0` stays `"0.0"`, etc. `value_to_json_tojson` honours the repr
        // when f64 can round-trip it exactly, otherwise falls back to the
        // canonical f64 form (which matches jq's no-decnum rounding, cf. the
        // `13911860366432393` regression test).
        _ => Ok(Value::from_string(crate::value::value_to_json_tojson(v))),
    }
}

fn rt_tonumber(v: &Value) -> Result<Value> {
    match v {
        Value::Num(_, _) => Ok(v.clone()),
        Value::Str(s) => {
            let s_ref: &str = s.as_ref();
            if s_ref.is_empty() {
                bail!("{} cannot be parsed as a number", errdesc(v));
            }
            // jq rejects leading/trailing whitespace
            if s_ref.as_bytes()[0].is_ascii_whitespace() || s_ref.as_bytes()[s_ref.len()-1].is_ascii_whitespace() {
                bail!("{} cannot be parsed as a number", errdesc(v));
            }
            // Strip leading '+' for compatibility with jq
            let parse_str = s_ref.strip_prefix('+').unwrap_or(s_ref);
            match fast_float::parse(parse_str) {
                Ok(n) => {
                    // Mirror parse_json_number: preserve repr when the canonical
                    // decnum-style form differs from the f64 default. Without
                    // this, `"1e10" | tonumber` collapses to `10000000000`
                    // instead of jq's `1E+10`. See #428.
                    let canonical = crate::value::normalize_jq_repr(parse_str)
                        .unwrap_or_else(|| parse_str.to_string());
                    let f64_repr = crate::value::format_jq_number(n);
                    if canonical != f64_repr && crate::value::is_valid_json_number(parse_str) {
                        Ok(Value::number_with_repr(n, Rc::from(parse_str)))
                    } else {
                        Ok(Value::number(n))
                    }
                }
                Err(_) => bail!("{} cannot be parsed as a number", errdesc(v)),
            }
        }
        _ => bail!("{} cannot be parsed as a number", errdesc(v)),
    }
}

fn rt_ascii_downcase(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => {
            if s.is_ascii() {
                let mut bytes = s.as_bytes().to_vec();
                bytes.make_ascii_lowercase();
                // Safety: input was ASCII, lowercase is still ASCII
                Ok(Value::from_string(unsafe { String::from_utf8_unchecked(bytes) }))
            } else {
                Ok(Value::from_string(s.chars().map(|c| if c.is_ascii() { c.to_ascii_lowercase() } else { c }).collect()))
            }
        }
        _ => bail!("explode input must be a string"),
    }
}

fn rt_ascii_upcase(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => {
            if s.is_ascii() {
                let mut bytes = s.as_bytes().to_vec();
                bytes.make_ascii_uppercase();
                Ok(Value::from_string(unsafe { String::from_utf8_unchecked(bytes) }))
            } else {
                Ok(Value::from_string(s.chars().map(|c| if c.is_ascii() { c.to_ascii_uppercase() } else { c }).collect()))
            }
        }
        _ => bail!("explode input must be a string"),
    }
}

fn rt_ltrim(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => {
            let trimmed = s.trim_start();
            if trimmed.len() == s.len() { Ok(v.clone()) }
            else { Ok(Value::from_str(trimmed)) }
        }
        _ => bail!("trim input must be a string"),
    }
}

fn rt_rtrim(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => {
            let trimmed = s.trim_end();
            if trimmed.len() == s.len() { Ok(v.clone()) }
            else { Ok(Value::from_str(trimmed)) }
        }
        _ => bail!("trim input must be a string"),
    }
}

fn rt_trim(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => Ok(Value::from_string(s.trim().to_string())),
        _ => bail!("trim input must be a string"),
    }
}

fn rt_explode(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => {
            let str = s.as_str();
            if str.is_ascii() {
                // ASCII fast path: byte value == codepoint, pre-allocate exact size
                let mut codepoints = Vec::with_capacity(str.len());
                for &b in str.as_bytes() {
                    codepoints.push(Value::number(b as f64));
                }
                Ok(Value::Arr(Rc::new(codepoints)))
            } else {
                let codepoints: Vec<Value> = str.chars()
                    .map(|c| Value::number(c as u32 as f64))
                    .collect();
                Ok(Value::Arr(Rc::new(codepoints)))
            }
        }
        _ => bail!("explode input must be a string"),
    }
}

fn rt_implode(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) => {
            let mut s = String::new();
            for item in a.iter() {
                match item {
                    Value::Num(n, _) => {
                        if n.is_nan() || n.is_infinite() {
                            bail!("{} can't be imploded, unicode codepoint needs to be numeric", errdesc(item));
                        }
                        let n_val = *n as i64;
                        // Negative, surrogate range (0xD800-0xDFFF), or > 0x10FFFF → replacement char
                        if n_val < 0 || (0xD800..=0xDFFF).contains(&n_val) || n_val > 0x10FFFF {
                            s.push('\u{FFFD}');
                        } else {
                            match char::from_u32(n_val as u32) {
                                Some(c) => s.push(c),
                                None => s.push('\u{FFFD}'),
                            }
                        }
                    }
                    Value::Str(sv) => {
                        bail!("{} can't be imploded, unicode codepoint needs to be numeric", errdesc(&Value::Str(sv.clone())));
                    }
                    _ => {
                        bail!("{} can't be imploded, unicode codepoint needs to be numeric", errdesc(item));
                    }
                }
            }
            Ok(Value::from_string(s))
        }
        _ => bail!("implode input must be an array"),
    }
}

fn rt_tojson(v: &Value) -> Result<Value> {
    // Preserve the lexical form of numbers so `1.0 | tojson` yields `"1.0"`
    // (not `"1"`). jq 1.8.1 does the same via decnum; jq-jit has only f64, so
    // reprs carrying more precision than f64 can hold (or overflowing to ±inf)
    // fall back to the f64-formatted form inside `value_to_json_tojson`.
    Ok(Value::from_string(crate::value::value_to_json_tojson(v)))
}

fn rt_fromjson(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => {
            // Try the native strict-JSON parser first (fast path for the common
            // case of valid JSON). On any failure, fall through to the
            // jq-compatible parser so error messages match jq 1.8.1 exactly.
            match crate::value::json_to_value(s.trim()) {
                Ok(v) => Ok(v),
                Err(_) => crate::value::json_to_value_fromjson(s),
            }
        }
        _ => bail!("{} only strings can be parsed", errdesc(v)),
    }
}

fn rt_to_entries(v: &Value) -> Result<Value> {
    match v {
        Value::Obj(ObjInner(o)) => {
            let entries: Vec<Value> = o.iter().map(|(k, v)| {
                let mut entry = new_objmap();
                entry.insert("key".into(), Value::from_str(k));
                entry.insert("value".into(), v.clone());
                Value::object_from_map(entry)
            }).collect();
            Ok(Value::Arr(Rc::new(entries)))
        }
        Value::Arr(a) => {
            let entries: Vec<Value> = a.iter().enumerate().map(|(i, v)| {
                let mut entry = new_objmap();
                entry.insert("key".into(), Value::number(i as f64));
                entry.insert("value".into(), v.clone());
                Value::object_from_map(entry)
            }).collect();
            Ok(Value::Arr(Rc::new(entries)))
        }
        // jq implements `to_entries` on top of `keys`, so the error
        // wording is shared: `<type> (<value>) has no keys`. See #448.
        _ => bail!("{} has no keys", errdesc(v)),
    }
}

fn rt_from_entries(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) => {
            let mut obj = new_objmap();
            for entry in a.iter() {
                match entry {
                    Value::Obj(ObjInner(o)) => {
                        // jq resolves the key via `.key // .key_ // .Key // .name // .Name`
                        // — `//` skips null/false — and errors when the resolved key is
                        // not a string. Matches jq 1.8.1 wording.
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
                            Value::Str(s) => s.to_string(),
                            other => bail!(
                                "Cannot use {} ({}) as object key",
                                other.type_name(),
                                crate::value::value_to_json(other),
                            ),
                        };
                        obj.insert(KeyStr::from(key_str), val);
                    }
                    // jq's from_entries desugars to `map(...) | with_entries(.)`.
                    // For null entries, `.key`/`.name` resolve to null, which
                    // then reaches object construction and bails as
                    // `Cannot use null (null) as object key` (#517).
                    Value::Null => bail!("Cannot use null (null) as object key"),
                    // For non-null non-object entries, jq's `.key` access on
                    // the entry surfaces the standard `Cannot index <type>
                    // with string "key"` error.
                    other => bail!(
                        "Cannot index {} with string \"key\"",
                        other.type_name(),
                    ),
                }
            }
            Ok(Value::object_from_map(obj))
        }
        // jq's from_entries desugars to `map(...) | add + {}`. On {} the map yields [],
        // add yields null, and null + {} is {} — so empty objects round-trip to {}.
        Value::Obj(ObjInner(o)) if o.is_empty() => Ok(Value::object_from_map(new_objmap())),
        // jq's stdlib starts with `map(...)` which iterates the input — non-array
        // (and non-empty-object) inputs fail there with the standard
        // `Cannot iterate over <type> (<value>)` wording (#481/#505/#517).
        _ => bail!("Cannot iterate over {}", errdesc(v)),
    }
}

fn rt_transpose(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) => {
            if a.is_empty() {
                return Ok(Value::Arr(Rc::new(vec![])));
            }
            let max_len = a.iter().filter_map(|v| {
                if let Value::Arr(inner) = v { Some(inner.len()) } else { None }
            }).max().unwrap_or(0);

            let mut result = Vec::with_capacity(max_len);
            for i in 0..max_len {
                let mut row = Vec::with_capacity(a.len());
                for item in a.iter() {
                    if let Value::Arr(inner) = item {
                        row.push(inner.get(i).cloned().unwrap_or(Value::Null));
                    } else {
                        row.push(Value::Null);
                    }
                }
                result.push(Value::Arr(Rc::new(row)));
            }
            Ok(Value::Arr(Rc::new(result)))
        }
        _ => bail!("{} cannot be transposed", v.type_name()),
    }
}

fn rt_not(v: &Value) -> Result<Value> {
    Ok(Value::from_bool(!v.is_true()))
}

// -----------------------------------------------------------------------
// Binary builtins
// -----------------------------------------------------------------------

fn rt_has(v: &Value, key: &Value) -> Result<Value> {
    match (v, key) {
        (Value::Obj(ObjInner(o)), Value::Str(k)) => Ok(Value::from_bool(o.contains_key(k.as_str()))),
        (Value::Arr(a), Value::Num(n, _)) => {
            if n.is_nan() || n.is_infinite() { return Ok(Value::False); }
            let idx = *n as i64;
            if idx < 0 { return Ok(Value::False); }
            Ok(Value::from_bool((idx as usize) < a.len()))
        }
        (Value::Null, _) => Ok(Value::False),
        _ => bail!("Cannot check whether {} has a {} key", v.type_name(), key.type_name()),
    }
}

fn rt_in(v: &Value, container: &Value) -> Result<Value> {
    // `in(x)` is defined as `. as $k | x | has($k)`, so the containment check
    // reuses `has` with input as the key. This preserves jq's type-mismatch
    // errors (e.g. non-string key on object) instead of silently returning false.
    rt_has(container, v)
}

fn rt_contains(a: &Value, b: &Value) -> Result<Value> {
    // jq treats true/false as distinct kinds at the top level: contains
    // requires both operands to share a kind, otherwise it errors. Nested
    // comparisons (inside arrays/objects) silently fall back to false.
    if jq_kind_tag(a) != jq_kind_tag(b) {
        bail!(
            "{} and {} cannot have their containment checked",
            errdesc(a),
            errdesc(b)
        );
    }
    Ok(Value::from_bool(value_contains(a, b)))
}

fn jq_kind_tag(v: &Value) -> u8 {
    match v {
        Value::Null => 0,
        Value::False => 1,
        Value::True => 2,
        Value::Num(_, _) => 3,
        Value::Str(_) => 4,
        Value::Arr(_) => 5,
        Value::Obj(_) => 6,
        Value::Error(_) => 7,
    }
}

fn value_contains(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Str(x), Value::Str(y)) => x.contains(y.as_str()),
        (Value::Arr(x), Value::Arr(y)) => {
            y.iter().all(|yv| x.iter().any(|xv| value_contains(xv, yv)))
        }
        (Value::Obj(ObjInner(x)), Value::Obj(ObjInner(y))) => {
            y.iter().all(|(k, yv)| {
                x.get(k).is_some_and(|xv| value_contains(xv, yv))
            })
        }
        _ => values_equal(a, b),
    }
}

fn rt_startswith(v: &Value, prefix: &Value) -> Result<Value> {
    match (v, prefix) {
        (Value::Str(s), Value::Str(p)) => Ok(Value::from_bool(s.starts_with(p.as_str()))),
        _ => bail!("startswith() requires string inputs"),
    }
}

fn rt_endswith(v: &Value, suffix: &Value) -> Result<Value> {
    match (v, suffix) {
        (Value::Str(s), Value::Str(p)) => Ok(Value::from_bool(s.ends_with(p.as_str()))),
        _ => bail!("endswith() requires string inputs"),
    }
}

fn exec_spawn(input: &Value, cmd: &Value) -> Result<std::process::Output> {
    let cmd_str = match cmd {
        Value::Str(s) => s.as_str().to_string(),
        _ => bail!("exec requires a string command"),
    };
    let stdin_data = match input {
        Value::Null => None,
        Value::Str(s) => Some(s.as_str().to_string()),
        other => Some(crate::value::value_to_json(other)),
    };
    let mut child = std::process::Command::new("sh")
        .args(["-c", &cmd_str])
        .stdin(if stdin_data.is_some() {
            std::process::Stdio::piped()
        } else {
            std::process::Stdio::null()
        })
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("exec: failed to spawn: {}", e))?;
    if let Some(data) = stdin_data {
        use std::io::Write;
        if let Some(ref mut stdin) = child.stdin {
            let _ = stdin.write_all(data.as_bytes());
        }
    }
    child.wait_with_output()
        .map_err(|e| anyhow::anyhow!("exec: failed to wait: {}", e))
}

fn rt_exec(input: &Value, cmd: &Value) -> Result<Value> {
    let output = exec_spawn(input, cmd)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output.status.code().unwrap_or(-1);
        bail!("exec: command exited with code {}: {}", code, stderr.trim_end());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let trimmed = stdout.trim_end_matches('\n');
    Ok(Value::from_str(trimmed))
}

fn rt_execv(input: &Value, cmd: &Value) -> Result<Value> {
    let output = exec_spawn(input, cmd)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let code = output.status.code().unwrap_or(-1);
    let mut obj = new_objmap();
    obj.insert(KeyStr::const_new("exitcode"), Value::number(code as f64));
    obj.insert(KeyStr::const_new("stdout"), Value::from_str(stdout.trim_end_matches('\n')));
    obj.insert(KeyStr::const_new("stderr"), Value::from_str(stderr.trim_end_matches('\n')));
    Ok(Value::object_from_map(obj))
}

fn rt_ltrimstr(v: &Value, prefix: &Value) -> Result<Value> {
    match (v, prefix) {
        (Value::Str(s), Value::Str(p)) => {
            if let Some(rest) = s.strip_prefix(p.as_str()) {
                Ok(Value::from_str(rest))
            } else {
                Ok(v.clone())
            }
        }
        _ => bail!("startswith() requires string inputs"),
    }
}

fn rt_rtrimstr(v: &Value, suffix: &Value) -> Result<Value> {
    match (v, suffix) {
        // rtrimstr("") is defined as `if endswith($s) then .[:-($s|length)] end`.
        // With $s="" that reduces to .[:-0]; jq's slice semantics make -0 end-indexed,
        // so the result is the empty string.
        (Value::Str(_), Value::Str(p)) if p.is_empty() => Ok(Value::from_str("")),
        (Value::Str(s), Value::Str(p)) => {
            if let Some(rest) = s.strip_suffix(p.as_str()) {
                Ok(Value::from_str(rest))
            } else {
                Ok(v.clone())
            }
        }
        _ => bail!("endswith() requires string inputs"),
    }
}

fn rt_split(v: &Value, sep: &Value) -> Result<Value> {
    match (v, sep) {
        (Value::Str(s), Value::Str(p)) => {
            if s.is_empty() {
                return Ok(Value::Arr(Rc::new(Vec::new())));
            }
            if p.is_empty() {
                // split("") = each character as a separate element
                let mut parts = Vec::with_capacity(s.len());
                if s.is_ascii() {
                    // ASCII fast path: &str[i..i+1] avoids char→String allocation
                    for i in 0..s.len() {
                        parts.push(Value::from_str(&s.as_str()[i..i+1]));
                    }
                } else {
                    let mut buf = [0u8; 4];
                    for c in s.chars() {
                        parts.push(Value::from_str(c.encode_utf8(&mut buf)));
                    }
                }
                Ok(Value::Arr(Rc::new(parts)))
            } else {
                let parts: Vec<Value> = s.split(p.as_str())
                    .map(Value::from_str)
                    .collect();
                Ok(Value::Arr(Rc::new(parts)))
            }
        }
        _ => bail!("split input and separator must be strings"),
    }
}

fn rt_regex_split(v: &Value, re: &Value, flags: &Value) -> Result<Value> {
    match (v, re) {
        (Value::Str(s), Value::Str(r)) => {
            let (pat, _global) = apply_regex_flags(r.as_str(), flags)?;
            with_regex(&pat, |regex| {
                // jq enumerates an empty match adjacent to a non-empty
                // one; Rust's `Regex::split` skips it. Walk
                // `jq_match_spans` instead so the slices line up with
                // jq's output for zero-width regexes like `a*`. See #444.
                let s_str = s.as_str();
                let spans = jq_match_spans(regex, s_str);
                let mut parts: Vec<Value> = Vec::with_capacity(spans.len() + 1);
                let mut last = 0;
                for (start, end) in spans {
                    parts.push(Value::from_str(&s_str[last..start]));
                    last = end;
                }
                parts.push(Value::from_str(&s_str[last..]));
                Value::Arr(Rc::new(parts))
            })
        }
        // Two-arg split shares the regex error wording with sub/gsub/scan/match.
        _ => bail!("{} cannot be matched, as it is not a string", errdesc(v)),
    }
}

fn rt_join(v: &Value, sep: &Value) -> Result<Value> {
    // jq treats `null` as the additive identity for strings (`"a" + null == "a"`),
    // and `join` is internally a reduce over `acc + sep + item`, so a `null`
    // separator collapses to the empty string. See #430.
    let sep_bytes: &[u8] = match sep {
        Value::Str(s) => s.as_bytes(),
        Value::Null => b"",
        _ => bail!("join requires array and string"),
    };
    // jq's `join` desugars to a reduce that iterates the input. Both arrays
    // and objects are accepted (#533); for objects we walk the values in
    // insertion order, just like `.[]` does.
    let items: Vec<&Value> = match v {
        Value::Arr(a) => a.iter().collect(),
        Value::Obj(ObjInner(o)) => o.values().collect(),
        // Non-iterable input: jq emits the standard `Cannot iterate over ...`
        // wording adopted across other iteration sites (#481/#505/#517).
        _ => bail!("Cannot iterate over {}", errdesc(v)),
    };
    let cap = items.len() * (8 + sep_bytes.len());
    let mut buf: Vec<u8> = Vec::with_capacity(cap);
    for (i, item) in items.iter().enumerate() {
        if i > 0 { buf.extend_from_slice(sep_bytes); }
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
            _ => {
                // jq errors when trying to add string to object/array
                let partial = Value::from_string(unsafe { String::from_utf8_unchecked(buf) });
                bail!("{} and {} cannot be added", errdesc(&partial), errdesc(item));
            }
        }
    }
    Ok(Value::from_string(unsafe { String::from_utf8_unchecked(buf) }))
}

fn rt_str_index(v: &Value, target: &Value, is_rindex: bool) -> Result<Value> {
    match (v, target) {
        (Value::Str(s), Value::Str(t)) => {
            if t.is_empty() {
                return Ok(Value::Null);
            }
            // ASCII fast path: byte positions == char positions
            if s.is_ascii() && t.is_ascii() {
                if is_rindex {
                    return Ok(match s.rfind(t.as_str()) {
                        Some(pos) => Value::number(pos as f64),
                        None => Value::Null,
                    });
                } else {
                    return Ok(match s.find(t.as_str()) {
                        Some(pos) => Value::number(pos as f64),
                        None => Value::Null,
                    });
                }
            }
            // Unicode: use char indices
            let chars: Vec<char> = s.chars().collect();
            let tchars: Vec<char> = t.chars().collect();
            if is_rindex {
                for i in (0..chars.len()).rev() {
                    if i + tchars.len() <= chars.len() && chars[i..i+tchars.len()] == tchars[..] {
                        return Ok(Value::number(i as f64));
                    }
                }
                Ok(Value::Null)
            } else {
                for i in 0..chars.len() {
                    if i + tchars.len() <= chars.len() && chars[i..i+tchars.len()] == tchars[..] {
                        return Ok(Value::number(i as f64));
                    }
                }
                Ok(Value::Null)
            }
        }
        (Value::Arr(a), _) => {
            if is_rindex {
                for (i, item) in a.iter().enumerate().rev() {
                    if values_equal(item, target) {
                        return Ok(Value::number(i as f64));
                    }
                }
                Ok(Value::Null)
            } else {
                for (i, item) in a.iter().enumerate() {
                    if values_equal(item, target) {
                        return Ok(Value::number(i as f64));
                    }
                }
                Ok(Value::Null)
            }
        }
        // jq's def is `index(x): indices(x) | .[0]` (and `rindex(x): ... | .[-1:][0]`),
        // so non-string/array input falls through to `.[target]`-style indexing.
        _ => {
            let inner = rt_indices_dot_fallback(v, target)?;
            match &inner {
                Value::Null => Ok(Value::Null),
                _ => {
                    // Scalar from object lookup: jq's `.[0]` / `.[-1:][0]` on a
                    // scalar errors with these specific wordings.
                    if is_rindex {
                        bail!("Cannot index {} with object", inner.type_name())
                    } else {
                        bail!("Cannot index {} with number", inner.type_name())
                    }
                }
            }
        }
    }
}

/// Mimic jq's `.[target]` semantics used when indices/index/rindex fall through
/// to builtin.jq's `.[$i]` branch on non-string/array input.
fn rt_indices_dot_fallback(v: &Value, target: &Value) -> Result<Value> {
    match (v, target) {
        (Value::Null, Value::Str(_)) => Ok(Value::Null),
        (Value::Null, Value::Num(_, _)) => Ok(Value::Null),
        (Value::Null, Value::Obj(_)) => Ok(Value::Null),
        (Value::Obj(ObjInner(o)), Value::Str(k)) => Ok(o.get(k.as_str()).cloned().unwrap_or(Value::Null)),
        _ => bail!("Cannot index {} with {}", v.type_name(), index_err_desc(target)),
    }
}

fn rt_indices(v: &Value, target: &Value) -> Result<Value> {
    match (v, target) {
        (Value::Str(s), Value::Str(t)) => {
            let mut indices = Vec::new();
            if t.is_empty() {
                return Ok(Value::Arr(Rc::new(indices)));
            }
            // ASCII fast path: byte positions == char positions
            if s.is_ascii() && t.is_ascii() {
                let sb = s.as_bytes();
                let tb = t.as_bytes();
                let tlen = tb.len();
                if tlen == 1 {
                    // Single-byte search: use memchr for SIMD acceleration
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
                } else if tlen <= sb.len() {
                    for i in 0..=sb.len() - tlen {
                        if &sb[i..i+tlen] == tb {
                            indices.push(Value::number(i as f64));
                        }
                    }
                }
                return Ok(Value::Arr(Rc::new(indices)));
            }
            // Unicode: use character indices
            let chars: Vec<char> = s.chars().collect();
            let tchars: Vec<char> = t.chars().collect();
            for i in 0..chars.len() {
                if i + tchars.len() <= chars.len() && chars[i..i+tchars.len()] == tchars[..] {
                    indices.push(Value::number(i as f64));
                }
            }
            Ok(Value::Arr(Rc::new(indices)))
        }
        (Value::Arr(a), Value::Arr(sub)) => {
            // Subarray search
            let mut indices = Vec::new();
            if sub.is_empty() {
                return Ok(Value::Arr(Rc::new(indices)));
            }
            for i in 0..a.len() {
                if i + sub.len() <= a.len() {
                    let mut matches = true;
                    for j in 0..sub.len() {
                        if !values_equal(&a[i+j], &sub[j]) { matches = false; break; }
                    }
                    if matches {
                        indices.push(Value::number(i as f64));
                    }
                }
            }
            Ok(Value::Arr(Rc::new(indices)))
        }
        (Value::Arr(a), _) => {
            // Element search
            let mut indices = Vec::new();
            for (i, item) in a.iter().enumerate() {
                if values_equal(item, target) {
                    indices.push(Value::number(i as f64));
                }
            }
            Ok(Value::Arr(Rc::new(indices)))
        }
        // jq's `def indices($i)` falls through to `.[$i]` for non-string/array
        // input, so the result here can be a scalar (object value) or null —
        // not an array.
        _ => rt_indices_dot_fallback(v, target),
    }
}

fn rt_pow(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::number(x.powf(*y))),
        _ => bail!("{} number required", errdesc(if !matches!(a, Value::Num(..)) { a } else { b })),
    }
}

pub fn rt_getpath(v: &Value, path: &Value) -> Result<Value> {
    match path {
        Value::Arr(p) => {
            let mut current = v.clone();
            for key in p.iter() {
                match (&current, key) {
                    (Value::Obj(ObjInner(o)), Value::Str(k)) => {
                        current = o.get(k.as_str()).cloned().unwrap_or(Value::Null);
                    }
                    (Value::Arr(a), Value::Num(n, _)) => {
                        let idx = *n as i64;
                        let actual = if idx < 0 { (a.len() as i64 + idx) as usize } else { idx as usize };
                        current = a.get(actual).cloned().unwrap_or(Value::Null);
                    }
                    // Slice path element `{start: N, end: M}`. Without this,
                    // `.[N:M] |= f` would lose the LHS value (#535) — eval's
                    // Update branch calls `rt_getpath` to fetch the current
                    // slice value before applying the update closure, and
                    // bailing here means the closure runs on `null` instead.
                    (Value::Arr(a), Value::Obj(ObjInner(slice_spec)))
                        if slice_spec.contains_key("start") && slice_spec.contains_key("end") =>
                    {
                        let len = a.len() as i64;
                        let (si, ei) = slice_indices(slice_spec, len);
                        current = Value::Arr(Rc::new(a[si..ei].to_vec()));
                    }
                    (Value::Str(s), Value::Obj(ObjInner(slice_spec)))
                        if slice_spec.contains_key("start") && slice_spec.contains_key("end") =>
                    {
                        // String slice indexes by UTF-8 code points, matching
                        // `.[N:M]` semantics on strings.
                        let chars: Vec<char> = s.chars().collect();
                        let len = chars.len() as i64;
                        let (si, ei) = slice_indices(slice_spec, len);
                        let sliced: String = chars[si..ei].iter().collect();
                        current = Value::from_string(sliced);
                    }
                    // jq short-circuits getpath on null for string/number/object keys
                    // (matching `.[k]` on null), but still errors for null/bool/array keys.
                    (Value::Null, Value::Str(_)) | (Value::Null, Value::Num(_, _)) | (Value::Null, Value::Obj(_)) => {
                        current = Value::Null;
                    }
                    // jq errors on type-incompatible path elements (issue #77).
                    (_, _) => bail!(
                        "Cannot index {} with {}",
                        current.type_name(),
                        index_err_desc(key),
                    ),
                }
            }
            Ok(current)
        }
        _ => bail!("Path must be specified as an array"),
    }
}

/// Resolve a `{start, end}` slice spec into normalised `(start, end)` byte/element
/// indices for a sequence of length `len`. Mirrors `rt_setpath`'s slice arm and
/// jq's `.[N:M]` semantics: missing/null endpoints open to `(0, len)`, negative
/// indices count from the end, and the result is clamped to `[0, len]`.
fn slice_indices(slice_spec: &crate::value::ObjMap, len: i64) -> (usize, usize) {
    let start = match slice_spec.get("start") {
        Some(Value::Num(n, _)) => n.floor() as i64,
        _ => 0,
    };
    let end = match slice_spec.get("end") {
        Some(Value::Num(n, _)) => n.ceil() as i64,
        Some(Value::Null) | None => len,
        _ => 0,
    };
    let si_raw = if start < 0 { (len + start).max(0) } else { start.min(len) };
    let ei_raw = if end < 0 { (len + end).max(0) } else { end.min(len) };
    let si = si_raw as usize;
    let ei = (ei_raw as usize).max(si);
    (si, ei)
}

/// Match jq 1.8.1's "Cannot index X with Y" wording for the Y side.
/// Numbers omit the value; strings keep the quoted content; others show the type name.
fn index_err_desc(key: &Value) -> String {
    match key {
        Value::Str(s) => format!("string \"{}\"", s),
        Value::Num(_, _) => "number".to_string(),
        Value::Null => "null".to_string(),
        Value::True | Value::False => "boolean".to_string(),
        Value::Arr(_) => "array".to_string(),
        Value::Obj(_) => "object".to_string(),
        Value::Error(_) => "error".to_string(),
    }
}

pub fn rt_setpath(v: &Value, path: &Value, val: &Value) -> Result<Value> {
    match path {
        Value::Arr(p) if p.is_empty() => Ok(val.clone()),
        Value::Arr(p) => {
            let key = &p[0];
            let rest = Value::Arr(Rc::new(p[1..].to_vec()));
            match (v, key) {
                (Value::Obj(ObjInner(o)), Value::Str(k)) => {
                    let inner = o.get(k.as_str()).cloned().unwrap_or(Value::Null);
                    let new_inner = rt_setpath(&inner, &rest, val)?;
                    let mut new_obj = (**o).clone();
                    new_obj.insert(KeyStr::from(k.as_str()), new_inner);
                    Ok(Value::object_from_map(new_obj))
                }
                (Value::Arr(a), Value::Num(n, _)) => {
                    if n.is_nan() { bail!("Cannot set array element at NaN index"); }
                    let idx = *n as i64;
                    let actual = if idx < 0 {
                        // Out-of-range negative indices must error — clamping silently
                        // rewrote the first element, see issue #77.
                        let adj = a.len() as i64 + idx;
                        if adj < 0 { bail!("Out of bounds negative array index"); }
                        adj as usize
                    } else { idx as usize };
                    if actual > 536870911 { bail!("Array index too large"); }
                    let inner = a.get(actual).cloned().unwrap_or(Value::Null);
                    let new_inner = rt_setpath(&inner, &rest, val)?;
                    let mut new_arr = (**a).clone();
                    while new_arr.len() <= actual {
                        new_arr.push(Value::Null);
                    }
                    new_arr[actual] = new_inner;
                    Ok(Value::Arr(Rc::new(new_arr)))
                }
                (Value::Null, Value::Str(k)) => {
                    let new_inner = rt_setpath(&Value::Null, &rest, val)?;
                    let mut obj = new_objmap();
                    obj.insert(KeyStr::from(k.as_str()), new_inner);
                    Ok(Value::object_from_map(obj))
                }
                (Value::Null, Value::Num(n, _)) => {
                    if n.is_nan() { bail!("Cannot set array element at NaN index"); }
                    if *n < 0.0 { bail!("Out of bounds negative array index"); }
                    let idx = *n as usize;
                    if idx > 536870911 { bail!("Array index too large"); }
                    let new_inner = rt_setpath(&Value::Null, &rest, val)?;
                    let mut arr = vec![Value::Null; idx + 1];
                    arr[idx] = new_inner;
                    Ok(Value::Arr(Rc::new(arr)))
                }
                (Value::Obj(_), Value::Num(_, _)) => {
                    // jq: number keys omit the value (#440).
                    bail!("Cannot index object with number");
                }
                (Value::Arr(_), Value::Str(k)) => {
                    // jq: string keys are quoted without parens (#440).
                    bail!("Cannot index array with string \"{}\"", k);
                }
                // Slice assignment: path element is {start: N, end: N}
                (_, Value::Obj(ObjInner(slice_spec))) if slice_spec.contains_key("start") && slice_spec.contains_key("end") => {
                    // jq applies floor(start) / ceil(end) when consuming a slice path,
                    // and treats null as the open endpoint (0 for start, len for end).
                    let len = match v { Value::Arr(a) => a.len() as i64, _ => 0 };
                    let start = match slice_spec.get("start") {
                        Some(Value::Num(n, _)) => n.floor() as i64,
                        _ => 0,
                    };
                    let end = match slice_spec.get("end") {
                        Some(Value::Num(n, _)) => n.ceil() as i64,
                        Some(Value::Null) | None => len,
                        _ => 0,
                    };
                    match v {
                        Value::Arr(a) => {
                            let len = a.len() as i64;
                            let si_raw = if start < 0 { (len + start).max(0) } else { start.min(len) };
                            let ei_raw = if end < 0 { (len + end).max(0) } else { end.min(len) };
                            let si = si_raw as usize;
                            let ei = (ei_raw as usize).max(si);
                            let new_val = rt_setpath(&Value::Null, &rest, val)?;
                            let replacement = match &new_val {
                                Value::Arr(r) => r.as_ref().clone(),
                                _ => bail!("A slice of an array can only be assigned another array"),
                            };
                            let mut result = a[..si].to_vec();
                            result.extend(replacement);
                            result.extend_from_slice(&a[ei..]);
                            Ok(Value::Arr(Rc::new(result)))
                        }
                        Value::Str(_) => bail!("Cannot update string slices"),
                        Value::Null => {
                            let new_val = rt_setpath(&Value::Null, &rest, val)?;
                            let replacement = match &new_val {
                                Value::Arr(r) => r.as_ref().clone(),
                                _ => bail!("A slice of an array can only be assigned another array"),
                            };
                            Ok(Value::Arr(Rc::new(replacement)))
                        }
                        _ => bail!("Cannot set path"),
                    }
                }
                // jq emits a special message when both the container and the
                // key are arrays: `Cannot update field at array index of array`.
                // All other type-mismatch combinations use the same wording as
                // the read side (`rt_getpath`, `.[k]`).
                (Value::Arr(_), Value::Arr(_)) => {
                    bail!("Cannot update field at array index of array");
                }
                (_, _) => bail!(
                    "Cannot index {} with {}",
                    v.type_name(),
                    index_err_desc(key),
                ),
            }
        }
        _ => bail!("Path must be specified as an array"),
    }
}

/// In-place setpath: modifies `v` directly when Rc refcount allows.
/// For use in reduce contexts where TakeVar ensures exclusive ownership.
pub fn rt_setpath_mut(v: &mut Value, path: &[Value], val: Value) -> Result<()> {
    if path.is_empty() {
        *v = val;
        return Ok(());
    }
    let key = &path[0];
    let rest = &path[1..];
    match key {
        Value::Str(k) => {
            // Ensure v is an Obj (or convert Null to Obj)
            if matches!(v, Value::Null) {
                *v = Value::object_from_map(new_objmap());
            }
            if let Value::Obj(ObjInner(o)) = v {
                let obj = Rc::make_mut(o);
                if rest.is_empty() {
                    obj.insert(KeyStr::from(k.as_str()), val);
                } else {
                    let inner = obj.get_mut(k.as_str());
                    if let Some(inner_val) = inner {
                        rt_setpath_mut(inner_val, rest, val)?;
                    } else {
                        let mut inner_val = Value::Null;
                        rt_setpath_mut(&mut inner_val, rest, val)?;
                        obj.insert(KeyStr::from(k.as_str()), inner_val);
                    }
                }
                Ok(())
            } else {
                bail!("Cannot index {} with string \"{}\"", v.type_name(), k);
            }
        }
        Value::Num(n, _) => {
            if n.is_nan() { bail!("Cannot set array element at NaN index"); }
            let idx = *n as i64;
            // Ensure v is an Arr (or convert Null to Arr)
            if matches!(v, Value::Null) {
                if idx < 0 { bail!("Out of bounds negative array index"); }
                let uidx = idx as usize;
                if uidx > 536870911 { bail!("Array index too large"); }
                *v = Value::Arr(Rc::new(vec![Value::Null; uidx + 1]));
            }
            if let Value::Arr(a) = v {
                let arr = Rc::make_mut(a);
                let actual = if idx < 0 {
                    // Out-of-range negative indices error — the previous `.max(0)`
                    // clamp silently rewrote the first element (issue #77).
                    let adj = arr.len() as i64 + idx;
                    if adj < 0 { bail!("Out of bounds negative array index"); }
                    adj as usize
                } else { idx as usize };
                if actual > 536870911 { bail!("Array index too large"); }
                while arr.len() <= actual {
                    arr.push(Value::Null);
                }
                if rest.is_empty() {
                    arr[actual] = val;
                } else {
                    rt_setpath_mut(&mut arr[actual], rest, val)?;
                }
                Ok(())
            } else {
                // jq: number keys omit the value (#440).
                let _ = n;
                bail!("Cannot index {} with number", v.type_name());
            }
        }
        // jq's special-case wording for array-key on array-container; the
        // other combinations share the read-side `Cannot index <container>
        // with <key_type>` form.
        Value::Arr(_) if matches!(v, Value::Arr(_)) => {
            bail!("Cannot update field at array index of array");
        }
        _ => bail!(
            "Cannot index {} with {}",
            v.type_name(),
            index_err_desc(key),
        ),
    }
}

pub fn rt_delpaths(v: &Value, paths: &Value) -> Result<Value> {
    match paths {
        Value::Arr(ps) => {
            // Sort ascending so any path that is a strict prefix of another
            // appears first; that lets us drop the longer path as subsumed.
            // jq's `delpaths_sorted` (jv_aux.c) achieves the same by grouping
            // paths sharing a first key and skipping descent when one path in
            // the group is a "delete this whole key" path. Without this,
            // `delpaths([["a"],["a","b"]]) on {"a":1,...}` errors on the
            // second path (descending into `.a` = number); jq returns the
            // object minus `.a`. See #432.
            let mut sorted_paths: Vec<&Value> = ps.iter().collect();
            sorted_paths.sort_by(|a, b| compare_values(a, b));
            let mut filtered: Vec<&Value> = Vec::with_capacity(sorted_paths.len());
            for path in sorted_paths {
                let p_arr = match path {
                    Value::Arr(p) => p,
                    _ => return Err(anyhow::anyhow!(
                        "Path must be specified as array, not {}",
                        path.type_name()
                    )),
                };
                let subsumed = filtered.iter().any(|prev| {
                    let prev_arr = match prev {
                        Value::Arr(q) => q,
                        _ => return false,
                    };
                    prev_arr.len() <= p_arr.len()
                        && prev_arr.iter().zip(p_arr.iter()).all(|(a, b)| a == b)
                });
                if !subsumed {
                    filtered.push(path);
                }
            }
            // Iterate filtered paths in reverse to preserve array index
            // stability when multiple sibling indices are deleted.
            let mut result = v.clone();
            for path in filtered.into_iter().rev() {
                result = delete_path(&result, path)?;
            }
            Ok(result)
        }
        _ => bail!("Paths must be specified as an array"),
    }
}

fn delete_path(v: &Value, path: &Value) -> Result<Value> {
    match path {
        Value::Arr(p) if p.is_empty() => Ok(Value::Null),
        Value::Arr(p) if p.len() == 1 => {
            match (v, &p[0]) {
                (Value::Null, _) => Ok(Value::Null),
                (Value::Obj(ObjInner(o)), Value::Str(k)) => {
                    let mut new_obj = (**o).clone();
                    new_obj.shift_remove(k.as_str());
                    Ok(Value::object_from_map(new_obj))
                }
                (Value::Arr(a), Value::Num(n, _)) => {
                    let ni = *n as i64;
                    let idx = if ni < 0 { a.len() as i64 + ni } else { ni };
                    if idx >= 0 && (idx as usize) < a.len() {
                        let mut new_arr = (**a).clone();
                        new_arr.remove(idx as usize);
                        Ok(Value::Arr(Rc::new(new_arr)))
                    } else {
                        Ok(v.clone())
                    }
                }
                // Match jq 1.8.1's error wording for type-incompatible paths (issue #77).
                (Value::Obj(_), key) => bail!("Cannot delete {} field of object", index_err_desc(key)),
                (Value::Arr(_), key) => bail!("Cannot delete {} element of array", index_err_desc(key)),
                (other, _) => bail!("Cannot delete fields from {}", other.type_name()),
            }
        }
        Value::Arr(p) => {
            let key = &p[0];
            let rest = Value::Arr(Rc::new(p[1..].to_vec()));
            match (v, key) {
                (Value::Null, _) => Ok(Value::Null),
                (Value::Obj(ObjInner(o)), Value::Str(k)) => {
                    if let Some(inner) = o.get(k.as_str()) {
                        let new_inner = delete_path(inner, &rest)?;
                        let mut new_obj = (**o).clone();
                        new_obj.insert(KeyStr::from(k.as_str()), new_inner);
                        Ok(Value::object_from_map(new_obj))
                    } else {
                        Ok(v.clone())
                    }
                }
                (Value::Arr(a), Value::Num(n, _)) => {
                    let ni = *n as i64;
                    let idx = if ni < 0 { a.len() as i64 + ni } else { ni };
                    if idx >= 0 && (idx as usize) < a.len() {
                        let uidx = idx as usize;
                        let inner = &a[uidx];
                        let new_inner = delete_path(inner, &rest)?;
                        let mut new_arr = (**a).clone();
                        new_arr[uidx] = new_inner;
                        Ok(Value::Arr(Rc::new(new_arr)))
                    } else {
                        Ok(v.clone())
                    }
                }
                (Value::Obj(_), key) => bail!("Cannot delete {} field of object", index_err_desc(key)),
                (Value::Arr(_), key) => bail!("Cannot delete {} element of array", index_err_desc(key)),
                (other, _) => bail!("Cannot delete fields from {}", other.type_name()),
            }
        }
        _ => Ok(v.clone()),
    }
}

// -----------------------------------------------------------------------
// Regex (with compile cache — jq-jit is single-threaded)
// -----------------------------------------------------------------------

/// Cached regex: stores the last compiled pattern to avoid re-compilation.
/// jq filters typically use a single fixed regex pattern, so a 1-entry cache
/// hits >99% of the time. For the rare case of multiple patterns, we fall
/// back to a small LRU of 8 entries.
struct RegexCache {
    entries: Vec<(String, regex::Regex)>,
}

const REGEX_CACHE_SIZE: usize = 8;

impl RegexCache {
    fn new() -> Self {
        Self { entries: Vec::with_capacity(REGEX_CACHE_SIZE) }
    }

    fn get(&mut self, pattern: &str) -> Result<&regex::Regex> {
        // Check if already cached (most recent first for common case)
        if let Some(pos) = self.entries.iter().position(|(p, _)| p == pattern) {
            // Move to front (MRU)
            if pos > 0 {
                let entry = self.entries.remove(pos);
                self.entries.insert(0, entry);
            }
            return Ok(&self.entries[0].1);
        }
        // Compile and cache
        let re = regex::Regex::new(pattern)
            .map_err(|e| anyhow::anyhow!("Invalid regex: {}", e))?;
        if self.entries.len() >= REGEX_CACHE_SIZE {
            self.entries.pop();
        }
        self.entries.insert(0, (pattern.to_string(), re));
        Ok(&self.entries[0].1)
    }
}

// Per-thread regex cache. Each `cargo test` worker keeps its own cache so
// parallel test runs cannot trample the LRU; the previous global cache was
// the kind of thing `--test-threads=1` was hiding.
thread_local! {
    static REGEX_CACHE: std::cell::RefCell<Option<RegexCache>> = const { std::cell::RefCell::new(None) };
}

fn with_regex<R>(pattern: &str, f: impl FnOnce(&regex::Regex) -> R) -> Result<R> {
    REGEX_CACHE.with_borrow_mut(|cache| {
        let cache = cache.get_or_insert_with(RegexCache::new);
        let re = cache.get(pattern)?;
        Ok(f(re))
    })
}

/// Iterate matches with jq/Oniguruma-style global semantics: after a
/// non-zero-width match ending at `e`, attempt a zero-width match at `e`
/// before advancing past `e`. The Rust `regex` crate's default `find_iter`
/// suppresses that zero-width attempt to prevent infinite loops, which makes
/// `match("a*"; "g")` on `"abc"` emit one fewer match than jq (issue #158).
///
/// The walker yields `(start, end)` pairs and re-runs `find_at` on demand —
/// callers map the spans back to `regex::Match` / `regex::Captures` as
/// needed (cheap because the spans are already validated).
fn jq_match_spans(regex: &regex::Regex, s: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut pos: usize = 0;
    let mut just_emitted_empty_at_end: Option<usize> = None;
    while pos <= s.len() {
        let m = match regex.find_at(s, pos) {
            Some(m) => m,
            None => break,
        };
        let (start, end) = (m.start(), m.end());
        let is_empty = start == end;
        if is_empty && just_emitted_empty_at_end == Some(end) {
            // Already yielded a zero-width match at this position; advance one
            // codepoint to avoid an infinite loop.
            let next = next_char_boundary(s, pos);
            if next == pos { break; }
            pos = next;
            just_emitted_empty_at_end = None;
            continue;
        }
        spans.push((start, end));
        if is_empty {
            just_emitted_empty_at_end = Some(end);
            pos = end;
        } else {
            just_emitted_empty_at_end = None;
            pos = end;
        }
    }
    spans
}

fn next_char_boundary(s: &str, pos: usize) -> usize {
    if pos >= s.len() { return s.len() + 1; }
    let bytes = s.as_bytes();
    // Advance one codepoint by skipping bytes whose top two bits are `10`.
    let mut p = pos + 1;
    while p < s.len() && (bytes[p] & 0xC0) == 0x80 {
        p += 1;
    }
    p
}

/// Parse jq regex flags string and return (pattern_with_inline_flags, is_global).
/// Supported flags: i (case-insensitive), x (extended), s (dotall/single-line), g (global).
/// Coerce a regex pattern arg for builtins that accept either a plain
/// string OR a `[pattern, flags?]` array (test, match, capture). Mirrors
/// jq's wording for invalid types (#461). The `allow_array` flag is true
/// only for the 1-arg form (test(re)) — the 2-arg form (test(re; flags))
/// rejects array patterns, since jq internally wraps the args into
/// `[re, flags]` before calling _match_impl.
fn coerce_regex_pat_or_array(re: &Value, allow_array: bool) -> Result<(String, String)> {
    match re {
        Value::Str(s) => Ok((s.as_str().to_string(), String::new())),
        Value::Arr(a) if allow_array => {
            let pat = match a.first() {
                Some(Value::Str(s)) => s.as_str().to_string(),
                Some(other) => bail!("{} is not a string", errdesc(other)),
                None => bail!("array not a string or array"),
            };
            let flags = match a.get(1) {
                Some(Value::Str(s)) => s.as_str().to_string(),
                Some(Value::Null) | None => String::new(),
                Some(other) => bail!("{} is not a string", errdesc(other)),
            };
            Ok((pat, flags))
        }
        Value::Arr(_) => bail!("{} is not a string", errdesc(re)),
        _ => bail!("{} not a string or array", re.type_name()),
    }
}

/// Coerce a regex pattern arg for builtins that require a plain string
/// (scan, sub, gsub, splits). jq emits errdesc-style wording (#461).
fn coerce_regex_pat_str(re: &Value) -> Result<String> {
    match re {
        Value::Str(s) => Ok(s.as_str().to_string()),
        _ => bail!("{} is not a string", errdesc(re)),
    }
}

fn apply_regex_flags(pattern: &str, flags: &Value) -> Result<(String, bool)> {
    let flags_str = match flags {
        Value::Str(s) => s.as_str(),
        Value::Null => return Ok((pattern.to_string(), false)),
        // jq rejects non-string non-null flags with the standard errdesc
        // wording (#461).
        _ => bail!("{} is not a string", errdesc(flags)),
    };
    // jq accepts g, i, l, m, n, p, s, x. Any other character — even mixed in
    // with valid ones — rejects the whole modifier string.
    if !flags_str.chars().all(|c| matches!(c, 'g' | 'i' | 'l' | 'm' | 'n' | 'p' | 's' | 'x')) {
        bail!("{} is not a valid modifier string", flags_str);
    }
    let mut inline = String::new();
    let mut global = false;
    for ch in flags_str.chars() {
        match ch {
            'i' | 'x' | 's' => inline.push(ch),
            'g' => global = true,
            // l, m, n, p have no inline equivalent in onig — accepted but ignored.
            _ => {}
        }
    }
    let pat = if inline.is_empty() {
        pattern.to_string()
    } else {
        format!("(?{}){}", inline, pattern)
    };
    Ok((pat, global))
}

fn rt_test(v: &Value, re: &Value) -> Result<Value> {
    match (v, re) {
        (Value::Str(s), Value::Str(r)) => {
            let matched = with_regex(r, |regex| regex.is_match(s))?;
            Ok(Value::from_bool(matched))
        }
        _ => bail!("test requires string and regex"),
    }
}

fn rt_match(v: &Value, re: &Value) -> Result<Value> {
    match (v, re) {
        (Value::Str(s), Value::Str(r)) => {
            with_regex(r, |regex| {
                let capture_names: Vec<Option<&str>> = regex.capture_names().skip(1).collect();
                let num_groups = regex.captures_len();
                if num_groups <= 1 {
                    match regex.find(s) {
                        Some(m) => Ok(build_match_obj(&m, None, &capture_names, s)),
                        None => bail!("match failed"),
                    }
                } else {
                    match regex.captures(s) {
                        Some(caps) => {
                            let m = caps.get(0).unwrap();
                            Ok(build_match_obj(&m, Some(&caps), &capture_names, s))
                        }
                        None => bail!("match failed"),
                    }
                }
            })?
        }
        _ => bail!("match requires string and regex"),
    }
}

fn build_match_obj(m: &regex::Match, caps: Option<&regex::Captures>, capture_names: &[Option<&str>], s: &str) -> Value {
    let byte_offset = m.start();
    let char_offset = s[..byte_offset].chars().count();
    let char_length = m.as_str().chars().count();
    let mut result = new_objmap();
    result.insert("offset".into(), Value::number(char_offset as f64));
    result.insert("length".into(), Value::number(char_length as f64));
    result.insert("string".into(), Value::from_str(m.as_str()));
    let mut captures_vec = Vec::new();
    if let Some(caps) = caps {
        for (i, name_opt) in capture_names.iter().enumerate() {
            let name_val = match name_opt {
                Some(name) => Value::from_str(name),
                None => Value::Null,
            };
            if let Some(cap) = caps.get(i + 1) {
                let cap_byte_offset = cap.start();
                let cap_char_offset = s[..cap_byte_offset].chars().count();
                let cap_char_length = cap.as_str().chars().count();
                let mut c = new_objmap();
                c.insert("offset".into(), Value::number(cap_char_offset as f64));
                c.insert("length".into(), Value::number(cap_char_length as f64));
                c.insert("string".into(), Value::from_str(cap.as_str()));
                c.insert("name".into(), name_val);
                captures_vec.push(Value::object_from_map(c));
            } else {
                let mut c = new_objmap();
                c.insert("offset".into(), Value::number(-1.0));
                c.insert("length".into(), Value::number(0.0));
                c.insert("string".into(), Value::Null);
                c.insert("name".into(), name_val);
                captures_vec.push(Value::object_from_map(c));
            }
        }
    }
    result.insert("captures".into(), Value::Arr(Rc::new(captures_vec)));
    Value::object_from_map(result)
}

fn rt_match_global(v: &Value, re: &Value) -> Result<Value> {
    match (v, re) {
        (Value::Str(s), Value::Str(r)) => {
            with_regex(r, |regex| {
                let capture_names: Vec<Option<&str>> = regex.capture_names().skip(1).collect();
                let num_groups = regex.captures_len();
                let mut results = Vec::new();
                let spans = jq_match_spans(regex, s);
                if num_groups <= 1 {
                    for (start, _) in &spans {
                        if let Some(m) = regex.find_at(s, *start) {
                            if m.start() == *start {
                                results.push(build_match_obj(&m, None, &capture_names, s));
                            }
                        }
                    }
                } else {
                    for (start, _) in &spans {
                        if let Some(caps) = regex.captures_at(s, *start) {
                            if let Some(m) = caps.get(0) {
                                if m.start() == *start {
                                    results.push(build_match_obj(&m, Some(&caps), &capture_names, s));
                                }
                            }
                        }
                    }
                }
                if results.is_empty() {
                    bail!("match failed");
                }
                Ok(Value::Arr(Rc::new(results)))
            })?
        }
        _ => bail!("match requires string and regex"),
    }
}

fn rt_capture(v: &Value, re: &Value) -> Result<Value> {
    match (v, re) {
        (Value::Str(s), Value::Str(r)) => {
            with_regex(r, |regex| {
                match regex.captures(s) {
                    Some(caps) => {
                        let mut result = new_objmap();
                        for name in regex.capture_names().flatten() {
                            if let Some(m) = caps.name(name) {
                                result.insert(KeyStr::from(name), Value::from_str(m.as_str()));
                            } else {
                                result.insert(KeyStr::from(name), Value::Null);
                            }
                        }
                        Ok(Value::object_from_map(result))
                    }
                    None => bail!("capture failed"),
                }
            })?
        }
        _ => bail!("capture requires string and regex"),
    }
}

pub fn rt_capture_global(v: &Value, re: &Value) -> Result<Value> {
    match (v, re) {
        (Value::Str(s), Value::Str(r)) => {
            with_regex(r, |regex| {
                let mut results: Vec<Value> = Vec::new();
                for (start, _) in jq_match_spans(regex, s) {
                    let caps = match regex.captures_at(s, start) {
                        Some(c) if c.get(0).map(|m| m.start()) == Some(start) => c,
                        _ => continue,
                    };
                    let mut obj = new_objmap();
                    for name in regex.capture_names().flatten() {
                        if let Some(m) = caps.name(name) {
                            obj.insert(KeyStr::from(name), Value::from_str(m.as_str()));
                        } else {
                            obj.insert(KeyStr::from(name), Value::Null);
                        }
                    }
                    results.push(Value::object_from_map(obj));
                }
                Ok(Value::Arr(Rc::new(results)))
            })?
        }
        _ => bail!("capture requires string and regex"),
    }
}

fn rt_scan(v: &Value, re: &Value) -> Result<Value> {
    match (v, re) {
        (Value::Str(s), Value::Str(r)) => {
            with_regex(r, |regex| {
                let has_captures = regex.captures_len() > 1;
                let spans = jq_match_spans(regex, s);
                let results: Vec<Value> = if has_captures {
                    spans.iter()
                        .filter_map(|(start, _)| regex.captures_at(s, *start)
                            .filter(|c| c.get(0).map(|m| m.start()) == Some(*start)))
                        .map(|caps| {
                            let groups: Vec<Value> = (1..caps.len())
                                .map(|i| match caps.get(i) {
                                    Some(m) => Value::from_str(m.as_str()),
                                    None => Value::from_str(""),
                                })
                                .collect();
                            Value::Arr(Rc::new(groups))
                        })
                        .collect()
                } else {
                    spans.iter()
                        .map(|(start, end)| Value::from_str(&s[*start..*end]))
                        .collect()
                };
                Value::Arr(Rc::new(results))
            }).map(Ok)?
        }
        _ => bail!("scan requires string and regex"),
    }
}

fn rt_sub_gsub(v: &Value, re: &Value, replacement: &Value, global: bool) -> Result<Value> {
    match (v, re, replacement) {
        (Value::Str(s), Value::Str(r), Value::Str(rep)) => {
            let result = with_regex(r, |regex| {
                if global {
                    regex.replace_all(s, rep.as_str()).to_string()
                } else {
                    regex.replace(s, rep.as_str()).to_string()
                }
            })?;
            Ok(Value::from_string(result))
        }
        _ => bail!("sub/gsub requires string, regex, and replacement"),
    }
}

/// A match segment for sub/gsub with capture-aware replacement.
/// Each segment is either a literal string (non-matched part) or a capture object.
pub struct SubGsubSegment {
    /// The literal text before this match (or trailing text for the last segment).
    pub literal: String,
    /// If Some, this is a capture object (named captures as keys → matched strings as values).
    pub captures: Option<Value>,
}

/// Find regex matches and return segments for capture-aware sub/gsub replacement.
/// Returns segments alternating: literal, capture_obj, literal, capture_obj, ..., literal.
pub fn sub_gsub_segments(input: &str, pattern: &str, flags: &Value, global: bool) -> Result<Vec<SubGsubSegment>> {
    let (pat, flag_global) = apply_regex_flags(pattern, flags)?;
    let is_global = global || flag_global;
    with_regex(&pat, |regex| {
        let mut segments = Vec::new();
        let mut last_end = 0;

        let mut process_match = |m: regex::Match, caps: Option<&regex::Captures>| {
            // Add literal text before this match
            let literal = input[last_end..m.start()].to_string();
            // Build capture object
            let mut obj = new_objmap();
            if let Some(caps) = caps {
                for name in regex.capture_names().flatten() {
                    if let Some(cm) = caps.name(name) {
                        obj.insert(KeyStr::from(name), Value::from_str(cm.as_str()));
                    } else {
                        obj.insert(KeyStr::from(name), Value::Null);
                    }
                }
            }
            segments.push(SubGsubSegment {
                literal,
                captures: Some(Value::object_from_map(obj)),
            });
            last_end = m.end();
        };

        let has_captures = regex.captures_len() > 1;
        if is_global {
            // jq enumerates an empty match adjacent to a non-empty one for
            // zero-width regexes (`a*`, etc.). `captures_iter` /
            // `find_iter` skip those. Walk `jq_match_spans` so the
            // segment list matches jq's output. See #444.
            let spans = jq_match_spans(regex, input);
            if has_captures {
                for (start, _end) in spans {
                    if let Some(caps) = regex.captures_at(input, start) {
                        if let Some(m) = caps.get(0) {
                            if m.start() == start {
                                process_match(m, Some(&caps));
                            }
                        }
                    }
                }
            } else {
                for (start, _end) in spans {
                    if let Some(m) = regex.find_at(input, start) {
                        if m.start() == start {
                            process_match(m, None);
                        }
                    }
                }
            }
        } else if has_captures {
            if let Some(caps) = regex.captures(input) {
                let m = caps.get(0).unwrap();
                process_match(m, Some(&caps));
            }
        } else if let Some(m) = regex.find(input) {
            process_match(m, None);
        }

        // Trailing literal
        segments.push(SubGsubSegment {
            literal: input[last_end..].to_string(),
            captures: None,
        });
        segments
    })
}

// -----------------------------------------------------------------------
// Date/Time
// -----------------------------------------------------------------------

fn rt_gmtime(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n, _) => {
            // jq preserves fractional seconds in tm_sec: trunc toward zero
            // for the libc call, then add |n - trunc| (always non-negative)
            // back. `(-1.5) | gmtime | .[5]` is `59.5`, not `58.5`. See #436.
            let secs = n.trunc() as i64;
            let frac = (n - n.trunc()).abs();
            Ok(libc_gmtime(secs, frac))
        }
        _ => bail!("gmtime() requires numeric inputs"),
    }
}

fn libc_gmtime(secs: i64, frac_sec: f64) -> Value {
    use libc::{gmtime_r, time_t, tm};
    let t = secs as time_t;
    let mut result: tm = unsafe { std::mem::zeroed() };
    unsafe { gmtime_r(&t, &mut result) };
    // jq format: [year+1900, month(0-based), mday, hour, min, sec, wday, yday]
    // Wait - jq actually uses [tm_year+1900, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday]
    // But test expects [2015,2,5,23,51,47,4,63] for epoch 1425599507
    // That matches: year=2015, mon=2(march, 0-indexed), mday=5, hour=23, min=51, sec=47, wday=4(thu), yday=63
    Value::Arr(Rc::new(vec![
        Value::number((result.tm_year + 1900) as f64),
        Value::number(result.tm_mon as f64),
        Value::number(result.tm_mday as f64),
        Value::number(result.tm_hour as f64),
        Value::number(result.tm_min as f64),
        Value::number(result.tm_sec as f64 + frac_sec),
        Value::number(result.tm_wday as f64),
        Value::number(result.tm_yday as f64),
    ]))
}

fn rt_localtime(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n, _) => {
            // Same fractional-seconds preservation as gmtime (#436).
            let secs = n.trunc() as i64;
            let frac = (n - n.trunc()).abs();
            use libc::{localtime_r, time_t, tm};
            let t = secs as time_t;
            let mut result: tm = unsafe { std::mem::zeroed() };
            unsafe { localtime_r(&t, &mut result) };
            Ok(Value::Arr(Rc::new(vec![
                Value::number((result.tm_year + 1900) as f64),
                Value::number(result.tm_mon as f64),
                Value::number(result.tm_mday as f64),
                Value::number(result.tm_hour as f64),
                Value::number(result.tm_min as f64),
                Value::number(result.tm_sec as f64 + frac),
                Value::number(result.tm_wday as f64),
                Value::number(result.tm_yday as f64),
            ])))
        }
        _ => bail!("localtime() requires numeric inputs"),
    }
}

fn time_arr_to_tm(a: &[Value]) -> Result<libc::tm> {
    let get = |i: usize| -> f64 { a.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0) };
    // Validate that first element is a number
    if !a.is_empty() {
        if let Value::Str(_) = &a[0] {
            bail!("mktime requires parsed datetime inputs");
        }
    }
    let mut t: libc::tm = unsafe { std::mem::zeroed() };
    // jq's broken-down-time arrays hold the literal year (e.g. 2023) at
    // index 0, which is offset by 1900 before handing to libc. When the
    // year field is absent we keep tm_year at 0 so `strftime("%Y")` renders
    // as 1900 (libc's "0 years since 1900"), matching jq 1.8.1's default
    // tm_year fill for `[] | strftime(...)`.
    if !a.is_empty() {
        t.tm_year = get(0) as i32 - 1900;
    }
    t.tm_mon = get(1) as i32;
    // jq leaves missing mday at 0 (which strftime renders as "00" and
    // timegm normalises to the previous day); don't default to 1 here.
    // See `[2024] | strftime("%Y-%m-%d")` → `"2023-12-31"` in jq 1.8.1.
    t.tm_mday = get(2) as i32;
    t.tm_hour = get(3) as i32;
    t.tm_min = get(4) as i32;
    t.tm_sec = get(5) as i32;
    t.tm_wday = get(6) as i32;
    t.tm_yday = get(7) as i32;
    Ok(t)
}

fn rt_mktime(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) if a.len() >= 2 => {
            let mut t = time_arr_to_tm(a)?;
            // Use timegm for UTC
            let result = unsafe { libc::timegm(&mut t) };
            Ok(Value::number(result as f64))
        }
        Value::Arr(a) if a.is_empty() => {
            // jq's mktime on `[]` falls through to the broken-down-time
            // validator and bails with "invalid gmtime representation"
            // (#525) — the array-vs-not check only catches non-arrays.
            bail!("invalid gmtime representation");
        }
        Value::Arr(a) => {
            // Single-element array: jq still calls into the broken-down-
            // time validator, which trips on the missing fields.
            if let Value::Str(_) = &a[0] {
                bail!("mktime requires parsed datetime inputs");
            }
            bail!("mktime requires array of time components");
        }
        _ => bail!("mktime requires array inputs"),
    }
}

fn rt_strftime(v: &Value, fmt: &Value) -> Result<Value> {
    let fmt_str = match fmt {
        Value::Str(f) => f.as_ref(),
        _ => bail!("strftime/1 requires a string format"),
    };
    match v {
        Value::Arr(a) => {
            if !a.is_empty() {
                if let Value::Str(_) = &a[0] {
                    bail!("strftime/1 requires parsed datetime inputs");
                }
            }
            let t = time_arr_to_tm(a)?;
            Ok(Value::from_str(&format_tm(&t, fmt_str)))
        }
        Value::Num(n, _) => {
            // Convert epoch to gmtime first, then format
            let secs = *n as i64;
            use libc::{gmtime_r, time_t, tm};
            let t_val = secs as time_t;
            let mut t: tm = unsafe { std::mem::zeroed() };
            unsafe { gmtime_r(&t_val, &mut t) };
            Ok(Value::from_str(&format_tm(&t, fmt_str)))
        }
        _ => bail!("strftime/1 requires parsed datetime inputs"),
    }
}

fn format_tm(t: &libc::tm, fmt: &str) -> String {
    use std::ffi::CString;
    let fmt_c = CString::new(fmt).unwrap_or_default();
    let mut buf = vec![0u8; 256];
    let len = unsafe {
        libc::strftime(buf.as_mut_ptr() as *mut libc::c_char, buf.len(), fmt_c.as_ptr(), t)
    };
    if len == 0 { String::new() }
    else { String::from_utf8_lossy(&buf[..len]).into_owned() }
}

fn rt_strptime(v: &Value, fmt: &Value) -> Result<Value> {
    match (v, fmt) {
        (Value::Str(s), Value::Str(f)) => {
            use std::ffi::CString;
            let s_c = CString::new(s.as_str()).unwrap_or_default();
            let f_c = CString::new(f.as_str()).unwrap_or_default();
            let mut t: libc::tm = unsafe { std::mem::zeroed() };
            unsafe { libc::strptime(s_c.as_ptr(), f_c.as_ptr(), &mut t) };
            // Compute yday and wday using mktime
            let mut t2 = t;
            unsafe { libc::timegm(&mut t2) };
            t.tm_wday = t2.tm_wday;
            t.tm_yday = t2.tm_yday;
            Ok(Value::Arr(Rc::new(vec![
                Value::number((t.tm_year + 1900) as f64),
                Value::number(t.tm_mon as f64),
                Value::number(t.tm_mday as f64),
                Value::number(t.tm_hour as f64),
                Value::number(t.tm_min as f64),
                Value::number(t.tm_sec as f64),
                Value::number(t.tm_wday as f64),
                Value::number(t.tm_yday as f64),
            ])))
        }
        _ => bail!("strptime/1 requires string inputs and arguments"),
    }
}

fn rt_strflocaltime_impl(input: &Value, fmt: &Value) -> Result<Value> {
    let fmt_str = match fmt {
        Value::Str(s) => s.clone(),
        _ => bail!("strflocaltime/1 requires a string format"),
    };
    match input {
        Value::Arr(a) => {
            if !a.is_empty() {
                if let Value::Str(_) = &a[0] {
                    bail!("strflocaltime/1 requires parsed datetime inputs");
                }
            }
            let t = time_arr_to_tm(a)?;
            Ok(Value::from_str(&format_tm(&t, fmt_str.as_str())))
        }
        Value::Num(n, _) => {
            // Convert epoch to localtime first, then format
            let secs = *n as i64;
            use libc::{localtime_r, time_t, tm};
            let t_val = secs as time_t;
            let mut t: tm = unsafe { std::mem::zeroed() };
            unsafe { localtime_r(&t_val, &mut t) };
            Ok(Value::from_str(&format_tm(&t, fmt_str.as_str())))
        }
        _ => bail!("strflocaltime/1 requires parsed datetime inputs"),
    }
}

// -----------------------------------------------------------------------
// ISO 8601 date conversion
// -----------------------------------------------------------------------

fn rt_fromisodate(v: &Value) -> Result<Value> {
    use chrono::NaiveDateTime;

    // jq's `fromdateiso8601` (and its alias `fromdate`) validates with
    // strptime against the literal format `%Y-%m-%dT%H:%M:%SZ` — integer
    // seconds, mandatory `T`, mandatory uppercase `Z`. Fractional seconds
    // and timezone offsets are rejected. See #458.
    let s = match v {
        Value::Str(s) => s.as_str(),
        _ => bail!("strptime/1 requires string inputs and arguments"),
    };

    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ") {
        return Ok(Value::number(ndt.and_utc().timestamp() as f64));
    }

    bail!(r#"date "{}" does not match format "%Y-%m-%dT%H:%M:%SZ""#, s)
}

fn rt_toisodate(v: &Value) -> Result<Value> {
    use chrono::DateTime;

    // jq defines `todate`/`todateiso8601` as
    // `def todate: strftime("%Y-%m-%dT%H:%M:%SZ")`, so broken-down-time
    // arrays are accepted and non-numeric/non-array inputs surface the
    // standard `strftime/1 requires parsed datetime inputs` wording.
    // Delegate to `rt_strftime` for the array branch and the type-error
    // wording (#531). Number input still goes through the chrono path
    // below so the existing fractional-second handling is preserved.
    if !matches!(v, Value::Num(_, _)) {
        let fmt = Value::from_str("%Y-%m-%dT%H:%M:%SZ");
        return rt_strftime(v, &fmt);
    }
    let epoch = match v {
        Value::Num(n, _) => *n,
        _ => unreachable!(),
    };

    let secs = epoch.floor() as i64;
    let frac = epoch - epoch.floor();

    if frac.abs() < 1e-9 {
        // Integer epoch: no fractional part
        let dt = DateTime::from_timestamp(secs, 0)
            .ok_or_else(|| anyhow::anyhow!("toisodate: invalid epoch: {}", epoch))?;
        let formatted = dt.format("%Y-%m-%dT%H:%M:%SZ").to_string();
        Ok(Value::from_str(&formatted))
    } else {
        // Float epoch: include milliseconds
        // Use round(frac * 1000) to get millis to avoid precision loss
        let millis = (frac * 1000.0).round() as u32;
        let nanos = millis * 1_000_000;
        let dt = DateTime::from_timestamp(secs, nanos)
            .ok_or_else(|| anyhow::anyhow!("toisodate: invalid epoch: {}", epoch))?;
        let formatted = format!("{}.{:03}Z", dt.format("%Y-%m-%dT%H:%M:%S"), millis);
        Ok(Value::from_str(&formatted))
    }
}

// -----------------------------------------------------------------------
// Environment
// -----------------------------------------------------------------------

fn rt_env() -> Value {
    let mut env = new_objmap();
    for (k, v) in std::env::vars() {
        env.insert(KeyStr::from(k), Value::from_string(v));
    }
    Value::object_from_map(env)
}

pub fn rt_builtins() -> Value {
    // Source of truth for the `builtins/0` advertised list. Every spec
    // appears at most once (jq 1.8.1 emits each name+arity exactly once),
    // and any builtin callable from filter source is listed here so that
    // feature-probing scripts (`builtins | contains(...)`) can find it.
    let builtins = vec![
        "length/0", "utf8bytelength/0", "keys/0", "keys_unsorted/0", "values/0",
        "has/1", "in/1",
        "contains/1", "inside/1",
        "startswith/1", "endswith/1", "ltrimstr/1", "rtrimstr/1", "trimstr/1",
        "split/1", "split/2", "splits/1", "splits/2", "join/1",
        "test/1", "test/2", "match/1", "match/2",
        "capture/1", "capture/2", "scan/1", "scan/2",
        "sub/2", "sub/3", "gsub/2", "gsub/3",
        "tostring/0", "tonumber/0", "type/0",
        "empty/0", "error/0", "error/1",
        "not/0",
        "map/1", "map_values/1", "select/1",
        "add/0", "add/1",
        "any/0", "any/1", "any/2", "all/0", "all/1", "all/2",
        "flatten/0", "flatten/1",
        "range/1", "range/2", "range/3",
        "floor/0", "ceil/0", "round/0",
        "sqrt/0", "fabs/0", "abs/0",
        "pow/2", "log/0", "log2/0", "log10/0",
        "exp/0", "exp2/0", "exp10/0",
        "expm1/0", "log1p/0",
        "erf/0", "erfc/0",
        "sin/0", "cos/0", "tan/0", "asin/0", "acos/0", "atan/0",
        "sinh/0", "cosh/0", "tanh/0", "asinh/0", "acosh/0", "atanh/0",
        "atan2/2", "hypot/2",
        "copysign/2", "fdim/2", "fmax/2", "fmin/2", "fmod/2",
        "nextafter/2", "nexttoward/2", "jn/2", "yn/2",
        "significand/0", "logb/0",
        "nearbyint/0", "trunc/0", "rint/0",
        "j0/0", "j1/0", "y0/0", "y1/0", "cbrt/0",
        "gamma/0", "tgamma/0", "lgamma/0", "lgamma_r/0",
        "frexp/0", "fma/3", "ldexp/2", "scalb/2", "scalbln/2",
        "drem/2", "remainder/2",
        "min/0", "max/0", "min_by/1", "max_by/1",
        "sort/0", "sort_by/1", "group_by/1", "unique/0", "unique_by/1",
        "reverse/0",
        "to_entries/0", "from_entries/0", "with_entries/1",
        "paths/0", "paths/1", "path/1",
        "getpath/1", "setpath/2", "delpaths/1",
        "tojson/0", "fromjson/0",
        "ascii_downcase/0", "ascii_upcase/0",
        "explode/0", "implode/0",
        "indices/1", "index/1", "rindex/1",
        "ltrim/0", "rtrim/0", "trim/0",
        "nan/0", "infinite/0",
        "isinfinite/0", "isnan/0", "isnormal/0", "isfinite/0",
        "finites/0", "normals/0",
        // Type-filter generators (each yields the input iff it has the
        // matching type, otherwise empty). All of these are callable in
        // jq-jit but were missing from the advertisement.
        "arrays/0", "booleans/0", "nulls/0", "numbers/0", "objects/0",
        "strings/0", "iterables/0", "scalars/0",
        "env/0", "debug/0", "debug/1", "stderr/0",
        "input/0", "inputs/0", "input_line_number/0", "input_filename/0",
        "combinations/0", "combinations/1", "modf/0",
        "limit/2", "first/0", "first/1", "last/0", "last/1",
        "nth/1", "nth/2",
        "while/2", "until/2", "repeat/1",
        "recurse/0", "recurse/1", "recurse/2",
        "transpose/0",
        "now/0", "gmtime/0", "localtime/0", "mktime/0",
        "strftime/1", "strflocaltime/1", "strptime/1",
        "todate/0", "fromdate/0", "date/0",
        "fromdateiso8601/0", "todateiso8601/0",
        "modulemeta/0", "builtins/0",
        "isempty/1",
        "del/1", "pick/1",
        "halt/0", "halt_error/0", "halt_error/1",
        "bsearch/1",
        "walk/1",
        "INDEX/1", "INDEX/2", "IN/1", "IN/2",
        "JOIN/2", "JOIN/3", "JOIN/4",
        "skip/2",
        "have_decnum/0", "have_literal_numbers/0",
        "exec/1", "exec/2", "execv/1",
        "fromcsv/0", "fromtsv/0", "fromcsvh/0", "fromcsvh/1", "fromtsvh/0", "fromtsvh/1",
        "toboolean/0",
        "format/1",
    ];
    let arr: Vec<Value> = builtins.iter()
        .map(|&name| Value::from_str(name))
        .collect();
    Value::Arr(Rc::new(arr))
}
