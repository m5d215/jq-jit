//! Runtime helpers for jq operations.
//!
//! These functions implement jq's built-in operations using our Value type.
//! They're called by both the interpreter and (eventually) JIT-compiled code.

use std::rc::Rc;

use anyhow::{Result, bail};
use indexmap::IndexMap;

use crate::value::Value;

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
        "flatten" => unary_op(args, |v| rt_flatten(v, None)),
        "unique" => unary_op(args, rt_unique),
        "min" => unary_op(args, rt_min),
        "max" => unary_op(args, rt_max),
        "add" => unary_op(args, rt_add_all),
        "any" => unary_op(args, rt_any),
        "all" => unary_op(args, rt_all),
        "floor" => unary_op(args, rt_floor),
        "ceil" => unary_op(args, rt_ceil),
        "round" => unary_op(args, rt_round),
        "fabs" | "abs" => unary_op(args, rt_fabs),
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
        "nan" => Ok(Value::Num(f64::NAN)),
        "infinite" => Ok(Value::Num(f64::INFINITY)),
        "isinfinite" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::from_bool(n.is_infinite())),
            _ => Ok(Value::False),
        }),
        "isnan" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::from_bool(n.is_nan())),
            _ => Ok(Value::False),
        }),
        "isnormal" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::from_bool(n.is_normal())),
            _ => Ok(Value::False),
        }),
        "isfinite" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::from_bool(n.is_finite())),
            _ => Ok(Value::False),
        }),
        "ascii" => unary_op(args, rt_ascii),
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
        "now" => Ok(Value::Num(std::time::SystemTime::now()
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
        "ltrimstr" => binary_arg(args, rt_ltrimstr),
        "rtrimstr" => binary_arg(args, rt_rtrimstr),
        "split" => binary_arg(args, rt_split),
        "join" => binary_arg(args, rt_join),
        "index" | "rindex" => binary_arg(args, |a, b| rt_str_index(a, b, name == "rindex")),
        "indices" | "rindices" => binary_arg(args, rt_indices),
        "test" => binary_arg(args, rt_test),
        "match" => binary_arg(args, rt_match),
        "capture" => binary_arg(args, rt_capture),
        "scan" => binary_arg(args, rt_scan),
        "sub" | "gsub" => {
            // These need 3 args: input, regex, replacement
            if args.len() >= 3 {
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
        "map" | "select" | "map_values" | "with_entries" | "from_entries" | "to_entries" => {
            Ok(args.first().cloned().unwrap_or(Value::Null))
        }
        "flatten" if args.len() >= 2 => {
            // flatten(depth)
            if let Value::Num(depth) = &args[1] {
                rt_flatten(&args[0], Some(*depth as usize))
            } else {
                rt_flatten(&args[0], None)
            }
        }
        "pow" => binary_arg(args, rt_pow),
        "atan2" => binary_arg(args, |a, b| match (a, b) {
            (Value::Num(y), Value::Num(x)) => Ok(Value::Num(y.atan2(*x))),
            _ => bail!("atan2 requires numbers"),
        }),
        "log" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::Num(n.ln())),
            _ => bail!("log requires number"),
        }),
        "log2" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::Num(n.log2())),
            _ => bail!("log2 requires number"),
        }),
        "log10" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::Num(n.log10())),
            _ => bail!("log10 requires number"),
        }),
        "exp" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::Num(n.exp())),
            _ => bail!("exp requires number"),
        }),
        "exp2" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::Num(2f64.powf(*n))),
            _ => bail!("exp2 requires number"),
        }),
        "sin" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::Num(n.sin())),
            _ => bail!("sin requires number"),
        }),
        "cos" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::Num(n.cos())),
            _ => bail!("cos requires number"),
        }),
        "tan" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::Num(n.tan())),
            _ => bail!("tan requires number"),
        }),
        "asin" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::Num(n.asin())),
            _ => bail!("asin requires number"),
        }),
        "acos" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::Num(n.acos())),
            _ => bail!("acos requires number"),
        }),
        "atan" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::Num(n.atan())),
            _ => bail!("atan requires number"),
        }),
        "cbrt" => unary_op(args, |v| match v {
            Value::Num(n) => Ok(Value::Num(n.cbrt())),
            _ => bail!("cbrt requires number"),
        }),
        "significand" | "exponent" | "logb" | "nearbyint" | "trunc" | "rint" | "j0" | "j1" => {
            unary_op(args, |v| match v {
                Value::Num(n) => {
                    let r = match name {
                        "significand" => {
                            // significand(x) = x * 2^(-ilogb(x))
                            let e = libm::ilogb(*n);
                            *n * 2f64.powi(-e)
                        }
                        "exponent" => libm::ilogb(*n) as f64,
                        "logb" => {
                            let e = libm::ilogb(*n);
                            e as f64
                        }
                        "nearbyint" => libm::rint(*n),
                        "trunc" => n.trunc(),
                        "rint" => libm::rint(*n),
                        "j0" => libm::j0(*n),
                        "j1" => libm::j1(*n),
                        _ => unreachable!(),
                    };
                    Ok(Value::Num(r))
                }
                _ => bail!("{} requires number", name),
            })
        }
        "gmtime" => unary_op(args, rt_gmtime),
        "mktime" => unary_op(args, rt_mktime),
        "strftime" => binary_arg(args, rt_strftime),
        "strptime" => binary_arg(args, rt_strptime),
        "modulemeta" => {
            bail!("modulemeta is not supported");
        }
        "have_decnum" => {
            // We don't have arbitrary precision, return false
            Ok(Value::False)
        }
        "have_sql" => Ok(Value::False),
        "have_bom" => Ok(Value::True),
        _ => {
            // Unknown builtin - try to handle common patterns
            bail!("unknown builtin: {} (nargs={})", name, args.len());
        }
    }
}

// -----------------------------------------------------------------------
// Helper macros
// -----------------------------------------------------------------------

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

// -----------------------------------------------------------------------
// Arithmetic
// -----------------------------------------------------------------------

fn rt_add(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x + y)),
        (Value::Str(x), Value::Str(y)) => {
            Ok(Value::from_string(format!("{}{}", x, y)))
        }
        (Value::Arr(x), Value::Arr(y)) => {
            let mut result = (**x).clone();
            result.extend((**y).iter().cloned());
            Ok(Value::Arr(Rc::new(result)))
        }
        (Value::Obj(x), Value::Obj(y)) => {
            let mut result = (**x).clone();
            for (k, v) in y.iter() {
                result.insert(k.clone(), v.clone());
            }
            Ok(Value::Obj(Rc::new(result)))
        }
        (Value::Null, x) | (x, Value::Null) => Ok(x.clone()),
        _ => bail!(
            "{} and {} cannot be added",
            a.type_name(),
            b.type_name()
        ),
    }
}

fn rt_sub(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x - y)),
        (Value::Arr(x), Value::Arr(y)) => {
            let result: Vec<Value> = x.iter()
                .filter(|v| !y.contains(v))
                .cloned()
                .collect();
            Ok(Value::Arr(Rc::new(result)))
        }
        _ => bail!(
            "{} and {} cannot be subtracted",
            a.type_name(),
            b.type_name()
        ),
    }
}

fn rt_mul(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x * y)),
        (Value::Str(s), Value::Num(n)) | (Value::Num(n), Value::Str(s)) => {
            if *n <= 0.0 {
                Ok(Value::Null)
            } else {
                Ok(Value::from_string(s.repeat(*n as usize)))
            }
        }
        (Value::Obj(x), Value::Obj(y)) => {
            // Object multiplication = recursive merge
            Ok(Value::Obj(Rc::new(merge_objects(x, y))))
        }
        (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
        _ => bail!(
            "{} and {} cannot be multiplied",
            a.type_name(),
            b.type_name()
        ),
    }
}

fn merge_objects(a: &IndexMap<String, Value>, b: &IndexMap<String, Value>) -> IndexMap<String, Value> {
    let mut result = a.clone();
    for (k, v) in b.iter() {
        if let Some(existing) = result.get(k) {
            if let (Value::Obj(ea), Value::Obj(eb)) = (existing, v) {
                result.insert(k.clone(), Value::Obj(Rc::new(merge_objects(ea, eb))));
                continue;
            }
        }
        result.insert(k.clone(), v.clone());
    }
    result
}

fn rt_div(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => {
            if *y == 0.0 {
                bail!("{} and {} cannot be divided because the divisor is zero", x, y);
            }
            Ok(Value::Num(x / y))
        }
        (Value::Str(s), Value::Str(sep)) => {
            // String division = split
            let parts: Vec<Value> = s.split(sep.as_str())
                .map(|p| Value::from_str(p))
                .collect();
            Ok(Value::Arr(Rc::new(parts)))
        }
        _ => bail!(
            "{} and {} cannot be divided",
            a.type_name(),
            b.type_name()
        ),
    }
}

fn rt_mod(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => {
            if *y == 0.0 {
                bail!("{} and {} cannot be divided (remainder) because the divisor is zero", x, y);
            }
            Ok(Value::Num(x % y))
        }
        _ => bail!(
            "{} and {} cannot be divided (remainder)",
            a.type_name(),
            b.type_name()
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

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::True, Value::True) => true,
        (Value::False, Value::False) => true,
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Arr(x), Value::Arr(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| values_equal(a, b))
        }
        (Value::Obj(x), Value::Obj(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|yv| values_equal(v, yv)))
        }
        _ => false,
    }
}

fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let type_order = |v: &Value| -> u8 {
        match v {
            Value::Null => 0,
            Value::False => 1,
            Value::True => 2,
            Value::Num(_) => 3,
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
        (Value::Num(x), Value::Num(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
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
        (Value::Obj(x), Value::Obj(y)) => {
            // Compare by sorted keys then values
            let mut xkeys: Vec<&String> = x.keys().collect();
            let mut ykeys: Vec<&String> = y.keys().collect();
            xkeys.sort();
            ykeys.sort();
            for (xk, yk) in xkeys.iter().zip(ykeys.iter()) {
                let kord = xk.cmp(yk);
                if kord != Ordering::Equal {
                    return kord;
                }
                let xv = x.get(xk.as_str()).unwrap();
                let yv = y.get(yk.as_str()).unwrap();
                let vord = compare_values(xv, yv);
                if vord != Ordering::Equal {
                    return vord;
                }
            }
            x.len().cmp(&y.len())
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
        Value::Str(s) => Ok(Value::Num(s.len() as f64)),
        _ => v.length(),
    }
}

fn rt_type(v: &Value) -> Result<Value> {
    Ok(Value::from_str(v.type_name()))
}

fn rt_keys(v: &Value, sorted: bool) -> Result<Value> {
    match v {
        Value::Obj(o) => {
            let mut keys: Vec<Value> = o.keys().map(|k| Value::from_str(k)).collect();
            if sorted {
                keys.sort_by(|a, b| compare_values(a, b));
            }
            Ok(Value::Arr(Rc::new(keys)))
        }
        Value::Arr(a) => {
            let keys: Vec<Value> = (0..a.len()).map(|i| Value::Num(i as f64)).collect();
            Ok(Value::Arr(Rc::new(keys)))
        }
        _ => bail!("{} has no keys", v.type_name()),
    }
}

fn rt_values(v: &Value) -> Result<Value> {
    match v {
        Value::Obj(o) => {
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
            sorted.sort_by(compare_values);
            Ok(Value::Arr(Rc::new(sorted)))
        }
        _ => bail!("{} is not an array", v.type_name()),
    }
}

fn rt_reverse(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) => {
            let mut reversed = (**a).clone();
            reversed.reverse();
            Ok(Value::Arr(Rc::new(reversed)))
        }
        Value::Str(s) => {
            Ok(Value::from_string(s.chars().rev().collect()))
        }
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
        _ => bail!("{} is not an array", v.type_name()),
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
            sorted.sort_by(compare_values);
            sorted.dedup_by(|a, b| values_equal(a, b));
            Ok(Value::Arr(Rc::new(sorted)))
        }
        _ => bail!("{} is not an array", v.type_name()),
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
        _ => bail!("{} is not an array", v.type_name()),
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
        _ => bail!("{} is not an array", v.type_name()),
    }
}

fn rt_add_all(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) if a.is_empty() => Ok(Value::Null),
        Value::Arr(a) => {
            let mut result = a[0].clone();
            for item in &a[1..] {
                result = rt_add(&result, item)?;
            }
            Ok(result)
        }
        _ => bail!("{} is not an array", v.type_name()),
    }
}

fn rt_any(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) => Ok(Value::from_bool(a.iter().any(|v| v.is_true()))),
        _ => bail!("{} is not an array", v.type_name()),
    }
}

fn rt_all(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) => Ok(Value::from_bool(a.iter().all(|v| v.is_true()))),
        _ => bail!("{} is not an array", v.type_name()),
    }
}

fn rt_floor(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n) => Ok(Value::Num(n.floor())),
        _ => bail!("{} cannot be floored", v.type_name()),
    }
}

fn rt_ceil(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n) => Ok(Value::Num(n.ceil())),
        _ => bail!("{} cannot be ceiled", v.type_name()),
    }
}

fn rt_round(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n) => Ok(Value::Num(n.round())),
        _ => bail!("{} cannot be rounded", v.type_name()),
    }
}

fn rt_fabs(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n) => Ok(Value::Num(n.abs())),
        _ => bail!("{} is not a number", v.type_name()),
    }
}

fn rt_sqrt(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n) => Ok(Value::Num(n.sqrt())),
        _ => bail!("{} is not a number", v.type_name()),
    }
}

fn rt_tostring(v: &Value) -> Result<Value> {
    match v {
        Value::Str(_) => Ok(v.clone()),
        _ => Ok(Value::from_string(crate::value::value_to_json(v))),
    }
}

fn rt_tonumber(v: &Value) -> Result<Value> {
    match v {
        Value::Num(_) => Ok(v.clone()),
        Value::Str(s) => {
            let s = s.trim();
            match s.parse::<f64>() {
                Ok(n) => Ok(Value::Num(n)),
                Err(_) => bail!("Cannot convert {:?} to number", s),
            }
        }
        _ => bail!("{} cannot be converted to number", v.type_name()),
    }
}

fn rt_ascii_downcase(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => Ok(Value::from_string(s.to_lowercase())),
        _ => bail!("{} cannot be lowercased", v.type_name()),
    }
}

fn rt_ascii_upcase(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => Ok(Value::from_string(s.to_uppercase())),
        _ => bail!("{} cannot be uppercased", v.type_name()),
    }
}

fn rt_ltrim(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => Ok(Value::from_string(s.trim_start().to_string())),
        _ => bail!("{} cannot be trimmed", v.type_name()),
    }
}

fn rt_rtrim(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => Ok(Value::from_string(s.trim_end().to_string())),
        _ => bail!("{} cannot be trimmed", v.type_name()),
    }
}

fn rt_trim(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => Ok(Value::from_string(s.trim().to_string())),
        _ => bail!("{} cannot be trimmed", v.type_name()),
    }
}

fn rt_explode(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => {
            let codepoints: Vec<Value> = s.chars()
                .map(|c| Value::Num(c as u32 as f64))
                .collect();
            Ok(Value::Arr(Rc::new(codepoints)))
        }
        _ => bail!("{} cannot be exploded", v.type_name()),
    }
}

fn rt_implode(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) => {
            let mut s = String::new();
            for item in a.iter() {
                match item {
                    Value::Num(n) => {
                        let cp = *n as u32;
                        match char::from_u32(cp) {
                            Some(c) => s.push(c),
                            None => bail!("Invalid codepoint: {}", cp),
                        }
                    }
                    _ => bail!("implode requires array of numbers"),
                }
            }
            Ok(Value::from_string(s))
        }
        _ => bail!("{} cannot be imploded", v.type_name()),
    }
}

fn rt_tojson(v: &Value) -> Result<Value> {
    Ok(Value::from_string(crate::value::value_to_json(v)))
}

fn rt_fromjson(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => crate::value::json_to_value(s),
        _ => bail!("{} cannot be parsed as JSON", v.type_name()),
    }
}

fn rt_to_entries(v: &Value) -> Result<Value> {
    match v {
        Value::Obj(o) => {
            let entries: Vec<Value> = o.iter().map(|(k, v)| {
                let mut entry = IndexMap::new();
                entry.insert("key".to_string(), Value::from_str(k));
                entry.insert("value".to_string(), v.clone());
                Value::Obj(Rc::new(entry))
            }).collect();
            Ok(Value::Arr(Rc::new(entries)))
        }
        _ => bail!("{} has no entries", v.type_name()),
    }
}

fn rt_from_entries(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) => {
            let mut obj = IndexMap::new();
            for entry in a.iter() {
                match entry {
                    Value::Obj(o) => {
                        let key = o.get("key").or_else(|| o.get("name"))
                            .cloned().unwrap_or(Value::Null);
                        let val = o.get("value").cloned().unwrap_or(Value::Null);
                        let key_str = match &key {
                            Value::Str(s) => s.as_ref().clone(),
                            Value::Num(n) => crate::value::format_jq_number(*n),
                            _ => crate::value::value_to_json(&key),
                        };
                        obj.insert(key_str, val);
                    }
                    _ => bail!("from_entries requires array of objects"),
                }
            }
            Ok(Value::Obj(Rc::new(obj)))
        }
        _ => bail!("{} cannot be converted from entries", v.type_name()),
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

fn rt_ascii(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n) => {
            let cp = *n as u32;
            match char::from_u32(cp) {
                Some(c) => Ok(Value::from_string(c.to_string())),
                None => bail!("Invalid ASCII codepoint: {}", cp),
            }
        }
        _ => bail!("{} cannot be converted to ASCII", v.type_name()),
    }
}

// -----------------------------------------------------------------------
// Binary builtins
// -----------------------------------------------------------------------

fn rt_has(v: &Value, key: &Value) -> Result<Value> {
    match (v, key) {
        (Value::Obj(o), Value::Str(k)) => Ok(Value::from_bool(o.contains_key(k.as_str()))),
        (Value::Arr(a), Value::Num(n)) => {
            let idx = *n as usize;
            Ok(Value::from_bool(idx < a.len()))
        }
        _ => bail!("{} cannot have {} as key", v.type_name(), key.type_name()),
    }
}

fn rt_in(v: &Value, container: &Value) -> Result<Value> {
    match (v, container) {
        (Value::Str(k), Value::Obj(o)) => Ok(Value::from_bool(o.contains_key(k.as_str()))),
        (Value::Num(n), Value::Arr(a)) => {
            let idx = *n as usize;
            Ok(Value::from_bool(idx < a.len()))
        }
        _ => Ok(Value::False),
    }
}

fn rt_contains(a: &Value, b: &Value) -> Result<Value> {
    Ok(Value::from_bool(value_contains(a, b)))
}

fn value_contains(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Str(x), Value::Str(y)) => x.contains(y.as_str()),
        (Value::Arr(x), Value::Arr(y)) => {
            y.iter().all(|yv| x.iter().any(|xv| value_contains(xv, yv)))
        }
        (Value::Obj(x), Value::Obj(y)) => {
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
        _ => bail!("startswith requires strings"),
    }
}

fn rt_endswith(v: &Value, suffix: &Value) -> Result<Value> {
    match (v, suffix) {
        (Value::Str(s), Value::Str(p)) => Ok(Value::from_bool(s.ends_with(p.as_str()))),
        _ => bail!("endswith requires strings"),
    }
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
        _ => bail!("ltrimstr requires strings"),
    }
}

fn rt_rtrimstr(v: &Value, suffix: &Value) -> Result<Value> {
    match (v, suffix) {
        (Value::Str(s), Value::Str(p)) => {
            if let Some(rest) = s.strip_suffix(p.as_str()) {
                Ok(Value::from_str(rest))
            } else {
                Ok(v.clone())
            }
        }
        _ => bail!("rtrimstr requires strings"),
    }
}

fn rt_split(v: &Value, sep: &Value) -> Result<Value> {
    match (v, sep) {
        (Value::Str(s), Value::Str(p)) => {
            let parts: Vec<Value> = s.split(p.as_str())
                .map(|p| Value::from_str(p))
                .collect();
            Ok(Value::Arr(Rc::new(parts)))
        }
        _ => bail!("split requires strings"),
    }
}

fn rt_join(v: &Value, sep: &Value) -> Result<Value> {
    match (v, sep) {
        (Value::Arr(a), Value::Str(s)) => {
            let parts: Vec<String> = a.iter().map(|v| match v {
                Value::Str(s) => s.as_ref().clone(),
                Value::Null => String::new(),
                _ => crate::value::value_to_json(v),
            }).collect();
            Ok(Value::from_string(parts.join(s.as_str())))
        }
        _ => bail!("join requires array and string"),
    }
}

fn rt_str_index(v: &Value, target: &Value, is_rindex: bool) -> Result<Value> {
    match (v, target) {
        (Value::Str(s), Value::Str(t)) => {
            if is_rindex {
                match s.rfind(t.as_str()) {
                    Some(pos) => Ok(Value::Num(pos as f64)),
                    None => Ok(Value::Null),
                }
            } else {
                match s.find(t.as_str()) {
                    Some(pos) => Ok(Value::Num(pos as f64)),
                    None => Ok(Value::Null),
                }
            }
        }
        (Value::Arr(a), _) => {
            if is_rindex {
                for (i, item) in a.iter().enumerate().rev() {
                    if values_equal(item, target) {
                        return Ok(Value::Num(i as f64));
                    }
                }
                Ok(Value::Null)
            } else {
                for (i, item) in a.iter().enumerate() {
                    if values_equal(item, target) {
                        return Ok(Value::Num(i as f64));
                    }
                }
                Ok(Value::Null)
            }
        }
        _ => bail!("index/rindex requires string or array"),
    }
}

fn rt_indices(v: &Value, target: &Value) -> Result<Value> {
    match (v, target) {
        (Value::Str(s), Value::Str(t)) => {
            let mut indices = Vec::new();
            let mut start = 0;
            while let Some(pos) = s[start..].find(t.as_str()) {
                indices.push(Value::Num((start + pos) as f64));
                start += pos + 1;
            }
            Ok(Value::Arr(Rc::new(indices)))
        }
        (Value::Arr(a), _) => {
            let mut indices = Vec::new();
            for (i, item) in a.iter().enumerate() {
                if values_equal(item, target) {
                    indices.push(Value::Num(i as f64));
                }
            }
            Ok(Value::Arr(Rc::new(indices)))
        }
        _ => bail!("indices requires string or array"),
    }
}

fn rt_pow(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x.powf(*y))),
        _ => bail!("pow requires numbers"),
    }
}

fn rt_getpath(v: &Value, path: &Value) -> Result<Value> {
    match path {
        Value::Arr(p) => {
            let mut current = v.clone();
            for key in p.iter() {
                match (&current, key) {
                    (Value::Obj(o), Value::Str(k)) => {
                        current = o.get(k.as_str()).cloned().unwrap_or(Value::Null);
                    }
                    (Value::Arr(a), Value::Num(n)) => {
                        let idx = *n as i64;
                        let actual = if idx < 0 { (a.len() as i64 + idx) as usize } else { idx as usize };
                        current = a.get(actual).cloned().unwrap_or(Value::Null);
                    }
                    (Value::Null, _) => {
                        current = Value::Null;
                    }
                    _ => return Ok(Value::Null),
                }
            }
            Ok(current)
        }
        _ => bail!("getpath requires array path"),
    }
}

fn rt_setpath(v: &Value, path: &Value, val: &Value) -> Result<Value> {
    match path {
        Value::Arr(p) if p.is_empty() => Ok(val.clone()),
        Value::Arr(p) => {
            let key = &p[0];
            let rest = Value::Arr(Rc::new(p[1..].to_vec()));
            match (v, key) {
                (Value::Obj(o), Value::Str(k)) => {
                    let inner = o.get(k.as_str()).cloned().unwrap_or(Value::Null);
                    let new_inner = rt_setpath(&inner, &rest, val)?;
                    let mut new_obj = (**o).clone();
                    new_obj.insert(k.as_ref().clone(), new_inner);
                    Ok(Value::Obj(Rc::new(new_obj)))
                }
                (Value::Arr(a), Value::Num(n)) => {
                    let idx = *n as i64;
                    let actual = if idx < 0 { (a.len() as i64 + idx) as usize } else { idx as usize };
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
                    let mut obj = IndexMap::new();
                    obj.insert(k.as_ref().clone(), new_inner);
                    Ok(Value::Obj(Rc::new(obj)))
                }
                (Value::Null, Value::Num(n)) => {
                    let idx = *n as usize;
                    let new_inner = rt_setpath(&Value::Null, &rest, val)?;
                    let mut arr = vec![Value::Null; idx + 1];
                    arr[idx] = new_inner;
                    Ok(Value::Arr(Rc::new(arr)))
                }
                _ => bail!("Cannot set path"),
            }
        }
        _ => bail!("setpath requires array path"),
    }
}

fn rt_delpaths(v: &Value, paths: &Value) -> Result<Value> {
    match paths {
        Value::Arr(ps) => {
            let mut result = v.clone();
            // Sort paths in reverse order so deletions don't affect indices
            let mut sorted_paths: Vec<&Value> = ps.iter().collect();
            sorted_paths.sort_by(|a, b| compare_values(b, a));
            for path in sorted_paths {
                result = delete_path(&result, path)?;
            }
            Ok(result)
        }
        _ => bail!("delpaths requires array of paths"),
    }
}

fn delete_path(v: &Value, path: &Value) -> Result<Value> {
    match path {
        Value::Arr(p) if p.is_empty() => Ok(Value::Null),
        Value::Arr(p) if p.len() == 1 => {
            match (v, &p[0]) {
                (Value::Obj(o), Value::Str(k)) => {
                    let mut new_obj = (**o).clone();
                    new_obj.shift_remove(k.as_str());
                    Ok(Value::Obj(Rc::new(new_obj)))
                }
                (Value::Arr(a), Value::Num(n)) => {
                    let idx = *n as usize;
                    let mut new_arr = (**a).clone();
                    if idx < new_arr.len() {
                        new_arr.remove(idx);
                    }
                    Ok(Value::Arr(Rc::new(new_arr)))
                }
                _ => Ok(v.clone()),
            }
        }
        Value::Arr(p) => {
            let key = &p[0];
            let rest = Value::Arr(Rc::new(p[1..].to_vec()));
            match (v, key) {
                (Value::Obj(o), Value::Str(k)) => {
                    let inner = o.get(k.as_str()).cloned().unwrap_or(Value::Null);
                    let new_inner = delete_path(&inner, &rest)?;
                    let mut new_obj = (**o).clone();
                    new_obj.insert(k.as_ref().clone(), new_inner);
                    Ok(Value::Obj(Rc::new(new_obj)))
                }
                (Value::Arr(a), Value::Num(n)) => {
                    let idx = *n as usize;
                    if idx < a.len() {
                        let inner = &a[idx];
                        let new_inner = delete_path(inner, &rest)?;
                        let mut new_arr = (**a).clone();
                        new_arr[idx] = new_inner;
                        Ok(Value::Arr(Rc::new(new_arr)))
                    } else {
                        Ok(v.clone())
                    }
                }
                _ => Ok(v.clone()),
            }
        }
        _ => Ok(v.clone()),
    }
}

// -----------------------------------------------------------------------
// Regex
// -----------------------------------------------------------------------

fn rt_test(v: &Value, re: &Value) -> Result<Value> {
    match (v, re) {
        (Value::Str(s), Value::Str(r)) => {
            let regex = regex::Regex::new(r)
                .map_err(|e| anyhow::anyhow!("Invalid regex: {}", e))?;
            Ok(Value::from_bool(regex.is_match(s)))
        }
        _ => bail!("test requires string and regex"),
    }
}

fn rt_match(v: &Value, re: &Value) -> Result<Value> {
    match (v, re) {
        (Value::Str(s), Value::Str(r)) => {
            let regex = regex::Regex::new(r)
                .map_err(|e| anyhow::anyhow!("Invalid regex: {}", e))?;
            match regex.find(s) {
                Some(m) => {
                    let mut result = IndexMap::new();
                    result.insert("offset".to_string(), Value::Num(m.start() as f64));
                    result.insert("length".to_string(), Value::Num(m.len() as f64));
                    result.insert("string".to_string(), Value::from_str(m.as_str()));
                    // Add captures
                    let mut captures = Vec::new();
                    if let Some(caps) = regex.captures(s) {
                        for i in 1..caps.len() {
                            if let Some(cap) = caps.get(i) {
                                let mut c = IndexMap::new();
                                c.insert("offset".to_string(), Value::Num(cap.start() as f64));
                                c.insert("length".to_string(), Value::Num(cap.len() as f64));
                                c.insert("string".to_string(), Value::from_str(cap.as_str()));
                                c.insert("name".to_string(), Value::Null);
                                captures.push(Value::Obj(Rc::new(c)));
                            } else {
                                let mut c = IndexMap::new();
                                c.insert("offset".to_string(), Value::Num(-1.0));
                                c.insert("length".to_string(), Value::Num(0.0));
                                c.insert("string".to_string(), Value::Null);
                                c.insert("name".to_string(), Value::Null);
                                captures.push(Value::Obj(Rc::new(c)));
                            }
                        }
                    }
                    result.insert("captures".to_string(), Value::Arr(Rc::new(captures)));
                    Ok(Value::Obj(Rc::new(result)))
                }
                None => bail!("match failed"),
            }
        }
        _ => bail!("match requires string and regex"),
    }
}

fn rt_capture(v: &Value, re: &Value) -> Result<Value> {
    match (v, re) {
        (Value::Str(s), Value::Str(r)) => {
            let regex = regex::Regex::new(r)
                .map_err(|e| anyhow::anyhow!("Invalid regex: {}", e))?;
            match regex.captures(s) {
                Some(caps) => {
                    let mut result = IndexMap::new();
                    for name in regex.capture_names().flatten() {
                        if let Some(m) = caps.name(name) {
                            result.insert(name.to_string(), Value::from_str(m.as_str()));
                        } else {
                            result.insert(name.to_string(), Value::Null);
                        }
                    }
                    Ok(Value::Obj(Rc::new(result)))
                }
                None => bail!("capture failed"),
            }
        }
        _ => bail!("capture requires string and regex"),
    }
}

fn rt_scan(v: &Value, re: &Value) -> Result<Value> {
    match (v, re) {
        (Value::Str(s), Value::Str(r)) => {
            let regex = regex::Regex::new(r)
                .map_err(|e| anyhow::anyhow!("Invalid regex: {}", e))?;
            let results: Vec<Value> = regex.find_iter(s)
                .map(|m| Value::Arr(Rc::new(vec![Value::from_str(m.as_str())])))
                .collect();
            Ok(Value::Arr(Rc::new(results)))
        }
        _ => bail!("scan requires string and regex"),
    }
}

fn rt_sub_gsub(v: &Value, re: &Value, replacement: &Value, global: bool) -> Result<Value> {
    match (v, re, replacement) {
        (Value::Str(s), Value::Str(r), Value::Str(rep)) => {
            let regex = regex::Regex::new(r)
                .map_err(|e| anyhow::anyhow!("Invalid regex: {}", e))?;
            let result = if global {
                regex.replace_all(s, rep.as_str()).to_string()
            } else {
                regex.replace(s, rep.as_str()).to_string()
            };
            Ok(Value::from_string(result))
        }
        _ => bail!("sub/gsub requires string, regex, and replacement"),
    }
}

// -----------------------------------------------------------------------
// Date/Time
// -----------------------------------------------------------------------

fn rt_gmtime(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n) => {
            let secs = *n as i64;
            // Convert to broken-down time
            let time = libc_gmtime(secs);
            Ok(time)
        }
        _ => bail!("gmtime requires number"),
    }
}

fn libc_gmtime(secs: i64) -> Value {
    use libc::{gmtime_r, time_t, tm};
    let t = secs as time_t;
    let mut result: tm = unsafe { std::mem::zeroed() };
    unsafe { gmtime_r(&t, &mut result) };
    Value::Arr(Rc::new(vec![
        Value::Num(result.tm_sec as f64),
        Value::Num(result.tm_min as f64),
        Value::Num(result.tm_hour as f64),
        Value::Num(result.tm_mday as f64),
        Value::Num(result.tm_mon as f64),
        Value::Num(result.tm_year as f64),
        Value::Num(result.tm_wday as f64),
        Value::Num(result.tm_yday as f64),
    ]))
}

fn rt_mktime(v: &Value) -> Result<Value> {
    match v {
        Value::Arr(a) if a.len() >= 8 => {
            use libc::{mktime, tm};
            let mut t: tm = unsafe { std::mem::zeroed() };
            t.tm_sec = a[0].as_f64().unwrap_or(0.0) as i32;
            t.tm_min = a[1].as_f64().unwrap_or(0.0) as i32;
            t.tm_hour = a[2].as_f64().unwrap_or(0.0) as i32;
            t.tm_mday = a[3].as_f64().unwrap_or(0.0) as i32;
            t.tm_mon = a[4].as_f64().unwrap_or(0.0) as i32;
            t.tm_year = a[5].as_f64().unwrap_or(0.0) as i32;
            t.tm_wday = a[6].as_f64().unwrap_or(0.0) as i32;
            t.tm_yday = a[7].as_f64().unwrap_or(0.0) as i32;
            let result = unsafe { mktime(&mut t) };
            Ok(Value::Num(result as f64))
        }
        _ => bail!("mktime requires array of 8 numbers"),
    }
}

fn rt_strftime(v: &Value, fmt: &Value) -> Result<Value> {
    match (v, fmt) {
        (Value::Arr(_), Value::Str(f)) => {
            // TODO: Implement strftime
            Ok(Value::from_str(""))
        }
        _ => bail!("strftime requires time array and format string"),
    }
}

fn rt_strptime(v: &Value, fmt: &Value) -> Result<Value> {
    match (v, fmt) {
        (Value::Str(_s), Value::Str(_f)) => {
            // TODO: Implement strptime
            Ok(Value::Arr(Rc::new(vec![Value::Num(0.0); 8])))
        }
        _ => bail!("strptime requires string and format"),
    }
}

// -----------------------------------------------------------------------
// Environment
// -----------------------------------------------------------------------

fn rt_env() -> Value {
    let mut env = IndexMap::new();
    for (k, v) in std::env::vars() {
        env.insert(k, Value::from_string(v));
    }
    Value::Obj(Rc::new(env))
}

fn rt_builtins() -> Value {
    let builtins = vec![
        "length", "utf8bytelength", "keys", "values", "has", "in", "contains", "inside",
        "startswith", "endswith", "ltrimstr", "rtrimstr", "split", "join",
        "test", "match", "capture", "scan", "sub", "gsub",
        "tostring", "tonumber", "type", "empty", "error", "null", "true", "false",
        "not", "map", "select", "if_then_else", "reduce", "foreach",
        "add", "any", "all", "flatten", "range", "floor", "ceil", "round",
        "sqrt", "fabs", "abs", "pow", "log", "log2", "log10", "exp", "exp2",
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "min", "max", "sort", "sort_by", "group_by", "unique", "unique_by",
        "reverse", "keys_unsorted", "to_entries", "from_entries", "with_entries",
        "paths", "getpath", "setpath", "delpaths", "path",
        "tojson", "fromjson", "ascii_downcase", "ascii_upcase",
        "explode", "implode", "indices", "index", "rindex",
        "ltrim", "rtrim", "trim",
        "nan", "infinite", "isinfinite", "isnan", "isnormal", "isfinite",
        "env", "debug", "stderr", "input", "inputs",
        "limit", "first", "last", "nth", "while", "until", "repeat",
        "recurse", "recurse_down", "transpose",
        "ascii", "now", "gmtime", "mktime", "strftime", "strptime",
        "modulemeta", "builtins", "have_decnum", "have_sql", "have_bom",
        "min_by", "max_by", "map_values",
        "@base64", "@base64d", "@uri", "@csv", "@tsv", "@html", "@json", "@text", "@sh",
    ];
    let arr: Vec<Value> = builtins.iter()
        .map(|&name| Value::from_str(name))
        .collect();
    Value::Arr(Rc::new(arr))
}
