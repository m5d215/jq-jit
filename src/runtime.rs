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
        "nan" => Ok(Value::Num(f64::NAN, None)),
        "infinite" => Ok(Value::Num(f64::INFINITY, None)),
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
            Value::Num(n, _) => Ok(Value::from_bool(n.is_finite())),
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
            .as_secs_f64(), None)),
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
        // flatten with depth already handled above
        "pow" => ternary_arg(args, rt_pow),
        "atan2" => ternary_arg(args, |a, b| match (a, b) {
            (Value::Num(y, _), Value::Num(x, _)) => Ok(Value::Num(y.atan2(*x), None)),
            _ => bail!("atan2 requires numbers"),
        }),
        "log" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::Num(n.ln(), None)),
            _ => bail!("log requires number"),
        }),
        "log2" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::Num(n.log2(), None)),
            _ => bail!("log2 requires number"),
        }),
        "log10" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::Num(n.log10(), None)),
            _ => bail!("log10 requires number"),
        }),
        "exp" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::Num(n.exp(), None)),
            _ => bail!("exp requires number"),
        }),
        "exp2" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::Num(2f64.powf(*n), None)),
            _ => bail!("exp2 requires number"),
        }),
        "sin" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::Num(n.sin(), None)),
            _ => bail!("sin requires number"),
        }),
        "cos" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::Num(n.cos(), None)),
            _ => bail!("cos requires number"),
        }),
        "tan" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::Num(n.tan(), None)),
            _ => bail!("tan requires number"),
        }),
        "asin" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::Num(n.asin(), None)),
            _ => bail!("asin requires number"),
        }),
        "acos" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::Num(n.acos(), None)),
            _ => bail!("acos requires number"),
        }),
        "atan" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::Num(n.atan(), None)),
            _ => bail!("atan requires number"),
        }),
        "cbrt" => unary_op(args, |v| match v {
            Value::Num(n, _) => Ok(Value::Num(n.cbrt(), None)),
            _ => bail!("cbrt requires number"),
        }),
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
                    Ok(Value::Num(r, None))
                }
                _ => bail!("{} requires number", name),
            })
        }
        "gmtime" => unary_op(args, rt_gmtime),
        "mktime" => unary_op(args, rt_mktime),
        "strftime" => binary_arg(args, rt_strftime),
        "strptime" => binary_arg(args, rt_strptime),
        "todate" => unary_op(args, |v| rt_strftime(v, &Value::from_str("%Y-%m-%dT%H:%M:%SZ"))),
        "fromdate" => unary_op(args, |v| rt_strptime(v, &Value::from_str("%Y-%m-%dT%H:%M:%SZ"))),
        "date" => unary_op(args, |v| rt_strftime(v, &Value::from_str("%Y-%m-%dT%H:%M:%SZ"))),
        "trimstr" => binary_arg(args, |a, b| {
            let v = rt_ltrimstr(a, b)?;
            rt_rtrimstr(&v, b)
        }),
        "modulemeta" => {
            // modulemeta takes a module name string as input and returns metadata
            let input = &args[0];
            let name = match input {
                Value::Str(s) => s.as_str().to_string(),
                _ => bail!("modulemeta requires string input"),
            };
            // Need lib_dirs from env - they're passed through args[1] if available
            // For now, try to find module in common paths
            return Ok(Value::Obj(Rc::new({
                let mut m = IndexMap::new();
                m.insert("version".to_string(), Value::Num(0.1, None));
                m.insert("deps".to_string(), Value::Arr(Rc::new(vec![])));
                m.insert("defs".to_string(), Value::Arr(Rc::new(vec![])));
                m
            })));
        }
        "have_decnum" => {
            // We don't have arbitrary precision, return false
            Ok(Value::False)
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
                            std::cmp::Ordering::Equal => return Ok(Value::Num(mid as f64, None)),
                            std::cmp::Ordering::Less => lo = mid + 1,
                            std::cmp::Ordering::Greater => hi = mid - 1,
                        }
                    }
                    Ok(Value::Num(-(lo as f64) - 1.0, None))
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
    let json = crate::value::value_to_json(v);
    let tn = v.type_name();
    let jlen = json.as_bytes().len();
    if json.starts_with('"') {
        // String: threshold 28 bytes of JSON (including quotes)
        if jlen > 28 {
            let mut end = 25.min(jlen);
            while end > 0 && !json.is_char_boundary(end) { end -= 1; }
            format!("{} ({}...\")", tn, &json[..end])
        } else {
            format!("{} ({})", tn, json)
        }
    } else {
        // Non-string: threshold 28 bytes
        if jlen > 28 {
            let mut end = 26.min(jlen);
            while end > 0 && !json.is_char_boundary(end) { end -= 1; }
            format!("{} ({}...)", tn, &json[..end])
        } else {
            format!("{} ({})", tn, json)
        }
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
        (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::Num(x + y, None)),
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
            errdesc(a),
            errdesc(b)
        ),
    }
}

pub fn rt_sub(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::Num(x - y, None)),
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
        (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::Num(x * y, None)),
        (Value::Str(s), Value::Num(n, _)) | (Value::Num(n, _), Value::Str(s)) => {
            if n.is_nan() || *n < 0.0 {
                Ok(Value::Null)
            } else {
                let count = *n as usize;
                let result_len = s.len().saturating_mul(count);
                if result_len > 536870911 {
                    bail!("Repeat string result too long");
                }
                Ok(Value::from_string(s.repeat(count)))
            }
        }
        (Value::Obj(x), Value::Obj(y)) => {
            // Object multiplication = recursive merge
            Ok(Value::Obj(Rc::new(merge_objects(x, y))))
        }
        (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
        _ => bail!(
            "{} and {} cannot be multiplied",
            errdesc(a),
            errdesc(b)
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

pub fn rt_div(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x, _), Value::Num(y, _)) => {
            if *y == 0.0 {
                bail!("{} and {} cannot be divided because the divisor is zero", errdesc(a), errdesc(b));
            }
            Ok(Value::Num(x / y, None))
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
            errdesc(a),
            errdesc(b)
        ),
    }
}

pub fn rt_mod(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x, _), Value::Num(y, _)) => {
            // NaN inputs → NaN output
            if x.is_nan() || y.is_nan() {
                return Ok(Value::Num(f64::NAN, None));
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
                Ok(Value::Num(0.0, None))
            } else {
                Ok(Value::Num((xi % yi) as f64, None))
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
        (Value::Obj(x), Value::Obj(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|yv| values_equal(v, yv)))
        }
        _ => false,
    }
}

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
        (Value::Num(x, _), Value::Num(y, _)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
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
        Value::Str(s) => Ok(Value::Num(s.len() as f64, None)),
        _ => bail!("{} ({}) only strings have UTF-8 byte length", v.type_name(), crate::value::value_to_json(v)),
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
            let keys: Vec<Value> = (0..a.len()).map(|i| Value::Num(i as f64, None)).collect();
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
        Value::Null => Ok(Value::Arr(Rc::new(vec![]))),
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
        Value::Num(n, _) => Ok(Value::Num(n.floor(), None)),
        _ => bail!("{} cannot be floored", v.type_name()),
    }
}

fn rt_ceil(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n, _) => Ok(Value::Num(n.ceil(), None)),
        _ => bail!("{} cannot be ceiled", v.type_name()),
    }
}

fn rt_round(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n, _) => Ok(Value::Num(n.round(), None)),
        _ => bail!("{} cannot be rounded", v.type_name()),
    }
}

fn rt_fabs(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n, repr) => {
            if *n >= 0.0 { Ok(Value::Num(*n, repr.clone())) }
            else { Ok(Value::Num(n.abs(), None)) }
        }
        // abs on non-numbers returns the value for strings, errors for others
        Value::Str(_) => Ok(v.clone()),
        Value::Null => Ok(Value::Null),
        _ => v.length(),
    }
}

fn rt_sqrt(v: &Value) -> Result<Value> {
    match v {
        Value::Num(n, _) => Ok(Value::Num(n.sqrt(), None)),
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
        Value::Num(_, _) => Ok(v.clone()),
        Value::Str(s) => {
            let s_ref: &str = s.as_ref();
            if s_ref.is_empty() {
                bail!("Cannot convert empty string to number");
            }
            // jq rejects leading/trailing whitespace
            if s_ref.as_bytes()[0].is_ascii_whitespace() || s_ref.as_bytes()[s_ref.len()-1].is_ascii_whitespace() {
                bail!("Invalid numeric literal: {}", crate::value::value_to_json(v));
            }
            // Strip leading '+' for compatibility with jq
            let parse_str = if s_ref.starts_with('+') { &s_ref[1..] } else { s_ref };
            match parse_str.parse::<f64>() {
                Ok(n) => Ok(Value::Num(n, None)),
                Err(_) => bail!("Invalid numeric literal: {}", crate::value::value_to_json(v)),
            }
        }
        _ => bail!("{} cannot be converted to number", v.type_name()),
    }
}

fn rt_ascii_downcase(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => Ok(Value::from_string(s.chars().map(|c| if c.is_ascii() { c.to_ascii_lowercase() } else { c }).collect())),
        _ => bail!("{} cannot be lowercased", v.type_name()),
    }
}

fn rt_ascii_upcase(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => Ok(Value::from_string(s.chars().map(|c| if c.is_ascii() { c.to_ascii_uppercase() } else { c }).collect())),
        _ => bail!("{} cannot be uppercased", v.type_name()),
    }
}

fn rt_ltrim(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => Ok(Value::from_string(s.trim_start().to_string())),
        _ => bail!("trim input must be a string"),
    }
}

fn rt_rtrim(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => Ok(Value::from_string(s.trim_end().to_string())),
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
            let codepoints: Vec<Value> = s.chars()
                .map(|c| Value::Num(c as u32 as f64, None))
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
    Ok(Value::from_string(crate::value::value_to_json(v)))
}

fn rt_fromjson(v: &Value) -> Result<Value> {
    match v {
        Value::Str(s) => crate::value::json_to_value_libjq(s),
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
                        let key = o.get("key").or_else(|| o.get("Key"))
                            .or_else(|| o.get("name")).or_else(|| o.get("Name"))
                            .cloned().unwrap_or(Value::Null);
                        let val = o.get("value").or_else(|| o.get("Value"))
                            .cloned().unwrap_or(Value::Null);
                        let key_str = match &key {
                            Value::Str(s) => s.as_ref().clone(),
                            Value::Num(n, _) => crate::value::format_jq_number(*n),
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
        Value::Num(n, _) => {
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
        (Value::Arr(a), Value::Num(n, _)) => {
            if n.is_nan() || n.is_infinite() { return Ok(Value::False); }
            let idx = *n as i64;
            if idx < 0 { return Ok(Value::False); }
            Ok(Value::from_bool((idx as usize) < a.len()))
        }
        (Value::Null, _) => Ok(Value::False),
        _ => bail!("{} ({}) and {} ({}) cannot be has-tested", v.type_name(), crate::value::value_to_json(v), key.type_name(), crate::value::value_to_json(key)),
    }
}

fn rt_in(v: &Value, container: &Value) -> Result<Value> {
    match (v, container) {
        (Value::Str(k), Value::Obj(o)) => Ok(Value::from_bool(o.contains_key(k.as_str()))),
        (Value::Num(n, _), Value::Arr(a)) => {
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
        _ => bail!("startswith() requires string inputs"),
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
        _ => bail!("endswith() requires string inputs"),
    }
}

fn rt_split(v: &Value, sep: &Value) -> Result<Value> {
    match (v, sep) {
        (Value::Str(s), Value::Str(p)) => {
            if p.is_empty() {
                // split("") = each character as a separate element
                let parts: Vec<Value> = s.chars().map(|c| Value::from_string(c.to_string())).collect();
                Ok(Value::Arr(Rc::new(parts)))
            } else {
                let parts: Vec<Value> = s.split(p.as_str())
                    .map(|p| Value::from_str(p))
                    .collect();
                Ok(Value::Arr(Rc::new(parts)))
            }
        }
        _ => bail!("split requires strings"),
    }
}

fn rt_join(v: &Value, sep: &Value) -> Result<Value> {
    match (v, sep) {
        (Value::Arr(a), Value::Str(s)) => {
            let mut result = String::new();
            for (i, item) in a.iter().enumerate() {
                if i > 0 { result.push_str(s.as_str()); }
                match item {
                    Value::Str(sv) => result.push_str(sv.as_str()),
                    Value::Null => {},
                    Value::Num(n, _) => result.push_str(&crate::value::format_jq_number(*n)),
                    Value::True => result.push_str("true"),
                    Value::False => result.push_str("false"),
                    _ => {
                        // jq errors when trying to add string to object/array
                        let partial = Value::from_string(result);
                        bail!("{} and {} cannot be added", errdesc(&partial), errdesc(item));
                    }
                }
            }
            Ok(Value::from_string(result))
        }
        _ => bail!("join requires array and string"),
    }
}

fn rt_str_index(v: &Value, target: &Value, is_rindex: bool) -> Result<Value> {
    match (v, target) {
        (Value::Str(s), Value::Str(t)) => {
            if t.is_empty() {
                return Ok(Value::Null);
            }
            let chars: Vec<char> = s.chars().collect();
            let tchars: Vec<char> = t.chars().collect();
            if is_rindex {
                for i in (0..chars.len()).rev() {
                    if i + tchars.len() <= chars.len() && chars[i..i+tchars.len()] == tchars[..] {
                        return Ok(Value::Num(i as f64, None));
                    }
                }
                Ok(Value::Null)
            } else {
                for i in 0..chars.len() {
                    if i + tchars.len() <= chars.len() && chars[i..i+tchars.len()] == tchars[..] {
                        return Ok(Value::Num(i as f64, None));
                    }
                }
                Ok(Value::Null)
            }
        }
        (Value::Arr(a), _) => {
            if is_rindex {
                for (i, item) in a.iter().enumerate().rev() {
                    if values_equal(item, target) {
                        return Ok(Value::Num(i as f64, None));
                    }
                }
                Ok(Value::Null)
            } else {
                for (i, item) in a.iter().enumerate() {
                    if values_equal(item, target) {
                        return Ok(Value::Num(i as f64, None));
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
            // Use character indices, not byte indices
            let chars: Vec<char> = s.chars().collect();
            let tchars: Vec<char> = t.chars().collect();
            let mut indices = Vec::new();
            if tchars.is_empty() {
                return Ok(Value::Arr(Rc::new(indices)));
            }
            for i in 0..chars.len() {
                if i + tchars.len() <= chars.len() && chars[i..i+tchars.len()] == tchars[..] {
                    indices.push(Value::Num(i as f64, None));
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
                        indices.push(Value::Num(i as f64, None));
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
                    indices.push(Value::Num(i as f64, None));
                }
            }
            Ok(Value::Arr(Rc::new(indices)))
        }
        _ => bail!("indices requires string or array"),
    }
}

fn rt_pow(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Num(x, _), Value::Num(y, _)) => Ok(Value::Num(x.powf(*y), None)),
        _ => bail!("pow requires numbers"),
    }
}

pub fn rt_getpath(v: &Value, path: &Value) -> Result<Value> {
    match path {
        Value::Arr(p) => {
            let mut current = v.clone();
            for key in p.iter() {
                match (&current, key) {
                    (Value::Obj(o), Value::Str(k)) => {
                        current = o.get(k.as_str()).cloned().unwrap_or(Value::Null);
                    }
                    (Value::Arr(a), Value::Num(n, _)) => {
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

pub fn rt_setpath(v: &Value, path: &Value, val: &Value) -> Result<Value> {
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
                (Value::Arr(a), Value::Num(n, _)) => {
                    if n.is_nan() { bail!("Cannot set array element at NaN index"); }
                    let idx = *n as i64;
                    let actual = if idx < 0 { (a.len() as i64 + idx) as usize } else { idx as usize };
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
                    let mut obj = IndexMap::new();
                    obj.insert(k.as_ref().clone(), new_inner);
                    Ok(Value::Obj(Rc::new(obj)))
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
                (Value::Obj(_), Value::Num(n, _)) => {
                    bail!("Cannot index object with number ({})", crate::value::format_jq_number(*n));
                }
                (Value::Arr(_), Value::Str(k)) => {
                    bail!("Cannot index array with string (\"{}\")", k);
                }
                // Slice assignment: path element is {start: N, end: N}
                (_, Value::Obj(slice_spec)) if slice_spec.contains_key("start") && slice_spec.contains_key("end") => {
                    let start = match slice_spec.get("start") { Some(Value::Num(n, _)) => *n as i64, _ => 0 };
                    let end = match slice_spec.get("end") { Some(Value::Num(n, _)) => *n as i64, _ => 0 };
                    match v {
                        Value::Arr(a) => {
                            let len = a.len() as i64;
                            let si = start.max(0).min(len) as usize;
                            let ei = end.max(0).min(len) as usize;
                            let ei = ei.max(si);
                            let new_val = rt_setpath(&Value::Null, &rest, val)?;
                            let replacement = match &new_val {
                                Value::Arr(r) => r.as_ref().clone(),
                                _ => vec![new_val],
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
                                _ => vec![new_val],
                            };
                            Ok(Value::Arr(Rc::new(replacement)))
                        }
                        _ => bail!("Cannot set path"),
                    }
                }
                (_, Value::Arr(_)) => {
                    bail!("Cannot update field at array index of array");
                }
                (Value::Num(_, _), Value::Num(n, _)) => {
                    bail!("Cannot index number with number ({})", crate::value::format_jq_number(*n));
                }
                (Value::Num(_, _), Value::Str(k)) => {
                    bail!("Cannot index number with string (\"{}\")", k);
                }
                _ => bail!("Cannot set path"),
            }
        }
        _ => bail!("setpath requires array path"),
    }
}

pub fn rt_delpaths(v: &Value, paths: &Value) -> Result<Value> {
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
        _ => bail!("Paths must be specified as an array"),
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
                (Value::Arr(a), Value::Num(n, _)) => {
                    let ni = *n as i64;
                    let idx = if ni < 0 { (a.len() as i64 + ni) } else { ni };
                    if idx >= 0 && (idx as usize) < a.len() {
                        let mut new_arr = (**a).clone();
                        new_arr.remove(idx as usize);
                        Ok(Value::Arr(Rc::new(new_arr)))
                    } else {
                        Ok(v.clone())
                    }
                }
                _ => Ok(v.clone()),
            }
        }
        Value::Arr(p) => {
            let key = &p[0];
            let rest = Value::Arr(Rc::new(p[1..].to_vec()));
            match (v, key) {
                (Value::Obj(o), Value::Str(k)) => {
                    if let Some(inner) = o.get(k.as_str()) {
                        let new_inner = delete_path(inner, &rest)?;
                        let mut new_obj = (**o).clone();
                        new_obj.insert(k.as_ref().clone(), new_inner);
                        Ok(Value::Obj(Rc::new(new_obj)))
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
                    result.insert("offset".to_string(), Value::Num(m.start() as f64, None));
                    result.insert("length".to_string(), Value::Num(m.len() as f64, None));
                    result.insert("string".to_string(), Value::from_str(m.as_str()));
                    // Add captures
                    let mut captures = Vec::new();
                    if let Some(caps) = regex.captures(s) {
                        for i in 1..caps.len() {
                            if let Some(cap) = caps.get(i) {
                                let mut c = IndexMap::new();
                                c.insert("offset".to_string(), Value::Num(cap.start() as f64, None));
                                c.insert("length".to_string(), Value::Num(cap.len() as f64, None));
                                c.insert("string".to_string(), Value::from_str(cap.as_str()));
                                c.insert("name".to_string(), Value::Null);
                                captures.push(Value::Obj(Rc::new(c)));
                            } else {
                                let mut c = IndexMap::new();
                                c.insert("offset".to_string(), Value::Num(-1.0, None));
                                c.insert("length".to_string(), Value::Num(0.0, None));
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
        Value::Num(n, _) => {
            let secs = *n as i64;
            Ok(libc_gmtime(secs))
        }
        _ => bail!("gmtime requires number"),
    }
}

fn libc_gmtime(secs: i64) -> Value {
    use libc::{gmtime_r, time_t, tm};
    let t = secs as time_t;
    let mut result: tm = unsafe { std::mem::zeroed() };
    unsafe { gmtime_r(&t, &mut result) };
    // jq format: [year+1900, month(0-based), mday, hour, min, sec, wday, yday]
    // Wait - jq actually uses [tm_year+1900, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday]
    // But test expects [2015,2,5,23,51,47,4,63] for epoch 1425599507
    // That matches: year=2015, mon=2(march, 0-indexed), mday=5, hour=23, min=51, sec=47, wday=4(thu), yday=63
    Value::Arr(Rc::new(vec![
        Value::Num((result.tm_year + 1900) as f64, None),
        Value::Num(result.tm_mon as f64, None),
        Value::Num(result.tm_mday as f64, None),
        Value::Num(result.tm_hour as f64, None),
        Value::Num(result.tm_min as f64, None),
        Value::Num(result.tm_sec as f64, None),
        Value::Num(result.tm_wday as f64, None),
        Value::Num(result.tm_yday as f64, None),
    ]))
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
    t.tm_year = get(0) as i32 - 1900;
    t.tm_mon = get(1) as i32;
    t.tm_mday = if a.len() > 2 { get(2) as i32 } else { 1 };
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
            Ok(Value::Num(result as f64, None))
        }
        Value::Arr(a) if !a.is_empty() => {
            if let Value::Str(_) = &a[0] {
                bail!("mktime requires parsed datetime inputs");
            }
            bail!("mktime requires array of time components");
        }
        _ => bail!("mktime requires parsed datetime inputs"),
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
            let s_c = CString::new(s.as_ref().as_str()).unwrap_or_default();
            let f_c = CString::new(f.as_ref().as_str()).unwrap_or_default();
            let mut t: libc::tm = unsafe { std::mem::zeroed() };
            unsafe { libc::strptime(s_c.as_ptr(), f_c.as_ptr(), &mut t) };
            // Compute yday and wday using mktime
            let mut t2 = t;
            unsafe { libc::timegm(&mut t2) };
            t.tm_wday = t2.tm_wday;
            t.tm_yday = t2.tm_yday;
            Ok(Value::Arr(Rc::new(vec![
                Value::Num((t.tm_year + 1900) as f64, None),
                Value::Num(t.tm_mon as f64, None),
                Value::Num(t.tm_mday as f64, None),
                Value::Num(t.tm_hour as f64, None),
                Value::Num(t.tm_min as f64, None),
                Value::Num(t.tm_sec as f64, None),
                Value::Num(t.tm_wday as f64, None),
                Value::Num(t.tm_yday as f64, None),
            ])))
        }
        _ => bail!("strptime requires string and format"),
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
// Environment
// -----------------------------------------------------------------------

fn rt_env() -> Value {
    let mut env = IndexMap::new();
    for (k, v) in std::env::vars() {
        env.insert(k, Value::from_string(v));
    }
    Value::Obj(Rc::new(env))
}

pub fn rt_builtins() -> Value {
    let builtins = vec![
        "length/0", "utf8bytelength/0", "keys/0", "values/0",
        "has/1", "has/2", "in/1",
        "contains/1", "inside/1",
        "startswith/1", "endswith/1", "ltrimstr/1", "rtrimstr/1",
        "split/1", "split/2", "join/1",
        "test/1", "test/2", "match/1", "match/2",
        "capture/1", "capture/2", "scan/1", "scan/2",
        "sub/2", "sub/3", "gsub/2", "gsub/3",
        "tostring/0", "tonumber/0", "type/0",
        "empty/0", "error/0", "error/1",
        "null/0", "true/0", "false/0", "not/0",
        "map/1", "select/1",
        "add/0", "any/0", "any/1", "any/2", "all/0", "all/1", "all/2",
        "flatten/0", "flatten/1",
        "range/1", "range/2", "range/3",
        "floor/0", "ceil/0", "round/0",
        "sqrt/0", "fabs/0", "abs/0",
        "pow/2", "log/0", "log2/0", "log10/0",
        "exp/0", "exp2/0", "exp10/0",
        "sin/0", "cos/0", "tan/0", "asin/0", "acos/0", "atan/0",
        "atan/2", "atan2/2",
        "min/0", "max/0", "sort/0",
        "sort_by/1", "group_by/1", "unique/0", "unique_by/1",
        "reverse/0", "keys_unsorted/0",
        "to_entries/0", "from_entries/0", "with_entries/1",
        "paths/0", "paths/1", "path/1",
        "getpath/1", "setpath/2", "delpaths/1",
        "tojson/0", "fromjson/0",
        "ascii_downcase/0", "ascii_upcase/0",
        "explode/0", "implode/0",
        "indices/1", "index/1", "rindex/1",
        "ltrim/0", "rtrim/0", "trim/0",
        "nan/0", "infinite/0", "isinfinite/0", "isnan/0", "isnormal/0", "isfinite/0",
        "env/0", "debug/0", "debug/1", "stderr/0",
        "input/0", "inputs/0",
        "limit/2", "first/0", "first/1", "last/0", "last/1",
        "nth/1", "nth/2",
        "while/2", "until/2", "repeat/1",
        "recurse/0", "recurse/1", "recurse/2", "recurse_down/0",
        "transpose/0",
        "ascii/0", "now/0", "gmtime/0", "mktime/0",
        "strftime/1", "strptime/1",
        "todate/0", "fromdate/0", "dateadd/2", "datesub/2",
        "modulemeta/0", "builtins/0",
        "leaf_paths/0", "isempty/1",
        "del/1", "pick/1",
        "min_by/1", "max_by/1", "map_values/1",
        "label/1",
        "input_line_number/0",
        "bsearch/1",
        "walk/1",
        "significand/0", "exponent/0", "logb/0",
        "nearbyint/0", "trunc/0", "rint/0",
        "j0/0", "j1/0", "cbrt/0",
        "limit/2", "first/1", "last/1",
        "INDEX/1", "INDEX/2", "IN/1", "IN/2",
        "JOIN/2",
        "skip/2",
        "getpath/1", "setpath/2", "delpaths/1",
        "tojson/0", "fromjson/0",
        "have_decnum/0", "have_sql/0", "have_bom/0",
        "@base64/0", "@base64d/0", "@uri/0", "@csv/0", "@tsv/0",
        "@html/0", "@json/0", "@text/0", "@sh/0",
        "ascii_downcase/0", "ascii_upcase/0",
        "strflocaltime/1",
        "toboolean/0",
    ];
    let arr: Vec<Value> = builtins.iter()
        .map(|&name| Value::from_str(name))
        .collect();
    Value::Arr(Rc::new(arr))
}
