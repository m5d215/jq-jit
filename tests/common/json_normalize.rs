//! Value-level JSON normalisation for output comparison.
//!
//! `normalize` parses each non-empty line of `output` as a JSON value,
//! applies [`normalize_value`] (folds integer-valued floats to integers),
//! then re-serialises with [`serialize_sorted`] (object keys lexicographically
//! sorted, no whitespace). Two outputs with identical semantic values
//! compare equal regardless of textual formatting differences between
//! `jq-jit` and reference `jq`.

use serde_json::Value;

/// Parse one JSON value per non-empty line, normalise, and re-serialise.
/// Returns `Err` if any line is not valid JSON.
pub fn normalize(output: &str) -> Result<String, String> {
    let mut normalized_lines = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val: Value = serde_json::from_str(trimmed)
            .map_err(|e| format!("non-JSON line `{}`: {}", trimmed, e))?;
        normalized_lines.push(serialize_sorted(&normalize_value(val)));
    }
    Ok(normalized_lines.join("\n"))
}

/// Recursively fold integer-valued `f64` numbers into integers, so jq's
/// `1.0`-style float prints compare equal to jq-jit's `1`-style integer
/// prints. Values outside the f64 mantissa boundary are left as-is.
pub fn normalize_value(val: Value) -> Value {
    match val {
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                if f.is_finite() && f == (f as i64) as f64 && f.abs() < (1i64 << 53) as f64 {
                    return Value::Number(serde_json::Number::from(f as i64));
                }
            }
            Value::Number(n)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(normalize_value).collect()),
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| (k, normalize_value(v)))
                .collect(),
        ),
        other => other,
    }
}

/// Serialise `val` compactly with object keys lexicographically sorted.
pub fn serialize_sorted(val: &Value) -> String {
    match val {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap(),
        Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(serialize_sorted).collect();
            format!("[{}]", items.join(","))
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            let items: Vec<String> = entries
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap(),
                        serialize_sorted(v)
                    )
                })
                .collect();
            format!("{{{}}}", items.join(","))
        }
    }
}
