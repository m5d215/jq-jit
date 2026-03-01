//! Shared input reading and streaming JSON parsing.
//!
//! Used by both the JIT CLI (`jq-jit.rs`) and AOT runtime (`aot.rs`).
//! Consolidates duplicated `read_inputs`, `parse_json_values`, and
//! `serde_to_value` functions into a single module.

use std::collections::BTreeMap;
use std::io::{self, BufReader, Read};
use std::rc::Rc;

use crate::value::Value;

// ---------------------------------------------------------------------------
// serde_json::Value → Value conversion
// ---------------------------------------------------------------------------

/// Convert a `serde_json::Value` to our internal `Value` type.
///
/// This is the canonical implementation — both CLI and AOT runtime should use
/// this instead of maintaining their own copies.
pub fn serde_to_value(v: serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(b),
        serde_json::Value::Number(n) => Value::Num(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => Value::Str(Rc::new(s)),
        serde_json::Value::Array(a) => {
            Value::Arr(Rc::new(a.into_iter().map(serde_to_value).collect()))
        }
        serde_json::Value::Object(o) => {
            let map: BTreeMap<String, Value> =
                o.into_iter().map(|(k, v)| (k, serde_to_value(v))).collect();
            Value::Obj(Rc::new(map))
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming JSON parsing
// ---------------------------------------------------------------------------

/// Parse JSON values from a reader using serde_json's streaming deserializer.
///
/// Returns an iterator that yields one `Value` at a time.
/// Handles single JSON values, arrays, objects, NDJSON, and concatenated JSON.
///
/// EOF is treated as a clean termination, not an error.
pub fn stream_json_values<R: Read + 'static>(
    reader: R,
) -> Box<dyn Iterator<Item = Result<Value, String>>> {
    let deserializer = serde_json::Deserializer::from_reader(reader);
    Box::new(
        deserializer
            .into_iter::<serde_json::Value>()
            .map(|result| match result {
                Ok(v) => Ok(serde_to_value(v)),
                Err(e) if e.is_eof() => Err(String::new()), // sentinel for EOF
                Err(e) => Err(format!("parse error: {}", e)),
            })
            .filter(|r| !matches!(r, Err(s) if s.is_empty())), // strip EOF sentinels
    )
}

// ---------------------------------------------------------------------------
// Input source opening
// ---------------------------------------------------------------------------

/// Open input sources for reading.
///
/// Returns a boxed reader that can be passed to [`stream_json_values`].
///
/// - No files: reads from stdin
/// - One file: reads from that file
/// - Multiple files: chains readers in order
pub fn open_input(files: &[String]) -> Result<Box<dyn Read>, String> {
    if files.is_empty() {
        Ok(Box::new(BufReader::new(io::stdin())))
    } else if files.len() == 1 {
        let f = std::fs::File::open(&files[0])
            .map_err(|e| format!("{}: {}", files[0], e))?;
        Ok(Box::new(BufReader::new(f)))
    } else {
        // Multiple files: chain readers
        let mut chained: Box<dyn Read> = Box::new(io::empty());
        for path in files {
            let f = std::fs::File::open(path)
                .map_err(|e| format!("{}: {}", path, e))?;
            chained = Box::new(chained.chain(BufReader::new(f)));
        }
        Ok(chained)
    }
}

// ---------------------------------------------------------------------------
// Raw text reading (for -R mode)
// ---------------------------------------------------------------------------

/// Read raw text lines from files or stdin.
///
/// Used for `-R` (raw input) mode where input is treated as text, not JSON.
/// Each line becomes a separate string (without the trailing newline).
pub fn read_raw_lines(files: &[String]) -> Result<Vec<String>, String> {
    if files.is_empty() {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("reading stdin: {}", e))?;
        Ok(buf.lines().map(|l| l.to_string()).collect())
    } else {
        let mut lines = Vec::new();
        for path in files {
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("{}: {}", path, e))?;
            lines.extend(content.lines().map(|l| l.to_string()));
        }
        Ok(lines)
    }
}

/// Read entire raw input as a single string from files or stdin.
///
/// Used for `-Rs` (raw input + slurp) mode where the entire input
/// (including newlines) becomes one string value.
pub fn read_raw_string(files: &[String]) -> Result<String, String> {
    if files.is_empty() {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("reading stdin: {}", e))?;
        Ok(buf)
    } else {
        let mut combined = String::new();
        for path in files {
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("{}: {}", path, e))?;
            combined.push_str(&content);
        }
        Ok(combined)
    }
}
