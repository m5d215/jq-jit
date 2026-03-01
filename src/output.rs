//! Shared output formatting for JIT CLI and AOT runtime.
//!
//! Extracted from duplicated code in `src/bin/jq-jit.rs` and `src/aot.rs`.

use crate::value::{format_jq_number, value_to_json, Value};

// ---------------------------------------------------------------------------
// Output options
// ---------------------------------------------------------------------------

/// Formatting options shared between the JIT CLI and AOT runtime.
pub struct OutputOptions {
    pub raw_output: bool,
    pub compact_output: bool,
    pub tab_indent: bool,
    pub indent_n: usize,
}

impl Default for OutputOptions {
    fn default() -> Self {
        Self {
            raw_output: false,
            compact_output: false,
            tab_indent: false,
            indent_n: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// JSON string escaping
// ---------------------------------------------------------------------------

/// Escape a string for JSON output (wraps in double quotes).
pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\x20' => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ---------------------------------------------------------------------------
// Pretty-printing
// ---------------------------------------------------------------------------

/// Pretty-print a Value with the given indent unit and current depth.
pub fn pretty_print(v: &Value, indent_unit: &str, depth: usize) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Value::Num(n) => format_jq_number(*n),
        Value::Str(s) => json_escape(s),
        Value::Arr(a) => {
            if a.is_empty() {
                return "[]".to_string();
            }
            let inner_indent = indent_unit.repeat(depth + 1);
            let outer_indent = indent_unit.repeat(depth);
            let mut out = String::from("[\n");
            for (i, item) in a.iter().enumerate() {
                out.push_str(&inner_indent);
                out.push_str(&pretty_print(item, indent_unit, depth + 1));
                if i + 1 < a.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&outer_indent);
            out.push(']');
            out
        }
        Value::Obj(o) => {
            if o.is_empty() {
                return "{}".to_string();
            }
            let inner_indent = indent_unit.repeat(depth + 1);
            let outer_indent = indent_unit.repeat(depth);
            let mut out = String::from("{\n");
            for (i, (k, val)) in o.iter().enumerate() {
                out.push_str(&inner_indent);
                out.push_str(&json_escape(k));
                out.push_str(": ");
                out.push_str(&pretty_print(val, indent_unit, depth + 1));
                if i + 1 < o.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&outer_indent);
            out.push('}');
            out
        }
        Value::Error(e) => json_escape(e),
    }
}

// ---------------------------------------------------------------------------
// High-level formatting
// ---------------------------------------------------------------------------

/// Format a Value as a string according to the given options.
pub fn format_value(v: &Value, opts: &OutputOptions) -> String {
    if opts.compact_output {
        value_to_json(v)
    } else if opts.tab_indent {
        pretty_print(v, "\t", 0)
    } else {
        let indent_str = " ".repeat(opts.indent_n);
        pretty_print(v, &indent_str, 0)
    }
}

/// Print a Value to stdout according to the given options.
pub fn output_value(v: &Value, opts: &OutputOptions) {
    if opts.raw_output {
        if let Value::Str(s) = v {
            println!("{}", s);
            return;
        }
    }
    println!("{}", format_value(v, opts));
}
