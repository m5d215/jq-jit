//! Module system support for jq-jit.

use std::rc::Rc;
use anyhow::{Result, bail};
use crate::value::{Value, KeyStr, new_objmap};

/// Resolve a module name to a file path, searching lib_dirs.
fn resolve_module(name: &str, lib_dirs: &[String]) -> Result<String> {
    for dir in lib_dirs {
        // Try name/name.jq
        let path1 = format!("{}/{}/{}.jq", dir, name, name);
        if std::path::Path::new(&path1).exists() {
            return Ok(path1);
        }
        // Try name.jq
        let path2 = format!("{}/{}.jq", dir, name);
        if std::path::Path::new(&path2).exists() {
            return Ok(path2);
        }
    }
    bail!("Cannot find module '{}'", name)
}

/// Get module metadata for the `modulemeta` builtin.
pub fn get_modulemeta(input: &Value, lib_dirs: &[String]) -> Result<Value> {
    let name = match input {
        Value::Str(s) => s.as_str().to_string(),
        _ => bail!("modulemeta requires string input, got {}", input.type_name()),
    };

    let file_path = resolve_module(&name, lib_dirs)?;
    let content = std::fs::read_to_string(&file_path)
        .map_err(|e| anyhow::anyhow!("Cannot read module '{}': {}", file_path, e))?;

    parse_module_metadata(&content)
}

/// Parse a module file to extract metadata, deps, and defs.
fn parse_module_metadata(content: &str) -> Result<Value> {
    let mut metadata = new_objmap();
    let mut deps = Vec::new();
    let mut defs = Vec::new();

    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    // Simple line-based parser for module metadata extraction
    let full = content.trim();

    // Parse module statement
    if full.starts_with("module ") || full.starts_with("module{") {
        if let Some(semi_pos) = full.find(';') {
            let module_stmt = &full[7..semi_pos].trim();
            // Parse the metadata object
            if let Ok(meta_val) = parse_simple_object(module_stmt) {
                if let Value::Obj(obj) = meta_val {
                    for (k, v) in obj.iter() {
                        metadata.insert(k.clone(), v.clone());
                    }
                }
            }
            i = semi_pos + 1;
        }
    }

    // Parse the rest for imports and defs
    let rest = if i > 0 { &full[i..] } else { full };
    let rest = rest.trim();

    // Tokenize simply to find import/include/def statements
    let mut pos = 0;
    let chars: Vec<char> = rest.chars().collect();

    while pos < chars.len() {
        // Skip whitespace
        while pos < chars.len() && chars[pos].is_whitespace() { pos += 1; }
        if pos >= chars.len() { break; }

        // Read a word
        let word_start = pos;
        while pos < chars.len() && (chars[pos].is_alphanumeric() || chars[pos] == '_') { pos += 1; }
        let word: String = chars[word_start..pos].iter().collect();

        match word.as_str() {
            "import" => {
                // import "path" as alias {metadata};
                let dep = parse_import_dep(&chars, &mut pos);
                if let Some(d) = dep {
                    deps.push(d);
                }
            }
            "include" => {
                let dep = parse_include_dep(&chars, &mut pos);
                if let Some(d) = dep {
                    deps.push(d);
                }
            }
            "def" => {
                // def name(params): ...;
                let def_info = parse_def_info(&chars, &mut pos);
                if let Some(d) = def_info {
                    defs.push(d);
                }
            }
            _ => {
                // Skip to next semicolon or end
                while pos < chars.len() && chars[pos] != ';' { pos += 1; }
                if pos < chars.len() { pos += 1; }
            }
        }
    }

    let mut result = metadata;
    result.insert("deps".into(), Value::Arr(Rc::new(deps)));
    result.insert("defs".into(), Value::Arr(Rc::new(defs)));

    Ok(Value::Obj(Rc::new(result)))
}

fn parse_simple_object(s: &str) -> Result<Value> {
    let s = s.trim();
    if !s.starts_with('{') || !s.ends_with('}') {
        bail!("not an object");
    }
    let inner = &s[1..s.len()-1];
    let mut map = new_objmap();

    for part in split_top_level(inner, ',') {
        let part = part.trim();
        if part.is_empty() { continue; }
        if let Some(colon_pos) = part.find(':') {
            let key = part[..colon_pos].trim().trim_matches('"');
            let val_str = part[colon_pos+1..].trim();
            let val = parse_simple_value(val_str);
            map.insert(KeyStr::from(key), val);
        }
    }

    Ok(Value::Obj(Rc::new(map)))
}

fn parse_simple_value(s: &str) -> Value {
    let s = s.trim();
    match s {
        "null" => Value::Null,
        "true" => Value::True,
        "false" => Value::False,
        _ if s.starts_with('"') && s.ends_with('"') => {
            Value::from_str(&s[1..s.len()-1])
        }
        _ if s.parse::<f64>().is_ok() => {
            Value::Num(s.parse().unwrap(), None)
        }
        _ => Value::Null,
    }
}

fn split_top_level(s: &str, delim: char) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    let mut in_string = false;

    for ch in s.chars() {
        if in_string {
            current.push(ch);
            if ch == '"' { in_string = false; }
            continue;
        }
        match ch {
            '"' => { in_string = true; current.push(ch); }
            '{' | '[' => { depth += 1; current.push(ch); }
            '}' | ']' => { depth -= 1; current.push(ch); }
            c if c == delim && depth == 0 => {
                result.push(current.clone());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

fn parse_import_dep(chars: &[char], pos: &mut usize) -> Option<Value> {
    // Skip whitespace
    while *pos < chars.len() && chars[*pos].is_whitespace() { *pos += 1; }

    // Read quoted path
    if *pos >= chars.len() || chars[*pos] != '"' { return skip_to_semi(chars, pos); }
    *pos += 1;
    let mut path = String::new();
    while *pos < chars.len() && chars[*pos] != '"' {
        path.push(chars[*pos]);
        *pos += 1;
    }
    if *pos < chars.len() { *pos += 1; } // closing quote

    // Skip whitespace
    while *pos < chars.len() && chars[*pos].is_whitespace() { *pos += 1; }

    // Read "as"
    let mut word = String::new();
    while *pos < chars.len() && chars[*pos].is_alphabetic() {
        word.push(chars[*pos]);
        *pos += 1;
    }
    if word != "as" { return skip_to_semi(chars, pos); }

    // Skip whitespace
    while *pos < chars.len() && chars[*pos].is_whitespace() { *pos += 1; }

    // Read alias (could be $var or ident)
    let is_data = *pos < chars.len() && chars[*pos] == '$';
    if is_data { *pos += 1; }

    let mut alias = String::new();
    while *pos < chars.len() && (chars[*pos].is_alphanumeric() || chars[*pos] == '_') {
        alias.push(chars[*pos]);
        *pos += 1;
    }

    // Skip whitespace
    while *pos < chars.len() && chars[*pos].is_whitespace() { *pos += 1; }

    // Check for metadata {search:"..."}
    let mut dep_map = new_objmap();
    if *pos < chars.len() && chars[*pos] == '{' {
        *pos += 1;
        let mut meta_str = String::new();
        let mut depth = 1;
        while *pos < chars.len() && depth > 0 {
            if chars[*pos] == '{' { depth += 1; }
            if chars[*pos] == '}' { depth -= 1; if depth == 0 { *pos += 1; break; } }
            meta_str.push(chars[*pos]);
            *pos += 1;
        }
        // Parse metadata
        for part in meta_str.split(',') {
            let part = part.trim();
            if let Some(colon_pos) = part.find(':') {
                let key = part[..colon_pos].trim().trim_matches('"');
                let val = part[colon_pos+1..].trim().trim_matches('"');
                dep_map.insert(KeyStr::from(key), Value::from_str(val));
            }
        }
    }

    // Skip to semicolon
    while *pos < chars.len() && chars[*pos] != ';' { *pos += 1; }
    if *pos < chars.len() { *pos += 1; }

    dep_map.insert("as".into(), Value::from_str(&alias));
    dep_map.insert("is_data".into(), Value::from_bool(is_data));
    dep_map.insert("relpath".into(), Value::from_str(&path));

    Some(Value::Obj(Rc::new(dep_map)))
}

fn parse_include_dep(chars: &[char], pos: &mut usize) -> Option<Value> {
    while *pos < chars.len() && chars[*pos].is_whitespace() { *pos += 1; }

    if *pos >= chars.len() || chars[*pos] != '"' { return skip_to_semi(chars, pos); }
    *pos += 1;
    let mut path = String::new();
    while *pos < chars.len() && chars[*pos] != '"' {
        path.push(chars[*pos]);
        *pos += 1;
    }
    if *pos < chars.len() { *pos += 1; }

    while *pos < chars.len() && chars[*pos] != ';' { *pos += 1; }
    if *pos < chars.len() { *pos += 1; }

    let mut dep_map = new_objmap();
    dep_map.insert("is_data".into(), Value::False);
    dep_map.insert("relpath".into(), Value::from_str(&path));
    Some(Value::Obj(Rc::new(dep_map)))
}

fn parse_def_info(chars: &[char], pos: &mut usize) -> Option<Value> {
    while *pos < chars.len() && chars[*pos].is_whitespace() { *pos += 1; }

    let mut name = String::new();
    while *pos < chars.len() && (chars[*pos].is_alphanumeric() || chars[*pos] == '_') {
        name.push(chars[*pos]);
        *pos += 1;
    }

    while *pos < chars.len() && chars[*pos].is_whitespace() { *pos += 1; }

    // Count parameters
    let mut nargs = 0;
    if *pos < chars.len() && chars[*pos] == '(' {
        *pos += 1;
        let mut depth = 1;
        while *pos < chars.len() && depth > 0 {
            if chars[*pos] == '(' { depth += 1; }
            if chars[*pos] == ')' { depth -= 1; }
            if chars[*pos] == ';' && depth == 1 { nargs += 1; }
            *pos += 1;
        }
        nargs += 1; // last param
    }

    // Skip to end of def (matching semicolons with depth)
    // Skip : first
    while *pos < chars.len() && chars[*pos] != ':' && chars[*pos] != ';' { *pos += 1; }
    if *pos < chars.len() && chars[*pos] == ':' {
        *pos += 1;
        let mut depth = 0;
        loop {
            if *pos >= chars.len() { break; }
            match chars[*pos] {
                ';' if depth == 0 => { *pos += 1; break; }
                '{' | '[' | '(' => depth += 1,
                '}' | ']' | ')' => depth -= 1,
                _ => {}
            }
            *pos += 1;
        }
    } else if *pos < chars.len() {
        *pos += 1;
    }

    if name.is_empty() { return None; }
    Some(Value::from_str(&format!("{}/{}", name, nargs)))
}

fn skip_to_semi(chars: &[char], pos: &mut usize) -> Option<Value> {
    while *pos < chars.len() && chars[*pos] != ';' { *pos += 1; }
    if *pos < chars.len() { *pos += 1; }
    None
}
