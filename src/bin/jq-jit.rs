//! jq-jit: A JIT-compiled jq implementation.
//!
//! Usage: jq-jit [OPTIONS] <FILTER> [FILE...]

use std::io::{self, BufRead, Read, Write};
use std::process;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use jq_jit::value::{Value, json_to_value, json_stream, json_stream_offsets, json_stream_raw, json_stream_project, json_object_get_num, json_object_get_two_nums, json_object_get_field_raw, json_object_get_fields_raw, json_value_length, json_object_keys_to_buf, json_object_keys_unsorted_to_buf, json_object_has_key, json_type_byte, json_object_del_field, json_each_value_raw, json_to_entries_raw, is_json_compact, push_json_compact_raw, value_to_json_precise, value_to_json_pretty_ext, push_compact_line, push_pretty_line, push_jq_number_bytes, write_value_compact_ext, write_value_compact_line, write_value_pretty_line, pool_value};
use jq_jit::interpreter::Filter;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut filter_str = None;
    let mut files: Vec<String> = Vec::new();
    let mut compact = false;
    let mut raw_output = false;
    let mut raw_input = false;
    let mut null_input = false;
    let mut slurp = false;
    let mut join_output = false;
    let mut tab = false;
    let mut indent_n = 2usize;
    let mut sort_keys = false;
    let mut exit_status = false;
    let mut arg_vars: Vec<(String, Value)> = Vec::new();
    let mut argjson_vars: Vec<(String, Value)> = Vec::new();
    let mut lib_dirs: Vec<String> = Vec::new();

    // Expand args: split combined short flags like -ncr into ["-n", "-c", "-r"]
    let mut expanded_args: Vec<String> = Vec::new();
    for arg in &args[1..] {
        if arg.starts_with('-') && !arg.starts_with("--") && arg.len() > 2
            && arg[1..].chars().all(|c| c.is_ascii_alphabetic()) {
            for ch in arg[1..].chars() {
                expanded_args.push(format!("-{}", ch));
            }
        } else {
            expanded_args.push(arg.clone());
        }
    }

    let mut i = 0;
    while i < expanded_args.len() {
        let arg = &expanded_args[i];
        match arg.as_str() {
            "-c" | "--compact-output" => compact = true,
            "-r" | "--raw-output" => raw_output = true,
            "-R" | "--raw-input" => raw_input = true,
            "-n" | "--null-input" => null_input = true,
            "-s" | "--slurp" => slurp = true,
            "-j" | "--join-output" => { join_output = true; raw_output = true; }
            "-S" | "--sort-keys" => sort_keys = true,
            "-e" | "--exit-status" => exit_status = true,
            "--tab" => tab = true,
            "--indent" => {
                i += 1;
                if i < expanded_args.len() {
                    indent_n = expanded_args[i].parse().unwrap_or(2);
                }
            }
            "-f" | "--from-file" => {
                i += 1;
                if i < expanded_args.len() {
                    let path = std::path::Path::new(&expanded_args[i]);
                    let content = match std::fs::read_to_string(path) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("jq: Could not open file {}: {}", expanded_args[i], e);
                            process::exit(2);
                        }
                    };
                    filter_str = Some(content.trim_end().to_string());
                    // Add the filter file's directory to lib search path for import resolution
                    if let Some(parent) = path.canonicalize().ok().and_then(|p| p.parent().map(|d| d.to_string_lossy().into_owned())) {
                        if !lib_dirs.contains(&parent) {
                            lib_dirs.push(parent);
                        }
                    }
                }
            }
            "--arg" => {
                if i + 2 < expanded_args.len() {
                    let name = expanded_args[i + 1].clone();
                    let val = Value::from_str(&expanded_args[i + 2]);
                    arg_vars.push((name, val));
                    i += 2;
                }
            }
            "--argjson" => {
                if i + 2 < expanded_args.len() {
                    let name = expanded_args[i + 1].clone();
                    match json_to_value(&expanded_args[i + 2]) {
                        Ok(val) => argjson_vars.push((name, val)),
                        Err(e) => {
                            eprintln!("jq: Invalid JSON text passed to --argjson: {}", e);
                            process::exit(2);
                        }
                    }
                    i += 2;
                }
            }
            "-L" => {
                i += 1;
                if i < expanded_args.len() {
                    lib_dirs.push(expanded_args[i].clone());
                }
            }
            "--args" => break,
            "--version" => {
                println!("jq-jit-0.1.0");
                process::exit(0);
            }
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            s if s.starts_with('-') && filter_str.is_some() => {
                eprintln!("jq: Unknown option: {}", s);
                process::exit(2);
            }
            _ => {
                if filter_str.is_none() {
                    filter_str = Some(arg.clone());
                } else {
                    files.push(arg.clone());
                }
            }
        }
        i += 1;
    }

    let filter_str = match filter_str {
        Some(f) => f,
        None => {
            eprintln!("jq - commandline JSON processor");
            eprintln!("Usage: jq-jit [OPTIONS] <FILTER> [FILE...]");
            process::exit(2);
        }
    };

    // Prepend --arg / --argjson bindings to the filter expression
    let filter_str = {
        let mut prefix = String::new();
        for (name, val) in &arg_vars {
            prefix.push_str(&value_to_json_precise(val));
            prefix.push_str(" as $");
            prefix.push_str(name);
            prefix.push_str(" | ");
        }
        for (name, val) in &argjson_vars {
            prefix.push_str(&value_to_json_precise(val));
            prefix.push_str(" as $");
            prefix.push_str(name);
            prefix.push_str(" | ");
        }
        if prefix.is_empty() {
            filter_str
        } else {
            prefix.push_str(&filter_str);
            prefix
        }
    };

    // Skip JIT for null-input: compilation overhead exceeds savings for one-shot filters.
    let use_jit = !null_input;
    let filter = match Filter::with_options(&filter_str, &lib_dirs, use_jit) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("jq: error: {}", e);
            process::exit(3);
        }
    };

    // Projection pushdown: skip parsing values for unneeded fields.
    // Only worthwhile when extracting very few fields from wide objects.
    // Disabled for now — overhead exceeds savings for typical narrow objects.
    let projection_fields: Option<Vec<String>> = None;

    let stdout = io::stdout();
    let mut out = io::BufWriter::with_capacity(65536, stdout.lock());

    let mut any_output_false = false;

    let format_value = |v: &Value| -> String {
        if raw_output {
            if let Value::Str(s) = v {
                return s.to_string();
            }
        }
        if compact {
            value_to_json_precise(v)
        } else if tab {
            value_to_json_pretty_ext(v, 0, 1, true, sort_keys)
        } else {
            value_to_json_pretty_ext(v, 0, indent_n, false, sort_keys)
        }
    };

    let mut had_error = false;
    // Use Vec-based buffering for compact output to avoid per-value write_all overhead
    let use_compact_buf = compact && !raw_output && !sort_keys && !join_output;
    let use_pretty_buf = !compact && !raw_output && !sort_keys && !join_output && !tab;
    let field_access = if (use_compact_buf || use_pretty_buf) && !exit_status {
        filter.detect_field_access()
    } else { None };
    let field_remap = if use_compact_buf && !exit_status && field_access.is_none() {
        filter.detect_field_remap()
    } else { None };
    let field_binop = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_remap.is_none() {
        filter.detect_field_binop()
    } else { None };
    let field_str_concat = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_remap.is_none() && field_binop.is_none() {
        filter.detect_field_str_concat()
    } else { None };
    let select_cmp = if use_compact_buf && !exit_status && field_access.is_none() && field_remap.is_none() && field_binop.is_none() && field_str_concat.is_none() {
        filter.detect_select_field_cmp()
    } else { None };
    let array_field = if use_compact_buf && !exit_status && field_access.is_none() {
        filter.detect_array_field_access()
    } else { None };
    let multi_field = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && array_field.is_none() {
        filter.detect_multi_field_access()
    } else { None };
    let is_length = (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && multi_field.is_none() && select_cmp.is_none() && filter.is_length();
    let is_keys = use_compact_buf && !exit_status && field_access.is_none() && select_cmp.is_none() && !is_length && filter.is_keys();
    let is_keys_unsorted = use_compact_buf && !exit_status && !is_keys && !is_length && filter.is_keys_unsorted();
    let has_field = if (use_compact_buf || use_pretty_buf) && !exit_status && !is_length && !is_keys {
        filter.detect_has_field()
    } else { None };
    let is_type = (use_compact_buf || use_pretty_buf) && !exit_status && !is_length && !is_keys && has_field.is_none() && filter.is_type();
    let del_field = if use_compact_buf && !exit_status && !is_length && !is_keys && !is_type && has_field.is_none() {
        filter.detect_del_field()
    } else { None };
    let is_each = use_compact_buf && !exit_status && !is_length && !is_keys && !is_type && has_field.is_none() && del_field.is_none() && field_access.is_none() && filter.is_each();
    let is_to_entries = use_compact_buf && !exit_status && !is_each && filter.is_to_entries();
    let mut compact_buf: Vec<u8> = if use_compact_buf || use_pretty_buf { Vec::with_capacity(1 << 17) } else { Vec::new() };
    let process_input = |input: &Value, raw_bytes: Option<&[u8]>, out: &mut io::BufWriter<io::StdoutLock>, cbuf: &mut Vec<u8>, any_false: &mut bool, had_error: &mut bool| {
        let result = filter.execute_cb(input, &mut |result| {
            if let Value::Error(e) = result {
                eprintln!("jq: error: {}", e.as_str());
                *had_error = true;
                return Ok(true);
            }
            if exit_status && !result.is_true() {
                *any_false = true;
            }
            if use_compact_buf {
                // Raw passthrough: if result is the unmodified input and bytes are compact,
                // copy original bytes directly instead of re-serializing
                if let Some(raw) = raw_bytes {
                    if std::ptr::eq(result, input) && is_json_compact(raw) {
                        cbuf.extend_from_slice(raw);
                        cbuf.push(b'\n');
                        if cbuf.len() >= 1 << 17 {
                            let _ = out.write_all(cbuf);
                            cbuf.clear();
                        }
                        return Ok(true);
                    }
                }
                push_compact_line(cbuf, result);
                if cbuf.len() >= 1 << 17 {
                    let _ = out.write_all(cbuf);
                    cbuf.clear();
                }
            } else if use_pretty_buf {
                push_pretty_line(cbuf, result, indent_n, tab);
                if cbuf.len() >= 1 << 17 {
                    let _ = out.write_all(cbuf);
                    cbuf.clear();
                }
            } else if join_output {
                if compact && !raw_output {
                    let _ = write_value_compact_ext(out, result, sort_keys);
                } else {
                    let formatted = format_value(result);
                    let _ = write!(out, "{}", formatted);
                }
            } else if compact && !raw_output {
                let _ = write_value_compact_line(out, result, sort_keys);
            } else if !raw_output {
                let _ = write_value_pretty_line(out, result, indent_n, tab, sort_keys);
            } else {
                let formatted = format_value(result);
                let _ = writeln!(out, "{}", formatted);
            }
            Ok(true)
        });
        if let Err(e) = result {
            let msg = format!("{}", e);
            if let Some(jq_msg) = msg.strip_prefix("__jqerror__:") {
                eprintln!("jq: error: {}", jq_msg);
            } else {
                eprintln!("jq: error: {}", msg);
            }
            *had_error = true;
        }
    };

    if null_input {
        process_input(&Value::Null, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
    } else if files.is_empty() {
        // Read from stdin
        let stdin = io::stdin();
        if raw_input {
            let mut lines: Vec<Value> = Vec::new();
            for line in stdin.lock().lines() {
                match line {
                    Ok(l) => {
                        if slurp {
                            lines.push(Value::from_str(&l));
                        } else {
                            process_input(&Value::from_str(&l), None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    }
                    Err(e) => {
                        eprintln!("jq: error reading input: {}", e);
                        process::exit(2);
                    }
                }
            }
            if slurp {
                let arr = Value::Arr(std::rc::Rc::new(lines));
                process_input(&arr, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
            }
        } else {
            let mut input_str = String::new();
            stdin.lock().read_to_string(&mut input_str).unwrap_or(0);

            if slurp {
                // Parse all JSON values and collect into array
                let mut values = Vec::new();
                if let Err(e) = json_stream(&input_str, |v| {
                    values.push(v);
                    Ok(())
                }) {
                    eprintln!("jq: error (at <stdin>:0): {}", e);
                    process::exit(2);
                }
                let arr = Value::Arr(std::rc::Rc::new(values));
                process_input(&arr, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
            } else {
                let input_bytes = input_str.as_bytes();
                let parse_result = if filter.is_empty() {
                    json_stream_raw(&input_str, |_, _| Ok(()))
                } else if filter.is_identity() && use_compact_buf && !exit_status {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if is_json_compact(raw) {
                            compact_buf.extend_from_slice(raw);
                            compact_buf.push(b'\n');
                        } else {
                            push_json_compact_raw(&mut compact_buf, raw);
                            compact_buf.push(b'\n');
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref fa_field) = field_access {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if raw[0] == b'{' {
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, fa_field) {
                                let val_bytes = &raw[vs..ve];
                                let first = val_bytes[0];
                                if first != b'{' && first != b'[' {
                                    compact_buf.extend_from_slice(val_bytes);
                                    compact_buf.push(b'\n');
                                } else if use_compact_buf && is_json_compact(val_bytes) {
                                    compact_buf.extend_from_slice(val_bytes);
                                    compact_buf.push(b'\n');
                                } else {
                                    let v = json_to_value(unsafe { std::str::from_utf8_unchecked(val_bytes) })?;
                                    if use_compact_buf {
                                        push_compact_line(&mut compact_buf, &v);
                                    } else {
                                        push_pretty_line(&mut compact_buf, &v, indent_n, tab);
                                    }
                                }
                            } else {
                                compact_buf.extend_from_slice(b"null\n");
                            }
                        } else {
                            compact_buf.extend_from_slice(b"null\n");
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref af) = array_field {
                    let af_refs: Vec<&str> = af.iter().map(|s| s.as_str()).collect();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(ranges) = json_object_get_fields_raw(raw, 0, &af_refs) {
                            compact_buf.push(b'[');
                            for (i, (vs, ve)) in ranges.iter().enumerate() {
                                if i > 0 { compact_buf.push(b','); }
                                compact_buf.extend_from_slice(&raw[*vs..*ve]);
                            }
                            compact_buf.extend_from_slice(b"]\n");
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref mf) = multi_field {
                    let mf_refs: Vec<&str> = mf.iter().map(|s| s.as_str()).collect();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(ranges) = json_object_get_fields_raw(raw, 0, &mf_refs) {
                            for (vs, ve) in &ranges {
                                let val_bytes = &raw[*vs..*ve];
                                let first = val_bytes[0];
                                if first != b'{' && first != b'[' {
                                    compact_buf.extend_from_slice(val_bytes);
                                    compact_buf.push(b'\n');
                                } else if use_compact_buf && is_json_compact(val_bytes) {
                                    compact_buf.extend_from_slice(val_bytes);
                                    compact_buf.push(b'\n');
                                } else {
                                    let v = json_to_value(unsafe { std::str::from_utf8_unchecked(val_bytes) })?;
                                    if use_compact_buf {
                                        push_compact_line(&mut compact_buf, &v);
                                    } else {
                                        push_pretty_line(&mut compact_buf, &v, indent_n, tab);
                                    }
                                }
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref remap) = field_remap {
                    let input_fields: Vec<&str> = remap.iter().map(|(_, f)| f.as_str()).collect();
                    let mut key_prefixes: Vec<Vec<u8>> = Vec::with_capacity(remap.len());
                    for (i, (out_key, _)) in remap.iter().enumerate() {
                        let mut prefix = Vec::new();
                        if i == 0 { prefix.push(b'{'); } else { prefix.push(b','); }
                        prefix.push(b'"');
                        prefix.extend_from_slice(out_key.as_bytes());
                        prefix.extend_from_slice(b"\":");
                        key_prefixes.push(prefix);
                    }
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(ranges) = json_object_get_fields_raw(raw, 0, &input_fields) {
                            for (i, (vs, ve)) in ranges.iter().enumerate() {
                                compact_buf.extend_from_slice(&key_prefixes[i]);
                                compact_buf.extend_from_slice(&raw[*vs..*ve]);
                            }
                            compact_buf.extend_from_slice(b"}\n");
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref f1, ref op, ref f2)) = field_binop {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((a, b)) = json_object_get_two_nums(raw, 0, f1, f2) {
                            let result = match op {
                                BinOp::Add => a + b,
                                BinOp::Sub => a - b,
                                BinOp::Mul => a * b,
                                _ => unreachable!(),
                            };
                            push_jq_number_bytes(&mut compact_buf, result);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sc_field, ref suffix)) = field_str_concat {
                    let suffix_needs_escape = suffix.bytes().any(|b| b == b'"' || b == b'\\' || b < 0x20);
                    let suffix_escaped: Vec<u8> = if suffix_needs_escape {
                        let mut buf = Vec::new();
                        for &b in suffix.as_bytes() {
                            match b {
                                b'"' => buf.extend_from_slice(b"\\\""),
                                b'\\' => buf.extend_from_slice(b"\\\\"),
                                b'\n' => buf.extend_from_slice(b"\\n"),
                                b'\r' => buf.extend_from_slice(b"\\r"),
                                b'\t' => buf.extend_from_slice(b"\\t"),
                                c if c < 0x20 => {}
                                _ => buf.push(b),
                            }
                        }
                        buf
                    } else {
                        suffix.as_bytes().to_vec()
                    };
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if raw[0] == b'{' {
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sc_field) {
                                let val = &raw[vs..ve];
                                if val[0] == b'"' && !val[1..ve-vs-1].contains(&b'\\') {
                                    compact_buf.extend_from_slice(&val[..val.len()-1]);
                                    compact_buf.extend_from_slice(&suffix_escaped);
                                    compact_buf.extend_from_slice(b"\"\n");
                                } else {
                                    let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                }
                            } else {
                                compact_buf.push(b'"');
                                compact_buf.extend_from_slice(&suffix_escaped);
                                compact_buf.extend_from_slice(b"\"\n");
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref field, ref op, threshold)) = select_cmp {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(val) = json_object_get_num(raw, 0, field) {
                            let pass = match op {
                                BinOp::Gt => val > threshold,
                                BinOp::Lt => val < threshold,
                                BinOp::Ge => val >= threshold,
                                BinOp::Le => val <= threshold,
                                BinOp::Eq => val == threshold,
                                BinOp::Ne => val != threshold,
                                _ => false,
                            };
                            if pass {
                                if is_json_compact(raw) {
                                    compact_buf.extend_from_slice(raw);
                                    compact_buf.push(b'\n');
                                } else {
                                    push_json_compact_raw(&mut compact_buf, raw);
                                    compact_buf.push(b'\n');
                                }
                                if compact_buf.len() >= 1 << 17 {
                                    let _ = out.write_all(&compact_buf);
                                    compact_buf.clear();
                                }
                            }
                        }
                        Ok(())
                    })
                } else if is_length {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(len) = json_value_length(raw, 0) {
                            push_jq_number_bytes(&mut compact_buf, len as f64);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if is_keys {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if !json_object_keys_to_buf(raw, 0, &mut compact_buf) {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if is_keys_unsorted {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if !json_object_keys_unsorted_to_buf(raw, 0, &mut compact_buf) {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref hf) = has_field {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(found) = json_object_has_key(raw, 0, hf) {
                            compact_buf.extend_from_slice(if found { b"true\n" } else { b"false\n" });
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if is_type {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        compact_buf.extend_from_slice(json_type_byte(raw[0]));
                        compact_buf.push(b'\n');
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref df) = del_field {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_del_field(raw, 0, df, &mut compact_buf) {
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if is_each {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if !json_each_value_raw(raw, 0, &mut compact_buf) {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if is_to_entries {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if !json_to_entries_raw(raw, 0, &mut compact_buf) {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref pf) = projection_fields {
                    let field_refs: Vec<&str> = pf.iter().map(|s| s.as_str()).collect();
                    json_stream_project(&input_str, &field_refs, |v| {
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        pool_value(v);
                        Ok(())
                    })
                } else if use_compact_buf {
                    json_stream_offsets(&input_str, |v, start, end| {
                        let raw = &input_bytes[start..end];
                        process_input(&v, Some(raw), &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        pool_value(v);
                        Ok(())
                    })
                } else {
                    json_stream(&input_str, |v| {
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        pool_value(v);
                        Ok(())
                    })
                };
                if let Err(e) = parse_result {
                    eprintln!("jq: error (at <stdin>:0): {}", e);
                    process::exit(2);
                }
            }
        }
    } else {
        // Process files
        for file in &files {
            let f = match std::fs::File::open(file) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("jq: error: Could not open file {}: {}", file, e);
                    process::exit(2);
                }
            };
            let meta = f.metadata().unwrap();
            // Memory-map files to avoid heap allocation for file content
            let (mmap, content);
            if meta.len() > 0 {
                mmap = Some(unsafe { memmap2::Mmap::map(&f) }.unwrap_or_else(|e| {
                    eprintln!("jq: error: Could not mmap file {}: {}", file, e);
                    process::exit(2);
                }));
                // SAFETY: JSON is defined as UTF-8. Our parser validates structure
                // byte-by-byte, so we skip the upfront O(n) UTF-8 validation which
                // costs ~40% of total runtime on large files.
                content = unsafe { std::str::from_utf8_unchecked(mmap.as_ref().unwrap()) };
            } else {
                mmap = None;
                content = "";
            }
            let _ = &mmap; // keep mmap alive
            let parse_result = if filter.is_empty() {
                // Empty fast path: just validate JSON structure, produce no output.
                json_stream_raw(content, |_, _| Ok(()))
            } else if let Some(ref fa_field) = field_access {
                // Field access fast path: extract a single field's raw bytes, no full parse.
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if raw[0] == b'{' {
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, fa_field) {
                            let val_bytes = &raw[vs..ve];
                            let first = val_bytes[0];
                            if first != b'{' && first != b'[' {
                                // Scalar: same in compact and pretty
                                compact_buf.extend_from_slice(val_bytes);
                                compact_buf.push(b'\n');
                            } else if use_compact_buf && is_json_compact(val_bytes) {
                                compact_buf.extend_from_slice(val_bytes);
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(val_bytes) })?;
                                if use_compact_buf {
                                    push_compact_line(&mut compact_buf, &v);
                                } else {
                                    push_pretty_line(&mut compact_buf, &v, indent_n, tab);
                                }
                            }
                        } else {
                            compact_buf.extend_from_slice(b"null\n");
                        }
                    } else {
                        compact_buf.extend_from_slice(b"null\n");
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref af) = array_field {
                let content_bytes = content.as_bytes();
                let af_refs: Vec<&str> = af.iter().map(|s| s.as_str()).collect();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(ranges) = json_object_get_fields_raw(raw, 0, &af_refs) {
                        compact_buf.push(b'[');
                        for (i, (vs, ve)) in ranges.iter().enumerate() {
                            if i > 0 { compact_buf.push(b','); }
                            compact_buf.extend_from_slice(&raw[*vs..*ve]);
                        }
                        compact_buf.extend_from_slice(b"]\n");
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref mf) = multi_field {
                let content_bytes = content.as_bytes();
                let mf_refs: Vec<&str> = mf.iter().map(|s| s.as_str()).collect();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(ranges) = json_object_get_fields_raw(raw, 0, &mf_refs) {
                        for (vs, ve) in &ranges {
                            let val_bytes = &raw[*vs..*ve];
                            let first = val_bytes[0];
                            if first != b'{' && first != b'[' {
                                compact_buf.extend_from_slice(val_bytes);
                                compact_buf.push(b'\n');
                            } else if use_compact_buf && is_json_compact(val_bytes) {
                                compact_buf.extend_from_slice(val_bytes);
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(val_bytes) })?;
                                if use_compact_buf {
                                    push_compact_line(&mut compact_buf, &v);
                                } else {
                                    push_pretty_line(&mut compact_buf, &v, indent_n, tab);
                                }
                            }
                        }
                    } else {
                        // Missing field → fall back to parse + JIT for all values
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref field, ref op, threshold)) = select_cmp {
                // Select fast path: extract field without full parsing, copy raw bytes on match.
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(val) = json_object_get_num(raw, 0, field) {
                        let pass = match op {
                            BinOp::Gt => val > threshold,
                            BinOp::Lt => val < threshold,
                            BinOp::Ge => val >= threshold,
                            BinOp::Le => val <= threshold,
                            BinOp::Eq => val == threshold,
                            BinOp::Ne => val != threshold,
                            _ => false,
                        };
                        if pass {
                            if is_json_compact(raw) {
                                compact_buf.extend_from_slice(raw);
                                compact_buf.push(b'\n');
                            } else {
                                push_json_compact_raw(&mut compact_buf, raw);
                                compact_buf.push(b'\n');
                            }
                            if compact_buf.len() >= 1 << 17 {
                                let _ = out.write_all(&compact_buf);
                                compact_buf.clear();
                            }
                        }
                    }
                    Ok(())
                })
            } else if let Some(ref remap) = field_remap {
                // Object projection fast path: extract multiple fields' raw bytes and
                // construct output object directly, no Value construction needed.
                let content_bytes = content.as_bytes();
                let input_fields: Vec<&str> = remap.iter().map(|(_, f)| f.as_str()).collect();
                // Pre-build output key prefixes: {"key1":, ,"key2":, etc.
                let mut key_prefixes: Vec<Vec<u8>> = Vec::with_capacity(remap.len());
                for (i, (out_key, _)) in remap.iter().enumerate() {
                    let mut prefix = Vec::new();
                    if i == 0 { prefix.push(b'{'); } else { prefix.push(b','); }
                    prefix.push(b'"');
                    prefix.extend_from_slice(out_key.as_bytes());
                    prefix.extend_from_slice(b"\":");
                    key_prefixes.push(prefix);
                }
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(ranges) = json_object_get_fields_raw(raw, 0, &input_fields) {
                        for (i, (vs, ve)) in ranges.iter().enumerate() {
                            compact_buf.extend_from_slice(&key_prefixes[i]);
                            compact_buf.extend_from_slice(&raw[*vs..*ve]);
                        }
                        compact_buf.extend_from_slice(b"}\n");
                    } else {
                        // Missing field → fall back to parse + JIT
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref f1, ref op, ref f2)) = field_binop {
                // Arithmetic fast path: extract two numeric fields, compute, output.
                // Falls back to normal JIT if fields are missing or non-numeric.
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((a, b)) = json_object_get_two_nums(raw, 0, f1, f2) {
                        let result = match op {
                            BinOp::Add => a + b,
                            BinOp::Sub => a - b,
                            BinOp::Mul => a * b,
                            _ => unreachable!(),
                        };
                        push_jq_number_bytes(&mut compact_buf, result);
                        compact_buf.push(b'\n');
                    } else {
                        // Field missing or non-numeric: fall back to parse + JIT
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sc_field, ref suffix)) = field_str_concat {
                // String concat fast path: extract field's raw string bytes and append suffix.
                // Only works when the field value is a non-escaped string.
                let content_bytes = content.as_bytes();
                // Pre-escape suffix for JSON output
                let suffix_needs_escape = suffix.bytes().any(|b| b == b'"' || b == b'\\' || b < 0x20);
                let suffix_escaped: Vec<u8> = if suffix_needs_escape {
                    let mut buf = Vec::new();
                    for &b in suffix.as_bytes() {
                        match b {
                            b'"' => buf.extend_from_slice(b"\\\""),
                            b'\\' => buf.extend_from_slice(b"\\\\"),
                            b'\n' => buf.extend_from_slice(b"\\n"),
                            b'\r' => buf.extend_from_slice(b"\\r"),
                            b'\t' => buf.extend_from_slice(b"\\t"),
                            c if c < 0x20 => { let _ = std::fmt::Write::write_fmt(&mut String::new(), format_args!("\\u{:04x}", c)); }
                            _ => buf.push(b),
                        }
                    }
                    buf
                } else {
                    suffix.as_bytes().to_vec()
                };
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if raw[0] == b'{' {
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sc_field) {
                            let val = &raw[vs..ve];
                            // Only use fast path for simple strings (quoted, no backslash)
                            if val[0] == b'"' && !val[1..ve-vs-1].contains(&b'\\') {
                                // Copy everything except trailing quote, append suffix + quote + newline
                                compact_buf.extend_from_slice(&val[..val.len()-1]);
                                compact_buf.extend_from_slice(&suffix_escaped);
                                compact_buf.extend_from_slice(b"\"\n");
                            } else {
                                // Fall back for non-string or escaped string values
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            // Field not found → null + "str" = "str"
                            compact_buf.push(b'"');
                            compact_buf.extend_from_slice(&suffix_escaped);
                            compact_buf.extend_from_slice(b"\"\n");
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if is_length {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(len) = json_value_length(raw, 0) {
                        push_jq_number_bytes(&mut compact_buf, len as f64);
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if is_keys {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if !json_object_keys_to_buf(raw, 0, &mut compact_buf) {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if is_keys_unsorted {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if !json_object_keys_unsorted_to_buf(raw, 0, &mut compact_buf) {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref hf) = has_field {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(found) = json_object_has_key(raw, 0, hf) {
                        compact_buf.extend_from_slice(if found { b"true\n" } else { b"false\n" });
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if is_type {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    compact_buf.extend_from_slice(json_type_byte(raw[0]));
                    compact_buf.push(b'\n');
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref df) = del_field {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_del_field(raw, 0, df, &mut compact_buf) {
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if is_each {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if !json_each_value_raw(raw, 0, &mut compact_buf) {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if is_to_entries {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if !json_to_entries_raw(raw, 0, &mut compact_buf) {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if filter.is_identity() && use_compact_buf && !exit_status {
                // Identity fast path: skip JSON parsing entirely, just validate structure
                // and copy raw bytes directly. Falls back to parse+serialize for non-compact input.
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if is_json_compact(raw) {
                        compact_buf.extend_from_slice(raw);
                        compact_buf.push(b'\n');
                    } else {
                        push_json_compact_raw(&mut compact_buf, raw);
                        compact_buf.push(b'\n');
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref pf) = projection_fields {
                let field_refs: Vec<&str> = pf.iter().map(|s| s.as_str()).collect();
                json_stream_project(content, &field_refs, |v| {
                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    pool_value(v);
                    Ok(())
                })
            } else if use_compact_buf {
                let content_bytes = content.as_bytes();
                json_stream_offsets(content, |v, start, end| {
                    let raw = &content_bytes[start..end];
                    process_input(&v, Some(raw), &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    pool_value(v);
                    Ok(())
                })
            } else {
                json_stream(content, |v| {
                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    pool_value(v);
                    Ok(())
                })
            };
            if let Err(e) = parse_result {
                eprintln!("jq: error: {}", e);
                process::exit(2);
            }
        }
    }

    if !compact_buf.is_empty() {
        let _ = out.write_all(&compact_buf);
    }
    let _ = out.flush();

    if had_error {
        process::exit(5);
    }
    if exit_status && any_output_false {
        process::exit(5);
    }
}

fn print_usage() {
    eprintln!("Usage: jq-jit [OPTIONS] <FILTER> [FILE...]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -c, --compact-output   Compact output");
    eprintln!("  -r, --raw-output       Raw output (strings without quotes)");
    eprintln!("  -R, --raw-input        Raw input (each line is a string)");
    eprintln!("  -n, --null-input        Use null as input");
    eprintln!("  -s, --slurp            Slurp all inputs into array");
    eprintln!("  -S, --sort-keys        Sort object keys");
    eprintln!("  -e, --exit-status      Exit with non-zero if last output is false/null");
    eprintln!("  -f, --from-file FILE   Read filter from file");
    eprintln!("  --tab                  Use tabs for indentation");
    eprintln!("  --indent N             Use N spaces for indentation");
    eprintln!("  --arg NAME VALUE       Set variable $NAME to VALUE (string)");
    eprintln!("  --argjson NAME VALUE   Set variable $NAME to VALUE (JSON)");
    eprintln!("  --version              Show version");
    eprintln!("  -h, --help             Show this help");
}
