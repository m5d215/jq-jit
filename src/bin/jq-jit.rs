//! jq-jit: A JIT-compiled jq implementation.
//!
//! Usage: jq-jit [OPTIONS] <FILTER> [FILE...]

use std::io::{self, BufRead, Read, Write};
use std::process;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use jq_jit::value::{Value, json_to_value, json_stream, value_to_json_precise, value_to_json_pretty_ext, push_compact_line, write_value_compact_ext, write_value_compact_line, write_value_pretty_line};
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

    // Skip JIT for null-input: filter runs once, eval is faster than JIT compile + run
    let use_jit = !null_input;
    let filter = match Filter::with_options(&filter_str, &lib_dirs, use_jit) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("jq: error: {}", e);
            process::exit(3);
        }
    };

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
    let mut compact_buf: Vec<u8> = if use_compact_buf { Vec::with_capacity(1 << 17) } else { Vec::new() };
    let process_input = |input: &Value, out: &mut io::BufWriter<io::StdoutLock>, cbuf: &mut Vec<u8>, any_false: &mut bool, had_error: &mut bool| {
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
                push_compact_line(cbuf, result);
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
        process_input(&Value::Null, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
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
                            process_input(&Value::from_str(&l), &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
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
                process_input(&arr, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
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
                process_input(&arr, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
            } else {
                if let Err(e) = json_stream(&input_str, |v| {
                    process_input(&v, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    Ok(())
                }) {
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
                content = std::str::from_utf8(mmap.as_ref().unwrap()).unwrap_or_else(|e| {
                    eprintln!("jq: error: File {} is not valid UTF-8: {}", file, e);
                    process::exit(2);
                });
            } else {
                mmap = None;
                content = "";
            }
            let _ = &mmap; // keep mmap alive
            if let Err(e) = json_stream(content, |v| {
                process_input(&v, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                Ok(())
            }) {
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
