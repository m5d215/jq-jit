//! jq-jit2: A JIT-compiled jq implementation.
//!
//! Usage: jq-jit2 [OPTIONS] <FILTER> [FILE...]

use std::io::{self, BufRead, Read, Write};
use std::process;

use jq_jit2::value::{self, Value, json_to_value, json_stream, value_to_json, value_to_json_precise, value_to_json_pretty, write_value_compact};
use jq_jit2::interpreter::Filter;

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

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-c" | "--compact-output" => compact = true,
            "-r" | "--raw-output" => raw_output = true,
            "-R" | "--raw-input" => raw_input = true,
            "-n" | "--null-input" => null_input = true,
            "-s" | "--slurp" => slurp = true,
            "-j" | "--join-output" => {
                join_output = true;
                raw_output = true;
            }
            "-S" | "--sort-keys" => sort_keys = true,
            "-e" | "--exit-status" => exit_status = true,
            "--tab" => {
                tab = true;
            }
            "--indent" => {
                i += 1;
                if i < args.len() {
                    indent_n = args[i].parse().unwrap_or(2);
                }
            }
            "--arg" => {
                if i + 2 < args.len() {
                    let name = args[i + 1].clone();
                    let val = Value::from_str(&args[i + 2]);
                    arg_vars.push((name, val));
                    i += 2;
                }
            }
            "--argjson" => {
                if i + 2 < args.len() {
                    let name = args[i + 1].clone();
                    match json_to_value(&args[i + 2]) {
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
                if i < args.len() {
                    lib_dirs.push(args[i].clone());
                }
            }
            "--args" => {
                // Remaining args are positional args
                break;
            }
            "--version" => {
                println!("jq-jit2-0.1.0");
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
            eprintln!("Usage: jq-jit2 [OPTIONS] <FILTER> [FILE...]");
            process::exit(2);
        }
    };

    // Compile filter
    let filter = match Filter::with_lib_dirs(&filter_str, &lib_dirs) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("jq: error: {}", e);
            process::exit(3);
        }
    };

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    let mut any_output_false = false;

    let format_value = |v: &Value| -> String {
        if raw_output {
            if let Value::Str(s) = v {
                return s.as_ref().clone();
            }
        }
        if compact {
            value_to_json_precise(v)
        } else if tab {
            value_to_json_pretty(v, 0).replace("  ", "\t")
        } else {
            value_to_json_pretty(v, 0)
        }
    };

    let mut had_error = false;
    let process_input = |input: &Value, out: &mut io::BufWriter<io::StdoutLock>, any_false: &mut bool, had_error: &mut bool| {
        match filter.execute(input) {
            Ok(results) => {
                for result in &results {
                    if let Value::Error(e) = result {
                        eprintln!("jq: error: {}", e.as_str());
                        *had_error = true;
                        continue;
                    }
                    if exit_status && !result.is_true() {
                        *any_false = true;
                    }
                    // Use streaming write for compact mode to avoid intermediate String allocation
                    if compact && !raw_output {
                        let _ = write_value_compact(out, result);
                        if !join_output {
                            let _ = out.write_all(b"\n");
                        }
                    } else {
                        let formatted = format_value(result);
                        if join_output {
                            let _ = write!(out, "{}", formatted);
                        } else {
                            let _ = writeln!(out, "{}", formatted);
                        }
                    }
                }
            }
            Err(e) => {
                let msg = format!("{}", e);
                if let Some(jq_msg) = msg.strip_prefix("__jqerror__:") {
                    eprintln!("jq: error: {}", jq_msg);
                } else {
                    eprintln!("jq: error: {}", msg);
                }
                *had_error = true;
            }
        }
    };

    if null_input {
        process_input(&Value::Null, &mut out, &mut any_output_false, &mut had_error);
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
                            process_input(&Value::from_str(&l), &mut out, &mut any_output_false, &mut had_error);
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
                process_input(&arr, &mut out, &mut any_output_false, &mut had_error);
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
                process_input(&arr, &mut out, &mut any_output_false, &mut had_error);
            } else {
                if let Err(e) = json_stream(&input_str, |v| {
                    process_input(&v, &mut out, &mut any_output_false, &mut had_error);
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
            let content = match std::fs::read_to_string(file) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("jq: error: Could not open file {}: {}", file, e);
                    process::exit(2);
                }
            };
            if let Err(e) = json_stream(&content, |v| {
                process_input(&v, &mut out, &mut any_output_false, &mut had_error);
                Ok(())
            }) {
                eprintln!("jq: error: {}", e);
                process::exit(2);
            }
        }
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
    eprintln!("Usage: jq-jit2 [OPTIONS] <FILTER> [FILE...]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -c, --compact-output   Compact output");
    eprintln!("  -r, --raw-output       Raw output (strings without quotes)");
    eprintln!("  -R, --raw-input        Raw input (each line is a string)");
    eprintln!("  -n, --null-input        Use null as input");
    eprintln!("  -s, --slurp            Slurp all inputs into array");
    eprintln!("  -S, --sort-keys        Sort object keys");
    eprintln!("  -e, --exit-status      Exit with non-zero if last output is false/null");
    eprintln!("  --tab                  Use tabs for indentation");
    eprintln!("  --indent N             Use N spaces for indentation");
    eprintln!("  --arg NAME VALUE       Set variable $NAME to VALUE (string)");
    eprintln!("  --argjson NAME VALUE   Set variable $NAME to VALUE (JSON)");
    eprintln!("  --version              Show version");
    eprintln!("  -h, --help             Show this help");
}
