//! AOT runtime entry point for standalone compiled jq-jit binaries.
//!
//! When a user runs `jq-jit --compile '.+1' -o add1`, the resulting `add1`
//! binary contains a Cranelift-generated `main()` that calls `aot_run()`.
//!
//! This module intentionally avoids ALL libjq dependencies. JSON parsing uses
//! serde_json, and output formatting is self-contained.

use std::rc::Rc;

use crate::cache::deserialize_literals;
use crate::input;
use crate::output::{output_value, OutputOptions};
use crate::value::Value;

// ---------------------------------------------------------------------------
// Filter function type (same as JitFilter / CachedFilter)
// ---------------------------------------------------------------------------

type FilterFn =
    fn(*const Value, extern "C" fn(*const Value, *mut u8), *mut u8, *const *const Value);

// ---------------------------------------------------------------------------
// Callback for collecting filter results
// ---------------------------------------------------------------------------

extern "C" fn collect_callback(value_ptr: *const Value, ctx: *mut u8) {
    assert!(!value_ptr.is_null() && !ctx.is_null());
    let results = unsafe { &mut *(ctx as *mut Vec<Value>) };
    let value = unsafe { (*value_ptr).clone() };
    results.push(value);
}

// ---------------------------------------------------------------------------
// AOT CLI options (runtime-only subset)
// ---------------------------------------------------------------------------

struct AotOptions {
    files: Vec<String>,
    raw_output: bool,
    raw_input: bool,
    compact_output: bool,
    null_input: bool,
    slurp: bool,
    exit_status: bool,
    tab_indent: bool,
    indent_n: usize,
    show_help: bool,
}

impl Default for AotOptions {
    fn default() -> Self {
        Self {
            files: Vec::new(),
            raw_output: false,
            raw_input: false,
            compact_output: false,
            null_input: false,
            slurp: false,
            exit_status: false,
            tab_indent: false,
            indent_n: 2,
            show_help: false,
        }
    }
}

impl AotOptions {
    fn output_opts(&self) -> OutputOptions {
        OutputOptions {
            raw_output: self.raw_output,
            compact_output: self.compact_output,
            tab_indent: self.tab_indent,
            indent_n: self.indent_n,
        }
    }
}

// ---------------------------------------------------------------------------
// Argument parsing for AOT binaries
// ---------------------------------------------------------------------------

fn parse_aot_args(argc: i32, argv: *const *const i8) -> Result<AotOptions, String> {
    // Reconstruct argument strings from C argc/argv
    let args: Vec<String> = (1..argc as isize)
        .map(|i| {
            let c_str = unsafe { std::ffi::CStr::from_ptr(*argv.offset(i)) };
            c_str.to_string_lossy().into_owned()
        })
        .collect();

    let mut opts = AotOptions::default();
    let mut i = 0;
    let mut after_double_dash = false;

    while i < args.len() {
        let arg = &args[i];

        if after_double_dash {
            opts.files.push(arg.clone());
            i += 1;
            continue;
        }

        if arg == "--" {
            after_double_dash = true;
            i += 1;
            continue;
        }

        match arg.as_str() {
            "-h" | "--help" => {
                opts.show_help = true;
                return Ok(opts);
            }
            "-r" | "--raw-output" => opts.raw_output = true,
            "-R" | "--raw-input" => opts.raw_input = true,
            "-c" | "--compact-output" => opts.compact_output = true,
            "-n" | "--null-input" => opts.null_input = true,
            "-s" | "--slurp" => opts.slurp = true,
            "-e" | "--exit-status" => opts.exit_status = true,
            "--tab" => opts.tab_indent = true,
            "--indent" => {
                i += 1;
                if i >= args.len() {
                    return Err("--indent requires a numeric argument".to_string());
                }
                opts.indent_n = args[i]
                    .parse::<usize>()
                    .map_err(|_| format!("--indent: invalid number {:?}", args[i]))?;
            }
            _ => {
                if arg.starts_with('-') && arg.len() > 1 && !arg.starts_with("--") {
                    // Combined short flags like -rc, -Rns
                    let chars: Vec<char> = arg[1..].chars().collect();
                    for ch in &chars {
                        match ch {
                            'r' => opts.raw_output = true,
                            'R' => opts.raw_input = true,
                            'c' => opts.compact_output = true,
                            'n' => opts.null_input = true,
                            's' => opts.slurp = true,
                            'e' => opts.exit_status = true,
                            'h' => {
                                opts.show_help = true;
                                return Ok(opts);
                            }
                            _ => return Err(format!("Unknown option: -{}", ch)),
                        }
                    }
                } else {
                    // Positional args are input files
                    opts.files.push(arg.clone());
                }
            }
        }
        i += 1;
    }

    Ok(opts)
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

fn print_aot_help() {
    eprintln!(
        "\
Usage: <compiled-filter> [OPTIONS] [FILE...]

  A pre-compiled jq-jit filter.

Options:
  -r, --raw-output      Output raw strings (no quotes)
  -R, --raw-input       Read each input line as a string
  -c, --compact-output  Compact JSON output (no whitespace)
  -n, --null-input      Use null as input instead of reading JSON
  -s, --slurp           Read entire input into an array
  -e, --exit-status     Exit 1 if last output is false or null
  --tab                 Indent with tabs
  --indent N            Indent with N spaces (default: 2)
  -h, --help            Show this help message"
    );
}

// ---------------------------------------------------------------------------
// AOT entry point
// ---------------------------------------------------------------------------

/// AOT binary entry point. Called from Cranelift-generated main().
///
/// # Arguments
///
/// * `filter_fn` — pointer to the compiled jit_filter function (same object file)
/// * `literal_data` — pointer to serialized literal bytes embedded in .rodata
/// * `literal_data_len` — length of the serialized literal data
/// * `argc` — from C main()
/// * `argv` — from C main()
///
/// # Returns
///
/// Exit code: 0 = success, 1 = -e false/null, 2 = parse error, 5 = I/O error
#[unsafe(no_mangle)]
pub extern "C" fn aot_run(
    filter_fn: *const u8,
    literal_data: *const u8,
    literal_data_len: u64,
    argc: i32,
    argv: *const *const i8,
) -> i32 {
    // Catch panics to avoid unwinding across the C ABI boundary
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        aot_run_inner(filter_fn, literal_data, literal_data_len, argc, argv)
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            eprintln!("jq-jit: internal error (panic in AOT filter)");
            5
        }
    }
}

fn aot_run_inner(
    filter_fn: *const u8,
    literal_data: *const u8,
    literal_data_len: u64,
    argc: i32,
    argv: *const *const i8,
) -> i32 {
    // Step 1: Parse runtime arguments
    let opts = match parse_aot_args(argc, argv) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("jq-jit: {}", e);
            return 2;
        }
    };

    if opts.show_help {
        print_aot_help();
        return 0;
    }

    // Step 2: Deserialize literals from embedded .rodata
    let (literals, literal_ptrs) = if literal_data.is_null() || literal_data_len == 0 {
        // No literals embedded — use empty table
        (Vec::<Box<Value>>::new(), Vec::<*const Value>::new())
    } else {
        let data = unsafe { std::slice::from_raw_parts(literal_data, literal_data_len as usize) };
        match deserialize_literals(data) {
            Some(lits) => {
                let ptrs: Vec<*const Value> = lits.iter().map(|b| &**b as *const Value).collect();
                (lits, ptrs)
            }
            None => {
                eprintln!("jq-jit: failed to deserialize embedded literals");
                return 5;
            }
        }
    };

    // Step 3: Cast filter function pointer
    let filter: FilterFn = unsafe { std::mem::transmute::<*const u8, FilterFn>(filter_fn) };

    // Build output options once from AotOptions
    let out_opts = opts.output_opts();

    // Helper: execute filter and output results, tracking exit-status state.
    let execute_and_output =
        |input: &Value,
         filter: FilterFn,
         literal_ptrs: &[*const Value],
         out_opts: &OutputOptions,
         last_is_false_or_null: &mut bool,
         had_output: &mut bool| {
            let mut results: Vec<Value> = Vec::new();
            filter(
                input as *const Value,
                collect_callback,
                &mut results as *mut Vec<Value> as *mut u8,
                literal_ptrs.as_ptr(),
            );
            for result in &results {
                if let Value::Error(e) = result {
                    eprintln!("jq-jit: error: {}", e);
                    continue;
                }
                output_value(result, out_opts);
                *last_is_false_or_null = matches!(result, Value::Null | Value::Bool(false));
                *had_output = true;
            }
        };

    // Step 4+5: Read input(s) and execute filter (streaming where possible)
    let mut last_is_false_or_null = false;
    let mut had_output = false;

    if opts.null_input {
        // -n mode: run filter on null
        execute_and_output(
            &Value::Null,
            filter,
            &literal_ptrs,
            &out_opts,
            &mut last_is_false_or_null,
            &mut had_output,
        );
    } else if opts.raw_input {
        if opts.slurp {
            // -Rs: entire input as a single string (including newlines)
            let raw = match input::read_raw_string(&opts.files) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("jq-jit: {}", e);
                    return 5;
                }
            };
            let input_val = Value::Str(Rc::new(raw));
            execute_and_output(
                &input_val,
                filter,
                &literal_ptrs,
                &out_opts,
                &mut last_is_false_or_null,
                &mut had_output,
            );
        } else {
            // -R mode: each line as a separate string
            let lines = match input::read_raw_lines(&opts.files) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("jq-jit: {}", e);
                    return 5;
                }
            };
            for line in &lines {
                let input_val = Value::Str(Rc::new(line.clone()));
                execute_and_output(
                    &input_val,
                    filter,
                    &literal_ptrs,
                    &out_opts,
                    &mut last_is_false_or_null,
                    &mut had_output,
                );
            }
        }
    } else {
        // JSON mode — streaming
        let reader = match input::open_input(&opts.files) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("jq-jit: {}", e);
                return 5;
            }
        };

        if opts.slurp {
            // Slurp: collect all values, then filter as array
            let mut all = Vec::new();
            for result in input::stream_json_values(reader) {
                match result {
                    Ok(v) => all.push(v),
                    Err(e) => {
                        eprintln!("jq-jit: {}", e);
                        return 2;
                    }
                }
            }
            let arr = Value::Arr(Rc::new(all));
            execute_and_output(
                &arr,
                filter,
                &literal_ptrs,
                &out_opts,
                &mut last_is_false_or_null,
                &mut had_output,
            );
        } else {
            // Stream: parse one, execute one, output one
            for result in input::stream_json_values(reader) {
                match result {
                    Ok(input_val) => {
                        execute_and_output(
                            &input_val,
                            filter,
                            &literal_ptrs,
                            &out_opts,
                            &mut last_is_false_or_null,
                            &mut had_output,
                        );
                    }
                    Err(e) => {
                        eprintln!("jq-jit: {}", e);
                        return 2;
                    }
                }
            }
        }
    }

    // Step 6: Exit status
    if opts.exit_status && (!had_output || last_is_false_or_null) {
        return 1;
    }

    // Keep literals alive until after all filter calls complete
    drop(literals);

    0
}
