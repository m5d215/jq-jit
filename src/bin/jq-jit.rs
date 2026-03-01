//! jq-jit CLI — jq-compatible command-line interface for the JIT compiler.
//!
//! Usage: jq-jit [OPTIONS] <FILTER> [FILE...]

use std::process;
use std::rc::Rc;

use jq_jit::bytecode::JqState;
use jq_jit::cache::FilterCache;
use jq_jit::codegen::{compile_expr, compile_to_object};
use jq_jit::compiler::bytecode_to_ir;
use jq_jit::input;
use jq_jit::output::{output_value, OutputOptions};
use jq_jit::value::Value;

// ---------------------------------------------------------------------------
// CLI options
// ---------------------------------------------------------------------------

/// A --arg or --argjson binding: prepend `$name as $name |` to the filter.
struct ArgBinding {
    name: String,
    /// JSON-encoded value string (for --arg, the value is JSON-escaped).
    json_value: String,
}

struct CliOptions {
    filter: String,
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
    show_version: bool,
    from_file: Option<String>,
    arg_bindings: Vec<ArgBinding>,
    slurpfile_bindings: Vec<(String, String)>, // (name, filepath)
    no_cache: bool,
    clear_cache: bool,
    cache_dir: Option<String>,
    compile_output: Option<String>,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            filter: String::new(),
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
            show_version: false,
            from_file: None,
            arg_bindings: Vec::new(),
            slurpfile_bindings: Vec::new(),
            no_cache: false,
            clear_cache: false,
            cache_dir: None,
            compile_output: None,
        }
    }
}

impl CliOptions {
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
// Argument parsing (manual — no external crates)
// ---------------------------------------------------------------------------

fn parse_args() -> Result<CliOptions, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut opts = CliOptions::default();
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    let mut after_double_dash = false;
    let mut compile_mode = false;
    let mut compile_output: Option<String> = None;

    while i < args.len() {
        let arg = &args[i];

        // After `--`, all remaining args are positional
        if after_double_dash {
            positional.push(arg.clone());
            i += 1;
            continue;
        }

        // Handle `--` separator
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
            "--version" => {
                opts.show_version = true;
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
            "-f" | "--from-file" => {
                i += 1;
                if i >= args.len() {
                    return Err("-f/--from-file requires a filename argument".to_string());
                }
                opts.from_file = Some(args[i].clone());
            }
            "--arg" => {
                i += 1;
                if i + 1 >= args.len() {
                    return Err("--arg requires two arguments: name and value".to_string());
                }
                let name = args[i].clone();
                i += 1;
                let value = args[i].clone();
                // --arg binds as a string: JSON-escape the value
                opts.arg_bindings.push(ArgBinding {
                    name,
                    json_value: json_escape_for_arg(&value),
                });
            }
            "--argjson" => {
                i += 1;
                if i + 1 >= args.len() {
                    return Err("--argjson requires two arguments: name and JSON value".to_string());
                }
                let name = args[i].clone();
                i += 1;
                let json_value = args[i].clone();
                opts.arg_bindings.push(ArgBinding {
                    name,
                    json_value,
                });
            }
            "--compile" => compile_mode = true,
            "-o" => {
                i += 1;
                if i >= args.len() {
                    return Err("-o requires an output path argument".to_string());
                }
                compile_output = Some(args[i].clone());
            }
            "--no-cache" => opts.no_cache = true,
            "--clear-cache" => {
                opts.clear_cache = true;
                return Ok(opts);
            }
            "--cache-dir" => {
                i += 1;
                if i >= args.len() {
                    return Err("--cache-dir requires a directory argument".to_string());
                }
                opts.cache_dir = Some(args[i].clone());
            }
            "--slurpfile" => {
                i += 1;
                if i + 1 >= args.len() {
                    return Err("--slurpfile requires two arguments: name and filename".to_string());
                }
                let name = args[i].clone();
                i += 1;
                let filepath = args[i].clone();
                opts.slurpfile_bindings.push((name, filepath));
            }
            _ => {
                // Detect numeric filters like -1, -1.5, -.5 (not flags)
                let is_numeric_filter = arg.starts_with('-')
                    && arg.len() > 1
                    && (arg[1..].starts_with(|c: char| c.is_ascii_digit()) || arg[1..].starts_with('.'));

                if arg.starts_with('-') && arg.len() > 1 && !arg.starts_with("--") && !is_numeric_filter {
                    // Handle combined short flags like -rc, -rs, -Rn, etc.
                    let chars: Vec<char> = arg[1..].chars().collect();
                    let mut j = 0;
                    while j < chars.len() {
                        match chars[j] {
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
                            'f' => {
                                // -f with next arg as filename
                                i += 1;
                                if i >= args.len() {
                                    return Err("-f requires a filename argument".to_string());
                                }
                                opts.from_file = Some(args[i].clone());
                                // Remaining chars in this arg are ignored
                                break;
                            }
                            _ => {
                                return Err(format!("Unknown option: -{}", chars[j]));
                            }
                        }
                        j += 1;
                    }
                } else {
                    positional.push(arg.clone());
                }
            }
        }
        i += 1;
    }

    // Validate --compile / -o combination
    if compile_mode {
        match compile_output {
            Some(ref _path) => {
                opts.compile_output = compile_output;
            }
            None => {
                return Err("--compile requires -o <output-path>".to_string());
            }
        }
    } else if compile_output.is_some() {
        return Err("-o is only valid with --compile".to_string());
    }

    // If --from-file is set, read the filter from the file
    if let Some(ref path) = opts.from_file {
        let filter = std::fs::read_to_string(path)
            .map_err(|e| format!("reading filter file '{}': {}", path, e))?;
        opts.filter = filter.trim().to_string();
        // All positional args are input files
        opts.files = positional;
    } else {
        // First positional is the filter, rest are files
        if positional.is_empty() && !opts.show_help && !opts.show_version {
            return Err("No filter provided. Usage: jq-jit <FILTER> [FILE...]".to_string());
        }

        if !positional.is_empty() {
            opts.filter = positional.remove(0);
            opts.files = positional;
        }
    }

    Ok(opts)
}

/// JSON-escape a string value (for --arg string bindings).
fn json_escape_for_arg(s: &str) -> String {
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
// Help / Version
// ---------------------------------------------------------------------------

fn print_help() {
    eprintln!(
        "\
Usage: jq-jit [OPTIONS] <FILTER> [FILE...]

  A JIT-compiled jq processor.

Options:
  -r, --raw-output      Output raw strings (no quotes)
  -R, --raw-input        Read each input line as a string
  -c, --compact-output   Compact JSON output (no whitespace)
  -n, --null-input       Use null as input instead of reading JSON
  -s, --slurp            Read entire input into an array
  -e, --exit-status      Exit 1 if last output is false or null
  --tab                  Indent with tabs
  --indent N             Indent with N spaces (default: 2)
  -f, --from-file FILE   Read filter from FILE
  --arg NAME VALUE       Bind $NAME to string VALUE
  --argjson NAME VALUE   Bind $NAME to JSON VALUE
  --slurpfile NAME FILE  Bind $NAME to JSON array from FILE
  --compile              AOT compile the filter to a standalone binary
  -o PATH                Output path for --compile
  --no-cache             Disable filter cache (use JIT only)
  --clear-cache          Clear all cached filters and exit
  --cache-dir DIR        Use custom cache directory
  -h, --help             Show this help message
  --version              Show version

Examples:
  echo '{{\"foo\":1}}' | jq-jit '.foo'
  jq-jit '.[] | .name' data.json
  jq-jit -r '.name' data.json
  jq-jit -c '.' data.json
  jq-jit -n '1 + 2'
  jq-jit -R '.' <<< 'hello world'
  jq-jit --arg name Alice -n '$name'
  jq-jit -f filter.jq data.json
  jq-jit --compile '. + 1' -o add1"
    );
}

fn print_version() {
    println!("jq-jit {}", env!("CARGO_PKG_VERSION"));
}

// ---------------------------------------------------------------------------
// AOT compilation
// ---------------------------------------------------------------------------

/// AOT compile a filter to a standalone native binary.
fn compile_aot(filter: &str, output: &str) -> i32 {
    // Step 1: Compile filter to bytecode via libjq, then to IR
    let mut jq = match JqState::new() {
        Ok(jq) => jq,
        Err(e) => {
            eprintln!("jq-jit: error initializing jq: {}", e);
            return 5;
        }
    };

    let bc = match jq.compile(filter) {
        Ok(bc) => bc,
        Err(e) => {
            eprintln!("jq-jit: compile error: {}", e);
            return 3;
        }
    };

    let ir = match bytecode_to_ir(&bc) {
        Ok(ir) => ir,
        Err(e) => {
            eprintln!("jq-jit: IR translation error: {:#}", e);
            return 3;
        }
    };

    // Step 2: Generate .o file via Cranelift ObjectModule
    let obj_bytes = match compile_to_object(&ir) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("jq-jit: object compilation error: {:#}", e);
            return 3;
        }
    };

    let obj_path = format!("{}.o", output);
    if let Err(e) = std::fs::write(&obj_path, &obj_bytes) {
        eprintln!("jq-jit: failed to write object file '{}': {}", obj_path, e);
        return 5;
    }

    // Step 3: Find libjq_jit.a
    let lib_dir = find_runtime_lib();

    // Step 4: Link the object file with the runtime
    let native_libs = env!("NATIVE_STATIC_LIBS");
    let mut cmd = std::process::Command::new("cc");
    cmd.args(["-o", output, &obj_path]);
    cmd.arg(format!("-L{}", lib_dir));
    cmd.arg("-ljq_jit");

    // Add native static libs required by the Rust runtime
    for flag in native_libs.split_whitespace() {
        cmd.arg(flag);
    }

    // libjq and oniguruma — libjq_jit.a includes modules that reference
    // these symbols. The linker may or may not need them depending on what
    // gets pulled in, but including them is harmless when not needed.
    cmd.args(["-ljq", "-lonig"]);

    // Homebrew library path on macOS
    if cfg!(target_os = "macos") {
        if std::path::Path::new("/opt/homebrew/lib").exists() {
            cmd.arg("-L/opt/homebrew/lib");
        } else if std::path::Path::new("/usr/local/lib").exists() {
            cmd.arg("-L/usr/local/lib");
        }
    }

    eprintln!("jq-jit: linking: {:?}", cmd);

    let status = match cmd.status() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("jq-jit: failed to run cc: {}", e);
            // Clean up .o file
            std::fs::remove_file(&obj_path).ok();
            return 5;
        }
    };

    // Step 5: Clean up .o file
    std::fs::remove_file(&obj_path).ok();

    if status.success() {
        eprintln!("jq-jit: compiled to: {}", output);
        0
    } else {
        eprintln!("jq-jit: linking failed (exit code: {:?})", status.code());
        1
    }
}

/// Find the directory containing libjq_jit.a for AOT linking.
fn find_runtime_lib() -> String {
    // 1. Explicit env var override
    if let Ok(dir) = std::env::var("JQ_JIT_LIB_DIR") {
        return dir;
    }

    // 2. Next to the current executable (works after `cargo install`)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let lib = dir.join("libjq_jit.a");
            if lib.exists() {
                return dir.to_string_lossy().to_string();
            }
        }
    }

    // 3. target/release relative to CWD (dev workflow)
    let release_dir = "target/release";
    if std::path::Path::new(release_dir)
        .join("libjq_jit.a")
        .exists()
    {
        return release_dir.to_string();
    }

    // 4. Fallback to current directory
    ".".to_string()
}

// ---------------------------------------------------------------------------
// Core pipeline: compile filter, run on input(s)
// ---------------------------------------------------------------------------

/// Enum that wraps either a JitFilter or a CachedFilter so we can use a single
/// execution path regardless of how the filter was compiled.
enum CompiledFilter {
    Jit(jq_jit::codegen::JitFilter),
    Cached(jq_jit::cache::CachedFilter),
}

impl CompiledFilter {
    fn execute(&self, input: &Value) -> Vec<Value> {
        match self {
            CompiledFilter::Jit(f) => f.execute(input),
            CompiledFilter::Cached(f) => f.execute(input),
        }
    }
}

fn run(opts: &CliOptions) -> i32 {
    // Step 0: Build effective filter with --arg/--argjson/--slurpfile bindings
    let effective_filter = build_effective_filter(opts);

    // Step 1: Compile the filter (with optional cache)
    let filter: CompiledFilter = if opts.no_cache {
        // --no-cache: skip cache entirely, use normal JIT path
        match compile_jit(&effective_filter) {
            Ok(f) => CompiledFilter::Jit(f),
            Err(code) => return code,
        }
    } else {
        // Try cache first
        let cache = match &opts.cache_dir {
            Some(dir) => FilterCache::with_dir(std::path::PathBuf::from(dir)),
            None => FilterCache::new(),
        };

        if let Some(cached) = cache.load(&effective_filter) {
            CompiledFilter::Cached(cached)
        } else {
            // Cache miss: compile via libjq → IR, then try to cache
            let mut jq = match JqState::new() {
                Ok(jq) => jq,
                Err(e) => {
                    eprintln!("jq-jit: error initializing jq: {}", e);
                    return 5;
                }
            };

            let bc = match jq.compile(&effective_filter) {
                Ok(bc) => bc,
                Err(e) => {
                    eprintln!("jq-jit: compile error: {}", e);
                    return 3;
                }
            };

            let ir = match bytecode_to_ir(&bc) {
                Ok(ir) => ir,
                Err(e) => {
                    eprintln!("jq-jit: IR translation error: {:#}", e);
                    return 3;
                }
            };

            // Try to compile and store in cache
            match cache.compile_and_store(&effective_filter, &ir) {
                Ok(cached) => CompiledFilter::Cached(cached),
                Err(_) => {
                    // Cache store failed: fallback to normal JIT path
                    match compile_expr(&ir) {
                        Ok((jit_filter, _)) => CompiledFilter::Jit(jit_filter),
                        Err(e) => {
                            eprintln!("jq-jit: JIT compilation error: {}", e);
                            return 3;
                        }
                    }
                }
            }
        }
    };

    // Build output options once from CliOptions
    let out_opts = opts.output_opts();

    // Helper: execute filter and output results, tracking exit-status state.
    let execute_and_output =
        |input_val: &Value,
         filter: &CompiledFilter,
         out_opts: &OutputOptions,
         last_is_false_or_null: &mut bool,
         had_output: &mut bool| {
            let results = filter.execute(input_val);
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

    // Step 2+3: Read input(s) and execute filter (streaming where possible)
    let mut last_is_false_or_null = false;
    let mut had_output = false;

    if opts.null_input {
        // -n mode: run filter on null
        execute_and_output(&Value::Null, &filter, &out_opts, &mut last_is_false_or_null, &mut had_output);
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
            execute_and_output(&input_val, &filter, &out_opts, &mut last_is_false_or_null, &mut had_output);
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
                execute_and_output(&input_val, &filter, &out_opts, &mut last_is_false_or_null, &mut had_output);
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
            execute_and_output(&arr, &filter, &out_opts, &mut last_is_false_or_null, &mut had_output);
        } else {
            // Stream: parse one, execute one, output one
            for result in input::stream_json_values(reader) {
                match result {
                    Ok(input_val) => {
                        execute_and_output(&input_val, &filter, &out_opts, &mut last_is_false_or_null, &mut had_output);
                    }
                    Err(e) => {
                        eprintln!("jq-jit: {}", e);
                        return 2;
                    }
                }
            }
        }
    }

    // Step 4: Exit status
    if opts.exit_status {
        if !had_output || last_is_false_or_null {
            return 1;
        }
    }

    0
}

/// Compile a filter using the normal JIT path (no caching).
fn compile_jit(filter: &str) -> Result<jq_jit::codegen::JitFilter, i32> {
    let mut jq = match JqState::new() {
        Ok(jq) => jq,
        Err(e) => {
            eprintln!("jq-jit: error initializing jq: {}", e);
            return Err(5);
        }
    };

    let bc = match jq.compile(filter) {
        Ok(bc) => bc,
        Err(e) => {
            eprintln!("jq-jit: compile error: {}", e);
            return Err(3);
        }
    };

    let ir = match bytecode_to_ir(&bc) {
        Ok(ir) => ir,
        Err(e) => {
            eprintln!("jq-jit: IR translation error: {:#}", e);
            return Err(3);
        }
    };

    match compile_expr(&ir) {
        Ok((jit_filter, _)) => Ok(jit_filter),
        Err(e) => {
            eprintln!("jq-jit: JIT compilation error: {}", e);
            Err(3)
        }
    }
}

/// Build the effective filter string by prepending --arg/--argjson/--slurpfile bindings.
///
/// For `--arg name value`:   `"value" as $name | <filter>`
/// For `--argjson name val`: `val as $name | <filter>`
/// For `--slurpfile name f`: `[<file contents>] as $name | <filter>`
fn build_effective_filter(opts: &CliOptions) -> String {
    let mut prefix = String::new();

    // --arg and --argjson bindings
    for binding in &opts.arg_bindings {
        prefix.push_str(&binding.json_value);
        prefix.push_str(" as $");
        prefix.push_str(&binding.name);
        prefix.push_str(" | ");
    }

    // --slurpfile bindings
    for (name, filepath) in &opts.slurpfile_bindings {
        match std::fs::read_to_string(filepath) {
            Ok(content) => {
                // Parse file contents as JSON values and wrap in array
                let trimmed = content.trim();
                // Build a JSON array from the file contents
                // Try to parse as a single value first
                if trimmed.starts_with('[') || trimmed.starts_with('{') || trimmed.starts_with('"')
                    || trimmed == "null" || trimmed == "true" || trimmed == "false"
                    || trimmed.chars().next().map_or(false, |c| c.is_ascii_digit() || c == '-')
                {
                    // Wrap in array: [content]
                    prefix.push('[');
                    prefix.push_str(trimmed);
                    prefix.push(']');
                } else {
                    prefix.push_str("[]");
                }
                prefix.push_str(" as $");
                prefix.push_str(name);
                prefix.push_str(" | ");
            }
            Err(e) => {
                eprintln!("jq-jit: --slurpfile '{}': {}", filepath, e);
                // Push empty array as fallback
                prefix.push_str("[] as $");
                prefix.push_str(name);
                prefix.push_str(" | ");
            }
        }
    }

    format!("{}{}", prefix, opts.filter)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let opts = match parse_args() {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("jq-jit: {}", e);
            process::exit(2);
        }
    };

    if opts.show_help {
        print_help();
        process::exit(0);
    }

    if opts.show_version {
        print_version();
        process::exit(0);
    }

    if opts.clear_cache {
        let cache = match &opts.cache_dir {
            Some(dir) => FilterCache::with_dir(std::path::PathBuf::from(dir)),
            None => FilterCache::new(),
        };
        match cache.clear() {
            Ok(()) => {
                eprintln!("jq-jit: cache cleared");
                process::exit(0);
            }
            Err(e) => {
                eprintln!("jq-jit: error clearing cache: {}", e);
                process::exit(1);
            }
        }
    }

    // AOT compilation mode: compile filter to standalone binary and exit
    if let Some(ref output) = opts.compile_output {
        process::exit(compile_aot(&opts.filter, output));
    }

    let code = run(&opts);
    process::exit(code);
}
