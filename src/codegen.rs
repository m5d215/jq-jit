//! CPS IR → Cranelift CLIF code generation.
//!
//! This module translates the [`Expr`] tree (from [`crate::cps_ir`]) into
//! Cranelift CLIF IR, JIT-compiles it, and returns a callable function pointer.
//!
//! # JIT function signature (Phase 4: callback-based)
//!
//! ```text
//! fn jit_filter(input: *const Value, callback: extern "C" fn(*const Value, *mut u8), ctx: *mut u8, literals: *const *const Value)
//! ```
//!
//! - `input`: pointer to the input JSON value (caller-owned)
//! - `callback`: function pointer called once per output value
//! - `ctx`: opaque user data passed to the callback (e.g., `*mut Vec<Value>`)
//! - `literals`: pointer to table of pre-allocated literal Value pointers
//!
//! For 1→1 filters (Phase 1-3), callback is called exactly once.
//! For generators (Phase 4+), callback may be called 0 or N times.
//!
//! # Code generation strategy
//!
//! All `Value`s are handled as pointers (`i64` / pointer type).  Intermediate
//! values are stored in Cranelift `StackSlot`s (16 bytes each, 8-byte aligned).
//! Runtime helper functions (`rt_add`, `rt_sub`, etc.) are called to perform
//! operations on `Value` pointers.
//!
//! - `Expr::Input` → returns the `input` parameter pointer directly
//! - `Expr::Literal(Num(n))` → writes tag + f64 payload to a StackSlot, returns its address
//! - `Expr::Literal(Str(s))` → handled via a pre-allocated Vec of Values (indexed by slot)
//! - `Expr::BinOp` → recursively generates lhs/rhs, calls `rt_add`/etc., returns output slot addr
//! - `Expr::Index` → recursively generates expr/key, calls `rt_index`, returns output slot addr

use std::collections::HashMap;
use anyhow::{Context, Result};
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, BlockArg, InstBuilder, MemFlags, Signature, StackSlotData, StackSlotKind};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, Linkage, Module, default_libcall_names};
use cranelift_object::{ObjectBuilder, ObjectModule};

use crate::cps_ir::{BinOp, ClosureOp, Expr, Literal, UnaryOp};
use crate::runtime;
use crate::value::{TAG_BOOL, TAG_NULL, TAG_NUM, Value};

/// Phase 6-3: Type hint for type specialization.
///
/// Tracks the known type of a codegen expression result at compile time.
/// When the type is known (e.g., `Literal::Num` → `TypeHint::Num`),
/// we can generate specialized Cranelift instructions instead of calling
/// generic runtime helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TypeHint {
    /// Type is not known at compile time. Must use runtime dispatch.
    Unknown,
    /// Value is guaranteed to be a Num (f64). Can use f64 instructions directly.
    Num,
}

/// Holds pre-allocated literal values and the CLIF parameter for indirect referencing.
/// Rc-containing types (Str, Arr, Obj, Error) are stored here and loaded via
/// the `clif_param` pointer at runtime, instead of embedding heap addresses directly.
struct LiteralPool {
    /// Pre-allocated literal Values that JIT code references indirectly.
    values: Vec<Box<Value>>,
    /// CLIF Value for the `literals: *const *const Value` function parameter.
    /// Used by codegen_literal to generate `load` instructions.
    clif_param: cranelift_codegen::ir::Value,
}

/// A JIT-compiled jq filter, ready to execute.
pub struct JitFilter {
    /// The compiled function pointer.
    ///
    /// Signature: `fn(input: *const Value, callback: extern "C" fn(*const Value, *mut u8), ctx: *mut u8, literals: *const *const Value)`
    fn_ptr: *const u8,

    /// Keeps the JITModule alive so the compiled code remains valid.
    _module: JITModule,

    /// Pre-allocated literal Values that the JIT code references by pointer.
    /// These must live as long as the JIT code can be called.
    ///
    /// Each Value is individually Box-allocated so that its address remains
    /// stable even when more literals are added (unlike Vec<Value> which can
    /// reallocate and invalidate previously taken addresses).
    _literals: Vec<Box<Value>>,
}

/// Callback function that collects JIT output values into a Vec.
///
/// Called by JIT code via `call_indirect`.  `value_ptr` points to a Value
/// in JIT stack memory (raw bytes, no Rc refcount bump).  We clone it to
/// get proper Rc refcounting and push it into the results Vec.
extern "C" fn collect_callback(value_ptr: *const Value, ctx: *mut u8) {
    assert!(!value_ptr.is_null() && !ctx.is_null());
    let results = unsafe { &mut *(ctx as *mut Vec<Value>) };
    // Clone to get proper Rc refcounting (JIT code writes raw bytes via memcpy).
    let value = unsafe { (*value_ptr).clone() };
    results.push(value);
}

impl JitFilter {
    /// Execute the JIT-compiled filter.
    ///
    /// Returns a `Vec<Value>` of all output values produced by the filter.
    /// For 1→1 filters (Phase 1-3), this will contain exactly one value.
    /// For generators (Phase 4+), this may contain 0 or more values.
    ///
    /// # Safety
    ///
    /// The `fn_ptr` was produced by Cranelift's JIT compiler from verified
    /// CLIF IR.  The caller must provide a valid `input` Value.  The function
    /// calls the callback once per output value.
    pub fn execute(&self, input: &Value) -> Vec<Value> {
        let mut results: Vec<Value> = Vec::new();

        // Build literal pointer table
        let literal_ptrs: Vec<*const Value> = self._literals.iter()
            .map(|b| &**b as *const Value)
            .collect();

        type JitFn = fn(*const Value, extern "C" fn(*const Value, *mut u8), *mut u8, *const *const Value);
        let fn_typed: JitFn = unsafe {
            std::mem::transmute::<*const u8, JitFn>(self.fn_ptr)
        };

        fn_typed(
            input as *const Value,
            collect_callback,
            &mut results as *mut Vec<Value> as *mut u8,
            literal_ptrs.as_ptr(),
        );

        // Phase 9-8: Filter out Value::Error at toplevel (jq outputs errors to
        // stderr and skips them from stdout).  We print error messages to stderr
        // and remove them from the result set.
        results.retain(|v| {
            if let Value::Error(msg) = v {
                eprintln!("jq: error (at <stdin>:0): {}", msg);
                false
            } else {
                true
            }
        });

        results
    }
}

/// Register all runtime helper function symbols into the JIT builder.
fn register_jit_symbols(jit_builder: &mut JITBuilder) {
    // Binary (3-ptr): rt_*(out, a, b)
    jit_builder.symbol("rt_add", runtime::rt_add as *const u8);
    jit_builder.symbol("rt_sub", runtime::rt_sub as *const u8);
    jit_builder.symbol("rt_mul", runtime::rt_mul as *const u8);
    jit_builder.symbol("rt_div", runtime::rt_div as *const u8);
    jit_builder.symbol("rt_mod", runtime::rt_mod as *const u8);
    jit_builder.symbol("rt_index", runtime::rt_index as *const u8);
    // Comparison (3-ptr): rt_*(out, a, b)
    jit_builder.symbol("rt_eq", runtime::rt_eq as *const u8);
    jit_builder.symbol("rt_ne", runtime::rt_ne as *const u8);
    jit_builder.symbol("rt_lt", runtime::rt_lt as *const u8);
    jit_builder.symbol("rt_gt", runtime::rt_gt as *const u8);
    jit_builder.symbol("rt_le", runtime::rt_le as *const u8);
    jit_builder.symbol("rt_ge", runtime::rt_ge as *const u8);
    // Unary (2-ptr): rt_*(out, v)
    jit_builder.symbol("rt_length", runtime::rt_length as *const u8);
    jit_builder.symbol("rt_type", runtime::rt_type as *const u8);
    jit_builder.symbol("rt_tostring", runtime::rt_tostring as *const u8);
    jit_builder.symbol("rt_tonumber", runtime::rt_tonumber as *const u8);
    jit_builder.symbol("rt_keys", runtime::rt_keys as *const u8);
    jit_builder.symbol("rt_negate", runtime::rt_negate as *const u8);
    jit_builder.symbol("rt_make_error", runtime::rt_make_error as *const u8);
    jit_builder.symbol("rt_extract_error", runtime::rt_extract_error as *const u8);
    // Truthy check (Phase 3): fn(v: ptr) -> i32
    jit_builder.symbol("rt_is_truthy", runtime::rt_is_truthy as *const u8);
    // Error check (Phase 3 try-catch): fn(v: ptr) -> i32
    jit_builder.symbol("rt_is_error", runtime::rt_is_error as *const u8);
    // Phase 10-5: Try-catch generator wrapper callback
    jit_builder.symbol("rt_try_callback_wrapper", runtime::rt_try_callback_wrapper as *const u8);
    // Phase 12: Limit/Skip generator wrapper callbacks
    jit_builder.symbol("rt_limit_callback_wrapper", runtime::rt_limit_callback_wrapper as *const u8);
    jit_builder.symbol("rt_skip_callback_wrapper", runtime::rt_skip_callback_wrapper as *const u8);
    // Each (.[] iteration) helpers (Phase 4-3)
    jit_builder.symbol("rt_iter_length", runtime::rt_iter_length as *const u8);
    jit_builder.symbol("rt_iter_get", runtime::rt_iter_get as *const u8);
    jit_builder.symbol("rt_is_iterable", runtime::rt_is_iterable as *const u8);
    jit_builder.symbol("rt_iter_error", runtime::rt_iter_error as *const u8);
    // Phase 6-2: Object iteration optimization
    jit_builder.symbol("rt_iter_prepare", runtime::rt_iter_prepare as *const u8);

    // Collect helpers (Phase 5-1: array constructor)
    jit_builder.symbol("rt_collect_init", runtime::rt_collect_init as *const u8);
    jit_builder.symbol("rt_collect_append", runtime::rt_collect_append as *const u8);
    jit_builder.symbol("rt_collect_append_raw", runtime::rt_collect_append_raw as *const u8);

    // Phase 5-2: Additional unary builtins
    jit_builder.symbol("rt_sort", runtime::rt_sort as *const u8);
    jit_builder.symbol("rt_keys_unsorted", runtime::rt_keys_unsorted as *const u8);
    jit_builder.symbol("rt_floor", runtime::rt_floor as *const u8);
    jit_builder.symbol("rt_ceil", runtime::rt_ceil as *const u8);
    jit_builder.symbol("rt_round", runtime::rt_round as *const u8);
    jit_builder.symbol("rt_fabs", runtime::rt_fabs as *const u8);
    jit_builder.symbol("rt_explode", runtime::rt_explode as *const u8);
    jit_builder.symbol("rt_implode", runtime::rt_implode as *const u8);

    // Phase 5-2: Binary builtins (nargs=2)
    jit_builder.symbol("rt_split", runtime::rt_split as *const u8);
    jit_builder.symbol("rt_has", runtime::rt_has as *const u8);
    jit_builder.symbol("rt_startswith", runtime::rt_startswith as *const u8);
    jit_builder.symbol("rt_endswith", runtime::rt_endswith as *const u8);
    jit_builder.symbol("rt_join", runtime::rt_join as *const u8);

    // Phase 5-3: jq-defined functions as direct runtime calls
    jit_builder.symbol("rt_jq_add", runtime::rt_jq_add as *const u8);
    jit_builder.symbol("rt_reverse", runtime::rt_reverse as *const u8);
    jit_builder.symbol("rt_to_entries", runtime::rt_to_entries as *const u8);
    jit_builder.symbol("rt_from_entries", runtime::rt_from_entries as *const u8);
    jit_builder.symbol("rt_unique", runtime::rt_unique as *const u8);
    jit_builder.symbol("rt_ascii_downcase", runtime::rt_ascii_downcase as *const u8);
    jit_builder.symbol("rt_ascii_upcase", runtime::rt_ascii_upcase as *const u8);

    // Phase 8-1: Missing BinOp/UnaryOp runtime functions
    jit_builder.symbol("rt_contains", runtime::rt_contains as *const u8);
    jit_builder.symbol("rt_ltrimstr", runtime::rt_ltrimstr as *const u8);
    jit_builder.symbol("rt_rtrimstr", runtime::rt_rtrimstr as *const u8);
    jit_builder.symbol("rt_in", runtime::rt_in as *const u8);
    jit_builder.symbol("rt_min", runtime::rt_min as *const u8);
    jit_builder.symbol("rt_max", runtime::rt_max as *const u8);
    jit_builder.symbol("rt_flatten", runtime::rt_flatten as *const u8);
    jit_builder.symbol("rt_flatten_depth", runtime::rt_flatten_depth as *const u8);

    // Phase 8-2: Optional index access
    jit_builder.symbol("rt_index_opt", runtime::rt_index_opt as *const u8);

    // Phase 8-4: Object insert
    jit_builder.symbol("rt_obj_insert", runtime::rt_obj_insert as *const u8);

    // Phase 9-3: Format string functions
    jit_builder.symbol("rt_format_base64", runtime::rt_format_base64 as *const u8);
    jit_builder.symbol("rt_format_base64d", runtime::rt_format_base64d as *const u8);
    jit_builder.symbol("rt_format_html", runtime::rt_format_html as *const u8);
    jit_builder.symbol("rt_format_uri", runtime::rt_format_uri as *const u8);
    jit_builder.symbol("rt_format_urid", runtime::rt_format_urid as *const u8);
    jit_builder.symbol("rt_format_csv", runtime::rt_format_csv as *const u8);
    jit_builder.symbol("rt_format_tsv", runtime::rt_format_tsv as *const u8);
    jit_builder.symbol("rt_format_json", runtime::rt_format_json as *const u8);
    jit_builder.symbol("rt_format_sh", runtime::rt_format_sh as *const u8);
    jit_builder.symbol("rt_range", runtime::rt_range as *const u8);
    jit_builder.symbol("rt_range_step", runtime::rt_range_step as *const u8);
    jit_builder.symbol("rt_recurse", runtime::rt_recurse as *const u8);

    // Phase 9-6: Remaining builtin functions
    jit_builder.symbol("rt_any", runtime::rt_any as *const u8);
    jit_builder.symbol("rt_all", runtime::rt_all as *const u8);
    jit_builder.symbol("rt_indices", runtime::rt_indices as *const u8);
    jit_builder.symbol("rt_str_index", runtime::rt_str_index as *const u8);
    jit_builder.symbol("rt_str_rindex", runtime::rt_str_rindex as *const u8);
    jit_builder.symbol("rt_inside", runtime::rt_inside as *const u8);
    jit_builder.symbol("rt_tojson", runtime::rt_tojson as *const u8);
    jit_builder.symbol("rt_fromjson", runtime::rt_fromjson as *const u8);
    jit_builder.symbol("rt_getpath", runtime::rt_getpath as *const u8);
    jit_builder.symbol("rt_setpath", runtime::rt_setpath as *const u8);
    jit_builder.symbol("rt_delpaths", runtime::rt_delpaths as *const u8);
    jit_builder.symbol("rt_debug", runtime::rt_debug as *const u8);
    jit_builder.symbol("rt_env", runtime::rt_env as *const u8);
    jit_builder.symbol("rt_builtins", runtime::rt_builtins as *const u8);
    jit_builder.symbol("rt_infinite", runtime::rt_infinite as *const u8);
    jit_builder.symbol("rt_nan", runtime::rt_nan as *const u8);
    jit_builder.symbol("rt_isinfinite", runtime::rt_isinfinite as *const u8);
    jit_builder.symbol("rt_isnan", runtime::rt_isnan as *const u8);
    jit_builder.symbol("rt_isnormal", runtime::rt_isnormal as *const u8);

    // Math unary functions
    jit_builder.symbol("rt_sqrt", runtime::rt_sqrt as *const u8);
    jit_builder.symbol("rt_sin", runtime::rt_sin as *const u8);
    jit_builder.symbol("rt_cos", runtime::rt_cos as *const u8);
    jit_builder.symbol("rt_tan", runtime::rt_tan as *const u8);
    jit_builder.symbol("rt_asin", runtime::rt_asin as *const u8);
    jit_builder.symbol("rt_acos", runtime::rt_acos as *const u8);
    jit_builder.symbol("rt_atan", runtime::rt_atan as *const u8);
    jit_builder.symbol("rt_exp", runtime::rt_exp as *const u8);
    jit_builder.symbol("rt_exp2", runtime::rt_exp2 as *const u8);
    jit_builder.symbol("rt_exp10", runtime::rt_exp10 as *const u8);
    jit_builder.symbol("rt_log", runtime::rt_log as *const u8);
    jit_builder.symbol("rt_log2", runtime::rt_log2 as *const u8);
    jit_builder.symbol("rt_log10", runtime::rt_log10 as *const u8);
    jit_builder.symbol("rt_cbrt", runtime::rt_cbrt as *const u8);
    jit_builder.symbol("rt_significand", runtime::rt_significand as *const u8);
    jit_builder.symbol("rt_exponent", runtime::rt_exponent as *const u8);
    jit_builder.symbol("rt_logb", runtime::rt_logb as *const u8);
    jit_builder.symbol("rt_nearbyint", runtime::rt_nearbyint as *const u8);
    jit_builder.symbol("rt_trunc", runtime::rt_trunc as *const u8);
    jit_builder.symbol("rt_rint", runtime::rt_rint as *const u8);
    jit_builder.symbol("rt_j0", runtime::rt_j0 as *const u8);
    jit_builder.symbol("rt_j1", runtime::rt_j1 as *const u8);
    // Math binary functions
    jit_builder.symbol("rt_pow", runtime::rt_pow as *const u8);
    jit_builder.symbol("rt_atan2", runtime::rt_atan2 as *const u8);
    jit_builder.symbol("rt_drem", runtime::rt_drem as *const u8);
    jit_builder.symbol("rt_ldexp", runtime::rt_ldexp as *const u8);
    jit_builder.symbol("rt_scalb", runtime::rt_scalb as *const u8);
    jit_builder.symbol("rt_scalbln", runtime::rt_scalbln as *const u8);
    // bsearch
    jit_builder.symbol("rt_bsearch", runtime::rt_bsearch as *const u8);
    // Transpose, utf8bytelength, toboolean, trim, date/time
    jit_builder.symbol("rt_transpose", runtime::rt_transpose as *const u8);
    jit_builder.symbol("rt_utf8bytelength", runtime::rt_utf8bytelength as *const u8);
    jit_builder.symbol("rt_toboolean", runtime::rt_toboolean as *const u8);
    jit_builder.symbol("rt_trim", runtime::rt_trim as *const u8);
    jit_builder.symbol("rt_ltrim", runtime::rt_ltrim as *const u8);
    jit_builder.symbol("rt_rtrim", runtime::rt_rtrim as *const u8);
    jit_builder.symbol("rt_gmtime", runtime::rt_gmtime as *const u8);
    jit_builder.symbol("rt_mktime", runtime::rt_mktime as *const u8);
    jit_builder.symbol("rt_now", runtime::rt_now as *const u8);
    jit_builder.symbol("rt_strftime", runtime::rt_strftime as *const u8);
    jit_builder.symbol("rt_strptime", runtime::rt_strptime as *const u8);
    jit_builder.symbol("rt_strflocaltime", runtime::rt_strflocaltime as *const u8);

    // Phase 9-1: Closure-based array operations
    jit_builder.symbol("rt_sort_by_keys", runtime::rt_sort_by_keys as *const u8);
    jit_builder.symbol("rt_group_by_keys", runtime::rt_group_by_keys as *const u8);
    jit_builder.symbol("rt_unique_by_keys", runtime::rt_unique_by_keys as *const u8);
    jit_builder.symbol("rt_min_by_keys", runtime::rt_min_by_keys as *const u8);
    jit_builder.symbol("rt_max_by_keys", runtime::rt_max_by_keys as *const u8);

    // Phase 10-2: Regex functions
    jit_builder.symbol("rt_regex_test", runtime::rt_regex_test as *const u8);
    jit_builder.symbol("rt_regex_match", runtime::rt_regex_match as *const u8);
    jit_builder.symbol("rt_regex_capture", runtime::rt_regex_capture as *const u8);
    jit_builder.symbol("rt_regex_scan", runtime::rt_regex_scan as *const u8);
    jit_builder.symbol("rt_regex_sub", runtime::rt_regex_sub as *const u8);
    jit_builder.symbol("rt_regex_gsub", runtime::rt_regex_gsub as *const u8);

    // Phase 9-5 / 10-3: Path operations and update operators
    jit_builder.symbol("rt_path_of", runtime::rt_path_of as *const u8);
    jit_builder.symbol("rt_update", runtime::rt_update as *const u8);
    jit_builder.symbol("rt_slice_assign", runtime::rt_slice_assign as *const u8);
    jit_builder.symbol("rt_delpath", runtime::rt_delpath as *const u8);
}

/// Compile a CPS IR expression into a JIT-executable filter.
///
/// Returns a [`JitFilter`] that can be called with `execute()`.
/// Also returns the CLIF IR text for debugging.
pub fn compile_expr(expr: &Expr) -> Result<(JitFilter, String)> {

    // 1. Configure ISA for host machine
    let mut flag_builder = settings::builder();
    flag_builder
        .set("use_colocated_libcalls", "false")
        .context("setting use_colocated_libcalls")?;
    flag_builder
        .set("is_pic", "false")
        .context("setting is_pic")?;

    let isa_builder = cranelift_native::builder()
        .map_err(|e| anyhow::anyhow!("ISA detection failed: {}", e))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .context("finishing ISA")?;

    let ptr_ty = isa.pointer_type();

    // 2. Create JITModule with runtime symbols registered
    let mut jit_builder = JITBuilder::with_isa(isa, default_libcall_names());
    register_jit_symbols(&mut jit_builder);
    let mut module = JITModule::new(jit_builder);
    let mut ctx = module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();

    // 3. Build filter (generic over Module)
    let (func_id, literals, clif_text) = build_filter(&mut module, &mut ctx, &mut func_ctx, expr, ptr_ty)?;

    // 4. Finalize (JIT-specific: resolve relocations + get function pointer)
    module
        .finalize_definitions()
        .context("finalizing definitions")?;
    let code_ptr = module.get_finalized_function(func_id);

    Ok((
        JitFilter {
            fn_ptr: code_ptr,
            _module: module,
            _literals: literals,
        },
        clif_text,
    ))
}

/// Compile a CPS IR expression into a shared library (.dylib/.so).
///
/// The shared library exports a `jit_filter` function with the same signature
/// as the JIT-compiled version. Runtime symbols (rt_*) are left as undefined
/// references, resolved at dlopen time from the host process.
///
/// Returns the literal values needed for the filter to execute (must be
/// reconstructed at load time and passed as the 4th parameter).
pub fn compile_to_shared_object(expr: &Expr, output_path: &std::path::Path) -> Result<Vec<Box<Value>>> {
    // 1. Configure ISA with PIC enabled (required for shared libraries)
    let mut flag_builder = settings::builder();
    flag_builder
        .set("use_colocated_libcalls", "false")
        .context("setting use_colocated_libcalls")?;
    flag_builder
        .set("is_pic", "true")
        .context("setting is_pic")?;

    let isa_builder = cranelift_native::builder()
        .map_err(|e| anyhow::anyhow!("ISA detection failed: {}", e))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .context("finishing ISA")?;

    let ptr_ty = isa.pointer_type();

    // 2. Create ObjectModule (produces relocatable object file instead of JIT memory)
    let obj_builder = ObjectBuilder::new(isa, "jq_filter", default_libcall_names())
        .context("creating ObjectBuilder")?;
    let mut module = ObjectModule::new(obj_builder);
    let mut ctx = module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();

    // 3. Build filter (same codegen as JIT, thanks to Module trait abstraction)
    let (_func_id, literals, _clif) = build_filter(&mut module, &mut ctx, &mut func_ctx, expr, ptr_ty)?;

    // 4. Emit object file (.o)
    let product = module.finish();
    let obj_bytes = product.emit()
        .map_err(|e| anyhow::anyhow!("failed to emit object file: {}", e))?;

    let obj_path = output_path.with_extension("o");
    std::fs::write(&obj_path, &obj_bytes)
        .context("writing object file")?;

    // 5. Link into shared library using cc
    let shared_args: Vec<&str> = if cfg!(target_os = "macos") {
        // -undefined dynamic_lookup: allow unresolved symbols (rt_* functions)
        // to be resolved at dlopen time from the host process
        vec!["-shared", "-undefined", "dynamic_lookup"]
    } else {
        // Linux: shared libraries allow undefined symbols by default
        vec!["-shared"]
    };

    let status = std::process::Command::new("cc")
        .args(&shared_args)
        .arg("-o")
        .arg(output_path)
        .arg(&obj_path)
        .status()
        .context("failed to run cc")?;

    if !status.success() {
        // Clean up .o even on failure
        std::fs::remove_file(&obj_path).ok();
        return Err(anyhow::anyhow!("cc -shared failed with exit code: {}", status));
    }

    // 6. Clean up intermediate .o file
    std::fs::remove_file(&obj_path).ok();

    Ok(literals)
}

/// Compile a filter expression to a relocatable object file (.o) for AOT compilation.
///
/// The object file contains three symbols:
/// - `jit_filter`: the compiled filter function (via `build_filter`)
/// - `jq_literal_data`: serialized literal bytes in .rodata
/// - `main`: entry point that calls `aot_run(filter_fn, literal_data, literal_data_len, argc, argv)`
///
/// The `aot_run` symbol is left as an undefined import, resolved at link time
/// from `libjq_jit.a` (the static library built by cargo).
pub fn compile_to_object(expr: &Expr) -> Result<Vec<u8>> {
    // 1. Configure ISA with PIC enabled (required for ObjectModule)
    let mut flag_builder = settings::builder();
    flag_builder
        .set("use_colocated_libcalls", "false")
        .context("setting use_colocated_libcalls")?;
    flag_builder
        .set("is_pic", "true")
        .context("setting is_pic")?;

    let isa_builder = cranelift_native::builder()
        .map_err(|e| anyhow::anyhow!("ISA detection failed: {}", e))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .context("finishing ISA")?;

    let ptr_ty = isa.pointer_type();

    // 2. Create ObjectModule
    let obj_builder = ObjectBuilder::new(isa, "jq_aot_filter", default_libcall_names())
        .context("creating ObjectBuilder")?;
    let mut module = ObjectModule::new(obj_builder);
    let mut ctx = module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();

    // 3. Build the filter (jit_filter function) — same codegen as JIT/cache path
    let (filter_func_id, literals, _clif) =
        build_filter(&mut module, &mut ctx, &mut func_ctx, expr, ptr_ty)?;

    // 4. Serialize literals and embed as a data section (.rodata)
    let literal_bytes = crate::cache::serialize_literals(&literals);

    let data_id = module
        .declare_data("jq_literal_data", Linkage::Local, false, false)
        .context("declaring literal data")?;
    let mut data_desc = DataDescription::new();
    data_desc.define(literal_bytes.clone().into_boxed_slice());
    module
        .define_data(data_id, &data_desc)
        .map_err(|e| anyhow::anyhow!("defining literal data: {}", e))?;

    // 5. Declare aot_run as an imported function (resolved from libjq_jit.a at link time)
    let mut aot_run_sig = module.make_signature();
    aot_run_sig.params.push(AbiParam::new(ptr_ty));       // filter_fn: *const u8
    aot_run_sig.params.push(AbiParam::new(ptr_ty));       // literal_data: *const u8
    aot_run_sig.params.push(AbiParam::new(types::I64));   // literal_data_len: u64
    aot_run_sig.params.push(AbiParam::new(types::I32));   // argc: i32
    aot_run_sig.params.push(AbiParam::new(ptr_ty));       // argv: *const *const i8
    aot_run_sig.returns.push(AbiParam::new(types::I32));  // -> i32
    let aot_run_id = module
        .declare_function("aot_run", Linkage::Import, &aot_run_sig)
        .context("declaring aot_run")?;

    // 6. Generate main() function
    let mut main_sig = module.make_signature();
    main_sig.params.push(AbiParam::new(types::I32));  // argc
    main_sig.params.push(AbiParam::new(ptr_ty));      // argv
    main_sig.returns.push(AbiParam::new(types::I32)); // return i32
    let main_id = module
        .declare_function("main", Linkage::Export, &main_sig)
        .context("declaring main")?;

    // Build main() CLIF IR:
    //   int main(int argc, char** argv) {
    //       return aot_run(&jit_filter, &jq_literal_data, sizeof(jq_literal_data), argc, argv);
    //   }
    ctx.func.signature = main_sig;
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let argc = builder.block_params(entry_block)[0];
        let argv = builder.block_params(entry_block)[1];

        // Get function address of jit_filter (same object file, colocated)
        let filter_fn_ref = module.declare_func_in_func(filter_func_id, builder.func);
        let filter_fn_addr = builder.ins().func_addr(ptr_ty, filter_fn_ref);

        // Get address of literal data section
        let literal_data_gv = module.declare_data_in_func(data_id, builder.func);
        let literal_data_addr = builder.ins().global_value(ptr_ty, literal_data_gv);

        // Literal data length as an i64 constant
        let literal_data_len = builder.ins().iconst(types::I64, literal_bytes.len() as i64);

        // Call aot_run(filter_fn, literal_data, literal_data_len, argc, argv)
        let aot_run_ref = module.declare_func_in_func(aot_run_id, builder.func);
        let call = builder.ins().call(
            aot_run_ref,
            &[filter_fn_addr, literal_data_addr, literal_data_len, argc, argv],
        );
        let result = builder.inst_results(call)[0];

        builder.ins().return_(&[result]);
        builder.finalize();
    }

    module
        .define_function(main_id, &mut ctx)
        .context("compiling main")?;
    module.clear_context(&mut ctx);

    // 7. Finish and emit the object file bytes
    let product = module.finish();
    let obj_bytes = product
        .emit()
        .map_err(|e| anyhow::anyhow!("failed to emit object file: {}", e))?;

    Ok(obj_bytes)
}

/// Build CLIF IR for a jq filter and define it in the given module.
/// Works with both JITModule (JIT path) and ObjectModule (AOT/cache path).
fn build_filter<M: Module>(
    module: &mut M,
    ctx: &mut cranelift_codegen::Context,
    func_ctx: &mut FunctionBuilderContext,
    expr: &Expr,
    ptr_ty: types::Type,
) -> Result<(cranelift_module::FuncId, Vec<Box<Value>>, String)> {
    // Define function signature: fn(input: *const Value, callback: fn ptr, ctx: *mut u8, literals: *const *const Value)
    ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // input
    ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // callback
    ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // ctx
    ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // literals

    let func_id = module
        .declare_function(
            "jit_filter",
            cranelift_module::Linkage::Export,
            &ctx.func.signature,
        )
        .context("declaring jit_filter")?;

    // Collect string/non-numeric literals that need heap allocation.
    // We pre-allocate them on the Rust side and pass their addresses into a pointer table.
    // Using Box<Value> so each literal has a stable address regardless of Vec growth.
    let mut literal_values: Vec<Box<Value>> = Vec::new();

    // Build CLIF IR
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, func_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let input_ptr = builder.block_params(entry)[0]; // *const Value
        let callback_ptr = builder.block_params(entry)[1]; // callback fn ptr
        let ctx_ptr = builder.block_params(entry)[2]; // *mut u8 (user context)
        let literals_param = builder.block_params(entry)[3]; // *const *const Value (literals table)

        let mut literals = LiteralPool {
            values: std::mem::take(&mut literal_values),
            clif_param: literals_param,
        };

        // Declare runtime function signatures.
        // Binary: fn(out: ptr, a: ptr, b: ptr)
        let mut rt_bin_sig = module.make_signature();
        rt_bin_sig.params.push(AbiParam::new(ptr_ty)); // out
        rt_bin_sig.params.push(AbiParam::new(ptr_ty)); // a
        rt_bin_sig.params.push(AbiParam::new(ptr_ty)); // b

        // Unary: fn(out: ptr, v: ptr)
        let mut rt_unary_sig = module.make_signature();
        rt_unary_sig.params.push(AbiParam::new(ptr_ty)); // out
        rt_unary_sig.params.push(AbiParam::new(ptr_ty)); // v

        // Truthy check: fn(v: ptr) -> i32
        let mut rt_truthy_sig = module.make_signature();
        rt_truthy_sig.params.push(AbiParam::new(ptr_ty)); // v
        rt_truthy_sig.returns.push(AbiParam::new(types::I32)); // 0 or 1

        // Helper macro-like closure for declaring binary runtime functions
        let declare_bin = |module: &mut M, name: &str| -> cranelift_module::FuncId {
            module
                .declare_function(name, cranelift_module::Linkage::Import, &rt_bin_sig)
                .unwrap()
        };
        let declare_unary = |module: &mut M, name: &str| -> cranelift_module::FuncId {
            module
                .declare_function(name, cranelift_module::Linkage::Import, &rt_unary_sig)
                .unwrap()
        };

        // Declare binary runtime functions
        let rt_add_id = declare_bin(&mut *module, "rt_add");
        let rt_sub_id = declare_bin(&mut *module, "rt_sub");
        let rt_mul_id = declare_bin(&mut *module, "rt_mul");
        let rt_div_id = declare_bin(&mut *module, "rt_div");
        let rt_mod_id = declare_bin(&mut *module, "rt_mod");
        let rt_index_id = declare_bin(&mut *module, "rt_index");
        let rt_eq_id = declare_bin(&mut *module, "rt_eq");
        let rt_ne_id = declare_bin(&mut *module, "rt_ne");
        let rt_lt_id = declare_bin(&mut *module, "rt_lt");
        let rt_gt_id = declare_bin(&mut *module, "rt_gt");
        let rt_le_id = declare_bin(&mut *module, "rt_le");
        let rt_ge_id = declare_bin(&mut *module, "rt_ge");

        // Declare unary runtime functions
        let rt_length_id = declare_unary(&mut *module, "rt_length");
        let rt_type_id = declare_unary(&mut *module, "rt_type");
        let rt_tostring_id = declare_unary(&mut *module, "rt_tostring");
        let rt_tonumber_id = declare_unary(&mut *module, "rt_tonumber");
        let rt_keys_id = declare_unary(&mut *module, "rt_keys");
        let rt_negate_id = declare_unary(&mut *module, "rt_negate");
        let rt_make_error_id = declare_unary(&mut *module, "rt_make_error");
        let rt_extract_error_id = declare_unary(&mut *module, "rt_extract_error");

        // Phase 5-2: Additional unary builtins
        let rt_sort_id = declare_unary(&mut *module, "rt_sort");
        let rt_keys_unsorted_id = declare_unary(&mut *module, "rt_keys_unsorted");
        let rt_floor_id = declare_unary(&mut *module, "rt_floor");
        let rt_ceil_id = declare_unary(&mut *module, "rt_ceil");
        let rt_round_id = declare_unary(&mut *module, "rt_round");
        let rt_fabs_id = declare_unary(&mut *module, "rt_fabs");
        let rt_explode_id = declare_unary(&mut *module, "rt_explode");
        let rt_implode_id = declare_unary(&mut *module, "rt_implode");

        // Phase 5-3: jq-defined functions as direct runtime calls
        let rt_jq_add_id = declare_unary(&mut *module, "rt_jq_add");
        let rt_reverse_id = declare_unary(&mut *module, "rt_reverse");
        let rt_to_entries_id = declare_unary(&mut *module, "rt_to_entries");
        let rt_from_entries_id = declare_unary(&mut *module, "rt_from_entries");
        let rt_unique_id = declare_unary(&mut *module, "rt_unique");
        let rt_ascii_downcase_id = declare_unary(&mut *module, "rt_ascii_downcase");
        let rt_ascii_upcase_id = declare_unary(&mut *module, "rt_ascii_upcase");

        // Phase 5-2: Binary builtins (nargs=2)
        let rt_split_id = declare_bin(&mut *module, "rt_split");
        let rt_has_id = declare_bin(&mut *module, "rt_has");
        let rt_startswith_id = declare_bin(&mut *module, "rt_startswith");
        let rt_endswith_id = declare_bin(&mut *module, "rt_endswith");
        let rt_join_id = declare_bin(&mut *module, "rt_join");

        // Phase 8-1: Missing binary builtins
        let rt_contains_id = declare_bin(&mut *module, "rt_contains");
        let rt_ltrimstr_id = declare_bin(&mut *module, "rt_ltrimstr");
        let rt_rtrimstr_id = declare_bin(&mut *module, "rt_rtrimstr");
        let rt_in_id = declare_bin(&mut *module, "rt_in");

        // Phase 8-1: Missing unary builtins
        let rt_min_id = declare_unary(&mut *module, "rt_min");
        let rt_max_id = declare_unary(&mut *module, "rt_max");
        let rt_flatten_id = declare_unary(&mut *module, "rt_flatten");
        let rt_flatten_depth_id = declare_bin(&mut *module, "rt_flatten_depth");

        // Phase 9-3: Format string functions
        let rt_format_base64_id = declare_unary(&mut *module, "rt_format_base64");
        let rt_format_base64d_id = declare_unary(&mut *module, "rt_format_base64d");
        let rt_format_html_id = declare_unary(&mut *module, "rt_format_html");
        let rt_format_uri_id = declare_unary(&mut *module, "rt_format_uri");
        let rt_format_urid_id = declare_unary(&mut *module, "rt_format_urid");
        let rt_format_csv_id = declare_unary(&mut *module, "rt_format_csv");
        let rt_format_tsv_id = declare_unary(&mut *module, "rt_format_tsv");
        let rt_format_json_id = declare_unary(&mut *module, "rt_format_json");
        let rt_format_sh_id = declare_unary(&mut *module, "rt_format_sh");

        // Phase 8-2: Optional index access
        let rt_index_opt_id = declare_bin(&mut *module, "rt_index_opt");

        // Phase 8-4: Object insert — 4-ptr signature: fn(out, obj, key, val)
        let mut rt_obj_insert_sig = module.make_signature();
        rt_obj_insert_sig.params.push(AbiParam::new(ptr_ty)); // out
        rt_obj_insert_sig.params.push(AbiParam::new(ptr_ty)); // obj
        rt_obj_insert_sig.params.push(AbiParam::new(ptr_ty)); // key
        rt_obj_insert_sig.params.push(AbiParam::new(ptr_ty)); // val
        let rt_obj_insert_id = module
            .declare_function("rt_obj_insert", cranelift_module::Linkage::Import, &rt_obj_insert_sig)
            .unwrap();

        // Phase 9-2: Range — fn(from: ptr, to: ptr, callback: ptr, ctx: ptr)
        let mut rt_range_sig = module.make_signature();
        rt_range_sig.params.push(AbiParam::new(ptr_ty)); // from
        rt_range_sig.params.push(AbiParam::new(ptr_ty)); // to
        rt_range_sig.params.push(AbiParam::new(ptr_ty)); // callback
        rt_range_sig.params.push(AbiParam::new(ptr_ty)); // ctx
        let rt_range_id = module
            .declare_function("rt_range", cranelift_module::Linkage::Import, &rt_range_sig)
            .unwrap();

        // Phase 11: Range with step — fn(from: ptr, to: ptr, step: ptr, callback: ptr, ctx: ptr)
        let mut rt_range_step_sig = module.make_signature();
        rt_range_step_sig.params.push(AbiParam::new(ptr_ty)); // from
        rt_range_step_sig.params.push(AbiParam::new(ptr_ty)); // to
        rt_range_step_sig.params.push(AbiParam::new(ptr_ty)); // step
        rt_range_step_sig.params.push(AbiParam::new(ptr_ty)); // callback
        rt_range_step_sig.params.push(AbiParam::new(ptr_ty)); // ctx
        let rt_range_step_id = module
            .declare_function("rt_range_step", cranelift_module::Linkage::Import, &rt_range_step_sig)
            .unwrap();

        // Phase 9-4: Recursive descent — fn(input: ptr, callback: ptr, ctx: ptr)
        let mut rt_recurse_sig = module.make_signature();
        rt_recurse_sig.params.push(AbiParam::new(ptr_ty)); // input
        rt_recurse_sig.params.push(AbiParam::new(ptr_ty)); // callback
        rt_recurse_sig.params.push(AbiParam::new(ptr_ty)); // ctx
        let rt_recurse_id = module
            .declare_function("rt_recurse", cranelift_module::Linkage::Import, &rt_recurse_sig)
            .unwrap();

        // Phase 9-6: Remaining builtin functions
        let rt_any_id = declare_unary(&mut *module, "rt_any");
        let rt_all_id = declare_unary(&mut *module, "rt_all");
        let rt_indices_id = declare_bin(&mut *module, "rt_indices");
        let rt_str_index_id = declare_bin(&mut *module, "rt_str_index");
        let rt_str_rindex_id = declare_bin(&mut *module, "rt_str_rindex");
        let rt_inside_id = declare_bin(&mut *module, "rt_inside");
        let rt_tojson_id = declare_unary(&mut *module, "rt_tojson");
        let rt_fromjson_id = declare_unary(&mut *module, "rt_fromjson");
        let rt_getpath_id = declare_bin(&mut *module, "rt_getpath");
        let rt_delpath_id = declare_bin(&mut *module, "rt_delpath");
        let rt_delpaths_id = declare_bin(&mut *module, "rt_delpaths");
        let rt_debug_id = declare_unary(&mut *module, "rt_debug");
        let rt_env_id = declare_unary(&mut *module, "rt_env");
        let rt_builtins_id = declare_unary(&mut *module, "rt_builtins");
        let rt_infinite_id = declare_unary(&mut *module, "rt_infinite");
        let rt_nan_id = declare_unary(&mut *module, "rt_nan");
        let rt_isinfinite_id = declare_unary(&mut *module, "rt_isinfinite");
        let rt_isnan_id = declare_unary(&mut *module, "rt_isnan");
        let rt_isnormal_id = declare_unary(&mut *module, "rt_isnormal");

        // Math unary functions
        let rt_sqrt_id = declare_unary(&mut *module, "rt_sqrt");
        let rt_sin_id = declare_unary(&mut *module, "rt_sin");
        let rt_cos_id = declare_unary(&mut *module, "rt_cos");
        let rt_tan_id = declare_unary(&mut *module, "rt_tan");
        let rt_asin_id = declare_unary(&mut *module, "rt_asin");
        let rt_acos_id = declare_unary(&mut *module, "rt_acos");
        let rt_atan_id = declare_unary(&mut *module, "rt_atan");
        let rt_exp_id = declare_unary(&mut *module, "rt_exp");
        let rt_exp2_id = declare_unary(&mut *module, "rt_exp2");
        let rt_exp10_id = declare_unary(&mut *module, "rt_exp10");
        let rt_log_id = declare_unary(&mut *module, "rt_log");
        let rt_log2_id = declare_unary(&mut *module, "rt_log2");
        let rt_log10_id = declare_unary(&mut *module, "rt_log10");
        let rt_cbrt_id = declare_unary(&mut *module, "rt_cbrt");
        let rt_significand_id = declare_unary(&mut *module, "rt_significand");
        let rt_exponent_id = declare_unary(&mut *module, "rt_exponent");
        let rt_logb_id = declare_unary(&mut *module, "rt_logb");
        let rt_nearbyint_id = declare_unary(&mut *module, "rt_nearbyint");
        let rt_trunc_id = declare_unary(&mut *module, "rt_trunc");
        let rt_rint_id = declare_unary(&mut *module, "rt_rint");
        let rt_j0_id = declare_unary(&mut *module, "rt_j0");
        let rt_j1_id = declare_unary(&mut *module, "rt_j1");
        // Math binary functions
        let rt_pow_id = declare_bin(&mut *module, "rt_pow");
        let rt_atan2_id = declare_bin(&mut *module, "rt_atan2");
        let rt_drem_id = declare_bin(&mut *module, "rt_drem");
        let rt_ldexp_id = declare_bin(&mut *module, "rt_ldexp");
        let rt_scalb_id = declare_bin(&mut *module, "rt_scalb");
        let rt_scalbln_id = declare_bin(&mut *module, "rt_scalbln");
        // bsearch
        let rt_bsearch_id = declare_bin(&mut *module, "rt_bsearch");
        // Transpose, utf8bytelength, toboolean, trim, date/time
        let rt_transpose_id = declare_unary(&mut *module, "rt_transpose");
        let rt_utf8bytelength_id = declare_unary(&mut *module, "rt_utf8bytelength");
        let rt_toboolean_id = declare_unary(&mut *module, "rt_toboolean");
        let rt_trim_id = declare_unary(&mut *module, "rt_trim");
        let rt_ltrim_id = declare_unary(&mut *module, "rt_ltrim");
        let rt_rtrim_id = declare_unary(&mut *module, "rt_rtrim");
        let rt_gmtime_id = declare_unary(&mut *module, "rt_gmtime");
        let rt_mktime_id = declare_unary(&mut *module, "rt_mktime");
        let rt_now_id = declare_unary(&mut *module, "rt_now");
        let rt_strftime_id = declare_bin(&mut *module, "rt_strftime");
        let rt_strptime_id = declare_bin(&mut *module, "rt_strptime");
        let rt_strflocaltime_id = declare_bin(&mut *module, "rt_strflocaltime");

        // Phase 9-1: Closure-based array operations (3-ptr: same as binary)
        let rt_sort_by_keys_id = declare_bin(&mut *module, "rt_sort_by_keys");
        let rt_group_by_keys_id = declare_bin(&mut *module, "rt_group_by_keys");
        let rt_unique_by_keys_id = declare_bin(&mut *module, "rt_unique_by_keys");
        let rt_min_by_keys_id = declare_bin(&mut *module, "rt_min_by_keys");
        let rt_max_by_keys_id = declare_bin(&mut *module, "rt_max_by_keys");

        // Phase 9-6: setpath — 4-ptr signature: fn(out, input, path, val)
        let mut rt_setpath_sig = module.make_signature();
        rt_setpath_sig.params.push(AbiParam::new(ptr_ty)); // out
        rt_setpath_sig.params.push(AbiParam::new(ptr_ty)); // input
        rt_setpath_sig.params.push(AbiParam::new(ptr_ty)); // path
        rt_setpath_sig.params.push(AbiParam::new(ptr_ty)); // val
        let rt_setpath_id = module
            .declare_function("rt_setpath", cranelift_module::Linkage::Import, &rt_setpath_sig)
            .unwrap();

        // Phase 10-2: Regex functions

        // rt_regex_test: fn(out: ptr, input: ptr, re: ptr, flags: ptr) — ternary + output
        let mut rt_regex_test_sig = module.make_signature();
        rt_regex_test_sig.params.push(AbiParam::new(ptr_ty)); // out
        rt_regex_test_sig.params.push(AbiParam::new(ptr_ty)); // input
        rt_regex_test_sig.params.push(AbiParam::new(ptr_ty)); // re
        rt_regex_test_sig.params.push(AbiParam::new(ptr_ty)); // flags
        let rt_regex_test_id = module
            .declare_function("rt_regex_test", cranelift_module::Linkage::Import, &rt_regex_test_sig)
            .unwrap();

        // rt_regex_match: fn(input: ptr, re: ptr, flags: ptr, callback: ptr, ctx: ptr) — generator
        let mut rt_regex_match_sig = module.make_signature();
        rt_regex_match_sig.params.push(AbiParam::new(ptr_ty)); // input
        rt_regex_match_sig.params.push(AbiParam::new(ptr_ty)); // re
        rt_regex_match_sig.params.push(AbiParam::new(ptr_ty)); // flags
        rt_regex_match_sig.params.push(AbiParam::new(ptr_ty)); // callback
        rt_regex_match_sig.params.push(AbiParam::new(ptr_ty)); // ctx
        let rt_regex_match_id = module
            .declare_function("rt_regex_match", cranelift_module::Linkage::Import, &rt_regex_match_sig)
            .unwrap();

        // rt_regex_capture: fn(out: ptr, input: ptr, re: ptr, flags: ptr) — ternary + output
        let rt_regex_capture_id = module
            .declare_function("rt_regex_capture", cranelift_module::Linkage::Import, &rt_regex_test_sig)
            .unwrap();

        // rt_regex_scan: fn(input: ptr, re: ptr, flags: ptr, callback: ptr, ctx: ptr) — generator
        let rt_regex_scan_id = module
            .declare_function("rt_regex_scan", cranelift_module::Linkage::Import, &rt_regex_match_sig)
            .unwrap();

        // rt_regex_sub: fn(out: ptr, input: ptr, re: ptr, tostr: ptr, flags: ptr) — 5-ptr
        let mut rt_regex_sub_sig = module.make_signature();
        rt_regex_sub_sig.params.push(AbiParam::new(ptr_ty)); // out
        rt_regex_sub_sig.params.push(AbiParam::new(ptr_ty)); // input
        rt_regex_sub_sig.params.push(AbiParam::new(ptr_ty)); // re
        rt_regex_sub_sig.params.push(AbiParam::new(ptr_ty)); // tostr
        rt_regex_sub_sig.params.push(AbiParam::new(ptr_ty)); // flags
        let rt_regex_sub_id = module
            .declare_function("rt_regex_sub", cranelift_module::Linkage::Import, &rt_regex_sub_sig)
            .unwrap();

        // rt_regex_gsub: fn(out: ptr, input: ptr, re: ptr, tostr: ptr, flags: ptr) — 5-ptr
        let rt_regex_gsub_id = module
            .declare_function("rt_regex_gsub", cranelift_module::Linkage::Import, &rt_regex_sub_sig)
            .unwrap();

        // Phase 9-5: rt_path_of — generator: fn(input: ptr, descriptor: ptr, callback: ptr, ctx: ptr)
        let mut rt_path_of_sig = module.make_signature();
        rt_path_of_sig.params.push(AbiParam::new(ptr_ty)); // input
        rt_path_of_sig.params.push(AbiParam::new(ptr_ty)); // descriptor
        rt_path_of_sig.params.push(AbiParam::new(ptr_ty)); // callback
        rt_path_of_sig.params.push(AbiParam::new(ptr_ty)); // ctx
        let rt_path_of_id = module
            .declare_function("rt_path_of", cranelift_module::Linkage::Import, &rt_path_of_sig)
            .unwrap();

        // Phase 10-3: rt_update — fn(out: ptr, input: ptr, descriptor: ptr, update_fn: ptr, update_ctx: ptr)
        let mut rt_update_sig = module.make_signature();
        rt_update_sig.params.push(AbiParam::new(ptr_ty)); // out
        rt_update_sig.params.push(AbiParam::new(ptr_ty)); // input
        rt_update_sig.params.push(AbiParam::new(ptr_ty)); // descriptor
        rt_update_sig.params.push(AbiParam::new(ptr_ty)); // update_fn
        rt_update_sig.params.push(AbiParam::new(ptr_ty)); // update_ctx
        let rt_update_id = module
            .declare_function("rt_update", cranelift_module::Linkage::Import, &rt_update_sig)
            .unwrap();

        // rt_slice_assign — fn(out: ptr, input: ptr, slice_key: ptr, value: ptr)
        let mut rt_slice_assign_sig = module.make_signature();
        rt_slice_assign_sig.params.push(AbiParam::new(ptr_ty)); // out
        rt_slice_assign_sig.params.push(AbiParam::new(ptr_ty)); // input
        rt_slice_assign_sig.params.push(AbiParam::new(ptr_ty)); // slice_key
        rt_slice_assign_sig.params.push(AbiParam::new(ptr_ty)); // value
        let rt_slice_assign_id = module
            .declare_function("rt_slice_assign", cranelift_module::Linkage::Import, &rt_slice_assign_sig)
            .unwrap();

        // Declare truthy check function
        let rt_is_truthy_id = module
            .declare_function("rt_is_truthy", cranelift_module::Linkage::Import, &rt_truthy_sig)
            .unwrap();

        // Error check: fn(v: ptr) -> i32 (same signature as truthy)
        let rt_is_error_id = module
            .declare_function("rt_is_error", cranelift_module::Linkage::Import, &rt_truthy_sig)
            .unwrap();

        // Phase 10-5: Try-catch generator wrapper callback
        // fn(value_ptr: ptr, ctx: ptr) → void (same signature as callback)
        let mut rt_try_cb_sig = module.make_signature();
        rt_try_cb_sig.params.push(AbiParam::new(ptr_ty)); // value_ptr
        rt_try_cb_sig.params.push(AbiParam::new(ptr_ty)); // ctx (TryCallbackCtx*)
        let rt_try_callback_wrapper_id = module
            .declare_function("rt_try_callback_wrapper", cranelift_module::Linkage::Import, &rt_try_cb_sig)
            .unwrap();

        // Phase 12: Limit/Skip generator wrapper callbacks (same signature as callback)
        let rt_limit_callback_wrapper_id = module
            .declare_function("rt_limit_callback_wrapper", cranelift_module::Linkage::Import, &rt_try_cb_sig)
            .unwrap();
        let rt_skip_callback_wrapper_id = module
            .declare_function("rt_skip_callback_wrapper", cranelift_module::Linkage::Import, &rt_try_cb_sig)
            .unwrap();

        // Iteration helpers (Phase 4-3):
        // rt_iter_length: fn(v: ptr) -> i64
        let mut rt_iter_length_sig = module.make_signature();
        rt_iter_length_sig.params.push(AbiParam::new(ptr_ty));
        rt_iter_length_sig.returns.push(AbiParam::new(types::I64));
        let rt_iter_length_id = module
            .declare_function("rt_iter_length", cranelift_module::Linkage::Import, &rt_iter_length_sig)
            .unwrap();

        // rt_iter_get: fn(out: ptr, v: ptr, idx: i64)
        let mut rt_iter_get_sig = module.make_signature();
        rt_iter_get_sig.params.push(AbiParam::new(ptr_ty));  // out
        rt_iter_get_sig.params.push(AbiParam::new(ptr_ty));  // v
        rt_iter_get_sig.params.push(AbiParam::new(types::I64)); // idx
        let rt_iter_get_id = module
            .declare_function("rt_iter_get", cranelift_module::Linkage::Import, &rt_iter_get_sig)
            .unwrap();

        // rt_is_iterable: fn(v: ptr) -> i32 (same sig as truthy)
        let rt_is_iterable_id = module
            .declare_function("rt_is_iterable", cranelift_module::Linkage::Import, &rt_truthy_sig)
            .unwrap();
        let rt_iter_error_id = declare_unary(&mut *module, "rt_iter_error");

        // Phase 6-2: rt_iter_prepare: fn(out: ptr, v: ptr) — same sig as unary
        let rt_iter_prepare_id = module
            .declare_function("rt_iter_prepare", cranelift_module::Linkage::Import, &rt_unary_sig)
            .unwrap();

        // rt_collect_init: fn(out: ptr) → void
        let mut rt_collect_init_sig = Signature::new(CallConv::SystemV);
        rt_collect_init_sig.params.push(AbiParam::new(ptr_ty));
        let rt_collect_init_id = module
            .declare_function("rt_collect_init", cranelift_module::Linkage::Import, &rt_collect_init_sig)
            .unwrap();

        // rt_collect_append: fn(arr_ptr: ptr, elem_ptr: ptr) → void
        let mut rt_collect_append_sig = Signature::new(CallConv::SystemV);
        rt_collect_append_sig.params.push(AbiParam::new(ptr_ty));
        rt_collect_append_sig.params.push(AbiParam::new(ptr_ty));
        let rt_collect_append_id = module
            .declare_function("rt_collect_append", cranelift_module::Linkage::Import, &rt_collect_append_sig)
            .unwrap();
        // rt_collect_append_raw: same signature, but keeps Error values as array elements
        let mut rt_collect_append_raw_sig = Signature::new(CallConv::SystemV);
        rt_collect_append_raw_sig.params.push(AbiParam::new(ptr_ty));
        rt_collect_append_raw_sig.params.push(AbiParam::new(ptr_ty));
        let rt_collect_append_raw_id = module
            .declare_function("rt_collect_append_raw", cranelift_module::Linkage::Import, &rt_collect_append_raw_sig)
            .unwrap();

        // Get FuncRef for each runtime function
        let rt_add_ref = module.declare_func_in_func(rt_add_id, builder.func);
        let rt_sub_ref = module.declare_func_in_func(rt_sub_id, builder.func);
        let rt_mul_ref = module.declare_func_in_func(rt_mul_id, builder.func);
        let rt_div_ref = module.declare_func_in_func(rt_div_id, builder.func);
        let rt_mod_ref = module.declare_func_in_func(rt_mod_id, builder.func);
        let rt_index_ref = module.declare_func_in_func(rt_index_id, builder.func);
        let rt_eq_ref = module.declare_func_in_func(rt_eq_id, builder.func);
        let rt_ne_ref = module.declare_func_in_func(rt_ne_id, builder.func);
        let rt_lt_ref = module.declare_func_in_func(rt_lt_id, builder.func);
        let rt_gt_ref = module.declare_func_in_func(rt_gt_id, builder.func);
        let rt_le_ref = module.declare_func_in_func(rt_le_id, builder.func);
        let rt_ge_ref = module.declare_func_in_func(rt_ge_id, builder.func);
        let rt_length_ref = module.declare_func_in_func(rt_length_id, builder.func);
        let rt_type_ref = module.declare_func_in_func(rt_type_id, builder.func);
        let rt_tostring_ref = module.declare_func_in_func(rt_tostring_id, builder.func);
        let rt_tonumber_ref = module.declare_func_in_func(rt_tonumber_id, builder.func);
        let rt_keys_ref = module.declare_func_in_func(rt_keys_id, builder.func);
        let rt_negate_ref = module.declare_func_in_func(rt_negate_id, builder.func);
        let rt_make_error_ref = module.declare_func_in_func(rt_make_error_id, builder.func);
        let rt_extract_error_ref = module.declare_func_in_func(rt_extract_error_id, builder.func);
        let rt_is_truthy_ref = module.declare_func_in_func(rt_is_truthy_id, builder.func);
        let rt_is_error_ref = module.declare_func_in_func(rt_is_error_id, builder.func);
        let rt_try_callback_wrapper_ref = module.declare_func_in_func(rt_try_callback_wrapper_id, builder.func);
        let rt_limit_callback_wrapper_ref = module.declare_func_in_func(rt_limit_callback_wrapper_id, builder.func);
        let rt_skip_callback_wrapper_ref = module.declare_func_in_func(rt_skip_callback_wrapper_id, builder.func);
        let rt_iter_length_ref = module.declare_func_in_func(rt_iter_length_id, builder.func);
        let rt_iter_get_ref = module.declare_func_in_func(rt_iter_get_id, builder.func);
        let rt_is_iterable_ref = module.declare_func_in_func(rt_is_iterable_id, builder.func);
        let rt_iter_error_ref = module.declare_func_in_func(rt_iter_error_id, builder.func);
        let rt_iter_prepare_ref = module.declare_func_in_func(rt_iter_prepare_id, builder.func);
        let rt_collect_init_ref = module.declare_func_in_func(rt_collect_init_id, builder.func);
        let rt_collect_append_ref = module.declare_func_in_func(rt_collect_append_id, builder.func);
        let rt_collect_append_raw_ref = module.declare_func_in_func(rt_collect_append_raw_id, builder.func);

        // Phase 5-2: Additional unary builtins
        let rt_sort_ref = module.declare_func_in_func(rt_sort_id, builder.func);
        let rt_keys_unsorted_ref = module.declare_func_in_func(rt_keys_unsorted_id, builder.func);
        let rt_floor_ref = module.declare_func_in_func(rt_floor_id, builder.func);
        let rt_ceil_ref = module.declare_func_in_func(rt_ceil_id, builder.func);
        let rt_round_ref = module.declare_func_in_func(rt_round_id, builder.func);
        let rt_fabs_ref = module.declare_func_in_func(rt_fabs_id, builder.func);
        let rt_explode_ref = module.declare_func_in_func(rt_explode_id, builder.func);
        let rt_implode_ref = module.declare_func_in_func(rt_implode_id, builder.func);

        // Phase 5-3: jq-defined functions as direct runtime calls
        let rt_jq_add_ref = module.declare_func_in_func(rt_jq_add_id, builder.func);
        let rt_reverse_ref = module.declare_func_in_func(rt_reverse_id, builder.func);
        let rt_to_entries_ref = module.declare_func_in_func(rt_to_entries_id, builder.func);
        let rt_from_entries_ref = module.declare_func_in_func(rt_from_entries_id, builder.func);
        let rt_unique_ref = module.declare_func_in_func(rt_unique_id, builder.func);
        let rt_ascii_downcase_ref = module.declare_func_in_func(rt_ascii_downcase_id, builder.func);
        let rt_ascii_upcase_ref = module.declare_func_in_func(rt_ascii_upcase_id, builder.func);

        // Phase 5-2: Binary builtins (nargs=2)
        let rt_split_ref = module.declare_func_in_func(rt_split_id, builder.func);
        let rt_has_ref = module.declare_func_in_func(rt_has_id, builder.func);
        let rt_startswith_ref = module.declare_func_in_func(rt_startswith_id, builder.func);
        let rt_endswith_ref = module.declare_func_in_func(rt_endswith_id, builder.func);
        let rt_join_ref = module.declare_func_in_func(rt_join_id, builder.func);

        // Phase 8-1: Missing binary builtins
        let rt_contains_ref = module.declare_func_in_func(rt_contains_id, builder.func);
        let rt_ltrimstr_ref = module.declare_func_in_func(rt_ltrimstr_id, builder.func);
        let rt_rtrimstr_ref = module.declare_func_in_func(rt_rtrimstr_id, builder.func);
        let rt_in_ref = module.declare_func_in_func(rt_in_id, builder.func);

        // Phase 8-1: Missing unary builtins
        let rt_min_ref = module.declare_func_in_func(rt_min_id, builder.func);
        let rt_max_ref = module.declare_func_in_func(rt_max_id, builder.func);
        let rt_flatten_ref = module.declare_func_in_func(rt_flatten_id, builder.func);
        let rt_flatten_depth_ref = module.declare_func_in_func(rt_flatten_depth_id, builder.func);

        // Phase 9-3: Format string functions
        let rt_format_base64_ref = module.declare_func_in_func(rt_format_base64_id, builder.func);
        let rt_format_base64d_ref = module.declare_func_in_func(rt_format_base64d_id, builder.func);
        let rt_format_html_ref = module.declare_func_in_func(rt_format_html_id, builder.func);
        let rt_format_uri_ref = module.declare_func_in_func(rt_format_uri_id, builder.func);
        let rt_format_urid_ref = module.declare_func_in_func(rt_format_urid_id, builder.func);
        let rt_format_csv_ref = module.declare_func_in_func(rt_format_csv_id, builder.func);
        let rt_format_tsv_ref = module.declare_func_in_func(rt_format_tsv_id, builder.func);
        let rt_format_json_ref = module.declare_func_in_func(rt_format_json_id, builder.func);
        let rt_format_sh_ref = module.declare_func_in_func(rt_format_sh_id, builder.func);

        // Phase 8-2: Optional index access
        let rt_index_opt_ref = module.declare_func_in_func(rt_index_opt_id, builder.func);

        // Phase 8-4: Object insert
        let rt_obj_insert_ref = module.declare_func_in_func(rt_obj_insert_id, builder.func);

        // Phase 9-2: Range
        let rt_range_ref = module.declare_func_in_func(rt_range_id, builder.func);
        // Phase 11: Range with step
        let rt_range_step_ref = module.declare_func_in_func(rt_range_step_id, builder.func);

        // Phase 9-4: Recursive descent
        let rt_recurse_ref = module.declare_func_in_func(rt_recurse_id, builder.func);

        // Phase 9-6: Remaining builtin functions
        let rt_any_ref = module.declare_func_in_func(rt_any_id, builder.func);
        let rt_all_ref = module.declare_func_in_func(rt_all_id, builder.func);
        let rt_indices_ref = module.declare_func_in_func(rt_indices_id, builder.func);
        let rt_str_index_ref = module.declare_func_in_func(rt_str_index_id, builder.func);
        let rt_str_rindex_ref = module.declare_func_in_func(rt_str_rindex_id, builder.func);
        let rt_inside_ref = module.declare_func_in_func(rt_inside_id, builder.func);
        let rt_tojson_ref = module.declare_func_in_func(rt_tojson_id, builder.func);
        let rt_fromjson_ref = module.declare_func_in_func(rt_fromjson_id, builder.func);
        let rt_getpath_ref = module.declare_func_in_func(rt_getpath_id, builder.func);
        let rt_setpath_ref = module.declare_func_in_func(rt_setpath_id, builder.func);
        let rt_delpath_ref = module.declare_func_in_func(rt_delpath_id, builder.func);
        let rt_delpaths_ref = module.declare_func_in_func(rt_delpaths_id, builder.func);
        let rt_debug_ref = module.declare_func_in_func(rt_debug_id, builder.func);
        let rt_env_ref = module.declare_func_in_func(rt_env_id, builder.func);
        let rt_builtins_ref = module.declare_func_in_func(rt_builtins_id, builder.func);
        let rt_infinite_ref = module.declare_func_in_func(rt_infinite_id, builder.func);
        let rt_nan_ref = module.declare_func_in_func(rt_nan_id, builder.func);
        let rt_isinfinite_ref = module.declare_func_in_func(rt_isinfinite_id, builder.func);
        let rt_isnan_ref = module.declare_func_in_func(rt_isnan_id, builder.func);
        let rt_isnormal_ref = module.declare_func_in_func(rt_isnormal_id, builder.func);

        // Math unary functions
        let rt_sqrt_ref = module.declare_func_in_func(rt_sqrt_id, builder.func);
        let rt_sin_ref = module.declare_func_in_func(rt_sin_id, builder.func);
        let rt_cos_ref = module.declare_func_in_func(rt_cos_id, builder.func);
        let rt_tan_ref = module.declare_func_in_func(rt_tan_id, builder.func);
        let rt_asin_ref = module.declare_func_in_func(rt_asin_id, builder.func);
        let rt_acos_ref = module.declare_func_in_func(rt_acos_id, builder.func);
        let rt_atan_ref = module.declare_func_in_func(rt_atan_id, builder.func);
        let rt_exp_ref = module.declare_func_in_func(rt_exp_id, builder.func);
        let rt_exp2_ref = module.declare_func_in_func(rt_exp2_id, builder.func);
        let rt_exp10_ref = module.declare_func_in_func(rt_exp10_id, builder.func);
        let rt_log_ref = module.declare_func_in_func(rt_log_id, builder.func);
        let rt_log2_ref = module.declare_func_in_func(rt_log2_id, builder.func);
        let rt_log10_ref = module.declare_func_in_func(rt_log10_id, builder.func);
        let rt_cbrt_ref = module.declare_func_in_func(rt_cbrt_id, builder.func);
        let rt_significand_ref = module.declare_func_in_func(rt_significand_id, builder.func);
        let rt_exponent_ref = module.declare_func_in_func(rt_exponent_id, builder.func);
        let rt_logb_ref = module.declare_func_in_func(rt_logb_id, builder.func);
        let rt_nearbyint_ref = module.declare_func_in_func(rt_nearbyint_id, builder.func);
        let rt_trunc_ref = module.declare_func_in_func(rt_trunc_id, builder.func);
        let rt_rint_ref = module.declare_func_in_func(rt_rint_id, builder.func);
        let rt_j0_ref = module.declare_func_in_func(rt_j0_id, builder.func);
        let rt_j1_ref = module.declare_func_in_func(rt_j1_id, builder.func);
        // Math binary functions
        let rt_pow_ref = module.declare_func_in_func(rt_pow_id, builder.func);
        let rt_atan2_ref = module.declare_func_in_func(rt_atan2_id, builder.func);
        let rt_drem_ref = module.declare_func_in_func(rt_drem_id, builder.func);
        let rt_ldexp_ref = module.declare_func_in_func(rt_ldexp_id, builder.func);
        let rt_scalb_ref = module.declare_func_in_func(rt_scalb_id, builder.func);
        let rt_scalbln_ref = module.declare_func_in_func(rt_scalbln_id, builder.func);
        // bsearch
        let rt_bsearch_ref = module.declare_func_in_func(rt_bsearch_id, builder.func);
        // Transpose, utf8bytelength, toboolean, trim, date/time
        let rt_transpose_ref = module.declare_func_in_func(rt_transpose_id, builder.func);
        let rt_utf8bytelength_ref = module.declare_func_in_func(rt_utf8bytelength_id, builder.func);
        let rt_toboolean_ref = module.declare_func_in_func(rt_toboolean_id, builder.func);
        let rt_trim_ref = module.declare_func_in_func(rt_trim_id, builder.func);
        let rt_ltrim_ref = module.declare_func_in_func(rt_ltrim_id, builder.func);
        let rt_rtrim_ref = module.declare_func_in_func(rt_rtrim_id, builder.func);
        let rt_gmtime_ref = module.declare_func_in_func(rt_gmtime_id, builder.func);
        let rt_mktime_ref = module.declare_func_in_func(rt_mktime_id, builder.func);
        let rt_now_ref = module.declare_func_in_func(rt_now_id, builder.func);
        let rt_strftime_ref = module.declare_func_in_func(rt_strftime_id, builder.func);
        let rt_strptime_ref = module.declare_func_in_func(rt_strptime_id, builder.func);
        let rt_strflocaltime_ref = module.declare_func_in_func(rt_strflocaltime_id, builder.func);

        // Phase 9-1: Closure-based array operations
        let rt_sort_by_keys_ref = module.declare_func_in_func(rt_sort_by_keys_id, builder.func);
        let rt_group_by_keys_ref = module.declare_func_in_func(rt_group_by_keys_id, builder.func);
        let rt_unique_by_keys_ref = module.declare_func_in_func(rt_unique_by_keys_id, builder.func);
        let rt_min_by_keys_ref = module.declare_func_in_func(rt_min_by_keys_id, builder.func);
        let rt_max_by_keys_ref = module.declare_func_in_func(rt_max_by_keys_id, builder.func);

        // Phase 10-2: Regex functions
        let rt_regex_test_ref = module.declare_func_in_func(rt_regex_test_id, builder.func);
        let rt_regex_match_ref = module.declare_func_in_func(rt_regex_match_id, builder.func);
        let rt_regex_capture_ref = module.declare_func_in_func(rt_regex_capture_id, builder.func);
        let rt_regex_scan_ref = module.declare_func_in_func(rt_regex_scan_id, builder.func);
        let rt_regex_sub_ref = module.declare_func_in_func(rt_regex_sub_id, builder.func);
        let rt_regex_gsub_ref = module.declare_func_in_func(rt_regex_gsub_id, builder.func);

        // Phase 9-5 / 10-3: Path and update
        let rt_path_of_ref = module.declare_func_in_func(rt_path_of_id, builder.func);
        let rt_update_ref = module.declare_func_in_func(rt_update_id, builder.func);
        let rt_slice_assign_ref = module.declare_func_in_func(rt_slice_assign_id, builder.func);

        let rt_funcs = RuntimeFuncRefs {
            rt_add: rt_add_ref,
            rt_sub: rt_sub_ref,
            rt_mul: rt_mul_ref,
            rt_div: rt_div_ref,
            rt_mod: rt_mod_ref,
            rt_index: rt_index_ref,
            rt_eq: rt_eq_ref,
            rt_ne: rt_ne_ref,
            rt_lt: rt_lt_ref,
            rt_gt: rt_gt_ref,
            rt_le: rt_le_ref,
            rt_ge: rt_ge_ref,
            rt_length: rt_length_ref,
            rt_type: rt_type_ref,
            rt_tostring: rt_tostring_ref,
            rt_tonumber: rt_tonumber_ref,
            rt_keys: rt_keys_ref,
            rt_negate: rt_negate_ref,
            rt_make_error: rt_make_error_ref,
            rt_extract_error: rt_extract_error_ref,
            rt_is_truthy: rt_is_truthy_ref,
            rt_is_error: rt_is_error_ref,
            rt_try_callback_wrapper: rt_try_callback_wrapper_ref,
            rt_limit_callback_wrapper: rt_limit_callback_wrapper_ref,
            rt_skip_callback_wrapper: rt_skip_callback_wrapper_ref,
            rt_iter_length: rt_iter_length_ref,
            rt_iter_get: rt_iter_get_ref,
            rt_is_iterable: rt_is_iterable_ref,
            rt_iter_error: rt_iter_error_ref,
            rt_iter_prepare: rt_iter_prepare_ref,
            rt_collect_init: rt_collect_init_ref,
            rt_collect_append: rt_collect_append_ref,
            rt_collect_append_raw: rt_collect_append_raw_ref,
            // Phase 5-2: Additional unary builtins
            rt_sort: rt_sort_ref,
            rt_keys_unsorted: rt_keys_unsorted_ref,
            rt_floor: rt_floor_ref,
            rt_ceil: rt_ceil_ref,
            rt_round: rt_round_ref,
            rt_fabs: rt_fabs_ref,
            rt_explode: rt_explode_ref,
            rt_implode: rt_implode_ref,
            // Phase 5-2: Binary builtins (nargs=2)
            rt_split: rt_split_ref,
            rt_has: rt_has_ref,
            rt_startswith: rt_startswith_ref,
            rt_endswith: rt_endswith_ref,
            rt_join: rt_join_ref,
            // Phase 5-3: jq-defined functions as direct runtime calls
            rt_jq_add: rt_jq_add_ref,
            rt_reverse: rt_reverse_ref,
            rt_to_entries: rt_to_entries_ref,
            rt_from_entries: rt_from_entries_ref,
            rt_unique: rt_unique_ref,
            rt_ascii_downcase: rt_ascii_downcase_ref,
            rt_ascii_upcase: rt_ascii_upcase_ref,
            // Phase 8-1: Missing binary builtins
            rt_contains: rt_contains_ref,
            rt_ltrimstr: rt_ltrimstr_ref,
            rt_rtrimstr: rt_rtrimstr_ref,
            rt_in: rt_in_ref,
            // Phase 8-1: Missing unary builtins
            rt_min: rt_min_ref,
            rt_max: rt_max_ref,
            rt_flatten: rt_flatten_ref,
            rt_flatten_depth: rt_flatten_depth_ref,
            // Phase 8-2: Optional index access
            rt_index_opt: rt_index_opt_ref,
            // Phase 8-4: Object insert
            rt_obj_insert: rt_obj_insert_ref,
            // Phase 9-3: Format string functions
            rt_format_base64: rt_format_base64_ref,
            rt_format_base64d: rt_format_base64d_ref,
            rt_format_html: rt_format_html_ref,
            rt_format_uri: rt_format_uri_ref,
            rt_format_urid: rt_format_urid_ref,
            rt_format_csv: rt_format_csv_ref,
            rt_format_tsv: rt_format_tsv_ref,
            rt_format_json: rt_format_json_ref,
            rt_format_sh: rt_format_sh_ref,
            rt_range: rt_range_ref,
            rt_range_step: rt_range_step_ref,
            rt_recurse: rt_recurse_ref,
            // Phase 9-6: Remaining builtin functions
            rt_any: rt_any_ref,
            rt_all: rt_all_ref,
            rt_indices: rt_indices_ref,
            rt_str_index: rt_str_index_ref,
            rt_str_rindex: rt_str_rindex_ref,
            rt_inside: rt_inside_ref,
            rt_tojson: rt_tojson_ref,
            rt_fromjson: rt_fromjson_ref,
            rt_getpath: rt_getpath_ref,
            rt_setpath: rt_setpath_ref,
            rt_delpath: rt_delpath_ref,
            rt_delpaths: rt_delpaths_ref,
            rt_debug: rt_debug_ref,
            rt_env: rt_env_ref,
            rt_builtins: rt_builtins_ref,
            rt_infinite: rt_infinite_ref,
            rt_nan: rt_nan_ref,
            rt_isinfinite: rt_isinfinite_ref,
            rt_isnan: rt_isnan_ref,
            rt_isnormal: rt_isnormal_ref,
            // Math unary functions
            rt_sqrt: rt_sqrt_ref,
            rt_sin: rt_sin_ref,
            rt_cos: rt_cos_ref,
            rt_tan: rt_tan_ref,
            rt_asin: rt_asin_ref,
            rt_acos: rt_acos_ref,
            rt_atan: rt_atan_ref,
            rt_exp: rt_exp_ref,
            rt_exp2: rt_exp2_ref,
            rt_exp10: rt_exp10_ref,
            rt_log: rt_log_ref,
            rt_log2: rt_log2_ref,
            rt_log10: rt_log10_ref,
            rt_cbrt: rt_cbrt_ref,
            rt_significand: rt_significand_ref,
            rt_exponent: rt_exponent_ref,
            rt_logb: rt_logb_ref,
            rt_nearbyint: rt_nearbyint_ref,
            rt_trunc: rt_trunc_ref,
            rt_rint: rt_rint_ref,
            rt_j0: rt_j0_ref,
            rt_j1: rt_j1_ref,
            // Math binary functions
            rt_pow: rt_pow_ref,
            rt_atan2: rt_atan2_ref,
            rt_drem: rt_drem_ref,
            rt_ldexp: rt_ldexp_ref,
            rt_scalb: rt_scalb_ref,
            rt_scalbln: rt_scalbln_ref,
            // bsearch
            rt_bsearch: rt_bsearch_ref,
            // Transpose, utf8bytelength, toboolean, trim, date/time
            rt_transpose: rt_transpose_ref,
            rt_utf8bytelength: rt_utf8bytelength_ref,
            rt_toboolean: rt_toboolean_ref,
            rt_trim: rt_trim_ref,
            rt_ltrim: rt_ltrim_ref,
            rt_rtrim: rt_rtrim_ref,
            rt_gmtime: rt_gmtime_ref,
            rt_mktime: rt_mktime_ref,
            rt_now: rt_now_ref,
            rt_strftime: rt_strftime_ref,
            rt_strptime: rt_strptime_ref,
            rt_strflocaltime: rt_strflocaltime_ref,
            // Phase 9-1: Closure-based array operations
            rt_sort_by_keys: rt_sort_by_keys_ref,
            rt_group_by_keys: rt_group_by_keys_ref,
            rt_unique_by_keys: rt_unique_by_keys_ref,
            rt_min_by_keys: rt_min_by_keys_ref,
            rt_max_by_keys: rt_max_by_keys_ref,
            // Phase 10-2: Regex functions
            rt_regex_test: rt_regex_test_ref,
            rt_regex_match: rt_regex_match_ref,
            rt_regex_capture: rt_regex_capture_ref,
            rt_regex_scan: rt_regex_scan_ref,
            rt_regex_sub: rt_regex_sub_ref,
            rt_regex_gsub: rt_regex_gsub_ref,
            rt_path_of: rt_path_of_ref,
            rt_update: rt_update_ref,
            rt_slice_assign: rt_slice_assign_ref,
        };

        // Create the callback signature for call_indirect:
        // fn(value_ptr: *const Value, ctx: *mut u8) → void
        let mut callback_sig = Signature::new(CallConv::SystemV);
        callback_sig.params.push(AbiParam::new(ptr_ty)); // value_ptr
        callback_sig.params.push(AbiParam::new(ptr_ty)); // ctx
        let callback_sig_ref = builder.import_signature(callback_sig);

        // Variable slot mapping: var_index → StackSlot address (Cranelift Value)
        let mut var_slots: HashMap<u16, cranelift_codegen::ir::StackSlot> = HashMap::new();

        // Use codegen_generator as the top-level entry point.
        // This handles Comma (multiple outputs), Empty (zero outputs),
        // and delegates to codegen_expr for 1→1 expressions.
        codegen_generator(
            expr,
            input_ptr,
            callback_ptr,
            ctx_ptr,
            callback_sig_ref,
            &mut builder,
            ptr_ty,
            &rt_funcs,
            &mut literals,
            &mut var_slots,
        );

        builder.ins().return_(&[]);
        builder.finalize();

        literal_values = literals.values;
    }

    // Capture the CLIF IR text for debugging
    let clif_text = format!("{}", ctx.func.display());

    // Compile: define the function in the module
    module
        .define_function(func_id, ctx)
        .context("compiling jit_filter")?;
    module.clear_context(ctx);

    Ok((func_id, literal_values, clif_text))
}

/// References to imported runtime functions, used during codegen.
struct RuntimeFuncRefs {
    // Binary (3-ptr): arithmetic
    rt_add: cranelift_codegen::ir::FuncRef,
    rt_sub: cranelift_codegen::ir::FuncRef,
    rt_mul: cranelift_codegen::ir::FuncRef,
    rt_div: cranelift_codegen::ir::FuncRef,
    rt_mod: cranelift_codegen::ir::FuncRef,
    rt_index: cranelift_codegen::ir::FuncRef,
    // Binary (3-ptr): comparison
    rt_eq: cranelift_codegen::ir::FuncRef,
    rt_ne: cranelift_codegen::ir::FuncRef,
    rt_lt: cranelift_codegen::ir::FuncRef,
    rt_gt: cranelift_codegen::ir::FuncRef,
    rt_le: cranelift_codegen::ir::FuncRef,
    rt_ge: cranelift_codegen::ir::FuncRef,
    // Unary (2-ptr)
    rt_length: cranelift_codegen::ir::FuncRef,
    rt_type: cranelift_codegen::ir::FuncRef,
    rt_tostring: cranelift_codegen::ir::FuncRef,
    rt_tonumber: cranelift_codegen::ir::FuncRef,
    rt_keys: cranelift_codegen::ir::FuncRef,
    rt_negate: cranelift_codegen::ir::FuncRef,
    rt_make_error: cranelift_codegen::ir::FuncRef,
    rt_extract_error: cranelift_codegen::ir::FuncRef,
    // Truthy check (Phase 3): fn(v: ptr) -> i32
    rt_is_truthy: cranelift_codegen::ir::FuncRef,
    // Error check (Phase 3 try-catch): fn(v: ptr) -> i32
    rt_is_error: cranelift_codegen::ir::FuncRef,
    // Phase 10-5: Try-catch generator wrapper
    rt_try_callback_wrapper: cranelift_codegen::ir::FuncRef,
    // Phase 12: Limit/Skip generator wrappers
    rt_limit_callback_wrapper: cranelift_codegen::ir::FuncRef,
    rt_skip_callback_wrapper: cranelift_codegen::ir::FuncRef,
    // Iteration helpers (Phase 4-3)
    rt_iter_length: cranelift_codegen::ir::FuncRef,
    rt_iter_get: cranelift_codegen::ir::FuncRef,
    rt_is_iterable: cranelift_codegen::ir::FuncRef,
    rt_iter_error: cranelift_codegen::ir::FuncRef,
    // Phase 6-2: Object iteration optimization
    rt_iter_prepare: cranelift_codegen::ir::FuncRef,
    // Collect helpers (Phase 5-1: array constructor)
    rt_collect_init: cranelift_codegen::ir::FuncRef,
    rt_collect_append: cranelift_codegen::ir::FuncRef,
    rt_collect_append_raw: cranelift_codegen::ir::FuncRef,
    // Phase 5-2: Additional unary builtins
    rt_sort: cranelift_codegen::ir::FuncRef,
    rt_keys_unsorted: cranelift_codegen::ir::FuncRef,
    rt_floor: cranelift_codegen::ir::FuncRef,
    rt_ceil: cranelift_codegen::ir::FuncRef,
    rt_round: cranelift_codegen::ir::FuncRef,
    rt_fabs: cranelift_codegen::ir::FuncRef,
    rt_explode: cranelift_codegen::ir::FuncRef,
    rt_implode: cranelift_codegen::ir::FuncRef,
    // Phase 5-2: Binary builtins (nargs=2)
    rt_split: cranelift_codegen::ir::FuncRef,
    rt_has: cranelift_codegen::ir::FuncRef,
    rt_startswith: cranelift_codegen::ir::FuncRef,
    rt_endswith: cranelift_codegen::ir::FuncRef,
    rt_join: cranelift_codegen::ir::FuncRef,
    // Phase 5-3: jq-defined functions as direct runtime calls
    rt_jq_add: cranelift_codegen::ir::FuncRef,
    rt_reverse: cranelift_codegen::ir::FuncRef,
    rt_to_entries: cranelift_codegen::ir::FuncRef,
    rt_from_entries: cranelift_codegen::ir::FuncRef,
    rt_unique: cranelift_codegen::ir::FuncRef,
    rt_ascii_downcase: cranelift_codegen::ir::FuncRef,
    rt_ascii_upcase: cranelift_codegen::ir::FuncRef,
    // Phase 8-1: Missing binary builtins
    rt_contains: cranelift_codegen::ir::FuncRef,
    rt_ltrimstr: cranelift_codegen::ir::FuncRef,
    rt_rtrimstr: cranelift_codegen::ir::FuncRef,
    rt_in: cranelift_codegen::ir::FuncRef,
    // Phase 8-1: Missing unary builtins
    rt_min: cranelift_codegen::ir::FuncRef,
    rt_max: cranelift_codegen::ir::FuncRef,
    rt_flatten: cranelift_codegen::ir::FuncRef,
    rt_flatten_depth: cranelift_codegen::ir::FuncRef,
    // Phase 8-2: Optional index access
    rt_index_opt: cranelift_codegen::ir::FuncRef,
    // Phase 8-4: Object insert (4-ptr)
    rt_obj_insert: cranelift_codegen::ir::FuncRef,
    // Phase 9-3: Format string functions
    rt_format_base64: cranelift_codegen::ir::FuncRef,
    rt_format_base64d: cranelift_codegen::ir::FuncRef,
    rt_format_html: cranelift_codegen::ir::FuncRef,
    rt_format_uri: cranelift_codegen::ir::FuncRef,
    rt_format_urid: cranelift_codegen::ir::FuncRef,
    rt_format_csv: cranelift_codegen::ir::FuncRef,
    rt_format_tsv: cranelift_codegen::ir::FuncRef,
    rt_format_json: cranelift_codegen::ir::FuncRef,
    rt_format_sh: cranelift_codegen::ir::FuncRef,
    // Phase 9-2: Range generator
    rt_range: cranelift_codegen::ir::FuncRef,
    // Phase 11: Range with step
    rt_range_step: cranelift_codegen::ir::FuncRef,
    // Phase 9-4: Recursive descent (..)
    rt_recurse: cranelift_codegen::ir::FuncRef,
    // Phase 9-6: Remaining builtin functions
    rt_any: cranelift_codegen::ir::FuncRef,
    rt_all: cranelift_codegen::ir::FuncRef,
    rt_indices: cranelift_codegen::ir::FuncRef,
    rt_str_index: cranelift_codegen::ir::FuncRef,
    rt_str_rindex: cranelift_codegen::ir::FuncRef,
    rt_inside: cranelift_codegen::ir::FuncRef,
    rt_tojson: cranelift_codegen::ir::FuncRef,
    rt_fromjson: cranelift_codegen::ir::FuncRef,
    rt_getpath: cranelift_codegen::ir::FuncRef,
    rt_setpath: cranelift_codegen::ir::FuncRef,
    rt_delpath: cranelift_codegen::ir::FuncRef,
    rt_delpaths: cranelift_codegen::ir::FuncRef,
    rt_debug: cranelift_codegen::ir::FuncRef,
    rt_env: cranelift_codegen::ir::FuncRef,
    rt_builtins: cranelift_codegen::ir::FuncRef,
    rt_infinite: cranelift_codegen::ir::FuncRef,
    rt_nan: cranelift_codegen::ir::FuncRef,
    rt_isinfinite: cranelift_codegen::ir::FuncRef,
    rt_isnan: cranelift_codegen::ir::FuncRef,
    rt_isnormal: cranelift_codegen::ir::FuncRef,
    // Math unary functions
    rt_sqrt: cranelift_codegen::ir::FuncRef,
    rt_sin: cranelift_codegen::ir::FuncRef,
    rt_cos: cranelift_codegen::ir::FuncRef,
    rt_tan: cranelift_codegen::ir::FuncRef,
    rt_asin: cranelift_codegen::ir::FuncRef,
    rt_acos: cranelift_codegen::ir::FuncRef,
    rt_atan: cranelift_codegen::ir::FuncRef,
    rt_exp: cranelift_codegen::ir::FuncRef,
    rt_exp2: cranelift_codegen::ir::FuncRef,
    rt_exp10: cranelift_codegen::ir::FuncRef,
    rt_log: cranelift_codegen::ir::FuncRef,
    rt_log2: cranelift_codegen::ir::FuncRef,
    rt_log10: cranelift_codegen::ir::FuncRef,
    rt_cbrt: cranelift_codegen::ir::FuncRef,
    rt_significand: cranelift_codegen::ir::FuncRef,
    rt_exponent: cranelift_codegen::ir::FuncRef,
    rt_logb: cranelift_codegen::ir::FuncRef,
    rt_nearbyint: cranelift_codegen::ir::FuncRef,
    rt_trunc: cranelift_codegen::ir::FuncRef,
    rt_rint: cranelift_codegen::ir::FuncRef,
    rt_j0: cranelift_codegen::ir::FuncRef,
    rt_j1: cranelift_codegen::ir::FuncRef,
    // Math binary functions
    rt_pow: cranelift_codegen::ir::FuncRef,
    rt_atan2: cranelift_codegen::ir::FuncRef,
    rt_drem: cranelift_codegen::ir::FuncRef,
    rt_ldexp: cranelift_codegen::ir::FuncRef,
    rt_scalb: cranelift_codegen::ir::FuncRef,
    rt_scalbln: cranelift_codegen::ir::FuncRef,
    // bsearch
    rt_bsearch: cranelift_codegen::ir::FuncRef,
    // Transpose, utf8bytelength, toboolean, trim, date/time
    rt_transpose: cranelift_codegen::ir::FuncRef,
    rt_utf8bytelength: cranelift_codegen::ir::FuncRef,
    rt_toboolean: cranelift_codegen::ir::FuncRef,
    rt_trim: cranelift_codegen::ir::FuncRef,
    rt_ltrim: cranelift_codegen::ir::FuncRef,
    rt_rtrim: cranelift_codegen::ir::FuncRef,
    rt_gmtime: cranelift_codegen::ir::FuncRef,
    rt_mktime: cranelift_codegen::ir::FuncRef,
    rt_now: cranelift_codegen::ir::FuncRef,
    rt_strftime: cranelift_codegen::ir::FuncRef,
    rt_strptime: cranelift_codegen::ir::FuncRef,
    rt_strflocaltime: cranelift_codegen::ir::FuncRef,
    // Phase 9-1: Closure-based array operations (3-ptr: out, input, keys)
    rt_sort_by_keys: cranelift_codegen::ir::FuncRef,
    rt_group_by_keys: cranelift_codegen::ir::FuncRef,
    rt_unique_by_keys: cranelift_codegen::ir::FuncRef,
    rt_min_by_keys: cranelift_codegen::ir::FuncRef,
    rt_max_by_keys: cranelift_codegen::ir::FuncRef,
    // Phase 10-2: Regex functions
    rt_regex_test: cranelift_codegen::ir::FuncRef,
    rt_regex_match: cranelift_codegen::ir::FuncRef,
    rt_regex_capture: cranelift_codegen::ir::FuncRef,
    rt_regex_scan: cranelift_codegen::ir::FuncRef,
    rt_regex_sub: cranelift_codegen::ir::FuncRef,
    rt_regex_gsub: cranelift_codegen::ir::FuncRef,
    // Phase 9-5 / 10-3: Path operations and update operators
    rt_path_of: cranelift_codegen::ir::FuncRef,
    #[allow(dead_code)]
    rt_update: cranelift_codegen::ir::FuncRef,
    rt_slice_assign: cranelift_codegen::ir::FuncRef,
}

/// Recursively generate CLIF IR for an expression.
///
/// Returns a tuple of:
/// - A Cranelift `Value` (i64/ptr) that points to a `Value` in memory
/// - A `TypeHint` indicating the known type of the result (Phase 6-3)
///
/// The pointed-to Value may be:
/// - The input parameter (for Expr::Input)
/// - A StackSlot (for literals and operation results)
/// - A pre-allocated Rust-side Value (for Str/Arr/Obj literals)
fn codegen_expr(
    expr: &Expr,
    input_ptr: cranelift_codegen::ir::Value,
    builder: &mut FunctionBuilder,
    ptr_ty: types::Type,
    rt_funcs: &RuntimeFuncRefs,
    literals: &mut LiteralPool,
    var_slots: &mut HashMap<u16, cranelift_codegen::ir::StackSlot>,
) -> (cranelift_codegen::ir::Value, TypeHint) {
    match expr {
        Expr::Input => {
            // The input value is already at `input_ptr`.
            (input_ptr, TypeHint::Unknown)
        }

        Expr::Literal(lit) => {
            let hint = match lit {
                Literal::Num(_) => TypeHint::Num,
                _ => TypeHint::Unknown,
            };
            (codegen_literal(lit, builder, ptr_ty, literals), hint)
        }

        Expr::BinOp { op, lhs, rhs } => {
            // Generate code for both operands (with type hints)
            let (lhs_ptr, lhs_hint) = codegen_expr(lhs, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (rhs_ptr, rhs_hint) = codegen_expr(rhs, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Phase 6-3: Type specialization for numeric operations.
            //
            // Check if this BinOp can be specialized for f64:
            // - Both operands must be TypeHint::Num
            // - The op must be arithmetic (+,-,*,/,%) or comparison (<,>,<=,>=,==,!=)
            //
            // When both are Num, we know:
            // - No Error propagation needed (Literal::Num can't be Error)
            // - No type dispatch needed (both are f64)
            // - We can generate Cranelift f64 instructions directly

            // Div and Mod excluded from fast path: division by zero must produce Error
            let is_numeric_arith = matches!(op,
                BinOp::Add | BinOp::Sub | BinOp::Mul);
            let is_numeric_cmp = matches!(op,
                BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge | BinOp::Eq | BinOp::Ne);

            if lhs_hint == TypeHint::Num && rhs_hint == TypeHint::Num && (is_numeric_arith || is_numeric_cmp) {
                // === FAST PATH: Both operands are known Num ===
                // Read f64 payloads directly from offset 8
                let lhs_f64 = builder.ins().load(types::F64, MemFlags::trusted(), lhs_ptr, 8);
                let rhs_f64 = builder.ins().load(types::F64, MemFlags::trusted(), rhs_ptr, 8);

                // Allocate result slot
                let result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let result_addr = builder.ins().stack_addr(ptr_ty, result_slot, 0);

                if is_numeric_arith {
                    // Generate f64 arithmetic instruction
                    let result_f64 = match op {
                        BinOp::Add => builder.ins().fadd(lhs_f64, rhs_f64),
                        BinOp::Sub => builder.ins().fsub(lhs_f64, rhs_f64),
                        BinOp::Mul => builder.ins().fmul(lhs_f64, rhs_f64),
                        BinOp::Div => builder.ins().fdiv(lhs_f64, rhs_f64),
                        BinOp::Mod => {
                            // f64 remainder: a - trunc(a/b) * b (matching fmod/Rust % semantics)
                            let div = builder.ins().fdiv(lhs_f64, rhs_f64);
                            let trunced = builder.ins().trunc(div);
                            let prod = builder.ins().fmul(trunced, rhs_f64);
                            builder.ins().fsub(lhs_f64, prod)
                        }
                        _ => unreachable!(),
                    };

                    // Write tag=TAG_NUM at offset 0, f64 payload at offset 8
                    let tag = builder.ins().iconst(types::I64, TAG_NUM as i64);
                    builder.ins().stack_store(tag, result_slot, 0);
                    let bits = builder.ins().bitcast(types::I64, MemFlags::new(), result_f64);
                    builder.ins().stack_store(bits, result_slot, 8);

                    (result_addr, TypeHint::Num)
                } else {
                    // Generate f64 comparison instruction → Bool result
                    use cranelift_codegen::ir::condcodes::FloatCC;
                    let cc = match op {
                        BinOp::Lt => FloatCC::LessThan,
                        BinOp::Gt => FloatCC::GreaterThan,
                        BinOp::Le => FloatCC::LessThanOrEqual,
                        BinOp::Ge => FloatCC::GreaterThanOrEqual,
                        BinOp::Eq => FloatCC::Equal,
                        BinOp::Ne => FloatCC::NotEqual,
                        _ => unreachable!(),
                    };
                    let cmp_result = builder.ins().fcmp(cc, lhs_f64, rhs_f64);
                    // Convert i8 bool to i64 for tag payload
                    let bool_i64 = builder.ins().uextend(types::I64, cmp_result);

                    // Write tag=TAG_BOOL at offset 0, bool payload at offset 8
                    let tag = builder.ins().iconst(types::I64, TAG_BOOL as i64);
                    builder.ins().stack_store(tag, result_slot, 0);
                    builder.ins().stack_store(bool_i64, result_slot, 8);

                    (result_addr, TypeHint::Unknown)
                }
            } else if (lhs_hint == TypeHint::Num || rhs_hint == TypeHint::Num)
                && (is_numeric_arith || is_numeric_cmp)
            {
                // === PARTIAL FAST PATH: One operand is known Num ===
                // Check the unknown operand's tag at runtime.
                // If TAG_NUM → fast path (f64 instructions)
                // Otherwise → fallback to rt_* call

                // Identify which operand is unknown (type not known at compile time)
                let unknown_ptr = if lhs_hint == TypeHint::Num {
                    rhs_ptr
                } else {
                    lhs_ptr
                };

                // Read the unknown operand's tag
                let unknown_tag = builder.ins().load(types::I64, MemFlags::trusted(), unknown_ptr, 0);
                let num_tag = builder.ins().iconst(types::I64, TAG_NUM as i64);
                let is_num = builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::Equal, unknown_tag, num_tag,
                );

                let fast_block = builder.create_block();
                let slow_block = builder.create_block();
                let merge_block = builder.create_block();
                builder.append_block_param(merge_block, ptr_ty);

                let empty_args: &[BlockArg] = &[];
                builder.ins().brif(is_num, fast_block, empty_args, slow_block, empty_args);

                // Fast block: both are Num, do f64 ops
                builder.switch_to_block(fast_block);
                builder.seal_block(fast_block);

                let lhs_f64 = builder.ins().load(types::F64, MemFlags::trusted(), lhs_ptr, 8);
                let rhs_f64 = builder.ins().load(types::F64, MemFlags::trusted(), rhs_ptr, 8);

                let fast_result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let fast_result_addr = builder.ins().stack_addr(ptr_ty, fast_result_slot, 0);

                if is_numeric_arith {
                    let result_f64 = match op {
                        BinOp::Add => builder.ins().fadd(lhs_f64, rhs_f64),
                        BinOp::Sub => builder.ins().fsub(lhs_f64, rhs_f64),
                        BinOp::Mul => builder.ins().fmul(lhs_f64, rhs_f64),
                        BinOp::Div => builder.ins().fdiv(lhs_f64, rhs_f64),
                        BinOp::Mod => {
                            let div = builder.ins().fdiv(lhs_f64, rhs_f64);
                            let floored = builder.ins().floor(div);
                            let prod = builder.ins().fmul(floored, rhs_f64);
                            builder.ins().fsub(lhs_f64, prod)
                        }
                        _ => unreachable!(),
                    };
                    let tag = builder.ins().iconst(types::I64, TAG_NUM as i64);
                    builder.ins().stack_store(tag, fast_result_slot, 0);
                    let bits = builder.ins().bitcast(types::I64, MemFlags::new(), result_f64);
                    builder.ins().stack_store(bits, fast_result_slot, 8);
                } else {
                    use cranelift_codegen::ir::condcodes::FloatCC;
                    let cc = match op {
                        BinOp::Lt => FloatCC::LessThan,
                        BinOp::Gt => FloatCC::GreaterThan,
                        BinOp::Le => FloatCC::LessThanOrEqual,
                        BinOp::Ge => FloatCC::GreaterThanOrEqual,
                        BinOp::Eq => FloatCC::Equal,
                        BinOp::Ne => FloatCC::NotEqual,
                        _ => unreachable!(),
                    };
                    let cmp_result = builder.ins().fcmp(cc, lhs_f64, rhs_f64);
                    let bool_i64 = builder.ins().uextend(types::I64, cmp_result);
                    let tag = builder.ins().iconst(types::I64, TAG_BOOL as i64);
                    builder.ins().stack_store(tag, fast_result_slot, 0);
                    builder.ins().stack_store(bool_i64, fast_result_slot, 8);
                }

                let fast_arg = BlockArg::Value(fast_result_addr);
                builder.ins().jump(merge_block, &[fast_arg]);

                // Slow block: fallback to rt_* call
                builder.switch_to_block(slow_block);
                builder.seal_block(slow_block);

                let slow_result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let slow_result_addr = builder.ins().stack_addr(ptr_ty, slow_result_slot, 0);

                let func_ref = match op {
                    BinOp::Add => rt_funcs.rt_add,
                    BinOp::Sub => rt_funcs.rt_sub,
                    BinOp::Mul => rt_funcs.rt_mul,
                    BinOp::Div => rt_funcs.rt_div,
                    BinOp::Mod => rt_funcs.rt_mod,
                    BinOp::Eq => rt_funcs.rt_eq,
                    BinOp::Ne => rt_funcs.rt_ne,
                    BinOp::Lt => rt_funcs.rt_lt,
                    BinOp::Gt => rt_funcs.rt_gt,
                    BinOp::Le => rt_funcs.rt_le,
                    BinOp::Ge => rt_funcs.rt_ge,
                    _ => unreachable!(),
                };
                builder.ins().call(func_ref, &[slow_result_addr, lhs_ptr, rhs_ptr]);

                let slow_arg = BlockArg::Value(slow_result_addr);
                builder.ins().jump(merge_block, &[slow_arg]);

                // Merge
                builder.switch_to_block(merge_block);
                builder.seal_block(merge_block);

                let merged_ptr = builder.block_params(merge_block)[0];
                // Result type hint: arithmetic with one Num → could be Num or Error
                // For safety, mark as Unknown since the slow path may return non-Num
                (merged_ptr, TypeHint::Unknown)
            } else {
                // === GENERIC PATH: No type specialization ===
                let result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let result_addr = builder.ins().stack_addr(ptr_ty, result_slot, 0);

                let func_ref = match op {
                    BinOp::Add => rt_funcs.rt_add,
                    BinOp::Sub => rt_funcs.rt_sub,
                    BinOp::Mul => rt_funcs.rt_mul,
                    BinOp::Div => rt_funcs.rt_div,
                    BinOp::Mod => rt_funcs.rt_mod,
                    BinOp::Eq => rt_funcs.rt_eq,
                    BinOp::Ne => rt_funcs.rt_ne,
                    BinOp::Lt => rt_funcs.rt_lt,
                    BinOp::Gt => rt_funcs.rt_gt,
                    BinOp::Le => rt_funcs.rt_le,
                    BinOp::Ge => rt_funcs.rt_ge,
                    BinOp::Split => rt_funcs.rt_split,
                    BinOp::Has => rt_funcs.rt_has,
                    BinOp::StartsWith => rt_funcs.rt_startswith,
                    BinOp::EndsWith => rt_funcs.rt_endswith,
                    BinOp::Join => rt_funcs.rt_join,
                    BinOp::Contains => rt_funcs.rt_contains,
                    BinOp::Ltrimstr => rt_funcs.rt_ltrimstr,
                    BinOp::Rtrimstr => rt_funcs.rt_rtrimstr,
                    BinOp::In => rt_funcs.rt_in,
                    // Phase 9-6: Remaining binary builtins
                    BinOp::Inside => rt_funcs.rt_inside,
                    BinOp::Indices => rt_funcs.rt_indices,
                    BinOp::StrIndex => rt_funcs.rt_str_index,
                    BinOp::StrRindex => rt_funcs.rt_str_rindex,
                    BinOp::GetPath => rt_funcs.rt_getpath,
                    BinOp::DelPaths => rt_funcs.rt_delpaths,
                    BinOp::FlattenDepth => rt_funcs.rt_flatten_depth,
                    // Math binary functions
                    BinOp::Pow => rt_funcs.rt_pow,
                    BinOp::Atan2 => rt_funcs.rt_atan2,
                    BinOp::Drem => rt_funcs.rt_drem,
                    BinOp::Ldexp => rt_funcs.rt_ldexp,
                    BinOp::Scalb => rt_funcs.rt_scalb,
                    BinOp::Scalbln => rt_funcs.rt_scalbln,
                    // bsearch
                    BinOp::Bsearch => rt_funcs.rt_bsearch,
                    // Date/time with format argument
                    BinOp::Strftime => rt_funcs.rt_strftime,
                    BinOp::Strptime => rt_funcs.rt_strptime,
                    BinOp::Strflocaltime => rt_funcs.rt_strflocaltime,
                };

                builder.ins().call(func_ref, &[result_addr, lhs_ptr, rhs_ptr]);

                (result_addr, TypeHint::Unknown)
            }
        }

        Expr::Index { expr: obj, key } => {
            // Generate code for object/array and key
            let (obj_ptr, _) = codegen_expr(obj, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (key_ptr, _) = codegen_expr(key, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Allocate a StackSlot for the result (16 bytes, 8-byte aligned)
            let result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                16,
                8,
            ));
            let result_addr = builder.ins().stack_addr(ptr_ty, result_slot, 0);

            builder
                .ins()
                .call(rt_funcs.rt_index, &[result_addr, obj_ptr, key_ptr]);

            // Index result type is unknown (could be any field type)
            (result_addr, TypeHint::Unknown)
        }

        Expr::IndexOpt { expr: obj, key } => {
            // Phase 8-2: Optional index access `.foo?`.
            // Same as Index but calls rt_index_opt (returns null instead of Error).
            let (obj_ptr, _) = codegen_expr(obj, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (key_ptr, _) = codegen_expr(key, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            let result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                16,
                8,
            ));
            let result_addr = builder.ins().stack_addr(ptr_ty, result_slot, 0);

            builder
                .ins()
                .call(rt_funcs.rt_index_opt, &[result_addr, obj_ptr, key_ptr]);

            (result_addr, TypeHint::Unknown)
        }

        Expr::ObjectInsert { obj, key, value } => {
            // Phase 8-4: Object insert — insert key-value pair into object.
            // Evaluate all three operands.
            let (obj_ptr, _) = codegen_expr(obj, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (key_ptr, _) = codegen_expr(key, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (val_ptr, _) = codegen_expr(value, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Allocate a StackSlot for the result (16 bytes, 8-byte aligned)
            let result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                16,
                8,
            ));
            let result_addr = builder.ins().stack_addr(ptr_ty, result_slot, 0);

            // Call rt_obj_insert(out, obj, key, val)
            builder
                .ins()
                .call(rt_funcs.rt_obj_insert, &[result_addr, obj_ptr, key_ptr, val_ptr]);

            (result_addr, TypeHint::Unknown)
        }

        Expr::SetPath { input_expr, path, value } => {
            // Phase 9-6: setpath(path; value) — set a value at a path.
            let (input_val_ptr, _) = codegen_expr(input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (path_ptr, _) = codegen_expr(path, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (val_ptr, _) = codegen_expr(value, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            let result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let result_addr = builder.ins().stack_addr(ptr_ty, result_slot, 0);

            // Call rt_setpath(out, input, path, val)
            builder
                .ins()
                .call(rt_funcs.rt_setpath, &[result_addr, input_val_ptr, path_ptr, val_ptr]);

            (result_addr, TypeHint::Unknown)
        }

        Expr::UnaryOp { op, operand } => {
            // Generate code for the operand
            let (operand_ptr, _) =
                codegen_expr(operand, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Allocate a StackSlot for the result (16 bytes, 8-byte aligned)
            let result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                16,
                8,
            ));
            let result_addr = builder.ins().stack_addr(ptr_ty, result_slot, 0);

            // Call the appropriate unary runtime function (out, v)
            let func_ref = match op {
                UnaryOp::Length => rt_funcs.rt_length,
                UnaryOp::Type => rt_funcs.rt_type,
                UnaryOp::ToString => rt_funcs.rt_tostring,
                UnaryOp::ToNumber => rt_funcs.rt_tonumber,
                UnaryOp::Keys => rt_funcs.rt_keys,
                UnaryOp::Negate => rt_funcs.rt_negate,
                // Phase 5-2: additional unary builtins
                UnaryOp::Sort => rt_funcs.rt_sort,
                UnaryOp::KeysUnsorted => rt_funcs.rt_keys_unsorted,
                UnaryOp::Floor => rt_funcs.rt_floor,
                UnaryOp::Ceil => rt_funcs.rt_ceil,
                UnaryOp::Round => rt_funcs.rt_round,
                UnaryOp::Fabs => rt_funcs.rt_fabs,
                UnaryOp::Explode => rt_funcs.rt_explode,
                UnaryOp::Implode => rt_funcs.rt_implode,
                // Phase 5-3: jq-defined functions implemented as direct runtime calls
                UnaryOp::Add => rt_funcs.rt_jq_add,
                UnaryOp::Reverse => rt_funcs.rt_reverse,
                UnaryOp::ToEntries => rt_funcs.rt_to_entries,
                UnaryOp::FromEntries => rt_funcs.rt_from_entries,
                UnaryOp::Unique => rt_funcs.rt_unique,
                UnaryOp::AsciiDowncase => rt_funcs.rt_ascii_downcase,
                UnaryOp::AsciiUpcase => rt_funcs.rt_ascii_upcase,
                // Phase 8-1: Missing unary builtins
                UnaryOp::Flatten => rt_funcs.rt_flatten,
                UnaryOp::Min => rt_funcs.rt_min,
                UnaryOp::Max => rt_funcs.rt_max,
                // Phase 9-3: Format string functions
                UnaryOp::FormatBase64 => rt_funcs.rt_format_base64,
                UnaryOp::FormatBase64d => rt_funcs.rt_format_base64d,
                UnaryOp::FormatHtml => rt_funcs.rt_format_html,
                UnaryOp::FormatUri => rt_funcs.rt_format_uri,
                UnaryOp::FormatUrid => rt_funcs.rt_format_urid,
                UnaryOp::FormatCsv => rt_funcs.rt_format_csv,
                UnaryOp::FormatTsv => rt_funcs.rt_format_tsv,
                UnaryOp::FormatJson => rt_funcs.rt_format_json,
                UnaryOp::FormatSh => rt_funcs.rt_format_sh,
                // Phase 9-6: Remaining builtin functions
                UnaryOp::Any => rt_funcs.rt_any,
                UnaryOp::All => rt_funcs.rt_all,
                UnaryOp::ToJson => rt_funcs.rt_tojson,
                UnaryOp::FromJson => rt_funcs.rt_fromjson,
                UnaryOp::Debug => rt_funcs.rt_debug,
                UnaryOp::Env => rt_funcs.rt_env,
                UnaryOp::Builtins => rt_funcs.rt_builtins,
                UnaryOp::Infinite => rt_funcs.rt_infinite,
                UnaryOp::Nan => rt_funcs.rt_nan,
                UnaryOp::IsInfinite => rt_funcs.rt_isinfinite,
                UnaryOp::IsNan => rt_funcs.rt_isnan,
                UnaryOp::IsNormal => rt_funcs.rt_isnormal,
                // Phase 11: Make error value from input
                UnaryOp::MakeError => rt_funcs.rt_make_error,
                // Math unary functions
                UnaryOp::Sqrt => rt_funcs.rt_sqrt,
                UnaryOp::Sin => rt_funcs.rt_sin,
                UnaryOp::Cos => rt_funcs.rt_cos,
                UnaryOp::Tan => rt_funcs.rt_tan,
                UnaryOp::Asin => rt_funcs.rt_asin,
                UnaryOp::Acos => rt_funcs.rt_acos,
                UnaryOp::Atan => rt_funcs.rt_atan,
                UnaryOp::Exp => rt_funcs.rt_exp,
                UnaryOp::Exp2 => rt_funcs.rt_exp2,
                UnaryOp::Exp10 => rt_funcs.rt_exp10,
                UnaryOp::Log => rt_funcs.rt_log,
                UnaryOp::Log2 => rt_funcs.rt_log2,
                UnaryOp::Log10 => rt_funcs.rt_log10,
                UnaryOp::Cbrt => rt_funcs.rt_cbrt,
                UnaryOp::Significand => rt_funcs.rt_significand,
                UnaryOp::Exponent => rt_funcs.rt_exponent,
                UnaryOp::Logb => rt_funcs.rt_logb,
                UnaryOp::NearbyInt => rt_funcs.rt_nearbyint,
                UnaryOp::Trunc => rt_funcs.rt_trunc,
                UnaryOp::Rint => rt_funcs.rt_rint,
                UnaryOp::J0 => rt_funcs.rt_j0,
                UnaryOp::J1 => rt_funcs.rt_j1,
                // Transpose, utf8bytelength, toboolean, trim, date/time
                UnaryOp::Transpose => rt_funcs.rt_transpose,
                UnaryOp::Utf8ByteLength => rt_funcs.rt_utf8bytelength,
                UnaryOp::ToBoolean => rt_funcs.rt_toboolean,
                UnaryOp::Trim => rt_funcs.rt_trim,
                UnaryOp::Ltrim => rt_funcs.rt_ltrim,
                UnaryOp::Rtrim => rt_funcs.rt_rtrim,
                UnaryOp::Gmtime => rt_funcs.rt_gmtime,
                UnaryOp::Mktime => rt_funcs.rt_mktime,
                UnaryOp::Now => rt_funcs.rt_now,
                // Not yet implemented (require generator or special handling)
                UnaryOp::Values | UnaryOp::Not | UnaryOp::Ascii => {
                    panic!("codegen_expr: UnaryOp::{} not yet implemented (CALL_JQ function)", op);
                }
            };

            builder.ins().call(func_ref, &[result_addr, operand_ptr]);

            (result_addr, TypeHint::Unknown)
        }

        Expr::IfThenElse {
            cond,
            then_branch,
            else_branch,
        } => {
            // Phase 3: Conditional branching with multiple basic blocks.
            //
            // 1. Evaluate condition in current block
            // 2. Call rt_is_truthy to get an i32 (0 or 1)
            // 3. brif → then_block or else_block
            // 4. Each branch generates its code and jumps to merge_block
            // 5. merge_block receives the result via block parameter

            // Step 1: Generate condition value in the current block
            let (cond_ptr, _) = codegen_expr(cond, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Step 1b: Check if cond is Error — if so, propagate it directly
            let cond_error_call = builder.ins().call(rt_funcs.rt_is_error, &[cond_ptr]);
            let cond_is_error = builder.inst_results(cond_error_call)[0];

            let cond_ok_block = builder.create_block();
            let cond_error_block = builder.create_block();
            let merge_block = builder.create_block();

            // merge_block receives the result pointer as a block parameter
            builder.append_block_param(merge_block, ptr_ty);

            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(cond_is_error, cond_error_block, empty_args, cond_ok_block, empty_args);

            // cond is Error → propagate directly to merge
            builder.switch_to_block(cond_error_block);
            builder.seal_block(cond_error_block);
            let error_arg = BlockArg::Value(cond_ptr);
            builder.ins().jump(merge_block, &[error_arg]);

            // cond is OK → proceed with truthy check
            builder.switch_to_block(cond_ok_block);
            builder.seal_block(cond_ok_block);

            // Step 2: Call rt_is_truthy(cond_ptr) → i32
            let call_inst = builder.ins().call(rt_funcs.rt_is_truthy, &[cond_ptr]);
            let is_truthy = builder.inst_results(call_inst)[0]; // i32: 0 or 1

            // Step 3: Create basic blocks for then, else
            let then_block = builder.create_block();
            let else_block = builder.create_block();

            // Branch: if truthy → then_block, else → else_block
            builder.ins().brif(is_truthy, then_block, empty_args, else_block, empty_args);

            // Step 4a: Generate then branch
            builder.switch_to_block(then_block);
            builder.seal_block(then_block); // single predecessor: cond_ok_block
            let (then_result, _) = codegen_expr(then_branch, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let then_arg = BlockArg::Value(then_result);
            builder.ins().jump(merge_block, &[then_arg]);

            // Step 4b: Generate else branch
            builder.switch_to_block(else_block);
            builder.seal_block(else_block); // single predecessor: cond_ok_block
            let (else_result, _) = codegen_expr(else_branch, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let else_arg = BlockArg::Value(else_result);
            builder.ins().jump(merge_block, &[else_arg]);

            // Step 5: Merge block — three predecessors: cond_error_block, then_block, else_block
            builder.switch_to_block(merge_block);
            builder.seal_block(merge_block);

            // The result pointer is the block parameter
            // Type hint is Unknown since we don't know which branch was taken
            (builder.block_params(merge_block)[0], TypeHint::Unknown)
        }

        Expr::TryCatch {
            try_expr,
            catch_expr,
        } => {
            // Phase 3/11: Try-catch in scalar context.
            //
            // Special case: catch_expr is Empty (from `expr?` desugaring).
            // In scalar context, we can't produce 0 outputs, so we propagate
            // the Error value as-is. The outer TryCatch or generator context
            // will handle it properly.

            // Step 1: Generate try body
            let (try_result_ptr, _) = codegen_expr(try_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            if matches!(catch_expr.as_ref(), Expr::Empty) {
                // For `expr?` in scalar context: just pass through the result.
                // If it's an Error, it will be caught by an outer try/catch.
                // This enables chaining like `.foo?.bar?` where inner errors
                // propagate to the outermost handler.
                (try_result_ptr, TypeHint::Unknown)
            } else {
                // Normal try-catch with a real catch expression.
                // Step 2: Check if result is an error
                let call_inst = builder.ins().call(rt_funcs.rt_is_error, &[try_result_ptr]);
                let is_error = builder.inst_results(call_inst)[0]; // i32: 0 or 1

                // Step 3: Create basic blocks
                let catch_block = builder.create_block();
                let success_block = builder.create_block();
                let merge_block = builder.create_block();

                // merge_block receives the result pointer as a block parameter
                builder.append_block_param(merge_block, ptr_ty);

                // Branch: if error → catch_block, else → success_block
                let empty_args: &[BlockArg] = &[];
                builder.ins().brif(is_error, catch_block, empty_args, success_block, empty_args);

                // Step 4: Success block — pass through the try result
                builder.switch_to_block(success_block);
                builder.seal_block(success_block);
                let success_arg = BlockArg::Value(try_result_ptr);
                builder.ins().jump(merge_block, &[success_arg]);

                // Step 5: Catch block — extract error message and evaluate catch_expr
                builder.switch_to_block(catch_block);
                builder.seal_block(catch_block);

                // Extract error value: Value::Error(json) → original value
                let extracted_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let extracted_ptr = builder.ins().stack_addr(ptr_ty, extracted_slot, 0);
                builder.ins().call(rt_funcs.rt_extract_error, &[extracted_ptr, try_result_ptr]);

                let (catch_result_ptr, _) = codegen_expr(catch_expr, extracted_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
                let catch_arg = BlockArg::Value(catch_result_ptr);
                builder.ins().jump(merge_block, &[catch_arg]);

                // Step 6: Merge
                builder.switch_to_block(merge_block);
                builder.seal_block(merge_block);

                (builder.block_params(merge_block)[0], TypeHint::Unknown)
            }
        }

        Expr::LetBinding { var_index, value, body } => {
            // Phase 4-4: Variable binding.
            //
            // 1. Evaluate `value` to get a pointer to the bound value
            // 2. Allocate a StackSlot for the variable and copy the value into it
            // 3. Store the slot in var_slots so LoadVar can find it
            // 4. Evaluate `body` (which may reference the variable)
            // 5. Return the body's result

            // Step 1: Evaluate the value expression
            let (value_ptr, _) = codegen_expr(value, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Step 2: Allocate a stack slot and copy the value (16 bytes = sizeof(Value))
            let var_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                16,
                8,
            ));
            let var_addr = builder.ins().stack_addr(ptr_ty, var_slot, 0);

            // Copy 16 bytes from value_ptr to var_slot (memcpy equivalent)
            let val_lo = builder.ins().load(types::I64, MemFlags::trusted(), value_ptr, 0);
            let val_hi = builder.ins().load(types::I64, MemFlags::trusted(), value_ptr, 8);
            builder.ins().store(MemFlags::trusted(), val_lo, var_addr, 0);
            builder.ins().store(MemFlags::trusted(), val_hi, var_addr, 8);

            // Step 3: Register the variable slot
            let prev_slot = var_slots.insert(*var_index, var_slot);

            // Step 4: Evaluate the body
            let (body_result, body_hint) = codegen_expr(body, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Restore previous slot (for nested scopes with same var_index)
            if let Some(prev) = prev_slot {
                var_slots.insert(*var_index, prev);
            } else {
                var_slots.remove(var_index);
            }

            (body_result, body_hint)
        }

        Expr::LoadVar { var_index } => {
            // Phase 4-4: Variable reference.
            //
            // Return the address of the variable's stack slot.
            let var_slot = var_slots.get(var_index).unwrap_or_else(|| {
                panic!("codegen_expr: LoadVar references undefined variable {}", var_index);
            });
            (builder.ins().stack_addr(ptr_ty, *var_slot, 0), TypeHint::Unknown)
        }

        Expr::Reduce { source, init, var_index, acc_index, update } => {
            // Phase 4-5: Reduce (1→1).
            //
            // 1. Evaluate init to get the initial accumulator value
            // 2. Store it in acc_slot
            // 3. Loop over source elements:
            //    a. Store element in var_slot ($x)
            //    b. Load acc from acc_slot as input
            //    c. Evaluate update with acc as input
            //    d. Store result back in acc_slot
            // 4. Return final acc value

            // Step 1: Evaluate init
            let (init_ptr, _) = codegen_expr(init, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Step 2: Allocate acc_slot and copy init value
            let acc_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let acc_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            let init_lo = builder.ins().load(types::I64, MemFlags::trusted(), init_ptr, 0);
            let init_hi = builder.ins().load(types::I64, MemFlags::trusted(), init_ptr, 8);
            builder.ins().store(MemFlags::trusted(), init_lo, acc_addr, 0);
            builder.ins().store(MemFlags::trusted(), init_hi, acc_addr, 8);

            // Register acc_slot
            let prev_acc_slot = var_slots.insert(*acc_index, acc_slot);

            // Step 3: Evaluate source container.
            // If the source is a generator (e.g., range(n)), collect its outputs
            // into a temporary array first, then iterate over that.
            let raw_container_ptr = if is_generator(source) {
                // Collect generator outputs into a temporary array
                let gen_collect_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let gen_collect_ptr = builder.ins().stack_addr(ptr_ty, gen_collect_slot, 0);
                builder.ins().call(rt_funcs.rt_collect_init, &[gen_collect_ptr]);
                let collect_fn_addr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_collect_append);
                let callback_sig = builder.func.import_signature(cranelift_codegen::ir::Signature {
                    params: vec![AbiParam::new(ptr_ty), AbiParam::new(ptr_ty)],
                    returns: vec![],
                    call_conv: cranelift_codegen::isa::CallConv::SystemV,
                });
                codegen_generator(
                    source, input_ptr, collect_fn_addr, gen_collect_ptr, callback_sig,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );
                gen_collect_ptr
            } else {
                let (ptr, _) = codegen_expr(source, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
                ptr
            };

            // Phase 6-2: Prepare container for O(1) iteration (converts objects to arrays)
            let reduce_prep_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let container_ptr = builder.ins().stack_addr(ptr_ty, reduce_prep_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_prepare, &[container_ptr, raw_container_ptr]);

            // Get length
            let len_call = builder.ins().call(rt_funcs.rt_iter_length, &[container_ptr]);
            let len = builder.inst_results(len_call)[0]; // i64

            // Create loop blocks
            let loop_header = builder.create_block();
            let loop_body = builder.create_block();
            let loop_exit = builder.create_block();

            builder.append_block_param(loop_header, types::I64);

            let zero = builder.ins().iconst(types::I64, 0);
            let zero_arg = BlockArg::Value(zero);
            builder.ins().jump(loop_header, &[zero_arg]);

            // Loop header: check idx < len
            builder.switch_to_block(loop_header);
            let idx = builder.block_params(loop_header)[0];
            let cmp = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, idx, len,
            );
            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(cmp, loop_body, empty_args, loop_exit, empty_args);

            // Loop body
            builder.switch_to_block(loop_body);
            builder.seal_block(loop_body);

            // Get element
            let elem_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let elem_ptr = builder.ins().stack_addr(ptr_ty, elem_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_get, &[elem_ptr, container_ptr, idx]);

            // Store element in var_slot ($x)
            let var_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let var_addr = builder.ins().stack_addr(ptr_ty, var_slot, 0);
            let elem_lo = builder.ins().load(types::I64, MemFlags::trusted(), elem_ptr, 0);
            let elem_hi = builder.ins().load(types::I64, MemFlags::trusted(), elem_ptr, 8);
            builder.ins().store(MemFlags::trusted(), elem_lo, var_addr, 0);
            builder.ins().store(MemFlags::trusted(), elem_hi, var_addr, 8);

            let prev_var_slot = var_slots.insert(*var_index, var_slot);

            // Evaluate update with acc as input
            let current_acc_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            let (update_result, _) = codegen_expr(update, current_acc_addr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Store update result back in acc_slot
            let new_acc_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            let res_lo = builder.ins().load(types::I64, MemFlags::trusted(), update_result, 0);
            let res_hi = builder.ins().load(types::I64, MemFlags::trusted(), update_result, 8);
            builder.ins().store(MemFlags::trusted(), res_lo, new_acc_addr, 0);
            builder.ins().store(MemFlags::trusted(), res_hi, new_acc_addr, 8);

            // Restore var_slot
            if let Some(prev) = prev_var_slot {
                var_slots.insert(*var_index, prev);
            } else {
                var_slots.remove(var_index);
            }

            // Increment index and jump back
            let one = builder.ins().iconst(types::I64, 1);
            let next_idx = builder.ins().iadd(idx, one);
            let next_arg = BlockArg::Value(next_idx);
            builder.ins().jump(loop_header, &[next_arg]);

            // Loop exit
            builder.switch_to_block(loop_exit);
            builder.seal_block(loop_exit);
            builder.seal_block(loop_header);

            // Restore acc_slot
            if let Some(prev) = prev_acc_slot {
                var_slots.insert(*acc_index, prev);
            } else {
                var_slots.remove(acc_index);
            }

            // Return final acc address
            (builder.ins().stack_addr(ptr_ty, acc_slot, 0), TypeHint::Unknown)
        }

        Expr::Collect { generator, acc_index } => {
            // Phase 5-1: Array constructor [expr].
            //
            // 1. Allocate a StackSlot for the accumulator array (16 bytes = sizeof(Value))
            // 2. Call rt_collect_init(acc_slot) to write an empty array there
            // 3. Run the inner generator with rt_collect_append as callback
            //    and acc_slot address as ctx
            // 4. Return acc_slot address (the completed array)

            // Step 1: Allocate accumulator slot
            let acc_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                16,
                8,
            ));
            let acc_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);

            // Step 2: Initialize with empty array
            builder.ins().call(rt_funcs.rt_collect_init, &[acc_addr]);

            // Register acc_slot in var_slots so inner generator can reference it
            let prev_acc_slot = var_slots.insert(*acc_index, acc_slot);

            // Step 3: Get the address of rt_collect_append as a function pointer.
            // We use the declared FuncRef and func_addr to get a callable pointer.
            let collect_append_addr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_collect_append);

            // The callback signature is fn(value_ptr: ptr, ctx: ptr) → void
            // which matches rt_collect_append(elem_ptr: ptr, ctx: ptr) → void.
            // We pass acc_addr as the ctx so each element gets appended to it.
            let collect_acc_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);

            // Import the callback signature (same as the outer callback sig)
            let mut collect_cb_sig = Signature::new(CallConv::SystemV);
            collect_cb_sig.params.push(AbiParam::new(ptr_ty)); // value_ptr
            collect_cb_sig.params.push(AbiParam::new(ptr_ty)); // ctx
            let collect_cb_sig_ref = builder.import_signature(collect_cb_sig);

            // Run the inner generator with rt_collect_append as callback
            codegen_generator(
                generator, input_ptr,
                collect_append_addr, collect_acc_addr,
                collect_cb_sig_ref,
                builder, ptr_ty, rt_funcs, literals, var_slots,
            );

            // Restore var_slots
            if let Some(prev) = prev_acc_slot {
                var_slots.insert(*acc_index, prev);
            } else {
                var_slots.remove(acc_index);
            }

            // Step 4: Return the accumulator address (now contains the completed array)
            (builder.ins().stack_addr(ptr_ty, acc_slot, 0), TypeHint::Unknown)
        }

        Expr::Alternative { primary, fallback } => {
            // Phase 5-2/11: Alternative operator `//`.
            //
            // 1. Evaluate primary
            // 2. Check if result is truthy AND not an Error
            //    (Error values from try-catch propagation should fall through to fallback)
            // 3. If truthy & non-error → return primary result
            // 4. If falsy or error → evaluate fallback and return that

            // Step 1: Evaluate primary
            let (primary_ptr, _) = codegen_expr(primary, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Step 2: Check truthiness AND check not-error
            let truthy_inst = builder.ins().call(rt_funcs.rt_is_truthy, &[primary_ptr]);
            let is_truthy = builder.inst_results(truthy_inst)[0]; // i32: 0 or 1

            let error_inst = builder.ins().call(rt_funcs.rt_is_error, &[primary_ptr]);
            let is_error = builder.inst_results(error_inst)[0]; // i32: 0 or 1

            // use_primary = is_truthy AND NOT is_error
            // is_not_error = is_error XOR 1
            let one = builder.ins().iconst(types::I32, 1);
            let is_not_error = builder.ins().bxor(is_error, one);
            let use_primary = builder.ins().band(is_truthy, is_not_error);

            // Step 3: Create basic blocks
            let truthy_block = builder.create_block();
            let falsy_block = builder.create_block();
            let merge_block = builder.create_block();

            builder.append_block_param(merge_block, ptr_ty);

            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(use_primary, truthy_block, empty_args, falsy_block, empty_args);

            // Truthy: return primary result
            builder.switch_to_block(truthy_block);
            builder.seal_block(truthy_block);
            let truthy_arg = BlockArg::Value(primary_ptr);
            builder.ins().jump(merge_block, &[truthy_arg]);

            // Falsy: evaluate fallback
            builder.switch_to_block(falsy_block);
            builder.seal_block(falsy_block);
            let (fallback_ptr, _) = codegen_expr(fallback, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let falsy_arg = BlockArg::Value(fallback_ptr);
            builder.ins().jump(merge_block, &[falsy_arg]);

            // Merge
            builder.switch_to_block(merge_block);
            builder.seal_block(merge_block);

            (builder.block_params(merge_block)[0], TypeHint::Unknown)
        }

        Expr::ClosureApply { op, input_expr, key_expr } => {
            // Phase 9-1: Closure-based array operations (sort_by, group_by, etc.)
            //
            // 1. Evaluate input_expr to get the array
            // 2. Build a keys array by iterating over each element and applying key_expr
            //    (using the same loop pattern as Each + Collect)
            // 3. Call the appropriate rt_*_by_keys(out, input_arr, keys_arr) runtime function

            // Step 1: Evaluate the input array
            let (arr_ptr, _) = codegen_expr(
                input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots,
            );

            // Step 2: Build keys array — iterate over each element, apply key_expr
            // 2a. Prepare the container for iteration
            let prep_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let prep_ptr = builder.ins().stack_addr(ptr_ty, prep_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_prepare, &[prep_ptr, arr_ptr]);

            // 2b. Get length
            let len_inst = builder.ins().call(rt_funcs.rt_iter_length, &[prep_ptr]);
            let len = builder.inst_results(len_inst)[0];

            // 2c. Initialize keys accumulator array
            let keys_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let keys_addr = builder.ins().stack_addr(ptr_ty, keys_slot, 0);
            builder.ins().call(rt_funcs.rt_collect_init, &[keys_addr]);

            // 2d. Loop: for idx = 0..len, get element, apply key_expr, append to keys
            let loop_header = builder.create_block();
            let loop_body = builder.create_block();
            let loop_exit = builder.create_block();

            builder.append_block_param(loop_header, types::I64);

            let zero = builder.ins().iconst(types::I64, 0);
            let zero_arg = BlockArg::Value(zero);
            builder.ins().jump(loop_header, &[zero_arg]);

            builder.switch_to_block(loop_header);
            let idx = builder.block_params(loop_header)[0];
            let cmp = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, idx, len,
            );
            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(cmp, loop_body, empty_args, loop_exit, empty_args);

            builder.switch_to_block(loop_body);
            builder.seal_block(loop_body);

            // Get element
            let elem_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let elem_ptr = builder.ins().stack_addr(ptr_ty, elem_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_get, &[elem_ptr, prep_ptr, idx]);

            // Apply key_expr to element (codegen_expr with elem_ptr as input)
            let (key_val_ptr, _) = codegen_expr(
                key_expr, elem_ptr, builder, ptr_ty, rt_funcs, literals, var_slots,
            );

            // Append key to keys array
            // rt_collect_append signature: fn(elem_ptr, ctx=arr_ptr)
            builder.ins().call(rt_funcs.rt_collect_append, &[key_val_ptr, keys_addr]);

            // Increment index and loop back
            let one = builder.ins().iconst(types::I64, 1);
            let next_idx = builder.ins().iadd(idx, one);
            let next_arg = BlockArg::Value(next_idx);
            builder.ins().jump(loop_header, &[next_arg]);

            builder.switch_to_block(loop_exit);
            builder.seal_block(loop_exit);
            builder.seal_block(loop_header);

            // Step 3: Call the runtime function with (out, input_arr, keys_arr)
            let out_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let out_addr = builder.ins().stack_addr(ptr_ty, out_slot, 0);
            let keys_arr_addr = builder.ins().stack_addr(ptr_ty, keys_slot, 0);

            let rt_func = match op {
                ClosureOp::SortBy => rt_funcs.rt_sort_by_keys,
                ClosureOp::GroupBy => rt_funcs.rt_group_by_keys,
                ClosureOp::UniqueBy => rt_funcs.rt_unique_by_keys,
                ClosureOp::MinBy => rt_funcs.rt_min_by_keys,
                ClosureOp::MaxBy => rt_funcs.rt_max_by_keys,
            };
            builder.ins().call(rt_func, &[out_addr, arr_ptr, keys_arr_addr]);

            (builder.ins().stack_addr(ptr_ty, out_slot, 0), TypeHint::Unknown)
        }

        Expr::Comma { .. } | Expr::Empty | Expr::Each { .. } | Expr::EachOpt { .. } | Expr::Foreach { .. } | Expr::Recurse { .. } | Expr::Range { .. } | Expr::PathOf { .. } | Expr::While { .. } | Expr::Limit { .. } | Expr::Skip { .. } => {
            // Comma, Empty, Each, Foreach, Recurse, Range, PathOf, While are generator expressions that produce 0 or N outputs.
            // They should only be handled by codegen_generator, not codegen_expr.
            // If we reach here, it means a generator appeared in a scalar context,
            // which is a compiler bug.
            panic!(
                "codegen_expr: generator expression {:?} in scalar context — use codegen_generator instead",
                expr
            );
        }

        Expr::Until { input_expr, cond, update } => {
            // Phase 11: Until (1→1 expression).
            //
            // Starting with input_expr, loop: if cond is true, return current value.
            // Otherwise apply update and repeat.

            let out_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let out_addr = builder.ins().stack_addr(ptr_ty, out_slot, 0);

            // Evaluate input_expr to get initial value
            let (init_ptr, _) = codegen_expr(input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Allocate accumulator slot
            let acc_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let acc_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);

            // Copy initial value to accumulator
            let in_lo = builder.ins().load(types::I64, MemFlags::trusted(), init_ptr, 0);
            let in_hi = builder.ins().load(types::I64, MemFlags::trusted(), init_ptr, 8);
            builder.ins().store(MemFlags::trusted(), in_lo, acc_addr, 0);
            builder.ins().store(MemFlags::trusted(), in_hi, acc_addr, 8);

            // Create loop blocks
            let loop_header = builder.create_block();
            let loop_body = builder.create_block();
            let loop_exit = builder.create_block();

            builder.ins().jump(loop_header, &[]);

            // Loop header: evaluate cond with acc as input
            builder.switch_to_block(loop_header);

            let current_acc = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            let (cond_result, _) = codegen_expr(cond, current_acc, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Check truthiness
            let is_truthy_call = builder.ins().call(rt_funcs.rt_is_truthy, &[cond_result]);
            let is_truthy = builder.inst_results(is_truthy_call)[0]; // i32

            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(is_truthy, loop_exit, empty_args, loop_body, empty_args);

            // Loop body: apply update
            builder.switch_to_block(loop_body);
            builder.seal_block(loop_body);

            let body_acc = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            let (update_result, _) = codegen_expr(update, body_acc, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Store update result back into accumulator
            let new_acc_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            let res_lo = builder.ins().load(types::I64, MemFlags::trusted(), update_result, 0);
            let res_hi = builder.ins().load(types::I64, MemFlags::trusted(), update_result, 8);
            builder.ins().store(MemFlags::trusted(), res_lo, new_acc_addr, 0);
            builder.ins().store(MemFlags::trusted(), res_hi, new_acc_addr, 8);

            builder.ins().jump(loop_header, &[]);

            // Loop exit: copy accumulator to output
            builder.switch_to_block(loop_exit);
            builder.seal_block(loop_exit);
            builder.seal_block(loop_header);

            let final_acc = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            let final_lo = builder.ins().load(types::I64, MemFlags::trusted(), final_acc, 0);
            let final_hi = builder.ins().load(types::I64, MemFlags::trusted(), final_acc, 8);
            builder.ins().store(MemFlags::trusted(), final_lo, out_addr, 0);
            builder.ins().store(MemFlags::trusted(), final_hi, out_addr, 8);

            (out_addr, TypeHint::Unknown)
        }

        Expr::Update { input_expr, path_expr, update_expr, is_plain_assign } => {
            // Phase 10-3: Update operator (|= and variants).

            // Special case: Slice assignment (.[start:end] = value)
            // Detected when path_expr is Index { expr: Input, key: ObjectInsert{...} }
            // building a {"start":..., "end":...} slice key.
            if *is_plain_assign {
                if let Some(slice_key_expr) = extract_slice_key(path_expr) {
                    // Evaluate input
                    let (input_val_ptr, _) = codegen_expr(
                        input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots,
                    );
                    // Evaluate the slice key (ObjectInsert building {"start":..., "end":...})
                    let (slice_key_ptr, _) = codegen_expr(
                        &slice_key_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots,
                    );
                    // Evaluate the update value (against original input for plain assign)
                    let (value_ptr, _) = codegen_expr(
                        update_expr, input_val_ptr, builder, ptr_ty, rt_funcs, literals, var_slots,
                    );
                    // Call rt_slice_assign(out, input, slice_key, value)
                    let result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot, 16, 8,
                    ));
                    let result_addr = builder.ins().stack_addr(ptr_ty, result_slot, 0);
                    builder.ins().call(rt_funcs.rt_slice_assign, &[result_addr, input_val_ptr, slice_key_ptr, value_ptr]);
                    return (result_addr, TypeHint::Unknown);
                }
            }

            // General case: path-based update
            //
            // Strategy:
            // 1. Analyze path_expr to create a descriptor Value
            // 2. Evaluate input
            // 3. Collect all paths using rt_path_of
            // 4. For each path: getpath → apply update_expr → setpath
            //
            // Instead of using a Cranelift inner function, we implement the
            // getpath/update/setpath loop inline using Cranelift blocks.

            // Step 1: Create path descriptor from the path expression
            let descriptor = analyze_path_descriptor(path_expr);
            let desc_idx = literals.values.len();
            literals.values.push(Box::new(descriptor));
            let offset = (desc_idx as i32) * (ptr_ty.bytes() as i32);
            let desc_ptr = builder.ins().load(ptr_ty, MemFlags::trusted(), literals.clif_param, offset);

            // Step 2: Evaluate input
            let (input_val_ptr, _) = codegen_expr(
                input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots,
            );

            // Step 3: Collect all paths into a temporary array
            let collect_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let collect_ptr = builder.ins().stack_addr(ptr_ty, collect_slot, 0);
            builder.ins().call(rt_funcs.rt_collect_init, &[collect_ptr]);

            let collect_fn_addr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_collect_append);
            builder.ins().call(rt_funcs.rt_path_of, &[input_val_ptr, desc_ptr, collect_fn_addr, collect_ptr]);

            // Step 4: Loop over collected paths and apply updates
            // Prepare paths array for iteration
            let prep_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let prepared_ptr = builder.ins().stack_addr(ptr_ty, prep_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_prepare, &[prepared_ptr, collect_ptr]);

            let len_call = builder.ins().call(rt_funcs.rt_iter_length, &[prepared_ptr]);
            let len = builder.inst_results(len_call)[0];

            // Detect if update_expr is a try-catch-empty pattern.
            // In that case, Error means "empty" (0 outputs) → collect path for deletion
            // instead of propagating the error.
            let update_is_try_empty = matches!(
                update_expr.as_ref(),
                Expr::TryCatch { catch_expr, .. } if matches!(catch_expr.as_ref(), Expr::Empty)
            );

            // For try-empty updates: collect paths to delete after the main loop
            let del_collect_ptr = if update_is_try_empty {
                let del_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let del_ptr = builder.ins().stack_addr(ptr_ty, del_slot, 0);
                builder.ins().call(rt_funcs.rt_collect_init, &[del_ptr]);
                Some(del_ptr)
            } else {
                None
            };

            // Accumulator: starts as input value, updated after each path
            let acc_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let acc_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            // Copy input into accumulator (16 bytes)
            {
                let lo = builder.ins().load(types::I64, MemFlags::trusted(), input_val_ptr, 0);
                let hi = builder.ins().load(types::I64, MemFlags::trusted(), input_val_ptr, 8);
                builder.ins().store(MemFlags::trusted(), lo, acc_addr, 0);
                builder.ins().store(MemFlags::trusted(), hi, acc_addr, 8);
            }

            let loop_header = builder.create_block();
            let loop_body = builder.create_block();
            let loop_exit = builder.create_block();

            builder.append_block_param(loop_header, types::I64);

            let zero = builder.ins().iconst(types::I64, 0);
            let zero_arg = BlockArg::Value(zero);
            builder.ins().jump(loop_header, &[zero_arg]);

            builder.switch_to_block(loop_header);
            let idx = builder.block_params(loop_header)[0];
            let cmp = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, idx, len,
            );
            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(cmp, loop_body, empty_args, loop_exit, empty_args);

            builder.switch_to_block(loop_body);
            builder.seal_block(loop_body);

            // Get the path at this index
            let path_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let path_ptr = builder.ins().stack_addr(ptr_ty, path_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_get, &[path_ptr, prepared_ptr, idx]);

            // getpath(acc, path) → current value
            let current_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let current_ptr = builder.ins().stack_addr(ptr_ty, current_slot, 0);
            builder.ins().call(rt_funcs.rt_getpath, &[current_ptr, acc_addr, path_ptr]);

            // Apply update_expr:
            // For |= (is_plain_assign=false): evaluate against getpath result (current_ptr)
            // For = (is_plain_assign=true): evaluate against original input (input_val_ptr)
            let update_input = if *is_plain_assign { input_val_ptr } else { current_ptr };
            let (updated_ptr, _) = codegen_expr(
                update_expr, update_input, builder, ptr_ty, rt_funcs, literals, var_slots,
            );

            // Check if update result is an Error
            let updated_tag = builder.ins().load(types::I64, MemFlags::trusted(), updated_ptr, 0);
            let error_tag = builder.ins().iconst(types::I64, crate::value::TAG_ERROR as i64);
            let update_is_error = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::Equal, updated_tag, error_tag,
            );

            let update_ok_block = builder.create_block();
            let update_error_block = builder.create_block();
            let after_update_block = builder.create_block();
            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(update_is_error, update_error_block, empty_args, update_ok_block, empty_args);

            // --- update OK: setpath(acc, path, updated) ---
            builder.switch_to_block(update_ok_block);
            builder.seal_block(update_ok_block);

            let new_acc_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let new_acc_ptr = builder.ins().stack_addr(ptr_ty, new_acc_slot, 0);
            builder.ins().call(rt_funcs.rt_setpath, &[new_acc_ptr, acc_addr, path_ptr, updated_ptr]);

            // Check if setpath itself returned an Error
            let setpath_tag = builder.ins().load(types::I64, MemFlags::trusted(), new_acc_ptr, 0);
            let setpath_is_error = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::Equal, setpath_tag, error_tag,
            );
            let setpath_ok_block = builder.create_block();
            let setpath_error_block = builder.create_block();
            builder.ins().brif(setpath_is_error, setpath_error_block, empty_args, setpath_ok_block, empty_args);

            // setpath error: propagate error and exit loop
            builder.switch_to_block(setpath_error_block);
            builder.seal_block(setpath_error_block);
            {
                let lo = builder.ins().load(types::I64, MemFlags::trusted(), new_acc_ptr, 0);
                let hi = builder.ins().load(types::I64, MemFlags::trusted(), new_acc_ptr, 8);
                builder.ins().store(MemFlags::trusted(), lo, acc_addr, 0);
                builder.ins().store(MemFlags::trusted(), hi, acc_addr, 8);
            }
            builder.ins().jump(loop_exit, &[]);

            // setpath OK: update acc and continue
            builder.switch_to_block(setpath_ok_block);
            builder.seal_block(setpath_ok_block);
            {
                let lo = builder.ins().load(types::I64, MemFlags::trusted(), new_acc_ptr, 0);
                let hi = builder.ins().load(types::I64, MemFlags::trusted(), new_acc_ptr, 8);
                builder.ins().store(MemFlags::trusted(), lo, acc_addr, 0);
                builder.ins().store(MemFlags::trusted(), hi, acc_addr, 8);
            }
            builder.ins().jump(after_update_block, &[]);

            // --- update Error ---
            builder.switch_to_block(update_error_block);
            builder.seal_block(update_error_block);

            if let Some(del_ptr) = del_collect_ptr {
                // try-catch-empty: Error means "empty" → collect path for later deletion
                // Don't modify acc; just collect the path to delete after the loop.
                builder.ins().call(rt_funcs.rt_collect_append, &[path_ptr, del_ptr]);
                builder.ins().jump(after_update_block, &[]);
            } else {
                // Non-try error: propagate error and exit loop
                {
                    let lo = builder.ins().load(types::I64, MemFlags::trusted(), updated_ptr, 0);
                    let hi = builder.ins().load(types::I64, MemFlags::trusted(), updated_ptr, 8);
                    builder.ins().store(MemFlags::trusted(), lo, acc_addr, 0);
                    builder.ins().store(MemFlags::trusted(), hi, acc_addr, 8);
                }
                builder.ins().jump(loop_exit, &[]);
            }

            builder.switch_to_block(after_update_block);
            builder.seal_block(after_update_block);

            // Increment index and loop back
            let one = builder.ins().iconst(types::I64, 1);
            let next_idx = builder.ins().iadd(idx, one);
            let next_arg = BlockArg::Value(next_idx);
            builder.ins().jump(loop_header, &[next_arg]);

            builder.switch_to_block(loop_exit);
            builder.seal_block(loop_exit);
            builder.seal_block(loop_header);

            // For try-empty updates: apply collected delpaths after the loop.
            // del_collect_ptr is a Value::Arr containing paths to delete (e.g., [[1],[3],[4]]).
            // rt_delpaths handles sorting paths in reverse order to preserve indices.
            if let Some(del_ptr) = del_collect_ptr {
                // rt_delpaths(new_acc, acc, del_paths_array)
                let final_acc_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let final_acc_ptr = builder.ins().stack_addr(ptr_ty, final_acc_slot, 0);
                builder.ins().call(rt_funcs.rt_delpaths, &[final_acc_ptr, acc_addr, del_ptr]);

                // Copy final result back to acc
                {
                    let lo = builder.ins().load(types::I64, MemFlags::trusted(), final_acc_ptr, 0);
                    let hi = builder.ins().load(types::I64, MemFlags::trusted(), final_acc_ptr, 8);
                    builder.ins().store(MemFlags::trusted(), lo, acc_addr, 0);
                    builder.ins().store(MemFlags::trusted(), hi, acc_addr, 8);
                }
            }

            (acc_addr, TypeHint::Unknown)
        }

        Expr::RegexTest { input_expr, re, flags } => {
            // Phase 10-2: test(re; flags) — returns bool.
            let (input_val_ptr, _) = codegen_expr(input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (re_ptr, _) = codegen_expr(re, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (flags_ptr, _) = codegen_expr(flags, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            let result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let result_addr = builder.ins().stack_addr(ptr_ty, result_slot, 0);
            builder.ins().call(rt_funcs.rt_regex_test, &[result_addr, input_val_ptr, re_ptr, flags_ptr]);
            (result_addr, TypeHint::Unknown)
        }

        Expr::RegexCapture { input_expr, re, flags } => {
            // Phase 10-2: capture(re; flags) — returns named captures object.
            let (input_val_ptr, _) = codegen_expr(input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (re_ptr, _) = codegen_expr(re, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (flags_ptr, _) = codegen_expr(flags, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            let result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let result_addr = builder.ins().stack_addr(ptr_ty, result_slot, 0);
            builder.ins().call(rt_funcs.rt_regex_capture, &[result_addr, input_val_ptr, re_ptr, flags_ptr]);
            (result_addr, TypeHint::Unknown)
        }

        Expr::RegexSub { input_expr, re, tostr, flags } => {
            // Phase 10-2: sub(re; tostr; flags) — replace first match.
            let (input_val_ptr, _) = codegen_expr(input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (re_ptr, _) = codegen_expr(re, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (tostr_ptr, _) = codegen_expr(tostr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (flags_ptr, _) = codegen_expr(flags, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            let result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let result_addr = builder.ins().stack_addr(ptr_ty, result_slot, 0);
            builder.ins().call(rt_funcs.rt_regex_sub, &[result_addr, input_val_ptr, re_ptr, tostr_ptr, flags_ptr]);
            (result_addr, TypeHint::Unknown)
        }

        Expr::RegexGsub { input_expr, re, tostr, flags } => {
            // Phase 10-2: gsub(re; tostr; flags) — replace all matches.
            let (input_val_ptr, _) = codegen_expr(input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (re_ptr, _) = codegen_expr(re, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (tostr_ptr, _) = codegen_expr(tostr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (flags_ptr, _) = codegen_expr(flags, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            let result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let result_addr = builder.ins().stack_addr(ptr_ty, result_slot, 0);
            builder.ins().call(rt_funcs.rt_regex_gsub, &[result_addr, input_val_ptr, re_ptr, tostr_ptr, flags_ptr]);
            (result_addr, TypeHint::Unknown)
        }

        Expr::RegexMatch { .. } | Expr::RegexScan { .. } => {
            // match() and scan() are generators — they should go through codegen_generator.
            // If we reach here in scalar context, it means they're collected in [expr].
            panic!(
                "codegen_expr: generator regex expression {:?} in scalar context — use codegen_generator instead",
                expr
            );
        }
    }
}

/// Check if an expression is a generator (can produce 0 or more outputs).
///
/// Returns true for expressions handled specially by `codegen_generator`:
/// Comma, Empty, Each, EachOpt, IfThenElse (with generator branches),
/// TryCatch (with generator try_expr), Foreach, Range, Recurse,
/// RegexMatch, RegexScan, LetBinding/Collect wrapping generators.
///
/// Phase 10-5: Used to decide whether TryCatch needs the wrapper callback approach.
fn is_generator(expr: &Expr) -> bool {
    match expr {
        Expr::Comma { .. } => true,
        Expr::Empty => true,
        Expr::Each { .. } => true,
        Expr::EachOpt { .. } => true,
        Expr::Foreach { .. } => true,
        Expr::Range { .. } => true,
        Expr::While { .. } => true,
        Expr::Recurse { .. } => true,
        Expr::RegexMatch { .. } => true,
        Expr::RegexScan { .. } => true,
        // IfThenElse can be a generator if its branches are generators
        // or if its condition is a generator (e.g., `if empty then ...`)
        Expr::IfThenElse { cond, then_branch, else_branch } => {
            is_generator(cond) || is_generator(then_branch) || is_generator(else_branch)
        }
        // TryCatch can be a generator if try_expr is a generator OR catch_expr is Empty
        // (catch empty means 0 outputs on error, so the expression produces 0 or 1 outputs)
        Expr::TryCatch { try_expr, catch_expr } => {
            is_generator(try_expr) || matches!(catch_expr.as_ref(), Expr::Empty)
        }
        // LetBinding is a generator if its body is a generator
        Expr::LetBinding { body, .. } => is_generator(body),
        // PathOf is a generator (yields multiple paths)
        Expr::PathOf { .. } => true,
        // Limit is a generator (yields up to n outputs)
        Expr::Limit { .. } => true,
        // Skip is a generator (yields outputs after skipping n)
        Expr::Skip { .. } => true,
        // Collect is 1→1 (collects generator into array)
        Expr::Collect { .. } => false,
        // Update is 1→1 (returns the updated value)
        Expr::Update { .. } => false,
        // Check if any sub-expression is a generator (caused by pipe expansion
        // via substitute_input, e.g., `(1,2) | isnan` becomes `UnaryOp(Isnan, Comma(1,2))`)
        Expr::UnaryOp { operand, .. } => is_generator(operand),
        Expr::BinOp { lhs, rhs, .. } => is_generator(lhs) || is_generator(rhs),
        Expr::Alternative { primary, fallback } => is_generator(primary) || is_generator(fallback),
        Expr::Index { expr, key } => is_generator(expr) || is_generator(key),
        Expr::IndexOpt { expr, key } => is_generator(expr) || is_generator(key),
        // All other expressions are 1→1
        _ => false,
    }
}

/// Extract the innermost generator sub-expression from an expression,
/// returning (generator, template) where template has the generator
/// replaced with Expr::Input.
///
/// This is used to handle cases where pipe expansion (substitute_input)
/// places a generator inside a scalar expression, e.g.,
/// `(1,2) | isnan` → `UnaryOp(Isnan, Comma(1,2))`.
/// We extract Comma(1,2) as the generator and UnaryOp(Isnan, Input) as the template.
fn extract_inner_generator(expr: &Expr) -> Option<(&Expr, Expr)> {
    match expr {
        Expr::UnaryOp { op, operand } => {
            if is_generator(operand) {
                // Check if operand itself has a deeper generator
                if let Some((gen_expr, inner_template)) = extract_inner_generator(operand) {
                    Some((gen_expr, Expr::UnaryOp { op: op.clone(), operand: Box::new(inner_template) }))
                } else {
                    Some((operand.as_ref(), Expr::UnaryOp { op: op.clone(), operand: Box::new(Expr::Input) }))
                }
            } else {
                None
            }
        }
        Expr::BinOp { op, lhs, rhs } => {
            if is_generator(lhs) {
                if let Some((gen_expr, inner_template)) = extract_inner_generator(lhs) {
                    Some((gen_expr, Expr::BinOp { op: op.clone(), lhs: Box::new(inner_template), rhs: rhs.clone() }))
                } else {
                    Some((lhs.as_ref(), Expr::BinOp { op: op.clone(), lhs: Box::new(Expr::Input), rhs: rhs.clone() }))
                }
            } else if is_generator(rhs) {
                if let Some((gen_expr, inner_template)) = extract_inner_generator(rhs) {
                    Some((gen_expr, Expr::BinOp { op: op.clone(), lhs: lhs.clone(), rhs: Box::new(inner_template) }))
                } else {
                    Some((rhs.as_ref(), Expr::BinOp { op: op.clone(), lhs: lhs.clone(), rhs: Box::new(Expr::Input) }))
                }
            } else {
                None
            }
        }
        Expr::Index { expr: inner, key } => {
            if is_generator(inner) {
                if let Some((gen_expr, inner_template)) = extract_inner_generator(inner) {
                    Some((gen_expr, Expr::Index { expr: Box::new(inner_template), key: key.clone() }))
                } else {
                    Some((inner.as_ref(), Expr::Index { expr: Box::new(Expr::Input), key: key.clone() }))
                }
            } else if is_generator(key) {
                if let Some((gen_expr, inner_template)) = extract_inner_generator(key) {
                    Some((gen_expr, Expr::Index { expr: inner.clone(), key: Box::new(inner_template) }))
                } else {
                    Some((key.as_ref(), Expr::Index { expr: inner.clone(), key: Box::new(Expr::Input) }))
                }
            } else {
                None
            }
        }
        Expr::IndexOpt { expr: inner, key } => {
            if is_generator(inner) {
                if let Some((gen_expr, inner_template)) = extract_inner_generator(inner) {
                    Some((gen_expr, Expr::IndexOpt { expr: Box::new(inner_template), key: key.clone() }))
                } else {
                    Some((inner.as_ref(), Expr::IndexOpt { expr: Box::new(Expr::Input), key: key.clone() }))
                }
            } else if is_generator(key) {
                if let Some((gen_expr, inner_template)) = extract_inner_generator(key) {
                    Some((gen_expr, Expr::IndexOpt { expr: inner.clone(), key: Box::new(inner_template) }))
                } else {
                    Some((key.as_ref(), Expr::IndexOpt { expr: inner.clone(), key: Box::new(Expr::Input) }))
                }
            } else {
                None
            }
        }
        // Alternative with generator primary/fallback requires special handling
        // (fallback needs the original input, not generator output), so we don't
        // extract it here — it's handled directly in codegen_generator.
        Expr::Alternative { .. } => None,
        // For direct generators, there's no outer template
        _ => None,
    }
}

/// Recursively generate CLIF IR for a generator expression.
///
/// Unlike `codegen_expr` which returns a single result pointer, this function
/// handles expressions that can produce 0 or more output values.  Each output
/// value is delivered by calling the callback via `call_indirect`.
///
/// - `Comma { left, right }`: recursively generate left, then right
/// - `Empty`: do nothing (0 outputs)
/// - `IfThenElse`: generate condition, then branch to then/else generators
/// - `TryCatch`: try generator, on error run catch generator
/// - All other expressions: delegate to `codegen_expr` and call callback once
fn codegen_generator(
    expr: &Expr,
    input_ptr: cranelift_codegen::ir::Value,
    callback_ptr: cranelift_codegen::ir::Value,
    ctx_ptr: cranelift_codegen::ir::Value,
    callback_sig_ref: cranelift_codegen::ir::SigRef,
    builder: &mut FunctionBuilder,
    ptr_ty: types::Type,
    rt_funcs: &RuntimeFuncRefs,
    literals: &mut LiteralPool,
    var_slots: &mut HashMap<u16, cranelift_codegen::ir::StackSlot>,
) {
    match expr {
        Expr::Comma { left, right } => {
            // Generate all outputs from left, then all outputs from right.
            codegen_generator(
                left, input_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                builder, ptr_ty, rt_funcs, literals, var_slots,
            );
            codegen_generator(
                right, input_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                builder, ptr_ty, rt_funcs, literals, var_slots,
            );
        }

        Expr::Empty => {
            // Produce zero outputs — don't call callback at all.
        }

        Expr::IfThenElse {
            cond,
            then_branch,
            else_branch,
        } => {
            // Phase 9-5: Detect `paths` pattern.
            //
            // jq compiles `paths` as:
            //   if (length(path(., recurse(.))) > 0) then path(., recurse(.)) else empty end
            //
            // We recognize this pattern and emit a simple rt_path_of("recurse") call,
            // which inherently skips the empty root path (length > 0 filter).
            if matches!(else_branch.as_ref(), Expr::Empty) {
                if let Expr::PathOf { path_expr, .. } = then_branch.as_ref() {
                    if matches!(path_expr.as_ref(), Expr::Recurse { .. }) {
                        // This is the `paths` pattern — emit PathOf(Recurse) directly.
                        // rt_path_of with "recurse" descriptor already excludes the root path.
                        codegen_generator(
                            then_branch, input_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                            builder, ptr_ty, rt_funcs, literals, var_slots,
                        );
                        return;
                    }
                }
            }

            // If condition is a generator (e.g., `if empty then ...`), we need
            // to iterate over its outputs; for each, do the truthy check.
            // If the generator produces 0 outputs (empty), nothing happens.
            if is_generator(cond) {
                // Collect condition outputs
                let gen_collect_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let gen_collect_ptr = builder.ins().stack_addr(ptr_ty, gen_collect_slot, 0);
                builder.ins().call(rt_funcs.rt_collect_init, &[gen_collect_ptr]);

                let collect_fn_addr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_collect_append);
                let cb_sig = builder.func.import_signature(cranelift_codegen::ir::Signature {
                    params: vec![AbiParam::new(ptr_ty), AbiParam::new(ptr_ty)],
                    returns: vec![],
                    call_conv: cranelift_codegen::isa::CallConv::SystemV,
                });
                codegen_generator(
                    cond, input_ptr, collect_fn_addr, gen_collect_ptr, cb_sig,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );

                let prep_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let arr_ptr = builder.ins().stack_addr(ptr_ty, prep_slot, 0);
                builder.ins().call(rt_funcs.rt_iter_prepare, &[arr_ptr, gen_collect_ptr]);

                let len_call = builder.ins().call(rt_funcs.rt_iter_length, &[arr_ptr]);
                let len = builder.inst_results(len_call)[0];

                let loop_header = builder.create_block();
                let loop_body = builder.create_block();
                let loop_exit = builder.create_block();

                builder.append_block_param(loop_header, types::I64);
                let zero = builder.ins().iconst(types::I64, 0);
                let zero_arg = BlockArg::Value(zero);
                builder.ins().jump(loop_header, &[zero_arg]);

                builder.switch_to_block(loop_header);
                let idx = builder.block_params(loop_header)[0];
                let cmp_val = builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, idx, len,
                );
                let empty_args: &[BlockArg] = &[];
                builder.ins().brif(cmp_val, loop_body, empty_args, loop_exit, empty_args);

                builder.switch_to_block(loop_body);
                builder.seal_block(loop_body);

                let elem_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let elem_ptr = builder.ins().stack_addr(ptr_ty, elem_slot, 0);
                builder.ins().call(rt_funcs.rt_iter_get, &[elem_ptr, arr_ptr, idx]);

                let call_inst = builder.ins().call(rt_funcs.rt_is_truthy, &[elem_ptr]);
                let is_truthy = builder.inst_results(call_inst)[0];

                let then_block = builder.create_block();
                let else_block = builder.create_block();
                let iter_merge = builder.create_block();

                builder.ins().brif(is_truthy, then_block, empty_args, else_block, empty_args);

                builder.switch_to_block(then_block);
                builder.seal_block(then_block);
                codegen_generator(
                    then_branch, input_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );
                builder.ins().jump(iter_merge, &[]);

                builder.switch_to_block(else_block);
                builder.seal_block(else_block);
                codegen_generator(
                    else_branch, input_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );
                builder.ins().jump(iter_merge, &[]);

                builder.switch_to_block(iter_merge);
                builder.seal_block(iter_merge);

                let one = builder.ins().iconst(types::I64, 1);
                let next_idx = builder.ins().iadd(idx, one);
                let next_arg = BlockArg::Value(next_idx);
                builder.ins().jump(loop_header, &[next_arg]);

                builder.switch_to_block(loop_exit);
                builder.seal_block(loop_exit);
                builder.seal_block(loop_header);
                return;
            }

            // Evaluate condition, then dispatch to the appropriate branch
            // as a generator (so branches can contain Comma/Empty).
            let (cond_ptr, _) = codegen_expr(cond, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            let call_inst = builder.ins().call(rt_funcs.rt_is_truthy, &[cond_ptr]);
            let is_truthy = builder.inst_results(call_inst)[0];

            let then_block = builder.create_block();
            let else_block = builder.create_block();
            let merge_block = builder.create_block();

            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(is_truthy, then_block, empty_args, else_block, empty_args);

            // Then branch (as generator)
            builder.switch_to_block(then_block);
            builder.seal_block(then_block);
            codegen_generator(
                then_branch, input_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                builder, ptr_ty, rt_funcs, literals, var_slots,
            );
            builder.ins().jump(merge_block, &[]);

            // Else branch (as generator)
            builder.switch_to_block(else_block);
            builder.seal_block(else_block);
            codegen_generator(
                else_branch, input_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                builder, ptr_ty, rt_funcs, literals, var_slots,
            );
            builder.ins().jump(merge_block, &[]);

            // Merge
            builder.switch_to_block(merge_block);
            builder.seal_block(merge_block);
        }

        Expr::TryCatch {
            try_expr,
            catch_expr,
        } => {
            // Phase 10-5: Try-catch with proper generator semantics.
            //
            // Three cases:
            //
            // (A) try_expr is a 1→1 expression: evaluate, check error, dispatch.
            //     This is the simple case — no generator involved.
            //
            // (B) try_expr is a generator: use TryCallbackCtx wrapper approach.
            //     The wrapper intercepts each output from the generator:
            //     - Non-error outputs before any error → forwarded to real callback
            //     - First error → sets flag, saves error value, stops forwarding
            //     - All outputs after the first error → silently discarded
            //     After the generator completes, check the flag:
            //     - Flag set → run catch_expr with error value as input
            //     - Flag not set → nothing more to do

            if is_generator(try_expr) {
                // Case B: try_expr is a generator — use TryCallbackCtx wrapper.
                //
                // TryCallbackCtx layout (40 bytes):
                //   [0..8]   original_callback: fn ptr
                //   [8..16]  original_ctx: *mut u8
                //   [16..20] error_flag: i32
                //   [20..24] _padding: i32
                //   [24..40] error_value: Value (16 bytes)

                // Allocate TryCallbackCtx on the stack (40 bytes, 8-byte aligned)
                let ctx_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 40, 8,
                ));
                let ctx_addr = builder.ins().stack_addr(ptr_ty, ctx_slot, 0);

                // Store original_callback at offset 0
                builder.ins().store(MemFlags::trusted(), callback_ptr, ctx_addr, 0);
                // Store original_ctx at offset 8
                builder.ins().store(MemFlags::trusted(), ctx_ptr, ctx_addr, 8);
                // Store error_flag = 0 at offset 16
                let zero_i32 = builder.ins().iconst(types::I32, 0);
                builder.ins().store(MemFlags::trusted(), zero_i32, ctx_addr, 16);
                // Store _padding = 0 at offset 20
                builder.ins().store(MemFlags::trusted(), zero_i32, ctx_addr, 20);

                // Get the function pointer for rt_try_callback_wrapper
                let wrapper_fn_ptr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_try_callback_wrapper);

                // Run try_expr as a generator with wrapper callback + ctx
                codegen_generator(
                    try_expr, input_ptr, wrapper_fn_ptr, ctx_addr, callback_sig_ref,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );

                // After generator completes, check error_flag
                let flag_val = builder.ins().load(types::I32, MemFlags::trusted(), ctx_addr, 16);
                let catch_block = builder.create_block();
                let merge_block = builder.create_block();

                let empty_args: &[BlockArg] = &[];
                builder.ins().brif(flag_val, catch_block, empty_args, merge_block, empty_args);

                // Catch block: extract error message and run catch_expr
                builder.switch_to_block(catch_block);
                builder.seal_block(catch_block);
                let error_value_ptr = builder.ins().iadd_imm(ctx_addr, 24);
                let extracted_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let extracted_ptr = builder.ins().stack_addr(ptr_ty, extracted_slot, 0);
                builder.ins().call(rt_funcs.rt_extract_error, &[extracted_ptr, error_value_ptr]);
                codegen_generator(
                    catch_expr, extracted_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );
                builder.ins().jump(merge_block, &[]);

                // Merge
                builder.switch_to_block(merge_block);
                builder.seal_block(merge_block);
            } else {
                // Case A: try_expr is a 1→1 expression (original logic)
                let (try_result_ptr, _) = codegen_expr(try_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

                let call_inst = builder.ins().call(rt_funcs.rt_is_error, &[try_result_ptr]);
                let is_error = builder.inst_results(call_inst)[0];

                let catch_block = builder.create_block();
                let success_block = builder.create_block();
                let merge_block = builder.create_block();

                let empty_args: &[BlockArg] = &[];
                builder.ins().brif(is_error, catch_block, empty_args, success_block, empty_args);

                // Success: deliver try result via callback
                builder.switch_to_block(success_block);
                builder.seal_block(success_block);
                builder.ins().call_indirect(callback_sig_ref, callback_ptr, &[try_result_ptr, ctx_ptr]);
                builder.ins().jump(merge_block, &[]);

                // Catch: extract error message and run catch_expr as generator
                builder.switch_to_block(catch_block);
                builder.seal_block(catch_block);
                // Extract error value: Value::Error(json) → original value
                let extracted_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let extracted_ptr = builder.ins().stack_addr(ptr_ty, extracted_slot, 0);
                builder.ins().call(rt_funcs.rt_extract_error, &[extracted_ptr, try_result_ptr]);
                codegen_generator(
                    catch_expr, extracted_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );
                builder.ins().jump(merge_block, &[]);

                // Merge
                builder.switch_to_block(merge_block);
                builder.seal_block(merge_block);
            }
        }

        Expr::Each { input_expr, body } => {
            // .[] iteration: for each element in the container, generate the
            // body's outputs (body may itself be a generator).
            //
            // Phase 11: Added iterable check — non-iterable inputs produce an Error
            // value (delivered via callback), matching jq's error semantics.
            //
            // 1. Evaluate input_expr to get the container pointer
            // 1b. Check if iterable → if not, output Error and skip
            // 2. Call rt_iter_prepare to convert objects to arrays (O(1) indexed access)
            // 3. Call rt_iter_length to get the number of elements
            // 4. Loop: for idx = 0 .. len:
            //    a. Call rt_iter_get(out, container, idx) to get the element
            //    b. Run body as a generator with element as input
            // 5. Exit loop

            // Step 1: Evaluate the container expression
            let (raw_container_ptr, _) = codegen_expr(
                input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots,
            );

            // Step 1b: Check if iterable. If not, output an error and skip iteration.
            let check_inst = builder.ins().call(rt_funcs.rt_is_iterable, &[raw_container_ptr]);
            let is_iterable = builder.inst_results(check_inst)[0];

            let iter_block = builder.create_block();
            let error_block = builder.create_block();
            let final_exit = builder.create_block();
            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(is_iterable, iter_block, empty_args, error_block, empty_args);

            // Error block: generate "Cannot iterate over {type}" error and output it
            builder.switch_to_block(error_block);
            builder.seal_block(error_block);
            let err_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let err_ptr = builder.ins().stack_addr(ptr_ty, err_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_error, &[err_ptr, raw_container_ptr]);
            builder.ins().call_indirect(callback_sig_ref, callback_ptr, &[err_ptr, ctx_ptr]);
            builder.ins().jump(final_exit, &[]);

            // Iteration block: normal iteration
            builder.switch_to_block(iter_block);
            builder.seal_block(iter_block);

            // Step 2: Prepare container for O(1) iteration (Phase 6-2 optimization)
            let each_prep_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let container_ptr = builder.ins().stack_addr(ptr_ty, each_prep_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_prepare, &[container_ptr, raw_container_ptr]);

            // Step 3: Get the length
            let call_inst = builder.ins().call(rt_funcs.rt_iter_length, &[container_ptr]);
            let len = builder.inst_results(call_inst)[0]; // i64

            // Step 3: Create loop blocks
            let loop_header = builder.create_block();
            let loop_body = builder.create_block();
            let loop_exit = builder.create_block();

            // loop_header receives the index as a block parameter
            builder.append_block_param(loop_header, types::I64);

            // Jump to loop header with initial index = 0
            let zero = builder.ins().iconst(types::I64, 0);
            let zero_arg = BlockArg::Value(zero);
            builder.ins().jump(loop_header, &[zero_arg]);

            // Loop header: check idx < len
            builder.switch_to_block(loop_header);
            // Don't seal yet — loop_body jumps back
            let idx = builder.block_params(loop_header)[0]; // i64 index
            let cmp = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, idx, len);
            builder.ins().brif(cmp, loop_body, empty_args, loop_exit, empty_args);

            // Loop body: get element, run body generator
            builder.switch_to_block(loop_body);
            builder.seal_block(loop_body); // single predecessor: loop_header

            // Allocate stack slot for the element (16 bytes = sizeof(Value))
            let elem_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                16,
                8,
            ));
            let elem_ptr = builder.ins().stack_addr(ptr_ty, elem_slot, 0);

            // Call rt_iter_get(elem_ptr, container_ptr, idx)
            builder.ins().call(rt_funcs.rt_iter_get, &[elem_ptr, container_ptr, idx]);

            // Run body as generator with elem_ptr as input
            codegen_generator(
                body, elem_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                builder, ptr_ty, rt_funcs, literals, var_slots,
            );

            // Increment index and jump back to header
            let one = builder.ins().iconst(types::I64, 1);
            let next_idx = builder.ins().iadd(idx, one);
            let next_arg = BlockArg::Value(next_idx);
            builder.ins().jump(loop_header, &[next_arg]);

            // Loop exit → jump to final_exit
            builder.switch_to_block(loop_exit);
            builder.seal_block(loop_exit); // single predecessor: loop_header
            builder.ins().jump(final_exit, &[]);

            // Seal loop_header now (predecessors: iter_block, loop_body)
            builder.seal_block(loop_header);

            // Final exit: merge from error_block and loop_exit
            builder.switch_to_block(final_exit);
            builder.seal_block(final_exit);
        }

        Expr::EachOpt { input_expr, body } => {
            // Phase 8-3: .[]? optional iteration.
            // Same as Each but silently produces 0 outputs for non-iterable inputs.
            //
            // 1. Evaluate input_expr
            // 2. Check rt_is_iterable
            //    - If not iterable → skip (produce 0 outputs)
            //    - If iterable → same iteration loop as Each

            let (raw_container_ptr, _) = codegen_expr(
                input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots,
            );

            // Check if iterable
            let check_inst = builder.ins().call(rt_funcs.rt_is_iterable, &[raw_container_ptr]);
            let is_iterable = builder.inst_results(check_inst)[0];

            let iter_block = builder.create_block();
            let skip_block = builder.create_block();
            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(is_iterable, iter_block, empty_args, skip_block, empty_args);

            // iter_block: do the iteration (copy of Each logic)
            builder.switch_to_block(iter_block);
            builder.seal_block(iter_block);

            let each_prep_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let container_ptr = builder.ins().stack_addr(ptr_ty, each_prep_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_prepare, &[container_ptr, raw_container_ptr]);

            let call_inst = builder.ins().call(rt_funcs.rt_iter_length, &[container_ptr]);
            let len = builder.inst_results(call_inst)[0];

            let loop_header = builder.create_block();
            let loop_body = builder.create_block();
            let loop_exit = builder.create_block();

            builder.append_block_param(loop_header, types::I64);

            let zero = builder.ins().iconst(types::I64, 0);
            let zero_arg = BlockArg::Value(zero);
            builder.ins().jump(loop_header, &[zero_arg]);

            builder.switch_to_block(loop_header);
            let idx = builder.block_params(loop_header)[0];
            let cmp = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, idx, len);
            builder.ins().brif(cmp, loop_body, empty_args, loop_exit, empty_args);

            builder.switch_to_block(loop_body);
            builder.seal_block(loop_body);

            let elem_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let elem_ptr = builder.ins().stack_addr(ptr_ty, elem_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_get, &[elem_ptr, container_ptr, idx]);

            codegen_generator(
                body, elem_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                builder, ptr_ty, rt_funcs, literals, var_slots,
            );

            let one = builder.ins().iconst(types::I64, 1);
            let next_idx = builder.ins().iadd(idx, one);
            let next_arg = BlockArg::Value(next_idx);
            builder.ins().jump(loop_header, &[next_arg]);

            builder.switch_to_block(loop_exit);
            builder.seal_block(loop_exit);
            builder.seal_block(loop_header);

            // Jump to skip_block (merge point)
            builder.ins().jump(skip_block, empty_args);

            // skip_block: non-iterable → produce 0 outputs (do nothing)
            builder.switch_to_block(skip_block);
            builder.seal_block(skip_block);
        }

        Expr::LetBinding { var_index, value, body } => {
            // Phase 4-4: Variable binding in generator context.
            //
            // The body may contain generators (Comma, Each, etc.), so we
            // must process it as a generator rather than a scalar expression.
            //
            // If value is a generator (e.g., range(n)), we collect its outputs
            // into a temporary array, then iterate over each element, binding
            // it to $var and running body as a generator for each.

            if is_generator(value) {
                // Value is a generator: collect outputs, then iterate
                let gen_collect_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let gen_collect_ptr = builder.ins().stack_addr(ptr_ty, gen_collect_slot, 0);
                builder.ins().call(rt_funcs.rt_collect_init, &[gen_collect_ptr]);

                let collect_fn_addr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_collect_append);
                let cb_sig = builder.func.import_signature(cranelift_codegen::ir::Signature {
                    params: vec![AbiParam::new(ptr_ty), AbiParam::new(ptr_ty)],
                    returns: vec![],
                    call_conv: cranelift_codegen::isa::CallConv::SystemV,
                });
                codegen_generator(
                    value, input_ptr, collect_fn_addr, gen_collect_ptr, cb_sig,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );

                // Prepare for iteration
                let prep_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let arr_ptr = builder.ins().stack_addr(ptr_ty, prep_slot, 0);
                builder.ins().call(rt_funcs.rt_iter_prepare, &[arr_ptr, gen_collect_ptr]);

                let len_call = builder.ins().call(rt_funcs.rt_iter_length, &[arr_ptr]);
                let len = builder.inst_results(len_call)[0];

                let loop_header = builder.create_block();
                let loop_body = builder.create_block();
                let loop_exit = builder.create_block();

                builder.append_block_param(loop_header, types::I64);
                let zero = builder.ins().iconst(types::I64, 0);
                let zero_arg = BlockArg::Value(zero);
                builder.ins().jump(loop_header, &[zero_arg]);

                builder.switch_to_block(loop_header);
                let idx = builder.block_params(loop_header)[0];
                let cmp = builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, idx, len,
                );
                let empty_args: &[BlockArg] = &[];
                builder.ins().brif(cmp, loop_body, empty_args, loop_exit, empty_args);

                builder.switch_to_block(loop_body);
                builder.seal_block(loop_body);

                // Get element
                let elem_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let elem_ptr = builder.ins().stack_addr(ptr_ty, elem_slot, 0);
                builder.ins().call(rt_funcs.rt_iter_get, &[elem_ptr, arr_ptr, idx]);

                // Copy element to var slot
                let var_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let var_addr = builder.ins().stack_addr(ptr_ty, var_slot, 0);
                let el_lo = builder.ins().load(types::I64, MemFlags::trusted(), elem_ptr, 0);
                let el_hi = builder.ins().load(types::I64, MemFlags::trusted(), elem_ptr, 8);
                builder.ins().store(MemFlags::trusted(), el_lo, var_addr, 0);
                builder.ins().store(MemFlags::trusted(), el_hi, var_addr, 8);

                let prev_slot = var_slots.insert(*var_index, var_slot);

                // Process body as a generator for each element
                codegen_generator(
                    body, input_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );

                if let Some(prev) = prev_slot {
                    var_slots.insert(*var_index, prev);
                } else {
                    var_slots.remove(var_index);
                }

                // Increment index
                let one = builder.ins().iconst(types::I64, 1);
                let next_idx = builder.ins().iadd(idx, one);
                let next_arg = BlockArg::Value(next_idx);
                builder.ins().jump(loop_header, &[next_arg]);

                builder.switch_to_block(loop_exit);
                builder.seal_block(loop_exit);
                builder.seal_block(loop_header);
            } else {
                // Value is scalar: evaluate once and bind
                let (value_ptr, _) = codegen_expr(value, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

                let var_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    16,
                    8,
                ));
                let var_addr = builder.ins().stack_addr(ptr_ty, var_slot, 0);

                let val_lo = builder.ins().load(types::I64, MemFlags::trusted(), value_ptr, 0);
                let val_hi = builder.ins().load(types::I64, MemFlags::trusted(), value_ptr, 8);
                builder.ins().store(MemFlags::trusted(), val_lo, var_addr, 0);
                builder.ins().store(MemFlags::trusted(), val_hi, var_addr, 8);

                let prev_slot = var_slots.insert(*var_index, var_slot);

                codegen_generator(
                    body, input_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );

                if let Some(prev) = prev_slot {
                    var_slots.insert(*var_index, prev);
                } else {
                    var_slots.remove(var_index);
                }
            }
        }

        Expr::Foreach { source, init, var_index, acc_index, update } => {
            // Phase 4-5: Foreach (1→N generator).
            //
            // Same loop as Reduce, but yields each intermediate accumulator
            // value via callback instead of only the final value.
            //
            // 1. Evaluate init to get the initial accumulator value
            // 2. Store it in acc_slot
            // 3. Loop over source elements:
            //    a. Store element in var_slot ($x)
            //    b. Load acc from acc_slot as input
            //    c. Evaluate update with acc as input
            //    d. Store result back in acc_slot
            //    e. Call callback with the new acc value (yield)
            // 4. Done (no final output)

            // Step 1: Evaluate init
            let (init_ptr, _) = codegen_expr(init, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Step 2: Allocate acc_slot and copy init value
            let acc_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let acc_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            let init_lo = builder.ins().load(types::I64, MemFlags::trusted(), init_ptr, 0);
            let init_hi = builder.ins().load(types::I64, MemFlags::trusted(), init_ptr, 8);
            builder.ins().store(MemFlags::trusted(), init_lo, acc_addr, 0);
            builder.ins().store(MemFlags::trusted(), init_hi, acc_addr, 8);

            // Register acc_slot
            let prev_acc_slot = var_slots.insert(*acc_index, acc_slot);

            // Step 3: Evaluate source container.
            // If the source is a generator (e.g., range(n)), collect its outputs
            // into a temporary array first, then iterate over that.
            let raw_container_ptr = if is_generator(source) {
                let gen_collect_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let gen_collect_ptr = builder.ins().stack_addr(ptr_ty, gen_collect_slot, 0);
                builder.ins().call(rt_funcs.rt_collect_init, &[gen_collect_ptr]);
                let collect_fn_addr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_collect_append);
                let callback_sig = builder.func.import_signature(cranelift_codegen::ir::Signature {
                    params: vec![AbiParam::new(ptr_ty), AbiParam::new(ptr_ty)],
                    returns: vec![],
                    call_conv: cranelift_codegen::isa::CallConv::SystemV,
                });
                codegen_generator(
                    source, input_ptr, collect_fn_addr, gen_collect_ptr, callback_sig,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );
                gen_collect_ptr
            } else {
                let (ptr, _) = codegen_expr(source, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
                ptr
            };

            // Phase 6-2: Prepare container for O(1) iteration (converts objects to arrays)
            let foreach_prep_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let container_ptr = builder.ins().stack_addr(ptr_ty, foreach_prep_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_prepare, &[container_ptr, raw_container_ptr]);

            // Get length
            let len_call = builder.ins().call(rt_funcs.rt_iter_length, &[container_ptr]);
            let len = builder.inst_results(len_call)[0]; // i64

            // Create loop blocks
            let loop_header = builder.create_block();
            let loop_body = builder.create_block();
            let loop_exit = builder.create_block();

            builder.append_block_param(loop_header, types::I64);

            let zero = builder.ins().iconst(types::I64, 0);
            let zero_arg = BlockArg::Value(zero);
            builder.ins().jump(loop_header, &[zero_arg]);

            // Loop header: check idx < len
            builder.switch_to_block(loop_header);
            let idx = builder.block_params(loop_header)[0];
            let cmp = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, idx, len,
            );
            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(cmp, loop_body, empty_args, loop_exit, empty_args);

            // Loop body
            builder.switch_to_block(loop_body);
            builder.seal_block(loop_body);

            // Get element
            let elem_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let elem_ptr = builder.ins().stack_addr(ptr_ty, elem_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_get, &[elem_ptr, container_ptr, idx]);

            // Store element in var_slot ($x)
            let var_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let var_addr = builder.ins().stack_addr(ptr_ty, var_slot, 0);
            let elem_lo = builder.ins().load(types::I64, MemFlags::trusted(), elem_ptr, 0);
            let elem_hi = builder.ins().load(types::I64, MemFlags::trusted(), elem_ptr, 8);
            builder.ins().store(MemFlags::trusted(), elem_lo, var_addr, 0);
            builder.ins().store(MemFlags::trusted(), elem_hi, var_addr, 8);

            let prev_var_slot = var_slots.insert(*var_index, var_slot);

            // Evaluate update with acc as input
            let current_acc_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            let (update_result, _) = codegen_expr(update, current_acc_addr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Store update result back in acc_slot
            let new_acc_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            let res_lo = builder.ins().load(types::I64, MemFlags::trusted(), update_result, 0);
            let res_hi = builder.ins().load(types::I64, MemFlags::trusted(), update_result, 8);
            builder.ins().store(MemFlags::trusted(), res_lo, new_acc_addr, 0);
            builder.ins().store(MemFlags::trusted(), res_hi, new_acc_addr, 8);

            // Yield the new accumulator value via callback
            let yield_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            builder.ins().call_indirect(callback_sig_ref, callback_ptr, &[yield_addr, ctx_ptr]);

            // Restore var_slot
            if let Some(prev) = prev_var_slot {
                var_slots.insert(*var_index, prev);
            } else {
                var_slots.remove(var_index);
            }

            // Increment index and jump back
            let one = builder.ins().iconst(types::I64, 1);
            let next_idx = builder.ins().iadd(idx, one);
            let next_arg = BlockArg::Value(next_idx);
            builder.ins().jump(loop_header, &[next_arg]);

            // Loop exit
            builder.switch_to_block(loop_exit);
            builder.seal_block(loop_exit);
            builder.seal_block(loop_header);

            // Restore acc_slot
            if let Some(prev) = prev_acc_slot {
                var_slots.insert(*acc_index, prev);
            } else {
                var_slots.remove(acc_index);
            }
        }

        Expr::Range { from, to, step } => {
            // Phase 9-2: Range generator.
            //
            // Generates numeric values from `from` up to (but not including) `to`,
            // incrementing by `step` (default 1).
            //
            // If from, to, or step are generators (e.g., range(0,1;3,4)),
            // we collect each generator's outputs and iterate over all combinations
            // (Cartesian product), calling rt_range/rt_range_step for each.

            let has_gen_args = is_generator(from) || is_generator(to)
                || step.as_ref().map_or(false, |s| is_generator(s));

            if !has_gen_args {
                // Simple case: no generator arguments
                let (from_ptr, _) = codegen_expr(
                    from, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots,
                );
                let (to_ptr, _) = codegen_expr(
                    to, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots,
                );
                if let Some(step_expr) = step {
                    let (step_ptr, _) = codegen_expr(
                        step_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots,
                    );
                    builder.ins().call(rt_funcs.rt_range_step, &[from_ptr, to_ptr, step_ptr, callback_ptr, ctx_ptr]);
                } else {
                    builder.ins().call(rt_funcs.rt_range, &[from_ptr, to_ptr, callback_ptr, ctx_ptr]);
                }
            } else {
                // Generator arguments: collect each, iterate Cartesian product.
                // Helper: collect generator outputs or wrap scalar in single-element array
                let collect_arg = |expr: &Expr,
                    builder: &mut FunctionBuilder,
                    rt_funcs: &RuntimeFuncRefs,
                    literals: &mut LiteralPool,
                    var_slots: &mut HashMap<u16, cranelift_codegen::ir::StackSlot>|
                -> cranelift_codegen::ir::Value {
                    if is_generator(expr) {
                        let slot = builder.create_sized_stack_slot(StackSlotData::new(
                            StackSlotKind::ExplicitSlot, 16, 8,
                        ));
                        let ptr = builder.ins().stack_addr(ptr_ty, slot, 0);
                        builder.ins().call(rt_funcs.rt_collect_init, &[ptr]);
                        let fn_addr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_collect_append);
                        let sig = builder.func.import_signature(cranelift_codegen::ir::Signature {
                            params: vec![AbiParam::new(ptr_ty), AbiParam::new(ptr_ty)],
                            returns: vec![],
                            call_conv: cranelift_codegen::isa::CallConv::SystemV,
                        });
                        codegen_generator(
                            expr, input_ptr, fn_addr, ptr, sig,
                            builder, ptr_ty, rt_funcs, literals, var_slots,
                        );
                        ptr
                    } else {
                        // Wrap scalar in single-element array
                        let slot = builder.create_sized_stack_slot(StackSlotData::new(
                            StackSlotKind::ExplicitSlot, 16, 8,
                        ));
                        let ptr = builder.ins().stack_addr(ptr_ty, slot, 0);
                        builder.ins().call(rt_funcs.rt_collect_init, &[ptr]);
                        let (val, _) = codegen_expr(expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
                        builder.ins().call(rt_funcs.rt_collect_append, &[val, ptr]);
                        ptr
                    }
                };

                let from_arr = collect_arg(from, builder, rt_funcs, literals, var_slots);
                let to_arr = collect_arg(to, builder, rt_funcs, literals, var_slots);
                let step_arr = step.as_ref().map(|s| collect_arg(s, builder, rt_funcs, literals, var_slots));

                // Prepare arrays for iteration
                let from_prep = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 8));
                let from_prep_ptr = builder.ins().stack_addr(ptr_ty, from_prep, 0);
                builder.ins().call(rt_funcs.rt_iter_prepare, &[from_prep_ptr, from_arr]);
                let from_len_call = builder.ins().call(rt_funcs.rt_iter_length, &[from_prep_ptr]);
                let from_len = builder.inst_results(from_len_call)[0];

                let to_prep = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 8));
                let to_prep_ptr = builder.ins().stack_addr(ptr_ty, to_prep, 0);
                builder.ins().call(rt_funcs.rt_iter_prepare, &[to_prep_ptr, to_arr]);
                let to_len_call = builder.ins().call(rt_funcs.rt_iter_length, &[to_prep_ptr]);
                let to_len = builder.inst_results(to_len_call)[0];

                // Helper to create iteration loop
                macro_rules! make_loop {
                    ($arr:expr, $len:expr, $builder:expr, $body:expr) => {{
                        let header = $builder.create_block();
                        let body_blk = $builder.create_block();
                        let exit = $builder.create_block();
                        $builder.append_block_param(header, types::I64);
                        let zero = $builder.ins().iconst(types::I64, 0);
                        let z_arg = BlockArg::Value(zero);
                        $builder.ins().jump(header, &[z_arg]);
                        $builder.switch_to_block(header);
                        let idx = $builder.block_params(header)[0];
                        let cmp = $builder.ins().icmp(
                            cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, idx, $len,
                        );
                        let empty: &[BlockArg] = &[];
                        $builder.ins().brif(cmp, body_blk, empty, exit, empty);
                        $builder.switch_to_block(body_blk);
                        $builder.seal_block(body_blk);
                        let elem = $builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 8));
                        let elem_p = $builder.ins().stack_addr(ptr_ty, elem, 0);
                        $builder.ins().call(rt_funcs.rt_iter_get, &[elem_p, $arr, idx]);
                        #[allow(clippy::redundant_closure_call)]
                        $body(elem_p, $builder);
                        let one = $builder.ins().iconst(types::I64, 1);
                        let next = $builder.ins().iadd(idx, one);
                        let n_arg = BlockArg::Value(next);
                        $builder.ins().jump(header, &[n_arg]);
                        $builder.switch_to_block(exit);
                        $builder.seal_block(exit);
                        $builder.seal_block(header);
                    }};
                }

                if let Some(step_arr_val) = step_arr {
                    let step_prep = builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 8));
                    let step_prep_ptr = builder.ins().stack_addr(ptr_ty, step_prep, 0);
                    builder.ins().call(rt_funcs.rt_iter_prepare, &[step_prep_ptr, step_arr_val]);
                    let step_len_call = builder.ins().call(rt_funcs.rt_iter_length, &[step_prep_ptr]);
                    let step_len = builder.inst_results(step_len_call)[0];

                    // Triple nested loop: from × to × step
                    make_loop!(from_prep_ptr, from_len, builder, |f_ptr: cranelift_codegen::ir::Value, builder: &mut FunctionBuilder| {
                        make_loop!(to_prep_ptr, to_len, builder, |t_ptr: cranelift_codegen::ir::Value, builder: &mut FunctionBuilder| {
                            make_loop!(step_prep_ptr, step_len, builder, |s_ptr: cranelift_codegen::ir::Value, builder: &mut FunctionBuilder| {
                                builder.ins().call(rt_funcs.rt_range_step, &[f_ptr, t_ptr, s_ptr, callback_ptr, ctx_ptr]);
                            });
                        });
                    });
                } else {
                    // Double nested loop: from × to
                    make_loop!(from_prep_ptr, from_len, builder, |f_ptr: cranelift_codegen::ir::Value, builder: &mut FunctionBuilder| {
                        make_loop!(to_prep_ptr, to_len, builder, |t_ptr: cranelift_codegen::ir::Value, builder: &mut FunctionBuilder| {
                            builder.ins().call(rt_funcs.rt_range, &[f_ptr, t_ptr, callback_ptr, ctx_ptr]);
                        });
                    });
                }
            }
        }

        Expr::While { input_expr, cond, update } => {
            // Phase 11: While generator.
            //
            // Starting with input_expr, while cond is truthy:
            //   1. Yield the current value via callback
            //   2. Apply update to get next value
            // Stop when cond is falsy (don't yield that value).

            // Evaluate input_expr to get initial value
            let (init_ptr, _) = codegen_expr(input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Allocate accumulator slot
            let acc_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let acc_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);

            // Copy initial value to accumulator
            let in_lo = builder.ins().load(types::I64, MemFlags::trusted(), init_ptr, 0);
            let in_hi = builder.ins().load(types::I64, MemFlags::trusted(), init_ptr, 8);
            builder.ins().store(MemFlags::trusted(), in_lo, acc_addr, 0);
            builder.ins().store(MemFlags::trusted(), in_hi, acc_addr, 8);

            // Create loop blocks
            let loop_header = builder.create_block();
            let loop_body = builder.create_block();
            let loop_exit = builder.create_block();

            builder.ins().jump(loop_header, &[]);

            // Loop header: evaluate cond with acc as input
            builder.switch_to_block(loop_header);

            let current_acc = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            let (cond_result, _) = codegen_expr(cond, current_acc, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Check truthiness
            let is_truthy_call = builder.ins().call(rt_funcs.rt_is_truthy, &[cond_result]);
            let is_truthy = builder.inst_results(is_truthy_call)[0]; // i32

            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(is_truthy, loop_body, empty_args, loop_exit, empty_args);

            // Loop body: yield current value, then update
            builder.switch_to_block(loop_body);
            builder.seal_block(loop_body);

            // Yield current accumulator value
            let yield_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            builder.ins().call_indirect(callback_sig_ref, callback_ptr, &[yield_addr, ctx_ptr]);

            // Apply update to get next value
            let body_acc = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            let (update_result, _) = codegen_expr(update, body_acc, builder, ptr_ty, rt_funcs, literals, var_slots);

            // Store update result back into accumulator
            let new_acc_addr = builder.ins().stack_addr(ptr_ty, acc_slot, 0);
            let res_lo = builder.ins().load(types::I64, MemFlags::trusted(), update_result, 0);
            let res_hi = builder.ins().load(types::I64, MemFlags::trusted(), update_result, 8);
            builder.ins().store(MemFlags::trusted(), res_lo, new_acc_addr, 0);
            builder.ins().store(MemFlags::trusted(), res_hi, new_acc_addr, 8);

            builder.ins().jump(loop_header, &[]);

            // Loop exit
            builder.switch_to_block(loop_exit);
            builder.seal_block(loop_exit);
            builder.seal_block(loop_header);
        }

        Expr::Recurse { input_expr, body } => {
            // Phase 9-4: Recursive descent (..).
            //
            // Evaluate input_expr to get the value, then call rt_recurse which
            // recursively yields the value itself and all sub-values via callback.
            //
            // When body is Input (identity), we pass the external callback directly
            // to rt_recurse for maximum efficiency.
            // When body is not Input (e.g., `.. | numbers`), we generate an inner
            // callback that applies body to each recursed value, then calls the
            // outer callback for matching results.
            let (value_ptr, _) = codegen_expr(
                input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots,
            );

            if matches!(body.as_ref(), Expr::Input) {
                // Simple case: .. with no body transformation
                builder.ins().call(rt_funcs.rt_recurse, &[value_ptr, callback_ptr, ctx_ptr]);
            } else {
                // Complex case: .. | body
                // rt_recurse calls callback for each element.
                // We need a trampoline callback that applies body to each element.
                // Since Cranelift can't easily create closures, we call rt_recurse
                // with the external callback, then for each yielded element, apply body.
                //
                // Actually, a simpler approach: use the same callback but
                // codegen body as the generator applied to each recursed value.
                // We can do this by using rt_recurse with a wrapper.
                //
                // Simplest approach for now: call rt_recurse with external callback/ctx,
                // but wrap it so body is applied. Since we can't create a closure in CLIF,
                // we use a different strategy: collect all recursed values, then iterate
                // and apply body.
                //
                // But that's wasteful. Better approach: don't use rt_recurse at all for
                // the body case. Instead, use rt_recurse to collect into an array, then
                // iterate with Each-like loop applying body.
                //
                // Actually the simplest: just collect rt_recurse outputs into an internal
                // array, then iterate.

                // Step 1: Collect recursed values into a temporary array
                let collect_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let collect_ptr = builder.ins().stack_addr(ptr_ty, collect_slot, 0);
                builder.ins().call(rt_funcs.rt_collect_init, &[collect_ptr]);

                // Use rt_collect_append as the callback for rt_recurse
                // rt_collect_append signature: fn(value: *const Value, acc: *mut Value)
                // This matches the callback signature: fn(value_ptr, ctx_ptr)
                let collect_fn_addr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_collect_append);
                builder.ins().call(rt_funcs.rt_recurse, &[value_ptr, collect_fn_addr, collect_ptr]);

                // Step 2: Iterate over collected array, applying body to each element
                let prep_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let prepared_ptr = builder.ins().stack_addr(ptr_ty, prep_slot, 0);
                builder.ins().call(rt_funcs.rt_iter_prepare, &[prepared_ptr, collect_ptr]);

                let len_call = builder.ins().call(rt_funcs.rt_iter_length, &[prepared_ptr]);
                let len = builder.inst_results(len_call)[0];

                let loop_header = builder.create_block();
                let loop_body_block = builder.create_block();
                let loop_exit = builder.create_block();

                builder.append_block_param(loop_header, types::I64);
                let zero = builder.ins().iconst(types::I64, 0);
                let zero_arg = BlockArg::Value(zero);
                builder.ins().jump(loop_header, &[zero_arg]);

                builder.switch_to_block(loop_header);
                let idx = builder.block_params(loop_header)[0];
                let cmp = builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, idx, len,
                );
                let empty_args: &[BlockArg] = &[];
                builder.ins().brif(cmp, loop_body_block, empty_args, loop_exit, empty_args);

                builder.switch_to_block(loop_body_block);
                builder.seal_block(loop_body_block);

                let elem_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let elem_ptr = builder.ins().stack_addr(ptr_ty, elem_slot, 0);
                builder.ins().call(rt_funcs.rt_iter_get, &[elem_ptr, prepared_ptr, idx]);

                // Apply body as generator with elem_ptr as input
                codegen_generator(
                    body, elem_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );

                let one = builder.ins().iconst(types::I64, 1);
                let next_idx = builder.ins().iadd(idx, one);
                let next_arg = BlockArg::Value(next_idx);
                builder.ins().jump(loop_header, &[next_arg]);

                builder.switch_to_block(loop_exit);
                builder.seal_block(loop_exit);
                builder.seal_block(loop_header);
            }
        }

        Expr::Limit { count, generator } => {
            // Phase 12: limit(n; gen) — generator that yields at most n outputs.
            //
            // Uses LimitCallbackCtx wrapper: allocate context on stack, store
            // original callback/ctx and the count, then run generator with wrapper.
            //
            // If count < 0, emit an error: "limit doesn't support negative count"
            //
            // LimitCallbackCtx layout (24 bytes, 8-byte aligned):
            //   [0..8]   original_callback: fn ptr
            //   [8..16]  original_ctx: *mut u8
            //   [16..24] remaining: i64

            // Evaluate count expression to get the limit number
            let (count_ptr, _) = codegen_expr(count, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            // The count Value is a Num - extract the f64 payload directly.
            // Value::Num layout: tag=2 at offset 0, f64 at offset 8
            let count_f64 = builder.ins().load(types::F64, MemFlags::trusted(), count_ptr, 8);
            let count_i64 = builder.ins().fcvt_to_sint(types::I64, count_f64);

            // Check if count < 0 → emit error
            let zero_i64 = builder.ins().iconst(types::I64, 0);
            let is_negative = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, count_i64, zero_i64,
            );
            let error_block = builder.create_block();
            let normal_block = builder.create_block();
            let merge_block = builder.create_block();
            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(is_negative, error_block, empty_args, normal_block, empty_args);

            // Error block: emit error value via callback
            builder.switch_to_block(error_block);
            builder.seal_block(error_block);
            let err_msg = "limit doesn't support negative count";
            let err_val = Value::Error(std::rc::Rc::new(err_msg.to_string()));
            let err_idx = literals.values.len();
            literals.values.push(Box::new(err_val));
            let err_offset = (err_idx as i32) * (ptr_ty.bytes() as i32);
            let err_ptr = builder.ins().load(ptr_ty, MemFlags::trusted(), literals.clif_param, err_offset);
            builder.ins().call_indirect(callback_sig_ref, callback_ptr, &[err_ptr, ctx_ptr]);
            builder.ins().jump(merge_block, &[]);

            // Normal block: run generator with limit wrapper
            builder.switch_to_block(normal_block);
            builder.seal_block(normal_block);

            // Allocate LimitCallbackCtx on stack (24 bytes, 8-byte aligned)
            let limit_ctx_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 24, 8,
            ));
            let limit_ctx_addr = builder.ins().stack_addr(ptr_ty, limit_ctx_slot, 0);

            // Store original_callback at offset 0
            builder.ins().store(MemFlags::trusted(), callback_ptr, limit_ctx_addr, 0);
            // Store original_ctx at offset 8
            builder.ins().store(MemFlags::trusted(), ctx_ptr, limit_ctx_addr, 8);
            // Store remaining count at offset 16
            builder.ins().store(MemFlags::trusted(), count_i64, limit_ctx_addr, 16);

            // Get the function pointer for rt_limit_callback_wrapper
            let wrapper_fn_ptr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_limit_callback_wrapper);

            // Run generator with wrapper callback + limit ctx
            codegen_generator(
                generator, input_ptr, wrapper_fn_ptr, limit_ctx_addr, callback_sig_ref,
                builder, ptr_ty, rt_funcs, literals, var_slots,
            );
            builder.ins().jump(merge_block, &[]);

            builder.switch_to_block(merge_block);
            builder.seal_block(merge_block);
        }

        Expr::Skip { count, generator } => {
            // Phase 12: skip(n; gen) — generator that skips the first n outputs.
            //
            // If count < 0, emit an error: "skip doesn't support negative count"
            //
            // Uses SkipCallbackCtx wrapper: allocate context on stack, store
            // original callback/ctx and the skip count, then run generator with wrapper.
            //
            // SkipCallbackCtx layout (24 bytes, 8-byte aligned):
            //   [0..8]   original_callback: fn ptr
            //   [8..16]  original_ctx: *mut u8
            //   [16..24] remaining_skip: i64

            // Evaluate count expression to get the skip number
            let (count_ptr, _) = codegen_expr(count, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            // Extract f64 payload from Value::Num and convert to i64
            let count_f64 = builder.ins().load(types::F64, MemFlags::trusted(), count_ptr, 8);
            let count_i64 = builder.ins().fcvt_to_sint(types::I64, count_f64);

            // Check if count < 0 → emit error
            let zero_i64 = builder.ins().iconst(types::I64, 0);
            let is_negative = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, count_i64, zero_i64,
            );
            let error_block = builder.create_block();
            let normal_block = builder.create_block();
            let merge_block = builder.create_block();
            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(is_negative, error_block, empty_args, normal_block, empty_args);

            // Error block: emit error value via callback
            builder.switch_to_block(error_block);
            builder.seal_block(error_block);
            let err_msg = "skip doesn't support negative count";
            let err_val = Value::Error(std::rc::Rc::new(err_msg.to_string()));
            let err_idx = literals.values.len();
            literals.values.push(Box::new(err_val));
            let err_offset = (err_idx as i32) * (ptr_ty.bytes() as i32);
            let err_ptr = builder.ins().load(ptr_ty, MemFlags::trusted(), literals.clif_param, err_offset);
            builder.ins().call_indirect(callback_sig_ref, callback_ptr, &[err_ptr, ctx_ptr]);
            builder.ins().jump(merge_block, &[]);

            // Normal block: run generator with skip wrapper
            builder.switch_to_block(normal_block);
            builder.seal_block(normal_block);

            // Allocate SkipCallbackCtx on stack (24 bytes, 8-byte aligned)
            let skip_ctx_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 24, 8,
            ));
            let skip_ctx_addr = builder.ins().stack_addr(ptr_ty, skip_ctx_slot, 0);

            // Store original_callback at offset 0
            builder.ins().store(MemFlags::trusted(), callback_ptr, skip_ctx_addr, 0);
            // Store original_ctx at offset 8
            builder.ins().store(MemFlags::trusted(), ctx_ptr, skip_ctx_addr, 8);
            // Store remaining_skip count at offset 16
            builder.ins().store(MemFlags::trusted(), count_i64, skip_ctx_addr, 16);

            // Get the function pointer for rt_skip_callback_wrapper
            let wrapper_fn_ptr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_skip_callback_wrapper);

            // Run generator with wrapper callback + skip ctx
            codegen_generator(
                generator, input_ptr, wrapper_fn_ptr, skip_ctx_addr, callback_sig_ref,
                builder, ptr_ty, rt_funcs, literals, var_slots,
            );
            builder.ins().jump(merge_block, &[]);

            builder.switch_to_block(merge_block);
            builder.seal_block(merge_block);
        }

        Expr::PathOf { input_expr, path_expr } => {
            // Phase 9-5: path(expr) — generator yielding paths.
            //
            // 1. Create path descriptor from path_expr at compile time
            // 2. Evaluate input
            // 3. Call rt_path_of(input, descriptor, callback, ctx)
            let descriptor = analyze_path_descriptor(path_expr);
            let desc_idx = literals.values.len();
            literals.values.push(Box::new(descriptor));
            let desc_offset = (desc_idx as i32) * (ptr_ty.bytes() as i32);
            let desc_ptr = builder.ins().load(ptr_ty, MemFlags::trusted(), literals.clif_param, desc_offset);

            let (input_val_ptr, _) = codegen_expr(
                input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots,
            );

            builder.ins().call(rt_funcs.rt_path_of, &[input_val_ptr, desc_ptr, callback_ptr, ctx_ptr]);
        }

        Expr::RegexMatch { input_expr, re, flags } => {
            // Phase 10-2: match(re; flags) — generator (calls callback for each match).
            let (input_val_ptr, _) = codegen_expr(input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (re_ptr, _) = codegen_expr(re, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (flags_ptr, _) = codegen_expr(flags, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // rt_regex_match(input, re, flags, callback, ctx)
            builder.ins().call(rt_funcs.rt_regex_match, &[input_val_ptr, re_ptr, flags_ptr, callback_ptr, ctx_ptr]);
        }

        Expr::RegexScan { input_expr, re, flags } => {
            // Phase 10-2: scan(re; flags) — generator (calls callback for each match).
            let (input_val_ptr, _) = codegen_expr(input_expr, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (re_ptr, _) = codegen_expr(re, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let (flags_ptr, _) = codegen_expr(flags, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            // rt_regex_scan(input, re, flags, callback, ctx)
            builder.ins().call(rt_funcs.rt_regex_scan, &[input_val_ptr, re_ptr, flags_ptr, callback_ptr, ctx_ptr]);
        }

        // Alternative with generator primary: `.foo[] // .bar`
        // jq semantics: yield all truthy outputs from primary.
        // If NO truthy output exists, yield fallback instead.
        Expr::Alternative { primary, fallback } if is_generator(primary) => {
            // Use a runtime helper: rt_alt_filter_truthy
            // Collect primary's outputs, then filter.
            // Strategy: collect outputs, iterate twice.
            // Pass 1: check if any truthy element exists (store flag in stack slot).
            // Pass 2: if flag set, yield truthy elements; else yield fallback.

            let gen_collect_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let gen_collect_ptr = builder.ins().stack_addr(ptr_ty, gen_collect_slot, 0);
            builder.ins().call(rt_funcs.rt_collect_init, &[gen_collect_ptr]);

            let collect_fn_addr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_collect_append);
            let cb_sig = builder.func.import_signature(cranelift_codegen::ir::Signature {
                params: vec![AbiParam::new(ptr_ty), AbiParam::new(ptr_ty)],
                returns: vec![],
                call_conv: cranelift_codegen::isa::CallConv::SystemV,
            });
            codegen_generator(
                primary, input_ptr, collect_fn_addr, gen_collect_ptr, cb_sig,
                builder, ptr_ty, rt_funcs, literals, var_slots,
            );

            let prep_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let arr_ptr = builder.ins().stack_addr(ptr_ty, prep_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_prepare, &[arr_ptr, gen_collect_ptr]);

            let len_call = builder.ins().call(rt_funcs.rt_iter_length, &[arr_ptr]);
            let len = builder.inst_results(len_call)[0];

            // Pass 1: Check if any truthy element exists (use a stack slot for flag)
            let flag_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 8, 8,
            ));
            let flag_addr = builder.ins().stack_addr(ptr_ty, flag_slot, 0);
            let zero_i64 = builder.ins().iconst(types::I64, 0);
            builder.ins().store(MemFlags::trusted(), zero_i64, flag_addr, 0);

            let check_header = builder.create_block();
            let check_body = builder.create_block();
            let check_exit = builder.create_block();

            builder.append_block_param(check_header, types::I64); // idx
            let zero = builder.ins().iconst(types::I64, 0);
            let zero_arg = BlockArg::Value(zero);
            builder.ins().jump(check_header, &[zero_arg]);

            builder.switch_to_block(check_header);
            let idx = builder.block_params(check_header)[0];
            let cmp = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, idx, len,
            );
            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(cmp, check_body, empty_args, check_exit, empty_args);

            builder.switch_to_block(check_body);
            builder.seal_block(check_body);

            let elem_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let elem_ptr = builder.ins().stack_addr(ptr_ty, elem_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_get, &[elem_ptr, arr_ptr, idx]);

            let truthy_call = builder.ins().call(rt_funcs.rt_is_truthy, &[elem_ptr]);
            let is_truthy = builder.inst_results(truthy_call)[0];

            // If truthy, set flag to 1
            let set_flag_block = builder.create_block();
            let skip_flag_block = builder.create_block();
            builder.ins().brif(is_truthy, set_flag_block, empty_args, skip_flag_block, empty_args);

            builder.switch_to_block(set_flag_block);
            builder.seal_block(set_flag_block);
            let one_i64 = builder.ins().iconst(types::I64, 1);
            let flag_addr2 = builder.ins().stack_addr(ptr_ty, flag_slot, 0);
            builder.ins().store(MemFlags::trusted(), one_i64, flag_addr2, 0);
            builder.ins().jump(skip_flag_block, &[]);

            builder.switch_to_block(skip_flag_block);
            builder.seal_block(skip_flag_block);

            let one = builder.ins().iconst(types::I64, 1);
            let next_idx = builder.ins().iadd(idx, one);
            let next_arg = BlockArg::Value(next_idx);
            builder.ins().jump(check_header, &[next_arg]);

            builder.switch_to_block(check_exit);
            builder.seal_block(check_exit);
            builder.seal_block(check_header);

            // Read flag
            let flag_addr3 = builder.ins().stack_addr(ptr_ty, flag_slot, 0);
            let flag_val = builder.ins().load(types::I64, MemFlags::trusted(), flag_addr3, 0);
            let flag_nonzero = builder.ins().icmp_imm(
                cranelift_codegen::ir::condcodes::IntCC::NotEqual, flag_val, 0,
            );

            let has_truthy_block = builder.create_block();
            let no_truthy_block = builder.create_block();
            let alt_done = builder.create_block();

            builder.ins().brif(flag_nonzero, has_truthy_block, empty_args, no_truthy_block, empty_args);

            // Pass 2a: Has truthy outputs — yield only truthy ones
            builder.switch_to_block(has_truthy_block);
            builder.seal_block(has_truthy_block);

            let yield_header = builder.create_block();
            let yield_body = builder.create_block();
            let yield_exit = builder.create_block();

            builder.append_block_param(yield_header, types::I64);
            let zero2 = builder.ins().iconst(types::I64, 0);
            let zero2_arg = BlockArg::Value(zero2);
            builder.ins().jump(yield_header, &[zero2_arg]);

            builder.switch_to_block(yield_header);
            let y_idx = builder.block_params(yield_header)[0];
            let y_cmp = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, y_idx, len,
            );
            builder.ins().brif(y_cmp, yield_body, empty_args, yield_exit, empty_args);

            builder.switch_to_block(yield_body);
            builder.seal_block(yield_body);

            let y_elem_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let y_elem_ptr = builder.ins().stack_addr(ptr_ty, y_elem_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_get, &[y_elem_ptr, arr_ptr, y_idx]);

            let y_truthy_call = builder.ins().call(rt_funcs.rt_is_truthy, &[y_elem_ptr]);
            let y_is_truthy = builder.inst_results(y_truthy_call)[0];

            let do_yield = builder.create_block();
            let skip_yield = builder.create_block();
            builder.ins().brif(y_is_truthy, do_yield, empty_args, skip_yield, empty_args);

            builder.switch_to_block(do_yield);
            builder.seal_block(do_yield);
            builder.ins().call_indirect(callback_sig_ref, callback_ptr, &[y_elem_ptr, ctx_ptr]);
            builder.ins().jump(skip_yield, &[]);

            builder.switch_to_block(skip_yield);
            builder.seal_block(skip_yield);
            let y_one = builder.ins().iconst(types::I64, 1);
            let y_next = builder.ins().iadd(y_idx, y_one);
            let y_next_arg = BlockArg::Value(y_next);
            builder.ins().jump(yield_header, &[y_next_arg]);

            builder.switch_to_block(yield_exit);
            builder.seal_block(yield_exit);
            builder.seal_block(yield_header);
            builder.ins().jump(alt_done, &[]);

            // Pass 2b: No truthy outputs — yield fallback
            builder.switch_to_block(no_truthy_block);
            builder.seal_block(no_truthy_block);
            // Evaluate fallback against ORIGINAL input
            if is_generator(fallback) {
                codegen_generator(
                    fallback, input_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );
            } else {
                let (fb_ptr, _) = codegen_expr(fallback, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
                builder.ins().call_indirect(callback_sig_ref, callback_ptr, &[fb_ptr, ctx_ptr]);
            }
            builder.ins().jump(alt_done, &[]);

            builder.switch_to_block(alt_done);
            builder.seal_block(alt_done);
        }

        // Alternative with generator fallback: `expr // (gen1, gen2)`
        // If primary is truthy, yield it once; otherwise yield all outputs of fallback.
        Expr::Alternative { primary, fallback } if is_generator(fallback) => {
            let (primary_ptr, _) = codegen_expr(primary, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            let truthy_call = builder.ins().call(rt_funcs.rt_is_truthy, &[primary_ptr]);
            let is_truthy = builder.inst_results(truthy_call)[0];

            let use_primary = builder.create_block();
            let use_fallback = builder.create_block();
            let alt_exit = builder.create_block();

            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(is_truthy, use_primary, empty_args, use_fallback, empty_args);

            builder.switch_to_block(use_primary);
            builder.seal_block(use_primary);
            builder.ins().call_indirect(callback_sig_ref, callback_ptr, &[primary_ptr, ctx_ptr]);
            builder.ins().jump(alt_exit, &[]);

            builder.switch_to_block(use_fallback);
            builder.seal_block(use_fallback);
            codegen_generator(
                fallback, input_ptr, callback_ptr, ctx_ptr, callback_sig_ref,
                builder, ptr_ty, rt_funcs, literals, var_slots,
            );
            builder.ins().jump(alt_exit, &[]);

            builder.switch_to_block(alt_exit);
            builder.seal_block(alt_exit);
        }

        // BinOp with generator rhs (e.g., index(",","|"), join(",","/"), flatten(3,2,1)):
        // Collect the rhs generator outputs, then for each, evaluate the BinOp with
        // the ORIGINAL input_ptr as lhs and the generator element as rhs.
        // This avoids the extract_inner_generator problem where both lhs and rhs
        // would resolve to the generator element when lhs is also Expr::Input.
        Expr::BinOp { op, lhs, rhs } if !is_generator(lhs) && is_generator(rhs) => {
            // Collect rhs generator outputs into a temporary array
            let gen_collect_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let gen_collect_ptr = builder.ins().stack_addr(ptr_ty, gen_collect_slot, 0);
            builder.ins().call(rt_funcs.rt_collect_init, &[gen_collect_ptr]);

            let collect_fn_addr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_collect_append);
            let cb_sig = builder.func.import_signature(cranelift_codegen::ir::Signature {
                params: vec![AbiParam::new(ptr_ty), AbiParam::new(ptr_ty)],
                returns: vec![],
                call_conv: cranelift_codegen::isa::CallConv::SystemV,
            });
            codegen_generator(
                rhs, input_ptr, collect_fn_addr, gen_collect_ptr, cb_sig,
                builder, ptr_ty, rt_funcs, literals, var_slots,
            );

            // Prepare for iteration
            let prep_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let arr_ptr = builder.ins().stack_addr(ptr_ty, prep_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_prepare, &[arr_ptr, gen_collect_ptr]);

            let len_call = builder.ins().call(rt_funcs.rt_iter_length, &[arr_ptr]);
            let len = builder.inst_results(len_call)[0];

            // Loop over collected rhs values
            let loop_header = builder.create_block();
            let loop_body = builder.create_block();
            let loop_exit = builder.create_block();

            builder.append_block_param(loop_header, types::I64);
            let zero = builder.ins().iconst(types::I64, 0);
            let zero_arg = BlockArg::Value(zero);
            builder.ins().jump(loop_header, &[zero_arg]);

            builder.switch_to_block(loop_header);
            let idx = builder.block_params(loop_header)[0];
            let cmp = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, idx, len,
            );
            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(cmp, loop_body, empty_args, loop_exit, empty_args);

            builder.switch_to_block(loop_body);
            builder.seal_block(loop_body);

            // Get rhs element
            let elem_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let elem_ptr = builder.ins().stack_addr(ptr_ty, elem_slot, 0);
            builder.ins().call(rt_funcs.rt_iter_get, &[elem_ptr, arr_ptr, idx]);

            // Evaluate lhs with ORIGINAL input_ptr, then call the BinOp runtime
            // with original lhs result and the current rhs element.
            let (lhs_result, _) = codegen_expr(lhs, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);

            let binop_result_slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot, 16, 8,
            ));
            let binop_result_ptr = builder.ins().stack_addr(ptr_ty, binop_result_slot, 0);

            // Dispatch to appropriate runtime function based on BinOp type.
            // All BinOp runtime functions use the 3-arg convention: rt_func(out, lhs, rhs)
            match op {
                BinOp::Add => { builder.ins().call(rt_funcs.rt_add, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Sub => { builder.ins().call(rt_funcs.rt_sub, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Mul => { builder.ins().call(rt_funcs.rt_mul, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Div => { builder.ins().call(rt_funcs.rt_div, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Mod => { builder.ins().call(rt_funcs.rt_mod, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Eq => { builder.ins().call(rt_funcs.rt_eq, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Ne => { builder.ins().call(rt_funcs.rt_ne, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Lt => { builder.ins().call(rt_funcs.rt_lt, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Gt => { builder.ins().call(rt_funcs.rt_gt, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Le => { builder.ins().call(rt_funcs.rt_le, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Ge => { builder.ins().call(rt_funcs.rt_ge, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::StrIndex => { builder.ins().call(rt_funcs.rt_str_index, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::StrRindex => { builder.ins().call(rt_funcs.rt_str_rindex, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Indices => { builder.ins().call(rt_funcs.rt_indices, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Join => { builder.ins().call(rt_funcs.rt_join, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::FlattenDepth => { builder.ins().call(rt_funcs.rt_flatten_depth, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Contains => { builder.ins().call(rt_funcs.rt_contains, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Split => { builder.ins().call(rt_funcs.rt_split, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Has => { builder.ins().call(rt_funcs.rt_has, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::StartsWith => { builder.ins().call(rt_funcs.rt_startswith, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::EndsWith => { builder.ins().call(rt_funcs.rt_endswith, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Ltrimstr => { builder.ins().call(rt_funcs.rt_ltrimstr, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Rtrimstr => { builder.ins().call(rt_funcs.rt_rtrimstr, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Inside => { builder.ins().call(rt_funcs.rt_inside, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::In => { builder.ins().call(rt_funcs.rt_in, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::GetPath => { builder.ins().call(rt_funcs.rt_getpath, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::DelPaths => { builder.ins().call(rt_funcs.rt_delpaths, &[binop_result_ptr, lhs_result, elem_ptr]); }
                // Math binary functions
                BinOp::Pow => { builder.ins().call(rt_funcs.rt_pow, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Atan2 => { builder.ins().call(rt_funcs.rt_atan2, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Drem => { builder.ins().call(rt_funcs.rt_drem, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Ldexp => { builder.ins().call(rt_funcs.rt_ldexp, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Scalb => { builder.ins().call(rt_funcs.rt_scalb, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Scalbln => { builder.ins().call(rt_funcs.rt_scalbln, &[binop_result_ptr, lhs_result, elem_ptr]); }
                // bsearch
                BinOp::Bsearch => { builder.ins().call(rt_funcs.rt_bsearch, &[binop_result_ptr, lhs_result, elem_ptr]); }
                // Date/time with format argument
                BinOp::Strftime => { builder.ins().call(rt_funcs.rt_strftime, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Strptime => { builder.ins().call(rt_funcs.rt_strptime, &[binop_result_ptr, lhs_result, elem_ptr]); }
                BinOp::Strflocaltime => { builder.ins().call(rt_funcs.rt_strflocaltime, &[binop_result_ptr, lhs_result, elem_ptr]); }
            }

            builder.ins().call_indirect(callback_sig_ref, callback_ptr, &[binop_result_ptr, ctx_ptr]);

            let one = builder.ins().iconst(types::I64, 1);
            let next_idx = builder.ins().iadd(idx, one);
            let next_arg = BlockArg::Value(next_idx);
            builder.ins().jump(loop_header, &[next_arg]);

            builder.switch_to_block(loop_exit);
            builder.seal_block(loop_exit);
            builder.seal_block(loop_header);
        }

        // For expressions that contain inner generators (from pipe expansion via substitute_input),
        // extract the generator, collect its outputs, and iterate: for each output, evaluate the
        // template expression and yield via callback.
        other if is_generator(other) => {
            if let Some((inner_gen, template)) = extract_inner_generator(other) {
                // Collect generator outputs into a temporary array
                let gen_collect_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let gen_collect_ptr = builder.ins().stack_addr(ptr_ty, gen_collect_slot, 0);
                builder.ins().call(rt_funcs.rt_collect_init, &[gen_collect_ptr]);

                let collect_fn_addr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_collect_append);
                let cb_sig = builder.func.import_signature(cranelift_codegen::ir::Signature {
                    params: vec![AbiParam::new(ptr_ty), AbiParam::new(ptr_ty)],
                    returns: vec![],
                    call_conv: cranelift_codegen::isa::CallConv::SystemV,
                });
                codegen_generator(
                    inner_gen, input_ptr, collect_fn_addr, gen_collect_ptr, cb_sig,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );

                // Iterate over collected outputs
                let prep_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let arr_ptr = builder.ins().stack_addr(ptr_ty, prep_slot, 0);
                builder.ins().call(rt_funcs.rt_iter_prepare, &[arr_ptr, gen_collect_ptr]);

                let len_call = builder.ins().call(rt_funcs.rt_iter_length, &[arr_ptr]);
                let len = builder.inst_results(len_call)[0];

                let loop_header = builder.create_block();
                let loop_body = builder.create_block();
                let loop_exit = builder.create_block();

                builder.append_block_param(loop_header, types::I64);
                let zero = builder.ins().iconst(types::I64, 0);
                let zero_arg = BlockArg::Value(zero);
                builder.ins().jump(loop_header, &[zero_arg]);

                builder.switch_to_block(loop_header);
                let idx = builder.block_params(loop_header)[0];
                let cmp = builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, idx, len,
                );
                let empty_args: &[BlockArg] = &[];
                builder.ins().brif(cmp, loop_body, empty_args, loop_exit, empty_args);

                builder.switch_to_block(loop_body);
                builder.seal_block(loop_body);

                // Get element
                let elem_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let elem_ptr = builder.ins().stack_addr(ptr_ty, elem_slot, 0);
                builder.ins().call(rt_funcs.rt_iter_get, &[elem_ptr, arr_ptr, idx]);

                // Evaluate template with element as input, yield via callback
                let (result_ptr, _) = codegen_expr(&template, elem_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
                builder.ins().call_indirect(callback_sig_ref, callback_ptr, &[result_ptr, ctx_ptr]);

                let one = builder.ins().iconst(types::I64, 1);
                let next_idx = builder.ins().iadd(idx, one);
                let next_arg = BlockArg::Value(next_idx);
                builder.ins().jump(loop_header, &[next_arg]);

                builder.switch_to_block(loop_exit);
                builder.seal_block(loop_exit);
                builder.seal_block(loop_header);
            } else {
                // Generator at the top level — should be handled by specific patterns above.
                // This is a fallback: collect and yield each element.
                let gen_collect_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let gen_collect_ptr = builder.ins().stack_addr(ptr_ty, gen_collect_slot, 0);
                builder.ins().call(rt_funcs.rt_collect_init, &[gen_collect_ptr]);

                let collect_fn_addr = builder.ins().func_addr(ptr_ty, rt_funcs.rt_collect_append);
                let cb_sig = builder.func.import_signature(cranelift_codegen::ir::Signature {
                    params: vec![AbiParam::new(ptr_ty), AbiParam::new(ptr_ty)],
                    returns: vec![],
                    call_conv: cranelift_codegen::isa::CallConv::SystemV,
                });
                codegen_generator(
                    other, input_ptr, collect_fn_addr, gen_collect_ptr, cb_sig,
                    builder, ptr_ty, rt_funcs, literals, var_slots,
                );

                // Iterate and yield
                let prep_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let arr_ptr = builder.ins().stack_addr(ptr_ty, prep_slot, 0);
                builder.ins().call(rt_funcs.rt_iter_prepare, &[arr_ptr, gen_collect_ptr]);

                let len_call = builder.ins().call(rt_funcs.rt_iter_length, &[arr_ptr]);
                let len = builder.inst_results(len_call)[0];

                let loop_header = builder.create_block();
                let loop_body = builder.create_block();
                let loop_exit = builder.create_block();

                builder.append_block_param(loop_header, types::I64);
                let zero = builder.ins().iconst(types::I64, 0);
                let zero_arg = BlockArg::Value(zero);
                builder.ins().jump(loop_header, &[zero_arg]);

                builder.switch_to_block(loop_header);
                let idx = builder.block_params(loop_header)[0];
                let cmp = builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, idx, len,
                );
                let empty_args: &[BlockArg] = &[];
                builder.ins().brif(cmp, loop_body, empty_args, loop_exit, empty_args);

                builder.switch_to_block(loop_body);
                builder.seal_block(loop_body);

                let elem_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot, 16, 8,
                ));
                let elem_ptr = builder.ins().stack_addr(ptr_ty, elem_slot, 0);
                builder.ins().call(rt_funcs.rt_iter_get, &[elem_ptr, arr_ptr, idx]);
                builder.ins().call_indirect(callback_sig_ref, callback_ptr, &[elem_ptr, ctx_ptr]);

                let one = builder.ins().iconst(types::I64, 1);
                let next_idx = builder.ins().iadd(idx, one);
                let next_arg = BlockArg::Value(next_idx);
                builder.ins().jump(loop_header, &[next_arg]);

                builder.switch_to_block(loop_exit);
                builder.seal_block(loop_exit);
                builder.seal_block(loop_header);
            }
        }

        // For all 1→1 expressions (including LoadVar, Reduce), delegate to codegen_expr
        // and call callback once.
        other => {
            let (result_ptr, _) = codegen_expr(other, input_ptr, builder, ptr_ty, rt_funcs, literals, var_slots);
            builder.ins().call_indirect(callback_sig_ref, callback_ptr, &[result_ptr, ctx_ptr]);
        }
    }
}

/// Generate CLIF IR for a literal value.
///
/// For `Num` and `Null` and `Bool` literals: writes tag + payload directly to a StackSlot.
/// For `Str` literals: pre-allocates a Rust-side `Value::Str` and embeds its address as a constant.
fn codegen_literal(
    lit: &Literal,
    builder: &mut FunctionBuilder,
    ptr_ty: types::Type,
    literals: &mut LiteralPool,
) -> cranelift_codegen::ir::Value {
    match lit {
        Literal::Num(n) => {
            // Allocate a 16-byte StackSlot for Value::Num
            let slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                16,
                8,
            ));

            // Write tag (u64) at offset 0
            let tag = builder.ins().iconst(types::I64, TAG_NUM as i64);
            builder.ins().stack_store(tag, slot, 0);

            // Write f64 payload at offset 8
            // Cranelift stores f64 as bits, so we use f64const + bitcast to i64
            let payload = builder.ins().f64const(*n);
            let payload_bits = builder
                .ins()
                .bitcast(types::I64, MemFlags::new(), payload);
            builder.ins().stack_store(payload_bits, slot, 8);

            // Return address of the slot
            builder.ins().stack_addr(ptr_ty, slot, 0)
        }

        Literal::Null => {
            let slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                16,
                8,
            ));

            // Tag for Null = 0
            let tag = builder.ins().iconst(types::I64, TAG_NULL as i64);
            builder.ins().stack_store(tag, slot, 0);

            // Payload doesn't matter for Null, but zero it for safety
            let zero = builder.ins().iconst(types::I64, 0);
            builder.ins().stack_store(zero, slot, 8);

            builder.ins().stack_addr(ptr_ty, slot, 0)
        }

        Literal::Bool(b) => {
            let slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                16,
                8,
            ));

            let tag = builder.ins().iconst(types::I64, TAG_BOOL as i64);
            builder.ins().stack_store(tag, slot, 0);

            let payload = builder.ins().iconst(types::I64, if *b { 1 } else { 0 });
            builder.ins().stack_store(payload, slot, 8);

            builder.ins().stack_addr(ptr_ty, slot, 0)
        }

        Literal::Str(s) => {
            // String literals contain Rc<String> which can't be constructed in JIT code.
            // We pre-allocate the Value on the Rust side and load the pointer from the
            // literals table at runtime (instead of embedding a heap address directly).
            let val = Box::new(Value::Str(std::rc::Rc::new(s.clone())));
            let idx = literals.values.len();
            literals.values.push(val);
            let offset = (idx as i32) * (ptr_ty.bytes() as i32);
            builder.ins().load(ptr_ty, MemFlags::trusted(), literals.clif_param, offset)
        }

        Literal::EmptyArr => {
            // Empty array literal [].
            // Pre-allocate on Rust side; load pointer from literals table.
            let val = Box::new(Value::Arr(std::rc::Rc::new(Vec::new())));
            let idx = literals.values.len();
            literals.values.push(val);
            let offset = (idx as i32) * (ptr_ty.bytes() as i32);
            builder.ins().load(ptr_ty, MemFlags::trusted(), literals.clif_param, offset)
        }

        Literal::Arr(items) => {
            // Phase 8-8: Non-empty array literal [1,2,3].
            // Pre-allocate on Rust side; load pointer from literals table.
            let val = Box::new(Value::Arr(std::rc::Rc::new(items.clone())));
            let idx = literals.values.len();
            literals.values.push(val);
            let offset = (idx as i32) * (ptr_ty.bytes() as i32);
            builder.ins().load(ptr_ty, MemFlags::trusted(), literals.clif_param, offset)
        }

        Literal::EmptyObj => {
            // Phase 8-8: Empty object literal {}.
            // Pre-allocate on Rust side; load pointer from literals table.
            let val = Box::new(Value::Obj(std::rc::Rc::new(std::collections::BTreeMap::new())));
            let idx = literals.values.len();
            literals.values.push(val);
            let offset = (idx as i32) * (ptr_ty.bytes() as i32);
            builder.ins().load(ptr_ty, MemFlags::trusted(), literals.clif_param, offset)
        }

        Literal::Obj(map) => {
            // Non-empty object literal.
            // Pre-allocate on Rust side; load pointer from literals table.
            let val = Box::new(Value::Obj(std::rc::Rc::new(map.clone())));
            let idx = literals.values.len();
            literals.values.push(val);
            let offset = (idx as i32) * (ptr_ty.bytes() as i32);
            builder.ins().load(ptr_ty, MemFlags::trusted(), literals.clif_param, offset)
        }

        Literal::Error(msg) => {
            // Phase 10-4: Error literal.
            // Pre-allocate on Rust side; load pointer from literals table.
            let val = Box::new(Value::Error(std::rc::Rc::new(msg.clone())));
            let idx = literals.values.len();
            literals.values.push(val);
            let offset = (idx as i32) * (ptr_ty.bytes() as i32);
            builder.ins().load(ptr_ty, MemFlags::trusted(), literals.clif_param, offset)
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 10-3: Slice assignment detection
// ---------------------------------------------------------------------------

/// Extract the slice key expression from a path_expr if it is a slice pattern.
///
/// A slice pattern looks like:
/// ```text
/// Index { expr: Input, key: ObjectInsert { obj: ObjectInsert { obj: Literal(EmptyObj),
///         key: Literal(Str("start")), value: <start_expr> },
///         key: Literal(Str("end")), value: <end_expr> } }
/// ```
///
/// Returns the ObjectInsert sub-expression (the slice key) if the pattern matches.
fn extract_slice_key(path_expr: &Expr) -> Option<Expr> {
    if let Expr::Index { expr: input, key } = path_expr {
        if !matches!(input.as_ref(), Expr::Input) {
            return None;
        }
        // Check if key is an ObjectInsert chain building {"start": ..., "end": ...}
        if is_slice_object_insert(key) {
            return Some(*key.clone());
        }
    }
    None
}

/// Check if an expression is an ObjectInsert chain that builds a slice key
/// (an object with "start" and/or "end" fields).
fn is_slice_object_insert(expr: &Expr) -> bool {
    // Pattern: ObjectInsert { obj: ObjectInsert { obj: EmptyObj, key: "start", value: _ }, key: "end", value: _ }
    // Or: ObjectInsert { obj: ObjectInsert { obj: EmptyObj, key: "end", value: _ }, key: "start", value: _ }
    // Or just one of them
    match expr {
        Expr::ObjectInsert { obj, key, value: _ } => {
            if let Expr::Literal(Literal::Str(k)) = key.as_ref() {
                if k == "start" || k == "end" {
                    // Inner obj should be either EmptyObj or another ObjectInsert with the other key
                    match obj.as_ref() {
                        Expr::Literal(Literal::EmptyObj) => return true,
                        Expr::ObjectInsert { obj: inner_obj, key: inner_key, value: _ } => {
                            if let Expr::Literal(Literal::Str(ik)) = inner_key.as_ref() {
                                if (ik == "start" || ik == "end") && ik != k {
                                    if matches!(inner_obj.as_ref(), Expr::Literal(Literal::EmptyObj)) {
                                        return true;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            false
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Phase 9-5 / 10-3: Path descriptor analysis
// ---------------------------------------------------------------------------

/// Analyze a path expression IR and produce a path descriptor Value.
///
/// The descriptor tells the runtime what paths to extract from a value.
/// This is used by both `path(expr)` and `|=` (update operators).
///
/// Returns a Value that encodes the path pattern:
/// - `Value::Arr([...])` for static Index chains (`.a`, `.a.b`, `.[0]`)
/// - `Value::Str("each")` for `.[]`
/// - `Value::Str("each_opt")` for `.[]?`
/// - `Value::Str("recurse")` for `..`
fn analyze_path_descriptor(expr: &Expr) -> Value {
    // Try to extract a static path (chain of Index operations)
    if let Some(path) = extract_static_path(expr) {
        return Value::Arr(std::rc::Rc::new(path));
    }

    // Check for Each (.[] or .[]?)
    match expr {
        Expr::Each { input_expr, body } => {
            if matches!(input_expr.as_ref(), Expr::Input) && matches!(body.as_ref(), Expr::Input) {
                return Value::Str(std::rc::Rc::new("each".to_string()));
            }
            // .a[] or .a.b[] — prefix + each
            if let Some(prefix) = extract_static_path(input_expr) {
                // This is a compound path like .a[] — we need the descriptor to handle it
                let mut map = std::collections::BTreeMap::new();
                map.insert("type".to_string(), Value::Str(std::rc::Rc::new("index_then_each".to_string())));
                map.insert("keys".to_string(), Value::Arr(std::rc::Rc::new(prefix)));
                return Value::Obj(std::rc::Rc::new(map));
            }
            Value::Str(std::rc::Rc::new("each".to_string()))
        }
        Expr::EachOpt { input_expr, body } => {
            if matches!(input_expr.as_ref(), Expr::Input) && matches!(body.as_ref(), Expr::Input) {
                return Value::Str(std::rc::Rc::new("each_opt".to_string()));
            }
            Value::Str(std::rc::Rc::new("each_opt".to_string()))
        }
        Expr::Recurse { input_expr, body: _ } => {
            if matches!(input_expr.as_ref(), Expr::Input) {
                return Value::Str(std::rc::Rc::new("recurse".to_string()));
            }
            Value::Str(std::rc::Rc::new("recurse".to_string()))
        }
        // For Comma (e.g., path(.a, .b)), we'd need a compound descriptor
        // For now, fall back to a static path for the first branch
        Expr::Comma { left: _, right: _ } => {
            // Not directly supported as a single descriptor yet.
            // The caller should handle Comma at a higher level.
            // Return a placeholder that won't match anything.
            Value::Null
        }
        _ => {
            // Unknown pattern — return null (produces no paths)
            Value::Null
        }
    }
}

/// Try to extract a static path from an Index chain.
///
/// `.a` → Some([Str("a")])
/// `.a.b` → Some([Str("a"), Str("b")])
/// `.[0]` → Some([Num(0)])
/// Returns None if the expression is not a pure Index chain.
fn extract_static_path(expr: &Expr) -> Option<Vec<Value>> {
    match expr {
        Expr::Index { expr: inner, key } => {
            if matches!(inner.as_ref(), Expr::Input) {
                // Base case: .[key] where key is a literal
                if let Some(key_val) = literal_to_value(key) {
                    return Some(vec![key_val]);
                }
            } else if let Some(mut prefix) = extract_static_path(inner) {
                // Recursive case: expr[key]
                if let Some(key_val) = literal_to_value(key) {
                    prefix.push(key_val);
                    return Some(prefix);
                }
            }
            None
        }
        Expr::Input => {
            // Identity: path(.) = [] (empty path)
            Some(vec![])
        }
        _ => None,
    }
}

/// Convert a literal expression to a Value for use in path descriptors.
fn literal_to_value(expr: &Expr) -> Option<Value> {
    match expr {
        Expr::Literal(Literal::Str(s)) => Some(Value::Str(std::rc::Rc::new(s.clone()))),
        Expr::Literal(Literal::Num(n)) => Some(Value::Num(*n)),
        Expr::Literal(Literal::Null) => Some(Value::Null),
        Expr::Literal(Literal::Bool(b)) => Some(Value::Bool(*b)),
        // Handle negated number literals (e.g., .[-1] compiles as Negate(1))
        Expr::UnaryOp { op: UnaryOp::Negate, operand } => {
            if let Expr::Literal(Literal::Num(n)) = operand.as_ref() {
                Some(Value::Num(-n))
            } else {
                None
            }
        }
        // Handle nan (UnaryOp::Nan) — produces NaN
        Expr::UnaryOp { op: UnaryOp::Nan, .. } => {
            Some(Value::Num(f64::NAN))
        }
        // Handle infinite (UnaryOp::Infinite) — produces infinity
        Expr::UnaryOp { op: UnaryOp::Infinite, .. } => {
            Some(Value::Num(f64::INFINITY))
        }
        _ => None,
    }
}
