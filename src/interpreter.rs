//! Filter execution: parser → IR → tree-walking interpreter or JIT.
//!
//! All paths (parse / eval / JIT) run in pure Rust — the libjq fallback
//! was removed in the 1.3.0 line.


use anyhow::Result;

use crate::ir::CompiledFunc;
use crate::value::Value;

/// Forced-execution-mode bits, packed into a single atomic so the
/// per-record [`Filter::execute`] / [`Filter::execute_cb`] paths pay one
/// relaxed load regardless of how many force knobs exist (the ~55ns/record
/// NDJSON loops notice every extra load — see the classified-boundary-scan
/// notes in docs/maintenance.md).
///
/// - `FORCE_EVAL_BIT` — self-diff knob for issue #323
///   (`JQJIT_FORCE_INTERPRETER`): skip the typed fast path and JIT and run
///   the generic tree-walking interpreter, regardless of whether
///   `compile_jit` was called. See tests/selfdiff_jit_interp.rs.
/// - `FORCE_JITOP_BIT` — self-diff knob for issue #1059
///   (`JQJIT_FORCE_JITOP_INTERP`): skip the typed fast path and the
///   Cranelift JIT and run every flattenable filter on the direct JitOp
///   interpreter backend ([`crate::jit::JitProgram`]); filters the
///   flattener rejects fall back to the tree-walking eval path.
/// - `FORCE_CRANELIFT_BIT` — the Cranelift-side counterpart
///   (`JQJIT_FORCE_CRANELIFT`): identical routing, compiled backend.
///   Diffing the two pins the *backends* against each other over the same
///   lowering with no fast-path asymmetry. See
///   tests/selfdiff_jitop_backend.rs.
///
/// The binary sets these once at startup from the environment / CLI flags.
const FORCE_EVAL_BIT: u8 = 1;
const FORCE_JITOP_BIT: u8 = 2;
const FORCE_CRANELIFT_BIT: u8 = 4;

static FORCED_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn set_forced_mode_bit(bit: u8, on: bool) {
    if on {
        FORCED_MODE.fetch_or(bit, std::sync::atomic::Ordering::Relaxed);
    } else {
        FORCED_MODE.fetch_and(!bit, std::sync::atomic::Ordering::Relaxed);
    }
}

fn forced_mode() -> u8 {
    FORCED_MODE.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_force_interpreter(on: bool) {
    set_forced_mode_bit(FORCE_EVAL_BIT, on);
}

pub fn force_interpreter() -> bool {
    forced_mode() & FORCE_EVAL_BIT != 0
}

pub fn set_force_jitop_interp(on: bool) {
    set_forced_mode_bit(FORCE_JITOP_BIT, on);
}

pub fn force_jitop_interp() -> bool {
    forced_mode() & FORCE_JITOP_BIT != 0
}

pub fn set_force_cranelift(on: bool) {
    set_forced_mode_bit(FORCE_CRANELIFT_BIT, on);
}

pub fn force_cranelift() -> bool {
    forced_mode() & FORCE_CRANELIFT_BIT != 0
}

/// Layer-pinning knob for issue #685: when set, [`simplify_expr`] returns its
/// input unchanged, disabling every fast-path-detection rewrite (the (b)
/// layer in `docs/maintenance.md` §2). Used by
/// `tests/selfdiff_layers.rs` to isolate which fast-path layer is
/// responsible when the 2-way self-diff disagrees.
static DISABLE_SIMPLIFY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn set_disable_simplify(on: bool) {
    DISABLE_SIMPLIFY.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn disable_simplify() -> bool {
    DISABLE_SIMPLIFY.load(std::sync::atomic::Ordering::Relaxed)
}

// Fast-path shape enums and the `detect_*` / `classify_*` methods on
// [`Filter`] live in [`crate::classify`]; expression simplification lives
// in [`crate::simplify`] (#1029). Re-export the shape types so existing
// `crate::interpreter::*` paths keep working.
pub use crate::classify::{
    ArithExpr, BranchOutput, CmpVal, CondBranch, CondRhs, IfArrayCond, InterpPart, MathUnary,
    MixedCond, NumChainStep, RemapExpr, SplitConcatPart, StrBuiltin, StrFuncCond, StringAddPart,
    StringChainOp, StringChainTerminal,
};

/// A compiled jq filter, ready to execute.
pub struct Filter {
    /// Our parsed IR.
    pub(crate) parsed: (crate::ir::Expr, Vec<CompiledFunc>),
    /// Simplified expression for fast path detection (identity pipes stripped).
    pub(crate) simplified: crate::ir::Expr,
    /// JIT-compiled function (if JIT compilation succeeded).
    jit_fn: Option<crate::jit::JitFilterFn>,
    /// JIT compiler kept alive to own the compiled code.
    _jit_compiler: Option<Box<crate::jit::JitCompiler>>,
    /// JitOp program for the direct interpreter backend (#1059). Only built
    /// when `JQJIT_FORCE_JITOP_INTERP` routes execution through it.
    jit_program: Option<crate::jit::JitProgram>,
    lib_dirs: Vec<String>,
    /// Cached eval environment to avoid re-allocating per call.
    cached_env: std::cell::RefCell<Option<crate::eval::EnvRef>>,
    /// Number of `memoize(...)` slots the program needs. Used to size the
    /// per-program memo cache when the Env is materialized.
    memo_slots: u32,
    /// Per-slot upper bound on cache entries (CLI `--memo-max-entries`).
    memo_max_entries: usize,
}

impl Filter {
    /// Get the inner expression for pattern detection.
    ///
    /// Previously this stripped top-level `try EXPR` (TryCatch with Empty
    /// catch) on the assumption that the raw byte fast paths handled missing
    /// fields gracefully. They don't: `(.a)?` on a non-object needs to emit
    /// nothing (the error is caught), while the fast path emitted `null`.
    /// Leave TryCatch visible so the fast paths that can't honour `?`
    /// semantics simply don't match, and eval handles it correctly (see
    /// issue #50).
    pub(crate) fn detect_expr(&self) -> Option<&crate::ir::Expr> {
        Some(&self.simplified)
    }

    /// Probe the [`crate::fast_path::FastPath`]-shaped (typed) dispatch
    /// table for this filter. See `src/fast_path.rs` for the contract.
    ///
    /// Returns the fast path's verdict:
    ///
    /// - `Some(Ok(v))`  — the typed fast path produced `v`.
    /// - `Some(Err(e))` — the typed fast path detected the same error
    ///   jq would raise on this input.
    /// - `None`         — no typed fast path matched this filter shape,
    ///   or the fast path declined to handle this input type. The caller
    ///   should run the generic eval / jit path (authoritative).
    ///
    /// Currently wired paths: [`crate::fast_path::FieldAccessPath`]
    /// (single `.field` access). More paths migrate in follow-up PRs —
    /// each migration replaces a raw-byte detector whose type-dispatch
    /// obligations have historically leaked (see issue #83).
    pub fn try_typed_fast_path(&self, input: &Value) -> Option<Result<Value>> {
        use crate::fast_path::{FastPath, FieldAccessPath};
        if let Some(field) = self.detect_field_access() {
            return FieldAccessPath::new(field).run(input);
        }
        None
    }

    pub fn new(program: &str) -> Result<Self> {
        Self::with_options(program, &[], true)
    }

    pub fn with_lib_dirs(program: &str, lib_dirs: &[String]) -> Result<Self> {
        Self::with_options(program, lib_dirs, true)
    }

    pub fn with_options(program: &str, lib_dirs: &[String], use_jit: bool) -> Result<Self> {
        let result = crate::parser::Parser::parse_with_libs(program, lib_dirs)?;
        let memo_slots = result.memo_slots;
        let parsed = (result.expr, result.funcs);

        // Try JIT compilation for the parsed expression.
        let mut jit_fn = None;
        let mut jit_compiler = None;
        if use_jit {
            let (ref expr, ref funcs) = parsed;
            if crate::jit::is_jit_compilable_with_funcs(expr, funcs) {
                if let Ok(mut compiler) = crate::jit::JitCompiler::new() {
                    if let Ok(func) = compiler.compile_with_funcs(expr, funcs) {
                        jit_fn = Some(func);
                        jit_compiler = Some(Box::new(compiler));
                    }
                }
            }
        }

        let simplified = crate::simplify::simplify_expr(&parsed.0);

        let _ = program;
        Ok(Filter {
            parsed,
            simplified,
            jit_fn,
            _jit_compiler: jit_compiler,
            jit_program: None,
            lib_dirs: lib_dirs.to_vec(),
            cached_env: std::cell::RefCell::new(None),
            memo_slots,
            memo_max_entries: crate::eval::default_memo_max_entries(),
        })
    }

    /// Set the per-slot memoize cache cap. Honoured by the eval Env when
    /// it is first materialized; subsequent calls reuse the same cap.
    /// No-op for programs with no `memoize(...)` to keep `Env` slim.
    pub fn set_memo_max_entries(&mut self, n: usize) {
        self.memo_max_entries = n;
        if self.memo_slots == 0 { return; }
        if let Some(ref env) = *self.cached_env.borrow() {
            env.borrow_mut().set_memo_max_entries(n);
        }
    }

    /// Number of `memoize(...)` call sites in the parsed program.
    pub fn memo_slot_count(&self) -> u32 {
        self.memo_slots
    }

    /// Dump per-slot memoize cache stats (hits / misses / size) to the
    /// given writer. Used by `--debug-memo`. No-op if the program contains
    /// no `memoize` calls or if the Env was never materialized.
    pub fn dump_memo_stats(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        if self.memo_slots == 0 { return Ok(()); }
        let cached = self.cached_env.borrow();
        let env = match cached.as_ref() {
            Some(e) => e,
            None => return Ok(()),
        };
        let env_ref = env.borrow();
        let memo = match env_ref.memo.as_ref() {
            Some(m) => m,
            None => return Ok(()),
        };
        writeln!(w, "memoize stats: {} slot(s)", memo.slots.len())?;
        writeln!(w, "  {:>4}  {:>12}  {:>12}  {:>12}", "slot", "hits", "misses", "entries")?;
        for (i, slot) in memo.slots.iter().enumerate() {
            let s = slot.borrow();
            writeln!(w, "  {:>4}  {:>12}  {:>12}  {:>12}", i, s.hits, s.misses, s.entries.len())?;
        }
        Ok(())
    }

    /// Try to JIT-compile this filter if not already JIT'd.
    /// Call this after determining the input is large enough to justify compilation.
    pub fn compile_jit(&mut self) {
        if self.jit_fn.is_some() { return; }
        {
            let (ref expr, ref funcs) = self.parsed;
            if crate::jit::is_jit_compilable_with_funcs(expr, funcs) {
                if let Ok(mut compiler) = crate::jit::JitCompiler::new() {
                    if let Ok(func) = compiler.compile_with_funcs(expr, funcs) {
                        self.jit_fn = Some(func);
                        self._jit_compiler = Some(Box::new(compiler));
                        // Wire the parse-time memoize slot count into the
                        // delegated-eval config (#1059 Phase 3c).
                        crate::jit::publish_delegate_memo_config(
                            self.memo_slots, self.memo_max_entries);
                    }
                }
            }
        }
    }

    /// Returns true if this filter has a JIT-compiled function.
    pub fn has_jit(&self) -> bool {
        self.jit_fn.is_some()
    }

    /// Like `compile_jit`, but skips the default-routing heuristics
    /// entirely. Used by the forced-mode knobs (`JQJIT_FORCE_CRANELIFT`,
    /// `--force-jit`) so the backend self-diff covers every delegable
    /// filter. Since #1059 Phase 3.9 the default gate also accepts
    /// delegated programs, so the remaining difference is historical
    /// naming plus future heuristic divergence.
    pub fn compile_jit_with_delegates(&mut self) {
        if self.jit_fn.is_some() { return; }
        {
            let (ref expr, ref funcs) = self.parsed;
            if crate::jit::is_jit_compilable_with_delegates(expr, funcs) {
                if let Ok(mut compiler) = crate::jit::JitCompiler::new() {
                    if let Ok(func) = compiler.compile_with_funcs(expr, funcs) {
                        self.jit_fn = Some(func);
                        self._jit_compiler = Some(Box::new(compiler));
                        // Wire the parse-time memoize slot count into the
                        // delegated-eval config (#1059 Phase 3c).
                        crate::jit::publish_delegate_memo_config(
                            self.memo_slots, self.memo_max_entries);
                    }
                }
            }
        }
    }

    /// Build the JitOp program for the direct interpreter backend (#1059).
    /// Mirrors `compile_jit` but stops at the shared lowering — no Cranelift
    /// codegen. No-op if the flattener rejects the filter; execution then
    /// falls back to the tree-walking eval path.
    pub fn compile_jitop_program(&mut self) {
        if self.jit_program.is_some() { return; }
        let (ref expr, ref funcs) = self.parsed;
        // Forced-mode knob: accept delegated programs too, mirroring
        // `compile_jit_with_delegates` on the Cranelift side.
        if crate::jit::is_jit_compilable_with_delegates(expr, funcs) {
            if let Ok(prog) = crate::jit::JitProgram::compile(expr, funcs) {
                self.jit_program = Some(prog);
                crate::jit::publish_delegate_memo_config(
                    self.memo_slots, self.memo_max_entries);
            }
        }
    }

    /// Returns true if this filter has a compiled JitOp program (#1059).
    pub fn has_jitop_program(&self) -> bool {
        self.jit_program.is_some()
    }

    /// Like `compile_jitop_program`, but for the default dispatch (#1059
    /// Phase 2): additionally rejects programs whose loop bodies materialize
    /// slot Values, where the tree-walking evaluator measures 1.4-2.6x
    /// faster (see `JitProgram::eligible_for_default_routing`). The
    /// forced-mode knob keeps the ungated compile so the backend self-diff
    /// still covers every flattenable filter.
    pub fn compile_jitop_program_for_routing(&mut self) {
        if self.jit_program.is_some() { return; }
        let (ref expr, ref funcs) = self.parsed;
        if crate::jit::is_jit_compilable_with_funcs(expr, funcs) {
            if let Ok(prog) = crate::jit::JitProgram::compile(expr, funcs) {
                if prog.eligible_for_default_routing() {
                    self.jit_program = Some(prog);
                    crate::jit::publish_delegate_memo_config(
                        self.memo_slots, self.memo_max_entries);
                }
            }
        }
    }

    /// Returns true if this filter has loop constructs that benefit from JIT.
    /// Specifically: Update (.[] |= f), While/Until/Repeat, and Reduce/Foreach
    /// whose source references the input (e.g. `.[]` but not `range(N)`).
    /// For constant-range reduces on small inputs, eval.rs handles them efficiently.
    pub fn has_loop_constructs(&self) -> bool {
        use crate::ir::Expr;
        fn references_input(e: &Expr) -> bool {
            match e {
                Expr::Input => true,
                Expr::Pipe { left, right } | Expr::Comma { left, right }
                | Expr::BinOp { lhs: left, rhs: right, .. } => {
                    references_input(left) || references_input(right)
                }
                Expr::UnaryOp { operand, .. } | Expr::Negate { operand }
                | Expr::Collect { generator: operand } => references_input(operand),
                Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => {
                    references_input(expr) || references_input(key)
                }
                Expr::Range { from, to, step } => {
                    references_input(from) || references_input(to)
                    || step.as_ref().map_or(false, |s| references_input(s))
                }
                Expr::CallBuiltin { args, .. } => args.iter().any(|a| references_input(a)),
                _ => false,
            }
        }
        fn check(e: &Expr) -> bool {
            match e {
                Expr::Update { .. } | Expr::Mutate { .. } => true,
                Expr::While { .. } | Expr::Until { .. } | Expr::Repeat { .. } => true,
                Expr::Reduce { source, .. } | Expr::Foreach { source, .. } => {
                    references_input(source)
                }
                _ => false,
            }
        }
        fn walk(e: &Expr) -> bool {
            if check(e) { return true; }
            match e {
                Expr::Pipe { left, right } | Expr::Comma { left, right }
                | Expr::BinOp { lhs: left, rhs: right, .. }
                | Expr::Alternative { primary: left, fallback: right } => {
                    walk(left) || walk(right)
                }
                Expr::UnaryOp { operand, .. } | Expr::Negate { operand }
                | Expr::Collect { generator: operand } | Expr::Each { input_expr: operand }
                | Expr::EachOpt { input_expr: operand } | Expr::Recurse { input_expr: operand } => {
                    walk(operand)
                }
                Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => walk(expr) || walk(key),
                Expr::IfThenElse { cond, then_branch, else_branch } => {
                    walk(cond) || walk(then_branch) || walk(else_branch)
                }
                Expr::TryCatch { try_expr, catch_expr, .. } => walk(try_expr) || walk(catch_expr),
                Expr::Reduce { source, init, update, .. } => walk(source) || walk(init) || walk(update),
                Expr::Foreach { source, init, update, extract, .. } => {
                    walk(source) || walk(init) || walk(update) || extract.as_ref().map_or(false, |e| walk(e))
                }
                Expr::Slice { expr, from, to } => {
                    walk(expr) || from.as_ref().map_or(false, |e| walk(e))
                    || to.as_ref().map_or(false, |e| walk(e))
                }
                Expr::ObjectConstruct { pairs } => pairs.iter().any(|(k, v)| walk(k) || walk(v)),
                Expr::LetBinding { value, body, .. } => walk(value) || walk(body),
                Expr::Label { body, .. } => walk(body),
                Expr::CallBuiltin { args, .. } => args.iter().any(|a| walk(a)),
                Expr::Update { path_expr, update_expr } | Expr::Assign { path_expr, value_expr: update_expr } => {
                    walk(path_expr) || walk(update_expr)
                }
                Expr::Mutate { path_expr, value_expr, .. } => walk(path_expr) || walk(value_expr),
                _ => false,
            }
        }
        walk(&self.parsed.0)
    }

    /// Returns true if the filter uses `input` or `inputs` anywhere.
    pub fn uses_inputs(&self) -> bool {
        use crate::ir::Expr;
        // `funcs` lets us follow a `FuncCall` into its body: `def f: input; f`
        // hides the stream read behind the call, and without descending we'd
        // report no-inputs and skip seeding the queue (#853). `visited` guards
        // against recursive/mutually-recursive defs.
        fn walk(e: &Expr, funcs: &[crate::ir::CompiledFunc], visited: &mut Vec<crate::ir::FuncId>) -> bool {
            macro_rules! walk { ($x:expr) => { walk($x, funcs, visited) }; }
            match e {
                Expr::ReadInput | Expr::ReadInputs => true,
                Expr::Pipe { left, right } | Expr::Comma { left, right }
                | Expr::BinOp { lhs: left, rhs: right, .. }
                | Expr::Alternative { primary: left, fallback: right } => walk!(left) || walk!(right),
                Expr::UnaryOp { operand, .. } | Expr::Negate { operand }
                | Expr::Collect { generator: operand } | Expr::Each { input_expr: operand }
                | Expr::EachOpt { input_expr: operand } | Expr::Recurse { input_expr: operand } => walk!(operand),
                Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => walk!(expr) || walk!(key),
                Expr::IfThenElse { cond, then_branch, else_branch } => walk!(cond) || walk!(then_branch) || walk!(else_branch),
                Expr::TryCatch { try_expr, catch_expr, .. } => walk!(try_expr) || walk!(catch_expr),
                Expr::Reduce { source, init, update, .. } => walk!(source) || walk!(init) || walk!(update),
                Expr::Foreach { source, init, update, extract, .. } => walk!(source) || walk!(init) || walk!(update) || extract.as_ref().map_or(false, |e| walk!(e)),
                Expr::Slice { expr, from, to } => walk!(expr) || from.as_ref().map_or(false, |e| walk!(e)) || to.as_ref().map_or(false, |e| walk!(e)),
                Expr::ObjectConstruct { pairs } => pairs.iter().any(|(k, v)| walk!(k) || walk!(v)),
                Expr::LetBinding { value, body, .. } => walk!(value) || walk!(body),
                Expr::Label { body, .. } => walk!(body),
                Expr::CallBuiltin { args, .. } => args.iter().any(|a| walk!(a)),
                Expr::Update { path_expr, update_expr } | Expr::Assign { path_expr, value_expr: update_expr } => walk!(path_expr) || walk!(update_expr),
                Expr::Mutate { path_expr, value_expr, .. } => walk!(path_expr) || walk!(value_expr),
                Expr::ClosureOp { input_expr, key_expr, .. } => walk!(input_expr) || walk!(key_expr),
                // Path-expression forms wrap a sub-expression that may pull from
                // the input stream (`getpath([input])`, `setpath([input];9)`,
                // `delpaths([[input]])`, `path(input|.a)`). Omitting these made
                // `uses_inputs()` report false, so the binary never seeded the
                // input queue and `input` raised a bogus `break` (#853).
                Expr::GetPath { path } => walk!(path),
                Expr::SetPath { path, value } => walk!(path) || walk!(value),
                Expr::DelPaths { paths } => walk!(paths),
                Expr::PathExpr { expr: e } | Expr::Debug { expr: e } => walk!(e),
                Expr::StringInterpolation { parts } => parts.iter().any(|p| {
                    matches!(p, crate::ir::StringPart::Expr(e) if walk!(e))
                }),
                Expr::Format { expr: e, .. } => walk!(e),
                Expr::Limit { count, generator } => walk!(count) || walk!(generator),
                // `any(inputs; cond)` / `all(inputs; cond)` (and an
                // input-consuming predicate like `any(.[]; . == input)`) read
                // the stream. Omitting these made `uses_inputs()` report false,
                // so the binary never seeded the input queue under `-n` and the
                // generator yielded nothing (any -> false, all -> true). #928
                Expr::AnyShort { generator, predicate } | Expr::AllShort { generator, predicate } => walk!(generator) || walk!(predicate),
                // Regex builtins evaluate their sub-expressions against the
                // input, and `sub`/`gsub` evaluate the replacement (`tostr`)
                // as a generator that can pull from the stream
                // (`gsub(re; input)`). Omitting these made `uses_inputs()`
                // report false, so the binary never seeded the input queue:
                // `input` raised a bogus `break` and the main loop replayed
                // each remaining document as a separate top-level input,
                // producing the wrong output shape (#930).
                Expr::RegexTest { input_expr, re, flags }
                | Expr::RegexMatch { input_expr, re, flags }
                | Expr::RegexCapture { input_expr, re, flags }
                | Expr::RegexScan { input_expr, re, flags } => walk!(input_expr) || walk!(re) || walk!(flags),
                Expr::RegexSub { input_expr, re, tostr, flags }
                | Expr::RegexGsub { input_expr, re, tostr, flags } => walk!(input_expr) || walk!(re) || walk!(tostr) || walk!(flags),
                Expr::While { cond, update, .. } | Expr::Until { cond, update } => walk!(cond) || walk!(update),
                Expr::Repeat { update, .. } => walk!(update),
                Expr::Range { from, to, step } => {
                    walk!(from) || walk!(to) || step.as_ref().map_or(false, |s| walk!(s))
                }
                // A user-defined call hides the stream read in its body; descend
                // into it (guarding recursion via `visited`) plus its arguments.
                Expr::FuncCall { func_id, args } => {
                    if args.iter().any(|a| walk!(a)) { return true; }
                    if visited.contains(func_id) { return false; }
                    visited.push(*func_id);
                    funcs.get(func_id.idx()).map_or(false, |f| walk(&f.body, funcs, visited))
                }
                _ => false,
            }
        }
        walk(&self.parsed.0, &self.parsed.1, &mut Vec::new())
    }

    /// Returns true if the AST contains any runtime loop construct (Reduce,
    /// Foreach, Recurse, Update, While, Until, Repeat) anywhere — including
    /// inside `def` bodies. Used by the binary's null-input JIT gate: with no
    /// input to amortize against, eval.rs's "small input + constant-range
    /// reduce" exception (which makes `has_loop_constructs` gate on
    /// input-referencing source and walk only the top-level expression) no
    /// longer applies — any loop is a real runtime loop the JIT can usually
    /// speed up enough to cover compile cost.
    pub fn has_any_runtime_loop_construct(&self) -> bool {
        use crate::ir::Expr;
        fn walk(e: &Expr) -> bool {
            match e {
                Expr::Reduce { .. } | Expr::Foreach { .. } | Expr::Recurse { .. }
                | Expr::Update { .. } | Expr::Mutate { .. }
                | Expr::While { .. } | Expr::Until { .. }
                | Expr::Repeat { .. } => true,
                Expr::Pipe { left, right } | Expr::Comma { left, right }
                | Expr::BinOp { lhs: left, rhs: right, .. }
                | Expr::Alternative { primary: left, fallback: right } => walk(left) || walk(right),
                Expr::UnaryOp { operand, .. } | Expr::Negate { operand }
                | Expr::Collect { generator: operand } | Expr::Each { input_expr: operand }
                | Expr::EachOpt { input_expr: operand } => walk(operand),
                Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => walk(expr) || walk(key),
                Expr::IfThenElse { cond, then_branch, else_branch } =>
                    walk(cond) || walk(then_branch) || walk(else_branch),
                Expr::TryCatch { try_expr, catch_expr, .. } => walk(try_expr) || walk(catch_expr),
                Expr::Slice { expr, from, to } => walk(expr)
                    || from.as_ref().map_or(false, |e| walk(e))
                    || to.as_ref().map_or(false, |e| walk(e)),
                Expr::ObjectConstruct { pairs } => pairs.iter().any(|(k, v)| walk(k) || walk(v)),
                Expr::LetBinding { value, body, .. } => walk(value) || walk(body),
                Expr::Label { body, .. } => walk(body),
                Expr::Break { value, .. } => walk(value),
                Expr::CallBuiltin { args, .. } | Expr::FuncCall { args, .. } => args.iter().any(walk),
                Expr::Assign { path_expr, value_expr } => walk(path_expr) || walk(value_expr),
                Expr::ClosureOp { input_expr, key_expr, .. } => walk(input_expr) || walk(key_expr),
                Expr::Format { expr: e, .. } | Expr::PathExpr { expr: e } | Expr::Debug { expr: e } | Expr::Stderr { expr: e } => walk(e),
                Expr::Limit { count, generator } => walk(count) || walk(generator),
                Expr::AllShort { generator, predicate } | Expr::AnyShort { generator, predicate } => walk(generator) || walk(predicate),
                Expr::Range { from, to, step } => walk(from) || walk(to) || step.as_ref().map_or(false, |s| walk(s)),
                Expr::SetPath { path, value } => walk(path) || walk(value),
                Expr::GetPath { path } => walk(path),
                Expr::DelPaths { paths } => walk(paths),
                Expr::StringInterpolation { parts } => parts.iter().any(|p| match p {
                    crate::ir::StringPart::Expr(e) => walk(e),
                    crate::ir::StringPart::Literal(_) => false,
                }),
                Expr::Error { msg } => msg.as_ref().map_or(false, |e| walk(e)),
                Expr::AlternativeDestructure { alternatives } => alternatives.iter().any(walk),
                Expr::RegexTest { input_expr, re, flags, .. }
                | Expr::RegexMatch { input_expr, re, flags, .. }
                | Expr::RegexCapture { input_expr, re, flags, .. }
                | Expr::RegexScan { input_expr, re, flags, .. } => walk(input_expr) || walk(re) || walk(flags),
                Expr::RegexSub { input_expr, re, tostr, flags }
                | Expr::RegexGsub { input_expr, re, tostr, flags } => walk(input_expr) || walk(re) || walk(tostr) || walk(flags),
                _ => false,
            }
        }
        // Also walk def bodies — a reduce buried inside `def f: reduce ...; f`
        // is just as much a runtime loop as a top-level one (#658 follow-up).
        walk(&self.parsed.0) || self.parsed.1.iter().any(|f| walk(&f.body))
    }

    /// Counts AST nodes in the parsed filter. Proxies JIT compile cost: each
    /// node turns into one or more Cranelift IR instructions during codegen.
    /// Used by the binary's null-input heuristic — see `src/bin/jq-jit.rs`.
    pub fn ast_node_count(&self) -> usize {
        use crate::ir::{Expr, StringPart};
        fn count(e: &Expr) -> usize {
            1 + match e {
                Expr::Pipe { left, right } | Expr::Comma { left, right }
                | Expr::BinOp { lhs: left, rhs: right, .. }
                | Expr::Alternative { primary: left, fallback: right } => count(left) + count(right),
                Expr::UnaryOp { operand, .. } | Expr::Negate { operand }
                | Expr::Collect { generator: operand } | Expr::Each { input_expr: operand }
                | Expr::EachOpt { input_expr: operand } | Expr::Recurse { input_expr: operand } => count(operand),
                Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => count(expr) + count(key),
                Expr::IfThenElse { cond, then_branch, else_branch } => count(cond) + count(then_branch) + count(else_branch),
                Expr::TryCatch { try_expr, catch_expr, .. } => count(try_expr) + count(catch_expr),
                Expr::Reduce { source, init, update, .. } => count(source) + count(init) + count(update),
                Expr::Foreach { source, init, update, extract, .. } => {
                    count(source) + count(init) + count(update)
                        + extract.as_ref().map_or(0, |e| count(e))
                }
                Expr::Slice { expr, from, to } => {
                    count(expr) + from.as_ref().map_or(0, |e| count(e)) + to.as_ref().map_or(0, |e| count(e))
                }
                Expr::ObjectConstruct { pairs } => pairs.iter().map(|(k, v)| count(k) + count(v)).sum(),
                Expr::LetBinding { value, body, .. } => count(value) + count(body),
                Expr::Label { body, .. } => count(body),
                Expr::Break { value, .. } => count(value),
                Expr::CallBuiltin { args, .. } | Expr::FuncCall { args, .. } => args.iter().map(count).sum(),
                Expr::Update { path_expr, update_expr } | Expr::Assign { path_expr, value_expr: update_expr } => {
                    count(path_expr) + count(update_expr)
                }
                Expr::Mutate { path_expr, value_expr, .. } => count(path_expr) + count(value_expr),
                Expr::ClosureOp { input_expr, key_expr, .. } => count(input_expr) + count(key_expr),
                Expr::Format { expr, .. } | Expr::PathExpr { expr } | Expr::Debug { expr } | Expr::Stderr { expr } => count(expr),
                Expr::Limit { count: c, generator } => count(c) + count(generator),
                Expr::While { cond, update, .. } | Expr::Until { cond, update } => count(cond) + count(update),
                Expr::Repeat { update, .. } => count(update),
                Expr::AllShort { generator, predicate } | Expr::AnyShort { generator, predicate } => count(generator) + count(predicate),
                Expr::Range { from, to, step } => {
                    count(from) + count(to) + step.as_ref().map_or(0, |s| count(s))
                }
                Expr::SetPath { path, value } => count(path) + count(value),
                Expr::GetPath { path } => count(path),
                Expr::DelPaths { paths } => count(paths),
                Expr::StringInterpolation { parts } => parts.iter().map(|p| match p {
                    StringPart::Literal(_) => 0,
                    StringPart::Expr(e) => count(e),
                }).sum(),
                Expr::Error { msg } => msg.as_ref().map_or(0, |e| count(e)),
                Expr::AlternativeDestructure { alternatives } => alternatives.iter().map(count).sum(),
                Expr::RegexTest { input_expr, re, flags, .. }
                | Expr::RegexMatch { input_expr, re, flags, .. }
                | Expr::RegexCapture { input_expr, re, flags, .. }
                | Expr::RegexScan { input_expr, re, flags, .. } => count(input_expr) + count(re) + count(flags),
                Expr::RegexSub { input_expr, re, tostr, flags }
                | Expr::RegexGsub { input_expr, re, tostr, flags } => count(input_expr) + count(re) + count(tostr) + count(flags),
                _ => 0,
            }
        }
        // Include def bodies — the JIT inlines them, so they contribute to
        // codegen size as much as the top-level expression (#658 follow-up).
        count(&self.parsed.0) + self.parsed.1.iter().map(|f| count(&f.body)).sum::<usize>()
    }

    /// Execute the filter against an input value, collecting all results.
    pub fn execute(&self, input: &Value) -> Result<Vec<Value>> {
        let mode = forced_mode();
        let forced = mode & FORCE_EVAL_BIT != 0;
        let forced_jitop = mode & FORCE_JITOP_BIT != 0;
        let forced_cranelift = mode & FORCE_CRANELIFT_BIT != 0;
        // Typed fast path (issue #83): probed ahead of JIT / eval. Only
        // migrated filter shapes return `Some`; every other shape or
        // unhandled input type returns `None` and falls through to the
        // authoritative generic path below.
        if !forced && !forced_jitop && !forced_cranelift {
            if let Some(verdict) = self.try_typed_fast_path(input) {
                return verdict.map(|v| vec![v]);
            }
        }

        // JitOp interpreter backend (#1059). The program is compiled either
        // by the forced-mode knob (Phase 1 self-diff) or by the default
        // sub-threshold routing (Phase 2): whenever the Cranelift heuristics
        // decline to codegen, the binary builds the JitOp program instead and
        // execution lands here rather than on the tree-walking eval below.
        // Only `JQJIT_FORCE_INTERPRETER` skips it (that knob pins eval).
        if !forced {
            if let Some(ref prog) = self.jit_program {
                return crate::jit::execute_program(prog, input);
            }
        }

        // Try JIT execution first
        if !forced && !forced_jitop {
            if let Some(jit_fn) = self.jit_fn {
                return crate::jit::execute_jit(jit_fn, input);
            }
        }

        let (ref expr, ref funcs) = self.parsed;
        // Mirror execute_cb's env setup so memoize slots are wired up.
        let env = {
            let mut cached = self.cached_env.borrow_mut();
            if let Some(ref env) = *cached {
                env.clone()
            } else {
                let mut e = crate::eval::Env::with_lib_dirs(funcs.clone(), self.lib_dirs.clone());
                if self.memo_slots > 0 {
                    e.set_memo_slots(self.memo_slots);
                    e.set_memo_max_entries(self.memo_max_entries);
                }
                let env = std::rc::Rc::new(std::cell::RefCell::new(e));
                *cached = Some(env.clone());
                env
            }
        };
        let mut outputs = Vec::new();
        crate::eval::execute_ir_with_env_cb(expr, input.clone(), &env, &mut |v| {
            outputs.push(v);
            Ok(true)
        })?;
        Ok(outputs)
    }

    /// Execute the filter with a callback for each result (avoids Vec allocation).
    /// Returns Ok(true) if all values were processed, Ok(false) if stopped early.
    pub fn execute_cb(&self, input: &Value, cb: &mut dyn FnMut(&Value) -> Result<bool>) -> Result<bool> {
        let mode = forced_mode();
        let forced = mode & FORCE_EVAL_BIT != 0;
        let forced_jitop = mode & FORCE_JITOP_BIT != 0;
        let forced_cranelift = mode & FORCE_CRANELIFT_BIT != 0;
        // Typed fast path (issue #83) — see `execute` for rationale. The
        // current pilot emits a single value, so hitting it closes out the
        // generator: we invoke `cb` once with the verdict and return its
        // continue/stop decision to the caller.
        if !forced && !forced_jitop && !forced_cranelift {
            if let Some(verdict) = self.try_typed_fast_path(input) {
                let val = verdict?;
                return cb(&val);
            }
        }

        // JitOp interpreter backend (#1059) — see `execute` for rationale.
        if !forced {
            if let Some(ref prog) = self.jit_program {
                return crate::jit::execute_program_cb(prog, input, cb);
            }
        }

        if !forced && !forced_jitop {
            if let Some(jit_fn) = self.jit_fn {
                return crate::jit::execute_jit_cb(jit_fn, input, cb);
            }
        }

        let (ref expr, ref funcs) = self.parsed;
        // Use cached env to avoid re-allocation per call
        let env = {
            let mut cached = self.cached_env.borrow_mut();
            if let Some(ref env) = *cached {
                env.clone()
            } else {
                let mut e = crate::eval::Env::with_lib_dirs(funcs.clone(), self.lib_dirs.clone());
                if self.memo_slots > 0 {
                    e.set_memo_slots(self.memo_slots);
                    e.set_memo_max_entries(self.memo_max_entries);
                }
                let env = std::rc::Rc::new(std::cell::RefCell::new(e));
                *cached = Some(env.clone());
                env
            }
        };
        crate::eval::execute_ir_with_env_cb(
            expr, input.clone(), &env,
            &mut |val| cb(&val),
        )
    }
}
