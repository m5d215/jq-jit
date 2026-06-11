//! Default-dispatch routing guard for #1059 Phase 2: filters that the
//! Cranelift heuristics decline (sub-threshold input) must run on the JitOp
//! interpreter when they flatten fully, fall back to tree-walking eval when
//! the flattener bails, and keep Cranelift under `--force-jit`. The probe is
//! the `JQJIT_TRACE=1` generic-fallback label (`jit` / `jitop` / `eval`),
//! which reflects exactly the compile decision made in `src/bin/jq-jit.rs`.

use std::process::Command;

/// Run the binary on a one-line stdin input and return the traced
/// generic-fallback label (the `matched=` name of the first trace line).
fn traced_label(filter: &str, input: &str, extra_arg: Option<&str>) -> String {
    let bin = env!("CARGO_BIN_EXE_jq-jit");
    let mut cmd = Command::new(bin);
    if let Some(arg) = extra_arg {
        cmd.arg(arg);
    }
    cmd.arg("-c").arg(filter);
    cmd.env("JQJIT_TRACE", "1");
    cmd.env_remove("JQJIT_FORCE_INTERPRETER");
    cmd.env_remove("JQJIT_FORCE_JITOP_INTERP");
    cmd.env_remove("JQJIT_FORCE_CRANELIFT");
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn jq-jit");
    {
        use std::io::Write;
        let mut stdin = child.stdin.take().expect("stdin");
        let _ = stdin.write_all(input.as_bytes());
        let _ = stdin.write_all(b"\n");
    }
    let out = child.wait_with_output().expect("wait jq-jit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    for line in stderr.lines() {
        if let Some((_, name)) = line.rsplit_once("matched=") {
            return name.trim().to_string();
        }
    }
    panic!("no trace line emitted for filter {:?}; stderr: {}", filter, stderr);
}

/// A flattenable filter on sub-threshold input lands on the JitOp
/// interpreter, not tree-walking eval. The filter shape is chosen to miss
/// every raw-byte / typed fast path so the generic-fallback label is the
/// one traced.
#[test]
fn sub_threshold_flattenable_routes_to_jitop() {
    let label = traced_label(
        "(.a.b + 1) as $x | {r: $x, n: (.c | length)}",
        r#"{"a":{"b":3},"c":[1,2]}"#,
        None,
    );
    assert_eq!(label, "jitop");
}

/// A filter the flattener rejects keeps the eval fallback.
#[test]
fn non_flattenable_falls_back_to_eval() {
    let label = traced_label("memoize(.a)", r#"{"a":1}"#, None);
    assert_eq!(label, "eval");
}

/// The routing gate keeps Value-materializing loop bodies on eval: the
/// expansion shape `[range(.) | f] | ...` flattens fine (forced-mode
/// self-diff still runs it) but measures 1.4-2.6x slower on the
/// interpreter than on eval's fused builtins, so default dispatch must
/// not route it.
#[test]
fn value_materializing_loop_stays_on_eval() {
    let label = traced_label("[range(.) | . % 1000] | unique | length", "1000", None);
    assert_eq!(label, "eval");
}

/// Loops that stay on unboxed f64 variables pass the gate: the
/// constant-range reduce class (kept on eval by `has_loop_constructs`'
/// input-referencing-source rule, so it reaches the JitOp catch-all)
/// measures 1.6-1.8x faster on the interpreter than on eval.
#[test]
fn var_only_reduce_loop_routes_to_jitop() {
    let label = traced_label("reduce range(0; 1000) as $i (0; . + $i)", "null", None);
    assert_eq!(label, "jitop");
}

/// Fused `[range(n)]` collect is a single runtime call, not a loop span,
/// so a straight-line program around it passes the gate too.
#[test]
fn straight_line_collect_range_routes_to_jitop() {
    let label = traced_label("[range(.)] | length", "1000", None);
    assert_eq!(label, "jitop");
}

/// Programs that lower with a streaming eval delegate (#1059 Phase 3)
/// stay on whole-filter eval in default dispatch: per-record delegation
/// measures slower than eval when the delegate dominates. The forced-mode
/// knobs compile them (tests/selfdiff_jitop_backend.rs covers that side).
#[test]
fn delegated_program_stays_on_eval_by_default() {
    let label = traced_label("[path(.a // .b)]", r#"{"b":1}"#, None);
    assert_eq!(label, "eval");
}

/// `--force-jit` accepts delegated programs (debug knob, full coverage).
#[test]
fn force_jit_compiles_delegated_program() {
    let label = traced_label("[path(.a // .b)]", r#"{"b":1}"#, Some("--force-jit"));
    assert_eq!(label, "jit");
}

/// `--force-jit` keeps its "Cranelift or eval" debug semantics: the JitOp
/// routing must not capture it.
#[test]
fn force_jit_pins_cranelift() {
    let label = traced_label(
        "(.a.b + 1) as $x | {r: $x, n: (.c | length)}",
        r#"{"a":{"b":3},"c":[1,2]}"#,
        Some("--force-jit"),
    );
    assert_eq!(label, "jit");
}

/// A recursive def now lowers via the eval delegate (#1059 Phase 3c), but
/// delegated programs stay off the default dispatch — whole-filter eval
/// measures faster when the delegate dominates.
#[test]
fn recursive_def_stays_on_eval_by_default() {
    let label = traced_label(
        "def f: if . >= 5 then . else . + 1 | f end; . | f",
        "0",
        None,
    );
    assert_eq!(label, "eval");
}

/// `--force-jit` compiles the delegated recursive def (debug knob, full
/// coverage for the backend self-diff).
#[test]
fn force_jit_compiles_recursive_def_delegate() {
    let label = traced_label(
        "def f: if . >= 5 then . else . + 1 | f end; . | f",
        "0",
        Some("--force-jit"),
    );
    assert_eq!(label, "jit");
}
