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
/// stay on whole-filter eval in default dispatch when the delegate
/// *dominates* the program (few native ops — per-record delegation
/// measures ~1.1x over eval, #1059 Phase 3.9). The forced-mode knobs
/// compile them (tests/selfdiff_jitop_backend.rs covers that side).
#[test]
fn delegated_program_stays_on_eval_by_default() {
    let label = traced_label("[path(.a // .b)]", r#"{"b":1}"#, None);
    assert_eq!(label, "eval");
}

/// A *mixed* program (real native work plus a delegated subtree) routes
/// to the shared lowering by default since the #1059 Phase 3.9 flip —
/// the native part runs at JIT speed and measures faster than
/// whole-filter eval.
#[test]
fn mixed_delegated_program_routes_by_default() {
    let label = traced_label(
        "select(.x > 100) | path(. as [$a] | $a)",
        r#"{"x":500}"#,
        None,
    );
    assert_eq!(label, "jitop");
}

/// Programs observing per-read input state lower as a whole-program
/// eval delegate (see `observes_interleaved_input_state` in src/jit.rs),
/// which makes them delegate-dominant — default routing keeps them on
/// whole-filter eval.
#[test]
fn interleaved_input_state_stays_on_eval() {
    let label = traced_label("[inputs as $x | input_line_number]", "1", None);
    assert_eq!(label, "eval");
}

/// Run the binary under one forced-backend env knob with `-n -c` and
/// multi-line stdin, returning stdout.
fn forced_output(knob: &str, filter: &str, stdin_data: &str) -> String {
    let bin = env!("CARGO_BIN_EXE_jq-jit");
    let mut cmd = Command::new(bin);
    cmd.arg("-n").arg("-c").arg(filter);
    cmd.env(knob, "1");
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn jq-jit");
    {
        use std::io::Write;
        let mut stdin = child.stdin.take().expect("stdin");
        let _ = stdin.write_all(stdin_data.as_bytes());
    }
    let out = child.wait_with_output().expect("wait jq-jit");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// Forced-mode correctness for interleaved input state: such programs
/// lower as a single whole-program `DelegateGen`, preserving eval's lazy
/// read interleaving. The eager native lowering used to consume the
/// whole stream before the binding body ran, reporting post-consumption
/// line numbers ([3,3,3] instead of [1,2,3]).
#[test]
fn forced_modes_interleave_input_state_lazily() {
    for knob in ["JQJIT_FORCE_JITOP_INTERP", "JQJIT_FORCE_CRANELIFT"] {
        let out = forced_output(knob, "[inputs as $x | input_line_number]", "1\n2\n3\n");
        assert_eq!(out, "[1,2,3]", "{knob}");
        let out = forced_output(
            knob,
            "[input, input_line_number, input, input_line_number]",
            "10\n20\n30\n",
        );
        assert_eq!(out, "[10,1,20,2]", "{knob}");
    }
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

/// A recursive def now lowers via the eval delegate (#1059 Phase 3c).
/// This one is delegate-dominant (the call IS the program), so default
/// dispatch keeps whole-filter eval, which measures faster.
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
