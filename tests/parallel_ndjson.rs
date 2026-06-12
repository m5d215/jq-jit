//! Issue #1054: `--parallel[=N]` shards a stateless filter over an NDJSON
//! document stream across a worker pool and reassembles the output in record
//! order. The core correctness contract is that the stdout bytes and the
//! process exit code are *identical* to sequential mode for any
//! parallel-safe filter. These tests pin that equivalence by running the
//! same input through the binary with and without `--parallel`, exercising
//! CLI/stdin behavior the 3-line regression harness cannot express.
//!
//! The corpus is deliberately >256 records (the inline-fallback threshold)
//! so the real worker-pool path engages, and mixes shapes that hit the
//! compact raw-passthrough decision (non-canonical numbers, duplicate keys,
//! escaped solidus) plus per-record-stateful-but-parallel-safe constructs
//! (reduce / foreach / limit / regex / string ops).

use std::io::Write;
use std::process::{Command, Stdio};

/// Run the binary with the given extra args, feeding `input` on stdin.
/// Returns (stdout bytes, exit code).
fn run(args: &[&str], input: &str) -> (Vec<u8>, Option<i32>) {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn jq-jit");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .expect("failed to write stdin");
    let out = child.wait_with_output().expect("failed to wait jq-jit");
    (out.stdout, out.status.code())
}

/// A varied NDJSON corpus of `n` records (n well above the 256 inline
/// threshold), cycling through shapes that stress the formatter.
fn corpus(n: usize) -> String {
    let mut s = String::new();
    for i in 0..n {
        let line = match i % 8 {
            0 => format!(r#"{{"id":{i},"name":"u{i}","vals":[{a},{b},{c}],"on":{on}}}"#,
                         a = i % 7, b = (i * 3) % 11, c = i % 2, on = i % 2 == 0),
            1 => format!(r#"{{"id":{i},"f":{i}.5e2,"path":"a\/b\/c"}}"#),
            2 => format!(r#"{{"id":{i},"dup":1,"dup":{i},"s":"tab\there"}}"#),
            3 => format!(r#"[{i},{},"x{i}"]"#, i * 2),
            4 => format!(r#"{{"id":{i},"big":1E+{}}}"#, i % 10),
            5 => format!(r#""string-{i}""#),
            6 => format!("{i}"),
            _ => format!(r#"{{"id":{i},"nested":{{"k":[{i}]}},"flag":{}}}"#, i % 3 == 0),
        };
        s.push_str(&line);
        s.push('\n');
    }
    s
}

/// Assert sequential and `--parallel` produce byte-identical stdout and the
/// same exit code for `filter` under the given base flags.
fn assert_parallel_matches(base: &[&str], filter: &str, input: &str) {
    let mut seq_args: Vec<&str> = base.to_vec();
    seq_args.push(filter);
    let (seq_out, seq_code) = run(&seq_args, input);

    let mut par_args: Vec<&str> = base.to_vec();
    par_args.push("--parallel=4");
    par_args.push(filter);
    let (par_out, par_code) = run(&par_args, input);

    assert_eq!(
        seq_code, par_code,
        "exit code differs for filter `{filter}` (base {base:?})"
    );
    assert!(
        seq_out == par_out,
        "stdout differs for filter `{filter}` (base {base:?})\n--- seq ({} bytes) ---\n{}\n--- par ({} bytes) ---\n{}",
        seq_out.len(),
        String::from_utf8_lossy(&seq_out),
        par_out.len(),
        String::from_utf8_lossy(&par_out),
    );
}

const COMPACT_FILTERS: &[&str] = &[
    ".",
    ".|.",
    ".id",
    "{wrap: .}",
    "if type == \"object\" then {id, name} else . end",
    "try (.vals | add) catch \"err\"",
    "try (.vals | map(. * 2)) catch empty",
    "[paths] | length",
    "reduce (.vals // [])[] as $v (0; . + $v)",
    "foreach (.vals // [])[] as $v (0; . + $v; .)",
    "[limit(2; (.vals // [])[])]",
    "try (.name | ascii_upcase | explode | reverse | implode) catch empty",
    "try (.path | gsub(\"/\"; \"-\")) catch empty",
    "try (.s | @base64) catch empty",
    "try (.f * 2) catch empty",
    "..",
    "tostring | length",
];

#[test]
fn parallel_matches_sequential_compact() {
    let input = corpus(3000);
    for f in COMPACT_FILTERS {
        assert_parallel_matches(&["-c"], f, &input);
    }
}

#[test]
fn parallel_matches_sequential_pretty() {
    let input = corpus(2000);
    // Pretty (no -c) goes through push_pretty_line; default and explicit indent.
    for f in &[".", "{wrap: .}", "try {id, name} catch ."] {
        assert_parallel_matches(&[], f, &input);
        assert_parallel_matches(&["--indent", "4"], f, &input);
    }
}

#[test]
fn parallel_matches_sequential_exit_status() {
    let input = corpus(1500);
    // -e: exit code reflects the last output value across the whole stream.
    for f in &[
        "try (.id) catch empty",
        "try (.on) catch true",
        "select(type == \"object\")",
        "empty",
    ] {
        assert_parallel_matches(&["-ce"], f, &input);
    }
}

#[test]
fn parallel_preserves_record_order() {
    // A monotonically increasing field must come out strictly in order even
    // though records are processed out of order across workers.
    let mut input = String::new();
    for i in 0..5000 {
        input.push_str(&format!("{{\"n\":{i}}}\n"));
    }
    let (out, code) = run(&["-c", "--parallel=4", ".n"], &input);
    assert_eq!(code, Some(0));
    let got = String::from_utf8(out).unwrap();
    let expected: String = (0..5000).map(|i| format!("{i}\n")).collect();
    assert_eq!(got, expected, "record order not preserved under --parallel");
}

#[test]
fn parallel_malformed_stream_flushes_leading_then_exits_5() {
    // jq emits each valid leading document's result, then exits 5 at the
    // malformed token (`jq: parse error ...`). The parallel path scans record
    // boundaries with json_stream_raw, so it reproduces exactly that: flush
    // the 1000 valid `.a` values, then exit 5. (Note: the *sequential* `.a`
    // projection fast path has a latent divergence here — it exits 0 and
    // over-emits — so this case is pinned against jq's documented behavior
    // directly rather than against sequential mode.)
    let mut input = String::new();
    for i in 0..1000 {
        input.push_str(&format!("{{\"a\":{i}}}\n"));
    }
    input.push_str("{bad\n"); // malformed tail
    let (out, code) = run(&["-c", "--parallel=4", ".a"], &input);
    assert_eq!(code, Some(5), "malformed stream must exit 5");
    let got = String::from_utf8(out).unwrap();
    let expected: String = (0..1000).map(|i| format!("{i}\n")).collect();
    assert_eq!(got, expected, "leading valid records must be flushed in order");
}

#[test]
fn parallel_unsafe_filters_fall_back_and_match() {
    // Filters rejected by is_parallel_safe must fall back to the sequential
    // path transparently — output and exit code identical to no `--parallel`.
    let input = "1\n2\n3\n4\n5\n";
    for f in &[
        ". , input",
        "[inputs] | length",
        "input_line_number",
        "$__loc__ | .file",
    ] {
        assert_parallel_matches(&["-c"], f, input);
    }
    // A `def` hides the stream read behind a call; the classifier must still
    // descend into the body and reject it.
    assert_parallel_matches(&["-c"], "def f: input; ., f", input);
}
