//! Issue #1133: `--unbuffered` JSON input must flush each result as it arrives
//! on a continuous stream, instead of blocking on EOF. jq-jit previously
//! pre-read all of stdin (`read_to_string`) to size the JIT decision, so an
//! infinite stream (`pw-dump -m`, `tail -f`, a `while … echo` loop) produced no
//! output at all — the pre-read never returned.
//!
//! These cases shell out: a stream that emits, pauses, then emits again cannot
//! be expressed in the single-input regression harness.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

fn spawn(args: &[&str]) -> Child {
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    Command::new(jq_jit)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn jq-jit")
}

/// Drain the child's stdout line by line over a channel from a persistent
/// reader thread, so the test can pull lines with a timeout without ever losing
/// the handle (a line that arrives before stdin is closed proves the stream is
/// not buffered until EOF).
fn line_reader(stdout: ChildStdout) -> Receiver<String> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                break; // EOF
            }
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    rx
}

/// Next line within `timeout`, trimmed; None means nothing arrived (buffered).
fn next_line(rx: &Receiver<String>, timeout: Duration) -> Option<String> {
    rx.recv_timeout(timeout).ok().map(|l| l.trim().to_string())
}

#[test]
fn flushes_each_value_before_eof() {
    let mut child = spawn(&["-c", "--unbuffered", "."]);
    let mut stdin = child.stdin.take().unwrap();
    let rx = line_reader(child.stdout.take().unwrap());

    // First value, then keep stdin open (no EOF).
    stdin.write_all(b"{\"a\":1}\n").unwrap();
    stdin.flush().unwrap();
    assert_eq!(
        next_line(&rx, Duration::from_secs(5)).as_deref(),
        Some("{\"a\":1}"),
        "first value must reach stdout while stdin is still open"
    );

    // Second value, still no EOF.
    stdin.write_all(b"{\"a\":2}\n").unwrap();
    stdin.flush().unwrap();
    assert_eq!(
        next_line(&rx, Duration::from_secs(5)).as_deref(),
        Some("{\"a\":2}")
    );

    drop(stdin); // EOF
    assert_eq!(child.wait().unwrap().code(), Some(0));
}

#[test]
fn value_split_across_writes_is_not_truncated() {
    // A number split mid-token across two writes must emit `1234`, never `12`.
    let mut child = spawn(&["-c", "--unbuffered", "."]);
    let mut stdin = child.stdin.take().unwrap();
    let rx = line_reader(child.stdout.take().unwrap());

    stdin.write_all(b"12").unwrap();
    stdin.flush().unwrap();
    // Nothing should surface yet — the token boundary is unconfirmed.
    assert_eq!(
        next_line(&rx, Duration::from_millis(500)),
        None,
        "a partial number must not be emitted before its boundary is known"
    );

    stdin.write_all(b"34\n").unwrap();
    stdin.flush().unwrap();
    assert_eq!(next_line(&rx, Duration::from_secs(5)).as_deref(), Some("1234"));

    drop(stdin);
    assert_eq!(child.wait().unwrap().code(), Some(0));
}

#[test]
fn multi_line_value_streams_when_complete() {
    let mut child = spawn(&["-c", "--unbuffered", "."]);
    let mut stdin = child.stdin.take().unwrap();
    let rx = line_reader(child.stdout.take().unwrap());

    // Object spread over several lines; only the closing line completes it.
    stdin.write_all(b"{\n  \"a\": 1,\n").unwrap();
    stdin.flush().unwrap();
    assert_eq!(
        next_line(&rx, Duration::from_millis(500)),
        None,
        "an incomplete multi-line value must not emit early"
    );

    stdin.write_all(b"  \"b\": 2\n}\n").unwrap();
    stdin.flush().unwrap();
    assert_eq!(
        next_line(&rx, Duration::from_secs(5)).as_deref(),
        Some("{\"a\":1,\"b\":2}")
    );

    drop(stdin);
    assert_eq!(child.wait().unwrap().code(), Some(0));
}

#[test]
fn parse_error_flushes_leading_then_exits_5() {
    // Matches the batch stream path (#856): valid leading documents reach
    // stdout, then a malformed token exits 5.
    let jq_jit = env!("CARGO_BIN_EXE_jq-jit");
    let mut child = Command::new(jq_jit)
        .args(["-c", "--unbuffered", "."])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"1 2 xx").unwrap();
    let out = child.wait_with_output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim_end(), "1\n2");
    assert_eq!(out.status.code(), Some(5));
}
