# jq-jit

A JIT-compiling [jq](https://jqlang.github.io/jq/) in Rust using [Cranelift](https://cranelift.dev/) — **up to ~100x faster**, passing 100% of the official jq test suite (509/509).

> **The author can't read or write Rust** — essentially every line was written by an AI. See [How this was built](#how-this-was-built).

## Features

- **Targets full jq language compatibility** — passes the official jq test suite in full (509/509); any remaining divergence from jq is treated as a bug
- **JIT compilation** via Cranelift for hot execution paths
- **Raw byte fast paths** — 70+ filter shapes operate directly on raw bytes, skipping JSON parsing entirely; 180+ shapes in total route to specialized fast paths
- **Streaming JSON parser** for memory-efficient NDJSON processing
- **Memory-mapped file I/O** — mmap-based file reading with no upfront allocation
- **Optimized value representation** with compact strings, mimalloc, and inline Cranelift codegen
- **jqx extensions** — shell command execution (`exec`/`execv`), CSV/TSV parsing (`fromcsv`/`fromcsvh`/`fromtsv`/`fromtsvh`), function-result memoization (`memoize`), and the in-place path-update contract (`mutate`)

## Performance

On a 2M-line NDJSON file (typical ETL/data pipeline workload):

| Filter | jq-jit | jq | jaq | Speedup vs jq |
|--------|-------:|---:|----:|--------------:|
| `empty` | 0.02s | 1.38s | 0.91s | **64x** |
| `.name` (field access) | 0.12s | 2.08s | 1.76s | **18x** |
| `select(.x > 1500000)` | 0.09s | 2.24s | 1.81s | **25x** |
| `.x + .y` (arithmetic) | 0.09s | 2.21s | 2.18s | **24x** |
| `type` | 0.02s | 1.83s | 4.06s | **75x** |
| `to_entries` | 0.15s | 7.81s | 6.74s | **51x** |
| `keys` | 0.11s | 2.50s | 1.85s | **23x** |
| `.name \| gsub("_"; "-")` | 0.18s | 16.75s | 11.84s | **96x** |
| `walk(if type == "number" then . + 1 else . end)` | 0.15s | 16.38s | 13.27s | **109x** |

Measured on an Apple M4 Max (macOS 26.4.1, arm64), jq 1.7.1 and
[jaq](https://github.com/01mf02/jaq) 3.0.0, best of 3 runs over 2M NDJSON
objects. Speedups are machine-dependent — `jaq` is itself a fast Rust
reimplementation of jq, and jq-jit stays 15–167x ahead of it across these
filters. For per-version results across the full filter suite, see
[`docs/benchmark-history.md`](docs/benchmark-history.md). Run
`bash bench/comprehensive.sh` to reproduce on your own hardware.

## How this was built

**The author cannot read or write Rust. The entire architecture and essentially all of the code were written by [Claude](https://claude.ai/) (Anthropic) via [Claude Code](https://docs.anthropic.com/en/docs/claude-code)** — the libjq-bytecode backend, the choice of Cranelift, every optimization. The author made no code or design decisions; aside from a couple of small community fixes, every line is Claude's.

The human's role was twofold: setting the goal, and — once the tool first reached a working state — keeping a bug-hunting loop running with high-level directives ("go find bugs and fix them," "keep going"). Claude did the work on both ends: discovering the bugs (building its own test infrastructure, using Project Euler problems as a detector) and fixing them. The human never read the implementation and never pointed to a specific bug. How far that goes is the experiment: a real-world, performance-critical tool maintained for months by someone who never sees the code. See [`docs/development-story.md`](docs/development-story.md) for the full account, including the (small) number of messages the human actually sent.

## Installation

### Homebrew (macOS arm64, Linux x86_64)

Pre-built binaries from the [latest release](https://github.com/m5d215/jq-jit/releases/latest) are available via a personal tap:

```bash
brew install m5d215/tap/jq-jit
```

The Homebrew tap covers macOS arm64 and Linux x86_64. Windows x86_64 ships as a `.zip` on the releases page (see below); for any other platform, build from source.

### Prebuilt binaries (manual)

Download the archive for your platform from the [releases page](https://github.com/m5d215/jq-jit/releases/latest) and extract the binary onto your `PATH`. macOS / Linux ship as `.tar.gz`, Windows as `.zip` (`jq-jit.exe`).

## Building

### Prerequisites

- Rust toolchain (edition 2021)

jq-jit has no runtime C dependencies: parsing, evaluation, and JIT codegen
are all pure Rust. (Earlier versions linked against `libjq` and `libonig`;
those dependencies were removed as of 1.3.0.)

### Build

```bash
cargo build --release
```

The binary is output to `target/release/jq-jit`.

## Usage

```bash
jq-jit [OPTIONS] <FILTER> [FILE...]
```

### Options

| Flag | Description |
|------|-------------|
| `-c`, `--compact-output` | Compact JSON output |
| `-r`, `--raw-output` | Output strings without quotes |
| `--raw-output0` | Like `-r`, but terminate each output with NUL instead of newline |
| `-j`, `--join-output` | No newline after each output |
| `-R`, `--raw-input` | Treat each input line as a string |
| `-n`, `--null-input` | Use `null` as input |
| `-s`, `--slurp` | Collect all inputs into an array |
| `-S`, `--sort-keys` | Sort object keys |
| `-e`, `--exit-status` | Exit with 5 if last output is `false`/`null` |
| `-C`, `--color-output` | Force ANSI color output |
| `-M`, `--monochrome-output` | Disable color output |
| `--tab` | Use tabs for indentation |
| `--indent N` | Use N spaces for indentation (range -1..=7; -1 means tab; default: 2) |
| `--unbuffered` | Flush output after each value |
| `--seq` | Frame each output with `RS` (0x1E) per RFC 7464 |
| `-f`, `--from-file FILE` | Read filter from file |
| `-L`, `--library-path DIR` | Add DIR to the module search path (repeatable) |
| `--arg NAME VALUE` | Set `$NAME` to string VALUE |
| `--argjson NAME VALUE` | Set `$NAME` to JSON VALUE |
| `--slurpfile NAME FILE` | Set `$NAME` to array of JSON values from FILE |
| `--rawfile NAME FILE` | Set `$NAME` to string contents of FILE |
| `--args` | Remaining arguments are string `$ARGS.positional` |
| `--jsonargs` | Remaining arguments are JSON `$ARGS.positional` |
| `-V`, `--version` | Print version and exit |
| `-h`, `--help` | Print usage and exit |

`-a` / `--ascii-output` and `--stream` are accepted but not yet
implemented (tracked in #126); jq-jit exits with an explicit error rather
than falling through silently.

### Examples

```bash
# Identity
echo '{"name": "jq"}' | jq-jit '.'

# Field access
echo '{"name": "jq", "version": 1}' | jq-jit '.name'

# Process NDJSON file
jq-jit 'select(.age > 30)' data.jsonl

# Multiple filters
echo '[1,2,3]' | jq-jit 'map(. * 2)'

# Using variables
jq-jit --arg name "test" '.[$name]' data.json

# Positional arguments
jq-jit -n '$ARGS.positional' --args foo bar baz
```

## Extensions (jqx)

jq-jit ships with a small set of extensions beyond standard jq, collectively
referred to as **jqx**. They are part of the default build on `main` (no
separate branch or feature flag).

### Shell Command Execution

| Function | Description |
|----------|-------------|
| `exec("cmd")` | Execute command, return stdout. Non-zero exit raises a catchable error. |
| `exec(generator; "cmd")` | Pipe generator outputs to a single process's stdin, yield each stdout line. |
| `execv("cmd")` | Execute command, return `{exitcode, stdout, stderr}` object. |

When input is non-null, it is passed to the command's stdin (strings as-is, other values JSON-encoded). Trailing newlines are trimmed.

```bash
# Run a command and use its output
jq-jit -n 'exec("git rev-parse @") | "commit: \(.[:7])"'

# Pipe input to a command
echo '"hello"' | jq-jit 'exec("tr a-z A-Z")'  # → "HELLO"

# Stream a generator through a single process
jq-jit -n 'exec(range(0;10); "sed s/^/+/")'

# Capture exit code and stderr
jq-jit -n 'execv("ls /nope") | if .exitcode != 0 then .stderr else .stdout end'
```

### CSV/TSV Parsing

| Function | Description |
|----------|-------------|
| `fromcsv` | Parse CSV, yield `["field1","field2",...]` per row. |
| `fromcsvh` | Parse CSV with first row as headers, yield `{"col":"val",...}` per row. |
| `fromcsvh(["col1","col2"])` | Parse CSV with specified headers, yield objects per row. |
| `fromtsv` / `fromtsvh` / `fromtsvh(headers)` | Same as above, tab-delimited. |

All values are returned as strings. Parsing is RFC 4180 compliant (handles quoted fields, escaped quotes, commas/newlines within fields).

```bash
# Parse CSV into arrays
echo 'name,age\nAlice,30' | jq-jit -R 'fromcsv'

# Parse CSV file with headers into objects
jq-jit -Rsc 'fromcsvh' < data.csv

# Use custom headers
jq-jit -Rsc 'fromcsvh(["name","age"])' < no-header.csv

# Combine with exec
jq-jit -n 'exec("cat data.csv") | fromcsvh | select(.age | tonumber > 25)'
```

### Mutate

| Function | Description |
|----------|-------------|
| `mutate(path = v)` / `mutate(path \|= f)` / `mutate(path += v)` (and `-=`, `*=`, `/=`, `%=`, `//=`) | Contract that the wrapped path-update runs on the in-place fast path when the container's refcount allows. Semantically identical to the unmarked form. |

The wrapped expression must be exactly one path-update operator at the leaf —
composite forms (`if`/`then`/`else`, `reduce`, `,`, `|`) must distribute the
marker inward by wrapping each leaf separately. Anything else is a parse
error.

```bash
# Reduce accumulator stays in-place across iterations
jq-jit -n '[range(0; 1000) | 0] | reduce range(0; 1000) as $i (.; mutate(.[$i] = $i*$i))'

# Operator-assign forms work the same
echo '{"n":0}' | jq-jit 'mutate(.n += 1)'
```

`mutate(...)` is a contract, not a performance switch. The runtime check is
strict: refcount = 1 → in-place mutation, refcount > 1 → silent copy-on-write
fallback. The result is always equal to the unmarked form.

Since v1.8.0, the JIT's static escape analysis ([#697](https://github.com/m5d215/jq-jit/pull/697),
[#699](https://github.com/m5d215/jq-jit/pull/699),
[#700](https://github.com/m5d215/jq-jit/pull/700)) engages the in-place fast
path on the unmarked form for most shapes encountered in practice — including
the original P126 / P135 motivating cases. In those shapes, `mutate(...)`
and the unmarked form produce identical wall-clock times. The contract
remains useful as a documentation marker, a regression guard against future
JIT changes, and an observability anchor for `--trace-mutate`.

Use `--trace-mutate` to verify per-invocation whether the in-place path
engaged:

```bash
jq-jit --trace-mutate -n '[1,2,3] | mutate(.[0] = 99)'
# stderr: [trace:mutate] kind=assign container=array refcount=1 mode=in_place
```

As of v1.8.0, trace events are only emitted through the interpreter path
(e.g. `--force-interp`); JIT-mode tracing is tracked in
[#702](https://github.com/m5d215/jq-jit/issues/702).

`mutate` is a jq-jit extension; stock jq does not provide it. See issue
[#666](https://github.com/m5d215/jq-jit/issues/666) for the motivating
benchmarks and the persistent-vector follow-up tracked in
[#695](https://github.com/m5d215/jq-jit/issues/695).

### Memoization

| Function | Description |
|----------|-------------|
| `memoize(f)` | Cache the output sequence of `f` keyed by the current input value. |
| `memoize(f; key)` | Same, but use `key` (evaluated against the input) as the cache key instead of the input itself. |

Each lexical occurrence of `memoize(...)` gets its own cache; entries persist
for the lifetime of the program (across NDJSON input records). Keys compare by
structural equality, matching jq's `==` semantics (objects are order-independent,
arrays element-wise). The body is run to completion on first call — multi-output
generators are materialized so subsequent calls re-yield the same sequence.

Self-recursive memoization Just Works: jq's `def` binds the same name inside
the body, so recursive calls go through the memoized wrapper:

```bash
# Fibonacci — exponential without memo, linear with it
jq-jit -n 'def fib: memoize(if . < 2 then . else ((. - 1) | fib) + ((. - 2) | fib) end); 80 | fib'

# Collatz chain length — subgraph revisits become O(1)
jq-jit -n 'def collatz: memoize(if . == 1 then 0 else (if . % 2 == 0 then ./2 else 3*. + 1 end | collatz) + 1 end); 27 | collatz'

# 2-arg form: memoize a transformation by record id, ignoring the rest
jq-jit -c 'memoize(.value * 2; .id)' <<< '{"id":1,"value":10}'
```

Eviction defaults to "unbounded for program lifetime" up to a per-slot cap of
1,000,000 entries; control it with `--memo-max-entries N` on the CLI. Past the
cap, new inserts are silently dropped (the program continues, just without
caching new entries). Body errors do not poison the cache — the next call
re-evaluates.

Run with `--debug-memo` to print per-slot cache stats (hits / misses / entries)
to stderr at program exit. Useful for confirming a `memoize(...)` is actually
hitting:

```bash
jq-jit --debug-memo -n 'def fib: memoize(if . < 2 then . else ((. - 1) | fib) + ((. - 2) | fib) end); 30 | fib'
# memoize stats: 1 slot(s)
#   slot          hits        misses       entries
#      0            28            31            31
```

The 1-arg form keys only by the current input. If your body closes over a
`$var` that varies between calls, results will be stale — pull the var into
the key explicitly with the 2-arg form: `memoize(. + $x; [., $x])`.

## Testing

The official jq suite is the floor, not the ceiling — most bugs that mattered
were ones it never exercised. jq-jit is checked by several independent layers,
all run by a single `cargo test --release`:

| Layer | Cases | What it checks |
|-------|------:|----------------|
| Official jq test suite | 509 | Verbatim upstream `jq.test` — 100% pass |
| Regression corpus | 2,400+ | Issue-driven cases pinned from real bugs found during development |
| Differential vs jq | 1,200+ | Output compared against a reference `jq` build, case by case |
| Fast-path contracts | 400+ | Each raw-byte / in-place fast path must match its slow-path semantics |
| Self-differential | — | JIT output must equal the interpreter's on the same program |
| Metamorphic · fuzz · invariant enforcement | — | Generated programs, fuzzed inputs across multiple axes, and internal architectural contracts |

Thousands of corpus cases plus hundreds of contract and unit tests. See
[`docs/development-story.md`](docs/development-story.md) for how this test
infrastructure was built — and why passing 509/509 turned out to be the
beginning, not the end.

## Benchmarks

Measure jq-jit on your own hardware:

```bash
bash bench/comprehensive.sh
```

This generates 2M NDJSON objects and measures performance across ~250
filter patterns spanning NDJSON, generators, reduce/foreach, regex,
type conversion, and an external jaq filter corpus.

Set `JQ_JIT` to point at a different binary — e.g. `JQ_JIT=$(which jq)
bash bench/comprehensive.sh` for a sanity comparison against the
reference jq build.

## License

jq-jit's own source code is licensed under either of

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.

### Third-party components

When distributed as a compiled binary, jq-jit includes third-party Rust
code — most notably [Cranelift](https://cranelift.dev/) (Apache-2.0 WITH
LLVM-exception) — whose own license terms must be preserved. Because
several required dependencies (Cranelift, `ryu`, and others) do not offer
an MIT option, **a binary distribution in practice must comply with
Apache-2.0 terms for those components, regardless of which option a user
selects for jq-jit's own code**.

See [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md) for the full
attribution listing.
