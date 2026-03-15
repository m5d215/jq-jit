# jq-jit

A JIT-compiling implementation of [jq](https://jqlang.github.io/jq/) using [Cranelift](https://cranelift.dev/).

Passes 100% of the official jq test suite (509/509) while being **8x-180x faster** than jq for NDJSON workloads.

**This entire project — architecture, implementation, debugging, optimization — was autonomously developed by [Claude](https://claude.ai/) (Anthropic) via [Claude Code](https://docs.anthropic.com/en/docs/claude-code). No human-written code.** The goal was to see how far an AI can go building a real-world, performance-critical tool from scratch.

## Features

- **Full jq language compatibility** — drop-in replacement for `jq` (509/509 official tests)
- **JIT compilation** via Cranelift for hot execution paths
- **Raw byte fast paths** — 100+ filter patterns bypass JSON parsing entirely for maximum throughput
- **Streaming JSON parser** for memory-efficient NDJSON processing
- **Memory-mapped file I/O** — mmap-based file reading with no upfront allocation
- **Optimized value representation** with compact strings, mimalloc, and inline Cranelift codegen

## Performance

On a 2M-line NDJSON file (typical ETL/data pipeline workload):

| Filter | jq-jit | jq | Speedup |
|--------|--------|----|---------|
| `empty` | 0.01s | 0.85s | **85x** |
| `.name` (field access) | 0.05s | 4.83s | **97x** |
| `select(.x > N)` | 0.04s | 3.51s | **88x** |
| `.x + .y` (arithmetic) | 0.06s | 5.73s | **96x** |
| `type` | 0.01s | 1.26s | **126x** |
| `to_entries` | 0.11s | 8.03s | **73x** |
| `keys` | 0.12s | 4.69s | **39x** |
| `.name \| gsub("_"; "-")` | 0.31s | 28.5s | **92x** |
| `walk(if type == "number" then . + 1 else . end)` | 0.40s | 10.2s | **26x** |

Run `bash bench/run.sh` to benchmark on your machine.

## Building

### Prerequisites

- Rust toolchain (edition 2021)
- `libjq` (≥ 1.7) and `libonig` (development libraries)

On macOS (Homebrew):

```bash
brew install jq oniguruma
```

On Ubuntu/Debian:

```bash
sudo apt-get install libonig-dev
# For jq 1.8+, build from source:
git clone --depth 1 --branch jq-1.8.1 https://github.com/jqlang/jq.git /tmp/jq
cd /tmp/jq && git submodule update --init && autoreconf -i
./configure --prefix=/usr && make -j$(nproc) && sudo make install
```

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
| `-j`, `--join-output` | No newline after each output |
| `-R`, `--raw-input` | Treat each input line as a string |
| `-n`, `--null-input` | Use `null` as input |
| `-s`, `--slurp` | Collect all inputs into an array |
| `-S`, `--sort-keys` | Sort object keys |
| `-e`, `--exit-status` | Exit with 5 if last output is `false`/`null` |
| `-f`, `--from-file FILE` | Read filter from file |
| `--tab` | Use tabs for indentation |
| `--indent N` | Use N spaces for indentation (default: 2) |
| `--arg NAME VALUE` | Set `$NAME` to string VALUE |
| `--argjson NAME VALUE` | Set `$NAME` to JSON VALUE |
| `--slurpfile NAME FILE` | Set `$NAME` to array of JSON values from FILE |
| `--rawfile NAME FILE` | Set `$NAME` to string contents of FILE |
| `--args` | Remaining arguments are string `$ARGS.positional` |
| `--jsonargs` | Remaining arguments are JSON `$ARGS.positional` |

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

## Testing

Run the official jq test suite (509/509 passing):

```bash
cargo test --release -- --test-threads=1
```

## Benchmarks

Run benchmarks comparing against `jq`, `gojq`, and `jaq`:

```bash
bash bench/run.sh
```

This generates 2M NDJSON objects and measures performance across 100+ filter patterns (identity, field access, arithmetic, select, string operations, regex, object construction, and more).

## License

Licensed under either of

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
