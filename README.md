# jq-jit

A JIT-compiling implementation of [jq](https://jqlang.github.io/jq/) using [Cranelift](https://cranelift.dev/).

Passes 100% of the official jq test suite (509/509) while being significantly faster than jq for large inputs.

**This entire project — architecture, implementation, debugging, optimization — was autonomously developed by [Claude](https://claude.ai/) (Anthropic) via [Claude Code](https://docs.anthropic.com/en/docs/claude-code). No human-written code.** The goal was to see how far an AI can go building a real-world, performance-critical tool from scratch.

## Features

- Full jq language compatibility — drop-in replacement for `jq`
- JIT compilation via Cranelift for hot paths
- Streaming JSON parser for memory-efficient NDJSON processing
- Optimized value representation with compact strings and fast hashing

## Building

### Prerequisites

- Rust toolchain (edition 2021)
- `libjq` and `libonig` (development libraries)

On macOS (Homebrew):

```bash
brew install jq oniguruma
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
| `-R`, `--raw-input` | Treat each input line as a string |
| `-n`, `--null-input` | Use `null` as input |
| `-s`, `--slurp` | Collect all inputs into an array |
| `-S`, `--sort-keys` | Sort object keys |
| `-e`, `--exit-status` | Exit with 5 if last output is `false`/`null` |
| `--tab` | Use tabs for indentation |
| `--indent N` | Use N spaces for indentation (default: 2) |
| `--arg NAME VALUE` | Set `$NAME` to string VALUE |
| `--argjson NAME VALUE` | Set `$NAME` to JSON VALUE |
| `--args` | Remaining arguments are positional args |

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
```

## Testing

Run the official jq test suite (509/509 passing):

```bash
cargo test --release
```

## Benchmarks

Run benchmarks comparing against `jq`, `gojq`, and `jaq`:

```bash
bash bench/run.sh
```

This generates 2M NDJSON objects and measures performance across several filter patterns (identity, field access, arithmetic, select, string concatenation, object construction).

## License

Licensed under either of

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
