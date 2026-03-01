# jq-jit: JIT Compiler for jq Bytecode

jq のバイトコード VM をネイティブコードに JIT コンパイルし、高速実行するプロジェクト。

> **Note**: This project was generated 100% by AI (Claude) as a research experiment to explore the capabilities of AI-assisted systems programming. It is not intended for production use or ongoing maintenance.

## Performance

libjq (jq 1.8.1) との比較:

| カテゴリ | 例 | 高速化倍率 |
|---|---|---|
| スカラー演算 | `. + 1` | **19,124x** |
| 条件分岐 | `if . > 0 then ...` | **17,743x** |
| 文字列操作 | `ascii_downcase` | **5,306x** |
| ジェネレータ (100要素) | `.[] \| select(. > 50) \| . * 2` | **451x** |
| 集約 | `map(. * 2) \| add` (100要素) | **302x** |
| 大規模データ | `select(.age > 30)` (10K件) | **73x** |

> *Criterion マイクロベンチマーク（JIT 実行時間のみ）。JSON パースやプロセス起動を含む CLI wall-clock 比較は [docs/benchmark.md](docs/benchmark.md) を参照。*

CLI wall-clock (hyperfine):

| ワークロード | jq 比 |
|---|---|
| スカラー演算 (`echo 42 \| jq-jit '. + 1'`) | 1.0x (同等) |
| 大規模 select (10K records) | **1.7x 高速** |
| 大規模 select (100K records) | **1.3x 高速** |

### 実用ユースケース (hyperfine)

| ユースケース | Speedup |
|---|---|
| ログ分析 (100K NDJSON, select) | 1.2x |
| API レスポンス加工 (50K objects, group_by) | 1.4x |
| バッチ ETL (100 files, cache) | 1.15x |

詳細: [docs/usecase-benchmark.md](docs/usecase-benchmark.md)

## Usage

```bash
# Build
cargo build --release

# Run (jq-compatible CLI)
echo '{"name":"world"}' | ./target/release/jq-jit '"Hello, \(.name)!"'

# Examples
echo '[3,1,2]' | ./target/release/jq-jit 'sort | reverse'
echo '["hello","world"]' | ./target/release/jq-jit 'map(ascii_upcase) | join(" ")'
echo '{"a":1,"b":2}' | ./target/release/jq-jit 'to_entries | map(select(.value > 1)) | from_entries'
```

### AOT Compilation

Compile a filter to a standalone binary:

```bash
# Compile
./target/release/jq-jit --compile '.[] | select(. > 2)' -o filter

# Run (no jq-jit or jq needed at runtime)
echo '[1,2,3,4,5]' | ./filter
# Output:
# 3
# 4
# 5
```

### CLI Flags

| Flag | Description |
|---|---|
| `-r` | Raw string output (no quotes) |
| `-c` | Compact output |
| `-n` | Null input |
| `-s` | Slurp (read all inputs into array) |
| `-e` | Exit with 1 if last output is false/null |
| `-R` | Raw input (each line as string) |
| `-f FILE` | Read filter from file |
| `--arg NAME VAL` | Bind $NAME to string value |
| `--argjson NAME VAL` | Bind $NAME to JSON value |
| `--tab` | Indent with tabs |
| `--indent N` | Indent with N spaces |
| `--compile` | AOT compile filter to standalone binary |
| `-o FILE` | Output path for AOT compiled binary |

## Supported Features

- **Arithmetic**: `+`, `-`, `*`, `/`, `%`
- **Comparison**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Control flow**: `if-then-else`, `try-catch`, `//` (alternative)
- **Generators**: `,`, `empty`, `.[]`, `select`, `reduce`, `foreach`, `range`
- **Object construction**: `{key: .value}`, dynamic keys, shorthand
- **String interpolation**: `"Hello \(.name)"`
- **Regex**: `test`, `match`, `capture`, `scan`, `sub`, `gsub`
- **User-defined functions**: `def f: body; ...`, `def f(x): body; ...`
- **Math functions**: `sqrt`, `cbrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan(y;x)`, `exp`, `exp2`, `exp10`, `log`, `log2`, `log10`, `pow(x;y)`, `significand`, `exponent`, `logb`, `nearbyint`, `trunc`, `rint`, `j0`, `j1`, `fabs`, `floor`, `ceil`, `round`
- **Standard library**: `map`, `select`, `sort`, `sort_by`, `group_by`, `unique`, `unique_by`, `reverse`, `flatten`, `add`, `any`, `all`, `min`, `max`, `min_by`, `max_by`, `keys`, `values`, `has`, `in`, `contains`, `inside`, `to_entries`, `from_entries`, `ascii_downcase`, `ascii_upcase`, `ltrimstr`, `rtrimstr`, `startswith`, `endswith`, `split`, `join`, `indices`, `length`, `type`, `tostring`, `tonumber`, `tojson`, `fromjson`, `not`, `sort`, `reverse`, `unique`, `explode`, `implode`, `debug`, `env`, `builtins`, `getpath`, `setpath`, `delpaths`, `bsearch`, `infinite`, `nan`, `isinfinite`, `isnan`, `isnormal`
- **Format strings**: `@base64`, `@base64d`, `@html`, `@uri`, `@urid`, `@csv`, `@tsv`, `@json`, `@text`, `@sh`
- **Date/time**: `now`, `gmtime`, `mktime`, `strftime`, `strptime`, `strflocaltime`
- **String**: `trim`, `ltrim`, `rtrim`, `ascii_downcase`, `ascii_upcase`, `utf8bytelength`
- **Array**: `transpose`, `bsearch`
- **Recursive descent**: `..`
- **Optional access**: `.foo?`, `.[]?`

## Tech Stack

- **Language**: Rust
- **JIT backend**: Cranelift v0.129
- **Input**: jq bytecode via libjq FFI
- **Execution model**: CPS + callback (backtracking → normal function calls)
- **Value representation**: 16-byte `#[repr(C, u64)]` enum + Rc

## Project Structure

| Path | Role |
|---|---|
| `src/jq_ffi.rs` | libjq FFI bindings |
| `src/bytecode.rs` | Safe bytecode wrapper + disassembler |
| `src/compiler.rs` | Bytecode → CPS IR |
| `src/cps_ir.rs` | CPS intermediate representation |
| `src/codegen.rs` | CPS IR → Cranelift CLIF → native |
| `src/runtime.rs` | JIT runtime helpers (rt_*) |
| `src/value.rs` | Value type + JSON conversion |
| `src/input.rs` | Shared streaming JSON input (serde_json) |
| `src/output.rs` | Shared output formatting |
| `src/bin/jq-jit.rs` | CLI binary |
| `src/main.rs` | Differential test suite |
| `tests/compat.sh` | CLI compatibility tests |
| `tests/official/` | jq official test suite runner |
| `benches/` | Criterion benchmarks |

## Documents

| File | Content |
|---|---|
| [docs/plan/](docs/plan/) | Development plans & implementation records |
| [docs/benchmark.md](docs/benchmark.md) | Performance report (Criterion + hyperfine) |
| [docs/usecase-benchmark.md](docs/usecase-benchmark.md) | Real-world use case benchmarks |
| [STATUS.md](STATUS.md) | Current progress + technical insights |
| [docs/research/](docs/research/) | Technical research (6 documents) |

## Test Results

- **jq official test suite**: 340 PASS / 368 attempted (92.4%), 142 SKIP (unsupported), 28 FAIL / 510 total
- **Differential tests**: 649 PASS, 1 FAIL (known libjq divergence)
- **CLI compatibility**: 113 PASS, 1 FAIL, 2 SKIP
