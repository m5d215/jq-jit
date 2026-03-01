# jq-jit Performance Report

> 2026-03-01 測定。macOS (Darwin 25.3.0, Apple Silicon)

## TL;DR

- **JIT 実行のみ**: libjq 比 **55x〜18,000x** 高速（Criterion マイクロベンチマーク）
- **CLI wall-clock**: jq 比 **0.9x〜1.68x**（ストリーミング JSON パーサ最適化後）
- **AOT バイナリ**: jq 比 **1.04x** 高速（JIT コンパイルオーバーヘッドを完全除去）
- **フィルタキャッシュ**: 2回目以降は JIT コンパイルをスキップし dlopen で即ロード（~3ms → ~2.5ms）
- **損益分岐点**: 同一フィルタを 2回以上実行するなら JIT が有利。1-shot CLI では AOT バイナリが最速

## 測定環境

| 項目 | 値 |
|---|---|
| OS | macOS (Darwin 25.3.0, Apple Silicon) |
| jq | 1.7.1-apple (system) |
| jq-jit | v0.1.0 (Cranelift v0.129) |
| libjq (FFI) | 1.8.1 |
| ツール | Criterion 0.5 / hyperfine 1.20.0 |

## 1. JIT 実行性能（Criterion マイクロベンチマーク）

JIT コンパイル済みコードの実行時間のみを測定。JSON パース・プロセス起動は含まない。

### JIT コンパイル時間

| フィルタ | コンパイル時間 |
|---|---|
| `. + 1` | 419 µs |
| `.foo` | 396 µs |
| `.[] \| . + 1` | 447 µs |
| `if . > 0 then . * 2 else . * -1 end` | 469 µs |
| `map(. * 2) \| add` | 475 µs |
| `reduce .[] as $x (0; . + $x)` | 439 µs |

フィルタの複雑さによらず **400〜480 µs** で安定。Cranelift のセットアップコストが支配的。

### JIT 実行 vs libjq（run_jq 経由）

run_jq は libjq の compile + jq_start/jq_next を毎回実行する。jq-jit は compile 済みネイティブコードを直接呼ぶ。

| カテゴリ | フィルタ | 入力 | JIT | libjq | 倍率 |
|---|---|---|---|---|---|
| スカラー | `. + 1` | `42` | 17 ns | 309 µs | **18,152x** |
| スカラー | `.foo + .bar * 2` | `{"foo":10,"bar":20}` | 38 ns | 313 µs | **8,236x** |
| 条件分岐 | `if . > 0 then . * 2 else . * -1 end` | `5` | 19 ns | 320 µs | **16,842x** |
| ジェネレータ | `.[] \| . + 1` | [0..99] | 704 ns | 338 µs | **480x** |
| ジェネレータ | `.[] \| select(. > 50) \| . * 2` | [0..99] | 768 ns | 351 µs | **457x** |
| 集約 | `map(. * 2)` | [0..99] | 942 ns | 339 µs | **360x** |
| 集約 | `map(. * 2) \| add` | [0..99] | 1.28 µs | 351 µs | **274x** |
| 集約 | `reduce .[] as $x (0; . + $x)` | [0..999] | 5.29 µs | 520 µs | **98x** |
| 文字列 | `split(",") \| length` | "a,b,...,z" | 730 ns | 311 µs | **426x** |
| 文字列 | `ascii_downcase` | "HELLO WORLD" | 62 ns | 321 µs | **5,178x** |
| 大規模 | `.[] \| .name` | 10K objects | 134 µs | 7.36 ms | **55x** |
| 大規模 | `[.[] \| select(.age > 30)]` | 10K objects | 165 µs | 10.9 ms | **66x** |
| 大規模 | `.[] \| . + 1` | 1K-key object | 7.86 µs | 699 µs | **89x** |

### 倍率の傾向

- **スカラー演算 (5,000x〜18,000x)**: libjq は毎回 compile+interpret。JIT は関数呼び出し 1 回。圧倒的
- **ジェネレータ/集約 (100x〜500x)**: 要素数に比例してランタイムコール（Rc clone, Vec push）が増加。ネイティブコードの速さと Rust ランタイムのオーバーヘッドが拮抗
- **大規模データ (55x〜90x)**: 配列走査・オブジェクトアクセスのランタイムコストが支配的

## 2. CLI Wall-Clock 比較（hyperfine） — ストリーミングパーサ最適化後

実際の `jq` / `jq-jit` コマンドの end-to-end 実行時間。プロセス起動・JSON パース・JIT コンパイル・出力すべて含む。

> **2026-03-01 更新**: ストリーミング JSON パーサ最適化 (serde_json Deserializer + 入出力共有化) の反映後。以前の結果は「改善前比較」セクションを参照。

### 小〜中規模入力

| ベンチマーク | フィルタ | 入力 | jq (ms) | jq-jit (ms) | 比率 |
|---|---|---|---|---|---|
| scalar_add | `. + 1` | 3B | 2.7 | 2.6 | **1.03x** |
| scalar_field | `.foo + .bar * 2` | 20B | 2.8 | 2.7 | **1.11x** |
| conditional | `if . > 0 then ...` | 3B | 2.8 | 2.7 | **1.01x** |
| string_downcase | `ascii_downcase` | 14B | 2.8 | 2.7 | **1.04x** |
| string_split | `split(",") \| length` | 202B | 2.9 | 2.7 | **1.09x** |
| gen_each_100 | `.[] \| . + 1` | 391B | 2.8 | 2.8 | 1.00x |
| gen_select_100 | `.[] \| select(. > 50) \| . * 2` | 391B | 3.1 | 2.7 | **1.15x** |
| agg_map_100 | `map(. * 2)` | 391B | 2.9 | 2.7 | **1.08x** |
| agg_map_add_100 | `map(. * 2) \| add` | 391B | 2.8 | 2.7 | **1.04x** |
| agg_sort_100 | `sort \| reverse` | 391B | 2.8 | 2.7 | **1.03x** |

### 大規模入力

| ベンチマーク | フィルタ | 入力 | jq (ms) | jq-jit (ms) | 比率 |
|---|---|---|---|---|---|
| large_name_10k | `.[] \| .name` | 448KB | 11.9 | 12.5 | 0.95x |
| large_select_10k | `[.[] \| select(.age > 30)]` | 448KB | 18.9 | 11.3 | **1.68x** |
| large_length_10k | `length` | 448KB | 8.1 | 8.8 | 0.91x |
| xlarge_length_100k | `length` | 6.3MB | 70.7 | 77.0 | 0.92x |
| xlarge_first_100k | `.[0]` | 6.3MB | 70.5 | 76.4 | 0.92x |
| xlarge_select_100k | `[.[] \| select(.active)] \| length` | 6.3MB | 89.0 | 69.2 | **1.29x** |

### 追加ベンチマーク

| ベンチマーク | フィルタ | 入力 | jq (ms) | jq-jit (ms) | 比率 |
|---|---|---|---|---|---|
| select_10k_custom | `[.[] \| select(.age > 30)]` | 329KB | 18.0 | 10.0 | **1.81x** |
| length_100k_custom | `length` | 3.4MB | 42.9 | 51.8 | 0.83x |
| first_100k_custom | `.[0]` | 3.4MB | 43.0 | 52.1 | 0.83x |
| ndjson_100k | `.id` | 2.7MB (NDJSON) | 68.7 | 99.0 | 0.69x |
| scalar_simple | `. + 1` | echo 42 | 2.6 | 2.6 | **1.03x** |

### 改善前比較 (ストリーミングパーサ最適化前 → 後)

| ベンチマーク | Before | After | 改善幅 |
|---|---|---|---|
| scalar_add | 0.84x | **1.03x** | +23% |
| scalar_field | 0.84x | **1.11x** | +32% |
| agg_map_100 | 0.74x | **1.08x** | +46% |
| large_name_10k | 0.69x | **0.95x** | +38% |
| large_select_10k | 1.13x | **1.68x** | +49% |
| large_length_10k | 0.67x | **0.91x** | +36% |
| xlarge_length_100k | 0.62x | **0.92x** | +48% |
| xlarge_first_100k | 0.61x | **0.92x** | +51% |
| xlarge_select_100k | 0.83x | **1.29x** | +55% |

### Wall-Clock の分析

**改善の主因:**

1. **serde_json ストリーミングパーサ**: `serde_json::Deserializer::from_reader().into_iter()` により、入力全体をメモリに読み込んでからパースする方式から、読みながらパースする方式に変更。メモリ割り当てと memcpy が減少
2. **serde_json への統一**: JIT CLI が独自の JSON パーサを使っていたのを serde_json に統一。serde_json は高度に最適化されており、parse 性能が向上
3. **コード重複排除**: input.rs / output.rs への共通化で、JIT CLI と AOT のパスが同じ最適化コードを共有

**jq-jit が依然として遅いケース:**

- NDJSON (0.69x): 100K 行の NDJSON では、jq-jit の出力 I/O (100K 回の println) がボトルネック
- parse-only ワークロード (0.83x〜0.92x): `length`, `.[0]` など、フィルタ実行が軽くパース時間が支配的なケース。jq の jv パーサは内部表現に直接変換するため有利

### jq-jit が勝つ条件

- **計算集約的フィルタ**: `select + 配列構築` (1.3x〜1.8x)
- **小規模入力**: プロセス起動コストの差で jq-jit がわずかに高速 (1.0x〜1.15x)
- **フィルタキャッシュ有効時**: JIT コンパイル (~450µs) をスキップしてさらに高速

**傾向**: ストリーミングパーサ最適化により、jq-jit は小規模入力で jq と同等以上、計算集約的な中〜大規模入力で jq を大幅に上回るようになった。

## 3. AOT コンパイル性能

フィルタを事前にネイティブバイナリにコンパイルし、jq-jit なしでスタンドアロン実行する。JIT コンパイルオーバーヘッド (~450µs) を完全に除去。

### AOT CLI wall-clock（hyperfine）

| フィルタ | jq | jq-jit (JIT) | AOT バイナリ | AOT 対 jq |
|---|---|---|---|---|
| `. + 1` | 2.6ms | 2.6ms | 2.5ms | 1.04x 高速 |

### 分析

- AOT はプロセス起動コストが支配的な単純フィルタでは jq とほぼ同等
- JIT コンパイルオーバーヘッド (~450µs) を完全に除去
- 大規模データ・複雑フィルタでの差はより顕著になる見込み
- AOT バイナリサイズ: ~8MB (libjq_jit.a の rt_* 関数を含む)

### 使い方

```bash
# AOT コンパイル
jq-jit --compile '. + 1' -o add1

# 実行 (jq-jit も jq も不要)
echo 3 | ./add1  # → 4

# 性能比較
hyperfine 'echo 42 | jq ". + 1"' 'echo 42 | ./add1'
```

## 4. フィルタキャッシュ

同じフィルタの JIT コンパイル結果をディスクキャッシュし、2回目以降の実行では dlopen で即ロードする。

| 実行 | 処理 | wall-clock |
|---|---|---|
| 初回 | JIT コンパイル + 実行 | ~3ms |
| 2回目以降 | dlopen + 実行 | ~2.5ms |

JIT コンパイル (~450µs) をスキップすることで、繰り返し実行時のレイテンシが安定する。CLI の 1-shot 実行でもキャッシュが効くため、同じフィルタを複数回使うワークフローでは自動的に高速化される。

## 5. 性能特性の分析

### なぜ Criterion で 18,000x なのに CLI で 1.0x なのか

Criterion は **JIT 実行のみ** を測定する。libjq 側は毎回 compile+interpret を繰り返す。一方 CLI wall-clock では:

```
jq:     [startup 2ms] + [parse 0.01ms] + [compile 0.01ms] + [execute 0.01ms] = ~2.7ms
jq-jit: [startup 2ms] + [parse 0.01ms] + [JIT compile 0.45ms] + [execute 0.00002ms] = ~2.7ms
```

小規模入力ではプロセス起動がボトルネック。ストリーミングパーサ最適化後は jq と同等以上。

大規模入力 (計算集約的フィルタ) では:

```
jq:     [startup 2ms] + [parse 5ms] + [execute 12ms] = ~19ms (large_select_10k)
jq-jit: [startup 2ms] + [parse 5ms] + [JIT compile 0.45ms] + [execute 0.16ms] + [output 3ms] = ~11ms
```

JIT のネイティブ実行速度がフィルタの計算コストを大幅に削減し、パースの差を補って余りある。

大規模入力 (parse-only ワークロード) では:

```
jq:     [startup 2ms] + [parse 69ms] = ~71ms (xlarge_length_100k)
jq-jit: [startup 2ms] + [parse 75ms] = ~77ms
```

jq の jv パーサが serde_json + Value 変換よりわずかに高速。差は約 1.1 倍に縮小（以前は 1.6 倍）。

### jq-jit が真価を発揮するユースケース

1. **サーバーサイド**: フィルタを 1 回コンパイルし、リクエストごとに繰り返し実行
2. **ストリーム処理**: 大量の JSON レコードに同じフィルタを適用
3. **バッチ処理**: NDJSON の各行に同じフィルタを適用
4. **埋め込み**: Rust アプリケーションに組み込み、Value を直接渡す（JSON パース不要）

### ボトルネックと改善候補

| ボトルネック | 現状 | 改善案 |
|---|---|---|
| JSON パース | ~~serde_json (汎用)~~ ✅ serde_json streaming deserializer | simd-json (SIMD 加速) |
| JIT コンパイル | 400-480 µs/filter | ~~コンパイル結果キャッシュ~~ ✅ / ~~AOT モード~~ ✅ |
| Value clone | Rc ベース (ref count) | Arena allocator |
| 大量出力 | ~~Vec<Value> 蓄積 → 逐次出力~~ ✅ ストリーミング出力 | buffered writer |
| NDJSON 出力 I/O | 1行ごとに println (100K 回) | BufWriter でバッファリング |

## 6. 測定コマンド

```bash
# Criterion マイクロベンチマーク
cargo bench

# CLI wall-clock 比較 (hyperfine)
cargo build --release
bash benches/compare.sh

# 特定ベンチマークだけ実行
cargo bench -- scalar
cargo bench -- large_data

# AOT コンパイル + 性能比較
jq-jit --compile '. + 1' -o add1
hyperfine 'echo 42 | jq ". + 1"' 'echo 42 | ./add1'
```

## 8. 実用ユースケースベンチマーク

現実的なシナリオでの性能比較。詳細は [usecase-benchmark.md](usecase-benchmark.md) を参照。

| ユースケース | 代表フィルタ | Speedup |
|---|---|---|
| ログ分析 (100K NDJSON) | select + 条件フィルタ | 1.16x〜1.21x |
| API レスポンス加工 (50K objects) | flatten / select+transform / group_by | 1.27x〜1.38x |
| バッチ ETL (100 ファイル) | select + 変換 ×100 (キャッシュ有効) | 1.15x |

計算集約的なフィルタ（select + 変換 + 集計）で安定して jq を上回る。特に単一の大きな JSON 配列に対する group_by + map + 集計で **1.38x** を達成。

> **注意**: `[.items[].name]` のような generator-in-scalar フィルタはコンパイルエラーになる。`(.items | map(.name))` に書き換えが必要。

## 7. 再現データ

### Criterion 生データ

`target/criterion/` に HTML レポートが生成される。`open target/criterion/report/index.html` で閲覧可能。

### hyperfine 生データ

`benches/compare.sh` 実行時に各ベンチマークの JSON を一時ディレクトリに出力する。永続化が必要な場合はスクリプトの `TMPDIR` を変更すること。
