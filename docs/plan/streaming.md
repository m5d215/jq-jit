# jq-jit ストリーミング JSON パーサ最適化

> **状態: 完了** — 全 6 Step 完了。実装記録: [streaming-record.md](streaming-record.md)

## Context

jq-jit の CLI wall-clock 性能は jq に対して 0.6x〜1.15x と、大規模入力では劣位だった。主因は JSON パーサの差: jq-jit は入力全体を文字列に読み込んでからパースする 2-pass 方式で、JIT CLI と AOT ランタイムにそれぞれ独自の JSON パース実装が重複していた。

ストリーミング JSON パーサを導入し、入力を読みながらパースする 1-pass 方式に変更。同時に、JIT CLI と AOT で重複していた入力処理・出力フォーマットを共通モジュールに統合する。

## 目標

1. **性能改善**: 大規模入力での CLI wall-clock を jq と同等以上に
2. **コード重複排除**: JIT CLI / AOT の入力処理と出力フォーマットを DRY 化
3. **serde_json 統一**: 両パスのパーサを serde_json に統一

## アーキテクチャ

### 変更前

```
JIT CLI (jq-jit.rs):
  stdin → read_to_string → 独自 JSON パーサ → Vec<Value> → filter → output

AOT (aot.rs):
  stdin → read_to_string → serde_json::from_str → Vec<Value> → filter → output
```

### 変更後

```
共有モジュール:
  input.rs:  stdin → BufReader → serde_json::Deserializer::from_reader().into_iter() → Stream<Value>
  output.rs: Value → format → println

JIT CLI:
  input::open_input → input::stream_json_values → filter (1件ずつ) → output::output_value

AOT:
  input::open_input → input::stream_json_values → filter (1件ずつ) → output::output_value
```

## 実装ステップ

### Step 1: src/input.rs — 共有入力モジュール

`serde_to_value`, `stream_json_values`, `open_input`, `read_raw_lines`, `read_raw_string` を1つのモジュールに集約。

**変更ファイル**: `src/input.rs` (新規), `src/lib.rs`

### Step 2: AOT パス移行

`src/aot.rs` の独自 `serde_to_value` / `parse_json_values` を `input.rs` の関数に置換。

**変更ファイル**: `src/aot.rs`

### Step 3: JIT CLI パス移行

`src/bin/jq-jit.rs` の `read_inputs` / `parse_json_values` を `input.rs` に置換。

**変更ファイル**: `src/bin/jq-jit.rs`

### Step 4: ストリーミング実行

入力を `Vec<Value>` に一括収集してから処理する方式から、`stream_json_values` イテレータから1件ずつ取り出して即座にフィルタ実行する方式に変更。

**変更ファイル**: `src/bin/jq-jit.rs`

### Step 5: src/output.rs — 共有出力フォーマットモジュール

JIT CLI と AOT で重複していた `pretty_print`, `json_escape`, `format_value`, `output_value` を共通モジュールに抽出。

**変更ファイル**: `src/output.rs` (新規), `src/lib.rs`

### Step 6: テスト + ベンチマーク + ドキュメント更新

全テストスイートの回帰テスト、hyperfine ベンチマーク、ドキュメント更新。

## 検証方法

```bash
cargo build --release

# 回帰テスト
cargo run --bin jq-jit-test
bash tests/compat.sh target/release/jq-jit
bash tests/official/run.sh target/release/jq-jit tests/official/jq.test

# 性能比較
bash benches/compare.sh
hyperfine --warmup 3 --min-runs 10 \
  'cat /tmp/bench_10k.json | jq "[.[] | select(.age > 30)]"' \
  'cat /tmp/bench_10k.json | ./target/release/jq-jit "[.[] | select(.age > 30)]"'
```

## 技術的リスク

| リスク | レベル | 対策 |
|---|---|---|
| ストリーミングパーサの EOF 処理 | 中 | serde_json の is_eof() でセンチネル判定 |
| NDJSON の改行処理 | 低 | serde_json の into_iter() が自動的にハンドル |
| 出力フォーマットの互換性 | 低 | 全テストスイートで回帰テスト |
| -s (slurp) モードとの互換 | 中 | slurp は全入力を配列に集約するため、ストリーミングと併存する分岐が必要 |
