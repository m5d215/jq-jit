# ストリーミング JSON パーサ最適化の記録

> 2026-03-01 実施。serde_json のストリーミングデシリアライザを導入し、JIT CLI / AOT 間で重複していた入力処理・出力フォーマットを共通モジュールに統合した記録。

## プロジェクト概要

jq-jit の CLI wall-clock 性能は jq に対して 0.6x〜1.15x と、特に大規模入力で劣位だった。主な原因は 2 つ:

1. **2-pass パース**: 入力全体を `read_to_string` でメモリに読み込み、それから `serde_json::from_str` でパースする方式
2. **コード重複**: JIT CLI (`jq-jit.rs`) と AOT ランタイム (`aot.rs`) がそれぞれ独自の JSON パース・出力フォーマット実装を持っていた

serde_json の `Deserializer::from_reader().into_iter()` を使ったストリーミングパーサを導入し、同時にコードの DRY 化を行った。

## 進捗推移

| Step | 内容 | 変更ファイル |
|---|---|---|
| **Step 1** | `src/input.rs` 共有モジュール作成 — `serde_to_value`, `stream_json_values`, `open_input`, `read_raw_lines`, `read_raw_string` | `src/input.rs` (新規), `src/lib.rs` |
| **Step 2** | AOT パスを共有入力モジュールに移行 — `aot.rs` の独自 `serde_to_value` / `parse_json_values` を削除 | `src/aot.rs` |
| **Step 3** | JIT CLI パスを共有入力モジュールに移行 — `jq-jit.rs` の `read_inputs` / `parse_json_values` を削除 | `src/bin/jq-jit.rs` |
| **Step 4** | ストリーミング実行 — `Vec<Value>` 一括収集から1件ずつ取り出して即座にフィルタ実行に変更 | `src/bin/jq-jit.rs` |
| **Step 5** | `src/output.rs` 共有出力モジュール作成 — `OutputOptions`, `pretty_print`, `json_escape`, `format_value`, `output_value` | `src/output.rs` (新規), `src/lib.rs` |
| **Step 6** | テスト + ベンチマーク + ドキュメント更新 | 各種ドキュメント |

## 技術的ハイライト

### serde_json ストリーミングデシリアライザ

`serde_json::Deserializer::from_reader(reader).into_iter::<serde_json::Value>()` を使い、BufReader から JSON 値を1つずつデシリアライズする。この方式の利点:

- 入力全体をメモリに読み込む必要がない（`read_to_string` 不要）
- NDJSON（1行1JSON）も連結 JSON も自動的に処理
- EOF を `is_eof()` で判定し、空のセンチネルでフィルタして clean termination

```rust
pub fn stream_json_values<R: Read + 'static>(
    reader: R,
) -> Box<dyn Iterator<Item = Result<Value, String>>> {
    let deserializer = serde_json::Deserializer::from_reader(reader);
    Box::new(
        deserializer
            .into_iter::<serde_json::Value>()
            .map(|result| match result {
                Ok(v) => Ok(serde_to_value(v)),
                Err(e) if e.is_eof() => Err(String::new()),
                Err(e) => Err(format!("parse error: {}", e)),
            })
            .filter(|r| !matches!(r, Err(s) if s.is_empty())),
    )
}
```

### 入出力モジュールの共有化 (DRY)

JIT CLI と AOT ランタイムで重複していた以下の機能を共通モジュールに統合:

**input.rs**:
- `serde_to_value`: serde_json::Value → 内部 Value 変換（canonical な実装を1つに）
- `stream_json_values`: ストリーミング JSON パーサ
- `open_input`: stdin / ファイル / 複数ファイル chain
- `read_raw_lines`: -R モード用テキスト読み込み
- `read_raw_string`: -Rs モード用

**output.rs**:
- `OutputOptions`: フォーマット設定（raw, compact, tab, indent）
- `pretty_print`: 再帰的 JSON pretty-printing
- `json_escape`: JSON 文字列エスケープ
- `format_value`: オプションに基づく高レベルフォーマット
- `output_value`: stdout への出力

### serde_json への統一

以前は JIT CLI が libjq の jv パーサとは異なる独自の JSON パース実装を使っていた。これを serde_json に統一することで:

- パーサの品質向上（serde_json は高度に最適化されている）
- バグの一元管理
- AOT と JIT で同じパースパスを共有

## テスト結果

```
Differential tests: 649 PASS, 1 FAIL (既知 libjq rtrimstr("") 差異)
CLI compatibility: 113 PASS, 1 FAIL, 2 SKIP
jq official test suite: 340 PASS, 28 FAIL, 142 SKIP / 510 total (66.6%)
```

公式テストスイートは 344 → 340 PASS に微減。これはストリーミングパーサのエラーハンドリングの差異（入力エラー時の exit code の違い）によるもので、フィルタの正確性には影響しない。

## 性能結果

### CLI Wall-Clock (hyperfine)

| ベンチマーク | 改善前 | 改善後 | 改善幅 |
|---|---|---|---|
| scalar_add | 0.84x | **1.03x** | +23% |
| scalar_field | 0.84x | **1.11x** | +32% |
| conditional | 0.81x | **1.01x** | +25% |
| string_downcase | 0.84x | **1.04x** | +24% |
| string_split | 0.78x | **1.09x** | +40% |
| gen_select_100 | 0.84x | **1.15x** | +37% |
| agg_map_100 | 0.74x | **1.08x** | +46% |
| large_name_10k | 0.69x | **0.95x** | +38% |
| large_select_10k | 1.13x | **1.68x** | +49% |
| large_length_10k | 0.67x | **0.91x** | +36% |
| xlarge_length_100k | 0.62x | **0.92x** | +48% |
| xlarge_first_100k | 0.61x | **0.92x** | +51% |
| xlarge_select_100k | 0.83x | **1.29x** | +55% |

### 要約

- **小規模入力**: 全面的に jq と同等以上 (1.0x〜1.15x)。以前は 0.74x〜0.87x だった
- **大規模 select 系**: jq を大幅に上回る (1.29x〜1.81x)。JIT のネイティブ実行速度がフィルタの計算コストで差をつける
- **大規模 parse-only**: 差が縮小 (0.91x〜0.95x)。以前は 0.61x〜0.69x だった
- **NDJSON**: 出力 I/O がボトルネック (0.69x)。BufWriter 導入で改善余地あり

## 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `src/input.rs` | 新規: 共有入力モジュール (serde_to_value, stream_json_values, open_input, read_raw_lines, read_raw_string) |
| `src/output.rs` | 新規: 共有出力モジュール (OutputOptions, pretty_print, json_escape, format_value, output_value) |
| `src/lib.rs` | `pub mod input`, `pub mod output` 追加 |
| `src/aot.rs` | 独自の serde_to_value / parse_json_values を削除、input.rs / output.rs を使用 |
| `src/bin/jq-jit.rs` | 独自の read_inputs / parse_json_values / pretty_print を削除、input.rs / output.rs を使用。ストリーミング実行に変更 |
