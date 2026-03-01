# AOT コンパイル実装の記録

> 2026-02-28 実施。jq フィルタをスタンドアロン実行バイナリに AOT コンパイルする機能を実装した記録。フィルタキャッシュで構築した共通基盤を活用し、差分実装で完成させた。

## プロジェクト概要

jq-jit は Criterion ベンチマークで libjq 比 55x〜18,000x の高速化を達成しているが、CLI wall-clock ではプロセス起動と JIT コンパイルのオーバーヘッドにより jq と同等。AOT コンパイルで jq フィルタをスタンドアロンの実行バイナリに変換し、JIT コンパイルと libjq 依存を除去する。

```bash
jq-jit --compile '. + 1' -o add1
echo 3 | ./add1  # → 4
```

## 進捗推移

| Step | 内容 | 変更ファイル |
|---|---|---|
| **Step 1** | Cargo.toml + build.rs — staticlib 設定、`native-static-libs` キャプチャ | `Cargo.toml`, `build.rs` |
| **Step 2** | aot.rs — AOT ランタイムエントリポイント (`aot_run`, 525 行) | `src/aot.rs` (新規), `src/lib.rs` |
| **Step 3** | `compile_to_object` — Cranelift で `main()` 関数生成、リテラルデータ `.rodata` 埋め込み | `src/codegen.rs` |
| **Step 4** | `--compile` フラグ — フィルタ → `.o` → cc リンク → 実行バイナリ | `src/bin/jq-jit.rs` |
| **Step 5** | 統合テスト + ドキュメント更新 | テストスクリプト, STATUS.md |

## 技術的ハイライト

### キャッシュの共通基盤を活用

フィルタキャッシュの Step 1（Module トレイト抽象化 `build_filter<M: Module>`）と Step 2（リテラル間接参照化 `LiteralPool`）をそのまま活用。AOT 固有の実装は `compile_to_object` での main 関数生成と `aot.rs` エントリポイントのみで済んだ。

### Cranelift による main 関数生成

`compile_to_object` で `jit_filter` に加えて `main` 関数も Cranelift で生成。`func_addr` 命令で同一モジュール内の `jit_filter` のアドレスを取得し、`aot_run` に渡す。

### .rodata へのリテラル埋め込み

`DataDescription` を使い、シリアライズ済みリテラルデータを `.rodata` セクションに埋め込む。実行時に `aot_run` がデータを読み取り、Value オブジェクトを再構築する。

### libjq 完全非依存のランタイム

`aot_run` は libjq に一切依存しない。JSON パースは `serde_json` で行い、ランタイム関数群は `libjq_jit.a` (28MB) に全て含まれる。`catch_unwind` でパニック安全な C ABI 境界を実現。

### ランタイムライブラリの自動検索

`find_runtime_lib()` で `libjq_jit.a` を以下の優先順序で自動検索:

1. `JQ_JIT_LIB_DIR` 環境変数
2. 実行バイナリの隣接ディレクトリ
3. `target/release`

### build.rs による native-static-libs キャプチャ

`rustc --print native-static-libs` の出力を `build.rs` でキャプチャし、`env!` マクロでバイナリに埋め込む。AOT バイナリのリンク時に正しいシステムライブラリフラグを自動的に渡せる。

## 性能

| 対象 | 実行時間 (`. + 1`) | 対 jq 比 |
|---|---|---|
| AOT バイナリ | ~2.5ms | **1.05x** (最速) |
| jq-jit (JIT) | ~2.6ms | 1.00x |
| jq | ~2.6ms | 1.00x |

プロセス起動コストが支配的なため差は小さいが、AOT が一貫して最速。大規模データではより差が出る見込み。

## テスト結果

### AOT 統合テスト: 10/10 PASS

| # | テスト | 結果 |
|---|---|---|
| 1 | 基本算術 (`. + 1`) | PASS |
| 2 | 文字列操作 (`.name \| ascii_upcase`) | PASS |
| 3 | 配列フィルタ (`[.[] \| select(. > 2)]`) | PASS |
| 4 | オブジェクト構築 (`{name: .a, value: .b}`) | PASS |
| 5 | 文字列補間 (`"Hello, \(.name)!"`) | PASS |
| 6 | ランタイムフラグ (`-c`, `-r`) | PASS |
| 7 | 複数出力 (`.[]`) | PASS |
| 8 | null 入力 (`-n`) | PASS |
| 9 | 複合パイプライン (map + sort_by) | PASS |
| 10 | raw 出力フラグ (`.name -r`) | PASS |

### 回帰テスト

```
Differential tests: 649 PASS, 1 FAIL (既知 libjq 差異)
CLI compatibility: 113 PASS
```

## 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `src/aot.rs` | 新規: `aot_run` エントリポイント (525 行) |
| `src/codegen.rs` | `compile_to_object` 追加 (main 関数生成 + リテラル .rodata 埋め込み) |
| `src/bin/jq-jit.rs` | `--compile`, `-o` フラグ、`compile_aot` 関数、`find_runtime_lib` |
| `src/lib.rs` | `pub mod aot` 追加 |
| `Cargo.toml` | `[lib] crate-type` 追加、`serde_json` 追加 |
| `build.rs` | `native-static-libs` キャプチャ |
