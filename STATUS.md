# jq-jit: Project Status

> このファイルはセッションの冒頭で読み込み、作業終了時に更新する。
> コンテキスト compaction 後の状態復元に使う。

**順守** 作業を円滑に進めるために、非常に多くの権限を与えている。危険な操作は行わないこと。もし不明点があれば、必ずユーザーに確認すること。

## 現在のフェーズ

**Phase 10: 完全互換** — 全サブタスク完了

## テスト結果

- **jq 公式テストスイート**: 340 PASS, 28 FAIL, 142 SKIP / 510 total (66.6%)
  - SKIP 除外実質 PASS 率: 340/368 = **92.4%**
- **Differential tests**: 649 PASS, 1 FAIL (既知 libjq rtrimstr("") 差異)
- **CLI compatibility**: 113 PASS, 1 FAIL, 2 SKIP

## テスト実行方法

```bash
cargo build --release
cargo run --bin jq-jit-test          # differential tests (649件)
bash tests/compat.sh target/release/jq-jit  # CLI compatibility (116件)
bash tests/official/run.sh target/release/jq-jit tests/official/jq.test  # jq公式 (510件)
```

## 残り 24 FAIL

| カテゴリ | 件数 | 根本原因 |
|---|---|---|
| generator-in-scalar (exit 101) | 8 | codegen が Comma/generator を scalar context で処理できない |
| path() 操作 | 6 | path 追跡エンジンが select/map と連携しない |
| try/catch 優先順位 | 3 | bytecode レベルの構造問題 |
| 代入/更新の深いバグ | 3 | variable binding in path, recursive descent update |
| label/break 未実装 | 1 | LABEL/BREAK opcode 未対応 |
| f64 精度/フォーマット | 2 | 任意精度数値 (have_decnum) 未対応 |
| その他 | 1 | format edge case |

## 残り 142 SKIP の主要カテゴリ

未実装機能によりコンパイラが exit 3 を返すテスト。

| カテゴリ | 件数 | 実装難易度 |
|---|---|---|
| ?// alternative destructuring | 17 | 高（新構文） |
| try/catch 高度パターン | 15 | 高（bytecode 構造） |
| any/all with closure args | 13 | 中 |
| limit/first/last/nth | 10 | 高（generator-in-scalar） |
| modules/import | 10 | 高（新サブシステム） |
| reduce/foreach 高度 | 9 | 高（クロージャ） |
| higher-order builtins (IN/JOIN/INDEX) | 9 | 中 |
| have_decnum (任意精度) | 8 | 高（数値型変更） |
| def 高度（再帰・クロージャ） | 8 | 高（トランポリン必要） |
| foreach 高度 | 7 | 高 |
| datetime 追加 | 6 | 低〜中 |
| その他 | 30 | 混在 |

## ベンチマーク結果

- **Criterion (JIT 実行のみ)**: libjq 比 **55x〜18,000x** 高速
  - スカラー `. + 1`: 17ns vs 309µs = **18,152x**
  - 大規模 `.[] | .name` (10K): 134µs vs 7.36ms = **55x**
- **CLI wall-clock (hyperfine)**: jq 比 **0.9x〜1.68x**
  - ストリーミング JSON パーサ最適化により大幅改善
  - 小規模: jq と同等 (1.0x〜1.1x)
  - 大規模 (select 系): jq-jit が **1.3x〜1.68x** 高速
  - 大規模 (parse-only): jq がやや高速 (0.9x)

- **実用ユースケース (hyperfine)**: jq 比 **1.15x〜1.38x** 高速
  - ログ分析 (100K NDJSON): select + 条件フィルタで **1.16x〜1.21x**
  - API レスポンス加工 (50K objects): group_by + 集計で **1.38x**
  - バッチ ETL (100 ファイル): フィルタキャッシュ効果で **1.15x**

```bash
cargo bench                    # Criterion マイクロベンチマーク
bash benches/compare.sh        # CLI wall-clock 比較 (hyperfine)
bash benches/usecase.sh        # 実用ユースケースベンチマーク
```

詳細: [docs/benchmark.md](docs/benchmark.md), [docs/usecase-benchmark.md](docs/usecase-benchmark.md)

> **注意 (UC2 で発見)**: `[.items[].name]` のような generator-in-scalar フィルタは jq-jit ではコンパイルエラー (exit 101) になる。`(.items | map(.name))` に書き換えが必要。この制約は「最大のアーキテクチャ課題」セクションの generator-in-scalar-context 問題に起因する。

## 最大のアーキテクチャ課題

**generator-in-scalar-context**: CPS+callback モデルでは、複数値を返す式（generator）が単一値を期待する位置に出現すると処理できない。FAIL の 8 件と SKIP の多数がこの問題に帰着する。解決にはコードジェネレータの設計変更が必要。

## フィルタキャッシュ (plan-cache) — 完了

コンパイル済みフィルタを `.dylib` としてキャッシュし、2 回目以降は dlopen で即ロード。全 6 Step 完了。

### 完了した Step

| Step | 内容 | 変更ファイル |
|---|---|---|
| **Step 1** | codegen リファクタ — `build_filter<M: Module>` 抽出 | `src/codegen.rs` |
| **Step 2** | リテラル間接参照化 — `LiteralPool` + 4th param | `src/codegen.rs` |
| **Step 3** | `compile_to_shared_object` — ObjectModule → .o → cc -shared → .dylib | `src/codegen.rs`, `Cargo.toml` |
| **Step 4** | `cache.rs` — FilterCache + CachedFilter + リテラル直列化 | `src/cache.rs`, `src/lib.rs` |
| **Step 5** | CLI 統合 — CompiledFilter enum + run() にキャッシュを透過的に組み込み | `src/bin/jq-jit.rs` |
| **Step 6** | CLI フラグ — `--no-cache`, `--clear-cache`, `--cache-dir` | `src/bin/jq-jit.rs` |

### アーキテクチャ変更のポイント

- `build_filter<M: Module>` で JITModule/ObjectModule を抽象化（Step 1）
- リテラルを `iconst(ptr)` → `load(literals_param + idx * 8)` に間接参照化（Step 2）
- `jit_filter` シグネチャ: 3 引数 → 4 引数 (`literals: *const *const Value` 追加)
- `cranelift-object = "0.129"` 依存追加

### テスト結果

全テスト回帰なし: 649 PASS, 1 FAIL (既知), CLI 113 PASS

## AOT コンパイル (plan-aot) — 完了

jq フィルタをスタンドアロン実行バイナリに AOT コンパイルする機能。

### 使用方法

```bash
jq-jit --compile '. + 1' -o add1
echo 3 | ./add1  # → 4
```

### 実装した内容

| 内容 | 変更ファイル |
|---|---|
| compile_to_object — ObjectModule で .o 生成 (jit_filter + main + リテラル埋め込み) | src/codegen.rs |
| aot.rs — AOT バイナリのランタイムエントリポイント (aot_run) | src/aot.rs (新規) |
| staticlib ビルド — libjq_jit.a 生成 | Cargo.toml, build.rs |
| --compile フラグ — フィルタ → .o → cc リンク → 実行バイナリ | src/bin/jq-jit.rs |

### テスト結果

AOT 統合テスト: 10/10 PASS

| # | テスト | 結果 |
|---|---|---|
| 1 | 基本算術 (`. + 1`) | PASS — 出力: 4 |
| 2 | 文字列操作 (`.name \| ascii_upcase`) | PASS — 出力: "HELLO" |
| 3 | 配列フィルタ (`[.[] \| select(. > 2)]`) | PASS — 出力: [3,4,5] |
| 4 | オブジェクト構築 (`{name: .a, value: .b}`) | PASS — 出力: {"name":"x","value":1} |
| 5 | 文字列補間 (`"Hello, \(.name)!"`) | PASS — 出力: "Hello, world!" |
| 6 | ランタイムフラグ (`-c`, `-r`) | PASS — コンパクト出力・rawフラグ動作確認 |
| 7 | 複数出力 (`.[]`) | PASS — 出力: 1, 2, 3 |
| 8 | null 入力 (`-n`) | PASS — 出力: null |
| 9 | 複合パイプライン (map + sort_by) | PASS — 正しいソート結果 |
| 10 | raw 出力フラグ (`.name -r`) | PASS — 出力: hello (引用符なし) |

回帰テスト: 変化なし (649 PASS / 1 FAIL, CLI 113 PASS / 1 FAIL)

性能 (hyperfine, `. + 1`): AOT 2.5ms vs jq-jit 2.6ms vs jq 2.6ms — AOT がわずかに最速 (1.04x vs jq)

## ストリーミング JSON パーサ最適化 (plan-streaming) — 完了

serde_json のストリーミングデシリアライザを導入し、JSON パース性能を大幅に改善。

### 実装した内容

| Step | 内容 | 変更ファイル |
|---|---|---|
| **Step 1** | `src/input.rs` — 共有入力モジュール (`serde_to_value`, `stream_json_values`, `open_input`) | `src/input.rs` (新規) |
| **Step 2** | AOT パスを共有入力モジュールに移行 | `src/aot.rs` |
| **Step 3** | JIT CLI パスを共有入力モジュールに移行 | `src/bin/jq-jit.rs` |
| **Step 4** | ストリーミング実行 — 入力を1件ずつパース・実行 | `src/bin/jq-jit.rs` |
| **Step 5** | `src/output.rs` — 共有出力フォーマットモジュール | `src/output.rs` (新規) |
| **Step 6** | テスト + ベンチマーク + ドキュメント更新 | 各種 |

### アーキテクチャ変更のポイント

- `serde_json::Deserializer::from_reader().into_iter()` によるストリーミングパース (read-all-then-parse → parse-as-you-read)
- JIT CLI / AOT 共通の `input.rs` と `output.rs` でコード重複を排除 (DRY)
- serde_json への統一 (以前は JIT CLI が独自 JSON パーサを使用していた)

### 性能改善

| ベンチマーク | 改善前 | 改善後 | 改善幅 |
|---|---|---|---|
| large_select_10k | 1.13x | **1.68x** | +49% |
| xlarge_select_100k | 0.83x | **1.29x** | +55% (逆転勝利) |
| large_name_10k | 0.69x | **0.95x** | +38% |
| xlarge_length_100k | 0.62x | **0.91x** | +47% |
| xlarge_first_100k | 0.61x | **0.92x** | +51% |

### テスト結果

全テスト回帰なし: 649 PASS / 1 FAIL (既知), CLI 113 PASS / 1 FAIL

## 次のステップ

| プラン | 状態 | ファイル |
|---|---|---|
| **フィルタキャッシュ** | 完了 | [docs/plan/cache.md](docs/plan/cache.md) |
| **AOT コンパイル** | 完了 | [docs/plan/aot.md](docs/plan/aot.md) |
| **ストリーミング JSON パーサ** | 完了 | [docs/plan/streaming.md](docs/plan/streaming.md) |
| **ユースケースベンチマーク** | 完了 | [docs/plan/usecase.md](docs/plan/usecase.md) |

## 将来課題

| 課題 | 概要 | 期待効果 |
|---|---|---|
| **simd-json 導入** | JSON パーサを simd-json に置き換え、SIMD 加速でパース性能を改善 | parse-heavy ワークロード (length, .[0]) で jq を上回る可能性 |

## 開発フロー

### セッション管理

- **開始時**: STATUS.md を読み込んで現在の状態を把握する
- **終了時 / compaction 前**: STATUS.md を最新の状態に更新する

### 作業の流れ

1. **要望**: ユーザーが機能・改善を提案する
2. **プラン作成**: `docs/plan/<名前>.md` を作成し、`docs/plan/README.md`（インデックス）に登録する
3. **実装**: プランに従って作業する
   - Claude は PM として振る舞う。サブエージェントにタスクを委任し、自身のコンテキストを温存する
   - プランに記載されたタスクは必要に応じて随時細分化する
   - タスクの開始・終了時にプラン・STATUS.md・関連ドキュメントを更新する
4. **記録**: 完了時に `docs/plan/<名前>-record.md` を作成する（進捗推移・技術的ハイライト）

## ドキュメント規約

### ディレクトリ構成

```
README.md          プロジェクト概要（外向け）
STATUS.md          セッション状態管理（このファイル）
docs/
├── plan/
│   ├── README.md  開発プラン全体像（インデックス）
│   ├── xxx.md     個別プラン（設計・ステップ・検証方法）
│   └── xxx-record.md  実装記録（プランに対応）
├── benchmark.md   性能測定レポート
└── research/      技術調査アーカイブ
tests/
└── official/analysis.md  テスト FAIL 分析
```

### 命名規則

- **プラン**: `docs/plan/<名前>.md` — 設計・実装ステップ・検証方法を記載
- **実装記録**: `docs/plan/<名前>-record.md` — 対応するプランの実装時の記録（進捗推移・技術的ハイライト）
- **インデックス**: `docs/plan/README.md` — 全プランの関係・状態・推奨順序をまとめる。プラン追加時に更新する

### ドキュメント一覧

| ファイル | 内容 |
|---|---|
| [docs/plan/](docs/plan/) | 開発プラン全体像（インデックス） |
| [docs/plan/roadmap.md](docs/plan/roadmap.md) | Phase 0-10 ロードマップ |
| [docs/plan/roadmap-record.md](docs/plan/roadmap-record.md) | Phase 10-7 の実装記録 |
| [docs/plan/cache.md](docs/plan/cache.md) | フィルタキャッシュ実装プラン |
| [docs/plan/cache-record.md](docs/plan/cache-record.md) | フィルタキャッシュ実装記録 |
| [docs/plan/aot.md](docs/plan/aot.md) | AOT コンパイル実装プラン |
| [docs/plan/aot-record.md](docs/plan/aot-record.md) | AOT コンパイル実装記録 |
| [docs/plan/streaming.md](docs/plan/streaming.md) | ストリーミング JSON パーサ最適化プラン |
| [docs/plan/streaming-record.md](docs/plan/streaming-record.md) | ストリーミング JSON パーサ実装記録 |
| [docs/benchmark.md](docs/benchmark.md) | 性能測定レポート (Criterion + hyperfine) |
| [docs/usecase-benchmark.md](docs/usecase-benchmark.md) | 実用ユースケースベンチマーク結果 |
| [docs/plan/usecase.md](docs/plan/usecase.md) | ユースケースベンチマーク実装プラン |
| `tests/official/analysis.md` | jq 公式テスト FAIL 分析レポート |
| `docs/research/` | 技術調査 (6 documents) |
