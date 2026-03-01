# jq 代替実装のアーキテクチャと実行モデル調査

## 1. jaq (Rust)

GitHub: https://github.com/01mf02/jaq

### 全体アーキテクチャ

jaq は jq の Rust 製再実装で、正確性・速度・簡潔さを重視している。

2025-12-23 に **v3.0.0-beta** がリリースされた。主な変更点:
- `jaq-all` クレートの導入による API 組み込みの簡素化
- `jaq_json::Val` への `serde::Deserialize` 実装
- `Arena` の型定義の簡素化
- 文字列スライス更新、オブジェクトベースのインデックスなど新機能追加
- v3.0.0-alpha（2025-11-05）で導入された包括的マニュアル（約 500 のテスト済み例題）、マルチフォーマット対応（JSON/YAML/CBOR/TOML/XML）、XJON（JSON のスーパーセット）を継承

なお、v3.0 系ではコンパイル・実行・ネイティブフィルタ API に破壊的変更がある。ただし、実行モデル自体は AST インタプリタのままであり、バイトコード VM への移行はない。

ワークスペースは 7 つのクレートで構成される。

| クレート | 役割 |
|---|---|
| `jaq-core` | コンパイラ・実行エンジン・値トレイト定義 |
| `jaq-std` | 標準ライブラリ（jq 組み込み関数の定義） |
| `jaq-json` | JSON 値型の具体実装（`Val` enum） |
| `jaq-fmts` | YAML/CBOR/TOML/XML フォーマット対応 |
| `jaq-all` | 全クレートの集約 |
| `jaq` | CLI バイナリ |
| `jaq-play` | WebAssembly ベースの対話プレイグラウンド |

外部依存は極めて少ない。`jaq-core` の依存は `dyn-clone`、`once_cell`、`typed-arena` の 3 つのみ。

### パイプライン: パース → コンパイル → 実行

```
jq ソースコード
  → Parser (jaq-std: jq 文法の解析)
  → Compiler (jaq-core/src/compile.rs)
    → Term enum (AST/IR)
      → Filter<D> (コンパイル済みフィルタ + ルックアップテーブル)
  → Execution (filter.id.run((ctx, input)))
    → Iterator<Item = ValR> (結果ストリーム)
```

コンパイルは 2 パスで行われる:
1. **`def_pre`**: 全定義のプレースホルダを作成（再帰関数対応）
2. **`def_post`**: 各定義の本体をコンパイル

末尾再帰の最適化もコンパイル時に行われる。

### 実行モデル: Iterator ベースの AST インタプリタ

**jaq はバイトコード VM を持たない。** コンパイル済み AST（`Term` enum）を直接再帰的に評価する tree-walking interpreter。

各 `Term` ノードは `run(cv)` メソッドを持ち、`(Ctx, Value)` を受け取って `BoxIter<ValR>` を返す。

#### ジェネレータの実装

- **`Comma` (`,`)**: 左辺の Iterator と右辺の Iterator を `.chain()` で連結
- **`Pipe` (`|`)**: 左辺の各出力に対して右辺を `.flat_map()` で適用
- **`BoxIter<'a, T>`**: `Box<dyn Iterator<Item = T> + 'a>` — 型消去による動的ディスパッチ

重要な最適化として `next_if_one()` 関数がある:

> "This is one of the most important functions for performance in jaq. It enables optimisations for the case when a filter yields exactly one output, which is very common in typical jq programs."

`size_hint()` で出力が 1 つだけと判定できる場合、コンテキストの clone を回避し、単一値パスで処理する。

### 値の表現

```rust
pub enum Val {
    Null,
    Bool(bool),
    Num(Num),
    BStr(Box<Bytes>),      // バイナリ文字列
    TStr(Box<Bytes>),      // UTF-8 テキスト文字列
    Arr(Rc<Vec<Val>>),     // 配列
    Obj(Rc<Map<Val, Val>>), // オブジェクト
}
```

x86_64 で **16 バイト**。jq の `jv` と同サイズ。

### エラーハンドリング

jaq のエラーハンドリングは Rust の `Result` 型を基盤とする。

- **`ValX<T, V>`**: フィルタの出力型。`Result<T, Exn<V>>` として成功値またはエラーを返す
- **`Exn`**: 例外型。内部に `exn::Inner::Err(e)` としてエラー値をラップ
- **`try-catch`**: AST の `TryCatch(f, c)` ノードとして表現。`try_catch_run()` 関数がフィルタ `f` の出力を監視し、`Err(Exn(exn::Inner::Err(e)))` を検出すると catch ハンドラ `c` にエラー値を渡して回復
- **`?` 演算子**: `try-catch` の糖衣構文として実装。catch ハンドラが `empty`（出力なし）に設定される
- **エラー伝播**: フィルタチェイン内では `map(|v| Ok(...))` / `map_err(Exn::from)` パターンでエラーが伝播。`?` による短絡評価ではなく、Iterator の各要素レベルで `Result` を処理する

バックトラック方式の jq/gojq とは異なり、jaq ではエラーは値レベルの `Result` として Iterator 内を流れる。

### メモリ管理

- **`Rc<T>`** をデフォルトで使用（単一スレッド参照カウント）
- `sync` feature フラグで **`Arc<T>`** に切り替え可能
- 配列・オブジェクトのみ参照カウント。スカラー値は値渡し
- `typed-arena` を使用（コンパイル時のメモリ管理）

### パフォーマンス特性

AMD Ryzen 5 5500U での jaq-2.3 vs jq-1.8.1 vs gojq-0.12.17:

| ベンチマーク | n | jaq | jq | gojq | 最速 |
|---|---|---|---|---|---|
| empty (起動) | 512 | 330ms | 440ms | 290ms | gojq |
| sort | 1M | **100ms** | 450ms | 540ms | jaq |
| group_by | 1M | **340ms** | 1750ms | 1540ms | jaq |
| unique_by | 1M | **340ms** | 1180ms | 1070ms | jaq |
| min_by, max_by | 1M | **230ms** | 350ms | 600ms | jaq |
| add (strings) | — | **230ms** | 510ms | 860ms | jaq |
| add (arrays) | — | **460ms** | 560ms | 1130ms | jaq |
| reduce | 1M | **720ms** | 850ms | N/A | jaq |
| ascii_downcase | 1M | **370ms** | 690ms | 640ms | jaq |
| split | 1M | **400ms** | 530ms | 1120ms | jaq |
| test (regex) | 1M | **460ms** | 1020ms | 510ms | jaq |
| tree-flatten | 17 | 750ms | 340ms | **0ms** | gojq |
| tree-update | 17 | 1070ms | **290ms** | 340ms | jq |
| tree-paths | 17 | **340ms** | 810ms | 370ms | jaq |

**33 ベンチマーク中 23 で jaq が最速。**

---

## 2. gojq (Go)

GitHub: https://github.com/itchyny/gojq

### 全体アーキテクチャ

gojq は jq の Pure Go 実装。パイプラインは 3 段階:

```
jq ソースコード → Parser (goyacc) → Query AST → Compiler → Code (バイトコード) → Executor → Iter
```

### 実行モデル: バイトコード VM + 明示的フォーク

**gojq はバイトコード VM を採用。** goroutine/channel は使っていない。

開発の経緯（作者ブログより）:
1. 初期実装: goroutine + channel → **遅すぎて abandon**
2. リライト: スタックマシンベースのインタプリタに書き換え
3. 末尾呼び出し最適化の導入

#### オペコード（30 命令）

`nop`, `push`, `pop`, `dup`, `const`, `load`, `store`, `object`, `append`, `fork`, `forktrybegin`, `forktryend`, `forkalt`, `forklabel`, `backtrack`, `jump`, `jumpifnot`, `index`, `indexarray`, `call`, `callrec`, `pushpc`, `callpc`, `scope`, `ret`, `iter`, `expbegin`, `expend`, `pathbegin`, `pathend`

ジェネレータの実現は jq と同じ **フォーク/バックトラック方式**。

#### try-catch の動作モデル

gojq の `try-catch` は `forktrybegin` / `forktryend` オペコードで実装される。

1. **`forktrybegin`**: フォークポイントを push して try ブロックを開始。通常実行時は catch 先のアドレスを記録してフォークスタックに積む
2. **try ブロック内でエラー発生**: バックトラックが発動し、`forktrybegin` に戻る
3. **`forktrybegin` (backtrack 時)**: エラーの種別を判定:
   - `ValueError`: エラー値を取り出してスタックに push し、catch ブロックへ遷移
   - その他のエラー: エラーメッセージ文字列をスタックに push し、catch ブロックへ遷移
   - `breakError` / `HaltError`: try で捕捉せずそのまま上位へ伝播
4. **`forktryend`**: try ブロックの正常完了時はフォークポイントを push。バックトラック時にエラーがあれば `tryEndError` でラップして try ブロックの外へ伝播

`?` 演算子は `try-catch` の catch ハンドラが `empty` の場合と同等。

### 値の表現

Go の `any` (= `interface{}`) ベース。`*big.Int` による任意精度整数演算をサポート。

### パフォーマンス特性

- **起動時間**: gojq が最速（290ms @512 回）
- **33 ベンチマーク中 3 で最速** (empty, tree-flatten, indices)
- **tree-flatten で圧倒的に速い**（0ms vs jaq 750ms, jq 340ms）

---

## 3. jq 本体の VM

### ボトルネック分析

#### 1. 分岐予測を阻害するパターン

- **巨大 switch 文**: 約 70 の case。間接分岐は CPU の分岐予測器に難しい
- **バックトラック分岐**: 毎命令で `backtracking` フラグの条件分岐
- **`goto do_backtrack`**: 通常パスから外れるため予測ミスが頻発

#### 2. メモリアロケーション頻度

- **`jv` のアロケーション**: 文字列・配列・オブジェクトの生成で毎回 `jv_mem_alloc()`
- **フォークポイントの生成**: バックトラックのたびに push
- **フレームの push/pop**: 関数呼び出しのたびに生成

#### 3. 参照カウントのオーバーヘッド

- `jv_copy()`: refcount++。全ての値の複製で呼ばれる
- `jv_free()`: refcount--。0 になったら解放
- `jvp_refcnt_unshared()`: COW 判定。インプレース更新の前に毎回チェック

個々には軽量だが、jq のフィルタは値の生成・複製・破棄が極めて頻繁なため、累積的な影響は大きい。

---

## 4. 比較分析

### 基本情報

| 観点 | jq | jaq | gojq |
|---|---|---|---|
| 言語 | C | Rust | Go |
| 実行モデル | バイトコード VM | AST インタプリタ | バイトコード VM |
| ジェネレータ | FORK/BACKTRACK | Iterator チェイン | fork/backtrack |
| 命令数 | 43 (+ ON_BACKTRACK ハンドラ 11 個) | N/A | 30 |
| 末尾呼び出し最適化 | あり | あり | あり |
| ライセンス | MIT-like (jq license) | MIT | MIT |

### 値の表現

| 観点 | jq | jaq | gojq |
|---|---|---|---|
| 型 | `jv` (tagged union) | `Val` enum | `any` (interface{}) |
| サイズ | 16 bytes | 16 bytes | 16 bytes |
| Number | `double` (inline) | `Num` | `int`/`float64`/`*big.Int` |
| Array | refcounted `jv[]` | `Rc<Vec<Val>>` | `[]any` |
| Object | refcounted hash table | `Rc<Map<Val, Val>>` | `map[string]any` |

### メモリ管理

| 観点 | jq | jaq | gojq |
|---|---|---|---|
| 方式 | 参照カウント (手動) | Rc (Rust の半自動) | GC (Go ランタイム) |
| COW | あり | なし (clone 回避で代替) | N/A |
| オーバーヘッド | 毎操作の incref/decref | clone 時のみ | GC 停止時間 |

### モジュールシステム

| 観点 | jq | jaq | gojq |
|---|---|---|---|
| 対応状況 | フル対応 | v2.0 で対応（alpha） | フル対応 |
| `import` / `include` | 両方対応 | 両方対応（ネストされたモジュール読み込みに制限あり） | 両方対応 |
| `$ENV` / メタデータ | `modulemeta` オペコードで対応 | 対応 | `module` 文でメタデータ定義可能 |
| モジュールパス | `~/.jq`, `-L` オプション | `-L` オプション | `~/.jq`, `$ORIGIN/../lib/gojq`, `-L` オプション |
| ライブラリ利用時 | N/A | API 経由でモジュールローダーをカスタマイズ可能 | `NewModuleLoader` でカスタマイズ可能 |

### ジェネレータ実装の比較

| 特性 | jq | jaq | gojq |
|---|---|---|---|
| 方式 | 明示的フォーク + バックトラック | Rust Iterator チェイン | 明示的フォーク + バックトラック |
| 状態保存 | forkpoint (スタック位置全体) | なし (Iterator が状態を内包) | forks 配列 |
| メモリコスト | フォークポイントごとにスナップショット | Iterator オブジェクト (Box) | フォークポイントごとにスナップショット |

---

## 5. JIT 設計へのインプリケーション

### 各実装から学べる設計判断

**jaq から:**
1. `next_if_one()` — 出力 1 つのフィルタを検出してスカラーパスに特殊化すれば、フォーク/バックトラックを完全に排除可能
2. `ValT` トレイト — 値型の抽象化は JIT でも有用（unboxed/boxed の切り替え）
3. 末尾再帰のコンパイル時検出 — JIT のコンパイルパスでも使える
4. `typed-arena` — JIT コンパイルの中間表現管理に適用可能

**`next_if_one()` の JIT 応用 — スカラーパス特殊化:**

jaq の `next_if_one()` は `size_hint()` で出力が正確に 1 つと判定できる場合にコンテキストの clone を回避する最適化だが、JIT ではこれをさらに強力に応用できる:

- **静的解析によるスカラー判定**: `.foo`, `.bar`, `length`, `type`, 算術演算、比較演算などは常に出力が 1 つ。JIT コンパイル時にこれらを静的に検出し、フォークポイントの生成・バックトラック機構を完全にバイパスするネイティブコードを生成できる
- **スカラーパス専用コード生成**: スカラーフィルタの連鎖（`.foo | . + 1 | tostring`）に対して、Iterator のオーバーヘッドなしに直接値を受け渡すレジスタベースのコードを生成可能
- **ガード付き特殊化**: 実行時に出力数が 1 つであることを検証するガードを挿入し、ガード成功時はスカラーパス、失敗時はフォールバックのジェネレータパスに分岐する。これはトレースベース JIT の type guard と同じ発想
- **パイプライン全体の最適化**: スカラーフィルタが連続する区間を検出し、その区間全体を単一の関数呼び出しに畳み込むことで、中間値の Box アロケーションを排除

**gojq から:**
5. goroutine/channel の失敗例 — OS レベルのコンテキストスイッチは重すぎる。明示的な状態マシンが正解
6. バイトコード → ネイティブ変換の親和性 — 30 命令と小さいオペコードセットは JIT テンプレートの参考になる

**jq 本体から:**
7. FORK/BACKTRACK がパフォーマンスの中心 — スカラーパスの検出、インライン展開、フォークポイントの軽量化が鍵
8. ON_BACKTRACK() — 通常パスのみ生成し、バックトラックパスは lazy にコンパイルする設計の根拠
9. 16 バイト jv — JIT でも値表現は 16 バイト以内に。2 レジスタに収まる

### JIT に適用できるアイデア

1. **型特殊化**: `.price > 100` → 浮動小数点比較命令に直接変換
2. **トレースベース JIT**: ホットなバイトコードシーケンスをトレース → ネイティブコード化
3. **インライン化とエスケープ解析**: 小さな定義をインライン展開し、jv のアロケーションを排除
4. **Region-based メモリ管理**: パイプラインの各段階で arena を割り当て、一括解放
5. **SIMD ベースの値操作**: jv が 16 バイトなので SSE/NEON の 128 ビットレジスタに載る

### 避けるべきアンチパターン

1. **goroutine/channel ベースのジェネレータ** — gojq が証明した通り遅すぎる
2. **全命令の即時 JIT コンパイル** — ティアリング方式が妥当
3. **バックトラック機構の完全排除** — セマンティクスに依存。ホットパスの特殊化で迂回
4. **GC の導入** — 参照カウント + arena のほうがレイテンシ予測しやすい
5. **VM を捨てる** — JIT の出発点としてはバイトコード VM がネイティブコード変換に適している
