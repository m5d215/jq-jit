# Phase 10-7: jq 公式テストスイート対応の記録

> 2026-02-28 実施。jq-jit の jq 公式テストスイート (510テスト) への対応を 1 セッションで 29.2% → 67.4% に引き上げた記録。

## プロジェクト概要

jq-jit は jq のバイトコード VM をネイティブコードに JIT コンパイルするプロジェクト。Rust + Cranelift で実装し、libjq 比で最大 19,000 倍の高速化を達成している。

Phase 0（技術調査）から Phase 10（完全互換）まで段階的に開発を進め、Phase 10-7 で jq 公式テストスイートによる互換性検証を実施した。

## セッション開始時の状態

- 自作の differential テスト: 648 PASS, 0 FAIL
- CLI 互換テスト: 114 PASS
- jq 公式テストスイートは未導入

## 作業フロー

1 人間 + Claude Code (Opus) によるペアプログラミング。人間は方針決定と承認、Claude はサブエージェントへの委任と統合を担当。

各バッチの修正はサブエージェント (general-purpose agent, bypassPermissions) に委任し、結果を検証してから次のバッチに進めた。

## 進捗推移

| Batch | 内容 | PASS | 増分 | SKIP除外率 | 所要時間(目安) |
|---|---|---|---|---|---|
| 0 | テストスイート導入・初回実行 | 149/510 (29.2%) | — | — | 10min |
| 1 | ランタイム基本修正 (abs, flatten, join, split, div/0, string*num 等) | 193 (+44) | +44 | — | 35min |
| 2 | try/catch/error セマンティクス、? 演算子、.[] エラー伝播 | 233 (+40) | +40 | — | 48min |
| 3 | ソート深い比較、Unicode index、エラーメッセージ互換化、from_entries | 259 (+26) | +26 | — | 40min |
| 4 | Range Cartesian product、代入 (= vs \|=) 区別、変数スコープ (LOADV level)、//= | 298 (+39) | +39 | 91.1% | 85min |
| 5 | スライス代入、.[] \|= try expr、if error 伝播 | 303 (+5) | +5 | 92.7% | 62min |
| 6 | 数学関数 22個 + bsearch | 309 (+6) | +6 | 92.8% | 15min |
| 7 | transpose, trim, datetime, obj*obj再帰マージ, 非空オブジェクトリテラル, utf8bytelength, toboolean | 344 (+35) | +35 | 93.5% | 27min |
| **合計** | | **344/510 (67.4%)** | **+195** | **93.5%** | **~5h** |

## 最終結果

```
jq 公式テストスイート: 344 PASS, 24 FAIL, 142 SKIP / 510 total
  全体 PASS 率: 67.4%
  SKIP 除外 PASS 率: 93.5% (344/368)

Differential tests: 649 PASS, 1 FAIL (既知の libjq 差異)
CLI compatibility: 114 PASS
```

## 修正した主要バグ（技術的ハイライト）

### 1. error() 引数伝播

jq の `error("foo")` は引数をエラー値として投げ、`catch` がそれを受け取る。jq-jit は引数を無視して汎用 "error" 文字列を投げていた。

`UnaryOp::MakeError` (値 → Error) と `rt_extract_error` (Error → 値) を追加し、catch ブロックでエラー値を復元するようにした。

### 2. ? 演算子の desugar

`.foo?` は「エラーなら何も出力しない」というセマンティクス。jq-jit は null を返していた。

`INDEX_OPT` opcode を `TryCatch(Index, Empty)` に desugar する方式に変更。エラー時は Empty（0出力）になり、正しく抑制される。

### 3. 変数スコープ解決 (LOADV level)

jq のバイトコードは変数を `(level, index)` の 2 次元アドレスで参照する。level=0 は現スコープ、level=1 は親スコープ。jq-jit は level を完全に無視していた。

`var_index = scope_depth * 1000 + raw_index` でスコープ付き変数 ID を生成する方式に修正。これにより `//=` 演算子を含む多くのクロージャパターンが正しく動作するようになった。

### 4. 代入演算子 (= vs |=) の区別

jq の `.foo = .bar` は RHS を元の入力全体に対して評価する（plain assign）。`.foo |= expr` は RHS を LHS の現在値に対して評価する（update）。jq-jit は両方を update として処理していた。

`is_plain_assign` フラグを `Expr::Update` に追加し、plain assign の場合は RHS の入力を切り替えるようにした。

### 5. Range の Cartesian product 展開

`range(0,1;3,4)` のようにビルトインの引数がジェネレータ（複数値）の場合、全ての組み合わせで呼び出す必要がある。codegen に Comma 式の展開ロジックを追加した。

### 6. 非空オブジェクトリテラル

jq のバイトコードは `{a: 1, b: 2}` のような非空オブジェクトリテラルを定数として保持することがある。jq-jit は空オブジェクト `{}` しか定数化できなかった。`Literal::Obj(BTreeMap)` を追加し、codegen で実体化するようにした。これだけで 15 テストが一気に通った。

## 残り 24 FAIL の内訳

| カテゴリ | 件数 | 根本原因 |
|---|---|---|
| generator-in-scalar (exit 101) | 8 | CPS+callback モデルの構造的制約 |
| path() 操作 | 6 | path 追跡が select/map と連携しない |
| try/catch 優先順位 | 3 | bytecode レベルの構造問題 |
| 代入/更新の深いバグ | 3 | variable binding in path |
| label/break 未実装 | 1 | LABEL/BREAK opcode |
| f64 精度/フォーマット | 2 | 任意精度数値 (have_decnum) 未対応 |
| その他 | 1 | format edge case |

## 残り 142 SKIP の主要カテゴリ

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

## 技術スタック

- **言語**: Rust
- **JIT backend**: Cranelift v0.129
- **入力**: jq bytecode via libjq FFI (jq 1.8.1)
- **実行モデル**: CPS + callback（バックトラッキング → 通常の関数呼び出し）
- **値表現**: 16-byte `#[repr(C, u64)]` enum + Rc
- **追加依存**: regex, libm, libc

## 所感

- jq のバイトコードは想像以上に複雑で、特に generator（複数値を返す式）のセマンティクスが CPS+callback モデルと根本的に噛み合わない箇所がある
- 公式テストスイートの導入が品質向上に直結した。自作テストでは見つからなかったバグが大量に発覚
- サブエージェントへの委任モデルは効率的だったが、同じファイルを複数エージェントが触るとコンフリクトするため、逐次実行が基本になった
- 「簡単な修正」を先にやることで PASS 率を素早く上げ、モチベーションを維持できた
- 残り 24 FAIL と 142 SKIP の大半は generator-in-scalar-context という1つの根本問題に帰着する。これを解決するにはコードジェネレータの設計を見直す必要がある
