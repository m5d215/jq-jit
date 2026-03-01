# jq-jit Project Plan

## 概要

jq のバイトコード VM をネイティブコード JIT に置き換える。
段階的に進め、各フェーズで動くものを作りながら知見を積み上げる。

## 設計方針

Phase 0 の調査で確定。詳細は [research/design-decisions.md](../research/design-decisions.md)。

| 判断ポイント | Phase 1 方針 | アップグレードパス |
|---|---|---|
| バイトコード取得 | libjq FFI (手動 `#[repr(C)]`) | 独自 Rust パーサ（必要になった場合のみ） |
| 値の表現 | 16 バイト enum + Rc | NaN boxing（数値ホットパス向け） |
| メモリ管理 | 参照カウント (Rc) | Arena（配列ビルダー等）、refcount elision |
| バックトラッキング | CPS + コールバック | スタック深度ガード + トランポリン |
| コンパイル単位 | サブファンクション単位 | 選択的インライン化 |

---

## Phase 0: 技術調査 ✅ 完了

### 成果物

| ファイル | 内容 |
|---|---|
| [research/jq-bytecode.md](../research/jq-bytecode.md) | 43 opcodes の完全な一覧・セマンティクス、3 スタックモデル、バイトコード例 11 パターン |
| [research/backtracking.md](../research/backtracking.md) | forkpoint 機構、Prolog/Icon/Python/Lua 比較、7 戦略評価 → CPS + Cranelift 推奨 |
| [research/cranelift.md](../research/cranelift.md) | v0.129 API、FunctionBuilder パターン、try_call、メモリ管理、デバッグ手法 |
| [research/existing-impls.md](../research/existing-impls.md) | jaq/gojq/jq アーキテクチャ比較、ボトルネック分析、JIT 設計への示唆 |
| [research/jit-backends.md](../research/jit-backends.md) | 6 バックエンド比較 → Cranelift 推奨、Tiered compilation 戦略 |
| [research/design-decisions.md](../research/design-decisions.md) | 5 設計判断の分析・Phase 1 方針・検証項目 |

---

## Phase 1: Minimal JIT（算術フィルタのみ）✅ 完了

最小スコープで「バイトコード → ネイティブ実行」のパイプラインを通す。

### スコープ

- libjq FFI: `jq_compile()` → `struct bytecode*` を Rust から読む
- 対応 opcode: `TOP`, `LOADK`, `PUSHK_UNDER`, `DUP`, `POP`, `RET`, `SUBEXP_BEGIN`, `SUBEXP_END`, `CALL_BUILTIN` (`_plus`, `_minus`, `_multiply`, `_divide`, `_mod`), `INDEX` (`.foo`)
  - 注: jq の算術演算は専用 opcode (`PLUS` 等) ではなく `CALL_BUILTIN` 経由で実装されている（jq 1.8.1 で確認）
- 値型: `Value` enum `#[repr(C, u64)]` (Null, Bool, Num, Str, Arr, Obj) — 16 バイト
- Cranelift: 1 関数の JIT コンパイル + 実行。Value はポインタ渡し方式

### 判定基準 — すべて達成

- パイプラインが通ること（`jq '. + 1'` 相当がネイティブで動く）✅
- jq と同じ結果が返ること（differential testing）✅ 14 テスト全 PASS
- 速度はまだ気にしない

### サブタスク

- [x] 1-1. Rust プロジェクト scaffold (`cargo init`, Cranelift 0.129.1, libjq 1.8.1 dylib)
- [x] 1-2. libjq FFI バインディング (手動 `#[repr(C)]` 定義, safe wrapper, disassembler)
- [x] 1-3. バイトコード → CPS IR 変換（算術 opcode のみ）
- [x] 1-4. CPS IR → Cranelift CLIF 生成
- [x] 1-5. JIT 実行 + jq との differential testing (14 テスト)
- [x] 1-6. 動作確認 & Phase 1 振り返り

### Phase 1 で検証した設計仮説

1. ~~`bindgen` で jq ヘッダを処理できるか？~~ → 手動定義を選択。`bytecode.h` が非公開 + マクロが多い
2. ~~16 バイト `Value` の Cranelift calling convention~~ → ポインタ渡し方式で回避。直接渡しは Phase 6 最適化で検討
3. `extern "C"` コールバックのオーバーヘッド → Phase 1 では callback 未使用。Phase 3 で検証
4. JIT コンパイル時間 → 体感瞬時。Phase 6 で正式計測
5. ~~libjq を differential testing の oracle に使えるか~~ → 使える。`jq_runner.rs` で実装済み

### コード構成 (2,843 行)

| ファイル | 行数 | 役割 |
|---|---|---|
| `src/jq_ffi.rs` | 464 | FFI 型定義 (Jv, Bytecode, Opcode, extern 宣言) |
| `src/bytecode.rs` | 405 | Safe wrapper (JqState, BytecodeRef, disassembler) |
| `src/cps_ir.rs` | 157 | CPS 中間表現 (Expr, Literal, BinOp) |
| `src/compiler.rs` | 340 | バイトコード → CPS IR 変換 |
| `src/value.rs` | 344 | Value 型 + JSON 変換 |
| `src/runtime.rs` | 124 | JIT 用ランタイムヘルパー (rt_add 等) |
| `src/codegen.rs` | 402 | CPS IR → Cranelift CLIF 生成 |
| `src/jq_runner.rs` | 88 | libjq 経由の jq 実行 |
| `src/main.rs` | 519 | テスト (scaffold + FFI + IR + JIT + diff) |

---

## Phase 2: データ型と演算子拡張

Phase 1 の算術フィルタを拡張し、比較演算子・型操作・複合型の演算をサポート。
すべて `CALL_BUILTIN` パターンで、新しい opcode は不要。

### スコープ

**新規ビルトイン**:
- 比較: `_equal`, `_notequal`, `_less`, `_greater`, `_lesseq`, `_greatereq` (nargs=3)
- 型: `type` (nargs=1), `length` (nargs=1), `tostring` (nargs=1), `tonumber` (nargs=1)
- 配列/文字列: `keys` (nargs=1) — Phase 2 scope は基本のみ

**型の拡張**:
- `rt_add` を Str+Str (連結), Arr+Arr (連結) に拡張
- `rt_index` を Arr[Num] に拡張（Phase 1 で対応済みなら確認のみ）
- null/bool/string リテラルの IR → codegen 対応

**除外** (後続フェーズ):
- `.[]` (ジェネレータ → Phase 4)
- `not` (CALL_JQ + JUMP_F → Phase 3)
- `split`, `join` 等の文字列関数 (Phase 5)

### サブタスク

- [x] 2-1. IR + compiler + runtime 拡張 (比較演算子, 1引数ビルトイン, 型拡張)
- [x] 2-2. codegen 拡張 + null/bool/string リテラル対応
- [x] 2-3. differential testing (24 テスト追加、累計 38 PASS)

---

## Phase 3: 制御フローと関数 ✅ 完了

Phase 1-2 は「1 入力 → 1 出力、分岐なし」の直線的モデル。Phase 3 で制御フローを導入する。

### アーキテクチャ変更

- **CPS IR**: 現在の `Expr` 式木は値を返す式のみ。条件分岐 (`if`) やパイプ (`|`) を表現するノードの追加が必要
- **Cranelift コード生成**: 単一の基本ブロックから、複数ブロック + `brif` 分岐に拡張
- **CALL_JQ opcode**: サブファンクション (bytecode.subfunctions[]) を別の JIT 関数としてコンパイルし、呼び出す
- **パイプ `|`**: 1→1 パイプは式木の合成で済む可能性あり。ジェネレータ (Phase 4) まで CPS callback は不要かもしれない — 要調査

### スコープ

- `if-then-else` (`JUMP_F` opcode + 分岐)
- パイプ `|` (1→1 のケースのみ。ジェネレータパイプは Phase 4)
- `not` (Phase 2 から先送り: `CALL_JQ` + `JUMP_F`)
- `def` + `CALL_JQ` (サブファンクション呼び出し)
- `try-catch` (`TRY_BEGIN` / `TRY_END` opcode、Cranelift の `try_call`)

### 調査結果（2026-02-28）

- **パイプ `|`**: 1→1 パイプは式木合成で実装済み。新ノード不要。ジェネレータパイプは Phase 4
- **`not`**: jq では `if . then false else true end` のサブファンクション。`CALL_JQ` + `JUMP_F` を使う
- **if-then-else バイトコード**: `DUP` → 条件式 → `JUMP_F else` → then 式 → `JUMP end` → `POP` → else 式 → `RET`
- **try-catch バイトコード**: `TRY_BEGIN catch` → try 式 → `TRY_END` → `JUMP end` → catch 式 → `RET`
- **CALL_JQ**: `[nargs] [level] [fidx]` — `bytecode.subfunctions[fidx]` からサブファンクション取得

### サブタスク

- [x] 3-1. IR 拡張: `IfThenElse`, `TryCatch` ノードを CPS IR に追加
- [x] 3-2. Compiler + Codegen: `if-then-else` 対応（JUMP_F, JUMP → Cranelift 複数ブロック + brif）
- [x] 3-3. Compiler + Codegen: `CALL_JQ` + サブファンクション再帰コンパイル
- [x] 3-4. `not` 対応（3-2 + 3-3 の組み合わせ）
- [x] 3-5. Compiler + Codegen: `try-catch` 基本実装（TRY_BEGIN / TRY_END）
- [x] 3-6. Differential testing: Phase 3 全機能のテスト追加 (25 テスト、累計 92 PASS)

---

## Phase 4: ジェネレータとバックトラッキング ✅ 完了

CPS 変換の本領発揮。Phase 0 の backtracking.md 調査を元に実装。

### アーキテクチャ変更

**実行モデルの転換**: 1 入力 1 出力 → 1 入力 N 出力（CPS + コールバック）

```
// Before (Phase 1-3):
fn jit_filter(input: *const Value, output: *mut Value)

// After (Phase 4):
fn jit_filter(input: *const Value, callback: extern "C" fn(*const Value, *mut u8), ctx: *mut u8)
```

- 1→1 フィルタ: callback を 1 回呼ぶ（既存動作を保持）
- ジェネレータ: callback を N 回呼ぶ（`,`, `.[]` 等）
- `empty`: callback を呼ばない（0 出力）
- パイプ `|`: 左の callback 内で右を実行（ネスト）
- バックトラッキングが消滅し、通常のループ + 関数呼び出しに変換

### スコープ

- 実行モデル移行: callback-based（既存 92 テスト維持）
- `,` (カンマ = 複数出力 → CPS: 左右の callback を順に呼ぶ)
- `empty` (= callback を呼ばない)
- `.[]` (配列/オブジェクトの展開 → ループ + callback)
- `select` (if-then-else + empty の組み合わせ)
- `foreach`, `reduce`, `limit`
- `?//` (alternative operator)

### サブタスク

- [x] 4-1. 実行モデル移行: callback-based (既存 92 テスト維持)
- [x] 4-2. `,` (カンマ) + `empty` — 基本ジェネレータ (18 テスト追加、累計 110)
- [x] 4-3. `.[]` — 配列/オブジェクト展開 (20 テスト追加、累計 130)
- [x] 4-4. 変数 (STOREV/LOADV) + `select` (14 テスト追加、累計 144)
- [x] 4-5. `reduce` / `foreach` (12 テスト追加、累計 156)
- [x] 4-6. 包括テスト — 組み合わせ・エッジケース (18 テスト追加、累計 174)

### 未対応 (Phase 5+ で検討)

- `limit` — 未実装
- `?//` (alternative operator) — 未実装
- `try (generator) catch expr` — ジェネレータ全体をラップする try-catch のセマンティクス
- `reduce (generator | select(f)) as $x (...)` — reduce source 内のジェネレータ+select
- `[expr]` 配列コンストラクタ — APPEND opcode が必要

---

## Phase 5: 標準ライブラリ ✅ 完了

jq の builtin 関数群を JIT 対応。多くの jq 関数は jq 自身で定義され libjq がバイトコードにインライン展開するため、不足する opcode とランタイムヘルパーの追加が中心。

### 実装済み機能

**配列コンストラクタ + map**:
- `[expr]` — APPEND opcode + Collect IR ノード
- `map(f)` — CALL_JQ nargs=1 generic パターン + クロージャバインディング

**演算子**:
- `//` (alternative operator) — FORK パターン検出 + Alternative IR ノード

**ビルトイン関数 (C builtin — CALL_BUILTIN)**:
- 文字列: `split`, `startswith`, `endswith`
- 配列: `sort`
- オブジェクト: `keys_unsorted`, `has`
- 数学: `floor`, `ceil`, `round`, `fabs`
- その他: `explode`, `implode`

**jq-defined 関数 (ランタイム直接実装)**:
- `add`, `reverse`, `unique`, `to_entries`, `from_entries`, `ascii_downcase`, `ascii_upcase`

### 未対応 (Phase 6+ で検討)

- オブジェクト構築 `{key: value}`
- `values` (ジェネレータ、level>0 CALL_JQ)
- `join` (CALL_JQ nargs=1 引数あり)
- `map(select(f))` (ネスト高階関数のクロージャ解決)
- toplevel の Value::Error フィルタリング (jq はエラーを stderr にスキップ)
- Tier 1 残り: `sort_by`, `group_by`, `unique_by`, `flatten`, `first`, `last`, `nth`, `range`, `indices`, `limit`, `in`, `with_entries`, `ltrimstr`, `rtrimstr`, `test`, `sqrt`, `pow`, `log`, `nan`, `infinite`, `any`, `all`, `min`, `max`, `min_by`, `max_by`, `ascii`, `env`, `builtins`
- Tier 2 全般

### サブタスク

- [x] 5-1. 配列コンストラクタ `[expr]` — APPEND opcode + map(f) (11 テスト追加、累計 185)
- [x] 5-2. `//` (alternative) + C ビルトイン 13 関数 (23 テスト追加、累計 208)
- [x] 5-3. jq 定義関数 7 関数のランタイム直接実装 (33 テスト追加、累計 249)
- [x] 5-4. 包括テスト + バグ修正 (50 テスト追加、累計 294)

### 判定基準 — すべて達成

- 配列コンストラクタ `[expr]` と `map(f)` が動作 ✅
- `//` (alternative operator) が動作 ✅
- 主要ビルトイン関数が jq と同一結果を返す ✅
- 実用的なフィルタ組み合わせの differential test が PASS ✅
- 累計 294 テスト全 PASS (+ 13 SKIP)

---

## Phase 6: 統合と最適化 ✅ 完了

### スコープ

1. **ベンチマーク基盤** — jq / jaq / gojq との性能比較。ホットスポット特定
2. **即効性のある最適化** — ベンチマーク結果に基づく改善
3. **型特殊化** — `.price > 100` → f64 比較命令直接生成
4. **refcount elision** — 静的解析で不要な clone/drop を除去
5. **Tiered compilation** — Tier 0 インタプリタ → Tier 1 single_pass → Tier 2 regalloc

### サブタスク

- [x] 6-1. ベンチマーク基盤の構築と初期計測 — Criterion ベンチ 7 グループ、JIT vs libjq 比較
- [x] 6-2. ボトルネック特定と即効性のある最適化 — rt_iter_prepare (O(n^2)→O(n)) + fast_num_bin/cmp (4-7% 改善)
- [x] 6-3. 型特殊化 — TypeHint 導入 + BinOp 3 段階特殊化 (条件分岐 -26.4%, select+演算 -33.4%)
- [x] 6-4. 包括テスト + Phase 6 完了 — 27 テスト追加 (累計 320)、Phase 6 総合改善確認

### 初期計測結果 (Phase 6-1)

スカラー演算で **16,889x**、ジェネレータ (100 要素) で **300-400x**、大規模データ (10K) で **58-62x** の高速化を確認。
主要ボトルネック: Rc refcount、BTreeMap nth()、callback overhead、JIT コンパイル時間 (~350us)。

---

## Phase 7: CLI と配布 ✅ 完了

### サブタスク

- [x] 7-1/7-2. CLI バイナリ (`src/bin/jq-jit.rs`) — jq 互換フラグ (-r/-c/-n/-s/-e/--tab/--indent) + stdin/ファイル入力
- [x] 7-3. 互換性テストスイート (`tests/compat.sh`) — 105 テスト (102 PASS, 0 FAIL, 3 SKIP)

### 未実施 (ユーザー指示で保留)

- パッケージング（cargo install, Homebrew, crates.io）— 外部公開は未実施

---

## Phase 8: コア言語ギャップの解消 ✅ 完了

### サブタスク

- [x] 8-1. Quick Wins — contains/ltrimstr/rtrimstr/in/min/max/flatten ランタイム完成 (61 テスト追加)
- [x] 8-2. INDEX_OPT `.foo?` — null 返却版インデックス
- [x] 8-3. EACH_OPT `.[]?` — silent イテレーション
- [x] 8-4. オブジェクト構築 INSERT — `{key: .value}` 動的オブジェクト構築 (9 テスト追加)
- [x] 8-5. ネストクロージャ — level>0 CALL_JQ + nargs>=2。`map(select(f))` 動作 (22 テスト追加)
- [x] 8-7. 文字列補間 — `CALL_BUILTIN format` 対応。`"Hello \(.name)"` 動作 (14 テスト追加)
- [x] 8-8. 非空リテラル — `[1,2,3]`, `{}` リテラル対応
- [x] 8-9. 包括テスト + Phase 8 完了 — Obj+Obj マージ追加、SKIP 2 件解消、32 テスト追加 (累計 462)

### 判定基準 — すべて達成

- Phase 8 全機能が jq と同一結果を返す ✅
- 462 テスト全 PASS (+ 6 SKIP: 既知制限) ✅
- CLI 互換テスト 103 PASS, 0 FAIL ✅

---

## Phase 9: 拡張機能 ✅ 完了

**目標**: sort_by/group_by、range、format strings、再帰降下、path operations をカバー。500+ テスト PASS。

### サブタスク

- [x] 9-1. クロージャ付きビルトイン — sort_by(f), group_by(f), unique_by(f), min_by(f), max_by(f) (12 テスト)
- [x] 9-2. range — RANGE opcode。range(n), range(from;to) (5 テスト)
- [x] 9-3. フォーマット文字列 — @base64, @base64d, @html, @uri, @csv, @tsv, @json, @text (9 テスト)
- [x] 9-4. 再帰降下 `..` — ランタイム直接実装 rt_recurse (6 テスト)
- [x] 9-5. パス操作 — path(expr), paths, paths(f) + PATH_BEGIN/PATH_END。getpath/setpath/delpaths は 9-6 で実装済み (10-3 と一括)
- [x] 9-6. 残りビルトイン — any, all, indices, inside, tojson, fromjson, getpath, setpath, delpaths, infinite, nan, isinfinite, isnan, isnormal, debug, env, builtins (37 テスト)
- [x] 9-7. 先送り分解消 — values (自動対応), join(sep) (5 テスト)
- [x] 9-8. 変数操作拡張 — STOREVN, toplevel Error フィルタリング (2 テスト)
- [x] 9-9. 包括テスト + Phase 9 完了 — SKIP→PASS 変換 2 件、36 テスト追加 (累計 574)

### 判定基準 — すべて達成

- sort_by/group_by 等のクロージャ付きビルトインが動作 ✅
- range(n), range(from;to) が動作 ✅
- 8 種のフォーマット文字列が jq と同一結果を返す ✅
- 再帰降下 `..` が動作 ✅
- 19 関数の残りビルトインが動作 ✅
- 500+ テスト PASS ✅ (574 PASS, 0 FAIL, 6 SKIP)
- CLI 互換テスト 103 PASS, 0 FAIL ✅

---

## Phase 10: 完全互換

**目標**: def、regex、代入演算子、残り全 opcode。jq テストスイート 95%+ PASS。

### サブタスク

- [x] 10-1. def ユーザー定義関数 — 既存 CALL_JQ パスで動作確認 (13 PASS, 再帰は SKIP)
- [x] 10-2. 正規表現 — test, match, capture, scan, sub, gsub。regex クレート追加 (24 PASS)
- [x] 10-3. 代入演算子 + パス操作 — |=, +=, -=, *=, /=, %=, //= + path(expr), paths (25 PASS, 2 SKIP)
- [x] 10-4. 残り opcode — DUPN, DUP2, ERRORK, GENLABEL, DEPS, MODULEMETA, STORE_GLOBAL (4 PASS)
- [x] 10-5. try (generator) 修正 — エラーフラグ方式でジェネレータ中断 (7 PASS, 3 SKIP)
- [x] 10-6. 拡張 CLI — -R, -f, --arg, --argjson, --slurpfile (CLI 103→114 PASS)
- [ ] 10-7. jq テストスイート — jq 公式テスト実行。95%+ PASS ← 次
