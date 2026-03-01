# jq バックトラッキング（ジェネレータ）モデル — JIT 実装のための徹底調査

## 1. jq のバックトラッキング実装

### 1.1 概要: jq はジェネレータ言語

jq のすべてのフィルタは **0 個以上の値を生成するジェネレータ** として機能する。これは通常の「入力 → 1 つの出力」という関数モデルとは根本的に異なる。

数学的には、jq のフィルタはリストモナドの計算に相当する:

```
filter : Value → [Value]     -- 入力 1 つに対し、出力は 0 個以上のリスト
pipe   : concatMap            -- パイプ (|) はモナディックバインド (>>=)
comma  : append               -- カンマ (,) は結果リストの結合
empty  : []                   -- empty は空リスト（= バックトラックのトリガー）
```

jq の VM はこれをバックトラッキングで実現する。`jq_next()` を呼ぶたびに次の出力が 1 つ返り、出力が尽きると `jv_invalid()` が返る。

### 1.2 コアデータ構造

#### forkpoint 構造体

```c
// https://github.com/jqlang/jq/blob/master/src/execute.c
struct forkpoint {
  stack_ptr saved_data_stack;   // データスタックの位置
  stack_ptr saved_curr_frame;   // 現在のスタックフレーム
  int path_len, subexp_nest;    // パス式の状態
  jv value_at_path;             // パス式の値
  uint16_t* return_address;     // バックトラック時の復帰先 PC
};
```

#### スタック構成

jq VM は **単一の連続スタック** 上に 3 種類のブロックをインターリーブして積む:

```
┌─────────────────────┐ ← fork_top
│   forkpoint (最新)    │
├─────────────────────┤
│   データ値            │ ← stk_top
├─────────────────────┤
│   コールフレーム       │ ← curr_frame
├─────────────────────┤
│   forkpoint (古い)    │
├─────────────────────┤
│   データ値            │
└─────────────────────┘
```

### 1.3 stack_save / stack_restore

> 以下は [`src/execute.c`](https://github.com/jqlang/jq/blob/master/src/execute.c) のコードを簡略化して再構成したもの。実際のコードではエラーハンドリングやパス式の処理など、ここで省略している部分がある。

#### stack_save（フォークポイントの作成）

```c
void stack_save(jq_state *jq, uint16_t* retaddr, struct stack_pos sp) {
  jq->fork_top = stack_push_block(&jq->stk, jq->fork_top, sizeof(struct forkpoint));
  struct forkpoint* fork = stack_block(&jq->stk, jq->fork_top);
  fork->saved_data_stack = jq->stk_top;
  fork->saved_curr_frame = jq->curr_frame;
  fork->path_len = jv_get_kind(jq->path) == JV_KIND_ARRAY ?
    jv_array_length(jv_copy(jq->path)) : 0;
  fork->value_at_path = jv_copy(jq->value_at_path);
  fork->subexp_nest = jq->subexp_nest;
  fork->return_address = retaddr;
  jq->stk_top = sp.saved_data_stack;
  jq->curr_frame = sp.saved_curr_frame;
}
```

**ポイント**: スタックの「位置」を保存するだけで、値そのものはコピーしない。

#### stack_restore（フォークポイントへの復帰）

```c
uint16_t* stack_restore(jq_state *jq) {
  while (!stack_pop_will_free(&jq->stk, jq->fork_top)) {
    if (stack_pop_will_free(&jq->stk, jq->stk_top)) {
      jv_free(stack_pop(jq));
    } else if (stack_pop_will_free(&jq->stk, jq->curr_frame)) {
      frame_pop(jq);
    } else {
      assert(0);
    }
  }
  if (jq->fork_top == 0) return 0;

  struct forkpoint* fork = stack_block(&jq->stk, jq->fork_top);
  uint16_t* retaddr = fork->return_address;
  jq->stk_top = fork->saved_data_stack;
  jq->curr_frame = fork->saved_curr_frame;
  // パス状態の復元
  // ...
  jq->fork_top = stack_pop_block(&jq->stk, jq->fork_top, sizeof(struct forkpoint));
  return retaddr;
}
```

### 1.4 FORK / BACKTRACK オペコード

#### FORK

```c
case FORK: {
  stack_save(jq, pc - 1, stack_get_pos(jq));
  pc++;  // オフセットをスキップして左辺を実行
  break;
}
```

#### ON_BACKTRACK(FORK)

```c
case ON_BACKTRACK(FORK): {
  if (raising) goto do_backtrack;
  uint16_t offset = *pc++;
  pc += offset;  // 右辺のコードにジャンプ
  break;
}
```

#### BACKTRACK

```c
do_backtrack:
case BACKTRACK: {
  pc = stack_restore(jq);
  if (!pc) {
    return jv_invalid();  // フォークスタックが空 → 出力終了
  }
  backtracking = 1;
  break;
}
```

### 1.5 カンマ演算子 (`,`) の実装

コンパイル時に `gen_both()` で FORK + JUMP パターンに変換:

```c
block gen_both(block a, block b) {
  block jump = gen_op_targetlater(JUMP);
  block fork = gen_op_target(FORK, jump);
  block c = BLOCK(fork, a, jump, b);
  inst_set_target(jump, c);
  return c;
}
```

生成されるバイトコード:

```
FORK → [offset to JUMP]    ← (1) フォークポイント作成
... a のコード ...           ← (2) 左辺を実行
JUMP → [offset to end]      ← (3) 左辺の結果を出力後、末尾へ
... b のコード ...           ← (4) バックトラック時に右辺を実行
```

### 1.6 EACH（`.[]`）の実装

```c
case ON_BACKTRACK(EACH):
case ON_BACKTRACK(EACH_OPT): {
  int idx = jv_number_value(stack_pop(jq));
  jv container = stack_pop(jq);
  // インデックスを進める
  if (!keep_going || raising) {
    jv_free(container);
    goto do_backtrack;          // イテレーション終了
  } else if (is_last) {
    jv_free(container);
    stack_push(jq, value);      // 最後の要素 → フォークポイント不要
  } else {
    struct stack_pos spos = stack_get_pos(jq);
    stack_push(jq, container);
    stack_push(jq, jv_number(idx));
    stack_save(jq, pc - 1, spos); // 自分自身に戻るフォーク
    stack_push(jq, value);
  }
  break;
}
```

**最適化ポイント**: 最後の要素ではフォークポイントを作らない。

### 1.7 RET — 出力の生成と一時停止

トップレベルの RET は `jq_next()` から値を返す前にフォークポイントを作る。次に `jq_next()` が呼ばれるとバックトラックが発動し、次の値を生成する。


## 2. 類似する言語・処理系のアプローチ

### 2.1 Prolog WAM (Warren Abstract Machine)

**チョイスポイント（Choice Point）**: マシンの完全な状態（レジスタ、ヒープポインタ、トレイルポインタ、次の選択肢のアドレス）を保存。失敗時にチョイスポイントまで巻き戻し。

| 比較 | jq | WAM |
|---|---|---|
| フォークポイント | フォークポイント（スタックポインタのみ） | チョイスポイント（全レジスタ） |
| 目的 | JSON 変換 | ユニフィケーション |
| トレイル | 不要（不変値） | 必要（変数束縛の取消） |

**ネイティブコンパイルの実績**: GNU Prolog は WAM → ネイティブコードの直接コンパイルに成功。

### 2.2 Icon 言語

jq の設計に最も近い先行者。**goal-directed evaluation**（目標指向評価）を特徴とする。

- すべての式が「成功（値を返す）」または「失敗」を返す
- ジェネレータは `suspend` で値を返し、再開を待つ

Icon ライクな式評価システムを持つ Converge 言語での実測によると:

- failure frame 操作は全オペコードの 25-30%、実行時間の約 10%
- 約 80% のケースで failure frame のネスト深度は 3 以下

（出典: Laurence Tratt, "Experiences with an Icon-like Expression Evaluation System", 2010 — Converge 言語コンパイラの実行プロファイル結果）

### 2.3 Python ジェネレータ / Lua コルーチン

| 特性 | jq | Python ジェネレータ | Lua コルーチン |
|---|---|---|---|
| 複数ジェネレータの合成 | FORK で暗黙的に合成 | `itertools.chain` 等で明示的 | 手動で resume/yield |
| スタック保存 | ポインタのみ | フレームオブジェクト | スタック全体 |
| バックトラッキング | ネイティブサポート | なし | なし |
| ネスト | 制限なし | 1 レベルのみ | 制限なし |


## 3. JIT でバックトラッキングを実現する戦略

### 3.1 CPS (Continuation-Passing Style) 変換

すべてのフィルタを「次に何をするか」（continuation）を引数として受け取る関数に変換する。

```rust
type Continuation<'a> = Box<dyn FnMut(Value) + 'a>;

fn filter_pipe(left: Filter, right: Filter, input: Value, cont: Continuation) {
    left(input, |intermediate| {
        right(intermediate, |v| cont(v));
    });
}
```

| 利点 | 欠点 |
|---|---|
| バックトラッキングが通常の関数呼び出しに変換 | クロージャの大量生成（ヒープアロケーション） |
| Cranelift で素直にコンパイル可能 | 深いネストでスタックオーバーフローの可能性 |
| 末尾呼び出し最適化が適用可能 | コードサイズが膨張する傾向 |

### 3.2 スタックコピー / スタックスイッチ

FORK 時にスタックポインタを保存、バックトラック時に復元。jq のインタプリタ実装をネイティブコードに直接移植する方式。

**利点**: jq に最も忠実。GNU Prolog の実績あり。
**欠点**: Cranelift はスタック操作の直接制御を提供しない。**実装難易度: 非常に高い。**

### 3.3 コルーチン / Fiber

各ジェネレータを独立したコルーチンで実行し、yield で値を生成。

**利点**: メンタルモデルに最も自然。
**欠点**: Cranelift にコルーチンサポートなし。メモリオーバーヘッド大。

### 3.4 トランポリン

再帰をループに変換。各フィルタはサンクを返し、トランポリンループが繰り返し呼び出す。

**利点**: 実装がシンプル。段階的最適化に最適。
**欠点**: 間接呼び出しのオーバーヘッド。JIT の高速化を相殺する可能性。

### 3.5 セグメンテッドスタック

スタックを固定サイズのセグメントに分割し、オーバーフロー時に新しいセグメントを割り当てる。Go の初期ゴルーチン実装（Go 1.3 以前）で使われた手法。

**利点**: 深いバックトラッキングでもスタックオーバーフローしない。メモリ効率が良い（必要な分だけ割り当て）。
**欠点**: セグメント境界でのオーバーヘッド（"hot split" 問題 — Go が 1.4 で連続スタックに切り替えた理由）。Cranelift はスタックレイアウトを内部で管理しており、セグメンテッドスタックとの統合が困難。カスタムプロローグ/エピローグの挿入が必要になるが、Cranelift はこれをサポートしない。

### 3.6 Iterator 変換（jaq のアプローチ）

各フィルタを `Iterator<Item = Result<Value>>` に変換。パイプは `flat_map`、カンマは `chain`。

**利点**: Rust 型システムとの親和性最高。jaq が実証済み。
**欠点**: `Box<dyn Iterator>` による動的ディスパッチ。JIT との統合が不明確。

### 3.7 その他の検討候補

#### Delimited Continuations

jq の `FORK` は delimited continuations における `reset`（区切りの設定）に、`BACKTRACK` は `shift`（区切りまでの継続の捕捉）に概念的に対応する。理論的にはバックトラッキングを delimited continuations で表現可能だが、Cranelift には first-class continuations のサポートがなく、ランタイムでの continuation の捕捉・復元メカニズムを自前で実装する必要がある。結局のところ CPS 変換やスタックコピーと同等の実装コストになるため、独立した戦略としての採用は見送る。

#### Algebraic Effects

Algebraic effects は delimited continuations の構造化された形式であり、バックトラッキングを `Fail` / `Choose` エフェクトとしてモデル化できる。OCaml 5 の effect handlers や Koka 言語での実績がある。しかし、Cranelift 上で effect handler のセマンティクスを実現するには、やはりスタック操作の低レベル制御が必要となり、現時点では現実的でない。

**両候補とも理論的には elegantだが、Cranelift での実現コストが CPS 変換を大きく上回るため採用しない。**

### 3.8 戦略比較表

| 戦略 | Cranelift 親和性 | メモリ効率 | 実装難易度 | 性能ポテンシャル | 段階的導入 |
|---|---|---|---|---|---|
| CPS 変換 | 高い | クロージャ多い | 中~高 | 高い | 可能 |
| スタックコピー | 低い | 効率的 | 非常に高い | 最高 | 困難 |
| セグメンテッドスタック | 低い | 効率的 | 非常に高い | 高い | 困難 |
| コルーチン | 低い | オーバーヘッド大 | 中 | 中 | やや困難 |
| トランポリン | 高い | サンク多い | 低い | 低い | 最適 |
| Iterator 変換 | 低い（JIT時） | 中 | 低い（interp） | 中 | 容易 |


## 4. 推奨アプローチ

### 4.1 ハイブリッドアプローチ（段階的実装）

#### Phase 1: Iterator ベースインタプリタ（ベースライン）

jaq のアプローチを参考に、Iterator ベースのインタプリタを構築。正しさの検証とテストスイートの構築が目的。

#### Phase 2: ホットパスの CPS + Cranelift JIT

ホットなフィルタだけを CPS 変換 → Cranelift でコンパイル:

```rust
// JIT コンパイルされたフィルタのインターフェース
type JittedFilter = extern "C" fn(
    input: *const Value,
    callback: extern "C" fn(*const Value, *mut Context),
    ctx: *mut Context,
);
```

- カンマ: 左辺のコールバック呼び出し後、右辺のコールバック呼び出し
- パイプ: 左辺の出力をコールバックで受け、その中で右辺をコールバック呼び出し
- empty: コールバックを呼ばずに return
- **バックトラッキングが消滅し、通常の制御フロー（関数呼び出しとループ）になる**

#### CPS 変換後の概念的な Cranelift IR スケッチ

`.[] | . + 1`（配列の各要素に 1 を加える）を例に、CPS 変換後の Cranelift IR の概念的な構造を示す。

```
;; CPS 変換: .[] | . + 1
;; .[] は配列の各要素についてコールバックを呼ぶジェネレータ
;; . + 1 はコールバック内で適用される変換

function %each_plus_one(i64 %input, i64 %callback, i64 %ctx) {
block0(%input: i64, %callback: i64, %ctx: i64):
    ;; 配列の長さを取得
    %len = call %jv_array_length(%input)
    %zero = iconst.i64 0
    jump block1(%zero)

block1(%idx: i64):
    ;; ループ: idx < len の間、各要素を処理
    %cmp = icmp ult %idx, %len
    brif %cmp, block2, block3

block2:
    ;; 要素を取得
    %elem = call %jv_array_get(%input, %idx)
    ;; . + 1 を適用
    %one = call %jv_number(1.0)
    %result = call %jv_add(%elem, %one)
    ;; コールバック呼び出し（= 次のパイプラインステージに値を渡す）
    call_indirect %callback(%result, %ctx)
    ;; 次の要素へ（バックトラッキングではなく単なるループ）
    %next = iadd_imm %idx, 1
    jump block1(%next)

block3:
    ;; 全要素処理完了 → return（= empty 相当、コールバックを呼ばない）
    return
}
```

**ポイント**: jq の `FORK` + `BACKTRACK` によるバックトラッキングが、CPS 変換により `block1 → block2 → block1` の単純なループに変換される。コールバック呼び出しが「次の値を生成する」操作に対応し、バックトラッキングの概念がコードから完全に消えている。

#### Phase 3: 高度な最適化

インライン化、型特殊化、エスケープ解析、ループ融合。

### 4.2 なぜ CPS + コールバック方式が最適か

1. **Cranelift との自然な統合**: CPS 後のコードは通常の関数呼び出しグラフ
2. **バックトラッキングの消去**: ジェネレータが「コールバックを複数回呼ぶ関数」に変換
3. **既存実績**: CapyScheme（Rust + Cranelift + CPS）が同様のアプローチで成功
4. **段階的導入**: インタプリタが常にフォールバックとして機能
5. **jq の特性に合致**: 短いフィルタチェインは CPS 変換のオーバーヘッドが小さい


## 参考資料

- [jqlang/jq - Internals: backtracking (Wiki)](https://github.com/jqlang/jq/wiki/Internals:-backtracking)
- [jqlang/jq - src/execute.c](https://github.com/jqlang/jq/blob/master/src/execute.c)
- [jqlang/jq - src/compile.c](https://github.com/jqlang/jq/blob/master/src/compile.c)
- [01mf02/jaq - GitHub](https://github.com/01mf02/jaq)
- [Michael Farber - "Denotational Semantics and a Fast Interpreter for jq" (2023)](https://arxiv.org/abs/2302.10576)
- [Michael Farber - "A formal specification of the jq language" (2024)](https://arxiv.org/abs/2403.20132)
- [Warren Abstract Machine - Wikipedia](https://en.wikipedia.org/wiki/Warren_Abstract_Machine)
- [GNU Prolog - On the Implementation](https://arxiv.org/pdf/1012.2496)
- [Laurence Tratt - "Experiences with an Icon-like Expression Evaluation System"](https://tratt.net/laurie/research/pubs/html/tratt__experiences_with_an_icon_like_expression_evaluation_system/)
- [Cranelift JIT Demo](https://github.com/bytecodealliance/cranelift-jit-demo)
- [CapyScheme - CPS + Cranelift](https://github.com/playX18/capyscheme)
