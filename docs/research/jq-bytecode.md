# jq Bytecode VM Internal Architecture

> jq バイトコード VM の内部構造調査。ソースコードは [jqlang/jq](https://github.com/jqlang/jq) の `master` ブランチを参照。

## 1. Opcode 一覧とセマンティクス

### 1.1 opcode 定義

opcode は [`src/opcode_list.h`](https://github.com/jqlang/jq/blob/master/src/opcode_list.h) で `OP(name, imm_type, stack_in, stack_out)` マクロにより定義される。このマクロ展開により enum と記述テーブル（`opcode_descriptions[]`）が自動生成される。

命令語は **16-bit (`uint16_t`)** 単位。オペランドも 16-bit ワードとして後続する。

### 1.2 opcode フラグ

[`src/bytecode.h`](https://github.com/jqlang/jq/blob/master/src/bytecode.h) で定義されるフラグ:

| フラグ | 値 | 意味 |
|---|---|---|
| `OP_HAS_CONSTANT` | 2 | 定数プール索引を持つ |
| `OP_HAS_VARIABLE` | 4 | 変数参照（レベル + ローカルインデックス） |
| `OP_HAS_BRANCH` | 8 | 分岐オフセットを持つ |
| `OP_HAS_CFUNC` | 32 | C 関数参照 |
| `OP_HAS_UFUNC` | 64 | ユーザー定義関数参照 |
| `OP_IS_CALL_PSEUDO` | 128 | 呼び出し疑似命令 |
| `OP_HAS_BINDING` | 1024 | バインディング命令 |
| `OP_BIND_WILDCARD` | 2048 | break 用ワイルドカード |

### 1.3 全 opcode 一覧

合計 **43 opcodes**。

#### スタック操作

| Opcode | 即値型 | Stack In | Stack Out | 説明 |
|---|---|---|---|---|
| `LOADK` | CONSTANT | 1 | 1 | 定数プールから値をロードし、スタックトップを置換 |
| `DUP` | - | 1 | 2 | スタックトップを複製 |
| `DUPN` | - | 1 | 2 | DUP と同様だが、元の値を null で置換（move semantics） |
| `DUP2` | - | 2 | 3 | スタックの 2 番目を複製してトップに |
| `PUSHK_UNDER` | CONSTANT | 1 | 2 | 定数をトップの下に挿入 |
| `POP` | - | 1 | 0 | スタックトップを除去・解放 |

#### 変数操作

| Opcode | 即値型 | Stack In | Stack Out | 説明 |
|---|---|---|---|---|
| `LOADV` | VARIABLE | 1 | 1 | ローカル変数をスタックにコピー |
| `LOADVN` | VARIABLE | 1 | 1 | ローカル変数をロードし、変数側を null に（move） |
| `STOREV` | VARIABLE | 1 | 0 | スタックトップをローカル変数に格納 |
| `STOREVN` | VARIABLE | 1 | 0 | STOREV + バックトラック時の復元用セーブポイント設置 |
| `STORE_GLOBAL` | GLOBAL | 0 | 0 | グローバル変数に定数を格納。`$ENV` や `builtins` リストの初期化等で使用。プログラム起動時に一度だけ実行される初期化用 opcode であり、JIT では初期化フェーズで処理すれば以降は無視してよい |

#### インデックス・イテレーション

| Opcode | 即値型 | Stack In | Stack Out | 説明 |
|---|---|---|---|---|
| `INDEX` | - | 2 | 1 | `obj[key]` -- キーが存在しなければエラー |
| `INDEX_OPT` | - | 2 | 1 | `obj[key]` -- キーが存在しなければ `null` |
| `EACH` | - | 1 | 1 | 配列/オブジェクトの全要素を生成（ジェネレータ） |
| `EACH_OPT` | - | 1 | 1 | EACH の `?` 版（エラー時バックトラック） |

#### 制御フロー

| Opcode | 即値型 | Stack In | Stack Out | 説明 |
|---|---|---|---|---|
| `FORK` | BRANCH | 0 | 0 | フォークポイント設置（バックトラック時にオフセット先へ） |
| `JUMP` | BRANCH | 0 | 0 | 無条件分岐 |
| `JUMP_F` | BRANCH | 1 | 0 | false/null で分岐、それ以外はフォールスルー |
| `BACKTRACK` | - | 0 | 0 | 直前のフォークポイントに復帰 |

#### 例外処理

| Opcode | 即値型 | Stack In | Stack Out | 説明 |
|---|---|---|---|---|
| `TRY_BEGIN` | BRANCH | 0 | 0 | try ブロック開始。例外時にオフセット先（catch）へ |
| `TRY_END` | - | 0 | 0 | try ブロック終了。バックトラック時にエラーをラップ |
| `ERRORK` | CONSTANT | 1 | 0 | 定数メッセージでエラーを発生させバックトラック |
| `DESTRUCTURE_ALT` | BRANCH | 0 | 0 | デストラクチャリングの代替パターン分岐 |

#### パス追跡

| Opcode | 即値型 | Stack In | Stack Out | 説明 |
|---|---|---|---|---|
| `SUBEXP_BEGIN` | - | 1 | 2 | サブ式開始。`subexp_nest` カウンタをインクリメント |
| `SUBEXP_END` | - | 2 | 2 | サブ式終了。スタック値のスワップ |
| `PATH_BEGIN` | - | 1 | 2 | パス式開始。パス状態を保存し空配列で初期化 |
| `PATH_END` | - | 2 | 1 | パス式終了。パス配列を返却 |

#### 配列操作

| Opcode | 即値型 | Stack In | Stack Out | 説明 |
|---|---|---|---|---|
| `APPEND` | VARIABLE | 1 | 0 | 配列変数に値を追加 |
| `INSERT` | - | 4 | 2 | オブジェクトにキー・値ペアを挿入 |

#### ループ

| Opcode | 即値型 | Stack In | Stack Out | 説明 |
|---|---|---|---|---|
| `RANGE` | VARIABLE | 1 | 1 | range ループ。変数をインクリメントしてフォーク |

#### 関数呼び出し

| Opcode | 即値型 | Stack In | Stack Out | 説明 |
|---|---|---|---|---|
| `CALL_BUILTIN` | CFUNC | n | 1 | C ビルトイン関数呼び出し（引数数は可変） |
| `CALL_JQ` | UFUNC | 1 | 1 | jq ユーザー定義関数呼び出し |
| `TAIL_CALL_JQ` | UFUNC | 1 | 1 | 末尾呼び出し最適化版 CALL_JQ |
| `RET` | - | 1 | 1 | 関数から復帰。トップレベルなら値を yield |

#### クロージャ

| Opcode | 即値型 | Stack In | Stack Out | 説明 |
|---|---|---|---|---|
| `CLOSURE_CREATE` | DEFINITION | 0 | 0 | jq 関数のクロージャを生成 |
| `CLOSURE_CREATE_C` | DEFINITION | 0 | 0 | C 関数のクロージャを生成 |
| `CLOSURE_PARAM` | DEFINITION | 0 | 0 | クロージャ仮引数の定義 |
| `CLOSURE_PARAM_REGULAR` | DEFINITION | 0 | 0 | 通常引数としてのクロージャ仮引数 |
| `CLOSURE_REF` | CLOSURE_REF_IMM | 0 | 0 | クロージャ参照 |

#### その他

| Opcode | 即値型 | Stack In | Stack Out | 説明 |
|---|---|---|---|---|
| `TOP` | - | 0 | 0 | NOP（何もしない） |
| `GENLABEL` | - | 0 | 1 | ユニークラベルを生成してスタックに push |
| `DEPS` | CONSTANT | 0 | 0 | モジュール依存関係メタデータ。`import` / `include` 文で指定されたモジュールパスを記録し、リンカがモジュールのロード・解決に使用する。実行時にはデータフローに影響しないため、JIT では無視してよい |
| `MODULEMETA` | CONSTANT | 0 | 0 | モジュールメタデータ（`modulemeta` ビルトインで参照される）。`metadata` 式で指定されたメタデータを保持する。JIT ではメタデータの問い合わせ時にインタプリタにフォールバックすれば対応不要 |


## 2. VM の実行モデル

ソース: [`src/execute.c`](https://github.com/jqlang/jq/blob/master/src/execute.c)

### 2.1 3 つのスタック

jq VM は **単一の連続メモリ領域** 上に 3 種のスタックを重畳して管理する。

```
+----------------------------------------------+
|  データスタック  | フレームスタック | フォークスタック |
|  (stk_top)      | (curr_frame)    | (fork_top)      |
|  <- 上に伸びる   | <- 上に伸びる    | <- 上に伸びる    |
+----------------------------------------------+
```

#### データスタック

- jq の値（`jv` 型）を積む。`jv` 型は tagged union で、x86_64 上では **16 バイト**
- `stack_push()` / `stack_pop()` で操作
- 全ての値は参照カウント管理（`jv_copy` / `jv_free`）

#### フレームスタック

関数呼び出しごとに `struct frame`（`src/execute.c` で定義）を積む:

```c
struct frame {
    struct bytecode* bc;          // この関数のバイトコード
    stack_ptr       env;          // 親フレームポインタ（クロージャ環境）
    stack_ptr       retdata;      // 復帰時のデータスタック位置
    uint16_t*       retaddr;      // 復帰先アドレス
    union frame_entry entries[];  // クロージャ + ローカル変数（可変長）
};
```

`entries[]` 配列は以下の 2 部分から構成される:
- 先頭 `bc->nclosures` 個: クロージャ引数（サブファンクション参照 + 環境ポインタ）
- 残り `bc->nlocals` 個: ローカル変数（`jv` 値）

#### フォーク（バックトラック）スタック

ジェネレータの実現に使用。`stack_save()` 関数がスナップショットを `struct forkpoint` として保存する（`stack_save` は関数名であり構造体名ではない）:

```c
struct forkpoint {
    stack_ptr   saved_data_stack;  // データスタック位置
    stack_ptr   saved_curr_frame;  // フレームポインタ
    int         path_len;          // パス追跡状態
    int         subexp_nest;       // サブ式ネスト深度
    jv          value_at_path;     // パス式追跡中の値（jv 型）
    uint16_t*   return_address;    // 復帰先 PC
};
```

`jv` 型は tagged union で、x86_64 上では **16 バイト** のサイズを持つ。

`stack_restore()` で保存状態に巻き戻し、復帰先アドレスを返す。

### 2.2 メイン実行ループ

`jq_next()` が VM のメインループ。構造は以下の通り:

```
jq_next():
  while (true):
    opcode = *pc++
    switch (opcode):
      case DUP:  ...
      case LOADK: ...
      ...
      case BACKTRACK:
        pc = stack_restore()  // フォークスタックから復帰
        opcode_ON_BACKTRACK = *pc  // 復帰先 opcode の ON_BACKTRACK ハンドラへ
        switch (opcode_ON_BACKTRACK):
          case ON_BACKTRACK(EACH): ...
          case ON_BACKTRACK(TRY_BEGIN): ...
          case ON_BACKTRACK(FORK): ...
          ...
```

重要な特徴:
- **二重ディスパッチ**: 各 opcode は「順方向実行」と「バックトラック時」の 2 つのハンドラを持ちうる
- バックトラック時は `stack_restore()` で状態を巻き戻してから、保存された PC 位置の opcode の `ON_BACKTRACK` ハンドラに入る

### 2.3 ジェネレータの仕組み

jq の最も特徴的な機能。パイプライン中の各フィルタは **複数の値を生成** できる。

```
.[] | . + 1
```

1. `.[]` は `EACH` opcode で実装。最初の呼び出しでインデックス `-1` を初期化しフォークポイントを設置
2. 値を 1 つ生成するたびに下流（`. + 1`）に渡す
3. 下流の処理が完了すると `BACKTRACK` でフォークポイントに復帰
4. `ON_BACKTRACK(EACH)` がインデックスをインクリメントして次の要素を生成
5. 全要素を処理したら、さらに上流のフォークポイントにバックトラック

### 2.4 関数呼び出しとクロージャ

#### 通常呼び出し (`CALL_JQ`)

```
CALL_JQ [nargs] [level] [fidx]
```

1. 新しいフレームを `frame_push()` で生成
2. クロージャ引数を即値からコピー（`ARG_NEWCLOSURE` フラグで新規 vs 参照を区別）
3. 復帰先アドレス・データスタック位置を保存
4. PC をサブファンクションのコード先頭に設定

#### 末尾呼び出し (`TAIL_CALL_JQ`)

現在のフレームの復帰情報を再利用。フレームを pop してから push することで、スタックの無限成長を防ぐ。

#### クロージャ

jq では関数引数としてフィルタ（関数）を渡せる。例: `map(f)` の `f`。

- コンパイル時: 引数フィルタは `CLOSURE_CREATE` でサブファンクションとしてコンパイル
- 呼び出し時: `CALL_JQ` の即値でクロージャのバイトコードと環境ポインタが渡される
- 実行時: クロージャ内からは `env` ポインタ経由で外側のフレームの変数にアクセス

### 2.5 `struct bytecode` の構造

[`src/bytecode.h`](https://github.com/jqlang/jq/blob/master/src/bytecode.h):

```c
struct bytecode {
    uint16_t*           code;           // 命令列（16-bit ワード配列）
    int                 codelen;        // code の長さ（ワード数）
    int                 nlocals;        // ローカル変数の数
    int                 nclosures;      // クロージャ引数の数
    jv                  constants;      // 定数プール（jv 配列）
    struct symbol_table* globals;       // グローバルシンボルテーブル
    struct bytecode**   subfunctions;   // サブファンクション配列
    int                 nsubfunctions;  // サブファンクションの数
    struct bytecode*    parent;         // 親バイトコード（クロージャ用）
    jv                  debuginfo;      // デバッグ情報（行番号等）
};
```

バイトコードはツリー構造を形成する。トップレベルのプログラムが root で、定義された関数やクロージャは `subfunctions` として子ノードになる。


## 3. バイトコードのダンプ方法

### 3.1 コマンドラインオプション

```bash
jq --debug-dump-disasm '<filter>' < /dev/null
```

`--debug-dump-disasm` フラグにより、フィルタのコンパイル結果をバイトコードとして出力する。

### 3.2 ダンプの実装

[`src/bytecode.c`](https://github.com/jqlang/jq/blob/master/src/bytecode.c) に以下の関数群がある:

- `dump_disassembly()` -- メイン。関数パラメータ表示 + `dump_code()` 呼び出し + サブファンクションの再帰処理
- `dump_code()` -- 命令列を走査し `dump_operation()` を呼ぶ
- `dump_operation()` -- 個別命令のフォーマット出力（PC、opcode 名、オペランド）

### 3.3 出力例

```bash
$ echo 'null' | jq --debug-dump-disasm '.foo'
```

出力形式（概要）:
```
0000 DUP
0001 PUSHK_UNDER "foo"
0003 INDEX
0004 RET
```


## 4. 代表的なフィルタのバイトコード例

以下はソースコードのコンパイルパターンから推論したバイトコード列。コンパイラの `gen_*` 関数群とパーサー（[`src/parser.y`](https://github.com/jqlang/jq/blob/master/src/parser.y)、[`src/compile.c`](https://github.com/jqlang/jq/blob/master/src/compile.c)）を根拠とする。

### 4.1 `.foo` -- フィールドアクセス

```
0000  DUP                     ; 入力値を複製（パス追跡用）
0001  PUSHK_UNDER "foo"       ; 定数 "foo" をトップの下に挿入
0003  INDEX                   ; stack: [input, "foo"] -> input["foo"]
0004  RET                     ; 結果を返す
```

**実行トレース** (入力: `{"foo": 42}`)。スタックは `[底, ..., トップ]` の方向で表記:
1. `DUP` -- スタック: `[{"foo":42}, {"foo":42}]`
2. `PUSHK_UNDER "foo"` -- スタック: `[{"foo":42}, "foo", {"foo":42}]`
3. `INDEX` -- `{"foo":42}["foo"]` = `42` -- スタック: `[{"foo":42}, 42]`
4. `RET` -- 値 `42` を出力。スタック: `[]`

### 4.2 `.foo.bar` -- チェーンされたフィールドアクセス

```
0000  DUP
0001  PUSHK_UNDER "foo"
0003  INDEX                   ; .foo -> 中間オブジェクト
0004  DUP
0005  PUSHK_UNDER "bar"
0007  INDEX                   ; .bar -> 最終値
0008  RET
```

### 4.3 `. + 1` -- 算術演算

算術演算は C ビルトイン関数 `_plus` の呼び出しにコンパイルされる。

```
0000  SUBEXP_BEGIN            ; サブ式開始
0001  DUP                     ; 入力値を複製
0002  LOADK 1                 ; 定数 1 をロード
0003  SUBEXP_END              ; サブ式終了
0004  CALL_BUILTIN _plus/2    ; _plus(input, 1) を呼び出し
0005  RET
```

### 4.4 `select(.x > 0)` -- フィルタリング

`def select(f): if f then . else empty end;`

```
0000  DUP                     ; 入力を保存
0001  DUP                     ; select の引数用
; --- .x > 0 のコンパイル ---
0002  DUP
0003  PUSHK_UNDER "x"
0005  INDEX                   ; .x を取得
0006  SUBEXP_BEGIN
0007  DUP
0008  LOADK 0
0009  SUBEXP_END
0010  CALL_BUILTIN _greater/2 ; .x > 0
; --- if-then-else ---
0011  JUMP_F else             ; false なら else へ
0012  POP                     ; 条件結果を捨てる
0013  JUMP end
; else: empty
0014  POP
0015  BACKTRACK               ; empty = バックトラック（値を出力しない）
0016  RET                     ; end ラベル
```

### 4.5 `map(. * 2)` -- 高階関数

`def map(f): [.[] | f];`

```
0000  DUP                     ; 入力配列を保存
0001  LOADK []                ; 空配列を結果変数に初期化
0002  STOREV $result
; --- [.[] | f] の展開 ---
0003  FORK collect            ; バックトラック先 = collect
0004  DUP
0005  EACH                    ; .[] -- 要素を1つずつ生成
; --- f (. * 2) の適用 ---
0006  CALL_JQ closure_0       ; . * 2 を呼び出し
; --- 結果を配列に追加 ---
0007  APPEND $result          ; 結果配列に追加
0008  BACKTRACK               ; 次の要素へ
; collect:
0009  LOADVN $result          ; 完成した配列を取得
0010  RET
```

**実行トレース** (入力: `[1, 2, 3]`):
1. `EACH` で `1` を生成 -> `. * 2` -> `2` -> `APPEND` -> `$result = [2]` -> `BACKTRACK`
2. `ON_BACKTRACK(EACH)` で `2` を生成 -> `. * 2` -> `4` -> `APPEND` -> `$result = [2, 4]` -> `BACKTRACK`
3. `ON_BACKTRACK(EACH)` で `3` を生成 -> `. * 2` -> `6` -> `APPEND` -> `$result = [2, 4, 6]` -> `BACKTRACK`
4. `ON_BACKTRACK(EACH)` -- 要素なし -> さらにバックトラック
5. `ON_BACKTRACK(FORK)` -> collect ラベルへ -> `LOADVN $result` -> `[2, 4, 6]` を出力

### 4.6 `reduce .[] as $x (0; . + $x)` -- reduce

```
0000  DUP                     ; 入力を保存
0001  LOADK 0                 ; 初期値 0
0002  STOREV $acc             ; アキュムレータ = 0
0003  FORK done               ; バックトラック先 = done
0004  DUPN                    ; 入力を取得（null で置換）
0005  EACH                    ; .[] -- 要素を生成
0006  STOREV $x               ; as $x
0007  LOADVN $acc             ; アキュムレータを取得（move）
; --- . + $x ---
0008  SUBEXP_BEGIN
0009  DUP
0010  LOADV $x
0011  SUBEXP_END
0012  CALL_BUILTIN _plus/2    ; . + $x
0013  STOREV $acc             ; 結果をアキュムレータに戻す
0014  BACKTRACK               ; 次の要素へ
; done:
0015  LOADVN $acc             ; 最終結果を取得
0016  RET
```

### 4.7 `try .foo catch "default"` -- 例外処理

```
0000  TRY_BEGIN catch          ; 例外時 -> catch へ
0001  DUP
0002  PUSHK_UNDER "foo"
0004  INDEX                    ; .foo（存在しなければエラー）
0005  TRY_END                  ; try ブロック終了
0006  JUMP end                 ; 正常時は catch をスキップ
; catch:
0007  POP                      ; エラーメッセージを捨てる
0008  LOADK "default"          ; "default" を返す
; end:
0009  RET
```

### 4.8 `.[] | select(. > 3)` -- ジェネレータ + フィルタ

```
0000  EACH                    ; 要素を1つずつ生成（ジェネレータ）
; --- select(. > 3) ---
0001  DUP                     ; 入力を保存
0002  SUBEXP_BEGIN
0003  DUP
0004  LOADK 3
0005  SUBEXP_END
0006  CALL_BUILTIN _greater/2 ; . > 3
0007  JUMP_F else              ; false -> else
0008  POP                      ; 条件結果を捨てる
0009  JUMP end                 ; 値をそのまま通過
; else:
0010  POP
0011  BACKTRACK                ; empty -- 値を出力せずバックトラック
; end:
0012  RET
```

**実行トレース** (入力: `[1, 5, 2, 7]`):

| ステップ | EACH が生成 | `. > 3` | 結果 |
|---|---|---|---|
| 1 | `1` | `false` | `BACKTRACK`（出力なし） |
| 2 | `5` | `true` | `5` を出力 -> `BACKTRACK` |
| 3 | `2` | `false` | `BACKTRACK`（出力なし） |
| 4 | `7` | `true` | `7` を出力 -> `BACKTRACK` |
| 5 | 要素なし | - | 上流にバックトラック |

最終出力: `5`, `7`

### 4.9 `foreach .[] as $x (0; . + $x)` -- foreach

`foreach` は `reduce` と構造が似ているが、**各イテレーションの中間値を出力する** 点が異なる。`reduce` が最終結果のみを返すのに対し、`foreach` はアキュムレータの途中経過をストリームとして生成する。

```
0000  DUP                     ; 入力を保存
0001  LOADK 0                 ; 初期値 0
0002  STOREV $acc             ; アキュムレータ = 0
0003  FORK done               ; バックトラック先 = done
0004  DUPN                    ; 入力を取得（null で置換）
0005  EACH                    ; .[] -- 要素を生成
0006  STOREV $x               ; as $x
0007  LOADVN $acc             ; アキュムレータを取得（move）
; --- . + $x (update) ---
0008  SUBEXP_BEGIN
0009  DUP
0010  LOADV $x
0011  SUBEXP_END
0012  CALL_BUILTIN _plus/2    ; . + $x
0013  STOREV $acc             ; 結果をアキュムレータに戻す
; --- extract（省略時は . = アキュムレータの値をそのまま出力） ---
0014  LOADVN $acc             ; アキュムレータの現在値を出力
0015  RET                     ; 中間値を yield（→ 次回 jq_next() でバックトラック）
; done:
0016  BACKTRACK               ; 上流にバックトラック（foreach は最終値を出力しない）
```

**実行トレース** (入力: `[1, 2, 3]`)。スタックは `[底, ..., トップ]` の方向で表記:
1. `EACH` で `1` を生成 -> `$x = 1`, `$acc = 0` -> `. + $x` -> `1` -> `$acc = 1` -> 中間値 `1` を出力 -> `BACKTRACK`
2. `ON_BACKTRACK(EACH)` で `2` を生成 -> `$x = 2`, `$acc = 1` -> `. + $x` -> `3` -> `$acc = 3` -> 中間値 `3` を出力 -> `BACKTRACK`
3. `ON_BACKTRACK(EACH)` で `3` を生成 -> `$x = 3`, `$acc = 3` -> `. + $x` -> `6` -> `$acc = 6` -> 中間値 `6` を出力 -> `BACKTRACK`
4. `ON_BACKTRACK(EACH)` -- 要素なし -> `BACKTRACK` -> done -> さらに上流にバックトラック

最終出力: `1`, `3`, `6`

**`reduce` との違い**: `reduce` は `FORK done` でバックトラックが到達した時に `LOADVN $acc` で最終値を返す。`foreach` は各イテレーションで `RET` により中間値を yield し、最終的に `done` ラベルでは値を出力せずに上流にバックトラックする。

### 4.10 `label $out | foreach .[] as $x (0; . + $x; if . > 5 then ., break $out else . end)` -- GENLABEL と label-break

`label-break` パターンは `GENLABEL` で生成したユニークラベルを使い、`break` でそのラベルに対応するバックトラックポイントまで一気に巻き戻す仕組み。

```
; --- label $out ---
0000  GENLABEL                ; ユニークなラベル ID を生成してスタックに push
0001  STOREVN $out            ; ラベル ID を $out に保存（バックトラック時復元用）
0002  FORK label_end          ; label の終端（バックトラック時ここに飛ぶ）
; --- foreach .[] as $x (0; . + $x; extract) ---
0003  DUP
0004  LOADK 0
0005  STOREV $acc
0006  FORK foreach_done
0007  DUPN
0008  EACH                    ; .[] -- 要素を生成
0009  STOREV $x               ; as $x
0010  LOADVN $acc
; --- update: . + $x ---
0011  SUBEXP_BEGIN
0012  DUP
0013  LOADV $x
0014  SUBEXP_END
0015  CALL_BUILTIN _plus/2
0016  STOREV $acc
; --- extract: if . > 5 then ., break $out else . end ---
0017  LOADVN $acc
0018  DUP
0019  SUBEXP_BEGIN
0020  DUP
0021  LOADK 5
0022  SUBEXP_END
0023  CALL_BUILTIN _greater/2 ; . > 5
0024  JUMP_F else
; then: ., break $out
0025  POP
0026  DUP                     ; . を出力用に複製
0027  RET                     ; . を出力（1つ目の値）
; --- break $out の実装 ---
0028  LOADV $out              ; ラベル ID をロード
0029  BACKTRACK               ; ラベル ID 付きでバックトラック
                              ; → $out の FORK ポイントまで巻き戻す
; else:
0030  POP
0031  RET                     ; 中間値をそのまま出力
; foreach_done:
0032  BACKTRACK
; label_end:
0033  RET                     ; break で到達した場合、ここで終了
```

**`GENLABEL` の仕組み**: `GENLABEL` は実行ごとにインクリメントされるユニークな整数を生成する。`break $out` はこのラベル ID を `BACKTRACK` の raising メカニズムに載せ、フォークスタックを `label` の `FORK` ポイントまで一気に巻き戻す。途中のフォークポイント（`foreach` 内の `EACH` 等）はすべてスキップされる。

### 4.11 `.[] as {a: $a} ?// {b: $b}` -- DESTRUCTURE_ALT

`?//`（destructuring alternative）演算子は、パターンマッチが失敗した場合に次の代替パターンを試す。`DESTRUCTURE_ALT` opcode がこの分岐制御を担う。

```
0000  EACH                    ; .[] -- 要素を生成
; --- 1st pattern: {a: $a} ---
0001  DESTRUCTURE_ALT alt2    ; パターンマッチ失敗時 → alt2 へ
0002  DUP
0003  PUSHK_UNDER "a"
0004  INDEX                   ; .a を取得（存在しなければエラー → alt2 へ）
0005  STOREV $a               ; as $a
0006  JUMP match_end          ; マッチ成功 → 後続処理へ
; alt2: --- 2nd pattern: {b: $b} ---
0007  DUP
0008  PUSHK_UNDER "b"
0009  INDEX                   ; .b を取得（存在しなければエラー → バックトラック）
0010  STOREV $b               ; as $b
; match_end:
0011  RET
```

**`DESTRUCTURE_ALT` の動作**: `TRY_BEGIN` に類似した例外ハンドラとして機能する。現在のパターンでデストラクチャリングが失敗（`INDEX` が null/エラー）した場合、通常のバックトラックではなく、`DESTRUCTURE_ALT` で指定されたオフセット先（次の代替パターン）にジャンプする。全ての代替パターンが失敗した場合にのみ、通常のバックトラックが発動する。


## 5. JIT 化の観点からの所見

### 5.1 頻出 opcode の予測

| 頻度 | Opcodes | 理由 |
|---|---|---|
| 極高 | `DUP`, `LOADK`, `INDEX`, `RET` | ほぼ全てのフィールドアクセスで使用 |
| 高 | `EACH`, `FORK`, `BACKTRACK` | ジェネレータ/イテレーション |
| 高 | `CALL_BUILTIN`, `SUBEXP_BEGIN/END` | 算術・比較演算 |
| 中 | `STOREV`, `LOADV`, `LOADVN` | reduce, foreach, as パターン |
| 中 | `JUMP`, `JUMP_F`, `POP` | 条件分岐 |
| 低 | `TRY_BEGIN/END`, `ERRORK` | 例外処理 |
| 低 | `PATH_BEGIN/END`, `INSERT` | パス式、オブジェクト構築 |
| 極低 | `GENLABEL`, `DEPS`, `MODULEMETA`, `TOP` | メタ・特殊用途 |

### 5.2 JIT に向いている部分

#### (a) フィールドアクセスチェーン

`.foo.bar.baz` は `DUP + PUSHK_UNDER + INDEX` の繰り返し。型が既知ならハッシュテーブルルックアップに直接コンパイル可能。キー文字列が定数なので、ハッシュ値のプリコンピュートも効果的。

#### (b) 算術・比較演算

型が数値と判明していればネイティブの加算・比較命令に置換可能。`SUBEXP_BEGIN/END` のオーバーヘッドも除去できる。

#### (c) 線形パイプライン（バックトラックなし）

`.foo | . + 1 | tostring` のようなバックトラックのない直線的パイプライン。フォーク/バックトラック機構が不要なので、通常の関数呼び出しチェーンとしてコンパイル可能。

#### (d) `select` による単純フィルタリング

条件分岐 + 値の pass/skip として最適化可能。

### 5.3 JIT が難しい部分

#### (a) バックトラック機構全般

- 状態の保存/復元がスタック全体のスナップショット
- `BACKTRACK` のジャンプ先が実行時に動的に決まる
- ネストしたフォークの復帰位置が実行時の状態に依存

#### (b) `jv` 型の動的型付け

各演算で型チェックと参照カウント管理が必要。

#### (c) try-catch と例外伝播

例外はバックトラック機構の一部として実装されている。

#### (d) パス追跡 (`path()` 式)

全ての INDEX 操作に追加のオーバーヘッドが必要。

### 5.4 型特殊化の戦略

| 戦略 | 難易度 | 効果 |
|---|---|---|
| プロファイルベースの型特殊化 | 中 | インタープリタ実行中に型プロファイルを収集し、ホットパスの型をガード付きで特殊化 |
| 定数伝播 | 低 | `LOADK` の定数から型を推論 |
| 構造的型推論 | 高 | 入力スキーマが既知の場合、全フィールドの型を仮定して特殊化 |
| number 特殊化 | 中 | 算術ループを浮動小数点命令に特殊化。整数のみの場合は int64 に最適化 |

### 5.5 JIT 戦略の推奨

1. **Tier 1: テンプレート JIT** — 各 opcode を事前定義されたネイティブコードテンプレートに置換。ディスパッチオーバーヘッドの除去が主な効果
2. **Tier 2: トレーシング JIT** — ホットループを検出し、線形コードに展開。型プロファイル情報を使った型特殊化
3. **Tier 3: メソッド JIT** — バイトコード全体を SSA 形式の IR に変換。定数畳み込み、デッドコード除去、インライン化

初期仮説としてはトレーシング JIT が有力だったが、backtracking.md での詳細分析を経て **CPS + Cranelift によるメソッド JIT 方式** が推奨される。CPS 変換によりバックトラッキングが通常の関数呼び出しに消去され、Cranelift との統合が自然になる。段階的導入も可能（詳細は [backtracking.md](backtracking.md) 参照）。
