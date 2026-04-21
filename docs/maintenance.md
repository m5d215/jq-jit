# 保守・互換性監査メモ

jq 互換バグを探す・直す時に毎回踏む落とし穴と、使える手を集約。
新しい機能追加時の invariant もここに追記していく。

---

## 1. 互換性バグの探し方

### 基準実装

| | バイナリ | バージョン |
|---|---|---|
| **基準（これに合わせる）** | `/opt/homebrew/opt/jq/bin/jq` | jq-1.8.1 |
| 参考（Apple 同梱） | `/usr/bin/jq` | jq-1.7.1-apple |

基準 1.8.1 にする。`/usr/bin/jq` は古いので、1.8.1 で落ちない・1.7.1 で落ちるケースに注意。

### diff ループ

jq-jit 出力を jq と並べて比較するワンライナー。フィルタ一覧を `cases` に並べて回すのが効く:

```bash
JQ=/opt/homebrew/opt/jq/bin/jq; JJ=./target/release/jq-jit
cases=(
  '[.[] | select(. > 1)]'
  '{a:1, a:2}'
  '(.a, .b) += 100'
  # ...
)
for f in "${cases[@]}"; do
  for inp in '{"a":1,"b":[1,2,3]}' 'null'; do
    r1=$(echo "$inp" | timeout 5 $JQ -c "$f" 2>&1)
    r2=$(echo "$inp" | timeout 5 $JJ -c "$f" 2>&1)
    if [ "$r1" != "$r2" ] && ! (echo "$r1$r2" | grep -q "jq: error"); then
      echo "DIFF[$inp] $f"
      echo "  jq:  $r1"
      echo "  jit: $r2"
    fi
    if echo "$r2" | grep -q "Assertion failed\|panicked\|SIGSEGV"; then
      echo "CRASH[$inp] $f"
    fi
  done
done
```

エラーメッセージの差は無視（`jq: error` を含むペアは除外）。実値の差と crash だけを拾う。

### テスト出力

`cargo test --release` だけだと regression 通過数が非表示。必ず `--nocapture`:

```bash
cargo test --release -- --nocapture 2>&1 | grep -E "=== Regression|=== jq Off|PASS:|FAIL:|TOTAL:"
```

---

## 2. Fast path の地図

jq-jit は同じ filter 形を複数レイヤで最適化する。出力が jq と違ってエラーにならない時、だいたいどこかの fast path が先行して仕事してる。上から順に疑う:

### (a) `src/bin/jq-jit.rs` の `detect_*` 群（~2060-2400 行）

raw-byte 最適化。JSON をパースせずバイトコピーで済ませる系。ここでよくやらかすのは:

- **型情報の無視**: 数値だけ想定して `[.[] | select(. > 1)]` で非数値を全部捨てる、みたいなやつ
- **fallback 条件の取りこぼし**: detect が成功したら必ず jq と同じ出力を出す必要がある

該当パスが発火したかを知るには `eprintln!` を `detect_xxx` の内側と、`bin/jq-jit.rs` 内で `Some(ref xxx) = xxx` 分岐の手前に仕込む。`JQJIT_TRACE` みたいな専用フラグは現状ないので自前で足す（**TODO**: 常設したい）。

### (b) `src/interpreter.rs` の `simplify_expr`

構文木レベルの書き換え。`Pipe(Collect(…), UnaryOp(Add))` を `a + b + …` に畳む類。単一出力前提の置換が generator を壊すパターンをよく踏む（`Empty` branch、comma list、etc）。

### (c) `src/parser.rs` のパース直後の rewrite

`paths(f)` や `leaf_paths` のような builtin-as-macro 展開。ここで jq のセマンティクスを構文木で書き下ろすので、細部（空 path を落とすか、length > 0 の guard、etc）をサボると全出力に影響する。

### (d) `src/jit.rs` の Flattener

JIT にまで降りる filter はここ。`flatten_scalar` / `flatten_gen` の match 群。複雑 path の Update/Assign は `__update__:idx:idx` closure op で eval にバックアウトする。

---

## 3. 守るべき invariant

### オブジェクト重複キーの dedup

jq は `{a:1, a:2}` → `{"a":2}`（後勝ち、最初の位置を保持）。構築系 fast path すべてでこれを守る:

| 場所 | 関数 |
|---|---|
| `interpreter.rs` | `push_const_json`（`detect_literal_output` 経由） |
| `interpreter.rs` | `simplify_expr` の `{pairs} | length` 畳み込み |
| `interpreter.rs` | `detect_field_remap`（戻り値 dedup） |
| `interpreter.rs` | `detect_computed_remap`（戻り値 dedup） |

新しい object 構築 fast path を足す時は、リテラル string キーの後勝ち collapse を必ず通す。

### `[gen] | add` の Empty 扱い

`[a, b, c] | add` → `a + b + c` という rewrite は、各 branch が **必ず 1 値を yield** する時しか valid。`Empty` branch を含むと、jq 的には array から消えるだけだが、rewrite 後は `x + empty` → nothing になって全体が消える。

該当箇所:
- `src/parser.rs` の `finalize_pipe` 系
- `src/interpreter.rs` の `simplify_expr` pipe ハンドラ

追加時も `all(|e| !matches!(e, Expr::Empty))` の guard を必ず書く。

### `first()` / `limit(1)` 書き換えのスコープ

`limit(n; a, b, c, …)` で `n == 1` の時だけ先頭 branch に潰せる（`first(g)` 相当）。`n >= 1` だと 4 値欲しい時に 1 値しか返さなくなる。修正箇所は `interpreter.rs` の `simplify_expr` の `Expr::Limit` ハンドラ。

### JIT → eval 委譲時の env seeding

JIT が複雑な Update/Assign を処理する時、`__update__:path_idx:update_idx` / `__assign__:…` という closure op 経由で `eval_update_standalone` / `eval_assign_standalone` に落とす。このとき**新しい `eval::Env` を作る**ので、JIT が set していた let-binding 変数が消える。

具体例: `(.a, .b) += 100` は parser で
```text
let $r = 100 in update((.a, .b); . + $r)
```
に desugar される。JIT は `$r` を自分の var slot に入れるが、complex path で runtime 委譲した先の fresh Env は `$r` を知らず、`. + null` になって update が no-op 化する。

対策は `src/jit.rs` の `seed_eval_env_from_jit` で、委譲 expression の `LoadVar` 参照を集めて JIT env から eval env にコピーする。**新しい `__xxx__:` closure op を追加する時は必ず同じ seeding を通す**。

### `paths` / `leaf_paths` / `paths(f)` は root を落とす

jq の `paths` は `path(..) | select(length > 0)`。scalar 入力で空 path `[]` を yield しない。派生する `paths(f)` や `leaf_paths` も同じ不変性を持つ。rewrite を書き直す時は `length > 0` の guard を必ず最後に挟む。

### 型間比較の順序

jq は `null < false < true < number < string < array < object`。数値しか想定してない比較 fast path は、非数値 value を無視したり false 扱いしたりしやすい。`bin/jq-jit.rs` の `select_cmp_passes` ヘルパに合わせる（型判定は JSON 値の先頭バイトで足りる）。

---

## 4. 典型的なデバッグの流れ

1. **まず diff ループで再現** — input と filter の最小組を特定
2. **どの fast path が走ってるか特定** — `bin/jq-jit.rs` の `detect_*` 呼び出し手前に `eprintln!` を順に挿す。`detect_literal_output` → `detect_computed_remap` → ... と上から
3. **発火していない場合は eval / simplify** — `src/eval.rs` と `src/interpreter.rs` の `simplify_expr`
4. **JIT でしか起きない**: JIT は input が大きい時 or `has_loop_constructs` の時にしか使われない。`echo null | …` で再現する時は後者（Update/Reduce/Foreach を含む filter）
5. **直したら regression test に入れる** — `tests/regression.test` に 3 行形式で追加。1 修正 = 1 テスト以上
