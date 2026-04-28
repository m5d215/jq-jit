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

### 常設 differential harness

`cargo test --release --test differential` が `tests/differential/corpus.test` の
`(filter, input)` ペアを jq 1.8.x と突き合わせる。新しい compat バグを直したら
コーパスに 1 件追加すること（`tests/regression.test` の追加に加えて）。
jq バイナリは `JQ_BIN` → `/opt/homebrew/opt/jq/bin/jq` → `PATH` の順で解決する。

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

該当パスが発火したかを知るには環境変数 `JQJIT_TRACE=1` を付けて実行する。`bin/jq-jit.rs` の `detect_*` 呼び出しはトレースマクロでラップされてて、最初に match した fast path 名を一行 stderr に吐く:

```
$ JQJIT_TRACE=1 jq-jit '{a:.x, a:.x+1}' <<< '{"x":1}'
[trace] filter='{a:.x, a:.x+1}' matched=detect_computed_remap
```

どの fast path も match しない場合は `matched=eval` か `matched=jit` が出るので、generic path に落ちたことが判る。

### (b) `src/interpreter.rs` の `simplify_expr`

構文木レベルの書き換え。`Pipe(Collect(…), UnaryOp(Add))` を `a + b + …` に畳む類。単一出力前提の置換が generator を壊すパターンをよく踏む（`Empty` branch、comma list、etc）。

### (c) `src/parser.rs` のパース直後の rewrite

`paths(f)` のような builtin-as-macro 展開。ここで jq のセマンティクスを構文木で書き下ろすので、細部（空 path を落とすか、length > 0 の guard、etc）をサボると全出力に影響する。

### (d) `src/jit.rs` の Flattener

JIT にまで降りる filter はここ。`flatten_scalar` / `flatten_gen` の match 群。複雑 path の Update/Assign は `__update__:idx:idx` closure op で eval にバックアウトする。

---

## 3. 守るべき invariant

### オブジェクト重複キーの dedup

jq は `{a:1, a:2}` → `{"a":2}`（後勝ち、最初の位置を保持）。

**canonical な作り方**: `Value::Obj` を直接作らず、`src/value.rs` の
ファクトリを通す。dedup 不変性は関数シグネチャで強制される（"opt-out" 方式、#84）:

| 目的 | API |
|---|---|
| pair list から構築（dedup） | `Value::object_from_pairs(pairs)` |
| dedup 済みと分かっている場合 | `Value::object_from_normalized_pairs(pairs)` （debug_assert で重複検知） |
| `ObjMap` を insert/push_unique 経由で組み立て済み | `Value::object_from_map(map)` |

`Value::Obj(Rc::new(..))` を直書きすると
`tests/value_factory_enforcement.rs` が旧サイト以外で検知して fail する
（旧サイトの count は `tests/value_factory_enforcement.allowlist` に記録、
削減方向にしか更新できない）。

pair list を他形式（JSON バイト、Vec<(String, RemapExpr)> など）に変換する前に
dedup だけしたい場合は従来の `interpreter.rs::normalize_object_pairs<K, V>`
を使う（キー型は `PartialEq` でよい）。`| length` 系で個数だけ欲しい場合は
dedup 後の `.len()` を取れば jq と一致する。

invariant 回帰は `tests/regression.test` の "Issue #30" ブロックと
`tests/differential/corpus.test` の "duplicate-key collapse" ブロックで監視している。

### `[gen] | add` は single-valued 要素のみ畳める

`[a, b, c] | add` → `a + b + c` という rewrite は、各 branch が **必ず 1 値を yield** する時しか valid。0 値（`Empty`）も多値（`.[]`, `recurse`, `range`, `limit`, `match/scan/capture`）もダメ:

- 0 値: rewrite 後は `x + empty` → nothing になって全体が消える
- 多値: `[.[]] | add` が `.[]` に潰れて、1 つの fold 値じゃなく複数の要素を stream する（#56）

該当箇所:
- `src/parser.rs` の `finalize_pipe` 系
- `src/interpreter.rs` の `simplify_expr` pipe ハンドラ（`is_single_valued` ローカル helper）

新しい generator-ish な `Expr` variant を足したら `is_single_valued` の reject リストにも入れる。判定が微妙なら "safe default は false" で、rewrite を発火させない方にブレる。

### `first()` / `limit(1)` 書き換えのスコープ

`limit(n; a, b, c, …)` で `n == 1` の時だけ先頭 branch に潰せる（`first(g)` 相当）。`n >= 1` だと 4 値欲しい時に 1 値しか返さなくなる。修正箇所は `interpreter.rs` の `simplify_expr` の `Expr::Limit` ハンドラ。

### JIT → eval 委譲時の env seeding

JIT が複雑な Update/Assign/sort_by/paths(f)/path(..) などを処理する時、
`__update__:path_idx:update_idx` / `__assign__:…` / `__closure_op__:sort_by:idx`
/ `__paths_filtered__:idx` / `__path__:idx` といった closure op 経由で eval
側の `eval_*_standalone` に落とす。このとき**新しい `eval::Env` を作る**ので、
JIT が set していた let-binding 変数がそのままでは消える。

具体例: `(.a, .b) += 100` は parser で
```text
let $r = 100 in update((.a, .b); . + $r)
```
に desugar される。JIT は `$r` を自分の var slot に入れるが、complex path で
runtime 委譲した先の fresh Env は `$r` を知らず、`. + null` になって update が
no-op 化する。

対策は `src/jit.rs` の `new_delegated_env(&[&expr, ...])` /
`reset_delegated_env(&env, &[&expr, ...])` を**必ず**使うこと。内部で
`seed_eval_env_from_jit` を呼び、委譲 expression 内の `LoadVar` 参照を
`Flattener::collect_loadvar_indices` で exhaustive に歩いて JIT env から
eval env にコピーする。個別の handler は seeding を意識せずに済む。

つまり、新しい `__xxx__:` closure op を追加する時は:

- fresh Env が欲しいなら `new_delegated_env(&[&delegated_expr])`
- cached Env を reuse するなら `reset_delegated_env(&env, &[&delegated_expr])`

を呼ぶ。`Rc::new(RefCell::new(crate::eval::Env::new(vec![])))` を直接書いては
いけない（書くと let-binding がまた消える）。

また、`collect_loadvar_indices` は **全 `Expr` variant を再帰的に歩く契約**。
Index / ObjectConstruct / StringInterpolation / CallBuiltin / FuncCall 等に
LoadVar が埋もれていても拾える。新しい `Expr` variant を足した時は、この
walker にも再帰呼び出しを追加すること（足し忘れると let-binding が潜って
silently null になる）。

### `paths` / `paths(f)` は root を落とす

jq の `paths` は `path(..) | select(length > 0)`。scalar 入力で空 path `[]` を yield しない。派生する `paths(f)` も同じ不変性を持つ。rewrite を書き直す時は `length > 0` の guard を必ず最後に挟む。

### 型間比較の順序

jq は `null < false < true < number < string < array < object`。数値しか想定してない比較 fast path は、非数値 value を無視したり false 扱いしたりしやすい。`bin/jq-jit.rs` の `select_cmp_passes` ヘルパに合わせる（型判定は JSON 値の先頭バイトで足りる）。

### Fast path と `?` / `try` の相互作用

raw-byte fast path は「missing field を null で返す」動きをするものが多い。object 上の `.a` なら正しいが、**非 object 入力**（数値・文字列・boolean）では jq が `Cannot index ... with string` で error するところを null にしてしまう。すると `(.a)?` が「error → empty」ではなく「null → null を yield」になって divergence する（#50）。

対策:

- `detect_expr` は top-level `try EXPR`（Empty catch）を **剥がさない**。剥がすと fast path 一式が `(expr)?` でもそのまま発火して、null-masking が観測される。今は剥がしていない。
- 新しい raw-byte fast path を足す時は、**非期待型の入力でも正しく error を raise する**ように書くか、そうでないなら入力型を先頭バイトで判別して bail。bail は構造的に名前付きで返す: `src/fast_path.rs` の `RawApplyOutcome::Bail`（#83 Phase B、pilot は `apply_field_access_raw`）。caller 側で `RawApplyOutcome::Bail` の時は `process_input` 経由で `Filter::execute_cb` に落とすと、Phase A の typed probe + generic eval が引き継ぐ。inline `match raw[0]` の implicit fall-through だと「commit point」が review で見えないので、新規 apply-site は必ずこの形を踏襲する。
- 検出（手動）: `diff ループ` で `f` と `(f)?` の両方を jq と突き合わせる。`(f)?` だけ divergence したら null-masking。
- 検出（自動・常設）: `tests/fast_path_error_wrap_proptest.rs` (#172) が
  ランダムな single-valued な `(filter, input)` を生成して、jq が error する時
  `jq-jit -c '(<filter>)?' <input>` が必ず empty を返すことをアサーションする。
  raw-byte fast path が type-check 漏れで silent な値を吐くと、ここで shrinker
  が最小再現を出すので、`tests/regression.test` にそのまま貼って修正に進める。
  `cargo test --release` で default 実行（200 cases / <30s）、
  `JQJIT_PROPTEST_CASES=2000` で deep stress も可能。
  generator は `Pipe(_, all-constant-rhs)` のような既知の別バグ shape を
  `prop_filter` で除外している。修正で除外を緩めると検出力が広がる。

### simplify_expr は runtime 型チェックを畳み込まない

compile-time fold が「runtime で error するはずのケース」を正常値にしてしまうと、silent divergence になる。今まで踏んだやつ:

- `path(.a) → ["a"]` の畳み込み（#46）。入力が object じゃなければ jq は `Cannot index number with string "a"` で死ぬが、畳み込みはそれを殺す。
- `[lits] | .[N]` で negative overshoot を `.max(0)` でクランプ（#42）。`[1,2,3] | .[-4]` が `elems[0]` になって 1 を返す。
- `[gen] | .[N]` 系全般：`gen` が多値だと単一要素前提の fold で先頭しか拾えない。

原則: **畳み込んだ結果が "runtime でも絶対にその値"** である時だけ OK。型に依存する access（`path`, array index, object key）は基本畳めない。畳む時はレンジ外や型ミスマッチを "fall through して runtime に任せる" 方向に倒す（runtime は null を返すか error する）。

### JIT の f64 fast path は Num guard が必要

`ToF64Var` / `jit_rt_to_f64` は非 number を silent に `0.0` にする。fused f64 ループ（`Expr::Until` / `Expr::While` / `while` / `until` の narrow & general path）に raw 入力を突っ込むと、`"hello" | until(. > 5; . + 1)` が 6 を返す（#57）。

対策: fused f64 ループに入る前に runtime 型 guard:

```rust
self.emit(JitOp::TypeCmpBranch {
    src: input_slot,
    tags: 1u8 << 3,   // TAG_NUM (value.rs::TAG_NUM)
    then_label: fused_label,
    else_label: generic_label,
});
```

`else_label` 側は `flatten_scalar` で cond/update をセマンティックに評価する generic path に逃がす。cross-type 比較（`"hello" > 5` = true）を honour するのはそこだけ。新しい f64 fused path を書く時は必ずこの guard を付ける。タグ値は `src/value.rs` の `TAG_*` 定数。

---

## 4. 典型的なデバッグの流れ

1. **まず diff ループで再現** — input と filter の最小組を特定
2. **どの fast path が走ってるか特定** — `JQJIT_TRACE=1` を付けて stderr を見る（§2(a) 参照）。generic path 落ちは `matched=eval` / `matched=jit`
3. **発火していない場合は eval / simplify** — `src/eval.rs` と `src/interpreter.rs` の `simplify_expr`
4. **JIT でしか起きない**: JIT は input が大きい時 or `has_loop_constructs` の時にしか使われない。`echo null | …` で再現する時は後者（Update/Reduce/Foreach を含む filter）
5. **直したら regression test に入れる** — `tests/regression.test` に 3 行形式で追加。1 修正 = 1 テスト以上
