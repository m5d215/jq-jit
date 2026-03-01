# jq-jit AOT コンパイル: フィルタをスタンドアロン実行バイナリに

> **状態: 完了** — 全ステップ完了。実装記録: [aot-record.md](aot-record.md)

## Context

jq-jit は Criterion ベンチマークで libjq 比 55x〜18,000x の高速化を達成しているが、CLI wall-clock では jq より遅い (0.6x〜1.15x)。原因は毎回の JIT コンパイル (~450µs) と JSON パーサの差。

AOT (Ahead-of-Time) コンパイルで jq フィルタをスタンドアロンの実行バイナリに変換する。JIT コンパイルと libjq 依存を除去し、CLI 実行でも JIT の速さを活かす。

```bash
jq-jit --compile '. + 1' -o add1
echo 3 | ./add1  # → 4
```

## アーキテクチャ

### 現在の JIT パイプライン

```
filter → libjq bytecode → CPS IR → Cranelift JITModule → memory → execute
                                    ^^^^^^^^^^^^^^^^^^^^^^^^
                                    毎回 ~450µs
```

### AOT パイプライン

```
[コンパイル時]
filter → libjq bytecode → CPS IR → Cranelift ObjectModule → filter.o
                                                              ↓
                                              cc filter.o -ljq_jit_runtime → ./add1

[実行時]
stdin → JSON parse → jit_filter() [ネイティブコード] → output
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     JIT overhead = 0
```

### 核心のポイント

- `cranelift-object` の `ObjectModule` は `cranelift-jit` の `JITModule` と同じ `Module` トレイトを実装
- 既存の codegen ロジック（CLIF IR 生成部分）はそのまま流用可能
- ランタイム関数は既に `Linkage::Import` で宣言されており、ObjectModule でもそのまま動く
- 出力先が「メモリ」から「.o ファイル」に変わるだけ

## 実装ステップ

### Step 1: codegen.rs のリファクタ — Module 型の抽象化

`compile_expr` の中で JITModule 固有なのは以下だけ:
1. JITBuilder の作成とシンボル登録 (L158-349)
2. `module.finalize_definitions()` + `module.get_finalized_function()` (L1164-1170)

**方針**: CLIF codegen 部分 (L350-1162) をジェネリック関数に抽出。JIT/AOT は初期化と finalization だけ分岐。

```rust
// 共通コア: Module トレイトのジェネリック関数
fn build_filter<M: Module>(
    module: &mut M,
    ctx: &mut cranelift_codegen::Context,
    func_ctx: &mut FunctionBuilderContext,
    expr: &Expr,
    ptr_ty: types::Type,
) -> Result<(FuncId, Vec<Box<Value>>, String)> {
    // L354-1162 の内容をそのまま移動
    // declare_bin/declare_unary のクロージャを &mut M で受ける
}

// JIT パス (既存)
pub fn compile_expr(expr: &Expr) -> Result<(JitFilter, String)> {
    let isa = setup_isa()?;
    let mut jit_builder = JITBuilder::with_isa(isa.clone(), default_libcall_names());
    register_symbols(&mut jit_builder);  // 100+ symbols
    let mut module = JITModule::new(jit_builder);
    let (func_id, literals, clif) = build_filter(&mut module, ...)?;
    module.finalize_definitions()?;
    let code_ptr = module.get_finalized_function(func_id);
    Ok((JitFilter { fn_ptr: code_ptr, _module: module, _literals: literals }, clif))
}

// AOT パス (新規)
pub fn compile_to_object(expr: &Expr) -> Result<(Vec<u8>, String)> {
    let isa = setup_isa_for_aot()?;  // is_pic=true
    let obj_builder = ObjectBuilder::new(isa, "jq_filter", default_libcall_names())?;
    let mut module = ObjectModule::new(obj_builder);
    let (func_id, literals, clif) = build_filter(&mut module, ...)?;
    // リテラルを .rodata セクションに埋め込む (Step 6)
    // main() トランポリン関数を生成 (Step 2)
    let product = module.finish();
    let mut obj_bytes = Vec::new();
    product.object.write_stream(&mut obj_bytes)?;
    Ok((obj_bytes, clif))
}
```

**変更ファイル**: `src/codegen.rs`

**注意**: `declare_bin`/`declare_unary` クロージャが `&mut JITModule` を取っている箇所を `&mut M` に変更。`Module` トレイトの `declare_function` は共通 API なのでそのまま動く。

### Step 2: ObjectModule 用の main 関数生成

filter.o に `jit_filter` だけでなく `main` 関数も Cranelift で生成する。

```
main(argc, argv):
    filter_addr = func_addr(jit_filter)
    return call aot_run(filter_addr, argc, argv)
```

`aot_run` はランタイムライブラリ側に実装する Rust 関数 (Step 3)。
Cranelift の `func_addr` 命令で同一モジュール内の `jit_filter` のアドレスを取得して渡す。

**変更ファイル**: `src/codegen.rs` (`compile_to_object` 内に追加)

### Step 3: AOT ランタイムエントリ — `src/aot.rs`

スタンドアロンバイナリの実行ロジック。フィルタ関数ポインタを受け取り、stdin を処理する。

```rust
// src/aot.rs

type FilterFn = extern "C" fn(*const Value, extern "C" fn(*const Value, *mut u8), *mut u8);

/// AOT バイナリのエントリポイント。main() から呼ばれる。
#[no_mangle]
pub extern "C" fn aot_run(filter_fn: *const u8, argc: i64, argv: *const *const u8) -> i64 {
    // 1. argc/argv から CLI フラグをパース (-r, -c, -n, -s, --tab, --indent)
    // 2. stdin を読む
    // 3. JSON パース (serde_json)
    // 4. filter_fn を呼ぶ (collect_callback でVecに蓄積)
    // 5. 結果を出力
    // 6. exit code を返す
}
```

`jq-jit.rs` の `run()` 関数 (L409-) からフィルタのコンパイル部分を除いたもの。入出力ロジック・フラグパース・フォーマッティングを流用する。

**変更ファイル**: `src/aot.rs` (新規), `src/lib.rs` (pub mod aot 追加)

### Step 4: Static library ビルド

```toml
# Cargo.toml 変更
[lib]
crate-type = ["lib", "staticlib"]

[dependencies]
cranelift-object = "0.129"  # 追加
```

`cargo build --release` で `libjq_jit.a` が生成される。これに含まれるもの:
- 全 rt_* 関数 (~100個)
- Value 型と JSON 変換
- `aot_run` エントリポイント
- serde_json, regex 等の依存

### Step 5: リンカ呼び出し — `--compile` フラグ

```rust
// src/bin/jq-jit.rs に追加

fn compile_aot(filter: &str, output: &str) -> i32 {
    // 1. filter → bytecode → IR
    let ir = compile_filter_to_ir(filter)?;

    // 2. IR → object file
    let (obj_bytes, _) = compile_to_object(&ir)?;
    let obj_path = format!("{}.o", output);
    std::fs::write(&obj_path, &obj_bytes)?;

    // 3. リンク
    let status = std::process::Command::new("cc")
        .args([
            "-o", output,
            &obj_path,
            &format!("-L{}", runtime_lib_dir()),
            "-ljq_jit",
        ])
        .args(native_static_libs())  // -lSystem -lm etc.
        .status()?;

    // 4. クリーンアップ
    std::fs::remove_file(&obj_path).ok();
    if status.success() { 0 } else { 1 }
}
```

**リンカフラグの取得**: ビルド時に `rustc --print native-static-libs` の出力を `build.rs` でキャプチャし、バイナリに埋め込む (env! マクロ or include_str!)。

**ランタイムライブラリパス**: `jq-jit` バイナリと同じディレクトリ or `$JQ_JIT_LIB_DIR` 環境変数。

**変更ファイル**: `src/bin/jq-jit.rs`, `build.rs`

### Step 6: リテラル値の処理

現在の JIT では文字列リテラルを `Vec<Box<Value>>` にヒープ確保し、そのアドレスを定数として CLIF に埋め込む。AOT ではバイナリ実行時にアドレスが変わるため、この方式は使えない。

**方針**: ObjectModule の `.rodata` セクションにリテラルデータを配置し、リロケーション経由でアドレスを解決。

```rust
// ObjectModule のデータセクション機能を使う
let data_id = module.declare_data("lit_0", Linkage::Local, false, false)?;
let mut data_ctx = DataContext::new();
data_ctx.define(literal_bytes);
module.define_data(data_id, &data_ctx)?;

// CLIF IR 内でデータのアドレスを取得
let gv = module.declare_data_in_func(data_id, builder.func);
let addr = builder.ins().global_value(ptr_ty, gv);
```

Value は 16 bytes (`#[repr(C, u64)]`)。Num と Null/Bool はバイト列として埋め込める。
Str/Arr/Obj は Rc を含むため、初期化関数を生成して起動時に構築する。

## 技術的リスク

| リスク | レベル | 対策 |
|---|---|---|
| `cc` リンカの可用性 | 低 | macOS は Xcode CLT、Linux は build-essential。開発者なら入ってる |
| Rust std のリンクフラグ | 中 | `rustc --print native-static-libs` で取得。macOS/Linux で異なる |
| Str リテラルの AOT 対応 | 中 | 起動時初期化関数で Rc を構築。Num/Null/Bool は .rodata 直接埋め込み |
| static library のサイズ | 低 | Cranelift も含まれるが `-dead_strip` で除去される。最適化は後回し |
| `is_pic` 設定 | 中 | ObjectModule では `is_pic=true` が必要な場合がある（macOS PIE） |

## 既存コードの参照先

| 内容 | ファイル | 行 |
|---|---|---|
| JIT compile_expr | `src/codegen.rs` | L138-1180 |
| JITBuilder シンボル登録 | `src/codegen.rs` | L158-349 |
| CLIF codegen コア | `src/codegen.rs` | L354-1152 |
| finalize + get_fn_ptr | `src/codegen.rs` | L1164-1170 |
| declare_bin/declare_unary | `src/codegen.rs` | L402-411 |
| RuntimeFuncRefs 構造体 | `src/codegen.rs` | L1183+ |
| JitFilter::execute | `src/codegen.rs` | L104-131 |
| CLI run() 関数 | `src/bin/jq-jit.rs` | L409-444 |
| CLI フラグパース | `src/bin/jq-jit.rs` | 冒頭 |
| CLI 入力読み込み | `src/bin/jq-jit.rs` | L446+ |
| CLI 出力フォーマット | `src/bin/jq-jit.rs` | L380-403 |
| Value 型定義 | `src/value.rs` | |
| ランタイム関数群 | `src/runtime.rs` | ~100 関数 |
| lib.rs モジュール宣言 | `src/lib.rs` | |

## 検証方法

```bash
# ビルド
cargo build --release

# 基本動作
jq-jit --compile '. + 1' -o /tmp/add1
echo 3 | /tmp/add1                  # → 4

# 複雑なフィルタ
jq-jit --compile '.[] | select(. > 2)' -o /tmp/filter
echo '[1,2,3,4,5]' | /tmp/filter   # → 3\n4\n5

# CLI フラグ
jq-jit --compile '.name' -o /tmp/getname
echo '{"name":"world"}' | /tmp/getname -r   # → world (quotes なし)

# 性能比較
hyperfine 'echo 42 | jq ". + 1"' 'echo 42 | /tmp/add1'

# 既存テスト回帰なし
cargo run --bin jq-jit-test
bash tests/compat.sh target/release/jq-jit
bash tests/official/run.sh target/release/jq-jit tests/official/jq.test
```

## 実装順序

Step 1 (codegen リファクタ) → Step 4 (Cargo.toml) → Step 3 (aot.rs) → Step 2 (main 生成) → Step 6 (リテラル) → Step 5 (リンカ呼び出し)

Step 1 が最大の作業量。既存の JIT パスを壊さないよう慎重に進める。
