# Cranelift JIT コンパイラバックエンド 調査レポート

> 調査日: 2026-02-27
> 目的: jq バイトコード → ネイティブコード JIT コンパイルのバックエンド候補としての Cranelift 評価

## 1. 基本アーキテクチャ

### 概要

Cranelift は [Bytecode Alliance](https://bytecodealliance.org/) が開発する、高速・安全・比較的シンプルなコンパイラバックエンド。LLVM の約 1/100 のコード量（約 20 万行 vs 2000 万行超）で、コンパイル速度を優先した設計。

**ライセンス**: Apache-2.0 WITH LLVM-exception（商用利用・リンクに制約なし）。**ガバナンス**: Bytecode Alliance（Mozilla, Fastly, Intel, Red Hat 等が支援する非営利団体）が主導。Wasmtime の中核コンポーネントとして積極開発が継続。月次リリースサイクル。

- リポジトリ: https://github.com/bytecodealliance/wasmtime/tree/main/cranelift
- 公式サイト: https://cranelift.dev/

### クレート構成

| クレート | 役割 |
|---|---|
| `cranelift-codegen` | コア。CLIF IR → マシンコード生成 |
| `cranelift-frontend` | IR 構築ヘルパー。`FunctionBuilder` / `Variable` による SSA 変換を自動化 |
| `cranelift-module` | 複数の関数・データオブジェクトをまとめて管理 |
| `cranelift-jit` | JIT バックエンド。メモリ上にコード・データを emit し、直接実行可能にする |
| `cranelift-native` | ホストマシンの ISA を自動検出 |
| `cranelift` | 上記を re-export するメタクレート |

### CLIF IR の構造

CLIF (Cranelift IR Format) は SSA 形式の中間表現。

- **関数**: コンパイルの基本単位
- **基本ブロック (Block)**: 末尾は必ずターミネータで終わる。フォールスルーなし
- **SSA 値 (Value)**: 一度だけ定義、任意回数使用
- **ブロックパラメータ**: phi 命令の代わりに使用

```
function u0:0(i64, i64) -> i64 {
block0(v0: i64, v1: i64):
    v2 = iadd v0, v1
    return v2
}
```

**型システム**: `i8`-`i128`, `f32`, `f64`, ベクトル型, `iAddr`（ポインタサイズ整数）

### コンパイルパイプライン

```
CLIF IR (SSA)
    → Mid-end 最適化 (Acyclic E-graphs: GVN, LICM, 定数畳み込み)
    → Lowering / Instruction Selection (ISLE DSL)
    → VCode (仮想レジスタ、ターゲット依存)
    → Register Allocation (regalloc2)
    → Code Emission → ネイティブコード
```


## 2. JIT 実行の流れ

### 最小限の JIT 例: 整数加算

```rust
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Module};

fn main() {
    // 1. ISA 設定
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa_builder = cranelift_native::builder()
        .expect("host machine not supported");
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .unwrap();

    // 2. JITModule 作成
    let builder = JITBuilder::with_isa(isa, default_libcall_names());
    let mut module = JITModule::new(builder);
    let mut ctx = module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();

    // 3. シグネチャ定義: fn add(i64, i64) -> i64
    let int = module.target_config().pointer_type();
    ctx.func.signature.params.push(AbiParam::new(int));
    ctx.func.signature.params.push(AbiParam::new(int));
    ctx.func.signature.returns.push(AbiParam::new(int));

    // 4. 関数宣言
    let func_id = module
        .declare_function("add", cranelift_module::Linkage::Export, &ctx.func.signature)
        .unwrap();

    // 5. CLIF IR 構築
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let a = builder.block_params(entry)[0];
        let b = builder.block_params(entry)[1];
        let sum = builder.ins().iadd(a, b);
        builder.ins().return_(&[sum]);

        builder.finalize();
    }

    // 6. コンパイル
    module.define_function(func_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);

    // 7. リンク・リロケーション解決
    module.finalize_definitions().unwrap();

    // 8. 関数ポインタ取得 & 実行
    let code_ptr = module.get_finalized_function(func_id);
    let add_fn = unsafe { std::mem::transmute::<*const u8, fn(i64, i64) -> i64>(code_ptr) };

    let result = add_fn(3, 4);
    println!("3 + 4 = {}", result); // => 7
}
```

### Cargo.toml

```toml
[dependencies]
cranelift = "0.129"
cranelift-jit = "0.129"
cranelift-module = "0.129"
cranelift-native = "0.129"
```

> **Note**: 2026-02-25 時点の最新は 0.129.1。Cranelift の全クレートは Wasmtime と同期リリースされるため、バージョンを必ず揃えること。

### JITModule 主要 API

| メソッド | 説明 |
|---|---|
| `declare_function(name, linkage, sig)` | 関数シンボルを登録 |
| `define_function(func_id, ctx)` | IR をコンパイル |
| `finalize_definitions()` | リロケーション解決 |
| `get_finalized_function(func_id)` | コード `*const u8` を取得 |
| `free_memory(self)` | コード/データメモリを解放 |


## 3. CLIF IR の構築パターン

### Variable と SSA

```rust
let x = Variable::new(0);                // 一意なインデックスで Variable を作成
builder.declare_var(x, types::I64);      // 型を宣言
builder.def_var(x, some_value);          // 変数に代入
let val = builder.use_var(x);            // 現在の値を取得（SSA 値）
```

`Variable::new(index)` のインデックスは関数内で一意であれば任意の `usize` でよい。`declare_var` は `Variable` と型を別引数で受け取る。

### 分岐（if-else）

```rust
let then_block = builder.create_block();
let else_block = builder.create_block();
let merge_block = builder.create_block();
builder.append_block_param(merge_block, types::I64); // phi の代替

builder.ins().brif(cond, then_block, &[], else_block, &[]);

// then → merge（値を渡す）
builder.switch_to_block(then_block);
builder.seal_block(then_block);
builder.ins().jump(merge_block, &[then_val]);

// else → merge（値を渡す）
builder.switch_to_block(else_block);
builder.seal_block(else_block);
builder.ins().jump(merge_block, &[else_val]);

// merge: ブロックパラメータで phi 値を受け取る
builder.switch_to_block(merge_block);
builder.seal_block(merge_block);
let result = builder.block_params(merge_block)[0];
```

### ループ

ループヘッダのシールはバックエッジ定義後に行う:

```rust
let loop_header = builder.create_block();
let loop_body = builder.create_block();
let loop_exit = builder.create_block();

builder.ins().jump(loop_header, &[]);

builder.switch_to_block(loop_header);
// ★ まだシールしない（バックエッジが未定義）
let cond = builder.ins().icmp(IntCC::SignedLessThan, i_val, limit);
builder.ins().brif(cond, loop_body, &[], loop_exit, &[]);

builder.switch_to_block(loop_body);
builder.seal_block(loop_body);
builder.ins().jump(loop_header, &[]); // バックエッジ

builder.seal_block(loop_header); // ★ バックエッジ定義完了後にシール

builder.switch_to_block(loop_exit);
builder.seal_block(loop_exit);
```


## 4. FFI と外部ランタイムの呼び出し

### アプローチ 1: JITBuilder にシンボル登録（推奨）

```rust
extern "C" fn jq_builtin_length(val: *const JvValue) -> i64 {
    unsafe { (*val).length() }
}

let mut jit_builder = JITBuilder::with_isa(isa, default_libcall_names());
jit_builder.symbol("jq_builtin_length", jq_builtin_length as *const u8);
```

JIT コード内から `Linkage::Import` + `call` で呼び出し。

### アプローチ 2: call_indirect（動的ディスパッチ）

```rust
let sig_ref = builder.import_signature(sig);
let fn_ptr = builder.ins().iconst(types::I64, fn_addr);
let call = builder.ins().call_indirect(sig_ref, fn_ptr, &[arg]);
```

### jq ビルトイン呼び出しの設計指針

1. すべてのビルトイン関数を `extern "C"` のラッパーとして用意
2. `JITBuilder::symbols()` で一括登録
3. ランタイムコンテキストはポインタで渡す


## 5. 制御フロー

### setjmp/longjmp 的な非局所脱出

**Cranelift 自体には setjmp/longjmp のネイティブサポートはない。**

- `setjmp` は「2 回返る関数」であり、SSA ベースの IR と根本的に相性が悪い
- Wasmtime でもかつて使っていたが `try_call` で置き換えられた

### try_call（例外ハンドリング）

2025年4月に初期実装マージ（[PR #10510](https://github.com/bytecodealliance/wasmtime/pull/10510)）、2025年8月にフル実装完了。Wasmtime で本番利用中。

```
try_call fn0(v1), block_normal(ret0), [ tag0: block_handler(exn0) ]
```

- `try_call` はブロックターミネータとして機能し、正常パスと例外ハンドラの両方を指定
- ハンドラは通常の基本ブロック。特殊な構造体は不要
- ゼロコスト例外処理（正常パスにオーバーヘッドなし）
- jq の `try-catch` に使える可能性あり

### コルーチン / ファイバー

Cranelift 自体にはサポートなし。jq のジェネレータ実装には:
1. CPS 変換（推奨）
2. ホスト側のスタック切り替え
3. イテレータパターン


## 6. パフォーマンス特性

### コンパイル速度

| 比較 | 結果 | 出典 |
|---|---|---|
| Cranelift vs WAVM/LLVM (Wasm) | Cranelift が **約 10x 高速** | Jangda et al. 2019 ([arxiv:2011.13127](https://arxiv.org/abs/2011.13127))、Cranelift README |
| Cranelift vs LLVM (Rust ビルド) | Cranelift で **約 40% 高速**（CPU 時間ベース） | [Wasmtime 1.0 Performance](https://bytecodealliance.org/articles/wasmtime-10-performance) |
| Cranelift vs LLVM (クエリコンパイル) | Cranelift が **20-35% 高速** | Engelke et al. 2024 ([CGO 2024 論文](https://home.cit.tum.de/~engelke/pubs/2403-cgo.pdf)) |

> **Note**: コンパイル速度の差はワークロードにより大きく異なる。Wasm モジュールのような比較的単純な関数群では 10x 以上の差が出るが、複雑な最適化が効くケースでは差は縮まる。

### 生成コードの品質

| 比較 | 結果 | 出典 |
|---|---|---|
| Cranelift vs WAVM/LLVM | Cranelift が **約 14% 遅い** | Jangda et al. 2019、[sightglass](https://github.com/bytecodealliance/sightglass) ベンチマーク |
| Cranelift vs V8 TurboFan | Cranelift が **約 2% 遅い** | 同上 |


## 7. 実例プロジェクト

| プロジェクト | 用途 |
|---|---|
| [Wasmtime](https://github.com/bytecodealliance/wasmtime) | WebAssembly ランタイム |
| [rustc_codegen_cranelift](https://github.com/rust-lang/rustc_codegen_cranelift) | Rust コンパイラ代替バックエンド |
| [cranelift-jit-demo](https://github.com/bytecodealliance/cranelift-jit-demo) | 教育用トイ言語 JIT |
| [revmc](https://github.com/paradigmxyz/revmc) | Ethereum VM JIT/AOT |


## 8. メモリ管理・GC 統合パターン

### 参照カウント操作

Cranelift 自体に GC や参照カウントのプリミティブはない。参照カウント操作は `extern "C"` 関数呼び出しで実現する:

```rust
// ホスト側
extern "C" fn jv_ref_inc(val: *mut JvValue) { unsafe { (*val).refcount += 1; } }
extern "C" fn jv_ref_dec(val: *mut JvValue) { unsafe { (*val).refcount -= 1; /* free if 0 */ } }

// JITBuilder に登録
jit_builder.symbol("jv_ref_inc", jv_ref_inc as *const u8);
jit_builder.symbol("jv_ref_dec", jv_ref_dec as *const u8);
```

JIT コード内では通常の `call` 命令で `jv_ref_inc` / `jv_ref_dec` を呼ぶ。インライン化はできないが、参照カウント操作は通常ボトルネックにならない。

### Safepoint / Stack Map（GC 統合）

Cranelift は `cranelift-frontend` の「User Stack Maps」機能で GC safepoint をサポートする。

- **Safepoint**: GC が安全に実行できるプログラムポイント（通常は `call` 命令）
- **Stack Map**: 各 safepoint で GC 管理オブジェクトがスタック上のどこにあるかを記録

2024年に大規模リデザインが行われ、stack map 生成がコンパイラバックエンドからフロントエンドに移動:

- フロントエンド（`cranelift-frontend`）が GC 参照の liveness 解析、スタックスロット割り当て、spill/reload 挿入を担当
- ミッドエンド・バックエンド・レジスタアロケータは GC 参照を特別扱いしない
- 最適化パスが spill/reload を「見える」ため、misoptimization を防止

詳細: [New Stack Maps for Wasmtime and Cranelift](https://fitzgen.com/2024/09/10/new-stack-maps-for-wasmtime.html)

### Arena ベースのメモリ管理との統合

jq のように短命なデータが大量に生成されるワークロードでは、Arena アロケータとの統合が有効:

- JIT 関数にランタイムコンテキスト（Arena ポインタ含む）をポインタで渡す
- alloc/free は `extern "C"` 経由で Arena API を呼ぶ
- Arena のリセットは JIT コードの外（ホスト側）で行う

### Wasmtime での GC 統合の実例

Wasmtime は WebAssembly GC proposal の実装を進めており、Cranelift の stack map 機能を活用している。`r32` / `r64` 型で GC 参照を表現し、safepoint での自動 spill/reload を行う。jq JIT で同等の仕組みを使う場合は Wasmtime の実装（[wasmtime/crates/cranelift/](https://github.com/bytecodealliance/wasmtime/tree/main/crates/cranelift)）が参考になる。


## 9. デバッグとトラブルシューティング

### CLIF IR の表示

```rust
// コンパイル前の CLIF IR を表示
println!("{}", ctx.func.display());
```

出力例:

```
function u0:0(i64, i64) -> i64 system_v {
block0(v0: i64, v1: i64):
    v2 = iadd v0, v1
    return v2
}
```

### 生成マシンコードの Disassembly 取得

```rust
// コンパイル前に disasm フラグを有効化
ctx.set_disasm(true);

// コンパイル実行
module.define_function(func_id, &mut ctx).unwrap();

// disassembly を取得
if let Some(ref code) = ctx.compiled_code() {
    if let Some(ref disasm) = code.disasm {
        println!("{}", disasm);
    }
}
```

> **Note**: `set_disasm(true)` はコンパイル速度に影響するため、デバッグ時のみ使用すること。

### Verifier の有効化

```rust
// 方法 1: settings で有効化（全関数に適用）
flag_builder.set("enable_verifier", "true").unwrap();

// 方法 2: 個別に検証
ctx.verify(isa.as_ref()).expect("CLIF verification failed");

// 方法 3: enable_verifier 設定に従って条件付き検証
ctx.verify_if(isa.as_ref()).expect("CLIF verification failed");
```

Verifier は CLIF IR の型整合性、支配関係（dominator tree）、制御フローグラフの一貫性を検査する。開発中は常に有効にしておくのを推奨。

### 最適化を無効にしてデバッグしやすい IR を得る

```rust
flag_builder.set("opt_level", "none").unwrap();
```

`opt_level` は `none` / `speed` / `speed_and_size` の 3 段階。`none` では E-graph ベースの最適化（GVN, LICM, 定数畳み込み等）がスキップされ、IR が入力に近い形で残るためデバッグしやすい。

### トラブルシューティングのフロー

1. `ctx.func.display()` で CLIF IR を確認 — 意図した IR が構築されているか
2. `ctx.verify()` で IR の整合性を検証 — 型エラーや支配関係の不整合を早期発見
3. `opt_level = "none"` で最適化起因の問題を切り分け
4. `set_disasm(true)` で生成マシンコードを確認 — lowering やレジスタ割り当ての問題を特定


## 10. try_call を jq バックトラッキングに使う設計スケッチ

jq の `FORK` / `BACKTRACK` バイトコードは「複数の出力を生成し、失敗時に次の分岐を試す」という非局所的な制御フローを表現する。`try_call` を使うことで、CPS 変換なしにこれをモデリングできる可能性がある。

### 概念

```
// jq バイトコード: FORK label → 「現在の継続を保存し、失敗したら label へ飛ぶ」
// Cranelift: try_call で各分岐を「関数呼び出し + 例外ハンドラ」としてモデル化

block_fork:
    try_call emit_first_alternative(ctx),
        block_success(ret),
        [ backtrack_tag: block_try_next() ]

block_try_next:
    try_call emit_second_alternative(ctx),
        block_success(ret),
        [ backtrack_tag: block_fail() ]

block_success(v0: i64):
    // 出力を yield
    ...

block_fail:
    // バックトラック失敗 → 上位へ例外伝播
    throw backtrack_tag
```

### CPS 方式との比較

| 観点 | try_call 方式 | CPS 変換方式 |
|---|---|---|
| 実装の自然さ | jq の FORK/BACKTRACK に直接対応 | 全体を継続渡しに変換する必要がある |
| スタック使用量 | 深いバックトラックでスタックが深くなる | ヒープに継続を確保するため制御可能 |
| 正常パスの性能 | ゼロコスト（例外が飛ばなければオーバーヘッドなし） | 継続オブジェクトの alloc/free コスト |
| 複雑なジェネレータ | 多段 try_call のネストが必要で複雑になりうる | 継続の合成で自然に表現可能 |
| Cranelift との親和性 | ネイティブ機能を直接使える | IR 構築が複雑になるが制約なし |

### 推奨

単純な `try-catch` パターンには `try_call` が適する。一方、jq のジェネレータ（`foreach`, `,` オペレータ等）のような多値出力は CPS 変換の方が自然。ハイブリッドアプローチ（ジェネレータは CPS、エラーハンドリングは `try_call`）も検討に値する。


## 11. 制限事項と注意点

| 制限 | 影響 |
|---|---|
| 可変引数関数 (varargs) 未サポート | ラッパー関数で対応 |
| setjmp/longjmp 未サポート | try_call またはホスト側で対応 |
| LLVM レベルの最適化なし | 生成コード品質は 10-15% 劣る |
| デバッグ情報 (DWARF) 限定的 | ソースレベルデバッグは困難 |

### Apple Silicon (AArch64) サポート

- **Tier 1 サポート**（x86-64 と同等）
- `cranelift_native::builder()` で自動検出、特別な対応不要
- W^X ポリシーは `JITModule` が内部で処理

### jq JIT への推奨事項

1. **Cranelift は JIT バックエンドとして適切**
2. ビルトイン関数は `extern "C"` + `JITBuilder::symbols()` で登録
3. バックトラッキングは `try_call` + CPS のハイブリッドが有望（詳細はセクション 10 参照）
4. Apple Silicon サポートは問題なし
5. デバッグは `ctx.func.display()` と `ctx.verify()` から始める（詳細はセクション 9 参照）


## 参考資料

### 公式ドキュメント・API

- [Cranelift IR ドキュメント](https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/docs/ir.md)
- [FunctionBuilder docs.rs](https://docs.rs/cranelift-frontend/latest/cranelift_frontend/struct.FunctionBuilder.html)
- [JITModule docs.rs](https://docs.rs/cranelift-jit/latest/cranelift_jit/struct.JITModule.html)
- [Context docs.rs](https://docs.rs/cranelift-codegen/latest/cranelift_codegen/struct.Context.html)
- [cranelift-jit-demo](https://github.com/bytecodealliance/cranelift-jit-demo)

### ベンチマーク・パフォーマンス

- [Wasmtime 1.0 Performance](https://bytecodealliance.org/articles/wasmtime-10-performance) — コンパイル速度・生成コード品質の公式ベンチマーク
- [Cranelift Progress in 2022](https://bytecodealliance.org/articles/cranelift-progress-2022) — E-graph 最適化導入時のパフォーマンス改善
- [Not So Fast: Analyzing the Performance of WebAssembly vs. Native Code](https://arxiv.org/abs/2011.13127) — Cranelift vs LLVM vs V8 の比較数値の原典
- [Compile-Time Analysis of Compiler Frameworks for Query Compilation (CGO 2024)](https://home.cit.tum.de/~engelke/pubs/2403-cgo.pdf) — クエリコンパイル文脈での比較
- [sightglass ベンチマークスイート](https://github.com/bytecodealliance/sightglass) — Bytecode Alliance 公式ベンチマークツール

### 例外処理・try_call

- [Exceptions in Cranelift and Wasmtime](https://cfallin.org/blog/2025/11/06/exceptions/) — try_call の設計と実装経緯
- [try_call 初期実装 PR #10510](https://github.com/bytecodealliance/wasmtime/pull/10510)

### GC・メモリ管理

- [New Stack Maps for Wasmtime and Cranelift](https://fitzgen.com/2024/09/10/new-stack-maps-for-wasmtime.html) — safepoint / stack map リデザインの詳細
