# jq-jit フィルタキャッシュ: 同じフィルタの再コンパイルを省く

> **状態: 完了** — 全 6 Step 完了。実装記録: [cache-record.md](cache-record.md)

## Context

jq-jit の CLI wall-clock は ~3ms（うち JIT コンパイル ~450µs、プロセス起動 ~2ms）。同じフィルタを繰り返し実行する場合、毎回 Cranelift コンパイルを走らせるのは無駄。

フィルタをコンパイル済みの共有ライブラリ (.dylib/.so) としてキャッシュし、2回目以降は dlopen で即ロードする。

```bash
echo 42 | jq-jit '. + 1'   # 初回: compile + cache + execute (~3ms)
echo 42 | jq-jit '. + 1'   # 2回目: dlopen + execute (~2.5ms, JIT 0)
```

## AOT (PLAN-AOT.md) との関係

| 共通 | キャッシュ固有 | AOT 固有 |
|---|---|---|
| Step 1: codegen リファクタ (Module 抽象化) | dlopen/dlsym でロード | main 関数生成 |
| Step 6: リテラル処理 | キャッシュディレクトリ管理 | aot.rs エントリポイント |
| ObjectModule → .o 出力 | `cc -shared` で .dylib 生成 | `cc` でフルリンク |
| | | static library ビルド |
| | | リンカフラグ管理 |

**キャッシュを先にやれば、AOT の Step 1 と Step 6 が自然に完成する。**

## アーキテクチャ

### フロー

```
jq-jit '. + 1'
  ↓
filter_hash = sha256(filter_string + jq_jit_version)
  ↓
~/.cache/jq-jit/<hash>.dylib が存在する？
  ├── YES → dlopen → dlsym("jit_filter") → execute
  └── NO  → libjq compile → IR → ObjectModule → .o
            → cc -shared → .dylib → cache に保存
            → dlopen → execute
```

### 共有ライブラリのリンク

フィルタの .o には rt_* への未解決参照がある。通常のリンクでは全 rt_* を含むライブラリが必要だが、共有ライブラリでは不要:

```bash
# macOS: 未解決シンボルを実行時解決に委ねる
cc -shared -undefined dynamic_lookup -o filter.dylib filter.o

# Linux: 共有ライブラリのデフォルト動作
cc -shared -o filter.so filter.o
```

dlopen 時に jq-jit プロセスが持つ rt_* シンボルで解決される。ランタイムライブラリのリンクは一切不要。

## 実装ステップ

### Step 1: codegen リファクタ (AOT と共通)

`compile_expr` から CLIF コード生成部分を抽出してジェネリック化。

```rust
fn build_filter<M: Module>(module: &mut M, ...) -> Result<(FuncId, Vec<Box<Value>>, String)>
```

JITModule 固有の部分:
- JITBuilder のシンボル登録 → JIT パスにのみ残す
- finalize + get_fn_ptr → JIT パスにのみ残す

ObjectModule 固有の部分:
- ObjectBuilder::new (is_pic=true)
- module.finish() → .emit() で .o バイト列

**変更ファイル**: `src/codegen.rs`
**依存追加**: `cranelift-object = "0.129"`

### Step 2: リテラル処理 (AOT と共通)

**問題**: 現在のリテラル処理は `Box<Value>` のヒープアドレスを `iconst` でハードコードする。別プロセスで dlopen した共有ライブラリでは、このアドレスは無効。

**解決策**: リテラルを共有ライブラリの外から注入する。

```
[現在 - JIT]
iconst(0x7fff12345678)  ← Box<Value> の絶対アドレス

[変更後 - JIT/AOT 共通]
load(literals_base + idx * 8)  ← リテラルテーブルからの間接参照
```

具体的には:

1. `jit_filter` のシグネチャに `literals: *const *const Value` を追加:
   ```
   fn jit_filter(input: *const Value, callback: fn, ctx: *mut u8, literals: *const *const Value)
   ```

2. `literal_to_value` で `iconst(ptr)` の代わりに `load(literals_param + idx * ptr_size)` を生成

3. `JitFilter::execute` が `literals` テーブルを構築して渡す

4. キャッシュロード時も同じ `literals` テーブルを再構築して渡す

**リテラル再構築**: フィルタの IR から決定論的に再構築可能。IR を serialize して .dylib と一緒にキャッシュする。

**変更ファイル**: `src/codegen.rs`, `src/codegen.rs` (JitFilter::execute)

### Step 3: compile_to_shared_object

ObjectModule でフィルタを .o にコンパイルし、`cc -shared` で .dylib を生成。

```rust
// src/codegen.rs に追加

pub fn compile_to_shared_object(expr: &Expr, output_path: &Path) -> Result<LiteralTable> {
    let isa = setup_isa_pic()?;  // is_pic=true
    let obj_builder = ObjectBuilder::new(isa, "jq_filter", default_libcall_names())?;
    let mut module = ObjectModule::new(obj_builder);

    let (func_id, literals, _clif) = build_filter(&mut module, ...)?;

    // .o ファイルを出力
    let product = module.finish();
    let obj_path = output_path.with_extension("o");
    let mut f = File::create(&obj_path)?;
    product.object.write_stream(&mut f)?;

    // cc -shared で .dylib/.so を生成
    let shared_flag = if cfg!(target_os = "macos") {
        vec!["-shared", "-undefined", "dynamic_lookup"]
    } else {
        vec!["-shared"]
    };
    Command::new("cc")
        .args(&shared_flag)
        .args(["-o", output_path.to_str().unwrap(), obj_path.to_str().unwrap()])
        .status()?;

    std::fs::remove_file(&obj_path)?;  // .o クリーンアップ

    // リテラル情報を返す (キャッシュ時にシリアライズ)
    Ok(LiteralTable::from_expr(expr))
}
```

**変更ファイル**: `src/codegen.rs`

### Step 4: キャッシュマネージャ — `src/cache.rs`

```rust
// src/cache.rs

use std::path::PathBuf;

pub struct FilterCache {
    cache_dir: PathBuf,  // ~/.cache/jq-jit/
}

impl FilterCache {
    pub fn new() -> Self {
        let dir = dirs::cache_dir()  // or home_dir + ".cache"
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("jq-jit");
        std::fs::create_dir_all(&dir).ok();
        Self { cache_dir: dir }
    }

    /// フィルタ文字列からキャッシュキーを生成
    fn cache_key(&self, filter: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        filter.hash(&mut hasher);
        env!("CARGO_PKG_VERSION").hash(&mut hasher);  // バージョン変更で無効化
        format!("{:016x}", hasher.finish())
    }

    /// キャッシュから JitFilter をロード。なければ None。
    pub fn load(&self, filter: &str) -> Option<CachedFilter> {
        let key = self.cache_key(filter);
        let dylib_path = self.cache_dir.join(format!("{}.dylib", key));
        let meta_path = self.cache_dir.join(format!("{}.meta", key));

        if !dylib_path.exists() || !meta_path.exists() {
            return None;
        }

        // dlopen
        let lib = unsafe { libc::dlopen(dylib_path.to_str()?.as_ptr() as *const _, libc::RTLD_NOW) };
        if lib.is_null() { return None; }

        // dlsym
        let sym = unsafe { libc::dlsym(lib, b"jit_filter\0".as_ptr() as *const _) };
        if sym.is_null() { return None; }

        // リテラル再構築 (meta ファイルから)
        let literals = LiteralTable::deserialize(&meta_path).ok()?;

        Some(CachedFilter { fn_ptr: sym, _lib: lib, literals })
    }

    /// コンパイルしてキャッシュに保存
    pub fn compile_and_store(&self, filter: &str, expr: &Expr) -> Result<CachedFilter> {
        let key = self.cache_key(filter);
        let dylib_path = self.cache_dir.join(format!("{}.dylib", key));
        let meta_path = self.cache_dir.join(format!("{}.meta", key));

        let lit_table = compile_to_shared_object(expr, &dylib_path)?;
        lit_table.serialize(&meta_path)?;

        self.load(filter).ok_or_else(|| anyhow!("failed to load cached filter"))
    }
}
```

**変更ファイル**: `src/cache.rs` (新規), `src/lib.rs` (pub mod cache 追加)
**依存追加**: なし (libc は既存依存)

### Step 5: CLI 統合

`src/bin/jq-jit.rs` の `run()` を変更して、キャッシュを透過的に使用。

```rust
fn run(opts: &CliOptions) -> i32 {
    let effective_filter = build_effective_filter(opts);
    let cache = FilterCache::new();

    // キャッシュヒット？
    let filter = if let Some(cached) = cache.load(&effective_filter) {
        cached
    } else {
        // キャッシュミス: 通常のコンパイルパイプライン
        let mut jq = JqState::new().unwrap();
        let bc = jq.compile(&effective_filter).unwrap();
        let ir = bytecode_to_ir(&bc).unwrap();

        // キャッシュに保存 (失敗しても JIT にフォールバック)
        match cache.compile_and_store(&effective_filter, &ir) {
            Ok(cached) => cached,
            Err(_) => {
                // フォールバック: 従来の JIT パス
                let (jit_filter, _) = compile_expr(&ir).unwrap();
                return run_with_jit_filter(&jit_filter, opts);
            }
        }
    };

    run_with_cached_filter(&filter, opts)
}
```

**変更ファイル**: `src/bin/jq-jit.rs`

### Step 6: CLI フラグ

```
--no-cache     キャッシュを使わない (デバッグ用)
--clear-cache  キャッシュを全削除
--cache-dir    キャッシュディレクトリを指定
```

**変更ファイル**: `src/bin/jq-jit.rs`

## リテラルテーブルの設計

### シリアライズ形式 (.meta ファイル)

```
[version: u32]
[num_literals: u32]
[literal_0_type: u8] [literal_0_data...]
[literal_1_type: u8] [literal_1_data...]
...
```

| Type | Format |
|---|---|
| Null (0) | なし |
| Bool (1) | u8 (0/1) |
| Num (2) | f64 (8 bytes) |
| Str (3) | u32 len + UTF-8 bytes |
| EmptyArr (4) | なし |
| Arr (5) | 再帰的に Value をシリアライズ |
| EmptyObj (6) | なし |
| Obj (7) | u32 num_entries + (key_len + key + value)* |
| Error (8) | u32 len + UTF-8 bytes |

### ランタイム再構築

```rust
impl LiteralTable {
    fn reconstruct(&self) -> Vec<Box<Value>> {
        self.entries.iter().map(|entry| {
            Box::new(match entry {
                LitEntry::Null => Value::Null,
                LitEntry::Num(n) => Value::Num(*n),
                LitEntry::Str(s) => Value::Str(Rc::new(s.clone())),
                // ...
            })
        }).collect()
    }
}
```

## 技術的リスク

| リスク | レベル | 対策 |
|---|---|---|
| dlopen のシンボル解決 | 中 | macOS は `-undefined dynamic_lookup`。Linux はデフォルトで OK。テストで確認 |
| `cc` の可用性 | 低 | 開発者マシンにはほぼ入ってる。なければ JIT フォールバック |
| リテラルテーブルの互換性 | 低 | バージョンをキャッシュキーに含めるので、バージョン更新で自動無効化 |
| キャッシュ肥大化 | 低 | TTL or LRU で古いエントリを削除。初期実装では手動 `--clear-cache` のみ |
| macOS の code signing | 中 | dlopen する .dylib は署名不要 (ad-hoc で十分)。`cc -shared` が自動で処理 |
| jit_filter シグネチャ変更 | 中 | 既存テスト全修正が必要。ただし変更は機械的 (引数追加) |

## 実装順序

```
Step 1 (codegen リファクタ)
  → Step 2 (リテラル間接参照化)
    → Step 3 (compile_to_shared_object)
      → Step 4 (cache.rs)
        → Step 5 (CLI 統合)
          → Step 6 (CLI フラグ)
```

Step 1-2 が最重要。これが動けば Step 3-6 は straightforward。

## 検証方法

```bash
cargo build --release

# 初回 (キャッシュミス)
echo 42 | jq-jit '. + 1'   # → 43, ~/.cache/jq-jit/<hash>.dylib が生成される

# 2回目 (キャッシュヒット)
echo 42 | jq-jit '. + 1'   # → 43, JIT コンパイルをスキップ

# キャッシュの確認
ls -la ~/.cache/jq-jit/

# 性能比較
hyperfine --warmup 1 \
  'echo 42 | jq-jit --no-cache ". + 1"' \
  'echo 42 | jq-jit ". + 1"'

# キャッシュクリア
jq-jit --clear-cache

# 既存テスト回帰なし
cargo run --bin jq-jit-test
bash tests/compat.sh target/release/jq-jit
bash tests/official/run.sh target/release/jq-jit tests/official/jq.test
```

## 既存コードの参照先

| 内容 | ファイル | 行 |
|---|---|---|
| リテラル → iconst | `src/codegen.rs` | L4822-4881 |
| JitFilter::execute | `src/codegen.rs` | L104-131 |
| compile_expr 全体 | `src/codegen.rs` | L138-1180 |
| CLI run() | `src/bin/jq-jit.rs` | L409-444 |
| Value 型定義 | `src/value.rs` | |
| CPS IR Literal enum | `src/cps_ir.rs` | |

## AOT への橋渡し

キャッシュ実装後、AOT に必要な追加作業:
- **Step 2 (main 生成)**: `compile_to_object` で main トランポリン関数を追加
- **Step 3 (aot.rs)**: aot_run エントリポイント
- **Step 4 (staticlib)**: Cargo.toml に `crate-type = ["lib", "staticlib"]`
- **Step 5 (リンカ)**: `cc` でフルリンク (runtime + system libs)

Step 1 (codegen リファクタ), Step 2 (リテラル), Step 3 (ObjectModule) はキャッシュで完成済みなので、AOT は差分だけ。
