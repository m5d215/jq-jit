# フィルタキャッシュ実装の記録

> 2026-02-28 実施。コンパイル済みフィルタを `.dylib` としてキャッシュし、2 回目以降は dlopen で即ロードする機能を全 6 Step で実装した記録。

## プロジェクト概要

jq-jit は jq のバイトコード VM をネイティブコードに JIT コンパイルするプロジェクト。CLI 実行で毎回 JIT コンパイル (~450µs) を走らせるのは同じフィルタの繰り返し実行では無駄であるため、コンパイル結果を共有ライブラリとしてディスクにキャッシュし、2 回目以降は dlopen で即ロードする仕組みを導入した。

## 進捗推移

| Step | 内容 | 変更ファイル |
|---|---|---|
| **Step 1** | codegen リファクタ — `build_filter<M: Module>` 抽出、`register_jit_symbols` ヘルパー | `src/codegen.rs` |
| **Step 2** | リテラル間接参照化 — `LiteralPool` 構造体、`jit_filter` 4 引数化、`iconst` → `load` 変換 | `src/codegen.rs` |
| **Step 3** | `compile_to_shared_object` — `ObjectModule` + `cc -shared` → `.dylib` 生成 | `src/codegen.rs`, `Cargo.toml` |
| **Step 4** | `cache.rs` — `FilterCache`, `CachedFilter`, リテラルシリアライズ/デシリアライズ | `src/cache.rs` (新規), `src/lib.rs` |
| **Step 5** | CLI 統合 — `CompiledFilter` enum、`run()` にキャッシュを透過的に組み込み | `src/bin/jq-jit.rs` |
| **Step 6** | CLI フラグ — `--no-cache`, `--clear-cache`, `--cache-dir` | `src/bin/jq-jit.rs` |

## 技術的ハイライト

### Module トレイトのジェネリック化

`compile_expr` 内の CLIF IR 生成ロジックを `build_filter<M: Module>` としてジェネリック関数に抽出。JITModule / ObjectModule で同じコード生成パスを共有するようにした。JIT 固有の処理（JITBuilder シンボル登録、finalize + get_fn_ptr）と ObjectModule 固有の処理（ObjectBuilder 作成、finish → emit）だけが分岐する。この抽象化が後続の Cache / AOT 実装の基盤となった。

### リテラルの間接参照化

従来の JIT ではリテラル値を `Box<Value>` としてヒープに確保し、そのアドレスを `iconst(ptr)` で CLIF IR にハードコードしていた。共有ライブラリを dlopen した場合、このアドレスは無効になる。

解決策として `jit_filter` のシグネチャに 4 番目のパラメータ `literals: *const *const Value` を追加し、`iconst(ptr)` を `load(literals_param + idx * ptr_size)` に変換した。`LiteralPool` 構造体で CLIF パラメータと Value の Vec をバンドルし、JIT/Cache/AOT のいずれのパスでも同じ方式でリテラルを扱える。

### 共有ライブラリのリンク

フィルタの `.o` には `rt_*` への未解決参照がある。macOS では `-undefined dynamic_lookup` フラグで未解決シンボルを実行時解決に委ね、dlopen 時に jq-jit プロセスが持つ `rt_*` シンボルで解決される。ランタイムライブラリのリンクは不要。

### キャッシュキーとフォールバック

キャッシュキーは `filter_string + CARGO_PKG_VERSION` のハッシュで生成。バージョン変更時は自動的にキャッシュが無効化される。キャッシュの生成・ロードに失敗した場合は、従来の JIT パスにフォールバックする設計とし、キャッシュ機能の不具合が実行に影響しないようにした。

## テスト結果

```
全テスト回帰なし:
  Differential tests: 649 PASS, 1 FAIL (既知 libjq 差異)
  CLI compatibility: 113 PASS
キャッシュ動作:
  キャッシュ生成・ヒット・クリア動作確認済み
```

## 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `src/codegen.rs` | 大幅変更: `build_filter` 抽出、`LiteralPool`、`compile_to_shared_object` |
| `src/cache.rs` | 新規: `FilterCache`、`CachedFilter`、シリアライズ |
| `src/lib.rs` | `pub mod cache` 追加 |
| `src/bin/jq-jit.rs` | `CompiledFilter` enum、キャッシュ統合、フラグ追加 |
| `Cargo.toml` | `cranelift-object` 追加 |

## AOT への橋渡し

キャッシュ実装で完成した Step 1（Module 抽象化）と Step 2（リテラル間接参照化）は、AOT コンパイルの基盤としてそのまま活用された。AOT は差分実装（compile_to_object での main 関数生成、aot.rs エントリポイント、staticlib ビルド、リンカ呼び出し）のみで済んだ。
