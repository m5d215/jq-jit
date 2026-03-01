# jq JIT コンパイラ: バックエンド比較評価

> 調査日: 2026-02-27
> 目的: jq バイトコードのネイティブ JIT コンパイルに最適なバックエンドの選定

## 1. バックエンド個別評価

### 1.1 Cranelift

- リポジトリ: [bytecodealliance/wasmtime](https://github.com/bytecodealliance/wasmtime) (cranelift/)
- crate: [cranelift-jit](https://crates.io/crates/cranelift-jit)

Wasmtime プロジェクト由来。約 20 万行の Rust コードベース（LLVM の 1/100）。最新バージョンは **0.129.1**（2026-02-25 リリース）。

**メンテナンス体制**: Bytecode Alliance（Mozilla, Fastly, Intel, Red Hat 等が支援する非営利団体）が主導。Wasmtime の中核コンポーネントとして積極開発が継続しており、Rust コンパイラの公式バックエンド（`rustc_codegen_cranelift`）としても採用が進行中。

| 項目 | 評価 |
|---|---|
| コンパイル速度 | LLVM 比 **約 10 倍高速** |
| 生成コード品質 | V8 TurboFan 比 **約 2% 遅い**、LLVM 比 **約 14% 遅い** |
| API | Rust ネイティブ、型安全。`FunctionBuilder` で段階的に IR 構築 |
| AArch64 | Tier 1 サポート。Apple Silicon で Wasmtime が本番運用済み |
| 非局所制御フロー | `try_call` 命令（2025 年追加）で例外ハンドリング対応 |
| バイナリサイズ | 数 MB のオーバーヘッド |

### 1.2 LLVM ORC JIT

- Rust バインディング: [inkwell](https://github.com/TheDan64/inkwell)

**メンテナンス体制**: LLVM プロジェクト（Apache Foundation 下、Google, Apple, ARM, Intel 等が主要コントリビュータ）。世界最大のコンパイラ基盤で、長期的な存続に懸念なし。

| 項目 | 評価 |
|---|---|
| コンパイル速度 | Cranelift の **約 1/10** |
| 生成コード品質 | **最高水準** |
| API | inkwell で改善するが本質的に複雑。LLVM バージョン管理が煩雑 |
| AArch64 | フル対応 |
| 非局所制御フロー | `invoke`/`landingpad`、setjmp/longjmp 完全対応 |
| バイナリサイズ | 最小でも数 MB、最適化パス込みで **10MB+** |

### 1.3 dynasm-rs

- リポジトリ: [CensoredUsername/dynasm-rs](https://github.com/CensoredUsername/dynasm-rs)

Rust proc-macro ベースの動的アセンブラ。レジスタアロケータ・最適化パスなし。

**メンテナンス体制**: 個人プロジェクト（CensoredUsername）。メンテナンスは継続しているが、コミュニティは小規模。

| 項目 | 評価 |
|---|---|
| コンパイル速度 | **最速クラス**。memcpy + パッチに近い速度 |
| 生成コード品質 | **開発者の腕次第**。スケールしない |
| API | マクロは直感的だがアセンブリ知識必須 |
| AArch64 | ARMv8.4 対応（SVE なし） |
| 非局所制御フロー | アセンブリレベルで完全自由 |
| バイナリサイズ | **極小**（~100 KB） |

### 1.4 Copy-and-Patch

- 論文: [Xu & Kjolstad, OOPSLA 2021](https://arxiv.org/abs/2011.13127)
- CPython 実装: [PEP 744](https://peps.python.org/pep-0744/)

事前コンパイルしたバイナリテンプレート（ステンシル）をコピー + パッチして連結。

**メンテナンス体制**: 特定のライブラリではなく手法・論文。CPython での実装は Brandt Bucher（元 Microsoft Faster CPython チーム、現在も個人で継続）が主導。

| 項目 | 評価 |
|---|---|
| コンパイル速度 | **理論上最速**（memcpy + 数回のポインタ書き込み） |
| 生成コード品質 | Clang `-O3` の恩恵を継承。ステンシル間最適化は不可 |
| Rust での実装 | **非常に困難**。guaranteed tail call / `preserve_none` ABI が Rust にない |
| AArch64 | アーキテクチャごとにステンシル生成が必要 |
| エコシステム | CPython 以外の実績が少ない |

**CPython JIT の進捗（PEP 744）:**
- **Python 3.13**: 実験的に導入（`--enable-experimental-jit` ビルドオプション）。Copy-and-Patch 方式でバイトコードをネイティブ化
- **Python 3.14**: JIT ウォームアップ閾値を 16 → 4096 に引き上げ。より保守的なコンパイル戦略に変更
- **Python 3.15（開発中）**: スタックアンワインディングの JIT 対応、スレッドセーフ化が主要課題。free-threaded CPython がデフォルトに向かう中で、JIT のスレッド安全性が重要に
- **注意**: 2025 年に Microsoft が Faster CPython チームの大部分をレイオフ。主開発者の Brandt Bucher は個人として開発継続を表明しているが、組織的支援の後退はリスク要因

### 1.5 GCC libgccjit

- Rust バインディング: [gccjit](https://crates.io/crates/gccjit)

**メンテナンス体制**: GCC プロジェクト（FSF / GNU）。GCC 本体の一部として長期メンテナンスされるが、libgccjit 自体のユーザーコミュニティは LLVM に比べて小さい。

| 項目 | 評価 |
|---|---|
| コンパイル速度 | LLVM と同等かやや遅い |
| 生成コード品質 | GCC の全最適化パスの恩恵。LLVM と同等クラス |
| 配布 | `libgccjit.so` が数十 MB。CLI ツールに同梱は非現実的 |
| Apple Silicon | サポート不安定 |

### 1.6 MIR (Medium Internal Representation)

- リポジトリ: [vnmakarov/mir](https://github.com/vnmakarov/mir)

約 55K 行の C コード。CRuby の JIT バックエンド候補として議論されたが、最終的に CRuby は YJIT（Shopify 主導、Rust ベース）を採用。MIR の作者 Vladimir Makarov 自身も YJIT の方向性を評価し、MIR JIT の Ruby 向け開発は事実上停止している。

**メンテナンス体制**: 個人プロジェクト（Vladimir Makarov、元 Red Hat）。コミュニティは小規模で、長期的なメンテナンスの継続性にリスクがある。

| 項目 | 評価 |
|---|---|
| コンパイル速度 | GCC `-O2` 比 **約 80 倍高速** |
| 生成コード品質 | GCC `-O2` の **91%** |
| Rust バインディング | **なし**。FFI 自作が必要 |
| AArch64 | Apple M1 対応済み |
| バイナリサイズ | **961 KB** |
| エコシステム | 個人プロジェクト、コミュニティ小 |


## 2. 総合比較

| 評価軸 | Cranelift | LLVM ORC | dynasm-rs | Copy-and-Patch | libgccjit | MIR |
|---|---|---|---|---|---|---|
| コンパイル速度 | ◎ | △ | ◎+ | ◎+ | △ | ◎ |
| 生成コード品質 | ○ | ◎+ | △ | ○ | ◎+ | ○ |
| API の使いやすさ | ◎ | △ | ○ | × | △ | ○ |
| Rust 親和性 | ◎+ | △ | ◎+ | × | △ | △ |
| AArch64 対応 | ◎ | ◎ | ○ | ○ | △ | ○ |
| エコシステム成熟度 | ○ | ◎ | ○ | △ | ○ | △ |
| バイナリサイズ | ○ | × | ◎+ | ◎ | × | ◎ |
| 非局所制御フロー | ○ | ◎ | ◎+ | ○ | ○ | △ |
| ライセンス | Apache-2.0 WITH LLVM-exception | Apache-2.0 WITH LLVM-exception | MPL-2.0 | N/A（手法） | GPL-3.0（リンク形態で伝播注意） | MIT |
| デバッグ支援 | ◎（CLIF IR ダンプ、disassembly、DWARF 生成対応） | ◎+（IR ダンプ、全レベルの disassembly、完全な DWARF/CodeView） | △（手動 disassembly のみ、デバッグ情報生成なし） | △（ステンシル単位の disassembly、デバッグ情報は手動） | ◎（GCC デバッグ情報フル対応、DWARF 生成） | ○（MIR IR ダンプ、disassembly 対応、DWARF は限定的） |

凡例: ◎+ = 最優秀, ◎ = 優秀, ○ = 良好, △ = 課題あり, × = 不適


## 3. jq ユースケース別の最適解

### パターン 1: 短命フィルタ (`jq '.foo'`)

- ボトルネック: **コンパイル時間**
- 最適解: **コンパイルしない（インタプリタのまま）** か **dynasm-rs / MIR レベルの超高速コンパイル**

### パターン 2: ストリーム処理 (`jq -c 'select(.status == "active")' huge.jsonl`)

- ボトルネック: **実行速度**
- 最適解: **Cranelift**（コンパイル速度と生成コード品質のバランスが最適）


## 4. Tiered Compilation 戦略

```
Tier 0: バイトコードインタプリタ（即座に実行開始）
  ↓ 実行回数カウント or 入力サイズ判定
Tier 1: 高速 JIT（Cranelift single_pass）
  ↓ ホットパス検出
Tier 2: 最適化 JIT（Cranelift full optimization）
```

| 条件 | 動作 |
|---|---|
| 入力が小さい（< 1KB）or フィルタが単純 | Tier 0 のまま |
| 入力がパイプ or ファイルサイズ > 1MB | Tier 1 を即座にトリガー |
| 同一フィルタの実行回数 > N | Tier 2 に昇格 |

Cranelift は single_pass / backtracking の 2 つのレジスタアロケータモードを持ち、同じ IR で Tier 1 / Tier 2 をカバーできる。


## 5. 推奨と結論

### 最終推奨: Cranelift

| 推奨理由 |
|---|
| コンパイル速度（LLVM の 10 倍）と生成コード品質（LLVM の 86%）の両立 |
| Pure Rust。cargo で依存追加するだけ |
| Apple Silicon で Wasmtime として本番運用されている信頼性 |
| `try_call` によるバックトラッキングの例外ハンドリングモデリングの可能性 |
| single_pass / backtracking の 2 モードで Tiered compilation に対応 |
| Bytecode Alliance が積極開発（最新 0.129.1、2026-02-25）。Rust コンパイラの公式バックエンドとしても採用進行中 |

### リスクと緩和策

| リスク | 緩和策 |
|---|---|
| バイナリサイズ（数 MB 増） | strip + LTO で圧縮。CLI ツールとしては許容範囲 |
| 短命フィルタのコンパイルオーバーヘッド | Tier 0（インタプリタ）を維持し JIT 対象を選択的に |
| バックトラッキングの `try_call` マッピングの複雑さ | プロトタイプで検証。代替として CPS も検討 |

### 次善の選択肢

- **MIR**: Cranelift に匹敵する速度と品質。Rust バインディングの自作が必要
- **dynasm-rs**: Tier 1（超高速ベースライン）として部分的に採用する価値あり
- **LLVM ORC**: コード品質最優先の場合の最終手段


## 参考リンク

- [Cranelift 公式](https://cranelift.dev/) / [cranelift-jit-demo](https://github.com/bytecodealliance/cranelift-jit-demo)
- [Cranelift 例外ハンドリング設計](https://cfallin.org/blog/2025/11/06/exceptions/)
- [LLVM ORC v2 Design](https://llvm.org/docs/ORCv2.html) / [inkwell](https://github.com/TheDan64/inkwell)
- [dynasm-rs](https://github.com/CensoredUsername/dynasm-rs)
- [Copy-and-Patch 論文](https://arxiv.org/abs/2011.13127) / [チュートリアル](https://transactional.blog/copy-and-patch/tutorial)
- [MIR](https://github.com/vnmakarov/mir) / [解説](https://developers.redhat.com/blog/2020/01/20/mir-a-lightweight-jit-compiler-project)
