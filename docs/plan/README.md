# jq-jit: 開発プラン全体像

## 開発の流れ

```
Phase 0-10 (完了)          次のステップ (未着手)
━━━━━━━━━━━━━━━━         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
技術調査                    codegen リファクタ (共通基盤)
  ↓                           ↓
Minimal JIT                 リテラル間接参照 (共通基盤)
  ↓                           ↓
段階的に機能追加            ┌─────────┴──────────┐
(条件分岐→文字列→          ↓                    ↓
 配列→オブジェクト→     フィルタキャッシュ    AOT コンパイル
 正規表現→ユーザー定義   (dlopen で即ロード)  (スタンドアロンバイナリ)
 関数→完全互換)
  ↓
性能測定
```

## プラン一覧

| プラン | ファイル | 状態 | 概要 |
|---|---|---|---|
| **Phase 0-10 ロードマップ** | [roadmap.md](roadmap.md) | 完了 | 技術調査から完全互換までの段階的開発 |
| **フィルタキャッシュ** | [cache.md](cache.md) | 完了 | コンパイル済みフィルタを .dylib としてキャッシュし、2 回目以降は dlopen で即ロード |
| **AOT コンパイル** | [aot.md](aot.md) | 完了 | jq フィルタをスタンドアロン実行バイナリに変換。JIT オーバーヘッド = 0 |
| **ストリーミング JSON パーサ** | [streaming.md](streaming.md) | 完了 | serde_json ストリーミングデシリアライザ + 入出力共有化で CLI 性能を大幅改善 |
| **ユースケースベンチマーク** | [usecase.md](usecase.md) | 完了 | 実用シナリオ（ログ分析・API 加工・バッチ ETL）での性能測定 |
| **simd-json 導入** | — | 検討中 | JSON パーサを simd-json に置き換え、SIMD 加速でパース性能を改善 |

## キャッシュと AOT の関係

両プランは Step 1-2 を共有している。キャッシュを先にやれば AOT は差分で済む。

```
          共通基盤
        ┌──────────────────────────────────┐
        │ Step 1: codegen リファクタ       │
        │   compile_expr を build_filter   │
        │   <M: Module> にジェネリック化   │
        │                                  │
        │ Step 2: リテラル間接参照         │
        │   iconst(絶対アドレス) →          │
        │   load(literals_base + idx)      │
        └────────┬─────────────┬───────────┘
                 │             │
    ┌────────────▼──┐  ┌──────▼────────────┐
    │ キャッシュ     │  │ AOT               │
    │                │  │                   │
    │ ObjectModule   │  │ ObjectModule      │
    │   → .o         │  │   → .o            │
    │   → cc -shared │  │   → cc (フルリンク) │
    │   → .dylib     │  │   → 実行バイナリ    │
    │                │  │                   │
    │ dlopen で      │  │ main() 生成       │
    │ ロード         │  │ aot_run() 呼び出し │
    │ (rt_* は       │  │ staticlib に      │
    │  プロセス内で  │  │ rt_* を含む       │
    │  解決)         │  │                   │
    └────────────────┘  └───────────────────┘
```

### 推奨実装順序

1. **キャッシュ** — 共通基盤 (Step 1-2) + 共有ライブラリ生成。AOT より単純で、JIT フォールバックも容易
2. **AOT** — キャッシュで完成した共通基盤の上に main 生成・staticlib・リンカ呼び出しを追加

## 現在の性能

| 測定 | 結果 | 詳細 |
|---|---|---|
| JIT 実行 (Criterion) | libjq 比 55x〜18,000x 高速 | [benchmark.md](../benchmark.md) |
| CLI wall-clock (hyperfine) | jq 比 0.9x〜1.68x | ストリーミングパーサ最適化で大幅改善 |

キャッシュで JIT コンパイルを省略、AOT で完全に除去、ストリーミングパーサで JSON パース性能を改善。計算集約的フィルタでは jq を 1.3x〜1.8x 上回る。

## 関連ドキュメント

| ファイル | 内容 |
|---|---|
| [benchmark.md](../benchmark.md) | 性能測定レポート (Criterion + hyperfine) |
| [roadmap-record.md](roadmap-record.md) | Phase 10-7 の詳細な作業記録 |
| [cache-record.md](cache-record.md) | フィルタキャッシュ実装記録 |
| [aot-record.md](aot-record.md) | AOT コンパイル実装記録 |
| [streaming-record.md](streaming-record.md) | ストリーミング JSON パーサ実装記録 |
| [research/](../research/) | 技術調査 (6 documents) |
