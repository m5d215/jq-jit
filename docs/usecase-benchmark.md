# 実用ユースケースベンチマーク

> 2026-03-01 測定。macOS (Darwin 25.3.0, Apple Silicon)

## TL;DR

- **ログ分析 (100K NDJSON)**: jq 比 **1.16x〜1.21x** 高速
- **API レスポンス加工 (50K objects)**: jq 比 **1.27x〜1.38x** 高速
- **バッチ ETL (100 ファイル)**: jq 比 **1.15x** 高速（フィルタキャッシュ効果 1.17x）

jq-jit は計算集約的なフィルタ（select + 変換 + 集計）で安定して jq を上回る。データが大きくフィルタが複雑なほど差が開く。

## 測定環境

| 項目 | 値 |
|---|---|
| OS | macOS (Darwin 25.3.0, Apple Silicon) |
| jq | 1.7.1-apple (system) |
| jq-jit | v0.1.0 (Cranelift v0.129) |
| ツール | hyperfine 1.20.0 |

## UC1: ログ分析 — 構造化ログから異常検出

**シナリオ**: アプリケーションの構造化 JSON ログ（CloudWatch Logs / Datadog 等）から、特定条件に合致するエントリを抽出。

**データ**: 100,000 件の NDJSON ログエントリ（24MB）。各エントリは timestamp, level, service, request_id, duration_ms, status, message, metadata（endpoint, method, user_id）を含む。

| ベンチマーク | フィルタ | jq (ms) | jq-jit (ms) | Speedup |
|---|---|---|---|---|
| select_error | `select(.level == "error")` | 286 | 237 | **1.21x** |
| select_slow_error | `select(.level == "error" and .duration_ms > 1000)` | 283 | 236 | **1.21x** |
| extract_errors | `select(.level == "error") \| {timestamp, service, message, endpoint: .metadata.endpoint}` | 270 | 234 | **1.16x** |

**分析**: NDJSON 100K 行の処理で安定して 1.2x 前後の高速化。条件が単純な select でも複合条件でもほぼ同じ速度。フィールド抽出 + オブジェクト構築を追加しても jq-jit 側は大きく変わらず、jq 側の実行時間が減る（フィルタで除外された行の出力が減るため）。

## UC2: API レスポンス加工 — 大きな JSON から必要データを変換

**シナリオ**: REST API の巨大レスポンス（ユーザー一覧 + ネストされたプロフィール）を加工してダッシュボード用 JSON を生成。

**データ**: 50,000 件のユーザーオブジェクト配列（15MB）。各ユーザーは id, name, email, profile（department, role, joined, skills[]）, activity（last_login, login_count, projects[]）を含む。

| ベンチマーク | フィルタ | jq (ms) | jq-jit (ms) | Speedup |
|---|---|---|---|---|
| flatten | `[.[] \| {name, department: .profile.department, skill_count: (.profile.skills \| length)}]` | 260 | 204 | **1.27x** |
| select_transform | `[.[] \| select(.activity.login_count > 100) \| {name, logins: .activity.login_count, projects: (.activity.projects \| map(.name))}]` | 270 | 201 | **1.34x** |
| group_aggregate | `group_by(.profile.department) \| map({dept: .[0].profile.department, count: length, avg_logins: (map(.activity.login_count) \| add / length)})` | 252 | 183 | **1.38x** |

**分析**: 最も差が出たユースケース。単一の大きな JSON 配列に対するフィルタ実行で、jq-jit の JIT ネイティブコードが jq のインタプリタを大幅に上回る。特に group_by + map + 集計のような多段パイプラインで **1.38x** を達成。フィルタが複雑なほど jq-jit の優位性が増す傾向。

## UC3: バッチ ETL — 同一フィルタの繰り返し適用

**シナリオ**: 日次エクスポートされた JSON ファイル群に同じフィルタを適用（ログローテーション、パーティション分割データ）。

**データ**: 100 ファイル × 1,000 件の購買レコード（各ファイル約 200KB）。各レコードは id, type, amount, currency, customer_id, items[] を含む。

| モード | 合計時間 (ms) | vs jq |
|---|---|---|
| jq | 624 | — |
| jq-jit (キャッシュ有効) | 540 | **1.15x** |
| jq-jit (キャッシュ無効) | 633 | 0.99x |

**フィルタ**: `[.[] | select(.type == "purchase" and .amount > 1000) | {id, amount, items: (.items | length)}]`

**分析**: フィルタキャッシュの効果が明確に出ている。キャッシュなしでは jq とほぼ同等だが、キャッシュが効くと 100 回のうち 99 回は JIT コンパイルをスキップし dlopen で即ロードするため、合計で **1.15x** の高速化。1 ファイルあたり約 0.9ms の節約（JIT コンパイル約 450µs + α）。

## jq-jit が有効なユースケースまとめ

| ユースケース | 条件 | 期待される高速化 |
|---|---|---|
| ログ分析 | 大量 NDJSON + select フィルタ | 1.1x〜1.2x |
| API レスポンス加工 | 大きな JSON 配列 + 変換・集計 | 1.3x〜1.4x |
| バッチ処理 | 同一フィルタを多数ファイルに適用 | 1.1x〜1.2x（キャッシュ効果） |
| 組み込み利用 | Rust アプリで Value を直接渡す | JSON パース不要で大幅高速化 |

**jq-jit が不向きなケース**:
- parse-only ワークロード（`length`, `.[0]` など）— jq の方が若干高速
- generator-in-scalar を使うフィルタ（`[.items[].name]` 形式）— コンパイルエラー。`map(.name)` に書き換えが必要
- NDJSON の単純フィールド抽出 — 出力 I/O がボトルネックで jq が優位

## 測定コマンド

```bash
cargo build --release
bash benches/usecase.sh
```
