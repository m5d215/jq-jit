# 実用ユースケースベンチマーク

> **状態: 完了** — 全 4 Step 完了。結果レポート: [usecase-benchmark.md](../usecase-benchmark.md)

## Context

jq-jit は合成ベンチマーク（compare.sh）で性能特性を把握済みだが、「現実の何に使えるか」が見えない。ユースケースを明確化し、現実的なシナリオで jq との性能差を示すことで、jq-jit の実用的な価値を証明する。

併せて simd-json 検討を将来課題として記録する。

## 成果物

1. `benches/usecase.sh` — 3つの実用シナリオベンチマークスクリプト
2. `docs/usecase-benchmark.md` — 結果レポート + ユースケース説明
3. `STATUS.md` / `docs/plan.md` 更新 — simd-json を将来課題として記録

## ユースケース設計

### UC1: ログ分析 — 構造化ログから異常検出

**シナリオ**: アプリケーションの構造化 JSON ログ（CloudWatch Logs / Datadog 等）から、特定条件に合致するエントリを抽出。

**データ**: 10万件の JSON ログエントリ（NDJSON）
```json
{"timestamp":"2026-03-01T10:23:45Z","level":"error","service":"api-gateway","request_id":"abc-123","duration_ms":1523,"status":500,"message":"upstream timeout","metadata":{"endpoint":"/v1/users","method":"POST","user_id":"u-789"}}
```

**フィルタ**:
- `select(.level == "error")` — エラーだけ抽出
- `select(.level == "error" and .duration_ms > 1000)` — 遅いエラーだけ
- `select(.level == "error") | {timestamp, service, message, endpoint: .metadata.endpoint}` — エラーから必要フィールドを抽出・整形

**期待**: select + フィールド抽出が計算集約的 → jq-jit が有利

### UC2: API レスポンス加工 — 大きな JSON から必要データを変換

**シナリオ**: REST API の巨大レスポンス（ユーザー一覧 + ネストされたプロフィール）を加工してダッシュボード用 JSON を生成。

**データ**: 5万件のユーザーオブジェクト配列（1つの大きな JSON）
```json
[{"id":1,"name":"Alice","email":"alice@example.com","profile":{"department":"engineering","role":"senior","joined":"2023-01-15","skills":["rust","go","python"]},"activity":{"last_login":"2026-02-28","login_count":342,"projects":[{"name":"alpha","role":"lead"},{"name":"beta","role":"member"}]}}]
```

**フィルタ**:
- `[.[] | {name, department: .profile.department, skill_count: (.profile.skills | length)}]` — フラット化 + 集計
- `[.[] | select(.activity.login_count > 100) | {name, logins: .activity.login_count, projects: (.activity.projects | map(.name))}]` — 条件抽出 + ネスト展開（※ 元の `[.activity.projects[].name]` は generator-in-scalar 制約によりコンパイルエラー。`map(.name)` に書き換え）
- `group_by(.profile.department) | map({dept: .[0].profile.department, count: length, avg_logins: (map(.activity.login_count) | add / length)})` — 部門別集計

**期待**: 深いネストアクセス + 変換 + 集計で jq-jit が有利

### UC3: バッチ ETL — 同一フィルタの繰り返し適用

**シナリオ**: 日次エクスポートされた JSON ファイル群に同じフィルタを適用（ログローテーション、パーティション分割データ）。

**データ**: 1000件のオブジェクトを含む JSON ファイル × 100個
```json
[{"id":1,"type":"purchase","amount":1250,"currency":"JPY","customer_id":"c-001","items":[{"sku":"A-100","qty":2,"price":500},{"sku":"B-200","qty":1,"price":250}]}]
```

**フィルタ**:
- `[.[] | select(.type == "purchase" and .amount > 1000) | {id, amount, items: (.items | length)}]`

**計測方法**: 100 ファイルに対して同一フィルタを逐次適用するシェルループ
```bash
# jq: 毎回パース + インタプリタ実行
for f in data/*.json; do jq -c '<filter>' "$f"; done

# jq-jit: 2回目以降はキャッシュヒット
for f in data/*.json; do jq-jit -c '<filter>' "$f"; done

# jq-jit (キャッシュなし): 比較用
for f in data/*.json; do jq-jit -c --no-cache '<filter>' "$f"; done
```

**期待**: jq-jit のフィルタキャッシュにより、繰り返し実行で差が拡大

## 実装ステップ

### Step 1: benches/usecase.sh 作成

既存の `compare.sh` を参考に、3 ユースケース用のベンチマークスクリプトを作成。

- Python3 でリアルなテストデータを生成
- hyperfine で各フィルタを計測
- UC3 はシェルループ全体を hyperfine で計測（キャッシュあり/なし比較）
- 結果を markdown テーブルで集約出力

### Step 2: ベンチマーク実行 + 結果記録

```bash
cargo build --release
bash benches/usecase.sh
```

### Step 3: docs/usecase-benchmark.md 作成

各ユースケースの説明 + 結果 + 分析を記載。「こういう場面で使うと速い」が一目でわかるドキュメント。

### Step 4: ドキュメント更新

- `STATUS.md`: ユースケースベンチマーク完了を記録 + simd-json を将来課題に追加
- `docs/plan.md`: 本プランを登録

## 影響を受けるファイル

| ファイル | 変更内容 |
|---|---|
| `benches/usecase.sh` | 新規: 実用ユースケースベンチマーク |
| `docs/usecase-benchmark.md` | 新規: ユースケース別性能レポート |
| `docs/plan.md` | インデックスに追加 |
| `STATUS.md` | ユースケースベンチマーク + simd-json 将来課題 |

## 検証方法

```bash
cargo build --release
bash benches/usecase.sh          # 全3ユースケース実行
# 結果の markdown テーブルが stdout に出力される
# それを docs/usecase-benchmark.md に転記
```
