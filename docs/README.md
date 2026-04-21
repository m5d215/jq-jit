# docs/

jq-jit v1.0.0 リリース後に、開発中の Claude Code セッションログ（1.5GB超）を分析してまとめたドキュメント群。

## ファイル一覧

| ファイル | 内容 |
|---|---|
| [development-story.md](development-story.md) | 開発の全体像。タイムライン、アーキテクチャ、性能、最適化カタログ |
| [human-instructions.md](human-instructions.md) | 人間が AI に送った全指示（タイムスタンプ付き・全量） |
| [insights.md](insights.md) | ログから掘り出したインサイト。コミット時間分布、失敗の記録、劇的な改善など |
| [team-playbook.md](team-playbook.md) | Claude Code を長時間自律駆動させるためのプレイブック。チーム展開向け |
| [benchmark-history.md](benchmark-history.md) | 15バージョンにわたるベンチマーク推移（NDJSON、文字列、数値、配列、reduce 等） |
| [maintenance.md](maintenance.md) | 互換性バグの探し方、fast path の地図、守るべき invariant、デバッグの流れ |
