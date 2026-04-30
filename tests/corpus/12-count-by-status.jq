.entries
| group_by(.status)
| map({status: .[0].status, count: length})
| sort_by(-.count)
