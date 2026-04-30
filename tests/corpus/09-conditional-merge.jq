. + {
  status: (if .errors == 0 then "ok"
           elif .errors < 5 then "warn"
           else "fail" end),
  rate: (if .total > 0 then (.errors / .total) else 0 end)
}
