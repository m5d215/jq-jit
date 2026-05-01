# Benchmark History

Recent slice (last 5 columns). Full history lives in
[`benchmark-history.tsv`](benchmark-history.tsv) (long format,
`section / benchmark / version / time_seconds`).

```text
--- NDJSON workloads (2M objects) ---
  Benchmark                v0.10.0  v1.0.0  v1.1.0  v1.4.3  v1.4.4
  ---                      -------  ------  ------  ------  ------
  empty                    0.016s   0.017s  0.018s  0.017s  0.017s
  identity -c              0.083s   0.081s  0.077s  0.082s  0.082s
  identity (pretty)        0.110s   0.103s  0.099s  0.108s  0.099s
  field access .name       0.068s   0.067s  0.066s  0.080s  0.080s
  nested .x,.y,.name       0.101s   0.096s  0.096s  0.111s  0.123s
  arithmetic .x + .y       0.065s   0.064s  0.064s  0.064s  0.094s
  select .x > 1500000      0.048s   0.046s  0.047s  0.066s  0.115s
  string concat            0.079s   0.075s  0.074s  0.090s  0.089s
  object construct         0.087s   0.084s  0.083s  0.087s  0.127s
  array construct          0.090s   0.086s  0.081s  0.097s  0.102s
  .[]                      0.081s   0.080s  0.081s  0.085s  0.098s
  to_entries               0.149s   0.143s  0.142s  0.104s  0.112s
  keys                     0.086s   0.085s  0.085s  0.088s  0.102s
  keys_unsorted            0.077s   0.076s  0.075s  0.078s  0.091s
  length                   0.066s   0.063s  0.065s  0.073s  0.085s
  has("x")                 0.028s   0.025s  0.025s  0.030s  0.030s
  type                     0.019s   0.018s  0.019s  0.019s  0.019s
  del(.name)               0.091s   0.089s  0.088s  0.098s  0.095s

--- String operations (2M objects) ---
  Benchmark                v0.10.0  v1.0.0  v1.1.0  v1.4.3  v1.4.4
  ---                      -------  ------  ------  ------  ------
  ascii_downcase           0.088s   0.082s  0.084s  0.090s  0.088s
  ascii_upcase             0.088s   0.085s  0.083s  0.090s  0.088s
  ltrimstr                 0.085s   0.082s  0.082s  0.090s  0.090s
  rtrimstr                 0.087s   0.082s  0.083s  0.089s  0.090s
  split                    0.160s   0.153s  0.149s  0.150s  0.151s
  case+split               0.111s   0.106s  0.109s  0.117s  0.116s
  join                     0.083s   0.081s  0.078s  0.087s  0.092s
  startswith               0.084s   0.082s  0.080s  0.084s  0.086s
  endswith                 0.083s   0.080s  0.079s  0.085s  0.088s
  tostring                 0.043s   0.043s  0.043s  0.043s  0.095s
  tonumber                 0.096s   0.095s  0.092s  0.108s  0.104s
  string interpolation     0.100s   0.096s  0.097s  0.104s  0.106s

--- String ops (200K objects) ---
  Benchmark                v0.10.0  v1.0.0  v1.1.0  v1.4.3  v1.4.4
  ---                      -------  ------  ------  ------  ------
  test (regex)             0.011s   0.011s  0.011s  0.013s  0.013s
  match (regex)            0.037s   0.036s  0.037s  0.031s  0.032s
  @base64                  0.010s   0.010s  0.010s  0.011s  0.011s
  @uri                     0.010s   0.010s  0.011s  0.011s  0.011s
  @html                    0.010s   0.010s  0.010s  0.012s  0.011s
  @csv (array)             0.013s   0.012s  0.013s  0.013s  0.014s
  @tsv (array)             0.013s   0.012s  0.013s  0.013s  0.014s
  gsub                     0.018s   0.017s  0.018s  0.018s  0.018s
  case+gsub                0.176s   0.171s  0.176s  0.181s  0.180s
  case+test                0.109s   0.108s  0.110s  0.116s  0.115s
  ltrim+tonum+arith        0.086s   0.084s  0.083s  0.107s  0.107s

--- Numeric & math (2M objects) ---
  Benchmark                v0.10.0  v1.0.0  v1.1.0  v1.4.3  v1.4.4
  ---                      -------  ------  ------  ------  ------
  floor                    0.042s   0.041s  0.041s  0.042s  0.091s
  sqrt                     0.072s   0.069s  0.071s  0.072s  0.115s
  modulo                   0.073s   0.070s  0.070s  0.045s  0.095s
  if-elif-else             0.106s   0.103s  0.107s  0.102s  0.134s
  select|del               0.071s   0.069s  0.069s  0.073s  0.126s
  select|merge             0.105s   0.101s  0.101s  0.105s  0.157s
  select(test)|merge       0.020s   0.019s  0.020s  0.022s  0.021s

--- Array generators ---
  Benchmark                v0.10.0  v1.0.0  v1.1.0  v1.4.3  v1.4.4
  ---                      -------  ------  ------  ------  ------
  range(2M) | length       0.010s   0.010s  0.011s  0.011s  0.011s
  reverse(2M)              0.017s   0.016s  0.017s  0.017s  0.017s
  sort(2M)                 0.022s   0.021s  0.022s  0.022s  0.022s
  unique(1M)               0.029s   0.028s  0.028s  0.028s  0.028s
  flatten(500K)            0.010s   0.009s  0.010s  0.010s  0.010s
  min, max(2M)             0.021s   0.018s  0.017s  0.017s  0.018s
  add numbers(2M)          0.012s   0.011s  0.012s  0.012s  0.012s
  any/all(2M)              0.027s   0.027s  0.027s  0.028s  0.027s
  limit(10; range(10M))    0.002s   0.002s  0.003s  0.002s  0.002s
  first(range(10M))        0.002s   0.002s  0.002s  0.002s  0.002s
  last(range(2M))          0.002s   0.002s  0.003s  0.002s  0.002s
  indices(1M)              0.015s   0.015s  0.015s  0.015s  0.015s

--- Reduce & foreach ---
  Benchmark                v0.10.0  v1.0.0  v1.1.0  v1.4.3  v1.4.4
  ---                      -------  ------  ------  ------  ------
  reduce (sum)             0.008s   0.008s  0.009s  0.009s  0.009s
  reduce (array build)     0.004s   0.004s  0.004s  0.004s  0.004s
  reduce (obj build)       0.009s   0.009s  0.010s  0.010s  0.010s
  reduce (setpath)         0.017s   0.015s  0.017s  0.016s  0.016s
  foreach (running sum)    0.010s   0.009s  0.010s  0.010s  0.010s
  foreach + emit           0.009s   0.009s  0.009s  0.010s  0.010s
  reduce (sum-of-squares)  0.035s   0.032s  0.032s  0.032s  0.032s
  reduce (conditional)     0.035s   0.034s  0.034s  0.034s  0.034s
  reduce (product)         0.035s   0.036s  0.034s  0.034s  0.034s
  foreach (conditional)    0.010s   0.010s  0.010s  0.010s  0.010s
  until (100M)             0.311s   0.294s  0.293s  0.295s  0.294s
  reduce (harmonic)        0.034s   0.031s  0.032s  0.032s  0.032s
  reduce (floor pipe)      0.034s   0.031s  0.032s  0.032s  0.032s
  reduce (sqrt pipe)       0.034s   0.032s  0.033s  0.032s  0.032s
  reduce (sin+cos)         0.052s   0.050s  0.051s  0.052s  0.052s

--- Object operations ---
  Benchmark                v0.10.0  v1.0.0  v1.1.0  v1.4.3  v1.4.4
  ---                      -------  ------  ------  ------  ------
  large obj construct      0.004s   0.003s  0.004s  0.004s  0.004s
  large obj keys           0.010s   0.009s  0.010s  0.011s  0.010s
  large obj to_entries     0.012s   0.010s  0.011s  0.011s  0.012s
  with_entries             0.009s   0.008s  0.009s  0.009s  0.009s

--- Assignment operators ---
  Benchmark                v0.10.0  v1.0.0  v1.1.0  v1.4.3  v1.4.4
  ---                      -------  ------  ------  ------  ------
  .[] |= f (100K)          0.004s   0.004s  0.005s  0.005s  0.005s
  .[] += 1 (100K)          0.005s   0.004s  0.005s  0.005s  0.005s
  .[k] = v reduce(50K)     0.008s   0.008s  0.009s  0.008s  0.008s

--- String-heavy generators ---
  Benchmark                v0.10.0  v1.0.0  v1.1.0  v1.4.3  v1.4.4
  ---                      -------  ------  ------  ------  ------
  gsub(100K)               0.018s   0.018s  0.019s  0.025s  0.025s
  join large(100K)         0.005s   0.005s  0.005s  0.006s  0.005s
  explode/implode(100K)    0.029s   0.027s  0.029s  0.029s  0.028s
  reduce str concat(100K)  0.007s   0.007s  0.008s  0.008s  0.008s

--- Try-catch & alternative ---
  Benchmark                v0.10.0  v1.0.0  v1.1.0  v1.4.3  v1.4.4
  ---                      -------  ------  ------  ------  ------
  alternative //           0.032s   0.031s  0.032s  0.031s  0.031s
  try-catch                0.023s   0.022s  0.023s  0.023s  0.023s
  label-break              0.004s   0.003s  0.004s  0.004s  0.004s

--- Type conversion ---
  Benchmark                v0.10.0  v1.0.0  v1.1.0  v1.4.3  v1.4.4
  ---                      -------  ------  ------  ------  ------
  tojson/fromjson(100K)    0.023s   0.021s  0.022s  0.021s  0.021s
  null propagation(2M)     0.088s   0.101s  0.087s  0.089s  0.090s

--- jaq-derived ---
  Benchmark                v0.10.0  v1.0.0  v1.1.0  v1.4.3  v1.4.4
  ---                      -------  ------  ------  ------  ------
  jaq: reverse             0.010s   0.010s  0.010s  0.010s  0.011s
  jaq: sort                0.017s   0.016s  0.017s  0.018s  0.018s
  jaq: group-by            0.038s   0.036s  0.037s  0.034s  0.035s
  jaq: min-max             0.010s   0.010s  0.011s  0.010s  0.010s
  jaq: ex-implode          0.019s   0.018s  0.019s  0.019s  0.019s
  jaq: repeat              0.011s   0.011s  0.012s  0.011s  0.011s
  jaq: from                0.005s   0.005s  0.006s  0.006s  0.006s
  jaq: last                0.002s   0.002s  0.003s  0.002s  0.002s
  jaq: cumsum              0.010s   0.009s  0.010s  0.010s  0.012s
  jaq: cumsum-xy           0.017s   0.016s  0.017s  0.016s  0.018s
  jaq: try-catch           0.087s   0.085s  0.090s  0.084s  0.088s
  jaq: add                 0.041s   0.039s  0.042s  0.039s  0.042s
  jaq: reduce              0.083s   0.079s  0.092s  0.084s  0.094s
  jaq: reduce-update       0.004s   0.004s  0.005s  0.005s  0.005s
  jaq: kv                  0.014s   0.013s  0.015s  0.014s  0.014s
  jaq: kv-update           0.017s   0.017s  0.018s  0.019s  0.018s
  jaq: kv-entries          0.057s   0.056s  0.055s  0.056s  0.056s
  jaq: pyramid             0.016s   0.016s  0.016s  0.016s  0.015s
  jaq: upto                0.006s   0.007s  0.007s  0.007s  0.007s
  jaq: tree-flatten        0.003s   0.003s  0.003s  0.003s  0.003s
  jaq: tree-update         0.006s   0.006s  0.007s  0.007s  0.007s
  jaq: to-fromjson         0.005s   0.004s  0.005s  0.005s  0.005s
  jaq: str-slice           0.014s   0.013s  0.014s  0.014s  0.013s
```
