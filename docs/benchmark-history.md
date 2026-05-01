# Benchmark History

Recent slice (last 5 columns). Full history lives in
[`benchmark-history.tsv`](benchmark-history.tsv) (long format,
`section / benchmark / version / time_seconds`).

```text
--- NDJSON workloads (2M objects) ---
  Benchmark                v1.0.0  v1.1.0  v1.4.3  v1.4.4  3d440ca
  ---                      ------  ------  ------  ------  -------
  empty                    0.017s  0.018s  0.017s  0.017s  0.017s
  identity -c              0.081s  0.077s  0.082s  0.082s  0.078s
  identity (pretty)        0.103s  0.099s  0.108s  0.099s  0.101s
  field access .name       0.067s  0.066s  0.080s  0.080s  0.087s
  nested .x,.y,.name       0.096s  0.096s  0.111s  0.123s  0.144s
  arithmetic .x + .y       0.064s  0.064s  0.064s  0.094s  0.082s
  select .x > 1500000      0.046s  0.047s  0.066s  0.115s  0.069s
  string concat            0.075s  0.074s  0.090s  0.089s  0.096s
  object construct         0.084s  0.083s  0.087s  0.127s  0.112s
  array construct          0.086s  0.081s  0.097s  0.102s  0.116s
  .[]                      0.080s  0.081s  0.085s  0.098s  0.100s
  to_entries               0.143s  0.142s  0.104s  0.112s  0.161s
  keys                     0.085s  0.085s  0.088s  0.102s  0.098s
  keys_unsorted            0.076s  0.075s  0.078s  0.091s  0.094s
  length                   0.063s  0.065s  0.073s  0.085s  0.080s
  has("x")                 0.025s  0.025s  0.030s  0.030s  0.029s
  type                     0.018s  0.019s  0.019s  0.019s  0.022s
  del(.name)               0.089s  0.088s  0.098s  0.095s  0.098s
  @csv                     -       -       -       -       0.140s
  split/join               -       -       -       -       0.093s
  select|field             -       -       -       -       0.112s
  select|remap             -       -       -       -       0.095s
  computed remap           -       -       -       -       0.211s
  [.x,.y]|add              -       -       -       -       0.082s
  [.x,.y]|avg              -       -       -       -       0.110s
  map(*2)|add              -       -       -       -       0.110s
  keys|length              -       -       -       -       0.254s
  .+{z=0}                  -       -       -       -       0.143s
  split|first              -       -       -       -       0.092s
  slice[0..5]              -       -       -       -       0.095s
  dynkey {(.name)}         -       -       -       -       0.118s
  .x += 1                  -       -       -       -       0.069s
  {a}+{b} merge            -       -       -       -       0.143s
  .x*2+1                   -       -       -       -       0.049s
  .x+.y*2                  -       -       -       -       0.102s
  .x > .y                  -       -       -       -       0.076s
  to_entries|len           -       -       -       -       0.397s
  .x|.+1 (pipe)            -       -       -       -       0.047s
  .x|.*2|.+1               -       -       -       -       0.049s
  .name|.+"_x"             -       -       -       -       0.098s
  .x>N | not               -       -       -       -       0.040s
  and (2 cmp)              -       -       -       -       0.080s
  if-then-else             -       -       -       -       0.043s
  sel(and)|field           -       -       -       -       0.076s
  sel(and)|remap           -       -       -       -       0.074s
  arith|cmp                -       -       -       -       0.044s
  if cmp .field            -       -       -       -       0.102s
  split|length             -       -       -       -       0.094s
  [x,y]|min                -       -       -       -       0.090s
  [x,y]|max                -       -       -       -       0.091s
  [x,y]|sort|.[0]          -       -       -       -       0.087s
  .name|len>5              -       -       -       -       0.096s
  sel(len>5)|.x            -       -       -       -       0.125s
  if .x>.y .name           -       -       -       -       0.087s
  sel(.x>.y)|.name         -       -       -       -       0.071s
  .x*2|tostring            -       -       -       -       0.047s
  .x*.x+1                  -       -       -       -       0.056s
  {k=.name,v=tostr}        -       -       -       -       0.160s
  str add chain            -       -       -       -       0.386s
  if>.y .name|empty        -       -       -       -       0.071s
  if .x%2==0               -       -       -       -       0.045s
  if .x*2+1>1M             -       -       -       -       0.045s
  sel(.x%2==0)|.name       -       -       -       -       0.074s
  sel(.x*2+1>1M)           -       -       -       -       0.142s
  .x|@json                 -       -       -       -       0.045s
  .x|@text                 -       -       -       -       0.045s
  .name|@json              -       -       -       -       0.104s
  sel|[arr]                -       -       -       -       0.148s
  sel(and)|[arr]           -       -       -       -       0.076s
  if>.y [arr]              -       -       -       -       0.199s
  if sw then .f            -       -       -       -       0.135s
  dynkey {(.n)=.x*2}       -       -       -       -       0.131s
  sel(and)|.x*.y           -       -       -       -       0.075s
  sel>N|str chain          -       -       -       -       0.158s
  .f+"_"+arith_ts          -       -       -       -       0.145s
  sel(sw)|str ch           -       -       -       -       0.317s
  split|rev|join           -       -       -       -       0.123s
  dynkey+static            -       -       -       -       0.324s
  if>.y str chain          -       -       -       -       0.186s
  remap+str chain          -       -       -       -       0.171s
  sel(len>8)               -       -       -       -       0.163s
  up|split|join            -       -       -       -       0.101s
  .name|index              -       -       -       -       0.126s
  .name|index+1            -       -       -       -       0.127s
  .name|rindex             -       -       -       -       0.131s
  .name|indices            -       -       -       -       0.153s
  [x,y]|sort               -       -       -       -       0.154s
  .name|scan               -       -       -       -       0.218s
  .name|gsub               -       -       -       -       0.175s
  walk(if num .+1)         -       -       -       -       0.140s
  tojson                   -       -       -       -       0.106s
  {name,x}                 -       -       -       -       0.144s
  .z//.name                -       -       -       -       0.165s
  .x|=test(re)             -       -       -       -       0.123s
  ./sep|first              -       -       -       -       0.131s
  .y=(.x*2)                -       -       -       -       0.112s
  .y=(.x+.y)               -       -       -       -       0.161s
  objects                  -       -       -       -       0.081s
  .tag|=if..then N         -       -       -       -       0.613s
  .x=(.x+1)                -       -       -       -       0.070s
  sel>N|.y+=1              -       -       -       -       0.085s
  sel(and)|.x+=1           -       -       -       -       0.100s
  sel(sw)|.x+=1            -       -       -       -       0.132s
  match(re)                -       -       -       -       0.369s
  capture(re)              -       -       -       -       0.306s
  first(.name,.x)          -       -       -       -       0.090s
  if .x==null              -       -       -       -       0.043s
  we(sw(.key))             -       -       -       -       0.111s
  sel(sw or ew)            -       -       -       -       0.211s
  path(.name,.x)           -       -       -       -       0.276s
  sel(str+num+num)         -       -       -       -       0.142s
  nested if|field          -       -       -       -       0.076s
  .f|floor|.*2             -       -       -       -       0.050s
  split|len>1              -       -       -       -       0.122s
  .name|len|.*2            -       -       -       -       0.106s
  if len>5 .x .y           -       -       -       -       0.137s
  sel(len>5)|remap         -       -       -       -       0.230s
  .x|tostr|len             -       -       -       -       0.056s
  if .x>.y .x .y           -       -       -       -       0.093s
  split|last|tonum         -       -       -       -       0.100s
  split|rev|.[0]           -       -       -       -       0.097s
  split|.[0]+.[1]          -       -       -       -       0.122s
  .[]|strings              -       -       -       -       0.096s
  .[]|numbers              -       -       -       -       0.107s
  [x,y]|any(>1M)           -       -       -       -       0.082s
  sel(dc|sw)               -       -       -       -       0.104s
  [[x,y],[n]]|flat         -       -       -       -       0.475s
  .x|floor|.*2             -       -       -       -       0.051s
  tojson|fromjson          -       -       -       -       0.083s
  [.x]|add                 -       -       -       -       0.048s
  if>N {o}+.               -       -       -       -       0.123s
  if>N .+{o}               -       -       -       -       0.124s
  if .n=="s" .+{o}         -       -       -       -       0.167s
  sel(.n>"s")              -       -       -       -       0.095s
  [x,y,z]|min              -       -       -       -       0.311s
  if .n|len>5 l s          -       -       -       -       0.107s
  if .x|flr>N b s          -       -       -       -       0.047s
  if .n|test l e           -       -       -       -       0.111s
  if .n|sw l e             -       -       -       -       0.090s
  if .n|ew l e             -       -       -       -       0.092s
  .n|len|tostr             -       -       -       -       0.097s

--- String operations (2M objects) ---
  Benchmark                v1.0.0  v1.1.0  v1.4.3  v1.4.4  3d440ca
  ---                      ------  ------  ------  ------  -------
  ascii_downcase           0.082s  0.084s  0.090s  0.088s  0.110s
  ascii_upcase             0.085s  0.083s  0.090s  0.088s  0.108s
  ltrimstr                 0.082s  0.082s  0.090s  0.090s  0.101s
  rtrimstr                 0.082s  0.083s  0.089s  0.090s  0.105s
  split                    0.153s  0.149s  0.150s  0.151s  0.169s
  case+split               0.106s  0.109s  0.117s  0.116s  0.126s
  join                     0.081s  0.078s  0.087s  0.092s  0.101s
  startswith               0.082s  0.080s  0.084s  0.086s  0.101s
  endswith                 0.080s  0.079s  0.085s  0.088s  0.103s
  tostring                 0.043s  0.043s  0.043s  0.095s  0.052s
  tonumber                 0.095s  0.092s  0.108s  0.104s  0.117s
  string interpolation     0.096s  0.097s  0.104s  0.106s  0.135s

--- String ops (200K objects) ---
  Benchmark                v1.0.0  v1.1.0  v1.4.3  v1.4.4  3d440ca
  ---                      ------  ------  ------  ------  -------
  test (regex)             0.011s  0.011s  0.013s  0.013s  0.014s
  match (regex)            0.036s  0.037s  0.031s  0.032s  0.033s
  @base64                  0.010s  0.010s  0.011s  0.011s  0.013s
  @uri                     0.010s  0.011s  0.011s  0.011s  0.013s
  @html                    0.010s  0.010s  0.012s  0.011s  0.013s
  @csv (array)             0.012s  0.013s  0.013s  0.014s  0.020s
  @tsv (array)             0.012s  0.013s  0.013s  0.014s  0.018s
  gsub                     0.017s  0.018s  0.018s  0.018s  0.020s
  case+gsub                0.171s  0.176s  0.181s  0.180s  0.193s
  case+test                0.108s  0.110s  0.116s  0.115s  0.130s
  ltrim+tonum+arith        0.084s  0.083s  0.107s  0.107s  0.118s

--- Numeric & math (2M objects) ---
  Benchmark                v1.0.0  v1.1.0  v1.4.3  v1.4.4  3d440ca
  ---                      ------  ------  ------  ------  -------
  floor                    0.041s  0.041s  0.042s  0.091s  0.048s
  sqrt                     0.069s  0.071s  0.072s  0.115s  0.078s
  modulo                   0.070s  0.070s  0.045s  0.095s  0.051s
  if-elif-else             0.103s  0.107s  0.102s  0.134s  0.125s
  select|del               0.069s  0.069s  0.073s  0.126s  0.079s
  select|merge             0.101s  0.101s  0.105s  0.157s  0.110s
  select(test)|merge       0.019s  0.020s  0.022s  0.021s  0.022s

--- Array generators ---
  Benchmark                v1.0.0  v1.1.0  v1.4.3  v1.4.4  3d440ca
  ---                      ------  ------  ------  ------  -------
  range(2M) | length       0.010s  0.011s  0.011s  0.011s  0.011s
  reverse(2M)              0.016s  0.017s  0.017s  0.017s  0.018s
  sort(2M)                 0.021s  0.022s  0.022s  0.022s  0.023s
  unique(1M)               0.028s  0.028s  0.028s  0.028s  0.029s
  flatten(500K)            0.009s  0.010s  0.010s  0.010s  0.010s
  min, max(2M)             0.018s  0.017s  0.017s  0.018s  0.017s
  add numbers(2M)          0.011s  0.012s  0.012s  0.012s  0.012s
  any/all(2M)              0.027s  0.027s  0.028s  0.027s  0.028s
  limit(10; range(10M))    0.002s  0.003s  0.002s  0.002s  0.002s
  first(range(10M))        0.002s  0.002s  0.002s  0.002s  0.002s
  last(range(2M))          0.002s  0.003s  0.002s  0.002s  0.002s
  indices(1M)              0.015s  0.015s  0.015s  0.015s  0.015s

--- Reduce & foreach ---
  Benchmark                v1.0.0  v1.1.0  v1.4.3  v1.4.4  3d440ca
  ---                      ------  ------  ------  ------  -------
  reduce (sum)             0.008s  0.009s  0.009s  0.009s  0.009s
  reduce (array build)     0.004s  0.004s  0.004s  0.004s  0.004s
  reduce (obj build)       0.009s  0.010s  0.010s  0.010s  0.010s
  reduce (setpath)         0.015s  0.017s  0.016s  0.016s  0.016s
  foreach (running sum)    0.009s  0.010s  0.010s  0.010s  0.010s
  foreach + emit           0.009s  0.009s  0.010s  0.010s  0.010s
  reduce (sum-of-squares)  0.032s  0.032s  0.032s  0.032s  0.032s
  reduce (conditional)     0.034s  0.034s  0.034s  0.034s  0.035s
  reduce (product)         0.036s  0.034s  0.034s  0.034s  0.035s
  foreach (conditional)    0.010s  0.010s  0.010s  0.010s  0.011s
  until (100M)             0.294s  0.293s  0.295s  0.294s  0.297s
  reduce (harmonic)        0.031s  0.032s  0.032s  0.032s  0.033s
  reduce (floor pipe)      0.031s  0.032s  0.032s  0.032s  0.033s
  reduce (sqrt pipe)       0.032s  0.033s  0.032s  0.032s  0.032s
  reduce (sin+cos)         0.050s  0.051s  0.052s  0.052s  0.052s

--- Object operations ---
  Benchmark                v1.0.0  v1.1.0  v1.4.3  v1.4.4  3d440ca
  ---                      ------  ------  ------  ------  -------
  large obj construct      0.003s  0.004s  0.004s  0.004s  0.004s
  large obj keys           0.009s  0.010s  0.011s  0.010s  0.011s
  large obj to_entries     0.010s  0.011s  0.011s  0.012s  0.012s
  with_entries             0.008s  0.009s  0.009s  0.009s  0.009s

--- Assignment operators ---
  Benchmark                v1.0.0  v1.1.0  v1.4.3  v1.4.4  3d440ca
  ---                      ------  ------  ------  ------  -------
  .[] |= f (100K)          0.004s  0.005s  0.005s  0.005s  0.005s
  .[] += 1 (100K)          0.004s  0.005s  0.005s  0.005s  0.005s
  .[k] = v reduce(50K)     0.008s  0.009s  0.008s  0.008s  0.008s

--- String-heavy generators ---
  Benchmark                v1.0.0  v1.1.0  v1.4.3  v1.4.4  3d440ca
  ---                      ------  ------  ------  ------  -------
  gsub(100K)               0.018s  0.019s  0.025s  0.025s  0.025s
  join large(100K)         0.005s  0.005s  0.006s  0.005s  0.006s
  explode/implode(100K)    0.027s  0.029s  0.029s  0.028s  0.029s
  reduce str concat(100K)  0.007s  0.008s  0.008s  0.008s  0.008s

--- Try-catch & alternative ---
  Benchmark                v1.0.0  v1.1.0  v1.4.3  v1.4.4  3d440ca
  ---                      ------  ------  ------  ------  -------
  alternative //           0.031s  0.032s  0.031s  0.031s  0.032s
  try-catch                0.022s  0.023s  0.023s  0.023s  0.023s
  label-break              0.003s  0.004s  0.004s  0.004s  0.004s

--- Type conversion ---
  Benchmark                v1.0.0  v1.1.0  v1.4.3  v1.4.4  3d440ca
  ---                      ------  ------  ------  ------  -------
  tojson/fromjson(100K)    0.021s  0.022s  0.021s  0.021s  0.022s
  null propagation(2M)     0.101s  0.087s  0.089s  0.090s  0.088s

--- jaq-derived ---
  Benchmark                v1.0.0  v1.1.0  v1.4.3  v1.4.4  3d440ca
  ---                      ------  ------  ------  ------  -------
  jaq: reverse             0.010s  0.010s  0.010s  0.011s  0.011s
  jaq: sort                0.016s  0.017s  0.018s  0.018s  0.018s
  jaq: group-by            0.036s  0.037s  0.034s  0.035s  0.037s
  jaq: min-max             0.010s  0.011s  0.010s  0.010s  0.011s
  jaq: ex-implode          0.018s  0.019s  0.019s  0.019s  0.019s
  jaq: repeat              0.011s  0.012s  0.011s  0.011s  0.011s
  jaq: from                0.005s  0.006s  0.006s  0.006s  0.006s
  jaq: last                0.002s  0.003s  0.002s  0.002s  0.002s
  jaq: cumsum              0.009s  0.010s  0.010s  0.012s  0.010s
  jaq: cumsum-xy           0.016s  0.017s  0.016s  0.018s  0.017s
  jaq: try-catch           0.085s  0.090s  0.084s  0.088s  0.083s
  jaq: add                 0.039s  0.042s  0.039s  0.042s  0.040s
  jaq: reduce              0.079s  0.092s  0.084s  0.094s  0.081s
  jaq: reduce-update       0.004s  0.005s  0.005s  0.005s  0.005s
  jaq: kv                  0.013s  0.015s  0.014s  0.014s  0.015s
  jaq: kv-update           0.017s  0.018s  0.019s  0.018s  0.018s
  jaq: kv-entries          0.056s  0.055s  0.056s  0.056s  0.055s
  jaq: pyramid             0.016s  0.016s  0.016s  0.015s  0.016s
  jaq: upto                0.007s  0.007s  0.007s  0.007s  0.007s
  jaq: tree-flatten        0.003s  0.003s  0.003s  0.003s  0.003s
  jaq: tree-update         0.006s  0.007s  0.007s  0.007s  0.007s
  jaq: to-fromjson         0.004s  0.005s  0.005s  0.005s  0.005s
  jaq: str-slice           0.013s  0.014s  0.014s  0.013s  0.014s
```
