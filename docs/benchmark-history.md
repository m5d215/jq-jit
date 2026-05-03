# Benchmark History

Recent slice (last 5 columns). Full history lives in
[`benchmark-history.tsv`](benchmark-history.tsv) (long format,
`section / benchmark / version / time_seconds`).

```text
--- NDJSON workloads (2M objects) ---
  Benchmark                v1.4.3  v1.4.4  3d440ca  v1.4.5  v1.5.0
  ---                      ------  ------  -------  ------  ------
  empty                    0.017s  0.017s  0.017s   0.017s  0.017s
  identity -c              0.082s  0.082s  0.078s   0.081s  0.082s
  identity (pretty)        0.108s  0.099s  0.101s   0.105s  0.098s
  field access .name       0.080s  0.080s  0.087s   0.091s  0.096s
  nested .x,.y,.name       0.111s  0.123s  0.144s   0.143s  0.150s
  arithmetic .x + .y       0.064s  0.094s  0.082s   0.081s  0.083s
  select .x > 1500000      0.066s  0.115s  0.069s   0.073s  0.081s
  string concat            0.090s  0.089s  0.096s   0.096s  0.092s
  object construct         0.087s  0.127s  0.112s   0.114s  0.113s
  array construct          0.097s  0.102s  0.116s   0.120s  0.106s
  .[]                      0.085s  0.098s  0.100s   0.099s  0.103s
  to_entries               0.104s  0.112s  0.161s   0.156s  0.164s
  keys                     0.088s  0.102s  0.098s   0.101s  0.104s
  keys_unsorted            0.078s  0.091s  0.094s   0.093s  0.095s
  length                   0.073s  0.085s  0.080s   0.083s  0.087s
  has("x")                 0.030s  0.030s  0.029s   0.030s  0.036s
  type                     0.019s  0.019s  0.022s   0.022s  0.022s
  del(.name)               0.098s  0.095s  0.098s   0.099s  0.103s
  @csv                     -       -       0.140s   0.137s  0.127s
  split/join               -       -       0.093s   0.098s  0.091s
  select|field             -       -       0.112s   0.112s  0.096s
  select|remap             -       -       0.095s   0.096s  0.096s
  computed remap           -       -       0.211s   0.214s  0.187s
  [.x,.y]|add              -       -       0.082s   0.083s  0.084s
  [.x,.y]|avg              -       -       0.110s   0.112s  0.108s
  map(*2)|add              -       -       0.110s   0.113s  0.104s
  keys|length              -       -       0.254s   0.254s  0.254s
  .+{z=0}                  -       -       0.143s   0.147s  0.151s
  split|first              -       -       0.092s   0.098s  0.090s
  slice[0..5]              -       -       0.095s   0.100s  0.093s
  dynkey {(.name)}         -       -       0.118s   0.121s  0.108s
  .x += 1                  -       -       0.069s   0.070s  0.073s
  {a}+{b} merge            -       -       0.143s   0.147s  0.132s
  .x*2+1                   -       -       0.049s   0.050s  0.058s
  .x+.y*2                  -       -       0.102s   0.105s  0.097s
  .x > .y                  -       -       0.076s   0.077s  0.077s
  to_entries|len           -       -       0.397s   0.395s  0.398s
  .x|.+1 (pipe)            -       -       0.047s   0.048s  0.058s
  .x|.*2|.+1               -       -       0.049s   0.050s  0.058s
  .name|.+"_x"             -       -       0.098s   0.098s  0.093s
  .x>N | not               -       -       0.040s   0.041s  0.049s
  and (2 cmp)              -       -       0.080s   0.080s  0.080s
  if-then-else             -       -       0.043s   0.043s  0.051s
  sel(and)|field           -       -       0.076s   0.076s  0.078s
  sel(and)|remap           -       -       0.074s   0.076s  0.076s
  arith|cmp                -       -       0.044s   0.047s  0.054s
  if cmp .field            -       -       0.102s   0.107s  0.112s
  split|length             -       -       0.094s   0.099s  0.090s
  [x,y]|min                -       -       0.090s   0.092s  0.090s
  [x,y]|max                -       -       0.091s   0.095s  0.095s
  [x,y]|sort|.[0]          -       -       0.087s   0.090s  0.089s
  .name|len>5              -       -       0.096s   0.100s  0.093s
  sel(len>5)|.x            -       -       0.125s   0.127s  0.105s
  if .x>.y .name           -       -       0.087s   0.090s  0.088s
  sel(.x>.y)|.name         -       -       0.071s   0.072s  0.073s
  .x*2|tostring            -       -       0.047s   0.048s  0.056s
  .x*.x+1                  -       -       0.056s   0.057s  0.065s
  {k=.name,v=tostr}        -       -       0.160s   0.162s  0.143s
  str add chain            -       -       0.386s   0.398s  0.381s
  if>.y .name|empty        -       -       0.071s   0.073s  0.073s
  if .x%2==0               -       -       0.045s   0.047s  0.053s
  if .x*2+1>1M             -       -       0.045s   0.048s  0.057s
  sel(.x%2==0)|.name       -       -       0.074s   0.076s  0.085s
  sel(.x*2+1>1M)           -       -       0.142s   0.144s  0.156s
  .x|@json                 -       -       0.045s   0.045s  0.047s
  .x|@text                 -       -       0.045s   0.045s  0.048s
  .name|@json              -       -       0.104s   0.107s  0.101s
  sel|[arr]                -       -       0.148s   0.150s  0.143s
  sel(and)|[arr]           -       -       0.076s   0.078s  0.079s
  if>.y [arr]              -       -       0.199s   0.199s  0.180s
  if sw then .f            -       -       0.135s   0.138s  0.138s
  dynkey {(.n)=.x*2}       -       -       0.131s   0.132s  0.112s
  sel(and)|.x*.y           -       -       0.075s   0.077s  0.077s
  sel>N|str chain          -       -       0.158s   0.158s  0.154s
  .f+"_"+arith_ts          -       -       0.145s   0.148s  0.130s
  sel(sw)|str ch           -       -       0.317s   0.310s  0.302s
  split|rev|join           -       -       0.123s   0.122s  0.114s
  dynkey+static            -       -       0.324s   0.331s  0.332s
  if>.y str chain          -       -       0.186s   0.189s  0.167s
  remap+str chain          -       -       0.171s   0.178s  0.155s
  sel(len>8)               -       -       0.163s   0.170s  0.161s
  up|split|join            -       -       0.101s   0.104s  0.096s
  .name|index              -       -       0.126s   0.128s  0.119s
  .name|index+1            -       -       0.127s   0.133s  0.130s
  .name|rindex             -       -       0.131s   0.134s  0.124s
  .name|indices            -       -       0.153s   0.160s  0.152s
  [x,y]|sort               -       -       0.154s   0.156s  0.154s
  .name|scan               -       -       0.218s   0.219s  0.216s
  .name|gsub               -       -       0.175s   0.178s  0.169s
  walk(if num .+1)         -       -       0.140s   0.141s  0.143s
  tojson                   -       -       0.106s   0.110s  0.107s
  {name,x}                 -       -       0.144s   0.144s  0.128s
  .z//.name                -       -       0.165s   0.164s  0.158s
  .x|=test(re)             -       -       0.123s   0.124s  0.121s
  ./sep|first              -       -       0.131s   0.132s  0.129s
  .y=(.x*2)                -       -       0.112s   0.115s  0.125s
  .y=(.x+.y)               -       -       0.161s   0.159s  0.172s
  objects                  -       -       0.081s   0.085s  0.088s
  .tag|=if..then N         -       -       0.613s   0.614s  0.625s
  .x=(.x+1)                -       -       0.070s   0.072s  0.072s
  sel>N|.y+=1              -       -       0.085s   0.087s  0.094s
  sel(and)|.x+=1           -       -       0.100s   0.102s  0.111s
  sel(sw)|.x+=1            -       -       0.132s   0.133s  0.132s
  match(re)                -       -       0.369s   0.367s  0.367s
  capture(re)              -       -       0.306s   0.303s  0.307s
  first(.name,.x)          -       -       0.090s   0.093s  0.097s
  if .x==null              -       -       0.043s   0.044s  0.047s
  we(sw(.key))             -       -       0.111s   0.111s  0.112s
  sel(sw or ew)            -       -       0.211s   0.214s  0.211s
  path(.name,.x)           -       -       0.276s   0.277s  0.277s
  sel(str+num+num)         -       -       0.142s   0.145s  0.154s
  nested if|field          -       -       0.076s   0.077s  0.077s
  .f|floor|.*2             -       -       0.050s   0.051s  0.060s
  split|len>1              -       -       0.122s   0.124s  0.117s
  .name|len|.*2            -       -       0.106s   0.107s  0.102s
  if len>5 .x .y           -       -       0.137s   0.133s  0.117s
  sel(len>5)|remap         -       -       0.230s   0.233s  0.200s
  .x|tostr|len             -       -       0.056s   0.056s  0.060s
  if .x>.y .x .y           -       -       0.093s   0.094s  0.094s
  split|last|tonum         -       -       0.100s   0.104s  0.096s
  split|rev|.[0]           -       -       0.097s   0.099s  0.093s
  split|.[0]+.[1]          -       -       0.122s   0.123s  0.113s
  .[]|strings              -       -       0.096s   0.095s  0.095s
  .[]|numbers              -       -       0.107s   0.106s  0.106s
  [x,y]|any(>1M)           -       -       0.082s   0.082s  0.084s
  sel(dc|sw)               -       -       0.104s   0.107s  0.098s
  [[x,y],[n]]|flat         -       -       0.475s   0.475s  0.452s
  .x|floor|.*2             -       -       0.051s   0.051s  0.061s
  tojson|fromjson          -       -       0.083s   0.081s  0.088s
  [.x]|add                 -       -       0.048s   0.048s  0.060s
  if>N {o}+.               -       -       0.123s   0.125s  0.145s
  if>N .+{o}               -       -       0.124s   0.125s  0.136s
  if .n=="s" .+{o}         -       -       0.167s   0.161s  0.162s
  sel(.n>"s")              -       -       0.095s   0.094s  0.089s
  [x,y,z]|min              -       -       0.311s   0.312s  0.303s
  if .n|len>5 l s          -       -       0.107s   0.106s  0.102s
  if .x|flr>N b s          -       -       0.047s   0.045s  0.057s
  if .n|test l e           -       -       0.111s   0.112s  0.105s
  if .n|sw l e             -       -       0.090s   0.091s  0.083s
  if .n|ew l e             -       -       0.092s   0.091s  0.084s
  .n|len|tostr             -       -       0.097s   0.099s  0.092s

--- String operations (2M objects) ---
  Benchmark                v1.4.3  v1.4.4  3d440ca  v1.4.5  v1.5.0
  ---                      ------  ------  -------  ------  ------
  ascii_downcase           0.090s  0.088s  0.110s   0.112s  0.104s
  ascii_upcase             0.090s  0.088s  0.108s   0.109s  0.102s
  ltrimstr                 0.090s  0.090s  0.101s   0.102s  0.096s
  rtrimstr                 0.089s  0.090s  0.105s   0.107s  0.098s
  split                    0.150s  0.151s  0.169s   0.172s  0.169s
  case+split               0.117s  0.116s  0.126s   0.128s  0.115s
  join                     0.087s  0.092s  0.101s   0.104s  0.094s
  startswith               0.084s  0.086s  0.101s   0.103s  0.095s
  endswith                 0.085s  0.088s  0.103s   0.104s  0.098s
  tostring                 0.043s  0.095s  0.052s   0.053s  0.061s
  tonumber                 0.108s  0.104s  0.117s   0.117s  0.111s
  string interpolation     0.104s  0.106s  0.135s   0.134s  0.119s

--- String ops (200K objects) ---
  Benchmark                v1.4.3  v1.4.4  3d440ca  v1.4.5  v1.5.0
  ---                      ------  ------  -------  ------  ------
  test (regex)             0.013s  0.013s  0.014s   0.014s  0.014s
  match (regex)            0.031s  0.032s  0.033s   0.033s  0.031s
  @base64                  0.011s  0.011s  0.013s   0.012s  0.012s
  @uri                     0.011s  0.011s  0.013s   0.013s  0.012s
  @html                    0.012s  0.011s  0.013s   0.013s  0.012s
  @csv (array)             0.013s  0.014s  0.020s   0.018s  0.016s
  @tsv (array)             0.013s  0.014s  0.018s   0.017s  0.016s
  gsub                     0.018s  0.018s  0.020s   0.020s  0.019s
  case+gsub                0.181s  0.180s  0.193s   0.193s  0.181s
  case+test                0.116s  0.115s  0.130s   0.130s  0.121s
  ltrim+tonum+arith        0.107s  0.107s  0.118s   0.118s  0.110s

--- Numeric & math (2M objects) ---
  Benchmark                v1.4.3  v1.4.4  3d440ca  v1.4.5  v1.5.0
  ---                      ------  ------  -------  ------  ------
  floor                    0.042s  0.091s  0.048s   0.048s  0.057s
  sqrt                     0.072s  0.115s  0.078s   0.078s  0.079s
  modulo                   0.045s  0.095s  0.051s   0.051s  0.059s
  if-elif-else             0.102s  0.134s  0.125s   0.124s  0.127s
  select|del               0.073s  0.126s  0.079s   0.079s  0.095s
  select|merge             0.105s  0.157s  0.110s   0.107s  0.120s
  select(test)|merge       0.022s  0.021s  0.022s   0.022s  0.022s

--- Array generators ---
  Benchmark                v1.4.3  v1.4.4  3d440ca  v1.4.5  v1.5.0
  ---                      ------  ------  -------  ------  ------
  range(2M) | length       0.011s  0.011s  0.011s   0.011s  0.011s
  reverse(2M)              0.017s  0.017s  0.018s   0.017s  0.017s
  sort(2M)                 0.022s  0.022s  0.023s   0.023s  0.023s
  unique(1M)               0.028s  0.028s  0.029s   0.029s  0.029s
  flatten(500K)            0.010s  0.010s  0.010s   0.010s  0.010s
  min, max(2M)             0.017s  0.018s  0.017s   0.017s  0.018s
  add numbers(2M)          0.012s  0.012s  0.012s   0.012s  0.012s
  any/all(2M)              0.028s  0.027s  0.028s   0.027s  0.028s
  limit(10; range(10M))    0.002s  0.002s  0.002s   0.002s  0.002s
  first(range(10M))        0.002s  0.002s  0.002s   0.002s  0.002s
  last(range(2M))          0.002s  0.002s  0.002s   0.002s  0.002s
  indices(1M)              0.015s  0.015s  0.015s   0.015s  0.016s

--- Reduce & foreach ---
  Benchmark                v1.4.3  v1.4.4  3d440ca  v1.4.5  v1.5.0
  ---                      ------  ------  -------  ------  ------
  reduce (sum)             0.009s  0.009s  0.009s   0.009s  0.010s
  reduce (array build)     0.004s  0.004s  0.004s   0.004s  0.005s
  reduce (obj build)       0.010s  0.010s  0.010s   0.009s  0.010s
  reduce (setpath)         0.016s  0.016s  0.016s   0.017s  0.017s
  foreach (running sum)    0.010s  0.010s  0.010s   0.010s  0.010s
  foreach + emit           0.010s  0.010s  0.010s   0.010s  0.011s
  reduce (sum-of-squares)  0.032s  0.032s  0.032s   0.032s  0.035s
  reduce (conditional)     0.034s  0.034s  0.035s   0.035s  0.036s
  reduce (product)         0.034s  0.034s  0.035s   0.034s  0.034s
  foreach (conditional)    0.010s  0.010s  0.011s   0.010s  0.011s
  until (100M)             0.295s  0.294s  0.297s   0.295s  0.304s
  reduce (harmonic)        0.032s  0.032s  0.033s   0.033s  0.033s
  reduce (floor pipe)      0.032s  0.032s  0.033s   0.033s  0.034s
  reduce (sqrt pipe)       0.032s  0.032s  0.032s   0.032s  0.034s
  reduce (sin+cos)         0.052s  0.052s  0.052s   0.052s  0.053s

--- Object operations ---
  Benchmark                v1.4.3  v1.4.4  3d440ca  v1.4.5  v1.5.0
  ---                      ------  ------  -------  ------  ------
  large obj construct      0.004s  0.004s  0.004s   0.004s  0.004s
  large obj keys           0.011s  0.010s  0.011s   0.011s  0.012s
  large obj to_entries     0.011s  0.012s  0.012s   0.012s  0.014s
  with_entries             0.009s  0.009s  0.009s   0.009s  0.010s

--- Assignment operators ---
  Benchmark                v1.4.3  v1.4.4  3d440ca  v1.4.5  v1.5.0
  ---                      ------  ------  -------  ------  ------
  .[] |= f (100K)          0.005s  0.005s  0.005s   0.005s  0.006s
  .[] += 1 (100K)          0.005s  0.005s  0.005s   0.005s  0.006s
  .[k] = v reduce(50K)     0.008s  0.008s  0.008s   0.008s  0.009s

--- String-heavy generators ---
  Benchmark                v1.4.3  v1.4.4  3d440ca  v1.4.5  v1.5.0
  ---                      ------  ------  -------  ------  ------
  gsub(100K)               0.025s  0.025s  0.025s   0.025s  0.027s
  join large(100K)         0.006s  0.005s  0.006s   0.005s  0.006s
  explode/implode(100K)    0.029s  0.028s  0.029s   0.028s  0.028s
  reduce str concat(100K)  0.008s  0.008s  0.008s   0.008s  0.008s

--- Try-catch & alternative ---
  Benchmark                v1.4.3  v1.4.4  3d440ca  v1.4.5  v1.5.0
  ---                      ------  ------  -------  ------  ------
  alternative //           0.031s  0.031s  0.032s   0.032s  0.033s
  try-catch                0.023s  0.023s  0.023s   0.023s  0.024s
  label-break              0.004s  0.004s  0.004s   0.004s  0.004s

--- Type conversion ---
  Benchmark                v1.4.3  v1.4.4  3d440ca  v1.4.5  v1.5.0
  ---                      ------  ------  -------  ------  ------
  tojson/fromjson(100K)    0.021s  0.021s  0.022s   0.022s  0.023s
  null propagation(2M)     0.089s  0.090s  0.088s   0.087s  0.088s

--- jaq-derived ---
  Benchmark                v1.4.3  v1.4.4  3d440ca  v1.4.5  v1.5.0
  ---                      ------  ------  -------  ------  ------
  jaq: reverse             0.010s  0.011s  0.011s   0.010s  0.011s
  jaq: sort                0.018s  0.018s  0.018s   0.017s  0.018s
  jaq: group-by            0.034s  0.035s  0.037s   0.037s  0.038s
  jaq: min-max             0.010s  0.010s  0.011s   0.010s  0.012s
  jaq: ex-implode          0.019s  0.019s  0.019s   0.019s  0.020s
  jaq: repeat              0.011s  0.011s  0.011s   0.011s  0.012s
  jaq: from                0.006s  0.006s  0.006s   0.006s  0.006s
  jaq: last                0.002s  0.002s  0.002s   0.002s  0.002s
  jaq: cumsum              0.010s  0.012s  0.010s   0.010s  0.011s
  jaq: cumsum-xy           0.016s  0.018s  0.017s   0.017s  0.018s
  jaq: try-catch           0.084s  0.088s  0.083s   0.090s  0.078s
  jaq: add                 0.039s  0.042s  0.040s   0.040s  0.041s
  jaq: reduce              0.084s  0.094s  0.081s   0.086s  0.086s
  jaq: reduce-update       0.005s  0.005s  0.005s   0.005s  0.006s
  jaq: kv                  0.014s  0.014s  0.015s   0.014s  0.015s
  jaq: kv-update           0.019s  0.018s  0.018s   0.018s  0.019s
  jaq: kv-entries          0.056s  0.056s  0.055s   0.055s  0.059s
  jaq: pyramid             0.016s  0.015s  0.016s   0.016s  0.017s
  jaq: upto                0.007s  0.007s  0.007s   0.007s  0.009s
  jaq: tree-flatten        0.003s  0.003s  0.003s   0.003s  0.004s
  jaq: tree-update         0.007s  0.007s  0.007s   0.007s  0.007s
  jaq: to-fromjson         0.005s  0.005s  0.005s   0.005s  0.005s
  jaq: str-slice           0.014s  0.013s  0.014s   0.014s  0.015s
```
