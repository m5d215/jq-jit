# Benchmark History

Recent slice (last 5 columns). Full history lives in
[`benchmark-history.tsv`](benchmark-history.tsv) (long format,
`section / benchmark / version / time_seconds`).

```text
--- NDJSON workloads (2M objects) ---
  Benchmark                    v1.5.5  v1.5.6  v1.6.0  v1.6.1  v1.7.0
  ---                          ------  ------  ------  ------  ------
  empty                        0.017s  0.017s  0.017s  0.018s  0.017s
  identity -c                  0.087s  0.087s  0.088s  0.087s  0.086s
  identity (pretty)            0.102s  0.105s  0.106s  0.107s  0.106s
  field access .name           0.094s  0.095s  0.096s  0.097s  0.093s
  nested .x,.y,.name           0.150s  0.150s  0.155s  0.150s  0.147s
  arithmetic .x + .y           0.086s  0.083s  0.086s  0.084s  0.083s
  select .x > 1500000          0.083s  0.082s  0.083s  0.087s  0.084s
  string concat                0.093s  0.093s  0.092s  0.095s  0.092s
  object construct             0.115s  0.111s  0.115s  0.116s  0.114s
  array construct              0.106s  0.103s  0.107s  0.106s  0.105s
  .[]                          0.106s  0.101s  0.105s  0.104s  0.102s
  to_entries                   0.162s  0.159s  0.159s  0.159s  0.163s
  keys                         0.106s  0.109s  0.105s  0.107s  0.105s
  keys_unsorted                0.095s  0.095s  0.095s  0.097s  0.094s
  length                       0.086s  0.085s  0.087s  0.088s  0.084s
  has("x")                     0.037s  0.034s  0.039s  0.039s  0.037s
  type                         0.023s  0.022s  0.023s  0.023s  0.023s
  del(.name)                   0.102s  0.104s  0.103s  0.107s  0.101s
  @csv                         0.124s  0.123s  0.124s  0.120s  0.118s
  split/join                   0.091s  0.089s  0.090s  0.094s  0.090s
  select|field                 0.097s  0.095s  0.101s  0.099s  0.097s
  select|remap                 0.100s  0.095s  0.098s  0.100s  0.099s
  computed remap               0.192s  0.191s  0.190s  0.192s  0.191s
  [.x,.y]|add                  0.086s  0.082s  0.084s  0.084s  0.083s
  [.x,.y]|avg                  0.112s  0.106s  0.109s  0.111s  0.106s
  map(*2)|add                  0.106s  0.103s  0.106s  0.107s  0.104s
  keys|length                  0.252s  0.251s  0.253s  0.259s  0.255s
  .+{z=0}                      0.147s  0.146s  0.150s  0.150s  0.146s
  split|first                  0.090s  0.090s  0.091s  0.092s  0.089s
  slice[0..5]                  0.092s  0.092s  0.093s  0.095s  0.094s
  dynkey {(.name)}             0.104s  0.102s  0.106s  0.108s  0.104s
  .x += 1                      0.125s  0.125s  0.129s  0.128s  0.121s
  {a}+{b} merge                0.130s  0.135s  0.132s  0.133s  0.130s
  .x*2+1                       0.060s  0.060s  0.060s  0.061s  0.060s
  .x+.y*2                      0.100s  0.096s  0.100s  0.099s  0.100s
  .x > .y                      0.081s  0.077s  0.079s  0.078s  0.077s
  to_entries|len               0.394s  0.400s  0.400s  0.405s  0.397s
  .x|.+1 (pipe)                0.058s  0.058s  0.058s  0.059s  0.059s
  .x|.*2|.+1                   0.060s  0.059s  0.060s  0.060s  0.060s
  .name|.+"_x"                 0.092s  0.092s  0.094s  0.097s  0.094s
  .x>N | not                   0.050s  0.049s  0.049s  0.051s  0.049s
  and (2 cmp)                  0.085s  0.080s  0.085s  0.082s  0.080s
  if-then-else                 0.052s  0.052s  0.051s  0.053s  0.052s
  sel(and)|field               0.083s  0.077s  0.080s  0.079s  0.076s
  sel(and)|remap               0.083s  0.077s  0.079s  0.079s  0.077s
  arith|cmp                    0.055s  0.053s  0.054s  0.056s  0.054s
  if cmp .field                0.114s  0.114s  0.113s  0.118s  0.113s
  split|length                 0.088s  0.090s  0.089s  0.093s  0.089s
  [x,y]|min                    0.095s  0.090s  0.092s  0.092s  0.090s
  [x,y]|max                    0.097s  0.092s  0.096s  0.097s  0.095s
  [x,y]|sort|.[0]              0.094s  0.090s  0.093s  0.092s  0.089s
  .name|len>5                  0.092s  0.092s  0.093s  0.094s  0.090s
  sel(len>5)|.x                0.110s  0.111s  0.113s  0.111s  0.107s
  if .x>.y .name               0.092s  0.089s  0.093s  0.092s  0.093s
  sel(.x>.y)|.name             0.076s  0.072s  0.075s  0.074s  0.071s
  .x*2|tostring                0.057s  0.056s  0.057s  0.058s  0.057s
  .x*.x+1                      0.066s  0.066s  0.066s  0.067s  0.067s
  {k=.name,v=tostr}            0.146s  0.144s  0.151s  0.147s  0.144s
  str add chain                0.378s  0.385s  0.388s  0.388s  0.385s
  if>.y .name|empty            0.080s  0.073s  0.077s  0.073s  0.073s
  if .x%2==0                   0.056s  0.054s  0.055s  0.054s  0.055s
  if .x*2+1>1M                 0.056s  0.054s  0.055s  0.055s  0.056s
  sel(.x%2==0)|.name           0.083s  0.083s  0.083s  0.084s  0.084s
  sel(.x*2+1>1M)               0.159s  0.159s  0.160s  0.157s  0.159s
  .x|@json                     0.047s  0.048s  0.048s  0.049s  0.047s
  .x|@text                     0.048s  0.047s  0.048s  0.049s  0.048s
  .name|@json                  0.104s  0.103s  0.103s  0.102s  0.101s
  sel|[arr]                    0.146s  0.143s  0.147s  0.143s  0.149s
  sel(and)|[arr]               0.082s  0.077s  0.081s  0.077s  0.079s
  if>.y [arr]                  0.176s  0.173s  0.178s  0.175s  0.179s
  if sw then .f                0.138s  0.139s  0.143s  0.143s  0.139s
  dynkey {(.n)=.x*2}           0.113s  0.114s  0.115s  0.117s  0.115s
  sel(and)|.x*.y               0.082s  0.076s  0.079s  0.077s  0.077s
  sel>N|str chain              0.154s  0.151s  0.156s  0.152s  0.155s
  .f+"_"+arith_ts              0.133s  0.132s  0.136s  0.134s  0.133s
  sel(sw)|str ch               0.303s  0.307s  0.304s  0.308s  0.307s
  split|rev|join               0.114s  0.114s  0.118s  0.117s  0.117s
  dynkey+static                0.344s  0.331s  0.338s  0.340s  0.338s
  if>.y str chain              0.167s  0.165s  0.166s  0.167s  0.171s
  remap+str chain              0.153s  0.148s  0.159s  0.156s  0.156s
  sel(len>8)                   0.160s  0.165s  0.161s  0.163s  0.162s
  up|split|join                0.099s  0.097s  0.098s  0.098s  0.095s
  .name|index                  0.121s  0.121s  0.120s  0.121s  0.118s
  .name|index+1                0.125s  0.123s  0.124s  0.126s  0.122s
  .name|rindex                 0.129s  0.127s  0.130s  0.131s  0.127s
  .name|indices                0.157s  0.146s  0.159s  0.161s  0.155s
  [x,y]|sort                   0.153s  0.152s  0.156s  0.154s  0.153s
  .name|scan                   0.209s  0.208s  0.209s  0.214s  0.207s
  .name|gsub                   0.163s  0.168s  0.168s  0.170s  0.167s
  walk(if num .+1)             0.145s  0.143s  0.142s  0.140s  0.141s
  tojson                       0.106s  0.104s  0.106s  0.106s  0.104s
  {name,x}                     0.129s  0.133s  0.130s  0.129s  0.128s
  .z//.name                    0.157s  0.157s  0.157s  0.158s  0.154s
  .x|=test(re)                 0.173s  0.180s  0.171s  0.176s  0.172s
  ./sep|first                  0.184s  0.182s  0.190s  0.185s  0.182s
  .y=(.x*2)                    0.179s  0.178s  0.181s  0.180s  0.174s
  .y=(.x+.y)                   0.229s  0.233s  0.238s  0.234s  0.228s
  objects                      0.128s  0.138s  0.134s  0.138s  0.134s
  .tag|=if..then N             0.607s  0.614s  0.617s  0.611s  0.627s
  .x=(.x+1)                    0.122s  0.128s  0.129s  0.127s  0.124s
  sel>N|.y+=1                  0.123s  0.121s  0.124s  0.122s  0.122s
  sel(and)|.x+=1               0.109s  0.111s  0.111s  0.111s  0.109s
  sel(sw)|.x+=1                0.163s  0.164s  0.165s  0.163s  0.159s
  match(re)                    0.361s  0.369s  0.364s  0.367s  0.362s
  capture(re)                  0.298s  0.296s  0.297s  0.299s  0.302s
  first(.name,.x)              0.096s  0.097s  0.098s  0.097s  0.094s
  if .x==null                  0.047s  0.046s  0.048s  0.047s  0.046s
  we(sw(.key))                 0.107s  0.110s  0.110s  0.105s  0.108s
  sel(sw or ew)                0.207s  0.205s  0.202s  0.207s  0.205s
  path(.name,.x)               0.274s  0.274s  0.276s  0.278s  0.278s
  sel(str+num+num)             0.151s  0.152s  0.152s  0.153s  0.150s
  nested if|field              0.082s  0.076s  0.081s  0.077s  0.077s
  .f|floor|.*2                 0.061s  0.061s  0.061s  0.061s  0.060s
  split|len>1                  0.118s  0.114s  0.118s  0.119s  0.115s
  .name|len|.*2                0.102s  0.103s  0.104s  0.103s  0.103s
  if len>5 .x .y               0.115s  0.113s  0.118s  0.113s  0.113s
  sel(len>5)|remap             0.206s  0.207s  0.207s  0.202s  0.201s
  .x|tostr|len                 0.058s  0.059s  0.062s  0.060s  0.059s
  if .x>.y .x .y               0.098s  0.093s  0.097s  0.094s  0.094s
  split|last|tonum             0.095s  0.096s  0.096s  0.097s  0.096s
  split|rev|.[0]               0.094s  0.090s  0.096s  0.091s  0.091s
  split|.[0]+.[1]              0.115s  0.114s  0.113s  0.115s  0.113s
  .[]|strings                  0.106s  0.104s  0.106s  0.106s  0.105s
  .[]|numbers                  0.124s  0.124s  0.129s  0.125s  0.124s
  [x,y]|any(>1M)               0.087s  0.082s  0.084s  0.082s  0.082s
  sel(dc|sw)                   0.102s  0.097s  0.098s  0.101s  0.098s
  [[x,y],[n]]|flat             0.456s  0.460s  0.460s  0.462s  0.464s
  .x|floor|.*2                 0.061s  0.061s  0.061s  0.060s  0.061s
  tojson|fromjson              0.087s  0.085s  0.087s  0.084s  0.082s
  [.x]|add                     0.060s  0.060s  0.060s  0.060s  0.059s
  if>N {o}+.                   0.134s  0.135s  0.136s  0.135s  0.138s
  if>N .+{o}                   0.134s  0.138s  0.135s  0.135s  0.136s
  if .n=="s" .+{o}             0.161s  0.161s  0.158s  0.161s  0.157s
  sel(.n>"s")                  0.089s  0.089s  0.089s  0.090s  0.091s
  [x,y,z]|min                  0.309s  0.309s  0.308s  0.311s  0.307s
  if .n|len>5 l s              0.102s  0.100s  0.100s  0.101s  0.100s
  if .x|flr>N b s              0.056s  0.056s  0.056s  0.055s  0.055s
  if .n|test l e               0.103s  0.106s  0.105s  0.105s  0.105s
  if .n|sw l e                 0.083s  0.084s  0.083s  0.085s  0.082s
  if .n|ew l e                 0.084s  0.084s  0.084s  0.086s  0.085s
  .n|len|tostr                 0.091s  0.092s  0.092s  0.091s  0.090s

--- String operations (2M objects) ---
  Benchmark                    v1.5.5  v1.5.6  v1.6.0  v1.6.1  v1.7.0
  ---                          ------  ------  ------  ------  ------
  ascii_downcase               0.108s  0.105s  0.105s  0.106s  0.104s
  ascii_upcase                 0.108s  0.103s  0.104s  0.106s  0.103s
  ltrimstr                     0.096s  0.095s  0.096s  0.099s  0.097s
  rtrimstr                     0.099s  0.098s  0.098s  0.099s  0.097s
  split                        0.168s  0.162s  0.170s  0.165s  0.162s
  case+split                   0.113s  0.115s  0.118s  0.116s  0.121s
  join                         0.090s  0.091s  0.096s  0.093s  0.093s
  startswith                   0.095s  0.095s  0.098s  0.098s  0.095s
  endswith                     0.097s  0.096s  0.098s  0.098s  0.096s
  tostring                     0.062s  0.063s  0.063s  0.063s  0.063s
  tonumber                     0.114s  0.110s  0.112s  0.111s  0.110s
  string interpolation         0.115s  0.118s  0.120s  0.116s  0.118s

--- String ops (200K objects) ---
  Benchmark                    v1.5.5  v1.5.6  v1.6.0  v1.6.1  v1.7.0
  ---                          ------  ------  ------  ------  ------
  test (regex)                 0.014s  0.015s  0.015s  0.015s  0.014s
  match (regex)                0.032s  0.033s  0.032s  0.033s  0.032s
  @base64                      0.012s  0.012s  0.012s  0.012s  0.012s
  @uri                         0.013s  0.012s  0.012s  0.013s  0.012s
  @html                        0.013s  0.012s  0.012s  0.013s  0.012s
  @csv (array)                 0.015s  0.015s  0.016s  0.016s  0.016s
  @tsv (array)                 0.014s  0.015s  0.015s  0.015s  0.015s
  gsub                         0.018s  0.019s  0.019s  0.019s  0.019s
  case+gsub                    0.176s  0.181s  0.181s  0.181s  0.180s
  case+test                    0.118s  0.119s  0.116s  0.124s  0.120s
  ltrim+tonum+arith            0.112s  0.111s  0.113s  0.113s  0.111s

--- Numeric & math (2M objects) ---
  Benchmark                    v1.5.5  v1.5.6  v1.6.0  v1.6.1  v1.7.0
  ---                          ------  ------  ------  ------  ------
  floor                        0.057s  0.057s  0.059s  0.056s  0.057s
  sqrt                         0.079s  0.079s  0.078s  0.079s  0.079s
  modulo                       0.058s  0.057s  0.057s  0.057s  0.059s
  if-elif-else                 0.124s  0.121s  0.124s  0.125s  0.123s
  select|del                   0.090s  0.092s  0.090s  0.090s  0.091s
  select|merge                 0.119s  0.120s  0.121s  0.118s  0.121s
  select(test)|merge           0.021s  0.022s  0.021s  0.022s  0.021s

--- Array generators ---
  Benchmark                    v1.5.5  v1.5.6  v1.6.0  v1.6.1  v1.7.0
  ---                          ------  ------  ------  ------  ------
  range(2M) | length           0.011s  0.011s  0.012s  0.012s  0.012s
  reverse(2M)                  0.018s  0.018s  0.018s  0.018s  0.018s
  sort(2M)                     0.023s  0.023s  0.023s  0.023s  0.023s
  unique(1M)                   0.030s  0.030s  0.031s  0.030s  0.030s
  flatten(500K)                0.010s  0.010s  0.011s  0.011s  0.011s
  min, max(2M)                 0.021s  0.018s  0.022s  0.021s  0.019s
  add numbers(2M)              0.013s  0.013s  0.013s  0.013s  0.013s
  any/all(2M)                  0.028s  0.028s  0.029s  0.028s  0.029s
  limit(10; range(10M))        0.002s  0.002s  0.003s  0.002s  0.003s
  first(range(10M))            0.002s  0.002s  0.003s  0.002s  0.003s
  last(range(2M))              0.002s  0.002s  0.003s  0.002s  0.003s
  indices(1M)                  0.016s  0.016s  0.017s  0.017s  0.017s

--- Reduce & foreach ---
  Benchmark                    v1.5.5  v1.5.6  v1.6.0  v1.6.1  v1.7.0
  ---                          ------  ------  ------  ------  ------
  reduce (sum)                 0.009s  0.009s  0.009s  0.009s  0.009s
  reduce (array build)         0.004s  0.004s  0.004s  0.004s  0.004s
  reduce (obj build)           0.009s  0.009s  0.010s  0.010s  0.010s
  reduce (setpath)             0.016s  0.016s  0.017s  0.016s  0.016s
  foreach (running sum)        0.010s  0.010s  0.010s  0.010s  0.010s
  foreach + emit               0.010s  0.010s  0.011s  0.011s  0.010s
  reduce (sum-of-squares)      0.033s  0.033s  0.036s  0.034s  0.034s
  reduce (conditional)         0.036s  0.035s  0.037s  0.036s  0.035s
  reduce (product)             0.034s  0.034s  0.036s  0.035s  0.034s
  foreach (conditional)        0.010s  0.010s  0.011s  0.010s  0.010s
  until (100M)                 0.300s  0.301s  0.307s  0.303s  0.305s
  reduce (harmonic)            0.033s  0.033s  0.035s  0.033s  0.033s
  reduce (floor pipe)          0.033s  0.032s  0.035s  0.034s  0.033s
  reduce (sqrt pipe)           0.033s  0.032s  0.035s  0.033s  0.033s
  reduce (sin+cos)             0.052s  0.052s  0.052s  0.052s  0.052s

--- Object operations ---
  Benchmark                    v1.5.5  v1.5.6  v1.6.0  v1.6.1  v1.7.0
  ---                          ------  ------  ------  ------  ------
  large obj construct          0.004s  0.004s  0.004s  0.004s  0.004s
  large obj keys               0.011s  0.011s  0.012s  0.011s  0.011s
  large obj to_entries         0.012s  0.012s  0.012s  0.012s  0.012s
  with_entries                 0.009s  0.009s  0.009s  0.009s  0.009s

--- Assignment operators ---
  Benchmark                    v1.5.5  v1.5.6  v1.6.0  v1.6.1  v1.7.0
  ---                          ------  ------  ------  ------  ------
  .[] |= f (100K)              0.005s  0.005s  0.005s  0.005s  0.005s
  .[] += 1 (100K)              0.005s  0.005s  0.006s  0.005s  0.005s
  .[k] = v reduce(50K)         0.008s  0.008s  0.008s  0.008s  0.008s

--- String-heavy generators ---
  Benchmark                    v1.5.5  v1.5.6  v1.6.0  v1.6.1  v1.7.0
  ---                          ------  ------  ------  ------  ------
  gsub(100K)                   0.026s  0.027s  0.027s  0.027s  0.027s
  join large(100K)             0.005s  0.006s  0.006s  0.005s  0.005s
  explode/implode(100K)        0.027s  0.028s  0.028s  0.028s  0.027s
  reduce str concat(100K)      0.008s  0.008s  0.008s  0.008s  0.008s

--- Try-catch & alternative ---
  Benchmark                    v1.5.5  v1.5.6  v1.6.0  v1.6.1  v1.7.0
  ---                          ------  ------  ------  ------  ------
  alternative //               0.032s  0.032s  0.035s  0.035s  0.035s
  try-catch                    0.023s  0.023s  0.024s  0.024s  0.024s
  label-break                  0.004s  0.004s  0.004s  0.004s  0.004s

--- Type conversion ---
  Benchmark                    v1.5.5  v1.5.6  v1.6.0  v1.6.1  v1.7.0
  ---                          ------  ------  ------  ------  ------
  tojson/fromjson(100K)        0.022s  0.022s  0.023s  0.022s  0.022s
  null propagation(2M)         0.089s  0.090s  0.093s  0.091s  0.091s

--- jaq-derived ---
  Benchmark                    v1.5.5  v1.5.6  v1.6.0  v1.6.1  v1.7.0
  ---                          ------  ------  ------  ------  ------
  jaq: reverse                 -       -       -       -       -
  jaq: sort                    -       -       -       -       -
  jaq: group-by                -       -       -       -       -
  jaq: min-max                 -       -       -       -       -
  jaq: ex-implode              -       -       -       -       -
  jaq: repeat                  -       -       -       -       -
  jaq: from                    -       -       -       -       -
  jaq: last                    -       -       -       -       -
  jaq: cumsum                  -       -       -       -       -
  jaq: cumsum-xy               -       -       -       -       -
  jaq: try-catch               -       -       -       -       -
  jaq: add                     -       -       -       -       -
  jaq: reduce                  -       -       -       -       -
  jaq: reduce-update           -       -       -       -       -
  jaq: kv                      -       -       -       -       -
  jaq: kv-update               -       -       -       -       -
  jaq: kv-entries              -       -       -       -       -
  jaq: pyramid                 -       -       -       -       -
  jaq: upto                    -       -       -       -       -
  jaq: tree-flatten            -       -       -       -       -
  jaq: tree-update             -       -       -       -       -
  jaq: to-fromjson             -       -       -       -       -
  jaq: str-slice               -       -       -       -       -

--- Memoization (jqx) ---
  Benchmark                    v1.5.5  v1.5.6  v1.6.0  v1.6.1  v1.7.0
  ---                          ------  ------  ------  ------  ------
  memo fib (1K)                -       -       0.004s  0.003s  0.003s
  memo collatz sum (10K)       -       -       0.017s  0.017s  0.016s
  memo by .id (100K, 1K keys)  -       -       0.021s  0.020s  0.020s
```
