# Benchmark History

Recent slice (last 5 columns). Full history lives in
[`benchmark-history.tsv`](benchmark-history.tsv) (long format,
`section / benchmark / version / time_seconds`).

```text
--- NDJSON workloads (2M objects) ---
  Benchmark                    v1.6.1  v1.7.0  v1.7.1  v1.8.0  v1.8.1
  ---                          ------  ------  ------  ------  ------
  empty                        0.018s  0.017s  0.017s  0.017s  0.017s
  identity -c                  0.087s  0.086s  0.084s  0.084s  0.083s
  identity (pretty)            0.107s  0.106s  0.102s  0.099s  0.103s
  field access .name           0.097s  0.093s  0.090s  0.089s  0.103s
  nested .x,.y,.name           0.150s  0.147s  0.146s  0.149s  0.171s
  arithmetic .x + .y           0.084s  0.083s  0.082s  0.079s  0.083s
  select .x > 1500000          0.087s  0.084s  0.080s  0.077s  0.082s
  string concat                0.095s  0.092s  0.092s  0.088s  0.090s
  object construct             0.116s  0.114s  0.110s  0.109s  0.124s
  array construct              0.106s  0.105s  0.105s  0.100s  0.123s
  .[]                          0.104s  0.102s  0.101s  0.098s  0.111s
  to_entries                   0.159s  0.163s  0.154s  0.150s  0.169s
  keys                         0.107s  0.105s  0.100s  0.098s  0.101s
  keys_unsorted                0.097s  0.094s  0.092s  0.090s  0.094s
  length                       0.088s  0.084s  0.087s  0.084s  0.086s
  has("x")                     0.039s  0.037s  0.036s  0.035s  0.038s
  type                         0.023s  0.023s  0.022s  0.021s  0.022s
  del(.name)                   0.107s  0.101s  0.099s  0.098s  0.113s
  @csv                         0.120s  0.118s  0.117s  0.119s  0.125s
  split/join                   0.094s  0.090s  0.089s  0.086s  0.089s
  select|field                 0.099s  0.097s  0.094s  0.093s  0.098s
  select|remap                 0.100s  0.099s  0.097s  0.093s  0.097s
  computed remap               0.192s  0.191s  0.187s  0.186s  0.206s
  [.x,.y]|add                  0.084s  0.083s  0.083s  0.081s  0.082s
  [.x,.y]|avg                  0.111s  0.106s  0.106s  0.105s  0.105s
  map(*2)|add                  0.107s  0.104s  0.102s  0.099s  0.104s
  keys|length                  0.259s  0.255s  0.251s  0.249s  0.252s
  .+{z=0}                      0.150s  0.146s  0.145s  0.142s  0.147s
  split|first                  0.092s  0.089s  0.088s  0.086s  0.087s
  slice[0..5]                  0.095s  0.094s  0.094s  0.088s  0.091s
  dynkey {(.name)}             0.108s  0.104s  0.106s  0.100s  0.114s
  .x += 1                      0.128s  0.121s  0.122s  0.124s  0.127s
  {a}+{b} merge                0.133s  0.130s  0.129s  0.128s  0.148s
  .x*2+1                       0.061s  0.060s  0.059s  0.057s  0.059s
  .x+.y*2                      0.099s  0.100s  0.096s  0.094s  0.095s
  .x > .y                      0.078s  0.077s  0.076s  0.074s  0.078s
  to_entries|len               0.405s  0.397s  0.397s  0.389s  0.393s
  .x|.+1 (pipe)                0.059s  0.059s  0.056s  0.055s  0.057s
  .x|.*2|.+1                   0.060s  0.060s  0.058s  0.056s  0.059s
  .name|.+"_x"                 0.097s  0.094s  0.092s  0.088s  0.092s
  .x>N | not                   0.051s  0.049s  0.048s  0.047s  0.050s
  and (2 cmp)                  0.082s  0.080s  0.080s  0.078s  0.081s
  if-then-else                 0.053s  0.052s  0.051s  0.049s  0.052s
  sel(and)|field               0.079s  0.076s  0.076s  0.073s  0.077s
  sel(and)|remap               0.079s  0.077s  0.077s  0.074s  0.078s
  arith|cmp                    0.056s  0.054s  0.053s  0.051s  0.053s
  if cmp .field                0.118s  0.113s  0.110s  0.108s  0.113s
  split|length                 0.093s  0.089s  0.088s  0.085s  0.087s
  [x,y]|min                    0.092s  0.090s  0.089s  0.087s  0.090s
  [x,y]|max                    0.097s  0.095s  0.092s  0.090s  0.094s
  [x,y]|sort|.[0]              0.092s  0.089s  0.088s  0.085s  0.089s
  .name|len>5                  0.094s  0.090s  0.089s  0.088s  0.091s
  sel(len>5)|.x                0.111s  0.107s  0.107s  0.105s  0.106s
  if .x>.y .name               0.092s  0.093s  0.088s  0.088s  0.089s
  sel(.x>.y)|.name             0.074s  0.071s  0.070s  0.067s  0.071s
  .x*2|tostring                0.058s  0.057s  0.055s  0.053s  0.056s
  .x*.x+1                      0.067s  0.067s  0.064s  0.063s  0.066s
  {k=.name,v=tostr}            0.147s  0.144s  0.145s  0.138s  0.157s
  str add chain                0.388s  0.385s  0.378s  0.372s  0.380s
  if>.y .name|empty            0.073s  0.073s  0.072s  0.070s  0.073s
  if .x%2==0                   0.054s  0.055s  0.054s  0.052s  0.054s
  if .x*2+1>1M                 0.055s  0.056s  0.054s  0.052s  0.054s
  sel(.x%2==0)|.name           0.084s  0.084s  0.081s  0.080s  0.084s
  sel(.x*2+1>1M)               0.157s  0.159s  0.154s  0.151s  0.157s
  .x|@json                     0.049s  0.047s  0.047s  0.046s  0.048s
  .x|@text                     0.049s  0.048s  0.047s  0.045s  0.048s
  .name|@json                  0.102s  0.101s  0.101s  0.099s  0.102s
  sel|[arr]                    0.143s  0.149s  0.144s  0.140s  0.160s
  sel(and)|[arr]               0.077s  0.079s  0.076s  0.076s  0.079s
  if>.y [arr]                  0.175s  0.179s  0.174s  0.178s  0.198s
  if sw then .f                0.143s  0.139s  0.135s  0.132s  0.139s
  dynkey {(.n)=.x*2}           0.117s  0.115s  0.112s  0.111s  0.116s
  sel(and)|.x*.y               0.077s  0.077s  0.078s  0.074s  0.079s
  sel>N|str chain              0.152s  0.155s  0.153s  0.151s  0.153s
  .f+"_"+arith_ts              0.134s  0.133s  0.132s  0.130s  0.133s
  sel(sw)|str ch               0.308s  0.307s  0.298s  0.301s  0.303s
  split|rev|join               0.117s  0.117s  0.112s  0.111s  0.112s
  dynkey+static                0.340s  0.338s  0.332s  0.332s  0.339s
  if>.y str chain              0.167s  0.171s  0.168s  0.167s  0.170s
  remap+str chain              0.156s  0.156s  0.154s  0.145s  0.160s
  sel(len>8)                   0.163s  0.162s  0.159s  0.154s  0.159s
  up|split|join                0.098s  0.095s  0.100s  0.095s  0.096s
  .name|index                  0.121s  0.118s  0.117s  0.114s  0.121s
  .name|index+1                0.126s  0.122s  0.122s  0.117s  0.122s
  .name|rindex                 0.131s  0.127s  0.125s  0.122s  0.124s
  .name|indices                0.161s  0.155s  0.149s  0.145s  0.159s
  [x,y]|sort                   0.154s  0.153s  0.154s  0.152s  0.168s
  .name|scan                   0.214s  0.207s  0.213s  0.208s  0.206s
  .name|gsub                   0.170s  0.167s  0.166s  0.162s  0.168s
  walk(if num .+1)             0.140s  0.141s  0.137s  0.136s  0.137s
  tojson                       0.106s  0.104s  0.105s  0.100s  0.107s
  {name,x}                     0.129s  0.128s  0.128s  0.123s  0.150s
  .z//.name                    0.158s  0.154s  0.155s  0.146s  0.164s
  .x|=test(re)                 0.176s  0.172s  0.170s  0.165s  0.170s
  ./sep|first                  0.185s  0.182s  0.180s  0.173s  0.179s
  .y=(.x*2)                    0.180s  0.174s  0.175s  0.172s  0.176s
  .y=(.x+.y)                   0.234s  0.228s  0.226s  0.224s  0.232s
  objects                      0.138s  0.134s  0.137s  0.130s  0.125s
  .tag|=if..then N             0.611s  0.627s  0.614s  0.590s  0.607s
  .x=(.x+1)                    0.127s  0.124s  0.126s  0.123s  0.128s
  sel>N|.y+=1                  0.122s  0.122s  0.120s  0.119s  0.122s
  sel(and)|.x+=1               0.111s  0.109s  0.109s  0.106s  0.106s
  sel(sw)|.x+=1                0.163s  0.159s  0.156s  0.156s  0.164s
  match(re)                    0.367s  0.362s  0.368s  0.355s  0.363s
  capture(re)                  0.299s  0.302s  0.299s  0.294s  0.299s
  first(.name,.x)              0.097s  0.094s  0.095s  0.092s  0.106s
  if .x==null                  0.047s  0.046s  0.046s  0.045s  0.047s
  we(sw(.key))                 0.105s  0.108s  0.107s  0.104s  0.109s
  sel(sw or ew)                0.207s  0.205s  0.199s  0.198s  0.206s
  path(.name,.x)               0.278s  0.278s  0.271s  0.264s  0.273s
  sel(str+num+num)             0.153s  0.150s  0.151s  0.145s  0.150s
  nested if|field              0.077s  0.077s  0.077s  0.074s  0.078s
  .f|floor|.*2                 0.061s  0.060s  0.061s  0.058s  0.062s
  split|len>1                  0.119s  0.115s  0.113s  0.110s  0.117s
  .name|len|.*2                0.103s  0.103s  0.100s  0.097s  0.101s
  if len>5 .x .y               0.113s  0.113s  0.115s  0.110s  0.112s
  sel(len>5)|remap             0.202s  0.201s  0.199s  0.198s  0.199s
  .x|tostr|len                 0.060s  0.059s  0.058s  0.057s  0.060s
  if .x>.y .x .y               0.094s  0.094s  0.094s  0.091s  0.093s
  split|last|tonum             0.097s  0.096s  0.095s  0.092s  0.097s
  split|rev|.[0]               0.091s  0.091s  0.090s  0.089s  0.092s
  split|.[0]+.[1]              0.115s  0.113s  0.112s  0.109s  0.112s
  .[]|strings                  0.106s  0.105s  0.104s  0.106s  0.108s
  .[]|numbers                  0.125s  0.124s  0.124s  0.125s  0.126s
  [x,y]|any(>1M)               0.082s  0.082s  0.083s  0.079s  0.082s
  sel(dc|sw)                   0.101s  0.098s  0.098s  0.094s  0.099s
  [[x,y],[n]]|flat             0.462s  0.464s  0.456s  0.450s  0.459s
  .x|floor|.*2                 0.060s  0.061s  0.061s  0.059s  0.062s
  tojson|fromjson              0.084s  0.082s  0.084s  0.080s  0.086s
  [.x]|add                     0.060s  0.059s  0.059s  0.057s  0.064s
  if>N {o}+.                   0.135s  0.138s  0.136s  0.127s  0.137s
  if>N .+{o}                   0.135s  0.136s  0.132s  0.131s  0.132s
  if .n=="s" .+{o}             0.161s  0.157s  0.159s  0.154s  0.162s
  sel(.n>"s")                  0.090s  0.091s  0.087s  0.085s  0.088s
  [x,y,z]|min                  0.311s  0.307s  0.312s  0.305s  0.313s
  if .n|len>5 l s              0.101s  0.100s  0.099s  0.095s  0.100s
  if .x|flr>N b s              0.055s  0.055s  0.054s  0.053s  0.055s
  if .n|test l e               0.105s  0.105s  0.104s  0.102s  0.109s
  if .n|sw l e                 0.085s  0.082s  0.084s  0.080s  0.085s
  if .n|ew l e                 0.086s  0.085s  0.083s  0.080s  0.084s
  .n|len|tostr                 0.091s  0.090s  0.092s  0.085s  0.090s

--- String operations (2M objects) ---
  Benchmark                    v1.6.1  v1.7.0  v1.7.1  v1.8.0  v1.8.1
  ---                          ------  ------  ------  ------  ------
  ascii_downcase               0.106s  0.104s  0.104s  0.100s  0.104s
  ascii_upcase                 0.106s  0.103s  0.102s  0.099s  0.101s
  ltrimstr                     0.099s  0.097s  0.096s  0.093s  0.096s
  rtrimstr                     0.099s  0.097s  0.097s  0.095s  0.099s
  split                        0.165s  0.162s  0.164s  0.156s  0.166s
  case+split                   0.116s  0.121s  0.116s  0.113s  0.117s
  join                         0.093s  0.093s  0.094s  0.091s  0.092s
  startswith                   0.098s  0.095s  0.094s  0.090s  0.096s
  endswith                     0.098s  0.096s  0.097s  0.091s  0.096s
  tostring                     0.063s  0.063s  0.062s  0.060s  0.073s
  tonumber                     0.111s  0.110s  0.109s  0.108s  0.108s
  string interpolation         0.116s  0.118s  0.119s  0.114s  0.120s

--- String ops (200K objects) ---
  Benchmark                    v1.6.1  v1.7.0  v1.7.1  v1.8.0  v1.8.1
  ---                          ------  ------  ------  ------  ------
  test (regex)                 0.015s  0.014s  0.014s  0.014s  0.014s
  match (regex)                0.033s  0.032s  0.033s  0.033s  0.032s
  @base64                      0.012s  0.012s  0.012s  0.011s  0.012s
  @uri                         0.013s  0.012s  0.012s  0.011s  0.012s
  @html                        0.013s  0.012s  0.012s  0.011s  0.012s
  @csv (array)                 0.016s  0.016s  0.016s  0.015s  0.016s
  @tsv (array)                 0.015s  0.015s  0.015s  0.014s  0.015s
  gsub                         0.019s  0.019s  0.019s  0.018s  0.019s
  case+gsub                    0.181s  0.180s  0.176s  0.177s  0.182s
  case+test                    0.124s  0.120s  0.118s  0.114s  0.122s
  ltrim+tonum+arith            0.113s  0.111s  0.111s  0.107s  0.111s

--- Numeric & math (2M objects) ---
  Benchmark                    v1.6.1  v1.7.0  v1.7.1  v1.8.0  v1.8.1
  ---                          ------  ------  ------  ------  ------
  floor                        0.056s  0.057s  0.056s  0.054s  0.057s
  sqrt                         0.079s  0.079s  0.078s  0.077s  0.080s
  modulo                       0.057s  0.059s  0.057s  0.055s  0.059s
  if-elif-else                 0.125s  0.123s  0.123s  0.123s  0.126s
  select|del                   0.090s  0.091s  0.091s  0.087s  0.097s
  select|merge                 0.118s  0.121s  0.118s  0.114s  0.119s
  select(test)|merge           0.022s  0.021s  0.021s  0.020s  0.022s

--- Array generators ---
  Benchmark                    v1.6.1  v1.7.0  v1.7.1  v1.8.0  v1.8.1
  ---                          ------  ------  ------  ------  ------
  range(2M) | length           0.012s  0.012s  0.011s  0.011s  0.012s
  reverse(2M)                  0.018s  0.018s  0.018s  0.017s  0.018s
  sort(2M)                     0.023s  0.023s  0.023s  0.022s  0.024s
  unique(1M)                   0.030s  0.030s  0.030s  0.029s  0.031s
  flatten(500K)                0.011s  0.011s  0.011s  0.010s  0.011s
  min, max(2M)                 0.021s  0.019s  0.020s  0.017s  0.020s
  add numbers(2M)              0.013s  0.013s  0.013s  0.012s  0.013s
  any/all(2M)                  0.028s  0.029s  0.028s  0.028s  0.028s
  limit(10; range(10M))        0.002s  0.003s  0.002s  0.002s  0.002s
  first(range(10M))            0.002s  0.003s  0.002s  0.002s  0.002s
  last(range(2M))              0.002s  0.003s  0.002s  0.002s  0.002s
  indices(1M)                  0.017s  0.017s  0.017s  0.016s  0.016s

--- Reduce & foreach ---
  Benchmark                    v1.6.1  v1.7.0  v1.7.1  v1.8.0  v1.8.1
  ---                          ------  ------  ------  ------  ------
  reduce (sum)                 0.009s  0.009s  0.009s  0.009s  0.009s
  reduce (array build)         0.004s  0.004s  0.004s  0.004s  0.004s
  reduce (obj build)           0.010s  0.010s  0.009s  0.009s  0.010s
  reduce (setpath)             0.016s  0.016s  0.016s  0.016s  0.017s
  foreach (running sum)        0.010s  0.010s  0.010s  0.009s  0.010s
  foreach + emit               0.011s  0.010s  0.010s  0.010s  0.010s
  reduce (sum-of-squares)      0.034s  0.034s  0.033s  0.032s  0.033s
  reduce (conditional)         0.036s  0.035s  0.035s  0.033s  0.035s
  reduce (product)             0.035s  0.034s  0.034s  0.035s  0.034s
  foreach (conditional)        0.010s  0.010s  0.010s  0.010s  0.010s
  until (100M)                 0.303s  0.305s  0.303s  0.301s  0.301s
  reduce (harmonic)            0.033s  0.033s  0.033s  0.032s  0.032s
  reduce (floor pipe)          0.034s  0.033s  0.033s  0.032s  0.033s
  reduce (sqrt pipe)           0.033s  0.033s  0.033s  0.032s  0.033s
  reduce (sin+cos)             0.052s  0.052s  0.052s  0.051s  0.052s

--- Object operations ---
  Benchmark                    v1.6.1  v1.7.0  v1.7.1  v1.8.0  v1.8.1
  ---                          ------  ------  ------  ------  ------
  large obj construct          0.004s  0.004s  0.004s  0.004s  0.004s
  large obj keys               0.011s  0.011s  0.011s  0.011s  0.011s
  large obj to_entries         0.012s  0.012s  0.012s  0.012s  0.012s
  with_entries                 0.009s  0.009s  0.009s  0.008s  0.009s

--- Assignment operators ---
  Benchmark                    v1.6.1  v1.7.0  v1.7.1  v1.8.0  v1.8.1
  ---                          ------  ------  ------  ------  ------
  .[] |= f (100K)              0.005s  0.005s  0.005s  0.005s  0.005s
  .[] += 1 (100K)              0.005s  0.005s  0.006s  0.005s  0.005s
  .[k] = v reduce(50K)         0.008s  0.008s  0.008s  0.008s  0.008s
  p126-shape nested reduce     -       -       -       0.103s  0.110s
  p126-shape w/ mutate         -       -       -       0.105s  0.112s
  p150-shape serial mutate     -       -       -       0.010s  0.012s

--- String-heavy generators ---
  Benchmark                    v1.6.1  v1.7.0  v1.7.1  v1.8.0  v1.8.1
  ---                          ------  ------  ------  ------  ------
  gsub(100K)                   0.027s  0.027s  0.026s  0.026s  0.027s
  join large(100K)             0.005s  0.005s  0.006s  0.005s  0.006s
  explode/implode(100K)        0.028s  0.027s  0.028s  0.027s  0.028s
  reduce str concat(100K)      0.008s  0.008s  0.008s  0.007s  0.008s

--- Try-catch & alternative ---
  Benchmark                    v1.6.1  v1.7.0  v1.7.1  v1.8.0  v1.8.1
  ---                          ------  ------  ------  ------  ------
  alternative //               0.035s  0.035s  0.035s  0.034s  0.035s
  try-catch                    0.024s  0.024s  0.024s  0.023s  0.023s
  label-break                  0.004s  0.004s  0.004s  0.004s  0.004s

--- Type conversion ---
  Benchmark                    v1.6.1  v1.7.0  v1.7.1  v1.8.0  v1.8.1
  ---                          ------  ------  ------  ------  ------
  tojson/fromjson(100K)        0.022s  0.022s  0.022s  0.022s  0.022s
  null propagation(2M)         0.091s  0.091s  0.093s  0.090s  0.092s

--- jaq-derived ---
  Benchmark                    v1.6.1  v1.7.0  v1.7.1  v1.8.0  v1.8.1
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
  Benchmark                    v1.6.1  v1.7.0  v1.7.1  v1.8.0  v1.8.1
  ---                          ------  ------  ------  ------  ------
  memo fib (1K)                0.003s  0.003s  0.003s  0.003s  0.003s
  memo collatz sum (10K)       0.017s  0.016s  0.016s  0.016s  0.017s
  memo by .id (100K, 1K keys)  0.020s  0.020s  0.020s  0.020s  0.020s
```
