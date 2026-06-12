# Benchmark History

Recent slice (last 5 columns). Full history lives in
[`benchmark-history.tsv`](benchmark-history.tsv) (long format,
`section / benchmark / version / time_seconds`).

```text
--- NDJSON workloads (2M objects) ---
  Benchmark                    v1.8.2  v1.8.3  v1.9.0  v1.9.1  v1.9.2
  ---                          ------  ------  ------  ------  ------
  empty                        0.017s  0.017s  0.018s  0.018s  0.019s
  identity -c                  0.085s  0.085s  0.088s  0.085s  0.087s
  identity (pretty)            0.110s  0.106s  0.109s  0.112s  0.111s
  field access .name           0.110s  0.114s  0.084s  0.081s  0.084s
  nested .x,.y,.name           0.184s  0.200s  0.119s  0.123s  0.121s
  arithmetic .x + .y           0.087s  0.084s  0.088s  0.090s  0.088s
  select .x > 1500000          0.086s  0.081s  0.083s  0.083s  0.084s
  string concat                0.096s  0.093s  0.092s  0.094s  0.096s
  object construct             0.135s  0.137s  0.118s  0.120s  0.121s
  array construct              0.135s  0.136s  0.108s  0.105s  0.106s
  .[]                          0.117s  0.116s  0.120s  0.120s  0.120s
  to_entries                   0.149s  0.148s  0.155s  0.151s  0.150s
  keys                         0.104s  0.103s  0.107s  0.104s  0.104s
  keys_unsorted                0.100s  0.099s  0.101s  0.102s  0.103s
  length                       0.086s  0.084s  0.090s  0.087s  0.092s
  has("x")                     0.037s  0.039s  0.040s  0.039s  0.040s
  type                         0.020s  0.020s  0.021s  0.020s  0.020s
  del(.name)                   0.115s  0.113s  0.117s  0.115s  0.114s
  @csv                         0.119s  0.118s  0.123s  0.117s  0.122s
  split/join                   0.089s  0.088s  0.090s  0.088s  0.089s
  select|field                 0.100s  0.097s  0.097s  0.098s  0.097s
  select|remap                 0.101s  0.102s  0.100s  0.101s  0.099s
  computed remap               0.222s  0.224s  0.206s  0.205s  0.201s
  [.x,.y]|add                  0.086s  0.084s  0.090s  0.091s  0.089s
  [.x,.y]|avg                  0.111s  0.112s  0.112s  0.114s  0.113s
  map(*2)|add                  0.106s  0.102s  0.108s  0.108s  0.108s
  keys|length                  0.263s  0.253s  0.263s  0.260s  0.261s
  .+{z=0}                      0.154s  0.150s  0.150s  0.152s  0.158s
  split|first                  0.086s  0.086s  0.088s  0.090s  0.088s
  slice[0..5]                  0.093s  0.091s  0.091s  0.093s  0.095s
  dynkey {(.name)}             0.113s  0.117s  0.115s  0.115s  0.115s
  .x += 1                      0.128s  0.128s  0.133s  0.135s  0.133s
  {a}+{b} merge                0.161s  0.166s  0.132s  0.131s  0.137s
  .x*2+1                       0.061s  0.062s  0.062s  0.062s  0.062s
  .x+.y*2                      0.100s  0.100s  0.100s  0.101s  0.101s
  .x > .y                      0.081s  0.078s  0.080s  0.082s  0.082s
  to_entries|len               0.387s  0.387s  0.392s  0.388s  0.389s
  .x|.+1 (pipe)                0.058s  0.058s  0.058s  0.058s  0.058s
  .x|.*2|.+1                   0.061s  0.061s  0.062s  0.062s  0.062s
  .name|.+"_x"                 0.095s  0.093s  0.093s  0.095s  0.094s
  .x>N | not                   0.049s  0.051s  0.052s  0.051s  0.052s
  and (2 cmp)                  0.084s  0.085s  0.085s  0.087s  0.084s
  if-then-else                 0.053s  0.053s  0.054s  0.053s  0.052s
  sel(and)|field               0.080s  0.079s  0.080s  0.083s  0.086s
  sel(and)|remap               0.080s  0.080s  0.083s  0.085s  0.084s
  arith|cmp                    0.055s  0.054s  0.057s  0.055s  0.056s
  if cmp .field                0.119s  0.122s  0.115s  0.116s  0.113s
  split|length                 0.089s  0.086s  0.090s  0.087s  0.089s
  [x,y]|min                    0.095s  0.093s  0.098s  0.098s  0.096s
  [x,y]|max                    0.100s  0.097s  0.099s  0.102s  0.099s
  [x,y]|sort|.[0]              0.095s  0.093s  0.099s  0.097s  0.094s
  .name|len>5                  0.087s  0.085s  0.088s  0.087s  0.086s
  sel(len>5)|.x                0.110s  0.108s  0.112s  0.109s  0.113s
  if .x>.y .name               0.092s  0.092s  0.094s  0.098s  0.092s
  sel(.x>.y)|.name             0.075s  0.071s  0.076s  0.076s  0.077s
  .x*2|tostring                0.059s  0.057s  0.057s  0.060s  0.058s
  .x*.x+1                      0.068s  0.067s  0.067s  0.068s  0.069s
  {k=.name,v=tostr}            0.167s  0.170s  0.144s  0.143s  0.145s
  str add chain                0.396s  0.385s  0.403s  0.396s  0.395s
  if>.y .name|empty            0.075s  0.074s  0.078s  0.078s  0.077s
  if .x%2==0                   0.055s  0.054s  0.057s  0.055s  0.055s
  if .x*2+1>1M                 0.056s  0.057s  0.057s  0.057s  0.056s
  sel(.x%2==0)|.name           0.083s  0.084s  0.084s  0.086s  0.085s
  sel(.x*2+1>1M)               0.156s  0.153s  0.161s  0.160s  0.162s
  .x|@json                     0.051s  0.060s  0.061s  0.062s  0.061s
  .x|@text                     0.050s  0.060s  0.062s  0.059s  0.062s
  .name|@json                  0.107s  0.104s  0.105s  0.100s  0.103s
  sel|[arr]                    0.172s  0.180s  0.184s  0.173s  0.175s
  sel(and)|[arr]               0.081s  0.080s  0.084s  0.080s  0.080s
  if>.y [arr]                  0.205s  0.217s  0.221s  0.216s  0.214s
  if sw then .f                0.147s  0.142s  0.144s  0.141s  0.143s
  dynkey {(.n)=.x*2}           0.118s  0.119s  0.117s  0.113s  0.117s
  sel(and)|.x*.y               0.080s  0.079s  0.082s  0.084s  0.080s
  sel>N|str chain              0.157s  0.166s  0.167s  0.161s  0.166s
  .f+"_"+arith_ts              0.133s  0.137s  0.137s  0.135s  0.140s
  sel(sw)|str ch               0.312s  0.309s  0.317s  0.309s  0.320s
  split|rev|join               0.114s  0.116s  0.115s  0.117s  0.118s
  dynkey+static                0.340s  0.351s  0.343s  0.348s  0.344s
  if>.y str chain              0.171s  0.173s  0.171s  0.173s  0.172s
  remap+str chain              0.162s  0.178s  0.165s  0.167s  0.163s
  sel(len>8)                   0.161s  0.161s  0.163s  0.159s  0.163s
  up|split|join                0.097s  0.096s  0.097s  0.097s  0.098s
  .name|index                  0.120s  0.118s  0.119s  0.121s  0.121s
  .name|index+1                0.126s  0.121s  0.128s  0.127s  0.127s
  .name|rindex                 0.129s  0.129s  0.127s  0.128s  0.130s
  .name|indices                0.150s  0.147s  0.151s  0.148s  0.151s
  [x,y]|sort                   0.175s  0.178s  0.178s  0.179s  0.183s
  .name|scan                   0.205s  0.204s  0.204s  0.203s  0.210s
  .name|gsub                   0.167s  0.164s  0.168s  0.166s  0.166s
  walk(if num .+1)             0.141s  0.142s  0.141s  0.142s  0.143s
  tojson                       0.115s  0.117s  0.119s  0.118s  0.119s
  {name,x}                     0.154s  0.166s  0.127s  0.135s  0.131s
  .z//.name                    0.175s  0.174s  0.177s  0.175s  0.179s
  .x|=test(re)                 0.172s  0.174s  0.173s  0.171s  0.174s
  ./sep|first                  0.186s  0.180s  0.185s  0.189s  0.188s
  .y=(.x*2)                    0.179s  0.179s  0.183s  0.180s  0.184s
  .y=(.x+.y)                   0.235s  0.237s  0.234s  0.235s  0.238s
  objects                      0.137s  0.146s  0.150s  0.145s  0.140s
  .tag|=if..then N             0.635s  0.624s  0.613s  0.615s  0.612s
  .x=(.x+1)                    0.131s  0.127s  0.133s  0.131s  0.138s
  sel>N|.y+=1                  0.124s  0.125s  0.127s  0.127s  0.129s
  sel(and)|.x+=1               0.108s  0.108s  0.109s  0.107s  0.108s
  sel(sw)|.x+=1                0.169s  0.165s  0.169s  0.165s  0.166s
  match(re)                    0.379s  0.372s  0.382s  0.369s  0.376s
  capture(re)                  0.306s  0.303s  0.301s  0.294s  0.305s
  first(.name,.x)              0.109s  0.116s  0.083s  0.082s  0.081s
  if .x==null                  0.047s  0.048s  0.049s  0.050s  0.048s
  we(sw(.key))                 0.118s  0.117s  0.118s  0.120s  0.120s
  sel(sw or ew)                0.205s  0.199s  0.208s  0.203s  0.210s
  path(.name,.x)               0.320s  0.300s  0.308s  0.311s  0.307s
  sel(str+num+num)             0.151s  0.154s  0.155s  0.154s  0.152s
  nested if|field              0.083s  0.079s  0.084s  0.083s  0.082s
  .f|floor|.*2                 0.062s  0.062s  0.062s  0.063s  0.063s
  split|len>1                  0.114s  0.110s  0.110s  0.111s  0.113s
  .name|len|.*2                0.102s  0.102s  0.104s  0.100s  0.103s
  if len>5 .x .y               0.115s  0.114s  0.114s  0.113s  0.112s
  sel(len>5)|remap             0.209s  0.203s  0.204s  0.205s  0.206s
  .x|tostr|len                 0.061s  0.059s  0.061s  0.061s  0.061s
  if .x>.y .x .y               0.099s  0.096s  0.097s  0.104s  0.099s
  split|last|tonum             0.095s  0.094s  0.096s  0.095s  0.093s
  split|rev|.[0]               0.091s  0.091s  0.094s  0.089s  0.092s
  split|.[0]+.[1]              0.114s  0.111s  0.112s  0.113s  0.113s
  .[]|strings                  0.108s  0.105s  0.110s  0.103s  0.106s
  .[]|numbers                  0.125s  0.124s  0.127s  0.125s  0.127s
  [x,y]|any(>1M)               0.086s  0.084s  0.087s  0.088s  0.088s
  sel(dc|sw)                   0.098s  0.095s  0.095s  0.097s  0.097s
  [[x,y],[n]]|flat             0.520s  0.488s  0.460s  0.460s  0.459s
  .x|floor|.*2                 0.060s  0.062s  0.064s  0.063s  0.062s
  tojson|fromjson              0.086s  0.084s  0.085s  0.087s  0.089s
  [.x]|add                     0.066s  0.071s  0.048s  0.049s  0.050s
  if>N {o}+.                   0.135s  0.137s  0.137s  0.136s  0.138s
  if>N .+{o}                   0.134s  0.133s  0.132s  0.137s  0.142s
  if .n=="s" .+{o}             0.162s  0.159s  0.160s  0.158s  0.157s
  sel(.n>"s")                  0.092s  0.088s  0.089s  0.087s  0.087s
  [x,y,z]|min                  0.311s  0.313s  0.315s  0.318s  0.316s
  if .n|len>5 l s              0.101s  0.101s  0.101s  0.100s  0.100s
  if .x|flr>N b s              0.055s  0.056s  0.056s  0.056s  0.056s
  if .n|test l e               0.103s  0.104s  0.105s  0.106s  0.101s
  if .n|sw l e                 0.086s  0.083s  0.086s  0.086s  0.083s
  if .n|ew l e                 0.083s  0.084s  0.086s  0.084s  0.085s
  .n|len|tostr                 0.089s  0.089s  0.092s  0.091s  0.092s

--- String operations (2M objects) ---
  Benchmark                    v1.8.2  v1.8.3  v1.9.0  v1.9.1  v1.9.2
  ---                          ------  ------  ------  ------  ------
  ascii_downcase               0.095s  0.092s  0.095s  0.094s  0.095s
  ascii_upcase                 0.094s  0.094s  0.094s  0.096s  0.095s
  ltrimstr                     0.097s  0.095s  0.095s  0.098s  0.095s
  rtrimstr                     0.095s  0.097s  0.094s  0.098s  0.096s
  split                        0.158s  0.157s  0.159s  0.162s  0.164s
  case+split                   0.116s  0.116s  0.114s  0.117s  0.120s
  join                         0.094s  0.096s  0.092s  0.097s  0.096s
  startswith                   0.090s  0.090s  0.090s  0.091s  0.090s
  endswith                     0.090s  0.088s  0.093s  0.091s  0.089s
  tostring                     0.068s  0.064s  0.065s  0.066s  0.065s
  tonumber                     0.110s  0.108s  0.113s  0.111s  0.113s
  string interpolation         0.115s  0.127s  0.126s  0.124s  0.128s

--- String ops (200K objects) ---
  Benchmark                    v1.8.2  v1.8.3  v1.9.0  v1.9.1  v1.9.2
  ---                          ------  ------  ------  ------  ------
  test (regex)                 0.015s  0.014s  0.015s  0.015s  0.015s
  match (regex)                0.033s  0.033s  0.033s  0.033s  0.033s
  @base64                      0.012s  0.011s  0.012s  0.012s  0.012s
  @uri                         0.013s  0.011s  0.013s  0.012s  0.012s
  @html                        0.013s  0.012s  0.013s  0.012s  0.012s
  @csv (array)                 0.015s  0.016s  0.016s  0.016s  0.016s
  @tsv (array)                 0.015s  0.015s  0.016s  0.017s  0.017s
  gsub                         0.018s  0.018s  0.019s  0.019s  0.019s
  case+gsub                    0.179s  0.181s  0.178s  0.181s  0.180s
  case+test                    0.117s  0.118s  0.119s  0.117s  0.119s
  ltrim+tonum+arith            0.110s  0.110s  0.116s  0.112s  0.111s

--- Numeric & math (2M objects) ---
  Benchmark                    v1.8.2  v1.8.3  v1.9.0  v1.9.1  v1.9.2
  ---                          ------  ------  ------  ------  ------
  floor                        0.056s  0.058s  0.058s  0.060s  0.059s
  sqrt                         0.081s  0.079s  0.081s  0.080s  0.079s
  modulo                       0.058s  0.059s  0.059s  0.060s  0.059s
  if-elif-else                 0.130s  0.131s  0.127s  0.127s  0.130s
  select|del                   0.097s  0.096s  0.100s  0.099s  0.098s
  select|merge                 0.123s  0.121s  0.121s  0.121s  0.123s
  select(test)|merge           0.021s  0.021s  0.023s  0.022s  0.022s

--- Array generators ---
  Benchmark                    v1.8.2  v1.8.3  v1.9.0  v1.9.1  v1.9.2
  ---                          ------  ------  ------  ------  ------
  range(2M) | length           0.012s  0.011s  0.009s  0.009s  0.009s
  reverse(2M)                  0.018s  0.018s  0.016s  0.016s  0.018s
  sort(2M)                     0.023s  0.024s  0.025s  0.026s  0.027s
  unique(1M)                   0.033s  0.033s  0.034s  0.034s  0.035s
  flatten(500K)                0.011s  0.010s  0.010s  0.010s  0.010s
  min, max(2M)                 0.021s  0.020s  0.016s  0.017s  0.017s
  add numbers(2M)              0.013s  0.013s  0.011s  0.011s  0.010s
  any/all(2M)                  0.029s  0.028s  0.031s  0.032s  0.031s
  limit(10; range(10M))        0.002s  0.002s  0.003s  0.003s  0.003s
  first(range(10M))            0.002s  0.002s  0.003s  0.002s  0.003s
  last(range(2M))              0.002s  0.002s  0.003s  0.002s  0.003s
  indices(1M)                  0.017s  0.017s  0.018s  0.017s  0.018s

--- Reduce & foreach ---
  Benchmark                    v1.8.2  v1.8.3  v1.9.0  v1.9.1  v1.9.2
  ---                          ------  ------  ------  ------  ------
  reduce (sum)                 0.009s  0.009s  0.010s  0.010s  0.010s
  reduce (array build)         0.004s  0.004s  0.005s  0.005s  0.005s
  reduce (obj build)           0.009s  0.010s  0.010s  0.010s  0.010s
  reduce (setpath)             0.016s  0.016s  0.018s  0.017s  0.017s
  foreach (running sum)        0.010s  0.010s  0.011s  0.011s  0.011s
  foreach + emit               0.010s  0.010s  0.011s  0.011s  0.011s
  reduce (sum-of-squares)      0.033s  0.033s  0.036s  0.034s  0.036s
  reduce (conditional)         0.036s  0.035s  0.037s  0.037s  0.037s
  reduce (product)             0.035s  0.034s  0.036s  0.035s  0.036s
  foreach (conditional)        0.011s  0.010s  0.011s  0.011s  0.011s
  until (100M)                 0.308s  0.312s  0.322s  0.325s  0.324s
  reduce (harmonic)            0.033s  0.035s  0.036s  0.035s  0.035s
  reduce (floor pipe)          0.033s  0.033s  0.035s  0.034s  0.035s
  reduce (sqrt pipe)           0.033s  0.034s  0.035s  0.035s  0.035s
  reduce (sin+cos)             0.052s  0.052s  0.053s  0.053s  0.052s

--- Object operations ---
  Benchmark                    v1.8.2  v1.8.3  v1.9.0  v1.9.1  v1.9.2
  ---                          ------  ------  ------  ------  ------
  large obj construct          0.004s  0.004s  0.004s  0.004s  0.004s
  large obj keys               0.011s  0.011s  0.011s  0.011s  0.011s
  large obj to_entries         0.012s  0.012s  0.012s  0.012s  0.013s
  with_entries                 0.009s  0.009s  0.009s  0.010s  0.010s

--- Assignment operators ---
  Benchmark                    v1.8.2  v1.8.3  v1.9.0  v1.9.1  v1.9.2
  ---                          ------  ------  ------  ------  ------
  .[] |= f (100K)              0.005s  0.005s  0.005s  0.006s  0.005s
  .[] += 1 (100K)              0.005s  0.005s  0.006s  0.006s  0.006s
  .[k] = v reduce(50K)         0.008s  0.008s  0.009s  0.009s  0.009s
  p126-shape nested reduce     0.120s  0.121s  0.118s  0.120s  0.119s
  p126-shape w/ mutate         0.129s  0.124s  0.124s  0.120s  0.132s
  p150-shape serial mutate     0.011s  0.012s  0.013s  0.012s  0.013s

--- String-heavy generators ---
  Benchmark                    v1.8.2  v1.8.3  v1.9.0  v1.9.1  v1.9.2
  ---                          ------  ------  ------  ------  ------
  gsub(100K)                   0.029s  0.028s  0.029s  0.029s  0.029s
  join large(100K)             0.006s  0.006s  0.006s  0.006s  0.006s
  explode/implode(100K)        0.028s  0.027s  0.029s  0.028s  0.028s
  reduce str concat(100K)      0.008s  0.008s  0.008s  0.008s  0.008s

--- Try-catch & alternative ---
  Benchmark                    v1.8.2  v1.8.3  v1.9.0  v1.9.1  v1.9.2
  ---                          ------  ------  ------  ------  ------
  alternative //               0.032s  0.033s  0.034s  0.034s  0.034s
  try-catch                    0.024s  0.024s  0.024s  0.024s  0.025s
  label-break                  0.004s  0.004s  0.004s  0.004s  0.004s

--- Type conversion ---
  Benchmark                    v1.8.2  v1.8.3  v1.9.0  v1.9.1  v1.9.2
  ---                          ------  ------  ------  ------  ------
  tojson/fromjson(100K)        0.022s  0.022s  0.022s  0.023s  0.022s
  null propagation(2M)         0.092s  0.090s  0.091s  0.090s  0.090s

--- jaq-derived ---
  Benchmark                    v1.8.2  v1.8.3  v1.9.0  v1.9.1  v1.9.2
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
  Benchmark                    v1.8.2  v1.8.3  v1.9.0  v1.9.1  v1.9.2
  ---                          ------  ------  ------  ------  ------
  memo fib (1K)                0.003s  0.003s  0.003s  0.003s  0.003s
  memo collatz sum (10K)       0.016s  0.014s  0.015s  0.015s  0.015s
  memo by .id (100K, 1K keys)  0.020s  0.020s  0.020s  0.021s  0.021s
```
