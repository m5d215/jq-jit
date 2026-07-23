# Benchmark History

Recent slice (last 5 columns). Full history lives in
[`benchmark-history.tsv`](benchmark-history.tsv) (long format,
`section / benchmark / version / time_seconds`).

```text
--- NDJSON workloads (2M objects) ---
  Benchmark                    v1.9.0  v1.9.1  v1.9.2  v1.10.0  v1.11.0
  ---                          ------  ------  ------  -------  -------
  empty                        0.018s  0.018s  0.019s  0.019s   0.017s
  identity -c                  0.088s  0.085s  0.087s  0.088s   0.089s
  identity (pretty)            0.109s  0.112s  0.111s  0.109s   0.104s
  field access .name           0.084s  0.081s  0.084s  0.085s   0.081s
  nested .x,.y,.name           0.119s  0.123s  0.121s  0.121s   0.119s
  arithmetic .x + .y           0.088s  0.090s  0.088s  0.091s   0.083s
  select .x > 1500000          0.083s  0.083s  0.084s  0.088s   0.082s
  string concat                0.092s  0.094s  0.096s  0.095s   0.092s
  object construct             0.118s  0.120s  0.121s  0.124s   0.117s
  array construct              0.108s  0.105s  0.106s  0.111s   0.103s
  .[]                          0.120s  0.120s  0.120s  0.121s   0.118s
  to_entries                   0.155s  0.151s  0.150s  0.151s   0.148s
  keys                         0.107s  0.104s  0.104s  0.109s   0.104s
  keys_unsorted                0.101s  0.102s  0.103s  0.104s   0.100s
  length                       0.090s  0.087s  0.092s  0.086s   0.085s
  has("x")                     0.040s  0.039s  0.040s  0.040s   0.037s
  type                         0.021s  0.020s  0.020s  0.021s   0.019s
  del(.name)                   0.117s  0.115s  0.114s  0.112s   0.101s
  @csv                         0.123s  0.117s  0.122s  0.118s   0.116s
  split/join                   0.090s  0.088s  0.089s  0.094s   0.088s
  select|field                 0.097s  0.098s  0.097s  0.099s   0.099s
  select|remap                 0.100s  0.101s  0.099s  0.107s   0.096s
  computed remap               0.206s  0.205s  0.201s  0.201s   0.201s
  [.x,.y]|add                  0.090s  0.091s  0.089s  0.090s   0.086s
  [.x,.y]|avg                  0.112s  0.114s  0.113s  0.115s   0.110s
  map(*2)|add                  0.108s  0.108s  0.108s  0.107s   0.102s
  keys|length                  0.263s  0.260s  0.261s  0.263s   0.260s
  .+{z=0}                      0.150s  0.152s  0.158s  0.159s   0.156s
  split|first                  0.088s  0.090s  0.088s  0.089s   0.086s
  slice[0..5]                  0.091s  0.093s  0.095s  0.096s   0.089s
  dynkey {(.name)}             0.115s  0.115s  0.115s  0.114s   0.111s
  .x += 1                      0.133s  0.135s  0.133s  0.138s   0.129s
  {a}+{b} merge                0.132s  0.131s  0.137s  0.132s   0.130s
  .x*2+1                       0.062s  0.062s  0.062s  0.064s   0.059s
  .x+.y*2                      0.100s  0.101s  0.101s  0.103s   0.099s
  .x > .y                      0.080s  0.082s  0.082s  0.083s   0.078s
  to_entries|len               0.392s  0.388s  0.389s  0.409s   0.395s
  .x|.+1 (pipe)                0.058s  0.058s  0.058s  0.058s   0.057s
  .x|.*2|.+1                   0.062s  0.062s  0.062s  0.062s   0.060s
  .name|.+"_x"                 0.093s  0.095s  0.094s  0.094s   0.093s
  .x>N | not                   0.052s  0.051s  0.052s  0.054s   0.050s
  and (2 cmp)                  0.085s  0.087s  0.084s  0.086s   0.083s
  if-then-else                 0.054s  0.053s  0.052s  0.055s   0.052s
  sel(and)|field               0.080s  0.083s  0.086s  0.082s   0.079s
  sel(and)|remap               0.083s  0.085s  0.084s  0.085s   0.079s
  arith|cmp                    0.057s  0.055s  0.056s  0.057s   0.055s
  if cmp .field                0.115s  0.116s  0.113s  0.118s   0.111s
  split|length                 0.090s  0.087s  0.089s  0.089s   0.086s
  [x,y]|min                    0.098s  0.098s  0.096s  0.095s   0.091s
  [x,y]|max                    0.099s  0.102s  0.099s  0.101s   0.099s
  [x,y]|sort|.[0]              0.099s  0.097s  0.094s  0.098s   0.092s
  .name|len>5                  0.088s  0.087s  0.086s  0.092s   0.084s
  sel(len>5)|.x                0.112s  0.109s  0.113s  0.116s   0.111s
  if .x>.y .name               0.094s  0.098s  0.092s  0.095s   0.090s
  sel(.x>.y)|.name             0.076s  0.076s  0.077s  0.076s   0.073s
  .x*2|tostring                0.057s  0.060s  0.058s  0.059s   0.058s
  .x*.x+1                      0.067s  0.068s  0.069s  0.070s   0.066s
  {k=.name,v=tostr}            0.144s  0.143s  0.145s  0.147s   0.140s
  str add chain                0.403s  0.396s  0.395s  0.397s   0.385s
  if>.y .name|empty            0.078s  0.078s  0.077s  0.077s   0.076s
  if .x%2==0                   0.057s  0.055s  0.055s  0.057s   0.053s
  if .x*2+1>1M                 0.057s  0.057s  0.056s  0.058s   0.057s
  sel(.x%2==0)|.name           0.084s  0.086s  0.085s  0.086s   0.082s
  sel(.x*2+1>1M)               0.161s  0.160s  0.162s  0.167s   0.157s
  .x|@json                     0.061s  0.062s  0.061s  0.051s   0.048s
  .x|@text                     0.062s  0.059s  0.062s  0.052s   0.048s
  .name|@json                  0.105s  0.100s  0.103s  0.101s   0.093s
  sel|[arr]                    0.184s  0.173s  0.175s  0.184s   0.168s
  sel(and)|[arr]               0.084s  0.080s  0.080s  0.084s   0.080s
  if>.y [arr]                  0.221s  0.216s  0.214s  0.216s   0.207s
  if sw then .f                0.144s  0.141s  0.143s  0.145s   0.134s
  dynkey {(.n)=.x*2}           0.117s  0.113s  0.117s  0.114s   0.109s
  sel(and)|.x*.y               0.082s  0.084s  0.080s  0.083s   0.077s
  sel>N|str chain              0.167s  0.161s  0.166s  0.167s   0.158s
  .f+"_"+arith_ts              0.137s  0.135s  0.140s  0.139s   0.133s
  sel(sw)|str ch               0.317s  0.309s  0.320s  0.326s   0.303s
  split|rev|join               0.115s  0.117s  0.118s  0.122s   0.109s
  dynkey+static                0.343s  0.348s  0.344s  0.360s   0.303s
  if>.y str chain              0.171s  0.173s  0.172s  0.181s   0.172s
  remap+str chain              0.165s  0.167s  0.163s  0.175s   0.161s
  sel(len>8)                   0.163s  0.159s  0.163s  0.168s   0.155s
  up|split|join                0.097s  0.097s  0.098s  0.098s   0.092s
  .name|index                  0.119s  0.121s  0.121s  0.125s   0.117s
  .name|index+1                0.128s  0.127s  0.127s  0.129s   0.120s
  .name|rindex                 0.127s  0.128s  0.130s  0.132s   0.122s
  .name|indices                0.151s  0.148s  0.151s  0.151s   0.143s
  [x,y]|sort                   0.178s  0.179s  0.183s  0.179s   0.175s
  .name|scan                   0.204s  0.203s  0.210s  0.205s   0.202s
  .name|gsub                   0.168s  0.166s  0.166s  0.166s   0.166s
  walk(if num .+1)             0.141s  0.142s  0.143s  0.143s   0.138s
  tojson                       0.119s  0.118s  0.119s  0.126s   0.122s
  {name,x}                     0.127s  0.135s  0.131s  0.132s   0.125s
  .z//.name                    0.177s  0.175s  0.179s  0.184s   0.173s
  .x|=test(re)                 0.173s  0.171s  0.174s  0.175s   0.170s
  ./sep|first                  0.185s  0.189s  0.188s  0.192s   0.182s
  .y=(.x*2)                    0.183s  0.180s  0.184s  0.187s   0.173s
  .y=(.x+.y)                   0.234s  0.235s  0.238s  0.244s   0.225s
  objects                      0.150s  0.145s  0.140s  0.152s   0.133s
  .tag|=if..then N             0.613s  0.615s  0.612s  0.634s   0.602s
  .x=(.x+1)                    0.133s  0.131s  0.138s  0.140s   0.129s
  sel>N|.y+=1                  0.127s  0.127s  0.129s  0.130s   0.121s
  sel(and)|.x+=1               0.109s  0.107s  0.108s  0.108s   0.105s
  sel(sw)|.x+=1                0.169s  0.165s  0.166s  0.167s   0.159s
  match(re)                    0.382s  0.369s  0.376s  0.378s   0.371s
  capture(re)                  0.301s  0.294s  0.305s  0.315s   0.293s
  first(.name,.x)              0.083s  0.082s  0.081s  0.085s   0.080s
  if .x==null                  0.049s  0.050s  0.048s  0.049s   0.047s
  we(sw(.key))                 0.118s  0.120s  0.120s  0.120s   0.114s
  sel(sw or ew)                0.208s  0.203s  0.210s  0.212s   0.197s
  path(.name,.x)               0.308s  0.311s  0.307s  0.354s   0.313s
  sel(str+num+num)             0.155s  0.154s  0.152s  0.162s   0.147s
  nested if|field              0.084s  0.083s  0.082s  0.084s   0.078s
  .f|floor|.*2                 0.062s  0.063s  0.063s  0.065s   0.062s
  split|len>1                  0.110s  0.111s  0.113s  0.116s   0.110s
  .name|len|.*2                0.104s  0.100s  0.103s  0.105s   0.097s
  if len>5 .x .y               0.114s  0.113s  0.112s  0.121s   0.109s
  sel(len>5)|remap             0.204s  0.205s  0.206s  0.216s   0.194s
  .x|tostr|len                 0.061s  0.061s  0.061s  0.063s   0.059s
  if .x>.y .x .y               0.097s  0.104s  0.099s  0.101s   0.092s
  split|last|tonum             0.096s  0.095s  0.093s  0.099s   0.091s
  split|rev|.[0]               0.094s  0.089s  0.092s  0.094s   0.087s
  split|.[0]+.[1]              0.112s  0.113s  0.113s  0.116s   0.109s
  .[]|strings                  0.110s  0.103s  0.106s  0.109s   0.104s
  .[]|numbers                  0.127s  0.125s  0.127s  0.130s   0.122s
  [x,y]|any(>1M)               0.087s  0.088s  0.088s  0.092s   0.083s
  sel(dc|sw)                   0.095s  0.097s  0.097s  0.100s   0.092s
  [[x,y],[n]]|flat             0.460s  0.460s  0.459s  0.551s   0.489s
  .x|floor|.*2                 0.064s  0.063s  0.062s  0.063s   0.060s
  tojson|fromjson              0.085s  0.087s  0.089s  0.090s   0.083s
  [.x]|add                     0.048s  0.049s  0.050s  0.051s   0.047s
  if>N {o}+.                   0.137s  0.136s  0.138s  0.144s   0.134s
  if>N .+{o}                   0.132s  0.137s  0.142s  0.143s   0.135s
  if .n=="s" .+{o}             0.160s  0.158s  0.157s  0.168s   0.160s
  sel(.n>"s")                  0.089s  0.087s  0.087s  0.091s   0.084s
  [x,y,z]|min                  0.315s  0.318s  0.316s  0.305s   0.300s
  if .n|len>5 l s              0.101s  0.100s  0.100s  0.104s   0.095s
  if .x|flr>N b s              0.056s  0.056s  0.056s  0.058s   0.054s
  if .n|test l e               0.105s  0.106s  0.101s  0.108s   0.102s
  if .n|sw l e                 0.086s  0.086s  0.083s  0.087s   0.080s
  if .n|ew l e                 0.086s  0.084s  0.085s  0.088s   0.082s
  .n|len|tostr                 0.092s  0.091s  0.092s  0.090s   0.088s

--- String operations (2M objects) ---
  Benchmark                    v1.9.0  v1.9.1  v1.9.2  v1.10.0  v1.11.0
  ---                          ------  ------  ------  -------  -------
  ascii_downcase               0.095s  0.094s  0.095s  0.095s   0.091s
  ascii_upcase                 0.094s  0.096s  0.095s  0.095s   0.090s
  ltrimstr                     0.095s  0.098s  0.095s  0.099s   0.092s
  rtrimstr                     0.094s  0.098s  0.096s  0.098s   0.093s
  split                        0.159s  0.162s  0.164s  0.161s   0.151s
  case+split                   0.114s  0.117s  0.120s  0.120s   0.110s
  join                         0.092s  0.097s  0.096s  0.097s   0.090s
  startswith                   0.090s  0.091s  0.090s  0.089s   0.085s
  endswith                     0.093s  0.091s  0.089s  0.092s   0.086s
  tostring                     0.065s  0.066s  0.065s  0.067s   0.064s
  tonumber                     0.113s  0.111s  0.113s  0.113s   0.105s
  string interpolation         0.126s  0.124s  0.128s  0.123s   0.108s

--- String ops (200K objects) ---
  Benchmark                    v1.9.0  v1.9.1  v1.9.2  v1.10.0  v1.11.0
  ---                          ------  ------  ------  -------  -------
  test (regex)                 0.015s  0.015s  0.015s  0.015s   0.014s
  match (regex)                0.033s  0.033s  0.033s  0.035s   0.032s
  @base64                      0.012s  0.012s  0.012s  0.012s   0.011s
  @uri                         0.013s  0.012s  0.012s  0.012s   0.011s
  @html                        0.013s  0.012s  0.012s  0.013s   0.010s
  @csv (array)                 0.016s  0.016s  0.016s  0.017s   0.015s
  @tsv (array)                 0.016s  0.017s  0.017s  0.018s   0.015s
  gsub                         0.019s  0.019s  0.019s  0.020s   0.018s
  case+gsub                    0.178s  0.181s  0.180s  0.184s   0.176s
  case+test                    0.119s  0.117s  0.119s  0.124s   0.116s
  ltrim+tonum+arith            0.116s  0.112s  0.111s  0.119s   0.107s

--- Numeric & math (2M objects) ---
  Benchmark                    v1.9.0  v1.9.1  v1.9.2  v1.10.0  v1.11.0
  ---                          ------  ------  ------  -------  -------
  floor                        0.058s  0.060s  0.059s  0.062s   0.057s
  sqrt                         0.081s  0.080s  0.079s  0.084s   0.078s
  modulo                       0.059s  0.060s  0.059s  0.062s   0.058s
  if-elif-else                 0.127s  0.127s  0.130s  0.136s   0.127s
  select|del                   0.100s  0.099s  0.098s  0.104s   0.089s
  select|merge                 0.121s  0.121s  0.123s  0.128s   0.120s
  select(test)|merge           0.023s  0.022s  0.022s  0.023s   0.021s

--- Array generators ---
  Benchmark                    v1.9.0  v1.9.1  v1.9.2  v1.10.0  v1.11.0
  ---                          ------  ------  ------  -------  -------
  range(2M) | length           0.009s  0.009s  0.009s  0.009s   0.008s
  reverse(2M)                  0.016s  0.016s  0.018s  0.017s   0.015s
  sort(2M)                     0.025s  0.026s  0.027s  0.026s   0.024s
  unique(1M)                   0.034s  0.034s  0.035s  0.034s   0.034s
  flatten(500K)                0.010s  0.010s  0.010s  0.010s   0.010s
  min, max(2M)                 0.016s  0.017s  0.017s  0.016s   0.017s
  add numbers(2M)              0.011s  0.011s  0.010s  0.011s   0.009s
  any/all(2M)                  0.031s  0.032s  0.031s  0.031s   0.030s
  limit(10; range(10M))        0.003s  0.003s  0.003s  0.003s   0.002s
  first(range(10M))            0.003s  0.002s  0.003s  0.003s   0.002s
  last(range(2M))              0.003s  0.002s  0.003s  0.003s   0.002s
  indices(1M)                  0.018s  0.017s  0.018s  0.018s   0.017s

--- Reduce & foreach ---
  Benchmark                    v1.9.0  v1.9.1  v1.9.2  v1.10.0  v1.11.0
  ---                          ------  ------  ------  -------  -------
  reduce (sum)                 0.010s  0.010s  0.010s  0.010s   0.009s
  reduce (array build)         0.005s  0.005s  0.005s  0.005s   0.004s
  reduce (obj build)           0.010s  0.010s  0.010s  0.011s   0.009s
  reduce (setpath)             0.018s  0.017s  0.017s  0.019s   0.017s
  foreach (running sum)        0.011s  0.011s  0.011s  0.011s   0.010s
  foreach + emit               0.011s  0.011s  0.011s  0.011s   0.010s
  reduce (sum-of-squares)      0.036s  0.034s  0.036s  0.037s   0.032s
  reduce (conditional)         0.037s  0.037s  0.037s  0.038s   0.035s
  reduce (product)             0.036s  0.035s  0.036s  0.040s   0.034s
  foreach (conditional)        0.011s  0.011s  0.011s  0.012s   0.011s
  until (100M)                 0.322s  0.325s  0.324s  0.348s   0.298s
  reduce (harmonic)            0.036s  0.035s  0.035s  0.037s   0.032s
  reduce (floor pipe)          0.035s  0.034s  0.035s  0.036s   0.032s
  reduce (sqrt pipe)           0.035s  0.035s  0.035s  0.037s   0.032s
  reduce (sin+cos)             0.053s  0.053s  0.052s  0.054s   0.051s

--- Object operations ---
  Benchmark                    v1.9.0  v1.9.1  v1.9.2  v1.10.0  v1.11.0
  ---                          ------  ------  ------  -------  -------
  large obj construct          0.004s  0.004s  0.004s  0.004s   0.004s
  large obj keys               0.011s  0.011s  0.011s  0.012s   0.011s
  large obj to_entries         0.012s  0.012s  0.013s  0.013s   0.012s
  with_entries                 0.009s  0.010s  0.010s  0.010s   0.009s

--- Assignment operators ---
  Benchmark                    v1.9.0  v1.9.1  v1.9.2  v1.10.0  v1.11.0
  ---                          ------  ------  ------  -------  -------
  .[] |= f (100K)              0.005s  0.006s  0.005s  0.006s   0.005s
  .[] += 1 (100K)              0.006s  0.006s  0.006s  0.006s   0.005s
  .[k] = v reduce(50K)         0.009s  0.009s  0.009s  0.009s   0.008s
  p126-shape nested reduce     0.118s  0.120s  0.119s  0.114s   0.105s
  p126-shape w/ mutate         0.124s  0.120s  0.132s  0.117s   0.103s
  p150-shape serial mutate     0.013s  0.012s  0.013s  0.013s   0.012s

--- String-heavy generators ---
  Benchmark                    v1.9.0  v1.9.1  v1.9.2  v1.10.0  v1.11.0
  ---                          ------  ------  ------  -------  -------
  gsub(100K)                   0.029s  0.029s  0.029s  0.029s   0.028s
  join large(100K)             0.006s  0.006s  0.006s  0.006s   0.005s
  explode/implode(100K)        0.029s  0.028s  0.028s  0.029s   0.028s
  reduce str concat(100K)      0.008s  0.008s  0.008s  0.009s   0.008s

--- Try-catch & alternative ---
  Benchmark                    v1.9.0  v1.9.1  v1.9.2  v1.10.0  v1.11.0
  ---                          ------  ------  ------  -------  -------
  alternative //               0.034s  0.034s  0.034s  0.035s   0.033s
  try-catch                    0.024s  0.024s  0.025s  0.026s   0.024s
  label-break                  0.004s  0.004s  0.004s  0.005s   0.004s

--- Type conversion ---
  Benchmark                    v1.9.0  v1.9.1  v1.9.2  v1.10.0  v1.11.0
  ---                          ------  ------  ------  -------  -------
  tojson/fromjson(100K)        0.022s  0.023s  0.022s  0.023s   0.022s
  null propagation(2M)         0.091s  0.090s  0.090s  0.096s   0.089s

--- jaq-derived ---
  Benchmark                    v1.9.0  v1.9.1  v1.9.2  v1.10.0  v1.11.0
  ---                          ------  ------  ------  -------  -------
  jaq: reverse                 -       -       -       -        -
  jaq: sort                    -       -       -       -        -
  jaq: group-by                -       -       -       -        -
  jaq: min-max                 -       -       -       -        -
  jaq: ex-implode              -       -       -       -        -
  jaq: repeat                  -       -       -       -        -
  jaq: from                    -       -       -       -        -
  jaq: last                    -       -       -       -        -
  jaq: cumsum                  -       -       -       -        -
  jaq: cumsum-xy               -       -       -       -        -
  jaq: try-catch               -       -       -       -        -
  jaq: add                     -       -       -       -        -
  jaq: reduce                  -       -       -       -        -
  jaq: reduce-update           -       -       -       -        -
  jaq: kv                      -       -       -       -        -
  jaq: kv-update               -       -       -       -        -
  jaq: kv-entries              -       -       -       -        -
  jaq: pyramid                 -       -       -       -        -
  jaq: upto                    -       -       -       -        -
  jaq: tree-flatten            -       -       -       -        -
  jaq: tree-update             -       -       -       -        -
  jaq: to-fromjson             -       -       -       -        -
  jaq: str-slice               -       -       -       -        -

--- Memoization (jqx) ---
  Benchmark                    v1.9.0  v1.9.1  v1.9.2  v1.10.0  v1.11.0
  ---                          ------  ------  ------  -------  -------
  memo fib (1K)                0.003s  0.003s  0.003s  0.004s   0.003s
  memo collatz sum (10K)       0.015s  0.015s  0.015s  0.016s   0.014s
  memo by .id (100K, 1K keys)  0.020s  0.021s  0.021s  0.023s   0.020s
```
