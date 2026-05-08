# Benchmark History

Recent slice (last 5 columns). Full history lives in
[`benchmark-history.tsv`](benchmark-history.tsv) (long format,
`section / benchmark / version / time_seconds`).

```text
--- NDJSON workloads (2M objects) ---
  Benchmark                3d440ca  v1.4.5  v1.5.0  v1.5.1  v1.5.2
  ---                      -------  ------  ------  ------  ------
  empty                    0.017s   0.017s  0.017s  0.018s  0.017s
  identity -c              0.078s   0.081s  0.085s  0.089s  0.086s
  identity (pretty)        0.101s   0.105s  0.103s  0.108s  0.108s
  field access .name       0.087s   0.091s  0.093s  0.095s  0.092s
  nested .x,.y,.name       0.144s   0.143s  0.145s  0.154s  0.154s
  arithmetic .x + .y       0.082s   0.081s  0.082s  0.086s  0.084s
  select .x > 1500000      0.069s   0.073s  0.080s  0.084s  0.085s
  string concat            0.096s   0.096s  0.091s  0.093s  0.096s
  object construct         0.112s   0.114s  0.111s  0.112s  0.115s
  array construct          0.116s   0.120s  0.103s  0.109s  0.103s
  .[]                      0.100s   0.099s  0.104s  0.106s  0.107s
  to_entries               0.161s   0.156s  0.159s  0.159s  0.158s
  keys                     0.098s   0.101s  0.101s  0.105s  0.106s
  keys_unsorted            0.094s   0.093s  0.094s  0.096s  0.095s
  length                   0.080s   0.083s  0.085s  0.086s  0.084s
  has("x")                 0.029s   0.030s  0.038s  0.036s  0.036s
  type                     0.022s   0.022s  0.022s  0.023s  0.023s
  del(.name)               0.098s   0.099s  0.102s  0.104s  0.107s
  @csv                     0.140s   0.137s  0.121s  0.119s  0.122s
  split/join               0.093s   0.098s  0.090s  0.088s  0.091s
  select|field             0.112s   0.112s  0.099s  0.095s  0.099s
  select|remap             0.095s   0.096s  0.096s  0.099s  0.101s
  computed remap           0.211s   0.214s  0.188s  0.187s  0.190s
  [.x,.y]|add              0.082s   0.083s  0.085s  0.084s  0.083s
  [.x,.y]|avg              0.110s   0.112s  0.108s  0.111s  0.106s
  map(*2)|add              0.110s   0.113s  0.103s  0.105s  0.104s
  keys|length              0.254s   0.254s  0.250s  0.253s  0.253s
  .+{z=0}                  0.143s   0.147s  0.151s  0.148s  0.149s
  split|first              0.092s   0.098s  0.087s  0.089s  0.091s
  slice[0..5]              0.095s   0.100s  0.091s  0.093s  0.093s
  dynkey {(.name)}         0.118s   0.121s  0.104s  0.104s  0.104s
  .x += 1                  0.069s   0.070s  0.125s  0.128s  0.128s
  {a}+{b} merge            0.143s   0.147s  0.130s  0.127s  0.131s
  .x*2+1                   0.049s   0.050s  0.058s  0.060s  0.060s
  .x+.y*2                  0.102s   0.105s  0.096s  0.096s  0.097s
  .x > .y                  0.076s   0.077s  0.077s  0.078s  0.078s
  to_entries|len           0.397s   0.395s  0.397s  0.403s  0.396s
  .x|.+1 (pipe)            0.047s   0.048s  0.057s  0.058s  0.058s
  .x|.*2|.+1               0.049s   0.050s  0.058s  0.061s  0.059s
  .name|.+"_x"             0.098s   0.098s  0.092s  0.092s  0.095s
  .x>N | not               0.040s   0.041s  0.049s  0.050s  0.049s
  and (2 cmp)              0.080s   0.080s  0.081s  0.082s  0.082s
  if-then-else             0.043s   0.043s  0.050s  0.052s  0.052s
  sel(and)|field           0.076s   0.076s  0.076s  0.080s  0.079s
  sel(and)|remap           0.074s   0.076s  0.077s  0.081s  0.077s
  arith|cmp                0.044s   0.047s  0.052s  0.055s  0.055s
  if cmp .field            0.102s   0.107s  0.111s  0.115s  0.115s
  split|length             0.094s   0.099s  0.089s  0.089s  0.092s
  [x,y]|min                0.090s   0.092s  0.090s  0.092s  0.090s
  [x,y]|max                0.091s   0.095s  0.094s  0.097s  0.095s
  [x,y]|sort|.[0]          0.087s   0.090s  0.088s  0.093s  0.091s
  .name|len>5              0.096s   0.100s  0.089s  0.093s  0.091s
  sel(len>5)|.x            0.125s   0.127s  0.110s  0.113s  0.108s
  if .x>.y .name           0.087s   0.090s  0.089s  0.092s  0.090s
  sel(.x>.y)|.name         0.071s   0.072s  0.072s  0.072s  0.071s
  .x*2|tostring            0.047s   0.048s  0.055s  0.057s  0.057s
  .x*.x+1                  0.056s   0.057s  0.064s  0.066s  0.066s
  {k=.name,v=tostr}        0.160s   0.162s  0.149s  0.150s  0.144s
  str add chain            0.386s   0.398s  0.382s  0.386s  0.384s
  if>.y .name|empty        0.071s   0.073s  0.074s  0.076s  0.073s
  if .x%2==0               0.045s   0.047s  0.053s  0.054s  0.054s
  if .x*2+1>1M             0.045s   0.048s  0.053s  0.056s  0.056s
  sel(.x%2==0)|.name       0.074s   0.076s  0.081s  0.083s  0.083s
  sel(.x*2+1>1M)           0.142s   0.144s  0.156s  0.157s  0.159s
  .x|@json                 0.045s   0.045s  0.047s  0.047s  0.048s
  .x|@text                 0.045s   0.045s  0.047s  0.048s  0.048s
  .name|@json              0.104s   0.107s  0.099s  0.100s  0.100s
  sel|[arr]                0.148s   0.150s  0.144s  0.145s  0.148s
  sel(and)|[arr]           0.076s   0.078s  0.077s  0.078s  0.077s
  if>.y [arr]              0.199s   0.199s  0.177s  0.177s  0.173s
  if sw then .f            0.135s   0.138s  0.137s  0.138s  0.140s
  dynkey {(.n)=.x*2}       0.131s   0.132s  0.112s  0.116s  0.114s
  sel(and)|.x*.y           0.075s   0.077s  0.078s  0.078s  0.076s
  sel>N|str chain          0.158s   0.158s  0.153s  0.156s  0.152s
  .f+"_"+arith_ts          0.145s   0.148s  0.134s  0.130s  0.131s
  sel(sw)|str ch           0.317s   0.310s  0.298s  0.307s  0.309s
  split|rev|join           0.123s   0.122s  0.114s  0.111s  0.115s
  dynkey+static            0.324s   0.331s  0.334s  0.339s  0.329s
  if>.y str chain          0.186s   0.189s  0.170s  0.163s  0.165s
  remap+str chain          0.171s   0.178s  0.151s  0.150s  0.151s
  sel(len>8)               0.163s   0.170s  0.162s  0.164s  0.161s
  up|split|join            0.101s   0.104s  0.096s  0.096s  0.095s
  .name|index              0.126s   0.128s  0.114s  0.123s  0.122s
  .name|index+1            0.127s   0.133s  0.121s  0.123s  0.124s
  .name|rindex             0.131s   0.134s  0.126s  0.127s  0.129s
  .name|indices            0.153s   0.160s  0.154s  0.155s  0.155s
  [x,y]|sort               0.154s   0.156s  0.152s  0.152s  0.155s
  .name|scan               0.218s   0.219s  0.204s  0.208s  0.211s
  .name|gsub               0.175s   0.178s  0.161s  0.172s  0.167s
  walk(if num .+1)         0.140s   0.141s  0.137s  0.137s  0.139s
  tojson                   0.106s   0.110s  0.107s  0.105s  0.113s
  {name,x}                 0.144s   0.144s  0.132s  0.127s  0.125s
  .z//.name                0.165s   0.164s  0.153s  0.159s  0.154s
  .x|=test(re)             0.123s   0.124s  0.169s  0.179s  0.171s
  ./sep|first              0.131s   0.132s  0.184s  0.183s  0.185s
  .y=(.x*2)                0.112s   0.115s  0.178s  0.176s  0.180s
  .y=(.x+.y)               0.161s   0.159s  0.228s  0.230s  0.230s
  objects                  0.081s   0.085s  0.123s  0.137s  0.130s
  .tag|=if..then N         0.613s   0.614s  0.607s  0.602s  0.615s
  .x=(.x+1)                0.070s   0.072s  0.124s  0.128s  0.128s
  sel>N|.y+=1              0.085s   0.087s  0.118s  0.123s  0.121s
  sel(and)|.x+=1           0.100s   0.102s  0.109s  0.109s  0.108s
  sel(sw)|.x+=1            0.132s   0.133s  0.160s  0.159s  0.161s
  match(re)                0.369s   0.367s  0.359s  0.360s  0.364s
  capture(re)              0.306s   0.303s  0.292s  0.291s  0.297s
  first(.name,.x)          0.090s   0.093s  0.093s  0.095s  0.096s
  if .x==null              0.043s   0.044s  0.046s  0.048s  0.047s
  we(sw(.key))             0.111s   0.111s  0.107s  0.106s  0.105s
  sel(sw or ew)            0.211s   0.214s  0.201s  0.206s  0.205s
  path(.name,.x)           0.276s   0.277s  0.271s  0.276s  0.274s
  sel(str+num+num)         0.142s   0.145s  0.151s  0.151s  0.153s
  nested if|field          0.076s   0.077s  0.077s  0.077s  0.078s
  .f|floor|.*2             0.050s   0.051s  0.059s  0.061s  0.061s
  split|len>1              0.122s   0.124s  0.115s  0.117s  0.117s
  .name|len|.*2            0.106s   0.107s  0.100s  0.101s  0.103s
  if len>5 .x .y           0.137s   0.133s  0.115s  0.113s  0.111s
  sel(len>5)|remap         0.230s   0.233s  0.200s  0.202s  0.204s
  .x|tostr|len             0.056s   0.056s  0.059s  0.059s  0.059s
  if .x>.y .x .y           0.093s   0.094s  0.094s  0.096s  0.095s
  split|last|tonum         0.100s   0.104s  0.097s  0.095s  0.093s
  split|rev|.[0]           0.097s   0.099s  0.095s  0.090s  0.091s
  split|.[0]+.[1]          0.122s   0.123s  0.113s  0.113s  0.113s
  .[]|strings              0.096s   0.095s  0.106s  0.106s  0.109s
  .[]|numbers              0.107s   0.106s  0.129s  0.125s  0.128s
  [x,y]|any(>1M)           0.082s   0.082s  0.082s  0.082s  0.082s
  sel(dc|sw)               0.104s   0.107s  0.096s  0.097s  0.100s
  [[x,y],[n]]|flat         0.475s   0.475s  0.452s  0.457s  0.459s
  .x|floor|.*2             0.051s   0.051s  0.059s  0.062s  0.061s
  tojson|fromjson          0.083s   0.081s  0.088s  0.087s  0.087s
  [.x]|add                 0.048s   0.048s  0.059s  0.059s  0.063s
  if>N {o}+.               0.123s   0.125s  0.133s  0.135s  0.136s
  if>N .+{o}               0.124s   0.125s  0.135s  0.136s  0.139s
  if .n=="s" .+{o}         0.167s   0.161s  0.161s  0.161s  0.157s
  sel(.n>"s")              0.095s   0.094s  0.087s  0.088s  0.087s
  [x,y,z]|min              0.311s   0.312s  0.306s  0.309s  0.311s
  if .n|len>5 l s          0.107s   0.106s  0.100s  0.100s  0.100s
  if .x|flr>N b s          0.047s   0.045s  0.054s  0.054s  0.055s
  if .n|test l e           0.111s   0.112s  0.105s  0.102s  0.104s
  if .n|sw l e             0.090s   0.091s  0.082s  0.083s  0.083s
  if .n|ew l e             0.092s   0.091s  0.084s  0.083s  0.084s
  .n|len|tostr             0.097s   0.099s  0.091s  0.091s  0.088s

--- String operations (2M objects) ---
  Benchmark                3d440ca  v1.4.5  v1.5.0  v1.5.1  v1.5.2
  ---                      -------  ------  ------  ------  ------
  ascii_downcase           0.110s   0.112s  0.103s  0.104s  0.105s
  ascii_upcase             0.108s   0.109s  0.102s  0.105s  0.102s
  ltrimstr                 0.101s   0.102s  0.099s  0.097s  0.098s
  rtrimstr                 0.105s   0.107s  0.099s  0.099s  0.096s
  split                    0.169s   0.172s  0.165s  0.165s  0.166s
  case+split               0.126s   0.128s  0.116s  0.114s  0.117s
  join                     0.101s   0.104s  0.093s  0.091s  0.094s
  startswith               0.101s   0.103s  0.095s  0.096s  0.093s
  endswith                 0.103s   0.104s  0.095s  0.097s  0.095s
  tostring                 0.052s   0.053s  0.061s  0.063s  0.063s
  tonumber                 0.117s   0.117s  0.110s  0.109s  0.109s
  string interpolation     0.135s   0.134s  0.122s  0.117s  0.119s

--- String ops (200K objects) ---
  Benchmark                3d440ca  v1.4.5  v1.5.0  v1.5.1  v1.5.2
  ---                      -------  ------  ------  ------  ------
  test (regex)             0.014s   0.014s  0.014s  0.014s  0.015s
  match (regex)            0.033s   0.033s  0.032s  0.032s  0.032s
  @base64                  0.013s   0.012s  0.011s  0.012s  0.012s
  @uri                     0.013s   0.013s  0.012s  0.011s  0.012s
  @html                    0.013s   0.013s  0.012s  0.012s  0.013s
  @csv (array)             0.020s   0.018s  0.015s  0.016s  0.016s
  @tsv (array)             0.018s   0.017s  0.015s  0.015s  0.015s
  gsub                     0.020s   0.020s  0.018s  0.019s  0.019s
  case+gsub                0.193s   0.193s  0.178s  0.176s  0.179s
  case+test                0.130s   0.130s  0.115s  0.116s  0.119s
  ltrim+tonum+arith        0.118s   0.118s  0.108s  0.113s  0.112s

--- Numeric & math (2M objects) ---
  Benchmark                3d440ca  v1.4.5  v1.5.0  v1.5.1  v1.5.2
  ---                      -------  ------  ------  ------  ------
  floor                    0.048s   0.048s  0.056s  0.057s  0.056s
  sqrt                     0.078s   0.078s  0.078s  0.079s  0.078s
  modulo                   0.051s   0.051s  0.056s  0.057s  0.058s
  if-elif-else             0.125s   0.124s  0.124s  0.124s  0.125s
  select|del               0.079s   0.079s  0.090s  0.091s  0.094s
  select|merge             0.110s   0.107s  0.118s  0.117s  0.121s
  select(test)|merge       0.022s   0.022s  0.021s  0.021s  0.021s

--- Array generators ---
  Benchmark                3d440ca  v1.4.5  v1.5.0  v1.5.1  v1.5.2
  ---                      -------  ------  ------  ------  ------
  range(2M) | length       0.011s   0.011s  0.011s  0.012s  0.012s
  reverse(2M)              0.018s   0.017s  0.018s  0.018s  0.018s
  sort(2M)                 0.023s   0.023s  0.025s  0.023s  0.023s
  unique(1M)               0.029s   0.029s  0.030s  0.030s  0.031s
  flatten(500K)            0.010s   0.010s  0.010s  0.011s  0.011s
  min, max(2M)             0.017s   0.017s  0.018s  0.018s  0.022s
  add numbers(2M)          0.012s   0.012s  0.013s  0.013s  0.013s
  any/all(2M)              0.028s   0.027s  0.028s  0.028s  0.029s
  limit(10; range(10M))    0.002s   0.002s  0.002s  0.002s  0.002s
  first(range(10M))        0.002s   0.002s  0.002s  0.002s  0.003s
  last(range(2M))          0.002s   0.002s  0.002s  0.002s  0.002s
  indices(1M)              0.015s   0.015s  0.016s  0.016s  0.016s

--- Reduce & foreach ---
  Benchmark                3d440ca  v1.4.5  v1.5.0  v1.5.1  v1.5.2
  ---                      -------  ------  ------  ------  ------
  reduce (sum)             0.009s   0.009s  0.009s  0.009s  0.009s
  reduce (array build)     0.004s   0.004s  0.004s  0.004s  0.004s
  reduce (obj build)       0.010s   0.009s  0.009s  0.010s  0.010s
  reduce (setpath)         0.016s   0.017s  0.016s  0.016s  0.016s
  foreach (running sum)    0.010s   0.010s  0.010s  0.010s  0.010s
  foreach + emit           0.010s   0.010s  0.010s  0.010s  0.010s
  reduce (sum-of-squares)  0.032s   0.032s  0.032s  0.034s  0.034s
  reduce (conditional)     0.035s   0.035s  0.035s  0.036s  0.036s
  reduce (product)         0.035s   0.034s  0.034s  0.034s  0.035s
  foreach (conditional)    0.011s   0.010s  0.010s  0.011s  0.011s
  until (100M)             0.297s   0.295s  0.300s  0.303s  0.303s
  reduce (harmonic)        0.033s   0.033s  0.032s  0.033s  0.034s
  reduce (floor pipe)      0.033s   0.033s  0.032s  0.033s  0.034s
  reduce (sqrt pipe)       0.032s   0.032s  0.034s  0.033s  0.033s
  reduce (sin+cos)         0.052s   0.052s  0.052s  0.052s  0.052s

--- Object operations ---
  Benchmark                3d440ca  v1.4.5  v1.5.0  v1.5.1  v1.5.2
  ---                      -------  ------  ------  ------  ------
  large obj construct      0.004s   0.004s  0.004s  0.004s  0.004s
  large obj keys           0.011s   0.011s  0.011s  0.011s  0.011s
  large obj to_entries     0.012s   0.012s  0.012s  0.012s  0.012s
  with_entries             0.009s   0.009s  0.009s  0.009s  0.009s

--- Assignment operators ---
  Benchmark                3d440ca  v1.4.5  v1.5.0  v1.5.1  v1.5.2
  ---                      -------  ------  ------  ------  ------
  .[] |= f (100K)          0.005s   0.005s  0.005s  0.005s  0.005s
  .[] += 1 (100K)          0.005s   0.005s  0.005s  0.006s  0.005s
  .[k] = v reduce(50K)     0.008s   0.008s  0.008s  0.008s  0.008s

--- String-heavy generators ---
  Benchmark                3d440ca  v1.4.5  v1.5.0  v1.5.1  v1.5.2
  ---                      -------  ------  ------  ------  ------
  gsub(100K)               0.025s   0.025s  0.028s  0.027s  0.027s
  join large(100K)         0.006s   0.005s  0.006s  0.005s  0.006s
  explode/implode(100K)    0.029s   0.028s  0.028s  0.027s  0.027s
  reduce str concat(100K)  0.008s   0.008s  0.008s  0.008s  0.008s

--- Try-catch & alternative ---
  Benchmark                3d440ca  v1.4.5  v1.5.0  v1.5.1  v1.5.2
  ---                      -------  ------  ------  ------  ------
  alternative //           0.032s   0.032s  0.032s  0.033s  0.033s
  try-catch                0.023s   0.023s  0.023s  0.023s  0.023s
  label-break              0.004s   0.004s  0.004s  0.004s  0.004s

--- Type conversion ---
  Benchmark                3d440ca  v1.4.5  v1.5.0  v1.5.1  v1.5.2
  ---                      -------  ------  ------  ------  ------
  tojson/fromjson(100K)    0.022s   0.022s  0.022s  0.022s  0.022s
  null propagation(2M)     0.088s   0.087s  0.089s  0.090s  0.090s

--- jaq-derived ---
  Benchmark                3d440ca  v1.4.5  v1.5.0  v1.5.1  v1.5.2
  ---                      -------  ------  ------  ------  ------
  jaq: reverse             0.011s   0.010s  0.011s  -       -
  jaq: sort                0.018s   0.017s  0.018s  -       -
  jaq: group-by            0.037s   0.037s  0.037s  -       -
  jaq: min-max             0.011s   0.010s  0.011s  -       -
  jaq: ex-implode          0.019s   0.019s  0.019s  -       -
  jaq: repeat              0.011s   0.011s  0.011s  -       -
  jaq: from                0.006s   0.006s  0.005s  -       -
  jaq: last                0.002s   0.002s  0.002s  -       -
  jaq: cumsum              0.010s   0.010s  0.010s  -       -
  jaq: cumsum-xy           0.017s   0.017s  0.017s  -       -
  jaq: try-catch           0.083s   0.090s  0.077s  -       -
  jaq: add                 0.040s   0.040s  0.041s  -       -
  jaq: reduce              0.081s   0.086s  0.078s  -       -
  jaq: reduce-update       0.005s   0.005s  0.005s  -       -
  jaq: kv                  0.015s   0.014s  0.015s  -       -
  jaq: kv-update           0.018s   0.018s  0.018s  -       -
  jaq: kv-entries          0.055s   0.055s  0.056s  -       -
  jaq: pyramid             0.016s   0.016s  0.016s  -       -
  jaq: upto                0.007s   0.007s  0.007s  -       -
  jaq: tree-flatten        0.003s   0.003s  0.003s  -       -
  jaq: tree-update         0.007s   0.007s  0.222s  -       -
  jaq: to-fromjson         0.005s   0.005s  0.005s  -       -
  jaq: str-slice           0.014s   0.014s  0.014s  -       -
```
