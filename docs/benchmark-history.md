# Benchmark History

Recent slice (last 5 columns). Full history lives in
[`benchmark-history.tsv`](benchmark-history.tsv) (long format,
`section / benchmark / version / time_seconds`).

```text
--- NDJSON workloads (2M objects) ---
  Benchmark                v1.4.5  v1.5.0  v1.5.1  v1.5.2  v1.5.3
  ---                      ------  ------  ------  ------  ------
  empty                    0.017s  0.017s  0.018s  0.017s  0.017s
  identity -c              0.081s  0.085s  0.089s  0.086s  0.092s
  identity (pretty)        0.105s  0.103s  0.108s  0.108s  0.109s
  field access .name       0.091s  0.093s  0.095s  0.092s  0.097s
  nested .x,.y,.name       0.143s  0.145s  0.154s  0.154s  0.155s
  arithmetic .x + .y       0.081s  0.082s  0.086s  0.084s  0.084s
  select .x > 1500000      0.073s  0.080s  0.084s  0.085s  0.083s
  string concat            0.096s  0.091s  0.093s  0.096s  0.099s
  object construct         0.114s  0.111s  0.112s  0.115s  0.113s
  array construct          0.120s  0.103s  0.109s  0.103s  0.105s
  .[]                      0.099s  0.104s  0.106s  0.107s  0.104s
  to_entries               0.156s  0.159s  0.159s  0.158s  0.158s
  keys                     0.101s  0.101s  0.105s  0.106s  0.104s
  keys_unsorted            0.093s  0.094s  0.096s  0.095s  0.095s
  length                   0.083s  0.085s  0.086s  0.084s  0.084s
  has("x")                 0.030s  0.038s  0.036s  0.036s  0.035s
  type                     0.022s  0.022s  0.023s  0.023s  0.023s
  del(.name)               0.099s  0.102s  0.104s  0.107s  0.104s
  @csv                     0.137s  0.121s  0.119s  0.122s  0.124s
  split/join               0.098s  0.090s  0.088s  0.091s  0.093s
  select|field             0.112s  0.099s  0.095s  0.099s  0.096s
  select|remap             0.096s  0.096s  0.099s  0.101s  0.101s
  computed remap           0.214s  0.188s  0.187s  0.190s  0.187s
  [.x,.y]|add              0.083s  0.085s  0.084s  0.083s  0.086s
  [.x,.y]|avg              0.112s  0.108s  0.111s  0.106s  0.108s
  map(*2)|add              0.113s  0.103s  0.105s  0.104s  0.105s
  keys|length              0.254s  0.250s  0.253s  0.253s  0.253s
  .+{z=0}                  0.147s  0.151s  0.148s  0.149s  0.151s
  split|first              0.098s  0.087s  0.089s  0.091s  0.093s
  slice[0..5]              0.100s  0.091s  0.093s  0.093s  0.099s
  dynkey {(.name)}         0.121s  0.104s  0.104s  0.104s  0.108s
  .x += 1                  0.070s  0.125s  0.128s  0.128s  0.130s
  {a}+{b} merge            0.147s  0.130s  0.127s  0.131s  0.132s
  .x*2+1                   0.050s  0.058s  0.060s  0.060s  0.060s
  .x+.y*2                  0.105s  0.096s  0.096s  0.097s  0.097s
  .x > .y                  0.077s  0.077s  0.078s  0.078s  0.077s
  to_entries|len           0.395s  0.397s  0.403s  0.396s  0.399s
  .x|.+1 (pipe)            0.048s  0.057s  0.058s  0.058s  0.058s
  .x|.*2|.+1               0.050s  0.058s  0.061s  0.059s  0.060s
  .name|.+"_x"             0.098s  0.092s  0.092s  0.095s  0.096s
  .x>N | not               0.041s  0.049s  0.050s  0.049s  0.048s
  and (2 cmp)              0.080s  0.081s  0.082s  0.082s  0.081s
  if-then-else             0.043s  0.050s  0.052s  0.052s  0.052s
  sel(and)|field           0.076s  0.076s  0.080s  0.079s  0.078s
  sel(and)|remap           0.076s  0.077s  0.081s  0.077s  0.078s
  arith|cmp                0.047s  0.052s  0.055s  0.055s  0.054s
  if cmp .field            0.107s  0.111s  0.115s  0.115s  0.115s
  split|length             0.099s  0.089s  0.089s  0.092s  0.091s
  [x,y]|min                0.092s  0.090s  0.092s  0.090s  0.091s
  [x,y]|max                0.095s  0.094s  0.097s  0.095s  0.094s
  [x,y]|sort|.[0]          0.090s  0.088s  0.093s  0.091s  0.091s
  .name|len>5              0.100s  0.089s  0.093s  0.091s  0.093s
  sel(len>5)|.x            0.127s  0.110s  0.113s  0.108s  0.109s
  if .x>.y .name           0.090s  0.089s  0.092s  0.090s  0.088s
  sel(.x>.y)|.name         0.072s  0.072s  0.072s  0.071s  0.073s
  .x*2|tostring            0.048s  0.055s  0.057s  0.057s  0.056s
  .x*.x+1                  0.057s  0.064s  0.066s  0.066s  0.065s
  {k=.name,v=tostr}        0.162s  0.149s  0.150s  0.144s  0.145s
  str add chain            0.398s  0.382s  0.386s  0.384s  0.389s
  if>.y .name|empty        0.073s  0.074s  0.076s  0.073s  0.071s
  if .x%2==0               0.047s  0.053s  0.054s  0.054s  0.055s
  if .x*2+1>1M             0.048s  0.053s  0.056s  0.056s  0.055s
  sel(.x%2==0)|.name       0.076s  0.081s  0.083s  0.083s  0.087s
  sel(.x*2+1>1M)           0.144s  0.156s  0.157s  0.159s  0.161s
  .x|@json                 0.045s  0.047s  0.047s  0.048s  0.049s
  .x|@text                 0.045s  0.047s  0.048s  0.048s  0.048s
  .name|@json              0.107s  0.099s  0.100s  0.100s  0.103s
  sel|[arr]                0.150s  0.144s  0.145s  0.148s  0.146s
  sel(and)|[arr]           0.078s  0.077s  0.078s  0.077s  0.079s
  if>.y [arr]              0.199s  0.177s  0.177s  0.173s  0.175s
  if sw then .f            0.138s  0.137s  0.138s  0.140s  0.144s
  dynkey {(.n)=.x*2}       0.132s  0.112s  0.116s  0.114s  0.116s
  sel(and)|.x*.y           0.077s  0.078s  0.078s  0.076s  0.079s
  sel>N|str chain          0.158s  0.153s  0.156s  0.152s  0.155s
  .f+"_"+arith_ts          0.148s  0.134s  0.130s  0.131s  0.132s
  sel(sw)|str ch           0.310s  0.298s  0.307s  0.309s  0.307s
  split|rev|join           0.122s  0.114s  0.111s  0.115s  0.114s
  dynkey+static            0.331s  0.334s  0.339s  0.329s  0.337s
  if>.y str chain          0.189s  0.170s  0.163s  0.165s  0.168s
  remap+str chain          0.178s  0.151s  0.150s  0.151s  0.157s
  sel(len>8)               0.170s  0.162s  0.164s  0.161s  0.166s
  up|split|join            0.104s  0.096s  0.096s  0.095s  0.098s
  .name|index              0.128s  0.114s  0.123s  0.122s  0.124s
  .name|index+1            0.133s  0.121s  0.123s  0.124s  0.127s
  .name|rindex             0.134s  0.126s  0.127s  0.129s  0.133s
  .name|indices            0.160s  0.154s  0.155s  0.155s  0.155s
  [x,y]|sort               0.156s  0.152s  0.152s  0.155s  0.155s
  .name|scan               0.219s  0.204s  0.208s  0.211s  0.209s
  .name|gsub               0.178s  0.161s  0.172s  0.167s  0.169s
  walk(if num .+1)         0.141s  0.137s  0.137s  0.139s  0.139s
  tojson                   0.110s  0.107s  0.105s  0.113s  0.109s
  {name,x}                 0.144s  0.132s  0.127s  0.125s  0.124s
  .z//.name                0.164s  0.153s  0.159s  0.154s  0.159s
  .x|=test(re)             0.124s  0.169s  0.179s  0.171s  0.173s
  ./sep|first              0.132s  0.184s  0.183s  0.185s  0.188s
  .y=(.x*2)                0.115s  0.178s  0.176s  0.180s  0.184s
  .y=(.x+.y)               0.159s  0.228s  0.230s  0.230s  0.239s
  objects                  0.085s  0.123s  0.137s  0.130s  0.137s
  .tag|=if..then N         0.614s  0.607s  0.602s  0.615s  0.611s
  .x=(.x+1)                0.072s  0.124s  0.128s  0.128s  0.127s
  sel>N|.y+=1              0.087s  0.118s  0.123s  0.121s  0.124s
  sel(and)|.x+=1           0.102s  0.109s  0.109s  0.108s  0.114s
  sel(sw)|.x+=1            0.133s  0.160s  0.159s  0.161s  0.165s
  match(re)                0.367s  0.359s  0.360s  0.364s  0.374s
  capture(re)              0.303s  0.292s  0.291s  0.297s  0.299s
  first(.name,.x)          0.093s  0.093s  0.095s  0.096s  0.098s
  if .x==null              0.044s  0.046s  0.048s  0.047s  0.047s
  we(sw(.key))             0.111s  0.107s  0.106s  0.105s  0.110s
  sel(sw or ew)            0.214s  0.201s  0.206s  0.205s  0.211s
  path(.name,.x)           0.277s  0.271s  0.276s  0.274s  0.277s
  sel(str+num+num)         0.145s  0.151s  0.151s  0.153s  0.155s
  nested if|field          0.077s  0.077s  0.077s  0.078s  0.078s
  .f|floor|.*2             0.051s  0.059s  0.061s  0.061s  0.061s
  split|len>1              0.124s  0.115s  0.117s  0.117s  0.120s
  .name|len|.*2            0.107s  0.100s  0.101s  0.103s  0.105s
  if len>5 .x .y           0.133s  0.115s  0.113s  0.111s  0.113s
  sel(len>5)|remap         0.233s  0.200s  0.202s  0.204s  0.204s
  .x|tostr|len             0.056s  0.059s  0.059s  0.059s  0.060s
  if .x>.y .x .y           0.094s  0.094s  0.096s  0.095s  0.094s
  split|last|tonum         0.104s  0.097s  0.095s  0.093s  0.099s
  split|rev|.[0]           0.099s  0.095s  0.090s  0.091s  0.096s
  split|.[0]+.[1]          0.123s  0.113s  0.113s  0.113s  0.119s
  .[]|strings              0.095s  0.106s  0.106s  0.109s  0.104s
  .[]|numbers              0.106s  0.129s  0.125s  0.128s  0.125s
  [x,y]|any(>1M)           0.082s  0.082s  0.082s  0.082s  0.082s
  sel(dc|sw)               0.107s  0.096s  0.097s  0.100s  0.100s
  [[x,y],[n]]|flat         0.475s  0.452s  0.457s  0.459s  0.454s
  .x|floor|.*2             0.051s  0.059s  0.062s  0.061s  0.061s
  tojson|fromjson          0.081s  0.088s  0.087s  0.087s  0.089s
  [.x]|add                 0.048s  0.059s  0.059s  0.063s  0.060s
  if>N {o}+.               0.125s  0.133s  0.135s  0.136s  0.140s
  if>N .+{o}               0.125s  0.135s  0.136s  0.139s  0.135s
  if .n=="s" .+{o}         0.161s  0.161s  0.161s  0.157s  0.163s
  sel(.n>"s")              0.094s  0.087s  0.088s  0.087s  0.091s
  [x,y,z]|min              0.312s  0.306s  0.309s  0.311s  0.309s
  if .n|len>5 l s          0.106s  0.100s  0.100s  0.100s  0.104s
  if .x|flr>N b s          0.045s  0.054s  0.054s  0.055s  0.056s
  if .n|test l e           0.112s  0.105s  0.102s  0.104s  0.108s
  if .n|sw l e             0.091s  0.082s  0.083s  0.083s  0.086s
  if .n|ew l e             0.091s  0.084s  0.083s  0.084s  0.087s
  .n|len|tostr             0.099s  0.091s  0.091s  0.088s  0.090s

--- String operations (2M objects) ---
  Benchmark                v1.4.5  v1.5.0  v1.5.1  v1.5.2  v1.5.3
  ---                      ------  ------  ------  ------  ------
  ascii_downcase           0.112s  0.103s  0.104s  0.105s  0.106s
  ascii_upcase             0.109s  0.102s  0.105s  0.102s  0.105s
  ltrimstr                 0.102s  0.099s  0.097s  0.098s  0.099s
  rtrimstr                 0.107s  0.099s  0.099s  0.096s  0.104s
  split                    0.172s  0.165s  0.165s  0.166s  0.169s
  case+split               0.128s  0.116s  0.114s  0.117s  0.119s
  join                     0.104s  0.093s  0.091s  0.094s  0.094s
  startswith               0.103s  0.095s  0.096s  0.093s  0.099s
  endswith                 0.104s  0.095s  0.097s  0.095s  0.101s
  tostring                 0.053s  0.061s  0.063s  0.063s  0.063s
  tonumber                 0.117s  0.110s  0.109s  0.109s  0.113s
  string interpolation     0.134s  0.122s  0.117s  0.119s  0.118s

--- String ops (200K objects) ---
  Benchmark                v1.4.5  v1.5.0  v1.5.1  v1.5.2  v1.5.3
  ---                      ------  ------  ------  ------  ------
  test (regex)             0.014s  0.014s  0.014s  0.015s  0.014s
  match (regex)            0.033s  0.032s  0.032s  0.032s  0.032s
  @base64                  0.012s  0.011s  0.012s  0.012s  0.012s
  @uri                     0.013s  0.012s  0.011s  0.012s  0.012s
  @html                    0.013s  0.012s  0.012s  0.013s  0.013s
  @csv (array)             0.018s  0.015s  0.016s  0.016s  0.016s
  @tsv (array)             0.017s  0.015s  0.015s  0.015s  0.015s
  gsub                     0.020s  0.018s  0.019s  0.019s  0.019s
  case+gsub                0.193s  0.178s  0.176s  0.179s  0.179s
  case+test                0.130s  0.115s  0.116s  0.119s  0.121s
  ltrim+tonum+arith        0.118s  0.108s  0.113s  0.112s  0.114s

--- Numeric & math (2M objects) ---
  Benchmark                v1.4.5  v1.5.0  v1.5.1  v1.5.2  v1.5.3
  ---                      ------  ------  ------  ------  ------
  floor                    0.048s  0.056s  0.057s  0.056s  0.056s
  sqrt                     0.078s  0.078s  0.079s  0.078s  0.078s
  modulo                   0.051s  0.056s  0.057s  0.058s  0.057s
  if-elif-else             0.124s  0.124s  0.124s  0.125s  0.123s
  select|del               0.079s  0.090s  0.091s  0.094s  0.092s
  select|merge             0.107s  0.118s  0.117s  0.121s  0.120s
  select(test)|merge       0.022s  0.021s  0.021s  0.021s  0.021s

--- Array generators ---
  Benchmark                v1.4.5  v1.5.0  v1.5.1  v1.5.2  v1.5.3
  ---                      ------  ------  ------  ------  ------
  range(2M) | length       0.011s  0.011s  0.012s  0.012s  0.011s
  reverse(2M)              0.017s  0.018s  0.018s  0.018s  0.018s
  sort(2M)                 0.023s  0.025s  0.023s  0.023s  0.023s
  unique(1M)               0.029s  0.030s  0.030s  0.031s  0.030s
  flatten(500K)            0.010s  0.010s  0.011s  0.011s  0.011s
  min, max(2M)             0.017s  0.018s  0.018s  0.022s  0.022s
  add numbers(2M)          0.012s  0.013s  0.013s  0.013s  0.013s
  any/all(2M)              0.027s  0.028s  0.028s  0.029s  0.028s
  limit(10; range(10M))    0.002s  0.002s  0.002s  0.002s  0.002s
  first(range(10M))        0.002s  0.002s  0.002s  0.003s  0.002s
  last(range(2M))          0.002s  0.002s  0.002s  0.002s  0.002s
  indices(1M)              0.015s  0.016s  0.016s  0.016s  0.016s

--- Reduce & foreach ---
  Benchmark                v1.4.5  v1.5.0  v1.5.1  v1.5.2  v1.5.3
  ---                      ------  ------  ------  ------  ------
  reduce (sum)             0.009s  0.009s  0.009s  0.009s  0.009s
  reduce (array build)     0.004s  0.004s  0.004s  0.004s  0.004s
  reduce (obj build)       0.009s  0.009s  0.010s  0.010s  0.010s
  reduce (setpath)         0.017s  0.016s  0.016s  0.016s  0.016s
  foreach (running sum)    0.010s  0.010s  0.010s  0.010s  0.010s
  foreach + emit           0.010s  0.010s  0.010s  0.010s  0.010s
  reduce (sum-of-squares)  0.032s  0.032s  0.034s  0.034s  0.033s
  reduce (conditional)     0.035s  0.035s  0.036s  0.036s  0.035s
  reduce (product)         0.034s  0.034s  0.034s  0.035s  0.034s
  foreach (conditional)    0.010s  0.010s  0.011s  0.011s  0.010s
  until (100M)             0.295s  0.300s  0.303s  0.303s  0.302s
  reduce (harmonic)        0.033s  0.032s  0.033s  0.034s  0.036s
  reduce (floor pipe)      0.033s  0.032s  0.033s  0.034s  0.034s
  reduce (sqrt pipe)       0.032s  0.034s  0.033s  0.033s  0.034s
  reduce (sin+cos)         0.052s  0.052s  0.052s  0.052s  0.052s

--- Object operations ---
  Benchmark                v1.4.5  v1.5.0  v1.5.1  v1.5.2  v1.5.3
  ---                      ------  ------  ------  ------  ------
  large obj construct      0.004s  0.004s  0.004s  0.004s  0.004s
  large obj keys           0.011s  0.011s  0.011s  0.011s  0.011s
  large obj to_entries     0.012s  0.012s  0.012s  0.012s  0.012s
  with_entries             0.009s  0.009s  0.009s  0.009s  0.009s

--- Assignment operators ---
  Benchmark                v1.4.5  v1.5.0  v1.5.1  v1.5.2  v1.5.3
  ---                      ------  ------  ------  ------  ------
  .[] |= f (100K)          0.005s  0.005s  0.005s  0.005s  0.005s
  .[] += 1 (100K)          0.005s  0.005s  0.006s  0.005s  0.005s
  .[k] = v reduce(50K)     0.008s  0.008s  0.008s  0.008s  0.008s

--- String-heavy generators ---
  Benchmark                v1.4.5  v1.5.0  v1.5.1  v1.5.2  v1.5.3
  ---                      ------  ------  ------  ------  ------
  gsub(100K)               0.025s  0.028s  0.027s  0.027s  0.026s
  join large(100K)         0.005s  0.006s  0.005s  0.006s  0.005s
  explode/implode(100K)    0.028s  0.028s  0.027s  0.027s  0.027s
  reduce str concat(100K)  0.008s  0.008s  0.008s  0.008s  0.008s

--- Try-catch & alternative ---
  Benchmark                v1.4.5  v1.5.0  v1.5.1  v1.5.2  v1.5.3
  ---                      ------  ------  ------  ------  ------
  alternative //           0.032s  0.032s  0.033s  0.033s  0.032s
  try-catch                0.023s  0.023s  0.023s  0.023s  0.023s
  label-break              0.004s  0.004s  0.004s  0.004s  0.004s

--- Type conversion ---
  Benchmark                v1.4.5  v1.5.0  v1.5.1  v1.5.2  v1.5.3
  ---                      ------  ------  ------  ------  ------
  tojson/fromjson(100K)    0.022s  0.022s  0.022s  0.022s  0.022s
  null propagation(2M)     0.087s  0.089s  0.090s  0.090s  0.090s

--- jaq-derived ---
  Benchmark                v1.4.5  v1.5.0  v1.5.1  v1.5.2  v1.5.3
  ---                      ------  ------  ------  ------  ------
  jaq: reverse             0.010s  0.011s  -       -       -
  jaq: sort                0.017s  0.018s  -       -       -
  jaq: group-by            0.037s  0.037s  -       -       -
  jaq: min-max             0.010s  0.011s  -       -       -
  jaq: ex-implode          0.019s  0.019s  -       -       -
  jaq: repeat              0.011s  0.011s  -       -       -
  jaq: from                0.006s  0.005s  -       -       -
  jaq: last                0.002s  0.002s  -       -       -
  jaq: cumsum              0.010s  0.010s  -       -       -
  jaq: cumsum-xy           0.017s  0.017s  -       -       -
  jaq: try-catch           0.090s  0.077s  -       -       -
  jaq: add                 0.040s  0.041s  -       -       -
  jaq: reduce              0.086s  0.078s  -       -       -
  jaq: reduce-update       0.005s  0.005s  -       -       -
  jaq: kv                  0.014s  0.015s  -       -       -
  jaq: kv-update           0.018s  0.018s  -       -       -
  jaq: kv-entries          0.055s  0.056s  -       -       -
  jaq: pyramid             0.016s  0.016s  -       -       -
  jaq: upto                0.007s  0.007s  -       -       -
  jaq: tree-flatten        0.003s  0.003s  -       -       -
  jaq: tree-update         0.007s  0.222s  -       -       -
  jaq: to-fromjson         0.005s  0.005s  -       -       -
  jaq: str-slice           0.014s  0.014s  -       -       -
```
