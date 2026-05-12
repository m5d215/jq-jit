# Benchmark History

Recent slice (last 5 columns). Full history lives in
[`benchmark-history.tsv`](benchmark-history.tsv) (long format,
`section / benchmark / version / time_seconds`).

```text
--- NDJSON workloads (2M objects) ---
  Benchmark                v1.5.2  v1.5.3  v1.5.4  v1.5.5  v1.5.6
  ---                      ------  ------  ------  ------  ------
  empty                    0.017s  0.017s  0.017s  0.017s  0.017s
  identity -c              0.086s  0.092s  0.088s  0.087s  0.087s
  identity (pretty)        0.108s  0.109s  0.103s  0.102s  0.105s
  field access .name       0.092s  0.097s  0.096s  0.094s  0.095s
  nested .x,.y,.name       0.154s  0.155s  0.151s  0.150s  0.150s
  arithmetic .x + .y       0.084s  0.084s  0.085s  0.086s  0.083s
  select .x > 1500000      0.085s  0.083s  0.083s  0.083s  0.082s
  string concat            0.096s  0.099s  0.094s  0.093s  0.093s
  object construct         0.115s  0.113s  0.115s  0.115s  0.111s
  array construct          0.103s  0.105s  0.107s  0.106s  0.103s
  .[]                      0.107s  0.104s  0.105s  0.106s  0.101s
  to_entries               0.158s  0.158s  0.157s  0.162s  0.159s
  keys                     0.106s  0.104s  0.107s  0.106s  0.109s
  keys_unsorted            0.095s  0.095s  0.094s  0.095s  0.095s
  length                   0.084s  0.084s  0.085s  0.086s  0.085s
  has("x")                 0.036s  0.035s  0.037s  0.037s  0.034s
  type                     0.023s  0.023s  0.022s  0.023s  0.022s
  del(.name)               0.107s  0.104s  0.103s  0.102s  0.104s
  @csv                     0.122s  0.124s  0.127s  0.124s  0.123s
  split/join               0.091s  0.093s  0.091s  0.091s  0.089s
  select|field             0.099s  0.096s  0.103s  0.097s  0.095s
  select|remap             0.101s  0.101s  0.100s  0.100s  0.095s
  computed remap           0.190s  0.187s  0.194s  0.192s  0.191s
  [.x,.y]|add              0.083s  0.086s  0.084s  0.086s  0.082s
  [.x,.y]|avg              0.106s  0.108s  0.107s  0.112s  0.106s
  map(*2)|add              0.104s  0.105s  0.105s  0.106s  0.103s
  keys|length              0.253s  0.253s  0.254s  0.252s  0.251s
  .+{z=0}                  0.149s  0.151s  0.152s  0.147s  0.146s
  split|first              0.091s  0.093s  0.090s  0.090s  0.090s
  slice[0..5]              0.093s  0.099s  0.093s  0.092s  0.092s
  dynkey {(.name)}         0.104s  0.108s  0.109s  0.104s  0.102s
  .x += 1                  0.128s  0.130s  0.128s  0.125s  0.125s
  {a}+{b} merge            0.131s  0.132s  0.136s  0.130s  0.135s
  .x*2+1                   0.060s  0.060s  0.060s  0.060s  0.060s
  .x+.y*2                  0.097s  0.097s  0.098s  0.100s  0.096s
  .x > .y                  0.078s  0.077s  0.078s  0.081s  0.077s
  to_entries|len           0.396s  0.399s  0.397s  0.394s  0.400s
  .x|.+1 (pipe)            0.058s  0.058s  0.058s  0.058s  0.058s
  .x|.*2|.+1               0.059s  0.060s  0.059s  0.060s  0.059s
  .name|.+"_x"             0.095s  0.096s  0.094s  0.092s  0.092s
  .x>N | not               0.049s  0.048s  0.051s  0.050s  0.049s
  and (2 cmp)              0.082s  0.081s  0.081s  0.085s  0.080s
  if-then-else             0.052s  0.052s  0.051s  0.052s  0.052s
  sel(and)|field           0.079s  0.078s  0.078s  0.083s  0.077s
  sel(and)|remap           0.077s  0.078s  0.077s  0.083s  0.077s
  arith|cmp                0.055s  0.054s  0.055s  0.055s  0.053s
  if cmp .field            0.115s  0.115s  0.115s  0.114s  0.114s
  split|length             0.092s  0.091s  0.089s  0.088s  0.090s
  [x,y]|min                0.090s  0.091s  0.089s  0.095s  0.090s
  [x,y]|max                0.095s  0.094s  0.094s  0.097s  0.092s
  [x,y]|sort|.[0]          0.091s  0.091s  0.091s  0.094s  0.090s
  .name|len>5              0.091s  0.093s  0.094s  0.092s  0.092s
  sel(len>5)|.x            0.108s  0.109s  0.111s  0.110s  0.111s
  if .x>.y .name           0.090s  0.088s  0.091s  0.092s  0.089s
  sel(.x>.y)|.name         0.071s  0.073s  0.070s  0.076s  0.072s
  .x*2|tostring            0.057s  0.056s  0.057s  0.057s  0.056s
  .x*.x+1                  0.066s  0.065s  0.066s  0.066s  0.066s
  {k=.name,v=tostr}        0.144s  0.145s  0.154s  0.146s  0.144s
  str add chain            0.384s  0.389s  0.393s  0.378s  0.385s
  if>.y .name|empty        0.073s  0.071s  0.073s  0.080s  0.073s
  if .x%2==0               0.054s  0.055s  0.054s  0.056s  0.054s
  if .x*2+1>1M             0.056s  0.055s  0.056s  0.056s  0.054s
  sel(.x%2==0)|.name       0.083s  0.087s  0.084s  0.083s  0.083s
  sel(.x*2+1>1M)           0.159s  0.161s  0.158s  0.159s  0.159s
  .x|@json                 0.048s  0.049s  0.048s  0.047s  0.048s
  .x|@text                 0.048s  0.048s  0.048s  0.048s  0.047s
  .name|@json              0.100s  0.103s  0.108s  0.104s  0.103s
  sel|[arr]                0.148s  0.146s  0.147s  0.146s  0.143s
  sel(and)|[arr]           0.077s  0.079s  0.078s  0.082s  0.077s
  if>.y [arr]              0.173s  0.175s  0.183s  0.176s  0.173s
  if sw then .f            0.140s  0.144s  0.139s  0.138s  0.139s
  dynkey {(.n)=.x*2}       0.114s  0.116s  0.113s  0.113s  0.114s
  sel(and)|.x*.y           0.076s  0.079s  0.077s  0.082s  0.076s
  sel>N|str chain          0.152s  0.155s  0.159s  0.154s  0.151s
  .f+"_"+arith_ts          0.131s  0.132s  0.140s  0.133s  0.132s
  sel(sw)|str ch           0.309s  0.307s  0.310s  0.303s  0.307s
  split|rev|join           0.115s  0.114s  0.117s  0.114s  0.114s
  dynkey+static            0.329s  0.337s  0.339s  0.344s  0.331s
  if>.y str chain          0.165s  0.168s  0.174s  0.167s  0.165s
  remap+str chain          0.151s  0.157s  0.160s  0.153s  0.148s
  sel(len>8)               0.161s  0.166s  0.161s  0.160s  0.165s
  up|split|join            0.095s  0.098s  0.095s  0.099s  0.097s
  .name|index              0.122s  0.124s  0.123s  0.121s  0.121s
  .name|index+1            0.124s  0.127s  0.126s  0.125s  0.123s
  .name|rindex             0.129s  0.133s  0.129s  0.129s  0.127s
  .name|indices            0.155s  0.155s  0.155s  0.157s  0.146s
  [x,y]|sort               0.155s  0.155s  0.156s  0.153s  0.152s
  .name|scan               0.211s  0.209s  0.211s  0.209s  0.208s
  .name|gsub               0.167s  0.169s  0.166s  0.163s  0.168s
  walk(if num .+1)         0.139s  0.139s  0.142s  0.145s  0.143s
  tojson                   0.113s  0.109s  0.108s  0.106s  0.104s
  {name,x}                 0.125s  0.124s  0.134s  0.129s  0.133s
  .z//.name                0.154s  0.159s  0.155s  0.157s  0.157s
  .x|=test(re)             0.171s  0.173s  0.172s  0.173s  0.180s
  ./sep|first              0.185s  0.188s  0.187s  0.184s  0.182s
  .y=(.x*2)                0.180s  0.184s  0.179s  0.179s  0.178s
  .y=(.x+.y)               0.230s  0.239s  0.229s  0.229s  0.233s
  objects                  0.130s  0.137s  0.137s  0.128s  0.138s
  .tag|=if..then N         0.615s  0.611s  0.618s  0.607s  0.614s
  .x=(.x+1)                0.128s  0.127s  0.128s  0.122s  0.128s
  sel>N|.y+=1              0.121s  0.124s  0.122s  0.123s  0.121s
  sel(and)|.x+=1           0.108s  0.114s  0.110s  0.109s  0.111s
  sel(sw)|.x+=1            0.161s  0.165s  0.165s  0.163s  0.164s
  match(re)                0.364s  0.374s  0.375s  0.361s  0.369s
  capture(re)              0.297s  0.299s  0.306s  0.298s  0.296s
  first(.name,.x)          0.096s  0.098s  0.097s  0.096s  0.097s
  if .x==null              0.047s  0.047s  0.047s  0.047s  0.046s
  we(sw(.key))             0.105s  0.110s  0.108s  0.107s  0.110s
  sel(sw or ew)            0.205s  0.211s  0.214s  0.207s  0.205s
  path(.name,.x)           0.274s  0.277s  0.278s  0.274s  0.274s
  sel(str+num+num)         0.153s  0.155s  0.150s  0.151s  0.152s
  nested if|field          0.078s  0.078s  0.077s  0.082s  0.076s
  .f|floor|.*2             0.061s  0.061s  0.061s  0.061s  0.061s
  split|len>1              0.117s  0.120s  0.120s  0.118s  0.114s
  .name|len|.*2            0.103s  0.105s  0.104s  0.102s  0.103s
  if len>5 .x .y           0.111s  0.113s  0.116s  0.115s  0.113s
  sel(len>5)|remap         0.204s  0.204s  0.209s  0.206s  0.207s
  .x|tostr|len             0.059s  0.060s  0.060s  0.058s  0.059s
  if .x>.y .x .y           0.095s  0.094s  0.094s  0.098s  0.093s
  split|last|tonum         0.093s  0.099s  0.095s  0.095s  0.096s
  split|rev|.[0]           0.091s  0.096s  0.091s  0.094s  0.090s
  split|.[0]+.[1]          0.113s  0.119s  0.113s  0.115s  0.114s
  .[]|strings              0.109s  0.104s  0.105s  0.106s  0.104s
  .[]|numbers              0.128s  0.125s  0.124s  0.124s  0.124s
  [x,y]|any(>1M)           0.082s  0.082s  0.082s  0.087s  0.082s
  sel(dc|sw)               0.100s  0.100s  0.099s  0.102s  0.097s
  [[x,y],[n]]|flat         0.459s  0.454s  0.464s  0.456s  0.460s
  .x|floor|.*2             0.061s  0.061s  0.061s  0.061s  0.061s
  tojson|fromjson          0.087s  0.089s  0.087s  0.087s  0.085s
  [.x]|add                 0.063s  0.060s  0.058s  0.060s  0.060s
  if>N {o}+.               0.136s  0.140s  0.138s  0.134s  0.135s
  if>N .+{o}               0.139s  0.135s  0.137s  0.134s  0.138s
  if .n=="s" .+{o}         0.157s  0.163s  0.168s  0.161s  0.161s
  sel(.n>"s")              0.087s  0.091s  0.090s  0.089s  0.089s
  [x,y,z]|min              0.311s  0.309s  0.314s  0.309s  0.309s
  if .n|len>5 l s          0.100s  0.104s  0.102s  0.102s  0.100s
  if .x|flr>N b s          0.055s  0.056s  0.054s  0.056s  0.056s
  if .n|test l e           0.104s  0.108s  0.108s  0.103s  0.106s
  if .n|sw l e             0.083s  0.086s  0.083s  0.083s  0.084s
  if .n|ew l e             0.084s  0.087s  0.085s  0.084s  0.084s
  .n|len|tostr             0.088s  0.090s  0.090s  0.091s  0.092s

--- String operations (2M objects) ---
  Benchmark                v1.5.2  v1.5.3  v1.5.4  v1.5.5  v1.5.6
  ---                      ------  ------  ------  ------  ------
  ascii_downcase           0.105s  0.106s  0.107s  0.108s  0.105s
  ascii_upcase             0.102s  0.105s  0.104s  0.108s  0.103s
  ltrimstr                 0.098s  0.099s  0.099s  0.096s  0.095s
  rtrimstr                 0.096s  0.104s  0.101s  0.099s  0.098s
  split                    0.166s  0.169s  0.168s  0.168s  0.162s
  case+split               0.117s  0.119s  0.116s  0.113s  0.115s
  join                     0.094s  0.094s  0.095s  0.090s  0.091s
  startswith               0.093s  0.099s  0.096s  0.095s  0.095s
  endswith                 0.095s  0.101s  0.099s  0.097s  0.096s
  tostring                 0.063s  0.063s  0.063s  0.062s  0.063s
  tonumber                 0.109s  0.113s  0.113s  0.114s  0.110s
  string interpolation     0.119s  0.118s  0.121s  0.115s  0.118s

--- String ops (200K objects) ---
  Benchmark                v1.5.2  v1.5.3  v1.5.4  v1.5.5  v1.5.6
  ---                      ------  ------  ------  ------  ------
  test (regex)             0.015s  0.014s  0.014s  0.014s  0.015s
  match (regex)            0.032s  0.032s  0.032s  0.032s  0.033s
  @base64                  0.012s  0.012s  0.012s  0.012s  0.012s
  @uri                     0.012s  0.012s  0.012s  0.013s  0.012s
  @html                    0.013s  0.013s  0.012s  0.013s  0.012s
  @csv (array)             0.016s  0.016s  0.016s  0.015s  0.015s
  @tsv (array)             0.015s  0.015s  0.015s  0.014s  0.015s
  gsub                     0.019s  0.019s  0.018s  0.018s  0.019s
  case+gsub                0.179s  0.179s  0.177s  0.176s  0.181s
  case+test                0.119s  0.121s  0.118s  0.118s  0.119s
  ltrim+tonum+arith        0.112s  0.114s  0.115s  0.112s  0.111s

--- Numeric & math (2M objects) ---
  Benchmark                v1.5.2  v1.5.3  v1.5.4  v1.5.5  v1.5.6
  ---                      ------  ------  ------  ------  ------
  floor                    0.056s  0.056s  0.056s  0.057s  0.057s
  sqrt                     0.078s  0.078s  0.078s  0.079s  0.079s
  modulo                   0.058s  0.057s  0.058s  0.058s  0.057s
  if-elif-else             0.125s  0.123s  0.124s  0.124s  0.121s
  select|del               0.094s  0.092s  0.092s  0.090s  0.092s
  select|merge             0.121s  0.120s  0.118s  0.119s  0.120s
  select(test)|merge       0.021s  0.021s  0.022s  0.021s  0.022s

--- Array generators ---
  Benchmark                v1.5.2  v1.5.3  v1.5.4  v1.5.5  v1.5.6
  ---                      ------  ------  ------  ------  ------
  range(2M) | length       0.012s  0.011s  0.012s  0.011s  0.011s
  reverse(2M)              0.018s  0.018s  0.018s  0.018s  0.018s
  sort(2M)                 0.023s  0.023s  0.023s  0.023s  0.023s
  unique(1M)               0.031s  0.030s  0.030s  0.030s  0.030s
  flatten(500K)            0.011s  0.011s  0.011s  0.010s  0.010s
  min, max(2M)             0.022s  0.022s  0.019s  0.021s  0.018s
  add numbers(2M)          0.013s  0.013s  0.013s  0.013s  0.013s
  any/all(2M)              0.029s  0.028s  0.028s  0.028s  0.028s
  limit(10; range(10M))    0.002s  0.002s  0.002s  0.002s  0.002s
  first(range(10M))        0.003s  0.002s  0.002s  0.002s  0.002s
  last(range(2M))          0.002s  0.002s  0.002s  0.002s  0.002s
  indices(1M)              0.016s  0.016s  0.016s  0.016s  0.016s

--- Reduce & foreach ---
  Benchmark                v1.5.2  v1.5.3  v1.5.4  v1.5.5  v1.5.6
  ---                      ------  ------  ------  ------  ------
  reduce (sum)             0.009s  0.009s  0.009s  0.009s  0.009s
  reduce (array build)     0.004s  0.004s  0.004s  0.004s  0.004s
  reduce (obj build)       0.010s  0.010s  0.010s  0.009s  0.009s
  reduce (setpath)         0.016s  0.016s  0.017s  0.016s  0.016s
  foreach (running sum)    0.010s  0.010s  0.010s  0.010s  0.010s
  foreach + emit           0.010s  0.010s  0.010s  0.010s  0.010s
  reduce (sum-of-squares)  0.034s  0.033s  0.034s  0.033s  0.033s
  reduce (conditional)     0.036s  0.035s  0.036s  0.036s  0.035s
  reduce (product)         0.035s  0.034s  0.034s  0.034s  0.034s
  foreach (conditional)    0.011s  0.010s  0.010s  0.010s  0.010s
  until (100M)             0.303s  0.302s  0.302s  0.300s  0.301s
  reduce (harmonic)        0.034s  0.036s  0.033s  0.033s  0.033s
  reduce (floor pipe)      0.034s  0.034s  0.034s  0.033s  0.032s
  reduce (sqrt pipe)       0.033s  0.034s  0.033s  0.033s  0.032s
  reduce (sin+cos)         0.052s  0.052s  0.052s  0.052s  0.052s

--- Object operations ---
  Benchmark                v1.5.2  v1.5.3  v1.5.4  v1.5.5  v1.5.6
  ---                      ------  ------  ------  ------  ------
  large obj construct      0.004s  0.004s  0.004s  0.004s  0.004s
  large obj keys           0.011s  0.011s  0.011s  0.011s  0.011s
  large obj to_entries     0.012s  0.012s  0.012s  0.012s  0.012s
  with_entries             0.009s  0.009s  0.009s  0.009s  0.009s

--- Assignment operators ---
  Benchmark                v1.5.2  v1.5.3  v1.5.4  v1.5.5  v1.5.6
  ---                      ------  ------  ------  ------  ------
  .[] |= f (100K)          0.005s  0.005s  0.005s  0.005s  0.005s
  .[] += 1 (100K)          0.005s  0.005s  0.005s  0.005s  0.005s
  .[k] = v reduce(50K)     0.008s  0.008s  0.008s  0.008s  0.008s

--- String-heavy generators ---
  Benchmark                v1.5.2  v1.5.3  v1.5.4  v1.5.5  v1.5.6
  ---                      ------  ------  ------  ------  ------
  gsub(100K)               0.027s  0.026s  0.027s  0.026s  0.027s
  join large(100K)         0.006s  0.005s  0.005s  0.005s  0.006s
  explode/implode(100K)    0.027s  0.027s  0.028s  0.027s  0.028s
  reduce str concat(100K)  0.008s  0.008s  0.008s  0.008s  0.008s

--- Try-catch & alternative ---
  Benchmark                v1.5.2  v1.5.3  v1.5.4  v1.5.5  v1.5.6
  ---                      ------  ------  ------  ------  ------
  alternative //           0.033s  0.032s  0.033s  0.032s  0.032s
  try-catch                0.023s  0.023s  0.023s  0.023s  0.023s
  label-break              0.004s  0.004s  0.004s  0.004s  0.004s

--- Type conversion ---
  Benchmark                v1.5.2  v1.5.3  v1.5.4  v1.5.5  v1.5.6
  ---                      ------  ------  ------  ------  ------
  tojson/fromjson(100K)    0.022s  0.022s  0.022s  0.022s  0.022s
  null propagation(2M)     0.090s  0.090s  0.090s  0.089s  0.090s

--- jaq-derived ---
  Benchmark                v1.5.2  v1.5.3  v1.5.4  v1.5.5  v1.5.6
  ---                      ------  ------  ------  ------  ------
  jaq: reverse             -       -       -       -       -
  jaq: sort                -       -       -       -       -
  jaq: group-by            -       -       -       -       -
  jaq: min-max             -       -       -       -       -
  jaq: ex-implode          -       -       -       -       -
  jaq: repeat              -       -       -       -       -
  jaq: from                -       -       -       -       -
  jaq: last                -       -       -       -       -
  jaq: cumsum              -       -       -       -       -
  jaq: cumsum-xy           -       -       -       -       -
  jaq: try-catch           -       -       -       -       -
  jaq: add                 -       -       -       -       -
  jaq: reduce              -       -       -       -       -
  jaq: reduce-update       -       -       -       -       -
  jaq: kv                  -       -       -       -       -
  jaq: kv-update           -       -       -       -       -
  jaq: kv-entries          -       -       -       -       -
  jaq: pyramid             -       -       -       -       -
  jaq: upto                -       -       -       -       -
  jaq: tree-flatten        -       -       -       -       -
  jaq: tree-update         -       -       -       -       -
  jaq: to-fromjson         -       -       -       -       -
  jaq: str-slice           -       -       -       -       -
```
