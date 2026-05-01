#!/bin/bash
# Comprehensive jq-jit benchmark — broad coverage across NDJSON, generators,
# reduce/foreach, regex, type conversion, and an external jaq filter corpus.
# Source dataset for docs/benchmark-history.md release columns.
#
# Usage:
#   bench/comprehensive.sh                        # benchmark target/release/jq-jit
#   JQ_JIT=path/to/binary bench/comprehensive.sh  # benchmark a different build, or set
#                                                 # JQ_JIT=$(which jq) for a sanity comparison
#
# bench/run.sh — quick daily checks (142 NDJSON patterns).
# This script — broader coverage (~92 patterns + jaq-derived filters).
set -e

JQ_JIT="${JQ_JIT:-target/release/jq-jit}"
TIMEOUT=15
RUNS=3

if [ ! -x "$JQ_JIT" ]; then
    echo "Error: $JQ_JIT not found. Run: cargo build --release"
    exit 1
fi

# Generate test data if needed
NDJSON="/tmp/bench_2m.json"
NDJSON_200K="/tmp/bench_200k.json"
if [ ! -f "$NDJSON" ]; then
    bash "$(dirname "$0")/generate_data.sh" "$NDJSON"
fi
if [ ! -f "$NDJSON_200K" ]; then
    python3 -c "
import json, sys
for i in range(200000):
    sys.stdout.write(json.dumps({'x': i, 'y': i * 2, 'name': 'item_' + str(i)}) + '\n')
" > "$NDJSON_200K"
fi

# Benchmark runner: NDJSON file input
bench_ndjson() {
    local label="$1" flags="$2" filter="$3" datafile="${4:-$NDJSON}"
    printf "  %-35s" "$label"
    local best=999 failed=0
    for ((i=0; i<RUNS; i++)); do
        local t
        t=$( { TIMEFORMAT='%U'; time timeout "$TIMEOUT" "$JQ_JIT" $flags "$filter" "$datafile" > /dev/null 2>&1; } 2>&1 ) && rc=0 || rc=$?
        if [ $rc -ne 0 ]; then failed=1; break; fi
        if (( $(echo "$t < $best" | bc -l) )); then best=$t; fi
    done
    if [ $failed -eq 1 ]; then
        printf "  %-14s\n" "FAIL/TIMEOUT"
    else
        printf "  %-14s\n" "${best}s"
    fi
}

# Benchmark runner: single JSON input via echo
bench_gen() {
    local label="$1" n="$2" filter="$3"
    printf "  %-35s" "$label"
    local best=999 failed=0
    for ((i=0; i<RUNS; i++)); do
        local t
        t=$( { TIMEFORMAT='%U'; time echo "$n" | timeout "$TIMEOUT" "$JQ_JIT" "$filter" > /dev/null 2>&1; } 2>&1 ) && rc=0 || rc=$?
        if [ $rc -ne 0 ]; then failed=1; break; fi
        if (( $(echo "$t < $best" | bc -l) )); then best=$t; fi
    done
    if [ $failed -eq 1 ]; then
        printf "  %-14s\n" "FAIL/TIMEOUT"
    else
        printf "  %-14s\n" "${best}s"
    fi
}

header() {
    printf "  %-35s  %-14s\n" "Benchmark" "time"
    printf "  %-35s  %-14s\n" "---" "---"
}

echo "=== Comprehensive jq-jit Benchmark Suite ==="
echo "Binary: $JQ_JIT"
echo "Runs: best of $RUNS, timeout ${TIMEOUT}s"
echo ""

# --- NDJSON workloads (2M objects) ---
echo "--- NDJSON workloads (2M objects) ---"
header
# Format per entry: 'label:flags:filter'. The filter is everything after the
# second `:` so embedded colons (e.g. `{name: .name}`) are preserved verbatim.
NDJSON_TESTS=(
    'empty::empty'
    'identity -c:-c:.'
    'identity (pretty)::.'
    'field access .name:-c:.name'
    'nested .x,.y,.name:-c:.x,.y,.name'
    'arithmetic .x + .y:-c:.x + .y'
    'select .x > 1500000:-c:select(.x > 1500000)'
    'string concat:-c:.name + "_x"'
    'object construct:-c:{a: .x, b: .y}'
    'array construct:-c:[.name, .x]'
    '.[]:-c:.[]'
    'to_entries:-c:to_entries'
    'keys:-c:keys'
    'keys_unsorted:-c:keys_unsorted'
    'length:-c:length'
    'has("x"):-c:has("x")'
    'type:-c:type'
    'del(.name):-c:del(.name)'
    '@csv:-c:[.name, .x] | @csv'
    'split/join:-c:.name | split("a") | join("b")'
    'select|field:-c:select(.x > 1000000) | .name'
    'select|remap:-c:select(.x > 1000000) | {n:.name, v:.y}'
    'computed remap:-c:{name: .name, double: (.x * 2), sum: (.x + .y)}'
    '[.x,.y]|add:-c:[.x, .y] | add'
    '[.x,.y]|avg:-c:[.x, .y] | add / length'
    'map(*2)|add:-c:[.x, .y] | map(. * 2) | add'
    'keys|length:-c:keys | length'
    '.+{z=0}:-c:. + {z: 0}'
    'split|first:-c:.name | split("_") | .[0]'
    'slice[0..5]:-c:.name[0:5]'
    'dynkey {(.name)}:-c:{(.name): .x}'
    '.x += 1:-c:.x += 1'
    '{a}+{b} merge:-c:{a: .name} + {b: .x}'
    '.x*2+1:-c:.x * 2 + 1'
    '.x+.y*2:-c:.x + .y * 2'
    '.x > .y:-c:.x > .y'
    'to_entries|len:-c:to_entries | length'
    '.x|.+1 (pipe):-c:.x | . + 1'
    '.x|.*2|.+1:-c:.x | . * 2 | . + 1'
    '.name|.+"_x":-c:.name | . + "_x"'
    '.x>N | not:-c:.x > 1000000 | not'
    'and (2 cmp):-c:.x > 100 and .y < 500'
    'if-then-else:-c:if .x > 1000000 then "big" else "small" end'
    'sel(and)|field:-c:select(.x > 100 and .y < 500) | .name'
    'sel(and)|remap:-c:select(.x > 100 and .y < 500) | {n:.name, v:.y}'
    'arith|cmp:-c:.x | . * 2 | . + 1 | . > 1000000'
    'if cmp .field:-c:if .x > 1000000 then .name else .y end'
    'split|length:-c:.name | split("_") | length'
    '[x,y]|min:-c:[.x, .y] | min'
    '[x,y]|max:-c:[.x, .y] | max'
    '[x,y]|sort|.[0]:-c:[.x, .y] | sort | .[0]'
    '.name|len>5:-c:.name | length > 5'
    'sel(len>5)|.x:-c:select(.name | length > 5) | .x'
    'if .x>.y .name:-c:if .x > .y then .name else .x end'
    'sel(.x>.y)|.name:-c:select(.x > .y) | .name'
    '.x*2|tostring:-c:.x * 2 | tostring'
    '.x*.x+1:-c:.x | . * . + 1'
    '{k:.name,v:tostr}:-c:{key:.name,val:(.x|tostring)}'
    'str add chain:-c:.name + ":" + (.x|tostring)'
    'if>.y .name|empty:-c:if .x > .y then .name else empty end'
    'if .x%2==0:-c:if .x % 2 == 0 then "even" else "odd" end'
    'if .x*2+1>1M:-c:if .x * 2 + 1 > 1000000 then "big" else "small" end'
    'sel(.x%2==0)|.name:-c:select(.x % 2 == 0) | .name'
    'sel(.x*2+1>1M):-c:select(.x * 2 + 1 > 1000000)'
    '.x|@json:-c:.x | @json'
    '.x|@text:-c:.x | @text'
    '.name|@json:-c:.name | @json'
    'sel|[arr]:-c:select(.x > 500) | [.name, .x]'
    'sel(and)|[arr]:-c:select(.x > 100 and .y < 500) | [.name, .x, .y]'
    'if>.y [arr]:-c:if .x > .y then [.name, .x] else [.name, .y] end'
    'if sw then .f:-c:if .name | startswith("user_1") then .x else .y end'
    'dynkey {(.n):.x*2}:-c:{(.name): (.x * 2)}'
    'sel(and)|.x*.y:-c:select(.x > 500 and .y < 1000) | .x * .y'
    'sel>N|str chain:-c:select(.x > 1000) | (.name + ":" + (.x | tostring))'
    '.f+"_"+arith_ts:-c:.name + "_" + (.x * 2 | tostring)'
    'sel(sw)|str ch:-c:select(.name | startswith("item_1")) | (.name + ":" + (.x | tostring))'
    'split|rev|join:-c:.name | split("_") | reverse | join("-")'
    'dynkey+static:-c:{(.name): .x, total: (.x + .y)}'
    'if>.y str chain:-c:if .x > .y then .name + ":big" else .name + ":small" end'
    'remap+str chain:-c:{a: (.name + "_" + (.x | tostring)), b: .y}'
    'sel(len>8):-c:select(.name | length > 8)'
    'up|split|join:-c:.name | ascii_upcase | split("_") | join("-")'
    '.name|index:-c:.name | index("_")'
    '.name|index+1:-c:.name | index("_") + 1'
    '.name|rindex:-c:.name | rindex("_")'
    '.name|indices:-c:.name | indices("_")'
    '[x,y]|sort:-c:[.x, .y] | sort'
    '.name|scan:-c:.name | scan("[0-9]+")'
    '.name|gsub:-c:.name | gsub("_"; "-")'
    'walk(if num .+1):-c:walk(if type == "number" then . + 1 else . end)'
    'tojson:-c:tojson'
    '{name,x}:-c:{name,x}'
    '.z//.name:-c:.z // .name'
    '.x|=test(re):-c:.name |= test("^item_[0-9]+")'
    './sep|first:-c:.name |= (. / "_" | .[0])'
    '.y=(.x*2):-c:.y = (.x * 2)'
    '.y=(.x+.y):-c:.y = (.x + .y)'
    'objects:-c:objects'
    '.tag|=if..then N:-c:.tag |= if . == "abc" then 1 else 0 end'
    '.x=(.x+1):-c:.x = (.x + 1)'
    'sel>N|.y+=1:-c:select(.x > 1000000) | .y += 1'
    'sel(and)|.x+=1:-c:select(.x > 500 and .y < 1000) | .x += 1'
    'sel(sw)|.x+=1:-c:select(.name | startswith("item_1")) | .x += 1'
    'match(re):-c:.name | match("([a-z]+)_([0-9]+)")'
    'capture(re):-c:.name | capture("(?P<w>[a-z]+)_(?P<n>[0-9]+)")'
    'first(.name,.x):-c:first(.name, .x)'
    'if .x==null:-c:if .x == null then "none" else "some" end'
    'we(sw(.key)):-c:with_entries(select(.key | startswith("n")))'
    'sel(sw or ew):-c:select((.name | startswith("item_1")) or (.name | endswith("_0")))'
    'path(.name,.x):-c:path(.name, .x)'
    'sel(str+num+num):-c:select((.name | contains("_1")) and .x > 500 and .y < 1000)'
    'nested if|field:-c:if .x > 100 then if .y < 500 then .name else empty end else empty end'
    '.f|floor|.*2:-c:.x | floor | . * 2'
    'split|len>1:-c:.name | split("_") | length > 1'
    '.name|len|.*2:-c:.name | length | . * 2'
    'if len>5 .x .y:-c:if .name | length > 5 then .x else .y end'
    'sel(len>5)|remap:-c:select(.name | length > 5) | {n:.name, v:.x}'
    '.x|tostr|len:-c:.x | tostring | length'
    'if .x>.y .x .y:-c:if .x > .y then .x else .y end'
    'split|last|tonum:-c:.name | split("_") | last | tonumber'
    'split|rev|.[0]:-c:.name | split("_") | reverse | .[0]'
    'split|.[0]+.[1]:-c:.name | split("_") | .[0] + "-" + .[1]'
    '.[]|strings:-c:.[] | strings'
    '.[]|numbers:-c:.[] | numbers'
    '[x,y]|any(>1M):-c:[.x, .y] | any(. > 1000000)'
    'sel(dc|sw):-c:select(.name | ascii_downcase | startswith("user"))'
    '[[x,y],[n]]|flat:-c:[[.x,.y],[.name]] | flatten'
    '.x|floor|.*2:-c:.x | floor | . * 2'
    'tojson|fromjson:-c:tojson | fromjson'
    '[.x]|add:-c:[.x] | add'
    'if>N {o}+.:-c:if .x > 1000000 then {status:"high"} + . else . end'
    'if>N .+{o}:-c:if .x > 1000000 then . + {status:"high"} else . end'
    'if .n=="s" .+{o}:-c:if .name == "user_1" then . + {found:true} else . end'
    'sel(.n>"s"):-c:select(.name > "user_5")'
    '[x,y,z]|min:-c:[.x, .y, .x] | min'
    'if .n|len>5 l s:-c:if .name | length > 5 then "long" else "short" end'
    'if .x|flr>N b s:-c:if .x | floor > 1000000 then "big" else "small" end'
    'if .n|test l e:-c:if .name | test("^user") then "yes" else "no" end'
    'if .n|sw l e:-c:if .name | startswith("user") then "yes" else "no" end'
    'if .n|ew l e:-c:if .name | endswith("_0") then "yes" else "no" end'
    '.n|len|tostr:-c:.name | length | tostring'
)
for t in "${NDJSON_TESTS[@]}"; do
    IFS=: read -r label flags filter <<< "$t"
    bench_ndjson "$label" "$flags" "$filter"
done

echo ""
echo "--- String operations (2M objects) ---"
header
bench_ndjson "ascii_downcase"          "-c" ".name | ascii_downcase"
bench_ndjson "ascii_upcase"            "-c" ".name | ascii_upcase"
bench_ndjson "ltrimstr"                "-c" '.name | ltrimstr("item_")'
bench_ndjson "rtrimstr"                "-c" '.name | rtrimstr("0")'
bench_ndjson "split"                   "-c" '.name | split("_")'
bench_ndjson "case+split"              "-c" '.name | ascii_downcase | split("_")'
bench_ndjson "join"                    "-c" '[.name, "x"] | join(",")'
bench_ndjson "startswith"              "-c" '.name | startswith("item_1")'
bench_ndjson "endswith"                "-c" '.name | endswith("0")'
bench_ndjson "tostring"                "-c" ".x | tostring"
bench_ndjson "tonumber"                "-c" '.name | ltrimstr("item_") | tonumber'
bench_ndjson "string interpolation"    "-c" '"\(.name)=\(.x)"'

echo ""
echo "--- String ops (200K objects) ---"
header
bench_ndjson "test (regex)"            "-c" '.name | test("^item_[0-9]+$")' "$NDJSON_200K"
bench_ndjson "match (regex)"           "-c" '.name | match("([0-9]+)")' "$NDJSON_200K"
bench_ndjson "@base64"                 "-c" ".name | @base64" "$NDJSON_200K"
bench_ndjson "@uri"                    "-c" ".name | @uri" "$NDJSON_200K"
bench_ndjson "@html"                   "-c" ".name | @html" "$NDJSON_200K"
bench_ndjson "@csv (array)"            "-c" '[.name, .x, .y] | @csv' "$NDJSON_200K"
bench_ndjson "@tsv (array)"            "-c" '[.name, .x, .y] | @tsv' "$NDJSON_200K"
bench_ndjson "gsub"                    "-c" '.name | gsub("_"; "-")' "$NDJSON_200K"
bench_ndjson "case+gsub"               "-c" '.name | ascii_downcase | gsub("_"; " ")'
bench_ndjson "case+test"               "-c" '.name | ascii_downcase | test("user")'
bench_ndjson "ltrim+tonum+arith"       "-c" '.name | ltrimstr("item_") | tonumber | . * 2'

echo ""
echo "--- Numeric & math (2M objects) ---"
header
bench_ndjson "floor"                   "-c" ".x / 3 | floor"
bench_ndjson "sqrt"                    "-c" ".x | sqrt"
bench_ndjson "modulo"                  "-c" ".x % 7"
bench_ndjson "if-elif-else"            "-c" "if .x > 1000000 then .x elif .x > 500000 then .y else 0 end"
bench_ndjson "select|del"              "-c" 'select(.x > 1000000) | del(.name)'
bench_ndjson "select|merge"            "-c" 'select(.x > 1000000) | .+{status:"high"}'
bench_ndjson "select(test)|merge"      "-c" 'select(.name | test("_[12]")) | .+{tag:"match"}' "$NDJSON_200K"

echo ""
echo "--- Array generators ---"
header
bench_gen "range(2M) | length"          2000000   "[range(.)] | length"
bench_gen "reverse(2M)"                 2000000   "[range(.)] | reverse | length"
bench_gen "sort(2M)"                    2000000   "[range(.) * -1] | sort | length"
bench_gen "unique(1M)"                  1000000   "[range(.) | . % 1000] | unique | length"
bench_gen "flatten(500K)"               500000    '[[range(.)], [range(.)]] | flatten | length'
bench_gen "min, max(2M)"                2000000   "[range(.)] | [min, max]"
bench_gen "add numbers(2M)"             2000000   "[range(.)] | add"
bench_gen "any/all(2M)"                 2000000   "[range(.)] | [any(. > 1000), all(. >= 0)]"
bench_gen "limit(10; range(10M))"       10000000  "limit(10; range(.))"
bench_gen "first(range(10M))"           10000000  "first(range(.))"
bench_gen "last(range(2M))"             2000000   "last(range(.))"
bench_gen "indices(1M)"                 1000000   "[range(.) | . % 100] | indices(42) | length"

echo ""
echo "--- Reduce & foreach ---"
header
bench_gen "reduce (sum)"              2000000   "reduce range(.) as \$x (0; . + \$x)"
bench_gen "reduce (array build)"      500000    "reduce range(.) as \$x ([]; . + [\$x]) | length"
bench_gen "reduce (obj build)"        50000     'reduce range(.) as $x ({}; . + {("k\($x)"): $x}) | length'
bench_gen "reduce (setpath)"          100000    'reduce range(.) as $i ({}; setpath(["k\($i)"]; $i)) | length'
bench_gen "foreach (running sum)"     1000000   "[foreach range(.) as \$x (0; . + \$x)] | length"
bench_gen "foreach + emit"            1000000   "[foreach range(.) as \$x (0; . + \$x; . * 2)] | length"
bench_gen "reduce (sum-of-squares)"  10000000  "reduce range(1; . + 1) as \$x (0; . + \$x * \$x)"
bench_gen "reduce (conditional)"     10000000  'reduce range(.) as $x (0; if $x % 3 == 0 or $x % 5 == 0 then . + $x else . end)'
bench_gen "reduce (product)"         10000000  "reduce range(1; . + 1) as \$x (1; . * \$x)"
bench_gen "foreach (conditional)"    1000000   '[foreach range(.) as $x (0; if $x % 2 == 0 then . + $x else . + 1 end)] | length'
bench_gen "until (100M)"             100000000 '. as $n | 0 | until(. >= $n; . + 1)'

bench_gen "reduce (harmonic)"       10000000  "reduce range(1; . + 1) as \$x (0; . + 1 / \$x)"
bench_gen "reduce (floor pipe)"     10000000  "reduce range(.) as \$x (0; . + (\$x / 3 | floor))"
bench_gen "reduce (sqrt pipe)"      10000000  "reduce range(1; . + 1) as \$x (0; . + (\$x | sqrt))"
bench_gen "reduce (sin+cos)"         5000000  "reduce range(.) as \$x (0; . + (\$x | sin) + (\$x | cos))"

echo ""
echo "--- Object operations ---"
header
bench_gen "large obj construct"        10000     '[range(.) | {("k\(.)"): .}] | add'
bench_gen "large obj keys"             50000     '[range(.) | {("k\(.)"): .}] | add | keys | length'
bench_gen "large obj to_entries"       50000     '[range(.) | {("k\(.)"): .}] | add | to_entries | length'
bench_gen "with_entries"               10000     '[range(.) | {("k\(.)"): .}] | add | with_entries(.value += 1) | length'

echo ""
echo "--- Assignment operators ---"
header
bench_gen ".[] |= f (100K)"            100000    '[range(.)] | [.[] |= . + 1] | length'
bench_gen ".[] += 1 (100K)"            100000    '[range(.)] | .[] += 1 | length'
bench_gen ".[k] = v reduce(50K)"       50000    'reduce range(.) as $i ({}; .["k\($i)"] = $i) | length'

echo ""
echo "--- String-heavy generators ---"
header
bench_gen "gsub(100K)"                  100000    '[range(.) | "hello_world_\(.)"] | map(gsub("_"; "-")) | length'
bench_gen "join large(100K)"            100000    '[range(.) | tostring] | join(",") | length'
bench_gen "explode/implode(100K)"       100000    '[range(.) | "abc\(.)"] | map(explode | map(. + 1) | implode) | length'
bench_gen "reduce str concat(100K)"    100000    'reduce range(.) as $x (""; . + ($x | tostring)) | length'

echo ""
echo "--- Try-catch & alternative ---"
header
bench_gen "alternative //"              1000000   '[range(.) | if . % 2 == 0 then null else . end] | map(. // 0) | length'
bench_gen "try-catch"                   1000000   '[range(.)] | map(try (1/.) catch 0) | length'
bench_gen "label-break"                 2000000   'label $out | foreach range(.) as $x (0; . + $x; if . > 1000 then ., break $out else . end) | length'

echo ""
echo "--- Type conversion ---"
header
bench_gen "tojson/fromjson(100K)"       100000    '[range(.) | {a: ., b: "x"}] | map(tojson | fromjson) | length'
bench_gen "null propagation(2M)"        2000000   '[range(.) | null] | map(.a.b.c?) | length'

echo ""
echo "--- jaq-derived ---"
header
BENCH_DIR="$HOME/src/github.com/01mf02/jaq/examples/benches"
if [ -d "$BENCH_DIR" ]; then
    for name in reverse sort group-by min-max ex-implode repeat from last cumsum cumsum-xy try-catch add reduce reduce-update kv kv-update kv-entries pyramid upto tree-flatten tree-update to-fromjson str-slice; do
        f="$BENCH_DIR/$name.jq"
        if [ -f "$f" ]; then
            case "$name" in
                to-fromjson)   n=65536  ;;
                reduce-update) n=16384  ;;
                str-slice)     n=8192   ;;
                range-prop)    n=128    ;;
                tree-*)        n=17     ;;
                pyramid)       n=524288 ;;
                upto)          n=8192   ;;
                kv*|from)      n=131072 ;;
                *)             n=1048576;;
            esac
            filter="$(cat "$f") | length"
            bench_gen "jaq: $name" "$n" "$filter"
        fi
    done
else
    echo "  (jaq repo not found at $BENCH_DIR — clone it for the external corpus section)"
fi

echo ""
echo "Times: user CPU seconds (best of $RUNS). Lower is better."
