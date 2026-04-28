#!/bin/bash
# Comprehensive jq-jit benchmark suite
# Sources: jq, gojq, jaq bench patterns + jq-jit weak-pattern analysis
#
# Usage:
#   bench/comprehensive.sh          # full run (best of 3)
#   bench/comprehensive.sh --quick  # quick run (1 pass, 15s timeout)
#
# bench/run.sh — quick daily checks (17 NDJSON patterns, ~30s, colored ratios)
# This script — thorough analysis (80+ patterns: generators, strings, regex, etc.)
set -e

JQ_JIT="${JQ_JIT:-target/release/jq-jit}"
JQ="$(command -v jq 2>/dev/null || true)"
JAQ="$(command -v jaq 2>/dev/null || true)"
TIMEOUT=30
RUNS=3

if [ "$1" = "--quick" ]; then
    #RUNS=1
    TIMEOUT=15
    JQ=
    JAQ=
fi

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
    for tool in "$JQ_JIT" "$JQ" "$JAQ"; do
        if [ -z "$tool" ]; then printf "  %-14s" "N/A"; continue; fi
        local best=999 failed=0
        for ((i=0; i<RUNS; i++)); do
            local t
            t=$( { TIMEFORMAT='%U'; time timeout "$TIMEOUT" "$tool" $flags "$filter" "$datafile" > /dev/null 2>&1; } 2>&1 ) && rc=0 || rc=$?
            if [ $rc -ne 0 ]; then failed=1; break; fi
            if (( $(echo "$t < $best" | bc -l) )); then best=$t; fi
        done
        if [ $failed -eq 1 ]; then
            printf "  %-14s" "FAIL/TIMEOUT"
        else
            printf "  %-14s" "${best}s"
        fi
    done
    echo ""
}

# Benchmark runner: single JSON input via echo
bench_gen() {
    local label="$1" n="$2" filter="$3"
    printf "  %-35s" "$label"
    for tool in "$JQ_JIT" "$JQ" "$JAQ"; do
        if [ -z "$tool" ]; then printf "  %-14s" "N/A"; continue; fi
        local best=999 failed=0
        for ((i=0; i<RUNS; i++)); do
            local t
            t=$( { TIMEFORMAT='%U'; time echo "$n" | timeout "$TIMEOUT" "$tool" "$filter" > /dev/null 2>&1; } 2>&1 ) && rc=0 || rc=$?
            if [ $rc -ne 0 ]; then failed=1; break; fi
            if (( $(echo "$t < $best" | bc -l) )); then best=$t; fi
        done
        if [ $failed -eq 1 ]; then
            printf "  %-14s" "FAIL/TIMEOUT"
        else
            printf "  %-14s" "${best}s"
        fi
    done
    echo ""
}

header() {
    printf "  %-35s  %-14s  %-14s  %-14s\n" "Benchmark" "jq-jit" "jq" "jaq"
    printf "  %-35s  %-14s  %-14s  %-14s\n" "---" "---" "---" "---"
}

echo "=== Comprehensive jq-jit Benchmark Suite ==="
echo "Tools: jq-jit | jq $(jq --version 2>&1) | jaq $(jaq --version 2>&1 | awk '{print $2}')"
echo "Runs: best of $RUNS, timeout ${TIMEOUT}s"
echo ""

# --- NDJSON workloads (2M objects) ---
echo "--- NDJSON workloads (2M objects) ---"
header
bench_ndjson "empty"                   ""   "empty"
bench_ndjson "identity -c"             "-c" "."
bench_ndjson "identity (pretty)"       ""   "."
bench_ndjson "field access .name"      "-c" ".name"
bench_ndjson "nested .x,.y,.name"      "-c" ".x,.y,.name"
bench_ndjson "arithmetic .x + .y"      "-c" ".x + .y"
bench_ndjson "select .x > 1500000"     "-c" "select(.x > 1500000)"
bench_ndjson "string concat"           "-c" '.name + "_x"'
bench_ndjson "object construct"        "-c" '{a: .x, b: .y}'
bench_ndjson "array construct"         "-c" '[.name, .x]'
bench_ndjson ".[]"                     "-c" ".[]"
bench_ndjson "to_entries"              "-c" "to_entries"
bench_ndjson "keys"                    "-c" "keys"
bench_ndjson "keys_unsorted"           "-c" "keys_unsorted"
bench_ndjson "length"                  "-c" "length"
bench_ndjson 'has("x")'               "-c" 'has("x")'
bench_ndjson "type"                    "-c" "type"
bench_ndjson "del(.name)"             "-c" "del(.name)"

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
    echo "  (jaq repo not found at $BENCH_DIR — run: git clone https://github.com/01mf02/jaq /tmp/jaq-repo)"
fi

echo ""
echo "Times: user CPU seconds (best of $RUNS). Lower is better."
