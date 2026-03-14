#!/bin/bash
# Quick daily benchmark — 17 NDJSON workloads, ~30s, colored output with ratios
# For thorough analysis (80+ patterns), use: bench/comprehensive.sh
# Usage: bench/run.sh [data_file]
set -e

DATAFILE="${1:-/tmp/bench_2m.json}"
RUNS=3

# Generate data if missing
if [ ! -f "$DATAFILE" ]; then
    echo "Data file not found. Generating..."
    bash "$(dirname "$0")/generate_data.sh" "$DATAFILE"
fi

COUNT=$(awk 'END{print NR}' "$DATAFILE")
echo "=== jq-jit Benchmark ($COUNT NDJSON objects) ==="
echo ""

# Detect available tools
JQ_JIT="${JQ_JIT:-target/release/jq-jit}"
if [ ! -x "$JQ_JIT" ]; then
    echo "Error: $JQ_JIT not found. Run: cargo build --release"
    exit 1
fi

TOOLS=("$JQ_JIT")
NAMES=("jq-jit")

for cmd in jq gojq jaq; do
    if command -v "$cmd" > /dev/null 2>&1; then
        TOOLS+=("$(command -v "$cmd")")
        case "$cmd" in
            jq)   NAMES+=("jq $($cmd --version 2>&1 | head -1)") ;;
            gojq) NAMES+=("gojq $(gojq --version 2>&1 | awk '{print $2}')") ;;
            jaq)  NAMES+=("jaq $(jaq --version 2>&1 | awk '{print $2}')") ;;
        esac
    fi
done

echo "Tools: ${NAMES[*]}"
echo "Data:  $DATAFILE ($COUNT NDJSON objects)"
echo "Runs:  best of $RUNS"
echo ""

# Run a benchmark: best user CPU time of $RUNS
bench() {
    local tool="$1"; shift
    local best=999
    for ((i=0; i<RUNS; i++)); do
        local t
        t=$( { TIMEFORMAT='%U'; time "$tool" "$@" "$DATAFILE" > /dev/null 2>&1; } 2>&1 )
        if (( $(echo "$t < $best" | bc -l) )); then
            best=$t
        fi
    done
    echo "$best"
}

# Header
printf "%-28s" "Workload"
for name in "${NAMES[@]}"; do
    printf "  %-12s" "$name"
done
echo ""
printf "%-28s" "----------------------------"
for name in "${NAMES[@]}"; do
    printf "  %-12s" "------------"
done
echo ""

# Tests — NDJSON format: each line is one JSON object
# Filters operate on individual objects (no .[] needed)
TESTS=(
    'empty::empty'
    'identity (.):-c:.'
    'identity (pretty)::.'
    'field access::.name'
    'arithmetic::.x + .y'
    'select:-c:select(.x > 1500000)'
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
    'tostring:-c:.x | tostring'
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
    'computed remap:-c:{name: .name, double: (.x * 2), sum: (.x + .y)}'
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
)

for test in "${TESTS[@]}"; do
    IFS=: read -r label flags filter <<< "$test"
    printf "%-28s" "$label"
    first=""
    for tool in "${TOOLS[@]}"; do
        t=$(bench "$tool" $flags "$filter")
        if [ -z "$first" ]; then
            first=$t
            printf "  \033[32m%-12s\033[0m" "${t}s"
        else
            ratio=$(echo "scale=1; $t / $first" | bc -l)
            printf "  %-12s" "${t}s (${ratio}x)"
        fi
    done
    echo ""
done

echo ""
echo "Times are user CPU seconds (best of $RUNS). Lower is better."
echo "Ratio is relative to jq-jit (higher = slower)."
