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
