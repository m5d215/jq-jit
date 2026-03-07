#!/bin/bash
# Benchmark jq-jit against jq, gojq, jaq
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
JQ_JIT2="${JQ_JIT2:-target/release/jq-jit}"
if [ ! -x "$JQ_JIT2" ]; then
    echo "Error: $JQ_JIT2 not found. Run: cargo build --release"
    exit 1
fi

TOOLS=("$JQ_JIT2")
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
    'field access::.name'
    'arithmetic::.x + .y'
    'select:-c:select(.x > 1500000)'
    'string concat:-c:.name + "_x"'
    'object construct:-c:{a: .x, b: .y}'
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
