#!/bin/bash
# Project Euler benchmark — compare jq-jit vs jq on Euler solutions
# Usage: bench/euler.sh
set -e

JQ_JIT="${JQ_JIT:-target/release/jq-jit}"
JQ="$(command -v jq 2>/dev/null || true)"
JAQ="$(command -v jaq 2>/dev/null || true)"
EULER_DIR="/Users/dmanabe/src/github.com/m5d215/projecteuler-jq/src"
RUNS=3

if [ ! -x "$JQ_JIT" ]; then
    echo "Error: $JQ_JIT not found. Run: cargo build --release"
    exit 1
fi

if [ ! -d "$EULER_DIR" ]; then
    echo "Error: Euler solutions not found at $EULER_DIR"
    exit 1
fi

echo "=== Project Euler Benchmark ==="
echo "Tools: jq-jit | jq $(jq --version 2>&1) | jaq $(jaq --version 2>&1 | awk '{print $2}')"
echo "Runs: best of $RUNS"
echo ""

printf "  %-12s  %-14s  %-14s  %-14s\n" "Problem" "jq-jit" "jq" "jaq"
printf "  %-12s  %-14s  %-14s  %-14s\n" "---" "---" "---" "---"

for f in $(ls "$EULER_DIR"/*.jq | grep -v lib.jq | sort -t/ -k9 -n 2>/dev/null || ls "$EULER_DIR"/[0-9]*.jq | sort -t/ -k9 -n); do
    num=$(basename "$f" .jq)
    lib_flag=""
    if grep -q 'import\|include' "$f" 2>/dev/null; then
        lib_flag="-L $EULER_DIR"
    fi
    printf "  Euler %-5s " "$num"
    for tool in "$JQ_JIT" "$JQ" "$JAQ"; do
        if [ -z "$tool" ]; then printf "  %-14s" "N/A"; continue; fi
        best=999 failed=0
        for ((i=0; i<RUNS; i++)); do
            t=$( { TIMEFORMAT='%U'; time echo 'null' | timeout 10 "$tool" $lib_flag -f "$f" > /dev/null 2>&1; } 2>&1 ) && rc=0 || rc=$?
            if [ $rc -ne 0 ]; then failed=1; break; fi
            if (( $(echo "$t < $best" | bc -l) )); then best=$t; fi
        done
        if [ $failed -eq 1 ]; then
            printf "  %-14s" "FAIL"
        else
            printf "  %-14s" "${best}s"
        fi
    done
    echo ""
done

echo ""
echo "Times: user CPU seconds (best of $RUNS). Lower is better."
