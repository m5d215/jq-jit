#!/bin/bash
# Wall-clock comparison: jq (system) vs jq-jit
# Requires: hyperfine, jq, target/release/jq-jit
# Usage: bash benches/compare.sh

set -euo pipefail

JQ="${JQ:-jq}"
JQ_JIT="${JQ_JIT:-target/release/jq-jit}"
WARMUP=3
RUNS=50

echo "=== jq vs jq-jit Wall-Clock Comparison ==="
echo ""
echo "jq:     $($JQ --version 2>&1)"
echo "jq-jit: $JQ_JIT"
echo ""

# Generate test data files
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Small scalar input
echo '42' > "$TMPDIR/num.json"
echo '{"foo":10,"bar":20}' > "$TMPDIR/obj.json"
echo '"HELLO WORLD"' > "$TMPDIR/str.json"

# Array of 100 numbers
python3 -c "import json; print(json.dumps(list(range(100))))" > "$TMPDIR/arr100.json"

# Array of 1000 numbers
python3 -c "import json; print(json.dumps(list(range(1000))))" > "$TMPDIR/arr1k.json"

# Array of 10000 objects
python3 -c "
import json
objs = [{'name': f'user{i}', 'age': 20 + i % 60, 'score': i * 10 % 100} for i in range(10000)]
print(json.dumps(objs))
" > "$TMPDIR/objs10k.json"

# Large JSON (100K objects)
python3 -c "
import json
objs = [{'id': i, 'name': f'user{i}', 'age': 20 + i % 60, 'active': i % 3 == 0} for i in range(100000)]
print(json.dumps(objs))
" > "$TMPDIR/objs100k.json"

# CSV-like string
python3 -c "print('\"' + ','.join(chr(97 + i % 26) for i in range(100)) + '\"')" > "$TMPDIR/csv.json"

echo "Test data generated."
echo ""

run_bench() {
    local name="$1"
    local filter="$2"
    local input_file="$3"
    local extra_flags="${4:-}"

    echo "--- $name ---"
    echo "  filter: $filter"
    echo "  input:  $(basename "$input_file") ($(wc -c < "$input_file" | tr -d ' ') bytes)"
    echo ""

    hyperfine \
        --warmup "$WARMUP" \
        --runs "$RUNS" \
        --export-json "$TMPDIR/result_${name}.json" \
        -n "jq" "cat '$input_file' | $JQ -c $extra_flags '$filter' > /dev/null" \
        -n "jq-jit" "cat '$input_file' | $JQ_JIT -c $extra_flags '$filter' > /dev/null" \
        2>&1

    echo ""
}

# --- Benchmarks ---

# 1. Scalar
run_bench "scalar_add" ". + 1" "$TMPDIR/num.json"
run_bench "scalar_field" ".foo + .bar * 2" "$TMPDIR/obj.json"

# 2. Conditional
run_bench "conditional" "if . > 0 then . * 2 else . * -1 end" "$TMPDIR/num.json"

# 3. String
run_bench "string_downcase" "ascii_downcase" "$TMPDIR/str.json"
run_bench "string_split" 'split(",") | length' "$TMPDIR/csv.json"

# 4. Generator (100 elements)
run_bench "gen_each_100" ".[] | . + 1" "$TMPDIR/arr100.json"
run_bench "gen_select_100" ".[] | select(. > 50) | . * 2" "$TMPDIR/arr100.json"

# 5. Aggregation
run_bench "agg_map_100" "map(. * 2)" "$TMPDIR/arr100.json"
run_bench "agg_map_add_100" "map(. * 2) | add" "$TMPDIR/arr100.json"
run_bench "agg_sort_100" "sort | reverse" "$TMPDIR/arr100.json"

# 6. Large data (10K objects)
run_bench "large_name_10k" ".[] | .name" "$TMPDIR/objs10k.json"
run_bench "large_select_10k" "[.[] | select(.age > 30)]" "$TMPDIR/objs10k.json"
run_bench "large_length_10k" "length" "$TMPDIR/objs10k.json"

# 7. Very large data (100K objects)
run_bench "xlarge_length_100k" "length" "$TMPDIR/objs100k.json"
run_bench "xlarge_first_100k" ".[0]" "$TMPDIR/objs100k.json"
run_bench "xlarge_select_100k" "[.[] | select(.active)] | length" "$TMPDIR/objs100k.json"

# --- Summary ---
echo ""
echo "=== Summary ==="
echo ""

# Extract results from JSON
python3 -c "
import json, glob, os, sys

results = []
for f in sorted(glob.glob('$TMPDIR/result_*.json')):
    name = os.path.basename(f).replace('result_', '').replace('.json', '')
    with open(f) as fh:
        data = json.load(fh)

    times = {}
    for r in data['results']:
        times[r['command']] = r['median']

    if 'jq' in times and 'jq-jit' in times:
        jq_ms = times['jq'] * 1000
        jit_ms = times['jq-jit'] * 1000
        ratio = times['jq'] / times['jq-jit']
        results.append((name, jq_ms, jit_ms, ratio))

# Print table
print(f'| {\"Benchmark\":<25} | {\"jq (ms)\":>10} | {\"jq-jit (ms)\":>12} | {\"Speedup\":>10} |')
print(f'|{\"-\"*27}|{\"-\"*12}|{\"-\"*14}|{\"-\"*12}|')
for name, jq_ms, jit_ms, ratio in results:
    print(f'| {name:<25} | {jq_ms:>10.3f} | {jit_ms:>12.3f} | {ratio:>9.1f}x |')
" 2>&1

echo ""
echo "Note: Wall-clock times include process startup, JSON parsing, and output."
echo "jq-jit's JIT compilation overhead is included in every invocation."
