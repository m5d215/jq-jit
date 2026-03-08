#!/bin/bash
# PGO (Profile-Guided Optimization) build for jq-jit
# Uses LLVM's PGO passes: instrument → profile → optimize
set -e

DATAFILE="${1:-/tmp/bench_2m.json}"
PGO_DIR="/tmp/pgo-data"

# Find llvm-profdata
PROFDATA=""
for p in \
    "$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's|host: ||p')/bin/llvm-profdata" \
    /Library/Developer/CommandLineTools/usr/bin/llvm-profdata \
    /usr/bin/llvm-profdata \
    llvm-profdata; do
    if command -v "$p" &>/dev/null || [ -x "$p" ]; then
        PROFDATA="$p"
        break
    fi
done
if [ -z "$PROFDATA" ]; then
    echo "Error: llvm-profdata not found"
    exit 1
fi

# Generate data if missing
if [ ! -f "$DATAFILE" ]; then
    echo "Generating benchmark data..."
    bash "$(dirname "$0")/generate_data.sh" "$DATAFILE"
fi

# Step 1: Instrumented build
echo "=== Step 1/4: Building with instrumentation ==="
rm -rf "$PGO_DIR"
mkdir -p "$PGO_DIR"
RUSTFLAGS="-Cprofile-generate=$PGO_DIR" cargo build --release

# Step 2: Run representative workloads
echo "=== Step 2/4: Collecting profile data ==="
JQ_JIT=target/release/jq-jit
"$JQ_JIT" -c .         "$DATAFILE" > /dev/null
"$JQ_JIT" '.name'      "$DATAFILE" > /dev/null
"$JQ_JIT" '.x + .y'    "$DATAFILE" > /dev/null
"$JQ_JIT" -c 'select(.x > 1500000)' "$DATAFILE" > /dev/null
"$JQ_JIT" -c '.name + "_x"'         "$DATAFILE" > /dev/null
"$JQ_JIT" -c '{a: .x, b: .y}'       "$DATAFILE" > /dev/null

# Step 3: Merge profiles
echo "=== Step 3/4: Merging profile data ==="
"$PROFDATA" merge -o "$PGO_DIR/merged.profdata" "$PGO_DIR"/*.profraw

# Step 4: PGO-optimized build
echo "=== Step 4/4: Building with PGO optimization ==="
RUSTFLAGS="-Cprofile-use=$PGO_DIR/merged.profdata" cargo build --release

echo ""
echo "PGO build complete: target/release/jq-jit"
echo "Run bench/run.sh to verify improvements."
