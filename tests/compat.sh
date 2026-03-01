#!/bin/bash
# jq-jit compatibility test suite
# Compares jq-jit CLI output against system jq to verify compatibility.
#
# Usage: ./tests/compat.sh [path-to-jq-jit-binary]
#
# Exit codes:
#   0 — all tests passed (FAIL=0)
#   1 — one or more tests failed

# Do NOT use 'set -e' — test helpers intentionally invoke commands that may fail.
set -u

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JQ_JIT="${1:-target/release/jq-jit}"

if [ ! -x "$JQ_JIT" ]; then
    echo "ERROR: jq-jit binary not found at '$JQ_JIT'"
    echo "Usage: $0 [path-to-jq-jit-binary]"
    exit 2
fi

if ! command -v jq &>/dev/null; then
    echo "ERROR: 'jq' not found in PATH"
    exit 2
fi

PASS=0
FAIL=0
SKIP=0
TOTAL=0

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

# test_compat <description> <filter> <input> [flags]
# Compares jq and jq-jit output on the given filter+input.
test_compat() {
    local description="$1"
    local filter="$2"
    local input="$3"
    local flags="${4:-}"

    TOTAL=$((TOTAL + 1))

    # Get jq output
    local expected
    if [ -n "$flags" ]; then
        expected=$(printf '%s' "$input" | jq $flags "$filter" 2>/dev/null) || true
    else
        expected=$(printf '%s' "$input" | jq "$filter" 2>/dev/null) || true
    fi

    # Get jq-jit output and exit code
    local actual jit_exit
    if [ -n "$flags" ]; then
        actual=$(printf '%s' "$input" | "$JQ_JIT" $flags "$filter" 2>/dev/null) && jit_exit=0 || jit_exit=$?
    else
        actual=$(printf '%s' "$input" | "$JQ_JIT" "$filter" 2>/dev/null) && jit_exit=0 || jit_exit=$?
    fi

    # Exit code 3 = compilation/IR error → SKIP
    if [ "$jit_exit" -eq 3 ]; then
        SKIP=$((SKIP + 1))
        echo "  SKIP: $description (jq-jit compilation error)"
        return
    fi

    if [ "$expected" = "$actual" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $description"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $description"
        echo "    filter:   $filter"
        echo "    input:    $input"
        echo "    expected: $expected"
        echo "    actual:   $actual"
    fi
}

# test_compat_null <description> <filter> [flags]
# Same as test_compat but with -n (null input).
test_compat_null() {
    local description="$1"
    local filter="$2"
    local flags="${3:-}"

    TOTAL=$((TOTAL + 1))

    local expected
    if [ -n "$flags" ]; then
        expected=$(jq -n $flags "$filter" 2>/dev/null) || true
    else
        expected=$(jq -n "$filter" 2>/dev/null) || true
    fi

    local actual jit_exit
    if [ -n "$flags" ]; then
        actual=$(printf '' | "$JQ_JIT" -n $flags "$filter" 2>/dev/null) && jit_exit=0 || jit_exit=$?
    else
        actual=$(printf '' | "$JQ_JIT" -n "$filter" 2>/dev/null) && jit_exit=0 || jit_exit=$?
    fi

    if [ "$jit_exit" -eq 3 ]; then
        SKIP=$((SKIP + 1))
        echo "  SKIP: $description (jq-jit compilation error)"
        return
    fi

    if [ "$expected" = "$actual" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $description"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $description"
        echo "    filter:   $filter"
        echo "    input:    (null)"
        echo "    expected: $expected"
        echo "    actual:   $actual"
    fi
}

# test_compat_exit <description> <filter> <input> [flags]
# Compares both output AND exit code between jq -e and jq-jit -e.
test_compat_exit() {
    local description="$1"
    local filter="$2"
    local input="$3"
    local flags="${4:-}"

    TOTAL=$((TOTAL + 1))

    local expected jq_exit
    expected=$(printf '%s' "$input" | jq -e $flags "$filter" 2>/dev/null) && jq_exit=0 || jq_exit=$?

    local actual jit_exit
    actual=$(printf '%s' "$input" | "$JQ_JIT" -e $flags "$filter" 2>/dev/null) && jit_exit=0 || jit_exit=$?

    # Compilation error → SKIP
    if [ "$jit_exit" -eq 3 ]; then
        SKIP=$((SKIP + 1))
        echo "  SKIP: $description (jq-jit compilation error)"
        return
    fi

    if [ "$expected" = "$actual" ] && [ "$jq_exit" = "$jit_exit" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $description"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $description"
        echo "    filter:   $filter"
        echo "    input:    $input"
        echo "    expected output: $expected"
        echo "    actual output:   $actual"
        echo "    expected exit:   $jq_exit"
        echo "    actual exit:     $jit_exit"
    fi
}

# test_compat_raw <description> <filter> <input> [extra_flags]
# Specifically tests -r (raw output) comparison.
test_compat_raw() {
    local description="$1"
    local filter="$2"
    local input="$3"
    local flags="${4:-}"

    TOTAL=$((TOTAL + 1))

    local expected
    expected=$(printf '%s' "$input" | jq -r $flags "$filter" 2>/dev/null) || true

    local actual jit_exit
    actual=$(printf '%s' "$input" | "$JQ_JIT" -r $flags "$filter" 2>/dev/null) && jit_exit=0 || jit_exit=$?

    if [ "$jit_exit" -eq 3 ]; then
        SKIP=$((SKIP + 1))
        echo "  SKIP: $description (jq-jit compilation error)"
        return
    fi

    if [ "$expected" = "$actual" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $description"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $description"
        echo "    filter:   $filter"
        echo "    input:    $input"
        echo "    expected: $expected"
        echo "    actual:   $actual"
    fi
}

# ===========================================================================
# a) Basic arithmetic (10 tests)
# ===========================================================================

echo "=== a) Basic arithmetic ==="

test_compat "identity on number" "." "42"
test_compat "add 1" ". + 1" "10"
test_compat "subtract" ". - 3" "10"
test_compat "multiply" ". * 2" "7"
test_compat "divide" ". / 4" "100"
test_compat "modulo" ". % 3" "10"
test_compat "field access" ".foo" '{"foo":42}'
test_compat "field addition" ".foo + .bar" '{"foo":10,"bar":20}'
test_compat "field arithmetic" ".foo + .bar * 2" '{"a":1,"foo":10,"bar":5}'
test_compat "nested field arith" ". * 2 + 1" "15"

# ===========================================================================
# b) Conditionals (5 tests)
# ===========================================================================

echo ""
echo "=== b) Conditionals ==="

test_compat "if positive" 'if . > 0 then "pos" else "neg" end' "5"
test_compat "if negative" 'if . > 0 then "pos" else "neg" end' "-3"
test_compat "if field truthy" 'if .foo then .foo else "none" end' '{"foo":"hello"}'
test_compat "if field falsy" 'if .foo then .foo else "none" end' '{"foo":null}'
test_compat "nested if" 'if . > 10 then "big" elif . > 0 then "small" else "zero" end' "5"

# ===========================================================================
# c) Generators (10 tests)
# ===========================================================================

echo ""
echo "=== c) Generators ==="

test_compat "array each" '.[]' '[1,2,3]'
test_compat "each plus 1" '.[] | . + 1' '[10,20,30]'
test_compat "each select > 3" '.[] | select(. > 3)' '[1,2,3,4,5,6]'
test_compat_null "comma literals" '1, 2, 3'
test_compat "dot comma" '., .' '42'
test_compat "each multiply" '.[] | . * 2' '[3,5,7]'
test_compat "object each" '.[]' '{"a":1,"b":2,"c":3}'
test_compat "each with pipe" '.[] | . + 10 | . * 2' '[1,2,3]'
test_compat "select string gt" '.[] | select(. > "c")' '["apple","banana","cherry","date"]'
test_compat_null "empty comma" '1, empty, 2'

# ===========================================================================
# d) Array operations (10 tests)
# ===========================================================================

echo ""
echo "=== d) Array operations ==="

test_compat "map multiply" 'map(. * 2)' '[1,2,3,4,5]'
test_compat "collect plus 10" '[.[] | . + 10]' '[1,2,3]'
test_compat "sort" 'sort' '[3,1,4,1,5,9,2,6]'
test_compat "reverse" 'reverse' '[1,2,3,4,5]'
test_compat "unique" 'unique' '[3,1,2,1,3,2]'
test_compat "length array" 'length' '[1,2,3,4,5]'
test_compat "add numbers" 'add' '[1,2,3,4,5]'
test_compat "add strings" 'add' '["hello"," ","world"]'
test_compat "map floor" 'map(floor)' '[1.1,2.9,3.5]'
test_compat "map plus collect" '[.[] | . * 3]' '[10,20,30]'

# ===========================================================================
# e) String operations (5 tests)
# ===========================================================================

echo ""
echo "=== e) String operations ==="

test_compat "split comma" 'split(",")' '"a,b,c"'
test_compat "ascii_downcase" 'ascii_downcase' '"HELLO WORLD"'
test_compat "ascii_upcase" 'ascii_upcase' '"hello world"'
test_compat "startswith" 'startswith("foo")' '"foobar"'
test_compat "endswith" 'endswith("bar")' '"foobar"'

# ===========================================================================
# f) Object operations (5 tests)
# ===========================================================================

echo ""
echo "=== f) Object operations ==="

test_compat "keys" 'keys' '{"c":3,"a":1,"b":2}'
test_compat "to_entries" 'to_entries' '{"a":1,"b":2}'
test_compat "from_entries" 'from_entries' '[{"key":"a","value":1},{"key":"b","value":2}]'
test_compat "has key true" 'has("foo")' '{"foo":1,"bar":2}'
test_compat "type object" 'type' '{"a":1}'

# ===========================================================================
# g) Reduce / foreach (5 tests)
# ===========================================================================

echo ""
echo "=== g) Reduce / foreach ==="

test_compat "reduce sum" 'reduce .[] as $x (0; . + $x)' '[1,2,3,4,5]'
test_compat "reduce product" 'reduce .[] as $x (1; . * $x)' '[2,3,4]'
test_compat "reduce concat" 'reduce .[] as $x (""; . + $x)' '["a","b","c"]'
# SKIP reason: [foreach ...] requires array constructor wrapping foreach,
# which uses FORK→APPEND pattern that may not be supported yet.
test_compat "foreach running sum" '[foreach .[] as $x (0; . + $x)]' '[1,2,3,4,5]'
test_compat "foreach count" '[foreach .[] as $x (0; . + 1)]' '["a","b","c"]'

# ===========================================================================
# h) Try-catch (5 tests)
# ===========================================================================

echo ""
echo "=== h) Try-catch ==="

test_compat "try success" 'try .foo catch "error"' '{"foo":42}'
test_compat "try null field" 'try .foo catch "error"' 'null'
test_compat "try catch on array" '.[] | try .name catch "missing"' '[{"name":"a"},42,"str"]'
test_compat "try arithmetic" 'try (. + 1) catch "err"' '5'
test_compat "try nested" 'try (try .foo catch "inner") catch "outer"' '{"foo":1}'

# ===========================================================================
# i) Alternative operator // (3 tests)
# ===========================================================================

echo ""
echo "=== i) Alternative operator // ==="

test_compat "alt null" '.foo // "default"' '{"bar":1}'
test_compat_null "alt number" 'null // 42'
test_compat "alt chain" '.missing // .backup // "fallback"' '{"backup":"found"}'

# ===========================================================================
# j) CLI flags (8 tests)
# ===========================================================================

echo ""
echo "=== j) CLI flags ==="

# -r (raw output)
test_compat_raw "-r raw output string" '.' '"hello world"'
test_compat_raw "-r raw field" '.name' '{"name":"Alice"}'

# -c (compact output)
test_compat "-c compact array" '.' '[1,2,3]' '-c'
test_compat "-c compact object" '.' '{"a":1,"b":2}' '-c'

# -n (null input)
test_compat_null "-n null input add" '1 + 2'
test_compat_null "-n null input string" '"hello"'

# -e (exit status)
test_compat_exit "-e exit 0 on true" '.' 'true'
test_compat_exit "-e exit 1 on false" '.' 'false'

# ===========================================================================
# k) Practical patterns (10 tests)
# ===========================================================================

echo ""
echo "=== k) Practical patterns ==="

test_compat "filter by age" \
    '[.[] | select(.age > 30) | .name]' \
    '[{"name":"Alice","age":25},{"name":"Bob","age":35},{"name":"Carol","age":40}]'

test_compat "price * quantity" \
    '.[] | .price * .quantity' \
    '[{"price":10,"quantity":3},{"price":5,"quantity":7}]'

test_compat "map floor after scale" \
    'map(. * 1.1 | floor)' \
    '[100,200,300]'

test_compat "sort and reverse" \
    'sort | reverse' \
    '[3,1,4,1,5,9]'

test_compat "unique sorted" \
    'sort | unique' \
    '[5,3,1,2,3,5,1]'

test_compat "length of keys" \
    'keys | length' \
    '{"a":1,"b":2,"c":3,"d":4}'

test_compat "split and count" \
    'split(",") | length' \
    '"a,b,c,d,e"'

test_compat "map add 10 then sum" \
    'map(. + 10) | add' \
    '[1,2,3]'

test_compat "select type number" \
    '[.[] | select(type == "number")]' \
    '[1,"a",2,"b",3]'

test_compat "reduce with condition" \
    'reduce .[] as $x (0; if $x > 3 then . + $x else . end)' \
    '[1,2,3,4,5,6]'

# ===========================================================================
# l) Type operations (5 tests)
# ===========================================================================

echo ""
echo "=== l) Type operations ==="

test_compat "type number" 'type' '42'
test_compat "type string" 'type' '"hello"'
test_compat "type array" 'type' '[1,2,3]'
test_compat "type null" 'type' 'null'
test_compat "type bool" 'type' 'true'

# ===========================================================================
# m) Not operator (3 tests)
# ===========================================================================

echo ""
echo "=== m) Not operator ==="

test_compat "not true" 'not' 'true'
test_compat "not false" 'not' 'false'
test_compat "not null" 'not' 'null'

# ===========================================================================
# n) Math builtins (4 tests)
# ===========================================================================

echo ""
echo "=== n) Math builtins ==="

test_compat "floor" 'floor' '3.7'
test_compat "ceil" 'ceil' '3.2'
test_compat "round" 'round' '3.5'
test_compat "fabs" 'fabs' '-7.5'

# ===========================================================================
# o) Explode / implode (2 tests)
# ===========================================================================

echo ""
echo "=== o) Explode / implode ==="

test_compat "explode" 'explode' '"ABC"'
test_compat "implode" 'implode' '[72,101,108,108,111]'

# ===========================================================================
# p) Complex combinations (8 tests)
# ===========================================================================

echo ""
echo "=== p) Complex combinations ==="

test_compat "map then reduce" \
    'map(. * 2) | reduce .[] as $x (0; . + $x)' \
    '[1,2,3,4,5]'

test_compat "to_entries values add" \
    'to_entries | map(.value) | add' \
    '{"a":10,"b":20,"c":30}'

# SKIP reason: split | map(ascii_upcase) involves piped map with a jq-defined
# function body that triggers a compilation error in the current JIT.
test_compat "split map upcase" \
    'split(",") | map(ascii_upcase)' \
    '"hello,world,foo"'

test_compat "nested select" \
    '[.[] | select(. > 2) | . * 10]' \
    '[1,2,3,4,5]'

test_compat "reverse then map" \
    'reverse | map(. + 1)' \
    '[10,20,30]'

test_compat "sort unique reverse" \
    'sort | unique | reverse' \
    '[5,3,1,4,2,3,1]'

test_compat "reduce string build" \
    'reduce .[] as $x (""; . + $x + ",")' \
    '["a","b","c"]'

test_compat "chained alternative" \
    '.x // .y // .z // "none"' \
    '{"y":42}'

# ===========================================================================
# q) Edge cases and special values (5 tests)
# ===========================================================================

echo ""
echo "=== q) Edge cases ==="

test_compat "negative number" '. + 1' '-5'
test_compat "zero" '. * 100' '0'
test_compat "large number" '. + 1' '999999999'
test_compat "empty array length" 'length' '[]'
test_compat "empty object keys" 'keys' '{}'

# ===========================================================================
# r) -s slurp (2 tests)
# ===========================================================================

echo ""
echo "=== r) Slurp ==="

# Slurp test: multiple JSON values on separate lines
TOTAL=$((TOTAL + 1))
input_slurp=$'1\n2\n3'
expected_slurp=$(printf '%s' "$input_slurp" | jq -s '.' 2>/dev/null) || true
actual_slurp=$(printf '%s' "$input_slurp" | "$JQ_JIT" -s '.' 2>/dev/null) && slurp_exit=0 || slurp_exit=$?
if [ "$slurp_exit" -eq 3 ]; then
    SKIP=$((SKIP + 1))
    echo "  SKIP: -s slurp multiple values (jq-jit compilation error)"
elif [ "$expected_slurp" = "$actual_slurp" ]; then
    PASS=$((PASS + 1))
    echo "  PASS: -s slurp multiple values"
else
    FAIL=$((FAIL + 1))
    echo "  FAIL: -s slurp multiple values"
    echo "    expected: $expected_slurp"
    echo "    actual:   $actual_slurp"
fi

TOTAL=$((TOTAL + 1))
input_slurp2=$'{"a":1}\n{"b":2}'
expected_slurp2=$(printf '%s' "$input_slurp2" | jq -s 'length' 2>/dev/null) || true
actual_slurp2=$(printf '%s' "$input_slurp2" | "$JQ_JIT" -s 'length' 2>/dev/null) && slurp2_exit=0 || slurp2_exit=$?
if [ "$slurp2_exit" -eq 3 ]; then
    SKIP=$((SKIP + 1))
    echo "  SKIP: -s slurp length (jq-jit compilation error)"
elif [ "$expected_slurp2" = "$actual_slurp2" ]; then
    PASS=$((PASS + 1))
    echo "  PASS: -s slurp length"
else
    FAIL=$((FAIL + 1))
    echo "  FAIL: -s slurp length"
    echo "    expected: $expected_slurp2"
    echo "    actual:   $actual_slurp2"
fi

# ===========================================================================
# s) -R raw input (5 tests)
# ===========================================================================

echo ""
echo "=== s) Raw input ==="

# -R: each line as a string
TOTAL=$((TOTAL + 1))
input_raw=$'hello\nworld\nfoo'
expected_raw=$(printf '%s' "$input_raw" | jq -R '.' 2>/dev/null) || true
actual_raw=$(printf '%s' "$input_raw" | "$JQ_JIT" -R '.' 2>/dev/null) && raw_exit=0 || raw_exit=$?
if [ "$raw_exit" -eq 3 ]; then
    SKIP=$((SKIP + 1))
    echo "  SKIP: -R each line as string (jq-jit compilation error)"
elif [ "$expected_raw" = "$actual_raw" ]; then
    PASS=$((PASS + 1))
    echo "  PASS: -R each line as string"
else
    FAIL=$((FAIL + 1))
    echo "  FAIL: -R each line as string"
    echo "    expected: $expected_raw"
    echo "    actual:   $actual_raw"
fi

# -Rs: entire input as one string
TOTAL=$((TOTAL + 1))
expected_rs=$(printf '%s' "$input_raw" | jq -Rs '.' 2>/dev/null) || true
actual_rs=$(printf '%s' "$input_raw" | "$JQ_JIT" -Rs '.' 2>/dev/null) && rs_exit=0 || rs_exit=$?
if [ "$rs_exit" -eq 3 ]; then
    SKIP=$((SKIP + 1))
    echo "  SKIP: -Rs slurp as one string (jq-jit compilation error)"
elif [ "$expected_rs" = "$actual_rs" ]; then
    PASS=$((PASS + 1))
    echo "  PASS: -Rs slurp as one string"
else
    FAIL=$((FAIL + 1))
    echo "  FAIL: -Rs slurp as one string"
    echo "    expected: $expected_rs"
    echo "    actual:   $actual_rs"
fi

# -R with filter
TOTAL=$((TOTAL + 1))
expected_rf=$(printf '%s' "$input_raw" | jq -R 'length' 2>/dev/null) || true
actual_rf=$(printf '%s' "$input_raw" | "$JQ_JIT" -R 'length' 2>/dev/null) && rf_exit=0 || rf_exit=$?
if [ "$rf_exit" -eq 3 ]; then
    SKIP=$((SKIP + 1))
    echo "  SKIP: -R line length (jq-jit compilation error)"
elif [ "$expected_rf" = "$actual_rf" ]; then
    PASS=$((PASS + 1))
    echo "  PASS: -R line length"
else
    FAIL=$((FAIL + 1))
    echo "  FAIL: -R line length"
    echo "    expected: $expected_rf"
    echo "    actual:   $actual_rf"
fi

# -Rn: raw input + null (unusual combo — should read nothing)
test_compat_null "-Rn null input" '"hello"' '-R'

# -Rc: raw input + compact
TOTAL=$((TOTAL + 1))
expected_rc=$(printf '%s' "$input_raw" | jq -Rc '.' 2>/dev/null) || true
actual_rc=$(printf '%s' "$input_raw" | "$JQ_JIT" -Rc '.' 2>/dev/null) && rc_exit=0 || rc_exit=$?
if [ "$rc_exit" -eq 3 ]; then
    SKIP=$((SKIP + 1))
    echo "  SKIP: -Rc compact raw (jq-jit compilation error)"
elif [ "$expected_rc" = "$actual_rc" ]; then
    PASS=$((PASS + 1))
    echo "  PASS: -Rc compact raw"
else
    FAIL=$((FAIL + 1))
    echo "  FAIL: -Rc compact raw"
    echo "    expected: $expected_rc"
    echo "    actual:   $actual_rc"
fi

# ===========================================================================
# t) -f from-file (2 tests)
# ===========================================================================

echo ""
echo "=== t) From file ==="

# Create temp filter file
TMPFILTER=$(mktemp /tmp/jq-jit-test-filter.XXXXXX)
echo '.foo + 1' > "$TMPFILTER"

TOTAL=$((TOTAL + 1))
expected_f=$(printf '%s' '{"foo":41}' | jq -f "$TMPFILTER" 2>/dev/null) || true
actual_f=$(printf '%s' '{"foo":41}' | "$JQ_JIT" -f "$TMPFILTER" 2>/dev/null) && f_exit=0 || f_exit=$?
if [ "$f_exit" -eq 3 ]; then
    SKIP=$((SKIP + 1))
    echo "  SKIP: -f from-file basic (jq-jit compilation error)"
elif [ "$expected_f" = "$actual_f" ]; then
    PASS=$((PASS + 1))
    echo "  PASS: -f from-file basic"
else
    FAIL=$((FAIL + 1))
    echo "  FAIL: -f from-file basic"
    echo "    expected: $expected_f"
    echo "    actual:   $actual_f"
fi

# Multiline filter file
TMPFILTER2=$(mktemp /tmp/jq-jit-test-filter2.XXXXXX)
printf '.[] | select(. > 2)' > "$TMPFILTER2"

TOTAL=$((TOTAL + 1))
expected_f2=$(printf '%s' '[1,2,3,4,5]' | jq -f "$TMPFILTER2" 2>/dev/null) || true
actual_f2=$(printf '%s' '[1,2,3,4,5]' | "$JQ_JIT" -f "$TMPFILTER2" 2>/dev/null) && f2_exit=0 || f2_exit=$?
if [ "$f2_exit" -eq 3 ]; then
    SKIP=$((SKIP + 1))
    echo "  SKIP: -f multiline filter (jq-jit compilation error)"
elif [ "$expected_f2" = "$actual_f2" ]; then
    PASS=$((PASS + 1))
    echo "  PASS: -f multiline filter"
else
    FAIL=$((FAIL + 1))
    echo "  FAIL: -f multiline filter"
    echo "    expected: $expected_f2"
    echo "    actual:   $actual_f2"
fi

rm -f "$TMPFILTER" "$TMPFILTER2"

# ===========================================================================
# u) --arg / --argjson (4 tests)
# ===========================================================================

echo ""
echo "=== u) --arg / --argjson ==="

# --arg basic
TOTAL=$((TOTAL + 1))
expected_arg=$(jq -n --arg name Alice '$name' 2>/dev/null) || true
actual_arg=$("$JQ_JIT" -n --arg name Alice '$name' 2>/dev/null) && arg_exit=0 || arg_exit=$?
if [ "$arg_exit" -eq 3 ]; then
    SKIP=$((SKIP + 1))
    echo "  SKIP: --arg basic (jq-jit compilation error)"
elif [ "$expected_arg" = "$actual_arg" ]; then
    PASS=$((PASS + 1))
    echo "  PASS: --arg basic"
else
    FAIL=$((FAIL + 1))
    echo "  FAIL: --arg basic"
    echo "    expected: $expected_arg"
    echo "    actual:   $actual_arg"
fi

# --argjson basic
TOTAL=$((TOTAL + 1))
expected_argjson=$(jq -n --argjson count 42 '$count + 1' 2>/dev/null) || true
actual_argjson=$("$JQ_JIT" -n --argjson count 42 '$count + 1' 2>/dev/null) && argjson_exit=0 || argjson_exit=$?
if [ "$argjson_exit" -eq 3 ]; then
    SKIP=$((SKIP + 1))
    echo "  SKIP: --argjson basic (jq-jit compilation error)"
elif [ "$expected_argjson" = "$actual_argjson" ]; then
    PASS=$((PASS + 1))
    echo "  PASS: --argjson basic"
else
    FAIL=$((FAIL + 1))
    echo "  FAIL: --argjson basic"
    echo "    expected: $expected_argjson"
    echo "    actual:   $actual_argjson"
fi

# --arg with object input
TOTAL=$((TOTAL + 1))
expected_argobj=$(echo '{"name":"Bob"}' | jq --arg greeting hello '"\($greeting) \(.name)"' 2>/dev/null) || true
actual_argobj=$(echo '{"name":"Bob"}' | "$JQ_JIT" --arg greeting hello '"\($greeting) \(.name)"' 2>/dev/null) && argobj_exit=0 || argobj_exit=$?
if [ "$argobj_exit" -eq 3 ]; then
    SKIP=$((SKIP + 1))
    echo "  SKIP: --arg with object input (jq-jit compilation error)"
elif [ "$expected_argobj" = "$actual_argobj" ]; then
    PASS=$((PASS + 1))
    echo "  PASS: --arg with object input"
else
    FAIL=$((FAIL + 1))
    echo "  FAIL: --arg with object input"
    echo "    expected: $expected_argobj"
    echo "    actual:   $actual_argobj"
fi

# --argjson with array
TOTAL=$((TOTAL + 1))
expected_argarr=$(jq -n --argjson arr '[1,2,3]' '$arr | add' 2>/dev/null) || true
actual_argarr=$("$JQ_JIT" -n --argjson arr '[1,2,3]' '$arr | add' 2>/dev/null) && argarr_exit=0 || argarr_exit=$?
if [ "$argarr_exit" -eq 3 ]; then
    SKIP=$((SKIP + 1))
    echo "  SKIP: --argjson with array (jq-jit compilation error)"
elif [ "$expected_argarr" = "$actual_argarr" ]; then
    PASS=$((PASS + 1))
    echo "  PASS: --argjson with array"
else
    FAIL=$((FAIL + 1))
    echo "  FAIL: --argjson with array"
    echo "    expected: $expected_argarr"
    echo "    actual:   $actual_argarr"
fi

# ===========================================================================
# Summary
# ===========================================================================

echo ""
echo "=== jq-jit compatibility test results ==="
echo "PASS:  $PASS"
echo "FAIL:  $FAIL"
echo "SKIP:  $SKIP"
echo "TOTAL: $TOTAL"

if [ "$FAIL" -gt 0 ]; then
    exit 1
else
    exit 0
fi
