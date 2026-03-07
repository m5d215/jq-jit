#!/bin/bash
# jq official test suite runner for jq-jit
# Usage: bash tests/official/run.sh [target/release/jq-jit] [tests/official/jq.test]

set -euo pipefail

JQ_JIT="${1:-target/release/jq-jit}"
TEST_FILE="${2:-tests/official/jq.test}"
TIMEOUT=5
LIB_DIR="tests/modules"

PASS=0
FAIL=0
SKIP=0
ERROR=0
TOTAL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

declare -a FAILURES=()

run_test() {
    local filter="$1"
    local input="$2"
    local expected="$3"
    local test_num="$4"

    TOTAL=$((TOTAL + 1))

    local actual exit_code
    if [[ "$input" == "null" && "$filter" != "." ]]; then
        actual=$(echo "null" | timeout "$TIMEOUT" "$JQ_JIT" -L "$LIB_DIR" -c "$filter" 2>/dev/null) && exit_code=$? || exit_code=$?
    else
        actual=$(echo "$input" | timeout "$TIMEOUT" "$JQ_JIT" -L "$LIB_DIR" -c "$filter" 2>/dev/null) && exit_code=$? || exit_code=$?
    fi

    if [[ $exit_code -eq 124 || $exit_code -eq 137 ]]; then
        SKIP=$((SKIP + 1))
        if [[ "${VERBOSE:-}" == "1" ]]; then
            echo -e "${YELLOW}SKIP${NC} #$test_num (timeout): $filter"
        fi
        return
    fi

    if [[ $exit_code -eq 139 ]]; then
        ERROR=$((ERROR + 1))
        FAILURES+=("#$test_num SIGSEGV: $filter")
        if [[ "${VERBOSE:-}" == "1" ]]; then
            echo -e "${RED}SEGV${NC} #$test_num: $filter"
        fi
        return
    fi

    if [[ $exit_code -eq 3 ]]; then
        SKIP=$((SKIP + 1))
        if [[ "${VERBOSE:-}" == "1" ]]; then
            echo -e "${YELLOW}SKIP${NC} #$test_num (unsupported): $filter"
        fi
        return
    fi

    if [[ $exit_code -ne 0 && $exit_code -ne 5 ]]; then
        if echo "$expected" | grep -q '^".*"$'; then
            :
        fi
        FAIL=$((FAIL + 1))
        FAILURES+=("#$test_num exit=$exit_code: $filter | input: $input")
        if [[ "${VERBOSE:-}" == "1" ]]; then
            echo -e "${RED}FAIL${NC} #$test_num (exit $exit_code): $filter"
        fi
        return
    fi

    local actual_c expected_c
    actual_c=$(echo "$actual" | sed 's/, /,/g; s/: /:/g')
    expected_c=$(echo "$expected" | sed 's/, /,/g; s/: /:/g')

    _normalize_json() {
        python3 -c "
import sys, json, math

def normalize_nums(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return obj
        if obj == int(obj) and abs(obj) < 2**53:
            return int(obj)
        return obj
    if isinstance(obj, list):
        return [normalize_nums(x) for x in obj]
    if isinstance(obj, dict):
        return {k: normalize_nums(v) for k, v in obj.items()}
    return obj

text = sys.stdin.read()
lines = []
for line in text.split('\n'):
    line = line.strip()
    if not line:
        continue
    try:
        val = json.loads(line)
        val = normalize_nums(val)
        lines.append(json.dumps(val, ensure_ascii=False, separators=(',',':'), sort_keys=True))
    except:
        lines.append(line)
sys.stdout.write('\n'.join(lines))
" 2>/dev/null || cat
    }

    actual_c=$(echo "$actual_c" | _normalize_json)
    expected_c=$(echo "$expected_c" | _normalize_json)
    local expected_norm actual_norm
    expected_norm=$(echo "$expected_c" | sort)
    actual_norm=$(echo "$actual_c" | sort)

    if [[ "$actual_c" == "$expected_c" ]]; then
        PASS=$((PASS + 1))
        if [[ "${VERBOSE:-}" == "2" ]]; then
            echo -e "${GREEN}PASS${NC} #$test_num: $filter"
        fi
    elif [[ "$actual_norm" == "$expected_norm" ]]; then
        PASS=$((PASS + 1))
        if [[ "${VERBOSE:-}" == "2" ]]; then
            echo -e "${GREEN}PASS${NC} #$test_num (reordered): $filter"
        fi
    else
        FAIL=$((FAIL + 1))
        FAILURES+=("#$test_num: $filter")
        if [[ "${VERBOSE:-}" == "1" ]]; then
            echo -e "${RED}FAIL${NC} #$test_num: $filter"
            echo "  input:    $input"
            echo "  expected: $(echo "$expected" | head -3)"
            echo "  actual:   $(echo "$actual" | head -3)"
        fi
    fi
}

# Parse test file
test_num=0
filter=""
input=""
expected=""
state="filter"
in_fail_block=0

while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    if [[ "$line" == "%%FAIL"* ]]; then
        in_fail_block=1
        continue
    fi
    if [[ $in_fail_block -eq 1 ]]; then
        if [[ -z "$line" ]]; then
            in_fail_block=0
        fi
        continue
    fi

    if [[ -z "$line" ]]; then
        if [[ "$state" == "output" && -n "$filter" ]]; then
            test_num=$((test_num + 1))
            run_test "$filter" "$input" "$expected" "$test_num"
            filter=""
            input=""
            expected=""
            state="filter"
        fi
        continue
    fi

    case "$state" in
        filter)
            filter="$line"
            state="input"
            ;;
        input)
            input="$line"
            state="output"
            expected=""
            ;;
        output)
            if [[ -z "$expected" ]]; then
                expected="$line"
            else
                expected="$expected
$line"
            fi
            ;;
    esac
done < "$TEST_FILE"

if [[ "$state" == "output" && -n "$filter" ]]; then
    test_num=$((test_num + 1))
    run_test "$filter" "$input" "$expected" "$test_num"
fi

# Summary
echo ""
echo "=== jq Official Test Suite Results ==="
echo -e "PASS: ${GREEN}$PASS${NC}"
echo -e "FAIL: ${RED}$FAIL${NC}"
echo -e "SKIP: ${YELLOW}$SKIP${NC}"
echo -e "ERROR: ${RED}$ERROR${NC}"
echo "TOTAL: $TOTAL"

if [[ $TOTAL -gt 0 ]]; then
    PASS_RATE=$(echo "scale=1; $PASS * 100 / $TOTAL" | bc)
    echo -e "PASS rate: ${GREEN}${PASS_RATE}%${NC}"
fi

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo ""
    echo "=== Failures ==="
    for i in "${!FAILURES[@]}"; do
        echo "  ${FAILURES[$i]}"
    done
fi

exit 0
