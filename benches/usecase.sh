#!/bin/bash
# Real-world use case benchmarks: jq vs jq-jit
# UC1: Log analysis (100K NDJSON)
# UC2: API response processing (50K objects)
# UC3: Batch ETL (100 files x 1000 records)
# Requires: hyperfine, jq, target/release/jq-jit, python3
# Usage: bash benches/usecase.sh

set -euo pipefail

JQ="${JQ:-jq}"
JQ_JIT="${JQ_JIT:-target/release/jq-jit}"

# Validate binaries
command -v "$JQ" >/dev/null 2>&1 || { echo "Error: jq not found at '$JQ'" >&2; exit 1; }
[[ -x "$JQ_JIT" ]] || { echo "Error: jq-jit not found at '$JQ_JIT' (run: cargo build --release)" >&2; exit 1; }
command -v hyperfine >/dev/null 2>&1 || { echo "Error: hyperfine is not installed" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is not installed" >&2; exit 1; }

echo "=== Use Case Benchmark: jq vs jq-jit ==="
echo ""
echo "jq:     $($JQ --version 2>&1)"
echo "jq-jit: $JQ_JIT"
echo ""

TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/jqjit-usecase.XXXXXX")
trap 'rm -rf "$TMPDIR"' EXIT

# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

echo "Generating test data..."

echo "  [1/3] UC1: 100K NDJSON log entries..."
python3 -c "
import json, random, hashlib
levels = ['info'] * 70 + ['warn'] * 15 + ['error'] * 10 + ['debug'] * 5
services = ['api-gateway', 'auth-service', 'user-service', 'payment-service', 'notification-service']
endpoints = ['/v1/users', '/v1/orders', '/v1/payments', '/v1/auth/login', '/v1/notifications']
methods = ['GET', 'POST', 'PUT', 'DELETE']
messages = {
    'info': ['request completed', 'cache hit', 'health check ok'],
    'warn': ['slow query detected', 'rate limit approaching', 'retry attempt'],
    'error': ['upstream timeout', 'connection refused', 'internal server error'],
    'debug': ['parsing request body', 'validating token', 'building response']
}
random.seed(42)
for i in range(100000):
    level = random.choice(levels)
    svc = random.choice(services)
    entry = {
        'timestamp': f'2026-03-01T{10 + i // 3600:02d}:{(i // 60) % 60:02d}:{i % 60:02d}Z',
        'level': level,
        'service': svc,
        'request_id': hashlib.md5(f'{i}'.encode()).hexdigest()[:12],
        'duration_ms': random.randint(1, 50) if level != 'error' else random.randint(100, 5000),
        'status': 200 if level in ('info', 'debug') else (429 if level == 'warn' else random.choice([500, 502, 503])),
        'message': random.choice(messages[level]),
        'metadata': {
            'endpoint': random.choice(endpoints),
            'method': random.choice(methods),
            'user_id': f'u-{random.randint(1, 10000):05d}'
        }
    }
    print(json.dumps(entry, separators=(',', ':')))
" > "$TMPDIR/logs.ndjson"

echo "  [2/3] UC2: 50K user objects (single JSON array)..."
python3 -c "
import json, random
random.seed(42)
departments = ['engineering', 'product', 'design', 'marketing', 'sales', 'support', 'hr', 'finance']
roles = ['junior', 'mid', 'senior', 'lead', 'principal', 'manager', 'director']
skills_pool = ['rust', 'go', 'python', 'typescript', 'java', 'kotlin', 'swift', 'c++', 'sql', 'react', 'vue', 'aws', 'gcp', 'docker', 'k8s']
project_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa']

users = []
for i in range(50000):
    dept = random.choice(departments)
    n_skills = random.randint(1, 6)
    n_projects = random.randint(0, 4)
    users.append({
        'id': i + 1,
        'name': f'user_{i+1:05d}',
        'email': f'user{i+1}@example.com',
        'profile': {
            'department': dept,
            'role': random.choice(roles),
            'joined': f'{random.randint(2020, 2025)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
            'skills': random.sample(skills_pool, n_skills)
        },
        'activity': {
            'last_login': f'2026-{random.randint(1,2):02d}-{random.randint(1,28):02d}',
            'login_count': random.randint(0, 500),
            'projects': [{'name': random.choice(project_names), 'role': random.choice(['lead', 'member', 'reviewer'])} for _ in range(n_projects)]
        }
    })
print(json.dumps(users, separators=(',', ':')))
" > "$TMPDIR/users.json"

echo "  [3/3] UC3: 100 batch files (1000 records each)..."
BATCH_DIR="$TMPDIR/batches"
python3 -c "
import json, random, os, sys
random.seed(42)
outdir = sys.argv[1]
os.makedirs(outdir, exist_ok=True)
skus = [f'{chr(65+i)}-{j:03d}' for i in range(5) for j in range(1, 21)]
types = ['purchase'] * 60 + ['refund'] * 20 + ['exchange'] * 20

for fnum in range(100):
    records = []
    for i in range(1000):
        rid = fnum * 1000 + i + 1
        t = random.choice(types)
        n_items = random.randint(1, 5)
        records.append({
            'id': rid,
            'type': t,
            'amount': random.randint(100, 50000),
            'currency': random.choice(['JPY', 'USD', 'EUR']),
            'customer_id': f'c-{random.randint(1, 5000):04d}',
            'items': [{'sku': random.choice(skus), 'qty': random.randint(1, 10), 'price': random.randint(50, 5000)} for _ in range(n_items)]
        })
    with open(os.path.join(outdir, f'batch_{fnum:03d}.json'), 'w') as f:
        json.dump(records, f, separators=(',', ':'))
" "$BATCH_DIR"

echo ""
echo "Test data sizes:"
echo "  logs.ndjson:   $(wc -c < "$TMPDIR/logs.ndjson" | tr -d ' ') bytes ($(wc -l < "$TMPDIR/logs.ndjson" | tr -d ' ') lines)"
echo "  users.json:    $(wc -c < "$TMPDIR/users.json" | tr -d ' ') bytes"
echo "  batches/:      $(ls "$BATCH_DIR" | wc -l | tr -d ' ') files"
echo ""

# ---------------------------------------------------------------------------
# UC1: Log Analysis (NDJSON)
# ---------------------------------------------------------------------------

echo "========================================="
echo " UC1: Log Analysis (100K NDJSON entries)"
echo "========================================="
echo ""

WARMUP=3
RUNS=50

# UC1-1: Select errors
echo "--- log_select_error ---"
echo '  filter: select(.level == "error")'
echo ""
hyperfine \
    --warmup "$WARMUP" --runs "$RUNS" \
    --export-json "$TMPDIR/result_uc1_select_error.json" \
    -n "jq" "$JQ -c 'select(.level == \"error\")' $TMPDIR/logs.ndjson > /dev/null" \
    -n "jq-jit" "$JQ_JIT -c 'select(.level == \"error\")' $TMPDIR/logs.ndjson > /dev/null" \
    2>&1
echo ""

# UC1-2: Select slow errors
echo "--- log_select_slow_error ---"
echo '  filter: select(.level == "error" and .duration_ms > 1000)'
echo ""
hyperfine \
    --warmup "$WARMUP" --runs "$RUNS" \
    --export-json "$TMPDIR/result_uc1_select_slow_error.json" \
    -n "jq" "$JQ -c 'select(.level == \"error\" and .duration_ms > 1000)' $TMPDIR/logs.ndjson > /dev/null" \
    -n "jq-jit" "$JQ_JIT -c 'select(.level == \"error\" and .duration_ms > 1000)' $TMPDIR/logs.ndjson > /dev/null" \
    2>&1
echo ""

# UC1-3: Extract error details
echo "--- log_extract_errors ---"
echo '  filter: select(.level == "error") | {timestamp, service, message, endpoint: .metadata.endpoint}'
echo ""
hyperfine \
    --warmup "$WARMUP" --runs "$RUNS" \
    --export-json "$TMPDIR/result_uc1_extract_errors.json" \
    -n "jq" "$JQ -c 'select(.level == \"error\") | {timestamp, service, message, endpoint: .metadata.endpoint}' $TMPDIR/logs.ndjson > /dev/null" \
    -n "jq-jit" "$JQ_JIT -c 'select(.level == \"error\") | {timestamp, service, message, endpoint: .metadata.endpoint}' $TMPDIR/logs.ndjson > /dev/null" \
    2>&1
echo ""

# ---------------------------------------------------------------------------
# UC2: API Response Processing (single large JSON array)
# ---------------------------------------------------------------------------

echo "========================================="
echo " UC2: API Response (50K user objects)"
echo "========================================="
echo ""

# UC2-1: Flatten
echo "--- api_flatten ---"
echo '  filter: [.[] | {name, department: .profile.department, skill_count: (.profile.skills | length)}]'
echo ""
hyperfine \
    --warmup "$WARMUP" --runs "$RUNS" \
    --export-json "$TMPDIR/result_uc2_flatten.json" \
    -n "jq" "$JQ -c '[.[] | {name, department: .profile.department, skill_count: (.profile.skills | length)}]' $TMPDIR/users.json > /dev/null" \
    -n "jq-jit" "$JQ_JIT -c '[.[] | {name, department: .profile.department, skill_count: (.profile.skills | length)}]' $TMPDIR/users.json > /dev/null" \
    2>&1
echo ""

# UC2-2: Select and transform
echo "--- api_select_transform ---"
echo '  filter: [.[] | select(.activity.login_count > 100) | {name, logins: .activity.login_count, projects: (.activity.projects | map(.name))}]'
echo ""
hyperfine \
    --warmup "$WARMUP" --runs "$RUNS" \
    --export-json "$TMPDIR/result_uc2_select_transform.json" \
    -n "jq" "$JQ -c '[.[] | select(.activity.login_count > 100) | {name, logins: .activity.login_count, projects: (.activity.projects | map(.name))}]' $TMPDIR/users.json > /dev/null" \
    -n "jq-jit" "$JQ_JIT -c '[.[] | select(.activity.login_count > 100) | {name, logins: .activity.login_count, projects: (.activity.projects | map(.name))}]' $TMPDIR/users.json > /dev/null" \
    2>&1
echo ""

# UC2-3: Group and aggregate
echo "--- api_group_aggregate ---"
echo '  filter: group_by(.profile.department) | map({dept: .[0].profile.department, count: length, avg_logins: (map(.activity.login_count) | add / length)})'
echo ""
hyperfine \
    --warmup "$WARMUP" --runs "$RUNS" \
    --export-json "$TMPDIR/result_uc2_group_aggregate.json" \
    -n "jq" "$JQ -c 'group_by(.profile.department) | map({dept: .[0].profile.department, count: length, avg_logins: (map(.activity.login_count) | add / length)})' $TMPDIR/users.json > /dev/null" \
    -n "jq-jit" "$JQ_JIT -c 'group_by(.profile.department) | map({dept: .[0].profile.department, count: length, avg_logins: (map(.activity.login_count) | add / length)})' $TMPDIR/users.json > /dev/null" \
    2>&1
echo ""

# ---------------------------------------------------------------------------
# UC3: Batch ETL (100 files, same filter)
# ---------------------------------------------------------------------------

echo "========================================="
echo " UC3: Batch ETL (100 files x 1000 records)"
echo "========================================="
echo ""

ETL_FILTER='[.[] | select(.type == "purchase" and .amount > 1000) | {id, amount, items: (.items | length)}]'

# Wrapper scripts with full binary paths substituted
cat > "$TMPDIR/run_jq.sh" <<SCRIPT
for f in "$BATCH_DIR"/batch_*.json; do
    $JQ -c '$ETL_FILTER' "\$f" > /dev/null
done
SCRIPT

cat > "$TMPDIR/run_jqjit.sh" <<SCRIPT
for f in "$BATCH_DIR"/batch_*.json; do
    $JQ_JIT -c '$ETL_FILTER' "\$f" > /dev/null
done
SCRIPT

cat > "$TMPDIR/run_jqjit_nocache.sh" <<SCRIPT
for f in "$BATCH_DIR"/batch_*.json; do
    $JQ_JIT -c --no-cache '$ETL_FILTER' "\$f" > /dev/null
done
SCRIPT

# Clear any existing cache before benchmark
$JQ_JIT --clear-cache 2>/dev/null || true

echo "--- batch_etl ---"
echo "  filter: $ETL_FILTER"
echo "  files:  100 x 1000 records"
echo ""
hyperfine \
    --warmup 1 --runs 10 \
    --export-json "$TMPDIR/result_uc3_batch_etl.json" \
    -n "jq" "bash $TMPDIR/run_jq.sh" \
    -n "jq-jit (cache)" "bash $TMPDIR/run_jqjit.sh" \
    -n "jq-jit (no-cache)" "bash $TMPDIR/run_jqjit_nocache.sh" \
    2>&1
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "=== Use Case Benchmark Results ==="
echo ""

python3 -c "
import json, os, sys

tmpdir = '$TMPDIR'

def load_result(filename):
    path = os.path.join(tmpdir, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def get_times(data):
    \"\"\"Return dict of command_name -> median_ms.\"\"\"
    times = {}
    for r in data['results']:
        times[r['command']] = r['median'] * 1000
    return times

# UC1: Log Analysis
print('## UC1: Log Analysis (100K NDJSON entries)')
print('')
print(f'| {\"Benchmark\":<25} | {\"jq (ms)\":>10} | {\"jq-jit (ms)\":>12} | {\"Speedup\":>10} |')
print(f'|{\"-\"*27}|{\"-\"*12}|{\"-\"*14}|{\"-\"*12}|')

uc1_files = [
    ('result_uc1_select_error.json', 'select_error'),
    ('result_uc1_select_slow_error.json', 'select_slow_error'),
    ('result_uc1_extract_errors.json', 'extract_errors'),
]
for filename, label in uc1_files:
    data = load_result(filename)
    if data is None:
        continue
    t = get_times(data)
    jq_ms = t.get('jq', 0)
    jit_ms = t.get('jq-jit', 0)
    ratio = jq_ms / jit_ms if jit_ms > 0 else 0
    print(f'| {label:<25} | {jq_ms:>10.1f} | {jit_ms:>12.1f} | {ratio:>9.1f}x |')

print('')

# UC2: API Response Processing
print('## UC2: API Response Processing (50K objects)')
print('')
print(f'| {\"Benchmark\":<25} | {\"jq (ms)\":>10} | {\"jq-jit (ms)\":>12} | {\"Speedup\":>10} |')
print(f'|{\"-\"*27}|{\"-\"*12}|{\"-\"*14}|{\"-\"*12}|')

uc2_files = [
    ('result_uc2_flatten.json', 'flatten'),
    ('result_uc2_select_transform.json', 'select_transform'),
    ('result_uc2_group_aggregate.json', 'group_aggregate'),
]
for filename, label in uc2_files:
    data = load_result(filename)
    if data is None:
        continue
    t = get_times(data)
    jq_ms = t.get('jq', 0)
    jit_ms = t.get('jq-jit', 0)
    ratio = jq_ms / jit_ms if jit_ms > 0 else 0
    print(f'| {label:<25} | {jq_ms:>10.1f} | {jit_ms:>12.1f} | {ratio:>9.1f}x |')

print('')

# UC3: Batch ETL
print('## UC3: Batch ETL (100 files x 1000 records)')
print('')
print(f'| {\"Benchmark\":<25} | {\"jq (ms)\":>10} | {\"jq-jit (ms)\":>12} | {\"no-cache (ms)\":>14} | {\"Cache Speedup\":>14} | {\"vs jq\":>10} |')
print(f'|{\"-\"*27}|{\"-\"*12}|{\"-\"*14}|{\"-\"*16}|{\"-\"*16}|{\"-\"*12}|')

data = load_result('result_uc3_batch_etl.json')
if data is not None:
    t = get_times(data)
    jq_ms = t.get('jq', 0)
    jit_cache_ms = t.get('jq-jit (cache)', 0)
    jit_nocache_ms = t.get('jq-jit (no-cache)', 0)
    cache_speedup = jit_nocache_ms / jit_cache_ms if jit_cache_ms > 0 else 0
    vs_jq = jq_ms / jit_cache_ms if jit_cache_ms > 0 else 0
    print(f'| {\"batch_etl\":<25} | {jq_ms:>10.1f} | {jit_cache_ms:>12.1f} | {jit_nocache_ms:>14.1f} | {cache_speedup:>13.1f}x | {vs_jq:>9.1f}x |')

print('')
" 2>&1

echo "Note: All times are median wall-clock (ms) including process startup, parsing, and output."
echo "UC3 times represent the full loop over 100 files."
