#!/bin/bash
# Generate benchmark data: 2M JSON objects (newline-delimited)
set -e

OUTFILE="${1:-/tmp/bench_2m.json}"
COUNT="${2:-2000000}"

echo "Generating $COUNT JSON objects to $OUTFILE ..."
python3 -c "
import json, sys
for i in range($COUNT):
    sys.stdout.write(json.dumps({'x': i, 'y': i * 2, 'name': 'item_' + str(i)}) + '\n')
" > "$OUTFILE"

SIZE=$(du -h "$OUTFILE" | cut -f1)
echo "Done: $OUTFILE ($SIZE, $COUNT objects)"
