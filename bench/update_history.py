#!/usr/bin/env python3
"""Append a new version column to docs/benchmark-history.{tsv,md}.

Workflow:
    bench/update_history.py <version-label> [bench_output_file]

  - If a bench output file is given, parses it and appends rows to the TSV.
  - Otherwise, runs `bench/comprehensive.sh` first, captures its stdout,
    then parses + appends.
  - Always regenerates docs/benchmark-history.md as the last KEEP_LAST
    columns of the (now updated) TSV.

The TSV is the source of truth (long format,
`section / benchmark / version / time_seconds`); the markdown is a slim
human-readable view derived from it.
"""
import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TSV = ROOT / 'docs/benchmark-history.tsv'
MD = ROOT / 'docs/benchmark-history.md'
BENCH = ROOT / 'bench/comprehensive.sh'
KEEP_LAST = 5  # columns retained in the slim markdown

ROW_RE = re.compile(r'^  (.+?)\s{2,}(\S+)\s*$')
SECTION_RE = re.compile(r'^--- (.+) ---\s*$')


def parse_bench_output(text):
    """Yield (section, benchmark, time_seconds) for each data row."""
    section = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = SECTION_RE.match(line)
        if m:
            section = m.group(1)
            i += 3  # skip section header, "Benchmark / time" header, "--- / ---" rule
            while i < len(lines) and lines[i].strip():
                row = lines[i]
                rm = ROW_RE.match(row)
                if rm:
                    label = rm.group(1).rstrip()
                    val = rm.group(2)
                    if val.endswith('s'):
                        val = val[:-1]
                    elif val == 'FAIL/TIMEOUT':
                        val = ''
                    yield section, label, val
                i += 1
            continue
        i += 1


def append_to_tsv(rows, version):
    with TSV.open('a') as f:
        for section, bench, t in rows:
            f.write(f'{section}\t{bench}\t{version}\t{t}\n')


def regenerate_md():
    versions_order = []
    sections_order = []
    benchmarks_order = defaultdict(list)
    data = defaultdict(dict)  # (section, benchmark) -> {version: time}

    with TSV.open() as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # header
        for row in reader:
            section, bench, version, t = row
            if version not in versions_order:
                versions_order.append(version)
            if section not in sections_order:
                sections_order.append(section)
            if bench not in benchmarks_order[section]:
                benchmarks_order[section].append(bench)
            data[(section, bench)][version] = t

    slim_versions = versions_order[-KEEP_LAST:]

    def fmt_time(t):
        return f'{t}s' if t else '-'

    bench_w = max(
        (len(b) for s in sections_order for b in benchmarks_order[s]),
        default=10,
    ) + 2

    col_w = []
    for v in slim_versions:
        w = len(v)
        for times in data.values():
            if v in times:
                w = max(w, len(fmt_time(times[v])))
        col_w.append(w + 1)

    def fmt_row(label, values):
        parts = [f'  {label:<{bench_w}}']
        for w, v in zip(col_w, values):
            parts.append(f'{v:<{w}} ')
        return ''.join(parts).rstrip()

    out = [
        '# Benchmark History',
        '',
        f'Recent slice (last {KEEP_LAST} columns). Full history lives in',
        '[`benchmark-history.tsv`](benchmark-history.tsv) (long format,',
        '`section / benchmark / version / time_seconds`).',
        '',
        '```text',
    ]
    for idx, section in enumerate(sections_order):
        if idx > 0:
            out.append('')
        out.append(f'--- {section} ---')
        out.append(fmt_row('Benchmark', slim_versions))
        out.append(fmt_row('---', ['-' * (w - 1) for w in col_w]))
        for bench in benchmarks_order[section]:
            times = data[(section, bench)]
            values = [fmt_time(times.get(v, '')) for v in slim_versions]
            out.append(fmt_row(bench, values))
    out.append('```')

    MD.write_text('\n'.join(out) + '\n')
    return len(versions_order), slim_versions


def run_bench():
    print(f'Running {BENCH} (this can take ~10 minutes) ...', file=sys.stderr)
    result = subprocess.run(
        [str(BENCH)], capture_output=True, text=True, cwd=ROOT,
    )
    if result.returncode != 0:
        sys.exit(f'comprehensive.sh exited {result.returncode}\n{result.stderr}')
    return result.stdout


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: update_history.py <version> [bench_output_file]')
    version = sys.argv[1]
    if len(sys.argv) >= 3:
        text = Path(sys.argv[2]).read_text()
    else:
        text = run_bench()

    rows = list(parse_bench_output(text))
    if not rows:
        sys.exit('no rows parsed from bench output')
    append_to_tsv(rows, version)
    print(f'Appended {len(rows)} rows for version {version}', file=sys.stderr)

    total, slim = regenerate_md()
    print(f'Regenerated {MD.relative_to(ROOT)} '
          f'(last {KEEP_LAST} of {total} versions: {slim})', file=sys.stderr)


if __name__ == '__main__':
    main()
