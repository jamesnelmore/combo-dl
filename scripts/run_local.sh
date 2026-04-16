#!/bin/bash
# Run Cayley ILP search locally. Uses quadratic formulation.
#
# Usage:
#   scripts/run_local.sh dsrg dsrg_parameters.csv    # directed, nonabelian only
#   scripts/run_local.sh srg srg_open_cases.csv       # undirected, all groups
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:?Usage: $0 <dsrg|srg> <params.csv>}"
PARAMS="${2:?Usage: $0 <dsrg|srg> <params.csv>}"
OUTDIR="cayley_ilp_results/${MODE}"
TIME_LIMIT=120

UNDIRECTED_FLAG=""
if [[ "$MODE" == "srg" ]]; then
    UNDIRECTED_FLAG="--undirected"
fi

mkdir -p "$OUTDIR"

echo "=== Cayley ILP local sweep ==="
echo "Mode:       $MODE"
echo "Params:     $PARAMS"
echo "Output:     $OUTDIR"
echo "Time limit: ${TIME_LIMIT}s per group"
echo ""

# Generate task list
uv run python3 scripts/cayley_ilp_submit.py "$PARAMS" $UNDIRECTED_FLAG --time-limit 00:02:30 --dry-run 2>&1 | head -20

# Read task list and run sequentially
uv run python3 -c "
import json, subprocess, sys, time

tasks = json.load(open('cayley_ilp_tasks.json'))
print(f'Running {len(tasks)} tasks sequentially\n')

found = 0
infeasible = 0
timeout = 0
errors = 0

for i, t in enumerate(tasks):
    label = f'({t[\"n\"]},{t[\"k\"]},{t[\"t\"]},{t[\"lambda\"]},{t[\"mu\"]}) g{t[\"lib_id\"]} {t[\"group_name\"]}'
    sys.stdout.write(f'[{i+1}/{len(tasks)}] {label} ... ')
    sys.stdout.flush()

    cmd = [
        'uv', 'run', 'python3', 'scripts/cayley_ilp_worker.py',
        '--n', str(t['n']), '--k', str(t['k']), '--t', str(t['t']),
        '--lambda', str(t['lambda']), '--mu', str(t['mu']),
        '--lib-id', str(t['lib_id']),
        '--output-dir', '$OUTDIR',
        '--time-limit', '$TIME_LIMIT',
        $( [[ -n "$UNDIRECTED_FLAG" ]] && echo "'--undirected'," )
    ]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=${TIME_LIMIT}+60)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f'ERROR ({elapsed:.1f}s)')
        errors += 1
        continue

    try:
        r = json.loads(result.stdout.strip().split('\n')[-1])
        status = r['status']
        if status == 'Optimal' and r.get('connection_set'):
            print(f'FOUND in {r[\"solve_seconds\"]:.1f}s')
            found += 1
        elif status == 'Infeasible':
            print(f'infeasible in {r[\"solve_seconds\"]:.1f}s')
            infeasible += 1
        elif status == 'TimeLimit':
            print(f'timeout in {r[\"solve_seconds\"]:.1f}s')
            timeout += 1
        else:
            print(f'{status} in {r[\"solve_seconds\"]:.1f}s')
    except Exception as e:
        print(f'parse error: {e}')
        errors += 1

print(f'\n{\"=\"*60}')
print(f'Done: {found} found, {infeasible} infeasible, {timeout} timeout, {errors} errors')
"

echo ""
echo "Aggregating results..."
uv run python3 scripts/cayley_ilp_aggregate.py "$OUTDIR"
