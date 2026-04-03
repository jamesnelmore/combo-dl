#!/usr/bin/env python3
"""Run Cayley ILP search locally. One solve per (param_set, group), 2 min each.

Usage:
    uv run python3 scripts/run_local.py dsrg dsrg_parameters.csv [--filter-n 48] [--filter-status open]
    uv run python3 scripts/run_local.py srg srg_open_cases.csv
"""
import json
import subprocess
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else None
PARAMS = sys.argv[2] if len(sys.argv) > 2 else None
if MODE not in ("dsrg", "srg") or not PARAMS:
    print(f"Usage: {sys.argv[0]} <dsrg|srg> <params.csv> [--filter-n N] [--filter-status S]")
    sys.exit(1)

# Collect extra flags to pass through to submit script
EXTRA_FLAGS = sys.argv[3:]

OUTDIR = f"cayley_ilp_results/{MODE}"
TIME_LIMIT = 120
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

# Generate task list via submit script (dry run)
undirected = ["--undirected"] if MODE == "srg" else []
subprocess.run(
    ["uv", "run", "python3", "scripts/cayley_ilp_submit.py", PARAMS]
    + undirected
    + ["--time-limit", "00:02:30", "--dry-run"]
    + EXTRA_FLAGS,
    check=True,
)

tasks = json.loads(Path("cayley_ilp_tasks.json").read_text())
print(f"\nRunning {len(tasks)} tasks sequentially, {TIME_LIMIT}s limit each\n")

found = 0
infeasible = 0
timeout = 0
errors = 0

for i, t in enumerate(tasks):
    label = f"({t['n']},{t['k']},{t['t']},{t['lambda']},{t['mu']}) g{t['lib_id']} {t['group_name']}"
    print(f"[{i+1}/{len(tasks)}] {label} ... ", end="", flush=True)

    cmd = [
        "uv", "run", "python3", "scripts/cayley_ilp_worker.py",
        "--n", str(t["n"]), "--k", str(t["k"]), "--t", str(t["t"]),
        "--lambda", str(t["lambda"]), "--mu", str(t["mu"]),
        "--lib-id", str(t["lib_id"]),
        "--output-dir", OUTDIR,
        "--time-limit", str(TIME_LIMIT),
    ]
    if MODE == "srg":
        cmd.append("--undirected")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIME_LIMIT + 60)
    except subprocess.TimeoutExpired:
        print("killed (subprocess timeout)")
        errors += 1
        continue

    if result.returncode != 0:
        print(f"ERROR (rc={result.returncode})")
        if result.stderr:
            print(f"  {result.stderr.strip()[:200]}")
        errors += 1
        continue

    try:
        # Last line of stdout is the JSON result
        r = json.loads(result.stdout.strip().split("\n")[-1])
        status = r["status"]
        if status == "Optimal" and r.get("connection_set"):
            orig = r.get("connection_set_original_indices", [])
            print(f"FOUND in {r['solve_seconds']:.1f}s  S={{{','.join(str(x) for x in orig[:8])}{',...' if len(orig)>8 else ''}}}")
            found += 1
        elif status == "Infeasible":
            print(f"infeasible in {r['solve_seconds']:.1f}s")
            infeasible += 1
        elif status == "TimeLimit":
            print(f"timeout in {r['solve_seconds']:.1f}s")
            timeout += 1
        else:
            print(f"{status} in {r.get('solve_seconds', '?')}s")
    except Exception as e:
        print(f"parse error: {e}")
        errors += 1

print(f"\n{'='*60}")
print(f"Done: {found} found, {infeasible} infeasible, {timeout} timeout, {errors} errors")
print(f"\nAggregating...")
subprocess.run(["uv", "run", "python3", "scripts/cayley_ilp_aggregate.py", OUTDIR])
