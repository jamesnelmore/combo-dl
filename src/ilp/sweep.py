import csv
import json
import time
from pathlib import Path

import pulp as pl

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dsrg import build_dsrg_lp

PARAMS_CSV = Path(__file__).parent / "dsrg_params.csv"
RESULTS_JSON = Path(__file__).parent / "sweep_results.json"


def load_params():
    with open(PARAMS_CSV) as f:
        return [
            (int(r["n"]), int(r["k"]), int(r["t"]), int(r["lambda"]), int(r["mu"]))
            for r in csv.DictReader(f)
        ]


def solve_one(n, k, t, lambda_param, mu):
    prob, _ = build_dsrg_lp(n, k, t, lambda_param, mu)
    t0 = time.perf_counter()
    prob.solve(pl.GUROBI(msg=False, threads=-1))
    elapsed = time.perf_counter() - t0
    status = pl.LpStatus[prob.status]
    return {"status": status, "wall_seconds": round(elapsed, 4)}


def main():
    results = []
    if RESULTS_JSON.exists():
        results = json.loads(RESULTS_JSON.read_text())

    done = {(r["n"], r["k"], r["t"], r["lambda"], r["mu"]) for r in results}

    for n, k, t, lam, mu in load_params():
        if (n, k, t, lam, mu) in done:
            print(f"  skip ({n},{k},{t},{lam},{mu}) — already done")
            continue

        print(f"Solving ({n},{k},{t},{lam},{mu}) ...", end=" ", flush=True)
        result = solve_one(n, k, t, lam, mu)
        result.update({"n": n, "k": k, "t": t, "lambda": lam, "mu": mu})
        results.append(result)
        print(f"{result['status']} in {result['wall_seconds']:.2f}s")

        RESULTS_JSON.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
