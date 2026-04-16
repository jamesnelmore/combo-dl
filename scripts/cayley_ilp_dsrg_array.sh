#!/bin/bash
#SBATCH --job-name=cayley-dsrg
#SBATCH --time=00:02:30
#SBATCH --output=slurm_logs/cayley_dsrg_%A_%a.out
#SBATCH --error=slurm_logs/cayley_dsrg_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs cayley_ilp_results

# Read task from JSON task list
TASK=$(python3 -c "
import json, sys
tasks = json.load(open('cayley_ilp_tasks.json'))
t = tasks[$SLURM_ARRAY_TASK_ID]
print(f'--n {t["n"]} --k {t["k"]} --t {t["t"]} --lambda {t["lambda"]} --mu {t["mu"]} --lib-id {t["lib_id"]}')
")

uv run python3 scripts/cayley_ilp_worker.py \
    $TASK \
    --output-dir cayley_ilp_results \
    --time-limit 60
