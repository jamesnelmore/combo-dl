#!/bin/bash
#SBATCH --job-name=cayley-c1
#SBATCH --time=06:00:00
#SBATCH -N 1
#SBATCH --output=slurm_logs/cayley_c1_%A_%a.out
#SBATCH --error=slurm_logs/cayley_c1_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs cayley_ilp_results

TASK=$(python3 -c "
import json
tasks = json.load(open('cayley_ilp_cluster1_tasks.json'))
t = tasks[$SLURM_ARRAY_TASK_ID]
print(f'--n {t[\"n\"]} --k {t[\"k\"]} --t {t[\"t\"]} --lambda {t[\"lambda\"]} --mu {t[\"mu\"]} --lib-id {t[\"lib_id\"]}')
")

uv run python3 scripts/cayley_ilp_worker.py \
    $TASK \
    --output-dir cayley_ilp_results \
    --time-limit 21300 \
    --threads 32
