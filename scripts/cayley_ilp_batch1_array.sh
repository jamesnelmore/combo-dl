#!/bin/bash
#SBATCH --job-name=cayley-b1
#SBATCH --time=72:00:00
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jnelmore@wm.edu
#SBATCH --output=slurm_logs/cayley_b1_%A_%a.out
#SBATCH --error=slurm_logs/cayley_b1_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs cayley_ilp_results

TASK=$(python3 -c "
import json
tasks = json.load(open('cayley_ilp_batch1_tasks.json'))
t = tasks[$SLURM_ARRAY_TASK_ID]
lam = t['lambda']
print(f'--n {t[\"n\"]} --k {t[\"k\"]} --t {t[\"t\"]} --lambda {lam} --mu {t[\"mu\"]} --lib-id {t[\"lib_id\"]}')
")

uv run python3 scripts/cayley_ilp_worker.py \
    $TASK \
    --linear \
    --output-dir cayley_ilp_results \
    --time-limit 259200 \
    --threads 32
