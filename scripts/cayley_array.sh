#!/bin/bash
#SBATCH --job-name=cayley
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Cayley DSRG search: one parameter set per array task.
#
# Each task reads row $SLURM_ARRAY_TASK_ID from the params CSV, searches
# all nonabelian groups of order n for DSRGs, and saves results to
#   cayley_data/{n}_{k}_{t}_{lambda}_{mu}/
#
# Progress is tracked in a per-task CSV that's written before search
# starts and updated after each group completes, so it's easy to tell
# if a task finished or timed out.
#
# Usage:
#   sbatch --array=0-199 scripts/cayley_array.sh              # all 200 rows
#   sbatch --array=0-199%20 scripts/cayley_array.sh            # max 20 concurrent
#   PARAMS=my_params.csv sbatch --array=0-49 scripts/cayley_array.sh
#
# After completion, each task dir contains:
#   progress.csv              — one row per group, status/timing/counts
#   dsrg_{n}_{k}_{t}_{l}_{m}_g{id}.npz  — adjacency matrices (if any found)
#   slurm<jobID>.out/err     — stdout/stderr logs (one file per submission)

set -euo pipefail

# ── Project setup ─────────────────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

# ── Configurable parameters ───────────────────────────────────────────────
PARAMS="${PARAMS:-larger_params.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-cayley_data}"
BATCH_SIZE="${BATCH_SIZE:-100000}"

# ── Resolve task output directory from params CSV ─────────────────────────
# Read the row to figure out the directory name before redirecting output.
TASK_DIR=$(python -c "
import pandas as pd; from pathlib import Path
pf = Path('${PARAMS}')
df = pd.read_excel(pf) if pf.suffix in ('.xls','.xlsx','.xlsm','.xlsb','.ods') else pd.read_csv(pf)
df.columns = df.columns.str.strip()
for c in df.columns:
    if df[c].dtype == object: df[c] = df[c].str.strip()
df = df.dropna(subset=['n']).reset_index(drop=True)
r = df.iloc[${SLURM_ARRAY_TASK_ID}]
print(f'${OUTPUT_DIR}/{int(r.n)}_{int(r.k)}_{int(r.t)}_{int(r[\"lambda\"])}_{int(r.mu)}')
")
mkdir -p "${TASK_DIR}"
exec > "${TASK_DIR}/slurm${SLURM_ARRAY_JOB_ID}.out" 2> "${TASK_DIR}/slurm${SLURM_ARRAY_JOB_ID}.err"

# ── Logging ───────────────────────────────────────────────────────────────
echo "=== Cayley DSRG Search ==="
echo "Job ID:        ${SLURM_ARRAY_JOB_ID}"
echo "Task ID:       ${SLURM_ARRAY_TASK_ID}"
echo "Node:          $(hostname)"
echo "GPUs:          ${SLURM_GPUS:-none}"
echo "Params CSV:    ${PARAMS}"
echo "Output dir:    ${TASK_DIR}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Start:         $(date -Iseconds)"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────
python -u -m cayley_search.run_single \
    --params "${PARAMS}" \
    --index "${SLURM_ARRAY_TASK_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --noninteractive

echo ""
echo "End:           $(date -Iseconds)"
