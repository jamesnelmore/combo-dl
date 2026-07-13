#!/bin/bash
#SBATCH --job-name=cayley-dsrg
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs/cayley_dsrg_%A_%a.out
#SBATCH --error=slurm_logs/cayley_dsrg_%A_%a.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=email@jameselmore.org

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

TASK_FILE="${TASK_FILE:-cayley_ilp_tasks.json}"
OUTPUT_DIR="${OUTPUT_DIR:-cayley_ilp_results-post-thesis}"
BATCH_SIZE="${BATCH_SIZE:-1}"
# TASK_OFFSET lets us cover >999 tasks across multiple array submissions
# (Slurm caps array indices at 999). Each submission handles a contiguous block.
TASK_OFFSET="${TASK_OFFSET:-0}"
NTASKS=$(python3 -c "import json; print(len(json.load(open('$TASK_FILE'))))")
# Gurobi time limit: keep the full Slurm wall per ILP (minus 5 min overhead).
# Derive from the actual allocation so it tracks --time without hardcoding.
SLURM_SECS=$(( ${SLURM_JOB_END_TIME:-0} - ${SLURM_JOB_START_TIME:-0} ))
if [ "$SLURM_SECS" -le 0 ]; then SLURM_SECS=$(( 72 * 3600 )); fi
GUROBI_LIMIT=$(( SLURM_SECS / BATCH_SIZE - 300 ))

mkdir -p slurm_logs "$OUTPUT_DIR"

TASK_BASE=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE + TASK_OFFSET ))

for i in $(seq 0 $(( BATCH_SIZE - 1 ))); do
  TASK_IDX=$(( TASK_BASE + i ))
  if [ "$TASK_IDX" -ge "$NTASKS" ]; then break; fi

  TASK=$(python3 - "$TASK_IDX" "$TASK_FILE" <<'PYEOF'
import json, sys
tasks = json.load(open(sys.argv[2]))
t = tasks[int(sys.argv[1])]
print(f"--n {t['n']} --k {t['k']} --t {t['t']} --lambda {t['lambda']} --mu {t['mu']} --lib-id {t['lib_id']}")
PYEOF
  )

  echo "=== ILP task $TASK_IDX (batch item $i) ==="
  uv run python3 scripts/cayley_ilp_worker.py \
      $TASK \
      --output-dir "$OUTPUT_DIR" \
      --time-limit "$GUROBI_LIMIT" \
      --threads "${SLURM_CPUS_PER_TASK:-8}"
done
