#!/bin/bash
#SBATCH --array=0-35
#SBATCH --job-name=dce-bench
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# DCE (Deep Cross-Entropy) SRG benchmark.
# Runs one parameter set per array task from srg_params_n50.csv (36 rows).
# Wall-time limited to 8h; training runs until solution found or time expires.
#
# Usage:
#   sbatch scripts/dce_bench_array.sh
#
# To target specific indices:
#   sbatch --array=5,12,17 scripts/dce_bench_array.sh

set -euo pipefail

# ── Project setup ─────────────────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

# ── Configurable parameters ───────────────────────────────────────────────
PARAMS_CSV="src/ilp/srg_params_n50.csv"
OUTPUT_DIR="bench_output/${SLURM_JOB_NAME}"

# ── Create task directory and redirect SLURM output there ────────────────
TASK_DIR="${OUTPUT_DIR}/$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")"
mkdir -p "${TASK_DIR}"
exec > "${TASK_DIR}/slurm.out" 2> "${TASK_DIR}/slurm.err"

# ── Extract SRG parameters for this task ─────────────────────────────────
read -r N K LAMBDA MU < <(python3 -c "
import csv
with open('${PARAMS_CSV}') as f:
    rows = list(csv.DictReader(f))
    r = rows[${SLURM_ARRAY_TASK_ID}]
    print(r['n'], r['k'], r['lambda'], r['mu'])
")

EXPERIMENT_NAME="$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")_srg_${N}_${K}_${LAMBDA}_${MU}"

# ── Scale MLP width with n: width = max(n^2 / 2, 1024) ───────────────────
WIDTH=$(python3 -c "print(int(max(${N}**2 / 2, 1024)))")
HIDDEN="[${WIDTH},${WIDTH},${WIDTH},64]"

# ── Early stopping patience: ≈ 1 hour of no-improvement at this n ────────
# Iter time on A40 at batch=16384 scales ~0.0025 s/edge.
PATIENCE=$(python3 -c "
n = ${N}
edges = n * (n - 1) // 2
iter_s = max(0.0025 * edges, 0.05)
print(max(int(3600 / iter_s), 500))
")

# ── Logging ───────────────────────────────────────────────────────────────
echo "=== DCE SRG Benchmark ==="
echo "Job ID:        ${SLURM_ARRAY_JOB_ID}"
echo "Task ID:       ${SLURM_ARRAY_TASK_ID}"
echo "Node:          $(hostname)"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "SRG:           (${N},${K},${LAMBDA},${MU})"
echo "Experiment:    ${EXPERIMENT_NAME}"
echo "Output dir:    ${TASK_DIR}"
echo "Hidden layers: ${HIDDEN}"
echo "Patience:      ${PATIENCE} iters (~1h)"
echo "Start:         $(date -Iseconds)"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────
python -m experiments.mlp_dce \
    --config-name=dce_srg_bench \
    graph.n="${N}" \
    graph.k="${K}" \
    graph.lambda_param="${LAMBDA}" \
    graph.mu="${MU}" \
    model.hidden_layer_sizes="${HIDDEN}" \
    training.early_stopping_patience="${PATIENCE}" \
    experiment_name="${EXPERIMENT_NAME}" \
    save_dir="${TASK_DIR}"

echo ""
echo "End:           $(date -Iseconds)"
