#!/bin/bash
#SBATCH --array=0-35%20
#SBATCH --job-name=relaxed_ilp
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# SRG ILP Benchmark: relaxed formulation with v0+v1 neighbour fixing.
#
# Minimises the number of λ/μ violations while enforcing k-regularity.
# Objective = 0 certifies a valid SRG.
#
# Usage:
#   sbatch scripts/srg_bench_relaxed_array.sh
#
# After all tasks finish:
#   python scripts/aggregate_bench.py bench_output/relaxed_ilp

set -euo pipefail

# ── Project setup ─────────────────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

# ── Configurable parameters ───────────────────────────────────────────────
PARAMS_CSV="src/ilp/srg_params_n50.csv"
MODEL="srg_relaxed"
TIMEOUT=14100        # seconds (leave 300s buffer before SLURM kills)
HEURISTICS=0.3       # elevated for feasibility problem
SEED=0               # reproducibility
OUTPUT_DIR="bench_output/${SLURM_JOB_NAME}"

# ── Create task directory and redirect SLURM output there ────────────────
TASK_DIR="${OUTPUT_DIR}/$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")"
mkdir -p "${TASK_DIR}"
exec > "${TASK_DIR}/slurm.out" 2> "${TASK_DIR}/slurm.err"

# ── Logging ───────────────────────────────────────────────────────────────
echo "=== SRG ILP Benchmark (relaxed) ==="
echo "Job ID:        ${SLURM_ARRAY_JOB_ID}"
echo "Task ID:       ${SLURM_ARRAY_TASK_ID}"
echo "Node:          $(hostname)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Model:         ${MODEL}"
echo "Params CSV:    ${PARAMS_CSV}"
echo "Timeout:       ${TIMEOUT}s"
echo "Heuristics:    ${HEURISTICS}"
echo "Seed:          ${SEED}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Start:         $(date -Iseconds)"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────
python -m ilp bench-single \
    --params "${PARAMS_CSV}" \
    --index "${SLURM_ARRAY_TASK_ID}" \
    --model "${MODEL}" \
    --fix-neighbors --fix-v1 \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --timeout "${TIMEOUT}" \
    --heuristics "${HEURISTICS}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "End:           $(date -Iseconds)"
