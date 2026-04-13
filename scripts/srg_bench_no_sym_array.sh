#!/bin/bash
#SBATCH --array=0-35
#SBATCH --job-name=srg-no-sym
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# SRG ILP Benchmark: exact formulation with no symmetry breaking (baseline).
#
# Usage:
#   sbatch scripts/srg_bench_no_sym_array.sh
#
# To target specific indices only:
#   sbatch --array=5,12,17 scripts/srg_bench_no_sym_array.sh

set -euo pipefail

# ── Project setup ─────────────────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

# ── Configurable parameters ───────────────────────────────────────────────
PARAMS_CSV="src/ilp/srg_params_n50.csv"
MODEL="srg_exact"
TIMEOUT=28500        # seconds (leave 300s buffer before SLURM kills at 8h)
HEURISTICS=0.3       # elevated for feasibility problem
SEED=0               # reproducibility
OUTPUT_DIR="bench_output/${SLURM_JOB_NAME}"

# ── Create task directory and redirect SLURM output there ────────────────
TASK_DIR="${OUTPUT_DIR}/$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")"
mkdir -p "${TASK_DIR}"
exec > "${TASK_DIR}/slurm.out" 2> "${TASK_DIR}/slurm.err"

# ── Logging ───────────────────────────────────────────────────────────────
echo "=== SRG ILP Benchmark (no symmetry breaking) ==="
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
    --no-fix-neighbors --lex none \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --timeout "${TIMEOUT}" \
    --heuristics "${HEURISTICS}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "End:           $(date -Iseconds)"
