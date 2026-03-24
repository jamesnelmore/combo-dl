#!/bin/bash
#SBATCH --array=0-35
#SBATCH --job-name=srg-lex-leader
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --output=bench_output/lex-leader/%A_%a.out
#SBATCH --error=bench_output/lex-leader/%A_%a.err

# SRG ILP Benchmark: exact formulation with lex-leader symmetry breaking.
#
# Same parameter sets and solver settings as srg_bench_array.sh, but uses
# lex-leader row ordering instead of v0+v1 neighbour fixing.
#
# Usage:
#   mkdir -p bench_output && sbatch scripts/srg_bench_lex_array.sh
#
# After all tasks finish:
#   python scripts/aggregate_bench.py bench_output/<JOB_ID>

set -euo pipefail

# ── Project setup ─────────────────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

mkdir -p bench_output

# ── Configurable parameters ───────────────────────────────────────────────
PARAMS_CSV="src/ilp/srg_params_n50.csv"
MODEL="srg_exact"
TIMEOUT=14100        # seconds (leave 300s buffer before SLURM kills)
HEURISTICS=0.3       # elevated for feasibility problem
SEED=0               # reproducibility
OUTPUT_DIR="bench_output/${SLURM_ARRAY_JOB_ID}"

# ── Logging ───────────────────────────────────────────────────────────────
echo "=== SRG ILP Benchmark (lex-leader) ==="
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
    --no-fix-neighbors --lex lex_leader \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --timeout "${TIMEOUT}" \
    --heuristics "${HEURISTICS}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "End:           $(date -Iseconds)"
