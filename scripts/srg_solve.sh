#!/bin/bash
#SBATCH --job-name=srg-99-14-1-2
#SBATCH --output=slurm_logs/%j-srg-ham.out
#SBATCH --error=slurm_logs/%j-srg-ham.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G

set -euo pipefail

cd "/sciclone/home/jnelmore/thesis/"
source .venv/bin/activate

mkdir -p slurm_logs

echo "=== SRG ILP Solve ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Tasks:     $SLURM_NTASKS"
echo "Start:     $(date)"
echo ""

# 71h55m time limit leaves 5 minutes for Gurobi to flush its solution
# before the 72h SBATCH wall-clock kills the process.
python src/ilp/srg.py 99 14 1 2 \
    --fix-ham \
    --threads "$SLURM_NTASKS" \
    --time-limit 259500 \
    --mip-focus 1 \
    --heuristics 0.5

echo ""
echo "End:       $(date)"
