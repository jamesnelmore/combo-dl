#!/bin/bash
#SBATCH --job-name=rrhc-test
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/rrhc_test_%j.out
#SBATCH --error=slurm_logs/rrhc_test_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
# Single-chunk probe to calibrate RRHC runtime at given (n, k, t, lambda, mu).
# Usage:
#   sbatch --export=ALL,N=96,K=10,T=5,LAMBDA=1,MU=1,START=1,END=16 scripts/rrhc_test.sh
#   sbatch --export=ALL,N=96,K=35,T=20,LAMBDA=19,MU=9,START=1,END=16 scripts/rrhc_test.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs rrhc_test_results

: "${N:?N required}"
: "${K:?K required}"
: "${T:?T required}"
: "${LAMBDA:?LAMBDA required}"
: "${MU:?MU required}"
: "${START:=1}"
: "${END:=16}"
: "${TRIALS:=300}"

OUT="rrhc_test_results/n${N}_k${K}_t${T}_l${LAMBDA}_m${MU}_g${START}-${END}.txt"
: > "$OUT"

echo "=== RRHC probe ==="
echo "n=$N k=$K t=$T lambda=$LAMBDA mu=$MU"
echo "Groups: SmallGroup($N, $START..$END)"
echo "Trials: $TRIALS"
echo "Output: $OUT"
echo "Start:  $(date -Iseconds)"
echo ""

gap -q -b <<GAPEOF
Read("src/random_restart.g");
RRHCindices(${N}, ${START}, ${END}, ${K}, ${T}, ${LAMBDA}, ${MU}, ${TRIALS}, "${OUT}");
QUIT;
GAPEOF

echo ""
echo "End:    $(date -Iseconds)"
