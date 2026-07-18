#!/bin/bash
#SBATCH --job-name=rrhc-n64
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/rrhc_n64_%A_%a.out
#SBATCH --error=slurm_logs/rrhc_n64_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
# Random-restart hill-climbing sweep for DSRG Cayley sets at n=64.
#
# Each array task handles one (parameter set, chunk of 32 groups) pair.
# 34 open params * 8 chunks (256 groups / 32) = 272 array tasks.
# Submit with: sbatch --array=0-271 scripts/rrhc_n64_array.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs rrhc_n64_results

# All open (k, t, lambda, mu) sets for n=64 (Status=open in dsrg_parameters.csv).
PARAMS=(
  "13 6 1 3"
  "14 7 3 3"
  "17 8 3 5"
  "19 8 7 5"
  "22 18 10 6"
  "23 13 12 6"
  "25 19 6 12"
  "26 24 8 12"
  "27 18 9 13"
  "27 18 17 7"
  "28 14 10 14"
  "29 17 12 14"
  "29 27 14 12"
  "30 15 13 15"
  "30 21 15 13"
  "30 24 16 12"
  "33 18 17 17"
  "33 21 16 18"
  "33 24 15 19"
  "33 27 14 20"
  "34 32 16 20"
  "35 21 20 18"
  "36 27 21 19"
  "37 35 22 20"
  "38 32 24 20"
  "39 27 26 20"
  "40 30 22 30"
  "41 37 24 30"
  "44 33 29 33"
  "45 36 31 33"
  "46 37 33 33"
  "47 36 35 33"
  "49 42 37 39"
  "50 43 39 39"
)

N=64
GROUPS_PER_CHUNK=16
N_CHUNKS=16                  # 256 / 32
TRIALS=300

TID=${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}
PARAM_IDX=$(( TID / N_CHUNKS ))
CHUNK_IDX=$(( TID % N_CHUNKS ))
START=$(( CHUNK_IDX * GROUPS_PER_CHUNK + 1 ))
END=$(( (CHUNK_IDX + 1) * GROUPS_PER_CHUNK ))

if [ -z "${PARAMS[$PARAM_IDX]:-}" ]; then
  echo "Task $TID out of range (param_idx=$PARAM_IDX)" >&2
  exit 1
fi
read -r K T LAMBDA MU <<<"${PARAMS[$PARAM_IDX]}"

OUT="rrhc_n64_results/n${N}_k${K}_t${T}_l${LAMBDA}_m${MU}_g${START}-${END}.txt"
: > "$OUT"   # truncate any previous result

echo "=== RRHC sweep ==="
echo "Task:    $TID  param_idx=$PARAM_IDX  chunk_idx=$CHUNK_IDX"
echo "Params:  n=$N k=$K t=$T lambda=$LAMBDA mu=$MU"
echo "Groups:  SmallGroup($N, $START..$END)"
echo "Trials:  $TRIALS"
echo "Output:  $OUT"
echo "Start:   $(date -Iseconds)"
echo ""

gap -q -b <<GAPEOF
Read("src/random_restart.g");
RRHCindices(${N}, ${START}, ${END}, ${K}, ${T}, ${LAMBDA}, ${MU}, ${TRIALS}, "${OUT}");
QUIT;
GAPEOF

echo ""
echo "End:     $(date -Iseconds)"
