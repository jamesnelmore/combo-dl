#!/bin/bash
#SBATCH --job-name=rrhc
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs/rrhc_%A_%a.out
#SBATCH --error=slurm_logs/rrhc_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=email@jameselmore.org
# JSON-driven RRHC sweep. Each array task reads one or more (param, group chunk)
# entries from rrhc_tasks.json and runs RRHCindices.
#
# Generate task file + submit command:
#   python scripts/rrhc_submit.py --n-min N --n-max M [--batch-size B]
# Submit with:
#   sbatch --array=0-<A-1> --export=ALL,BATCH_SIZE=<B> scripts/rrhc_array.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs rrhc_results

TASK_FILE="${TASK_FILE:-rrhc_tasks.json}"
TRIALS="${TRIALS:-300}"
TASK_OFFSET="${TASK_OFFSET:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NTASKS=$(python3 -c "import json; print(len(json.load(open('$TASK_FILE'))))")

TASK_BASE=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE + TASK_OFFSET ))

for i in $(seq 0 $(( BATCH_SIZE - 1 ))); do
  TASK_IDX=$(( TASK_BASE + i ))
  if [ "$TASK_IDX" -ge "$NTASKS" ]; then break; fi

  read -r N K T LAMBDA MU START END < <(python3 - "$TASK_IDX" "$TASK_FILE" <<'PYEOF'
import json, sys
tasks = json.load(open(sys.argv[2]))
t = tasks[int(sys.argv[1])]
print(t["n"], t["k"], t["t"], t["lambda"], t["mu"], t["start"], t["end"])
PYEOF
  )

  OUT="rrhc_results/n${N}_k${K}_t${T}_l${LAMBDA}_m${MU}_g${START}-${END}.txt"
  : > "$OUT"

  echo "=== RRHC ==="
  echo "Task:    array=$SLURM_ARRAY_TASK_ID batch_item=$i idx=$TASK_IDX"
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
done
