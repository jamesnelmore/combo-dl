#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/slurm_logs}"

DEFAULT_TIME_LIMIT="02:00:00"
DEFAULT_GPUS="1"

TIME_LIMIT="${TIME:-$DEFAULT_TIME_LIMIT}"
GPUS_REQUEST="${GPUS:-$DEFAULT_GPUS}"
JOB_NAME="${JOB_NAME:-combo-dl}"

if [[ $# -lt 1 ]]; then
	echo "Usage: TIME=HH:MM:SS GPUS=N $0 <python.module> [hydra overrides...]" >&2
	exit 1
fi

PYTHON_MODULE="$1"
shift

mkdir -p "${LOG_DIR}"

SBATCH_ARGS=(
	"--job-name=${JOB_NAME}"
	"--time=${TIME_LIMIT}"
	"--output=${LOG_DIR}/%j.out"
	"--error=${LOG_DIR}/%j.err"
)

if [[ -n "${GPUS_REQUEST}" && "${GPUS_REQUEST}" != "0" ]]; then
	SBATCH_ARGS+=("--gres=gpu:${GPUS_REQUEST}")
fi

CMD=(uv run python -m "${PYTHON_MODULE}")
if [[ $# -gt 0 ]]; then
	CMD+=("$@")
fi

CMD_STRING=""
printf -v CMD_STRING '%q ' "${CMD[@]}"
CMD_STRING="${CMD_STRING% }"

sbatch "${SBATCH_ARGS[@]}" --wrap="source /etc/profile && module load cuda && cd ${PROJECT_ROOT} && ${CMD_STRING}"

