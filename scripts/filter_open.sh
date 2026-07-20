#!/bin/bash
# Build a slurm job manifest from a parameter CSV: emit one line per *open* row,
# as space-separated "n k t lambda mu" (no header) so that array task N maps to
# manifest line N and the columns drop straight into `rrhc dsrg`.
#
#   scripts/filter_open.sh new_parameters.csv > jobs.txt
#   sbatch --array=1-"$(wc -l < jobs.txt)" scripts/slurm/rrhc.sh
set -euo pipefail
# sub(/\r$/,"") strips a trailing CR so CRLF-terminated CSVs still match.
awk -F, '{ sub(/\r$/, "") } NR>1 && $6=="open" {print $1, $2, $3, $4, $5}' "${1:-new_parameters.csv}"
