#!/bin/bash
#SBATCH --job-name=survivor-srg
# Array range: (number of params * NUM_RUNS) - 1 = (8 * 5) - 1 = 39
#SBATCH --array=0-39
#SBATCH --output=slurm_logs/%a_%A-survivor-srg-small.out
#SBATCH --error=slurm_logs/%a_%A-survivor-srg-small.out
#SBATCH --time=02:00:00
#SBATCH --gpus=1

# Array of survivor parameter values to test
PARAMS=(0.0 0.01 0.02 0.05 0.08 0.09 0.10 0.15)
NUM_RUNS=5

# Calculate parameter index and run number
# Each parameter gets NUM_RUNS experiments
PARAM_IDX=$((SLURM_ARRAY_TASK_ID / NUM_RUNS))
RUN_NUM=$((SLURM_ARRAY_TASK_ID % NUM_RUNS))

# Get the parameter value for this task
PARAM=${PARAMS[$PARAM_IDX]}

# Use different seed for each run (42, 43, ..., 51)
SEED=$((42 + RUN_NUM))


# Run the experiment
cd "/sciclone/home/jnelmore/thesis/"
source .venv/bin/activate

echo "Running experiment:"
echo "  Survivor proportion: $PARAM"
echo "  Run number: $((RUN_NUM + 1))/$NUM_RUNS"
echo "  Seed: $SEED"
echo "  Array task ID: $SLURM_ARRAY_TASK_ID"

python -m experiments.mlp_dce \
    --config-name=survivor_test \
    +training.survivor_proportion=$PARAM \
    +seed=$SEED \
    +graph.n=10 \ 
    +graph.k=3 \ 
    +graph.lambda_param=1 \
    +graph.mu=0
    experiment_name="survivor_${PARAM}_run${RUN_NUM}_small"

echo "Experiment completed successfully"

