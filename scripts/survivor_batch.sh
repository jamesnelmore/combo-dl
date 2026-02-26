#!/bin/bash
#SBATCH --job-name=survivor-test
#SBATCH --array=0-5
#SBATCH --output=slurm_logs/survivor-%A_%a.out
#SBATCH --time=00:45:00
#SBATCH --gpus=1

# Array of parameter values
PARAMS=(0.0 0.01 0.02 0.05 0.08) # Add back .09

# Get the parameter for this task
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}

# Run the experiment
cd "/sciclone/home/jnelmore/thesis/"
source .venv/bin/activate.csh
python -m experiments.mlp_dce --config-name=survivor_test +training.survivor_proportion=$PARAM