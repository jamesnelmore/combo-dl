#!/bin/bash
#SBATCH --job-name=skrikhande
#SBATCH -N 1 -n 1 -c 8
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jnelmore@wm.edu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Load required modules
source /usr/local/etc/sciclone.bashrc
module load python
module load cuda

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export PYTHONPATH=${PWD}/src
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

# Change to working directory
cd ${PWD}

# Run the experiment
python src/combo_dl/main.py --config-name palay_large_model