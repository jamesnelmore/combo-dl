# Slurm Setup for William & Mary HPC

This guide helps you configure and run your experiments on William & Mary's HPC cluster using Hydra's Slurm launcher, based on the [official W&M HPC documentation](https://www.wm.edu/offices/it/services/researchcomputing/using/running_jobs_slurm/).

## Quick Start

1. **Install the Slurm launcher plugin:**
   ```bash
   uv sync  # This will install hydra-submitit-launcher
   ```

2. **Choose your W&M cluster and update the constraint:**
   Edit `config/launcher/slurm.yaml` to specify which W&M cluster to use.

3. **Submit your job:**
   ```bash
   make run-slurm CONFIG=palay_large_model
   ```

## W&M HPC Cluster Information

Based on the [W&M HPC documentation](https://www.wm.edu/offices/it/services/researchcomputing/using/running_jobs_slurm/), W&M uses **constraints** instead of traditional partitions to specify clusters:

### Available W&M Clusters:
- **james** (`-C jm`): Main compute cluster
- **kuro** (`-C ku`): High-memory cluster  
- **bora** (`-C bo`): General compute cluster
- **hima** (`-C hi`): Accessed through bora front-end

### Key W&M HPC Features:
- Jobs start in submission directory (not home directory)
- Must source `/usr/local/etc/sciclone.bashrc` for main-campus clusters
- Can SSH into allocated nodes during job execution
- Uses `srun` for MPI job launching

## Configuration Steps

### 1. Update `config/launcher/slurm.yaml`

**Key parameters for W&M:**
- `constraint`: Set to specify which cluster (e.g., `jm` for james)
- `gres`: Request GPUs if needed
- `mem_gb` and `cpus_per_task`: Set based on your requirements
- `setup` commands: Must include W&M-specific environment sourcing

**Example for james cluster with GPU:**
```yaml
constraint: jm        # Use james cluster
gres: gpu:1          # Request 1 GPU
mem_gb: 32           # 32GB memory
cpus_per_task: 8     # 8 CPU cores
```

### 2. Environment Setup for W&M
The setup section is pre-configured for W&M with uv:
```yaml
setup:
  - "source /usr/local/etc/sciclone.bashrc"  # Required for W&M
  - "cd ${hydra:runtime.cwd}"
  - "module load python"                     # W&M's Python module
  - "export PYTHONPATH=${hydra:runtime.cwd}/src:$PYTHONPATH"
  # Note: uv handles virtual environment activation automatically
```

## Running Jobs

### Basic job submission:
```bash
make run-slurm CONFIG=palay_large_model
```

### With custom parameters:
```bash
make run-slurm-with-args CONFIG=palay_large_model ARGS="seed=999 algorithm.iterations=1000"
```

### Direct Hydra command:
```bash
PYTHONPATH=./src uv run python -m thesis.main --config-name palay_large_model hydra/launcher=slurm
```

### Specify W&M cluster at runtime:
```bash
make run-slurm-with-args CONFIG=palay_large_model ARGS="hydra.launcher.constraint=jm"
```

## Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

Check job output:
```bash
tail -f hydra_outputs/palay_large_model/YYYY-MM-DD_HH-MM-SS/slurm-JOBID.out
```

View job details:
```bash
scontrol show job JOBID
```

## W&M HPC Specific Commands

Based on the [W&M documentation](https://www.wm.edu/offices/it/services/researchcomputing/using/running_jobs_slurm/), useful commands include:

```bash
# Check node status and availability
sinfo

# Check specific node details
scontrol show node nodename

# Check job efficiency after completion
seff JOBID

# Cancel a job
scancel JOBID

# Check available modules
module avail
```

## W&M Cluster Selection

Choose the appropriate cluster constraint for your job:

- **james** (`constraint: jm`): Main compute cluster, good for most jobs
- **kuro** (`constraint: ku`): High-memory cluster for memory-intensive jobs  
- **bora** (`constraint: bo`): General compute cluster
- **hima** (`constraint: hi`): Submit from bora front-end

## Troubleshooting

1. **Job fails immediately**: Check the Slurm error file for module loading issues
2. **Environment not found**: Ensure your conda environment exists and is accessible
3. **Module loading fails**: Verify W&M's current module names with `module avail`
4. **GPU not found**: Check if the target cluster has GPU nodes available
5. **Permission denied**: Make sure you're submitting from the correct front-end node

## Configuration Files

- `config/launcher/slurm.yaml` - Main Slurm configuration for W&M HPC
- `config/palay_large_model.yaml` - Experiment config (includes `launcher: slurm`)
- `Makefile` - Contains `run-slurm` and `run-slurm-with-args` targets
- `SLURM_SETUP.md` - This documentation file

## Next Steps

1. Install the Slurm launcher: `uv sync`
2. Choose your target W&M cluster and set `constraint` in `config/launcher/slurm.yaml`
3. Verify your conda environment name matches what's in the setup commands
4. Submit your first test job: `make run-slurm CONFIG=palay_large_model`
