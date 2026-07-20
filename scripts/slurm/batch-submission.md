# Submitting To Slurm

## Random-restart Hillclimbing
1. Ensure that the directory `rrhc-out` exists. The scruot will not create it by default
2. Generate a manifest of parameter sets to run. Each row should represent a single set with parameters seperated by spaces. For example
```
100 16 7 0 3
100 17 9 2 3
100 18 9 4 3
100 21 9 2 5
100 22 11 4 5
...
```
3. Modify the SBATCH commands at the top of `rrhc.sh` as needed to fit your cluster.
4. Submit the array job with
```bash
sbatch --array=1-$(wc -l < manifest.txt) scripts/slurm/rrhc.sh
```
