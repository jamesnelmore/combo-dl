#!/bin/bash

# ==============================================================================
# run_parallel.sh
#
# This script runs the `cayley_graphs.g` GAP script in parallel to check all
# 15 groups of order 24. It launches multiple background GAP processes, with
# each process checking a different group.
#
# Usage:
# 1. Make the script executable:
#    chmod +x run_parallel.sh
#
# 2. Run the script:
#    ./run_parallel.sh
# ==============================================================================

# --- Configuration ---
# Set the maximum number of GAP processes to run at the same time.
# A good starting point is the number of CPU cores on your system.
# You can find this on macOS with: sysctl -n hw.ncpu
# Or on Linux with: nproc
# WARNING: Each GAP process can be memory-intensive. If you have limited RAM,
# use a smaller number.
MAX_JOBS=10

# Directory to store the output logs from each GAP process.
LOG_DIR="logs"

# --- Script Body ---

# Create the log directory if it doesn't exist.
mkdir -p "$LOG_DIR"

# Clean up old logs from previous runs.
rm -f "$LOG_DIR"/group_*.log

echo "Starting parallel search for 15 groups with up to $MAX_JOBS concurrent jobs."
echo "Output for each group will be saved to the '$LOG_DIR' directory."
echo "This process will take a very long time."

# Loop through all 15 groups of order 24.
for i in {1..15}
do
  # The command to run for each group.
  # We execute `gap` with the script and the group index `i`.
  # The output (both stdout and stderr) is redirected to a log file.
  # The `&` at the end runs the command in the background.
  echo "Launching job for group $i..."
  gap cayley_graphs.g "$i" > "$LOG_DIR/group_$i.log" 2>&1 &

  # Check the number of currently running background jobs.
  # If it's greater than or equal to MAX_JOBS, wait for one to finish
  # before launching the next one.
  # `jobs -r -p` lists the process IDs (PIDs) of running jobs.
  # `wc -l` counts the number of lines, which gives us the job count.
  while [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; do
    # `wait -n` waits for the next background job to terminate.
    # It's more efficient than a simple `sleep`.
    wait -n
  done
done

# Wait for all remaining background jobs to complete.
echo "All 15 jobs have been launched. Waiting for the last few to complete..."
wait

echo ""
echo "--- All Jobs Complete ---"
echo "Search finished for all 15 groups."
echo "Check the log files in the '$LOG_DIR' directory for results."
echo "You can search for the word 'FOUND' to quickly see if any matches were discovered:"
echo "grep -r 'FOUND' $LOG_DIR"
