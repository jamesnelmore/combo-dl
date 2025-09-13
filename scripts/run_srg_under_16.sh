#!/bin/bash

# Script to run 9 SRG parameter sets as separate Hydra jobs
# Based on the parameter sets from srg_search_up_to_16.yaml

set -e  # Exit on any error

# Activate virtual environment
source .venv/bin/activate

# Define the 9 parameter sets to test (from Brouwer's table)
PARAM_SETS=(
    "[5,2,0,1]"
    "[9,4,1,2]"
    "[10,3,0,1]"
    "[10,6,4,4]"
    "[13,6,2,3]"
    "[15,6,1,3]"
    "[15,8,4,4]"
    "[16,5,0,2]"
    "[16,6,2,2]"
)

# Base config to use
CONFIG_NAME="srg_ff_general"

echo "Starting 9 SRG parameter tests..."
echo "Config: $CONFIG_NAME"
echo "Total jobs: ${#PARAM_SETS[@]}"
echo "=================================="

# Run each parameter set as a separate job
for i in "${!PARAM_SETS[@]}"; do
    params="${PARAM_SETS[$i]}"
    echo ""
    echo "Job $((i+1))/${#PARAM_SETS[@]}: Testing SRG parameters $params"
    echo "----------------------------------------"
    
    # Run the job
    python -m combo_dl.main \
        --config-name="$CONFIG_NAME" \
        "srg_params=$params"
    
    # Check if job succeeded
    if [ $? -eq 0 ]; then
        echo "✅ Job $((i+1)) completed successfully"
    else
        echo "❌ Job $((i+1)) failed"
        exit 1
    fi
done

echo ""
echo "🎉 All 9 SRG parameter tests completed successfully!"
echo "=================================="
