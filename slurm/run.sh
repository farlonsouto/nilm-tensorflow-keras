#!/bin/bash

# Usage check
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <ARCH> <PURPOSE> <APPLIANCE> <KEY>"
    echo "Example: $0 bert train 'dish washer' simpleLossFn"
    exit 1
fi

PURPOSE=$1
ARCH=$2
APPLIANCE=$3
KEY=$4

# Sanitize appliance name for filename usage
SAFE_APPLIANCE=$(echo "$APPLIANCE" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')

# Build job name and output file name
JOBNAME="job_${ARCH}_${SAFE_APPLIANCE}_${KEY}"
LOGFILE="out_${ARCH}_${SAFE_APPLIANCE}_${KEY}.log"

# Submit the job
sbatch \
  --export=ALL,ARCH=$ARCH,PURPOSE=$PURPOSE,APPLIANCE="$APPLIANCE" \
  --job-name="$JOBNAME" \
  --output="$LOGFILE" \
  executor.job
