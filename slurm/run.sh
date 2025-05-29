#!/bin/bash

# Usage check
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <PURPOSE> <ARCH> <APPLIANCE> <REGULARIZER> <CONTINUATION> <MASK> <KEY>"
    echo "Example: $0 bert train 'dish washer' l1l2 first unmasked G2"
    exit 1
fi

PURPOSE=$1
ARCH=$2
APPLIANCE=$3
REGULARIZER=$4
IS_CONTINUATION=$5
MASKED=$6
KEY=$7

# Sanitize appliance name for filename usage
SAFE_APPLIANCE=$(echo "$APPLIANCE" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')

# Build job name and output file name
JOBNAME="job_${ARCH}_${SAFE_APPLIANCE}_${KEY}"
LOGFILE="out_${ARCH}_${SAFE_APPLIANCE}_${KEY}.log"

# Submit the job
sbatch \
  --export=ALL,ARCH="$ARCH",PURPOSE="$PURPOSE",APPLIANCE="$APPLIANCE",REGULARIZER="$REGULARIZER",IS_CONTINUATION="$IS_CONTINUATION",MASKED="$MASKED" \
  --job-name="$JOBNAME" \
  --output="$LOGFILE" \
  executor.job
