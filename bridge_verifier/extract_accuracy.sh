#!/bin/bash
set -e

LOG_DIR="logs"

printf "%-40s %-20s\n" "Checkpoint" "Overall Accuracy"
printf "%-40s %-20s\n" "----------------------------------------" "--------------------"

for log_file in $LOG_DIR/*.log; do
    ckpt=$(basename "$log_file" | sed 's/^eval_//' | sed 's/\.log$//')

    # Extract line with "Overall accuracy"
    acc_line=$(grep -E "Overall accuracy" "$log_file" | tail -n 1)

    if [ -n "$acc_line" ]; then
        # Extract numeric accuracy (e.g., 0.400)
        acc_value=$(echo "$acc_line" | grep -oE "[0-9]+\.[0-9]+")
        printf "%-40s %-20s\n" "$ckpt" "$acc_value"
    else
        printf "%-40s %-20s\n" "$ckpt" "N/A"
    fi
done
