#!/bin/bash
# Slow pull: 30 pulls/hour = 120s between pulls

IMAGES_FILE="/home/claude/work/rllm-origin/r2e_gym_images.txt"
LOG_FILE="/home/claude/work/logs/docker_pull/slow_pull.log"
START_IDX=${1:-0}
DELAY=${2:-120}  # 120s = 30 pulls/hour

echo "=== Slow Docker Pull (30/hour) ===" | tee -a "$LOG_FILE"
echo "Start index: $START_IDX, Delay: ${DELAY}s" | tee -a "$LOG_FILE"
echo "Started at $(date)" | tee -a "$LOG_FILE"

i=0
while IFS= read -r image; do
    if [ $i -lt $START_IDX ]; then
        i=$((i + 1))
        continue
    fi
    
    if docker image inspect "$image" > /dev/null 2>&1; then
        echo "[$i] SKIP: ${image##*/}" | tee -a "$LOG_FILE"
    else
        echo "[$i] PULL: ${image##*/}" | tee -a "$LOG_FILE"
        if docker pull "$image" > /dev/null 2>&1; then
            echo "[$i] OK" | tee -a "$LOG_FILE"
        else
            echo "[$i] FAILED (will retry later)" | tee -a "$LOG_FILE"
        fi
        # Wait only after actual pull attempt (not skip)
        echo "[$i] Waiting ${DELAY}s..." | tee -a "$LOG_FILE"
        sleep $DELAY
    fi
    i=$((i + 1))
done < "$IMAGES_FILE"

echo "Done at $(date)" | tee -a "$LOG_FILE"
