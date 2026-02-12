#!/bin/bash
# Pull R2E-Gym Docker images with conservative rate limiting
# Adds 60s delay between each pull to avoid 429 errors
# Usage: ./pull_r2e_images_slow.sh [start_index] [end_index]

set -e

IMAGES_FILE="/home/claude/work/rllm-origin/r2e_gym_images.txt"
LOG_DIR="/home/claude/work/logs/docker_pull"
DELAY_SECONDS=60  # Wait 60s between pulls

START_IDX=${1:-0}
END_IDX=${2:-$(wc -l < "$IMAGES_FILE")}
END_IDX=$((END_IDX - 1))

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROGRESS_FILE="$LOG_DIR/progress_slow_${START_IDX}_${END_IDX}_${TIMESTAMP}.log"

echo "=== R2E-Gym Slow Puller (60s delay) ==="
echo "Range: $START_IDX to $END_IDX"
echo "Progress log: $PROGRESS_FILE"
echo ""

mapfile -t ALL_IMAGES < "$IMAGES_FILE"
TOTAL=$((END_IDX - START_IDX + 1))
CURRENT=0
PULLED=0
SKIPPED=0

for i in $(seq $START_IDX $END_IDX); do
    image="${ALL_IMAGES[$i]}"
    CURRENT=$((CURRENT + 1))
    
    echo -n "[$CURRENT/$TOTAL] $image ... "
    
    if docker image inspect "$image" > /dev/null 2>&1; then
        echo "SKIP (exists)"
        SKIPPED=$((SKIPPED + 1))
    else
        if docker pull "$image" > /dev/null 2>&1; then
            echo "OK"
            PULLED=$((PULLED + 1))
            echo "$i,$image,OK" >> "$PROGRESS_FILE"
            # Wait before next pull to avoid rate limit
            if [ $CURRENT -lt $TOTAL ]; then
                echo "   Waiting ${DELAY_SECONDS}s before next pull..."
                sleep $DELAY_SECONDS
            fi
        else
            echo "RETRY after 120s..."
            sleep 120
            if docker pull "$image" > /dev/null 2>&1; then
                echo "   OK (retry)"
                PULLED=$((PULLED + 1))
            else
                echo "   FAILED"
            fi
        fi
    fi
done

echo ""
echo "=== Summary ==="
echo "Pulled: $PULLED, Skipped: $SKIPPED"
