#!/bin/bash
# Pull R2E-Gym Docker images in parallel with rate limit protection
# Usage: ./pull_r2e_images.sh [start_index] [end_index] [parallel_jobs]
#
# Examples:
#   ./pull_r2e_images.sh              # Pull all 4578 images
#   ./pull_r2e_images.sh 0 2288       # Pull first half (for .37)
#   ./pull_r2e_images.sh 2289 4577    # Pull second half (for .38)
#   ./pull_r2e_images.sh 0 100 4      # Pull first 100 images with 4 parallel jobs

set -e

IMAGES_FILE="/home/claude/work/rllm-origin/r2e_gym_images.txt"
LOG_DIR="/home/claude/work/logs/docker_pull"
PARALLEL_JOBS=${3:-2}  # Default 2 parallel pulls to avoid rate limit

START_IDX=${1:-0}
END_IDX=${2:-$(wc -l < "$IMAGES_FILE")}
END_IDX=$((END_IDX - 1))  # Convert to 0-based

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROGRESS_FILE="$LOG_DIR/progress_${START_IDX}_${END_IDX}_${TIMESTAMP}.log"
ERROR_FILE="$LOG_DIR/errors_${START_IDX}_${END_IDX}_${TIMESTAMP}.log"

echo "=== R2E-Gym Docker Image Puller ==="
echo "Images file: $IMAGES_FILE"
echo "Range: $START_IDX to $END_IDX"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Progress log: $PROGRESS_FILE"
echo "Error log: $ERROR_FILE"
echo ""

# Count total images to pull
TOTAL=$((END_IDX - START_IDX + 1))
echo "Total images to pull: $TOTAL"
echo ""

# Function to pull a single image
pull_image() {
    local idx=$1
    local image=$2
    local start_time=$(date +%s)
    
    # Check if already exists
    if docker image inspect "$image" > /dev/null 2>&1; then
        echo "[$idx] SKIP (exists): $image"
        echo "$idx,$image,SKIP,0" >> "$PROGRESS_FILE"
        return 0
    fi
    
    # Pull with retry
    local max_retries=3
    local retry=0
    while [ $retry -lt $max_retries ]; do
        if docker pull "$image" > /dev/null 2>&1; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            echo "[$idx] OK (${duration}s): $image"
            echo "$idx,$image,OK,$duration" >> "$PROGRESS_FILE"
            return 0
        fi
        retry=$((retry + 1))
        if [ $retry -lt $max_retries ]; then
            echo "[$idx] RETRY $retry: $image"
            sleep 10  # Wait before retry (rate limit)
        fi
    done
    
    echo "[$idx] FAILED: $image"
    echo "$idx,$image,FAILED,0" >> "$PROGRESS_FILE"
    echo "$image" >> "$ERROR_FILE"
    return 1
}

export -f pull_image
export PROGRESS_FILE ERROR_FILE

# Read images and create indexed list
mapfile -t ALL_IMAGES < "$IMAGES_FILE"

# Process images in range
CURRENT=0
PULLED=0
SKIPPED=0
FAILED=0

echo "Starting pull at $(date)"
echo ""

for i in $(seq $START_IDX $END_IDX); do
    image="${ALL_IMAGES[$i]}"
    CURRENT=$((CURRENT + 1))
    
    # Progress indicator
    PCT=$((CURRENT * 100 / TOTAL))
    echo -ne "\rProgress: $CURRENT/$TOTAL ($PCT%) - Pulled: $PULLED, Skipped: $SKIPPED, Failed: $FAILED"
    
    # Pull image
    if docker image inspect "$image" > /dev/null 2>&1; then
        SKIPPED=$((SKIPPED + 1))
        echo "$i,$image,SKIP,0" >> "$PROGRESS_FILE"
    else
        if docker pull "$image" > /dev/null 2>&1; then
            PULLED=$((PULLED + 1))
            echo "$i,$image,OK,0" >> "$PROGRESS_FILE"
        else
            # Retry once after 30s (rate limit)
            sleep 30
            if docker pull "$image" > /dev/null 2>&1; then
                PULLED=$((PULLED + 1))
                echo "$i,$image,OK,0" >> "$PROGRESS_FILE"
            else
                FAILED=$((FAILED + 1))
                echo "$image" >> "$ERROR_FILE"
                echo "$i,$image,FAILED,0" >> "$PROGRESS_FILE"
            fi
        fi
    fi
done

echo ""
echo ""
echo "=== Summary ==="
echo "Total: $TOTAL"
echo "Pulled: $PULLED"
echo "Skipped (already exists): $SKIPPED"
echo "Failed: $FAILED"
echo "Completed at $(date)"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed images saved to: $ERROR_FILE"
fi
