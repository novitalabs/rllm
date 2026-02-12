#!/bin/bash
# Sync Docker images from youyun.38 (10.83.115.12) to youyun.37
# Usage: ./sync_docker_images.sh [start] [end]

REMOTE_HOST="10.83.115.12"
START=${1:-0}
END=${2:-100}
LOG_DIR=~/work/logs/docker_sync
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/sync_$(date +%Y%m%d_%H%M%S).log"

echo "Fetching image list from remote..." | tee -a "$LOG_FILE"

# Get list of images to sync (on remote but not local)
REMOTE_IMAGES=$(ssh $REMOTE_HOST "docker images --format '{{.Repository}}:{{.Tag}}' | grep -E 'namanjain12|slimshetty'" | sort)
LOCAL_IMAGES=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -E 'namanjain12|slimshetty' | sort)

# Find images that need to be synced
IMAGES=$(comm -23 <(echo "$REMOTE_IMAGES") <(echo "$LOCAL_IMAGES"))

# Convert to array
mapfile -t IMG_ARRAY <<< "$IMAGES"
TOTAL=${#IMG_ARRAY[@]}

echo "Total images to sync: $TOTAL" | tee -a "$LOG_FILE"
echo "Processing range: $START to $END" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

SYNCED=0
SKIPPED=0
FAILED=0

for ((i=START; i<END && i<TOTAL; i++)); do
    IMG="${IMG_ARRAY[$i]}"
    [ -z "$IMG" ] && continue
    
    echo "[$((i+1))/$TOTAL] Syncing: $IMG" | tee -a "$LOG_FILE"
    
    # Check if already exists locally
    if docker image inspect "$IMG" &>/dev/null; then
        echo "  SKIP: Already exists" | tee -a "$LOG_FILE"
        ((SKIPPED++))
        continue
    fi
    
    # Transfer via pipe
    START_TIME=$(date +%s)
    if ssh $REMOTE_HOST "docker save '$IMG'" 2>/dev/null | docker load 2>/dev/null; then
        END_TIME=$(date +%s)
        echo "  OK: $((END_TIME-START_TIME))s" | tee -a "$LOG_FILE"
        ((SYNCED++))
    else
        echo "  FAIL" | tee -a "$LOG_FILE"
        ((FAILED++))
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "=== Summary ===" | tee -a "$LOG_FILE"
echo "Synced: $SYNCED" | tee -a "$LOG_FILE"
echo "Skipped: $SKIPPED" | tee -a "$LOG_FILE"
echo "Failed: $FAILED" | tee -a "$LOG_FILE"
