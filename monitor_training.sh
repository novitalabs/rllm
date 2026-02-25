#!/bin/bash
# Training monitor script for DeepSWE
# Watches deepswe_full.log and reports key events

LOG=~/work/logs/deepswe_full.log
INTERVAL=60  # check every 60 seconds

echo "=== DeepSWE Training Monitor ==="
echo "Started at $(date)"
echo "Watching: $LOG"
echo ""

while true; do
    if [ ! -f "$LOG" ]; then
        echo "[$(date +%H:%M:%S)] WARNING: Log file not found"
        sleep $INTERVAL
        continue
    fi

    LINES=$(wc -l < "$LOG")
    SIZE=$(du -h "$LOG" | cut -f1)

    # Check for key events
    STEP=$(grep -a "step.*started" "$LOG" 2>/dev/null | tail -1)
    LATEST_REWARD=$(grep -a "Trajectory.*Reward" "$LOG" 2>/dev/null | tail -5)
    ERRORS=$(grep -a -c "Error\|error\|Traceback\|FATAL\|zmq.error" "$LOG" 2>/dev/null)
    ZMQ_WARN=$(grep -a -c "zmq.error.Again (transient)" "$LOG" 2>/dev/null)
    TIMING=$(grep -a "timing_s" "$LOG" 2>/dev/null | tail -1)
    PG_LOSS=$(grep -a "pg_loss" "$LOG" 2>/dev/null | tail -1)
    CKPT=$(grep -a "Saving checkpoint" "$LOG" 2>/dev/null | tail -1)

    echo "---"
    echo "[$(date +%H:%M:%S)] Lines: $LINES | Size: $SIZE | Errors: $ERRORS | ZMQ retries: $ZMQ_WARN"
    [ -n "$STEP" ] && echo "  Latest step: $(echo "$STEP" | strings | sed "s/.*\(epoch.*\)/\1/")"
    [ -n "$TIMING" ] && echo "  Timing: $(echo "$TIMING" | strings | head -1)"
    [ -n "$PG_LOSS" ] && echo "  PG Loss: $(echo "$PG_LOSS" | strings | head -1)"
    [ -n "$CKPT" ] && echo "  Checkpoint: $(echo "$CKPT" | strings | head -1)"
    if [ -n "$LATEST_REWARD" ]; then
        REWARD_GT0=$(grep -a "Reward is [1-9]\|Reward is 0\.[1-9]" "$LOG" 2>/dev/null | wc -l)
        REWARD_TOTAL=$(grep -a "Reward is" "$LOG" 2>/dev/null | wc -l)
        echo "  Rewards: $REWARD_GT0/$REWARD_TOTAL > 0"
    fi

    # Check if training process is alive
    PROCS=$(ps aux | grep train_agent_ppo | grep -v grep | wc -l)
    if [ "$PROCS" -eq 0 ]; then
        echo "  [!] Training process NOT running!"
    fi

    sleep $INTERVAL
done
