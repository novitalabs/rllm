#!/bin/bash
# cleanup_checkpoints.sh - Clean old FSDP checkpoints on both training nodes
# Usage: ./cleanup_checkpoints.sh [--keep N] [--execute]
# Default: dry-run mode, keep latest 5 checkpoints

set -uo pipefail

KEEP=5
DRY_RUN=true
CKPT_DIR="/home/claude/work/rllm-origin/checkpoints/deepswe-full/tp16-n8-full"
NODES=("localhost" "10.83.115.12")
NODE_NAMES=("youyun.37" "youyun.38")
# Skip checkpoints modified within this many seconds (protect in-progress saves)
PROTECT_RECENT_SECS=300

while [[ $# -gt 0 ]]; do
    case $1 in
        --keep)
            KEEP="$2"
            shift 2
            ;;
        --execute)
            DRY_RUN=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--keep N] [--execute]"
            echo "  --keep N     Keep latest N checkpoints (default: 5)"
            echo "  --execute    Actually delete (default: dry-run)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checkpoint cleanup started"
if $DRY_RUN; then
    echo "Mode: DRY RUN (use --execute to actually delete)"
else
    echo "Mode: EXECUTE"
fi
echo "Keeping latest $KEEP checkpoints"
echo ""

# run_on_node <node> <command>
run_on_node() {
    local node="$1"
    shift
    if [[ "$node" == "localhost" ]]; then
        bash -c "$*"
    else
        ssh -o ConnectTimeout=10 -o BatchMode=yes "$node" "$*" 2>/dev/null
    fi
}

for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    name="${NODE_NAMES[$i]}"
    echo "--- $name ($node) ---"

    # List checkpoints sorted by step number, with mtime (single command)
    # Output format: <mtime_epoch> <path>
    ckpt_info=$(run_on_node "$node" \
        "for d in ${CKPT_DIR}/global_step_*; do [ -d \"\$d\" ] && echo \"\$(stat -c '%Y' \"\$d\") \$d\"; done" \
        | sort -t_ -k3 -n || true)

    if [[ -z "$ckpt_info" ]]; then
        echo "  No checkpoints found."
        echo ""
        continue
    fi

    total=$(echo "$ckpt_info" | wc -l)
    to_delete=$((total - KEEP))

    echo "  Total checkpoints: $total"

    if [[ $to_delete -le 0 ]]; then
        echo "  Nothing to delete ($total <= $KEEP)"
        echo ""
        continue
    fi

    # Split into delete candidates and keep lists
    delete_candidates=$(echo "$ckpt_info" | head -n "$to_delete")
    keep_list=$(echo "$ckpt_info" | tail -n "$KEEP")

    # Filter out recently-modified checkpoints from delete list
    now=$(date +%s)
    safe_delete_paths=()
    skipped=0

    while IFS= read -r line; do
        mtime=$(echo "$line" | awk '{print $1}')
        path=$(echo "$line" | awk '{print $2}')
        step=$(basename "$path" | sed 's/global_step_//')
        age=$((now - mtime))
        if [[ $age -lt $PROTECT_RECENT_SECS ]]; then
            echo "  [SKIP]   step $step (modified ${age}s ago, < ${PROTECT_RECENT_SECS}s)"
            skipped=$((skipped + 1))
        else
            echo "  [DELETE] step $step (age: ${age}s)"
            safe_delete_paths+=("$path")
        fi
    done <<< "$delete_candidates"

    while IFS= read -r line; do
        path=$(echo "$line" | awk '{print $2}')
        step=$(basename "$path" | sed 's/global_step_//')
        echo "  [KEEP]   step $step"
    done <<< "$keep_list"

    if [[ $skipped -gt 0 ]]; then
        echo "  Skipped $skipped recently-modified checkpoint(s)"
    fi

    if ! $DRY_RUN && [[ ${#safe_delete_paths[@]} -gt 0 ]]; then
        # Build single rm command with all paths
        paths_str="${safe_delete_paths[*]}"
        run_on_node "$node" "rm -rf $paths_str"
        echo "  >>> Deleted ${#safe_delete_paths[@]} checkpoint(s)."
    fi

    # Show disk usage
    run_on_node "$node" "df -h /data" | tail -1 \
        | awk '{print "  Disk: " $3 " used / " $2 " total (" $5 " used, " $4 " free)"}'
    echo ""
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
