#!/bin/bash
# Distribute patched files to all worker nodes
# Usage: bash distribute_patches.sh
#
# This script copies patched site-packages files from the head node
# to all worker nodes. Run this AFTER applying patches on the head node.

set -e

WORKERS="10.83.115.21 10.83.115.22 10.83.115.23 10.83.115.25 10.83.115.26 10.83.115.27 10.83.115.28"

# Files to distribute (site-packages patches)
VERL_ROLLOUT="/usr/local/lib/python3.10/dist-packages/verl/workers/rollout/vllm_rollout"
VLLM_WORKER="/usr/local/lib/python3.10/dist-packages/vllm/v1/worker"
VERL_WORKERS="/usr/local/lib/python3.10/dist-packages/verl/workers"

FILES=(
    "${VERL_ROLLOUT}/vllm_rollout_spmd.py"
    "${VERL_ROLLOUT}/vllm_async_server.py"
    "${VLLM_WORKER}/gpu_model_runner.py"
    "${VERL_WORKERS}/fsdp_workers.py"
)

# Also distribute the ABI shim
SHIM="/tmp/torch_compat_shim.so"

echo "Distributing patches to worker nodes..."

for node in $WORKERS; do
    echo "=== Node: $node ==="
    for f in "${FILES[@]}"; do
        if [ -f "$f" ]; then
            echo "  Copying $f"
            scp -q "$f" "root@${node}:${f}"
        else
            echo "  WARNING: $f not found, skipping"
        fi
    done
    if [ -f "$SHIM" ]; then
        echo "  Copying $SHIM"
        scp -q "$SHIM" "root@${node}:${SHIM}"
    fi
    echo "  Done."
done

echo ""
echo "All patches distributed successfully."
