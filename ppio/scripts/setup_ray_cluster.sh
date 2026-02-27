#!/bin/bash
# Setup Ray cluster for multi-node training
# Usage: bash setup_ray_cluster.sh [head|worker]
#
# Run on head node first, then on each worker node.

set -e

HEAD_IP="10.83.115.18"
HEAD_PORT="6379"
NUM_GPUS=8

# Fix CUBLAS version mismatch on all nodes
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}

MODE="${1:-head}"

if [ "$MODE" = "head" ]; then
    echo "Starting Ray HEAD node..."
    ray stop --force 2>/dev/null || true
    sleep 2
    ray start --head --port=$HEAD_PORT --num-gpus=$NUM_GPUS
    echo ""
    echo "Head node started at ${HEAD_IP}:${HEAD_PORT}"
    echo "Run on each worker: bash setup_ray_cluster.sh worker"
elif [ "$MODE" = "worker" ]; then
    echo "Starting Ray WORKER node..."
    ray stop --force 2>/dev/null || true
    sleep 2
    ray start --address=${HEAD_IP}:${HEAD_PORT} --num-gpus=$NUM_GPUS
    echo ""
    echo "Worker connected to ${HEAD_IP}:${HEAD_PORT}"
else
    echo "Usage: bash setup_ray_cluster.sh [head|worker]"
    exit 1
fi

echo ""
ray status
