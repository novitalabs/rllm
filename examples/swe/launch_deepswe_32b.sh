#!/bin/bash
set -ex

# ============================================================
# Multi-node launch script for DeepSWE 32B training
# Uses Ray cluster across 8 nodes
# ============================================================

HEAD_NODE="10.83.115.18"
WORKER_NODES=(
    "10.83.115.21"
    "10.83.115.22"
    "10.83.115.23"
    "10.83.115.25"
    "10.83.115.26"
    "10.83.115.27"
    "10.83.115.28"
)
ALL_NODES=("$HEAD_NODE" "${WORKER_NODES[@]}")

RAY_PORT=6379
RAY_HEAD_ADDRESS="${HEAD_NODE}:${RAY_PORT}"

# Environment variables to propagate to all nodes
# LD_PRELOAD shim for vllm-flash-attn FA2 ABI symbol
ENV_VARS="export LD_PRELOAD=/tmp/torch_compat_shim.so && \
export VLLM_ATTENTION_BACKEND=FLASH_ATTN && \
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False' && \
export VLLM_USE_V1=1 && \
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 && \
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000"

# ============================================================
# Step 1: Stop any existing Ray instances on all nodes
# ============================================================
echo ">>> Stopping existing Ray instances and vLLM processes on all nodes..."
for node in "${ALL_NODES[@]}"; do
    ssh -o StrictHostKeyChecking=no "$node" "ray stop -f; pkill -9 -f 'VLLM::' 2>/dev/null; pkill -9 -f 'vllm.entrypoints' 2>/dev/null; pkill -9 -f 'EngineCore' 2>/dev/null" 2>/dev/null || true
done
sleep 3

# ============================================================
# Step 2: Start Ray head node
# ============================================================
echo ">>> Starting Ray head node on ${HEAD_NODE}..."
ssh -o StrictHostKeyChecking=no "$HEAD_NODE" "${ENV_VARS} && ray start --head --port=${RAY_PORT} --num-cpus=128"
sleep 5

# ============================================================
# Step 3: Start Ray worker nodes
# ============================================================
echo ">>> Starting Ray worker nodes..."
for node in "${WORKER_NODES[@]}"; do
    echo "  Starting worker on ${node}..."
    ssh -o StrictHostKeyChecking=no "$node" "${ENV_VARS} && ray start --address='${RAY_HEAD_ADDRESS}' --num-cpus=128" &
done
wait
sleep 10

# ============================================================
# Step 4: Verify Ray cluster
# ============================================================
echo ">>> Verifying Ray cluster status..."
ssh -o StrictHostKeyChecking=no "$HEAD_NODE" "ray status"

# ============================================================
# Step 5: Launch training on head node
# ============================================================
echo ">>> Launching DeepSWE 32B training..."
TRAIN_SCRIPT="/root/develop/ref/rllm/examples/swe/train_deepswe_32b.sh"
ssh -o StrictHostKeyChecking=no "$HEAD_NODE" "cd /root/develop/ref/rllm && ${ENV_VARS} && bash ${TRAIN_SCRIPT}"
