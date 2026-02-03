#!/bin/bash
set -x

# =============================================================================
# DeepSWE Evaluation Script for Qwen3-32B on SWE-Bench-Verified
# Uses vLLM server for inference
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RLLM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="$RLLM_DIR:$PYTHONPATH"

echo "=============================================="
echo "DeepSWE Evaluation - Qwen3-32B"
echo "RLLM_DIR: $RLLM_DIR"
echo "=============================================="

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# PPIO API key
if [ -z "$PPIO_API_KEY" ]; then
    if [ -f "$SCRIPT_DIR/.env" ]; then
        export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
    elif [ -f "$RLLM_DIR/.env" ]; then
        export $(grep -v '^#' "$RLLM_DIR/.env" | xargs)
    fi
fi

if [ -z "$PPIO_API_KEY" ]; then
    echo "Error: PPIO_API_KEY not set."
    exit 1
fi

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------

# Default model (can be overridden with --model flag)
MODEL="${MODEL:-Qwen/Qwen3-32B}"

# For trained checkpoints, use:
# MODEL="/path/to/checkpoint"

# vLLM server settings
VLLM_HOST="${VLLM_HOST:-localhost}"
VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL=8
MAX_CONTEXT_LEN=65536

# -----------------------------------------------------------------------------
# Start vLLM Server (if not running)
# -----------------------------------------------------------------------------

start_vllm_server() {
    echo "Starting vLLM server..."
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve $MODEL \
        --tensor-parallel-size $TENSOR_PARALLEL \
        --max-model-len $MAX_CONTEXT_LEN \
        --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
        --enable-prefix-caching \
        --host 0.0.0.0 \
        --port $VLLM_PORT &

    VLLM_PID=$!
    echo "vLLM server PID: $VLLM_PID"

    # Wait for server to start
    echo "Waiting for vLLM server to be ready..."
    for i in {1..60}; do
        if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            return 0
        fi
        sleep 5
    done

    echo "Error: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null
    exit 1
}

# Check if server is already running
if ! curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" > /dev/null 2>&1; then
    start_vllm_server
else
    echo "vLLM server already running at ${VLLM_HOST}:${VLLM_PORT}"
fi

# -----------------------------------------------------------------------------
# Run Evaluation
# -----------------------------------------------------------------------------

echo "=============================================="
echo "Running evaluation on SWE-Bench-Verified"
echo "Model: $MODEL"
echo "vLLM endpoint: http://${VLLM_HOST}:${VLLM_PORT}"
echo "=============================================="

cd $RLLM_DIR
python3 examples/swe/run_deepswe.py \
    --model $MODEL \
    --api-base "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
    --temperature 0.0 \
    --max-tokens 16384 \
    --dataset swe-bench-verified \
    --output-dir "$RLLM_DIR/results/eval_$(date +%Y%m%d_%H%M%S)"
