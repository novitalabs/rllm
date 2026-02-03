#!/bin/bash

# =============================================================================
# Environment Check Script for DeepSWE Training
# Verifies all dependencies and configurations before training
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLLM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=============================================="
echo "DeepSWE Environment Check"
echo "=============================================="

ERRORS=0

# -----------------------------------------------------------------------------
# Check Python and dependencies
# -----------------------------------------------------------------------------

echo ""
echo "[1/6] Checking Python environment..."

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "  Python: $PYTHON_VERSION"
else
    echo "  ERROR: python3 not found"
    ERRORS=$((ERRORS + 1))
fi

# Check key packages
for pkg in torch vllm rllm transformers; do
    if python3 -c "import $pkg" 2>/dev/null; then
        VERSION=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
        echo "  $pkg: $VERSION"
    else
        echo "  ERROR: $pkg not installed"
        ERRORS=$((ERRORS + 1))
    fi
done

# -----------------------------------------------------------------------------
# Check GPU availability
# -----------------------------------------------------------------------------

echo ""
echo "[2/6] Checking GPU availability..."

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "  GPU count: $GPU_COUNT"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        echo "    $line"
    done

    if [ "$GPU_COUNT" -lt 8 ]; then
        echo "  WARNING: Expected 8 GPUs for full training, found $GPU_COUNT"
    fi
else
    echo "  ERROR: nvidia-smi not found"
    ERRORS=$((ERRORS + 1))
fi

# -----------------------------------------------------------------------------
# Check PPIO API key
# -----------------------------------------------------------------------------

echo ""
echo "[3/6] Checking PPIO configuration..."

if [ -z "$PPIO_API_KEY" ]; then
    if [ -f "$SCRIPT_DIR/.env" ]; then
        source <(grep PPIO_API_KEY "$SCRIPT_DIR/.env")
    elif [ -f "$RLLM_DIR/.env" ]; then
        source <(grep PPIO_API_KEY "$RLLM_DIR/.env")
    fi
fi

if [ -n "$PPIO_API_KEY" ]; then
    echo "  PPIO_API_KEY: ${PPIO_API_KEY:0:10}...${PPIO_API_KEY: -4}"

    # Test PPIO connectivity
    if python3 -c "
from ppio_sdk import Sandbox
import os
os.environ['PPIO_API_KEY'] = '$PPIO_API_KEY'
sb = Sandbox()
sb.close()
print('  PPIO connectivity: OK')
" 2>/dev/null; then
        :
    else
        echo "  WARNING: PPIO connectivity test failed"
    fi
else
    echo "  ERROR: PPIO_API_KEY not set"
    ERRORS=$((ERRORS + 1))
fi

# -----------------------------------------------------------------------------
# Check training data
# -----------------------------------------------------------------------------

echo ""
echo "[4/6] Checking training data..."

TRAIN_DATA="$RLLM_DIR/data/swe/R2E_Gym_Subset.parquet"
VAL_DATA="$RLLM_DIR/data/swe/SWE_Bench_Verified.parquet"

if [ -f "$TRAIN_DATA" ]; then
    SIZE=$(du -h "$TRAIN_DATA" | cut -f1)
    COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_DATA')))" 2>/dev/null || echo "?")
    echo "  Training data: $SIZE ($COUNT instances)"
else
    echo "  ERROR: Training data not found: $TRAIN_DATA"
    echo "         Run: python3 examples/swe/prepare_swe_data.py"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "$VAL_DATA" ]; then
    SIZE=$(du -h "$VAL_DATA" | cut -f1)
    COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$VAL_DATA')))" 2>/dev/null || echo "?")
    echo "  Validation data: $SIZE ($COUNT instances)"
else
    echo "  ERROR: Validation data not found: $VAL_DATA"
    ERRORS=$((ERRORS + 1))
fi

# -----------------------------------------------------------------------------
# Check model access
# -----------------------------------------------------------------------------

echo ""
echo "[5/6] Checking model access..."

MODEL="Qwen/Qwen3-32B"
if python3 -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('$MODEL', trust_remote_code=True)
print(f'  Model: $MODEL')
print(f'  Hidden size: {config.hidden_size}')
print(f'  Num layers: {config.num_hidden_layers}')
" 2>/dev/null; then
    :
else
    echo "  WARNING: Cannot access model config (may need HF token or network)"
fi

# -----------------------------------------------------------------------------
# Check disk space
# -----------------------------------------------------------------------------

echo ""
echo "[6/6] Checking disk space..."

DISK_AVAIL=$(df -h $RLLM_DIR | tail -1 | awk '{print $4}')
echo "  Available disk space: $DISK_AVAIL"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
if [ $ERRORS -eq 0 ]; then
    echo "All checks passed! Ready to train."
    echo ""
    echo "To start training, run:"
    echo "  bash $SCRIPT_DIR/train_qwen3_32b_8h200.sh"
else
    echo "Found $ERRORS error(s). Please fix before training."
fi
echo "=============================================="

exit $ERRORS
