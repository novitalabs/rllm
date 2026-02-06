#!/bin/bash

# =============================================================================
# DeepSWE Training Environment Setup for H200 GPUs
# One-click installation of all dependencies
#
# Usage:
#   # With proxy (recommended for China)
#   https_proxy=http://127.0.0.1:1083 bash ppio/scripts/setup_h200_env.sh
#
#   # Without proxy
#   bash ppio/scripts/setup_h200_env.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLLM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=============================================="
echo "DeepSWE H200 Environment Setup"
echo "=============================================="
echo "RLLM_DIR: $RLLM_DIR"
echo "Date: $(date)"
echo ""

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Version requirements (tested on H200)
VLLM_VERSION="0.10.2"
FLASHINFER_VERSION="0.6.1"
VERL_VERSION="0.6.1"  # rllm is compatible with verl 0.6.1

# R2E-Gym repository
R2EGYM_REPO="https://github.com/agentica-project/R2E-Gym.git"

# Show proxy status
if [ -n "$https_proxy" ] || [ -n "$HTTPS_PROXY" ]; then
    echo "[Config] Using proxy: ${https_proxy:-$HTTPS_PROXY}"
else
    echo "[Config] No proxy configured"
fi
echo ""

# -----------------------------------------------------------------------------
# Step 1: Verify Python and CUDA
# -----------------------------------------------------------------------------

echo "[1/8] Verifying Python and CUDA..."

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python: $PYTHON_VERSION"

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install CUDA drivers."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "  GPUs: $GPU_COUNT"

# -----------------------------------------------------------------------------
# Step 2: Install verl (RL training framework)
# -----------------------------------------------------------------------------

echo ""
echo "[2/8] Installing verl==$VERL_VERSION..."

cd "$RLLM_DIR"

# Check if local verl directory exists
if [ -d "./verl" ]; then
    pip install -e ./verl -q
    pip install -e "./verl[vllm]" -q
    echo "  verl: installed from local"
else
    # Install from PyPI with specific version (rllm compatible with verl 0.6.1)
    pip install "verl==$VERL_VERSION" -q
    echo "  verl: $VERL_VERSION"
fi

# -----------------------------------------------------------------------------
# Step 3: Install rllm
# -----------------------------------------------------------------------------

echo ""
echo "[3/8] Installing rllm..."

pip install -e . -q 2>/dev/null || pip install -e ".[swe]" -q 2>/dev/null || true
echo "  rllm: installed"

# -----------------------------------------------------------------------------
# Step 4: Install vllm (compatible version)
# -----------------------------------------------------------------------------

echo ""
echo "[4/8] Installing vllm==$VLLM_VERSION..."

# This will also install compatible torch version
pip install "vllm==$VLLM_VERSION" -q
echo "  vllm: $VLLM_VERSION"

# Get the installed torch version
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
echo "  torch: $TORCH_VERSION (installed with vllm)"

# -----------------------------------------------------------------------------
# Step 5: Rebuild flash-attn for current torch
# -----------------------------------------------------------------------------

echo ""
echo "[5/8] Building flash-attn for torch $TORCH_VERSION..."

# Remove existing flash-attn
pip uninstall flash-attn -y -q 2>/dev/null || true

# Build from source (required for ABI compatibility)
echo "  Compiling flash-attn (this may take a few minutes)..."
pip install flash-attn --no-build-isolation --no-cache-dir -q

FLASH_VERSION=$(python3 -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || echo "unknown")
echo "  flash-attn: $FLASH_VERSION"

# -----------------------------------------------------------------------------
# Step 6: Install flashinfer (matching version)
# -----------------------------------------------------------------------------

echo ""
echo "[6/8] Installing flashinfer-python==$FLASHINFER_VERSION..."

pip install "flashinfer-python==$FLASHINFER_VERSION" -q
echo "  flashinfer: $FLASHINFER_VERSION"

# -----------------------------------------------------------------------------
# Step 7: Install PPIO Sandbox SDK
# -----------------------------------------------------------------------------

echo ""
echo "[7/8] Installing ppio_sandbox..."

pip install ppio_sandbox -q
echo "  ppio_sandbox: installed"

# -----------------------------------------------------------------------------
# Step 8: Install R2E-Gym
# -----------------------------------------------------------------------------

echo ""
echo "[8/8] Installing R2E-Gym..."

# Check if already installed
if python3 -c "import r2egym" 2>/dev/null; then
    echo "  r2egym: already installed"
else
    pip install git+${R2EGYM_REPO} -q
    echo "  r2egym: installed"
fi

# -----------------------------------------------------------------------------
# Install additional dependencies
# -----------------------------------------------------------------------------

echo ""
echo "[Extra] Installing additional dependencies..."

pip install pylatexenc pandas datasets -q
echo "  Additional packages: installed"

# -----------------------------------------------------------------------------
# Step 9: Download model (optional but recommended)
# -----------------------------------------------------------------------------

echo ""
echo "[9/9] Downloading Qwen3-32B model (if not cached)..."

MODEL_NAME="Qwen/Qwen3-32B"
MODEL_CACHE_DIR="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-32B"

# Check if model is fully cached
if [ -d "$MODEL_CACHE_DIR" ]; then
    SNAPSHOT_DIR=$(ls -d "$MODEL_CACHE_DIR/snapshots"/*/ 2>/dev/null | head -1)
    if [ -n "$SNAPSHOT_DIR" ] && [ -f "${SNAPSHOT_DIR}tokenizer.json" ]; then
        echo "  Model already cached: $MODEL_NAME"
    else
        echo "  Model cache incomplete, downloading..."
        python3 -c "
from transformers import AutoTokenizer, AutoConfig
print('Downloading tokenizer and config...')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME', trust_remote_code=True)
config = AutoConfig.from_pretrained('$MODEL_NAME', trust_remote_code=True)
print('Model files cached successfully')
" || echo "  WARNING: Model download failed (may need proxy)"
    fi
else
    echo "  Downloading model..."
    python3 -c "
from transformers import AutoTokenizer, AutoConfig
print('Downloading tokenizer and config...')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME', trust_remote_code=True)
config = AutoConfig.from_pretrained('$MODEL_NAME', trust_remote_code=True)
print('Model files cached successfully')
" || echo "  WARNING: Model download failed (may need proxy)"
fi

# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

ERRORS=0

# Check all packages
declare -A PACKAGES=(
    ["torch"]="import torch; print(torch.__version__)"
    ["vllm"]="import vllm; print(vllm.__version__)"
    ["flash_attn"]="import flash_attn; print(flash_attn.__version__)"
    ["transformers"]="import transformers; print(transformers.__version__)"
    ["ppio_sandbox"]="from ppio_sandbox.core import Sandbox; print('OK')"
    ["r2egym"]="import r2egym; print('OK')"
    ["rllm"]="import rllm; print('OK')"
)

for pkg in torch vllm flash_attn transformers ppio_sandbox r2egym rllm; do
    if VERSION=$(python3 -c "${PACKAGES[$pkg]}" 2>/dev/null); then
        echo "  $pkg: $VERSION"
    else
        echo "  ERROR: $pkg not working"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check multi-step environment
if python3 -c "from rllm.environments.swe_ppio.swe_ppio_multistep import SWEBenchPPIOMultiStepEnv" 2>/dev/null; then
    echo "  SWEBenchPPIOMultiStepEnv: OK"
else
    echo "  ERROR: SWEBenchPPIOMultiStepEnv not available"
    ERRORS=$((ERRORS + 1))
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
if [ $ERRORS -eq 0 ]; then
    echo "Environment setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Set PPIO API Key:"
    echo "     export PPIO_API_KEY=sk_xxxxx"
    echo ""
    echo "  2. Prepare training data (if not done):"
    echo "     python3 examples/swe/prepare_swe_data.py"
    echo ""
    echo "  3. Verify environment:"
    echo "     bash ppio/scripts/check_env.sh"
    echo ""
    echo "  4. Start training:"
    echo "     bash ppio/scripts/train_qwen3_32b_8h200.sh"
else
    echo "Setup completed with $ERRORS error(s)."
    echo "Please check the errors above."
fi
echo "=============================================="

exit $ERRORS
