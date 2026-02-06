#!/bin/bash

# =============================================================================
# DeepSWE Training Environment Setup Script
# Installs all dependencies including r2egym, ppio_sandbox, and rllm
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLLM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=============================================="
echo "DeepSWE Environment Setup"
echo "RLLM_DIR: $RLLM_DIR"
echo "=============================================="

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Set proxy if provided via environment
if [ -n "$https_proxy" ] || [ -n "$HTTPS_PROXY" ]; then
    echo "[Setup] Using proxy: ${https_proxy:-$HTTPS_PROXY}"
fi

# R2E-Gym repository
R2EGYM_REPO="https://github.com/agentica-project/R2E-Gym.git"
R2EGYM_DIR="/tmp/R2E-Gym"

# Version requirements
VLLM_VERSION="0.10.2"
FLASHINFER_VERSION="0.6.1"

# -----------------------------------------------------------------------------
# Step 1: Check Python environment
# -----------------------------------------------------------------------------

echo ""
echo "[1/7] Checking Python environment..."

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.10" ]]; then
    echo "WARNING: Python 3.10+ recommended, found $PYTHON_VERSION"
fi

# -----------------------------------------------------------------------------
# Step 2: Install verl (RL training framework)
# -----------------------------------------------------------------------------

echo ""
echo "[2/7] Installing verl..."

cd "$RLLM_DIR"

if [ -d "./verl" ]; then
    pip install -e ./verl
    pip install -e "./verl[vllm]"
    echo "  verl installed"
else
    echo "WARNING: verl directory not found, skipping"
fi

# -----------------------------------------------------------------------------
# Step 3: Install rllm
# -----------------------------------------------------------------------------

echo ""
echo "[3/7] Installing rllm..."

pip install -e ".[swe]" || pip install -e .
echo "  rllm installed"

# -----------------------------------------------------------------------------
# Step 4: Install R2E-Gym
# -----------------------------------------------------------------------------

echo ""
echo "[4/7] Installing R2E-Gym..."

# Check if r2egym is already installed
if python3 -c "import r2egym" 2>/dev/null; then
    R2EGYM_VERSION=$(python3 -c "import r2egym; print(getattr(r2egym, '__version__', 'installed'))")
    echo "  r2egym already installed: $R2EGYM_VERSION"
else
    echo "  Cloning R2E-Gym from GitHub..."

    # Clone if not exists
    if [ ! -d "$R2EGYM_DIR" ]; then
        git clone "$R2EGYM_REPO" "$R2EGYM_DIR"
    fi

    # Install
    cd "$R2EGYM_DIR"
    pip install -e .
    cd "$RLLM_DIR"

    # Verify installation
    if python3 -c "import r2egym" 2>/dev/null; then
        echo "  r2egym installed successfully"
    else
        echo "WARNING: r2egym installation may have failed"
    fi
fi

# -----------------------------------------------------------------------------
# Step 5: Install PPIO Sandbox SDK
# -----------------------------------------------------------------------------

echo ""
echo "[5/7] Installing PPIO Sandbox SDK..."

pip install ppio_sandbox

if python3 -c "from ppio_sandbox.core import Sandbox" 2>/dev/null; then
    echo "  ppio_sandbox installed"
else
    echo "WARNING: ppio_sandbox installation may have failed"
fi

# -----------------------------------------------------------------------------
# Step 6: Install compatible versions of key dependencies
# -----------------------------------------------------------------------------

echo ""
echo "[6/7] Installing compatible dependency versions..."

# Install specific vllm version
echo "  Installing vllm==$VLLM_VERSION..."
pip install "vllm==$VLLM_VERSION"

# Rebuild flash-attn for current torch
echo "  Rebuilding flash-attn..."
pip uninstall flash-attn -y 2>/dev/null || true
pip install flash-attn --no-build-isolation --no-cache-dir

# Install matching flashinfer
echo "  Installing flashinfer-python==$FLASHINFER_VERSION..."
pip install "flashinfer-python==$FLASHINFER_VERSION"

# Install other dependencies
echo "  Installing additional dependencies..."
pip install pylatexenc pandas datasets

# -----------------------------------------------------------------------------
# Step 7: Verify installation
# -----------------------------------------------------------------------------

echo ""
echo "[7/7] Verifying installation..."

ERRORS=0

# Check core packages
for pkg in torch vllm transformers ppio_sandbox; do
    if python3 -c "import $pkg" 2>/dev/null; then
        VERSION=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', 'OK'))" 2>/dev/null)
        echo "  $pkg: $VERSION"
    else
        echo "  ERROR: $pkg not installed"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check rllm
if python3 -c "import rllm" 2>/dev/null; then
    echo "  rllm: OK"
else
    echo "  ERROR: rllm not installed"
    ERRORS=$((ERRORS + 1))
fi

# Check r2egym
if python3 -c "import r2egym" 2>/dev/null; then
    echo "  r2egym: OK"
else
    echo "  WARNING: r2egym not installed (optional but recommended)"
fi

# Check multi-step environment
if python3 -c "from rllm.environments.swe_ppio.swe_ppio_multistep import SWEBenchPPIOMultiStepEnv" 2>/dev/null; then
    echo "  SWEBenchPPIOMultiStepEnv: OK"
else
    echo "  WARNING: SWEBenchPPIOMultiStepEnv import failed"
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
    echo "  1. Set PPIO API Key:"
    echo "     export PPIO_API_KEY=sk_xxxxx"
    echo ""
    echo "  2. Prepare training data:"
    echo "     python3 examples/swe/prepare_swe_data.py"
    echo ""
    echo "  3. Verify environment:"
    echo "     bash $SCRIPT_DIR/check_env.sh"
    echo ""
    echo "  4. Start training:"
    echo "     bash $SCRIPT_DIR/train_qwen3_32b_8h200.sh"
else
    echo "Setup completed with $ERRORS error(s)."
    echo "Please check the errors above and fix manually."
fi
echo "=============================================="

exit $ERRORS
