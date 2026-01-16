#!/bin/bash
# Build R2E-Gym Base Template for PPIO Sandbox
#
# Prerequisites:
# 1. Docker installed and running
# 2. ppio-sandbox-cli installed (npm i -g ppio-sandbox-cli)
# 3. PPIO_ACCESS_TOKEN set
#
# Usage:
#   export PPIO_ACCESS_TOKEN="your_token"
#   ./build_r2e_gym_template.sh

set -e

TEMPLATE_NAME="r2e-gym-base"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================="
echo "Building PPIO Template: $TEMPLATE_NAME"
echo "============================================="

# Check prerequisites
if [ -z "$PPIO_ACCESS_TOKEN" ]; then
    echo "Error: PPIO_ACCESS_TOKEN is not set"
    echo ""
    echo "Get your access token from: https://ppio.com/settings/key-management"
    echo "Then run: export PPIO_ACCESS_TOKEN='your_token'"
    exit 1
fi

# Check Docker
if ! docker info &> /dev/null; then
    echo "Error: Docker is not running"
    echo "Please start Docker first"
    exit 1
fi

# Check ppio-sandbox-cli
if ! command -v ppio-sandbox-cli &> /dev/null; then
    echo "Installing ppio-sandbox-cli..."
    npm install -g ppio-sandbox-cli
fi

echo ""
echo "Configuration:"
echo "  Template name: $TEMPLATE_NAME"
echo "  Dockerfile: $SCRIPT_DIR/r2e-gym-base.Dockerfile"
echo "  CPU: 4 cores"
echo "  Memory: 4096 MB"
echo ""

# Build Docker image locally first
echo "[1/3] Building Docker image locally..."
docker build -f "$SCRIPT_DIR/r2e-gym-base.Dockerfile" -t "r2e-gym-base:latest" "$SCRIPT_DIR"

echo ""
echo "[2/3] Pushing to PPIO..."

# Build the PPIO template
# Note: ppio-sandbox-cli will push the image and convert to microVM
cd "$SCRIPT_DIR"
ppio-sandbox-cli template build \
    -n "$TEMPLATE_NAME" \
    -f "r2e-gym-base.Dockerfile" \
    -c "/usr/bin/supervisord -c /etc/supervisord.conf" \
    --cpu-count 4 \
    --memory-mb 4096

echo ""
echo "[3/3] Verifying template..."
ppio-sandbox-cli template list | grep -E "$TEMPLATE_NAME|NAME" || echo "Template may take a few minutes to appear"

echo ""
echo "============================================="
echo "Template build complete!"
echo "============================================="
echo ""
echo "Usage in code:"
echo ""
echo "  # Environment variable"
echo "  export PPIO_SANDBOX_TEMPLATE='$TEMPLATE_NAME'"
echo ""
echo "  # Or in Python"
echo "  from ppio_sandbox import Sandbox"
echo "  sandbox = Sandbox.create(template='$TEMPLATE_NAME', api_key='...')"
echo ""
echo "To use in training:"
echo "  Add to train_deepswe_4h100_ppio.sh:"
echo "  export PPIO_SANDBOX_TEMPLATE='$TEMPLATE_NAME'"
