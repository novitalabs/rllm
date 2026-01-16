#!/bin/bash
# Build R2E-Gym Templates for PPIO Sandbox
#
# Prerequisites:
# 1. Docker installed and running
# 2. ppio-sandbox-cli installed (npm i -g ppio-sandbox-cli)
# 3. PPIO_ACCESS_TOKEN set
#
# Usage:
#   export PPIO_ACCESS_TOKEN="your_token"
#   ./build_template.sh r2e-gym-base
#   ./build_template.sh r2e-gym-scientific
#   ./build_template.sh r2e-gym-pillow
#   ./build_template.sh all  # Build all templates

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Available templates
TEMPLATES=("r2e-gym-base" "r2e-gym-scientific" "r2e-gym-pillow")

build_template() {
    local TEMPLATE_NAME=$1
    local DOCKERFILE="${TEMPLATE_NAME}.Dockerfile"

    echo "============================================="
    echo "Building PPIO Template: $TEMPLATE_NAME"
    echo "============================================="

    if [ ! -f "$SCRIPT_DIR/$DOCKERFILE" ]; then
        echo "Error: Dockerfile not found: $SCRIPT_DIR/$DOCKERFILE"
        return 1
    fi

    echo ""
    echo "Configuration:"
    echo "  Template name: $TEMPLATE_NAME"
    echo "  Dockerfile: $SCRIPT_DIR/$DOCKERFILE"
    echo "  CPU: 4 cores"
    echo "  Memory: 4096 MB"
    echo ""

    # Build Docker image locally first
    echo "[1/3] Building Docker image locally..."
    docker build -f "$SCRIPT_DIR/$DOCKERFILE" -t "${TEMPLATE_NAME}:latest" "$SCRIPT_DIR"

    echo ""
    echo "[2/3] Pushing to PPIO..."

    # Build the PPIO template
    cd "$SCRIPT_DIR"
    ppio-sandbox-cli template build \
        -n "$TEMPLATE_NAME" \
        -f "$DOCKERFILE" \
        -c "/usr/bin/supervisord -c /etc/supervisord.conf" \
        --cpu-count 4 \
        --memory-mb 4096

    echo ""
    echo "[3/3] Verifying template..."
    ppio-sandbox-cli template list | grep -E "$TEMPLATE_NAME|NAME" || echo "Template may take a few minutes to appear"

    echo ""
    echo "Template '$TEMPLATE_NAME' build complete!"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    if [ -z "$PPIO_ACCESS_TOKEN" ]; then
        echo "Error: PPIO_ACCESS_TOKEN is not set"
        echo ""
        echo "Get your access token from: https://ppio.com/settings/key-management"
        echo "Then run: export PPIO_ACCESS_TOKEN='your_token'"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo "Error: Docker is not running"
        echo "Please start Docker first"
        exit 1
    fi

    if ! command -v ppio-sandbox-cli &> /dev/null; then
        echo "Installing ppio-sandbox-cli..."
        npm install -g ppio-sandbox-cli
    fi
}

# Main
if [ $# -eq 0 ]; then
    echo "Usage: $0 <template-name|all>"
    echo ""
    echo "Available templates:"
    for t in "${TEMPLATES[@]}"; do
        echo "  - $t"
    done
    echo "  - all (build all templates)"
    exit 1
fi

check_prerequisites

if [ "$1" == "all" ]; then
    for t in "${TEMPLATES[@]}"; do
        build_template "$t"
        echo ""
    done
    echo "============================================="
    echo "All templates built successfully!"
    echo "============================================="
else
    build_template "$1"
fi

echo ""
echo "Usage in code:"
echo ""
echo "  # Environment variable"
echo "  export PPIO_SANDBOX_TEMPLATE='<template-name>'"
echo ""
echo "  # Or in Python"
echo "  from ppio_sandbox import Sandbox"
echo "  sandbox = Sandbox.create(template='<template-name>', api_key='...')"
