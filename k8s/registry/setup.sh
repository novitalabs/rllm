#!/usr/bin/env bash
# Setup Docker Hub pull-through cache on the registry node.
# Run this once on the node that will host the registry (default: 10.83.115.18).
#
# Usage:
#   ssh 10.83.115.18 'bash -s' < k8s/registry/setup.sh

set -euo pipefail

REGISTRY_DIR="/data/registry-cache"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Setting up Docker Hub pull-through cache ==="

# 1. Create storage directory
mkdir -p "${REGISTRY_DIR}/data"
echo "[1/4] Created storage directory: ${REGISTRY_DIR}/data"

# 2. Copy registry config
cp "${SCRIPT_DIR}/config.yml" "${REGISTRY_DIR}/config.yml"
echo "[2/4] Installed registry config: ${REGISTRY_DIR}/config.yml"

# 3. Pull registry image (needs proxy)
export HTTP_PROXY=http://127.0.0.1:1083
export HTTPS_PROXY=http://127.0.0.1:1083
export NO_PROXY=localhost,127.0.0.1,10.0.0.0/8
docker pull registry:2
echo "[3/4] Pulled registry:2 image"

# 4. Install and start systemd service
cp "${SCRIPT_DIR}/docker-registry-mirror.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable docker-registry-mirror
systemctl start docker-registry-mirror
echo "[4/4] Started docker-registry-mirror service"

# Verify
sleep 2
if curl -sf http://127.0.0.1:5000/v2/_catalog >/dev/null; then
    echo "=== Registry mirror running at :5000 ==="
    curl -s http://127.0.0.1:5000/v2/_catalog
    echo
else
    echo "ERROR: Registry not responding on :5000"
    systemctl status docker-registry-mirror --no-pager
    exit 1
fi
