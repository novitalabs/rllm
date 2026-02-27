#!/bin/bash
# Setup Docker for R2E-Gym SWE environment containers
# Run on the head node (where agent trajectories execute)
#
# Prerequisites: containerd must be running

set -e

echo "=== Docker Setup for SWE Training ==="

# Check if docker is already working
if docker info &>/dev/null; then
    echo "Docker is already running."
    docker --version
    exit 0
fi

# Install Docker CE
echo "Installing Docker CE..."
apt-get update
apt-get install -y docker-ce docker-ce-cli || {
    echo "Handling containerd config conflict..."
    dpkg --configure --force-confold containerd.io
    apt-get install -y -f
    apt-get install -y docker-ce docker-ce-cli
}

# Fix containerd config for v3 (required for containerd 2.2.x)
CONTAINERD_VERSION=$(containerd --version 2>/dev/null | grep -oP '\d+\.\d+' | head -1)
echo "Containerd version: $CONTAINERD_VERSION"

if [[ "$CONTAINERD_VERSION" == "2."* ]]; then
    echo "Generating containerd v3 config..."
    cp /etc/containerd/config.toml /etc/containerd/config.toml.bak 2>/dev/null || true
    containerd config default > /etc/containerd/config.toml
    systemctl restart containerd
fi

# Start Docker
systemctl restart docker
sleep 2

# Verify
echo "Verifying Docker..."
docker run --rm hello-world

echo ""
echo "Docker setup complete."
docker --version
