# PPIO Sandbox Custom Template Build Guide

This document explains how to build custom PPIO Sandbox templates for R2E-Gym evaluation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Build Steps](#build-steps)
3. [Dockerfile Examples](#dockerfile-examples)
4. [Common Templates](#common-templates)
5. [Build Troubleshooting](#build-troubleshooting)

---

## Prerequisites

### Install Dependencies

```bash
# Install PPIO CLI
npm i -g ppio-sandbox-cli

# Ensure Docker is installed
docker info
```

### Set Authentication

```bash
# Get from env.local.md
source .env.local
export PPIO_ACCESS_TOKEN=$PPIO_API_KEY
```

---

## Build Steps

### Step 1: Create Template Directory

```bash
mkdir -p templates/my-template
cd templates/my-template
```

### Step 2: Write Dockerfile

```bash
cat > ppio.Dockerfile << 'EOF'
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl wget \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip3 install pytest numpy pandas

WORKDIR /testbed
EOF
```

### Step 3: Build Template

```bash
# With proxy (China network)
ppio-sandbox-cli tpl build \
  --build-arg HTTP_PROXY=$HTTP_PROXY \
  --build-arg HTTPS_PROXY=$HTTPS_PROXY

# Without proxy
ppio-sandbox-cli tpl build
```

### Step 4: Get Template ID

```bash
ppio-sandbox-cli tpl list
```

Example output:
```
┌─────────────────────┬───────────────────┬─────────────┐
│ Template ID         │ Name              │ Created     │
├─────────────────────┼───────────────────┼─────────────┤
│ xzegiq0xmuinwrclqr8a│ r2e-gym-orange3   │ 2026-02-02  │
│ 8k14f17ixvkgis2guf51│ r2e-gym-scientific│ 2026-01-15  │
└─────────────────────┴───────────────────┴─────────────┘
```

---

## Dockerfile Examples

### R2E-Gym Base Template

```dockerfile
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3-pip \
    git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Scientific packages
RUN pip3 install --no-cache-dir \
    pytest numpy scipy pandas scikit-learn \
    matplotlib pillow requests

WORKDIR /testbed
```

### Pre-cloned Repository Template

```dockerfile
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /testbed

# Pre-clone repository
RUN git clone --depth 1 https://github.com/biolab/orange3.git /testbed

# Set permissions (important!)
RUN chmod -R 777 /testbed
```

### sandbox-fusion Compatible Template

```dockerfile
FROM volcengine/sandbox-fusion:server-20250609

# Uses same environment as local Docker sandbox
# Supports /run_code HTTP API
```

---

## Common Templates

| Template ID | Name | Description | Use Case |
|-------------|------|-------------|----------|
| `code-interpreter-v1` | Official default | Python/JS/R/Java/Bash | General code execution |
| `8k14f17ixvkgis2guf51` | r2e-gym-scientific | Scientific packages | Requires numpy/pandas |
| `xzegiq0xmuinwrclqr8a` | r2e-gym-orange3 | Pre-cloned orange3 | Orange3 bug fixes |
| `f62brfz8qc6cjsz96kpz` | r2e-gym-pillow | Pillow image processing | Image-related tasks |

See [../tinker/env.local.md](../tinker/env.local.md) for latest template IDs.

---

## Build Troubleshooting

### Docker Not Running

```
Error: Docker is required
```

**Solution**: Start Docker service
```bash
sudo systemctl start docker
# or
docker info
```

### BuildKit Network Issues

```
Error: failed to resolve source metadata
```

**Solution**: Disable BuildKit
```bash
export DOCKER_BUILDKIT=0
ppio-sandbox-cli tpl build ...
```

### Image Pull Timeout

**Solution**: Configure Docker registry mirrors
```bash
# /etc/docker/daemon.json
{
    "registry-mirrors": [
        "https://dockerpull.org",
        "https://dockerhub.icu"
    ]
}
```

```bash
sudo systemctl restart docker
```

### Rate Limit

```
Error: Rate limit exceeded
```

**Solution**: Wait a few minutes and retry

### Long Build Time

First build requires:
1. Pull Docker image
2. Upload to PPIO remote repository
3. Convert to Firecracker microVM

**Expected time**: 10-15 minutes

---

## Related Documentation

- [api_guide.md](api_guide.md) - API Guide
- [troubleshooting.md](troubleshooting.md) - Troubleshooting
