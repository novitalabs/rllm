# R2E-Gym Base Template for PPIO Sandbox

Custom PPIO template with pre-installed dependencies for SWE-bench training.

## Contents

- `r2e-gym-base.Dockerfile` - Template definition
- `build_r2e_gym_template.sh` - Build and deploy script

## Pre-installed Packages

### Testing
- pytest, pytest-cov, pytest-timeout, pytest-xdist
- hypothesis, coverage

### Code Analysis
- tree_sitter_languages
- chardet

### Scientific Computing
- numpy (<2.3 for Numba compatibility)
- scipy, mpmath, sympy
- cython, numexpr

### Web/Async
- aiohttp, yarl, multidict

### Utilities
- ipython, tqdm, requests
- pyyaml, toml

### System Tools
- uv (fast Python package manager)
- git (configured for fast clones)
- build-essential, gfortran, openblas, lapack

## Build Instructions

### Prerequisites

1. Docker installed and running
2. Node.js (for ppio-sandbox-cli)
3. PPIO access token

### Steps

```bash
# 1. Get access token from https://ppio.com/settings/key-management
export PPIO_ACCESS_TOKEN="your_token"

# 2. Build template (takes 10-15 minutes)
./build_r2e_gym_template.sh
```

### Manual Build

```bash
# Install CLI
npm i -g ppio-sandbox-cli

# Build locally
docker build -f r2e-gym-base.Dockerfile -t r2e-gym-base:latest .

# Push to PPIO
ppio-sandbox-cli template build \
    -n "r2e-gym-base" \
    -f "r2e-gym-base.Dockerfile" \
    -c "/usr/bin/supervisord -c /etc/supervisord.conf" \
    --cpu-count 4 \
    --memory-mb 4096
```

## Usage

### Environment Variable

```bash
export PPIO_SANDBOX_TEMPLATE="r2e-gym-base"
./train_deepswe_4h100_ppio.sh
```

### In Python

```python
from ppio_sandbox import Sandbox

sandbox = Sandbox.create(
    template="r2e-gym-base",
    api_key="sk_xxx"
)
```

### In ppio_reward.py

```python
class PPIOSandboxManager:
    def __init__(self, template: str = None, ...):
        self.template = template or os.environ.get(
            "PPIO_SANDBOX_TEMPLATE", "base"
        )
```

## Comparison

| Feature | base | sandbox-fusion | r2e-gym-base |
|---------|------|----------------|--------------|
| Python packages | Minimal | Standard | Full scientific |
| uv installer | No | No | Yes |
| numpy | Basic | Yes | <2.3 (Numba) |
| pytest/hypothesis | No | No | Yes |
| tree_sitter | No | No | Yes |
| Build tools | No | Basic | Full (gfortran) |

## Template Size

- Base image: ~2GB (sandbox-fusion)
- Additional packages: ~1.5GB
- Total: ~3.5GB

First sandbox creation may take 2-3 minutes for image pull.
Subsequent creations are instant due to caching.
