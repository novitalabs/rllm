---
name: ppio-sandbox
description: PPIO Sandbox API for cloud-based code execution in R2E-Gym training
metadata:
  tags: ppio, sandbox, cloud, code-execution, docker, template
---

## When to use

Use this skill when working with:
- PPIO cloud sandbox for code execution
- Building custom Docker templates for R2E-Gym
- Managing sandbox lifecycle (create, run, pause, resume, kill)
- Debugging sandbox issues (timeout, rate limit, SSL errors)

## Environment Setup

Set the following environment variables:
- `PPIO_API_KEY` - API authentication
- `PPIO_ACCESS_TOKEN` - CLI authentication (same as API key)

```bash
export PPIO_API_KEY=sk_xxxxx
export PPIO_ACCESS_TOKEN=$PPIO_API_KEY
```

## Documentation

| Document | Description |
|----------|-------------|
| [api_guide.md](api_guide.md) | **Complete API Guide** |
| [template_build.md](template_build.md) | **Custom Template Build Guide** |
| [optimization.md](optimization.md) | Performance Optimization |
| [troubleshooting.md](troubleshooting.md) | Troubleshooting Guide |

## Quick Start

### Python SDK

```python
from ppio_sandbox.core import Sandbox
import os

# Create sandbox
sandbox = Sandbox(
    api_key=os.environ['PPIO_API_KEY'],
    template=os.environ.get('PPIO_TEMPLATE_ID', 'code-interpreter-v1'),
    timeout=3600
)

# Run command
output, exit_code = sandbox.run("python3 --version")
print(output)

# Cleanup
sandbox.kill()
```

### CLI Commands

```bash
# List templates
ppio-sandbox-cli tpl list

# Build custom template
ppio-sandbox-cli tpl build

# List running sandboxes
ppio-sandbox-cli sandbox list
```

## Performance Overview

| Metric | Sequential | Batched | Parallel (pool=32) |
|--------|------------|---------|-------------------|
| API calls/step | ~88,000 | ~512 | ~512 |
| Sandboxes used | 1 | 1 | 32 |
| Reward time | ~31 min | ~8.4 min | **~50 sec** |
| Speedup | - | 3.7x | **37x** |

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Rate Limit (429) | >150 concurrent | Use connection pool |
| Sandbox Expired (502) | >300s timeout | Rebuild or use pause/resume |
| Upload Timeout | Large files | Use pre-cloned templates |
| SSL Error | Proxy issues | Check HTTP_PROXY settings |

See [troubleshooting.md](troubleshooting.md) for detailed solutions.
