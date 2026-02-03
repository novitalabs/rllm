# PPIO Sandbox Performance Optimization Guide

This document covers performance optimization techniques for PPIO Sandbox in R2E-Gym training.

## Bottleneck Analysis

### Typical R2E-Gym Training Flow

```
1. Create sandbox (~5-10s)
2. Upload code (~5-30s depending on size)
3. Execute N steps (~2-5s each)
4. Run tests (~10-60s)
5. Kill sandbox (~1s)
```

### Main Bottlenecks

| Stage | Problem | Impact |
|-------|---------|--------|
| Code upload | Large files slow | +30s per rollout |
| Test execution | Sequential execution | Slows with multiple tests |
| Sandbox creation | Frequent create/destroy | Extra overhead |

---

## Optimization Strategies

### 1. Use Pre-cloned Templates

**Problem**: Uploading repository code takes too long

**Solution**: Build templates with pre-cloned code

```dockerfile
FROM ubuntu:20.04
WORKDIR /testbed
RUN git clone --depth 1 https://github.com/biolab/orange3.git /testbed
RUN chmod -R 777 /testbed
```

**Result**:
- Upload time: 30s → 0s
- Sandbox creation: 10s → 7s

### 2. Batched Test Execution

**Problem**: Running tests one by one

**Solution**: Combine into single execution

```python
# Not recommended: multiple calls
for test in tests:
    sandbox.run(f'pytest {test}')

# Recommended: single call
sandbox.run(f'pytest {" ".join(tests)}')
```

**Result**: API calls reduced by N times

### 3. Sandbox Connection Pool

**Problem**: Frequent sandbox create/destroy

**Solution**: Use connection pool for reuse

```python
from concurrent.futures import ThreadPoolExecutor

class SandboxPool:
    def __init__(self, size=32):
        self.sandboxes = [
            Sandbox(api_key=os.environ['PPIO_API_KEY'])
            for _ in range(size)
        ]

    def get(self):
        return self.sandboxes.pop()

    def release(self, sandbox):
        # Clean up and return to pool
        sandbox.run('rm -rf /testbed/*')
        self.sandboxes.append(sandbox)
```

### 4. Parallel Processing

**Problem**: Single process handling multiple rollouts

**Solution**: Multi-threaded/AsyncIO parallel processing

```python
from concurrent.futures import ThreadPoolExecutor

def process_rollout(instance):
    sandbox = pool.get()
    try:
        # ... execute rollout
        return result
    finally:
        pool.release(sandbox)

with ThreadPoolExecutor(max_workers=32) as executor:
    results = list(executor.map(process_rollout, instances))
```

### 5. Pause/Resume (Beta)

**Problem**: Sandbox needs rebuild after 300s timeout

**Solution**: Use pause/resume to preserve state

```python
# Pause sandbox (save state)
sandbox.beta_pause()
sandbox_id = sandbox.sandbox_id

# ... do other work ...

# Resume sandbox (~1s)
sandbox = Sandbox.connect(sandbox_id, api_key=api_key)
```

**Note**: Pause time is proportional to RAM (~4s/GB)

---

## Performance Comparison

### Standard vs Optimized Configuration

| Configuration | Reward Time | Speedup |
|---------------|-------------|---------|
| Sequential (baseline) | ~31 min | 1x |
| Batched tests | ~8.4 min | 3.7x |
| +Parallel (pool=32) | ~50 sec | 37x |
| +AsyncIO (pool=2000) | ~40 sec | **46x** |

### Recommended Configuration

```python
# Standard configuration (512 concurrent)
POOL_SIZE = 512
MAX_CONCURRENT = 1024

# High performance configuration (AsyncIO)
POOL_SIZE = 2000
MAX_CONCURRENT = 4000
USE_ASYNC = True
ENABLE_PAUSE_RESUME = True
```

---

## Monitoring Metrics

### Key Metrics

| Metric | Target | Warning Threshold |
|--------|--------|-------------------|
| Sandbox creation time | <10s | >30s |
| Command execution latency | <100ms | >500ms |
| 429 error rate | 0% | >5% |
| 502 error rate | 0% | >1% |

### Log Monitoring

```bash
# View training logs
tail -f logs/rl_training.log | grep -E "(sandbox|reward|timeout)"
```

---

## Related Documentation

- [api_guide.md](api_guide.md) - API Guide
- [troubleshooting.md](troubleshooting.md) - Troubleshooting
