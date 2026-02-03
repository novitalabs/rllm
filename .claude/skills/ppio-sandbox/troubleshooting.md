# PPIO Sandbox Troubleshooting Guide

This document collects common issues and solutions when using PPIO Sandbox.

## Table of Contents

1. [HTTP Errors](#http-errors)
2. [Sandbox Issues](#sandbox-issues)
3. [File Operation Issues](#file-operation-issues)
4. [Execution Issues](#execution-issues)
5. [Template Issues](#template-issues)
6. [Network Issues](#network-issues)

---

## HTTP Errors

### 429 Too Many Requests

**Symptoms**:
```
HTTPError: 429 Too Many Requests
Rate limit exceeded
```

**Cause**: Exceeded 150 concurrent limit

**Solution**:
1. Use semaphore to control concurrency
```python
import asyncio
semaphore = asyncio.Semaphore(100)

async def run_with_limit():
    async with semaphore:
        return await sandbox.run(cmd)
```

2. Add retry logic
```python
import time
for attempt in range(3):
    try:
        result = sandbox.run(cmd)
        break
    except HTTPError as e:
        if '429' in str(e):
            time.sleep(2 ** attempt)
        else:
            raise
```

### 502 Bad Gateway

**Symptoms**:
```
HTTPError: 502 Bad Gateway
Sandbox expired
```

**Cause**: Sandbox exceeded 300s timeout

**Solution**:
1. Recreate sandbox
2. Use pause/resume to preserve state
3. Increase timeout parameter
```python
sandbox = Sandbox(api_key=api_key, timeout=3600)  # 1 hour max
```

### 404 Not Found

**Symptoms**:
```
HTTPError: 404 Not Found
Sandbox not found
```

**Cause**: Sandbox was destroyed or ID is wrong

**Solution**:
1. Check if sandbox exists
```bash
ppio-sandbox-cli sandbox list
```
2. Recreate sandbox

---

## Sandbox Issues

### Sandbox Creation Timeout

**Symptoms**:
```
TimeoutError: Sandbox creation timed out
```

**Cause**: PPIO service busy or network issues

**Solution**:
1. Increase creation timeout
```python
sandbox = Sandbox(api_key=api_key, creation_timeout=60)
```
2. Retry
3. Check PPIO service status

### Sandbox Auto-destroyed

**Symptoms**: Sandbox becomes unavailable during use

**Cause**: Exceeded default 5-minute timeout

**Solution**:
```python
# Set longer timeout at creation
sandbox = Sandbox(api_key=api_key, timeout=3600)

# Or periodically update timeout
sandbox.set_timeout(3600)
```

---

## File Operation Issues

### Permission Denied

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied: '/testbed/file.py'
```

**Cause**: Pre-cloned template files owned by root

**Solution**:

1. Add chmod in Dockerfile
```dockerfile
RUN chmod -R 777 /testbed
```

2. Modify permissions in sandbox
```python
sandbox.run('chmod -R 777 /testbed')
```

### Upload Timeout

**Symptoms**:
```
TimeoutError: File upload timed out
```

**Cause**: File too large or slow network

**Solution**:
1. Use pre-cloned templates to avoid upload
2. Compress files before upload
3. Upload in batches
```python
# Batch upload
for batch in chunks(files, 100):
    sandbox.files.write(batch)
```

### SSL Error

**Symptoms**:
```
SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING]
```

**Cause**: Proxy or network issues

**Solution**:
1. Check proxy settings
```bash
echo $HTTP_PROXY
echo $HTTPS_PROXY
```
2. Retry
3. Use different network

---

## Execution Issues

### Command Execution Timeout

**Symptoms**:
```
TimeoutError: Command execution timed out
```

**Solution**:
```python
# Increase timeout
output, exit_code = sandbox.run('pytest', timeout=300)
```

### Output Truncated

**Symptoms**: Command output incomplete

**Cause**: Output exceeds buffer limit

**Solution**:
```python
# Redirect output to file
sandbox.run('pytest > /tmp/output.txt 2>&1')
output = sandbox.files.read('/tmp/output.txt')
```

### input() Not Supported

**Symptoms**:
```
EOFError: EOF when reading a line
```

**Cause**: IPython does not support native input()

**Solution**: Use file or environment variables for input
```python
# Via environment variable
sandbox.run('python script.py', envs={'USER_INPUT': 'value'})

# Via file
sandbox.files.write('/tmp/input.txt', 'value')
sandbox.run('python script.py < /tmp/input.txt')
```

---

## Template Issues

### Template Not Found

**Symptoms**:
```
Error: Template not found
```

**Solution**:
1. List available templates
```bash
ppio-sandbox-cli tpl list
```
2. Check if template ID is correct
3. Test with default template
```python
sandbox = Sandbox(api_key=api_key)  # Uses default template
```

### Template Build Failed

**Symptoms**:
```
Error: Template build failed
```

**Solution**:
1. Check Dockerfile syntax
2. Test build locally
```bash
docker build -f ppio.Dockerfile .
```
3. Disable BuildKit
```bash
export DOCKER_BUILDKIT=0
```

---

## Network Issues

### Proxy Configuration

**China network requires proxy**:
```bash
export HTTP_PROXY=http://172.17.0.1:1081
export HTTPS_PROXY=http://172.17.0.1:1081
```

**Docker build proxy**:
```bash
ppio-sandbox-cli tpl build \
  --build-arg HTTP_PROXY=$HTTP_PROXY \
  --build-arg HTTPS_PROXY=$HTTPS_PROXY
```

### DNS Resolution Failed

**Solution**:
1. Use IP address instead of domain
2. Configure DNS servers
```bash
# /etc/resolv.conf
nameserver 8.8.8.8
nameserver 114.114.114.114
```

---

## Cleanup Scripts

### Cleanup All Sandboxes

```python
from ppio_sandbox.core import Sandbox
import os

api_key = os.environ['PPIO_API_KEY']

# List all sandboxes
sandboxes = Sandbox.list(api_key=api_key)

# Cleanup
for sb in sandboxes:
    try:
        Sandbox.connect(sb['sandbox_id'], api_key=api_key).kill()
        print(f"Killed {sb['sandbox_id']}")
    except:
        pass
```

### Cleanup Paused Sandboxes

```python
sandboxes = Sandbox.list(api_key=api_key, state='paused')
for sb in sandboxes:
    Sandbox.connect(sb['sandbox_id'], api_key=api_key).kill()
```

---

## Related Documentation

- [api_guide.md](api_guide.md) - API Guide
- [template_build.md](template_build.md) - Template Build Guide
- [optimization.md](optimization.md) - Performance Optimization
