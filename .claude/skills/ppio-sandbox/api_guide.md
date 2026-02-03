# PPIO Sandbox API Complete Guide

This document is based on PPIO official documentation, adapted for R2E-Gym training scenarios.

## Table of Contents

1. [SDK Installation](#sdk-installation)
2. [Sandbox Lifecycle](#sandbox-lifecycle)
3. [File System Operations](#file-system-operations)
4. [Command Execution](#command-execution)
5. [Code Interpreter](#code-interpreter)
6. [Environment Variables](#environment-variables)
7. [Sandbox Persistence](#sandbox-persistence)
8. [API Reference](#api-reference)

---

## SDK Installation

### Python SDK

```bash
# Stable version
pip install ppio-sandbox

# Beta version (supports pause/resume)
pip install ppio-sandbox --pre
```

### CLI Tool

```bash
npm i -g ppio-sandbox-cli
```

---

## Sandbox Lifecycle

### Create Sandbox

```python
from ppio_sandbox.core import Sandbox
import os

sandbox = Sandbox(
    api_key=os.environ['PPIO_API_KEY'],
    template='code-interpreter-v1',  # or custom template ID
    timeout=3600  # max 1 hour
)
```

### Timeout Configuration

| Parameter | Default | Maximum |
|-----------|---------|---------|
| timeout | 300s (5 min) | 3600s (1 hour) |

### Kill Sandbox

```python
sandbox.kill()
```

### Get Sandbox Info

```python
info = sandbox.get_info()
# Returns: sandboxId, templateId, state, startedAt, endAt, cpuCount, memoryMB
```

---

## File System Operations

### Write Files

```python
# Single file
sandbox.files.write('/testbed/test.py', 'print("hello")')

# Multiple files
sandbox.files.write([
    {'path': '/testbed/a.py', 'data': 'code_a'},
    {'path': '/testbed/b.py', 'data': 'code_b'}
])
```

### Read Files

```python
content = sandbox.files.read('/testbed/test.py')
```

### Upload Local Directory

```python
import os

def upload_directory(sandbox, local_dir, remote_dir):
    files = []
    for root, _, filenames in os.walk(local_dir):
        for filename in filenames:
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, local_dir)
            remote_path = os.path.join(remote_dir, rel_path)
            with open(local_path, 'r') as f:
                files.append({'path': remote_path, 'data': f.read()})
    sandbox.files.write(files)
```

---

## Command Execution

### Basic Execution

```python
output, exit_code = sandbox.run('ls -la /testbed')
print(f'Exit code: {exit_code}')
print(f'Output: {output}')
```

### Timeout Control

```python
output, exit_code = sandbox.run('python test.py', timeout=60)
```

### Background Execution

```python
# Start background process
handle = sandbox.run('python server.py', background=True)

# Kill later
handle.kill()
```

---

## Code Interpreter

### Supported Languages

Using `code-interpreter-v1` template:
- Python
- JavaScript/TypeScript
- R
- Java
- Bash

### Python Execution

```python
result = sandbox.run_code('''
import sys
print(sys.version)
''', language='python')

print(result.logs)
print(result.exit_code)
```

### Execution with Environment Variables

```python
result = sandbox.run_code('''
import os
print(os.environ.get('MY_VAR'))
''', language='python', envs={'MY_VAR': 'hello'})
```

---

## Environment Variables

### Set at Creation

```python
sandbox = Sandbox(
    api_key=os.environ['PPIO_API_KEY'],
    envs={
        'PYTHONPATH': '/testbed',
        'DEBUG': '1'
    }
)
```

### Set at Runtime

```python
output, _ = sandbox.run('echo $MY_VAR', envs={'MY_VAR': 'value'})
```

---

## Sandbox Persistence

### Pause Sandbox (Beta)

```python
sandbox.beta_pause()
sandbox_id = sandbox.sandbox_id  # Save ID for resume
```

### Resume Sandbox (Beta)

```python
sandbox = Sandbox.connect(sandbox_id, api_key=os.environ['PPIO_API_KEY'])
```

### Performance Notes

| Operation | Time |
|-----------|------|
| Pause | ~4 sec/GB RAM |
| Resume | ~1 sec |
| Data retention | Up to 30 days |

---

## API Reference

| Category | Method | Description |
|----------|--------|-------------|
| **Sandbox** | `Sandbox(api_key, template, timeout)` | Create sandbox |
| | `sandbox.kill()` | Kill sandbox |
| | `sandbox.get_info()` | Get info |
| | `sandbox.beta_pause()` | Pause sandbox |
| | `Sandbox.connect(id)` | Resume sandbox |
| **Files** | `sandbox.files.read(path)` | Read file |
| | `sandbox.files.write(path, data)` | Write file |
| **Commands** | `sandbox.run(cmd, timeout, envs)` | Execute command |
| | `sandbox.run_code(code, language)` | Run code |

---

## Related Documentation

- [template_build.md](template_build.md) - Custom Template Build Guide
- [optimization.md](optimization.md) - Performance Optimization
- [troubleshooting.md](troubleshooting.md) - Troubleshooting Guide
