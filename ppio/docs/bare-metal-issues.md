# DeepSWE Bare-Metal Training - Issues & Solutions

## Environment

- **Cluster**: 8 nodes × 8 H200 GPUs (64 GPUs total)
  - Head: 10.83.115.18
  - Workers: 10.83.115.21, .22, .23, .25, .26, .27, .28
- **Software**: torch 2.8.0+cu128, vLLM 0.10.2, flash-attn 2.8.3 (rebuilt from source), verl 0.6.1, NVIDIA driver 580.126.09
  - (Previously tried: torch 2.10.0 + vLLM 0.11.2, had persistent ABI and "Cannot access data pointer" issues)
- **Model**: Qwen3-32B at `/data/models/Qwen3-32B`
- **Training**: RLOO (PPO variant) with FSDP + vLLM hybrid engine, async rollout mode

### Ray Cluster Setup

```bash
# Head node:
ray start --head --port=6379 --num-gpus=8

# Worker nodes:
ray start --address=10.83.115.18:6379 --num-gpus=8
```

All 8 nodes must have identical Python environments (torch, vllm, verl, flash-attn, etc.).

---

## Issue 1: ABI Incompatibility (vllm-flash-attn FA2 vs torch 2.10.0)

**Symptom**: `undefined symbol` errors when loading vllm-flash-attn's FA2 `.so` file.

**Root Cause**: torch 2.10.0 changed C++ ABI signatures:
- `c10::cuda::c10_cuda_check_implementation` changed `line` param from `int` to `unsigned int`
- `c10::MessageLogger` constructor changed signature

**Solution**: Build a C++ shim shared library that bridges the old symbols to new ones, loaded via `LD_PRELOAD`.

**Files**: `ppio/patches/torch_compat_shim.cpp`, `ppio/scripts/build_shim.sh`

---

## Issue 2: vLLM 0.11.0 Incompatible with torch 2.10.0

**Symptom**: Various import errors and runtime crashes with pip-installed vLLM 0.11.0.

**Root Cause**: vLLM 0.11.0 was built against an older torch version. The CUDA/torch ABI differences cause crashes.

**Solution**: Build vLLM 0.11.2 from source against torch 2.10.0:
```bash
git clone https://github.com/vllm-project/vllm.git -b v0.11.2
cd vllm
MAX_JOBS=64 pip install -e . --no-build-isolation
```

---

## Issue 3: verl 0.6.1 API Incompatibilities with vLLM 0.11.2

**Symptom**: Various `AttributeError` and `TypeError` when verl calls vLLM APIs.

**Root Cause**: verl 0.6.1 was written for an older vLLM API. vLLM 0.11.2 changed several interfaces.

**Solution**: Patch verl files. Key changes in `vllm_rollout_spmd.py` and `vllm_async_server.py`. See patches directory.

---

## Issue 4: CUBLAS Version Mismatch (Half-Precision GEMM Failures)

**Symptom**: `CUBLAS_STATUS_INVALID_VALUE` errors during BF16/FP16 GEMM operations. All matrix multiplications fail.

**Root Cause**: Version mismatch between:
- pip `nvidia-cublas-cu12` (12.8.4.1) provides `libcublas.so.12` (CUDA 12.8 API)
- System CUDA 12.9 provides `libcublasLt.so.12.9.0.13`

When `libcublas.so.12` (12.8) calls `libcublasLt.so.12` (12.9), the version skew causes internal API mismatches for half-precision kernels.

**Solution**: Prepend the pip cublas library path to `LD_LIBRARY_PATH` so both `libcublas.so.12` and `libcublasLt.so.12` come from the same package (12.8):

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
```

**Files**: Added to `train_deepswe_32b.sh` line 4.

---

## Issue 5: Worker Nodes Have Independent Filesystems

**Symptom**: Patches applied on head node don't take effect on worker nodes.

**Root Cause**: Each of the 8 nodes has its own filesystem. Ray distributes code via serialization but Python site-packages are loaded locally.

**Solution**: All patches to site-packages files must be distributed to all 8 nodes via SCP. Use `ppio/scripts/distribute_patches.sh`.

---

## Issue 6: Model and Data Distribution

**Symptom**: Workers can't find model or data files.

**Root Cause**: `/data/models/Qwen3-32B` and `/root/develop/ref/rllm/data/swe/` need to exist on all nodes.

**Solution**: SCP from source node (10.83.115.14) to all experiment nodes:
```bash
# From 10.83.115.14:
for node in 10.83.115.18 21 22 23 25 26 27 28; do
  scp -r /data/models/Qwen3-32B root@10.83.115.$node:/data/models/
  scp -r /root/develop/rllm/data/swe root@10.83.115.$node:/root/develop/ref/rllm/data/swe/
done
```

---

## Issue 7: flash-attn Build Timeout

**Symptom**: `pip install flash-attn` takes >1 hour and often times out.

**Solution**: Use `MAX_JOBS=64` to parallelize compilation:
```bash
MAX_JOBS=64 pip install flash-attn --no-build-isolation
```

---

## Issue 8: Docker Not Available for SWE Environment

**Symptom**: R2E-Gym containers fail with `docker: command not found` or `Failed to start container after 3 attempts`.

**Root Cause**: The head node only had `nerdctl` and `containerd`, not Docker CE.

**Solution**: Install Docker CE:
```bash
apt-get update
apt-get install -y docker-ce docker-ce-cli
# If containerd.io config conflict:
dpkg --configure --force-confold containerd.io
```

---

## Issue 9: Containerd Shim Version Incompatibility

**Symptom**: `failed to create task for container: Unimplemented: failed to start shim: start failed: unsupported shim version (3)`

**Root Cause**: containerd 2.2.1 uses shim v3 format, but the old config file was version 2.

**Solution**: Generate default v3 config:
```bash
containerd config default > /etc/containerd/config.toml
systemctl restart containerd
systemctl restart docker
```

---

## Issue 10: R2E-Gym Defaults to Kubernetes Backend

**Symptom**: `HTTPSConnectionPool(host='apiserver.cluster.local', port=6443)` - trying to connect to Kubernetes API.

**Root Cause**: The `SWEEnv` class in `rllm/environments/swe/swe.py` defaults to `backend="kubernetes"`.

**Solution**: Add `+rllm.env.env_args.backend=docker` to the Hydra training config. The `+` prefix is required because `env_args.backend` is a new key not in the base config.

**Files**: Added to `train_deepswe_32b.sh` line 71.

---

## Issue 11: CUDA OOM During Training

**Symptom**: `torch.OutOfMemoryError: CUDA out of memory` during FSDP training or vLLM inference.

**Solution**: Multiple mitigations:
- `gpu_memory_utilization=0.4` (vLLM uses only 40% of GPU memory for KV cache)
- `param_offload=True` + `optimizer_offload=True` (FSDP offloads to CPU)
- `free_cache_engine=False` (keep KV cache allocated to avoid reallocation overhead)
- `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"`

---

## Issue 12: `_loop_forever` ZMQ Server Breaks on Error (Deadlock)

**Symptom**: After any error in the vLLM worker's ZMQ message loop, the entire TP group deadlocks. Training hangs indefinitely.

**Root Cause**: The original `_loop_forever()` in `vllm_rollout_spmd.py` had a `break` statement in the `except` block. When any worker in a TP group encounters an error, it breaks out of the message loop. The other TP workers wait forever for the broken worker's response, causing a deadlock.

**Solution**: Remove the `break` statement. Continue processing messages after errors:
```python
async def _loop_forever(self):
    while True:
        try:
            message = await self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = await self._execute_method(method, *args, **kwargs)
            await self.socket.send(pickle.dumps(result))
        except Exception as e:
            logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
            try:
                await self.socket.send(pickle.dumps(e))
            except Exception:
                pass
            # Do NOT break - continue processing to avoid deadlocking the TP group
```

**Files**: `ppio/patches/vllm_rollout_spmd_loop_fix.patch`

---

## Issue 13: `enable_sleep_mode` Causes cumem Allocator Errors

**Symptom**: `RuntimeError: cumem allocator not initialized` when `enable_sleep_mode=True`.

**Root Cause**: vLLM's cumem (CUDA Unified Memory) allocator requires specific initialization. When `free_cache_engine=False`, sleep mode is not needed.

**Solution**: Set `enable_sleep_mode` to match `free_cache_engine`:
```python
# In vllm_async_server.py line 252:
"enable_sleep_mode": self.config.free_cache_engine,  # was: True
```

---

## Issue 14 (OPEN): "Cannot access data pointer of Tensor that doesn't have storage"

**Symptom**: First `execute_model()` after `update_weights()` fails with:
```
RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
```

**Error Location**: Inside `flash_attn_varlen_func` → `torch.ops._vllm_fa3_C.fwd` (Flash Attention 3 C kernel).

**Call Stack**:
```
vllm_rollout_spmd.py _loop_forever
  → _execute_method → execute_model
  → gpu_model_runner.py execute_model → _model_forward
  → qwen3.py forward → qwen2.py forward (decoder layers)
  → qwen3.py self_attn → Attention.forward
  → unified_attention_with_output → flash_attn.py forward
  → flash_attn_varlen_func → torch.ops._vllm_fa3_C.fwd
  RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
```

**Timeline**:
1. vLLM engine initializes with `load_format=dummy` (HYBRID mode)
2. `profile_run()` succeeds (model forward with dummy data works fine)
3. KV cache is allocated and reshaped via `_reshape_kv_cache_tensors()`
4. `update_weights()` is called - copies FSDP weights into vLLM model via `model.load_weights(weights)`
5. First real `execute_model()` dispatched via ZMQ → **FAILS**

**Investigation So Far**:
- Diagnostic logging added to check all model parameters and KV cache for `has_storage()` before model forward - no issues detected at Python level
- KV cache view tensor references saved to prevent GC - did not fix the issue
- The error is at the C++ level inside FA3 kernel, one of the tensor arguments to the kernel has lost its storage
- Possible tensor arguments: q, k (key_cache), v (value_cache), out, block_table, scheduler_metadata, q/k/v_descale, sinks

**Suspected Root Causes**:
1. `model.load_weights()` may replace parameter tensors (not in-place copy), causing old tensor storage to be freed. If any cached tensor (e.g., from compilation/tracing) still references old storage, it would fail.
2. The KV cache view chain `.view(dtype).view(shape).permute()` may lose underlying storage between profile_run and first execute_model if intermediate references are garbage collected, despite our fix attempt.
3. A tensor allocated during `profile_run` and cached for reuse (e.g., output buffer, scale factors) may have its storage freed during memory cleanup between profile and execute.

**Status**: Under investigation. Diagnostic logging deployed to all nodes.

---

## Issue 15: torch 2.10.0 "Cannot access data pointer of Tensor that doesn't have storage"

**Symptom**: `RuntimeError: Cannot access data pointer of Tensor that doesn't have storage` during vLLM FA3 forward pass after `update_weights` in the hybrid engine cycle.

**Root Cause**: PyTorch version issue - torch 2.10.0 has a bug where tensor storage can be invalidated after certain weight update + garbage collection patterns in the hybrid engine (FSDP weight offload → vLLM weight load → inference).

**Solution**: Downgrade to torch 2.8.0 which is a confirmed working version:
```bash
pip install torch==2.8.0 --no-deps
```

---

## Issue 16: vLLM 0.11.2 incompatible with torch 2.8.0

**Symptom**: `undefined symbol: _ZNK3c106SymInt22maybe_as_int_slow_pathEv` when importing vllm_flash_attn with torch 2.8.0.

**Root Cause**: vLLM 0.11.2 pip package was compiled against torch 2.10.0+ where `SymInt::maybe_as_int_slow_path()` and `SymInt::sym_ne_slow_path()` are separate library functions. In torch 2.8.0 these are inlined in the header.

**Solution**: Use vLLM 0.10.2 which was built against torch 2.8.0:
```bash
pip uninstall vllm -y
pip install vllm==0.10.2
```

---

## Issue 17: flash-attn 2.8.3 pip wheel ABI mismatch with torch 2.8.0

**Symptom**: `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb` when importing flash_attn.

**Root Cause**: The pip cached wheel for flash-attn was compiled against a newer torch where `c10_cuda_check_implementation` takes `unsigned int` (jb) instead of `int` (ib) for the line number parameter.

**Solution**: Force rebuild from source against torch 2.8.0:
```bash
pip uninstall flash-attn -y
MAX_JOBS=64 TORCH_CUDA_ARCH_LIST="9.0" pip install flash-attn --no-build-isolation --no-cache-dir --no-binary flash-attn
```
Then distribute the built package to worker nodes:
```bash
cd /usr/local/lib/python3.10/dist-packages
tar czf /tmp/flash_attn_28.tar.gz flash_attn/ flash_attn_2_cuda* flash_attn-*.dist-info/
# scp to workers and extract
```

---

## Issue 18: verl 0.6.1 `init_app_state()` API mismatch with vLLM 0.10.2

**Symptom**: `TypeError: init_app_state() missing 1 required positional argument: 'args'`

**Root Cause**: vLLM 0.10.2 `init_app_state()` takes 4 args `(engine_client, vllm_config, state, args)` while verl 0.6.1 was written for vLLM 0.11.2 which takes 3 args `(engine_client, state, args)`.

**Solution**: Edit `verl/workers/rollout/vllm_rollout/vllm_async_server.py` line 363:
```python
# Before (for vLLM 0.11.2):
await init_app_state(engine_client, app.state, args)
# After (for vLLM 0.10.2):
await init_app_state(engine_client, vllm_config, app.state, args)
```

---

## Issue 19 (OPEN): LLM Call Has No Timeout in Agent Trajectory Loop

**Symptom**: Single trajectory (#22) blocked entire training step for 3+ hours. Observed in Run 34, Step 1.

**Root Cause**: Three-layer timeout design flaw in `agent_execution_engine.py`:

```
Timeout Chain (innermost to outermost):
├── get_model_response()           → NO timeout (line 243)
├── env.step()                     → HAS timeout: trajectory_timeout - elapsed (line 266)
├── trajectory_timeout check       → Only checked AFTER both LLM+env complete (line 341)
├── run_agent_trajectory_async()   → asyncio.wait_for(timeout=7200s) (line 498)
└── retry_limit=3                  → Up to 3 × 7200s = 6 hours total (line 495)
```

```python
# Line 243 - NO timeout on LLM call:
model_output = await self.get_model_response(prompt_messages, application_id, **kwargs)

# Line 266 - env.step() HAS timeout:
next_observation, ... = await asyncio.wait_for(
    loop.run_in_executor(self.executor, env.step, action),
    timeout=(self.trajectory_timeout - total_time)
)

# Line 341 - timeout check only AFTER both LLM + env complete:
if total_time >= self.trajectory_timeout:
    break
```

**Full Timeline of Trajectory #22 (Run 34, Step 1)**:

| Time | Event |
|------|-------|
| 13:11 | Trajectory #22 started (along with 63 others) |
| ~14:30 | Most other trajectories completing (ENV_DONE / ENV_TIMEOUT) |
| 15:06 | 63/64 trajectories done. Only #22 remaining |
| ~15:11 | 7200s hard timeout fires on attempt 1. Error: `asyncio.CancelledError` on `server.generate.remote()` → `asyncio.TimeoutError` |
| 15:11 | Retry attempt 2 starts. New Docker container created (pandas) |
| 15:11-16:10 | Retry attempt 2 running. GPU utilization at 0% for ~1 hour (LLM not doing inference) |
| 16:10 | vLLM server process still alive (logging event stats) but not processing requests. 82 total `HandlePushTaskActor` calls, 0 active |

**Error Trace**:
```
agent_execution_engine.py:498 run_agent_trajectory_with_retry
  → asyncio.wait_for(run_agent_trajectory_async(...), timeout=7200)
    → agent_execution_engine.py:243 get_model_response (NO TIMEOUT)
      → verl_engine.py:80 server_manager.generate()
        → agent_loop.py:109 server.generate.remote() [Ray remote call]
          → asyncio.CancelledError (from 7200s timeout)
            → asyncio.TimeoutError
```

**Impact**: One slow trajectory can block the entire training step for up to `retry_limit × 7200s = 21600s` (6 hours). During this time, 63 completed trajectories sit idle, 64 GPUs are unused, and Docker containers remain running.

**Proposed Fix**:
1. Add `asyncio.wait_for` around the LLM call with remaining trajectory time:
```python
remaining_time = self.trajectory_timeout - total_time
if remaining_time <= 0:
    termination_reason = "TIMEOUT"
    break
model_output = await asyncio.wait_for(
    self.get_model_response(prompt_messages, application_id, **kwargs),
    timeout=remaining_time
)
```
2. Reduce outer hard timeout from 7200s to `trajectory_timeout + 600` (headroom for reward computation)
3. Reduce `retry_limit` from 3 to 1 (or 0) — retrying a stuck trajectory wastes hours
4. Add per-step timeout: if a single LLM call exceeds `max_single_llm_timeout` (e.g., 600s), abort the trajectory

**Status**: Fix not yet applied. Training Run 34 was killed due to this issue.

---

## Issue 20: `compute_final_reward()` Has No Timeout in Agent Loop

**Symptom**: After trajectory timeout, `compute_final_reward()` runs test suites in Docker containers with no timeout control from the agent loop.

**Root Cause**: In `agent_execution_engine.py`, after the trajectory loop breaks:
```python
reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
```
The inner `_calculate_reward()` has its own timeout (300s), but there's no outer `asyncio.wait_for` from the agent loop. If the Docker container hangs, this can block indefinitely.

**Impact**: Adds 300+ seconds after trajectory completion, extending already-long trajectories.

---

## Issue 21: Retokenization Mismatch Masks Out Many Trajectories [FIXED]

**Status**: Fixed. See [code-issues.md](code-issues.md) Issue 1 for full details and fix.

`traj/token_mismatch_mean` 从 0.37 降至 0.0。根因是每步重新 tokenize 导致 BPE 边界不一致，修复为跨步累积 token IDs 直接传给 vLLM。

---

## Issue 22: vLLM Server Actor Unresponsive During Retry (Run 34)

**Symptom**: After trajectory #22's first attempt timed out (7200s) and retry attempt 2 started, GPU utilization remained at 0% for over 1 hour. The vLLM server process was alive (logging event stats every minute) but not processing any inference requests.

**Evidence**:
- Ray core worker log for vLLM server (PID 3066090): `HandlePushTaskActor - 82 total (0 active)` — no active requests being processed
- All 8 GPUs on head node at 0% utilization during retry
- New Docker container (pandas) created at 15:11 for retry, running for 1+ hour
- The `AsyncLLMServerManager` uses sticky sessions via `request_id`. On retry, a new `application_id` (UUID) is generated, so the retry request goes to a potentially different vLLM server via least-requests load balancing

**Analysis**:
When the first attempt's `asyncio.wait_for(timeout=7200)` fires, it sends `CancelledError` to the `server.generate.remote()` Ray call. The cancellation propagation chain:
1. `asyncio` cancels the `wait_for` future
2. The Ray `ObjectRef` from `server.generate.remote()` is garbage collected
3. The vLLM server's `generate()` method may still be running (processing the async generator from `self.engine.generate()`)
4. If the vLLM V1 engine has an in-flight request that was never properly cancelled, it could be holding resources or be in a stuck state

The vLLM server process was alive but may have been in a state where:
- The async event loop was blocked waiting for the engine to finish the old request
- The engine's scheduler was stalled due to the cancelled but not cleaned up request
- OR: the retry trajectory was stuck on a different operation (env.reset, Docker setup) before reaching the LLM call

**Possible Root Causes**:
1. **Dangling async generator**: `vLLMHttpServer.generate()` calls `self.engine.generate()` which returns an async generator. If the Ray actor method was cancelled externally, the async generator may not have been properly closed, leaving the engine with a stuck in-flight request
2. **AsyncLLMServerManager load balancing**: With sticky sessions and a new UUID, the retry might be routed to a different server on a different node, while the local vLLM server on the head node shows 0% GPU. The actual inference might be happening on another node (not verified)
3. **Environment setup blocking**: The retry's `env.reset()` creates a new Docker container. If the env setup itself is slow or blocked, the LLM call hasn't been reached yet

**Status**: Inconclusive. The vLLM server was killed by a diagnostic signal (SIGUSR1) at 16:10:33 before the investigation could determine which code path the retry was stuck in.

---

## Issue 23: Diagnostic SIGUSR1 Killed vLLM Server Actor (Run 34)

**Symptom**: vLLM server process (PID 3066090) was terminated at 16:10:33 during diagnostic investigation.

**Root Cause**: During investigation of trajectory #22, `os.kill(3066090, signal.SIGUSR1)` was sent to dump Python thread stacks. The default action for SIGUSR1 in Linux is to terminate the process. The vLLM server Ray actor did not have a custom SIGUSR1 handler.

**Evidence**:
- Core worker log: last event stats at 16:10:07 (process alive)
- Ray event: `RAY_WORKER_FAILURE` at timestamp 1772179833 (16:10:33)
- Death detail: "Worker unexpectedly exits with a connection error code 2. End of file"
- Ray actor state notification: `state: DEAD, num_restarts: 0`

**Impact**: Killed the vLLM inference server, making the training unrecoverable. All subsequent LLM calls from the TaskRunner will fail with `RayActorError`.

**Lesson**: Never send signals to Ray actor processes. Use `ray.util.inspect_serializability()` or Ray dashboard for diagnostics instead.

**Resolution**: Training Run 34 must be killed and restarted.
