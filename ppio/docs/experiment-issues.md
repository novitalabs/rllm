# DeepSWE 32B RL Training - Experiment Issues & Solutions

## Environment

- **Cluster**: 8 nodes x 8 H200 GPUs (64 GPUs total)
  - Head: 10.83.115.18
  - Workers: 10.83.115.21, .22, .23, .25, .26, .27, .28
- **Software**: torch 2.8.0+cu128, vLLM 0.10.2, flash-attn 2.8.3 (rebuilt from source), verl 0.6.1, NVIDIA driver 580.126.09
  - (Previously tried: torch 2.10.0 + vLLM 0.11.2, had persistent ABI and "Cannot access data pointer" issues)
- **Model**: Qwen3-32B at `/data/models/Qwen3-32B`
- **Training**: RLOO (PPO variant) with FSDP + vLLM hybrid engine, async rollout mode

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

## Ray Cluster Setup

```bash
# Head node:
ray start --head --port=6379 --num-gpus=8

# Worker nodes:
ray start --address=10.83.115.18:6379 --num-gpus=8
```

All 8 nodes must have identical Python environments (torch, vllm, verl, flash-attn, etc.).

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
