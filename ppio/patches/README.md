# Patches for DeepSWE 32B Training

These patches fix compatibility issues between verl 0.6.1, vLLM 0.11.2, and torch 2.10.0.

## Patch Files

### `torch_compat_shim.cpp`
ABI compatibility shim for vllm-flash-attn FA2 on torch 2.10.x. Build with `../scripts/build_shim.sh`.

### `vllm_async_server.patch`
Patches `verl/workers/rollout/vllm_rollout/vllm_async_server.py`:
- Sets `enable_sleep_mode` to `self.config.free_cache_engine` (was hardcoded `True`)

### `vllm_rollout_spmd.patch`
Patches `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`:
- Removes `break` in `_loop_forever()` to prevent TP group deadlock on errors

### `gpu_model_runner_kv_cache.patch`
Patches `vllm/v1/worker/gpu_model_runner.py`:
- Stores intermediate KV cache view tensor references to prevent GC

## Applying Patches

1. Apply patches on the head node
2. Run `../scripts/distribute_patches.sh` to copy to all worker nodes
