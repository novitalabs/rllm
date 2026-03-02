#!/usr/bin/env python3
"""Apply all training patches at Docker build time."""

import os
import pathlib
import sys


def patch_vllm_async_server():
    """Patch 1: enable_sleep_mode fix."""
    f = pathlib.Path("/workspace/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py")
    t = f.read_text()
    old = '"enable_sleep_mode": True,'
    new = '"enable_sleep_mode": self.config.free_cache_engine,'
    if old in t:
        f.write_text(t.replace(old, new))
        print("Patch 1 applied: vllm_async_server.py (enable_sleep_mode)")
    elif "self.config.free_cache_engine" in t:
        print("Patch 1: already applied")
    else:
        print("WARNING: Patch 1 pattern not found", file=sys.stderr)
        return False
    return True


def patch_vllm_rollout_spmd():
    """Patch 2: _loop_forever no-break fix.

    verl v0.6.1 already has logger.exception + socket.send, but still has 'break'.
    We just need to remove the 'break' and add a comment.
    """
    f = pathlib.Path("/workspace/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py")
    t = f.read_text()
    # verl v0.6.1 pattern: exception + send + break
    old = (
        '                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")\n'
        "                await self.socket.send(pickle.dumps(e))\n"
        "                break"
    )
    new = (
        '                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")\n'
        "                try:\n"
        "                    await self.socket.send(pickle.dumps(e))\n"
        "                except Exception:\n"
        "                    pass\n"
        "                # Do NOT break - continue processing to avoid deadlocking the TP group"
    )
    if old in t:
        f.write_text(t.replace(old, new))
        print("Patch 2 applied: vllm_rollout_spmd.py (_loop_forever no-break)")
    elif "Do NOT break" in t:
        print("Patch 2: already applied")
    else:
        print("WARNING: Patch 2 pattern not found", file=sys.stderr)
        return False
    return True


def patch_gpu_model_runner():
    """Patch 3: KV cache GC prevention.

    vllm 0.11.0 has the code on 3 lines (chained calls, no parens wrapping).
    """
    import vllm

    vllm_dir = pathlib.Path(os.path.dirname(vllm.__file__))
    f = vllm_dir / "v1" / "worker" / "gpu_model_runner.py"
    t = f.read_text()
    # vllm 0.11.0 pattern: chained on 3 lines
    old = (
        "                    kv_caches[layer_name] = kv_cache_raw_tensors[\n"
        "                        layer_name].view(dtype).view(kv_cache_shape).permute(\n"
        "                            *inv_order)"
    )
    new = (
        "                    _raw = kv_cache_raw_tensors[layer_name]\n"
        "                    _typed = _raw.view(dtype)\n"
        "                    _shaped = _typed.view(kv_cache_shape)\n"
        "                    _permuted = _shaped.permute(*inv_order)\n"
        "                    kv_caches[layer_name] = _permuted\n"
        '                    if not hasattr(self, "_kv_cache_view_refs"):\n'
        "                        self._kv_cache_view_refs = []\n"
        "                    self._kv_cache_view_refs.extend([_raw, _typed, _shaped])"
    )
    if old in t:
        f.write_text(t.replace(old, new))
        print("Patch 3 applied: gpu_model_runner.py (KV cache GC prevention)")
    elif "_kv_cache_view_refs" in t:
        print("Patch 3: already applied")
    else:
        print("WARNING: Patch 3 pattern not found", file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    ok = True
    ok &= patch_vllm_async_server()
    ok &= patch_vllm_rollout_spmd()
    ok &= patch_gpu_model_runner()
    if not ok:
        print("Some patches failed to apply!", file=sys.stderr)
        sys.exit(1)
    print("All patches applied successfully")
