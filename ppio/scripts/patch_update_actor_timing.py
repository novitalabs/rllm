#!/usr/bin/env python3
"""
Patch verl's fsdp_workers.py and dp_actor.py to add sub-phase timing
instrumentation inside update_actor.

This adds timing for:
- fsdp_workers.py: load_model, load_optimizer, update_policy, offload_model, offload_optimizer
- dp_actor.py: forward, backward, optimizer_step per mini-batch

Usage:
    python3 patch_update_actor_timing.py [--dry-run]

Run inside Docker container on both head and worker nodes:
    docker exec deepswe-train python3 /root/develop/rllm/ppio/scripts/patch_update_actor_timing.py
    ssh 10.83.115.17 "docker exec deepswe-train python3 /root/develop/rllm/ppio/scripts/patch_update_actor_timing.py"
"""

import sys
import shutil
from pathlib import Path

VERL_DIR = Path("/usr/local/lib/python3.12/dist-packages/verl")
DRY_RUN = "--dry-run" in sys.argv


def patch_fsdp_workers():
    """Add sub-phase timing to update_actor in fsdp_workers.py"""
    fpath = VERL_DIR / "workers" / "fsdp_workers.py"
    content = fpath.read_text()

    # Check if already patched
    if "timing/load_model_to_gpu" in content:
        print(f"[SKIP] {fpath} already patched")
        return

    # Backup
    if not DRY_RUN:
        shutil.copy2(fpath, str(fpath) + ".bak")

    # Add 'import time' after 'import datetime' if not present
    if "import time\n" not in content:
        content = content.replace("import datetime\n", "import datetime\nimport time\n", 1)

    # Replace the update_actor method body
    old_code = """\
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on actor.update_policy

            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output"""

    new_code = """\
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        _phase_timing = {}

        _t0 = time.monotonic()
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        _phase_timing["load_model_to_gpu"] = time.monotonic() - _t0

        _t0 = time.monotonic()
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())
        _phase_timing["load_optimizer_to_gpu"] = time.monotonic() - _t0

        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on actor.update_policy

            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            _phase_timing["update_policy"] = delta_time

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = output.to("cpu")

        _t0 = time.monotonic()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        _phase_timing["offload_model_to_cpu"] = time.monotonic() - _t0

        _t0 = time.monotonic()
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)
        _phase_timing["offload_optimizer_to_cpu"] = time.monotonic() - _t0

        # Add sub-phase timing to metrics
        for k, v in _phase_timing.items():
            output.meta_info["metrics"][f"timing/{k}"] = v

        return output"""

    if old_code not in content:
        print(f"[ERROR] {fpath}: Could not find update_actor method to patch. Code may have changed.")
        return False

    content = content.replace(old_code, new_code)

    if DRY_RUN:
        print(f"[DRY-RUN] Would patch {fpath}")
    else:
        fpath.write_text(content)
        print(f"[OK] Patched {fpath}")
    return True


def patch_dp_actor():
    """Add per-phase timing to update_policy in dp_actor.py"""
    fpath = VERL_DIR / "workers" / "actor" / "dp_actor.py"
    content = fpath.read_text()

    # Check if already patched
    if "timing/forward_s" in content:
        print(f"[SKIP] {fpath} already patched")
        return

    # Backup
    if not DRY_RUN:
        shutil.copy2(fpath, str(fpath) + ".bak")

    # Add 'import time' after existing imports if not present
    if "import time\n" not in content:
        content = content.replace("import torch\n", "import time\nimport torch\n", 1)

    # Patch the update_policy method to add timing around forward, backward, optimizer_step
    # We need to be surgical here. Let's add timing counters.

    # 1. Add timing accumulators after "metrics = {}"
    old_metrics_init = """        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):"""

    new_metrics_init = """        metrics = {}
        _timing_forward = 0.0
        _timing_backward = 0.0
        _timing_optim_step = 0.0
        _n_micro_batches = 0
        _n_mini_batches = 0
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):"""

    if old_metrics_init not in content:
        print(f"[ERROR] {fpath}: Could not find metrics init block to patch")
        return False

    content = content.replace(old_metrics_init, new_metrics_init, 1)

    # 2. Add timing around _forward_micro_batch
    old_forward = """                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )"""

    new_forward = """                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    _t_fwd = time.monotonic()
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _timing_forward += time.monotonic() - _t_fwd
                    _n_micro_batches += 1"""

    if old_forward not in content:
        print(f"[ERROR] {fpath}: Could not find forward block to patch")
        return False

    content = content.replace(old_forward, new_forward, 1)

    # 3. Add timing around loss.backward()
    old_backward = """                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()"""

    new_backward = """                    _t_bwd = time.monotonic()
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _timing_backward += time.monotonic() - _t_bwd"""

    if old_backward not in content:
        print(f"[ERROR] {fpath}: Could not find backward block to patch")
        return False

    content = content.replace(old_backward, new_backward, 1)

    # 4. Add timing around _optimizer_step
    old_optim = """                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics"""

    new_optim = """                _t_opt = time.monotonic()
                grad_norm = self._optimizer_step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _timing_optim_step += time.monotonic() - _t_opt
                _n_mini_batches += 1
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        metrics["timing/forward_s"] = [_timing_forward]
        metrics["timing/backward_s"] = [_timing_backward]
        metrics["timing/optim_step_s"] = [_timing_optim_step]
        metrics["timing/n_micro_batches"] = [_n_micro_batches]
        metrics["timing/n_mini_batches"] = [_n_mini_batches]
        return metrics"""

    if old_optim not in content:
        print(f"[ERROR] {fpath}: Could not find optimizer step block to patch")
        return False

    content = content.replace(old_optim, new_optim, 1)

    if DRY_RUN:
        print(f"[DRY-RUN] Would patch {fpath}")
    else:
        fpath.write_text(content)
        print(f"[OK] Patched {fpath}")
    return True


if __name__ == "__main__":
    print("=== Patching verl for update_actor sub-phase timing ===")
    ok1 = patch_fsdp_workers()
    ok2 = patch_dp_actor()
    if ok1 and ok2:
        print("\nAll patches applied successfully.")
        print("Timing metrics will appear as:")
        print("  timing/load_model_to_gpu    - FSDP param reload from CPU")
        print("  timing/load_optimizer_to_gpu - Optimizer state reload from CPU")
        print("  timing/update_policy         - Total PPO training loop")
        print("  timing/offload_model_to_cpu  - FSDP param offload to CPU")
        print("  timing/offload_optimizer_to_cpu - Optimizer state offload to CPU")
        print("  timing/forward_s             - Sum of all forward passes")
        print("  timing/backward_s            - Sum of all backward passes")
        print("  timing/optim_step_s          - Sum of all optimizer steps")
        print("  timing/n_micro_batches       - Number of micro-batches processed")
        print("  timing/n_mini_batches        - Number of mini-batches processed")
    else:
        print("\nSome patches failed. Check errors above.")
        sys.exit(1)
