"""Patch verl optimizer to work with megatron-core 0.16.0 API."""
import verl.utils.megatron.optimizer as m
fpath = m.__file__
with open(fpath) as f:
    content = f.read()

old = """def get_megatron_optimizer(
    model,
    config: OptimizerConfig,
    no_weight_decay_cond=None,
    scale_lr_cond=None,
    lr_mult=1.0,
):
    # Base optimizer.
    return get_megatron_optimizer_native(
        config=config,
        model_chunks=model,
        no_weight_decay_cond=no_weight_decay_cond,
        scale_lr_cond=scale_lr_cond,
        lr_mult=lr_mult,
    )"""

new = """def get_megatron_optimizer(
    model,
    config: OptimizerConfig,
    no_weight_decay_cond=None,
    scale_lr_cond=None,
    lr_mult=1.0,
):
    # Base optimizer (adapted for megatron-core 0.16.0 API).
    import inspect
    sig = inspect.signature(get_megatron_optimizer_native)
    if "no_weight_decay_cond" in sig.parameters:
        # Old API (megatron-core < 0.16)
        return get_megatron_optimizer_native(
            config=config,
            model_chunks=model,
            no_weight_decay_cond=no_weight_decay_cond,
            scale_lr_cond=scale_lr_cond,
            lr_mult=lr_mult,
        )
    else:
        # New API (megatron-core >= 0.16)
        return get_megatron_optimizer_native(
            config=config,
            model_chunks=model,
        )"""

if old in content:
    content = content.replace(old, new)
    with open(fpath, "w") as f:
        f.write(content)
    print("Patched optimizer.py")
else:
    print("Already patched or content mismatch")
