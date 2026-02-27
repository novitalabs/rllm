# DeepSWE 32B Training Quick Start Guide

## Prerequisites

- 8 nodes, each with 8x H200 GPUs
- NVIDIA driver 580.x, CUDA 12.8+
- Python 3.10, torch 2.10.0+cu128
- Network connectivity between all nodes (Ray uses port 6379)

## Step 1: Install Dependencies on All Nodes

```bash
# On each node:
pip install verl==0.6.1
MAX_JOBS=64 pip install flash-attn --no-build-isolation

# Build vLLM 0.11.2 from source (required for torch 2.10.0 compatibility)
git clone https://github.com/vllm-project/vllm.git -b v0.11.2
cd vllm
MAX_JOBS=64 pip install -e . --no-build-isolation

# Install R2E-Gym (on head node only, for SWE environments)
pip install r2e-gym
```

## Step 2: Build ABI Shim (All Nodes)

```bash
cd /path/to/rllm/ppio
bash scripts/build_shim.sh /tmp/torch_compat_shim.so
```

## Step 3: Apply Patches (Head Node)

Apply the verl/vLLM patches listed in `patches/README.md` to the head node's site-packages.

## Step 4: Distribute Patches to Workers

```bash
bash ppio/scripts/distribute_patches.sh
```

## Step 5: Distribute Model & Data

Ensure `/data/models/Qwen3-32B` and training data exist on all nodes.

## Step 6: Setup Docker (Head Node)

```bash
bash ppio/scripts/setup_docker.sh
```

## Step 7: Start Ray Cluster

```bash
# Head node:
bash ppio/scripts/setup_ray_cluster.sh head

# Each worker node:
bash ppio/scripts/setup_ray_cluster.sh worker
```

Verify: `ray status` should show 8 nodes with 64 GPUs total.

## Step 8: Launch Training

```bash
nohup bash examples/swe/train_deepswe_32b.sh > train.log 2>&1 &
```

## Monitoring

```bash
# Training log
tail -f train.log

# Ray dashboard
# http://<head_ip>:8265

# Ray worker logs
ls -lt /tmp/ray/session_latest/logs/worker-*.err

# Docker containers (SWE environments)
docker ps | wc -l
```

## Key Configuration Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `trainer.nnodes` | 8 | Number of nodes |
| `trainer.n_gpus_per_node` | 8 | GPUs per node |
| `rollout.tensor_model_parallel_size` | 8 | TP size for vLLM |
| `rollout.gpu_memory_utilization` | 0.4 | GPU memory for KV cache |
| `rollout.free_cache_engine` | False | Keep KV cache during training |
| `actor.fsdp_config.param_offload` | True | Offload FSDP params to CPU |
| `actor.fsdp_config.optimizer_offload` | True | Offload optimizer to CPU |
| `+rllm.env.env_args.backend` | docker | Use Docker for SWE containers |
| `rollout.n` | 8 | Samples per prompt |
| `data.max_response_length` | 32768 | Max response tokens |

## Environment Variables

```bash
# Required:
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/tmp/torch_compat_shim.so
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
```
