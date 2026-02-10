# 32xH200 Multi-Node Experiment Log

Date: 2026-02-10
Cluster: 4 nodes x 8x NVIDIA H200 (141GB HBM3e each)
Model: Qwen3-32B
Docker Base: vllm/vllm-openai:v0.10.2

## Cluster Topology

| External IP    | Internal IP    | Hostname             | Role   |
|----------------|----------------|----------------------|--------|
| 111.6.123.35   | 10.83.115.14   | host-10-83-115-14    | HEAD   |
| 111.6.123.36   | 10.83.115.17   | host-10-83-115-17    | Worker |
| 111.6.123.37   | 10.83.115.10   | host-10-83-115-10    | Worker |
| 111.6.123.38   | 10.83.115.12   | host-10-83-115-12    | Worker |

- External IPs (111.6.123.x) are NOT interconnected between nodes
- Inter-node communication uses internal IPs (10.83.115.x) via SSH
- `/root/develop` and `/data` are **local per-node** (not shared NFS)
- Each node has ~2TB CPU RAM, 8x H200 GPUs
- Network: `b_manage0` for management/NCCL TCP; `GPU4-GPU7` interfaces for RDMA (MTU 9000)

## Issues Encountered

### 1. Docker Image: Missing `flash-attn` Package

**Symptom:**
```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error:
the package flash_attn seems to be not installed.
```

**Root Cause:** The `vllm/vllm-openai:v0.10.2` base image does NOT include `flash-attn`. vLLM uses its own attention backend internally, but FSDP training workers use HuggingFace `transformers` which requires the `flash-attn` Python package for FlashAttention2.

**Fix:** Added `pip install flash-attn --no-build-isolation` to `Dockerfile.32h200`. This installs the pre-built wheel (flash-attn 2.8.3). Must use `--no-build-isolation` to leverage the existing torch/cuda in the image.

### 2. Docker Build: Proxy Not Accessible Inside Build Context

**Symptom:** `pip install` commands hang or fail during `docker build` when using `https_proxy=http://127.0.0.1:1083`.

**Root Cause:** Docker build runs in a separate network namespace. `127.0.0.1` inside the build container refers to the container's own loopback, not the host.

**Fix:** Use `docker build --network=host` to share the host's network namespace during build. Pass proxy via build args:
```bash
docker build --network=host \
  --build-arg http_proxy=http://127.0.0.1:1083 \
  --build-arg https_proxy=http://127.0.0.1:1083 \
  -t deepswe-train:latest -f ppio/scripts/Dockerfile.32h200 .
```

### 3. Docker Build: `blinker` Package Conflict

**Symptom:**
```
Cannot uninstall 'blinker'. It is a distutils installed project...
```

**Root Cause:** The base vllm image has `blinker 1.4` installed via distutils, which pip cannot uninstall. R2E-Gym's dependencies require a newer version.

**Fix:** `pip install --ignore-installed blinker` before installing R2E-Gym.

### 4. vLLM Image ENTRYPOINT Override

**Symptom:** Container starts the vLLM API server instead of staying idle for Ray worker use.

**Root Cause:** `vllm/vllm-openai:v0.10.2` has `ENTRYPOINT` set to launch the OpenAI-compatible API server.

**Fix:** Override in Dockerfile: `ENTRYPOINT ["/bin/bash"]`. At runtime: `docker run --entrypoint /bin/bash ... -c 'sleep infinity'`.

### 5. Parquet Loading: Nested Data Arrow Error

**Symptom:**
```
pyarrow.lib.ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs
```

**Root Cause:** The `prepare_swe_data.py` script creates `_verl.parquet` files with deeply nested structures (`extra_info` contains `List(Struct(...))` with further nesting). The `datasets` library's parquet loader cannot handle these nested Arrow types.

**Fix:** Created a conversion script that:
- Keeps `prompt` column as native `List(Struct({content, role}))` (required by verl for chat template)
- Serializes `reward_model` and `extra_info` columns to JSON strings
- Saves as `*_verl_flat.parquet` files

```python
import pyarrow as pa, pyarrow.parquet as pq, polars as pl, json

df = pl.read_parquet("train_verl.parquet")
rows = df.to_dicts()
prompts, reward_models, extra_infos = [], [], []
for row in rows:
    prompts.append(row["prompt"])
    reward_models.append(json.dumps(row["reward_model"]))
    extra_infos.append(json.dumps(row["extra_info"]))

prompt_type = pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))
table = pa.table({
    "prompt": pa.array(prompts, type=prompt_type),
    "reward_model": pa.array(reward_models, type=pa.string()),
    "extra_info": pa.array(extra_infos, type=pa.string()),
})
pq.write_table(table, "train_verl_flat.parquet")
```

### 6. Hydra Config: `ray_init` Key Not in Struct

**Symptom:**
```
Could not override 'ray_init.address'.
Key 'ray_init' is not in struct
```

**Root Cause:** The verl base config does not define `ray_init` at the top level (it's under `ray_kwargs.ray_init`). Hydra's strict mode rejects overriding non-existent keys.

**Fix:** Use `+ray_init.address=auto` (with `+` prefix to add a new key) instead of `ray_init.address=auto` in the training script.

### 7. NCCL Interface Configuration

**Symptom:** NCCL hangs or fails during cross-node communication.

**Root Cause:** Default `NCCL_SOCKET_IFNAME=eth0` doesn't match the actual management interface on PPIO nodes.

**Fix:** Set `NCCL_SOCKET_IFNAME=b_manage0` which is the management network interface available on all nodes. The GPU RDMA interfaces (`GPU4-GPU7`) have MTU 9000 for high-bandwidth inter-node communication, but NCCL TCP fallback uses `b_manage0`.

### 8. Ray Worker Node Dropping

**Symptom:** `ray status` shows only 3/4 nodes (24 GPUs instead of 32).

**Root Cause:** Worker node `10.83.115.10` (111.6.123.37) Ray process died, possibly during flash-attn installation or model download.

**Fix:** Manually restart Ray on the missing worker:
```bash
ssh root@111.6.123.37 "docker exec deepswe-train bash -c 'ray stop --force; ray start --address=10.83.115.14:6379 --num-cpus=64 --num-gpus=8'"
```

### 9. NCCL Init Crash: ActorUnavailableError (UNDER INVESTIGATION)

**Symptom:**
```
ray.exceptions.ActorUnavailableError: The actor is temporarily unavailable:
RpcError: RPC Error message: Socket closed
```
Followed by TCPStore connection failures across multiple ranks.

**Root Cause:** During FSDP model wrapping (after loading checkpoint shards), one actor becomes unavailable. NCCL process group initialization partially completes but the socket to one worker closes. This cascades to all ranks failing.

**Status:** Under investigation. Possible causes:
- NCCL timeout during cross-node all-gather for FSDP parameter sharding
- Network transient failure on `b_manage0` interface
- Ray actor memory pressure during 32-worker model loading

**Potential Fixes to Try:**
1. Set `NCCL_SOCKET_TIMEOUT` to a higher value
2. Use `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` (already set but verify propagation)
3. Reduce `data.filter_overlong_prompts_workers` from 64 to reduce concurrent processes
4. Consider using `NCCL_IB_DISABLE=0` with RDMA interfaces instead of TCP
5. Check Ray actor restart settings

### 10. NumPy Version Conflict with Numba (vLLM)

**Symptom:**
```
ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.4.
```

**Root Cause:** Installing `flash-attn` (via pip) pulls in NumPy 2.4 as a dependency upgrade. However, vLLM's `numba` dependency (used for ngram speculative decoding) requires NumPy <= 2.2.

**Fix:** Pin NumPy after flash-attn install:
```dockerfile
RUN pip install --no-cache-dir flash-attn --no-build-isolation && \
    pip install --no-cache-dir "numpy<2.3"
```

### 11. Container Restart Loses Pip-Installed Packages

**Symptom:** After containers crash and are recreated, packages installed with `pip install` inside the container (like flash-attn) are lost.

**Root Cause:** Containers are created from the base image. Any runtime `pip install` changes are not persisted unless the image is rebuilt.

**Fix:** Rebuild the Docker image (`Dockerfile.32h200`) with all required packages baked in. Build on each node:
```bash
docker build --network=host \
    --build-arg http_proxy=http://127.0.0.1:1083 \
    --build-arg https_proxy=http://127.0.0.1:1083 \
    -t deepswe-train:latest \
    -f ppio/scripts/Dockerfile.32h200 .
```

### 12. NCCL NVLS Warnings: Cuda failure 1 'invalid argument'

**Symptom:**
```
transport/nvls.cc:621 NCCL WARN Cuda failure 1 'invalid argument'
```
Repeated ~1500+ times across all ranks during FSDP all-gather.

**Root Cause:** `NCCL_CUMEM_ENABLE=0` disables cuMem support, but NCCL still attempts NVLS (NVLink Scale) transport which requires cuMem. Each failed attempt logs a warning.

**Impact:** Non-fatal. NCCL falls back to NVLink/Socket transports. Training continues normally.

**Mitigation:** These warnings can be reduced by setting `NCCL_NVLS_ENABLE=0` if desired, but they don't affect correctness.

### 13. Data Format: Use Pre-built `data/swe/*.parquet` (NOT `prepare_swe_data.py`)

**Symptom:** Using `prepare_swe_data.py` generates `_verl.parquet` files with deeply nested Arrow structs that crash with:
```
pyarrow.lib.ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs
```
Serializing `extra_info` to JSON string as a workaround causes:
```
AttributeError: 'str' object has no attribute 'get'
```
in verl's `rl_dataset.py` which expects `extra_info` to be a dict.

**Root Cause:** `prepare_swe_data.py` stores the ENTIRE HuggingFace row as `extra_info`, including complex nested fields (e.g., `modified_entity_summaries` as `List(Struct(...))`). The `datasets` library's Arrow reader cannot handle these deeply nested types.

**Fix:** Use the pre-built `data/swe/*.parquet` files which have clean schemas:
- `data/swe/R2E_Gym_Subset.parquet` (4578 rows, 9.6MB) - training data
- `data/swe/SWE_Bench_Verified.parquet` (500 rows, 2.0MB) - validation data

These files have `extra_info` as a native struct with only flat scalar fields (strings, ints, simple lists). No JSON serialization needed. verl's `RLHFDataset` loads them directly.

## Model & Data Setup

### Model Download
- Used `hf-mirror.com` (accessible without proxy): `HF_ENDPOINT=https://hf-mirror.com`
- Downloaded to `/data/models/Qwen3-32B` on all 4 nodes in parallel
- Size: ~62GB per node (17 safetensors + config files)
- Duration: ~40 minutes per node with parallel downloads

### Data
- Use pre-built parquet files from `data/swe/` directory (synced to all nodes)
- `R2E_Gym_Subset.parquet`: 4578 training samples with clean schema
- `SWE_Bench_Verified.parquet`: 500 validation samples
- Data synced via `rsync` from head to all workers

## Key Configuration

### Docker Run Command
```bash
docker run -d \
  --name deepswe-train \
  --entrypoint /bin/bash \
  --gpus all \
  --net=host \
  --ipc=host \
  --shm-size=64g \
  --cap-add=SYS_ADMIN \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /root/develop:/root/develop \
  -v /data:/data \
  -v /root/.ssh:/root/.ssh:ro \
  -v /tmp:/tmp \
  -e PPIO_API_KEY=sk_xxx \
  -e NCCL_DEBUG=INFO \
  -e NCCL_SOCKET_IFNAME=b_manage0 \
  deepswe-train:latest \
  -c 'sleep infinity'
```

### Training Parameters
| Parameter | 8xH200 | 16xH200 | 32xH200 |
|-----------|---------|---------|---------|
| nnodes | 1 | 2 | 4 |
| n_gpus_per_node | 8 | 8 | 8 |
| train_batch_size | 8 | 16 | 32 |
| ppo_mini_batch_size | 8 | 16 | 32 |
| rollout_n | 8 | 8 | 8 |
| tensor_parallel | 8 | 8 | 8 |
| sequence_parallel | 8 | 8 | 8 |
| val_batch_size | 128 | 256 | 512 |
| gpu_memory_utilization | 0.7 | 0.7 | 0.7 |

### 16xH200 Training Status (2026-02-10)
- Nodes: 111.6.123.35 (HEAD) + 111.6.123.36 (Worker)
- Script: `ppio/scripts/train_qwen3_32b_16h200.sh`
- Data: `data/swe/R2E_Gym_Subset.parquet` (train) + `data/swe/SWE_Bench_Verified.parquet` (val)
- Dataset loaded: 4578 train, 500 val
- NCCL initialized successfully across 2 nodes (16 channels, Socket transport)
- vLLM engines started on both nodes with CUDA graph capture
- First batch of PPIO sandbox environments created and rollouts in progress
