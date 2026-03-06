#!/bin/bash
set -euo pipefail

# ============================================================
# DeepSWE K8s Entrypoint
# Three-phase startup: dockerd → Ray → training
# rllm is mounted at /workspace/rllm (not baked into image)
# ============================================================

# ---------- Environment ----------
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# NCCL RDMA configuration (8x 400G RoCE v2 NICs, 1:1 GPU-NIC PIX mapping)
export NCCL_CUMEM_ENABLE=0  # Disable cuMem (avoids NVLS allocation issues)
export NCCL_NVLS_ENABLE=0   # Disable NVLS transport explicitly (fixes OOM/invalid arg errors on some nodes)
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=b_manage0
export NCCL_DEBUG=INFO

# Gloo (FSDP CPU backend) must use the management NIC, not loopback
export GLOO_SOCKET_IFNAME=b_manage0

NUM_GPUS=8
RAY_PORT=6379
HEAD_SVC="${HEAD_SVC_NAME:-deepswe-training-headless}"
POD_NS="${POD_NAMESPACE:-deepswe}"

# ---------- Detect pod ordinal ----------
# With hostNetwork, hostname is the node's hostname, not the pod name.
# Use POD_NAME from Downward API (set in StatefulSet env).
MY_POD_NAME="${POD_NAME:-$(hostname)}"
ORDINAL="${MY_POD_NAME##*-}"
echo "=== Pod ${MY_POD_NAME} (ordinal ${ORDINAL}) starting ==="

# ---------- Add rllm to PYTHONPATH (mounted, not installed) ----------
if [ -d /workspace/rllm ]; then
    export PYTHONPATH="/workspace/rllm:${PYTHONPATH:-}"
    echo "[setup] Added /workspace/rllm to PYTHONPATH"
fi

# ---------- Patch verl to handle JSON string extra_info ----------
# verl's RLHFDataset.__getitem__ calls .get() on extra_info expecting a dict,
# but our parquet stores it as a JSON string. Patch in-place at startup.
python3 << 'PYEOF'
import verl.utils.dataset.rl_dataset as m
fpath = m.__file__
with open(fpath) as f:
    content = f.read()
if 'json.loads(row_dict["extra_info"])' not in content:
    # Add json import
    content = content.replace("import copy\n", "import copy\nimport json\n", 1)
    # Add JSON deserialization after the None check
    old = '        if "extra_info" not in row_dict or row_dict["extra_info"] is None:\n            row_dict["extra_info"] = dict()'
    new = old + '\n        elif isinstance(row_dict["extra_info"], str):\n            row_dict["extra_info"] = json.loads(row_dict["extra_info"])'
    content = content.replace(old, new)
    with open(fpath, "w") as f:
        f.write(content)
    print("[setup] Patched verl rl_dataset.py to handle JSON string extra_info")
else:
    print("[setup] verl rl_dataset.py already patched")
PYEOF

# ---------- Phase 1: Start dockerd (DinD) ----------
echo "[phase-1] Starting dockerd..."

# Configure Docker daemon proxy for pulling images (hostNetwork shares host's loopback proxy)
export HTTP_PROXY=http://127.0.0.1:1083
export HTTPS_PROXY=http://127.0.0.1:1083
export NO_PROXY=localhost,127.0.0.1,10.0.0.0/8
echo "[phase-1] Docker proxy configured: ${HTTPS_PROXY}"

# Configure registry mirror (pull-through cache) and storage driver via daemon.json
REGISTRY_MIRROR="${REGISTRY_MIRROR:-}"
mkdir -p /etc/docker
if [ -n "${REGISTRY_MIRROR}" ]; then
    cat > /etc/docker/daemon.json <<DEOF
{
    "storage-driver": "overlay2",
    "registry-mirrors": ["${REGISTRY_MIRROR}"],
    "insecure-registries": ["${REGISTRY_MIRROR#http://}"]
}
DEOF
    echo "[phase-1] Registry mirror configured: ${REGISTRY_MIRROR}"
    dockerd --host=unix:///var/run/docker.sock &
else
    dockerd --storage-driver=overlay2 --host=unix:///var/run/docker.sock &
fi
DOCKERD_PID=$!

# Poll until dockerd is ready (up to 60s)
for i in $(seq 1 60); do
    if docker info &>/dev/null; then
        echo "[phase-1] dockerd ready after ${i}s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "[phase-1] ERROR: dockerd failed to start within 60s"
        exit 1
    fi
    sleep 1
done

# DockerHub login if credentials provided
if [ -n "${DOCKERHUB_USERNAME:-}" ] && [ -n "${DOCKERHUB_TOKEN:-}" ]; then
    echo "[phase-1] Logging into DockerHub..."
    echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
fi

# Pre-warm images if requested (typically first run only)
if [ "${PREWARM_IMAGES:-false}" = "true" ]; then
    echo "[phase-1] Pre-warming Docker images..."
    python3 /usr/local/bin/prewarm-images.py
    echo "[phase-1] Image pre-warming complete"
fi

# ---------- Phase 2: Start Ray ----------
echo "[phase-2] Starting Ray (ordinal=${ORDINAL})..."

# With hostNetwork, hostname may not resolve (gethostbyname fails).
# Explicitly detect node IP from b_manage0 interface.
NODE_IP=$(ip -4 addr show b_manage0 2>/dev/null | grep -oP 'inet \K[0-9.]+' || hostname -I | awk '{print $1}')
echo "[phase-2] Detected node IP: ${NODE_IP}"

# With hostNetwork, Ray's random port picks for internal components can collide
# with the worker_ports range (default 10002-19999). Move worker ports to 30000-39999
# and pin other ports outside both ranges to prevent conflicts.
RAY_COMMON_ARGS="--node-ip-address=${NODE_IP} --num-gpus=${NUM_GPUS} \
    --min-worker-port=30000 --max-worker-port=39999 \
    --metrics-export-port=20100 \
    --runtime-env-agent-port=20400 \
    --dashboard-agent-grpc-port=20200 \
    --dashboard-agent-listen-port=20300"

if [ "$ORDINAL" = "0" ]; then
    # Head node
    ray start --head --port=${RAY_PORT} ${RAY_COMMON_ARGS}
    echo "[phase-2] Ray head started on port ${RAY_PORT}"
else
    # Worker node: resolve head pod's IP via headless service
    # Derive head pod name from current pod name (e.g., deepswe-30b-1 -> deepswe-30b-0)
    POD_PREFIX="${MY_POD_NAME%-*}"  # Remove ordinal suffix
    HEAD_HOST="${POD_PREFIX}-0.${HEAD_SVC}.${POD_NS}.svc.cluster.local"

    echo "[phase-2] Waiting for Ray head at ${HEAD_HOST}:${RAY_PORT}..."
    for i in $(seq 1 120); do
        if ray start --address="${HEAD_HOST}:${RAY_PORT}" ${RAY_COMMON_ARGS} 2>/dev/null; then
            echo "[phase-2] Ray worker connected after ${i}s"
            break
        fi
        if [ "$i" -eq 120 ]; then
            echo "[phase-2] ERROR: Could not connect to Ray head within 120s"
            exit 1
        fi
        sleep 1
    done
fi

# ---------- Phase 3: Launch training (head only) / Wait (workers) ----------
if [ "$ORDINAL" = "0" ]; then
    echo "[phase-3] Head node: waiting for all Ray nodes..."

    EXPECTED_NODES=2
    for i in $(seq 1 300); do
        ACTIVE_NODES=$(python3 -c "
import ray
ray.init(address='auto', ignore_reinit_error=True)
nodes = ray.nodes()
active = sum(1 for n in nodes if n['Alive'])
print(active)
ray.shutdown()
" 2>/dev/null || echo "0")

        if [ "$ACTIVE_NODES" -ge "$EXPECTED_NODES" ]; then
            echo "[phase-3] All ${EXPECTED_NODES} Ray nodes are ready"
            break
        fi
        if [ "$i" -eq 300 ]; then
            echo "[phase-3] WARNING: Only ${ACTIVE_NODES}/${EXPECTED_NODES} nodes after 300s, proceeding anyway"
            break
        fi
        if [ $((i % 10)) -eq 0 ]; then
            echo "[phase-3] Waiting for nodes: ${ACTIVE_NODES}/${EXPECTED_NODES} (${i}s)..."
        fi
        sleep 1
    done

    echo "[phase-3] Launching training..."
    ray status
    # Training script can be passed as first argument or via TRAIN_SCRIPT env var
    TRAIN_SCRIPT="${1:-${TRAIN_SCRIPT:-train_deepswe_32b_k8s.sh}}"
    bash /workspace/rllm/k8s/scripts/${TRAIN_SCRIPT} || true

    echo "[phase-3] Training script exited, keeping pod alive for debugging"
    # Keep pod alive so we can inspect logs / re-run manually
    wait $DOCKERD_PID
else
    echo "[phase-3] Worker node: idling (Ray worker active)"
    # Keep the container alive so Ray worker stays up
    wait $DOCKERD_PID
fi
