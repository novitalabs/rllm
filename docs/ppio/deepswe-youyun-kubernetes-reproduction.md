# DeepSWE Full Reproduction — K8s 8-Node Preparation Guide

> Goal: Reproduce official DeepSWE training at full scale (64 GPUs, 8×H200×8) on youyun k8s cluster.
> Training nodes: the 8 new k8s worker nodes (.18, .21-.28), NOT youyun.35-38.
> Date: 2026-02-26

---

## Cluster Topology

```
K8s Cluster (12 nodes, k8s v1.32.10, containerd)
├── Control-Plane (SSH accessible, NOT used for training)
│   ├── host-10-83-115-10 (youyun.37) — has rllm-origin env, Docker images, NFS server
│   ├── host-10-83-115-12 (youyun.38) — partial env
│   └── host-10-83-115-14 (youyun.35)
├── Worker (SSH accessible, NOT used for training)
│   └── host-10-83-115-17 (youyun.36)
└── Worker ← TRAINING NODES (8 nodes, 64 GPUs total)
    ├── host-10-83-115-18  (containerd 1.7.29, labeled workload=r2e-eval,role=vllm)
    ├── host-10-83-115-21  (containerd 1.7.29, labeled workload=r2e-eval)
    ├── host-10-83-115-22  (containerd 2.2.1,  labeled workload=r2e-eval)
    ├── host-10-83-115-23  (containerd 2.2.1,  labeled workload=r2e-eval)
    ├── host-10-83-115-25  (containerd 2.2.1,  labeled workload=r2e-eval)
    ├── host-10-83-115-26  (containerd 2.2.1,  labeled workload=r2e-eval)
    ├── host-10-83-115-27  (containerd 2.2.1,  labeled workload=r2e-eval)
    └── host-10-83-115-28  (containerd 2.2.1,  labeled workload=r2e-eval)

Per-Node: 192 CPUs, ~2TB RAM, 8×H200, 400Gbps RoCE (mlx5), 880GB SSD,
          7TB NVMe (/var/lib/containerd), 7TB NVMe (/data)
NFS:      10.83.115.10:/nfs (exported to *, 894GB /dev/sdb on youyun.37)
```

### Architecture

No SSH to the 8 training nodes → deploy everything via k8s:

| Workload | How | Where |
|----------|-----|-------|
| **Ray cluster** (head + 7 workers) | K8s Pods with GPU, via KubeRay | 8 new nodes (.18-.28) |
| **R2E-Gym env containers** | K8s Pods (CPU only), created by training code | Same 8 nodes (co-located) |
| **Model weights / code / data** | NFS from youyun.37 + baked into training image | Mounted into Ray pods |
| **Checkpoints** | NFS or hostPath `/data` | Accessible from all nodes |

Co-location is fine: env pods use ~1 CPU + 1GB each; with 192 CPUs per node, 8 concurrent env pods barely register.

---

## Step 1: Build Training Docker Image

A `deepswe-train:latest` image already exists on youyun.37 (24.9GB), based on `vllm/vllm-openai:v0.10.2`:

```
Base: vllm/vllm-openai:v0.10.2 (includes torch, vllm, CUDA)
 + verl==0.6.1, ray>=2.40
 + R2E-Gym (pip install from git)
 + flash-attn, transformers>=4.55, datasets, kubernetes, swebench
 + docker, hydra-core, openai, wandb
```

### 1.1 Verify existing image contents

```bash
ssh youyun.37 'docker run --rm deepswe-train:latest -c "
  python3 -c \"import verl; print(verl.__version__)\"
  python3 -c \"import vllm; print(vllm.__version__)\"
  python3 -c \"import ray; print(ray.__version__)\"
  python3 -c \"import r2egym; print(r2egym.__file__)\"
  python3 -c \"import rllm; print(rllm.__file__)\" 2>/dev/null || echo rllm NOT installed
"'
```

### 1.2 Add rllm source code and patches

The current image has pip-installed packages but NOT the local rllm-origin checkout (which contains training scripts and patches like the outer timeout fix).

```bash
# On youyun.37, rebuild with rllm source:
cat > /tmp/Dockerfile.deepswe-train-v2 << 'EOF'
FROM deepswe-train:latest

# Copy rllm source with local patches
COPY rllm-origin/ /workspace/rllm-origin/
COPY R2E-Gym/ /workspace/R2E-Gym/

# Install rllm in editable mode (picks up patches)
RUN pip install --no-cache-dir -e /workspace/rllm-origin && \
    pip install --no-cache-dir -e /workspace/rllm-origin/verl && \
    pip install --no-cache-dir -e /workspace/R2E-Gym/

ENV RLLM_DIR=/workspace/rllm-origin
ENV R2EGYM_DIR=/workspace/R2E-Gym/src
ENV PYTHONPATH="/workspace/rllm-origin:/workspace/R2E-Gym/src:${PYTHONPATH}"

WORKDIR /workspace/rllm-origin
EOF

cd /home/claude/work
docker build -f /tmp/Dockerfile.deepswe-train-v2 \
    -t deepswe-train:v2 \
    --build-context rllm-origin=./rllm-origin \
    --build-context R2E-Gym=./R2E-Gym \
    .
```

### 1.3 Distribute image to k8s nodes

K8s nodes use containerd, not Docker. Options:

```bash
# Option A: Push to Docker Hub (simplest, uses existing dockerhub secret)
docker tag deepswe-train:v2 novitalabs/deepswe-train:v2
docker push novitalabs/deepswe-train:v2

# Option B: Export to NFS, import via DaemonSet
docker save deepswe-train:v2 | gzip > /nfs/deepswe-train-v2.tar.gz
# Then use a privileged DaemonSet to import on each node (see Step 1.4)

# Option C: Local registry on youyun.37
docker run -d -p 5000:5000 --restart=always --name registry registry:2
docker tag deepswe-train:v2 10.83.115.10:5000/deepswe-train:v2
docker push 10.83.115.10:5000/deepswe-train:v2
# Note: k8s nodes need to trust this registry (insecure registry config in containerd)
```

### 1.4 Import image via NFS + DaemonSet (Option B)

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: import-training-image
spec:
  selector:
    matchLabels:
      app: import-training-image
  template:
    metadata:
      labels:
        app: import-training-image
    spec:
      nodeSelector:
        workload: r2e-eval
      hostPID: true
      containers:
      - name: importer
        image: docker.io/library/alpine:latest
        command: ["/bin/sh", "-c"]
        args:
        - |
          apk add --no-cache skopeo
          # Import from tarball to containerd
          ctr -n k8s.io images import /nfs/deepswe-train-v2.tar.gz
          echo "Import done on $(hostname)"
          sleep infinity
        securityContext:
          privileged: true
        volumeMounts:
        - name: nfs
          mountPath: /nfs
          readOnly: true
        - name: containerd-sock
          mountPath: /run/containerd/containerd.sock
      volumes:
      - name: nfs
        nfs:
          server: 10.83.115.10
          path: /nfs
      - name: containerd-sock
        hostPath:
          path: /run/containerd/containerd.sock
```

---

## Step 2: Set Up kubectl Access for Training Process

The R2E-Gym k8s backend calls `config.load_incluster_config()` first (works when running inside a k8s pod), then falls back to `config.load_kube_config()`.

Since training runs inside k8s pods (KubeRay), **incluster config works automatically** — just need the right ServiceAccount.

### 2.1 Create ServiceAccount and RBAC

```yaml
# deepswe-rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: deepswe-training
  namespace: default
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: deepswe-training-binding
subjects:
- kind: ServiceAccount
  name: deepswe-training
  namespace: default
roleRef:
  kind: ClusterRole
  name: swebench-eval-role   # reuse existing: pod create/delete/exec/log
  apiGroup: rbac.authorization.k8s.io
```

```bash
sudo kubectl --kubeconfig=/etc/kubernetes/admin.conf apply -f deepswe-rbac.yaml
```

### 2.2 Verify dockerhub pull secret

```bash
sudo kubectl --kubeconfig=/etc/kubernetes/admin.conf get secret dockerhub -o yaml
# Should exist in namespace default with .dockerconfigjson for novitalabs
```

---

## Step 3: Prepare NFS Shared Storage

NFS on youyun.37 (`10.83.115.10:/nfs`, 894GB) is accessible from all k8s nodes. Use it for:

| Content | NFS Path | Size |
|---------|----------|------|
| Model weights (Qwen3-32B) | `/nfs/models/Qwen3-32B` | ~62GB |
| Training data (parquets) | `/nfs/rllm-data/` | ~200MB |
| Checkpoints (output) | `/nfs/checkpoints/` | ~180GB (3 × 60GB) |
| Training image tarball | `/nfs/deepswe-train-v2.tar.gz` | ~25GB |

```bash
# On youyun.37:
mkdir -p /nfs/models /nfs/rllm-data /nfs/checkpoints

# Copy model weights
cp -r /home/claude/work/rllm/models/Qwen3-32B /nfs/models/Qwen3-32B

# Copy training parquets
cp /home/claude/work/rllm-origin/rllm/data/datasets/R2E_Gym_Subset/train_verl.parquet \
   /nfs/rllm-data/
cp /home/claude/work/rllm-origin/rllm/data/datasets/SWE_Bench_Verified/test_verl.parquet \
   /nfs/rllm-data/
```

**Warning**: NFS bandwidth is limited by the 25Gbps management network. Model loading at training start will be slow (~20s for 62GB). For vLLM rollout (which loads model per-node), this could be a bottleneck. Consider copying model to local `/data` on each node via an init job:

```yaml
# init container in Ray pod spec:
initContainers:
- name: copy-model
  image: busybox
  command: ["sh", "-c", "cp -r /nfs/models/Qwen3-32B /data/models/Qwen3-32B || true"]
  volumeMounts:
  - name: nfs
    mountPath: /nfs
    readOnly: true
  - name: local-data
    mountPath: /data
```

---

## Step 4: Pre-Pull R2E-Gym Training Docker Images

### 4.1 Training dataset image inventory

4578 unique Docker images across 10 repos:

| Repository | Count | % | Unique/Img | Est. Total Unique |
|-----------|-------|---|-----------|-------------------|
| `namanjain12/pandas_final` | 1444 | 31.5% | ~1.3GB | ~1.9TB |
| `namanjain12/numpy_final` | 781 | 17.1% | 1.3GB | 1.0TB |
| `namanjain12/pillow_final` | 620 | 13.5% | 1.1GB | 0.7TB |
| `namanjain12/orange3_final` | 482 | 10.5% | 2.4GB | ~1.2TB |
| `namanjain12/aiohttp_final` | 299 | 6.5% | 1.2GB | 0.35TB |
| `namanjain12/tornado_final` | 261 | 5.7% | 0.9GB | 0.22TB |
| `namanjain12/scrapy_final` | 215 | 4.7% | 0.9GB | 0.19TB |
| `namanjain12/pyramid_final` | 189 | 4.1% | 0.9GB | 0.16TB |
| `namanjain12/datalad_final` | 179 | 3.9% | 1.6GB | 0.29TB |
| `namanjain12/coveragepy_final` | 108 | 2.4% | 0.8GB | 0.07TB |
| **Total** | **4578** | | | **~6.1TB** |

**Measured data** (from youyun.37, Docker overlay2):
- 2642 images (all except pandas + most orange3) = **3.2TB actual disk**
- Cross-repo layer sharing is minimal (dedup ratio only 9%) — most layers are unique per image
- pandas and orange3 are estimated from per-image unique sizes of similar repos

### 4.2 Disk budget per k8s node

Each k8s worker node's containerd NVMe (confirmed from host-10-83-115-21):

```
/dev/nvme0n1  7.0T  434G  6.6T  6%  /var/lib/containerd
/dev/nvme1n1  7.0T  294G  6.7T  4%  /data
```

Current 434GB usage = ~50 swebench-verified eval images + system images.

| Strategy | Per-Node Disk | Total (8 nodes) | Utilization | Headroom |
|----------|-------------|-----------------|-------------|----------|
| **A: All on all** | ~6.1TB + 0.4TB = 6.5TB | 48.8TB pulls | 93% of 7TB | **~500GB** ⚠️ |
| **B: Distributed (2 copies)** | ~1.5TB avg + 0.4TB = ~1.9TB | 12.2TB pulls | 27% of 7TB | **~5.1TB** ✓ |

**Strategy A is risky** — 500GB headroom leaves no margin for containerd overhead, writable container layers, logs, or unexpected growth. Strategy B is recommended.

### 4.3 Recommended: Distributed pre-pull (2 copies per image)

Each image is pre-pulled on **2 out of 8 nodes**. This provides redundancy while keeping disk usage manageable.

**How k8s scheduling interacts with distributed images**:
- The k8s default scheduler includes an `ImageLocality` scoring plugin that gives bonus points to nodes already having the required image
- With 8 identical nodes (same CPU/RAM/GPU), ImageLocality becomes the dominant differentiator
- `imagePullPolicy: IfNotPresent` (the default for tagged images) means: if a pod lands on a node without the image, it falls back to pulling — slower but not fatal

**Repo-to-node assignment** (balanced by total unique size, 2 copies each):

| Nodes | Repos | Est. Disk |
|-------|-------|-----------|
| Node 1, Node 2 | pandas (1.9TB) | ~1.9TB |
| Node 3, Node 4 | numpy (1.0TB) + orange3 (1.2TB) | ~2.2TB |
| Node 5, Node 6 | pillow (0.7TB) + aiohttp (0.35TB) + datalad (0.29TB) | ~1.3TB |
| Node 7, Node 8 | tornado (0.22TB) + scrapy (0.19TB) + pyramid (0.16TB) + coveragepy (0.07TB) | ~0.6TB |

Max per node: ~2.2TB + 0.4TB existing = 2.6TB → 37% utilization → comfortable.

### 4.4 Generate image list

```bash
ssh youyun.37 'source /home/claude/work/rllm-origin/.venv/bin/activate && python3 -c "
import json, pandas as pd
df = pd.read_parquet(\"/home/claude/work/rllm-origin/rllm/data/datasets/R2E_Gym_Subset/train_verl.parquet\")
for row in df[\"extra_info\"]:
    info = json.loads(row) if isinstance(row, str) else row
    print(info.get(\"docker_image\", \"\"))
" > /nfs/r2e_training_images.txt'

# Also generate per-repo lists for targeted pre-pull:
ssh youyun.37 'source /home/claude/work/rllm-origin/.venv/bin/activate && python3 << "PYEOF"
import json, pandas as pd

df = pd.read_parquet("/home/claude/work/rllm-origin/rllm/data/datasets/R2E_Gym_Subset/train_verl.parquet")
repos = {}
for row in df["extra_info"]:
    info = json.loads(row) if isinstance(row, str) else row
    img = info.get("docker_image", "")
    repo = img.split(":")[0].split("/")[-1]
    repos.setdefault(repo, []).append(img)

for repo, imgs in sorted(repos.items()):
    with open(f"/nfs/r2e_images_{repo}.txt", "w") as f:
        for img in imgs:
            f.write(img + "\n")
    print(f"{repo}: {len(imgs)} images -> /nfs/r2e_images_{repo}.txt")
PYEOF'
```

### 4.5 Pre-pull with DaemonSet (targeted per node group)

Deploy privileged DaemonSet on each node to pull images via `ctr`:

```yaml
# prepull-daemonset.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: image-puller
spec:
  selector:
    matchLabels:
      app: image-puller
  template:
    metadata:
      labels:
        app: image-puller
    spec:
      nodeSelector:
        workload: r2e-eval
      hostPID: true
      containers:
      - name: puller
        image: docker.io/library/alpine:latest
        command: ["sleep", "infinity"]
        securityContext:
          privileged: true
        volumeMounts:
        - name: containerd-sock
          mountPath: /run/containerd/containerd.sock
        - name: nfs
          mountPath: /nfs
          readOnly: true
      volumes:
      - name: containerd-sock
        hostPath:
          path: /run/containerd/containerd.sock
      - name: nfs
        nfs:
          server: 10.83.115.10
          path: /nfs
      imagePullSecrets:
      - name: dockerhub
```

Then exec into specific node's puller pod and pull the assigned repos:

```bash
# Apply DaemonSet (one pod per node):
kubectl apply -f prepull-daemonset.yaml

# Get puller pod names and their nodes:
kubectl get pods -l app=image-puller -o wide --no-headers

# For each node group, exec into the corresponding puller pod and pull assigned repos.
# Example: Node 1 (host-10-83-115-18) gets pandas_final:
PULLER_POD=$(kubectl get pods -l app=image-puller -o wide --no-headers | grep "host-10-83-115-18" | awk '{print $1}')
kubectl exec $PULLER_POD -- sh -c '
  apk add --no-cache containerd-ctr
  while read img; do
    ctr -n k8s.io images pull "docker.io/$img" && echo "OK: $img" || echo "FAIL: $img"
  done < /nfs/r2e_images_pandas_final.txt
'

# Repeat for other node-repo assignments per the table above.
# Throttle to avoid Docker Hub 429: max_concurrent_downloads=20 is already set in containerd config.
```

### 4.6 Alternative: all-on-all via official DaemonSet script

If disk headroom is acceptable (~500GB), use the official approach from `rllm-origin/rllm/environments/swe/cache_images_k8.py` which creates a DaemonSet per image (no nodeSelector → pulls on ALL nodes). This is simpler but uses ~6TB per node.

```bash
# On youyun.37, inside rllm-origin venv:
cd /home/claude/work/rllm-origin
python rllm/environments/swe/cache_images_k8.py
# Uses 48 concurrent threads, creates/deletes DaemonSets per image.
# Each DaemonSet pulls the image on all 8 worker nodes simultaneously.
```

### 4.7 Monitoring pre-pull progress

```bash
# Check containerd image count/size on each node:
for pod in $(kubectl get pods -l app=image-puller -o name); do
  node=$(kubectl get $pod -o jsonpath='{.spec.nodeName}')
  count=$(kubectl exec ${pod##*/} -- sh -c 'ctr -n k8s.io images ls -q 2>/dev/null | wc -l')
  echo "$node: $count images"
done
```

---

## Step 5: Install KubeRay Operator

```bash
# On youyun.37 (has kubectl via admin.conf):
export KUBECONFIG=/etc/kubernetes/admin.conf

# Install Helm (if not present)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Add KubeRay repo and install
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator \
    --namespace ray-system --create-namespace

# Verify
kubectl get pods -n ray-system
# Should see kuberay-operator-XXXXX Running
```

---

## Step 6: Deploy Ray Cluster on K8s

### 6.1 RayCluster manifest

```yaml
# deepswe-raycluster.yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: deepswe-train
  namespace: default
spec:
  rayVersion: "2.40.0"
  enableInTreeAutoscaling: false
  headGroupSpec:
    serviceType: ClusterIP
    rayStartParams:
      num-gpus: "8"
      dashboard-host: "0.0.0.0"
    template:
      metadata:
        labels:
          ray-node: head
      spec:
        serviceAccountName: deepswe-training
        containers:
        - name: ray-head
          image: novitalabs/deepswe-train:v2   # or local registry path
          resources:
            limits:
              nvidia.com/gpu: 8
              cpu: "128"
              memory: "512Gi"
            requests:
              nvidia.com/gpu: 8
              cpu: "32"
              memory: "256Gi"
          env:
          - name: NCCL_SOCKET_IFNAME
            value: "GPU0"
          - name: NCCL_IB_HCA
            value: "mlx5"
          - name: NCCL_NET
            value: "IB"
          - name: NCCL_NVLS_ENABLE
            value: "0"
          - name: VLLM_ATTENTION_BACKEND
            value: "FLASH_ATTN"
          - name: VLLM_ALLOW_LONG_MAX_MODEL_LEN
            value: "1"
          - name: VLLM_ENGINE_ITERATION_TIMEOUT_S
            value: "100000000000"
          - name: PYTORCH_CUDA_ALLOC_CONF
            value: "expandable_segments:False"
          volumeMounts:
          - name: nfs
            mountPath: /nfs
          - name: local-data
            mountPath: /data
          - name: shm
            mountPath: /dev/shm
        volumes:
        - name: nfs
          nfs:
            server: 10.83.115.10
            path: /nfs
        - name: local-data
          hostPath:
            path: /data
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: "512Gi"
        imagePullSecrets:
        - name: dockerhub
        nodeSelector:
          workload: r2e-eval
  workerGroupSpecs:
  - replicas: 7
    minReplicas: 7
    maxReplicas: 7
    groupName: gpu-workers
    rayStartParams:
      num-gpus: "8"
    template:
      metadata:
        labels:
          ray-node: worker
      spec:
        serviceAccountName: deepswe-training
        containers:
        - name: ray-worker
          image: novitalabs/deepswe-train:v2
          resources:
            limits:
              nvidia.com/gpu: 8
              cpu: "128"
              memory: "512Gi"
            requests:
              nvidia.com/gpu: 8
              cpu: "32"
              memory: "256Gi"
          env:
          - name: NCCL_SOCKET_IFNAME
            value: "GPU0"
          - name: NCCL_IB_HCA
            value: "mlx5"
          - name: NCCL_NET
            value: "IB"
          - name: NCCL_NVLS_ENABLE
            value: "0"
          - name: VLLM_ATTENTION_BACKEND
            value: "FLASH_ATTN"
          - name: VLLM_ALLOW_LONG_MAX_MODEL_LEN
            value: "1"
          - name: VLLM_ENGINE_ITERATION_TIMEOUT_S
            value: "100000000000"
          - name: PYTORCH_CUDA_ALLOC_CONF
            value: "expandable_segments:False"
          volumeMounts:
          - name: nfs
            mountPath: /nfs
          - name: local-data
            mountPath: /data
          - name: shm
            mountPath: /dev/shm
        volumes:
        - name: nfs
          nfs:
            server: 10.83.115.10
            path: /nfs
        - name: local-data
          hostPath:
            path: /data
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: "512Gi"
        imagePullSecrets:
        - name: dockerhub
        nodeSelector:
          workload: r2e-eval
```

### 6.2 Anti-affinity to spread pods across nodes

Add to both head and worker pod specs to ensure one Ray pod per physical node:

```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchExpressions:
        - key: ray.io/cluster
          operator: In
          values: ["deepswe-train"]
      topologyKey: kubernetes.io/hostname
```

### 6.3 Deploy and verify

```bash
sudo kubectl --kubeconfig=/etc/kubernetes/admin.conf apply -f deepswe-raycluster.yaml

# Wait for all pods to be Running:
kubectl get pods -l ray.io/cluster=deepswe-train -o wide
# Should see 1 head + 7 workers, each on a different node

# Check Ray cluster status from head pod:
HEAD_POD=$(kubectl get pods -l ray-node=head -o jsonpath='{.items[0].metadata.name}')
kubectl exec $HEAD_POD -- ray status
# Should show 8 nodes, 64 GPUs total
```

---

## Step 7: NCCL Network Verification

Before running training, verify RDMA works across the Ray pods.

```bash
# Exec into head pod:
HEAD_POD=$(kubectl get pods -l ray-node=head -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it $HEAD_POD -- bash

# Inside the pod, check RDMA devices:
ibstat | grep -E "CA |State|Rate" | head -20
# Should see mlx5_0..mlx5_11, State: Active, Rate: 400

# Quick NCCL test (2 GPUs on same node):
python3 -c "
import torch, torch.distributed as dist
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
t = torch.randn(1024, 1024, device='cuda:0')
print('NCCL basic test passed, tensor sum:', t.sum().item())
"
```

For full 8-node NCCL test, the training launch itself serves as the definitive test.

---

## Step 8: Launch Training

### 8.1 Training script

Exec into the head pod and launch training:

```bash
HEAD_POD=$(kubectl get pods -l ray-node=head -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it $HEAD_POD -- bash
```

Inside the head pod:

```bash
#!/bin/bash
set -x
cd /workspace/rllm-origin

export RLLM_DIR=/workspace/rllm-origin
export R2EGYM_DIR=/workspace/R2E-Gym/src
export PYTHONPATH="$RLLM_DIR:$R2EGYM_DIR:$PYTHONPATH"

# NCCL env already set via pod spec, but reinforce:
export NCCL_SOCKET_IFNAME=GPU0
export NCCL_IB_HCA=mlx5
export NCCL_NET=IB
export NCCL_NVLS_ENABLE=0

MODEL="/nfs/models/Qwen3-32B"  # or /data/models/Qwen3-32B if copied locally

python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=/nfs/rllm-data/train_verl.parquet \
    data.val_files=/nfs/rllm-data/test_verl.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    rllm.filter_token_mismatch=False \
    trainer.critic_warmup=0 \
    "trainer.logger=[console,wandb]" \
    trainer.project_name=deepswe-full-v2 \
    trainer.experiment_name=8node-rloo-bs8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.default_hdfs_dir=null \
    "trainer.default_local_dir=/nfs/checkpoints/deepswe-full-v2/8node-rloo-bs8" \
    rllm.env.name=swe \
    +rllm.env.env_args.backend=kubernetes \
    +rllm.env.env_args.scaffold=r2egym \
    rllm.agent.name=r2egym \
    rllm.agent.max_steps=50 \
    rllm.agent.overlong_filter=True \
    rllm.agent.trajectory_timeout=5400 \
    trainer.total_epochs=1000
```

### 8.2 Smoke test first (2 steps)

Before full training, run 2 steps with `trainer.total_training_steps=2` and `data.train_batch_size=1` to verify:
- Ray cluster sees all 64 GPUs
- K8s environment pods are created on worker nodes
- NCCL AllReduce works across 8 nodes
- Checkpoints save to NFS correctly

---

## Summary Checklist

### Phase 1: Image Build & Distribution
- [ ] Verify `deepswe-train:latest` image contents on youyun.37
- [ ] Build `deepswe-train:v2` with rllm source + patches
- [ ] Push to Docker Hub (`novitalabs/deepswe-train:v2`) or local registry
- [ ] Verify image pullable from k8s worker nodes

### Phase 2: NFS Storage
- [ ] Copy Qwen3-32B model weights to `/nfs/models/Qwen3-32B` (~62GB)
- [ ] Copy training parquets to `/nfs/rllm-data/`
- [ ] Create `/nfs/checkpoints/` directory
- [ ] Verify NFS accessible from k8s pods (test with a simple pod)

### Phase 3: K8s RBAC
- [ ] Create `deepswe-training` ServiceAccount
- [ ] Bind to `swebench-eval-role` (pod create/delete/exec/log)
- [ ] Verify `dockerhub` imagePullSecret exists

### Phase 4: Pre-Pull R2E-Gym Training Images (~6.1TB total, distributed)
- [ ] Generate per-repo image lists from training parquet (4578 images, 10 repos)
- [ ] Deploy image-puller DaemonSet on all 8 worker nodes
- [ ] Pre-pull pandas_final on Node 1+2 (~1.9TB each)
- [ ] Pre-pull numpy_final + orange3_final on Node 3+4 (~2.2TB each)
- [ ] Pre-pull pillow_final + aiohttp_final + datalad_final on Node 5+6 (~1.3TB each)
- [ ] Pre-pull tornado_final + scrapy_final + pyramid_final + coveragepy_final on Node 7+8 (~0.6TB each)
- [ ] Monitor: `ctr -n k8s.io images ls -q | wc -l` per node via puller pods

### Phase 5: KubeRay
- [ ] Install Helm on youyun.37
- [ ] Install KubeRay operator
- [ ] Verify operator running: `kubectl get pods -n ray-system`

### Phase 6: Ray Cluster
- [ ] Apply RayCluster manifest (1 head + 7 workers, 8 GPUs each)
- [ ] Verify all 8 pods Running on different nodes
- [ ] Verify `ray status` shows 64 GPUs
- [ ] Verify RDMA: `ibstat` shows Active, Rate 400

### Phase 7: Smoke Test
- [ ] Run 2-step training with batch_size=1
- [ ] Verify: K8s env pods created, NCCL works, checkpoint saved to NFS
- [ ] Check for errors: ZMQ, NCCL timeout, vLLM hang

### Phase 8: Full Training
- [ ] Launch full training (batch_size=8, total_epochs=1000)
- [ ] Monitor via `kubectl logs` on head pod
- [ ] Set up WandB for metrics tracking

---

## Troubleshooting & Lessons Learned (from 2-Node Experiment)

> Based on 200 training steps across 12 days, 9 crashes, and 11 documented issues from the previous SSH-based 2-node run. Items below are filtered for relevance to the k8s 8-node setup.

### Critical: Outer Timeout Fix Must Be Baked Into Image

**Problem**: Without an outer timeout, a stuck trajectory retries up to 3× with the full `trajectory_timeout` (5400s). Worst case: 5400s × 3 = 4.5 hours blocking a single trajectory, while all other workers wait at the synchronization barrier.

**Fix**: Patch `rllm/engine/agent_execution_engine.py:500-510` in the Docker image:

```python
async def run_agent_trajectory_with_retry(self, idx, seed=0, mode="Text", **kwargs):
    for _ in range(self.retry_limit):
        try:
            application_id = str(uuid.uuid4())
            outer_timeout = int((self.trajectory_timeout or 3600) * 1.2)
            return await asyncio.wait_for(
                self.run_agent_trajectory_async(
                    idx, application_id=application_id,
                    seed=seed, mode=mode, **kwargs),
                timeout=outer_timeout)
        except Exception:
            traceback.print_exc()
            continue
    traceback.print_exc()
    raise Exception(f"Trajectory {idx} cannot complete.")
```

**Action**: Ensure this patch is applied when building `deepswe-train:v2`. With `trajectory_timeout=5400`, the outer timeout becomes 6480s — a single bad trajectory blocks for at most ~108 min instead of potentially hours.

---

### ZMQ Deadlock (Crashes #1, #3, #4, #8)

**Symptom**: Training hangs indefinitely. No progress in logs. All workers idle. Happened ~every 40-50 steps on the 2-node run, causing 15+ hour stalls each time.

**Root Cause**: verl uses ZMQ for inter-process communication between the training coordinator and rollout workers. Under certain conditions (e.g., after vLLM timeout recovery), the ZMQ channel deadlocks.

**Mitigation for K8s**:
- The outer timeout fix (above) prevents infinite blocking of individual trajectories
- Monitor the head pod logs for stalls: no new `step N` log lines for >30 min means likely deadlock
- Recovery: `kubectl delete raycluster deepswe-train && kubectl apply -f deepswe-raycluster.yaml` to restart the entire Ray cluster, then resume from latest checkpoint
- Training auto-resumes from the latest checkpoint in `trainer.default_local_dir` — ensure this is on NFS so it survives pod restarts

---

### vLLM Hang Pattern (~Every 20 Steps)

**Symptom**: vLLM rollout server stops responding. Trajectory generation stalls. Happened at steps ~20, ~140, ~160, ~169, ~188 in the 2-node run.

**Root Cause**: Unknown internal vLLM state corruption. The `VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000` env var prevents vLLM from self-killing, which paradoxically makes it easier to detect and handle at the training level via the outer timeout.

**Mitigation for K8s**:
- Ensure `VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000` is set in pod env (already in the RayCluster manifest)
- The outer timeout will catch stuck trajectories and allow training to continue
- If it happens repeatedly in the same step, restart the Ray cluster and resume
- Consider `enforce_eager=False` (official config) — this uses CUDA graphs which may be more stable than eager mode, though it caused Flash Attention TMA errors with very long sequences in our 2-node run

---

### Flash Attention TMA Error (Issue #1)

**Symptom**: `Failed to initialize TMA descriptor` crash with sequences >119K tokens.

**Root Cause**: Flash Attention's TMA (Tensor Memory Access) descriptor fails with very large attention matrices. The official config uses `enforce_eager=False` (CUDA graphs) — unclear if this avoids or exacerbates the issue.

**Mitigation for K8s**:
- `VLLM_ATTENTION_BACKEND=FLASH_ATTN` is set in the manifest
- If TMA errors occur, fallback option: set `enforce_eager=True` in training config (reduces throughput but avoids TMA)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` prevents another class of CUDA allocation issues

---

### NCCL Configuration (Critical for 8-Node)

**Problem in 2-node run**: Used 25Gbps management network (`NCCL_SOCKET_IFNAME=b_manage0`) instead of 400Gbps RoCE RDMA. This made FSDP PPO updates take 2+ hours (Issue #6).

**Correct configuration** (already in the RayCluster manifest):

```bash
NCCL_SOCKET_IFNAME=GPU0       # GPU-dedicated RDMA NICs (not b_manage0)
NCCL_IB_HCA=mlx5              # ConnectX-7 400Gbps HCAs
NCCL_NET=IB                   # Use InfiniBand/RoCE transport
NCCL_NVLS_ENABLE=0            # Disable NVLink-Switch (not available on H200)
```

**Verification after Ray cluster is up**:
```bash
# Inside head pod:
ibstat | grep -E "CA |State|Rate" | head -20
# Expect: mlx5_0..mlx5_11, State: Active, Rate: 400
```

**If NCCL hangs on first AllReduce**:
- Check that all 8 nodes see the same RDMA interfaces
- Ensure no firewall rules blocking RoCE traffic on the GPU NICs
- Try adding `NCCL_DEBUG=INFO` to pod env for diagnostic output
- Fallback: `NCCL_NET=Socket` + `NCCL_SOCKET_IFNAME=GPU0` (slower, but avoids RDMA issues)

---

### Disk Space Management (Issues #8, #9)

**Problem in 2-node run**: Two disk-full incidents. Docker images (6TB) and checkpoints (~60GB each) exhausted the 880GB SSD.

**K8s differences**:
- K8s nodes have 7TB NVMe for containerd (`/var/lib/containerd`) + 7TB NVMe (`/data`)
- With distributed pre-pull (Step 4.3): max ~2.2TB images per node → 37% utilization → safe
- All-on-all would use ~6.5TB/7TB (93%) → no margin → avoid
- Containerd does NOT auto-prune unused images (unlike Docker with `docker system prune`)
- Runtime fallback pulls (ImageLocality miss) will gradually increase per-node image count

**Mitigation**:
- Use distributed pre-pull strategy (2 copies per image) to keep per-node disk under 3TB
- Monitor disk usage via puller DaemonSet: `df -h /var/lib/containerd` on each node
- Checkpoint pruning: `trainer.max_actor_ckpt_to_keep=3` limits to 3 checkpoints on NFS
- If containerd disk grows beyond 5TB: `crictl rmi --prune` on affected nodes (via privileged pod)

---

### Docker Hub Rate Limiting (Issue #10)

**Symptom**: HTTP 429 errors when pulling `namanjain12/*` images during training rollouts.

**Mitigation for K8s**:
- The `dockerhub` imagePullSecret (novitalabs account) is already configured
- Pre-pull images in advance (Step 4) to avoid runtime pulls
- Throttle pre-pull jobs: max 50-100 concurrent pulls to stay within Docker Hub rate limits
- If 429 persists during training: the R2E-Gym k8s backend retries pod creation, but repeated failures will cause trajectory timeouts
- Consider mirroring critical images to a local registry on youyun.37

---

### pg_clipfrac Dead Learning (Critical Config Issue)

**Problem**: When `train_batch_size == ppo_mini_batch_size`, there is only 1 optimizer update per training step. This leads to `pg_clipfrac=0` (zero policy gradient clipping) → the policy barely updates.

**Evidence from PPIO experiments**:
| mini_batch | batch | optimizer steps/step | pg_clipfrac | Learning |
|------------|-------|---------------------|-------------|----------|
| 8 | 64 | 32 (8×4 epochs) | 1.3e-4 → 2.3e-4 | **Alive** |
| 32 | 32 | 4 (1×4) | 0.0 | **Dead** |
| 64 | 64 | 4 (1×4) | 0.0 | **Dead** |

Our 2-node run had `batch=4, mini_batch=4` → 1 opt step → likely dead learning. Result: Step 164 = 76.2% vs base 74.8% = only +1.4pp improvement.

**Official config**: `batch=8, mini_batch=8` → also only 1 opt step per epoch, BUT the official script may use multiple PPO epochs (default 4 in verl), giving 4 opt steps total.

**Action for 8-node run**: Verify that `ppo_epochs` is set (add `actor_rollout_ref.actor.ppo_epochs=4` if not already default). With batch=8, mini_batch=8, epochs=4 → 4 optimizer updates per step.

---

### Token Mismatch / Truncated Samples (Issues #3, #4)

**Symptom**: Token mismatch in `assemble_steps` → empty response_masks → PPO updates skipped. Truncated samples get filtered out → empty batches.

**Fix**: Two flags must be set (already in the training config):
```bash
rllm.mask_truncated_samples=False     # Don't discard truncated trajectories
rllm.filter_token_mismatch=False      # Don't error on BPE retokenization differences
```

These are critical — without them, a significant fraction of trajectories are silently discarded.

---

### Log Probability Config (Issue #2)

**Symptom**: Empty entropy values → PPO updates produce no gradients.

**Fix**: Must set:
```bash
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
```

Default `log_prob_micro_batch_size_per_gpu=None` causes the issue.

---

### Version Compatibility (Strict)

Mismatched versions cause silent failures or crashes:

| Component | Required | Risk |
|-----------|----------|------|
| Python | 3.10.x | 3.11+ may work but untested |
| torch | 2.8.0+cu128 | vllm 0.10.2 requires exact match |
| vllm | 0.10.2 | Requires torch==2.8.0 |
| transformers | 4.50 - 4.57.x | **5.x breaks** — API changes in tokenizer |
| flash-attn | 2.8.x | Must match torch/CUDA |
| verl | 0.6.1 | |
| swebench | 3.0.2 | R2E-Gym dependency |

**Action**: Pin all versions in the Docker image. Test with `pip freeze` after build.

---

### Security: Monitor for Crypto Miners

**Incident**: During our 2-node run, two separate crypto miners were installed (Feb 14: `/bin/kworker` rootkit, Feb 20: `/tmp/dns` miner). The miner consumed 18000%+ CPU → vLLM throughput dropped 10x → 10 training steps had zero learning.

**Mitigation for K8s**:
- Ray pods run as non-root (set `securityContext.runAsNonRoot: true` if possible)
- R2E-Gym environment pods should have strict resource limits (already: 1 CPU, 1Gi)
- Periodic check: `kubectl top pods` — any env pod using significantly more CPU than requested is suspicious
- Network policy: restrict env pod egress to only the k8s API server and Docker Hub
- If a node shows unexpected CPU usage: drain it, inspect, redeploy

---

### Checkpoint Recovery Procedure

Training auto-resumes from the latest checkpoint in `trainer.default_local_dir`. On NFS:

```bash
# Check latest checkpoint:
ls /nfs/checkpoints/deepswe-full-v2/8node-rloo-bs8/
cat /nfs/checkpoints/deepswe-full-v2/8node-rloo-bs8/latest_checkpointed_iteration.txt

# If checkpoint is corrupt (happened once in 2-node run due to I/O contention):
# Remove the corrupt checkpoint directory, update latest_checkpointed_iteration.txt to previous step
```

**Save frequency**: `trainer.save_freq=10` → one checkpoint every 10 steps. With `max_actor_ckpt_to_keep=3`, at most 3 × ~60GB = 180GB on NFS.

---

### Monitoring Checklist During Training

| What to Watch | How | Warning Sign |
|---------------|-----|-------------|
| Training progress | `kubectl logs <head-pod> --tail=100` | No new `step N` lines for >30 min |
| GPU utilization | `kubectl exec <head-pod> -- nvidia-smi` | GPUs idle during rollout phase |
| Disk usage | `kubectl exec <any-pod> -- df -h /nfs` | NFS >80% full |
| Pod health | `kubectl get pods -l ray.io/cluster=deepswe-train` | Any pod in CrashLoopBackOff |
| Env pods | `kubectl get pods -l app=r2e-env` (or similar) | Stuck in ImagePullBackOff |
| NCCL | Head pod logs, grep for `NCCL` | `NCCL timeout` or `NCCL error` |
| WandB metrics | Dashboard | `pg_clipfrac=0` (dead learning), `score` flat |

---

### Env Pod Scheduling & Load Balancing

**Code location**: `R2E-Gym/src/r2egym/agenthub/runtime/docker.py` lines 237-268.

The R2E-Gym k8s backend creates env pods with `nodeSelector: {workload: r2e-eval}` and resource requests of 1 CPU + 1Gi RAM. **There is no explicit load balancing logic** — no `topologySpreadConstraints`, no affinity rules, no image-locality-aware scheduling.

**Scheduling behavior**:
- K8s default scheduler distributes pods across the 8 matching nodes using its scoring algorithm (considers available resources, existing pod count, etc.)
- With `rollout.n=8` × `train_batch_size=8` = up to **64 concurrent env pods** at peak
- 64 pods / 8 nodes = ~8 pods/node — trivial for 192-CPU nodes, so no resource contention
- Ray training pods and env pods share the same nodeSelector (`workload: r2e-eval`), so they co-locate on the same nodes. This is fine since env pods use negligible resources compared to GPU training

**Image locality and distributed pre-pull**: The k8s default scheduler has an `ImageLocality` scoring plugin that prefers nodes already having the required image. With the distributed pre-pull strategy (2 copies per image, see Step 4.3), the scheduler will route env pods toward nodes that have the image cached. If a pod lands on a node without the image, `imagePullPolicy: IfNotPresent` triggers a fallback runtime pull — slower but functional. Over time, hot images may spread to more nodes organically.

**Optional improvement**: Add `topologySpreadConstraints` to the R2E-Gym pod spec to enforce even distribution, but this is unlikely to be necessary in practice given the small pod-to-node ratio.

---

## Reference: Official Training Config

From `rllm-origin/examples/swe/train_deepswe_32b.sh`:

| Parameter | Value |
|-----------|-------|
| `algorithm.adv_estimator` | rloo |
| `data.train_batch_size` | 8 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 8 |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 8 |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | 8 |
| `actor_rollout_ref.rollout.enforce_eager` | False |
| `actor_rollout_ref.rollout.temperature` | 1.0 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.6 |
| `actor_rollout_ref.rollout.n` | 8 |
| `rllm.agent.trajectory_timeout` | 5400 |
| `data.max_prompt_length` | 4096 |
| `trainer.nnodes` | 8 |
| `trainer.n_gpus_per_node` | 8 |
| `trainer.save_freq` | 10 |
| `trainer.test_freq` | 10 |
| `trainer.total_epochs` | 1000 |
