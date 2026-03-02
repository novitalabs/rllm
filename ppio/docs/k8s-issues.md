# DeepSWE K8s Training - Issues & Solutions

## Environment

- **Cluster**: 4 nodes × 8 H200 GPUs (32 GPUs total), K8s StatefulSet
  - node-0 (head): 10.83.115.18 → pod deepswe-training-2
  - node-1: 10.83.115.21 → pod deepswe-training-1
  - node-2: 10.83.115.22 → pod deepswe-training-3
  - node-3: 10.83.115.23 → pod deepswe-training-0 (Ray head)
- **K8s**: kubeadm 1.32, Flannel CNI, containerd
- **Containers**: Docker-in-Docker (DinD) inside privileged pods
- **Image**: `rllm-deepswe:latest` based on `vllm/vllm-openai:v0.10.2`
- **Networking**: `hostNetwork: true` (required for RDMA)
- **Storage**: PVC for `/var/lib/docker` (200Gi local PV per node), hostPath for model weights/data/checkpoints

---

## Issue 1: verl `extra_info` JSON String Deserialization Failure

**Symptom**: `AttributeError: 'str' object has no attribute 'get'` in verl's `RLHFDataset.__getitem__`.

**Root Cause**: verl's `rl_dataset.py` (line ~442) calls `.get()` on `extra_info` expecting a dict:
```python
index = row_dict.get("extra_info", {}).get("index", 0)
```
But the DeepSWE parquet files store `extra_info` as a JSON-encoded string, not a dict. The `None` check passes because the value is a non-empty string, so it flows into `.get()` on a `str` object.

**Solution**: Patch verl's `rl_dataset.py` at container startup to deserialize JSON strings. Added to `k8s/scripts/entrypoint.sh`:

```python
# Adds: import json
# Adds after the None check:
elif isinstance(row_dict["extra_info"], str):
    row_dict["extra_info"] = json.loads(row_dict["extra_info"])
```

The patch is idempotent — it checks if it has already been applied before modifying the file.

**Files**: `k8s/scripts/entrypoint.sh` (lines 47-67)

---

## Issue 2: Ray Workers Bind to 127.0.0.1 with hostNetwork

**Symptom**: After starting all 4 pods, `ray status` shows only 1 node (8/32 GPUs). Worker Ray logs show `Local node IP: 127.0.0.1`.

**Root Cause**: With `hostNetwork: true`, the pod's hostname is the node's hostname (e.g., `host-10-83-115-21`), not a DNS-resolvable pod name. Python's `socket.gethostbyname()` fails for these hostnames, so Ray falls back to `127.0.0.1` as the node IP. Workers register with the head using `127.0.0.1`, making them unreachable.

**Solution**: Explicitly detect the node IP from the management NIC and pass it to Ray via `--node-ip-address`:

```bash
NODE_IP=$(ip -4 addr show b_manage0 2>/dev/null | grep -oP 'inet \K[0-9.]+' || hostname -I | awk '{print $1}')
ray start --head --node-ip-address=${NODE_IP} ...
```

**Files**: `k8s/scripts/entrypoint.sh` (lines 111-113, 118)

---

## Issue 3: Ray Port Conflict — runtime_env_agent vs worker_ports

**Symptom**: Worker pod fails to start Ray:
```
ValueError: Ray component worker_ports is trying to use a port number 37715 that is used by other components
```

**Root Cause**: With `hostNetwork: true`, Ray's internal components (metrics exporter, dashboard agent, runtime env agent) randomly select ports that may collide with the worker port range. The default worker port range is 10002-19999, but was already configured to 30000-39999 to avoid conflicts. However, `runtime_env_agent` randomly picked port 37715, which falls within the custom worker port range.

**Solution**: Pin all Ray internal component ports to fixed values outside the worker port range:

```bash
RAY_COMMON_ARGS="--node-ip-address=${NODE_IP} --num-gpus=${NUM_GPUS} \
    --min-worker-port=30000 --max-worker-port=39999 \
    --metrics-export-port=20100 \
    --runtime-env-agent-port=20400 \
    --dashboard-agent-grpc-port=20200 \
    --dashboard-agent-listen-port=20300"
```

**Lesson**: When using `hostNetwork`, pin ALL Ray ports to fixed values. Random port selection will eventually collide across components.

**Files**: `k8s/scripts/entrypoint.sh` (lines 118-123)

---

## Issue 4: GCS Mismatch from Rolling Update (24/32 GPUs)

**Symptom**: After `kubectl rollout restart`, only 3/4 Ray nodes visible (24 GPUs instead of 32). One worker shows:
```
Failed to connect to GCS within 60 seconds
```

**Root Cause**: `kubectl rollout restart` on a StatefulSet with `OrderedReady` policy restarts pods in **reverse order** (3 → 2 → 1 → 0). So worker-3 restarts first, connects to the **old head's GCS**, then when the head (pod-0) restarts with a **new GCS ID**, worker-3's GCS connection becomes stale and the raylet eventually dies.

The restart sequence:
1. Pod-3 (worker) restarts → connects to old head's GCS ✓
2. Pod-2 (worker) restarts → connects to old head's GCS ✓
3. Pod-1 (worker) restarts → connects to old head's GCS ✓
4. Pod-0 (head) restarts → **new GCS starts**, old GCS dies
5. Pod-3's raylet loses GCS connection → `Failed to connect to GCS` → dies
6. Only pods 0, 1, 2 remain = 24 GPUs

**Solution**: Delete all pods simultaneously instead of using rolling restart:

```bash
kubectl delete pods -l app=deepswe-training -n deepswe
```

This ensures all pods start fresh and connect to the same new GCS. With `OrderedReady` policy, pod-0 (head) starts first, then workers follow.

**Lesson**: Never use `kubectl rollout restart` for Ray StatefulSets. Always delete all pods simultaneously.

---

## Issue 5: DinD Docker Cannot Pull Images — Proxy Not Configured

**Symptom**: Training reaches SWE agent execution (step 0) but all trajectory container creations fail:
```
Error response from daemon: Get "https://registry-1.docker.io/v2/": context deadline exceeded
```

**Root Cause**: The DinD dockerd inside each pod had no proxy configuration. The cluster nodes require an HTTP proxy (`http://127.0.0.1:1083`) to reach external registries like DockerHub. With `hostNetwork: true`, the host's loopback proxy is accessible from within the container, but `dockerd` doesn't inherit shell proxy environment variables — it uses its own daemon-level config or process environment.

**Solution**: Set proxy environment variables before starting `dockerd` in the entrypoint:

```bash
export HTTP_PROXY=http://127.0.0.1:1083
export HTTPS_PROXY=http://127.0.0.1:1083
export NO_PROXY=localhost,127.0.0.1,10.0.0.0/8
dockerd --storage-driver=overlay2 --host=unix:///var/run/docker.sock &
```

Since `dockerd` is a child process of the entrypoint shell, it inherits the environment variables. This approach is simpler than creating `/etc/docker/daemon.json` or `/etc/systemd/system/docker.service.d/proxy.conf`.

**Files**: `k8s/scripts/entrypoint.sh` (lines 72-76)

---

## Issue 6: Gloo Backend Fails on Wrong Network Interface

**Symptom**: FSDP CPU-backend operations (param_offload/optimizer_offload) hang or fail with connection errors during `update_actor`.

**Root Cause**: PyTorch's Gloo backend (used for FSDP CPU communication) defaults to the first network interface, which with `hostNetwork` may be loopback or an RDMA NIC that doesn't support TCP. The management NIC `b_manage0` is the correct interface for TCP-based Gloo communication.

**Solution**: Set `GLOO_SOCKET_IFNAME` to the management NIC:

```bash
export GLOO_SOCKET_IFNAME=b_manage0
```

**Files**: `k8s/scripts/entrypoint.sh` (line 27)

---

## Issue 7: NCCL NVLS Transport Warning (Cuda failure 1 'invalid argument')

**Symptom**: Repeated NCCL warnings across all workers:
```
transport/nvls.cc:621 NCCL WARN Cuda failure 1 'invalid argument'
```

**Root Cause**: NCCL's NVLS (NVLink SHARP) transport attempts to use CUDA Unified Memory (cumem) allocator for cross-GPU communication. On H200 nodes in this cluster configuration, the NVLS transport encounters an incompatibility, likely due to the CUDA 12.8 + driver 580 combination or the specific NVLink topology.

**Solution**: Disable NVLS transport via:

```bash
export NCCL_CUMEM_ENABLE=0
```

This forces NCCL to use standard NVLink/P2P/IB transports instead. The warning is cosmetic (NCCL falls back automatically), but disabling it avoids log noise.

**Files**: `k8s/scripts/entrypoint.sh` (line 18)

---

## Issue 8: Pod Hostname Not Matching Pod Name (hostNetwork)

**Symptom**: Pod ordinal detection fails. With `hostNetwork: true`, `hostname` returns the node's hostname (e.g., `host-10-83-115-23`), not the StatefulSet pod name (e.g., `deepswe-training-0`). The entrypoint script can't determine which pod is the Ray head.

**Root Cause**: `hostNetwork: true` shares the host's network namespace, including its hostname. The StatefulSet pod name is not reflected in the container's hostname.

**Solution**: Use the Kubernetes Downward API to inject the pod name as an environment variable, then extract the ordinal:

```yaml
# In statefulset.yaml:
env:
  - name: POD_NAME
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
```

```bash
# In entrypoint.sh:
MY_POD_NAME="${POD_NAME:-$(hostname)}"
ORDINAL="${MY_POD_NAME##*-}"
```

The `${POD_NAME##*-}` bash parameter expansion strips everything up to the last `-`, giving the ordinal (0, 1, 2, 3).

**Files**: `k8s/statefulset.yaml` (lines 82-85), `k8s/scripts/entrypoint.sh` (lines 37-38)

---

## Issue 9: 8-Node Bare-Metal Run Crash — ActorUnavailableError (Run 35)

**Symptom**: Training run on 8 nodes (64 GPUs) crashed at step 0 (61/64 trajectories completed) with:
```
ray.exceptions.ActorUnavailableError: The actor ... is unavailable:
  RpcError: RPC error: Socket closed rpc_code: 14
```
Followed by TCPStore failures across all workers:
```
[c10d] recvValue failed on SocketImpl: Failed to recv, got 0 bytes.
  Connection was likely closed. Did the remote server shutdown or crash?
```

**Root Cause**: One of the Ray actors (likely the TaskRunner on the head node) lost its RPC connection. This cascaded:
1. The head node's Raylet terminated unexpectedly
2. All FSDP workers lost their TCPStore connection (hosted on head node)
3. NCCL heartbeat monitors detected the TCPStore failure
4. All workers reported ProcessGroupNCCL failures

The Raylet's last logs showed `memory_monitor.cc:197: Got negative used memory for cgroup -1` repeated every 5 seconds, suggesting cgroup memory monitoring was malfunctioning. This is likely a symptom, not the cause.

**Possible Root Causes**:
1. **OOM on head node**: The head runs TaskRunner (Docker containers) + Ray GCS + vLLM server + FSDP worker. With 8 nodes worth of trajectory Docker containers potentially scheduled on the head, memory pressure could kill a critical process.
2. **Network instability**: The 8-node setup uses inter-node RDMA. A transient network partition or NIC reset could disconnect the Ray RPC layer.
3. **Raylet bug**: The negative cgroup memory readings suggest Raylet's memory monitor was confused, possibly leading to incorrect OOM kill decisions.

**Status**: Not investigated further. The 8-node bare-metal setup was replaced by the 4-node K8s deployment.

**Files**: Log at `/tmp/train_run35.log`

---

## Issue 10: K8s API Server Unreachable After Training Run

**Symptom**: `kubectl` commands return `net/http: TLS handshake timeout` or `context deadline exceeded` after training has been running for extended periods.

**Root Cause**: The kube-apiserver container is running (`crictl ps` shows it alive) and listening on port 6443, but not accepting new connections. Possible causes:
1. **Proxy interference**: The training pods set `HTTP_PROXY`/`HTTPS_PROXY` environment variables. If these leaked to the kubelet or node-level processes, API server connections could be routed through the proxy incorrectly.
2. **etcd pressure**: With 32 DinD Docker daemons creating/destroying containers and the Flannel CNI managing overlay networking, etcd may be under heavy write pressure.
3. **Connection exhaustion**: With `hostNetwork: true`, all pods share the host's network stack. 64+ concurrent agent trajectories making outbound connections + Ray inter-node traffic + Docker registry pulls could exhaust connection table limits.

**Workaround**: Access pod containers directly via `crictl` (bypasses kube-apiserver):
```bash
# List pods
crictl pods --name deepswe
# Get container ID
CONTAINER_ID=$(crictl ps --name training -q | head -1)
# View logs
crictl logs $CONTAINER_ID
# Execute commands
crictl exec $CONTAINER_ID <command>
```

For cross-node access, SSH to the node and use `crictl` locally:
```bash
ssh root@10.83.115.23 "crictl logs $(crictl ps --name training -q | head -1)"
```

**Status**: Workaround in place. API server recovery may require restarting the kube-apiserver pod or the entire control plane.

---

## Issue 11: Docker Hub Rate Limiting Kills >50% Trajectories (batch_size=32)

**Symptom**: Steps 7-9 of the batch_size=32 run had 137/150/117 failed trajectories (out of 256), causing `traj/steps_mean` to drop from ~21 to ~9 and `response_length/mean` from ~17K to ~7K. Docker logs show:
```
429 Client Error: Too Many Requests for url: http+docker://localhost/v1.53/images/create?tag=...&fromImage=namanjain12%2Ftornado_final
500 Server Error: Internal Server Error for url: http+docker://localhost/v1.53/images/create?...
```

**Root Cause**: Docker Hub's unauthenticated rate limit is **100 pulls per 6 hours per IP**. With `train_batch_size=32` (256 trajectories/step), each step potentially pulls hundreds of distinct r2e-gym Docker images (one per SWE-bench task, identified by git commit hash). After ~5 steps (~768 cumulative image pulls), the quota was exhausted.

Error chain:
```
Docker Hub → 429 Too Many Requests (on /images/create)
→ DockerRuntime.__init__: container = None
→ AttributeError: 'NoneType' object has no attribute 'id'
→ run_agent_trajectory_with_retry: 3 retries all fail (same image, same 429)
→ launch_one_trajectory_task: returns dummy result (reward=0, response=[pad_id])
```

**Per-step Docker errors** (training data collection):

| Step | 429 Errors | 500 Errors | Dummy (failed) |
|------|-----------|-----------|----------------|
| 3 | 0 | 0 | 0 |
| 4 | 0 | 5 | 1 |
| 5 | 0 | 5 | 0 |
| 6 | **34** | 20 | 5 |
| 7 | **346** | 51 | **137** |
| 8 | **402** | 49 | **150** |
| 9 | **329** | 44 | **117** |
| 10 | 0 | 5 | 0 |
| 11 | 0 | 5 | 0 |

**Why steps 7-9**: The 6-hour rate limit window starts when the first pulls occur (~21:00). By step 6 (~03:33, +6.5h), the quota is exhausted. Steps 7-9 (~04:50-07:20) are deep in the rate limit. Step 10 validation (~08:47-10:24, ~100min) provides cooldown. Step 11 (10:24+) fully recovered.

**Impact**: 404 trajectories returned dummy results across 3 steps. Steps 7-9 contributed minimal effective gradient. ~3 wasted training steps (~4h, 128 GPU-hours).

**Fix**: Deploy a local Docker registry mirror (pull-through cache) on the cluster to cache r2e-gym images. See `k8s/registry/` for deployment files.

**Note**: This issue does NOT affect `batch_size=8` (64 trajectories/step) because the pull rate stays within Docker Hub's limit.

---

## Deployment Checklist

Based on all issues encountered, here's the checklist for deploying DeepSWE training on K8s:

### Pod Configuration
- [x] `hostNetwork: true` + `dnsPolicy: ClusterFirstWithHostNet` (RDMA)
- [x] `privileged: true` (DinD)
- [x] Pod name via Downward API (`POD_NAME` env var) — Issue 8
- [x] `podAntiAffinity` (one pod per node)

### Ray Cluster
- [x] Detect node IP from NIC, not hostname — Issue 2
- [x] Pin ALL Ray ports (`--metrics-export-port`, `--runtime-env-agent-port`, `--dashboard-agent-grpc-port`, `--dashboard-agent-listen-port`) — Issue 3
- [x] Never use `kubectl rollout restart`; delete all pods — Issue 4

### Networking
- [x] `NCCL_SOCKET_IFNAME=b_manage0` (NCCL)
- [x] `GLOO_SOCKET_IFNAME=b_manage0` (FSDP Gloo backend) — Issue 6
- [x] `NCCL_CUMEM_ENABLE=0` (disable NVLS transport) — Issue 7
- [x] `HTTP_PROXY`/`HTTPS_PROXY` before `dockerd` start — Issue 5

### Software Patches
- [x] verl `extra_info` JSON deserialization patch — Issue 1
- [x] verl `init_app_state` 4-arg fix for vLLM 0.10.2
- [x] verl `enable_sleep_mode` fix
- [x] vLLM `_loop_forever` no-break fix
