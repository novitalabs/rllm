# 迁移实验环境：4节点(.18/.21/.22/.23) → 2节点(.14/.17)

## 现状

### 当前集群 (4×8 H200 = 32 GPUs)
- 节点: 10.83.115.18(head), .21, .22, .23
- 训练进度: step 15, save_freq=10, 已保存 global_step_1/2/10
- 每个 checkpoint ~92G/节点 (FSDP world_size=32), 共 275G/节点
- Docker image: `rllm-deepswe:latest` (基于 vllm/vllm-openai:v0.10.2)
- Registry mirror: 10.83.115.18:5000 (刚部署，缓存为空)
- 代码: `/root/develop/ref/rllm/` (本机 + .18 同步)

### 新节点 (2×8 H200 = 16 GPUs)
| 项目 | 10.83.115.14 | 10.83.115.17 |
|------|-------------|-------------|
| GPU | 8× H200 ✅ | 8× H200 ✅ |
| /data 可用 | 1.3T (83%) | 1.7T (77%) |
| Docker | 28.5.2 ✅ | 28.5.2 ✅ |
| kubectl/kubeadm | v1.32.10 ✅ | v1.32.10 ✅ |
| 模型权重 | Qwen3-32B ✅ | Qwen3-32B ✅ |
| Proxy | ✅ | ✅ |
| RDMA/nvidia_peermem | ✅ | ✅ |
| b_manage0 | ✅ | ✅ |
| Ray | ❌ (容器内) | ❌ (容器内) |
| inotify | 128 (需调高) | 128 (需调高) |

---

## TODO 清单

### 1. 系统配置 (两节点都执行)

```bash
# 在 .14 和 .17 上执行:
sysctl -w fs.inotify.max_user_instances=1024
echo "fs.inotify.max_user_instances=1024" >> /etc/sysctl.conf
```

### 2. 同步代码到 .14

```bash
# 从本机同步 rllm 代码到 .14
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    /root/develop/ref/rllm/ 10.83.115.14:/root/develop/ref/rllm/

# 或者直接在 .14 上 git clone
ssh 10.83.115.14 "mkdir -p /root/develop/ref && cd /root/develop/ref && \
    git clone <repo_url> rllm && cd rllm && git checkout deepswe_ref"
```

### 3. K8s 集群搭建 (.14=head, .17=worker)

#### 3.1 清理旧 K8s 状态 (如有)

```bash
# 两节点都执行
ssh 10.83.115.14 "kubeadm reset -f; rm -rf /etc/cni /opt/cni /var/lib/etcd"
ssh 10.83.115.17 "kubeadm reset -f; rm -rf /etc/cni /opt/cni /var/lib/etcd"
```

#### 3.2 初始化 K8s head (.14)

```bash
ssh 10.83.115.14 "kubeadm init \
    --apiserver-advertise-address=10.83.115.14 \
    --pod-network-cidr=10.244.0.0/16 \
    --service-cidr=10.96.0.0/12"

# 配置 kubectl
ssh 10.83.115.14 "mkdir -p ~/.kube && cp /etc/kubernetes/admin.conf ~/.kube/config"
```

#### 3.3 安装 Flannel CNI

```bash
ssh 10.83.115.14 "kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml"
```

#### 3.4 Worker 加入 (.17)

```bash
# 用 kubeadm init 输出的 join 命令
ssh 10.83.115.17 "kubeadm join 10.83.115.14:6443 --token <token> --discovery-token-ca-cert-hash <hash>"
```

#### 3.5 安装 NVIDIA device plugin

```bash
ssh 10.83.115.14 "kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml"
```

#### 3.6 Taint 移除 (允许 head 运行 pod)

```bash
ssh 10.83.115.14 "kubectl taint nodes --all node-role.kubernetes.io/control-plane-"
```

### 4. 部署 Docker Registry Mirror (.14)

```bash
# 在 .14 上部署 registry:2 pull-through cache
ssh 10.83.115.14 "mkdir -p /data/registry-cache/data"
scp k8s/registry/config.yml 10.83.115.14:/data/registry-cache/config.yml
scp k8s/registry/docker-registry-mirror.service 10.83.115.14:/etc/systemd/system/

# 拉取 registry 镜像并启动
ssh 10.83.115.14 "
    export HTTP_PROXY=http://127.0.0.1:1083 HTTPS_PROXY=http://127.0.0.1:1083
    docker pull registry:2
    systemctl daemon-reload
    systemctl enable docker-registry-mirror
    systemctl start docker-registry-mirror
"

# 验证
curl http://10.83.115.14:5000/v2/_catalog
```

### 5. 构建/传输 Docker Image

```bash
# 方案A: 从旧节点传输 (推荐，避免重新构建)
ssh 10.83.115.18 "docker save rllm-deepswe:latest | gzip" | \
    ssh 10.83.115.14 "gunzip | docker load"

# 方案B: 在 .14 上重新构建
ssh 10.83.115.14 "cd /root/develop/ref/rllm && \
    export HTTP_PROXY=http://127.0.0.1:1083 HTTPS_PROXY=http://127.0.0.1:1083 && \
    docker build -f k8s/Dockerfile.k8s -t rllm-deepswe:latest ."

# .17 也需要镜像 (从 .14 传)
ssh 10.83.115.14 "docker save rllm-deepswe:latest | gzip" | \
    ssh 10.83.115.17 "gunzip | docker load"
```

### 6. 修改 K8s 配置 (2节点版)

需要修改的文件及关键变更:

#### 6.1 `k8s/statefulset.yaml`

```yaml
# replicas: 4 → 2
spec:
  replicas: 2

# REGISTRY_MIRROR 改为 .14
- name: REGISTRY_MIRROR
  value: "http://10.83.115.14:5000"
```

#### 6.2 `k8s/pv-nodes.yaml`

更新 PV 的 nodeAffinity 指向 .14 和 .17:
- pv-docker-0, pv-rllm-0, pv-model-0, pv-ckpt-0 → host-10-83-115-14
- pv-docker-1, pv-rllm-1, pv-model-1, pv-ckpt-1 → host-10-83-115-17
- 删除 ordinal 2 和 3 的 PV

#### 6.3 `k8s/scripts/train_deepswe_32b_k8s.sh`

```bash
# 关键变更:
trainer.nnodes=2                 # 4 → 2
data.train_batch_size=16         # 32 → 16 (保持每GPU相同)
# 或者保持 32 (每GPU更大batch)

# checkpoint 路径保持一致
trainer.default_hdfs_dir=null
```

#### 6.4 `k8s/scripts/entrypoint.sh`

已支持 `REGISTRY_MIRROR` 环境变量，无需修改。

### 7. 转移 Checkpoint (可选: 等 step 20 保存后)

**注意**: 当前 FSDP checkpoint 是 world_size=32 分片，新集群是 world_size=16。
需要先转换或使用 verl 的 checkpoint resharding 功能。

```bash
# 方案A: 转移 huggingface/ 格式 (已合并的权重，无需 resharding)
ssh 10.83.115.18 "ls /data/checkpoints/deepswe/deepscaler-agent/swe-agent-rl/global_step_10/huggingface/"

# 如果 huggingface/ 目录有完整权重:
rsync -avP 10.83.115.18:/data/checkpoints/deepswe/deepscaler-agent/swe-agent-rl/global_step_10/huggingface/ \
    10.83.115.14:/data/models/DeepSWE-Step10/

# 然后训练脚本中将 model.path 指向该 checkpoint:
# actor_rollout_ref.model.path=/data/models/DeepSWE-Step10
```

```bash
# 方案B: 直接从 Qwen3-32B 基础模型重新开始训练
# 不需要转移 checkpoint，两节点已有 Qwen3-32B
```

### 8. 创建 K8s 资源并启动

```bash
# 在 .14 上执行
ssh 10.83.115.14 "
    cd /root/develop/ref/rllm
    kubectl create namespace deepswe
    kubectl apply -f k8s/storageclass.yaml
    kubectl apply -f k8s/pv-nodes.yaml     # 修改后的2节点版
    kubectl apply -f k8s/statefulset.yaml   # 修改后的2节点版
"

# 检查 pod 启动
ssh 10.83.115.14 "kubectl -n deepswe get pods -o wide -w"
```

### 9. 验证训练启动

```bash
# 查看 head pod 日志
ssh 10.83.115.14 "kubectl -n deepswe logs deepswe-training-0 -f"

# 检查 Ray 集群
ssh 10.83.115.14 "kubectl -n deepswe exec deepswe-training-0 -- ray status"

# 检查 GPU 使用
ssh 10.83.115.14 "nvidia-smi"
ssh 10.83.115.17 "nvidia-smi"
```

---

## 配置差异对比 (4节点 vs 2节点)

| 参数 | 4节点 | 2节点 | 说明 |
|------|-------|-------|------|
| `trainer.nnodes` | 4 | 2 | |
| GPUs | 32 | 16 | |
| `data.train_batch_size` | 32 | 16 | 保持每GPU batch一致 |
| 每步 trajectories | 256 | 128 | batch_size × rollout.n(8) |
| `replicas` (StatefulSet) | 4 | 2 | |
| FSDP world_size | 32 | 16 | |
| vLLM replicas | 4 (TP=8 each) | 2 (TP=8 each) | |
| `REGISTRY_MIRROR` | http://10.83.115.18:5000 | http://10.83.115.14:5000 | |
| 预估步时间 | ~75min | ~75min | 推理瓶颈不变(并发减半,但vLLM也减半) |
| 预估 Docker 容器 | 256/step | 128/step | 更不容易触发 rate limit |

## 注意事项

1. **Checkpoint 兼容性**: world_size=32 的 FSDP checkpoint 不能直接用于 world_size=16。需要用 huggingface 格式的合并权重，或转换分片。
2. **/data 空间**: .14 只剩 1.3T，每个 checkpoint ~92G(16GPU) ≈ 46G/节点。可保存 ~20 个 checkpoint。如空间不足，清理旧的 deepswe-full/deepswe-v2。
3. **Docker Hub rate limit**: 128 并发容器 (vs 之前 256)，且有 registry mirror，基本不会触发 429。
4. **训练效率**: 2节点 16GPU，每步 128 trajectories。推理瓶颈不变(vLLM 半, 并发也半)，但 update_actor 快一倍(数据量半)。总步时间预计类似。
5. **模型权重**: 两节点已有 Qwen3-32B，无需传输。

---

## 迁移后实际状态 (2026-03-04)

### 已完成
- [x] K8s集群搭建 (.14=head, .17=worker), 两节点 Ready
- [x] Docker image 已部署到两节点
- [x] 从4节点 step 30 checkpoint 成功续训到 step 69
- [x] **Registry 本地化完成**: pull-through cache 转为 plain registry, 4619/4622 tags
  - 完全消除 Docker Hub 依赖，零 429 风险
  - 脚本: `k8s/scripts/push-to-registry-fast.sh`, `k8s/scripts/pull-missing-to-registry.sh`
- [x] 两节点 host Docker images 已清理 (释放 ~5T)

### 当前问题
- **训练在 step 70 崩溃**: .14 磁盘写满 (367G free / 7T)
  - Registry 2.8T + DinD cache 2.0T + Checkpoints 1.1T = 磁盘占满
  - 需清理旧 checkpoints (~1.1T 可回收)
- **策略更新不明显**: pg_clipfrac≈0, KL≈0, 学习率 1e-6 可能过低
- **验证得分**: 18.6% → 15.6% (step 40→60)

### 实际性能对比

| 指标 | 预估 | 实际 |
|------|------|------|
| 步时间 | ~75min | 30.3min (普通步), ~85min (验证步) |
| Checkpoint 大小 | ~46G/节点 | 184G/节点 (FSDP全量) |
| Registry mirror | pull-through cache | **plain registry** (4619 tags, 2.8T) |
| Docker Hub 429 | 可能偶发 | **完全消除** |

详细训练分析: [`experiment-analysis-2node-step69.md`](experiment-analysis-2node-step69.md)
