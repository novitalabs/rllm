# 32x H200 多节点训练指南 (4 节点 x 8 GPU)

## 集群信息

| 节点 | IP | 角色 | GPU |
|------|-----|------|-----|
| Node 0 | 111.6.123.35 | Ray Head (训练入口) | 8x H200 |
| Node 1 | 111.6.123.36 | Ray Worker | 8x H200 |
| Node 2 | 111.6.123.37 | Ray Worker | 8x H200 |
| Node 3 | 111.6.123.38 | Ray Worker | 8x H200 |

**总计**: 32x H200 GPU, ~4.5TB VRAM

## 前提条件

- 4 台节点之间的 SSH 免密登录
- 所有节点安装 Docker + NVIDIA Container Runtime
- 共享文件系统 (NFS/3FS) 挂载在 `/models/` 和 `/3fsdata/`
- PPIO API Key
- Docker 镜像: `vllm/vllm-openai:v0.10.2`

---

## 方法 A: 一键部署 (推荐)

从任意可 SSH 到所有节点的机器执行:

```bash
cd /path/to/rllm

# 设置环境变量
export PPIO_API_KEY=sk_xxxxx
export RLLM_HOST_DIR=/workspace/rllm     # rllm 在节点上的路径
export MODEL_HOST_DIR=/models             # 模型权重路径
export DATA_HOST_DIR=/3fsdata             # 共享数据路径

# 一键启动集群
bash ppio/scripts/setup_32h200_cluster.sh setup
```

启动完成后, SSH 到 head 节点运行训练:

```bash
ssh 111.6.123.35
docker exec -it deepswe-train bash
cd /workspace/rllm
export PPIO_API_KEY=sk_xxxxx
bash ppio/scripts/train_qwen3_32b_32h200.sh
```

---

## 方法 B: 手动部署 (逐步)

### 步骤 1: 在所有节点上启动 Docker 容器

在**每个节点**上执行:

```bash
docker run -d \
    --name deepswe-train \
    --runtime=nvidia \
    --gpus all \
    --net=host \
    --ipc=host \
    --shm-size=64g \
    --cap-add=SYS_ADMIN \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /workspace/rllm:/workspace/rllm \
    -v /models:/models \
    -v /3fsdata:/3fsdata \
    -v /tmp:/tmp \
    -e PPIO_API_KEY=sk_xxxxx \
    -e NCCL_DEBUG=INFO \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -e NCCL_IB_DISABLE=0 \
    vllm/vllm-openai:v0.10.2 \
    sleep infinity
```

关键 Docker 参数说明:
- `--net=host`: Ray 和 NCCL 跨节点通信必需
- `--ipc=host`: 共享内存用于 GPU 进程间通信
- `--shm-size=64g`: 保证足够的共享内存
- `-e NCCL_SOCKET_IFNAME=eth0`: 指定 NCCL 使用的网卡 (根据实际网卡名调整)

### 步骤 2: 在每个容器内安装依赖

在**每个节点**的容器内执行:

```bash
docker exec -it deepswe-train bash

cd /workspace/rllm

# 安装 verl
pip install -e ./verl
pip install -e "./verl[vllm]"

# 安装 rllm
pip install -e ".[swe]"

# 安装其他依赖
pip install ppio_sandbox pylatexenc pandas datasets
pip install git+https://github.com/agentica-project/R2E-Gym.git

# 安装 ray (如果镜像中没有)
pip install "ray[default]>=2.40"
```

### 步骤 3: 启动 Ray 集群

**在 Head 节点 (111.6.123.35) 的容器内:**

```bash
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-cpus=64 \
    --num-gpus=8
```

**在每个 Worker 节点 (36/37/38) 的容器内:**

```bash
ray start \
    --address=111.6.123.35:6379 \
    --num-cpus=64 \
    --num-gpus=8
```

### 步骤 4: 验证 Ray 集群

在 Head 节点容器内:

```bash
ray status
```

预期输出:
```
======== Autoscaler status ========
Node status
---------------------------------------------------------------
Active:
 1 node(s) 111.6.123.35 ...
 1 node(s) 111.6.123.36 ...
 1 node(s) 111.6.123.37 ...
 1 node(s) 111.6.123.38 ...

Resources
---------------------------------------------------------------
Total Usage: 0/256 CPU, 0/32 GPU
```

也可通过浏览器访问 Ray Dashboard: `http://111.6.123.35:8265`

### 步骤 5: 准备数据

在 Head 节点容器内:

```bash
cd /workspace/rllm
python3 examples/swe/prepare_swe_data.py

# 验证
ls -la data/swe/
# R2E_Gym_Subset.parquet
# SWE_Bench_Verified.parquet
```

### 步骤 6: 启动训练

```bash
cd /workspace/rllm
export PPIO_API_KEY=sk_xxxxx
bash ppio/scripts/train_qwen3_32b_32h200.sh
```

---

## 训练配置摘要

| 参数 | 8x H200 (单节点) | 32x H200 (4 节点) |
|------|-----|-----|
| 模型 | Qwen3-32B | Qwen3-32B |
| GPU 总数 | 8 | 32 |
| Tensor Parallel | 8 | 8 (节点内) |
| Sequence Parallel | 8 | 8 (节点内) |
| FSDP | 8 GPU | 32 GPU (跨节点) |
| Train Batch Size | 8 | 32 |
| PPO Mini Batch Size | 8 | 32 |
| Rollout N | 8 | 8 |
| Max Response Length | 32768 | 32768 |
| Total Epochs | 200 | 200 |
| Checkpoint Freq | 10 | 5 |

### 关键区别

- **FSDP 跨节点**: 模型参数和优化器状态分片到 32 块 GPU, 每 GPU 内存占用降低 ~4x
- **Batch Size 4x**: 每步训练吞吐量提升 4 倍
- **Ray 调度**: Ray 自动分配 Worker 到各节点的 GPU

---

## 自定义参数

训练脚本支持通过环境变量覆盖默认配置:

```bash
# 调整 batch size
TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 bash ppio/scripts/train_qwen3_32b_32h200.sh

# 调整实验名称和 checkpoint 目录
EXPERIMENT_NAME=qwen3-32b-run2 \
CHECKPOINT_DIR=/3fsdata/checkpoints/run2 \
bash ppio/scripts/train_qwen3_32b_32h200.sh

# 指定模型路径
MODEL_PATH=/models/Qwen3-32B \
bash ppio/scripts/train_qwen3_32b_32h200.sh

# 调整保存频率
SAVE_FREQ=10 TEST_FREQ=10 bash ppio/scripts/train_qwen3_32b_32h200.sh
```

---

## NCCL 网络配置

根据网络类型调整 NCCL 参数:

### InfiniBand (推荐, 高带宽)

```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5     # 根据实际 IB 设备调整
export NCCL_SOCKET_IFNAME=eth0
```

### RoCE (RDMA over Converged Ethernet)

```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
```

### TCP (普通以太网)

```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
```

查看节点网卡名:

```bash
ip link show    # 查看网卡
ibstat          # 查看 IB 设备 (如有)
```

---

## 常见问题

### Q: Ray Worker 无法加入集群

```bash
# 1. 检查网络连通性
ping 111.6.123.35   # 从 worker 节点 ping head

# 2. 检查 Ray 端口
nc -zv 111.6.123.35 6379

# 3. 确认 --net=host
docker inspect deepswe-train | grep NetworkMode
# 应为 "host"

# 4. 检查防火墙
iptables -L -n  # 确保 6379, 8265 等端口开放
```

### Q: NCCL 通信超时

```bash
# 开启详细日志
export NCCL_DEBUG=INFO

# 检查网卡名
ip addr show | grep 'state UP'
# 更新 NCCL_SOCKET_IFNAME 为实际活跃网卡名

# 如果没有 InfiniBand, 禁用 IB
export NCCL_IB_DISABLE=1
```

### Q: GPU OOM (显存不足)

```bash
# 降低 GPU 内存利用率
# 编辑 train_qwen3_32b_32h200.sh:
GPU_MEMORY_UTILIZATION=0.6

# 或降低 batch size
TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 bash ppio/scripts/train_qwen3_32b_32h200.sh
```

### Q: 共享文件系统问题

```bash
# 确认所有节点能访问相同路径
for node in 111.6.123.35 111.6.123.36 111.6.123.37 111.6.123.38; do
    ssh $node "ls /models/models/Qwen3-32B/ | head -3"
    ssh $node "ls /workspace/rllm/data/swe/"
done
```

### Q: 如何监控训练

```bash
# Ray Dashboard
# 浏览器打开: http://111.6.123.35:8265

# WandB (如已配置)
# 设置 WANDB_API_KEY 后, 训练日志自动上传到 WandB

# GPU 使用监控 (在任意节点)
watch -n 1 nvidia-smi
```

---

## 集群管理命令

```bash
# 查看集群状态
bash ppio/scripts/setup_32h200_cluster.sh status

# 仅重启 Ray 集群 (不重启容器)
bash ppio/scripts/setup_32h200_cluster.sh ray-stop
bash ppio/scripts/setup_32h200_cluster.sh ray-start

# 停止所有 (容器 + Ray)
bash ppio/scripts/setup_32h200_cluster.sh stop

# 重新启动所有
bash ppio/scripts/setup_32h200_cluster.sh setup
```
