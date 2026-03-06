# .22/.23 节点性能问题排查 (2026-03-05)

## 问题描述

在 .22/.23 节点上运行的 32B 实验 (deepswe-32b) 比 .14/.17 节点上相同配置的实验慢约 **3.6 倍**。

| 集群 | Step | Top-5 LLM 时间 | 每步时间 |
|------|------|---------------|---------|
| .14/.17 | 134 | 407s - 595s (~535s avg) | ~17 min |
| .22/.23 | 1 | 1423s - 2521s (~1924s avg) | ~50 min |

## 排查过程

### 1. 跨节点网络测试

#### RDMA 带宽 (ib_write_bw)
两组节点的 RDMA 带宽基本相同：
- .22 → .23: **390.77 Gb/s**
- .14 → .17: **390.54 Gb/s**

#### RDMA 延迟 (ib_write_lat)
| 指标 | .22/.23 | .14/.17 | 差异 |
|------|---------|---------|------|
| avg | 3.35 μs | 2.62 μs | 1.3x |
| stdev | 0.77 μs | 0.08 μs | **9.6x** |
| 99%ile | 7.62 μs | 2.77 μs | 2.75x |
| max | 13.11 μs | 5.40 μs | 2.4x |

.22/.23 的跨节点延迟抖动明显更高，但由于 rollout 在单节点内运行，这不应该直接影响 LLM 推理。

### 2. NCCL NVLS 问题

发现 .22/.23 节点的 NCCL NVLS (NVLink SHARP) 出现错误：
```
NCCL WARN Cuda failure 1 'invalid argument'
Failed to bind NVLink SHARP (NVLS) Multicast memory: CUDA error 2 'out of memory'
```

而 .14/.17 正常禁用了 NVLS (0 nvls channels)。

**修复**: 在 `ray_runtime_env.py` 和 `entrypoint.sh` 中添加：
```python
"NCCL_NVLS_ENABLE": "0",
```

修复后确认 NVLS 被禁用，但**性能没有改善**。

### 3. 单节点 NVLink 带宽测试 (根本原因)

使用 PyTorch 测试 GPU 间通信带宽：

```python
# GPU-to-GPU copy test (1GB tensor)
src = torch.randn(1024*1024*256, device="cuda:0")
dst.copy_(src)  # copy to cuda:1
```

**测试结果:**

| GPU 对 | .14 | .17 | .22 | .23 |
|--------|-----|-----|-----|-----|
| 0→1 | 342 GB/s | 359 GB/s | **311 GB/s** | 313 GB/s |
| 0→4 | 359 GB/s | 359 GB/s | **238 GB/s** | 310 GB/s |
| 2→5 | 360 GB/s | 359 GB/s | **291 GB/s** | 335 GB/s |
| 3→7 | 359 GB/s | 359 GB/s | **252 GB/s** | 335 GB/s |

**关键发现:**
- .14/.17 的 NVLink 带宽稳定在 **~359 GB/s** (理论峰值的 ~96%)
- .22 的跨 NUMA GPU 对 (0→4, 3→7) 带宽仅 **238-252 GB/s**，比正常低 **30-34%**
- .23 略低于正常但比 .22 好

## 根本原因

**.22 节点存在 NVLink 硬件问题**，导致 GPU 间通信带宽显著降低。

vLLM 使用 TP=8 (tensor parallelism) 进行推理，需要大量 all-reduce 操作在 8 个 GPU 之间同步。NVLink 带宽降低 30% 直接导致推理速度下降。

可能的硬件原因：
1. NVSwitch 故障或配置问题
2. GPU 与 NVSwitch 之间的连接问题
3. PCIe/CXL 链路问题

## 已实施的修复

### 1. NCCL NVLS 禁用

**文件: `rllm/trainer/verl/ray_runtime_env.py`**
```python
PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        ...
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_NVLS_ENABLE": "0",  # 新增
    },
}
```

**文件: `k8s/scripts/entrypoint.sh`**
```bash
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0  # 新增
```

### 2. 代码同步

由于 K8s StatefulSet 使用 hostPath 挂载源代码，需要手动同步到各节点：
```bash
rsync -avz /root/develop/ref/rllm/rllm/trainer/verl/ray_runtime_env.py \
    root@10.83.115.22:/root/develop/ref/rllm/rllm/trainer/verl/
rsync -avz /root/develop/ref/rllm/rllm/trainer/verl/ray_runtime_env.py \
    root@10.83.115.23:/root/develop/ref/rllm/rllm/trainer/verl/
```

## 建议

1. **联系基础设施团队**检查 .22 节点的 NVSwitch 和 GPU 连接
2. 运行 `nvidia-smi nvlink -s` 检查 NVLink 链路状态和错误计数
3. 考虑将 .22 节点**排除出训练集群**，或更换节点
4. 如果必须使用 .22/.23，可以考虑降低 TP size (如 TP=4)，但会显著降低吞吐量

## 诊断命令

### NVLink 带宽测试脚本
```python
import torch
import time

def test_nvlink_bw(src_gpu, dst_gpu):
    torch.cuda.set_device(src_gpu)
    src = torch.randn(1024*1024*256, device=f"cuda:{src_gpu}")
    dst = torch.empty_like(src, device=f"cuda:{dst_gpu}")
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    return (src.numel() * src.element_size() * 10) / elapsed / 1e9

# 测试关键 GPU 对
for pair in [(0,1), (0,4), (2,5), (3,7)]:
    bw = test_nvlink_bw(*pair)
    print(f"GPU{pair[0]} -> GPU{pair[1]}: {bw:.1f} GB/s")
```

### RDMA 带宽测试
```bash
# 服务端
ib_write_bw -d mlx5_0 -p 18515 --report_gbits

# 客户端
ib_write_bw -d mlx5_0 -p 18515 --report_gbits <server_ip>
```

### RDMA 延迟测试
```bash
# 服务端
ib_write_lat -d mlx5_0 -p 18515

# 客户端
ib_write_lat -d mlx5_0 -p 18515 <server_ip>
```

## 相关文件

- `k8s/statefulset-32b.yaml` - .22/.23 实验的 StatefulSet 配置
- `k8s/scripts/train_deepswe_32b_k8s.sh` - 训练脚本
- `k8s/scripts/entrypoint.sh` - NCCL 环境配置
- `rllm/trainer/verl/ray_runtime_env.py` - Ray runtime 环境变量
