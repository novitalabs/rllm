# DeepSWE 32B 实验 (.22/.23 节点)

## 实验配置

| 项目 | 值 |
|------|-----|
| 模型 | Qwen3-32B |
| 节点 | 10.83.115.22, 10.83.115.23 |
| GPU | 2×8 H200 (TP=8, DP=2) |
| Batch Size | 16 prompts × 8 rollouts = 128 trajectories/step |
| 算法 | RLOO |
| 学习率 | 1e-6 |
| Max Steps | 30 |
| Max Response Length | 32768 |
| Max Prompt Length | 4096 |
| Temperature | 1.0 |
| Ulysses SP Size | 8 |
| FSDP Offload | param + optimizer |

## 实验结果 (截至 Step 21)

### 训练指标详情

| Step | Score | Solve None | Solve Partial | Solve All | Steps Mean | Response Len | Step Time |
|------|-------|------------|---------------|-----------|------------|--------------|-----------|
| 16 | 8.6% | 12 | 4 | 0 | 24.2 | 19537 | 2957s (49m) |
| 17 | **17.2%** | 11 | 4 | 1 | 20.7 | 17270 | 3001s (50m) |
| 18 | **17.2%** | 10 | 5 | 1 | 20.6 | 16778 | 3041s (51m) |
| 19 | 12.5% | 10 | 6 | 0 | 22.9 | 17373 | 2617s (44m) |
| 20 | 7.0% | 11 | 5 | 0 | 21.4 | 17822 | 8101s (135m)* |

*Step 20 包含 validation testing (4770s)

### Validation 结果

| Step | Test Score | Pass@k |
|------|------------|--------|
| 20 | **7.2%** | 7.2% |

### Actor 训练指标

| Step | Entropy | PG Loss | Grad Norm | PPO KL | Clip Frac | MFU |
|------|---------|---------|-----------|--------|-----------|-----|
| 16 | 7878 | 9.58 | 664 | 4.2e-6 | 9.0e-5 | 11.5% |
| 17 | 6074 | 5.00 | 514 | -2.0e-5 | 8.0e-5 | 10.1% |
| 18 | 6889 | 10.04 | 638 | 8.0e-6 | 1.3e-4 | 9.6% |
| 19 | 6610 | 7.75 | 736 | -4.9e-6 | 1.9e-4 | 10.1% |
| 20 | 6981 | 3.85 | 593 | 1.6e-5 | 1.2e-4 | 10.6% |

### Trajectory 时间统计

| Step | LLM Mean | LLM Max | Env Mean | Env Max | Total Mean | Total Max |
|------|----------|---------|----------|---------|------------|-----------|
| 16 | 1061s | 2238s | 50s | 421s | 1111s | 2242s |
| 17 | 889s | 2391s | 18s | 275s | 907s | 2394s |
| 18 | 872s | 2430s | 60s | 479s | 932s | 2432s |
| 19 | 890s | 2000s | 46s | 364s | 937s | 2003s |
| 20 | 922s | 2571s | 51s | 487s | 973s | 2573s |

### Top-5 最慢 Trajectories

#### Step 17
| Rank | Repo | Steps | Total | LLM | Env | Reason |
|------|------|-------|-------|-----|-----|--------|
| 1 | coveragepy | 25 | 2394s | 2391s | 2s | ENV_DONE |
| 2 | numpy | 26 | 2182s | 2179s | 3s | ENV_DONE |
| 3 | pyramid | 29 | 1894s | 1891s | 3s | ENV_DONE |
| 4 | coveragepy | 25 | 1890s | 1887s | 3s | ENV_DONE |
| 5 | pyramid | 30 | 1771s | 1766s | 5s | ENV_DONE |

#### Step 18
| Rank | Repo | Steps | Total | LLM | Env | Reason |
|------|------|-------|-------|-----|-----|--------|
| 1 | pyramid | 20 | 2432s | 2430s | 2s | TRUNCATION |
| 2 | datalad | 28 | 2219s | 2034s | 186s | ENV_DONE |
| 3 | pandas | 30 | 2193s | 2093s | 100s | ENV_DONE |
| 4 | pillow | 30 | 2162s | 2159s | 3s | ENV_DONE |
| 5 | pandas | 29 | 1989s | 1714s | 276s | ENV_DONE |

#### Step 19
| Rank | Repo | Steps | Total | LLM | Env | Reason |
|------|------|-------|-------|-----|-----|--------|
| 1 | aiohttp | 28 | 2003s | 2000s | 3s | ENV_DONE |
| 2 | pandas | 29 | 1938s | 1932s | 6s | ENV_DONE |
| 3 | tornado | 29 | 1927s | 1563s | 364s | ENV_DONE |
| 4 | scrapy | 30 | 1713s | 1415s | 298s | ENV_DONE |
| 5 | pillow | 23 | 1646s | 1624s | 22s | ENV_DONE |

#### Step 20 (Rollout)
| Rank | Repo | Steps | Total | LLM | Env | Reason |
|------|------|-------|-------|-----|-----|--------|
| 1 | pillow | 30 | 2573s | 2571s | 2s | MAX_STEPS |
| 2 | aiohttp | 20 | 2164s | 2161s | 3s | ENV_DONE |
| 3 | pillow | 29 | 2077s | 1872s | 206s | ENV_DONE |
| 4 | orange3 | 18 | 2062s | 2053s | 9s | ENV_DONE |
| 5 | aiohttp | 30 | 2004s | 2000s | 4s | ENV_DONE |

#### Step 20 (Validation - Timeout Issues)
| Rank | Repo | Steps | Total | LLM | Env | Reason |
|------|------|-------|-------|-----|-----|--------|
| 1 | sphinx-doc | 1 | 4452s | 4452s | 0s | ENV_TIMEOUT |
| 2 | django | 1 | 4448s | 4448s | 0s | ENV_TIMEOUT |
| 3 | sympy | 8 | 4444s | 4263s | 181s | ENV_TIMEOUT |
| 4 | django | 1 | 4416s | 4416s | 0s | ENV_TIMEOUT |
| 5 | django | 1 | 4397s | 4397s | 0s | ENV_TIMEOUT |

#### Step 21
| Rank | Repo | Steps | Total | LLM | Env | Reason |
|------|------|-------|-------|-----|-----|--------|
| 1 | pandas | 28 | 2139s | 1955s | 184s | ENV_DONE |
| 2 | pandas | 30 | 2067s | 1971s | 96s | ENV_DONE |
| 3 | tornado | 18 | 2020s | 1838s | 182s | ENV_DONE |
| 4 | pandas | 30 | 1957s | 1952s | 5s | MAX_STEPS |
| 5 | pandas | 30 | 1896s | 1882s | 14s | ENV_DONE |

### 汇总统计

- 总 Trajectories: **1133**
- 成功 (Reward=1): **113** (~10%)
- 运行时间: ~20 小时
- 完成步数: 21 步 (从 checkpoint step 16 恢复)

### 资源使用

| 指标 | 值 |
|------|-----|
| GPU 内存 (allocated) | 136 GB |
| GPU 内存 (reserved) | 147 GB |
| CPU 内存 | ~101 GB |

### Timing 分解 (Step 20)

| 阶段 | 时间 |
|------|------|
| collect_trajectory | 2700s (45m) |
| transform_trajectory | 0.6s |
| old_log_prob | 51s |
| adv (advantage) | 51s |
| update_actor | 504s (8.4m) |
| save_checkpoint | 75s |
| testing (validation) | 4770s (79m) |
| **total step** | **8101s (135m)** |

## 与 .14/.17 对比

| 指标 | .22/.23 (Step 20) | .14/.17 (Step 69) |
|------|-------------------|-------------------|
| Test Score | 7.2% | ~26% |
| 每步时间 (无 val) | ~50 min | ~17 min |
| LLM 时间 (mean) | ~920s | ~500s |
| LLM 时间 (max) | ~2500s | ~600s |
| 性能比 | 1x | **~3x faster** |

## 问题分析

### 根本原因: NVLink 硬件问题

.22 节点存在 NVLink 带宽问题，导致 GPU 间通信显著变慢：

| GPU 对 | .14/.17 | .22 | 降幅 |
|--------|---------|-----|------|
| 0→1 | 359 GB/s | 311 GB/s | -13% |
| 0→4 | 359 GB/s | 238 GB/s | **-34%** |
| 2→5 | 359 GB/s | 291 GB/s | -19% |
| 3→7 | 359 GB/s | 252 GB/s | **-30%** |

跨 NUMA GPU 对 (0→4, 3→7) 降幅最大，影响 TP=8 的 all-reduce 性能。

详细排查记录见: [node-22-23-performance-issue.md](./node-22-23-performance-issue.md)

### 已实施修复

1. **NCCL NVLS 禁用**: 添加 `NCCL_NVLS_ENABLE=0` 解决 OOM/invalid arg 错误
2. 性能问题需要硬件团队检查 NVSwitch

## 结论

由于 .22 节点硬件问题，实验速度比正常慢 **~3 倍**，不适合继续在该节点运行。

**建议**:
1. 联系基础设施团队检查 .22 节点 NVSwitch
2. 将实验迁移到 .14/.17 节点
3. 或等待硬件修复后重新启动

## 相关文件

- StatefulSet: `k8s/statefulset-32b.yaml`
- 训练脚本: `k8s/scripts/train_deepswe_32b_k8s.sh`
- Checkpoint: `/data/checkpoints/deepswe-32b/` (on .22/.23 nodes)

## 时间线

- 2026-03-05 10:57 UTC: 实验启动 (从 step 16 checkpoint 恢复)
- 2026-03-06 07:15 UTC: 实验停止 (Step 21)
- 原因: 硬件性能问题，效率过低
