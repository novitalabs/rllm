# DeepSWE 2节点训练实验分析报告 (Step 31-69)

**日期**: 2026-03-04
**集群**: 2×8 H200 GPUs (10.83.115.14/.17)
**模型**: Qwen3-32B (从step 30 checkpoint续训)
**中断原因**: Step 70 checkpoint保存时磁盘空间不足 (367G free / 7.0T)

## 训练概要

| 指标 | 值 |
|------|-----|
| 完成步数 | 39 steps (step 31 → 69) |
| 最后完好checkpoint | step 60 |
| 总训练时间 | 19.7 小时 |
| GPU-hours | 630h (总), 389h (idle, 62%) |
| 平均步时间 | 30.3 分钟 (普通步), ~85 分钟 (验证步) |
| 总 Trajectories | 6,359 |
| Batch Size | 16 (每步128 trajectories, n=8) |

## 性能指标

### 时间分解
- **Collect Trajectory**: 61.7% (平均18.7分钟/步)
- **Update Actor**: 19.7% (平均6.0分钟/步)
- **Log Prob**: 平均55.2秒/步
- **MFU (训练阶段)**: 15.7%

### 与4节点对比

| 指标 | 4节点 (step 1-30) | 2节点 (step 31-69) | 变化 |
|------|-------------------|-------------------|------|
| 平均步时间 | 34.3 min | 30.3 min | -12% |
| Collect Trajectory | 19.9 min (58.2%) | 18.7 min (61.7%) | -6% |
| Update Actor | 6.2 min (17.9%) | 6.0 min (19.7%) | -3% |
| MFU | 16.7% | 15.7% | -6% |
| Tail Ratio | 2.22x | 2.64x | +19% |
| Agent Steps | 19.3 | 18.2 | -6% |
| Token Mismatch | 0.0% | 0.0% | - |

### Trajectory 效率
- **LLM推理时间**: 平均6-7分钟/trajectory, 最大17分钟
- **环境执行时间**: 平均4-23秒, 最大15分钟
- **Tail Ratio**: 平均2.64x (最慢trajectory vs 平均)
- **Agent Steps**: 平均18.2步/trajectory
- **Token Mismatch**: 0.0%

## 训练效果

### Score/Reward 趋势 (训练集)

| Step范围 | 平均Score | R=1比率 |
|---------|-----------|---------|
| 31-40 | 0.216 | 21.3% |
| 41-50 | 0.195 | 19.4% |
| 51-60 | 0.237 | 23.7% |
| 61-69 | 0.244 | 24.4% |

- **总体平均Score**: 0.222 (4节点: 0.196, +13%)
- **R=1 Trajectories**: 1,364/6,359 (21.4%)
- **Overlong过滤**: 926 (14.6%)

### Validation Score (SWE-bench Verified)

| Step | val/test_score | 趋势 |
|------|---------------|------|
| 40 | **18.6%** | baseline |
| 50 | 16.6% | -2.0% |
| 60 | 15.6% | -3.0% |

**验证集得分呈下降趋势**，从18.6%降至15.6%，但训练集score略有上升。这可能是:
1. 噪声波动 (128样本的验证集方差较大)
2. 轻微过拟合训练分布
3. 策略更新幅度极小 (clipfrac≈0)，变化在噪声范围内

### 训练动态

| 指标 | 平均值 | 趋势 | 诊断 |
|------|--------|------|------|
| PG Loss | 10.05 | 波动大 | 正常 |
| Grad Norm | 802.4 | 稳定偏高 | 可能需要clip |
| Entropy | 6,240 | 略下降 | 策略略有确定化 |
| pg_clipfrac | 0.02% | **接近0** | **策略更新极小** |
| KL | ~0 | **接近0** | **策略未偏离reference** |
| Response Len | 18,377 | 稳定 | 正常 |
| Clip Ratio | 1.2% | 低 | 正常 |
| Abort Ratio | 0.0% | - | 正常 |

### 关键问题: pg_clipfrac ≈ 0

pg_clipfrac 持续接近0%，说明PPO的clipping几乎没有被触发。结合KL≈0，表明:
- 策略在69步训练后仍几乎等于初始reference model
- 学习率 1e-6 可能**过低**，策略更新步长太小
- 虽然grad_norm ~800 (不小)，但optimizer step后的参数变化微乎其微

### 序列长度
- **Prompt Length**: 平均1,800 tokens (稳定)
- **Response Length**: 平均18,377 tokens (最大32,768)
- **Response Clip比率**: 1.2%

## Trajectory 结果分布

| 状态 | 数量 | 比率 |
|------|------|------|
| ENV_DONE (正常结束) | 5,433 | 85.4% |
| MAX_STEPS (达到30步) | 285 | 4.5% |
| TRUNCATION (token截断) | 641 | 10.1% |
| Overlong 过滤 | 926 | 14.6% |
| **Errors** | 112 | 1.8% |

**注意**:
- Step 40/50/60 为验证步 (包含~500 trajectories + testing)
- Step 69 的 112 errors 是磁盘空间不足/registry容器崩溃导致
- 排除 step 69 后 error rate 为 0%

## 验证步详情

| Step | Trajectories | R=1 | Overlong | Errors | Testing时间 | Val Score |
|------|-------------|------|----------|--------|-------------|-----------|
| 40 | 500 | 115 (23%) | 185 | 0 | ~59min | 18.6% |
| 50 | 500 | 91 (18%) | 188 | 0 | ~59min | 16.6% |
| 60 | 500 | 107 (21%) | 237 | 0 | ~63min | 15.6% |

验证步总时间 ~85分钟 (collect ~20min + testing ~60min)。

## 中断分析

### 磁盘使用 (.14节点, 7.0T)

| 路径 | 大小 | 说明 |
|------|------|------|
| /data/registry-cache/ | 2.8T | 本地registry (4619/4622 tags) |
| /data/docker-cache/ | 2.0T | DinD overlay2 (pod-0) |
| /data/checkpoints/deepswe/.../swe-agent-rl/ | 1.1T | Steps 10-60 (6×184G) + partial 70 |
| /data/docker/ | 358G | 宿主机Docker |
| /data/models/ | 245G | Qwen3-32B |
| /data/checkpoints/deepswe-full/ | 184G | 旧4节点实验 |
| **剩余** | **367G** | |

### 崩溃原因
Step 70 checkpoint保存时 `torch.save()` 写入失败:
```
RuntimeError: [enforce fail at inline_container.cc:858] . PytorchStreamWriter failed writing file data/0: file write failed
```

### 可清理空间

| 项目 | 大小 | 风险 |
|------|------|------|
| 旧4节点实验 (deepswe-full/) | 184G | 无 |
| Partial step 70 checkpoint | 741M | 无 |
| 旧checkpoints (steps 10-50) | 920G | 只保留step 60 |
| **总计** | **~1.1T** | |

清理后可用空间: ~1.5T

## 关键发现

### 正面
1. **2节点效率更优**: 步时间从34.3min降至30.3min (减少12%)
2. **零错误运行**: 排除最后step外，38步零error (4节点run有434 errors)
3. **Token Mismatch 已修复**: 0.0%
4. **Registry本地化完成**: 4619/4622 tags，完全消除Docker Hub依赖
5. **Score略有提升**: 从0.196提升到0.222 (+13%)

### 需要改进
1. **pg_clipfrac ≈ 0**: 策略更新极小，69步训练后策略几乎未变
2. **KL ≈ 0**: 策略未偏离reference model
3. **验证得分下降**: 18.6% → 15.6% (可能是噪声)
4. **GPU idle时间高**: 62%时间在等待trajectory收集
5. **磁盘空间不足**: .14节点只剩367G，需要清理

## 建议

### 训练超参数调整
1. **提高学习率**: 当前1e-6过低，建议尝试 **5e-6 或 1e-5**
   - pg_clipfrac和KL都接近0表明步长太小
   - Grad norm ~800说明梯度信号存在，只是step size太小
2. **考虑增大clip_ratio_high**: 当前0.28，如果提高lr后clipfrac仍低可以放宽
3. **关注entropy变化**: 当前6240→5688有轻微下降趋势，提高lr后需监控是否collapse

### 基础设施
4. **清理磁盘**: 删除旧checkpoints释放~1.1T空间
5. **减少checkpoint保存频率**: save_freq=10 → 20，每个checkpoint 184G
6. **监控.14磁盘**: 设置告警，避免再次写满

### 实验方向
7. **从step 60 checkpoint续训**: 使用更高学习率重启
8. **或从base model重新训练**: 如果认为前69步无效更新
9. **缩短验证频率**: test_freq=10 → 20，每次验证~60min

---

*报告生成时间: 2026-03-04*
*数据来源: kubectl logs deepswe-training-0 (deepswe namespace)*
