# DeepSWE 4节点训练实验分析报告 (Step 1-30)

**日期**: 2026-03-03
**集群**: 4×8 H200 GPUs (10.83.115.18/.21/.22/.23)
**模型**: Qwen3-32B

## 训练概要

| 指标 | 值 |
|------|-----|
| 完成步数 | 30 steps |
| 总训练时间 | 17.1 小时 |
| GPU-hours | 549h (总), 319h (idle, 58%) |
| 平均步时间 | 34.3 分钟 |
| 总 Trajectories | 4,892 |
| Batch Size | 32 (每步128 trajectories, n=8) |

## 性能指标

### 时间分解
- **Collect Trajectory**: 58.2% (平均19.9分钟/步)
- **Update Actor**: 17.9% (平均6.2分钟/步)
- **Log Prob**: 平均60.6秒/步
- **MFU (训练阶段)**: 16.7%

### Trajectory 效率
- **LLM推理时间**: 平均6-8分钟/trajectory, 最大14分钟
- **环境执行时间**: 平均4-33秒, 最大12分钟
- **Tail Ratio**: 平均2.22x (最慢trajectory vs 平均)
- **Agent Steps**: 平均19.3步/trajectory
- **Token Mismatch**: 0.0% (已修复)

## 训练效果

### Score/Reward 趋势
| Step范围 | 平均Score | R=1比率 |
|---------|-----------|---------|
| 1-10 | 0.188 | 17.9% |
| 11-20 | 0.207 | 20.6% |
| 21-30 | 0.203 | 19.6% |

- **总体平均Score**: 0.196
- **R=1 Trajectories**: 937/4892 (19.2%)
- **Overlong过滤**: 627 (12.8%)

### 训练动态
- **PG Loss**: 平均10.89, 波动范围-0.58到21.38
- **Grad Norm**: 平均792.7, 范围526-994
- **Entropy**: 平均6031 (变化不大,从5000到7370)
- **pg_clipfrac**: 0.02% (接近0, 策略更新幅度小)
- **KL散度**: ~0 (策略与reference model差异极小)

### 序列长度
- **Prompt Length**: 平均1800 tokens
- **Response Length**: 平均19,810 tokens (最大32,768)
- **Response Clip比率**: 1.8%

## Trajectory 结果分布

| 状态 | 数量 | 比率 |
|------|------|------|
| ENV_DONE (正常结束) | 4,206 | 87.0% |
| MAX_STEPS (达到30步) | 235 | 4.9% |
| TRUNCATION (token截断) | 392 | 8.1% |
| **Errors** | 434 | 9.0% |

**注意**: Step 10/20/30 包含 validation (500 trajectories)，errors主要集中在这些步骤。

## 关键发现

### 正面
1. **Token Mismatch 已修复**: 0.0%的mismatch率
2. **稳定的训练过程**: 30步无严重crash
3. **较好的 ENV_DONE 比率**: 87%的trajectory正常结束
4. **合理的 overlong 过滤**: 12.8%被过滤

### 需要改进
1. **Score提升有限**: 从初始~0.19到后期~0.20，提升不明显
2. **pg_clipfrac接近0**: 策略更新太保守，可能需要调整学习率
3. **KL接近0**: 策略几乎没有偏离reference model
4. **GPU idle时间过高**: 58%时间在等待trajectory收集
5. **Validation错误**: Step 10/20/30的验证出现大量errors

## 建议

1. **增加学习率**: 当前1e-6可能太小，考虑3e-6或5e-6
2. **监控entropy**: entropy稳定但没有明显下降趋势
3. **减少validation频率**: 或修复validation中的error问题
4. **优化Docker网络**: 新集群需要监控 docker bridge 网络状态

---

*报告生成时间: 2026-03-03*
