# DeepSWE 训练指标详解

本文档解释实验报告中各指标的含义、正常范围、诊断意义，以及当前实验暴露的问题。

## 目录

1. [Reward / Score 指标](#1-reward--score-指标)
2. [PPO 训练指标](#2-ppo-训练指标)
3. [序列长度指标](#3-序列长度指标)
4. [Trajectory 结果分类](#4-trajectory-结果分类)
5. [Timing / 性能指标](#5-timing--性能指标)
6. [当前实验诊断总结](#6-当前实验诊断总结)

---

## 1. Reward / Score 指标

### Score (critic/score/mean)

- **含义**: batch 内所有 trajectory 的平均 reward。本实验用 binary reward：解决问题=1，未解决=0。score=0.355 即 35.5% 的 trajectory 成功解决了问题。
- **正常范围**: 取决于数据集难度。R2E Gym 子集上通常 0.15-0.45。
- **诊断**: **最核心的训练效果指标**。持续上升说明模型在学会解决更多问题。
- **当前**: 从 0.202 (step 80-89) 上升至 0.355 (step 110-119)，后来在 0.35-0.45 横盘，训练 score 未见明显突破。

### R=1 / R=0

- **含义**: 成功/失败的 trajectory 计数。R1 Rate = R=1 / Total。
- **诊断**: 比 score 更直观，直接反映 solve rate。

### Advantages (adv_mean)

- **含义**: RLOO (REINFORCE Leave-One-Out) 估计的优势函数。对 batch 内每条 trajectory，用同一 prompt 下其他 trajectory 的平均 reward 作为 baseline，计算 `advantage = reward - baseline`。
- **为什么当前始终为负 (-0.01 ~ -0.06)**: 因为大多数 trajectory reward=0 (~75%)，只有少数 reward=1。大部分 trajectory 的 advantage = `0 - mean = -mean < 0`。只有 reward=1 的 trajectory 有正 advantage。这是 **sparse reward** 问题下的正常现象。
- **如果 adv_mean 接近 0**: 说明 reward 分布均匀，baseline 估计准确。
- **如果 adv_mean 大幅偏负**: 说明成功率极低，大部分更新在"惩罚"失败行为。

### solve_none / solve_partial / solve_all

- **含义**: 按 prompt 分组（每个 prompt 有 n=8 条 trajectory），统计：
  - `solve_none`: 8 条全部失败的 prompt 数
  - `solve_partial`: 部分成功的 prompt 数
  - `solve_all`: 8 条全部成功的 prompt 数
- **诊断**: 理想情况下 `solve_partial` 占多数（有正有负的 advantage，训练信号最丰富）。`solve_none` 过多说明 batch 偏难，`solve_all` 过多说明偏简单——两者的 advantage 方差都为 0，对训练无贡献。

---

## 2. PPO 训练指标

### PG Loss (actor/pg_loss)

- **含义**: PPO 策略梯度损失。计算方式为 `-advantage × clipped_ratio`，其中 ratio = π_new(a|s) / π_old(a|s)。直觉上衡量"当前策略与应该更新方向之间的差距"。
- **正常范围**: 随 batch 波动大（0.01 ~ 17.66），因为不同问题难度差异大。
- **趋势意义**: 下降意味着高 reward trajectory 的概率在增加，策略在逐步靠近"好的行为"。
- **当前**: 平均 8.04，从 11.28 降至 6.99，趋势健康。

### Grad Norm (actor/grad_norm)

- **含义**: 梯度的 L2 范数，衡量一步更新的"力度"。
- **正常范围**:
  - < 10: 梯度消失，学不动
  - 100-2000: 正常范围（取决于模型大小）
  - \> 10000: 梯度爆炸，训练不稳定
- **当前 ~785**: 梯度信号正常存在。但因为 lr=1e-6，实际参数更新量 `lr × grad ≈ 1e-6 × 800 ≈ 0.0008`，极其微小。**不是没有梯度，是步长太小。**
- **偶现尖峰** (如 step 92 的 1661): 某个 batch 的梯度偏大，但单次尖峰不影响稳定性。

### Entropy (actor/entropy)

- **含义**: 策略的信息熵，衡量"随机性/不确定性"。Entropy 高 = 模型在多个 token 间犹豫不决；Entropy 低 = 模型很确定要输出什么 token。
- **趋势意义**:
  - **缓慢下降**: 正常学习，策略在变得更确定（当前状况）
  - **快速下降 → 极低**: entropy collapse，模型退化成固定模式输出，需要立即干预
  - **持续不变**: 策略没学到东西
- **当前**: 6800 → 4300 (step ~286) 缓慢下降（学习信号），随后**反转上升至 8520 (step 341)**，超过训练初始值。这是 **lr=1e-6 下有效学习已到极限**的信号——模型开始在策略空间随机游走，熵在增大。

### pg_clipfrac (actor/pg_clipfrac)

- **含义**: PPO 的核心安全机制。当新旧策略的概率比 `ratio = π_new/π_old` 超出 `[1-ε, 1+ε]`（这里 ε = clip_ratio_high = 0.28，即 `[0.72, 1.28]`）时，梯度会被 clip 掉。pg_clipfrac 是被 clip 的 token 比例。
- **正常范围**: 5-15%。
- **当前**: **≈ 0.01%，几乎为零。这是核心问题指标。**
- **诊断**:
  - 新策略和旧策略概率比始终在 [0.72, 1.28] 内，PPO 的"安全带"从未被拉紧
  - 根因：lr=1e-6 太低，每步参数更新太小，策略几乎不变
  - 类比：给汽车装了限速器（120km/h），但车实际只开 5km/h，限速器从未生效

### pg_clipfrac_lower

- **含义**: 被**下界** clip 的比例（概率比低于 `1-ε`，即 ratio < 0.72）。
- **当前**: 0%。
- **如果不为 0**: 说明策略在某些 token 上的概率**大幅下降**，可能有 collapse 风险。

### KL Divergence (actor/ppo_kl)

- **含义**: 当前策略与 reference model（初始模型 / step 0）之间的 KL 散度。衡量"策略跑偏了多少"。
- **正常范围**: 0.01-0.1。
- **当前**: **≈ 0 (数量级 1e-5)**。
- **诊断**:
  - KL≈0 = 当前策略几乎等于初始模型，训练 59 步后参数变化极微
  - 与 pg_clipfrac≈0 互相印证：模型在极其缓慢地学习
  - 好消息：KL 没有变负或暴增，训练是稳定的

### clip_ratio_high 参数

- **含义**: PPO clip 的上界 ε=0.28，即允许 ratio 在 [0.72, 1.28] 范围内。
- **当前配置下**: 形同虚设，因为 ratio 从未接近边界。
- **如果提高 lr**: 需要关注 clipfrac 是否升至合理范围（5-15%），以及是否需要调整 ε。

---

## 3. 序列长度指标

### Prompt Length

- **含义**: 输入 prompt 的 token 长度（包含 system prompt + issue 描述）。
- **当前**: 平均 ~1800 tokens，非常稳定。受 issue 文本长度影响。
- **max_prompt_length=4096**: 超过此长度的 prompt 会被截断。当前无 prompt 被截断。

### Response Length

- **含义**: 模型输出的总 token 数，包含多轮 agent 交互的所有 thought + action 输出。
- **当前**: 平均 ~17k tokens，最大 32768 (=max_response_length 上限)。
- **趋势**: 略有下降（17800 → 17100），模型可能在学习更精简地解决问题。

### Response Clip Ratio

- **含义**: response 达到 max_response_length=32768 上限被强制截断的 trajectory 比例。
- **当前**: 0.7%，从 1.1% 降到 0.2%。说明越来越少的 trajectory 需要被截断。

### Abort Ratio

- **含义**: trajectory 因异常被中止的比例（如环境崩溃、超时等）。
- **当前**: 0.0%，所有 trajectory 都正常完成。

---

## 4. Trajectory 结果分类

### ENV_DONE

- **含义**: 环境判定任务结束（agent 提交了答案），不论对错。
- **当前**: 85.4%。绝大多数 trajectory 正常结束。

### MAX_STEPS

- **含义**: agent 达到 max_steps=30 步交互上限仍未提交答案。
- **当前**: 4.5%。少数问题上模型在"原地打转"——反复尝试但无法收敛到提交。
- **诊断**: 如果比例上升，说明模型在某类问题上 stuck。

### TRUNCATION

- **含义**: response 达到 32768 token 上限被截断。与 response_clip_ratio 相关但不完全相同（TRUNCATION 是从 trajectory 终止原因统计）。
- **当前**: 10.2%。通常是复杂问题需要更多交互轮次。

### Overlong (overlong_filter)

- **含义**: 过长 trajectory 被 overlong_filter 直接丢弃，不参与 PPO 更新。这是训练效率优化——过长 trajectory 会导致 PPO mini-batch 中 padding 过多、计算浪费。
- **当前**: 14.6%。从 ~15% 降到 ~5% (近期训练步)。
- **权衡**: 丢弃过长 trajectory 提高训练效率，但也丢失了复杂问题上的训练信号。

### Errors

- **含义**: trajectory 执行过程中的异常错误（Docker 容器崩溃、环境无法启动等）。
- **当前**: 0。59 步 10,017 条 trajectory 零错误，基础设施非常稳定。

---

## 5. Timing / 性能指标

### Collect Trajectory

- **含义**: 一个 step 中采集所有 128 条 trajectory 的总时间。包括：
  - vLLM 推理（模型生成 action）
  - Docker 容器中的环境执行（运行 bash 命令、测试等）
  - 多轮交互的串行时间
- **当前**: 占总步时间的 53%（~12-14 min/step）。
- **瓶颈**: batch 内最慢的 trajectory 决定总时间（所有 GPU 要等它完成）。

### Update Actor

- **含义**: FSDP 分布式 PPO 训练更新。把采集到的 trajectory 用 PPO 算法更新模型参数。
- **当前**: ~6 min/step，占 22%。这是实际的 GPU 训练计算时间。

### Log Prob (old_log_prob)

- **含义**: 用**当前模型**重新计算旧 trajectory 中每个 token 的 log probability。PPO 需要新旧概率比 `π_new/π_old`，旧概率在采集时已记录，新概率需要用更新后的模型重算。
- **当前**: ~55s/step，占比小。

### Transform Trajectory

- **含义**: 将原始 trajectory 数据转换为 PPO 训练格式（tokenization、padding、构造 DataProto 等）。
- **当前**: <1s/step，忽略不计。

### MFU (Model FLOPs Utilization)

- **含义**: 模型浮点运算利用率。理论上 GPU 每秒能做多少 FLOP，实际用了多少。只统计 update_actor 阶段。
- **当前**: 15.4%。
- **为什么低**:
  - FSDP param_offload + optimizer_offload: 参数和优化器状态在 CPU/GPU 间搬运
  - ulysses_sequence_parallel=8: 序列并行有通信开销
  - 32B 模型在 16 卡上的 batch 较小（ppo_mini_batch_size=8）
- **正常范围**: 此配置下 15-20% 是预期的。

### Tail Ratio

- **含义**: 最慢 trajectory 的完成时间 / 平均完成时间。衡量长尾效应。
- **当前**: 2.15x，即最慢的 trajectory 比平均慢 2.15 倍。
- **诊断**: 所有 GPU 必须等最慢的 trajectory 完成才能进入 update_actor。tail ratio 越高，GPU 空闲等待时间越长。
- **趋势**: 从 2.21x 改善到 2.08x，说明长尾在缩短。

### Token Mismatch

- **含义**: vLLM 推理产生的 token 序列与 HuggingFace tokenizer 重新编码后是否一致。PPO 的 importance sampling 依赖新旧 log_prob 对齐同一 token 序列。
- **当前**: 0.0%。之前版本有 bug（tokenizer 差异导致不匹配），现已修复。
- **如果不为 0**: PPO 的概率比计算会出错，训练信号失真。

### GPU Idle

- **含义**: GPU 没有做有效训练计算的时间占比。主要来自 collect_trajectory 阶段（GPU 做推理但大量时间在等环境返回）和 validation。
- **当前**: **77.9%**。16 块 H200 有近 78% 的时间是空闲的。
- **这是 agent RL 的固有问题**: 环境交互是串行瓶颈，不像纯语言模型 RL 那样全在 GPU 上完成。

---

## 6. 当前实验诊断总结

### 核心问题：学习率过低 (lr=1e-6)

三个指标共同指向同一结论：

| 指标 | 当前值 | 正常值 | 说明 |
|------|--------|--------|------|
| pg_clipfrac | ≈0% | 5-15% | PPO clip 从未触发 |
| KL | ≈0 | 0.01-0.1 | 策略几乎未偏离初始模型 |
| grad_norm | ~785 | ~785 | 梯度信号正常存在 |

**结论**: `grad_norm × lr = 800 × 1e-6 = 0.0008`，每步参数更新量极微。PPO 的 clip 机制形同虚设。模型在**正确的方向上极其缓慢地**学习。

### 积极信号

| 信号 | 证据 |
|------|------|
| 模型曾在学习 | score 从 0.20 → 0.43 (step ~183)，曾持续上升 |
| Validation 峰值 | R1 rate 在 step 290 达到 42.0% |
| 训练稳定 | 0 errors，grad_norm 无爆炸 |

### 效率问题

| 问题 | 数据 | 影响 |
|------|------|------|
| GPU idle 过高 | 48.7% (collect 阶段) | 16×H200 大量浪费 |
| Validation 耗时长 | ~65-101min/次 (每10步) | 占总时间比例大 |
| Overlong 丢弃 | 11.6% trajectory | 丢失复杂问题训练信号 |
| **Entropy 反转上升** | 4300→8520，超初始值 | 模型开始退化 |
| **Response 变长** | 17.5k→22k tokens | 策略变得更犹豫 |

### 建议的参数调整

| 参数 | 当前 | 建议 | 理由 |
|------|------|------|------|
| lr | 1e-6 | 5e-6 ~ 1e-5 | pg_clipfrac≈0，步长太小 |
| test_freq | 10 | 20 | 每次 validation ~86min，太频繁 |
| save_freq | 10 | 20 | 每个 checkpoint ~184G |

提高 lr 后需监控:
- pg_clipfrac 是否升至 5-15%（健康范围）
- entropy 是否急剧下降（collapse 风险）
- KL 是否暴增（策略跑偏风险）
- grad_norm 是否出现尖峰（不稳定风险）

---

*文档创建时间: 2026-03-06，更新: 2026-03-09*
*适用实验: DeepSWE 2节点训练 (Qwen3-32B, step 61-341)*
