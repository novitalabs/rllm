# DeepSWE Training - Code Issues & Fixes

rllm 框架代码层面的 bug 和优化，与部署环境无关。

---

## Issue 1: Retokenization Mismatch Masks Out 37% of Trajectories [FIXED]

**Symptom**: `traj/token_mismatch_mean ≈ 0.37`，37% 的 trajectory 的 `response_masks` 全为 0，不贡献梯度。日志输出 "detect the trajectory not accumulative at position X"。

**Root Cause**: 每个 agent step，`VerlEngine.get_model_response()` 将完整对话文本通过 HF tokenizer 重新编码生成 `prompt_ids`。但 `assemble_steps()` 用的是前一步 `prompt_ids + completion_ids` 拼接的 `accumulated_sequence`。由于 BPE 分词边界效应（在 added tokens `<|im_end|>`、`<|im_start|>` 与普通文本之间），每步约 2% 不匹配，25 步后累积到 37%。

**Fix**: 停止重新分词。在 agent 循环中直接累积 token IDs，后续 step 通过 `prompt_token_ids` 参数直接传给 vLLM，绕过文本重编码。BPE 边界在 added tokens 处是干净的（HF tokenizer 在 added tokens 前后独立 BPE 编码）。

**Changes**:
- `rllm/engine/rollout/verl_engine.py`: `get_model_response()` 增加 `prompt_token_ids` 参数，传入时跳过 chat template 渲染和 HF 编码
- `rllm/engine/agent_execution_engine.py`:
  - `get_model_response()`: 转发 `prompt_token_ids` 到 verl engine
  - `run_agent_trajectory_async()`: 维护 `accumulated_token_ids`，step 0 后累积 `eot_tokens + env_msg_tokens`

**Result**: `traj/token_mismatch_mean` 从 0.37 降至 0.0，有效训练 batch 从 ~40/64 恢复到 64/64。

---

## Issue 2: Overlong Initial Prompt Crashes Entire Training [FIXED]

**Symptom**: Step 10 validation 在 Trajectory 260 处崩溃，训练进程卡死 5 小时：
```
Exception: Trajectory 260: initial prompt length 4273 already exceeded max_prompt_length 4096, retrying
Exception: Trajectory 260 cannot complete. Please check the log message
```

**Root Cause**: 部分 SWE-bench 验证样本的 task description 较长，加上 system prompt 后 token 数超过 `max_prompt_length=4096`（如 4273）。原代码对此直接抛异常，`run_agent_trajectory_with_retry` 重试 3 次后仍然失败（prompt 是确定性的），然后 `trajectory_generator` 的 `raise e` 杀死 runner 线程，整个训练挂起。

注意 `data.filter_overlong_prompts=True` 只作用于 verl DataLoader 预过滤训练数据，不影响运行时 agent 动态组装的 prompt。

**Fix**:
1. `run_agent_trajectory_async()`: overlong prompt 不再抛异常，而是返回 `reward=0, response_tokens=[pad_id], response_masks=[0]` 的 dummy 结果
2. `trajectory_generator()`: 异常处理从 `raise e` 改为 `continue`，单个 trajectory 失败不中断整个 batch
3. `_transform_agent_trajectories()`: 增加 `numel() == 0` 安全检查，跳过空 trajectory

**Changes**:
- `rllm/engine/agent_execution_engine.py`: lines 221-268 (overlong prompt 优雅处理), lines 622-637 (trajectory_generator 容错)
- `rllm/trainer/verl/agent_ppo_trainer.py`: lines 612-627 (empty trajectory 过滤)

---

## Issue 3: Checkpoint 在 Validation 之后保存 [FIXED]

**Symptom**: Step 10 的 validation 崩溃后，没有 checkpoint，前 9 步训练全部丢失，必须从头开始。

**Root Cause**: 原代码顺序为 `validate → save_checkpoint`，如果 validation 崩溃，checkpoint 永远不会保存。

```python
# 修改前 (agent_ppo_trainer.py)
if test_freq > 0 and global_steps % test_freq == 0:
    val_metrics = self._validate_agent()     # ← 崩溃在这里
if save_freq > 0 and global_steps % save_freq == 0:
    self._save_checkpoint()                   # ← 永远执行不到
```

**Fix**: 将 `save_checkpoint` 移到 `validate` 之前。

**Changes**:
- `rllm/trainer/verl/agent_ppo_trainer.py`: 交换 save_checkpoint 和 validate 代码块顺序

---

## Issue 4: Validation 并发限制为 64（应为 500）[FIXED]

**Symptom**: 500 个 SWE-bench 验证样本以 64 并发执行，估计需要 8 轮 × ~1 小时 ≈ 8 小时才能完成一次 validation。

**Root Cause**: `trajectory_generator()` 的并发通过 `asyncio.Semaphore(self.n_parallel_agents)` 控制，`n_parallel_agents = train_batch_size * rollout.n = 8 * 8 = 64`。训练和验证共用同一个限制。

但实际上 `init_envs_and_agents()` 已经为验证创建了所有 500 个 env/agent，只是 semaphore 把并发限死在 64。

**Fix**: 将 semaphore 改为 `len(self.envs)` —— 训练时 envs=64（不变），验证时 envs=500（全部并发）。ThreadPoolExecutor 上限 256 workers。

```python
# 修改前
max_concurrency = self.n_parallel_agents  # always 64

# 修改后
max_concurrency = len(self.envs)  # 64 for train, 500 for val
```

**Changes**:
- `rllm/engine/agent_execution_engine.py`: `trajectory_generator()` 方法中 semaphore 和 ThreadPoolExecutor 的并发数

---

## Issue 5: Metrics Parsing Script Error Count Inflated ~3× [FIXED]

**Symptom**: `parse_training_metrics.py` 的 TRAJECTORY OUTCOMES 表中 Errors 列显示 274/300（steps 8-9），但实际失败 trajectory 只有 137/150。

**Root Cause**: `TRAJ_ERROR_RE` 正则匹配 `Trajectory (\d+).*cannot complete`，但每个失败的 trajectory 在日志中产生 3 行包含 "cannot complete" 的内容：
1. `raise Exception(f"Trajectory {idx} cannot complete...")` — traceback (stderr)
2. `Exception: Trajectory 10 cannot complete...` — exception 消息 (stderr)
3. `Trajectory 10 failed, returning dummy result: Trajectory 10 cannot complete...` — dummy 结果消息 (stdout)

加上 `run_agent_trajectory_with_retry` 的 3 次重试，每次重试也产生自己的 "cannot complete" 行。

**Fix**: 将正则从匹配 "cannot complete" 改为只匹配明确的 "failed, returning dummy result" 行（每个失败 trajectory 只输出一次）。

```python
# 修改前
TRAJ_ERROR_RE = re.compile(
    r"Trajectory (\d+).*cannot complete|Trajectory (\d+):.*exceeded max_prompt_length"
)

# 修改后
TRAJ_DUMMY_RE = re.compile(
    r"Trajectory (\d+) failed, returning dummy result"
)
```

**Changes**:
- `ppio/scripts/parse_training_metrics.py`: `TRAJ_ERROR_RE` → `TRAJ_DUMMY_RE`，正则改为匹配 "failed, returning dummy result"
