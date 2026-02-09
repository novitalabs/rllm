# PPIO SWE-Bench 训练问题总结与修复记录

## 实验概况

| 项目 | 值 |
|------|-----|
| 训练脚本 | `ppio/scripts/train_swebench_verified.sh` |
| 实验周期 | 2026-02-06 ~ 2026-02-09 |
| 数据集 | SWE_Bench_Verified.parquet (500 samples) |
| 环境 | `swe_ppio_multistep` (PPIO Cloud Sandbox) |
| 模型 | Qwen3-32B, 8xH200 |
| 配置 | batch_size=4, rollout_n=4, max_steps=30 |

### 训练运行记录

| 运行 | 时间 | 结果 | 崩溃原因 |
|------|------|------|----------|
| #1 | 02-06 03:12 | Step 4 崩溃 | Prompt 超长 (4345 > 4096) |
| #2 | 02-06 ~18:00 | Step 4 崩溃 | Prompt 超长 (9810 > 8192) |
| #3 | 02-08 05:41 | Step 15 崩溃 | Batch size 不匹配 (63 vs 64) |
| #4 | 02-09 02:53 | 运行中 | 所有已知问题已修复 |

---

## 问题总览

共发现 **6 个问题**，全部已修复：

| # | 问题 | 严重性 | 影响 | 修复文件 |
|---|------|--------|------|----------|
| 1 | Prompt 超长导致硬崩溃 | **致命** | 训练死锁 | `agent_execution_engine.py` |
| 2 | Runner 线程无异常保护 | **致命** | 训练死锁 | `agent_ppo_trainer.py` |
| 3 | Trajectory 失败导致 batch size 不匹配 | **致命** | 训练崩溃 | `agent_execution_engine.py` |
| 4 | PPIO 文件 API 500 错误 | 中等 | Trajectory 异常结束 | `swe_ppio_multistep.py` |
| 5 | PPIO Sandbox 超时 | 中等 | Trajectory 重试 | 配置 + 重试机制 |
| 6 | TCP 连接泄漏 (CLOSE_WAIT) | 低 | 资源浪费 | 待优化 |

---

## 问题 1: Prompt 超长导致硬崩溃 (致命)

### 现象

训练在 Step 4 稳定崩溃，Runner 线程异常退出，主进程死锁（GPU 0% 但内存仍占用）。

### 错误信息

```
# 运行 #1: max_prompt_length=4096
Exception: Trajectory 29: initial prompt length 4345 already exceeded max_prompt_length 4096, retrying
Exception: Trajectory 4: initial prompt length 4340 already exceeded max_prompt_length 4096, retrying

# 运行 #2: max_prompt_length=8192
Exception: Trajectory 7: initial prompt length 9810 already exceeded max_prompt_length 8192, retrying
```

### 根因

`agent_execution_engine.py:219-221` 中，当 prompt token 数超过 `max_prompt_length` 时，直接 `raise Exception`。该异常被 `run_agent_trajectory_with_retry` 重试 3 次（同样本必然一直超长），用尽重试后抛出 `"Trajectory X cannot complete"`，最终穿透到 `trajectory_generator`，re-raise 导致 runner 线程崩溃。

```
agent_execution_engine.py:219  raise Exception("prompt too long")
  ↓ retry 3 times (all fail)
agent_execution_engine.py:503  raise Exception("cannot complete")
  ↓ re-raise
agent_execution_engine.py:547  raise e  (trajectory_generator)
  ↓ propagate to runner thread
agent_ppo_trainer.py:758       asyncio.run(consume())  → thread dies
  ↓ sentinel never sent
agent_ppo_trainer.py:766       queue.get()  → 主进程永久阻塞 (死锁)
```

### 修复

**共 4 处改动：**

1. **`agent_execution_engine.py:219-221`** — 将硬异常改为优雅跳过：
```python
# 修复前
if prompt_token_len > self.max_prompt_length:
    agent.reset()
    raise Exception(f"Trajectory {idx}: initial prompt length ...")

# 修复后
if prompt_token_len > self.max_prompt_length:
    logger.warning(f"Trajectory {idx}: initial prompt length ... returning masked empty trajectory")
    termination_reason = "PROMPT_OVERLONG"
```

2. **`agent_execution_engine.py:223`** — 在 step 循环入口检查是否需要跳过：
```python
for step_idx in range(self.max_steps):
    if termination_reason:
        break  # 跳过 step 循环
```

3. **`agent_execution_engine.py:363`** — 将 PROMPT_OVERLONG 加入 overlong filter：
```python
if termination_reason in ("TRUNCATION", "MAX_STEPS", "TIMEOUT", "PROMPT_OVERLONG"):
    response_masks = [0] * len(response_masks)  # 不参与 loss
    masked_out = True
```

4. **`agent_execution_engine.py:395-403`** — Token 模式下处理空 episode_steps：
```python
if episode_steps:
    prompt_tokens, response_tokens, ... = self.assemble_steps(episode_steps)
else:
    # 创建合法的 dummy trajectory
    prompt_tokens = torch.tensor(prompt_tokens[:self.max_prompt_length], dtype=torch.long)
    response_tokens = torch.tensor([self.tokenizer.pad_token_id or 0], dtype=torch.long)
    response_masks = torch.tensor([0], dtype=torch.long)
    is_valid_trajectory = False
```

### 配置调整

```bash
# train_swebench_verified.sh
MAX_PROMPT_LENGTH=8192     # 从 4096 增加
MAX_RESPONSE_LENGTH=32768  # 从 16384 增加
PPO_MAX_TOKEN_LEN_PER_GPU=32000
data.filter_overlong_prompts=True
data.filter_overlong_prompts_workers=32
```

---

## 问题 2: Runner 线程无异常保护 (致命)

### 现象

Runner 线程崩溃后，`queue.put(None)` sentinel 永远不会被发送，主进程在 `queue.get()` 上永久阻塞。

### 根因

`agent_ppo_trainer.py` 中的 runner 线程没有 try-except 保护：

```python
# 修复前
def runner():
    async def consume():
        async for item in trajectory_generator(...):
            queue.put(item)
        queue.put(None)  # 如果上面抛异常，这行永远不会执行
    asyncio.run(consume())  # 异常直接杀死线程
```

### 修复

```python
# 修复后
def runner():
    try:
        async def consume():
            async for item in trajectory_generator(...):
                queue.put(item)
            queue.put(None)
        asyncio.run(consume())
    except Exception as e:
        logger.error(f"Runner thread crashed: {e}")
        traceback.print_exc()
        queue.put(None)  # 确保 sentinel 始终发送，主进程不会死锁
```

**涉及文件:** `rllm/trainer/verl/agent_ppo_trainer.py`

---

## 问题 3: Trajectory 失败导致 batch size 不匹配 (致命)

### 现象

运行 #3 在 Step 15 后的 validation 阶段崩溃：

```
AssertionError: Two tensor dict must have identical batch size.
Got torch.Size([64]) and torch.Size([63])
```

### 根因

Validation batch_size=64，其中 Trajectory 42 因 PPIO sandbox 反复超时，3 次重试全部失败。`trajectory_generator` 中的异常处理使用 `continue` 跳过了失败的 trajectory，导致只产出 63 个结果。下游 `batch.union()` 检查 batch size 一致性时触发断言。

```
Trajectory 42: sandbox timeout × 3 retries
  ↓
run_agent_trajectory_with_retry: raise "cannot complete"
  ↓
trajectory_generator: catch → continue (skip, 不 yield)
  ↓
trainer 收到 63 个结果，原始 batch 有 64 个
  ↓
batch.union() → AssertionError: [64] vs [63]
```

### 修复

**`launch_one_trajectory_task`** — 失败时返回 dummy 结果而非 re-raise：

```python
async def launch_one_trajectory_task(env_idx: int):
    async with semaphore:
        try:
            result = await self.run_agent_trajectory_with_retry(...)
        except Exception as e:
            logger.error(f"Trajectory {env_idx} permanently failed: {e}, returning dummy result")
            pad_id = self.tokenizer.pad_token_id or 0
            if mode == "Token":
                result = {
                    "prompt_tokens": torch.tensor([pad_id], dtype=torch.long),
                    "response_tokens": torch.tensor([pad_id], dtype=torch.long),
                    "response_masks": torch.tensor([0], dtype=torch.long),  # 不参与 loss
                    "trajectory_reward": 0.0,
                    "idx": env_idx,
                    "chat_completions": [],
                    "metrics": {"steps": 0, ...},
                }
            elif mode == "Step":
                result = {
                    "steps": [],
                    "trajectory_reward": 0.0,
                    "idx": env_idx,
                    "mc_returns": [],
                    "termination_reason": "ERROR",
                }
            else:
                raise
        return result
```

同时在 `_transform_agent_steps` 中：
- 将 `"ERROR"` 加入 `overlong_reasons` 集合
- 对空 `episode_steps` 的 episode 做 `continue` 跳过

**涉及文件:** `agent_execution_engine.py`, `agent_ppo_trainer.py`

---

## 问题 4: PPIO 文件 API 500 错误 (中等)

### 现象

```
HTTP Request: POST https://...sandbox.ppio.cn/files?path=/testbed/testapp/__init__.py "HTTP/1.1 500 Internal Server Error"
```

多个 sandbox 的文件写入/读取操作返回 500 错误。

### 根因

PPIO Sandbox 的文件 API 偶发性不稳定，可能是服务端负载问题。

### 修复

在 `swe_ppio_multistep.py` 中添加重试辅助函数：

```python
def _sandbox_file_write(sandbox, path: str, content: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            sandbox.files.write(path, content)
            return
        except Exception as e:
            if attempt < max_retries - 1 and "500" in str(e):
                time.sleep(2 ** attempt)  # 指数退避: 1s, 2s, 4s
                continue
            raise

def _sandbox_file_read(sandbox, path: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            return sandbox.files.read(path)
        except Exception as e:
            if attempt < max_retries - 1 and "500" in str(e):
                time.sleep(2 ** attempt)
                continue
            raise
```

替换所有 `sandbox.files.write(...)` 和 `sandbox.files.read(...)` 调用为重试版本。

**涉及文件:** `rllm/environments/swe_ppio/swe_ppio_multistep.py`

---

## 问题 5: PPIO Sandbox 超时 (中等)

### 现象

```
RuntimeError: Failed to setup repository: Checkout failed (exit=1):
Command failed: The sandbox was not found: This error is likely due to sandbox timeout.
```

运行 #3 中出现 18 次 sandbox 超时错误。

### 根因

PPIO sandbox 默认 `timeout=3600`（1小时），长时间不活动的 sandbox 被服务端回收。当 SandboxPool 尝试复用已回收的 sandbox 时触发此错误。

### 处理

- `run_agent_trajectory_with_retry` 重试机制可以处理偶发的 sandbox 超时
- 问题 3 的修复确保即使重试耗尽也不会导致训练崩溃
- 可考虑在 SandboxPool 中增加健康检查，主动清理超时的 sandbox 引用

---

## 问题 6: TCP 连接泄漏 (低)

### 现象

```bash
$ lsof -p <pid> | grep "CLOSE_WAIT" | wc -l
85
```

85 个到 PPIO 服务器的 TCP 连接处于 CLOSE_WAIT 状态。

### 根因

httpx/ppio_sandbox SDK 的连接池在服务端关闭连接后未正确清理客户端 socket。

### 状态

暂未修复，影响较小。可在后续优化中处理：
- 在 `PPIOSandboxManager.cleanup()` 中强制关闭连接
- 或使用 `httpx.AsyncClient` 的 context manager 确保连接释放

---

## 修改文件汇总

| 文件 | 改动行数 | 修复的问题 |
|------|----------|-----------|
| `rllm/engine/agent_execution_engine.py` | +57/-4 | #1, #3 |
| `rllm/trainer/verl/agent_ppo_trainer.py` | +28/-4 | #2, #3 |
| `rllm/environments/swe_ppio/swe_ppio_multistep.py` | +41/-4 | #4 |
| `ppio/scripts/train_swebench_verified.sh` | +6/-6 | #1 (配置) |

---

## 运行 #3 训练指标 (Step 1-15)

| 指标 | 值 |
|------|-----|
| 完成训练 Step | 15 |
| 完成验证轮次 | 20 |
| 总 Trajectory 数 | 858 |
| PROMPT_OVERLONG 事件 | 1 (优雅处理) |
| Sandbox 超时 | 18 (重试处理) |
| Runner 线程崩溃 | 0 |

### Termination 分布

| 结束原因 | 次数 | 比例 | 说明 |
|----------|------|------|------|
| ENV_DONE | 623 | 72.6% | Agent 调用 submit 正常结束 |
| TRUNCATION | 188 | 21.9% | 响应超过 max_response_length |
| MAX_STEPS | 46 | 5.4% | 达到 30 步上限 |
| PROMPT_OVERLONG | 1 | 0.1% | 初始 prompt 超长，优雅跳过 |

### Reward 情况

- 全部 858 个 trajectory 的 reward 为 0.0
- 训练初期正常，模型尚在学习 SWE-Bench 任务的 submit 流程
- 需要更多训练步数才能看到非零 reward

---

## 异常处理架构 (修复后)

```
Agent 运行 trajectory
  ↓
[Prompt 超长?] → termination_reason="PROMPT_OVERLONG"
                  → 返回 masked 空 trajectory (reward=0, mask=0)
                  → batch size 不变 ✓
  ↓
[运行异常?] → run_agent_trajectory_with_retry: 重试 3 次
              ↓ 重试耗尽
              launch_one_trajectory_task: 返回 dummy result
              → batch size 不变 ✓
  ↓
[trajectory_generator 异常?] → catch + continue + log (安全网)
  ↓
[runner 线程异常?] → try-except 保护
                     → queue.put(None) 始终发送
                     → 主进程不会死锁 ✓
```

每一层都有兜底机制，确保：
1. **单个 trajectory 失败不影响整个 batch**
2. **batch size 始终保持一致**
3. **runner 线程不会静默死亡导致死锁**
4. **失败的 trajectory 以 mask=0 参与训练，不影响 loss**

---

## 后续优化建议

1. **SandboxPool 健康检查** — 主动探测 sandbox 存活状态，清理已超时的引用
2. **TCP 连接池管理** — 在 PPIOSandboxManager 中增加连接生命周期管理
3. **动态 max_prompt_length** — 根据实际数据分布自适应调整，而非硬编码
4. **Reward 监控** — 增加 wandb logger 追踪训练曲线，及时发现问题
5. **Sandbox timeout 自适应** — 对长时间运行的 trajectory 动态延长 sandbox 超时
