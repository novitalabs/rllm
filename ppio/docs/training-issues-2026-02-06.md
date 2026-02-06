# 训练问题记录 - 2026-02-06

## 实验信息

| 项目 | 值 |
|------|-----|
| 训练脚本 | `ppio/scripts/train_swebench_verified.sh` |
| 开始时间 | 2026-02-06 03:12 |
| 卡住时间 | 2026-02-06 04:12 (Step 4 后) |
| 数据集 | SWE_Bench_Verified.parquet |
| 环境 | `swe_ppio_multistep` |
| 模型 | Qwen3-32B |

## 问题概述

训练在完成 4 个 step 后卡住，Runner 线程崩溃导致整个训练进程死锁。

---

## 问题 1: Prompt 长度超限 (主要原因)

### 错误信息
```
Exception: Trajectory 29: initial prompt length 4345 already exceeded max_prompt_length 4096, retrying
Exception: Trajectory 4: initial prompt length 4340 already exceeded max_prompt_length 4096, retrying
```

### 原因分析
- 配置 `data.max_prompt_length=4096`
- SWE-Bench Verified 部分样本的 `problem_statement` 较长
- Tokenize 后超过 4096 tokens
- 重试多次后抛出异常，导致 Runner 线程崩溃

### 影响
- Trajectory 无法完成
- Runner 线程崩溃
- 训练进程死锁

### 解决方案
```bash
# 方案 1: 增加 max_prompt_length
data.max_prompt_length=8192

# 方案 2: 预处理时过滤超长样本
data.filter_overlong_prompts=True
data.filter_overlong_prompts_workers=32
```

---

## 问题 2: PPIO API 500 错误

### 错误信息
```
2026-02-06 04:06:51 - HTTP Request: POST https://...sandbox.ppio.cn/files?path=/testbed/testapp/__init__.py "HTTP/1.1 500 Internal Server Error"
2026-02-06 04:07:25 - HTTP Request: POST https://...sandbox.ppio.cn/files?path=/testbed/dynamic_db_router/__init__.py "HTTP/1.1 500 Internal Server Error"
2026-02-06 04:08:00 - HTTP Request: POST https://...sandbox.ppio.cn/files?path=/testbed/test_app/__init__.py "HTTP/1.1 500 Internal Server Error"
```

### 原因分析
- PPIO Sandbox 文件写入 API 返回 500 错误
- 可能是 sandbox 实例不稳定或 API 服务端问题
- 多个不同 sandbox 都出现此问题

### 影响
- 文件创建操作失败
- Agent 无法正常创建测试文件
- 可能导致 trajectory 异常结束

### 解决方案
```python
# 在 ppio_reward.py 中增加文件操作重试
def write_file_with_retry(sandbox, path, content, max_retries=3):
    for attempt in range(max_retries):
        try:
            sandbox.files.write(path, content)
            return True
        except Exception as e:
            if "500" in str(e) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
```

---

## 问题 3: PPIO 连接泄漏

### 现象
```bash
$ lsof -p 1518837 | grep "CLOSE_WAIT" | wc -l
85
```

### 原因分析
- 85 个到 PPIO 服务器 (118.25.162.166) 的 TCP 连接处于 CLOSE_WAIT 状态
- 服务端关闭连接后，客户端未正确清理
- 可能是 httpx/aiohttp 连接池未正确管理

### 影响
- 文件描述符泄漏
- 可能导致后续连接失败
- 训练进程资源耗尽

### 解决方案
```python
# 确保使用 context manager 或显式关闭连接
async with httpx.AsyncClient() as client:
    response = await client.post(url, ...)

# 或在 PPIOSandboxManager.cleanup() 中强制关闭连接
```

---

## 问题 4: Runner 线程未处理异常

### 错误信息
```
Exception in thread Thread-10 (runner):
Traceback (most recent call last):
  File "/root/develop/tengwan/rllm/rllm/trainer/verl/agent_ppo_trainer.py", line 758, in runner
    asyncio.run(consume())
  ...
  File "/root/develop/tengwan/rllm/rllm/engine/agent_execution_engine.py", line 503, in run_agent_trajectory_with_retry
    raise Exception(f"Trajectory {idx} cannot complete. Please check the log message")
Exception: Trajectory 29 cannot complete. Please check the log message
```

### 原因分析
- `agent_execution_engine.py:503` 的异常未被上层捕获
- 异常导致整个 runner 线程退出
- 主进程未检测到线程死亡，继续等待

### 影响
- 训练进程死锁
- GPU 资源浪费 (0% 使用率但内存仍占用)

### 解决方案
```python
# 在 trajectory_generator 中增加异常处理
async def trajectory_generator(...):
    try:
        async for item in ...:
            yield item
    except Exception as e:
        logger.error(f"Trajectory generator failed: {e}")
        # 通知主进程
        raise

# 或在 runner 线程中增加 try-catch
def runner():
    try:
        asyncio.run(consume())
    except Exception as e:
        logger.error(f"Runner thread crashed: {e}")
        self._runner_failed = True
```

---

## 训练指标 (Step 1-4)

| Step | 平均步数 | Reward | 解决数 | 耗时 |
|------|---------|--------|--------|------|
| 1 | 12.9 | 0.0 | 0/4 | 335s |
| 2 | 13.3 | 0.0 | 0/4 | 405s |
| 3 | 12.0 | 0.0 | 0/4 | 308s |
| 4 | 17.0 | 0.0 | 0/4 | 475s |

### 观察
- 所有 reward 为 0.0 (模型尚未学会 submit)
- 大部分 trajectory 因 TRUNCATION 结束 (超出 max_response_length)
- 基础设施工作正常 (sandbox 创建、repo 检出成功)

---

## 建议的配置修改

```bash
# train_swebench_verified.sh 修改

# 1. 增加 prompt 长度限制
MAX_PROMPT_LENGTH=8192  # 从 4096 增加

# 2. 增加 response 长度 (可选)
MAX_RESPONSE_LENGTH=32768  # 从 16384 增加

# 3. 减少并行 trajectory 数量 (降低 PPIO 压力)
ROLLOUT_N=2  # 从 4 减少

# 4. 增加 trajectory timeout
rllm.agent.trajectory_timeout=7200  # 从 3600 增加到 2 小时
```

---

## 后续行动

1. [ ] 终止当前卡住的训练进程
2. [ ] 修改 `train_swebench_verified.sh` 配置
3. [ ] 在 `ppio_reward.py` 中增加文件操作重试逻辑
4. [ ] 在 `agent_execution_engine.py` 中改进异常处理
5. [ ] 重新启动训练

---

## 相关日志文件

- Worker 日志: `/tmp/ray/session_latest/logs/worker-ab4b4f97f0e3ac91c1da8960577d66a7984fcd8db642b3ba0c1cf28f-01000000-1518837.out`
- 错误日志: `/tmp/ray/session_latest/logs/worker-ab4b4f97f0e3ac91c1da8960577d66a7984fcd8db642b3ba0c1cf28f-01000000-1518837.err`
- Raylet 日志: `/tmp/ray/session_latest/logs/raylet.out`
