# PPIO SWE-Bench Reward Pipeline 修复与 V6 实验结论

## 实验概况

| 项目 | 值 |
|------|-----|
| 训练脚本 | `ppio/scripts/train_swebench_verified.sh` |
| 实验周期 | 2026-02-09 ~ 2026-02-10 |
| 数据集 | SWE_Bench_Verified.parquet (500 samples) |
| 环境 | `swe_ppio_multistep` (PPIO Cloud Sandbox) |
| 模型 | Qwen3-32B, 8xH200 |
| 基线目标 | Qwen3-32B 在 SWE-Bench Verified 预期 resolve rate ~23% |

### 实验版本

| 版本 | 时间 | Reward 情况 | 问题 |
|------|------|-------------|------|
| v1-v5 | 02-06 ~ 02-09 | **全部 0.0** | 训练稳定性 + reward 管道均有 bug |
| **v6** | 02-09 ~ 02-10 | **22.7% 非零** | 所有 reward 管道 bug 已修复 |

---

## 背景：为什么 v1-v5 所有 reward 都是 0.0

v1-v5 修复了训练框架的稳定性问题（prompt 超长、线程崩溃、batch size 不匹配等，见 `training-issues-2026-02-06-09.md`），但 reward 计算管道本身存在 **8 个 bug**，导致无论模型生成什么 patch，reward 始终为 0.0。

这 8 个 bug 可分为三类：
1. **Reward 返回通路错误** — compute_final_reward() 返回 0.0（Bug #1）
2. **测试执行错误** — 测试命令构建错误、test_patch 未应用（Bug #4, #6, #7, #8）
3. **结果解析错误** — 只支持 pytest、stdout 丢失（Bug #2, #3, #5）

---

## Bug #1: compute_final_reward() 返回 0.0 (致命)

### 现象
所有训练 trajectory 的 reward 都是 0.0，即使 agent 正确解决了问题。

### 根因
`agent_execution_engine.py` 在 episode 结束后调用 `env.compute_final_reward()` 并 **覆盖** 最后一步的 reward。两个 PPIO 环境的 `compute_final_reward()` 都返回了 0.0：

```python
# swe_ppio.py / swe_ppio_multistep.py (修复前)
def compute_final_reward(self) -> float:
    return 0.0  # ← 所有 reward 被覆盖为 0
```

### 执行引擎的 reward 覆盖逻辑

```python
# agent_execution_engine.py
final_reward = env.compute_final_reward()  # ← 必须返回实际 reward
episode_steps[-1].reward = final_reward    # ← 覆盖最后一步 reward
```

### 修复

```python
# 修复后
def __init__(self, ...):
    self._last_reward: Optional[float] = None

def step(self, action):
    ...  # 计算 reward
    self._last_reward = reward  # 存储 reward
    return observation, reward, done, info

def compute_final_reward(self) -> float:
    if self._last_reward is not None:
        return self._last_reward
    return 0.0
```

对于 multistep 环境，`_run_evaluation()` 在 episode 结束时如果 agent 未主动 submit，也会运行评估并设置 `self._last_reward`。

**涉及文件:** `swe_ppio.py`, `swe_ppio_multistep.py`

---

## Bug #2: parse_pytest_output() 解析格式不对齐 (中等)

### 现象
即使 pytest 输出了正确的测试结果，解析器也找不到任何测试。

### 根因
自定义的 `parse_pytest_output()` 与 R2E-Gym 的 `parse_log_pytest` 有三个差异：
1. 没有解析 "short test summary info" 段落
2. 使用了错误的测试名格式（完整路径 vs 带 `::` 的 dotted 格式）
3. 没有处理 ANSI escape codes

### 修复
重写 `parse_pytest_output()` 对齐 R2E-Gym 的 `parse_log_pytest`：
- 解析 `PASSED`/`FAILED`/`ERROR` 行
- 解析 "short test summary info" 段落
- 用 regex 去除 ANSI escape codes
- 测试名格式使用 `path::class::method`

**涉及文件:** `ppio_reward.py`

---

## Bug #3: SandboxPool.active_count 引用不存在的属性 (低)

### 现象
调用 `SandboxPool.active_count` 时抛出 `AttributeError: 'SandboxPool' has no attribute '_pool'`。

### 根因
代码引用了 `self._pool`，实际属性名为 `self._repo_pools`。

### 修复
```python
# 修复后
@property
def active_count(self) -> int:
    return sum(len(pool) for pool in self._repo_pools.values())
```

**涉及文件:** `ppio_reward.py`

---

## Bug #4: 只运行 FAIL_TO_PASS 测试，不运行 PASS_TO_PASS (中等)

### 现象
回归测试未被执行，模型可能在修 bug 的同时引入新 bug 而不被检测到。

### 根因
原始代码只将 FAIL_TO_PASS 测试名传给测试命令，忽略了 PASS_TO_PASS。

### 修复
同时传入 FAIL_TO_PASS 和 PASS_TO_PASS 测试名。（后续被 Bug #7 的修复取代，改用 `get_test_directives()`。）

**涉及文件:** `swe_ppio_multistep.py`, `ppio_reward.py`

---

## Bug #5: parse_pytest_output() 只支持 pytest 格式 (致命)

### 现象
非 pytest 仓库（django, sympy, matplotlib 等）的测试结果解析为 0 个测试，reward 始终为 0。

### 根因
SWE-bench 包含 23 个不同的仓库，每个仓库有各自的测试输出格式：
- **Django**: unittest 格式 (`test_name ... ok/FAIL`)
- **sympy**: 自定义格式
- **pytest repos**: `PASSED`/`FAILED` 格式

原始代码只实现了 pytest 格式的解析器。

### 修复
使用 swebench 的标准解析系统：

```python
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.grading import get_eval_tests_report, get_resolution_status

# 每个仓库使用对应的解析器
parser = MAP_REPO_TO_PARSER[repo]
test_status_map = parser(test_output, None)  # 23 种 repo-specific parser

# 使用标准评分逻辑
eval_report = get_eval_tests_report(test_status_map, ...)
is_resolved = get_resolution_status(eval_report) == "RESOLVED_FULL"
```

**关键发现:** `MAP_REPO_TO_PARSER` 有 23 个 repo-specific parser，签名为 `(log, test_spec)` 但 `test_spec` 参数并未被使用，可以传 `None`。

**涉及文件:** `ppio_reward.py`

---

## Bug #6: _run_command() 在非零退出码时丢失 stdout (致命)

### 现象
测试命令的输出完全为空，解析器找不到任何测试结果。

### 根因
PPIO SDK 在命令返回非零退出码时抛出 `CommandExitException`，而不是正常返回。原始代码只捕获了错误消息，丢失了 stdout：

```python
# 修复前
try:
    result = sandbox.commands.run(cmd)
    return result.exit_code, result.stdout
except Exception as e:
    return 1, str(e)  # ← stdout 完全丢失！
```

pytest 在有失败测试时返回 exit_code=1，在无测试收集时返回 exit_code=4。这意味着 **绝大多数测试运行的输出都会丢失**。

### 修复

```python
# 修复后
from ppio_sdk.error import CommandExitException

try:
    result = sandbox.commands.run(cmd)
    return result.exit_code, result.stdout
except CommandExitException as e:
    # CommandExitException 继承 CommandResult，有 .stdout, .stderr, .exit_code
    return e.exit_code, e.stdout  # ← 保留完整 stdout
except Exception as e:
    return 1, str(e)
```

**涉及文件:** `ppio_reward.py`（`PPIOSandboxManager._run_command()`）

---

## Bug #7: 测试命令使用 FAIL_TO_PASS 名称而非 test_patch directives (致命)

### 现象
Django 等仓库的测试命令收到无效参数，0 个测试被执行。

### 根因

**PPIO 的错误做法：**
```python
# 将 FAIL_TO_PASS / PASS_TO_PASS 名称直接作为测试命令参数
test_names = fail_to_pass + pass_to_pass
test_cmd = f"python -m pytest {' '.join(test_names)}"
```

**问题：** FAIL_TO_PASS/PASS_TO_PASS 是 **输出格式的测试标识符**，用于解析和评分，**不是** 测试命令参数。

例如 Django 的 PASS_TO_PASS 包含这样的内容：
```
"Check the creation and properties of a superuser."
"A superuser can be created (even if a superuser already exists)"
```
这些是 docstring 描述，不是可执行的测试路径！

**Standard (Docker/R2E-Gym) 的正确做法：**
```python
# 使用 swebench 的 get_test_directives() 从 test_patch 提取测试文件路径
from swebench.harness.test_spec.python import get_test_directives
directives = get_test_directives(instance)
# Django: tests/auth/test_login.py → "auth.test_login" (module notation)
# pytest repos: "tests/test_foo.py" (file path notation)
```

### 修复

新增 `build_eval_script()` 函数，使用 swebench 标准方式构建测试命令：

```python
def build_eval_script(entry: dict) -> tuple[str, str]:
    """Build evaluation script matching Standard's make_eval_script_list_py()."""
    from swebench.harness.test_spec.python import get_test_directives
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

    # 1. 从 test_patch 提取测试文件 directives
    directives = get_test_directives(instance)

    # 2. 从 MAP_REPO_VERSION_TO_SPECS 获取版本特定的测试命令
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    test_cmd = specs["test_cmd"]  # e.g., "./tests/runtests.py --settings=..." for Django

    # 3. 拼接: test_cmd + directives
    full_test_cmd = " ".join([test_cmd] + directives)
    ...
```

**涉及文件:** `swe_ppio_multistep.py`, `swe_ppio.py`, `ppio_reward.py`

---

## Bug #8: test_patch 从未被应用 (致命)

### 现象
FAIL_TO_PASS 测试在代码库中可能根本不存在，因为 `test_patch` 没有被应用。

### 根因

SWE-bench 的评估流程中，`test_patch` 包含新增/修改的测试文件，用于验证 bug 是否被修复。标准流程：

```
1. git checkout <base_commit> <test_files>    # 重置测试文件
2. git apply <test_patch>                      # 应用测试补丁
3. 运行测试命令                                 # 执行测试
4. git checkout <base_commit> <test_files>    # 恢复测试文件
```

PPIO 环境直接运行测试命令，跳过了步骤 1-2 和 4。

### 修复

`build_eval_script()` 生成完整的评估脚本，包含所有四个步骤：

```bash
#!/bin/bash
set -uxo pipefail
# 设置环境变量 (Django 需要 LANG=en_US.UTF-8 等)
export LANG=en_US.UTF-8
cd /testbed
# 重置测试文件到 base_commit
git checkout abc123 tests/test_foo.py tests/test_bar.py
# 应用 test_patch
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_foo.py b/tests/test_foo.py
...
EOF_114329324912
# 标记测试输出开始
: '>>>>> Start Test Output'
# 运行测试
python -m pytest tests/test_foo.py tests/test_bar.py
# 标记测试输出结束
: '>>>>> End Test Output'
# 恢复测试文件
git checkout abc123 tests/test_foo.py tests/test_bar.py
```

脚本通过 base64 编码写入 sandbox，避免 heredoc 中的特殊字符问题。

**涉及文件:** `swe_ppio_multistep.py`, `swe_ppio.py`

---

## 修改文件汇总

| 文件 | 修复的 Bug | 主要改动 |
|------|-----------|----------|
| `swe_ppio_multistep.py` | #1, #4, #7, #8 | 新增 `build_eval_script()`, `_run_eval_script()`; 重写 `_submit()`, `_run_evaluation()` |
| `swe_ppio.py` | #1, #7, #8 | 导入 `build_eval_script`; 重写 `step()` 使用 eval script |
| `ppio_reward.py` | #2, #3, #5, #6 | 重写 `parse_pytest_output()` 使用 swebench parser; 修复 `_run_command()` stdout 丢失 |

删除的旧代码：
- `REPO_TEST_CMDS` 字典（硬编码测试命令）
- `get_test_cmd_for_repo()` 函数
- `convert_test_names_for_django()` 函数

新增依赖（均为 swebench 已有模块）：
- `swebench.harness.test_spec.python.get_test_directives`
- `swebench.harness.test_spec.python.get_modified_files`
- `swebench.harness.constants.MAP_REPO_VERSION_TO_SPECS`
- `swebench.harness.log_parsers.MAP_REPO_TO_PARSER`
- `swebench.harness.grading.get_eval_tests_report`
- `swebench.harness.grading.get_resolution_status`

---

## V6 实验结果

### 总体统计 (截至 Step 18)

| 指标 | 值 |
|------|-----|
| 总 trajectory 数 | ~1800+ |
| 唯一 instance 数 | 436/500 |
| 非零 reward instance 数 | 99 (22.7%) |
| 非零 reward trajectory 比例 | ~11.4% |
| 平均 reward (非零) | ~0.85 |

### 预期对比

| 指标 | 预期值 | 实际值 | 状态 |
|------|--------|--------|------|
| Resolve rate (instance-level) | ~23% | 22.7% | ✅ 匹配 |
| 非零 reward | >0 | 99 instances | ✅ 确认 |

### 按仓库 Resolve 情况

| 仓库 | 解决数 | 说明 |
|------|--------|------|
| django | 59 | 最多，实例数也最多 |
| sympy | 18 | |
| matplotlib | 11 | |
| xarray | 7 | |
| requests | 2 | |
| pytest | 2 | |
| scikit-learn | 0 | C 扩展未编译 (sandbox 缺 make) |
| sphinx | 0 | tox 未安装 |
| astropy | 0 | 模型能力不足 |

### Eval Batch 学习趋势（同一 52 任务集）

| Eval Step | 解决数 | 解决率 | 平均 Reward |
|-----------|--------|--------|-------------|
| Step 5 (Epoch 0) | 6/52 | 11.5% | 0.1026 |
| Step 10 (Epoch 1) | 6/52 | 11.5% | 0.0929 |
| **Step 15 (Epoch 2)** | **8/52** | **15.4%** | **0.1410** |

### 训练 Batch 平均 Reward 趋势

| Epoch | Mean Reward | 非零比例 | 趋势 |
|-------|-------------|----------|------|
| 0 (Steps 1-4) | 0.026 | 4.7% | baseline |
| 1 (Steps 6-9) | 0.058 | 6.2% | +123% |
| 2 (Steps 11-14) | 0.063 | 6.2% | +8.6% |
| 3 (Steps 16-18) | 0.104 | 10.4% | +65% |

**Early (Steps 1-5) vs Late (Steps 15-18): Mean reward 提升 +104%**

---

## 结论

### 问题确认
v1-v5 所有 reward 为 0.0 是由 8 个 reward 管道 bug 共同造成的，其中 5 个是致命级别（任何一个单独存在都会导致全部 reward 为 0）。这些 bug 分布在三个层面：
1. **reward 返回通路** — compute_final_reward() 始终返回 0.0
2. **测试执行** — 测试命令参数错误、test_patch 未应用、PASS_TO_PASS 未执行
3. **结果解析** — 只支持 pytest 格式、stdout 在非零退出码时丢失

### 关键教训

1. **FAIL_TO_PASS/PASS_TO_PASS 是评分标识符，不是测试命令参数**。标准实现使用 `get_test_directives()` 从 `test_patch` 的 diff 中提取测试文件路径。

2. **test_patch 必须在测试前应用**。这是 SWE-bench 评估的核心流程：`test_patch` 包含了验证 bug 修复的测试代码。

3. **PPIO SDK 的 CommandExitException 包含 stdout**。非零退出码 ≠ 命令失败，pytest 返回 exit_code=1 只是表示有测试失败。必须从 exception 对象提取 `.stdout`。

4. **swebench 有 23 种 repo-specific parser**。不能只写一个 pytest parser。标准的 `MAP_REPO_TO_PARSER` + `get_eval_tests_report()` + `get_resolution_status()` 组合可以正确处理所有仓库。

5. **compute_final_reward() 是最终 reward 来源**。执行引擎会用它的返回值覆盖最后一步的 reward，必须返回实际的评估结果。

### RL 学习信号

v6 实验确认 RL 学习信号有效：
- Instance-level resolve rate 22.7%，匹配 Qwen3-32B 预期基线 ~23%
- Eval batch 解决率从 11.5% 提升到 15.4%（+33%）
- 训练 batch 平均 reward 单调递增，从 0.026 到 0.104（+300%）
- Step 15 新解决了 3 个之前从未解决的任务

---

## 已知限制与后续优化

### 环境限制（非代码 bug）

| 问题 | 影响仓库 | 原因 | 解决方案 |
|------|---------|------|---------|
| C 扩展未编译 | scikit-learn | sandbox 模板缺少 `make` 工具 | 更新 PPIO sandbox 模板，预编译 C 扩展 |
| tox 未安装 | sphinx | sandbox 模板缺少 `tox` | 更新 PPIO sandbox 模板，预装 tox |
| 部分仓库测试依赖缺失 | 待排查 | sandbox 模板环境不完整 | 逐仓库排查并更新模板 |

### 训练优化建议

1. **增加 sandbox 模板覆盖** — 补全 scikit-learn、sphinx 等仓库的环境依赖
2. **监控 reward 分布** — 添加 wandb 等 logger 追踪 per-repo reward 曲线
3. **调整 overlong_filter** — 当前导致每个 training step 产生大量 sub-batch，训练效率低
4. **增大 batch size** — 当前 batch_size=4 × rollout_n=4，reward 信号稀疏，增大 batch 有助于减少方差
