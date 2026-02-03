# DeepSWE 训练问题排查指南

本文档记录在 8x H200 环境上训练 Qwen3-32B 模型时遇到的问题及解决方案。

## 环境信息

- **GPU**: 8x NVIDIA H200 (143GB HBM3 each, 1.15TB total)
- **Python**: 3.12.3
- **CUDA**: 12.9
- **模型**: Qwen/Qwen3-32B
- **框架**: verl 0.6.1 + vLLM + FSDP

---

## 问题 1: vLLM 版本不兼容

### 症状
```
ModuleNotFoundError: No module named 'vllm.lora.models'
```

### 原因
安装的 vllm 0.14.0 与 verl 0.6.1 不兼容。verl 期望 vllm 0.10.x 版本的 API。

### 解决方案
```bash
pip install "vllm==0.10.2"
```

**注意**: 这会同时降级 torch 到 2.8.0。

---

## 问题 2: flash-attn 与 torch 版本不匹配

### 症状
```
ImportError: .../flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZNK3c106SymInt22maybe_as_int_slow_pathEv
```

### 原因
flash-attn 是针对 torch 2.9.1 编译的，但降级 vllm 后 torch 变为 2.8.0，ABI 不兼容。

### 解决方案
重新编译安装 flash-attn：
```bash
pip uninstall flash-attn -y
pip install flash-attn --no-build-isolation --no-cache-dir
```

**关键**: 必须使用 `--no-cache-dir` 强制重新编译，否则会使用缓存的旧 wheel。

---

## 问题 3: flashinfer 版本不匹配

### 症状
```
RuntimeError: flashinfer-cubin version (0.6.1) does not match flashinfer version (0.5.3)
```

或者运行时错误：
```
TypeError: Mismatched number of arguments when calling: `top_k_mask_logits(...)`
```

### 原因
系统预装了 flashinfer-cubin 0.6.1，但 vllm 0.10.2 安装了 flashinfer-python 0.5.3。

### 解决方案
安装匹配的 flashinfer-python 版本：
```bash
pip install flashinfer-python==0.6.1
```

**备选方案**: 设置环境变量绕过版本检查（不推荐，可能导致运行时错误）：
```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

---

## 问题 4: WandB API Key 缺失

### 症状
```
wandb.errors.errors.UsageError: No API key configured. Use `wandb login` to log in.
```

### 解决方案

**方案 A**: 配置 WandB API Key
```bash
wandb login
# 或
export WANDB_API_KEY=your-key
```

**方案 B**: 禁用 WandB（仅使用 console logger）

修改训练脚本中的 logger 配置：
```bash
# 原配置
trainer.logger=['console','wandb']

# 修改为
trainer.logger=['console']
```

---

## 问题 5: PPIO API Key 未配置

### 症状
```
Error: PPIO_API_KEY not set
```

### 解决方案
设置环境变量：
```bash
export PPIO_API_KEY=sk_xxxxx
```

或在 `.env` 文件中配置：
```
PPIO_API_KEY=sk_xxxxx
```

---

## 问题 6: 训练数据不存在

### 症状
```
Training data not found: /root/develop/tengwan/rllm/data/swe/R2E_Gym_Subset.parquet
```

### 解决方案
运行数据准备脚本：
```bash
python3 experiments/swebench_ppio/prepare_r2e_gym_data.py
```

这会从 HuggingFace 下载 R2E-Gym 数据集并保存为 parquet 格式。

---

## 问题 7: PPIO SSL 连接错误

### 症状
```
httpcore.ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol
httpcore.ConnectTimeout: _ssl.c:983: The handshake operation timed out
```

### 原因
HTTP/HTTPS 代理设置干扰了 PPIO SDK 与服务器的直接 SSL 连接。

### 解决方案
在训练脚本中清除代理设置：
```bash
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
```

---

## 问题 8: HuggingFace 连接超时

### 症状
```
MaxRetryError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded
ConnectTimeoutError: Connection to huggingface.co timed out
```

### 原因
清除代理后，无法直接访问 HuggingFace（某些网络环境需要代理）。

### 解决方案
使用离线模式（前提是模型已缓存到本地）：
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

检查模型是否已缓存：
```bash
ls ~/.cache/huggingface/hub/ | grep -i qwen
```

**注意**: 首次运行需要通过代理下载模型，之后可使用离线模式。

---

## 问题 9: PPIO Sandbox 限流/冲突错误

### 症状
```
ERROR: Response 429  (Too Many Requests)
ERROR: Response 409  (Conflict)
ERROR: Response 404  (Not Found)
```

训练日志停止更新，进程 CPU 时间不增长。

### 原因
- **429**: 请求频率过高，被 PPIO 限流
- **409**: Sandbox 操作冲突
- **404**: Sandbox 已被销毁或不存在

### 解决方案
1. 检查 PPIO 连接是否正常
2. 重启训练

```bash
# 测试 PPIO 连接
export PPIO_API_KEY=sk_xxxxx
python3 -c "
from ppio_sandbox.core import Sandbox
sb = Sandbox.create(timeout=60)
print(sb.commands.run('echo Hello').stdout)
sb.kill()
"

# 重启训练
pkill -9 -f train_agent_ppo
bash ppio/scripts/train_qwen3_32b_8h200.sh
```

---

## 问题 10: NCCL 警告

### 症状
```
NCCL WARN Cuda failure 1 'invalid argument'
```

### 说明
这通常是 NCCL NVLS (NVLink SHARP) 传输层的警告，不影响训练。NCCL 会自动 fallback 到其他传输方式。

### 解决方案（可选）
如果警告过多影响日志可读性，可以禁用 NVLS：
```bash
export NCCL_NVLS_ENABLE=0
```

---

## 完整环境配置步骤

```bash
# 1. 设置代理（下载依赖和模型时需要）
export https_proxy=http://127.0.0.1:1083
export http_proxy=http://127.0.0.1:1083

# 2. 安装兼容版本的 vllm
pip install "vllm==0.10.2"

# 3. 重新编译 flash-attn
pip uninstall flash-attn -y
pip install flash-attn --no-build-isolation --no-cache-dir

# 4. 安装匹配的 flashinfer
pip install flashinfer-python==0.6.1

# 5. 安装其他依赖
pip install pylatexenc ppio_sandbox

# 6. 安装 rllm
pip install -e ".[swe]"

# 7. 准备训练数据（需要代理访问 HuggingFace）
python3 experiments/swebench_ppio/prepare_r2e_gym_data.py

# 8. 设置训练环境变量
export PPIO_API_KEY=sk_xxxxx
export WANDB_API_KEY=your-wandb-key  # 可选
export FLASHINFER_DISABLE_VERSION_CHECK=1

# 9. 清除代理并启动训练（脚本内部会设置离线模式）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
bash ppio/scripts/train_qwen3_32b_8h200.sh
```

**重要**: 训练脚本会自动：
- 清除代理设置（PPIO 需要直连）
- 启用 HuggingFace 离线模式（使用本地缓存的模型）

---

## 版本兼容性矩阵

| 组件 | 推荐版本 | 备注 |
|------|---------|------|
| vllm | 0.10.2 | 与 verl 0.6.1 兼容 |
| torch | 2.8.0 | 随 vllm 0.10.2 安装 |
| flash-attn | 2.8.3 | 需针对 torch 2.8.0 编译 |
| flashinfer-python | 0.6.1 | 与预装 flashinfer-cubin 匹配 |
| verl | 0.6.1 | DeepSWE 训练框架 |
| transformers | 4.57.1 | HuggingFace transformers |

---

## 训练监控

```bash
# 查看训练日志
tail -f /root/develop/tengwan/rllm/training.log | grep -E "Epoch|step|loss|reward"

# GPU 使用情况
watch -n 5 nvidia-smi

# Ray Dashboard
http://127.0.0.1:8265

# 检查进程数
ps aux | grep -E "(train_agent|ray::|vLLM)" | grep -v grep | wc -l
```

---

## 问题 11: Sandbox 404 错误（sandbox 被销毁）

### 症状
```
ERROR: Response 404 (Not Found)
```

训练日志停止更新，sandbox pool 中的 sandbox 已被销毁。

### 原因
使用通用 template 时，每个 sandbox 需要克隆仓库和安装依赖，耗时长，容易超时被销毁。

### 解决方案
使用 **预构建模板 (Pre-built Templates)**，每个 SWE-bench 仓库有专门的模板：
- 模板中已预先克隆仓库到 `/testbed`
- 依赖已预先安装
- 只需 `git checkout` 到指定 commit

支持的仓库模板：
| 仓库 | 模板名称 |
|------|---------|
| django/django | swebench-django-django |
| sympy/sympy | swebench-sympy-sympy |
| pytest-dev/pytest | swebench-pytest-dev-pytest |
| matplotlib/matplotlib | swebench-matplotlib-matplotlib |
| scikit-learn/scikit-learn | swebench-scikit-learn-scikit-learn |
| astropy/astropy | swebench-astropy-astropy |
| sphinx-doc/sphinx | swebench-sphinx-doc-sphinx |
| pylint-dev/pylint | swebench-pylint-dev-pylint |
| pallets/flask | swebench-pallets-flask |
| psf/requests | swebench-psf-requests |
| pydata/xarray | swebench-pydata-xarray |
| mwaskom/seaborn | swebench-mwaskom-seaborn |

代码会自动根据仓库名选择对应模板，无需手动配置。

---

## 问题 12: Sandbox Pool 使用错误的 template

### 症状
训练日志显示：
```
[SandboxPool] Created sandbox [_default][5] with template=base
```

应该使用 repo-specific 模板但使用了 base 模板。

### 原因
1. `pool_size` 默认值不一致（配置文件和代码中有 16 和 32 两个版本）
2. `from_dict` 方法中的默认值可能覆盖了实际配置

### 解决方案
统一所有默认值为 32：
```python
# ppio_reward.py
self._pool_size: int = 32  # Pool size per repo (skill recommends 32+)

def configure(self, api_key: str, pool_size: int = 32, ...):

# swe_ppio.py
pool_size: int = 32,
pool_size=info.get("pool_size", 32),
```

---

## 问题 13: 评估脚本代理问题

### 症状
运行评估脚本时，PPIO SDK 连接被拒绝：
```
httpx.ConnectError: [Errno 111] Connection refused
```

### 原因
`rft-tinker-ppio/swebench_ppio_eval.py` 脚本设置了代理：
```python
os.environ.setdefault("HTTP_PROXY", "http://172.17.0.1:1081")
```

PPIO SDK 通过代理连接会失败。

### 解决方案
创建新的评估脚本 `ppio/scripts/run_qwen32b_eval.py`，在脚本开头清除代理：
```python
for key in list(os.environ.keys()):
    if 'proxy' in key.lower():
        del os.environ[key]
```

---

## 问题 14: vLLM 与训练冲突

### 症状
同时运行 vLLM server 和训练进程时，训练被 OOM killer 终止。

### 原因
Qwen3-32B 模型在 8x H200 上使用约 102GB 每卡。同时运行 vLLM server 和训练会超出 GPU 内存。

### 解决方案
1. 评估和训练不能同时进行
2. 评估前停止训练：
```bash
pkill -9 -f train_agent_ppo
```
3. 评估后重启训练：
```bash
bash ppio/scripts/train_qwen3_32b_8h200.sh
```

---

## 评估脚本使用

### 运行 Qwen32B 基线评估

```bash
# 1. 停止训练
pkill -9 -f train_agent_ppo

# 2. 启动 vLLM server (离线模式)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --tensor-parallel-size 8 \
    --port 30000 \
    --dtype bfloat16 \
    --disable-custom-all-reduce

# 3. 运行评估 (新终端)
export PPIO_API_KEY=sk_xxxxx
python ppio/scripts/run_qwen32b_eval.py \
    --data /path/to/data.jsonl \
    --num-eval 10 \
    --output outputs/results.jsonl

# 4. 评估完成后，重启训练
pkill -9 -f vllm
bash ppio/scripts/train_qwen3_32b_8h200.sh
```

---

---

## 问题 15: 预构建模板 git checkout 冲突

### 症状
使用预构建模板时，checkout 到特定 commit 失败：
```
error: Your local changes to the following files would be overwritten by checkout
Aborting
```

### 原因
预构建模板包含最新版本的代码，checkout 到旧 commit 时会有冲突。

### 解决方案
在 checkout 前强制重置：
```python
sandbox.commands.run(f"cd {DEFAULT_WORKDIR} && git reset --hard HEAD", timeout=30)
sandbox.commands.run(f"cd {DEFAULT_WORKDIR} && git clean -fd", timeout=30)
sandbox.commands.run(f"cd {DEFAULT_WORKDIR} && git checkout -f {commit}", timeout=120)
```

---

## 问题 16: r2egym 包未安装导致训练崩溃

### 症状
```
AttributeError: 'NoneType' object has no attribute 'from_string'
```

训练时在 `parse_xml_response` 函数崩溃。

### 原因
`rllm/agents/swe_agent.py` 中的 `SWEAction` 类从 `r2egym` 包导入。当 `r2egym` 未安装时，`SWEAction = None`，导致后续调用 `SWEAction.from_string()` 失败。

### 解决方案
在 `swe_agent.py` 中添加 fallback `SWEAction` 类实现：
```python
try:
    from r2egym.agenthub.action import Action as SWEAction
except ImportError:
    class SWEAction:
        """Fallback Action class compatible with R2E-Gym format."""
        def __init__(self, function_name: str = "", parameters: dict = None):
            self.function_name = function_name
            self.parameters = parameters or {}

        @classmethod
        def from_string(cls, action_str: str) -> "SWEAction":
            # Parse action from XML string
            ...

        def to_xml_string(self) -> str:
            # Convert action to XML string
            ...
```

---

## 问题 17: Sandbox `/tmp` 写入权限被拒绝

### 症状
```
SandboxException: 500: error creating file: open /tmp/patch.diff: permission denied
```

### 原因
PPIO sandbox 中 `/tmp` 目录可能没有写入权限。

### 解决方案
将 patch 文件写入工作目录而非 `/tmp`：
```python
# 使用 base64 编码避免特殊字符问题
patch_file = f"{self.workdir}/patch.diff"
patch_b64 = base64.b64encode(patch.encode()).decode()
self._run_command(f"echo '{patch_b64}' | base64 -d > {patch_file}")
```

---

## 问题 18: WandB API Key 缺失导致训练启动失败

### 症状
```
wandb.errors.errors.UsageError: No API key configured
```

### 解决方案
修改训练脚本，禁用 wandb：
```bash
# 原配置
trainer.logger=['console','wandb']

# 修改为
trainer.logger=['console']
```

---

## 问题 19: Paused Sandbox 超时被销毁

### 症状
```
[SandboxPool] Failed to resume sandbox [_default][X]: Paused sandbox xxx not found
```

### 原因
PPIO paused sandboxes 有超时限制，长时间不活动会被自动销毁。

### 影响
不影响训练，SandboxPool 会自动创建新的 sandbox 替代。

### 解决方案
无需手动处理，系统会自动重建 sandbox。如频繁出现，考虑减少 pool_size 或增加 sandbox 使用频率。

---

## 更新历史

- **2026-02-03**: 添加 paused sandbox 超时问题 (19)
- **2026-02-03**: 添加 sandbox /tmp 写入权限问题 (17)，WandB 问题 (18)
- **2026-02-03**: 添加 r2egym fallback 问题 (16)
- **2026-02-03**: 添加 git checkout 冲突问题 (15)
- **2026-02-03**: 添加评估相关问题 (12-14)，记录代理和内存冲突问题
- **2026-02-03**: 添加预构建模板支持，解决 sandbox 404 问题
- **2026-02-03**: 初始文档，记录 8x H200 环境配置问题
