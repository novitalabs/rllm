# PPIO DeepSWE 训练环境准备指南

本文档详细说明如何准备 PPIO + R2E-Gym 环境，用于 DeepSWE 训练实验。

## 目录

1. [前提条件](#前提条件)
2. [快速安装](#快速安装)
3. [详细安装步骤](#详细安装步骤)
4. [环境验证](#环境验证)
5. [数据准备](#数据准备)
6. [常见问题](#常见问题)

---

## 前提条件

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | 4x A100/H100 | 8x H200 |
| GPU 显存 | 320GB (4x80GB) | 640GB+ |
| CPU | 32 核 | 64 核+ |
| 内存 | 256GB | 512GB+ |
| 磁盘 | 500GB | 2TB+ SSD |

### 软件要求

- Python 3.10+ (推荐 3.10 或 3.12)
- CUDA 12.x
- Git
- Conda (推荐) 或 venv

### 账号/密钥

- **PPIO API Key**: 用于 PPIO Sandbox 访问
- **Hugging Face Token** (可选): 用于下载模型和数据集
- **WandB API Key** (可选): 用于训练监控

---

## 快速安装

```bash
# 一键安装（推荐）
cd /path/to/rllm
bash ppio/scripts/setup_env.sh

# 验证安装
export PPIO_API_KEY=sk_xxxxx
bash ppio/scripts/check_env.sh
```

---

## 详细安装步骤

### 步骤 1: 创建 Python 环境

```bash
# 使用 Conda（推荐）
conda create -n rllm python=3.10 -y
conda activate rllm

# 或使用 venv
python3 -m venv .venv
source .venv/bin/activate
```

### 步骤 2: 设置网络代理（如需要）

某些网络环境需要代理访问 GitHub 和 HuggingFace：

```bash
export https_proxy=http://127.0.0.1:1083
export http_proxy=http://127.0.0.1:1083
```

### 步骤 3: 安装 rllm 核心依赖

```bash
cd /path/to/rllm

# 安装 verl (RL 训练框架)
pip install -e ./verl
pip install -e "./verl[vllm]"

# 安装 rllm
pip install -e ".[swe]"
```

### 步骤 4: 安装 R2E-Gym

R2E-Gym 是 SWE-Bench 训练环境的核心组件，提供高质量的代码修复环境。

```bash
# 方法 1: 从 GitHub 安装（推荐）
git clone https://github.com/agentica-project/R2E-Gym.git /tmp/R2E-Gym
cd /tmp/R2E-Gym
pip install -e .
cd -

# 方法 2: 直接从 GitHub 安装
pip install git+https://github.com/agentica-project/R2E-Gym.git

# 验证安装
python3 -c "import r2egym; print(f'r2egym version: {r2egym.__version__}')"
```

### 步骤 5: 安装 PPIO Sandbox SDK

```bash
pip install ppio_sandbox

# 验证安装
python3 -c "from ppio_sandbox.core import Sandbox; print('ppio_sandbox OK')"
```

### 步骤 6: 安装兼容版本的关键依赖

由于版本兼容性问题，需要安装特定版本：

```bash
# 安装兼容的 vllm 版本
pip install "vllm==0.10.2"

# 重新编译 flash-attn（适配降级后的 torch）
pip uninstall flash-attn -y
pip install flash-attn --no-build-isolation --no-cache-dir

# 安装匹配的 flashinfer
pip install flashinfer-python==0.6.1

# 安装其他依赖
pip install pylatexenc pandas datasets
```

### 步骤 7: 配置 PPIO API Key

```bash
# 方法 1: 环境变量
export PPIO_API_KEY=sk_xxxxx

# 方法 2: .env 文件
echo "PPIO_API_KEY=sk_xxxxx" > ppio/scripts/.env

# 方法 3: 添加到 ~/.bashrc
echo 'export PPIO_API_KEY=sk_xxxxx' >> ~/.bashrc
source ~/.bashrc
```

---

## 环境验证

### 快速验证

```bash
bash ppio/scripts/check_env.sh
```

### 手动验证各组件

```bash
# 1. 验证 Python 环境
python3 --version

# 2. 验证核心包
python3 -c "
import torch
import vllm
import rllm
import transformers
import ppio_sandbox

print(f'torch: {torch.__version__}')
print(f'vllm: {vllm.__version__}')
print(f'rllm: OK')
print(f'transformers: {transformers.__version__}')
print(f'ppio_sandbox: OK')
"

# 3. 验证 r2egym
python3 -c "
try:
    from r2egym.agenthub.action import Action
    print('r2egym: OK (full)')
except ImportError:
    print('r2egym: NOT INSTALLED')
"

# 4. 验证 PPIO 连接
python3 -c "
import os
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

from ppio_sandbox.core import Sandbox
sb = Sandbox.create(timeout=60)
result = sb.commands.run('echo Hello PPIO')
print(f'PPIO Sandbox: {result.stdout.strip()}')
sb.kill()
"

# 5. 验证 GPU
nvidia-smi --query-gpu=name,memory.total --format=csv
```

### 验证多步环境

```bash
python3 -c "
from rllm.environments.swe_ppio.swe_ppio_multistep import SWEBenchPPIOMultiStepEnv
print('SWEBenchPPIOMultiStepEnv: OK')
"
```

---

## 数据准备

### 下载训练数据

```bash
# 设置代理（如需要访问 HuggingFace）
export https_proxy=http://127.0.0.1:1083

# 运行数据准备脚本
python3 examples/swe/prepare_swe_data.py

# 验证数据
ls -la data/swe/
# 应该看到:
# - R2E_Gym_Subset.parquet (训练数据)
# - SWE_Bench_Verified.parquet (验证数据)
```

### 检查数据内容

```bash
python3 -c "
import pandas as pd

train = pd.read_parquet('data/swe/R2E_Gym_Subset.parquet')
val = pd.read_parquet('data/swe/SWE_Bench_Verified.parquet')

print(f'Training samples: {len(train)}')
print(f'Validation samples: {len(val)}')
print(f'Columns: {list(train.columns)}')
"
```

### 下载模型（可选，离线训练需要）

```bash
# 使用代理下载
export https_proxy=http://127.0.0.1:1083

python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'Qwen/Qwen3-32B'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f'Model {model_name} cached successfully')
"

# 检查缓存
ls ~/.cache/huggingface/hub/ | grep -i qwen
```

---

## 版本兼容性矩阵

| 组件 | 推荐版本 | 备注 |
|------|---------|------|
| Python | 3.10.x | 3.12 也支持 |
| torch | 2.8.0 | 随 vllm 0.10.2 安装 |
| vllm | 0.10.2 | 与 verl 0.6.1 兼容 |
| flash-attn | 2.8.x | 需针对当前 torch 编译 |
| flashinfer-python | 0.6.1 | 与预装 cubin 匹配 |
| transformers | 4.50+ | |
| r2egym | latest | 从 GitHub 安装 |
| ppio_sandbox | latest | |
| verl | 0.6.1 | |

---

## 常见问题

### Q: r2egym 安装失败

```bash
# 错误: ModuleNotFoundError: No module named 'r2egym'

# 解决: 从 GitHub 安装
pip install git+https://github.com/agentica-project/R2E-Gym.git

# 如果网络问题，使用代理
https_proxy=http://127.0.0.1:1083 pip install git+https://github.com/agentica-project/R2E-Gym.git
```

### Q: PPIO 连接超时

```bash
# 错误: httpcore.ConnectTimeout

# 解决 1: 清除代理设置
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 解决 2: 检查 API Key
echo $PPIO_API_KEY

# 解决 3: 测试连接
python3 -c "
import os
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        del os.environ[k]
from ppio_sandbox.core import Sandbox
sb = Sandbox.create(timeout=120)
print(sb.commands.run('whoami').stdout)
sb.kill()
"
```

### Q: vllm 版本不兼容

```bash
# 错误: ModuleNotFoundError: No module named 'vllm.lora.models'

# 解决: 安装兼容版本
pip install "vllm==0.10.2"
```

### Q: flash-attn 符号错误

```bash
# 错误: undefined symbol: _ZNK3c106SymInt22maybe_as_int_slow_pathEv

# 解决: 重新编译
pip uninstall flash-attn -y
pip install flash-attn --no-build-isolation --no-cache-dir
```

### Q: 训练数据不存在

```bash
# 错误: Training data not found

# 解决: 运行数据准备脚本
python3 examples/swe/prepare_swe_data.py
```

---

## 下一步

环境准备完成后，可以开始训练：

```bash
# 1. 验证环境
bash ppio/scripts/check_env.sh

# 2. 设置 PPIO API Key
export PPIO_API_KEY=sk_xxxxx

# 3. 开始训练
bash ppio/scripts/train_qwen3_32b_8h200.sh
```

详见 [quickstart-8h200.md](./quickstart-8h200.md) 和 [troubleshooting.md](./troubleshooting.md)。
