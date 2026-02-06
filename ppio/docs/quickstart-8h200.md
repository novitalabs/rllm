# 8x H200 快速开始指南

## 前提条件

- 8x NVIDIA H200 GPU
- Python 3.10+ (推荐 3.10 或 3.12)
- CUDA 12.x
- PPIO API Key

## 一键环境配置

```bash
cd /path/to/rllm

# 一键安装所有依赖（推荐，使用代理）
https_proxy=http://127.0.0.1:1083 bash ppio/scripts/setup_h200_env.sh

# 或不使用代理
bash ppio/scripts/setup_h200_env.sh

# 准备数据
python3 examples/swe/prepare_swe_data.py
```

脚本会自动安装：
- verl (RL 训练框架)
- rllm (主库)
- vllm 0.10.2 (与 verl 兼容)
- flash-attn (针对当前 torch 编译)
- flashinfer 0.6.1
- ppio_sandbox (PPIO SDK)
- r2egym (SWE-bench 环境)

### 手动安装（可选）

```bash
# 安装 verl 和 rllm
pip install -e ./verl
pip install -e "./verl[vllm]"
pip install -e ".[swe]"

# 安装 R2E-Gym（重要！）
pip install git+https://github.com/agentica-project/R2E-Gym.git

# 安装兼容版本
pip install "vllm==0.10.2"
pip uninstall flash-attn -y && pip install flash-attn --no-build-isolation --no-cache-dir
pip install flashinfer-python==0.6.1 pylatexenc ppio_sandbox
```

## 环境验证

```bash
export PPIO_API_KEY=sk_xxxxx
bash ppio/scripts/check_env.sh
```

预期输出：
```
All checks passed! Ready to train.
```

## 启动训练

```bash
export PPIO_API_KEY=sk_xxxxx
bash ppio/scripts/train_qwen3_32b_8h200.sh
```

## 训练配置摘要

| 参数 | 值 |
|------|-----|
| 模型 | Qwen/Qwen3-32B |
| GPUs | 8x H200 |
| Tensor Parallel | 8 |
| Sequence Parallel | 8 |
| Batch Size | 8 |
| Max Response Length | 32768 |
| Total Epochs | 200 |
| Total Steps | 114,400 |

## 常见问题

详见 [troubleshooting.md](./troubleshooting.md)
