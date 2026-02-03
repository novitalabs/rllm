# 8x H200 快速开始指南

## 前提条件

- 8x NVIDIA H200 GPU
- Python 3.12+
- CUDA 12.x
- PPIO API Key

## 一键环境配置

```bash
# 设置代理（如需要）
export https_proxy=http://127.0.0.1:1083
export http_proxy=http://127.0.0.1:1083

# 安装兼容版本
pip install "vllm==0.10.2"
pip uninstall flash-attn -y && pip install flash-attn --no-build-isolation --no-cache-dir
pip install flashinfer-python==0.6.1 pylatexenc ppio_sandbox

# 安装 rllm
cd /path/to/rllm
pip install -e ".[swe]"

# 准备数据
python3 experiments/swebench_ppio/prepare_r2e_gym_data.py
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
