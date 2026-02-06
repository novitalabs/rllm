# DeepSWE Training Scripts for 8x H200

This directory contains scripts for training and evaluating DeepSWE on 8x H200 GPUs with PPIO sandbox backend.

## Scripts

| Script | Description |
|--------|-------------|
| `setup_h200_env.sh` | **One-click H200 environment setup** (recommended) |
| `setup_env.sh` | Generic environment setup script |
| `check_env.sh` | Verify environment setup before training |
| `train_qwen3_32b_8h200.sh` | Main training script for Qwen3-32B |
| `eval_qwen3_32b.sh` | Evaluation script using vLLM server |

## Quick Start

### 0. Setup Environment (First Time)

```bash
# Set proxy if needed
export https_proxy=http://127.0.0.1:1083

# One-click install all dependencies
bash ppio/scripts/setup_env.sh
```

This installs:
- verl (RL training framework)
- rllm (main library)
- **r2egym** (SWE-bench environment)
- ppio_sandbox (PPIO SDK)
- Compatible versions of vllm, flash-attn, flashinfer

### 1. Check Environment

```bash
bash ppio/scripts/check_env.sh
```

This verifies:
- Python dependencies (torch, vllm, rllm, transformers)
- GPU availability (expects 8x H200)
- PPIO API key configuration
- Training data existence
- Model access

### 2. Set PPIO API Key

```bash
export PPIO_API_KEY="sk-your-api-key"

# Or create .env file
echo "PPIO_API_KEY=sk-your-api-key" > ppio/scripts/.env
```

### 3. Prepare Data (if needed)

```bash
python3 examples/swe/prepare_swe_data.py
```

### 4. Start Training

```bash
bash ppio/scripts/train_qwen3_32b_8h200.sh
```

### 5. Evaluate (after training)

```bash
# Set model path to checkpoint
export MODEL="/path/to/checkpoint"
bash ppio/scripts/eval_qwen3_32b.sh
```

## Hardware Configuration

| Parameter | Value |
|-----------|-------|
| GPUs | 8x NVIDIA H200 (80GB HBM3 each) |
| Total VRAM | 640 GB |
| Tensor Parallel | 8 |
| Sequence Parallel | 8 |

## Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen/Qwen3-32B | 32B dense model |
| Batch Size | 8 | Same as original DeepSWE |
| Rollouts per sample | 8 | |
| Max prompt length | 4096 | |
| Max response length | 32768 | Long context for SWE tasks |
| Learning rate | 1e-6 | |
| Algorithm | GRPO++ with RLOO | |
| Environment | swe_ppio | PPIO sandbox backend |

## Expected Results

| Stage | Pass@1 |
|-------|--------|
| Baseline (pre-RL) | ~23-30% |
| After 200 steps | ~40-42% |

## Monitoring

Training logs will show:
- `batch/solve_rate`: Current solve rate
- `actor/pg_loss`: Policy gradient loss
- `response_length_clip_ratio`: Truncation rate (should decrease)

Use wandb for detailed monitoring:
```bash
export WANDB_API_KEY="your-key"
# Logs will appear in project: deepswe-8h200
```

## Troubleshooting

### OOM Errors
- Reduce `gpu_memory_utilization` from 0.7 to 0.6
- Enable more aggressive offloading

### PPIO 429 Rate Limit
- The sandbox pool handles this automatically
- Check `pool_size` setting if issues persist

### Slow Training
- Check network connectivity to PPIO
- Verify GPU utilization with `nvidia-smi`
