# SWE-bench Training with PPIO Sandbox

This example demonstrates how to train an SWE agent on SWE-bench tasks using PPIO sandbox for code execution, with the verl backend for distributed PPO training.

## Key Features

- **PPIO Sandbox**: Uses PPIO cloud sandbox instead of Docker for isolated code execution
- **verl Backend**: Uses verl's distributed PPO training with Ray and vLLM
- **SWE Agent**: Uses R2E-Gym compatible XML function format

## vs Tinker Backend

| Feature | Tinker | verl |
|---------|--------|------|
| Infrastructure | Anthropic cloud | Self-hosted Ray cluster |
| Distributed Training | Tinker API | Ray + FSDP |
| Model Inference | Tinker SamplingClient | vLLM |
| Checkpoints | Cloud storage | Local/HDFS |
| GPU Requirements | None (cloud) | 8+ GPUs |

## Prerequisites

1. **PPIO API Key**:
   ```bash
   export PPIO_API_KEY="your-api-key"
   ```

2. **Ray Cluster**: Either local or distributed
   ```bash
   ray start --head --num-gpus=8
   ```

3. **Required packages**:
   ```bash
   pip install verl vllm ray
   ```

## Usage

### Single Node (8 GPUs)

```bash
cd examples/swe_ppio
./train_swe_ppio.sh
```

### Multi-Node (8 nodes × 8 GPUs)

```bash
NNODES=8 ./train_swe_ppio.sh
```

### Custom Configuration

```bash
MODEL_PATH="Qwen/Qwen3-8B" BATCH_SIZE=4 GROUP_SIZE=4 MAX_STEPS=40 ./train_swe_ppio.sh
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `NNODES` | 1 | Number of nodes |
| `GPUS_PER_NODE` | 8 | GPUs per node |
| `MODEL_PATH` | Qwen/Qwen3-32B | Base model |
| `BATCH_SIZE` | 8 | Training batch size |
| `GROUP_SIZE` | 8 | GRPO group size |
| `MAX_STEPS` | 50 | Max steps per episode |
| `LR` | 1e-6 | Learning rate |

## Data

The training uses:
- **Train**: R2E_Gym_Subset (smaller, faster iteration)
- **Val**: SWE_Bench_Verified (full benchmark)

Prepare data:
```bash
python examples/swe/prepare_swe_data.py
```

## Monitoring

Training logs to WandB:
- Project: `swe-ppio-rl`
- Experiment: `swe-agent-ppio`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    verl PPO Trainer                         │
├─────────────────────────────────────────────────────────────┤
│  Ray Cluster                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Actor (FSDP)│  │ Rollout(vLLM)│  │ Ref Model   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  Agent Execution Engine                                     │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │ SWEAgent    │  │ PPIO Sandbox │                          │
│  │ (R2E-Gym)   │  │ Pool         │                          │
│  └─────────────┘  └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Compared to rft-tinker

This is a migration of the `rft-tinker/tinker_r2e_training_v3.py` training logic to use:
- rllm's agent/environment abstraction
- verl's distributed PPO instead of Tinker
- rllm's checkpoint management

The key patterns preserved:
- Retry with exponential backoff (via verl's infrastructure)
- Checkpoint resumption (via verl's checkpoint system)
- GRPO advantage computation (via RLOO estimator)
