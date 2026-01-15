# DeepSWE: Training a Fully Open-sourced, State-of-the-Art Coding Agent by Scaling RL

> Notes from: https://www.together.ai/blog/deepswe
> Published: July 2, 2025
> Authors: Michael Luo, Naman Jain, Jaskirat Singh, Sijun Tan, Ameen Patel, et al. (Agentica + Together AI)

## Overview

DeepSWE is a coding agent trained with reinforcement learning that achieves state-of-the-art performance on SWE-Bench. Key achievement: **42.2% Pass@1** on SWE-Bench-Verified, trained purely with RL (no SFT/distillation from proprietary models).

## Technical Details

### Base Model & Training

| Component | Details |
|-----------|---------|
| Base Model | Qwen3-32B |
| Framework | rLLM (Agentica's post-training system) |
| Dataset | 4,500 problems from R2E-Gym subset |
| Infrastructure | 64 H100 GPUs, 6 days |
| Algorithm | GRPO++ (enhanced with DAPO, Dr. GRPO, LOOP/RLOO improvements) |

### Algorithm: GRPO++

Key enhancements over standard GRPO:
- **Compact filtering**: Extends overlong filtering to multi-turn scenarios
- **No entropy loss**: More stable training
- **Length normalization**: Removes bias toward longer incorrect responses
- **Sparse outcome reward**: 1 for passing tests, 0 otherwise

### Environment Tools

- Bash execution
- Search
- File editor
- Finish/submit commands
- Kubernetes integration for scalable Docker container management

## Results

### SWE-Bench-Verified

| Metric | Score |
|--------|-------|
| Pass@1 | 42.2% (avg over 16 runs) |
| Pass@16 | 71.0% |
| With Hybrid TTS | 59% (beats previous SOTA open-weights by 12%) |

### Training Progress

- Validation score improved from 23% → 42% (+20%) in 200 RL steps
- Outperforms models using SFT/distillation from proprietary teachers

## Key Findings

### Emergent Behaviors

1. **Edge case consideration**: Model learns to consider edge cases and regression tests
2. **Adaptive token allocation**:
   - Complex reasoning: ~2K tokens
   - Simple operations: 100-200 tokens
3. **Testing habits**: Develops rigorous testing before final submission

### Test-Time Scaling (TTS)

- Hybrid approach (execution-based + execution-free verifiers) most effective
- Performance scales better with number of rollouts than context length
- Optimal gains at K=8 rollouts for practical scenarios

## Relevance to Our Work

This validates our approach with rllm + PPIO:

1. **RL-only training works**: No need for SFT/distillation
2. **Sparse rewards sufficient**: Simple pass/fail works for code tasks
3. **Hardware requirements**: 64 H100 GPUs for 32B model
   - Our RTX 4090 setup needs smaller models (0.6B-4B range)
4. **GRPO++ improvements**: Consider implementing compact filtering, length normalization
5. **R2E-Gym dataset**: 4,500 problems - same dataset we're exploring

## Open Source Release

All components publicly available:
- Dataset
- Training code
- Evaluation logs
- Model weights

## References

- Blog: https://www.together.ai/blog/deepswe
- rLLM Framework: https://github.com/agentica-project/rllm
- R2E-Gym: https://github.com/agentica-project/R2E-Gym
