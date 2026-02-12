# DeepSWE Training Guide on youyun Cluster

This document summarizes the experiments, issues encountered, and solutions found during DeepSWE reproduction on the youyun K8s cluster with Docker backend (Feb 9-12, 2026).

## Experiment Timeline

| Date | Activity | Key Finding |
|------|----------|-------------|
| Feb 9 | Docker-based training test | Successfully resolved astropy issue with reward=1.0 |
| Feb 10 | Fixed pytest quoting bug for Django | Evaluation accuracy jumped from 17% to 75% |
| Feb 11 | Started verl/rllm training on youyun.37/38 | TMA errors, disk full, rate limiting issues |
| Feb 11 | Created R2EGymAgent with detailed prompts | Success rate improved from 0% to 75% |
| Feb 11 | Fixed empty entropys issue | log_prob_micro_batch_size_per_gpu=1 is required |
| Feb 12 | Multi-node training (16 GPUs) | Stable training with all fixes applied |

---

## Cluster Environment

| Item | Value |
|------|-------|
| **Nodes** | youyun.37 (head) + youyun.38 (worker) |
| **GPUs** | 16x NVIDIA H200 (140GB VRAM each) |
| **Framework** | verl 0.6.1 + vLLM 0.10.2 |
| **Model** | Qwen3-32B |
| **Backend** | Docker |

---

## Issues Encountered and Solutions

### 1. Pytest Quoting Bug for Django Tests

**Problem**: Django test evaluation failed with shell syntax error:
```bash
python -m pytest test_ascii_validator (auth_tests.test_validators.UsernameValidatorsTests) -xvs
# Error: syntax error near unexpected token '('
```

**Root Cause**: Django test names contain parentheses that need shell escaping.

**Solution**: Quote test names in bash command:
```python
# Before (BUG)
f"python -m pytest {test} -xvs"

# After (FIXED)
f"python -m pytest '{test}' -xvs"
```

**Impact**: Evaluation accuracy jumped from 17% to 75% after fix.

---

### 2. Flash Attention TMA Error

**Problem**: Training crashed after ~26 minutes with long sequences (~119K tokens):
```
Error: Failed to initialize the TMA descriptor 710
globalDim (128,16,1,119713,1)
```

**Root Cause**: vLLM CUDA graph capture fails for very long sequences with Flash Attention.

**Solution**: Disable CUDA graphs:
```bash
actor_rollout_ref.rollout.enforce_eager=True
```

**Trade-off**: ~10-15% slower inference, but stable for long sequences.

---

### 3. Disk Space Issues

**Problem**: Root partition 100% full (879GB used), training failed to start.

**Root Cause**: Docker images accumulated to 575GB with no cleanup policy.

**Solution**:
```bash
# Clean unused Docker images
docker container prune -f && docker image prune -a -f

# Migrate Docker data-root to larger disk
sudo systemctl stop docker docker.socket
echo '{"data-root": "/data/docker"}' > /etc/docker/daemon.json
sudo rsync -aP /var/lib/docker/ /data/docker/
sudo systemctl start docker
```

---

### 4. Docker Hub Rate Limiting (429)

**Problem**: Training crashed when pulling instance-specific Docker images.

**Root Cause**: Docker Hub limits anonymous pulls to 100/6 hours.

**Solution**: Authenticate with Docker Hub:
```bash
echo 'YOUR_DOCKER_PAT' | docker login --username YOUR_USERNAME --password-stdin
```

| Account Type | Limit |
|--------------|-------|
| Anonymous | 100 pulls / 6 hours |
| Free (authenticated) | 200 pulls / 6 hours |

**Long-term**: Pre-pull images with batch script (see pull_r2e_images.sh).

---

### 5. Empty Batch Crash (micro_batch_size=None)

**Problem**: Training crashed when all samples in a batch were masked/filtered.

**Root Cause**: When ALL trajectories are filtered, micro_batch_size becomes None.

**Solution**: Add two-layer protection:

1. **dp_actor.py** - Return empty tensors gracefully
2. **agent_ppo_trainer.py** - Skip step when entropys is empty

---

### 6. Empty Entropys from compute_log_prob

**Problem**: Training skipped all PPO updates because compute_log_prob returned empty results.

**Root Cause**: log_prob_micro_batch_size_per_gpu defaults to None in verl config.

**Solution** (CRITICAL):
```bash
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
```

---

### 7. mask_truncated_samples Filter Issue

**Problem**: All samples filtered out, resulting in empty batch.

**Root Cause**: Rejection Sampling + mask_truncated_samples together filter all samples.

**Solution**:
```bash
rllm.mask_truncated_samples=False
```

---

### 8. BPE Retokenization Issue

**Problem**: Token mismatch in assemble_steps causing response_masks to be all zeros.

**Root Cause**: BPE tokenization is context-sensitive: tokenize(A+B) != tokenize(A) + tokenize(B).

**Solution**: Modified assemble_steps to use lenient token assembly, added config flag:
```bash
rllm.filter_token_mismatch=False
```

---

### 9. Agent Prompt Quality Issue

**Problem**: SWEAgent with minimal prompts had 0% success rate (0 rewards after 52 trajectories).

**Root Cause**: Minimal prompt does not guide model through systematic workflow.

**Comparison**:
| Agent | Prompt Style | Success Rate |
|-------|--------------|--------------|
| SWEAgent (sweagent) | Minimal | 0% |
| R2EGymAgent | 9-step detailed | 75% |

**Solution**: Created R2EGymAgent class with embedded R2E-Gym prompts:
- 4 tools: file_editor, execute_bash, search, finish
- 9-step workflow: explore -> reproduce -> fix -> verify -> test

**Files**: rllm/agents/r2egym_agent.py

---

## Recommended Training Configuration

```bash
#!/bin/bash
# train_deepswe_multinode.sh

# Algorithm (GRPO++ / RLOO)
algorithm.adv_estimator=rloo
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.clip_ratio_high=0.28
actor_rollout_ref.actor.entropy_coeff=0.0
algorithm.kl_ctrl.kl_coef=0.001

# Rollout (vLLM)
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.tensor_model_parallel_size=8
actor_rollout_ref.rollout.enforce_eager=True
actor_rollout_ref.rollout.gpu_memory_utilization=0.35
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True

# Actor (FSDP Training)
actor_rollout_ref.actor.ulysses_sequence_parallel_size=8
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# Agent
rllm.agent.name=r2egym
rllm.agent.max_steps=50
rllm.agent.trajectory_timeout=1800

# Bug fixes
rllm.mask_truncated_samples=False
rllm.filter_token_mismatch=False

# Multi-node
trainer.nnodes=2
trainer.n_gpus_per_node=8
```

---

## Multi-node Training Setup

```bash
# On head node (youyun.37)
ray start --head --port=6379 --dashboard-host=0.0.0.0

# On worker node (youyun.38)
ray start --address='youyun.37:6379'

# Verify cluster
ray status
```

---

## Monitoring Commands

```bash
# Watch training log
ssh youyun.37 "tail -f ~/work/logs/deepswe_multinode.log"

# Check GPU usage
ssh youyun.37 "nvidia-smi --query-gpu=index,memory.used --format=csv"
ssh youyun.38 "nvidia-smi --query-gpu=index,memory.used --format=csv"

# Check Docker containers
ssh youyun.37 "docker ps | wc -l"

# Check Ray cluster status
ssh youyun.37 "source ~/work/rllm/.venv/bin/activate && ray status"
```

---

## Key Metrics to Monitor

| Metric | Healthy Range | Warning Sign |
|--------|---------------|--------------|
| actor/entropy | 0.05-0.15 | < 0.01 (policy collapse) |
| critic/score/mean | 0.2-0.5 | Always 0 (no positive reward) |
| grad_norm | 0.1-1.0 | 0.0 (no gradient) |
| response_length/mean | 15K-25K | > 30K (truncation risk) |
| GPU memory | < 130GB | > 135GB (OOM risk) |

---

## DeepSWE Algorithm Summary

**GRPO++ Components** (from Together AI technical report):
1. **RLOO (Leave One Out)**: Uses other samples as baseline instead of critic
2. **Clip High (DAPO)**: clip_ratio_high=0.28 for stable training
3. **No KL Loss**: use_kl_loss=False
4. **Compact Filtering**: Masks max-context/max-steps/timeout trajectories
5. **No Entropy Loss**: entropy_coeff=0.0
6. **Length Normalization**: Per-token loss normalization

**Reward**: Sparse ORM (1 for passing tests, 0 otherwise)

---

## Related Files

| File | Purpose |
|------|---------|
| train_deepswe_multinode_r2egym.sh | Multi-node training script |
| pull_r2e_images.sh | Batch Docker image pull script |
| r2e_gym_images.txt | List of 4578 R2E-Gym Docker images |
| rllm/agents/r2egym_agent.py | R2EGymAgent with detailed prompts |

---

## Changelog

- **2026-02-09**: Docker-based training test succeeded
- **2026-02-10**: Fixed pytest quoting bug, evaluation accuracy 17% -> 75%
- **2026-02-10**: Set up youyun K8s cluster access
- **2026-02-11**: Started verl training, discovered TMA error and disk issues
- **2026-02-11**: Created R2EGymAgent, fixed empty batch crashes
- **2026-02-11**: Found log_prob_micro_batch_size fix for empty entropys
- **2026-02-12**: Multi-node training (16 GPUs) stable with all fixes
- **2026-02-12**: Documented all issues and solutions (Docker-only)
