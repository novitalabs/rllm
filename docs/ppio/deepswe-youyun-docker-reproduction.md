# DeepSWE Reproduction on youyun Cluster — Docker Backend (2-Node, 16 H200)

> Full experiment report for DeepSWE reproduction using rllm framework with R2E-Gym Docker backend on youyun.37/.38.
> Duration: Feb 10 – Feb 25, 2026 (16 days).
> Conclusion: Training completed 200 steps but yielded minimal improvement (+0.2pp) due to dead learning (insufficient optimizer steps per training step).

---

## 1. Overview

### Goal

Reproduce [Together AI's DeepSWE](https://together.ai/blog/deepswe-agentic-swe-bench) training on a 2-node cluster with Docker backend.

### Official DeepSWE Specs

| Item | Value |
|------|-------|
| Framework | rLLM (Agentica's RL post-training framework) |
| Algorithm | GRPO++ (RLOO advantage + Clip High + No KL Loss + Compact Filtering) |
| Base Model | Qwen3-32B (pure RL, no SFT) |
| Hardware | 64 H100 GPUs for 6 days |
| Batch Size | 64 with 8 passes (512 concurrent Docker containers) |
| Training Steps | 200 iterations |
| Dataset | 4,500 R2E-Gym tasks |
| Result | 42.2% Pass@1 on SWE-Bench-Verified |

### Our Setup

| Item | Value |
|------|-------|
| Nodes | youyun.37 (head, 10.83.115.10) + youyun.38 (worker, 10.83.115.12) |
| GPUs | 16× NVIDIA H200 (140GB VRAM each) |
| Network | 25Gbps management (b_manage0) + **400Gbps RoCE unused** (see §5) |
| Framework | `/home/claude/work/rllm-origin` — verl 0.6.1, vllm 0.10.2, Python 3.10 |
| Model | Qwen3-32B at `/home/claude/work/rllm/models/Qwen3-32B/` |
| R2E-Gym | `/home/claude/work/R2E-Gym/src` (swebench==3.0.2) |
| Backend | Docker (local containers, not k8s) |
| Agent | R2EGymAgent with 9-step workflow |

### Result

| Model | Eval (Custom XML) | Eval (Official Grading) |
|-------|-------------------|------------------------|
| Qwen3-32B base | 374/500 (74.8%) | 61/500 (12.2%) |
| DeepSWE-Step164 | 381/500 (76.2%) | — |
| DeepSWE-Step198 | 379/500 (75.8%) | 62/500 (12.4%) |

Fine-tuning improvement: **+0.2pp to +1.4pp** depending on eval method — essentially no meaningful gain.

---

## 2. Timeline

```
Feb 10   Feb 12    Feb 14    Feb 16    Feb 18    Feb 20    Feb 22    Feb 24
  |         |         |         |         |         |         |         |
  ▼         ▼         ▼         ▼         ▼         ▼         ▼         ▼
──┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬──
  │  SETUP  │  TRAIN  │  EVAL   │ STEADY  │  DISK   │ MINER!! │SECURITY │
  │  PHASE  │  START  │  FIXES  │ RUNNING │ CRISES  │DISCOVERY│HARDENING│
  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

  ●─── Run 1 (steps 1~18) ─────────● Crash: ZMQ deadlock (15h stall)
                                    ●── Run 2 (18~120) ──● ZMQ crash #3
                                                          ●─ Run 3 (120~122) ─● ZMQ crash #4
                                                                               ●── Run 4 (122~127) ──●
                                                                                     ● Disk full #1 ●
                                                                               ●───── Run 5 (127~160) ────● vLLM degraded
                                                                                                           │ (miner active
                                                                                                           │  since Feb 20)
                                                                               Steps 161-170: ZERO LEARNING│
                                                                                                           ● miner killed
                                                                                    ●─ Run 6 (from 160) ──● ZMQ crash #5
                                                                                    ●─ Run 7 (from 168) ──● vLLM hang 188
                                                                                    ●─ Run 8 (from 186) ──────────────● step 199 ✓
```

### Phase Details

| Date | Steps | Event | Duration |
|------|-------|-------|----------|
| Feb 10 | — | Env setup: pkg debugging, Docker fixes, pytest quoting fix, R2EGymAgent created | 1 day |
| Feb 12 | 1-3 | Training starts (single-node → multi-node), FSDP cross-node bottleneck discovered | ~1h |
| Feb 12-13 | 1-18 | Run 1: ZMQ deadlock at step 18 | ~15h |
| Feb 13-14 | 18-22 | Validation hang (scikit-learn, 14h stuck). `test_freq` changed 20→200 | 14h |
| Feb 14 | — | Option B validated (TP=16 rollout + Ulysses=8 FSDP). `rollout.n` 2→8, `total_steps` 20→200 | — |
| Feb 14 | 38-40 | Disk full on youyun.38 (checkpoints 3.6TB). Added `max_ckpt_to_keep=3` | — |
| Feb 15-17 | 3-122 | Run 2: Steady training. Step 56 peaked at 68.8% solve rate | ~2 days |
| Feb 17 | 122-123 | ZMQ crashes #3, #4 (3.3h apart). Cross-node TP=16 ZMQ unstable | — |
| Feb 18 | 128, 132 | Disk full on .37 (Docker images 6TB), I/O crash during cleanup | — |
| Feb 18-20 | 127-160 | Run 5: Steady progress | ~2 days |
| Feb 19 | 140 | OOM crash: 5 workers killed, NCCL broken | — |
| Feb 20 | 162-165 | Disk full again (7.0T/7.0T), ZMQ crash #5 | — |
| Feb 20-21 | 161-170 | **ZERO LEARNING**: crypto miner consuming 18000%+ CPU, vLLM 10× slower | ~1 day |
| Feb 22 | — | Miner killed. ZMQ retry patch applied. Step 164 eval: **76.2%**. Qwen3 base: **74.8%** | — |
| Feb 22-23 | 160-187 | Runs 6-7: ZMQ crash, vLLM hang at step 188. Outer timeout fix applied | — |
| Feb 23-24 | 186-199 | **Run 8**: Clean run (0 TimeoutErrors), steps 186→199 completed | ~8h |
| Feb 24 | 199+ | Validation triggered (SWE-Bench Verified). **Stuck** at batch 2/16 (vLLM hang) | — |
| Feb 25 | — | Training shutdown. Source committed. PPIO hyperparameter analysis. Eval v3 on k8s | — |

---

## 3. Configuration

### Training Config Evolution

| Parameter | Initial (Feb 12) | Final (Feb 24) | When Changed |
|-----------|------------------|----------------|--------------|
| `algorithm.adv_estimator` | `rloo` | `grpo` | Feb 12 (16:34, 5 runs in) |
| `total_training_steps` | 20 | 200 | Feb 14 |
| `test_freq` | 5 → 20 | 200 | Feb 14 (validation hang) |
| `save_freq` | 5 | 2 | Feb 14 |
| `rollout.n` | 2 | 8 | Feb 14 |
| `train_batch_size` | 4 | 4 | unchanged |
| `ppo_mini_batch_size` | 4 | 4 | unchanged |
| `trajectory_timeout` | 1800s | 3600s | Feb 21 |
| `max_actor_ckpt_to_keep` | not set | 3 | Feb 14 |
| `enforce_eager` | True | True | unchanged |

### Final Training Script

File: `/home/claude/work/rllm-origin/train_deepswe_full.sh` (on youyun.37)

```bash
#!/bin/bash
set -x
cd /home/claude/work/rllm-origin

export RLLM_DIR=/home/claude/work/rllm-origin
export R2EGYM_DIR=/home/claude/work/R2E-Gym/src
export PYTHONPATH="$RLLM_DIR:$R2EGYM_DIR:$PYTHONPATH"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export WANDB_MODE=disabled
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export NCCL_NVLS_ENABLE=0
export NCCL_SOCKET_IFNAME=eth0    # ← MISTAKE: should be GPU0 for RDMA
source $RLLM_DIR/.venv/bin/activate

python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=.../R2E_Gym_Subset/train_verl.parquet \
    data.val_files=.../SWE_Bench_Verified/test_verl.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=32 \
    data.max_prompt_length=32768 \
    data.max_response_length=32768 \
    data.shuffle=false \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    rllm.filter_token_mismatch=False \
    trainer.critic_warmup=0 \
    "trainer.logger=[console]" \
    trainer.project_name=deepswe-full \
    trainer.experiment_name=tp16-n8-full \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=2 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.test_freq=200 \
    rllm.env.name=swe \
    +rllm.env.env_args.backend=docker \
    +rllm.env.env_args.scaffold=r2egym \
    rllm.agent.name=r2egym \
    rllm.agent.max_steps=50 \
    rllm.agent.trajectory_timeout=3600 \
    trainer.total_training_steps=200
```

### Comparison with Official Config

| Parameter | Official (64 H100) | Ours (16 H200) | Impact |
|-----------|--------------------|-----------------| -------|
| `algorithm.adv_estimator` | `rloo` (official) | `grpo` | We switched from rloo to grpo on Feb 12; never switched back |
| `batch_size` | 64 | **4** | 16× smaller → less reward signal |
| `ppo_mini_batch_size` | 8 | **4** | |
| `ppo_epochs` | 4 (default) | **1** (implicit) | 32× fewer optimizer steps/step |
| `rollout.n` | 8 | 8 | Same |
| `max_steps` (agent) | 100 | **50** | Half the agent budget |
| `trajectory_timeout` | 5400s | **3600s** | |
| `backend` | kubernetes | **docker** | Docker slower for parallel containers |
| `scaffold` | sweagent | **r2egym** | Different prompt format |
| `tensor_parallel_size` | 8 | **16** (cross-node) | Cross-node ZMQ unstable |
| `ulysses_seq_parallel` | 8 | **8** (single-node) | Same (after Option B) |
| `max_prompt_length` | 4096 | **32768** | Ours 8× longer |
| `enforce_eager` | False | **True** | Avoids TMA error but slower |
| `temperature` | 1.0 | **0.6** | Different exploration |

---

## 4. Architecture: Option B

### Problem

Cross-node FSDP with Ulysses=16 caused PPO update to take **2+ hours** (vs ~50s expected).

### Root Cause

`NCCL_SOCKET_IFNAME=b_manage0` forced NCCL to use the **25Gbps management network** (TCP socket), instead of the 400Gbps RoCE RDMA NICs that exist on both nodes.

### Solution (Option B)

- **TP=16** for vLLM rollout (cross-node inference is fine over Ethernet)
- **Ulysses=8** for FSDP training (single-node, NVLink only)
- PPO update dropped to **~120s**

### Post-mortem (Feb 25)

Both nodes actually have **8× ConnectX-7 400Gbps RoCE NICs** (`mlx5_0`~`mlx5_11`, netdev `GPU0`~`GPU7`), each paired with a GPU. `/dev/infiniband/` devices are present and all ports are Active at Rate 400. The FSDP bottleneck was caused by NCCL being forced onto the 25Gbps management bond. Setting `NCCL_SOCKET_IFNAME=GPU0` or `NCCL_IB_HCA=mlx5` / `NCCL_NET=IB` would have enabled RDMA and potentially made cross-node Ulysses=16 viable without Option B.

---

## 5. Issues Encountered & Fixes

### Training Framework Issues (7)

| # | Issue | Symptom | Fix | Date |
|---|-------|---------|-----|------|
| 1 | Flash Attention TMA | `Failed to initialize TMA descriptor` with 119K tokens | `enforce_eager=True` | Feb 11 |
| 2 | Empty Entropys | PPO updates skipped; `log_prob_micro_batch_size_per_gpu=None` | Set to `1` + `log_prob_use_dynamic_bsz=True` | Feb 11 |
| 3 | BPE Retokenization | Token mismatch in `assemble_steps` → empty response_masks | `filter_token_mismatch=False` | Feb 11 |
| 4 | Truncated Samples | All samples filtered when combined with rejection | `mask_truncated_samples=False` | Feb 12 |
| 5 | Agent Prompt Quality | SWEAgent minimal prompts → 0% solve rate | R2EGymAgent with 9-step workflow → 75% | Feb 12 |
| 6 | Cross-node FSDP | PPO update 2+ hours (Ethernet, not RDMA) | Option B: TP=16 rollout, Ulysses=8 FSDP | Feb 12 |
| 7 | Outer Timeout | Hardcoded 7200s × 3 retries = 6h worst case | `timeout = trajectory_timeout × 1.2` | Feb 23 |

### Infrastructure Issues (4)

| # | Issue | Symptom | Fix | Date |
|---|-------|---------|-----|------|
| 8 | Disk Full (.38) | Checkpoint save failed, 7TB full | Move to `/data/nvme`, symlink, `max_ckpt_to_keep=3` | Feb 14 |
| 9 | Disk Full (.37) | Docker images 6.04TB (4262 images) | Prune non-active images, keep training set | Feb 18 |
| 10 | Docker Hub 429 | Rate limiting during image pulls | Docker login + pre-pull batch scripts | Feb 10 |
| 11 | Ray Duplicate GPU | 3 nodes instead of 2, duplicate CUDA device | Restart Ray with explicit addresses | Feb 12 |

### ZMQ Deadlock (Root Cause found Feb 22)

In `vllm_rollout_spmd.py`, the `_loop_forever` method catches ALL exceptions and **breaks** on any error. `zmq.error.Again` (EAGAIN) is a transient non-fatal error but was treated as fatal, permanently killing the ZMQ server loop.

**Fix**: Added `except zmq.error.Again` handler that retries with `asyncio.sleep(0.1)` instead of breaking. Applied on both nodes.

### Code Patch: Outer Timeout Fix

Applied to `rllm/engine/agent_execution_engine.py:500-510`:

```python
async def run_agent_trajectory_with_retry(self, idx, seed=0, mode="Text", **kwargs):
    for _ in range(self.retry_limit):
        try:
            application_id = str(uuid.uuid4())
            outer_timeout = int((self.trajectory_timeout or 3600) * 1.2)
            return await asyncio.wait_for(
                self.run_agent_trajectory_async(
                    idx, application_id=application_id,
                    seed=seed, mode=mode, **kwargs),
                timeout=outer_timeout)
        except Exception:
            traceback.print_exc()
            continue
    traceback.print_exc()
    raise Exception(f"Trajectory {idx} cannot complete.")
```

---

## 6. Crash Log (15 incidents)

| # | Date | Step | Type | Root Cause | Recovery From |
|---|------|------|------|------------|---------------|
| 1 | Feb 13 | 18 | ZMQ deadlock | `zmq.error.Again` breaks server loop (15h stall) | step 18 |
| 2 | Feb 13 | 20 (val) | Validation hang | scikit-learn tasks stuck 14h in retry loops | step 18 |
| 3 | Feb 14 | 40 | Disk full | youyun.38 at 100% (3.6TB checkpoints) | step 38 |
| 4 | Feb 17 | 122 | ZMQ/CancelledError | vLLM `CancelledError` cascade | step 120 |
| 5 | Feb 17 | 123 | ZMQ timeout | 3.3h after crash #4 | step 122 |
| 6 | Feb 18 | 128 | Disk full | youyun.37 at 100% (Docker 6.04TB) | step 126 |
| 7 | Feb 18 | 132 | Disk I/O | Concurrent `docker rmi` I/O contention | step 130 |
| 8 | Feb 19 | 140 | OOM | 5 Ray workers killed, NCCL broken | step 140 |
| 9 | Feb 20 | 162 | Trajectory hang | Multiple trajectories "cannot complete" | step 160 |
| 10 | Feb 20 | 164 | Disk full | /data 100% full again (7.0T/7.0T) | step 162 |
| 11 | Feb 20 | 165 | ZMQ timeout | ZMQ on youyun.38 | step 164 |
| 12 | Feb 21 | 161-170 | Zero learning | Crypto miner → vLLM 10× slower | step 160 |
| 13 | Feb 22 | 160 | NCCL failure | TCPStore recv failure after restart | step 160 |
| 14 | Feb 22 | 160 | Agent interference | Claude agent launched 3+ duplicate processes | step 160 |
| 15 | Feb 24 | 188 | vLLM hang | Same as step 169 pattern (every ~20 steps) | step 186 |

**vLLM Hang Pattern**: Every ~19-20 steps, the vLLM server stops responding. The outer timeout fix (Issue #7) prevents infinite blocking but doesn't prevent the hang itself.

---

## 7. Security Incident

Two malware families discovered Feb 22-23, active since Feb 14 and Feb 20 respectively:

### Malware #1: `/bin/kworker` (Ethereum Backdoor)

- **Installed**: Feb 14 13:37-13:55, as root, on .37, .38, .36
- **Mechanism**: SystemD service with `Restart=always`
- **Behavior**: Contacts Ethereum RPC endpoints (infura, llamarpc, blastapi, drpc.org)
- **MD5**: `d0ee21805e3793a525bb3cf727a7ac05`
- **Deployment gap**: 6-12 min between nodes → automated infrastructure deployment

### Malware #2: `/tmp/dns` (Crypto Miner)

- **Installed**: Feb 20 00:15, as user `claude`, on .37 and .38 only
- **Binary**: 9.8MB static-pie ELF, C2 server `103.127.134.124`
- **CPU Usage**: 18,000%+ (182 cores) per node
- **Persistence**: Crontab every 15 minutes
- **Impact**: **Root cause of 10× vLLM throughput degradation** (Steps 161-170 had ZERO learning — all ENV_TIMEOUT, pg_loss=0.0)

### Attack Vector

Most likely through PPIO hosting infrastructure (jumpserver SSH key or `admin@PPIO-DN` root authorized_key).

### Remediation

1. Killed all malware processes
2. Removed crontab entries, stopped/disabled kworker.service
3. Blocked C2 IP via iptables
4. Removed `admin@PPIO-DN` SSH key from root authorized_keys
5. Changed password on .37/.38
6. Restricted Ray port 6379 via iptables
7. Set up security monitoring cron (every 30 min)

---

## 8. Evaluation Results

### 8.1 Evaluation Script Evolution

The eval script went through 3 major revisions. Results varied dramatically:

| Version | Grading | Scaffold | Key Differences |
|---------|---------|----------|-----------------|
| v1 (196.2) | Custom (buggy, no P2P, F2P[:5]) | SWEAgent | Inflated by ~7pp |
| v2 (k8s) | Custom XML format | R2E-Gym XML fn-call | Per-test `pytest -xvs`, no `/run_tests.sh` |
| v3 (k8s) | Official `/run_tests.sh` + `get_eval_report()` | R2E-Gym training-aligned | Correct grading |

### 8.2 All Results

#### Early Evaluations (196.2 server, 8× RTX 4090, buggy scripts)

| Model | Script | Resolved | Rate | Notes |
|-------|--------|----------|------|-------|
| DeepSWE-Preview | v1 (no P2P, [:5]) | 363/500 | 72.6% | Inflated — no regression test |
| Qwen3-32B base | v2 (with P2P) | 357/500 | 71.4% | 23 errored (pytest quoting) |

#### K8s Evaluations (custom XML format, non-official grading)

| Model | Script | Workers | Resolved | Rate |
|-------|--------|---------|----------|------|
| Step164 | R2E-Gym OpenAI fn-call | 8 | 84/498 | 16.9% |
| Step164 | Custom XML | 8 | 381/500 | **76.2%** |
| Qwen3-32B base | Custom XML | 8 | 374/500 | **74.8%** |
| DeepSWE-Preview | Custom XML | 8 | 327/500 | 65.4% |
| Step198 | Custom XML (2-shard) | 16 | 379/500 | **75.8%** |

#### Final Evaluations (v3 training-aligned scaffold, official `/run_tests.sh` grading, 6 vLLM nodes)

| Model | Resolved | Rate | Delta vs Base |
|-------|----------|------|---------------|
| Qwen3-32B base | 61/500 | 12.2% | baseline |
| DeepSWE-Step198 | 62/500 | 12.4% | **+0.2pp** |
| agentica-org/DeepSWE-Preview | 63/500 | 12.6% | +0.4pp |

### 8.3 Key Eval Findings

1. **Scaffold/format dominates**: Same model goes from 16.9% to 76.2% by matching the training format (59pp difference from scaffold alone)
2. **Custom grading vs official grading**: 76.2% (v2 custom) → 12.2% (v3 official) — custom eval script had fundamental grading flaws (no test file reset, no test patch, per-test pytest vs full suite)
3. **Fine-tuning improvement is negligible**: +0.2pp (official) to +1.4pp (custom) — within noise
4. **Resolved instance overlap**: Despite similar totals (~62 each), the 3 models solve very different subsets. Union = 116 instances (nearly 2× any single model). Each model uniquely solves ~1/3 of its instances.
5. **DeepSWE-Preview misalignment**: Our eval had `enable_thinking=False`, `max_tokens=4096`, `max_steps=30`, `temperature=0` — all wrong for a model trained WITH thinking tokens at T=1.0, max_steps=100, max_tokens=32K+

---

## 9. Root Cause Analysis: Why Training Failed

### Primary Cause: Dead Learning (pg_clipfrac ≈ 0)

Analysis of PPIO's 19 experiment iterations (v1~v6.4) revealed the critical relationship between mini_batch_size and policy learning:

| mini_batch | batch | opt_steps/step | pg_clipfrac | Learning |
|------------|-------|----------------|-------------|----------|
| 8 | 64 | 32 (8×4 epochs) | 1.3e-4 → 2.3e-4 | **Most alive** |
| 16 | 64 | 16 (4×4) | 6.8e-5 → 1.8e-4 | Alive |
| 32 | 32 | 4 (1×4) | 0.0 | **Dead** |
| 64 | 64 | 4 (1×4) | 0.0 | **Dead** |
| **4 (ours)** | **4** | **1 (1×1)** | **likely 0** | **likely dead** |

Our config (`batch_size=4, mini_batch_size=4`) → **only 1 optimizer update per training step** → pg_clipfrac likely 0 → policy barely updates across 200 steps.

### Contributing Factors

1. **NCCL misconfiguration**: Forced onto 25Gbps management network instead of 400Gbps RDMA → required Option B workaround
2. **Crypto miner (Steps 161-170)**: 10 steps with zero learning due to CPU contention
3. **ZMQ instability**: 5 crashes from cross-node TP=16 ZMQ communication
4. **Disk space**: 3 disk-full incidents requiring manual cleanup
5. **Different adv_estimator**: Used `grpo` instead of `rloo` (official DeepSWE uses RLOO per `examples/swe/train_deepswe_32b.sh`)
6. **Different temperature**: 0.6 instead of 1.0 (official)
7. **Half agent budget**: `max_steps=50` instead of 100 (official)

### Recommended Fixes for Next Run

```bash
# Critical: increase optimizer steps per training step
data.train_batch_size=16                          # 4 → 16
actor_rollout_ref.actor.ppo_mini_batch_size=4     # keep 4
actor_rollout_ref.actor.ppo_epochs=4              # new → 16 opt steps/step

# Match official config
algorithm.adv_estimator=rloo                      # grpo → rloo
actor_rollout_ref.rollout.temperature=1.0         # 0.6 → 1.0
rllm.agent.max_steps=100                          # 50 → 100
rllm.agent.trajectory_timeout=5400                # 3600 → 5400
data.max_prompt_length=4096                       # 32768 → 4096

# Enable RDMA
export NCCL_SOCKET_IFNAME=GPU0                    # b_manage0 → GPU0
export NCCL_IB_HCA=mlx5
export NCCL_NET=IB
```

---

## 10. Step-Level Training Metrics

Extracted from console logs (`~/work/logs/deepswe_full*.log`). Steps 5-160 are missing — log files overwritten by subsequent restarts. Steps 165-170 impacted by crypto miner (zero learning). pg_clipfrac=0.0 for ALL steps, confirming dead learning.

**Phase 1: Initial Training (Feb 12, Run 1)**

| Step | Score | pg_loss | pg_clip | entropy | grad_norm | resp_len | traj_steps | llm_time | step_time | PPO_time | solve(a/p/n) |
|------|-------|---------|---------|---------|-----------|----------|------------|----------|-----------|----------|--------------|
| 1 | 0.0% | 0.00 | 0.0 | 1633 | 0 | 29187 | 30.5 | 570s | 1825s | 96s | 0/0/4 |
| 2 | 6.2% | 43.90 | 0.0 | 1354 | 1771 | 27627 | 31.4 | 660s | 1689s | 82s | 0/1/3 |
| 3 | 9.4% | 123.57 | 0.0 | 1308 | 2603 | 28923 | 28.6 | 671s | 1440s | 88s | 0/2/2 |
| 4 | 31.2% | 183.65 | 0.0 | 1600 | 3719 | 26906 | 33.0 | 540s | 1133s | 82s | 0/3/1 |

*Steps 5-160: No console logs preserved (overwritten by nohup restarts).*
*Partial diary records: Step 31=40.6%, Step 56=68.8% (peak), Step 109=46.9%, Step 119=40.6%, Step 121=28.1%, Step 127=37.5%.*

**Phase 2: Crypto Miner Period (Feb 21, Steps 161-170, ZERO LEARNING)**

| Step | Score | pg_loss | pg_clip | entropy | grad_norm | resp_len | traj_steps | llm_time | step_time | PPO_time | solve(a/p/n) |
|------|-------|---------|---------|---------|-----------|----------|------------|----------|-----------|----------|--------------|
| 161 | 6.2% | 3.76 | 0.0 | 918 | 1606 | 8861 | 6.0 | 4611s | 6876s | 264s | 0/1/3 |
| 165 | 0.0% | 0.00 | 0.0 | 557 | 0 | 6379 | 4.8 | 3008s | 11811s | 244s | 0/0/4 |
| 166 | 0.0% | 0.00 | 0.0 | 362 | 0 | 7504 | 4.1 | 2410s | 4407s | 249s | 0/0/4 |
| 167 | 0.0% | 0.00 | 0.0 | 482 | 0 | 6325 | 3.3 | 2751s | 4753s | 252s | 0/0/4 |
| 168 | 0.0% | 0.00 | 0.0 | 373 | 0 | 6120 | 4.2 | 2694s | 6154s | 250s | 0/0/4 |
| 169 | 0.0% | 0.00 | 0.0 | 425 | 0 | 4776 | 3.7 | 2623s | 4599s | 247s | 0/0/4 |
| 170 | 0.0% | 0.00 | 0.0 | 416 | 0 | 6865 | 4.0 | 2509s | 4684s | 247s | 0/0/4 |

*Steps 165-170: Crypto miner at 18000%+ CPU → vLLM 10x slower → agents timeout after 3-6 steps → all ENV_TIMEOUT → score=0, pg_loss=0, grad_norm=0.*

**Phase 3: Post-Miner (Feb 23-24, Steps 171-199, Runs 7-8)**

| Step | Score | pg_loss | pg_clip | entropy | grad_norm | resp_len | traj_steps | llm_time | step_time | PPO_time | solve(a/p/n) |
|------|-------|---------|---------|---------|-----------|----------|------------|----------|-----------|----------|--------------|
| 171 | 21.9% | 13.56 | 0.0 | 2735 | 3378 | 23702 | 21.6 | 1018s | 1894s | 121s | 0/2/2 |
| 172 | 12.5% | 11.46 | 0.0 | 2687 | 2590 | 25791 | 25.3 | 975s | 1559s | 114s | 0/1/3 |
| 173 | 31.2% | 51.28 | 0.0 | 3629 | 3741 | 24593 | 19.3 | 1145s | 2094s | 118s | 0/2/2 |
| 174 | 56.2% | 56.41 | 0.0 | 1542 | 3291 | 18138 | 19.1 | 767s | 1864s | 104s | 1/2/1 |
| 175 | 56.2% | 146.78 | 0.0 | 2798 | 4823 | 17939 | 15.6 | 998s | 1993s | 99s | 0/3/1 |
| 176 | 34.4% | 49.91 | 0.0 | 2871 | 4539 | 21693 | 21.5 | 942s | 1701s | 112s | 0/3/1 |
| 177 | 9.4% | 29.52 | 0.0 | 3179 | 2202 | 21831 | 16.5 | 1020s | 1998s | 113s | 0/1/3 |
| 178 | 18.8% | 19.51 | 0.0 | 3227 | 4226 | 22332 | 22.2 | 957s | 1790s | 112s | 0/3/1 |
| 179 | 25.0% | 34.46 | 0.0 | 4648 | 3912 | 23057 | 23.8 | 1192s | 2122s | 112s | 0/2/2 |
| 180 | 21.9% | 202.88 | 0.0 | 4480 | 4318 | 22172 | 17.2 | 1290s | 3448s | 121s | 0/2/2 |
| 181 | 37.5% | 62.47 | 0.0 | 4872 | 4309 | 25613 | 24.3 | 1123s | 1988s | 131s | 0/3/1 |
| 182 | 56.2% | -33.10 | 0.0 | 4289 | 5090 | 22082 | 20.2 | 992s | 2072s | 121s | 0/4/0 |
| 183 | 28.1% | 7.14 | 0.0 | 4612 | 3565 | 22900 | 22.7 | 1038s | 2149s | 110s | 0/2/2 |
| 184 | 65.6% | 62.27 | 0.0 | 3470 | 4210 | 20407 | 17.8 | 928s | 1977s | 103s | 0/3/1 |
| 185 | 37.5% | 92.55 | 0.0 | 5011 | 3764 | 22208 | 18.4 | 1083s | 2179s | 114s | 0/2/2 |
| 186 | 21.9% | 59.04 | 0.0 | 4872 | 3600 | 22750 | 21.0 | 1090s | 1769s | 117s | 0/2/2 |
| 187 | 28.1% | 21.88 | 0.0 | 4715 | 5021 | 24259 | 21.5 | 1125s | 1780s | 123s | 0/3/1 |
| 188 | 21.9% | 89.14 | 0.0 | 5160 | 4106 | 24658 | 21.3 | 1154s | 3268s | 122s | 0/2/2 |
| 189 | 18.8% | 17.14 | 0.0 | 6347 | 4628 | 25892 | 23.7 | 1166s | 1850s | 124s | 0/3/1 |
| 190 | 50.0% | 0.00 | 0.0 | 4050 | 0 | 21939 | 19.5 | 949s | 1865s | 111s | 2/0/2 |
| 191 | 25.0% | 28.70 | 0.0 | 4011 | 4577 | 22544 | 21.1 | 1012s | 2167s | 112s | 0/3/1 |
| 192 | 28.1% | 46.91 | 0.0 | 5193 | 5005 | 26329 | 22.6 | 1090s | 2307s | 128s | 0/3/1 |
| 193 | 37.5% | -12.19 | 0.0 | 5037 | 4990 | 24542 | 22.3 | 1073s | 2172s | 118s | 0/3/1 |
| 194 | 21.9% | 12.70 | 0.0 | 4728 | 2772 | 25305 | 21.1 | 946s | 1870s | 124s | 0/1/3 |
| 195 | 12.5% | 68.24 | 0.0 | 7444 | 4850 | 29151 | 26.5 | 1342s | 2853s | 138s | 0/2/2 |
| 196 | 21.9% | 1.16 | 0.0 | 6252 | 4297 | 27117 | 24.0 | 1126s | 1903s | 130s | 0/2/2 |
| 197 | 59.4% | 80.18 | 0.0 | 7472 | 7681 | 24488 | 22.4 | 1139s | 2085s | 118s | 0/3/1 |
| 198 | 21.9% | 47.45 | 0.0 | 7149 | 4345 | 24357 | 20.5 | 1012s | 1829s | 120s | 0/2/2 |
| 199 | 34.4% | 54.67 | 0.0 | 5486 | 6014 | 20085 | 18.2 | 971s | 2018s | 114s | 0/3/1 |

**Column Legend**:
- **Score**: `critic/score/mean` (fraction of rollouts that solved the task)
- **pg_loss**: `actor/pg_loss` (policy gradient loss)
- **pg_clip**: `actor/pg_clipfrac` (fraction of samples clipped by PPO — **0.0 for all steps = dead learning**)
- **entropy**: `actor/entropy` (policy entropy, higher = more exploration)
- **grad_norm**: `actor/grad_norm` (gradient L2 norm)
- **resp_len**: `response_length/mean` (tokens per response)
- **traj_steps**: `traj/steps_mean` (agent interaction steps per trajectory)
- **llm_time**: `traj/llm_time_mean` (LLM inference time per trajectory)
- **step_time**: `timing_s/collect_trajectory` (total rollout collection time)
- **PPO_time**: `timing_s/update_actor` (PPO update time)
- **solve(a/p/n)**: `batch/solve_all` / `solve_partial` / `solve_none` (4 prompts per batch, each with n=8 rollouts)

**Key Observations**:
1. **pg_clipfrac = 0.0 for ALL 39 recorded steps** — confirms dead learning (batch=4, mini_batch=4 → only 1 optimizer update/step)
2. **Entropy rising** from ~1300 (step 1) to ~5000-7000 (steps 190+) — policy becoming MORE random over training, not less
3. **Steps 165-170 fully zero**: miner caused ENV_TIMEOUT for all trajectories (traj_steps 3-5 vs normal 20+)
4. **Step 190 anomaly**: pg_loss=0, grad_norm=0 despite score=50% — all 8 rollouts either all-pass or all-fail per prompt, yielding zero advantage variance
5. **Score range 0-66%** with no trend — fluctuations from data difficulty, not policy improvement
6. **PPO update time stable** at ~110-130s (post-miner) — Option B (Ulysses=8 single-node) working correctly

**Source files** (on youyun.37, `/home/claude/work/logs/`):

| Log File | Steps | Run |
|----------|-------|-----|
| `deepswe_full_20260212_174943.log` | 1-3 | Run 1 start |
| `deepswe_full_resume.log` | 3-4 | Run 1 resume |
| `deepswe_full_20260221_step165_170.log` | 165-170 | Run 5 (miner) |
| `deepswe_full_20260222_zmqcrash_step161.log` | 161 | Run 6 (ZMQ crash) |
| `deepswe_full_restart_step168.log` | 169-187 | Run 7 |
| `deepswe_full_restart_step186.log` | 187-199 | Run 8 (final) |

## 11. Operational Runbook

### Start Ray Cluster

```bash
ssh youyun.37 "source ~/work/rllm/.venv/bin/activate && ray start --head --port=6379 --num-gpus=8"
ssh youyun.38 "source ~/work/rllm/.venv/bin/activate && ray start --address='10.83.115.10:6379' --num-gpus=8"
```

### Launch Training

```bash
ssh youyun.37 "cd ~/work/rllm-origin && nohup ./train_deepswe_full.sh > ~/work/logs/deepswe_full.log 2>&1 &"
```

### Kill Stuck Training

```bash
ssh youyun.37 "kill -9 \$(ps aux | grep train_agent_ppo | grep -v grep | awk '{print \$2}')"
ssh youyun.37 "sudo docker kill \$(sudo docker ps -q)"
# Then stop/restart Ray cluster
```

### Checkpoint Management

- **Location**: `/home/claude/work/rllm-origin/checkpoints/deepswe-full/tp16-n8-full/` (symlinked to `/data/nvme/...`)
- **Save frequency**: Every 2 steps
- **Max kept**: 3 most recent
- **Auto-resume**: Training resumes from `latest_checkpointed_iteration.txt`

### Merge FSDP → HuggingFace

```bash
cd ~/work/rllm-origin
python -m rllm.trainer.verl.merge_fsdp_to_hf \
    --checkpoint_dir checkpoints/deepswe-full/tp16-n8-full/global_step_198 \
    --output_dir ~/work/rllm/models/DeepSWE-Step198
```

---

## 12. Version Compatibility

| Component | Required | Notes |
|-----------|----------|-------|
| Python | 3.10.x | |
| torch | 2.8.0+cu128 | **Must match exactly** |
| vllm | 0.10.2 | Requires torch==2.8.0 |
| transformers | 4.50 - 4.57.x | **NOT 5.x** (API changes in tokenizer) |
| accelerate | 1.12.0 | |
| verl | 0.6.1 | |
| flash-attn | 2.8.x | |
| swebench | 3.0.2 | R2E-Gym requirement |

---

## 13. Key Lessons Learned

1. **pg_clipfrac is the single most important metric**: If it's 0, training is dead regardless of how many steps you run
2. **Optimizer steps per training step matter**: `batch=4, mini_batch=4` → 1 update → dead. Need at least 8-16 updates (via larger batch or more ppo_epochs)
3. **Scaffold/format dominates evaluation results**: Same model shows 16.9% vs 76.2% depending on prompt format (59pp difference)
4. **Eval methodology dramatically affects reported scores**: 76.2% (custom grading) → 12.2% (official grading) for the same model
5. **NCCL_SOCKET_IFNAME misconfiguration can silently degrade training**: 400Gbps hardware sitting idle while using 25Gbps management network
6. **Crypto miners cause silent 10× throughput degradation**: CPU-only mining doesn't show in GPU metrics
7. **ZMQ `zmq.error.Again` is transient**: Must retry, not break the server loop permanently
8. **Autonomous monitoring agents are unreliable**: Repeatedly launched duplicate training processes
9. **Disk space is the #1 operational risk**: Docker images (~6TB) + checkpoints (~184GB each) constantly fight for 7TB disk
10. **Do NOT kill Docker containers externally while asyncio is running**: Framework hangs indefinitely waiting for dead containers

---

## 14. Training Logs Archive

All training logs have been backed up to `/home/claude/work/rllm-origin/logs_backup/` (257MB total).
See `logs_backup/README.md` for full index.

### Directory Structure

```
logs_backup/                                          # 257MB total
├── ray_sessions/                                     # Compressed Ray worker logs
│   ├── session_2026-02-22_14-49-49_472914.tar.gz     #  190K — Run 5, Option A TP=8 attempt
│   ├── session_2026-02-22_15-43-40_745323.tar.gz     #  6.7M — Run 5/6, Option B early attempts
│   ├── session_2026-02-22_17-27-14_917894.tar.gz     #  1.9M — Run 6, batch=1 test
│   ├── session_2026-02-22_18-39-40_482380.tar.gz     #  1.9M — Run 7, crypto miner impacted
│   ├── session_2026-02-22_19-21-47_425549.tar.gz     #  2.4M — Run 7 restart
│   ├── session_2026-02-22_20-17-23_804434.tar.gz     #  8.1M — Run 7 continued (miner killed)
│   ├── session_2026-02-22_23-53-32.tar.gz            #   35M — Run 8 START (steps 0→~60)
│   ├── session_2026-02-23_15-03-44.tar.gz            #   72M — Run 8 RESTART (steps ~60→~140)
│   ├── session_2026-02-24_09-25-28.tar.gz            #  101M — Run 8 FINAL (steps ~140→200) ✓
│   └── session_2026-02-26_11-43-21.tar.gz            #   25M — Post-training evaluation session
├── hydra_configs/outputs/                            # Hydra config snapshots per launch
│   ├── 2026-02-10/  (23 sessions)                    # batch=8→2, tp=8 (single-node)
│   ├── 2026-02-11/  (4 sessions)                     # batch=2, tp=8
│   ├── 2026-02-12/  (15 sessions)                    # batch=4, tp=8 (Option A)
│   ├── 2026-02-13/  (1 session)                      # batch=4, tp=16 (Option B first)
│   ├── 2026-02-14~24/                                # batch=4, tp=16 (production)
│   └── Each session contains: .hydra/{config.yaml, hydra.yaml, overrides.yaml}
├── training_scripts/                                 # All 9 training shell scripts
│   ├── train_deepswe_full.sh                         # Final production script (Option B)
│   ├── train_deepswe_multinode.sh                    # Multi-node variant
│   └── ... (7 more variants)
├── experimental_scripts/                             # From git stash (4 files)
│   ├── train_deepswe_tp8.sh                          # TP=8 Option A variant
│   ├── train_deepswe_tp8_force_local.sh
│   ├── train_deepswe_no_ckpt.sh
│   └── train_minimal_test.sh
├── diagnostic_reports/                               # From git stash (6 reports)
│   ├── STATUS_REPORT.md
│   ├── debugging_summary.md
│   ├── final_diagnosis.md
│   ├── immediate_findings.md
│   ├── training_status.md
│   └── vllm_throughput_analysis.md
└── misc_logs/
    ├── vllm_server.log                               # 293K — Early standalone vLLM test
    └── monitor_output.log                            # 11K — Early Docker monitoring
```

### Run 8 Ray Session ↔ Hydra Config Mapping

| Segment | Ray Session | Hydra Config | Steps | Size (compressed) |
|---------|------------|-------------|-------|--------------------|
| Start | `session_2026-02-22_23-53-32` | `outputs/2026-02-22/23-54-23/` | 0 → ~60 | 35M |
| Restart 1 | `session_2026-02-23_15-03-44` | `outputs/2026-02-23/15-04-21/` | ~60 → ~140 | 72M |
| Restart 2 | `session_2026-02-24_09-25-28` | `outputs/2026-02-24/09-25-56/` | ~140 → 200 | 101M |

### Configuration Evolution (from Hydra overrides)

| Date Range | Sessions | batch_size | TP | Architecture |
|-----------|----------|------------|-----|--------------|
| Feb 10 (early) | 11 | 8 | 8 | Single-node, Option A |
| Feb 10 (late)–11 | 16 | 2 | 8 | Reduced batch, Option A |
| Feb 12 | 15 | 4 | 8 | Option A experiments |
| Feb 13–24 | 73 | 4 | 16 (mostly) | Option B production |

### Extracting Logs

```bash
# Extract a specific Ray session
cd /home/claude/work/rllm-origin/logs_backup/ray_sessions/
tar xzf session_2026-02-24_09-25-28.tar.gz

# View Hydra config for a specific launch
cat hydra_configs/outputs/2026-02-22/23-54-23/.hydra/overrides.yaml
```

### Not Backed Up

- **Checkpoints** (2.2TB in `checkpoints/`): Too large. 7 checkpoint dirs exist: `deepswe-full`, `deepswe-minimal`, `deepswe-minimal-optionb`, `deepswe-r2egym`, `deepswe-reproduction`, `merge_step164`, `merge_step198`
- **Original Ray logs** (5.7GB in `/tmp/ray/`): Ephemeral, will be lost on reboot. The compressed backups above are the preserved copies.
- **wandb logs**: Training ran with `WANDB_MODE=disabled`

---

## Related Documents

- [deepswe-youyun-kubernetes-reproduction.md](deepswe-youyun-kubernetes-reproduction.md) — K8s 8-node (64 GPU) reproduction plan
- [logs_backup/README.md](../../logs_backup/README.md) — Full log archive index (257MB)
- PPIO experiment docs: `/home/claude/work/rllm/ppio/docs/` (19 files, v1~v6.4)
- Diary entries: `diary-job/2026/0212.md` through `0225.md`
- Local memo: `diary-job/memo/rl-finetune/deepswe-youyun-reproduction.md`
