# DeepSWE Standard Training Pipeline Analysis

## Entry Point
**Script:** `examples/swe/train_deepswe_32b.sh`

## Overview

This document analyzes the standard DeepSWE training pipeline using Docker/r2egym backend.

---

## 1. Configuration Summary

```bash
# Environment
rllm.env.name=swe
rllm.agent.name=sweagent
rllm.agent.max_steps=50
rllm.agent.trajectory_timeout=5400

# Model
actor_rollout_ref.model.path=Qwen/Qwen3-32B

# Data
data.train_files=R2E_Gym_Subset.parquet
data.val_files=SWE_Bench_Verified.parquet
data.train_batch_size=8
data.max_prompt_length=4096
data.max_response_length=32768

# Distributed
trainer.n_gpus_per_node=8
trainer.nnodes=8  # 64 GPUs total
actor_rollout_ref.rollout.n=8
```

---

## 2. Module Dependency Tree

```
examples/swe/train_deepswe_32b.sh
└── python3 -m rllm.trainer.verl.train_agent_ppo
    ├── rllm/trainer/verl/train_agent_ppo.py (Entry Point)
    │   ├── run_ppo_agent(config)
    │   └── TaskRunner.run(config)
    │
    ├── rllm/trainer/verl/agent_ppo_trainer.py (Trainer)
    │   ├── AgentPPOTrainer
    │   │   ├── init_workers()
    │   │   ├── fit_agent()
    │   │   └── _validate_agent()
    │   └── AsyncAgentExecutionEngine
    │
    ├── rllm/trainer/env_agent_mappings.py (Registry)
    │   ├── ENV_CLASS_MAPPING["swe"] → SWEEnv
    │   └── AGENT_CLASS_MAPPING["sweagent"] → SWEAgent
    │
    ├── rllm/environments/swe/swe.py (Environment)
    │   ├── SWEEnv(BaseEnv)
    │   │   ├── reset() → r2egym.RepoEnv
    │   │   ├── step(action)
    │   │   └── compute_final_reward()
    │   └── Dependencies:
    │       └── r2egym (external package)
    │
    ├── rllm/agents/swe_agent.py (Agent)
    │   ├── SWEAgent(BaseAgent)
    │   │   ├── reset()
    │   │   ├── update_from_env()
    │   │   └── update_from_model()
    │   └── parse_xml_response()
    │
    ├── rllm/engine/agent_execution_engine.py (Execution)
    │   ├── AsyncAgentExecutionEngine
    │   │   ├── trajectory_generator()
    │   │   └── run_agent_trajectory_async()
    │   └── compute_trajectory_reward()
    │
    ├── rllm/engine/rollout/verl_engine.py (vLLM)
    │   └── VerlEngine.get_model_response()
    │
    └── rllm/parser/chat_template_parser.py (Tokenization)
        └── ChatTemplateParser.parse()
```

---

## 3. Core Modules Detail

### 3.1 Entry Point: `train_agent_ppo.py`

**File:** `rllm/trainer/verl/train_agent_ppo.py`

**Key Functions:**
- `main(config)` - Hydra decorator entry
- `run_ppo_agent(config)` - Initialize Ray cluster
- `TaskRunner.run(config)` - Main training orchestration

**Dependencies:**
- hydra-core (config management)
- ray (distributed computing)
- verl (RL framework)

### 3.2 Trainer: `agent_ppo_trainer.py`

**File:** `rllm/trainer/verl/agent_ppo_trainer.py`

**Class:** `AgentPPOTrainer`

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `init_workers()` | Setup vLLM, actor, critic workers |
| `fit_agent()` | Main training loop |
| `_transform_agent_trajectories()` | Convert trajectories to DataProto |
| `_validate_agent()` | Run validation pass |

**Training Loop:**
```python
for epoch in range(total_epochs):
    for batch in train_dataloader:
        # 1. Initialize envs/agents
        envs, agents = init_envs_and_agents(batch)

        # 2. Generate trajectories
        trajectories = engine.trajectory_generator(envs, agents)

        # 3. Transform to tensors
        data_proto = _transform_agent_trajectories(trajectories)

        # 4. Compute advantages
        batch = compute_advantage(data_proto, gamma, lam)

        # 5. PPO update
        actor_loss = update_actor(batch)
        critic_loss = update_critic(batch)
```

### 3.3 Environment Registration: `env_agent_mappings.py`

**File:** `rllm/trainer/env_agent_mappings.py`

**Registry:**
```python
ENV_CLASSES = {
    "swe": safe_import("rllm.environments.swe.swe", "SWEEnv"),
    "swe_ppio": safe_import("rllm.environments.swe_ppio.swe_ppio", "SWEBenchPPIOEnv"),
    "swe_ppio_multistep": safe_import(..., "SWEBenchPPIOMultiStepEnv"),
}

AGENT_CLASSES = {
    "sweagent": safe_import("rllm.agents.swe_agent", "SWEAgent"),
}
```

### 3.4 SWE Environment: `swe.py`

**File:** `rllm/environments/swe/swe.py`

**Class:** `SWEEnv(BaseEnv)`

**Key Methods:**
```python
def reset(self) -> tuple[str, dict]:
    """Initialize r2egym RepoEnv with task data."""
    self.repo_env = r2egym.RepoEnv(
        repo=self.entry["repo"],
        commit=self.entry["base_commit"],
        ...
    )
    return observation, info

def step(self, action: str) -> tuple[str, float, bool, dict]:
    """Execute action in r2egym environment."""
    obs, reward, done, info = self.repo_env.step(action)
    return obs, reward, done, info

def compute_final_reward(self) -> float:
    """Get final test-based reward from r2egym."""
    return self.repo_env.compute_reward()

@staticmethod
def from_dict(info: dict) -> "SWEEnv":
    """Factory method for creating from task data."""
    return SWEEnv(entry=info)
```

**r2egym Integration:**
- Uses r2egym for Docker container management
- Test execution via r2egym's built-in evaluator
- Reward based on FAIL_TO_PASS test results

### 3.5 SWE Agent: `swe_agent.py`

**File:** `rllm/agents/swe_agent.py`

**Class:** `SWEAgent(BaseAgent)`

**Response Format:**
```xml
<function=execute_bash>
<parameter=command>ls -la /testbed</parameter>
</function>
```

**Supported Tools:**
| Tool | Purpose |
|------|---------|
| `execute_bash` | Run bash commands |
| `str_replace_editor` | File view/edit/create |
| `search` | Text search in files |
| `submit` | Submit solution |

### 3.6 Execution Engine: `agent_execution_engine.py`

**File:** `rllm/engine/agent_execution_engine.py`

**Class:** `AsyncAgentExecutionEngine`

**Async Execution Flow:**
```python
async def run_agent_trajectory_async(idx, env, agent):
    # Reset
    obs, info = await loop.run_in_executor(executor, env.reset)

    # Step loop
    for step in range(max_steps):
        # Get model response
        response = await vllm_engine.generate_async(prompt)

        # Parse action
        action = agent.update_from_model(response)

        # Execute in environment
        obs, reward, done, info = await loop.run_in_executor(
            executor, env.step, action
        )

        if done:
            break

    # Final reward
    final_reward = await loop.run_in_executor(
        executor, env.compute_final_reward
    )

    return trajectory
```

### 3.7 vLLM Engine: `verl_engine.py`

**File:** `rllm/engine/rollout/verl_engine.py`

**Class:** `VerlEngine`

**Model Response Generation:**
```python
def get_model_response(self, messages, sampling_params):
    # Parse messages to prompt
    prompt = self.chat_parser.parse(messages)

    # Tokenize
    prompt_ids = self.tokenizer.encode(prompt)

    # Generate via vLLM
    output = self.vllm_manager.generate(
        prompt_ids,
        temperature=sampling_params.temperature,
        max_tokens=sampling_params.max_tokens,
    )

    return ModelOutput(
        text=output.text,
        prompt_ids=prompt_ids,
        completion_ids=output.token_ids,
        logprobs=output.logprobs,
    )
```

---

## 4. Data Flow

### 4.1 Data Preparation

**Input:** HuggingFace SWE datasets
**Output:** Parquet files with structure:

```python
{
    "data_source": "swe",
    "prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem_statement}
    ],
    "extra_info": json.dumps({
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "abc123...",
        "FAIL_TO_PASS": ["tests/test.py::test_func"],
        "PASS_TO_PASS": [...],
        "problem_statement": "...",
    })
}
```

### 4.2 Training Data Flow

```
Parquet File
    ↓
DataLoader (batch of extra_info dicts)
    ↓
SWEEnv.from_dict(extra_info)
    ↓
env.reset() → r2egym.RepoEnv initialization
    ↓
Agent Loop (50 steps max):
    ├─ Model generates action
    ├─ env.step(action) → r2egym execution
    └─ Agent receives observation
    ↓
env.compute_final_reward() → Test results
    ↓
Trajectory (prompt_ids, response_ids, rewards)
    ↓
PPO Update
```

---

## 5. Reward Calculation

### 5.1 Per-Step Rewards
- Usually 0.0 during exploration
- Sparse reward only at submission

### 5.2 Final Reward
```python
def compute_final_reward(self):
    # r2egym runs tests
    test_results = self.repo_env.run_tests()

    # Calculate based on FAIL_TO_PASS
    if test_results.all_pass:
        return 1.0
    elif test_results.partial_pass:
        return passed_count / total_count
    else:
        return 0.0
```

### 5.3 Advantage Estimation
```python
# GAE (Generalized Advantage Estimation)
advantages = compute_advantage(
    batch,
    gamma=0.99,   # Discount factor
    lam=0.95      # GAE lambda
)
```

---

## 6. Key Configuration Parameters

### 6.1 Environment Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `rllm.env.name` | `swe` | Environment class |
| `rllm.agent.name` | `sweagent` | Agent class |
| `rllm.agent.max_steps` | `50` | Max steps per episode |
| `rllm.agent.trajectory_timeout` | `5400` | Episode timeout (90 min) |

### 6.2 Model Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `model.path` | `Qwen/Qwen3-32B` | Model checkpoint |
| `rollout.n` | `8` | Rollouts per sample |
| `rollout.temperature` | `1.0` | Sampling temperature |
| `rollout.tensor_model_parallel_size` | `8` | TP degree |

### 6.3 Training Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `train_batch_size` | `8` | Samples per batch |
| `max_prompt_length` | `4096` | Max prompt tokens |
| `max_response_length` | `32768` | Max response tokens |
| `actor.optim.lr` | `1e-6` | Learning rate |
| `actor.clip_ratio_high` | `0.28` | PPO clip ratio |

---

## 7. External Dependencies

### 7.1 r2egym
- Docker-based code execution environment
- Test runner for SWE-bench tasks
- Repository management (clone, checkout)

### 7.2 verl
- Distributed RL training framework
- FSDP/Megatron support
- vLLM integration

### 7.3 vLLM
- High-throughput LLM inference
- Tensor parallel support
- Async generation

---

## 8. Limitations

1. **Infrastructure:** Requires Docker and r2egym setup
2. **Scaling:** Limited by local compute resources
3. **Speed:** Docker container startup adds latency
4. **Cost:** Fixed infrastructure cost regardless of usage

---

## 9. File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `train_agent_ppo.py` | ~200 | Entry point, Ray setup |
| `agent_ppo_trainer.py` | ~600 | Main training loop |
| `env_agent_mappings.py` | ~50 | Registry |
| `swe.py` | ~150 | Environment wrapper |
| `swe_agent.py` | ~200 | Agent implementation |
| `agent_execution_engine.py` | ~400 | Async trajectory |
| `verl_engine.py` | ~150 | vLLM wrapper |
