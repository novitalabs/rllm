#!/usr/bin/env python3
"""
SWE-bench RL Training using PPIO Sandbox and verl backend.

This script trains an SWE agent on SWE-bench tasks using:
- PPIO sandbox for code execution (vs Docker in original SWEEnv)
- verl backend for distributed PPO training (vs Tinker)
- SWEAgent for agent policy

Based on rft-tinker v3 patterns:
- Checkpoint resumption
- Retry logic
- Proper state management
"""

import hydra
from omegaconf import DictConfig

from rllm.agents.swe_agent import SWEAgent
from rllm.data import DatasetRegistry
from rllm.environments.swe_ppio.swe_ppio import SWEBenchPPIOEnv
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config: DictConfig):
    """Main training function using verl backend with PPIO sandbox."""
    
    # Load datasets
    # R2E_Gym_Subset for training (smaller, faster iteration)
    # SWE_Bench_Verified for validation (full benchmark)
    train_dataset = DatasetRegistry.load_dataset("R2E_Gym_Subset", "train")
    val_dataset = DatasetRegistry.load_dataset("SWE_Bench_Verified", "test")
    
    # Environment args for PPIO sandbox
    env_args = {
        "timeout": 3600,           # 1 hour timeout per sandbox
        "workdir": "/testbed",     # Standard SWE-bench workdir
        "use_pool": True,          # Use sandbox pool to avoid 429 rate limits
        "pool_size": 32,           # Pool size matching batch size
    }
    
    # Agent args (r2egym scaffold for XML function format)
    agent_args = {
        "use_fn_calling": False,       # Use XML format, not function calling
        "format_model_response": False,
        "scaffold": "r2egym",          # R2E-Gym compatible format
    }
    
    # Create trainer with verl backend
    trainer = AgentTrainer(
        agent_class=SWEAgent,
        env_class=SWEBenchPPIOEnv,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        backend="verl",  # Use verl instead of tinker
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
