#!/usr/bin/env python3
"""
Local SWE-bench Training with PPIO Sandbox

A simpler training script that doesn't require a full Ray cluster.
Uses the same patterns as rft-tinker v3:
- Checkpoint manager with JSON persistence
- Retry with exponential backoff
- GRPO advantage computation

This is useful for:
- Development and testing
- Small-scale experiments
- Single-GPU training

For distributed training, use train_swe_ppio.sh instead.
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Checkpoint Manager (from rft-tinker v3)
# =============================================================================

class CheckpointManager:
    """Manages checkpoint state and paths."""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.state_file = os.path.join(checkpoint_dir, "checkpoint_state.json")
        os.makedirs(checkpoint_dir, exist_ok=True)

    def load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {
            "last_batch": 0,
            "best_reward": None,
            "best_model_path": None,
            "training_history": [],
        }

    def save_state(self, state: Dict):
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Checkpoint state saved to {self.state_file}")

    def record_batch(self, state: Dict, batch: int, avg_reward: float, model_path: str = None):
        state["last_batch"] = batch
        state["training_history"].append({
            "batch": batch,
            "avg_reward": avg_reward,
            "model_path": model_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        if state["best_reward"] is None or avg_reward > state["best_reward"]:
            state["best_reward"] = avg_reward
            state["best_model_path"] = model_path
            logger.info(f"New best reward: {avg_reward:.4f}")

        self.save_state(state)


# =============================================================================
# Retry Wrapper (from rft-tinker v3)
# =============================================================================

async def with_retry(operation, max_retries: int = 3, base_delay: float = 5.0):
    """Execute operation with exponential backoff retry."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return await operation()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Operation failed after {max_retries} attempts")
    raise last_error


# =============================================================================
# GRPO Advantage Computation (from rft-tinker v3)
# =============================================================================

def compute_grpo_advantages(rewards: List[float]) -> List[float]:
    """Compute GRPO advantages with std normalization."""
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
    std_reward = max(std_reward, 1e-8)
    return [(r - mean_reward) / std_reward for r in rewards]


# =============================================================================
# Training Loop
# =============================================================================

async def run_single_rollout(
    instance: Dict,
    rollout_idx: int,
    model,
    tokenizer,
    max_steps: int = 50,
):
    """Run a single rollout for an instance."""
    from rllm.environments.swe_ppio.swe_ppio import SWEBenchPPIOEnv
    from rllm.agents.swe_agent import SWEAgent

    env = SWEBenchPPIOEnv(
        entry=instance,
        timeout=3600,
        workdir="/testbed",
        use_pool=True,
        pool_size=32,
    )
    agent = SWEAgent(
        use_fn_calling=False,
        scaffold="r2egym",
    )

    try:
        obs, info = env.reset()
        agent.update_from_env(obs["task_instruction"], 0, False, info)

        trajectory = []
        done = False
        step = 0

        while not done and step < max_steps:
            messages = agent.chat_completions
            response = await generate_response(model, tokenizer, messages)
            action = agent.update_from_model(response)
            obs, reward, done, info = env.step(action.action)
            agent.update_from_env(obs, reward, done, info)

            trajectory.append({
                "action": action.action,
                "reward": reward,
                "done": done,
            })
            step += 1

        final_reward = env.compute_final_reward()
        total_reward = sum(t["reward"] for t in trajectory) + final_reward

        return {
            "reward": total_reward,
            "trajectory": trajectory,
            "instance_id": instance.get("instance_id", "unknown"),
            "rollout_idx": rollout_idx,
        }
    except Exception as e:
        logger.error(f"Rollout {rollout_idx} failed: {e}")
        return {
            "reward": 0.0,
            "trajectory": [],
            "instance_id": instance.get("instance_id", "unknown"),
            "rollout_idx": rollout_idx,
            "error": str(e),
        }
    finally:
        env.close()


async def train_batch(
    batch_idx: int,
    instances: List[Dict],
    model,
    tokenizer,
    group_size: int = 4,
    max_steps: int = 50,
    max_concurrent: int = 8,
):
    """Train on a batch of instances with parallel rollouts."""
    from asyncio import Semaphore

    sem = Semaphore(max_concurrent)

    async def limited_rollout(instance, idx):
        async with sem:
            return await run_single_rollout(instance, idx, model, tokenizer, max_steps)

    # Create all rollout tasks
    tasks = []
    task_info = []  # (instance_id, task)
    for instance in instances:
        instance_id = instance.get("instance_id", "unknown")
        for g in range(group_size):
            task = limited_rollout(instance, g)
            tasks.append(task)
            task_info.append(instance_id)

    logger.info(f"Starting {len(tasks)} parallel rollouts (max_concurrent={max_concurrent})")

    # Run all rollouts concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    batch_rewards = []
    results_by_instance = {}

    for i, result in enumerate(results):
        instance_id = task_info[i]
        if instance_id not in results_by_instance:
            results_by_instance[instance_id] = []

        if isinstance(result, Exception):
            logger.error(f"Rollout {i} exception: {result}")
            results_by_instance[instance_id].append(0.0)
        else:
            results_by_instance[instance_id].append(result["reward"])
            batch_rewards.append(result["reward"])

    # Aggregate by instance
    batch_trajectories = []
    for instance_id, rewards in results_by_instance.items():
        batch_trajectories.append({
            "instance_id": instance_id,
            "rewards": rewards,
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        })

    return batch_rewards, batch_trajectories


async def generate_response(model, tokenizer, messages):
    """Generate model response using vLLM."""
    # Format messages into prompt
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate with vLLM
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=4096,
        stop=["</function>"],
    )

    outputs = model.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text

    # Add back stop string if present in action
    if "<function=" in response and "</function>" not in response:
        response += "</function>"

    return response


async def main():
    """Main training loop."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Test without model loading")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU memory utilization")
    parser.add_argument("--max-concurrent", type=int, default=16, help="Maximum concurrent rollouts")
    args = parser.parse_args()

    logger.info(f"Starting SWE-PPIO local training")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Config: batch_size={args.batch_size}, group_size={args.group_size}, max_steps={args.max_steps}")

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    state = checkpoint_manager.load_state()
    start_batch = state.get("last_batch", 0)

    if start_batch > 0:
        logger.info(f"Resuming from batch {start_batch + 1}")

    # Load dataset
    instances = []
    with open(args.data_path, "r") as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    logger.info(f"Loaded {len(instances)} instances")

    if args.dry_run:
        logger.info("Dry run mode - skipping model loading")
        logger.info("Checkpoint manager and patterns demonstrated successfully")
        return

    # Load model with vLLM
    from vllm import LLM
    from transformers import AutoTokenizer

    logger.info(f"Loading model {args.model_path}...")
    model = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    logger.info("Model loaded successfully")

    # Training loop
    for batch_idx in range(start_batch, args.num_batches):
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch {batch_idx + 1}/{args.num_batches}")
        logger.info(f"{'='*60}")

        try:
            # Get batch instances
            batch_start = (batch_idx * args.batch_size) % len(instances)
            batch_instances = []
            for i in range(args.batch_size):
                idx = (batch_start + i) % len(instances)
                batch_instances.append(instances[idx])

            # Run training batch
            rewards, trajectories = await train_batch(
                batch_idx=batch_idx,
                instances=batch_instances,
                model=model,
                tokenizer=tokenizer,
                group_size=args.group_size,
                max_steps=args.max_steps,
                max_concurrent=args.max_concurrent,
            )

            # Compute metrics
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            logger.info(f"Batch {batch_idx + 1} - Avg reward: {avg_reward:.4f}")

            # Compute GRPO advantages
            advantages = compute_grpo_advantages(rewards)
            logger.info(f"Advantages: min={min(advantages):.2f}, max={max(advantages):.2f}")

            # Record checkpoint
            checkpoint_manager.record_batch(state, batch_idx + 1, avg_reward)

        except Exception as e:
            logger.error(f"Error in batch {batch_idx + 1}: {e}")
            logger.info("Attempting to continue with next batch...")
            continue

    logger.info("\nTraining complete!")
    logger.info(f"Best reward: {state.get('best_reward', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
