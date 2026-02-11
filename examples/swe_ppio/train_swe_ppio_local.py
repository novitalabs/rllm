#!/usr/bin/env python3
"""
Local SWE-bench Training with PPIO Sandbox

A simpler training script that doesn't require a full Ray cluster.
Uses HTTP API for inference (compatible with vLLM OpenAI server).

This is useful for:
- Development and testing
- Small-scale experiments
- Using existing inference servers

For distributed training, use train_swe_ppio.sh instead.
"""

import asyncio
import json
import logging
import os
import re
import requests
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
# Inference Configuration
# =============================================================================
INFERENCE_URL = "http://localhost:8000/v1/chat/completions"
MAX_TOKENS = 4096
TEMPERATURE = 1.0

# Rollout timeout to prevent hanging (45 min, safe within 1-hour sandbox limit)
ROLLOUT_TIMEOUT = 2700


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
# HTTP API Inference
# =============================================================================

async def generate_response(messages: List[Dict], inference_url: str = None) -> str:
    """Generate model response using HTTP API (vLLM OpenAI-compatible server)."""
    url = inference_url or INFERENCE_URL

    try:
        response = await asyncio.to_thread(
            requests.post,
            url,
            json={
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "stop": ["</function>"],
            },
            timeout=300,
            proxies={"http": None, "https": None},  # Disable proxy for localhost
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Remove <think>...</think> tags (Qwen3 reasoning)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = content.strip()

        # Add back stop string if present in action but not terminated
        if "<function=" in content and "</function>" not in content:
            content += "</function>"

        return content
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


# =============================================================================
# Training Loop
# =============================================================================

async def run_single_rollout(
    instance: Dict,
    rollout_idx: int,
    inference_url: str,
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
        obs, info = await asyncio.to_thread(env.reset)
        agent.update_from_env(obs["task_instruction"], 0, False, info)

        trajectory = []
        done = False
        step = 0

        while not done and step < max_steps:
            messages = agent.chat_completions
            response = await generate_response(messages, inference_url)
            action = agent.update_from_model(response)
            logger.info(f"Step {step+1}: {action.action[:200] if len(action.action) > 200 else action.action}...")
            obs, reward, done, info = await asyncio.to_thread(env.step, action.action)
            agent.update_from_env(obs, reward, done, info)

            trajectory.append({
                "action": action.action,
                "reward": reward,
                "done": done,
            })
            step += 1

        final_reward = env.compute_final_reward()
        total_reward = sum(t["reward"] for t in trajectory) + final_reward
        logger.info(f"Rollout {rollout_idx} completed: steps={step}, final_reward={final_reward}, total_reward={total_reward}")

        return {
            "reward": total_reward,
            "trajectory": trajectory,
            "instance_id": instance.get("instance_id", "unknown"),
            "rollout_idx": rollout_idx,
        }
    except Exception as e:
        logger.error(f"Rollout {rollout_idx} failed: {e}")
        import traceback
        traceback.print_exc()
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
    inference_url: str,
    group_size: int = 4,
    max_steps: int = 50,
    max_concurrent: int = 8,
):
    """Train on a batch of instances with parallel rollouts."""
    from asyncio import Semaphore

    sem = Semaphore(max_concurrent)

    async def limited_rollout(instance, idx):
        async with sem:
            try:
                return await asyncio.wait_for(
                    run_single_rollout(instance, idx, inference_url, max_steps),
                    timeout=ROLLOUT_TIMEOUT
                )
            except asyncio.TimeoutError:
                instance_id = instance.get("instance_id", "unknown")
                logger.error(f"Rollout {idx} for {instance_id} timed out after {ROLLOUT_TIMEOUT}s")
                return {
                    "reward": 0.0,
                    "trajectory": [],
                    "instance_id": instance_id,
                    "rollout_idx": idx,
                    "error": f"Timeout after {ROLLOUT_TIMEOUT}s",
                }

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


async def main():
    """Main training loop."""
    global INFERENCE_URL

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--inference-url", type=str, default=INFERENCE_URL, help="Inference server URL")
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Test without running rollouts")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Maximum concurrent rollouts")
    parser.add_argument("--instance", type=str, default=None, help="Run single instance (for testing)")
    args = parser.parse_args()

    INFERENCE_URL = args.inference_url

    logger.info(f"Starting SWE-PPIO local training")
    logger.info(f"Inference URL: {args.inference_url}")
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
                data = json.loads(line)
                if args.instance is None or data.get("instance_id") == args.instance:
                    instances.append(data)
                    if args.instance:
                        break
    logger.info(f"Loaded {len(instances)} instances")

    if not instances:
        logger.error(f"No instances found (filter: {args.instance})")
        return

    if args.dry_run:
        logger.info("Dry run mode - skipping rollouts")
        logger.info("Checkpoint manager and patterns demonstrated successfully")
        return

    # Test inference server
    try:
        test_resp = requests.post(
            args.inference_url,
            json={"messages": [{"role": "user", "content": "test"}], "max_tokens": 5},
            timeout=30,
            proxies={"http": None, "https": None},
        )
        logger.info(f"Inference server OK: {test_resp.status_code}")
    except Exception as e:
        logger.error(f"Inference server not available at {args.inference_url}: {e}")
        sys.exit(1)

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
                inference_url=args.inference_url,
                group_size=args.group_size,
                max_steps=args.max_steps,
                max_concurrent=args.max_concurrent,
            )

            # Compute metrics
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            logger.info(f"Batch {batch_idx + 1} - Avg reward: {avg_reward:.4f}")

            # Compute GRPO advantages
            if len(rewards) > 1:
                advantages = compute_grpo_advantages(rewards)
                logger.info(f"Advantages: min={min(advantages):.2f}, max={max(advantages):.2f}")

            # Record checkpoint
            checkpoint_manager.record_batch(state, batch_idx + 1, avg_reward)

        except Exception as e:
            logger.error(f"Error in batch {batch_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            logger.info("Attempting to continue with next batch...")
            continue

    logger.info("\nTraining complete!")
    logger.info(f"Best reward: {state.get('best_reward', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
