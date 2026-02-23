"""
Distributed trajectory worker for multi-node rollout execution.

Creates Ray actors (one per node) that each run a local AsyncAgentExecutionEngine,
distributing trajectory generation across all nodes instead of running everything
on the head node.
"""

import asyncio
import importlib
import logging
import traceback
from queue import Queue
from threading import Thread

import ray

logger = logging.getLogger(__name__)


class StubRolloutManager:
    """Minimal stand-in for AgentLoopManager on worker nodes.

    Workers don't manage vLLM lifecycle — the head node handles wake_up/sleep.
    This stub provides server_handles for AsyncLLMServerManager and an empty
    rollout_replicas list so VerlEngine.wake_up()/sleep() become no-ops.
    """

    def __init__(self, server_handles):
        self.server_handles = server_handles
        self.rollout_replicas = []  # No-op wake_up/sleep on workers


@ray.remote
class DistributedTrajectoryWorker:
    """Ray actor that runs trajectory generation on a single node.

    Each worker creates a local AsyncAgentExecutionEngine with a StubRolloutManager,
    generates trajectories for its assigned chunk, and returns results with
    globally-correct indices.
    """

    def __init__(
        self,
        config,
        server_handles,
        env_module: str,
        env_class_name: str,
        agent_module: str,
        agent_class_name: str,
        worker_id: int,
        n_parallel_agents: int,
    ):
        self.config = config
        self.worker_id = worker_id
        self.n_parallel_agents = n_parallel_agents

        # Import env/agent classes dynamically (avoids Ray serialization issues)
        env_mod = importlib.import_module(env_module)
        self.env_class = getattr(env_mod, env_class_name)

        agent_mod = importlib.import_module(agent_module)
        self.agent_class = getattr(agent_mod, agent_class_name)

        # Load tokenizer locally
        from transformers import AutoTokenizer

        model_path = config.actor_rollout_ref.model.path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Create local engine with stub rollout manager
        from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine

        stub_manager = StubRolloutManager(server_handles)
        self.engine = AsyncAgentExecutionEngine(
            rollout_engine=stub_manager,
            config=config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=model_path,
            max_steps=config.rllm.agent.max_steps,
            max_response_length=config.data.max_response_length,
            max_prompt_length=config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args={},
            env_class=self.env_class,
            env_args={},
            enforce_max_prompt_length=config.rllm.stepwise_advantage.enable,
            trajectory_timeout=config.rllm.agent.trajectory_timeout,
            overlong_filter=config.rllm.agent.get("overlong_filter", False),
            disable_thinking=config.rllm.disable_thinking,
            n_parallel_agents=n_parallel_agents,
        )

        logger.info(f"DistributedTrajectoryWorker {worker_id} initialized with {n_parallel_agents} parallel agents")

    def generate_trajectories(
        self,
        env_args_list: list[dict],
        full_agent_args: dict,
        base_env_args: dict,
        meta_info: dict,
        mode: str,
        idx_offset: int,
    ) -> list[dict]:
        """Generate trajectories for the assigned chunk of environments.

        Args:
            env_args_list: List of per-env argument dicts for this worker's chunk.
            full_agent_args: Agent constructor kwargs (same for all).
            base_env_args: Base env args merged into each env's args.
            meta_info: veRL generation metadata.
            mode: "Token" or "Step".
            idx_offset: Global index offset for this worker's chunk.

        Returns:
            List of trajectory result dicts with globally-correct idx values.
        """
        import json

        # Create envs and agents for this chunk
        envs = []
        for args in env_args_list:
            if isinstance(args, str):
                args = json.loads(args)
            envs.append(self.env_class.from_dict({**args, **base_env_args}))

        agents = [self.agent_class(**full_agent_args) for _ in range(len(envs))]

        self.engine.update_envs_and_agents(envs, agents)

        # Run trajectory generation in a thread (async -> sync bridge)
        results = []
        queue = Queue()

        def runner():
            try:
                async def consume():
                    async for item in self.engine.trajectory_generator(
                        timing_raw={}, mode=mode, meta_info=meta_info
                    ):
                        queue.put(item)
                    queue.put(None)

                asyncio.run(consume())
            except Exception as e:
                logger.error(f"Worker {self.worker_id} runner thread crashed: {e}")
                traceback.print_exc()
                queue.put(None)

        thread = Thread(target=runner, daemon=True)
        thread.start()

        while True:
            item = queue.get()
            if item is None:
                break
            # Remap local idx -> global idx
            item["idx"] = item["idx"] + idx_offset
            results.append(item)

        thread.join(timeout=10)
        return results
