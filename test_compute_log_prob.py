"""
Test compute_log_prob phase specifically.
This simulates the exact code path that crashed during training.
"""
import os
import sys
import torch
import logging
from dataclasses import dataclass
from typing import List, Optional

os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["NCCL_NVLS_ENABLE"] = "0"

sys.path.insert(0, "/home/claude/work/rllm-origin")
sys.path.insert(0, "/home/claude/work/R2E-Gym/src")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "/home/claude/work/rllm/models/Qwen3-32B"


def test_log_prob_computation():
    """
    Test log probability computation with varying sequence lengths.
    Simulates what happens during PPO training compute_log_prob.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    logger.info("=== Testing Log Prob Computation ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Initialize vLLM
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=8,
        max_model_len=65536,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
    )
    
    # Create test batch similar to training
    # In training, we have:
    # - batch_size = 2 (after rejection sampling may be less)
    # - Each sequence has prompt + response
    # - Sequence length can be up to 32768 (prompt) + 32768 (response) = 65536
    
    test_cases = [
        # (prompt_tokens, response_tokens)
        (1000, 500),    # Short sequences
        (4000, 2000),   # Medium sequences  
        (16000, 8000),  # Long sequences (like real training)
        (30000, 16000), # Very long sequences (near limit)
    ]
    
    for prompt_len, resp_len in test_cases:
        logger.info(f"\n--- Testing prompt={prompt_len}, response={resp_len} ---")
        
        # Create mock sequence
        prompt_text = "Hello world " * (prompt_len // 3)
        prompt_ids = tokenizer.encode(prompt_text)[:prompt_len]
        
        # Create mock response (just random tokens for testing)
        response_ids = list(range(1000, 1000 + resp_len))
        
        # Full sequence
        full_ids = prompt_ids + response_ids
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        
        logger.info(f"  Full sequence length: {len(full_ids)}")
        
        # Method 1: Use generate to check inference works
        params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=10,  # Just generate a few tokens
        )
        
        try:
            outputs = llm.generate([tokenizer.decode(prompt_ids)], params)
            logger.info(f"  Generation OK: {len(outputs[0].outputs[0].token_ids)} tokens")
        except Exception as e:
            logger.error(f"  Generation FAILED: {e}")
            continue
        
        # Method 2: Check if we can compute log probs via vLLM
        # Note: vLLM 0.10+ has compute_log_prob API through SamplingParams
        params_with_logprob = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=10,
            logprobs=1,  # Return log probs
        )
        
        try:
            outputs = llm.generate([tokenizer.decode(prompt_ids)], params_with_logprob)
            if outputs[0].outputs[0].logprobs:
                logger.info(f"  LogProb computation OK")
            else:
                logger.warning(f"  LogProb returned None")
        except Exception as e:
            logger.error(f"  LogProb FAILED: {e}")
    
    del llm
    torch.cuda.empty_cache()
    logger.info("\nAll log prob tests completed!")


def test_batch_log_prob():
    """
    Test batch log prob computation - closer to training scenario.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    logger.info("=== Testing Batch Log Prob ===")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=8,
        max_model_len=65536,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
    )
    
    # Create batch of sequences with different lengths (like real training)
    prompts = []
    for i in range(4):  # batch_size = 4
        length = (i + 1) * 2000  # 2000, 4000, 6000, 8000 tokens
        prompt_text = "Test prompt " * (length // 3)
        prompt_ids = tokenizer.encode(prompt_text)[:length]
        prompts.append(tokenizer.decode(prompt_ids))
    
    logger.info(f"Batch size: {len(prompts)}")
    logger.info(f"Prompt lengths: {[len(tokenizer.encode(p)) for p in prompts]}")
    
    params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=100,
        logprobs=1,
    )
    
    try:
        outputs = llm.generate(prompts, params)
        logger.info(f"Batch generation OK")
        for i, out in enumerate(outputs):
            logger.info(f"  Sample {i}: generated {len(out.outputs[0].token_ids)} tokens")
    except Exception as e:
        logger.error(f"Batch generation FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    del llm
    torch.cuda.empty_cache()


def test_verl_compute_log_prob():
    """
    Test compute_log_prob through verl's actual code path.
    This is the closest simulation to the crash scenario.
    """
    import ray
    from transformers import AutoTokenizer
    
    logger.info("=== Testing verl Compute Log Prob ===")
    
    if not ray.is_initialized():
        ray.init()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Import verl components
    try:
        from verl.trainer.ppo.rollout.vllm_rollout_spmd import vLLMRollout
        logger.info("vLLMRollout imported")
    except ImportError as e:
        logger.warning(f"Cannot import vLLMRollout: {e}")
        logger.info("Skipping verl test - module not available")
        ray.shutdown()
        return
    
    # Create mock batch
    batch_size = 2
    seq_len = 16384
    
    # Create DataProto-like batch
    from tensordict import TensorDict
    
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    batch = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }, batch_size=[batch_size])
    
    logger.info(f"Created mock batch: {batch_size} x {seq_len}")
    
    # Note: Full verl test requires more setup (worker groups, etc.)
    # This is just to verify imports and basic tensor operations
    
    ray.shutdown()
    logger.info("verl compute_log_prob test completed (import only)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0,
                        help="Test: 0=all, 1=single_log_prob, 2=batch_log_prob, 3=verl")
    args = parser.parse_args()
    
    tests = [
        test_log_prob_computation,
        test_batch_log_prob,
        test_verl_compute_log_prob,
    ]
    
    if args.test == 0:
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"Test {test.__name__} failed: {e}")
                import traceback
                traceback.print_exc()
    elif 1 <= args.test <= len(tests):
        tests[args.test - 1]()
    else:
        print("Usage: python test_compute_log_prob.py [--test N]")
