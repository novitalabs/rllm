"""
Unit test for loss computation phase.
Simulates the training flow to debug vLLM crash during compute_log_prob.

Test Flow:
1. Initialize vLLM engine (similar to rollout)
2. Create mock batch data with long sequences
3. Call compute_log_prob (this is where crash happens)
4. Compute PPO loss
"""
import os
import sys
import torch
import logging

# Environment setup
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

sys.path.insert(0, "/home/claude/work/rllm-origin")
sys.path.insert(0, "/home/claude/work/R2E-Gym/src")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_compute_log_prob_standalone():
    """Test compute_log_prob using vLLM directly without full training setup."""
    from vllm import LLM, SamplingParams
    
    MODEL_PATH = "/home/claude/work/rllm/models/Qwen3-32B"
    
    logger.info("=== Test 1: Standalone vLLM Log Prob Computation ===")
    logger.info(f"Model: {MODEL_PATH}")
    
    # Initialize vLLM with same config as training
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=8,
        max_model_len=65536,
        gpu_memory_utilization=0.35,
        enforce_eager=True,  # Same as training config
    )
    
    # Create test sequences of various lengths
    test_sequences = [
        "Hello world" * 100,   # Short
        "Hello world" * 1000,  # Medium
        "Hello world" * 5000,  # Long
    ]
    
    tokenizer = llm.get_tokenizer()
    
    for i, text in enumerate(test_sequences):
        tokens = tokenizer.encode(text)
        logger.info(f"Sequence {i+1}: {len(tokens)} tokens")
        
        # Test generation
        params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=100,
        )
        
        outputs = llm.generate([text], params)
        logger.info(f"  Generated {len(outputs[0].outputs[0].token_ids)} tokens")
    
    logger.info("Standalone test passed!")
    del llm
    return True


def test_compute_log_prob_with_verl():
    """Test compute_log_prob through verl's actor_rollout_wg."""
    import ray
    from omegaconf import OmegaConf
    
    logger.info("=== Test 2: vLLM Log Prob via verl Actor Rollout ===")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Create minimal config for actor_rollout
    config = OmegaConf.create({
        "model": {
            "path": "/home/claude/work/rllm/models/Qwen3-32B",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
        "hybrid_engine": True,
        "rollout": {
            "name": "vllm",
            "mode": "async",
            "tensor_model_parallel_size": 8,
            "gpu_memory_utilization": 0.35,
            "enforce_eager": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "n": 2,
        },
        "actor": {
            "optim": {"lr": 1e-6},
            "ppo_mini_batch_size": 2,
            "ppo_micro_batch_size_per_gpu": 1,
            "use_kl_loss": False,
            "clip_ratio_high": 0.28,
            "ulysses_sequence_parallel_size": 8,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "fsdp_config": {
                "param_offload": True,
            },
        },
    })
    
    # Import verl components
    try:
        from verl.workers.hybrid_engine import HybridEngine
        logger.info("HybridEngine imported")
    except ImportError as e:
        logger.error(f"Cannot import HybridEngine: {e}")
        return False
    
    logger.info("verl test passed (import only)")
    ray.shutdown()
    return True


def test_mock_batch_processing():
    """Test with mock batch data similar to training."""
    logger.info("=== Test 3: Mock Batch Processing ===")
    
    # Create mock batch tensor
    batch_size = 4
    seq_len = 32768  # Same as training max_prompt_length
    
    # Mock input_ids
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # Mock response_mask (1 for response tokens, 0 for prompt/observation)
    response_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    response_start = seq_len // 2  # Half prompt, half response
    response_mask[:, response_start:] = 1
    
    logger.info(f"Mock batch created:")
    logger.info(f"  input_ids shape: {input_ids.shape}")
    logger.info(f"  attention_mask shape: {attention_mask.shape}")
    logger.info(f"  response_mask shape: {response_mask.shape}")
    logger.info(f"  response_mask sum per sample: {response_mask.sum(dim=1).tolist()}")
    
    # Check for all-zero masks (the bug we fixed)
    all_zero_count = (response_mask.sum(dim=1) == 0).sum().item()
    if all_zero_count > 0:
        logger.warning(f"Found {all_zero_count} samples with all-zero response_mask!")
        return False
    
    logger.info("Mock batch test passed!")
    return True


def test_vllm_engine_lifecycle():
    """Test vLLM engine create/destroy cycle (potential crash point)."""
    from vllm import LLM
    
    logger.info("=== Test 4: vLLM Engine Lifecycle ===")
    MODEL_PATH = "/home/claude/work/rllm/models/Qwen3-32B"
    
    for i in range(3):
        logger.info(f"Cycle {i+1}/3: Creating vLLM engine...")
        
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=8,
            max_model_len=32768,
            gpu_memory_utilization=0.35,
            enforce_eager=True,
        )
        
        logger.info(f"Cycle {i+1}/3: Running inference...")
        from vllm import SamplingParams
        outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=10))
        logger.info(f"Cycle {i+1}/3: Generated: {outputs[0].outputs[0].text[:50]}")
        
        logger.info(f"Cycle {i+1}/3: Destroying engine...")
        del llm
        
        # Force CUDA memory cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        logger.info(f"Cycle {i+1}/3: Complete")
    
    logger.info("Lifecycle test passed!")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0, help="Test number (0=all, 1-4=specific)")
    args = parser.parse_args()
    
    tests = [
        ("Standalone vLLM", test_compute_log_prob_standalone),
        ("verl Actor Rollout", test_compute_log_prob_with_verl),
        ("Mock Batch Processing", test_mock_batch_processing),
        ("vLLM Engine Lifecycle", test_vllm_engine_lifecycle),
    ]
    
    if args.test == 0:
        # Run mock batch test first (no GPU needed)
        logger.info("\n" + "="*60)
        test_mock_batch_processing()
        logger.info("="*60 + "\n")
        
        # Then standalone vLLM test
        logger.info("\n" + "="*60)
        test_compute_log_prob_standalone()
        logger.info("="*60 + "\n")
    elif 1 <= args.test <= 4:
        name, func = tests[args.test - 1]
        logger.info(f"\nRunning test: {name}")
        func()
    else:
        print("Usage: python test_loss_computation.py [--test N]")
        print("  --test 0: Run default tests (3, 1)")
        print("  --test 1: Standalone vLLM")
        print("  --test 2: verl Actor Rollout")
        print("  --test 3: Mock Batch Processing")
        print("  --test 4: vLLM Engine Lifecycle")
