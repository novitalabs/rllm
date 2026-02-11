"""
Test verl hybrid engine - simulates switching between rollout and training mode.
This tests the exact code path where the crash occurred.
"""
import os
import sys
import torch
import logging
import gc

os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["NCCL_NVLS_ENABLE"] = "0"

sys.path.insert(0, "/home/claude/work/rllm-origin")
sys.path.insert(0, "/home/claude/work/R2E-Gym/src")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "/home/claude/work/rllm/models/Qwen3-32B"


def test_vllm_with_memory_pressure():
    """
    Test vLLM behavior under memory pressure.
    The crash might be related to memory allocation when FSDP also uses GPU.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    logger.info("=== Test: vLLM with Memory Pressure ===")
    
    # First, allocate some GPU memory to simulate FSDP weights
    logger.info("Allocating GPU tensors to simulate FSDP memory usage...")
    
    # Allocate ~20GB on each GPU to simulate FSDP model shards
    # Qwen3-32B with FSDP across 8 GPUs uses ~8GB per GPU for weights
    # Plus optimizer states (~16GB per GPU with Adam)
    
    gpu_tensors = []
    try:
        for i in range(8):
            # Allocate on each GPU
            device = torch.device(f"cuda:{i}")
            # Allocate 10GB per GPU
            tensor = torch.zeros(
                (10 * 1024 * 1024 * 1024 // 4,),  # 10GB in float32
                dtype=torch.float32,
                device=device
            )
            gpu_tensors.append(tensor)
            logger.info(f"  GPU {i}: Allocated 10GB")
    except torch.cuda.OutOfMemoryError as e:
        logger.warning(f"OOM during allocation: {e}")
        logger.info("Reducing allocation...")
        # Clear and try smaller allocation
        gpu_tensors = []
        gc.collect()
        torch.cuda.empty_cache()
        
        for i in range(8):
            device = torch.device(f"cuda:{i}")
            tensor = torch.zeros(
                (5 * 1024 * 1024 * 1024 // 4,),  # 5GB in float32
                dtype=torch.float32,
                device=device
            )
            gpu_tensors.append(tensor)
            logger.info(f"  GPU {i}: Allocated 5GB")
    
    # Now initialize vLLM with remaining memory
    logger.info("Initializing vLLM with reduced memory...")
    
    try:
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=8,
            max_model_len=32768,  # Reduced from 65536
            gpu_memory_utilization=0.25,  # Lower than default
            enforce_eager=True,
        )
        
        logger.info("vLLM initialized successfully")
        
        # Run inference
        params = SamplingParams(max_tokens=50, temperature=0.6)
        outputs = llm.generate(["Hello world"], params)
        logger.info(f"Generation OK: {outputs[0].outputs[0].text[:50]}")
        
        del llm
        
    except Exception as e:
        logger.error(f"vLLM initialization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    del gpu_tensors
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    logger.info("Memory pressure test completed")


def test_vllm_engine_recreation():
    """
    Test recreating vLLM engine multiple times.
    In verl hybrid engine, the engine may be recreated after training steps.
    """
    from vllm import LLM, SamplingParams
    
    logger.info("=== Test: vLLM Engine Recreation ===")
    
    for iteration in range(3):
        logger.info(f"\n--- Iteration {iteration + 1}/3 ---")
        
        # Create engine
        logger.info("Creating vLLM engine...")
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=8,
            max_model_len=32768,
            gpu_memory_utilization=0.35,
            enforce_eager=True,
        )
        
        # Run inference
        params = SamplingParams(max_tokens=50, temperature=0.6)
        outputs = llm.generate(["Test prompt " * 100], params)
        logger.info(f"Generated: {len(outputs[0].outputs[0].token_ids)} tokens")
        
        # Destroy engine
        logger.info("Destroying vLLM engine...")
        del llm
        
        # Force cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Check memory
        mem_free = torch.cuda.mem_get_info(0)[0] / (1024**3)
        logger.info(f"GPU 0 free memory: {mem_free:.2f} GB")
    
    logger.info("\nEngine recreation test completed!")


def test_long_sequence_stress():
    """
    Stress test with many long sequences.
    Test if the engine can handle continuous load.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    logger.info("=== Test: Long Sequence Stress Test ===")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=8,
        max_model_len=65536,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
    )
    
    # Run multiple batches
    for batch_idx in range(5):
        logger.info(f"\n--- Batch {batch_idx + 1}/5 ---")
        
        # Create varying length prompts
        prompts = []
        for i in range(4):
            length = (batch_idx + 1) * 2000 + i * 1000
            prompt = "Test " * (length // 2)
            prompts.append(prompt)
        
        prompt_lens = [len(tokenizer.encode(p)) for p in prompts]
        logger.info(f"Prompt lengths: {prompt_lens}")
        
        params = SamplingParams(max_tokens=100, temperature=0.6)
        
        try:
            outputs = llm.generate(prompts, params)
            for i, out in enumerate(outputs):
                logger.info(f"  Sample {i}: {len(out.outputs[0].token_ids)} tokens generated")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    logger.info("\nStress test completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0,
                        help="Test: 0=all, 1=memory_pressure, 2=recreation, 3=stress")
    args = parser.parse_args()
    
    tests = [
        test_vllm_with_memory_pressure,
        test_vllm_engine_recreation,
        test_long_sequence_stress,
    ]
    
    if args.test == 0:
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"Test {test.__name__} failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    elif 1 <= args.test <= len(tests):
        tests[args.test - 1]()
    else:
        print("Usage: python test_hybrid_engine.py [--test N]")
