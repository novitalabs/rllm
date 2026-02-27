#!/bin/bash
set -ex

export https_proxy=http://127.0.0.1:1083
export http_proxy=http://127.0.0.1:1083

PROJECT_DIR="/root/develop/ref/rllm"

# torch 2.10.0+cu128 and torchvision already installed on nodes

# Step 1: Install vllm without deps to avoid torch downgrade, then install its deps separately
pip3 install "vllm==0.11.0" --no-deps
pip3 install "verl==0.6.1" --no-deps

# Step 2: Install vllm/verl runtime deps (excluding torch which is already installed)
pip3 install \
    accelerate codetiming peft pybind11 tensordict tensorboard torchdata \
    qwen-vl-utils einops sentencepiece cachetools blake3 py-cpuinfo \
    prometheus_client "prometheus-fastapi-instrumentator>=7.0.0" \
    "lm-format-enforcer==0.11.3" "llguidance>=0.7.11,<0.8.0" \
    "outlines_core==0.2.11" "diskcache==5.6.3" "lark==1.2.2" \
    "xgrammar==0.1.25" partial-json-parser pyzmq msgspec gguf \
    "mistral_common>=1.8.2" "opencv-python-headless>=4.11.0" \
    "compressed-tensors==0.11.0" "depyf==0.19.0" cloudpickle watchfiles \
    python-json-logger scipy ninja pybase64 cbor2 setproctitle \
    "openai-harmony>=0.0.3" "numba==0.61.2"

# Step 3: Install flash-attn with parallel build (192 cores)
export MAX_JOBS=64
pip3 install "flash-attn>=2.8.1" --no-build-isolation

# Step 4: Install swe dependencies
pip3 install docker kubernetes swebench

# Step 5: Install rllm in editable mode (--no-deps to avoid torch conflict)
cd "$PROJECT_DIR"
pip3 install -e . --no-deps
# Install rllm's non-torch deps
pip3 install datasets pandas polars pillow ray numpy sympy pylatexenc \
    "antlr4-python3-runtime==4.9.3" mcp eval-protocol hydra-core openai \
    fastapi uvicorn tqdm pyyaml pydantic wrapt "asgiref>=3.7.0" wandb

# Step 6: Verify installation
python3 -c "
import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')
import vllm; print(f'vllm={vllm.__version__}')
import verl; print(f'verl={verl.__version__}')
import ray; print(f'ray={ray.__version__}')
import rllm; print(f'rllm imported OK')
"

echo "All dependencies installed successfully!"
