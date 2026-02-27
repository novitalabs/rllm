#!/bin/bash
set -ex

export https_proxy=http://127.0.0.1:1083
export http_proxy=http://127.0.0.1:1083

PROJECT_DIR="/root/develop/ref/rllm"

# Install all packages needed for worker nodes
# vllm/verl with --no-deps to avoid torch downgrade
pip3 install "vllm==0.11.0" --no-deps
pip3 install "verl==0.6.1" --no-deps

# Install runtime deps
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
    "openai-harmony>=0.0.3" "numba==0.61.2" \
    "transformers>=4.55.0,<5.0.0"

# flash-attn with parallel build
export MAX_JOBS=64
pip3 install "flash-attn>=2.8.1" --no-build-isolation

# swe deps
pip3 install docker kubernetes swebench

# rllm
cd "$PROJECT_DIR"
pip3 install -e . --no-deps
pip3 install datasets pandas polars pillow ray numpy sympy pylatexenc \
    "antlr4-python3-runtime==4.9.3" mcp eval-protocol hydra-core openai \
    fastapi uvicorn tqdm pyyaml pydantic wrapt "asgiref>=3.7.0" wandb

echo "Worker installation complete!"
