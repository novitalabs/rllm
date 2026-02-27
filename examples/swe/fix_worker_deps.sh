#!/bin/bash
set -ex

export https_proxy=http://127.0.0.1:1083
export http_proxy=http://127.0.0.1:1083

# Install ray and other rllm deps that were missed due to cd failure
pip3 install ray datasets pandas polars pillow numpy sympy pylatexenc \
    "antlr4-python3-runtime==4.9.3" mcp eval-protocol hydra-core openai \
    fastapi uvicorn tqdm pyyaml pydantic wrapt "asgiref>=3.7.0" wandb

# Install rllm wheel
pip3 install /tmp/rllm-0.2.1-py3-none-any.whl --no-deps

echo "Fix complete!"
