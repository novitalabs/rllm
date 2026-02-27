#!/bin/bash
# Build the torch ABI compatibility shim for vllm-flash-attn on torch 2.10.x
# Usage: bash build_shim.sh [output_path]

set -e

OUTPUT="${1:-/tmp/torch_compat_shim.so}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${SCRIPT_DIR}/../patches/torch_compat_shim.cpp"

TORCH_LIB=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path.replace('share/cmake/Torch','lib'))" 2>/dev/null || echo "/usr/local/lib/python3.10/dist-packages/torch/lib")

echo "Building ABI shim..."
echo "  Source: $SRC"
echo "  Output: $OUTPUT"
echo "  Torch lib: $TORCH_LIB"

g++ -shared -fPIC -o "$OUTPUT" "$SRC" \
    -L"$TORCH_LIB" -lc10 -lc10_cuda \
    -Wl,-rpath,"$TORCH_LIB"

echo "Built successfully: $OUTPUT"
echo "Usage: export LD_PRELOAD=$OUTPUT"
