# R2E-Gym PPIO Sandbox Templates

Custom PPIO sandbox templates optimized for R2E-Gym dataset training.

## Available Templates

| Template | Best For | Setup Time | Description |
|----------|----------|------------|-------------|
| `r2e-gym-base` | General | ~20-40s | Common dependencies for most repos |
| `r2e-gym-scientific` | numpy, pandas, orange3 | ~15-30s | Pre-built scientific stack |
| `r2e-gym-pillow` | pillow | ~15-25s | Image processing libraries |

### Repo → Template Mapping

| Repo | Recommended Template | Without Template | With Template |
|------|---------------------|-----------------|---------------|
| orange3 | `r2e-gym-scientific` | 134s | ~30s |
| pillow | `r2e-gym-pillow` | 81s | ~20s |
| pandas | `r2e-gym-scientific` | 67s | ~25s |
| numpy | `r2e-gym-scientific` | 64s | ~20s |
| aiohttp | `r2e-gym-base` | 60s | ~30s |
| datalad | `r2e-gym-base` | 41s | ~25s |
| scrapy | `r2e-gym-base` | 36s | ~20s |
| pyramid | `r2e-gym-base` | 27s | ~15s |
| coveragepy | `r2e-gym-base` | 25s | ~15s |
| tornado | `r2e-gym-base` | 23s | ~15s |

## Files

- `r2e-gym-base.Dockerfile` - General template
- `r2e-gym-scientific.Dockerfile` - Scientific computing template
- `r2e-gym-pillow.Dockerfile` - Image processing template
- `build_template.sh` - Build script for all templates
- `build_r2e_gym_template.sh` - Legacy build script (base only)

## Building Templates

### Prerequisites

1. Docker installed and running
2. Node.js and npm installed
3. PPIO access token

### Build Commands

```bash
# Set access token
export PPIO_ACCESS_TOKEN="your_token"

# Build a single template
./build_template.sh r2e-gym-base
./build_template.sh r2e-gym-scientific
./build_template.sh r2e-gym-pillow

# Build all templates
./build_template.sh all
```

## Usage

### Environment Variable

```bash
export PPIO_SANDBOX_TEMPLATE="r2e-gym-scientific"
python train_agent.py ...
```

### In Python

```python
from ppio_sandbox import Sandbox

# Use specific template
sandbox = Sandbox.create(
    template="r2e-gym-scientific",
    api_key="..."
)
```

### Dynamic Selection Based on Repo

```python
REPO_TEMPLATE_MAP = {
    'numpy': 'r2e-gym-scientific',
    'pandas': 'r2e-gym-scientific',
    'orange3': 'r2e-gym-scientific',
    'pillow': 'r2e-gym-pillow',
    # Others use r2e-gym-base
}

def get_template(repo_name: str) -> str:
    return REPO_TEMPLATE_MAP.get(repo_name, 'r2e-gym-base')
```

## Template Contents

### r2e-gym-base

- **Testing**: pytest, hypothesis, coverage, pytest-xdist
- **Scientific**: numpy (<2.3), scipy, mpmath, sympy
- **Code Analysis**: tree_sitter_languages, chardet
- **Web/Async**: aiohttp, yarl, multidict
- **Tools**: uv (fast installer), git configured
- **System**: gcc, g++, gfortran, openblas, lapack

### r2e-gym-scientific

Everything in base, plus:
- **Data Science**: pandas, scikit-learn (pre-built wheels)
- **Data Storage**: pyarrow, tables, h5py
- **Visualization**: matplotlib, pyqtgraph
- **Orange3 deps**: bottleneck, networkx, openpyxl, etc.
- **System**: hdf5, additional BLAS libraries

### r2e-gym-pillow

Everything in base, plus:
- **Image formats**: libjpeg, libpng, libtiff, libwebp, openjp2
- **Font rendering**: freetype, harfbuzz, fribidi
- **Color management**: lcms2
- **Pre-built pillow** with all features enabled

## Benchmarking

Test setup times with the benchmark script:

```bash
cd /home/claude/work/rllm
python experiments/swebench_ppio/test_repo_setup_time.py --repos numpy,pandas,pillow
```

## Template Sizes

| Template | Base Image | Additional | Total |
|----------|------------|------------|-------|
| r2e-gym-base | ~2GB | ~1.5GB | ~3.5GB |
| r2e-gym-scientific | ~2GB | ~3GB | ~5GB |
| r2e-gym-pillow | ~2GB | ~2GB | ~4GB |

First sandbox creation may take 2-3 minutes for image pull.
Subsequent creations are instant due to caching.
