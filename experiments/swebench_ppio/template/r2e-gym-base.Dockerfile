# R2E-Gym Base Template for PPIO Sandbox
# Pre-installed dependencies for SWE-bench training with R2E-Gym dataset
#
# Based on sandbox-fusion with additional packages for:
# - sympy, pandas, numpy, scrapy, tornado, statsmodels, pillow
# - pyramid, datalad, aiohttp, mypy, coveragepy, orange3, bokeh

FROM volcengine/sandbox-fusion:server-20250609

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# System dependencies for scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    ca-certificates \
    python3-dev \
    python3.11-dev \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Configure git for faster clones
RUN git config --global advice.detachedHead false && \
    git config --global core.compression 0 && \
    git config --global http.postBuffer 524288000

# Pre-install common dependencies using uv (global pip)
# These are commonly required by R2E-Gym repositories
RUN uv pip install --system \
    # Testing frameworks
    pytest>=7.0.0 \
    pytest-cov \
    pytest-timeout \
    pytest-xdist \
    hypothesis \
    coverage \
    # Code analysis
    tree_sitter_languages \
    chardet \
    # Scientific computing (common deps)
    numpy>=1.24.0,<2.3 \
    scipy \
    mpmath \
    # Utilities
    ipython \
    numexpr \
    cython \
    setuptools \
    wheel \
    pip \
    # Web/async (for aiohttp, tornado, scrapy)
    aiohttp \
    yarl \
    multidict \
    # For pillow
    pillow \
    # For sympy
    sympy \
    # Misc
    tqdm \
    requests \
    pyyaml \
    toml

# Create working directories
RUN mkdir -p /home/user/testbed /repos

# Set working directory
WORKDIR /home/user/testbed

# Environment variables for Python
ENV VIRTUAL_ENV=/home/user/testbed/.venv
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy supervisord config (inherited from sandbox-fusion)
# The sandbox-fusion image uses supervisord to manage services
