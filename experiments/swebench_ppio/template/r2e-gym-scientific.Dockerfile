# R2E-Gym Scientific Template for PPIO Sandbox
# Optimized for numpy, pandas, orange3 with pre-built wheels
#
# Reduces setup time from ~60-130s to ~15-30s by:
# - Pre-installing numpy, scipy, pandas, scikit-learn wheels
# - Including build dependencies for C extensions
# - Pre-compiling common Cython modules

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
    # Linear algebra libraries
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    # For pandas/numpy compilation
    libffi-dev \
    libssl-dev \
    pkg-config \
    # For HDF5 support (pandas)
    libhdf5-dev \
    # For compression
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Configure git
RUN git config --global advice.detachedHead false && \
    git config --global core.compression 0 && \
    git config --global http.postBuffer 524288000

# Pre-install scientific stack with pre-built wheels
# Use specific versions known to have wheels
RUN uv pip install --system \
    # Core scientific stack (pre-built wheels)
    "numpy>=1.24.0,<2.3" \
    "scipy>=1.10.0" \
    "pandas>=2.0.0" \
    "scikit-learn>=1.3.0" \
    # Testing frameworks
    pytest>=7.0.0 \
    pytest-cov \
    pytest-timeout \
    pytest-xdist \
    hypothesis \
    coverage \
    # Build tools
    cython>=3.0.0 \
    meson-python \
    meson \
    ninja \
    setuptools \
    wheel \
    pip \
    build \
    # Code analysis
    tree_sitter_languages \
    chardet \
    # Math libraries
    mpmath \
    sympy \
    # Data handling
    pyarrow \
    tables \
    h5py \
    numexpr \
    # Utilities
    ipython \
    tqdm \
    requests \
    pyyaml \
    toml \
    joblib \
    threadpoolctl

# Pre-install Orange3 dependencies (ML/data mining library)
RUN uv pip install --system \
    bottleneck \
    chardet \
    httpx \
    keyring \
    keyrings-alt \
    networkx \
    openpyxl \
    python-louvain \
    pyyaml \
    xlrd \
    xlsxwriter \
    # Visualization
    matplotlib \
    pyqtgraph \
    # Optional Orange deps
    serverfiles \
    AnyQt

# Create working directories
RUN mkdir -p /home/user/testbed /repos

WORKDIR /home/user/testbed

ENV VIRTUAL_ENV=/home/user/testbed/.venv
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
