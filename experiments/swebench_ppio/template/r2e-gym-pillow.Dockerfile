# R2E-Gym Pillow Template for PPIO Sandbox
# Optimized for python-pillow/Pillow development
#
# Reduces setup time from ~80s to ~20s by:
# - Pre-installing all image format libraries
# - Including development headers
# - Pre-built pillow with all features enabled

FROM volcengine/sandbox-fusion:server-20250609

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# System dependencies for image processing
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
    pkg-config \
    # Image format libraries
    libjpeg-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    # Font/text rendering
    libfreetype6-dev \
    libfribidi-dev \
    libharfbuzz-dev \
    # Other image formats
    liblcms2-dev \
    libxcb1-dev \
    # Compression
    zlib1g-dev \
    liblzma-dev \
    # For ImageMagick support
    libmagickwand-dev \
    # For tkinter
    tk-dev \
    tcl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Configure git
RUN git config --global advice.detachedHead false && \
    git config --global core.compression 0 && \
    git config --global http.postBuffer 524288000

# Pre-install pillow with all features and testing dependencies
RUN uv pip install --system \
    # Pre-built pillow with all features
    "pillow>=10.0.0" \
    # Testing frameworks
    pytest>=7.0.0 \
    pytest-cov \
    pytest-timeout \
    pytest-xdist \
    hypothesis \
    coverage \
    # Build tools
    cython \
    setuptools \
    wheel \
    pip \
    build \
    # Pillow test dependencies
    numpy \
    scipy \
    # Image comparison/testing
    defusedxml \
    packaging \
    # Utilities
    tqdm \
    requests

# Create working directories
RUN mkdir -p /home/user/testbed /repos

WORKDIR /home/user/testbed

ENV VIRTUAL_ENV=/home/user/testbed/.venv
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Environment variables for Pillow build
ENV MAX_CONCURRENCY=4
