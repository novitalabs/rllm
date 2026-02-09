# PPIO Sandbox Template for aio-libs/aiohttp
# R2E-Gym Training Dataset
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-venv python3-pip \
    git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    pytest pytest-asyncio pytest-cov \
    yarl multidict async-timeout \
    aiosignal frozenlist attrs \
    cython

WORKDIR /testbed

# Pre-clone repository
RUN git clone --depth 100 https://github.com/aio-libs/aiohttp.git /testbed

# Set permissions
RUN chmod -R 777 /testbed
