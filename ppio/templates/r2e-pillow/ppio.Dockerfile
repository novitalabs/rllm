# PPIO Sandbox Template for python-pillow/Pillow
# R2E-Gym Training Dataset
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# System dependencies for image processing
RUN apt-get update && apt-get install -y \
    python3.9 python3.9-dev python3.9-venv python3-pip \
    git curl wget build-essential \
    libjpeg-dev zlib1g-dev libpng-dev libtiff-dev \
    libwebp-dev libfreetype6-dev liblcms2-dev \
    libopenjp2-7-dev libimagequant-dev libraqm-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip3 install --no-cache-dir pytest

WORKDIR /testbed

# Pre-clone repository
RUN git clone --depth 100 https://github.com/python-pillow/Pillow.git /testbed

# Set permissions
RUN chmod -R 777 /testbed
