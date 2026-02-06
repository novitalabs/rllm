# PPIO Sandbox Template for biolab/orange3
# R2E-Gym Training Dataset
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV QT_QPA_PLATFORM=offscreen

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.9 python3.9-dev python3.9-venv python3-pip \
    git curl wget build-essential \
    libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-xinerama0 libxcb-xfixes0 libxcb-shape0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy scipy scikit-learn pandas \
    chardet serverfiles bottleneck \
    pytest AnyQt pyqtgraph

WORKDIR /testbed

# Pre-clone repository
RUN git clone --depth 100 https://github.com/biolab/orange3.git /testbed

# Set permissions
RUN chmod -R 777 /testbed
