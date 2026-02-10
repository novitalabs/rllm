#!/bin/bash

# =============================================================================
# DeepSWE 32xH200 Multi-Node Cluster Setup
#
# Sets up a Ray cluster across 4 nodes (4x8 H200) with Docker containers.
# Run this from the HEAD node (10.83.115.14).
#
# Usage:
#   bash ppio/scripts/setup_32h200_cluster.sh [setup|stop|status|sync|...]
#
# Node mapping:
#   External IP     Internal IP      Hostname
#   111.6.123.35    10.83.115.14     host-10-83-115-14 (HEAD)
#   111.6.123.36    10.83.115.17     host-10-83-115-17
#   111.6.123.37    10.83.115.10     host-10-83-115-10
#   111.6.123.38    10.83.115.12     host-10-83-115-12
#
# Prerequisites:
#   - SSH access from HEAD to all workers (via internal IPs)
#   - Docker with nvidia-container-toolkit on all nodes
#   - rllm repo at /root/develop/tengwan/rl/rllm on all nodes
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Cluster Configuration (internal IPs for inter-node communication)
# -----------------------------------------------------------------------------

HEAD_NODE="${HEAD_NODE:-10.83.115.14}"
if [ ${#WORKER_NODES[@]} -eq 0 ] 2>/dev/null; then
    WORKER_NODES=("10.83.115.17" "10.83.115.10" "10.83.115.12")
fi
ALL_NODES=("$HEAD_NODE" "${WORKER_NODES[@]}")

# Docker configuration
DOCKER_IMAGE="${DOCKER_IMAGE:-deepswe-train:latest}"
CONTAINER_NAME="deepswe-train"

# Ray configuration
RAY_PORT=6379
RAY_DASHBOARD_PORT=8265

# Paths (local on each node, not shared NFS)
RLLM_DIR="${RLLM_DIR:-/root/develop/tengwan/rl/rllm}"
DEVELOP_DIR="${DEVELOP_DIR:-/root/develop}"
DATA_DIR="${DATA_DIR:-/data}"
SHM_SIZE="${SHM_SIZE:-64g}"

# NCCL network interface (GPU RDMA interfaces have MTU 9000)
# Use b_manage0 for TCP fallback, GPU4-GPU7 for RDMA
NCCL_IFNAME="${NCCL_IFNAME:-b_manage0}"

# SSH options
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o LogLevel=ERROR"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

ssh_cmd() {
    local node=$1
    shift
    if [ "$node" = "$HEAD_NODE" ] || [ "$node" = "$(hostname -I | awk '{print $1}')" ]; then
        # Local execution for head node
        eval "$@"
    else
        ssh $SSH_OPTS "$node" "$@"
    fi
}

docker_exec() {
    local node=$1
    shift
    local cmd="$*"
    if [ "$node" = "$HEAD_NODE" ] || [ "$node" = "$(hostname -I | awk '{print $1}')" ]; then
        docker exec $CONTAINER_NAME bash -c "$cmd"
    else
        ssh $SSH_OPTS "$node" "docker exec $CONTAINER_NAME bash -c \"$(echo "$cmd" | sed 's/"/\\"/g')\""
    fi
}

# Check SSH connectivity
check_connectivity() {
    log "Checking SSH connectivity to all nodes..."
    for node in "${ALL_NODES[@]}"; do
        local hname
        hname=$(ssh_cmd "$node" "hostname" 2>/dev/null) || hname="FAILED"
        if [ "$hname" != "FAILED" ]; then
            log "  $node ($hname): OK"
        else
            log "  $node: FAILED"
            echo "ERROR: Cannot reach $node. Check SSH keys and network."
            exit 1
        fi
    done
    log "All nodes reachable."
}

# Check GPU status on all nodes
check_gpus() {
    log "Checking GPU status on all nodes..."
    for node in "${ALL_NODES[@]}"; do
        local info
        info=$(ssh_cmd "$node" "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null")
        local count
        count=$(echo "$info" | wc -l)
        local name
        name=$(echo "$info" | head -1)
        log "  $node: ${count}x ${name}"
        if [ "$count" -ne 8 ]; then
            echo "WARNING: Expected 8 GPUs on $node, found $count"
        fi
    done
}

# -----------------------------------------------------------------------------
# Sync Code to All Nodes
# -----------------------------------------------------------------------------

sync_code() {
    log "Syncing rllm code to all worker nodes..."
    for node in "${WORKER_NODES[@]}"; do
        log "  Syncing to $node..."
        rsync -az --delete \
            -e "ssh $SSH_OPTS" \
            --exclude='.git' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='.venv' \
            --exclude='outputs' \
            --exclude='wandb' \
            "${RLLM_DIR}/" "${node}:${RLLM_DIR}/" &
    done
    wait
    log "Code synced to all nodes."
}

# -----------------------------------------------------------------------------
# Start/Stop Containers
# -----------------------------------------------------------------------------

start_containers() {
    log "Starting Docker containers on all nodes..."

    for node in "${ALL_NODES[@]}"; do
        log "  Starting container on $node..."

        # Stop existing container if running
        ssh_cmd "$node" "docker rm -f $CONTAINER_NAME 2>/dev/null || true"

        # Start new container
        # --entrypoint: override vllm's default API server entrypoint
        # --net=host: required for Ray and NCCL cross-node communication
        # --gpus all: expose all GPUs (no --runtime=nvidia needed)
        # Mount both /root/develop and /data at same paths inside container
        ssh_cmd "$node" "docker run -d \
            --name $CONTAINER_NAME \
            --entrypoint /bin/bash \
            --gpus all \
            --net=host \
            --ipc=host \
            --shm-size=$SHM_SIZE \
            --cap-add=SYS_ADMIN \
            --ulimit memlock=-1 \
            --ulimit stack=67108864 \
            -v ${DEVELOP_DIR}:${DEVELOP_DIR} \
            -v ${DATA_DIR}:${DATA_DIR} \
            -v /root/.ssh:/root/.ssh:ro \
            -v /tmp:/tmp \
            -e PPIO_API_KEY=${PPIO_API_KEY:-} \
            -e WANDB_API_KEY=${WANDB_API_KEY:-} \
            -e NCCL_DEBUG=${NCCL_DEBUG:-INFO} \
            -e NCCL_SOCKET_IFNAME=${NCCL_IFNAME} \
            -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0} \
            $DOCKER_IMAGE \
            -c 'sleep infinity'"
    done

    sleep 2

    # Verify containers are running
    for node in "${ALL_NODES[@]}"; do
        local status
        status=$(ssh_cmd "$node" "docker inspect -f '{{.State.Status}}' $CONTAINER_NAME 2>/dev/null" || echo "not found")
        log "  $node container: $status"
    done
}

stop_containers() {
    log "Stopping containers on all nodes..."
    for node in "${ALL_NODES[@]}"; do
        ssh_cmd "$node" "docker rm -f $CONTAINER_NAME 2>/dev/null || true"
        log "  $node: stopped"
    done
}

# -----------------------------------------------------------------------------
# Install Dependencies Inside Containers
# -----------------------------------------------------------------------------

install_deps() {
    log "Installing dependencies inside containers on all nodes..."

    # Proxy setup: if host has https_proxy set, pass it to container pip commands
    local PROXY_PREFIX=""
    if [ -n "${https_proxy:-}" ]; then
        PROXY_PREFIX="https_proxy=${https_proxy} http_proxy=${http_proxy:-${https_proxy}}"
        log "  Using proxy: ${https_proxy}"
    fi

    for node in "${ALL_NODES[@]}"; do
        log "  Installing on $node..."
        docker_exec "$node" "cd ${RLLM_DIR} && \
            ${PROXY_PREFIX} pip install 'verl==0.6.1' -q 2>&1 | tail -1 && \
            ${PROXY_PREFIX} pip install 'verl[vllm]' -q 2>&1 | tail -1 && \
            ${PROXY_PREFIX} pip install -e '.[swe]' -q 2>&1 | tail -1 && \
            ${PROXY_PREFIX} pip install ppio_sandbox pylatexenc pandas datasets -q 2>&1 | tail -1 && \
            ${PROXY_PREFIX} pip install 'ray[default]>=2.40' -q 2>&1 | tail -1 && \
            ${PROXY_PREFIX} pip install git+https://github.com/agentica-project/R2E-Gym.git -q 2>&1 | tail -1 && \
            echo 'All deps installed on ${node}'" &
    done
    wait
    log "Dependencies installed on all nodes."
}

# -----------------------------------------------------------------------------
# Ray Cluster Management
# -----------------------------------------------------------------------------

start_ray_cluster() {
    log "Starting Ray cluster..."

    # Start Ray head on this node
    log "Starting Ray head on $HEAD_NODE..."
    docker_exec "$HEAD_NODE" "ray stop --force 2>/dev/null || true"
    docker_exec "$HEAD_NODE" "ray start --head \
        --port=$RAY_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$RAY_DASHBOARD_PORT \
        --num-cpus=64 \
        --num-gpus=8"

    sleep 5
    log "Ray head started on $HEAD_NODE:$RAY_PORT"

    # Start Ray workers on remaining nodes
    for node in "${WORKER_NODES[@]}"; do
        log "  Starting Ray worker on $node..."
        docker_exec "$node" "ray stop --force 2>/dev/null || true"
        docker_exec "$node" "ray start \
            --address=$HEAD_NODE:$RAY_PORT \
            --num-cpus=64 \
            --num-gpus=8"
        sleep 3
    done

    # Wait for all workers to join
    sleep 5

    # Check cluster status
    log "Checking Ray cluster status..."
    docker_exec "$HEAD_NODE" "ray status"
}

stop_ray_cluster() {
    log "Stopping Ray cluster..."
    for node in "${ALL_NODES[@]}"; do
        docker_exec "$node" "ray stop --force 2>/dev/null || true"
        log "  $node: Ray stopped"
    done
}

ray_status() {
    log "Ray cluster status:"
    docker_exec "$HEAD_NODE" "ray status" 2>/dev/null || echo "Ray cluster not running."
}

# -----------------------------------------------------------------------------
# Full Setup
# -----------------------------------------------------------------------------

full_setup() {
    check_connectivity
    check_gpus

    echo ""
    log "=== Phase 1: Sync Code ==="
    sync_code

    echo ""
    log "=== Phase 2: Start Containers ==="
    start_containers

    # Install rllm in editable mode (source is mounted via -v, deps are in image)
    echo ""
    log "=== Phase 3: Install rllm (editable) ==="
    for node in "${ALL_NODES[@]}"; do
        docker_exec "$node" "cd ${RLLM_DIR} && pip install -e . -q 2>&1 | tail -1" &
    done
    wait

    echo ""
    log "=== Phase 4: Start Ray Cluster ==="
    start_ray_cluster

    echo ""
    log "=============================================="
    log "Cluster setup complete!"
    log ""
    log "Ray Dashboard: http://${HEAD_NODE}:${RAY_DASHBOARD_PORT}"
    log ""
    log "To run training:"
    log "  docker exec -it $CONTAINER_NAME bash"
    log "  cd ${RLLM_DIR}"
    log "  export PPIO_API_KEY=sk_xxxxx"
    log "  bash ppio/scripts/train_qwen3_32b_32h200.sh"
    log "=============================================="
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

case "${1:-setup}" in
    setup|start)
        full_setup
        ;;
    stop)
        stop_ray_cluster
        stop_containers
        ;;
    sync)
        sync_code
        ;;
    containers-start)
        start_containers
        ;;
    containers-stop)
        stop_containers
        ;;
    install-deps)
        install_deps
        ;;
    ray-start)
        start_ray_cluster
        ;;
    ray-stop)
        stop_ray_cluster
        ;;
    status)
        check_connectivity
        check_gpus
        echo ""
        ray_status
        ;;
    *)
        echo "Usage: $0 {setup|stop|status|sync|containers-start|containers-stop|install-deps|ray-start|ray-stop}"
        echo ""
        echo "Commands:"
        echo "  setup             - Full setup: sync, containers, deps, Ray cluster"
        echo "  stop              - Stop Ray cluster and containers on all nodes"
        echo "  status            - Check connectivity, GPUs, and Ray cluster status"
        echo "  sync              - Rsync rllm code to all worker nodes"
        echo "  containers-start  - Start Docker containers only"
        echo "  containers-stop   - Stop Docker containers only"
        echo "  install-deps      - Install Python dependencies in containers"
        echo "  ray-start         - Start Ray cluster (containers must be running)"
        echo "  ray-stop          - Stop Ray cluster only"
        exit 1
        ;;
esac
