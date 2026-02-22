#!/bin/bash
set -e

# =============================================================================
# Cluster Setup Script for 64x H200 (8 nodes x 8 GPUs)
#
# This script helps set up the Ray cluster across 8 nodes.
#
# Prerequisites on each node:
#   1. Docker with NVIDIA runtime (nvidia-docker2)
#   2. NVIDIA driver installed (nvidia-smi works)
#   3. SSH access from head node to all workers
#   4. Model weights at /data/models/Qwen3-32B
#   5. Code repo at /root/develop/rllm (or adjust RLLM_DIR)
#
# Usage:
#   1. Edit the NODE_IPS array below with internal IPs
#   2. Run: bash setup_64h200_cluster.sh setup    # First time setup
#   3. Run: bash setup_64h200_cluster.sh start     # Start Ray cluster
#   4. Run: bash setup_64h200_cluster.sh stop      # Stop Ray cluster
#   5. Run: bash setup_64h200_cluster.sh status    # Check cluster status
#   6. Run: bash setup_64h200_cluster.sh firewall  # Apply iptables rules
# =============================================================================

# -----------------------------------------------------------------------------
# Node Configuration — EDIT THESE WITH INTERNAL IPs
# -----------------------------------------------------------------------------

# Head node (first entry) and workers (remaining entries)
NODE_IPS=(
    "10.83.115.14"   # Head node (existing)
    "10.83.115.17"   # Worker 1 (existing)
    "10.83.115.22"   # Worker 2 (new)
    "10.83.115.23"   # Worker 3 (new)
    "10.83.115.25"   # Worker 4 (new)
    "10.83.115.26"   # Worker 5 (new)
    "10.83.115.27"   # Worker 6 (new)
    "10.83.115.28"   # Worker 7 (new)
)

HEAD_IP="${NODE_IPS[0]}"
CONTAINER_NAME="deepswe-train"
DOCKER_IMAGE="${DOCKER_IMAGE:-}"  # Set to your training image

# -----------------------------------------------------------------------------
# NCCL network interface — EDIT THIS
# Run `ip addr` on a node to find the interface with the internal IP
# -----------------------------------------------------------------------------
NCCL_IFNAME="${NCCL_SOCKET_IFNAME:-b_manage0}"

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

run_on_node() {
    local ip=$1
    shift
    if [ "$ip" = "$HEAD_IP" ] && [ "$ip" = "$(hostname -I | awk '{print $1}')" ]; then
        eval "$@"
    else
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$ip" "$@"
    fi
}

run_in_container() {
    local ip=$1
    shift
    run_on_node "$ip" "docker exec $CONTAINER_NAME bash -c '$@'"
}

check_connectivity() {
    echo "Checking connectivity to all nodes..."
    for ip in "${NODE_IPS[@]}"; do
        echo -n "  $ip: "
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$ip" "hostname" 2>/dev/null; then
            echo "  OK"
        else
            echo "  FAILED"
            return 1
        fi
    done
    echo "All nodes reachable."
}

check_gpus() {
    echo "Checking GPUs on all nodes..."
    for ip in "${NODE_IPS[@]}"; do
        echo -n "  $ip: "
        local gpu_count
        gpu_count=$(run_on_node "$ip" "docker exec $CONTAINER_NAME nvidia-smi --query-gpu=count --format=csv,noheader | head -1" 2>/dev/null)
        if [ -n "$gpu_count" ]; then
            echo "${gpu_count} GPUs"
        else
            echo "FAILED (no container or no GPUs)"
        fi
    done
}

check_cuda() {
    echo "Checking CUDA in containers..."
    for ip in "${NODE_IPS[@]}"; do
        echo -n "  $ip: "
        run_in_container "$ip" "python3 -c 'import torch; print(f\"CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}\")'" 2>/dev/null || echo "FAILED"
    done
}

start_cluster() {
    echo "Starting Ray cluster..."
    echo "Head node: $HEAD_IP"

    # Start head
    echo "Starting Ray head on $HEAD_IP..."
    run_in_container "$HEAD_IP" \
        "ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus=8 --num-cpus=96 --object-store-memory=200000000000" 2>&1
    echo "Head started."

    # Start workers
    for i in $(seq 1 $((${#NODE_IPS[@]} - 1))); do
        local ip="${NODE_IPS[$i]}"
        echo "Starting Ray worker on $ip..."
        run_in_container "$ip" \
            "ray start --address='${HEAD_IP}:6379' --num-gpus=8 --num-cpus=96 --object-store-memory=200000000000" 2>&1
        echo "Worker $i ($ip) started."
    done

    # Verify
    sleep 3
    echo ""
    echo "Verifying cluster..."
    run_in_container "$HEAD_IP" \
        "python3 -c 'import ray; ray.init(address=\"auto\"); r=ray.cluster_resources(); print(f\"GPUs: {r.get(\"GPU\",0)}, CPUs: {r.get(\"CPU\",0)}, Nodes: {len([n for n in ray.nodes() if n[\"Alive\"]])}\"); ray.shutdown()'" 2>&1
}

stop_cluster() {
    echo "Stopping Ray on all nodes..."
    for ip in "${NODE_IPS[@]}"; do
        echo -n "  $ip: "
        run_in_container "$ip" "ray stop --force" 2>/dev/null && echo "OK" || echo "FAILED"
    done
}

apply_firewall() {
    echo "Applying iptables firewall rules on all nodes..."
    for node_ip in "${NODE_IPS[@]}"; do
        echo "  Configuring $node_ip..."
        for allowed_ip in "${NODE_IPS[@]}"; do
            for port in 6379 8265 10001; do
                run_on_node "$node_ip" "sudo iptables -C INPUT -p tcp -s $allowed_ip --dport $port -j ACCEPT 2>/dev/null || sudo iptables -I INPUT -p tcp -s $allowed_ip --dport $port -j ACCEPT"
            done
            run_on_node "$node_ip" "sudo iptables -C INPUT -p tcp -s $allowed_ip --dport 10002:19999 -j ACCEPT 2>/dev/null || sudo iptables -I INPUT -p tcp -s $allowed_ip --dport 10002:19999 -j ACCEPT"
        done
        # Allow localhost
        for port in 6379 8265; do
            run_on_node "$node_ip" "sudo iptables -C INPUT -p tcp -s 127.0.0.1 --dport $port -j ACCEPT 2>/dev/null || sudo iptables -I INPUT -p tcp -s 127.0.0.1 --dport $port -j ACCEPT"
        done
        # Drop external
        for port in 6379 8265 10001; do
            run_on_node "$node_ip" "sudo iptables -C INPUT -p tcp --dport $port -j DROP 2>/dev/null || sudo iptables -A INPUT -p tcp --dport $port -j DROP"
        done
        run_on_node "$node_ip" "sudo iptables -C INPUT -p tcp --dport 10002:19999 -j DROP 2>/dev/null || sudo iptables -A INPUT -p tcp --dport 10002:19999 -j DROP"
        echo "  $node_ip done."
    done
    echo "Firewall rules applied."
}

status() {
    check_connectivity
    echo ""
    check_gpus
    echo ""
    check_cuda
    echo ""
    echo "Ray cluster status:"
    run_in_container "$HEAD_IP" "ray status" 2>/dev/null || echo "  Ray not running."
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

case "${1:-status}" in
    setup)
        check_connectivity
        echo ""
        check_gpus
        echo ""
        check_cuda
        ;;
    start)
        start_cluster
        ;;
    stop)
        stop_cluster
        ;;
    status)
        status
        ;;
    firewall)
        apply_firewall
        ;;
    *)
        echo "Usage: $0 {setup|start|stop|status|firewall}"
        exit 1
        ;;
esac
