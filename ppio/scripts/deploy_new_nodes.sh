#!/bin/bash
set -e

# =============================================================================
# Deploy new nodes for 64x H200 cluster
#
# This script:
#   1. Installs Docker + NVIDIA Container Toolkit on new nodes
#   2. Transfers Docker image from head node
#   3. Copies model weights to /data/models/Qwen3-32B
#   4. Copies code repo to /root/develop/rllm
#   5. Creates and starts the deepswe-train container
#
# Usage:
#   bash deploy_new_nodes.sh [node_ip]        # Deploy a single node
#   bash deploy_new_nodes.sh all              # Deploy all new nodes
#   bash deploy_new_nodes.sh check            # Check status of all nodes
# =============================================================================

NEW_NODES=(10.83.115.22 10.83.115.23 10.83.115.25 10.83.115.26 10.83.115.27 10.83.115.28)
HEAD_NODE="10.83.115.14"
DOCKER_IMAGE="deepswe-train:latest"
CONTAINER_NAME="deepswe-train"
MODEL_PATH="/data/models/Qwen3-32B"
CODE_PATH="/root/develop/rllm"
IMAGE_TAR="/tmp/deepswe-train-image.tar"

ssh_cmd() {
    ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no "$@"
}

install_docker() {
    local ip=$1
    echo "[$ip] Installing Docker..."
    ssh_cmd "$ip" bash -s <<'INSTALL_EOF'
set -ex
# Install Docker if not present
if ! command -v docker &>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg 2>/dev/null || true
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin
fi

# Configure NVIDIA container runtime for Docker
if ! docker info 2>/dev/null | grep -q nvidia; then
    # nvidia-container-toolkit should already be installed (nvidia-container-cli exists)
    nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
    systemctl restart docker
fi

# Verify
docker info 2>&1 | grep -i runtime | head -3
echo "Docker installed and configured."
INSTALL_EOF
    echo "[$ip] Docker installation done."
}

save_image() {
    if [ -f "$IMAGE_TAR" ]; then
        echo "Docker image tar already exists at $IMAGE_TAR"
        return
    fi
    echo "Saving Docker image to $IMAGE_TAR (this may take a while)..."
    docker save "$DOCKER_IMAGE" -o "$IMAGE_TAR"
    ls -lh "$IMAGE_TAR"
    echo "Image saved."
}

transfer_image() {
    local ip=$1
    echo "[$ip] Transferring Docker image..."
    # Check if image already loaded
    if ssh_cmd "$ip" "docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -q '$DOCKER_IMAGE'"; then
        echo "[$ip] Image already present, skipping."
        return
    fi
    # Transfer and load
    scp -o ConnectTimeout=15 "$IMAGE_TAR" "$ip:/tmp/deepswe-train-image.tar"
    ssh_cmd "$ip" "docker load -i /tmp/deepswe-train-image.tar && rm -f /tmp/deepswe-train-image.tar"
    echo "[$ip] Image loaded."
}

copy_model() {
    local ip=$1
    echo "[$ip] Copying model weights..."
    if ssh_cmd "$ip" "[ -d $MODEL_PATH ] && [ -f $MODEL_PATH/config.json ]" 2>/dev/null; then
        echo "[$ip] Model already present, skipping."
        return
    fi
    ssh_cmd "$ip" "mkdir -p $MODEL_PATH"
    rsync -azP --timeout=30 "$MODEL_PATH/" "$ip:$MODEL_PATH/"
    echo "[$ip] Model copied."
}

copy_code() {
    local ip=$1
    echo "[$ip] Syncing code repo..."
    ssh_cmd "$ip" "mkdir -p $CODE_PATH"
    rsync -azP --timeout=30 --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        "$CODE_PATH/" "$ip:$CODE_PATH/"
    echo "[$ip] Code synced."
}

copy_checkpoint() {
    local ip=$1
    local ckpt_path="/data/checkpoints/deepswe-swebench-16h200/qwen3-32b-swebench-16h200-v1"
    echo "[$ip] Copying v1 checkpoint..."
    if ssh_cmd "$ip" "[ -d $ckpt_path/global_step_15 ]" 2>/dev/null; then
        echo "[$ip] Checkpoint already present, skipping."
        return
    fi
    ssh_cmd "$ip" "mkdir -p $ckpt_path"
    rsync -azP --timeout=30 "$ckpt_path/" "$ip:$ckpt_path/"
    echo "[$ip] Checkpoint copied."
}

create_container() {
    local ip=$1
    echo "[$ip] Creating container..."
    # Stop existing container if any
    ssh_cmd "$ip" "docker rm -f $CONTAINER_NAME 2>/dev/null || true"
    # Create container matching head node config
    ssh_cmd "$ip" "docker run -d \
        --name $CONTAINER_NAME \
        --network host \
        --ipc host \
        --gpus all \
        --shm-size=64g \
        -v /root/develop:/root/develop \
        -v /data:/data \
        -v /root/.ssh:/root/.ssh:ro \
        -v /tmp:/tmp \
        $DOCKER_IMAGE"
    echo "[$ip] Container created."
    # Verify CUDA
    echo -n "[$ip] CUDA check: "
    ssh_cmd "$ip" "docker exec $CONTAINER_NAME python3 -c 'import torch; print(f\"CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}\")'" 2>&1
}

deploy_node() {
    local ip=$1
    echo ""
    echo "=========================================="
    echo "Deploying $ip"
    echo "=========================================="
    install_docker "$ip"
    transfer_image "$ip"
    copy_model "$ip"
    copy_code "$ip"
    copy_checkpoint "$ip"
    create_container "$ip"
    echo "[$ip] Deployment complete!"
}

check_all() {
    echo "Checking all nodes..."
    echo ""
    printf "%-16s %-6s %-8s %-10s %-10s %-8s\n" "IP" "SSH" "Docker" "Container" "CUDA" "Model"
    printf "%-16s %-6s %-8s %-10s %-10s %-8s\n" "---" "---" "---" "---" "---" "---"
    for ip in "${NEW_NODES[@]}"; do
        local ssh_ok="NO"
        local docker_ok="NO"
        local container_ok="NO"
        local cuda_ok="NO"
        local model_ok="NO"

        if timeout 8 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$ip" "true" 2>/dev/null; then
            ssh_ok="OK"
            if ssh_cmd "$ip" "docker --version" 2>/dev/null | grep -q Docker; then
                docker_ok="OK"
            fi
            if ssh_cmd "$ip" "docker ps --format '{{.Names}}' 2>/dev/null | grep -q $CONTAINER_NAME"; then
                container_ok="OK"
            fi
            if ssh_cmd "$ip" "docker exec $CONTAINER_NAME python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null | grep -q True"; then
                cuda_ok="OK"
            fi
            if ssh_cmd "$ip" "[ -f $MODEL_PATH/config.json ]" 2>/dev/null; then
                model_ok="OK"
            fi
        fi
        printf "%-16s %-6s %-8s %-10s %-10s %-8s\n" "$ip" "$ssh_ok" "$docker_ok" "$container_ok" "$cuda_ok" "$model_ok"
    done
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

case "${1:-check}" in
    all)
        save_image
        for ip in "${NEW_NODES[@]}"; do
            deploy_node "$ip" || echo "FAILED: $ip, continuing..."
        done
        echo ""
        echo "All deployments attempted. Checking status..."
        check_all
        ;;
    check)
        check_all
        ;;
    *)
        # Single node deploy
        if [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            save_image
            deploy_node "$1"
        else
            echo "Usage: $0 {all|check|<node_ip>}"
            exit 1
        fi
        ;;
esac
