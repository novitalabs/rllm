#!/bin/bash
# Push local Docker images to plain registry and clean up host cache
# Usage: bash push-to-registry.sh [REGISTRY_ADDR] [REPO_FILTER]
# Example: bash push-to-registry.sh localhost:5001 namanjain12

set -euo pipefail

REGISTRY="${1:-localhost:5001}"
REPO_FILTER="${2:-namanjain12}"
MAX_RETRIES=3
RETRY_DELAY=5

echo "=== Push local images to ${REGISTRY} ==="
echo "Filter: ${REPO_FILTER}"
echo "Start: $(date)"
echo ""

push_with_retry() {
    local target="$1"
    local attempt=1
    while [ $attempt -le $MAX_RETRIES ]; do
        if docker push "${target}" > /dev/null 2>&1; then
            return 0
        fi
        echo "    retry ${attempt}/${MAX_RETRIES}..."
        sleep $RETRY_DELAY
        # Check if registry is alive
        if ! curl -sf "http://${REGISTRY}/v2/" > /dev/null 2>&1; then
            echo "    registry unresponsive, waiting 10s..."
            sleep 10
        fi
        attempt=$((attempt + 1))
    done
    return 1
}

# Get list of repos
REPOS=$(docker images --format '{{.Repository}}' | grep "${REPO_FILTER}" | sort -u)
TOTAL_IMAGES=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep "${REPO_FILTER}" | wc -l)
GLOBAL_PUSHED=0
GLOBAL_FAILED=0
GLOBAL_SKIPPED=0

echo "Total repos: $(echo "$REPOS" | wc -l), Total images: ${TOTAL_IMAGES}"
echo ""

for REPO in $REPOS; do
    TAGS=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep "^${REPO}:" | sort)
    TAG_COUNT=$(echo "$TAGS" | wc -l)

    # Check how many tags already in registry
    EXISTING=$(curl -sf "http://${REGISTRY}/v2/${REPO}/tags/list" 2>/dev/null \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('tags',[])))" 2>/dev/null || echo "0")

    echo "=== ${REPO}: ${TAG_COUNT} local, ${EXISTING} in registry ==="

    PUSHED=0
    FAILED=0
    SKIPPED=0

    for FULL_TAG in $TAGS; do
        TAG="${FULL_TAG#*:}"
        TARGET="${REGISTRY}/${REPO}:${TAG}"

        # Check if already in registry → skip push, just delete local
        if curl -sf -o /dev/null "http://${REGISTRY}/v2/${REPO}/manifests/${TAG}" \
            -H "Accept: application/vnd.docker.distribution.manifest.v2+json" 2>/dev/null; then
            SKIPPED=$((SKIPPED + 1))
            docker rmi "${FULL_TAG}" > /dev/null 2>&1 || true
            continue
        fi

        # Tag for registry
        docker tag "${FULL_TAG}" "${TARGET}" 2>/dev/null || continue

        # Push with retry
        if push_with_retry "${TARGET}"; then
            PUSHED=$((PUSHED + 1))
            GLOBAL_PUSHED=$((GLOBAL_PUSHED + 1))
        else
            FAILED=$((FAILED + 1))
            GLOBAL_FAILED=$((GLOBAL_FAILED + 1))
            echo "  FAIL: ${TAG}"
        fi

        # Clean up: remove registry-tagged copy + original
        docker rmi "${TARGET}" > /dev/null 2>&1 || true
        docker rmi "${FULL_TAG}" > /dev/null 2>&1 || true

        # Progress every 20 images
        DONE=$((PUSHED + FAILED + SKIPPED))
        if [ $((DONE % 20)) -eq 0 ] && [ $DONE -gt 0 ]; then
            echo "  [${DONE}/${TAG_COUNT}] pushed=${PUSHED} skip=${SKIPPED} fail=${FAILED}"
        fi
    done

    GLOBAL_SKIPPED=$((GLOBAL_SKIPPED + SKIPPED))

    # Prune dangling images after each repo
    docker image prune -f > /dev/null 2>&1 || true

    # Status
    AVAIL=$(df -h /data | awk 'NR==2{print $4}')
    REG_SIZE=$(du -sh /data/registry-local/data/ 2>/dev/null | awk '{print $1}')
    echo "  Result: pushed=${PUSHED} skip=${SKIPPED} fail=${FAILED}"
    echo "  /data avail: ${AVAIL}, registry: ${REG_SIZE}"
    echo "  Global: $((GLOBAL_PUSHED + GLOBAL_FAILED + GLOBAL_SKIPPED))/${TOTAL_IMAGES}"
    echo ""
done

echo "========================================="
echo "=== COMPLETE at $(date) ==="
echo "Pushed: ${GLOBAL_PUSHED}"
echo "Skipped: ${GLOBAL_SKIPPED}"
echo "Failed: ${GLOBAL_FAILED}"
echo ""
echo "Registry catalog:"
curl -s "http://${REGISTRY}/v2/_catalog" | python3 -m json.tool 2>/dev/null || echo "(unavailable)"
echo ""
echo "Registry: $(du -sh /data/registry-local/data/ 2>/dev/null | awk '{print $1}')"
echo "/data avail: $(df -h /data | awk 'NR==2{print $4}')"
