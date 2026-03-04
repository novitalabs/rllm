#!/bin/bash
# Fast parallel push of local Docker images to plain registry
# Usage: bash push-to-registry-fast.sh [REGISTRY] [CONCURRENCY] [REPO_FILTER]

set -uo pipefail

REGISTRY="${1:-localhost:5001}"
CONCURRENCY="${2:-16}"
REPO_FILTER="${3:-namanjain12}"

echo "=== Fast parallel push to ${REGISTRY} (concurrency=${CONCURRENCY}) ==="
echo "Start: $(date)"

# Collect all images (exclude registry-tagged copies)
mapfile -t ALL_IMAGES < <(docker images --format '{{.Repository}}:{{.Tag}}' | grep "^${REPO_FILTER}/" | sort)
TOTAL=${#ALL_IMAGES[@]}
echo "Total images: ${TOTAL}"
echo ""

# Counters
DONE=0
PUSHED=0
FAILED=0

# Process repo by repo
mapfile -t REPOS < <(printf '%s\n' "${ALL_IMAGES[@]}" | awk -F: '{print $1}' | sort -u)

for REPO in "${REPOS[@]}"; do
    # Get tags for this repo
    mapfile -t TAGS < <(printf '%s\n' "${ALL_IMAGES[@]}" | grep "^${REPO}:")
    REPO_COUNT=${#TAGS[@]}
    REPO_PUSHED=0
    REPO_FAILED=0

    echo "--- ${REPO}: ${REPO_COUNT} tags ---"

    # Parallel push using background jobs with concurrency limit
    RUNNING=0
    for FULL_TAG in "${TAGS[@]}"; do
        TAG="${FULL_TAG#*:}"
        TARGET="${REGISTRY}/${REPO}:${TAG}"

        (
            docker tag "${FULL_TAG}" "${TARGET}" 2>/dev/null || exit 1
            ok=0
            for attempt in 1 2 3; do
                if docker push "${TARGET}" > /dev/null 2>&1; then ok=1; break; fi
                sleep 3
            done
            docker rmi "${TARGET}" > /dev/null 2>&1 || true
            docker rmi "${FULL_TAG}" > /dev/null 2>&1 || true
            [ $ok -eq 1 ] && exit 0 || exit 1
        ) &

        RUNNING=$((RUNNING + 1))

        # Throttle: wait when hitting concurrency limit
        if [ $RUNNING -ge $CONCURRENCY ]; then
            wait -n 2>/dev/null || true
            RUNNING=$((RUNNING - 1))
        fi
    done

    # Wait for all remaining jobs in this repo
    wait

    # Count what's in registry now for this repo
    IN_REG=$(curl -sf "http://${REGISTRY}/v2/${REPO}/tags/list" 2>/dev/null \
        | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('tags',[])))" 2>/dev/null || echo "?")

    # Prune dangling
    docker image prune -f > /dev/null 2>&1 || true

    DONE=$((DONE + REPO_COUNT))
    AVAIL=$(df -h /data | awk 'NR==2{print $4}')
    REG_SIZE=$(du -sh /data/registry-local/data/ 2>/dev/null | awk '{print $1}')
    echo "  In registry: ${IN_REG} tags | Progress: ${DONE}/${TOTAL} | registry: ${REG_SIZE} | /data free: ${AVAIL}"
    echo ""
done

echo "========================================="
echo "=== COMPLETE at $(date) ==="
echo ""
echo "Registry:"
for repo in $(curl -s "http://${REGISTRY}/v2/_catalog" | python3 -c "import json,sys; [print(r) for r in json.load(sys.stdin).get('repositories',[])]" 2>/dev/null); do
    count=$(curl -s "http://${REGISTRY}/v2/${repo}/tags/list" | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('tags',[])))" 2>/dev/null)
    echo "  ${repo}: ${count} tags"
done
echo ""
echo "Size: $(du -sh /data/registry-local/data/ 2>/dev/null | awk '{print $1}')"
echo "Free: $(df -h /data | awk 'NR==2{print $4}')"
