#!/bin/bash
# Pull missing Docker images from Docker Hub and push to local registry
# Run this on .17: bash pull-missing-to-registry.sh 10.83.115.14:5000
set -uo pipefail

REGISTRY="${1:-10.83.115.14:5000}"
CONCURRENCY="${2:-8}"
PROXY="http://127.0.0.1:1083"

echo "=== Pull missing images → ${REGISTRY} (concurrency=${CONCURRENCY}) ==="
echo "Start: $(date)"
echo ""

export HTTP_PROXY="$PROXY"
export HTTPS_PROXY="$PROXY"
export NO_PROXY="localhost,127.0.0.1,10.0.0.0/8"

REPOS=(
    namanjain12/pandas_final
    namanjain12/orange3_final
    namanjain12/coveragepy_final
    namanjain12/pillow_final
    namanjain12/datalad_final
    namanjain12/pyramid_final
    namanjain12/aiohttp_final
    namanjain12/numpy_final
    namanjain12/scrapy_final
    namanjain12/tornado_final
)

GRAND_PULLED=0
GRAND_FAILED=0
GRAND_SKIPPED=0

for REPO in "${REPOS[@]}"; do
    echo "=== ${REPO} ==="

    # Get all tags from Docker Hub (paginated)
    HUB_TAGS=()
    PAGE=1
    while true; do
        RESULT=$(curl -sf "https://hub.docker.com/v2/repositories/${REPO}/tags/?page_size=100&page=${PAGE}" \
            -x "$PROXY" 2>/dev/null)
        [ -z "$RESULT" ] && break
        TAGS_PAGE=$(echo "$RESULT" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for t in d.get('results',[]):
    print(t['name'])
" 2>/dev/null)
        [ -z "$TAGS_PAGE" ] && break
        while IFS= read -r t; do HUB_TAGS+=("$t"); done <<< "$TAGS_PAGE"
        NEXT=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('next','') or '')" 2>/dev/null)
        [ -z "$NEXT" ] && break
        PAGE=$((PAGE + 1))
    done

    # Get tags already in local registry
    LOCAL_TAGS=$(curl -sf "http://${REGISTRY}/v2/${REPO}/tags/list" 2>/dev/null \
        | python3 -c "import json,sys; print('\n'.join(json.load(sys.stdin).get('tags',[])))" 2>/dev/null || echo "")

    # Find missing tags
    MISSING=()
    for tag in "${HUB_TAGS[@]}"; do
        if ! echo "$LOCAL_TAGS" | grep -qx "$tag"; then
            MISSING+=("$tag")
        fi
    done

    echo "  Hub: ${#HUB_TAGS[@]}, Local: $(echo "$LOCAL_TAGS" | grep -c . || echo 0), Missing: ${#MISSING[@]}"

    [ ${#MISSING[@]} -eq 0 ] && echo "  All synced!" && echo "" && continue

    PULLED=0
    FAILED=0
    RUNNING=0

    for TAG in "${MISSING[@]}"; do
        (
            SRC="${REPO}:${TAG}"
            DST="${REGISTRY}/${REPO}:${TAG}"

            # Pull from Docker Hub
            if ! docker pull "$SRC" > /dev/null 2>&1; then
                echo "  PULL_FAIL: ${TAG}" >&2
                exit 1
            fi

            # Tag and push to registry
            docker tag "$SRC" "$DST" 2>/dev/null
            ok=0
            for attempt in 1 2 3; do
                if docker push "$DST" > /dev/null 2>&1; then ok=1; break; fi
                sleep 3
            done

            # Clean up
            docker rmi "$DST" > /dev/null 2>&1 || true
            docker rmi "$SRC" > /dev/null 2>&1 || true

            [ $ok -eq 1 ] && exit 0 || exit 1
        ) &

        RUNNING=$((RUNNING + 1))
        if [ $RUNNING -ge $CONCURRENCY ]; then
            wait -n 2>/dev/null || true
            RUNNING=$((RUNNING - 1))
        fi
    done

    # Wait for all jobs in this repo
    wait

    # Count results from registry
    NEW_COUNT=$(curl -sf "http://${REGISTRY}/v2/${REPO}/tags/list" 2>/dev/null \
        | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('tags',[])))" 2>/dev/null || echo "?")
    NEWLY_ADDED=$((NEW_COUNT - $(echo "$LOCAL_TAGS" | grep -c . || echo 0)))
    REPO_FAILED=$((${#MISSING[@]} - NEWLY_ADDED))
    GRAND_PULLED=$((GRAND_PULLED + NEWLY_ADDED))
    GRAND_FAILED=$((GRAND_FAILED + REPO_FAILED))

    # Prune dangling
    docker image prune -f > /dev/null 2>&1 || true

    AVAIL=$(df -h /data | awk 'NR==2{print $4}')
    echo "  Result: added=${NEWLY_ADDED}/${#MISSING[@]} failed=${REPO_FAILED} | now=${NEW_COUNT} tags | /data free: ${AVAIL}"
    echo ""
done

echo "========================================="
echo "=== COMPLETE at $(date) ==="
echo "Pulled: ${GRAND_PULLED}"
echo "Failed: ${GRAND_FAILED}"
echo ""
echo "Registry totals:"
TOTAL=0
for REPO in "${REPOS[@]}"; do
    count=$(curl -sf "http://${REGISTRY}/v2/${REPO}/tags/list" 2>/dev/null \
        | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('tags',[])))" 2>/dev/null || echo "?")
    TOTAL=$((TOTAL + count))
    echo "  ${REPO}: ${count}"
done
echo "  TOTAL: ${TOTAL}"
echo ""
echo "/data free: $(df -h /data | awk 'NR==2{print $4}')"
