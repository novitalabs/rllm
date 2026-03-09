#!/usr/bin/env bash
# Finalize registry split between .14 (small repos) and .17 (large repos).
#
# Run AFTER rsync of blobs and repos to .17 is complete.
#
# .14 keeps: aiohttp, coveragepy, datalad, pyramid, scrapy, tornado  (1271 tags)
# .17 keeps: pandas, numpy, pillow, orange3                          (3351 tags)
#
# Usage:
#   bash k8s/scripts/finalize-registry-split.sh

set -euo pipefail

NODE14="10.83.115.14"
NODE17="10.83.115.17"

# Repos to keep on each node
KEEP_ON_14=(aiohttp_final coveragepy_final datalad_final pyramid_final scrapy_final tornado_final)
KEEP_ON_17=(pandas_final numpy_final pillow_final orange3_final)

REPO_BASE="/data/registry-cache/data/docker/registry/v2/repositories/namanjain12"

echo "=== Step 1: Verify .17 registry has the large repos ==="
for repo in "${KEEP_ON_17[@]}"; do
    count=$(ssh root@${NODE17} "ls ${REPO_BASE}/${repo}/_manifests/tags/ 2>/dev/null | wc -l" 2>/dev/null || echo 0)
    echo "  ${repo}: ${count} tags on .17"
    if [[ "$count" -lt 100 ]]; then
        echo "ERROR: ${repo} has too few tags on .17 (${count}). Rsync may not be complete."
        exit 1
    fi
done
echo "  [OK] All large repos present on .17"

echo ""
echo "=== Step 2: Delete large repos from .14 ==="
for repo in "${KEEP_ON_17[@]}"; do
    echo "  Deleting namanjain12/${repo} from .14..."
    ssh root@${NODE14} "rm -rf ${REPO_BASE}/${repo}" 2>/dev/null
done
echo "  [OK] Done"

echo ""
echo "=== Step 3: Delete small repos from .17 ==="
for repo in "${KEEP_ON_14[@]}"; do
    echo "  Deleting namanjain12/${repo} from .17 (if present)..."
    ssh root@${NODE17} "rm -rf ${REPO_BASE}/${repo}" 2>/dev/null || true
done
echo "  [OK] Done"

echo ""
echo "=== Step 4: Garbage collect .14 to reclaim blob space ==="
echo "  Restarting registry-mirror on .14 for GC..."
ssh root@${NODE14} "
  systemctl restart docker-registry-mirror 2>/dev/null || true
  sleep 3
  docker run --rm \
    -v /data/registry-cache/data:/var/lib/registry \
    registry:2 \
    garbage-collect /etc/docker/registry/config.yml --delete-untagged 2>&1 | tail -5
  du -sh /data/registry-cache/data/
"

echo ""
echo "=== Step 5: Garbage collect .17 ==="
ssh root@${NODE17} "
  docker run --rm \
    -v /data/registry-cache/data:/var/lib/registry \
    -v /data/registry-cache/config.yml:/etc/docker/registry/config.yml:ro \
    registry:2 \
    garbage-collect /etc/docker/registry/config.yml --delete-untagged 2>&1 | tail -5
  du -sh /data/registry-cache/data/
"

echo ""
echo "=== Done! ==="
echo ".14 registry (port 5000): ${KEEP_ON_14[*]}"
echo ".17 registry (port 5000): ${KEEP_ON_17[*]}"
echo ""
echo "Update statefulset.yaml REGISTRY_MIRROR if needed."
