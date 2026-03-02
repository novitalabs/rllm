#!/usr/bin/env python3
"""Pre-warm Docker images for R2E-Gym SWE tasks into the local DinD daemon.

Loads the R2E-Gym dataset, extracts unique docker_image values, checks which
are already cached locally, and pulls missing ones with parallel threads.

Usage:
    python3 prewarm-images.py [--workers 16] [--dataset R2E-Gym/R2E-Gym-V1]
"""

import argparse
import concurrent.futures
import subprocess
import sys
import threading

counter_lock = threading.Lock()
pull_count = 0


def get_local_images() -> set:
    """Return set of image tags already present in the local Docker daemon."""
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: docker images failed: {result.stderr}", file=sys.stderr)
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def pull_image(image: str, total: int) -> bool:
    """Pull a single Docker image. Returns True on success."""
    global pull_count
    try:
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"[FAIL] {image}: {result.stderr.strip()}")
            return False

        with counter_lock:
            pull_count += 1
            current = pull_count
        print(f"[OK] ({current}/{total}) {image}")
        return True
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {image}")
        return False
    except Exception as e:
        print(f"[ERROR] {image}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Pre-warm R2E-Gym Docker images")
    parser.add_argument("--workers", type=int, default=16, help="Parallel pull threads")
    parser.add_argument("--dataset", default="R2E-Gym/R2E-Gym-V1", help="HuggingFace dataset name")
    args = parser.parse_args()

    # Load dataset and extract unique images
    print(f"Loading dataset {args.dataset}...")
    from datasets import load_dataset
    dataset = load_dataset(args.dataset, split="train")

    unique_images = set()
    for entry in dataset:
        if "docker_image" in entry and entry["docker_image"]:
            unique_images.add(entry["docker_image"])

    print(f"Found {len(unique_images)} unique Docker images in dataset")

    # Check what's already cached
    local_images = get_local_images()
    missing = unique_images - local_images
    cached = unique_images & local_images

    print(f"Already cached: {len(cached)}")
    print(f"Need to pull: {len(missing)}")

    if not missing:
        print("All images already cached, nothing to do")
        return

    # Pull missing images in parallel
    succeeded = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(pull_image, img, len(missing)): img
            for img in missing
        }
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                succeeded += 1
            else:
                failed += 1

    print(f"\nPre-warm complete: {succeeded} pulled, {failed} failed, {len(cached)} already cached")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
