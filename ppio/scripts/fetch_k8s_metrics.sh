#!/usr/bin/env bash
# Fetch training metrics from K8s pod logs (including rotated logs on host).
# Outputs raw metric lines to stdout or a file for parse_training_metrics.py.
#
# Usage:
#   ./ppio/scripts/fetch_k8s_metrics.sh [-o OUTPUT_FILE] [-n NAMESPACE] [-p POD_NAME] [-H HOST_IP]
#
# Modes:
#   1. If HOST_IP is given, reads directly from host /var/log/pods/ (includes rotated logs)
#   2. Otherwise, uses kubectl logs (may miss older steps due to log rotation)

set -euo pipefail

NAMESPACE="${NAMESPACE:-deepswe}"
POD_NAME="${POD_NAME:-deepswe-training-0}"
HOST_IP=""
OUTPUT=""
KUBECTL="kubectl --server=https://127.0.0.1:6443 --insecure-skip-tls-verify"

while getopts "o:n:p:H:" opt; do
    case $opt in
        o) OUTPUT="$OPTARG" ;;
        n) NAMESPACE="$OPTARG" ;;
        p) POD_NAME="$OPTARG" ;;
        H) HOST_IP="$OPTARG" ;;
        *) echo "Usage: $0 [-o output] [-n namespace] [-p pod] [-H host_ip]" >&2; exit 1 ;;
    esac
done

fetch_from_host() {
    local host="$1"
    # Find the pod log directory
    local pod_uid
    pod_uid=$($KUBECTL -n "$NAMESPACE" get pod "$POD_NAME" -o jsonpath='{.metadata.uid}' 2>/dev/null)
    local log_dir="/var/log/pods/${NAMESPACE}_${POD_NAME}_${pod_uid}/training"

    # Read all log files (rotated gzipped + current) in chronological order
    ssh -o StrictHostKeyChecking=no "$host" "
        cd '$log_dir' 2>/dev/null || exit 1
        # Rotated gz files first (oldest to newest), then rotated plain, then current
        for f in \$(ls -t *.gz 2>/dev/null | tac); do zcat \"\$f\"; done
        for f in \$(ls -t *.log.* 2>/dev/null | grep -v '.gz$' | tac); do cat \"\$f\"; done
        cat 0.log 2>/dev/null
    "
}

fetch_from_kubectl() {
    # Use a large byte limit to get as much history as possible
    $KUBECTL -n "$NAMESPACE" logs "$POD_NAME" --limit-bytes=500000000 2>/dev/null
}

# Fetch raw logs
if [[ -n "$HOST_IP" ]]; then
    raw_logs=$(fetch_from_host "$HOST_IP")
else
    # Auto-detect host IP from pod
    HOST_IP=$($KUBECTL -n "$NAMESPACE" get pod "$POD_NAME" -o jsonpath='{.status.hostIP}' 2>/dev/null || true)
    if [[ -n "$HOST_IP" ]]; then
        raw_logs=$(fetch_from_host "$HOST_IP" 2>/dev/null || fetch_from_kubectl)
    else
        raw_logs=$(fetch_from_kubectl)
    fi
fi

if [[ -n "$OUTPUT" ]]; then
    echo "$raw_logs" > "$OUTPUT"
    echo "Raw logs saved to $OUTPUT" >&2
    echo "Lines: $(wc -l < "$OUTPUT")" >&2
else
    echo "$raw_logs"
fi
