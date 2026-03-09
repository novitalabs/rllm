Analyze the DeepSWE training experiment by fetching logs from K8s and parsing metrics.

## Quick Start (one command)

```bash
./ppio/scripts/fetch_k8s_metrics.sh | python3 ppio/scripts/parse_training_metrics.py -
```

## Steps

### 1. Fetch logs from K8s pod

```bash
# Auto-detects host and reads all log files (including rotated)
./ppio/scripts/fetch_k8s_metrics.sh -o /tmp/deepswe_raw.log -n deepswe -p deepswe-training-0
```

Options:
- `-o FILE` — save raw logs to file (recommended for re-analysis)
- `-n NS` — K8s namespace (default: `deepswe`)
- `-p POD` — pod name (default: `deepswe-training-0`)
- `-H IP` — force host IP (auto-detected from pod if omitted)

### 2. Parse and display metrics

```bash
python3 ppio/scripts/parse_training_metrics.py /tmp/deepswe_raw.log
```

Options:
- `--steps N-M` — filter step range (e.g., `--steps 5-9`, `--steps 10-`)
- `--json` — append JSON output for programmatic use
- `--csv` — append CSV output
- `--no-events` — skip per-step trajectory outcome table
- `-` — read from stdin (pipe from fetch script)

### 3. Output tables

The parser produces 6 sections:

| Section | Key Metrics |
|---------|-------------|
| **Timing Breakdown** | step time, collect_trajectory %, update_actor %, MFU |
| **Trajectory Metrics** | llm_time, env_time, total_time (mean/max), tail ratio, mismatch% |
| **Training Metrics** | score, pg_loss, grad_norm, pg_clipfrac, entropy, KL, resp_len, advantages |
| **Sequence Lengths** | prompt_length, response_length (mean/max/min), clip_ratio, abort_ratio |
| **Trajectory Outcomes** | ENV_DONE / MAX_STEPS / TRUNCATION counts, R=1/R=0, overlong, errors |
| **Summary** | aggregated averages, total GPU-hours, validation progress |

### 4. Summarize findings in Chinese

When presenting results, highlight:
- Per-step timing: collect_trajectory percentage, update_actor time
- Token mismatch rate (should be 0% after fix)
- Score trend and solve rate across steps
- Grad norm and entropy trends (signs of learning or divergence)
- pg_clipfrac (should be >0 once policy starts diverging from initial)
- Response length trends
- Trajectory outcomes: overlong filter rate, MAX_STEPS vs ENV_DONE ratio
- Validation progress and estimated resolve rate
- Any anomalies, errors, or timeouts

### 5. Additional analysis scripts

```bash
# LLM inference metrics (tok/s, latency breakdown)
python3 ppio/scripts/extract_llm_metrics.py /tmp/deepswe_raw.log

# Long-tail trajectory analysis
python3 ppio/scripts/analyze_long_tail.py /tmp/deepswe_raw.log
```

## Reference Documentation

- Overall analysis: `ppio/docs/training-timing-analysis.md`
- Long-tail deep dive: `ppio/docs/long-tail-trajectory-analysis.md`
- Issues & solutions: `ppio/docs/experiment-issues.md`

## K8s Environment

- Namespace: `deepswe`
- Pod: `deepswe-training-0` (head), `deepswe-training-{1,2,3}` (workers)
- Head node: 10.83.115.23
- kubectl: `kubectl --server=https://127.0.0.1:6443 --insecure-skip-tls-verify`
- Host log path: `/var/log/pods/deepswe_deepswe-training-0_<uid>/training/`

## Ray Worker Logs (补充数据源)

Kubelet pod 日志会轮转丢失历史数据，但 **Ray worker 日志保留了完整的训练指标**（从 Ray session 启动至今）。当 kubelet 日志缺失某些 step 时，用 Ray 日志恢复。

### 日志位置（pod 内）

```
/tmp/ray/session_latest/logs/worker-*.out
```

具体示例（当前 session）:
```
/tmp/ray/session_2026-03-03_20-21-04_588980_468/logs/worker-f05bbf21787bce039b205b45b8091c6229b5d0b44038e3443a6d57bf-02000000-13091.out
```

### 提取命令

```bash
# 提取指定 step 范围的训练指标 (例如 step 184-266)
kubectl exec -n deepswe deepswe-training-0 -- sh -c \
  "grep -P '^step:(1[89]\d|2[0-5]\d|26[0-6]) ' /tmp/ray/session_latest/logs/worker-*.out"

# 提取所有 val_score (temperature=0 greedy)
kubectl exec -n deepswe deepswe-training-0 -- sh -c \
  "grep -oP 'step:\d+.*?val/test_score/unknown:np\.float64\([0-9.]+\)' /tmp/ray/session_latest/logs/worker-*.out"

# 提取所有 step 指标（完整）
kubectl exec -n deepswe deepswe-training-0 -- sh -c \
  "grep -P '^step:\d+ ' /tmp/ray/session_latest/logs/worker-*.out"
```

### 注意事项

- Ray session 在 pod 重启后会重新创建，旧 session 日志会丢失
- 使用 `session_latest` 软链接而非硬编码 session 目录名
- Ray 日志包含完整指标但格式略不同于 kubelet 日志，需按 `key:value` 格式解析
