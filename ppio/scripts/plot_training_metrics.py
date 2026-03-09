#!/usr/bin/env python3
"""Plot DeepSWE training metrics as a multi-panel figure.

Reads step data from either:
  1. JSON file (output of parse_training_metrics.py --json)
  2. Raw training log (auto-detected, uses same regex as parse_training_metrics.py)

Usage:
    # From JSON (recommended):
    python parse_training_metrics.py --json /tmp/raw.log > metrics.json
    python plot_training_metrics.py metrics.json -o training_metrics.png

    # From raw log directly:
    python plot_training_metrics.py /tmp/raw.log -o training_metrics.png

    # Pipeline:
    ./fetch_k8s_metrics.sh | python parse_training_metrics.py --json - | python plot_training_metrics.py - -o out.png

Options:
    -o FILE         Output PNG path (default: training_metrics.png)
    --steps N-M     Only plot steps in range (e.g., 61-343, 100-)
    --no-smooth     Disable rolling average overlay
    --dpi N         Output DPI (default: 150)
    --title TEXT    Override figure title
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Regex from parse_training_metrics.py (for raw log parsing) ───────────

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CRI_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:\d{2})\s+\w+\s+\w\s+"
)
STEP_RE = re.compile(r"step:(\d+)\s")
KV_RE = re.compile(r"([\w/.]+):(?:np\.\w+\()?([-\d.e+]+)\)?")
TRAJ_COMPLETE_RE = re.compile(
    r"Trajectory (\d+) completed due to: (\w+)\. Reward is ([\d.]+?)\.?\s"
)


def parse_step_metrics(line: str) -> dict | None:
    clean = ANSI_RE.sub("", line).strip()
    clean = CRI_TS_RE.sub("", clean)
    clean = re.sub(r"\(TaskRunner pid=\d+\)\s*", "", clean)
    m = STEP_RE.search(clean)
    if not m:
        return None
    d = {"step": int(m.group(1))}
    for kv in KV_RE.finditer(clean):
        key, val = kv.group(1), kv.group(2)
        if key == "step":
            continue
        try:
            d[key] = float(val)
        except ValueError:
            pass
    return d


# ─── Data loading ─────────────────────────────────────────────────────────

def load_json(text: str) -> list[dict]:
    """Load step data from JSON (may have preamble text before the JSON array)."""
    # parse_training_metrics.py --json prints tables then "─── JSON ───" then JSON
    marker = "─── JSON ───"
    if marker in text:
        text = text[text.index(marker) + len(marker):]
    # Try parsing as JSON array directly
    data = json.loads(text.strip())
    if isinstance(data, list):
        return data
    raise ValueError("Expected a JSON array of step objects")


def load_raw_log(lines: list[str]) -> list[dict]:
    """Parse raw log lines into step dicts, including R1 event counts."""
    steps = []
    # First pass: find step metric lines
    step_indices = []
    for idx, line in enumerate(lines):
        d = parse_step_metrics(line)
        if d and d["step"] <= 10000:
            step_indices.append((idx, d))

    # Second pass: count R=1 events between step lines for R1 rate
    for i, (idx, d) in enumerate(step_indices):
        start = step_indices[i - 1][0] + 1 if i > 0 else 0
        end = idx
        r1 = 0
        total_traj = 0
        for j in range(start, end):
            m = TRAJ_COMPLETE_RE.search(ANSI_RE.sub("", lines[j]))
            if m:
                total_traj += 1
                if float(m.group(3)) >= 0.99:
                    r1 += 1
        if total_traj > 0:
            d["events/reward_1"] = r1
            d["events/total_completions"] = total_traj
        steps.append(d)

    return steps


def load_input(path: str) -> list[dict]:
    """Auto-detect JSON vs raw log format."""
    if path == "-":
        text = sys.stdin.read()
    else:
        text = Path(path).read_text()

    stripped = text.lstrip()
    # Heuristic: JSON starts with '[' or has the JSON marker
    if stripped.startswith("[") or "─── JSON ───" in text:
        return load_json(text)
    return load_raw_log(text.splitlines())


# ─── Plotting helpers ─────────────────────────────────────────────────────

def rolling_avg(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Compute rolling average with same-length output (edge-padded)."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    # Use 'valid' convolution then pad edges
    smoothed = np.convolve(values, kernel, mode="valid")
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    return np.concatenate([
        np.full(pad_left, smoothed[0]),
        smoothed,
        np.full(pad_right, smoothed[-1]),
    ])


def plot_metric(ax, steps, values, label, color="C0", ylabel=None, ylim=None,
                smooth=True, window=10):
    """Plot a single metric: raw (light) + rolling average (solid)."""
    arr = np.array(values, dtype=float)
    if smooth and len(arr) > window:
        ax.plot(steps, arr, color=color, alpha=0.25, linewidth=0.8)
        ax.plot(steps, rolling_avg(arr, window), color=color, linewidth=1.5, label=label)
    else:
        ax.plot(steps, arr, color=color, linewidth=1.2, label=label)
    ax.set_ylabel(ylabel or label, fontsize=9)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=8)


# ─── Main figure ──────────────────────────────────────────────────────────

def make_figure(steps_data: list[dict], smooth: bool = True, window: int = 10,
                title: str | None = None):
    """Create the 4x2 multi-panel training metrics figure."""
    x = [d["step"] for d in steps_data]

    def get(key, default=0.0):
        return [d.get(key, default) for d in steps_data]

    fig, axes = plt.subplots(4, 2, figsize=(16, 18), constrained_layout=True)

    # ── Panel 1: Score + Val Score ──
    ax = axes[0, 0]
    scores = get("critic/score/mean")
    plot_metric(ax, x, scores, "Train Score", color="C0", ylabel="Score",
                ylim=(0, 1), smooth=smooth, window=window)
    # Overlay val_score if present
    val_x, val_y = [], []
    for d in steps_data:
        vs = d.get("val/test_score/unknown")
        if vs is not None:
            val_x.append(d["step"])
            val_y.append(vs)
    if val_y:
        ax.scatter(val_x, val_y, color="red", marker="^", s=80, zorder=5,
                   label="Val Score", edgecolors="darkred", linewidths=0.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Score", fontsize=10, fontweight="bold")

    # ── Panel 2: Entropy ──
    ax = axes[0, 1]
    entropy = get("actor/entropy")
    plot_metric(ax, x, entropy, "Entropy", color="C1", ylabel="Entropy",
                smooth=smooth, window=window)
    ax.set_title("Entropy", fontsize=10, fontweight="bold")

    # ── Panel 3: PG Loss ──
    ax = axes[1, 0]
    pg_loss = get("actor/pg_loss")
    plot_metric(ax, x, pg_loss, "PG Loss", color="C2", ylabel="PG Loss",
                smooth=smooth, window=window)
    ax.set_title("Policy Gradient Loss", fontsize=10, fontweight="bold")

    # ── Panel 4: Grad Norm ──
    ax = axes[1, 1]
    grad_norm = get("actor/grad_norm")
    plot_metric(ax, x, grad_norm, "Grad Norm", color="C3", ylabel="Grad Norm",
                smooth=smooth, window=window)
    ax.set_title("Gradient Norm", fontsize=10, fontweight="bold")

    # ── Panel 5: KL Divergence ──
    ax = axes[2, 0]
    kl = get("actor/ppo_kl")
    plot_metric(ax, x, kl, "KL", color="C4", ylabel="KL Divergence",
                smooth=smooth, window=window)
    ax.set_title("KL Divergence", fontsize=10, fontweight="bold")

    # ── Panel 6: Response Length ──
    ax = axes[2, 1]
    resp_len = get("response_length/mean")
    plot_metric(ax, x, resp_len, "Resp Length", color="C5", ylabel="Tokens",
                smooth=smooth, window=window)
    ax.set_title("Response Length", fontsize=10, fontweight="bold")

    # ── Panel 7: pg_clipfrac ──
    ax = axes[3, 0]
    clipfrac = get("actor/pg_clipfrac")
    plot_metric(ax, x, clipfrac, "pg_clipfrac", color="C6", ylabel="Clip Fraction",
                ylim=(0, 1), smooth=smooth, window=window)
    ax.set_title("PPO Clip Fraction", fontsize=10, fontweight="bold")
    ax.set_xlabel("Training Step", fontsize=9)

    # ── Panel 8: R1 Rate ──
    ax = axes[3, 1]
    # Compute R1 rate from events if available
    r1_rates = []
    for d in steps_data:
        r1 = d.get("events/reward_1", 0)
        total = d.get("events/total_completions")
        if total is None:
            # Estimate from batch solve counts
            sn = d.get("batch/solve_none", 0)
            sp = d.get("batch/solve_partial", 0)
            sa = d.get("batch/solve_all", 0)
            total = sn + sp + sa
        if total and total > 0:
            r1_rates.append(r1 / total)
        else:
            r1_rates.append(0.0)
    plot_metric(ax, x, r1_rates, "R1 Rate", color="C8", ylabel="R1 Rate",
                ylim=(0, 1), smooth=smooth, window=window)
    ax.set_title("R1 Rate (Reward=1 / Total)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Training Step", fontsize=9)

    # ── Figure title ──
    fig_title = title or "DeepSWE Training Metrics (Qwen3-32B, 2x8 H200)"
    fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=1.01)

    return fig


# ─── CLI ──────────────────────────────────────────────────────────────────

def parse_step_range(spec: str) -> tuple[int | None, int | None]:
    if "-" in spec:
        parts = spec.split("-", 1)
        lo = int(parts[0]) if parts[0] else None
        hi = int(parts[1]) if parts[1] else None
        return lo, hi
    return int(spec), int(spec)


def main():
    parser = argparse.ArgumentParser(
        description="Plot DeepSWE training metrics as a multi-panel figure."
    )
    parser.add_argument("input", help="JSON file or raw log (use '-' for stdin)")
    parser.add_argument("-o", "--output", default="training_metrics.png",
                        help="Output PNG path (default: training_metrics.png)")
    parser.add_argument("--steps", default=None,
                        help="Step range filter (e.g., 61-343, 100-)")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable rolling average overlay")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI (default: 150)")
    parser.add_argument("--title", default=None,
                        help="Override figure title")
    parser.add_argument("--window", type=int, default=10,
                        help="Rolling average window size (default: 10)")
    args = parser.parse_args()

    # Load data
    steps_data = load_input(args.input)
    if not steps_data:
        print("No step data found in input.", file=sys.stderr)
        sys.exit(1)

    # Apply step range filter
    if args.steps:
        lo, hi = parse_step_range(args.steps)
        steps_data = [
            d for d in steps_data
            if (lo is None or d["step"] >= lo) and (hi is None or d["step"] <= hi)
        ]
        if not steps_data:
            print("No steps match the specified range.", file=sys.stderr)
            sys.exit(1)

    # Sort by step
    steps_data.sort(key=lambda d: d["step"])

    print(f"Plotting {len(steps_data)} steps "
          f"({int(steps_data[0]['step'])} -> {int(steps_data[-1]['step'])})")

    # Generate figure
    fig = make_figure(
        steps_data,
        smooth=not args.no_smooth,
        window=args.window,
        title=args.title,
    )

    # Save
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
