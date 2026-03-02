#!/usr/bin/env python3
"""Parse DeepSWE training metrics from log files.

Supports two input modes:
  1. Raw K8s pod logs (from fetch_k8s_metrics.sh or kubectl logs)
  2. Pre-extracted metric lines

Usage:
    # From raw K8s logs (auto-detects format):
    ./ppio/scripts/fetch_k8s_metrics.sh | python ppio/scripts/parse_training_metrics.py -

    # From a saved log file:
    python ppio/scripts/parse_training_metrics.py /tmp/training.log

    # Fetch + parse + save in one go:
    ./ppio/scripts/fetch_k8s_metrics.sh -o /tmp/raw.log && python ppio/scripts/parse_training_metrics.py /tmp/raw.log

Options:
    --json          Also output JSON data
    --csv           Also output CSV data
    --no-events     Skip per-step event summary
    --steps N-M     Only show steps in range (e.g., 1-9, 5-)
"""

import re
import sys
import json
import csv
import io
from collections import defaultdict
from dataclasses import dataclass, field


# ─── ANSI escape stripper ───────────────────────────────────────────────────

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


# ─── Log line timestamp extractor (K8s CRI format) ─────────────────────────

# CRI log format: 2026-02-28T21:21:15.720203643+08:00 stdout F <message>
CRI_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:\d{2})\s+\w+\s+\w\s+")

def extract_timestamp(line: str) -> str | None:
    m = CRI_TS_RE.match(line)
    return m.group(1) if m else None


# ─── Step metrics parser ───────────────────────────────────────────────────

STEP_RE = re.compile(r"step:(\d+)\s")
# Match: key:value or key:np.float64(value) or key:np.int64(value)
KV_RE = re.compile(r"([\w/.]+):(?:np\.\w+\()?([-\d.e+]+)\)?")

def parse_step_metrics(line: str) -> dict | None:
    clean = strip_ansi(line).strip()
    # Remove CRI timestamp prefix if present
    clean = CRI_TS_RE.sub("", clean)
    # Remove Ray prefix
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


# ─── Trajectory event parser ──────────────────────────────────────────────

TRAJ_COMPLETE_RE = re.compile(
    r"Trajectory (\d+) completed due to: (\w+)\. Reward is ([\d.]+?)\.?\s"
)
TRAJ_PROGRESS_RE = re.compile(
    r"Number of Trajectories (\d+)/(\d+) completed"
)
TRAJ_OVERLONG_RE = re.compile(
    r"Trajectory (\d+) is masked out due to overlong"
)
TRAJ_DUMMY_RE = re.compile(
    r"Trajectory (\d+) failed, returning dummy result"
)


@dataclass
class StepEvents:
    """Per-step trajectory events collected between step metric lines."""
    completions: dict = field(default_factory=lambda: defaultdict(int))  # reason -> count
    rewards_1: int = 0
    rewards_0: int = 0
    overlong: int = 0
    errors: int = 0
    total_in_batch: int = 0  # from N/M progress line

    def total_completions(self) -> int:
        return sum(self.completions.values())


def collect_events(lines: list[str], step_line_indices: list[int]) -> dict[int, StepEvents]:
    """Collect per-step trajectory events from log lines.

    Events between step N-1 metrics and step N metrics belong to step N.
    """
    # Build ranges: events before first step line -> step of first metric
    # events between step_line[i-1] and step_line[i] -> step of step_line[i]
    ranges = []
    for i, idx in enumerate(step_line_indices):
        start = step_line_indices[i - 1] + 1 if i > 0 else 0
        end = idx
        # Parse the step number from the metric line
        d = parse_step_metrics(lines[idx])
        if d:
            ranges.append((start, end, d["step"]))

    events_by_step: dict[int, StepEvents] = {}

    for start, end, step_num in ranges:
        ev = StepEvents()
        for j in range(start, end):
            clean = strip_ansi(lines[j])

            m = TRAJ_COMPLETE_RE.search(clean)
            if m:
                reason = m.group(2)
                reward = float(m.group(3))
                ev.completions[reason] += 1
                if reward >= 0.99:
                    ev.rewards_1 += 1
                else:
                    ev.rewards_0 += 1

            m = TRAJ_PROGRESS_RE.search(clean)
            if m:
                ev.total_in_batch = int(m.group(2))

            if TRAJ_OVERLONG_RE.search(clean):
                ev.overlong += 1

            if TRAJ_DUMMY_RE.search(clean):
                ev.errors += 1

        events_by_step[step_num] = ev

    return events_by_step


# ─── Table formatters ──────────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    """Format seconds as Xs or Xm."""
    if seconds >= 120:
        return f"{seconds/60:.0f}m"
    return f"{seconds:.0f}s"


def print_timing_table(steps: list[dict]):
    print("\n╔══════════════════════════════════════════════════════════════════════════════════╗")
    print("║                          TIMING BREAKDOWN (per step)                            ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════╝\n")
    hdr = f"{'Step':>4} │ {'Total':>7} │ {'Collect':>7} {'%':>5} │ {'Update':>7} {'%':>5} │ {'LogProb':>7} │ {'Xform':>6} │ {'MFU':>5}"
    print(hdr)
    print("─" * len(hdr))
    totals = defaultdict(float)
    for d in steps:
        s = d["step"]
        total = d.get("timing_s/step", 0)
        collect = d.get("timing_s/collect_trajectory", 0)
        update = d.get("timing_s/update_actor", 0)
        logprob = d.get("timing_s/old_log_prob", 0)
        xform = d.get("timing_s/transform_trajectory", 0)
        mfu = d.get("perf/mfu/actor", 0)
        cp = collect / total * 100 if total else 0
        up = update / total * 100 if total else 0
        print(f"{s:4.0f} │ {fmt_time(total):>7} │ {fmt_time(collect):>7} {cp:4.1f}% │ {fmt_time(update):>7} {up:4.1f}% │ {logprob:6.1f}s │ {xform:5.2f}s │ {mfu*100:4.1f}%")
        for k in ["timing_s/step", "timing_s/collect_trajectory", "timing_s/update_actor",
                   "timing_s/old_log_prob", "timing_s/transform_trajectory", "perf/mfu/actor"]:
            totals[k] += d.get(k, 0)
    n = len(steps)
    if n > 1:
        print("─" * len(hdr))
        total = totals["timing_s/step"] / n
        collect = totals["timing_s/collect_trajectory"] / n
        update = totals["timing_s/update_actor"] / n
        logprob = totals["timing_s/old_log_prob"] / n
        xform = totals["timing_s/transform_trajectory"] / n
        mfu = totals["perf/mfu/actor"] / n
        cp = collect / total * 100 if total else 0
        up = update / total * 100 if total else 0
        print(f" AVG │ {fmt_time(total):>7} │ {fmt_time(collect):>7} {cp:4.1f}% │ {fmt_time(update):>7} {up:4.1f}% │ {logprob:6.1f}s │ {xform:5.2f}s │ {mfu*100:4.1f}%")


def print_trajectory_table(steps: list[dict]):
    print("\n╔══════════════════════════════════════════════════════════════════════════════════╗")
    print("║                       TRAJECTORY METRICS (per step)                             ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════╝\n")
    hdr = f"{'Step':>4} │ {'LLM_mean':>8} {'LLM_max':>8} │ {'Env_mean':>8} {'Env_max':>8} │ {'Total_max':>9} │ {'Tail':>5} │ {'Steps':>5} │ {'Mis%':>5}"
    print(hdr)
    print("─" * len(hdr))
    for d in steps:
        s = d["step"]
        lm = d.get("traj/llm_time_mean", 0)
        lx = d.get("traj/llm_time_max", 0)
        em = d.get("traj/env_time_mean", 0)
        ex = d.get("traj/env_time_max", 0)
        tm = d.get("traj/total_time_mean", 0)
        tx = d.get("traj/total_time_max", 0)
        tail = tx / tm if tm > 0 else 0
        steps_m = d.get("traj/steps_mean", 0)
        mis = d.get("traj/token_mismatch_mean", 0)
        print(f"{s:4.0f} │ {fmt_time(lm):>8} {fmt_time(lx):>8} │ {fmt_time(em):>8} {fmt_time(ex):>8} │ {fmt_time(tx):>9} │ {tail:4.1f}x │ {steps_m:5.1f} │ {mis*100:4.1f}%")


def print_training_table(steps: list[dict]):
    print("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                                    TRAINING METRICS (per step)                                         ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n")
    hdr = (f"{'Step':>4} │ {'Score':>6} │ {'s_none':>6} {'s_part':>6} {'s_all':>5} │ "
           f"{'PG_loss':>9} │ {'Grad':>7} │ {'Clip%':>6} {'ClipL%':>6} │ "
           f"{'Entropy':>8} │ {'KL':>8} │ {'Resp_len':>8} {'Clip%':>5} │ "
           f"{'Adv_mean':>8} │ {'GPU_GB':>6} │ {'CPU_GB':>6}")
    print(hdr)
    print("─" * len(hdr))
    for d in steps:
        s = d["step"]
        score = d.get("critic/score/mean", 0)
        sn = d.get("batch/solve_none", 0)
        sp = d.get("batch/solve_partial", 0)
        sa = d.get("batch/solve_all", 0)
        pg = d.get("actor/pg_loss", 0)
        gn = d.get("actor/grad_norm", 0)
        cf = d.get("actor/pg_clipfrac", 0)
        cfl = d.get("actor/pg_clipfrac_lower", 0)
        ent = d.get("actor/entropy", 0)
        kl = d.get("actor/ppo_kl", 0)
        rl = d.get("response_length/mean", 0)
        rc = d.get("response_length/clip_ratio", 0)
        adv = d.get("critic/advantages/mean", 0)
        gpu = d.get("perf/max_memory_allocated_gb", 0)
        cpu = d.get("perf/cpu_memory_used_gb", 0)
        print(
            f"{s:4.0f} │ {score:6.3f} │ {sn:6.0f} {sp:6.0f} {sa:5.0f} │ "
            f"{pg:9.2f} │ {gn:7.1f} │ {cf*100:5.1f}% {cfl*100:5.1f}% │ "
            f"{ent:8.0f} │ {kl:8.5f} │ {rl:8.0f} {rc*100:4.1f}% │ "
            f"{adv:8.4f} │ {gpu:6.1f} │ {cpu:6.1f}"
        )


def print_response_length_table(steps: list[dict]):
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║                  SEQUENCE LENGTH STATS (per step)             ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    hdr = f"{'Step':>4} │ {'Prompt':>8} {'(max)':>7} │ {'Resp_mean':>9} {'Resp_max':>9} {'Resp_min':>9} │ {'Abort%':>6}"
    print(hdr)
    print("─" * len(hdr))
    for d in steps:
        s = d["step"]
        pm = d.get("prompt_length/mean", 0)
        px = d.get("prompt_length/max", 0)
        rm = d.get("response_length/mean", 0)
        rx = d.get("response_length/max", 0)
        rn = d.get("response_length/min", 0)
        ab = d.get("response/aborted_ratio", 0)
        print(f"{s:4.0f} │ {pm:8.0f} {px:7.0f} │ {rm:9.0f} {rx:9.0f} {rn:9.0f} │ {ab*100:5.1f}%")


def print_events_table(steps: list[dict], events: dict[int, StepEvents]):
    print("\n╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                    TRAJECTORY OUTCOMES (per step)                            ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝\n")
    hdr = f"{'Step':>4} │ {'Total':>5} │ {'ENV_DONE':>8} {'MAX_STEP':>8} {'TRUNC':>6} │ {'R=1':>4} {'R=0':>4} │ {'Overlong':>8} {'Errors':>6}"
    print(hdr)
    print("─" * len(hdr))
    sum_done = sum_max = sum_trunc = sum_r1 = sum_r0 = sum_ol = sum_err = 0
    for d in steps:
        s = int(d["step"])
        ev = events.get(s)
        if not ev:
            continue
        done = ev.completions.get("ENV_DONE", 0)
        maxs = ev.completions.get("MAX_STEPS", 0)
        trunc = ev.completions.get("TRUNCATION", 0)
        total = ev.total_in_batch or ev.total_completions()
        print(
            f"{s:4} │ {total:5} │ {done:8} {maxs:8} {trunc:6} │ "
            f"{ev.rewards_1:4} {ev.rewards_0:4} │ {ev.overlong:8} {ev.errors:6}"
        )
        sum_done += done; sum_max += maxs; sum_trunc += trunc
        sum_r1 += ev.rewards_1; sum_r0 += ev.rewards_0
        sum_ol += ev.overlong; sum_err += ev.errors
    n = len([d for d in steps if int(d["step"]) in events])
    if n > 1:
        print("─" * len(hdr))
        total_all = sum_done + sum_max + sum_trunc
        print(
            f" SUM │ {total_all:5} │ {sum_done:8} {sum_max:8} {sum_trunc:6} │ "
            f"{sum_r1:4} {sum_r0:4} │ {sum_ol:8} {sum_err:6}"
        )


def print_summary(steps: list[dict], events: dict[int, StepEvents]):
    n = len(steps)
    if n == 0:
        return

    avg = lambda key: sum(d.get(key, 0) for d in steps) / n

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║                         SUMMARY                             ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    print(f"  Steps analyzed:         {n} (step {int(steps[0]['step'])} → {int(steps[-1]['step'])})")
    print(f"  Avg step time:          {avg('timing_s/step')/60:.1f} min ({avg('timing_s/step'):.0f}s)")
    print(f"  Avg collect_trajectory: {avg('timing_s/collect_trajectory')/60:.1f} min ({avg('timing_s/collect_trajectory')/avg('timing_s/step')*100:.1f}%)")
    print(f"  Avg update_actor:       {avg('timing_s/update_actor'):.0f}s ({avg('timing_s/update_actor')/avg('timing_s/step')*100:.1f}%)")
    print(f"  Avg MFU (update only):  {avg('perf/mfu/actor')*100:.1f}%")
    print()
    print(f"  Avg tail ratio:         {sum(d.get('traj/total_time_max',0)/max(d.get('traj/total_time_mean',1),1) for d in steps)/n:.2f}x")
    print(f"  Avg mismatch rate:      {avg('traj/token_mismatch_mean')*100:.1f}%")
    print(f"  Avg agent steps:        {avg('traj/steps_mean'):.1f}")
    print()
    print(f"  Avg score (reward):     {avg('critic/score/mean'):.3f}")
    print(f"  Avg pg_loss:            {avg('actor/pg_loss'):.2f}")
    print(f"  Avg grad_norm:          {avg('actor/grad_norm'):.1f}")
    print(f"  Avg entropy:            {avg('actor/entropy'):.0f}")
    print(f"  Avg pg_clipfrac:        {avg('actor/pg_clipfrac')*100:.2f}%")
    print(f"  Avg resp_length:        {avg('response_length/mean'):.0f}")

    # Trajectory outcome totals
    total_r1 = sum(ev.rewards_1 for ev in events.values() if int(list(events.keys())[list(events.values()).index(ev)]) in {int(d["step"]) for d in steps})
    total_ol = sum(ev.overlong for ev in events.values())
    total_completions = sum(ev.total_completions() for ev in events.values())
    if total_completions > 0:
        print(f"\n  Total trajectories:     {total_completions}")
        print(f"  Reward=1 trajectories:  {total_r1} ({total_r1/total_completions*100:.1f}%)")
        print(f"  Overlong filtered:      {total_ol} ({total_ol/total_completions*100:.1f}%)")

    ngpus = 32
    total_step_time = sum(d.get("timing_s/step", 0) for d in steps)
    total_collect = sum(d.get("timing_s/collect_trajectory", 0) for d in steps)
    print(f"\n  Total wall time:        {total_step_time/3600:.1f}h")
    print(f"  GPU-hours (total):      {total_step_time*ngpus/3600:.0f}h")
    print(f"  GPU-hours (idle):       {total_collect*ngpus/3600:.0f}h (during trajectory collection)")


# ─── Validation progress parser ────────────────────────────────────────────

def check_validation_progress(lines: list[str]):
    """Check if there's an ongoing validation run and report progress."""
    # Look for the last N/500 pattern (validation uses 500 trajectories)
    last_val_progress = None
    val_reward_1 = 0
    val_started = False

    for line in lines:
        clean = strip_ansi(line)
        m = TRAJ_PROGRESS_RE.search(clean)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            if total > 64:  # Validation batch (>64 = not training batch)
                last_val_progress = (done, total)
                if not val_started:
                    val_started = True
                    val_reward_1 = 0  # Reset counter

        if val_started:
            m = TRAJ_COMPLETE_RE.search(clean)
            if m and float(m.group(3)) >= 0.99:
                val_reward_1 += 1

    if last_val_progress:
        done, total = last_val_progress
        pct = done / total * 100
        print(f"\n╔══════════════════════════════════════════════════════════════╗")
        print(f"║                    VALIDATION IN PROGRESS                    ║")
        print(f"╚══════════════════════════════════════════════════════════════╝\n")
        print(f"  Progress: {done}/{total} ({pct:.0f}%)")
        print(f"  Reward=1: {val_reward_1} (current rate: {val_reward_1/done*100:.1f}%)" if done > 0 else "")
        print(f"  Estimated resolve rate: {val_reward_1/total*100:.1f}% (extrapolated)" if done > total * 0.1 else "")


# ─── Main ──────────────────────────────────────────────────────────────────

def parse_step_range(spec: str) -> tuple[int | None, int | None]:
    """Parse step range spec like '1-9', '5-', '-9', '5'."""
    if "-" in spec:
        parts = spec.split("-", 1)
        lo = int(parts[0]) if parts[0] else None
        hi = int(parts[1]) if parts[1] else None
        return lo, hi
    return int(spec), int(spec)


def main():
    args = sys.argv[1:]
    log_file = None
    output_json = False
    output_csv = False
    show_events = True
    step_range = (None, None)

    i = 0
    while i < len(args):
        if args[i] == "--json":
            output_json = True
        elif args[i] == "--csv":
            output_csv = True
        elif args[i] == "--no-events":
            show_events = False
        elif args[i] == "--steps" and i + 1 < len(args):
            i += 1
            step_range = parse_step_range(args[i])
        elif not args[i].startswith("-") or args[i] == "-":
            log_file = args[i]
        i += 1

    if log_file == "-":
        lines = sys.stdin.readlines()
        source = "stdin"
    elif log_file:
        with open(log_file) as f:
            lines = f.readlines()
        source = log_file
    else:
        # Try default paths
        for path in ["/tmp/k8s_training_metrics.txt", "/tmp/training.log"]:
            try:
                with open(path) as f:
                    lines = f.readlines()
                source = path
                break
            except FileNotFoundError:
                continue
        else:
            print("No log file specified and no default found. Usage:", file=sys.stderr)
            print("  fetch_k8s_metrics.sh | python parse_training_metrics.py -", file=sys.stderr)
            print("  python parse_training_metrics.py LOG_FILE", file=sys.stderr)
            sys.exit(1)

    # Find all step metric lines
    step_line_indices = []
    steps = []
    for idx, line in enumerate(lines):
        d = parse_step_metrics(line)
        if d:
            step_num = d["step"]
            # Filter out false positives (timing values parsed as step numbers)
            if step_num > 10000:
                continue
            step_line_indices.append(idx)
            steps.append(d)

    if not steps:
        print(f"No training step metrics found in {source}", file=sys.stderr)
        # Still check for validation progress
        check_validation_progress(lines)
        sys.exit(0)

    # Apply step range filter
    lo, hi = step_range
    if lo is not None or hi is not None:
        filtered = []
        filtered_indices = []
        for d, idx in zip(steps, step_line_indices):
            s = d["step"]
            if (lo is None or s >= lo) and (hi is None or s <= hi):
                filtered.append(d)
                filtered_indices.append(idx)
        steps = filtered
        step_line_indices = filtered_indices

    if not steps:
        print("No steps match the specified range.", file=sys.stderr)
        sys.exit(1)

    # Collect per-step events
    events = collect_events(lines, step_line_indices) if show_events else {}

    print(f"Parsed {len(steps)} training steps from {source}")
    print(f"Steps: {int(steps[0]['step'])} → {int(steps[-1]['step'])}")

    # Print all tables
    print_timing_table(steps)
    print_trajectory_table(steps)
    print_training_table(steps)
    print_response_length_table(steps)
    if show_events and events:
        print_events_table(steps, events)
    print_summary(steps, events)
    check_validation_progress(lines)

    # Optional outputs
    if output_json:
        print("\n─── JSON ───")
        out = []
        for d in steps:
            row = dict(d)
            s = int(d["step"])
            if s in events:
                ev = events[s]
                row["events/env_done"] = ev.completions.get("ENV_DONE", 0)
                row["events/max_steps"] = ev.completions.get("MAX_STEPS", 0)
                row["events/truncation"] = ev.completions.get("TRUNCATION", 0)
                row["events/reward_1"] = ev.rewards_1
                row["events/overlong"] = ev.overlong
                row["events/errors"] = ev.errors
            out.append(row)
        print(json.dumps(out, indent=2))

    if output_csv:
        print("\n─── CSV ───")
        if steps:
            all_keys = sorted(set().union(*(d.keys() for d in steps)))
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=all_keys)
            writer.writeheader()
            for d in steps:
                writer.writerow({k: d.get(k, "") for k in all_keys})
            print(buf.getvalue())


if __name__ == "__main__":
    main()
