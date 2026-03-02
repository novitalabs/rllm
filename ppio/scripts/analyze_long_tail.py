#!/usr/bin/env python3
"""Analyze long-tail trajectory characteristics from training logs.

Usage:
    python ppio/scripts/analyze_long_tail.py [METRICS_FILE] [TRAIN_LOG]

Defaults:
    METRICS_FILE: /tmp/k8s_training_metrics.txt
    TRAIN_LOG:    /tmp/train_run35.log (optional, for completion order analysis)
"""

import re
import sys


def parse_metrics(filepath: str) -> list[dict]:
    with open(filepath) as f:
        lines = f.readlines()
    steps = []
    for line in lines:
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        m = re.match(r"\(TaskRunner pid=\d+\) step:(\d+)", clean)
        if not m:
            continue
        d = {"step": int(m.group(1))}
        for match in re.finditer(r"([\w/]+):(?:np\.\w+\()?([\d.e+-]+)\)?", clean):
            try:
                d[match.group(1)] = float(match.group(2))
            except ValueError:
                pass
        steps.append(d)
    return steps


def analyze_tail(steps: list[dict]):
    print("=" * 75)
    print("LONG-TAIL TRAJECTORY ANALYSIS")
    print("=" * 75)

    for d in steps:
        step = d["step"]
        total_max = d.get("traj/total_time_max", 0)
        total_mean = d.get("traj/total_time_mean", 0)
        llm_max = d.get("traj/llm_time_max", 0)
        env_max = d.get("traj/env_time_max", 0)
        steps_max = d.get("traj/steps_max", 0)
        steps_mean = d.get("traj/steps_mean", 0)
        collect = d.get("timing_s/collect_trajectory", 0)
        reward_max = d.get("traj/reward_time_max", 0)
        resp_max = d.get("response_length/max", 0)

        per_step_llm_max = llm_max / steps_max if steps_max > 0 else 0
        per_step_llm_avg = d.get("traj/llm_time_mean", 0) / steps_mean if steps_mean > 0 else 0

        print(f"\n--- Step {step:.0f} ---")
        print(f"  Batch wall-clock: {collect/60:.1f} min")
        print(f"  Tail trajectory:  {total_max/60:.1f} min (ratio: {total_max/total_mean:.2f}x mean)")
        print(f"  Bubble time:      {(collect - total_max)/60:.1f} min")
        print(f"  GPU-hours wasted: {32 * (collect - total_mean) / 3600:.1f} h")
        print(f"  Tail breakdown:")
        print(f"    LLM:      {llm_max/60:.1f} min ({llm_max/total_max*100:.1f}% of total)")
        print(f"    Env:      {env_max/60:.1f} min ({env_max/total_max*100:.1f}%)")
        print(f"    Reward:   {reward_max:.1f}s")
        print(f"    Steps:    {steps_max:.0f} (mean={steps_mean:.1f})")
        print(f"    Resp tok: {resp_max:.0f}")
        print(f"  Per-step LLM: tail={per_step_llm_max:.1f}s  avg={per_step_llm_avg:.1f}s  ratio={per_step_llm_max/per_step_llm_avg:.2f}x" if per_step_llm_avg > 0 else "")


def analyze_completion_order(log_path: str):
    """Parse trajectory completion order from training log."""
    print("\n" + "=" * 75)
    print("TRAJECTORY COMPLETION ORDER (last step in log)")
    print("=" * 75)

    try:
        with open(log_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"  Log file not found: {log_path}")
        return

    events = []
    for i, line in enumerate(lines):
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        m = re.search(r"Trajectory (\d+) completed due to: (\w+)\. Reward is ([0-9.]+)", clean)
        if m:
            reward_str = m.group(3).rstrip(".")
            events.append({
                "order": len(events) + 1,
                "traj_idx": int(m.group(1)),
                "reason": m.group(2),
                "reward": float(reward_str),
            })
        m2 = re.search(r"Trajectory (\d+) is masked out", clean)
        if m2 and events and events[-1]["traj_idx"] == int(m2.group(1)):
            events[-1]["masked"] = True

    if not events:
        print("  No trajectory events found")
        return

    # By reason
    by_reason = {}
    for e in events:
        by_reason.setdefault(e["reason"], []).append(e)

    print(f"\n  Total completions: {len(events)}")
    for reason, es in sorted(by_reason.items()):
        masked = sum(1 for e in es if e.get("masked"))
        rewards_pos = sum(1 for e in es if e["reward"] > 0)
        print(f"  {reason}: {len(es)} (masked={masked}, reward>0={rewards_pos})")

    print(f"\n  TAIL (last 10):")
    for e in sorted(events, key=lambda x: x["order"])[-10:]:
        m = " MASKED" if e.get("masked") else ""
        print(f"    #{e['order']:2d}/{len(events)}: Traj {e['traj_idx']:2d} → {e['reason']:12s} rwd={e['reward']:.1f}{m}")

    print(f"\n  HEAD (first 5):")
    for e in sorted(events, key=lambda x: x["order"])[:5]:
        m = " MASKED" if e.get("masked") else ""
        print(f"    #{e['order']:2d}/{len(events)}: Traj {e['traj_idx']:2d} → {e['reason']:12s} rwd={e['reward']:.1f}{m}")


def print_waste_summary(steps: list[dict]):
    n = len(steps)
    ngpus = 32

    print("\n" + "=" * 75)
    print("WASTE QUANTIFICATION")
    print("=" * 75)

    total_collect = sum(d.get("timing_s/collect_trajectory", 0) for d in steps)
    total_step = sum(d.get("timing_s/step", 0) for d in steps)
    total_update = sum(d.get("timing_s/update_actor", 0) for d in steps)
    total_mean_traj = sum(d.get("traj/total_time_mean", 0) for d in steps)

    print(f"\n  Over {n} steps:")
    print(f"    Total wall-clock:    {total_step/3600:.1f} h")
    print(f"    Total GPU-hours:     {total_step*ngpus/3600:.0f} h")
    print(f"    GPU-h training:      {total_update*ngpus/3600:.1f} h ({total_update/total_step*100:.1f}%)")
    print(f"    GPU-h idle (tail):   {(total_collect-total_mean_traj)*ngpus/3600:.0f} h")

    # max_steps sensitivity
    llm_max_sum = sum(d.get("traj/llm_time_max", 0) for d in steps)
    for target_steps in [35, 30, 25]:
        est = llm_max_sum * target_steps / 50
        saved = (llm_max_sum - est) / n
        print(f"\n  If max_steps={target_steps}:")
        print(f"    Est. savings/step:   {saved/60:.0f} min")
        print(f"    Est. daily (20 steps): {20*saved*ngpus/3600:.0f} GPU-h saved")


def main():
    metrics_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/k8s_training_metrics.txt"
    train_log = sys.argv[2] if len(sys.argv) > 2 else "/tmp/train_run35.log"

    steps = parse_metrics(metrics_file)
    if not steps:
        print(f"No metrics in {metrics_file}")
        sys.exit(1)

    analyze_tail(steps)
    analyze_completion_order(train_log)
    print_waste_summary(steps)


if __name__ == "__main__":
    main()
