#!/usr/bin/env python3
"""Extract LLM inference metrics from training logs.

Usage:
    python ppio/scripts/extract_llm_metrics.py [METRICS_FILE]

Outputs vLLM inference throughput estimates derived from trajectory-level timing.
"""

import re
import sys
import json


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


def compute_inference_metrics(steps: list[dict]):
    """Derive per-step and per-token inference metrics from trajectory data."""
    n_traj = 64  # batch_size × n_samples_per_prompt = 8 × 8

    print("=" * 75)
    print("vLLM INFERENCE METRICS (derived from trajectory timing)")
    print("=" * 75)

    print(f"\n{'Step':>4} | {'LLM/step':>9} | {'Tok/traj':>9} | {'Tok/s/traj':>10} | "
          f"{'Batch tok/s':>11} | {'ms/tok':>7} | {'Steps':>5}")
    print("-" * 80)

    all_tok_per_s = []
    all_ms_per_tok = []
    all_batch_tok_per_s = []

    for d in steps:
        step = d["step"]
        llm_mean = d.get("traj/llm_time_mean", 0)
        steps_mean = d.get("traj/steps_mean", 0)
        resp_mean = d.get("response_length/mean", 0)
        prompt_mean = d.get("prompt_length/mean", 0)

        # Per-step LLM time
        llm_per_step = llm_mean / steps_mean if steps_mean > 0 else 0

        # Total tokens generated per trajectory (response only)
        tok_per_traj = resp_mean

        # Throughput: tokens generated per second per trajectory
        tok_per_s = tok_per_traj / llm_mean if llm_mean > 0 else 0

        # Batch throughput: total tokens / collect_trajectory time
        collect = d.get("timing_s/collect_trajectory", 0)
        batch_tok_total = resp_mean * n_traj
        batch_tok_per_s = batch_tok_total / collect if collect > 0 else 0

        # Latency: ms per generated token (average)
        ms_per_tok = (llm_mean * 1000) / tok_per_traj if tok_per_traj > 0 else 0

        all_tok_per_s.append(tok_per_s)
        all_ms_per_tok.append(ms_per_tok)
        all_batch_tok_per_s.append(batch_tok_per_s)

        print(
            f"{step:4.0f} | {llm_per_step:8.1f}s | {tok_per_traj:9.0f} | {tok_per_s:9.1f} | "
            f"{batch_tok_per_s:10.0f} | {ms_per_tok:6.1f} | {steps_mean:5.1f}"
        )

    n = len(steps)
    if n > 0:
        print("-" * 80)
        print(f" AVG | {'':>9} | {'':>9} | {sum(all_tok_per_s)/n:9.1f} | "
              f"{sum(all_batch_tok_per_s)/n:10.0f} | {sum(all_ms_per_tok)/n:6.1f} |")

    return all_tok_per_s, all_ms_per_tok, all_batch_tok_per_s


def print_tail_inference(steps: list[dict]):
    """Show inference metrics for the tail (slowest) trajectory."""
    print("\n" + "=" * 75)
    print("TAIL TRAJECTORY INFERENCE (slowest per batch)")
    print("=" * 75)

    print(f"\n{'Step':>4} | {'LLM max':>8} | {'Steps':>5} | {'LLM/step':>9} | "
          f"{'Resp max':>8} | {'Tok/s':>7} | {'ms/tok':>7}")
    print("-" * 70)

    for d in steps:
        step = d["step"]
        llm_max = d.get("traj/llm_time_max", 0)
        steps_max = d.get("traj/steps_max", 0)
        resp_max = d.get("response_length/max", 0)

        llm_per_step = llm_max / steps_max if steps_max > 0 else 0
        tok_per_s = resp_max / llm_max if llm_max > 0 else 0
        ms_per_tok = (llm_max * 1000) / resp_max if resp_max > 0 else 0

        print(
            f"{step:4.0f} | {llm_max:7.0f}s | {steps_max:5.0f} | {llm_per_step:8.1f}s | "
            f"{resp_max:8.0f} | {tok_per_s:6.1f} | {ms_per_tok:6.1f}"
        )


def print_update_actor_metrics(steps: list[dict]):
    """Show update_actor (training forward/backward) per-token timing."""
    print("\n" + "=" * 75)
    print("TRAINING PHASE PER-TOKEN METRICS")
    print("=" * 75)

    print(f"\n{'Step':>4} | {'update_actor ms/tok':>20} | {'adv ms/tok':>12}")
    print("-" * 45)
    for d in steps:
        step = d["step"]
        ua = d.get("timing_per_token_ms/update_actor", 0)
        adv = d.get("timing_per_token_ms/adv", 0)
        print(f"{step:4.0f} | {ua:19.4f} | {adv:11.4f}")


def print_summary(steps: list[dict]):
    n = len(steps)
    avg = lambda k: sum(d.get(k, 0) for d in steps) / n

    print("\n" + "=" * 75)
    print("INFERENCE SUMMARY")
    print("=" * 75)

    llm_mean = avg("traj/llm_time_mean")
    resp_mean = avg("response_length/mean")
    steps_mean = avg("traj/steps_mean")
    prompt_mean = avg("prompt_length/mean")

    print(f"""
  Model:              Qwen3-32B (BF16)
  vLLM config:        TP=8, gpu_mem_util=0.6, CUDA graph, 2 replicas
  Concurrent trajs:   64

  Per-trajectory (average):
    Agent steps:      {steps_mean:.1f}
    Prompt tokens:    {prompt_mean:.0f}
    Response tokens:  {resp_mean:.0f}
    LLM time:         {llm_mean:.0f}s ({llm_mean/60:.1f} min)
    LLM time/step:    {llm_mean/steps_mean:.1f}s
    Tok/s (decode):   {resp_mean/llm_mean:.1f}
    ms/tok (e2e):     {llm_mean*1000/resp_mean:.1f}

  Per-trajectory (tail):
    Agent steps:      {avg('traj/steps_max'):.0f}
    Response tokens:  {avg('response_length/max'):.0f}
    LLM time:         {avg('traj/llm_time_max'):.0f}s ({avg('traj/llm_time_max')/60:.1f} min)
    LLM time/step:    {avg('traj/llm_time_max')/avg('traj/steps_max'):.1f}s
    Tok/s (decode):   {avg('response_length/max')/avg('traj/llm_time_max'):.1f}
    ms/tok (e2e):     {avg('traj/llm_time_max')*1000/avg('response_length/max'):.1f}

  Batch throughput:
    Total tokens/step:  {resp_mean*64:.0f}
    Wall-clock/step:    {avg('timing_s/collect_trajectory'):.0f}s
    Effective tok/s:    {resp_mean*64/avg('timing_s/collect_trajectory'):.0f}
""")


def main():
    metrics_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/k8s_training_metrics.txt"

    steps = parse_metrics(metrics_file)
    if not steps:
        print(f"No metrics in {metrics_file}")
        sys.exit(1)

    compute_inference_metrics(steps)
    print_tail_inference(steps)
    print_update_actor_metrics(steps)
    print_summary(steps)


if __name__ == "__main__":
    main()
