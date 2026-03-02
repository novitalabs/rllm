"""
DeepSWE Evaluation Log Visualizer

Visualizes r2egym-format evaluation logs from DeepSWE runs.
Supports browsing trajectories step-by-step with thinking/action/observation views.
Uses lazy loading to avoid loading all 13GB data into memory.

Usage:
    python deepswe_eval_visualizer.py --data_dir ./deepswe --server_port 12345
"""

import json
import os
import re
from collections import defaultdict

import gradio as gr
from fire import Fire


def build_index(data_dir: str):
    """Build a lightweight index of all records without loading full trajectory data."""
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".jsonl")])
    index = []  # list of (file_path, byte_offset, instance_id, reward, run_name, n_steps, exit_reason, repo)
    run_names = []
    instance_results = defaultdict(list)

    for fname in files:
        run_name = fname.replace(".jsonl", "")
        run_names.append(run_name)
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                rec = json.loads(line)
                instance_id = rec.get("ds", {}).get("instance_id", "unknown")
                reward = rec.get("reward", 0.0)
                n_steps = len(rec.get("trajectory_steps", []))
                exit_reason = rec.get("exit_reason", "N/A")
                repo = rec.get("ds", {}).get("repo", "N/A")
                index.append((fpath, offset, instance_id, reward, run_name, n_steps, exit_reason, repo))
                instance_results[instance_id].append(reward)

    return index, run_names, instance_results


def load_record(fpath, offset):
    """Load a single record from disk by file path and byte offset."""
    with open(fpath, "rb") as f:
        f.seek(offset)
        line = f.readline()
        return json.loads(line)


def main(data_dir: str = "./deepswe", server_port: int = 12345):
    print("Building index...")
    index, run_names, instance_results = build_index(data_dir)
    print(f"Indexed {len(index)} records across {len(run_names)} runs.")

    # Precompute pass@k
    n_instances = len(instance_results)
    pass_at_k = {}
    for k in [1, 2, 4, 8, 16]:
        passed = sum(1 for results in instance_results.values() if any(r > 0 for r in results[:k]))
        pass_at_k[k] = passed / n_instances if n_instances > 0 else 0

    def filter_indices(run_filter, reward_filter):
        filtered = []
        for i, (fpath, offset, inst_id, reward, run_name, n_steps, exit_reason, repo) in enumerate(index):
            if run_filter != "All Runs" and run_name != run_filter:
                continue
            if reward_filter == "Resolved (reward > 0)" and reward <= 0:
                continue
            if reward_filter == "Unresolved (reward = 0)" and reward > 0:
                continue
            filtered.append(i)
        return filtered

    def format_thought(thought):
        if not thought:
            return "*No thinking recorded*"
        think_match = re.search(r"<think>(.*?)</think>", thought, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        return thought.strip()

    def format_action(action):
        if not action:
            return "*No action*"
        if isinstance(action, dict):
            func_name = action.get("function", {}).get("name", "unknown")
            args = action.get("function", {}).get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            return f"**Function:** `{func_name}`\n```json\n{json.dumps(args, indent=2)[:3000]}\n```"
        return f"```\n{str(action)[:5000]}\n```"

    def format_observation(obs):
        if not obs:
            return "*No observation*"
        obs_str = str(obs)
        if len(obs_str) > 8000:
            return f"```\n{obs_str[:4000]}\n...\n(truncated {len(obs_str) - 8000} chars)\n...\n{obs_str[-4000:]}\n```"
        return f"```\n{obs_str}\n```"

    def update_view(traj_idx, step_idx, run_filter, reward_filter):
        empty = "*No data*"
        filtered = filter_indices(run_filter, reward_filter)

        if not filtered or traj_idx >= len(filtered):
            return (empty,) * 10

        idx = filtered[traj_idx]
        fpath, offset, instance_id, reward, run_name, n_steps, exit_reason, repo = index[idx]

        # Lazy load the full record
        rec = load_record(fpath, offset)
        steps = rec.get("trajectory_steps", [])
        n_steps = len(steps)

        if step_idx >= n_steps:
            step_idx = n_steps - 1 if n_steps > 0 else 0

        position = f"**Trajectory {traj_idx + 1}/{len(filtered)}**  |  **Step {step_idx + 1}/{n_steps}**"

        metadata = f"**Instance:** `{instance_id}`\n"
        metadata += f"**Repo:** `{repo}`\n"
        metadata += f"**Run:** `{run_name}`\n"
        metadata += f"**Docker:** `{rec.get('docker_image', 'N/A')}`"

        reward_icon = "\u2705" if reward > 0 else "\u274c"
        perf = f"**Reward:** {reward_icon} {reward}\n"
        perf += f"**Exit Reason:** `{exit_reason}`\n"
        perf += f"**Total Steps:** {n_steps}\n"
        perf += f"**Max Steps:** {rec.get('max_steps', 'N/A')} (abs: {rec.get('max_steps_absolute', 'N/A')})\n"
        perf += f"**Token Limit:** {rec.get('max_token_limit', 'N/A')}"

        inst_rewards = instance_results.get(instance_id, [])
        n_solved = sum(1 for r in inst_rewards if r > 0)
        cross_run = f"**Instance solve rate:** {n_solved}/{len(inst_rewards)} runs ({n_solved/len(inst_rewards)*100:.0f}%)" if inst_rewards else ""

        problem = rec.get("problem_statement", "N/A")
        if len(problem) > 3000:
            problem = problem[:3000] + "\n\n... (truncated)"
        problem_text = f"**Problem:**\n{problem}"

        if n_steps == 0:
            return (position, metadata, perf, cross_run, problem_text, empty, empty, empty, empty, empty)

        step = steps[step_idx]

        thinking_text = format_thought(step.get("thought", ""))
        action_text = format_action(step.get("action", ""))
        obs_text = format_observation(step.get("observation", ""))

        step_metrics = f"**Step:** {step.get('step_idx', step_idx)}\n"
        done_icon = "\u2705" if step.get("done", False) else "\u274c"
        step_metrics += f"**Done:** {done_icon}\n"
        step_metrics += f"**Prompt tokens:** {step.get('token_usage_prompt', 'N/A')}\n"
        step_metrics += f"**Completion tokens:** {step.get('token_usage_completion', 'N/A')}\n"
        step_metrics += f"**Total tokens:** {step.get('token_usage_total', 'N/A')}\n"
        step_metrics += f"**LLM time:** {step.get('llm_exec_time', 0):.1f}s\n"
        step_metrics += f"**Env time:** {step.get('env_exec_time', 0):.1f}s\n"
        step_metrics += f"**Total step time:** {step.get('total_step_time', 0):.1f}s\n"
        step_metrics += f"**Cumulative time:** {step.get('total_time_traj', 0):.1f}s"

        patch_text = ""
        if step_idx == n_steps - 1:
            patch = rec.get("output_patch", "")
            if patch:
                patch_text = f"**Output Patch:**\n```diff\n{patch[:5000]}\n```"
            else:
                patch_text = "*No output patch*"

        return (position, metadata, perf, cross_run, problem_text, thinking_text, action_text, obs_text, step_metrics, patch_text)

    def nav(traj_idx, step_idx, direction, level, run_filter, reward_filter):
        filtered = filter_indices(run_filter, reward_filter)
        n_trajs = len(filtered)
        if n_trajs == 0:
            return 0, 0

        traj_idx = int(traj_idx)
        step_idx = int(step_idx)

        if level == "trajectory":
            if direction == "next":
                traj_idx = (traj_idx + 1) % n_trajs
            else:
                traj_idx = (traj_idx - 1) % n_trajs
            step_idx = 0
        else:
            idx = filtered[traj_idx]
            n_steps = index[idx][5]
            if n_steps == 0:
                return traj_idx, 0
            if direction == "next":
                step_idx = (step_idx + 1) % n_steps
            else:
                step_idx = (step_idx - 1) % n_steps

        return traj_idx, step_idx

    # Stats summary
    total_records = len(index)
    total_resolved = sum(1 for _, _, _, r, _, _, _, _ in index if r > 0)
    stats_md = f"**Total:** {total_records} trajectories across {len(run_names)} runs\n"
    stats_md += f"**Resolved:** {total_resolved} ({total_resolved/total_records*100:.1f}%)\n"
    stats_md += f"**Pass@1:** {pass_at_k.get(1, 0)*100:.1f}% | "
    stats_md += f"**Pass@4:** {pass_at_k.get(4, 0)*100:.1f}% | "
    stats_md += f"**Pass@8:** {pass_at_k.get(8, 0)*100:.1f}% | "
    stats_md += f"**Pass@16:** {pass_at_k.get(16, 0)*100:.1f}%"

    custom_css = """
    .nav-button { min-width: 140px !important; }
    .step-display-box textarea {
        text-align: center !important; font-weight: bold !important;
        font-size: 1.1em !important;
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="DeepSWE Eval Visualizer") as interface:
        gr.Markdown("# DeepSWE Evaluation Log Visualizer")
        gr.Markdown(stats_md)

        traj_idx_state = gr.State(0)
        step_idx_state = gr.State(0)

        with gr.Row():
            run_dropdown = gr.Dropdown(
                choices=["All Runs"] + run_names,
                value="All Runs", label="Run Filter", interactive=True,
            )
            reward_dropdown = gr.Dropdown(
                choices=["All", "Resolved (reward > 0)", "Unresolved (reward = 0)"],
                value="All", label="Reward Filter", interactive=True,
            )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    prev_traj_btn = gr.Button("Prev Trajectory", elem_classes=["nav-button"])
                    next_traj_btn = gr.Button("Next Trajectory", elem_classes=["nav-button"])
                with gr.Row():
                    prev_step_btn = gr.Button("Prev Step", elem_classes=["nav-button"])
                    next_step_btn = gr.Button("Next Step", elem_classes=["nav-button"])
            with gr.Column(scale=2):
                position_display = gr.Textbox(label="Position", interactive=False, elem_classes=["step-display-box"])

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Metadata", open=True):
                    metadata_output = gr.Markdown()
                with gr.Accordion("Performance", open=True):
                    perf_output = gr.Markdown()
                with gr.Accordion("Cross-Run Stats", open=True):
                    cross_run_output = gr.Markdown()
                with gr.Accordion("Problem Statement", open=False):
                    problem_output = gr.Markdown()

            with gr.Column(scale=2):
                with gr.Accordion("Agent Thinking", open=True):
                    thinking_output = gr.Textbox(label="Thinking", lines=8, interactive=False)
                with gr.Accordion("Action", open=True):
                    action_output = gr.Markdown()
                with gr.Accordion("Observation", open=True):
                    obs_output = gr.Markdown()
                with gr.Accordion("Step Metrics", open=True):
                    step_metrics_output = gr.Markdown()
                with gr.Accordion("Output Patch", open=False):
                    patch_output = gr.Markdown()

        all_outputs = [
            position_display, metadata_output, perf_output, cross_run_output,
            problem_output, thinking_output, action_output, obs_output,
            step_metrics_output, patch_output,
        ]

        def reset_indices():
            return 0, 0

        prev_traj_btn.click(
            fn=lambda t, s, rf, rwf: nav(t, s, "prev", "trajectory", rf, rwf),
            inputs=[traj_idx_state, step_idx_state, run_dropdown, reward_dropdown],
            outputs=[traj_idx_state, step_idx_state],
        )
        next_traj_btn.click(
            fn=lambda t, s, rf, rwf: nav(t, s, "next", "trajectory", rf, rwf),
            inputs=[traj_idx_state, step_idx_state, run_dropdown, reward_dropdown],
            outputs=[traj_idx_state, step_idx_state],
        )
        prev_step_btn.click(
            fn=lambda t, s, rf, rwf: nav(t, s, "prev", "step", rf, rwf),
            inputs=[traj_idx_state, step_idx_state, run_dropdown, reward_dropdown],
            outputs=[traj_idx_state, step_idx_state],
        )
        next_step_btn.click(
            fn=lambda t, s, rf, rwf: nav(t, s, "next", "step", rf, rwf),
            inputs=[traj_idx_state, step_idx_state, run_dropdown, reward_dropdown],
            outputs=[traj_idx_state, step_idx_state],
        )

        run_dropdown.change(fn=reset_indices, outputs=[traj_idx_state, step_idx_state])
        reward_dropdown.change(fn=reset_indices, outputs=[traj_idx_state, step_idx_state])

        for state in [traj_idx_state, step_idx_state]:
            state.change(
                fn=update_view,
                inputs=[traj_idx_state, step_idx_state, run_dropdown, reward_dropdown],
                outputs=all_outputs,
            )
        run_dropdown.change(
            fn=update_view,
            inputs=[traj_idx_state, step_idx_state, run_dropdown, reward_dropdown],
            outputs=all_outputs,
        )
        reward_dropdown.change(
            fn=update_view,
            inputs=[traj_idx_state, step_idx_state, run_dropdown, reward_dropdown],
            outputs=all_outputs,
        )

        interface.load(
            fn=update_view,
            inputs=[traj_idx_state, step_idx_state, run_dropdown, reward_dropdown],
            outputs=all_outputs,
        )

    interface.launch(server_name="0.0.0.0", server_port=server_port)


if __name__ == "__main__":
    Fire(main)
