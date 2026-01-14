#!/usr/bin/env python3
"""
Prepare SWE-bench data in verl format for rllm training.
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd

def main():
    # Output directory
    output_dir = Path(__file__).parent.parent.parent / "data" / "swe"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SWE-bench Lite from HuggingFace...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    print(f"Loaded {len(dataset)} instances")

    # Convert to verl format
    processed_data = []
    for i, entry in enumerate(dataset):
        # Create verl-compatible format
        processed_entry = {
            "prompt": [{"role": "user", "content": "placeholder"}],
            "reward_model": {
                "style": "rule",
                "ground_truth": None,
            },
            # Store original entry as extra_info (as JSON string to ensure compatibility)
            "extra_info": json.dumps({
                "repo": entry["repo"],
                "instance_id": entry["instance_id"],
                "base_commit": entry["base_commit"],
                "patch": entry["patch"],
                "test_patch": entry["test_patch"],
                "problem_statement": entry["problem_statement"],
                "hints_text": entry.get("hints_text", ""),
                "version": entry.get("version", ""),
                "FAIL_TO_PASS": entry.get("FAIL_TO_PASS", ""),
                "PASS_TO_PASS": entry.get("PASS_TO_PASS", ""),
                "environment_setup_commit": entry.get("environment_setup_commit", ""),
            }),
        }
        processed_data.append(processed_entry)

    # Save as parquet
    df = pd.DataFrame(processed_data)
    output_path = output_dir / "SWE_Bench_Lite.parquet"
    df.to_parquet(str(output_path), index=False)
    print(f"Saved {len(df)} entries to {output_path}")

    # Verify
    df_verify = pd.read_parquet(str(output_path))
    print(f"\nVerification:")
    print(f"  Columns: {list(df_verify.columns)}")
    print(f"  Rows: {len(df_verify)}")

    # Show sample
    sample = df_verify.iloc[0]
    print(f"\nSample entry:")
    print(f"  prompt: {sample['prompt']}")
    print(f"  reward_model: {sample['reward_model']}")
    extra = json.loads(sample['extra_info'])
    print(f"  extra_info.instance_id: {extra['instance_id']}")
    print(f"  extra_info.repo: {extra['repo']}")

    print("\nData preparation complete!")

if __name__ == "__main__":
    main()
