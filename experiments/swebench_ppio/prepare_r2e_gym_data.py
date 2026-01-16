#!/usr/bin/env python3
"""
Prepare R2E-Gym data in verl format for DeepSWE reproduction training.
"""

import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd


def prepare_dataset(dataset_name: str, output_name: str, output_dir: Path):
    """Prepare a single dataset in verl format."""
    print(f"\nLoading {dataset_name} from HuggingFace...")

    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        return None

    # Get the appropriate split
    if "train" in dataset:
        data = dataset["train"]
    elif "test" in dataset:
        data = dataset["test"]
    else:
        # Use first available split
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]

    print(f"Loaded {len(data)} instances from {dataset_name}")

    # Convert to verl format
    processed_data = []
    for i, entry in enumerate(data):
        # Handle FAIL_TO_PASS and PASS_TO_PASS which may be lists or strings
        fail_to_pass = entry.get("FAIL_TO_PASS", "")
        pass_to_pass = entry.get("PASS_TO_PASS", "")

        # Convert to string if list
        if isinstance(fail_to_pass, list):
            fail_to_pass = str(fail_to_pass)
        if isinstance(pass_to_pass, list):
            pass_to_pass = str(pass_to_pass)

        # R2E-Gym uses different field names than SWE-Bench
        repo = entry.get("repo", "") or entry.get("repo_name", "")
        base_commit = entry.get("base_commit", "") or entry.get("commit_hash", "HEAD")
        instance_id = entry.get("instance_id", "") or f"{repo.replace('/', '__')}-{base_commit[:8]}" if repo else f"{output_name}_{i}"

        processed_entry = {
            "prompt": [{"role": "user", "content": "placeholder"}],
            "reward_model": {
                "style": "rule",
                "ground_truth": None,
            },
            "extra_info": {
                "index": i,
                "repo": repo,
                "instance_id": instance_id,
                "base_commit": base_commit,
                "patch": entry.get("patch", ""),
                "test_patch": entry.get("test_patch", ""),
                "problem_statement": entry.get("problem_statement", ""),
                "hints_text": entry.get("hints_text", ""),
                "version": entry.get("version", ""),
                "FAIL_TO_PASS": fail_to_pass,
                "PASS_TO_PASS": pass_to_pass,
                "environment_setup_commit": entry.get("environment_setup_commit", ""),
                # R2E-Gym specific fields
                "docker_image": entry.get("docker_image", ""),
                "expected_output_json": entry.get("expected_output_json", ""),
                "modified_files": entry.get("modified_files", []),
            },
        }
        processed_data.append(processed_entry)

    # Save as parquet
    df = pd.DataFrame(processed_data)
    output_path = output_dir / f"{output_name}.parquet"
    df.to_parquet(str(output_path), index=False)
    print(f"Saved {len(df)} entries to {output_path}")

    return output_path


def main():
    # Output directory
    output_dir = Path(__file__).parent.parent.parent / "data" / "swe"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Datasets to prepare
    datasets = [
        ("R2E-Gym/R2E-Gym-Subset", "R2E_Gym_Subset"),  # Training: 4,500 problems
        ("R2E-Gym/SWE-Bench-Verified", "SWE_Bench_Verified"),  # Validation: 500 problems
    ]

    results = {}
    for dataset_name, output_name in datasets:
        output_path = prepare_dataset(dataset_name, output_name, output_dir)
        if output_path:
            results[output_name] = output_path

    # Summary
    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print("=" * 50)

    for name, path in results.items():
        df = pd.read_parquet(str(path))
        print(f"\n{name}:")
        print(f"  Path: {path}")
        print(f"  Records: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        # Show sample
        sample = df.iloc[0]
        extra = sample['extra_info']
        print(f"  Sample instance_id: {extra.get('instance_id', 'N/A')}")
        print(f"  Sample repo: {extra.get('repo', 'N/A')}")


if __name__ == "__main__":
    main()
