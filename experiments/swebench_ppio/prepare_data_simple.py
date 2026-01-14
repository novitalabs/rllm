#!/usr/bin/env python3
"""
Simple data preparation script for SWE-bench Lite.
Downloads from HuggingFace and saves as parquet.
"""

import os
from pathlib import Path

from datasets import load_dataset

def main():
    # Output directory
    output_dir = Path(__file__).parent.parent.parent / "data" / "swe"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SWE-bench Lite from HuggingFace...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    print(f"Loaded {len(dataset)} instances")

    # Save as parquet
    output_path = output_dir / "SWE_Bench_Lite.parquet"
    dataset.to_parquet(str(output_path))
    print(f"Saved to {output_path}")

    # Show sample
    print("\nSample instance:")
    sample = dataset[0]
    print(f"  instance_id: {sample['instance_id']}")
    print(f"  repo: {sample['repo']}")
    print(f"  FAIL_TO_PASS: {len(sample.get('FAIL_TO_PASS', '').split()) if sample.get('FAIL_TO_PASS') else 0} tests")

    print("\nData preparation complete!")

if __name__ == "__main__":
    main()
