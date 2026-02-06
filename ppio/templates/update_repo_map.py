#!/usr/bin/env python3
"""
Update REPO_TEMPLATE_MAP in ppio_reward.py after building R2E-Gym templates.

Usage:
    python update_repo_map.py [--dry-run]

This script:
1. Lists templates from PPIO CLI
2. Matches template names to R2E-Gym repos
3. Updates REPO_TEMPLATE_MAP in ppio_reward.py
"""

import argparse
import os
import re
import subprocess
import sys

# R2E-Gym short name to full GitHub repo mapping
R2E_GYM_REPOS = {
    "pandas": "pandas",
    "numpy": "numpy",
    "pillow": "pillow",
    "orange3": "orange3",
    "aiohttp": "aiohttp",
    "tornado": "tornado",
    "scrapy": "scrapy",
    "pyramid": "pyramid",
    "datalad": "datalad",
    "coveragepy": "coveragepy",
}

PPIO_REWARD_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "rllm", "environments", "swe_ppio", "ppio_reward.py"
)


def get_ppio_templates():
    """List templates from PPIO CLI and parse output."""
    try:
        result = subprocess.run(
            ["ppio-sandbox-cli", "tpl", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running ppio-sandbox-cli: {e}")
        return {}

    # Parse table output
    templates = {}
    for line in output.split("\n"):
        # Look for lines with template data (contains │)
        if "│" in line:
            parts = [p.strip() for p in line.split("│")]
            if len(parts) >= 3:
                template_id = parts[1]
                template_name = parts[2]
                if template_id and not template_id.startswith("Template"):
                    templates[template_name] = template_id
    return templates


def match_templates(ppio_templates):
    """Match PPIO templates to R2E-Gym repos."""
    matches = {}
    for short_name, repo_key in R2E_GYM_REPOS.items():
        # Look for r2e-{name} pattern
        for tpl_name, tpl_id in ppio_templates.items():
            if f"r2e-{short_name}" in tpl_name.lower():
                matches[repo_key] = tpl_id
                break
    return matches


def update_ppio_reward(matches, dry_run=False):
    """Update REPO_TEMPLATE_MAP in ppio_reward.py."""
    ppio_reward_path = os.path.abspath(PPIO_REWARD_PATH)

    if not os.path.exists(ppio_reward_path):
        print(f"Error: {ppio_reward_path} not found")
        return False

    with open(ppio_reward_path, "r") as f:
        content = f.read()

    # Find REPO_TEMPLATE_MAP section
    pattern = r"(REPO_TEMPLATE_MAP\s*=\s*\{[^}]+\})"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("Error: Could not find REPO_TEMPLATE_MAP in ppio_reward.py")
        return False

    old_map = match.group(1)

    # Build new entries
    new_entries = []
    for repo, tpl_id in sorted(matches.items()):
        new_entries.append(f'    "{repo}": "{tpl_id}",')

    if not new_entries:
        print("No new templates to add")
        return True

    # Insert new entries before closing brace
    insert_point = old_map.rfind("}")
    updated_map = old_map[:insert_point]

    # Add comment for R2E-Gym section
    if "# R2E-Gym" not in updated_map:
        updated_map += "\n    # R2E-Gym training templates\n"

    updated_map += "\n".join(new_entries) + "\n"
    updated_map += "}"

    new_content = content.replace(old_map, updated_map)

    print("\n=== Changes to REPO_TEMPLATE_MAP ===")
    print(f"Adding {len(matches)} R2E-Gym templates:")
    for repo, tpl_id in sorted(matches.items()):
        print(f"  {repo}: {tpl_id}")

    if dry_run:
        print("\n[DRY RUN] Would update:", ppio_reward_path)
        return True

    with open(ppio_reward_path, "w") as f:
        f.write(new_content)

    print(f"\nUpdated: {ppio_reward_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Update REPO_TEMPLATE_MAP with R2E-Gym templates")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    args = parser.parse_args()

    print("Fetching PPIO templates...")
    templates = get_ppio_templates()

    if not templates:
        print("No templates found. Build templates first with:")
        print("  ./build_r2e_templates.sh all")
        return 1

    print(f"Found {len(templates)} templates")

    matches = match_templates(templates)

    if not matches:
        print("\nNo R2E-Gym templates found. Available templates:")
        for name, tid in sorted(templates.items()):
            print(f"  {name}: {tid}")
        return 1

    print(f"\nMatched {len(matches)} R2E-Gym templates")

    if update_ppio_reward(matches, dry_run=args.dry_run):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
