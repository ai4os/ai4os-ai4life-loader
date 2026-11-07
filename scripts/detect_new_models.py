"""Detect newly added models by comparing filtered_models.json versions.

Usage:
    python scripts/detect_new_models.py

Reads the current filtered_models.json and the previous version from git
history, then writes the list of newly added model keys to
scripts/new_models.txt (one per line). If no new models, the file is empty.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


FILTERED_JSON = "models/filtered_models.json"
OUTPUT_FILE = "scripts/new_models.txt"


def get_git_previous_version(filepath):
    """Get the previous version of a file from git history.

    If the BASE_REF environment variable is set (e.g. "main"), the file is
    compared with that branch instead of with its previous commit.
    """
    base_ref = os.environ.get("BASE_REF", "").strip()
    if base_ref:
        try:
            subprocess.run(
                ["git", "fetch", "origin", base_ref],
                capture_output=True, text=True, check=True,
            )
            result = subprocess.run(
                ["git", "show", f"origin/{base_ref}:{filepath}"],
                capture_output=True, text=True, check=True,
            )
            print(f"Comparing {filepath} with origin/{base_ref}")
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"ERROR: could not read {filepath} from {base_ref}: {e}")
            sys.exit(1)

    try:
        result = subprocess.run(
            [
                "git", "log", "--skip=1", "-1", "--format=%H",
                "--", filepath,
            ],
            capture_output=True, text=True, check=True,
        )
        prev_commit = result.stdout.strip()
        if not prev_commit:
            return None

        result = subprocess.run(
            ["git", "show", f"{prev_commit}:{filepath}"],
            capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def main():
    current_path = Path(FILTERED_JSON)
    if not current_path.exists():
        print(f"ERROR: {FILTERED_JSON} not found")
        sys.exit(1)

    with open(current_path) as f:
        current_models = json.load(f)

    previous_models = get_git_previous_version(FILTERED_JSON)
    if previous_models is None:
        print("No previous version found, treating all models as new")
        new_models = list(current_models.keys())
    else:
        old_keys = set(previous_models.keys())
        new_keys = set(current_models.keys())
        new_models = sorted(new_keys - old_keys)

    with open(OUTPUT_FILE, "w") as f:
        for model in new_models:
            f.write(model + "\n")

    print(f"Found {len(new_models)} new model(s):")
    for model in new_models:
        print(f"  + {model}")

    if not new_models:
        print("No new models to test.")


if __name__ == "__main__":
    main()
