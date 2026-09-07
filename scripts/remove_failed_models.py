"""Remove failed models from filtered_models.json.

Usage:
    python scripts/remove_failed_models.py scripts/test_results.json

Reads test_results.json, removes any failed model entries from
models/filtered_models.json, and writes the cleaned file back.
"""

import json
import os
import sys
from pathlib import Path

import ai4life as aimodel

FILTERED_JSON = os.path.join(aimodel.config.MODELS_PATH, "filtered_models.json")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/remove_failed_models.py <test_results.json>")
        sys.exit(1)

    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found")
        sys.exit(1)

    with open(results_file) as f:
        results = json.load(f)

    failed = results.get("failed", [])
    if not failed:
        print("No failed models to remove.")
        sys.exit(0)

    with open(FILTERED_JSON) as f:
        models = json.load(f)

    removed = []
    for item in failed:
        model_name = item["model"]
        # The test results store nickname only; try matching with icon too
        for key in list(models.keys()):
            if key.split(" ")[0] == model_name or key == model_name:
                del models[key]
                removed.append(key)
                break

    with open(FILTERED_JSON, "w") as f:
        json.dump(models, f, indent=4)

    print(f"Removed {len(removed)} failed model(s) from filtered_models.json:")
    for key in removed:
        print(f"  - {key}")


if __name__ == "__main__":
    main()
