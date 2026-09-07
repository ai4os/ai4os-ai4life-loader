"""Run pytest for new models and collect pass/fail results.

Usage:
    python scripts/test_new_models.py scripts/new_models.txt

Runs the existing tests/test_predictions/ suite with -k to select
only the new models, then parses the pytest output to identify
which models passed and which failed.

Results are written to scripts/test_results.json:
    {
        "passed": ["model1", "model2"],
        "failed": [{"model": "model3", "error": "..."}]
    }

Exit code 0 if all passed, 1 if any failed.
"""

import json
import os
import re
import subprocess
import sys


RESULTS_FILE = "scripts/test_results.json"


def build_k_filter(model_keys):
    """Build a pytest -k expression from model keys.

    Uses just the nickname part (before the emoji) since that's
    unique enough for -k substring matching.
    """
    nicknames = []
    for key in model_keys:
        nickname = key.split(" ")[0] if " " in key else key
        nicknames.append(nickname)
    return " or ".join(nicknames)


def parse_pytest_output(output):
    """Parse pytest -v output to extract pass/fail per model."""
    passed = []
    failed = []

    # Lines look like:
    # ...::test_predictions_type[affable-shark \U0001f988-application/json] PASSED
    # ...::test_predictions_type[affable-shark \U0001f988-application/json] ERROR
    # ...::test_predictions_type[affable-shark \U0001f988-application/json] FAILED
    pattern = re.compile(
        r"test_predictions_type\[(.+?)\s+\\U[0-9a-f]+\S*-application/json\]\s+(PASSED|FAILED|ERROR)"
    )

    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            model_name = match.group(1)
            status = match.group(2)
            if status == "PASSED":
                passed.append(model_name)
            else:
                failed.append({"model": model_name, "error": status})

    return passed, failed


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_new_models.py <new_models.txt>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found")
        sys.exit(1)

    with open(input_file) as f:
        model_keys = [line.strip() for line in f if line.strip()]

    if not model_keys:
        print("No new models to test.")
        with open(RESULTS_FILE, "w") as f:
            json.dump({"passed": [], "failed": []}, f, indent=2)
        sys.exit(0)

    k_filter = build_k_filter(model_keys)
    print(f"Running pytest with -k \"{k_filter}\"")

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/test_predictions/",
            "-v",
            f"-k", k_filter,
            "--tb=short",
            "-W", "ignore",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    passed, failed = parse_pytest_output(result.stdout)

    # For failed models, try to extract the error from the output
    if failed:
        # Split output into sections per failed test
        error_pattern = re.compile(
            r"_( ERROR at setup of | FAILED )test_predictions_type\[(.+?)\s+\\U[0-9a-f]+\S*-application/json\]_"
        )
        # Build a simple error map from the short traceback
        for item in failed:
            model = item["model"]
            # Look for the error line near the model name
            for line in result.stdout.splitlines():
                if model in line and ("Error" in line or "error" in line):
                    item["error"] = line.strip()[:200]
                    break

    results = {"passed": passed, "failed": failed}
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    for item in failed:
        print(f"  FAILED: {item['model']}: {item['error']}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
