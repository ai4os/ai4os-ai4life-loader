"""Generate email body from test results.

Usage:
    python scripts/generate_email.py scripts/test_results.json

Reads test_results.json and writes a formatted email body to
scripts/email_body.txt. Only generates content if there are failures.
"""

import json
import os
import sys
from datetime import datetime


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_email.py <test_results.json>")
        sys.exit(1)

    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found")
        sys.exit(1)

    with open(results_file) as f:
        results = json.load(f)

    passed = results.get("passed", [])
    failed = results.get("failed", [])

    if not failed:
        print("No failures to report.")
        sys.exit(0)

    lines = []
    lines.append("Subject: [AI4Life] New models failed prediction test")
    lines.append("")
    lines.append(
        f"The following new models were tested on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}:"
    )
    lines.append("")
    lines.append(f"Total tested: {len(passed) + len(failed)}")
    lines.append(f"Passed: {len(passed)}")
    lines.append(f"Failed: {len(failed)}")
    lines.append("")

    if passed:
        lines.append("PASSED:")
        for model in passed:
            lines.append(f"  + {model}")
        lines.append("")

    lines.append("FAILED (removed from filtered_models.json):")
    for item in failed:
        lines.append(f"  - {item['model']}")
        lines.append(f"    Error: {item['error']}")
    lines.append("")

    lines.append("The failed models have been removed from filtered_models.json.")
    lines.append("Repository: https://github.com/ai4os/ai4os-ai4life-loader")
    lines.append("Workflow: Test New Models")

    body = "\n".join(lines)

    with open("scripts/email_body.txt", "w") as f:
        f.write(body)

    print(body)


if __name__ == "__main__":
    main()
