"""Generate GitHub issue body from test results.

Usage:
    python scripts/generate_issue.py scripts/test_results.json

Reads test_results.json and writes a formatted GitHub issue body to
scripts/issue_body.md. Only generates content if there are failures.
"""

import json
import os
import sys
from datetime import datetime


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_issue.py <test_results.json>")
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
    lines.append(
        f"The **Test New Models** workflow found {len(failed)} model(s) that failed prediction testing"
        f" on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}."
    )
    lines.append("")
    lines.append(f"- Total tested: {len(passed) + len(failed)}")
    lines.append(f"- Passed: {len(passed)}")
    lines.append(f"- Failed: {len(failed)}")
    lines.append("")
    lines.append("The failed models have been **removed** from `filtered_models.json`.")
    lines.append("")

    if passed:
        lines.append("## Passed")
        lines.append("")
        for model in passed:
            lines.append(f"- {model}")
        lines.append("")

    lines.append("## Failed")
    lines.append("")
    lines.append("| Model | Error |")
    lines.append("|-------|-------|")
    for item in failed:
        model = item["model"]
        error = item["error"].replace("|", "\\|").replace("\n", " ")[:200]
        lines.append(f"| `{model}` | {error} |")
    lines.append("")
    lines.append("---")
    lines.append("Workflow: [Test New Models](https://github.com/ai4os/ai4os-ai4life-loader/actions)")

    body = "\n".join(lines)

    with open("scripts/issue_body.md", "w") as f:
        f.write(body)

    print(body)


if __name__ == "__main__":
    main()
