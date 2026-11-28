"""Record the prediction-test results in models/failed_models.json.

Usage:
    python scripts/update_failed_models.py scripts/test_results.json [open_issues.json]

- A model that failed its prediction test is added (or updated) with its
  error and, if given, the link to its open GitHub issue.
- A model that passed is removed from the report.

``open_issues.json`` is the output of
``gh issue list --state open --label automated --json title,url``: the
issues opened by the test workflow, titled
``[AI4Life] Prediction test failed: <model>``.

The model's version and source are read from ``models/index.json`` when
it is there. See ``ai4life/failure_report.py`` for the report format.
"""

import json
import os
import sys

from ai4life import config, failure_report

REPORT_JSON = os.path.join(config.MODELS_PATH, failure_report.REPORT_NAME)
INDEX_JSON = os.path.join(config.MODELS_PATH, "index.json")
ISSUE_TITLE = "[AI4Life] Prediction test failed: {model}"


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError):
        return default


def issue_links(issues):
    """Map model nickname -> issue URL, from ``gh issue list`` output."""
    prefix = ISSUE_TITLE.format(model="")
    links = {}
    for issue in issues or []:
        title = issue.get("title", "")
        if title.startswith(prefix) and issue.get("url"):
            links[title[len(prefix):].strip()] = issue["url"]
    return links


def latest_versions(index):
    """Map model nickname -> its latest version in the bioimage.io index."""
    versions = {}
    for item in (index or {}).get("items", []):
        if item.get("type") != "model" or not item.get("versions"):
            continue
        latest = max(item["versions"], key=lambda v: v.get("created_at") or "")
        versions[item["id"].split("/")[-1]] = latest
    return versions


def update_report(report, results, links, versions):
    """Return ``report`` updated with the test ``results``."""
    report = dict(report)
    for nickname in results.get("passed", []):
        entry = report.get(nickname)
        if entry and entry.get("stage") == failure_report.PREDICTION_TEST:
            del report[nickname]
    for item in results.get("failed", []):
        nickname = item["model"]
        version = versions.get(nickname, {})
        report[nickname] = failure_report.make_entry(
            failure_report.FAILED,
            failure_report.PREDICTION_TEST,
            item.get("error") or "prediction test failed",
            issue=links.get(nickname),
            version=version.get("version"),
            source=version.get("source"),
        )
    return report


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/update_failed_models.py "
            "<test_results.json> [open_issues.json]"
        )
        sys.exit(1)

    results = _read_json(sys.argv[1], None)
    if results is None:
        print(f"No test results in {sys.argv[1]}, report not changed.")
        sys.exit(0)

    issues = _read_json(sys.argv[2], []) if len(sys.argv) > 2 else []
    report = update_report(
        failure_report.load_report(REPORT_JSON),
        results,
        issue_links(issues),
        latest_versions(_read_json(INDEX_JSON, {})),
    )
    failure_report.save_report(REPORT_JSON, report)
    print(
        f"{REPORT_JSON}: {len(results.get('failed', []))} failed, "
        f"{len(results.get('passed', []))} passed, "
        f"{len(report)} model(s) in the report"
    )


if __name__ == "__main__":
    main()
