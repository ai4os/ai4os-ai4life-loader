"""Report of the bioimage.io models that are not offered on AI4EOSC, and why.

``models/failed_models.json`` sits next to ``filtered_models.json`` and is
keyed by model nickname (e.g. ``"famous-fish"``). Every entry has:

- ``status``: ``"not supported"`` (left out by design, e.g. TensorFlow
  weights or a v0.4 description) or ``"failed"`` (should work, but did
  not: the description could not be loaded, or the prediction test failed);
- ``stage``: ``"filter"`` or ``"prediction test"``;
- ``reason``: why, in one line or the error message;
- ``checked``: the date (UTC) of the check;
- ``loader_version``: the version of this loader (``VERSION`` file);
- optionally ``issue`` (the GitHub issue of a failed prediction test),
  ``version`` and ``source`` (of the model on bioimage.io).

The filter (``ai4life/filter_v0_5_models.py``) rewrites the ``filter``
entries on every run and keeps the ``prediction test`` entries, which the
test workflow (``scripts/update_failed_models.py``) adds and removes.

This module has no heavy imports, so the scripts can use it on its own.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPORT_NAME = "failed_models.json"

NOT_SUPPORTED = "not supported"
FAILED = "failed"

FILTER = "filter"
PREDICTION_TEST = "prediction test"

# Error messages can be long; keep the report readable.
MAX_REASON_LENGTH = 500

_VERSION_FILE = Path(__file__).resolve().parents[1] / "VERSION"


def report_path(filtered_json):
    """Return the path of the report next to ``filtered_json``."""
    folder = os.path.dirname(os.path.abspath(filtered_json))
    return os.path.join(folder, REPORT_NAME)


def loader_version():
    """Return the version of this loader, or None if it is unknown."""
    try:
        return _VERSION_FILE.read_text().strip() or None
    except OSError:
        return None


def today():
    """Return today's date (UTC) as YYYY-MM-DD."""
    return datetime.now(timezone.utc).date().isoformat()


def make_entry(status, stage, reason, **extra):
    """Build one report entry. Extra fields that are None are left out."""
    reason = " ".join(str(reason).split())  # one line
    if len(reason) > MAX_REASON_LENGTH:
        reason = reason[: MAX_REASON_LENGTH - 3] + "..."
    entry = {
        "status": status,
        "stage": stage,
        "reason": reason,
        "checked": today(),
        "loader_version": loader_version(),
    }
    entry.update(extra)
    return {key: value for key, value in entry.items() if value is not None}


def load_report(path):
    """Read a report; a missing or broken file gives an empty report."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, ValueError):
        return {}
    return report if isinstance(report, dict) else {}


def save_report(path, report):
    """Write a report, sorted by model nickname."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(dict(sorted(report.items())), file, indent=2, ensure_ascii=False)
        file.write("\n")
