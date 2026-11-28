"""Filter the bioimage.io collection for models supported on AI4EOSC.

Reads the bioimage.io collection index (``index.json``), loads the latest
version of every model and keeps the models that have a native v0.5
description and PyTorch weights (``torchscript`` or ``pytorch_state_dict``).

The result is written to ``filtered_models.json``, keyed by
``"<nickname> <emoji>"`` (e.g. ``"affable-shark 🦈"``). Each entry stores
the ``source`` URL of the model description, so the rest of the code loads
models from there and never needs the raw index.

Every model that is left out is written, with the reason, to
``failed_models.json`` next to it (see ``ai4life/failure_report.py``).

The index is published by bioimage.io at ``INDEX_URL``.
"""

import json
import urllib.request
from urllib.parse import urljoin

from bioimageio.core import load_description
from bioimageio.spec.model import v0_5

try:
    from ai4life import failure_report
except ImportError:  # run as a script without the package installed
    import failure_report

INDEX_URL = (
    "https://raw.githubusercontent.com/bioimage-io/collection/"
    "refs/heads/gh-pages/index.json"
)

# Weight formats that can run on the AI4EOSC platform, by runtime:
# - "torch" and "onnx" run in the loader's main environment;
# - "tf" runs in the separate TensorFlow environment of the image
#   (requirements-tf.txt), see ai4life/tf_runner.py.
# A model gets the first runtime, in this order, that its weights allow.
RUNTIME_WEIGHT_FORMATS = {
    "torch": ("torchscript", "pytorch_state_dict"),
    "onnx": ("onnx",),
    "tf": ("tensorflow_saved_model_bundle", "keras_hdf5"),
}
SUPPORTED_WEIGHT_FORMATS = tuple(
    wf for formats in RUNTIME_WEIGHT_FORMATS.values() for wf in formats
)

# BioImage.IO tests every model in BioEngine's standard environment, whose
# TensorFlow, numpy and bioimageio.core versions requirements-tf.txt copies.
# A TensorFlow model whose report shows a failed inference check there is
# left out. Without a report (or in an unknown format) the model is kept:
# the test workflow tests it in our own TensorFlow environment.
TEST_REPORT_URL = (
    "https://hypha.aicell.io/bioimage-io/artifacts/test-report-{id}"
    "/files/published/test_report.json"
)

# Models to skip, by nickname (e.g. {"stupendous-sheep"}).
EXCLUDED_IDS = set()


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle the case if the object type is unknown or non-serializable
        return str(obj)


def _latest_version(item):
    """Return the most recently created version of an index item."""
    versions = item.get("versions") or []
    if not versions:
        return None
    return max(versions, key=lambda v: v.get("created_at") or "")


def _load(source, perform_io_checks):
    """Load a model description from its source URL.

    If loading the URL with its ``?version=`` query fails, retry without
    the query (which loads the latest version of the same model).
    Returns ``(model, source_that_worked)``.
    """
    try:
        model = load_description(source, perform_io_checks=perform_io_checks)
        return model, source
    except Exception:
        if "?" not in source:
            raise
        plain = source.split("?", 1)[0]
        model = load_description(plain, perform_io_checks=perform_io_checks)
        return model, plain


def _weight_formats(model):
    """Return the names of all weight formats present in a model."""
    return [name for name, weights in model.weights if weights is not None]


def _runtime(model):
    """Return the runtime that can run ``model``, or None."""
    formats = set(_weight_formats(model))
    for runtime, runtime_formats in RUNTIME_WEIGHT_FORMATS.items():
        if formats.intersection(runtime_formats):
            return runtime
    return None


def _bioimageio_inference_status(nickname, timeout=30):
    """Return the inference-check status of a model's BioImage.IO test
    report (e.g. ``"passed"``), or None if it cannot be read."""
    url = TEST_REPORT_URL.format(id=nickname)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            report = json.load(response)
    except Exception as e:
        print(f"No BioImage.IO test report for {nickname}: {e}")
        return None
    check = report.get("inference_check") if isinstance(report, dict) else None
    status = check.get("status") if isinstance(check, dict) else None
    return status if isinstance(status, str) else None


def _cover_url(cover, source):
    """Return a cover image as a URL (relative paths resolved to source)."""
    path = str(getattr(cover, "source", cover))
    if path.startswith(("http://", "https://")):
        return path
    return urljoin(source, path)


def _make_entry(item, version, model, nickname, emoji, source):
    """Build the filtered_models.json entry for a supported model."""
    return {
        "id": nickname,
        "nickname": nickname,
        "nickname_icon": emoji,
        "type": "model",
        "name": model.name,
        "description": model.description,
        "format_version": model.format_version,
        "collection_id": item["id"],
        "version": version.get("version"),
        "created": version.get("created_at"),
        "sha256": version.get("sha256"),
        "source": source,
        "rdf_source": source,
        "tags": list(model.tags),
        "weight_formats": _weight_formats(model),
        "runtime": _runtime(model),
        "authors": [{"name": author.name} for author in model.authors],
        "covers": [_cover_url(cover, source) for cover in model.covers],
    }


def _left_out(report, nickname, status, reason, version=None, source=None):
    """Record in ``report`` why a model is left out, and print it."""
    report[nickname] = failure_report.make_entry(
        status,
        failure_report.FILTER,
        reason,
        version=(version or {}).get("version"),
        source=source or (version or {}).get("source"),
    )
    print(f"Skipping {nickname} ({status}): {reason}")


def _merge_reports(filter_report, previous_report, supported, in_index):
    """Return the report to save.

    The filter's own entries replace the previous ones. A previous
    prediction-test failure is kept while the model is still offered by
    the filter and still in the index: the test workflow removes it when
    the model passes again.
    """
    report = dict(filter_report)
    for nickname, entry in previous_report.items():
        if (
            entry.get("stage") == failure_report.PREDICTION_TEST
            and nickname in supported
            and nickname in in_index
            and nickname not in report
        ):
            report[nickname] = entry
    return report


def filter_and_load_models(
    input_json="index.json",
    output_json="filtered_models.json",
    perform_io_checks=False,
    report_json=None,
):
    """Write all supported models from ``input_json`` to ``output_json``.

    The models that are left out, and why, are written to ``report_json``
    (by default ``failed_models.json`` next to ``output_json``).
    """
    with open(input_json, "r") as file:
        data = json.load(file)

    models = [
        item for item in data.get("items", []) if item.get("type") == "model"
    ]

    models_v0_5 = {}
    failed_models = []
    filter_report = {}

    for item in models:
        nickname = item["id"].split("/")[-1]
        if nickname in EXCLUDED_IDS:
            _left_out(
                filter_report,
                nickname,
                failure_report.NOT_SUPPORTED,
                "excluded by the AI4EOSC maintainers (EXCLUDED_IDS)",
            )
            continue

        version = _latest_version(item)
        if not version or not version.get("source"):
            _left_out(
                filter_report,
                nickname,
                failure_report.FAILED,
                "no version with a source URL in the bioimage.io index",
                version,
            )
            continue

        try:
            model, source = _load(version["source"], perform_io_checks)
        except Exception as e:
            failed_models.append((nickname, str(e)))
            _left_out(
                filter_report,
                nickname,
                failure_report.FAILED,
                f"model description could not be loaded: {e}",
                version,
            )
            continue

        if not isinstance(model, v0_5.ModelDescr):
            if type(model).__name__ == "InvalidDescr":
                msg = "invalid model description"
                failed_models.append((nickname, msg))
                _left_out(
                    filter_report,
                    nickname,
                    failure_report.FAILED,
                    msg,
                    version,
                    source,
                )
            else:
                _left_out(
                    filter_report,
                    nickname,
                    failure_report.NOT_SUPPORTED,
                    f"format version {getattr(model, 'format_version', '?')}"
                    " (only native v0.5 descriptions are supported)",
                    version,
                    source,
                )
            continue

        runtime = _runtime(model)
        if runtime is None:
            formats = ", ".join(_weight_formats(model)) or "none"
            _left_out(
                filter_report,
                nickname,
                failure_report.NOT_SUPPORTED,
                f"weight formats: {formats} "
                f"(supported: {', '.join(SUPPORTED_WEIGHT_FORMATS)})",
                version,
                source,
            )
            continue

        if runtime == "tf":
            status = _bioimageio_inference_status(nickname)
            if status is not None and status != "passed":
                _left_out(
                    filter_report,
                    nickname,
                    failure_report.NOT_SUPPORTED,
                    "TensorFlow model whose BioImage.IO inference check "
                    f"is '{status}' (see {TEST_REPORT_URL.format(id=nickname)})",
                    version,
                    source,
                )
                continue

        emoji = getattr(model, "id_emoji", None) or ""
        key = f"{nickname} {emoji}"
        models_v0_5[key] = _make_entry(
            item, version, model, nickname, emoji, source
        )
        print(
            f"The model named {key} from AI4Life is supported"
            " on the AI4EOSC platform."
        )

    if failed_models:
        print(f"\n{len(failed_models)} model(s) failed to load:")
        for model_id, error in failed_models:
            print(f"  - {model_id}: {error}")

    if not models_v0_5:
        # Never replace a good list with an empty one (e.g. after a
        # dependency or network problem).
        raise SystemExit(
            f"No models passed the filter, not overwriting {output_json}."
        )

    print(f"\n{len(models_v0_5)} supported model(s) written to {output_json}")
    with open(output_json, "w") as names_file:
        json.dump(models_v0_5, names_file, indent=4, cls=CustomEncoder)

    report_json = report_json or failure_report.report_path(output_json)
    report = _merge_reports(
        filter_report,
        failure_report.load_report(report_json),
        supported={entry["id"] for entry in models_v0_5.values()},
        in_index={item["id"].split("/")[-1] for item in models},
    )
    failure_report.save_report(report_json, report)
    print(f"{len(report)} model(s) left out, with the reason, in {report_json}")
    return models_v0_5


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter models from the bioimage.io collection index."
    )
    parser.add_argument(
        "--input",
        required=True,
        help=f"Path to the downloaded index.json ({INDEX_URL})",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the filtered_models.json",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to save the models left out and why "
        "(default: failed_models.json next to --output)",
    )
    args = parser.parse_args()

    filter_and_load_models(args.input, args.output, report_json=args.report)
