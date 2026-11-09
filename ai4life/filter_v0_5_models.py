"""Filter the bioimage.io collection for models supported on AI4EOSC.

Reads the bioimage.io collection index (``index.json``), loads the latest
version of every model and keeps the models that have a native v0.5
description and PyTorch weights (``torchscript`` or ``pytorch_state_dict``).

The result is written to ``filtered_models.json``, keyed by
``"<nickname> <emoji>"`` (e.g. ``"affable-shark 🦈"``). Each entry stores
the ``source`` URL of the model description, so the rest of the code loads
models from there and never needs the raw index.

The index is published by bioimage.io at ``INDEX_URL``.
"""

import json
from urllib.parse import urljoin

from bioimageio.core import load_description
from bioimageio.spec.model import v0_5

INDEX_URL = (
    "https://raw.githubusercontent.com/bioimage-io/collection/"
    "refs/heads/gh-pages/index.json"
)

# Weight formats that can run on the AI4EOSC platform (for now).
SUPPORTED_WEIGHT_FORMATS = ("torchscript", "pytorch_state_dict")

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
        "authors": [{"name": author.name} for author in model.authors],
        "covers": [_cover_url(cover, source) for cover in model.covers],
    }


def filter_and_load_models(
    input_json="index.json",
    output_json="filtered_models.json",
    perform_io_checks=False,
):
    """Write all supported models from ``input_json`` to ``output_json``."""
    with open(input_json, "r") as file:
        data = json.load(file)

    models = [
        item for item in data.get("items", []) if item.get("type") == "model"
    ]

    models_v0_5 = {}
    failed_models = []

    for item in models:
        nickname = item["id"].split("/")[-1]
        if nickname in EXCLUDED_IDS:
            print(f"Skipping {nickname}: excluded")
            continue

        version = _latest_version(item)
        if not version or not version.get("source"):
            print(f"Skipping {nickname}: no version with a source URL")
            continue

        try:
            model, source = _load(version["source"], perform_io_checks)
        except Exception as e:
            failed_models.append((nickname, str(e)))
            print(f"Failed to load model '{nickname}': {e}")
            continue

        if not isinstance(model, v0_5.ModelDescr):
            if type(model).__name__ == "InvalidDescr":
                msg = "invalid model description"
                failed_models.append((nickname, msg))
                print(f"Failed to load model '{nickname}': {msg}")
            else:
                print(
                    f"Skipping {nickname}: format_version "
                    f"{getattr(model, 'format_version', '?')} "
                    "is not native v0.5"
                )
            continue

        if not any(
            getattr(model.weights, wf, None) is not None
            for wf in SUPPORTED_WEIGHT_FORMATS
        ):
            print(f"Skipping {nickname}: no torchscript/pytorch weights")
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
    args = parser.parse_args()

    filter_and_load_models(args.input, args.output)
