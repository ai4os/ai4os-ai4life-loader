import json
from collections import defaultdict
import ai4life as aimodel
import os

ARCHITECTURE_TAGS = [
    "segment-anything",
    "mobile-sam",
    "cellpose",
    "careamics",
    "n2v2",
    "noise2void",
    "unext-v1",
    "unext-v2",
    "unetr",
    "attention-unet",
    "multiresunet",
    "resunet-se",
    "resunet++",
    "resunet",
    "seunet",
    "unet",
    "empanada",
    "hylfm",
]


def _get_architecture(tags):
    lowered = {t.lower() for t in tags}
    for arch in ARCHITECTURE_TAGS:
        if arch in lowered:
            return arch
    return "other"


def _get_download_count(model):
    count = model.get("download_count", 0)
    return int(count) if str(count).isdigit() else 0


def select_models(data):
    model_groups = defaultdict(list)

    for model_id, model in data.items():
        model["model_id"] = model_id
        tags = model.get("tags", [])
        arch = _get_architecture(tags)
        model_groups[arch].append(model)

    selected_models = []
    for arch in sorted(model_groups):
        best = max(model_groups[arch], key=_get_download_count)
        selected_models.append(best["model_id"])

    return selected_models


def main():
    path = os.path.join(
        aimodel.config.MODELS_PATH, "filtered_models.json"
    )
    with open(path, "r") as file:
        models_data = json.load(file)

    selected = select_models(models_data)
    return selected


if __name__ == "__main__":
    selected = main()
