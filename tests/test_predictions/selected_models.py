import json
import os
from collections import defaultdict

import ai4life as aimodel

ARCHITECTURE_FAMILIES = {
    "sam": [
        "segment-anything",
        "mobile-sam",
    ],
    "cellpose": [
        "cellpose",
    ],
    "denoising": [
        "careamics",
        "n2v2",
        "noise2void",
    ],
    "unext": [
        "unext-v1",
        "unext-v2",
    ],
    "unetr": [
        "unetr",
    ],
    "unet": [
        "attention-unet",
        "multiresunet",
        "resunet-se",
        "resunet++",
        "resunet",
        "seunet",
        "unet",
    ],
    "empanada": [
        "empanada",
    ],
    "hylfm": [
        "hylfm",
    ],
}

TAG_TO_FAMILY = {
    tag: family
    for family, tags in ARCHITECTURE_FAMILIES.items()
    for tag in tags
}

MAX_MODELS_ENV_VAR = "AI4LIFE_TEST_MAX_MODELS"


def _get_family(tags):
    lowered = {t.lower() for t in tags}
    for tag, family in TAG_TO_FAMILY.items():
        if tag in lowered:
            return family
    return "other"


def _get_download_count(model):
    count = model.get("download_count", 0)
    return int(count) if str(count).isdigit() else 0


def _get_max_models():
    raw = os.environ.get(MAX_MODELS_ENV_VAR)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _rank_key(model):
    return (-_get_download_count(model), model["model_id"])


def select_models(data):
    model_groups = defaultdict(list)

    for model_id, model in data.items():
        model["model_id"] = model_id
        tags = model.get("tags", [])
        family = _get_family(tags)
        model_groups[family].append(model)

    selected_models = []
    for family in sorted(model_groups):
        best = min(model_groups[family], key=_rank_key)
        selected_models.append(best)

    selected_models.sort(key=_rank_key)

    max_models = _get_max_models()
    if max_models is not None:
        selected_models = selected_models[:max_models]

    return [model["model_id"] for model in selected_models]


def main():
    path = os.path.join(aimodel.config.MODELS_PATH, "filtered_models.json")
    with open(path, "r") as file:
        models_data = json.load(file)

    selected = select_models(models_data)
    return selected


if __name__ == "__main__":
    selected = main()
    print(json.dumps(selected, indent=2))
