import json
from collections import defaultdict
import ai4life as aimodel
import os


def select_models(data):
    def get_download_count(model):
        count = model.get("download_count", 0)
        return int(count) if str(count).isdigit() else 0

    # Group models by their primary function
    model_groups = defaultdict(list)
    selected_models = []

    for model_id, model in data.items():
        model["model_id"] = model_id

        # Determine model category based on tags and description
        tags = set(model.get("tags", []))

        if "instance-segmentation" in tags:
            if "segment-anything" in tags:
                model_groups["sam"].append(model)
            else:
                model_groups["segmentation"].append(model)
        elif "denoising" in tags:
            model_groups["denoising"].append(model)
        elif "image-reconstruction" in tags:
            model_groups["reconstruction"].append(model)
        elif "3d" in tags:
            model_groups["3d"].append(model)
        else:
            model_groups["other"].append(model)

    # Select best model from each group based on criteria
    # 1. Nucleus/Cell Segmentation (highest downloads)
    if model_groups["segmentation"]:
        best_seg = max(
            model_groups["segmentation"], key=get_download_count
        )
        selected_models.append(best_seg["model_id"])

    # 2. SAM-based model (prefer larger variant)
    if model_groups["sam"]:
        sam_model = next(
            (m for m in model_groups["sam"] if "vit_l" in m["name"]),
            None,
        )
        if sam_model:
            selected_models.append(sam_model["model_id"])
        else:
            selected_models.append(model_groups["sam"][0]["model_id"])

    # 3. 3D model
    if model_groups["3d"]:
        best_3d_model = model_groups["3d"][
            0
        ]  # Assume first model is fine if no criteria
        selected_models.append(best_3d_model["model_id"])

    # 4. Denoising (prefer newer architecture)
    if model_groups["denoising"]:
        n2v2_model = next(
            (
                m
                for m in model_groups["denoising"]
                if "N2V2" in m["name"]
            ),
            None,
        )
        if n2v2_model:
            selected_models.append(n2v2_model["model_id"])
        else:
            selected_models.append(
                model_groups["denoising"][0]["model_id"]
            )

    # 5. Reconstruction
    if model_groups["reconstruction"]:
        selected_models.append(
            model_groups["reconstruction"][0]["model_id"]
        )

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
