"""Utilities module for API endpoints and methods.
This module is used to define API utilities and helper functions. You can
use and edit any of the defined functions to improve or add methods to
your API.

The module shows simple but efficient example utilities. However, you may
need to modify them for your needs.
"""

import logging
import os
import sys
import json
import matplotlib.pyplot as plt
import io
import numpy as np
import ai4life as aimodel

from . import config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def ls_dirs(path):
    """Utility to return a list of directories available in `path` folder.

    Arguments:
        path -- Directory path to scan for folders.

    Returns:
        A list of strings for found subdirectories.
    """
    logger.debug("Scanning directories at: %s", path)
    # dirscan = (x.name for x in path.iterdir() if x.is_dir())
    with open(path, "r") as file:
        models_data = json.load(file)
    return models_data


def ls_files(path, pattern):
    """Utility to return a list of files available in `path` folder.

    Arguments:
        path -- Directory path to scan.
        pattern -- File pattern to filter found files. See glob.glob() python.

    Returns:
        A list of strings for files found according to the pattern.
    """
    logger.debug("Scanning for %s files at: %s", pattern, path)
    dirscan = (x.name for x in path.glob(pattern))
    return sorted(dirscan)


def generate_arguments(schema):
    """Function to generate arguments for DEEPaaS using schemas."""
    def arguments_function():  # fmt: skip
        logger.debug("Web args schema: %s", schema)
        return schema().fields
    return arguments_function


def predict_arguments(schema):
    """Decorator to inject schema as arguments to call predictions."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_predict_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema


def train_arguments(schema):
    """Decorator to inject schema as arguments to perform training."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_train_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema


# Function to display input and prediction output images
def show_images(input_array, output_):

    buffer = io.BytesIO()
    # Check for the number of channels to enable display
    input_array = np.squeeze(input_array)
    if len(input_array.shape) > 2:
        input_array = input_array[0]

    output_array = next(iter(output_.values()))

    # Check for the number of channels to enable display
    output_array = np.squeeze(output_array)
    if len(output_array.shape) > 2:
        output_array = output_array[0]

    fig = plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Input")
    ax1.axis("off")
    plt.imshow(np.asarray(input_array))
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Prediction")
    ax2.axis("off")
    ax2.imshow(output_array)
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)
    return buffer


def output_png(sample, output_):

    input_array = sample

    # if len(output_) == 1:
    output__ = {}
    if len(output_) > 1:
        output__["masks"] = np.array(output_.get("masks"))
    else:
        output__ = output_
    return show_images(input_array, output__)


def get_models_name():
    models_data = ls_dirs(
        os.path.join(config.MODELS_PATH, "collection.json")
    )
    # Filter models from collection
    models_list = [
        entry
        for entry in models_data["collection"]
        if entry["type"] == "model"
    ]
    model_name = aimodel.config.MODEL_NAME

    try:
        # Use next() with the filtered list directly
        model_nickname = next(
            (
                model["nickname_icon"]
                for model in models_list
                if model["name"] == model_name
            ),
            None,
        )
        if model_nickname:
            model_name = f"{model_name} {model_nickname}"

        return [model_name]
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error processing models_data: {e}")
        return [model_name]


def get_model_id_by_name(display_name):
    models_data = ls_dirs(
        os.path.join(config.MODELS_PATH, "collection.json")
    )
    models_list = [
        entry
        for entry in models_data["collection"]
        if entry["type"] == "model"
    ]

    # Extract the base name without the icon
    base_name = (
        display_name.split(" ")[0]
        if " " in display_name
        else display_name
    )

    # Find the model ID that matches the name
    for model in models_list:
        if model["name"] == base_name:
            return model["id"]

    return None


def hide_input():
    path = os.path.join(config.MODELS_PATH, "collection.json")
    model_name = aimodel.config.MODEL_NAME
    return aimodel.utils.load_models(
        model_name, path, perform_io_checks=False
    )
