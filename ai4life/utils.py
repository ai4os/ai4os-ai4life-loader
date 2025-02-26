"""Package to create datasets, pipelines and other utilities.

You can use this module for example, to write all the functions needed to
operate the methods defined at `__init__.py` or in your scripts.

All functions here are optional and you can add or remove them as you need.
"""

import logging
from pathlib import Path
import numpy as np
from ai4life import config
import json
from bioimageio.spec.model import v0_5
from bioimageio.core import load_description
import os
from typing import List, Tuple
import shutil
import imghdr
from typing_extensions import assert_never
from bioimageio.core.io import load_image
from bioimageio.core.axis import AxisId
from imageio.v3 import imread


logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle the case if the object type is unknown or non-serializable
        return str(obj)


def load_models(models_name, path, perform_io_checks=False):

    with open(path, "r") as file:
        models_data = json.load(file)
    model_entry = next(
        (
            entry
            for entry in models_data.get("collection", [])
            if entry.get("type") == "model"
            and entry.get("id") == models_name
        ),
        None,
    )

    model_id = next(
        (
            model_entry.get(key)
            for key in ["concept", "concept_doi", "source"]
            if model_entry.get(key)
        ),
        None,
    )

    if model_id:
        model = load_description(
            model_id, perform_io_checks=perform_io_checks
        )
        if isinstance(model, v0_5.ModelDescr):
            model_io_info = get_model_io_info(model)

            # Convert non-serializable fields to serializable format
            serializable_io_info = {
                key: (
                    value.__dict__
                    if hasattr(value, "__dict__")
                    else value
                )
                for key, value in model_io_info.items()
            }

            combined_entry = {**model_entry, **serializable_io_info}
            print(
                f"The model '{model.name}' with ID '{model_id}' "
                "has been correctly loaded."
            )

            # Directly return JSON string with proper formatting
            names_output_json = os.path.join(
                config.MODELS_PATH, "model_meta.json"
            )
            with open(names_output_json, "w") as names_file:
                json.dump(
                    combined_entry,
                    names_file,
                    indent=4,
                    cls=CustomEncoder,
                )
            # get the length of the input_model:

            return len(model_io_info["inputs"]) == 1


def _process_v0_5_input(input_descr) -> Tuple[List[int], List[int]]:
    """
    Process v0.5 input descriptor to extract shape information.

    Args:
        input_descr: Input descriptor object

    Returns:
        Tuple of (min_shape, step) lists
    """
    min_shape, step = [], []

    for axis in input_descr.axes:
        if isinstance(axis.size, v0_5.ParameterizedSize):
            min_shape.append(axis.size.min)
            step.append(axis.size.step)
        elif isinstance(axis.size, int):
            min_shape.append(axis.size)
            step.append(0)
        elif axis.size is None:
            axis.size = 1
            min_shape.append(axis.size)
            step.append(0)
        elif isinstance(axis.size, v0_5.SizeReference):
            raise NotImplementedError(
                f"Can't handle axes like '{axis}' yet"
            )
        else:
            assert_never(axis.size)

    return min_shape, step


def get_model_io_info(model):
    model_info = {
        "model name": model.name,
        "inputs": [],
        "outputs": [],
    }

    for ipt in model.inputs:
        min_shape, step = _process_v0_5_input(ipt)
        input_info = {
            "id": getattr(ipt, "id", None)
            or getattr(ipt, "name", None),
            "axis": ipt.axes,
            "shape": "The input shape for the model requires a minimum "
            f"size of {min_shape} and can increase by {step}",
            "data_description": getattr(ipt, "data", None),
            "test_tensor": (
                getattr(ipt, "test_tensor", None).source.absolute()
                if getattr(ipt, "test_tensor", None)
                else None
            ),
        }
        model_info["inputs"].append(input_info)

    # Collect output information
    for out in model.outputs:
        output_info = {
            "id": getattr(out, "id", None)
            or getattr(out, "name", None),
            "axes": out.axes,
            "data_description": getattr(out, "data", None),
            "test_tensor": (
                getattr(out, "test_tensor", None).source.absolute()
                if getattr(out, "test_tensor", None)
                else None
            ),
            "postprocessing": (
                [p for p in out.postprocessing]
                if getattr(out, "postprocessing", None)
                and len(out.postprocessing) > 1
                else None
            ),
        }
        model_info["outputs"].append(output_info)

    return model_info


def _copy_file_to_tmpdir(file, tmpdir, input_output_info):
    """Helper function to copy a file to a temporary directory"""
    """ and return the file path or image array."""
    # Copy file to temporary directory
    file_path = Path(tmpdir) / file.original_filename
    shutil.copy(file.filename, file_path)

    array = load_image(file_path)
    print(f"the shape of the array is {array.shape}")
    image_type = check_image_type(file_path)

    array_dim = _interprete_array_wo_known_axes(array)
    print(f"the array_dim is {array_dim}")
    axes_dim = input_output_info["inputs"][0]["axis"]
    axes_ids = (axis.id for axis in axes_dim)
    missing_axes = tuple(a for a in axes_ids if a not in array_dim)
    print(f"the missing axes are {missing_axes}")
    info, position, num_ch = check_channel_position(
        input_output_info["inputs"]
    )
    if image_type:

        if num_ch == 1:
            array = imread(file_path, mode="L")
        elif info:
            array = np.moveaxis(array, -1, position - 1)
            print(f"input array has shape {array.shape}")

    return array, missing_axes


def check_image_type(filename):
    image_type = imghdr.what(filename)
    if image_type in ["jpeg", "jpg", "png"]:
        return True

    else:
        return False


def check_channel_position(input_info):
    """
    Check if 'channels' exists in axes and determine its position

    Args:
        input_info: Dictionary containing model input information

    Returns:
        tuple: (bool, str) - (whether channels exists, position description)
    """
    # print(input_info)
    axes = input_info[0]["axis"]

    # Check if any axis has 'channels' name
    has_channels = any(
        "channel" in str(axis).lower() for axis in axes
    )

    if not has_channels:
        return False, "No channels dimension found"

    # Find position of channels
    for idx, axis in enumerate(axes):
        if "channel" in str(axis).lower():
            print(f"the channel pisition is {idx}")
            if hasattr(axis, "channel_names"):
                channel_names = axis.channel_names
                num_channels = len(channel_names)
            return True, idx, num_channels

    return False, "channels not found"


def _interprete_array_wo_known_axes(array):
    ndim = array.ndim
    if ndim == 2:
        current_axes = (
            v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[0]),
            v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[1]),
        )
    elif ndim == 3 and any(s <= 3 for s in array.shape):
        current_axes = (
            v0_5.ChannelAxis(
                channel_names=[
                    v0_5.Identifier(f"channel{i}")
                    for i in range(array.shape[0])
                ]
            ),
            v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[1]),
            v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[2]),
        )
    elif ndim == 3:
        current_axes = (
            v0_5.SpaceInputAxis(id=AxisId("z"), size=array.shape[0]),
            v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[1]),
            v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[2]),
        )
    elif ndim == 4:
        current_axes = (
            v0_5.ChannelAxis(
                channel_names=[
                    v0_5.Identifier(f"channel{i}")
                    for i in range(array.shape[0])
                ]
            ),
            v0_5.SpaceInputAxis(id=AxisId("z"), size=array.shape[1]),
            v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[2]),
            v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[3]),
        )
    elif ndim == 5:
        current_axes = (
            v0_5.BatchAxis(),
            v0_5.ChannelAxis(
                channel_names=[
                    v0_5.Identifier(f"channel{i}")
                    for i in range(array.shape[1])
                ]
            ),
            v0_5.SpaceInputAxis(id=AxisId("z"), size=array.shape[2]),
            v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[3]),
            v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[4]),
        )
    else:
        raise ValueError(
            f"Could not guess an axis mapping for {array.shape}"
        )

    return tuple(a.id for a in current_axes)


if __name__ == "__main__":

    model_name = "10.5281/zenodo.13219987"
    model = load_description(model_name)
    input_output_info = get_model_io_info(model)
    check_channel_position(input_output_info["inputs"])
    image_path = "/home/se1131/cat1.jpg"
    array = imread(image_path, mode="L")
    print(array.shape)
