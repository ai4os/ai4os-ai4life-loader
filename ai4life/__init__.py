"""Package to create dataset, build training and prediction pipelines.

This file should define or import all the functions needed to operate the
methods defined at ai4life/api.py. Complete the TODOs
with your own code or replace them importing your own functions.
For example:
```py
from your_module import your_function as predict
from your_module import your_function as training
```
"""

# TODO: add your imports here
import logging
import numpy as np

# Registers torchvision's operators (e.g. torchvision::nms) so that
# TorchScript models using them can be loaded.
import torchvision  # noqa: F401
import os

from bioimageio.core import predict as predict_
import tempfile
from bioimageio.core import Tensor
from ai4life import utils
from bioimageio.spec.model import v0_5
from bioimageio.core.digest_spec import (
    get_member_ids,
    create_sample_for_model,
)

from ai4life import config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def _load_v0_5_model(model_name):
    """Load a supported model from the source in filtered_models.json."""
    return utils.load_model_description(model_name)


def warm(**kwargs):
    """Main/public method to start up the model"""
    utils.load_models(config.MODEL_NAME, perform_io_checks=True)


def predict(model_name, **options):
    """Main/public method to perform prediction"""
    model = _load_v0_5_model(model_name)
    input_output_info = utils.get_model_io_info(model)
    input_ids = list(get_member_ids(model.inputs))
    output_ids = set(get_member_ids(model.outputs))
    input_data = {}

    if len(input_ids) == 1:
        with tempfile.TemporaryDirectory() as tmpdir:
            id = input_ids[0]
            input_data[id], missing_axes = utils._copy_file_to_tmpdir(
                options["input_file"], tmpdir, input_output_info
            )
            # 3D models grow in z, y and x at once, so a large block size
            # quickly needs tens of GB of memory. Use the largest block that
            # stays under a size limit for them.
            has_z = any(str(a.id) == "z" for a in model.inputs[0].axes)
            blocksize_parameter = (
                _choose_3d_blocksize(model) if has_z else 10
            )
            input_block_shape = model.get_tensor_sizes(
                get_ns(blocksize_parameter, model), batch_size=1
            ).inputs
            print(f"the target_tensor is {input_block_shape}")
            if "z" in missing_axes:
                raise ValueError(
                    "This model needs a 3D image as input, but "
                    "a 2D image is given."
                )
            input_tensor = Tensor.from_numpy(
                input_data[id], dims=model.inputs[0].axes
            )
            print(f"the input_tensor is {input_tensor.shape_tuple}")

            padded_input = input_tensor.pad_to(input_block_shape[id])
            print(f"the pad input is {padded_input.shape_tuple}")
            input_data_pad = {}
            input_data_pad[id] = padded_input
            sample = create_sample_for_model(
                model, inputs=input_data_pad, sample_id="sample_"
            )

            # bioimageio.core 0.11 can produce empty blocks for the blocked
            # prediction of some models (e.g. Cellpose, empanada): with one
            # block the result is empty, with several blocks it raises
            # "block with members [] does not overlap with sample members".
            # In both cases, predict the whole image at once instead.
            try:
                result = predict_(
                    model=model,
                    inputs=sample,
                    blocksize_parameter=blocksize_parameter,
                )
                blocked_ok = bool(result.members)
            except ValueError as err:
                if "does not overlap" not in str(err):
                    raise
                blocked_ok = False
            if not blocked_ok:
                # A new sample is needed: preprocessing changes the sample
                # in place.
                logger.warning(
                    "Blocked prediction returned no outputs, "
                    "predicting without blocking."
                )
                sample = create_sample_for_model(
                    model, inputs=input_data_pad, sample_id="sample_"
                )
                result = predict_(model=model, inputs=sample)
            return (result, output_ids, input_data[id])
    else:

        options_input = [
            "input_file",
            "box_prompts",
            "point_prompts",
            "point_labels",
            "mask_prompts",
            "embeddings",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for id, option in zip(input_ids, options_input):

                if id in [
                    "image",
                    "mask_prompts",
                    "embeddings",
                ] and options.get(option):

                    input_data[id], _ = utils._copy_file_to_tmpdir(
                        options[option], tmpdir, input_output_info
                    )
                    input_data[id] = input_data[id].astype(np.float32)

                elif options.get(option) is not None:
                    input_data[id] = np.array(options[option])
                    print(f"the shape of array {id} is {input_data[id].shape}")

                if (
                    "box_prompts" in input_data
                    and "mask_prompts" in input_data
                ):
                    # Example input shapes
                    box_prompts = input_data[
                        "box_prompts"
                    ]  # Shape: (1, num_boxes, 4)
                    mask_prompts = input_data[
                        "mask_prompts"
                    ]  # Shape: (1, num_masks, height, width)

                    # Check if padding is needed
                    if mask_prompts.shape[1] < box_prompts.shape[1]:
                        num_boxes = box_prompts.shape[1]
                        num_masks = mask_prompts.shape[1]

                        # Calculate how many masks to pad
                        pad_size = num_boxes - num_masks

                        # Create a zero array to pad with
                        padding = np.zeros(
                            (
                                mask_prompts.shape[0],
                                pad_size,
                                mask_prompts.shape[2],
                                mask_prompts.shape[3],
                                mask_prompts.shape[4],
                            )
                        )
                        print("the pading shape is ", padding.shape)
                        # Concatenate the zero array to mask_prompts
                        print("the mask shape is ", mask_prompts.shape)
                        mask_prompts = np.concatenate(
                            (mask_prompts, padding), axis=1
                        )

                    # Update the input data
                    input_data["mask_prompts"] = mask_prompts
            sample = create_sample_for_model(
                model, inputs=input_data, sample_id="sample_"
            )
            input_data = sample.members[input_ids[0]].data
            print("input shape is", input_data.shape)

            axes = input_output_info["inputs"][0]["axis"]
            if len(axes) != len(input_data.shape):
                raise ValueError(
                    "This model supports input image with "
                    f"dimensions {len(axes)}, but you input an "
                    "image with dimensions {len(input_data.shape)}."
                )

            # blocksize_parameter is not working with sam model
            return (
                predict_(model=model, inputs=sample),
                output_ids,
                input_data,
            )


def get_ns(n: int, model):
    return {
        (t.id, a.id): n
        for t in model.inputs
        for a in t.axes
        if isinstance(a.size, v0_5.ParameterizedSize)
    }


def _choose_3d_blocksize(model, max_voxels=2_500_000, max_n=10):
    """Largest block size whose input block stays under ``max_voxels``.

    Very small blocks are slow for models with a large halo (most of each
    block is border that is thrown away), very large blocks need too much
    memory. E.g. impartial-shrimp gets 1, loyal-ant gets 3.
    """
    input_id = model.inputs[0].id
    for n in range(max_n, 0, -1):
        sizes = model.get_tensor_sizes(get_ns(n, model), batch_size=1).inputs
        voxels = 1
        for axis_id, size in sizes[input_id].items():
            if str(axis_id) in ("z", "y", "x"):
                voxels *= size
        if voxels <= max_voxels:
            return n
    return 1


if __name__ == "__main__":
    from deepaas.model.v2.wrapper import UploadedFile

    model_name = "affectionate-cow \ud83d\udc04"
    options = {}
    path = "/home/se1131/ai4life_project/ai4life/data/affectionate-cow.npy"
    file_extension = os.path.splitext(path)[1]
    options["input_file"] = UploadedFile(
        "files",
        path,
        "application/octet-stream",
        f"files{file_extension}",
    )
    predict(model_name, **options)
