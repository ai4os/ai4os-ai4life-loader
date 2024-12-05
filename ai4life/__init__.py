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
import os

from bioimageio.core import predict as predict_
from bioimageio.core import load_description
import tempfile
from bioimageio.core import Tensor
from . import utils
from bioimageio.spec.model import v0_5
from bioimageio.core.digest_spec import (
    get_member_ids,
    create_sample_for_model,
)

from ai4life import config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def warm(**kwargs):
    """Main/public method to start up the model"""
    path = os.path.join(config.MODELS_PATH, "collection.json")
    model_name = config.MODEL_NAME
    utils.load_models(model_name, path, perform_io_checks=True)


def predict(model_name, **options):
    """Main/public method to perform prediction"""
    model_name, icon = model_name.split(" ", 1)
    model = load_description(model_name)
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
            blocksize_parameter = 10
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

            return (
                predict_(
                    model=model,
                    inputs=sample,
                    blocksize_parameter=blocksize_parameter,
                ),
                output_ids,
                input_data[id],
            )
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
                 #   input_data[id] = Tensor.from_numpy(
             #   input_data[id], dims=model.inputs[0].axes
          #  )

                elif options.get(option) is not None:
                    input_data[id] = np.array(options[option])

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
