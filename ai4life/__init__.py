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
import functools
import importlib.abc
import importlib.util
import inspect
import logging
import sys
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


class _Cellpose3Compat(importlib.abc.MetaPathFinder):
    """Provide Cellpose modules missing from the installed Cellpose version.

    - ``cellpose.vit`` (Cellpose 4 only): Cellpose-SAM/-DINO models (e.g.
      idealistic-eagle) import ``CPSAM``/``CPDINO`` from it.
    - ``cellpose.resnet_torch`` (Cellpose 3 only): Cellpose 3 models (e.g.
      famous-fish) import ``CPnet`` from it.

    The environment uses Cellpose 3; the vendored copies make the models of
    the other version work too. This finder is added at the end of
    ``sys.meta_path``, so it is only used when the installed Cellpose does
    not have the module itself.
    """

    _here = os.path.dirname(__file__)
    _modules = {
        "cellpose.vit": os.path.join(_here, "cellpose4_vit.py"),
        "cellpose.resnet_torch": os.path.join(
            _here, "cellpose_resnet_torch.py"
        ),
    }

    def find_spec(self, fullname, path=None, target=None):
        vendored = self._modules.get(fullname)
        if vendored is None:
            return None
        return importlib.util.spec_from_file_location(fullname, vendored)


if not any(isinstance(f, _Cellpose3Compat) for f in sys.meta_path):
    sys.meta_path.append(_Cellpose3Compat())


def _patch_biapy_upblock(blocks):
    """Let BiaPy >= 3.7 ``UpBlock`` accept calls written for BiaPy <= 3.6.

    BiaPy 3.7 inserted ``yx_down`` after ``k_size`` in ``UpBlock.__init__``.
    Models built with older BiaPy (e.g. efficient-raccoon) pass ``act`` as
    the 9th positional argument, which then lands in ``yx_down``. Such
    old-style calls get the default ``yx_down=2`` inserted; the layers (and
    so the weights) are the same in both versions.
    """
    up_block = getattr(blocks, "UpBlock", None)
    if up_block is None or getattr(up_block, "_ai4os_compat", False):
        return
    if "yx_down" not in inspect.signature(up_block.__init__).parameters:
        return  # old BiaPy, nothing to adapt
    original_init = up_block.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        # args[8] is yx_down (int or 2 ints) in new-style calls, but the
        # activation (str or None) in old-style calls
        if (
            len(args) > 8
            and "yx_down" not in kwargs
            and not isinstance(args[8], (int, tuple, list))
        ):
            args = args[:8] + (2,) + args[8:]
        original_init(self, *args, **kwargs)

    up_block.__init__ = __init__
    up_block._ai4os_compat = True


class _PatchingLoader(importlib.abc.Loader):
    """Run the real loader, then apply a compatibility patch."""

    def __init__(self, loader, patch):
        self._loader = loader
        self._patch = patch

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        self._patch(module)


class _PostImportPatcher(importlib.abc.MetaPathFinder):
    """Patch library modules right after they are imported."""

    _patches = {"biapy.models.blocks": _patch_biapy_upblock}

    def find_spec(self, fullname, path=None, target=None):
        patch = self._patches.get(fullname)
        if patch is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PatchingLoader(spec.loader, patch)
                return spec
        return None


if not any(isinstance(f, _PostImportPatcher) for f in sys.meta_path):
    sys.meta_path.insert(0, _PostImportPatcher())
if "biapy.models.blocks" in sys.modules:  # imported before ai4life
    _patch_biapy_upblock(sys.modules["biapy.models.blocks"])


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
