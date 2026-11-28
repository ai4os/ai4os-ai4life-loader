"""Run one prediction of a TensorFlow model in the TensorFlow environment.

This script is started by ``ai4life._run_prediction`` with the Python of the
separate TensorFlow environment (``config.TF_PYTHON``, see
requirements-tf.txt). It must only import what that environment has:
bioimageio.core and numpy, never the ``ai4life`` package.

Usage:
    <tf-python> tf_runner.py <model source> <work dir>

``<work dir>/inputs.json`` lists the input tensors::

    {"members": {"<member id>": {"file": "input_0.npy", "dims": ["b", "c", "y", "x"]}},
     "blocksize_parameter": 10}

``blocksize_parameter`` is optional. The script writes the output tensors
the same way to ``<work dir>/outputs.json``, or, when the prediction fails,
``<work dir>/error.json`` with ``{"type": ..., "message": ...}`` and exit
code 2.
"""

import json
import os
import sys
import traceback


def _predict(source, work_dir):
    import numpy as np
    from bioimageio.core import Tensor, load_description
    from bioimageio.core import predict as bioimageio_predict
    from bioimageio.core.digest_spec import create_sample_for_model

    with open(os.path.join(work_dir, "inputs.json"), encoding="utf-8") as f:
        request = json.load(f)

    model = load_description(source, perform_io_checks=False)
    if type(model).__name__ == "InvalidDescr":
        raise ValueError(f"Invalid model description: {source}")

    inputs = {
        member_id: Tensor.from_numpy(
            np.load(os.path.join(work_dir, spec["file"]), allow_pickle=False),
            dims=spec["dims"],
        )
        for member_id, spec in request["members"].items()
    }
    sample = create_sample_for_model(model, inputs=inputs, sample_id="sample_")

    kwargs = {}
    if request.get("blocksize_parameter") is not None:
        kwargs["blocksize_parameter"] = request["blocksize_parameter"]
    result = bioimageio_predict(model=model, inputs=sample, **kwargs)

    outputs = {}
    for i, (member_id, tensor) in enumerate(result.members.items()):
        file_name = f"output_{i}.npy"
        data = tensor.data
        np.save(
            os.path.join(work_dir, file_name),
            np.asarray(getattr(data, "values", data)),
            allow_pickle=False,
        )
        outputs[str(member_id)] = {
            "file": file_name,
            "dims": [str(dim) for dim in tensor.dims],
        }
    with open(os.path.join(work_dir, "outputs.json"), "w", encoding="utf-8") as f:
        json.dump({"members": outputs}, f)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    source, work_dir = sys.argv[1], sys.argv[2]
    # Let TensorFlow take GPU memory as needed instead of all of it.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    try:
        _predict(source, work_dir)
    except Exception as e:
        traceback.print_exc()
        with open(os.path.join(work_dir, "error.json"), "w", encoding="utf-8") as f:
            json.dump({"type": type(e).__name__, "message": str(e)}, f)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
