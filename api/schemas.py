"""Module for defining custom web fields to use on the API interface.
This module is used by the API server to generate the input form for the
prediction and training methods. You can use any of the defined schemas
to add new inputs to your API.

The module shows simple but efficient example schemas. However, you may
need to modify them for your needs.
"""

import marshmallow
from webargs import ValidationError, fields, validate
import json

from . import config, responses, utils


hide_input = utils.hide_input()


class BoxPromptField(fields.Field):
    def __init__(self, *args, **kwargs):

        self.metadata = kwargs.get("metadata", {})
        super().__init__(*args, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        try:
            return json.loads(value)
        except json.JSONDecodeError as err:
            raise ValidationError(f"Invalid JSON: `{err}`")

    def _validate(self, value):
        if not isinstance(value, list):
            raise ValidationError(
                "`prompt` must be a list of dictionaries."
            )
        for item in value:
            if not isinstance(item, list):
                raise ValidationError(
                    "Each item in the list must be a list."
                )
            if len(item) != 4:
                raise ValidationError(
                    "Each item in the list should be a list of"
                    " coordinate of the bounding box with "
                    "[x_min, y_min, x_max, y_max]."
                )
            for val in item:
                if not isinstance(val, int):
                    raise ValidationError(
                        "Value of bounding box coordinate must be an integer."
                    )


class PointPromptsField(fields.Field):
    def __init__(self, *args, **kwargs):
        self.metadata = kwargs.get("metadata", {})
        super().__init__(*args, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        try:
            return json.loads(value)
        except json.JSONDecodeError as err:
            raise ValidationError(f"Invalid JSON: `{err}`")

    def _validate(self, value):
        # Ensure value is a list
        if not isinstance(value, list):
            raise ValidationError(
                "`point_prompts` must be a list of lists."
            )

        for batch in value:
            # Check each batch dimension entry is a list
            if not isinstance(batch, list):
                raise ValidationError(
                    "Each item in `point_prompts` must be a list."
                )
            for obj in batch:
                # Check each object entry in batch is a list
                if not isinstance(obj, list):
                    raise ValidationError(
                        "Each object entry must be a list of points."
                    )
                for point in obj:
                    # Check each point entry in object is a list
                    #  with exactly 2 coordinates
                    if not isinstance(point, list) or len(point) != 2:
                        raise ValidationError(
                            "Each point must be a list with exactly two"
                            " coordinates [x, y]."
                        )
                    # Check each coordinate is an integer
                    for coord in point:
                        if not isinstance(coord, int):
                            raise ValidationError(
                                "Each coordinate must be an integer."
                            )


class PointLabelsField(fields.Field):
    """
    Custom field for point labels input, specifying details
    like data type, shape, and test tensor URL.
    """

    def _serialize(self, value, attr, obj, **kwargs):
        # Serialize field details
        return {
            "type": "int64",
            "shape": [1, 1, 1],  # Minimum shape required
            "description": "Point labels with [batch, object, point]"
            " dimensions",
            "data_description": {
                "range": [None, None],
                "unit": "arbitrary unit",
                "scale": 1.0,
                "offset": None,
            },
        }

    def _deserialize(self, value, attr, data, **kwargs):
        # Enforce the expected shape and type during deserialization
        if not isinstance(value, list) or len(value) != 3:
            raise ValidationError(
                "Point labels must be a list with [batch, object, point]"
                " dimensions."
            )
        if value != [1, 1, 1]:
            raise ValidationError(
                "Point labels shape must be exactly [1, 1, 1]."
            )
        return value


class ModelName(fields.String):
    """Field that takes a string and validates against current available
    models at config.MODELS_PATH.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if value not in utils.get_models_name():
            raise ValidationError(f"Checkpoint `{value}` not found.")
        return str(config.MODELS_PATH / value)


# EXAMPLE of Prediction Args description
# = HAVE TO MODIFY FOR YOUR NEEDS =
class PredArgsSchema(marshmallow.Schema):
    """Prediction arguments schema for api.predict function."""

    class Meta:
        ordered = True

    model_name = fields.String(
        metadata={
            "description": f"\nThe model '**{utils.get_models_name()[0]}**' "
            "has been loaded for inference. "
            f"For more information about the input and output "
            "of the model, please check the "
            f"**Metadata method**.",
        },
        validate=validate.OneOf(utils.get_models_name()),
        required=True,
    )

    input_file = fields.Field(
        metadata={
            "description": (
                "Image file predictions are either saved as "
                ".npy files or other "
                "image formats. Please refer to each model's metadata to "
                "determine the required dimensions for the input image."
            ),
            "type": "file",
            "location": "form",
        },
        required=True,
    )

    box_prompts = BoxPromptField(
        required=False,
        metadata={
            "description": "Bounding box prompt. Should be a list."
            " Each item in the list"
            " should be a list of coordinates of the bounding box "
            "with \n [x_min, y_min, x_max, y_max]. \n"
            "Example:[[0, 0, 100, 100], [255, 350, 400, 550]]\n"
        },
        load_default=None,
        dump_only=hide_input,
    )
    mask_prompts = fields.Field(
        metadata={
            "description": "npy or an image file. SAM will take"
            " this binary input mask as a hint or starting point"
            " and try to refine the segmentation around the "
            "provided mask area.",
            "type": "file",
            "location": "form",
        },
        required=False,
        dump_only=hide_input,
    )

    embeddings = fields.Field(
        metadata={
            "description": "The embeddings represent the image features"
            " that SAM uses for segmentation. It can vbe generated by"
            " the Generated by the image encoder part of SAM. "
            "Embedding input, with a fixed shape of [1, 256, 64, 64] "
            "and float32 type.",
            "type": "file",
            "location": "form",
        },
        dump_only=hide_input,
        required=False,
    )

    # fields.List(fields.List(fields.List(fields.List(fields.Int()))))

    point_prompts = PointPromptsField(
        metadata={
            "description": (
                "Point prompts input with shape "
                "[1, num_object, num_point, (x, y)]"
                " and int64 type, representing "
                "coordinates in 'x' and 'y' channels.\n\n"
                "Example:\n"
                "[\n"
                "    [  # Batch dimension\n"
                "        [  # First object\n"
                "            [10, 20],  # Point 1 (x=10, y=20)\n"
                "            [15, 25],  # Point 2 (x=15, y=25)\n"
                "            [20, 30]   # Point 3 (x=20, y=30)\n"
                "        ],\n"
                "        [  # Second object\n"
                "            [50, 60],  # Point 1 (x=50, y=60)\n"
                "            [55, 65],  # Point 2 (x=55, y=65)\n"
                "            [60, 70]   # Point 3 (x=60, y=70)\n"
                "        ]\n"
                "    ]\n"
                "]"
            )
        },
        dump_only=hide_input,
    )

    # fields.List(fields.List(fields.List(fields.Int())))
    point_labels = PointLabelsField(
        metadata={
            "description": "Point labels input with shape [1, 1, 1]"
            " and int64 type.",
        },
        required=False,
        dump_only=hide_input,
    )
    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        required=True,
        validate=validate.OneOf(list(responses.content_types)),
    )


# EXAMPLE of Training Args description
# = HAVE TO MODIFY FOR YOUR NEEDS =
class TrainArgsSchema(marshmallow.Schema):
    """Training arguments schema for api.train function."""

    class Meta:  # Keep order of the parameters as they are defined.
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

    pass
