"""Endpoint functions to integrate your model with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.ai4eosc.eu/
[2]: https://github.com/ai4os-hub/demo-advanced
"""
import logging

import ai4life as aimodel 
import os

from . import config, responses, schemas, utils

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def get_metadata():
    """Returns a dictionary containing metadata information about the module.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.API_NAME)
        metadata = {
            "author": config.API_METADATA.get("authors"),
            "author-email": config.API_METADATA.get("author-emails"),
            "description": config.API_METADATA.get("summary"),
            "license": config.API_METADATA.get("license"),
            "version": config.API_METADATA.get("version"),
            "datasets": utils.ls_files(config.DATA_PATH, '[A-Za-z0-9]*'),
            "models": utils.ls_dirs(os.path.join(config.MODELS_PATH, 'models_v0_5.json')),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        logger.error("Error collecting metadata: %s", err, exc_info=True)
        raise  # Reraise the exception after log


def warm():
    """Function to run preparation phase before anything else can start.

    Raises:
        RuntimeError: Unexpected errors aim to stop model loading.
    """
    try:  # Call your AI model warm() method
        logger.info("Warming up the model.api...")
        aimodel.warm()
    except Exception as err:
        logger.error("Error when warming up: %s", err, exc_info=True)
        raise RuntimeError(reason=err) from err


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict(model_name, accept='application/json', **options):
    """Performs model prediction from given input data and parameters.

    Arguments:
        model_name -- Model name from registry to use for prediction values.
        accept -- Response parser type, default is json.
        **options -- Arbitrary keyword arguments from PredArgsSchema.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        The predicted model values (dict or str) or files.
    """
    try:  # Call your AI model predict() method
        logger.info("Using model %s for predictions", model_name)
        #logger.debug("Loading data from input_file: %s", input_file.filename)
        logger.debug("Predict with options: %s", options)
        result, output_ids = aimodel.predict(model_name,  **options)
        logger.debug("Predict result: %s", result)
        logger.info("Returning content_type for: %s", accept)

        return responses.content_types[accept](result, output_ids, **options)
    except Exception as err:
        logger.error("Error calculating predictions: %s", err, exc_info=True)
        raise  # Reraise the exception after log




