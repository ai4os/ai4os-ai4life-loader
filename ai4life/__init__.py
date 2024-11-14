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
from pathlib import Path
 
from ai4life import config
import numpy as np
import os
from bioimageio.core import predict as predict_ 
from bioimageio.core import load_description
import tempfile
import shutil
from . import utils 
 
from bioimageio.core.digest_spec import get_member_ids,  create_sample_for_model
 
logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def warm(**kwargs):
    """Main/public method to start up the model
    """
    # if necessary, start the model
    utils.filter_and_load_models(os.path.join(config.MODELS_PATH, 'all_versions.json'))

def predict(model_name, **options):
    """Main/public method to perform prediction
    """
    model= load_description(model_name)
    input_output_info= utils.get_model_io_info(model)
    input_ids = list(get_member_ids(model.inputs))
    output_ids = set(get_member_ids(model.outputs))
    input_data = {}
    if len(input_ids)==1:
        with tempfile.TemporaryDirectory() as tmpdir:
            id=input_ids[0]
            input_data[id] = utils._copy_file_to_tmpdir(options['input_file'], tmpdir, input_output_info)
            sample = create_sample_for_model(
            model, inputs=input_data, sample_id='sample_'
        )  
            input_data = sample.members[input_ids[0]].data
            return predict_(model=model, inputs=sample, blocksize_parameter=1), output_ids ,input_data
    else:
        
        options_input = ['input_file', 'box_prompts', 'point_prompts', 'point_labels', 'mask_prompts', 'embeddings']
        with tempfile.TemporaryDirectory() as tmpdir:
            for id, option in zip(input_ids, options_input):
                
                    if id in ['image', 'mask_prompts'] and options.get(option):
                        input_data[id] = utils._copy_file_to_tmpdir(options[option], tmpdir,input_output_info)
                        
                             
                    elif options.get(option) is not None:
                            input_data[id] = np.array(options[option])
            sample = create_sample_for_model(
            model, inputs=input_data, sample_id='sample_'
        )  
            input_data = sample.members[input_ids[0]].data
            # blocksize_parameter is not working with sam model
            return  predict_(model=model, inputs=sample), output_ids, input_data
    logger.debug(f"[predict()]: {predict_result}")



 