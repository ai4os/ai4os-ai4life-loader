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
from bioimageio.core.digest_spec import get_member_ids

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
    input_ids = list(get_member_ids(model.inputs))
    output_ids = set(get_member_ids(model.outputs))
    input_data = {}
    if len(input_ids)==1:
        with tempfile.TemporaryDirectory() as tmpdir:
            id=input_ids[0]
            input_data[id] = utils._copy_file_to_tmpdir(options['input_file'], tmpdir)
            return predict_(model=model, inputs=input_data, blocksize_parameter=1), output_ids 
    else:
        
        options_input = ['input_file', 'box_prompts', 'point_prompts', 'point_labels', 'mask_prompts', 'embeddings']
        with tempfile.TemporaryDirectory() as tmpdir:
            for id, option in zip(input_ids, options_input):
                
                    if id in ['image', 'mask_prompts'] and options.get(option):
                        input_data[id] = utils._copy_file_to_tmpdir(options[option], tmpdir)
                    else:
                        if options.get(option) is not None:
                            input_data[id] = np.array(options[option])
                            
                        #else:    
                         #   input_data[id] = None 
                         #TODO
                          

            # blocksize_parameter=1
            return  predict_(model=model, inputs=input_data), output_ids
    logger.debug(f"[predict()]: {predict_result}")

    #return predict_result

# TODO: train
# = HAVE TO MODIFY FOR YOUR NEEDS =
def train(model_name, input_file, **options):
    """Main/public method to perform training
    """
    # prepare the dataset, e.g.
    # dtst.mkdata()
    
    # create model, e.g.
    # create_model()
    
    # train model
    # describe training steps

    # return training results
    train_result = {'result': 'not implemented'}
    logger.debug(f"[train()]: {train_result}")
    
    return train_result
