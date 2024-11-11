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
from bioimageio.spec import load_description
import json
from bioimageio.spec.model import v0_5
import os

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


# TODO: Is there any way to filter v0_5 model before loading them?

def filter_and_load_models(input_json='all_versions.json', output_json='filtered_models.json'):
    # Load the JSON file
    with open(input_json, 'r') as file:
        data = json.load(file)
        
    # Filter entries where "type" is "model"
    models = [entry for entry in data['entries'] if entry['type'] == 'model']

    models_v0_5 = {}

    for model_entry in models:
        model_id = None
        model = None
        
        if model_entry.get('concept'):
            model_id = model_entry['concept']
        elif model_entry.get('concept_doi'):
            model_id = model_entry['concept_doi']
        elif model_entry.get('source'):
            model_id = model_entry['source']

        if model_id:
            model = load_description(model_id)
            if isinstance(model, v0_5.ModelDescr):
                print(f"\nThe model '{model.name}' with ID '{model_id}' has been correctly loaded.")
                # Store model information in a dictionary
                models_v0_5[model_id] = get_model_io_info(model)

    # Define output path
    names_output_json = os.path.join(config.MODELS_PATH, 'models_v0_5.json')        
    
    # Write all model info to a JSON file
    with open(names_output_json, 'w') as names_file:
        json.dump(models_v0_5, names_file, indent=4)        

    return models_v0_5

def get_model_io_info(model):
    model_info = {
        'mode name': model.name,
        'inputs_shape': [],
        'outputs_shape': []
    }

    # Collect input information
    for ipt in model.inputs:
        input_info = {
            'id': getattr(ipt, 'id', None) or getattr(ipt, 'name', None),
            'axes': ipt.axes,
            'data_description': getattr(ipt, 'data', None),
            'test_tensor': getattr(ipt, 'test_tensor', None).source.absolute() if getattr(ipt, 'test_tensor', None) else None,
            'preprocessing': [p for p in ipt.preprocessing] if len(ipt.preprocessing) > 1 else None
        }
        model_info['inputs'].append(input_info)

    # Collect output information
    for out in model.outputs:
        output_info = {
            'id': getattr(out, 'id', None) or getattr(out, 'name', None),
            'axes': out.axes,
            'data_description': getattr(out, 'data', None),
            'test_tensor': getattr(out, 'test_tensor', None).source.absolute() if getattr(out, 'test_tensor', None) else None,
            'postprocessing': [p for p in out.postprocessing] if getattr(out, 'postprocessing', None) and len(out.postprocessing) > 1 else None
        }
        model_info['outputs'].append(output_info)

    return model_info

def warm(**kwargs):
    """Main/public method to start up the model
    """
    # if necessary, start the model
    filter_and_load_models((config.MODELS_PATH))

   


# TODO: predict
# = HAVE TO MODIFY FOR YOUR NEEDS =
def predict(model_name, input_file, **options):
    """Main/public method to perform prediction
    """
    # if necessary, preprocess data
    
    # choose AI model, load weights
    
    # return results of prediction
    predict_result = {'result': 'not implemented'}
    logger.debug(f"[predict()]: {predict_result}")

    return predict_result

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
