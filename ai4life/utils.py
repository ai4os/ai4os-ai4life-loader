"""Package to create datasets, pipelines and other utilities.

You can use this module for example, to write all the functions needed to
operate the methods defined at `__init__.py` or in your scripts.

All functions here are optional and you can add or remove them as you need.
"""
import logging
from pathlib import Path
import numpy as np
from ai4life import config
from bioimageio.core.digest_spec import get_member_ids
import json
from bioimageio.spec.model import v0_5
from bioimageio.core import   load_description
import os
from typing import List, Tuple
import shutil
from PIL import Image
from typing_extensions import  assert_never


logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


# TODO: Is there any way to filter v0_5 model before loading them?
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle the case if the object type is unknown or non-serializable
        return str(obj) 

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
        json.dump(models_v0_5, names_file, indent=4, cls=CustomEncoder)        

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
            raise NotImplementedError(f"Can't handle axes like '{axis}' yet")
        else:
            assert_never(axis.size)
            
    return min_shape, step

def get_model_io_info(model):
    model_info = {
        'model name': model.name,
        'inputs': [],
        'outputs': []
    }

    # Collect input information
    input_ids = set(get_member_ids(model.inputs))
    for ipt in model.inputs:
        min_shape, step = _process_v0_5_input(ipt)
        input_info = {
            'id': getattr(ipt, 'id', None) or getattr(ipt, 'name', None),
            'axis' : ipt.axes,
            'shape': f'The input shape for the model requires a minimum size of {min_shape} and can increase by {step}',#ipt.axis,
            'data_description': getattr(ipt, 'data', None),
            'test_tensor': getattr(ipt, 'test_tensor', None).source.absolute() if getattr(ipt, 'test_tensor', None) else None,
            
            #'preprocessing': [p for p in ipt.preprocessing] if len(ipt.preprocessing) > 1 else None
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

# make data
# = HAVE TO MODIFY FOR YOUR NEEDS =
def mkdata(input_filepath, output_filepath):
    """ Main/public function to run data processing to turn raw data
        from (data/raw) into cleaned data ready to be analyzed.
    """

    logger.info('Making final data set from raw data')

    # EXAMPLE for finding various files
    project_dir = Path(__file__).resolve().parents[2]


# create model
# = HAVE TO MODIFY FOR YOUR NEEDS =
 
def _copy_file_to_tmpdir(file, tmpdir, input_output_info):
    """Helper function to copy a file to a temporary directory and return the file path or image array."""
    # Copy file to temporary directory
    file_path = Path(tmpdir) / file.original_filename
    shutil.copy(file.filename, file_path)

    # Check if file is an image type
    image_type = check_image_type(file_path)
    if image_type:
        # Determine expected axes and check for supported shape
        axes = input_output_info['inputs'][0]['axis']
        if len(axes) not in (3, 4):
            raise ValueError('This model does not support images with unsupported dimensions.')

        # Open image and convert based on axes length
        with Image.open(file_path) as img:
            image_array = np.array(img.convert('RGB') if len(axes) == 4 else img)

        # Check channel position and adjust if necessary
        info, position = check_channel_position(input_output_info['inputs'])
        if info:
            image_array = np.moveaxis(image_array, -1, position - 1)
            print(f'Image array shape is {image_array.shape}')
        
        return image_array

    # Return file path if not an image
    return file_path
import imghdr
def check_image_type(filename):
    image_type = imghdr.what(filename)
    if image_type in ['jpeg', 'jpg','png']:
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
    #print(input_info)
    axes = input_info[0]['axis']
    
    # Check if any axis has 'channels' name
    has_channels = any('channel' in str(axis).lower() for axis in axes)
    
    if not has_channels:
        return False, "No channels dimension found"
    
    # Find position of channels
    for idx, axis in enumerate(axes):
        if 'channel' in str(axis).lower():
            print(f'the channel pisition is {idx}')
            return True, idx
    
    return False, "channels not found"    