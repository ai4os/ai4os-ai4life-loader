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
import imghdr
from typing_extensions import  assert_never
from bioimageio.core.io import load_image
from bioimageio.core import Tensor

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


# TODO: Is there any way to filter v0_5 model before loading them?
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle the case if the object type is unknown or non-serializable
        return str(obj) 

def filter_and_load_models(input_json='collection.json', 
                           output_json='filtered_models.json',
                           perform_io_checks= False):
    # Load the JSON file
    with open(input_json, 'r') as file:
        data = json.load(file)
        
    # Filter entries where "type" is "model"
    models = [entry for entry in data['collection'] if entry['type'] == 'model']
    
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
             
            model = load_description(model_id, 
                                     perform_io_checks=perform_io_checks)

            if isinstance(model, v0_5.ModelDescr):
                print(f"\nThe model '{model.name}' with ID '{model_id}' has been correctly loaded.")
                # Store model information in a dictionary
                model_io_info  = get_model_io_info(model)
                combined_entry = {**model_entry, **model_io_info}
            
                # Store the combined entry in the models_v0_5 dictionary
                models_v0_5[model_id] = combined_entry


    # Define output path
    names_output_json = os.path.join(config.MODELS_PATH, 'models_v0_5.json')        
    
    # Write all model info to a JSON file
    with open(names_output_json, 'w') as names_file:
        json.dump(models_v0_5, names_file, indent=4, cls=CustomEncoder)        
    return models_v0_5
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

 
 
def _copy_file_to_tmpdir(file, tmpdir, input_output_info):
    """Helper function to copy a file to a temporary directory and return the file path or image array."""
    # Copy file to temporary directory
    file_path = Path(tmpdir) / file.original_filename
    shutil.copy(file.filename, file_path)
    
    array= load_image(file_path)
    print(f'the shape of the array is {array.shape}')
    image_type = check_image_type(file_path)
    tensor_array= Tensor._interprete_array_wo_known_axes(array)
    print(f'the tensor_array is {tensor_array.shape_tuple}')
    
    array_dim= _interprete_array_wo_known_axes(  array )      
    print(f'the array_dim is {array_dim}')  
    axes_dim= input_output_info['inputs'][0]['axis']
    axes_ids = (axis.id for axis in axes_dim)
    missing_axes= tuple(a for a in axes_ids if a not in array_dim)
    print (f'the missing axes are {missing_axes}')
    if image_type:
        info, position = check_channel_position(input_output_info['inputs'])
        if info:
            array = np.moveaxis(array, -1, position - 1)
            print(f'input array has shape {array.shape}')

    # Check if file is an image type
   # image_type = check_image_type(file_path)
   # if image_type:
        # Determine expected axes and check for supported shape
    #    axes = input_output_info['inputs'][0]['axis']
      
        # Open image and convert based on axes length
     ##   with Image.open(file_path) as img:
        
     #       image_array = np.array(img.convert('RGB') if len(axes) == 4 else img)
        #if len(axes)-1!= image_array.shape:
         #   raise ValueError(f'This model support images with dimensions {len(axes)-1}.')
        # Check channel position and adjust if necessary
      #  info, position = check_channel_position(input_output_info['inputs'])
     #   if info:
        #    image_array = np.moveaxis(image_array, -1, position - 1)
     #       print(f'Image array shape is {image_array.shape}')
        
    return array, missing_axes
 

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
from bioimageio.core.axis import  AxisId

def _interprete_array_wo_known_axes(array):
        ndim = array.ndim
        if ndim == 2:
            current_axes = (
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[0]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[1]),
            )
        elif ndim == 3 and any(s <= 3 for s in array.shape):
            current_axes = (
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(array.shape[0])
                    ]
                ),
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[1]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[2]),
            )
        elif ndim == 3:
            current_axes = (
                v0_5.SpaceInputAxis(id=AxisId("z"), size=array.shape[0]),
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[1]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[2]),
            )
        elif ndim == 4:
            current_axes = (
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(array.shape[0])
                    ]
                ),
                v0_5.SpaceInputAxis(id=AxisId("z"), size=array.shape[1]),
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[2]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[3]),
            )
        elif ndim == 5:
            current_axes = (
                v0_5.BatchAxis(),
                v0_5.ChannelAxis(
                    channel_names=[
                        v0_5.Identifier(f"channel{i}") for i in range(array.shape[1])
                    ]
                ),
                v0_5.SpaceInputAxis(id=AxisId("z"), size=array.shape[2]),
                v0_5.SpaceInputAxis(id=AxisId("y"), size=array.shape[3]),
                v0_5.SpaceInputAxis(id=AxisId("x"), size=array.shape[4]),
            )
        else:
            raise ValueError(f"Could not guess an axis mapping for {array.shape}")

        return tuple(a.id for a in current_axes)
    
if __name__ == "__main__":
 
  filter_and_load_models(os.path.join(config.MODELS_PATH, 'collection.json'))