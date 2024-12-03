import os

import json
from bioimageio.core import load_description
from bioimageio.spec.model import v0_5
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
                #model_io_info  = get_model_io_info(model)
                combined_entry = {**model_entry}
            
                # Store the combined entry in the models_v0_5 dictionary
                models_v0_5[model_id] = combined_entry

                for weight in model.weights:
                    weight_format, weight_info = weight
                    #We support pytorch weights
                    #for now 
                   # if (weight_format == 'torchscript' or weight_format == 'pytorch_state_dict') and weight_info is not None:

                             # Define output path
                    #names_output_json = os.path.join(config.MODELS_PATH, 'models_v0_5.json')        
    
    # Write all model info to a JSON file
    with open(output_json, 'w') as names_file:
        json.dump(models_v0_5, names_file, indent=4, cls=CustomEncoder)        
    return models_v0_5

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Filter models from a JSON collection.")
    parser.add_argument('--input', required=True, help="Path to the input collection.json")
    parser.add_argument('--output', required=True, help="Path to save the filtered models.json")
    args = parser.parse_args()

    filter_and_load_models(args.input, args.output)