
import json
from bioimageio.core import load_description
from bioimageio.spec.model import v0_5

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        return str(obj)

def filter_and_load_models(input_json, output_json, perform_io_checks=False):
    with open(input_json, 'r') as file:
        data = json.load(file)

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
            model = load_description(model_id, perform_io_checks=perform_io_checks)

            if isinstance(model, v0_5.ModelDescr):
                print(f"The model '{model.name}' with ID '{model_id}' has been loaded.")
                combined_entry = {**model_entry}
                models_v0_5[model_id] = combined_entry

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