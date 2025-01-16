import json
import subprocess
import tempfile
import gradio as gr
import numpy as np
import os
from PIL import Image, ImageDraw#
import inspect
from pathlib import Path
from bioimageio.core import load_description
from bioimageio.spec._internal.io import download

main_path = Path(__file__).parent.absolute()

def load_npy_image(file_path):
    if file_path.endswith('.npy'):
        # Load the .npy file
        img_array = np.load(file_path)
        img_array = np.squeeze(img_array, axis=0)  # Remove batch size if present
        
        # Transpose if shape is (channels, height, width)
        if len(img_array.shape) == 3 and img_array.shape[0] in [1, 3, 4]:
        
            img_array = np.transpose(img_array, (1, 2, 0))  # Convert to (height, width, channels)

        # Convert to uint8 if not already
        if img_array.dtype != np.uint8:
            img_array = (255 * (img_array - img_array.min()) / (img_array.ptp() + 1e-5)).astype(np.uint8)

        # Convert to a PIL image
        img = Image.fromarray(img_array)

        # Save the image to a temporary directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_path = temp_file.name
            img.save(temp_path)

        return {'image': temp_path } # Return the path to the saved image
    else:
        raise ValueError("Provided file is not a .npy file.")


def input_files(model_name):
    """Fixture to provide options dictionary for the model."""
    model_name, icon = model_name.split(" ", 1)
    model = load_description(model_name, perform_io_checks=False)

    inputs = [d.test_tensor for d in model.inputs]
    options = {}

    for input_item in inputs:
        path = download(input_item).path
        filename = os.path.basename(path).split("-")[-1]

        if input_item == inputs[0]:
            options["input_file"] = path
        else:
            filename_without_extension = filename.split(".")[0]
            if filename_without_extension in ["mask_prompts", "embeddings"]:
                options[filename_without_extension] = path
            else:
                options[filename_without_extension] = np.load(path)

    return options



def api2gr_inputs(api_inp):
    """
    Transform DEEPaaS webargs to Gradio inputs.
    """
    inp_names = [i["name"] for i in api_inp]

    inp_types = {i["name"]: i.get("type", None) for i in api_inp}
    gr_inp = []
    for k, v in zip(inp_names, api_inp):

        if k == "accept":
            continue

        elif "enum" in v.keys() and v["type"] not in ["boolean"]:

            gr_inp.append(
                gr.Radio(
                    choices=v["enum"],
                    value=v.get("default", None),
                    label=k,
                )
            )
        elif v.get("type", None) == None:
            pass
        elif v["type"] in ["integer", "number", "float"]:
            if (v["type"] == "integer") and {
                "minimum",
                "maximum",
            }.issubset(v.keys()):
                gr_inp.append(
                    gr.Slider(
                        value=v.get("default", None),
                        minimum=v.get("minimum", None),
                        maximum=v.get("maximum", None),
                        step=1,
                        label=k,
                    )
                )
            else:
                gr_inp.append(
                    gr.Number(value=v.get("default", None), label=k)
                )
        elif v["type"] in ["boolean"]:
            gr_inp.append(
                gr.Checkbox(value=v.get("default", None), label=k)
            )
        elif v["type"] in ["string"]:
            gr_inp.append(
                gr.Textbox(value=v.get("default", None), label=k)
            )

        elif v["type"] in ["file"]:
            gr_inp.append(
                gr.File(
                    type="filepath",  # file_count="multiple",
                    label="Input Files (FASTA format)",
                )
            )
        else:
            raise Exception(
                f"UI does not support some of the input data types: `{k}` :: {v['type']}"
            )

    return gr_inp, inp_names, inp_types


def api2gr_outputs(struct):
    """
    Transform DEEPaaS webargs to Gradio outputs.
    """
    gr_out = []
    for k, v in struct.items():

        if v["type"] == "pdf":
            tmp = gr.outputs.File(type="file", label=k, accept=".pdf")
        elif v["type"] == "json":
            tmp = gr.outputs.JSON(label=k)

        else:
            raise Exception(
                f"UI does not support some of the output data types: {k} [{v['type']}]"
            )
        gr_out.append(tmp)

    return gr_out


def gr2api_input(params, inp_types):
    """
    Transform Gradio inputs to DEEPaaS webargs.
    """
    files = {}
    params = {
        k: v for k, v in params.items() if v is not None
    }  # Remove keys with None values

    for k, v in params.copy().items():

        if inp_types[k] == "integer":
            params[k] = int(v)
        elif inp_types[k] == "number":  # float
            params[k] = float(v)
        elif inp_types[k] == "string":
            params[k] = f"{v}"
        elif inp_types[k] == "boolean":
            params[k] = v
        elif inp_types[k] in ["file"] and v is not None:
            media = params.pop(k)
            path = media
           # if v not in {   "mask_prompts", "embeddings"}:
                
            files[k] = open(path, "rb") 
    
        elif v is not None:
            params[k] = json.dumps(v)
       
    return params, files

def visualize_prompts(image_path, point_prompts, point_labels, box_prompts):
    # Load image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw points
    for point, label in zip(point_prompts[0][0], point_labels[0][0]):
        color = 'green' if label == 1 else 'red'  # Green for foreground, red for background
        x, y = point
        draw.ellipse([x-5, y-5, x+5, y+5], fill=color)

    # Draw boxes
    for box in box_prompts[0]:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline='blue', width=2)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_path = temp_file.name
            image.save(temp_path)
    return temp_path

def get_parameter_default(param_name, api_inp):
    param_value = next(
        (
            param["default"]
            for param in api_inp
            if param["name"] == param_name
        ),
        None,
    )
    return param_value
def get_examples(model_name, api_inp):
        options = input_files(model_name)
        examples = []

        if len(api_inp) == 3:
            # Single image input
            examples.append(str(options["input_file"]))
        else:
            # Multiple inputs

            input_file = str(options["input_file"])
            examples.append(input_file)
            # examples.append(load_npy_image(input_file))
            box_prompts = (
                options["box_prompts"]
                if "box_prompts" in options
                else None
            )
            point_prompts = (
                options["point_prompts"]
                if "point_prompts" in options
                else None
            )
            point_labels = (
                options["point_labels"]
                if "point_labels" in options
                else None
            )
            image_data = load_npy_image(input_file)

            result_image = visualize_prompts(
                image_data["image"],
                point_prompts,
                point_labels,
                box_prompts,
            )
            examples.append({"image": result_image})
            mask_prompts = (
                str(options["mask_prompts"])
                if "mask_prompts" in options
                else None
            )
            examples.append(mask_prompts)
            embeddings = (
                str(options["embeddings"])
                if "embeddings" in options
                else None
            )
            examples.append(embeddings)

        return examples
    
def process_prompts(prompts):
    image = prompts["image"]
    points = prompts["points"]
    if points==[]:
        return image, None, None, None
    # Initialize lists to group points and labels inside boxes
    boxes = []
    points_by_box = []
    labels_by_box = []
    outlier_points = []
    outlier_labels = []

    # Process points and boxes from prompts
    for point in points:
        # Point format: [x, y, label, x2, y2, type]
        if point[-1] == 3:  # Box
            boxes.append([int(point[0]), int(point[1]), int(point[3]), int(point[4])])

    # Create empty lists to hold points and labels for each box
    points_by_box = [[] for _ in range(len(boxes))]
    labels_by_box = [[] for _ in range(len(boxes))]

    for point in points:
        if point[-1] == 4:  # Point
            x, y, label = int(point[0]), int(point[1]), int(point[2])
            point_in_box = False
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    points_by_box[i].append([x, y])
                    labels_by_box[i].append(label)
                    point_in_box = True
                    break
            if not point_in_box:
                outlier_points.append([x, y])
                outlier_labels.append(label)

    # Determine the maximum number of points across all boxes
    max_points_per_box = max([len(p) for p in points_by_box] + [len(outlier_points)])

    # Pad points and labels inside each box to ensure uniform shape
    for i in range(len(points_by_box)):
        while len(points_by_box[i]) < max_points_per_box:
            points_by_box[i].append([int(0), int(0)])  # Pad with dummy point
            labels_by_box[i].append(int(0))            # Pad with dummy label

    # Handle outlier points
    if outlier_points:
        while len(outlier_points) < max_points_per_box:
            outlier_points.append([int(0), int(0)])
            outlier_labels.append(int(0))
        points_by_box.append(outlier_points)
        labels_by_box.append(outlier_labels)
        boxes.append([int(0), int(0), int(0), int(0)])  # Add a dummy box for outliers


    return image, [points_by_box], [labels_by_box], [boxes]
 

def reverse_process_prompts(image, points_by_box, labels_by_box, boxes):
    """
    Converts box prompts and point labels back to the original prompt format
    
    Args:
        image: Input image
        points_by_box: List of point coordinates for each box
        labels_by_box: List of point labels for each box
        boxes: List of bounding boxes
    
    Returns:
        Dictionary of prompts
    """
    prompts = {"image":  next(iter(image.values())), "points": []}
    
    # Add box prompts
    for box in boxes[0]:
        x1, y1, x2, y2 = box
        prompts["points"].append([x1, y1, 2, x2, y2, 3])  # Box type marker
    
    # Add point prompts
    for box_points, box_labels, box in zip(points_by_box[0], labels_by_box[0], boxes[0]):
        for point, label in zip(box_points, box_labels):
          #  if point != [0, 0]:  # Skip dummy points
                x, y = point
                prompts["points"].append([x, y, label, 0, 0, 4])  # Point type marker
    
    return prompts

def generate_footer(metadata):

    # Retrieve git info
    git_commit = subprocess.run(
        ['git', 'log', '-1', '--format=%H'],
        stdout=subprocess.PIPE,
        text=True,
        cwd=main_path,
        ).stdout.strip()
    git_branch = subprocess.run(
        ['git', 'rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}'],
        stdout=subprocess.PIPE,
        text=True,
        cwd=main_path,
        ).stdout.strip()
    git_branch = git_branch.split('/')[-1]  # remove the "origin/" part

    version_text = f"deepaas_ui/{git_branch}@{git_commit[:5]}"
    version_link = f"https://github.com/ai4os/deepaas_ui/tree/{git_commit}"

    # Get module description
    description = metadata.get('description', '')
    if not description:
        # In old modules, description was named "summary"
        description = metadata.get('summary', '')

    # Get the appropriate logo (default is "ai4eosc")
    namespace = os.getenv('NOMAD_NAMESPACE', 'ai4eosc')
    namespace = namespace if namespace in ['imagine'] else 'ai4eosc'  # other namespace don't have logo
    homepages = {
        'ai4eosc': 'https://ai4eosc.eu/',
        'imagine': 'https://www.imagine-ai.eu/',
    }
    logo = f"""
        <a href="{homepages[namespace]}">
          <div align="center">
            <img src="https://raw.githubusercontent.com/ai4os/deepaas_ui/master/_static/images/logo-{namespace}.png" width="200" />
          </div>
        </a>
    """

    # Generate the footer
    author = metadata.get('author', '')+ [
    author.get('name', '') for author in metadata.get('model_info', {}).get('authors', [])
]
    if isinstance(author, list):
        author = ', '.join(author)
    footer = f"""
        <link href="https://use.fontawesome.com/releases/v5.13.0/css/all.css" rel="stylesheet">
        <b>Author(s)</b>: {author} <br>
        <b>Description</b>: {description} <br>
        <b>UI version</b>: <a href="{version_link}"><code>{version_text}</code></a> <br>
        <br><br>
        {logo}
    """
    footer = inspect.cleandoc(footer)
    return footer
if __name__=='__main__':
    prompts={}
    image=None
    box_prompts=np.array([[[  0,  31,  18,  55],
            [  0,  89,  16, 114]]])
    point_labels= np.array([[[1, 0, 0],
            [1, 0, 0]]])
    box_prompts=np.array([[[  0,  31,  18,  55],
            [  0,  89,  16, 114]]])
    reverse_process_prompts(image, box_prompts, point_labels, box_prompts)