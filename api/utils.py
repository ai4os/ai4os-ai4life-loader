"""Utilities module for API endpoints and methods.
This module is used to define API utilities and helper functions. You can
use and edit any of the defined functions to improve or add methods to
your API.

The module shows simple but efficient example utilities. However, you may
need to modify them for your needs.
"""
import logging
import subprocess
import sys
import json
import matplotlib.pyplot as plt
import io
from subprocess import TimeoutExpired
import numpy as np

from . import config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def ls_dirs(path):
    """Utility to return a list of directories available in `path` folder.

    Arguments:
        path -- Directory path to scan for folders.

    Returns:
        A list of strings for found subdirectories.
    """
    logger.debug("Scanning directories at: %s", path)
    #dirscan = (x.name for x in path.iterdir() if x.is_dir())
    with open(path, 'r') as file:
        models_data = json.load(file)
     #   print(models_data)

    
    return models_data


def ls_files(path, pattern):
    """Utility to return a list of files available in `path` folder.

    Arguments:
        path -- Directory path to scan.
        pattern -- File pattern to filter found files. See glob.glob() python.

    Returns:
        A list of strings for files found according to the pattern.
    """
    logger.debug("Scanning for %s files at: %s", pattern, path)
    dirscan = (x.name for x in path.glob(pattern))
    return sorted(dirscan)


def copy_remote(frompath, topath, timeout=600):
    """Copies remote (e.g. NextCloud) folder in your local deployment or
    vice versa for example:
        - `copy_remote('rshare:/data/images', '/srv/myapp/data/images')`

    Arguments:
        frompath -- Source folder to be copied.
        topath -- Destination folder.
        timeout -- Timeout in seconds for the copy command.

    Returns:
        A tuple with stdout and stderr from the command.
    """
    with subprocess.Popen(
        args=["rclone", "copy", f"{frompath}", f"{topath}"],
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.PIPE,  # Capture stderr
        text=True,  # Return strings rather than bytes
    ) as process:
        try:
            outs, errs = process.communicate(None, timeout)
            if errs:
                raise RuntimeError(errs)
        except TimeoutExpired:
            logger.error("Timeout when copying from/to remote directory.")
            process.kill()
            outs, errs = process.communicate()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error copying from/to remote directory\n %s", exc)
            process.kill()
            outs, errs = process.communicate()
    return outs, errs


def generate_arguments(schema):
    """Function to generate arguments for DEEPaaS using schemas."""
    def arguments_function():  # fmt: skip
        logger.debug("Web args schema: %s", schema)
        return schema().fields
    return arguments_function


def predict_arguments(schema):
    """Decorator to inject schema as arguments to call predictions."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_predict_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema


def train_arguments(schema):
    """Decorator to inject schema as arguments to perform training."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_train_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema


# Function to display input and prediction output images
def show_images(input_array, output_):

    buffer = io.BytesIO()
    # Check for the number of channels to enable display
    input_array = np.squeeze(input_array)
    if len(input_array.shape) > 2:
        input_array = input_array[0]

    output_array = next(iter(output_.values()))

    # Check for the number of channels to enable display
    output_array = np.squeeze(output_array)
    if len(output_array.shape) > 2:
        output_array = output_array[0]

    fig = plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Input")
    ax1.axis("off")
    plt.imshow(input_array)
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Prediction")
    ax2.axis("off")
    ax2.imshow(output_array)
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)
    return buffer

def show_masks_on_image(
    input_array, masks, scores, boxes=None, show_boxs=False
):
    """Displays masks on an image with optional bounding boxes.

    Args:
        raw_image: The original image as a NumPy array.
        masks: A tensor of masks with shape (N, H, W) or (N, 1, H, W).
        scores: A tensor of confidence scores with shape (N,) or (1, N).
        boxes: A tensor of bounding boxes with shape (N, 4).
        show_boxs: Whether to display bounding boxes. Defaults to False.

    Returns:
        A BytesIO buffer containing the image with masks and boxes.
    """

    _, nub_prediction, nb_masks = scores.shape
    print(f"Number of PREDICTION: {nub_prediction}")
    print(f"Number of masks: {masks.shape}")

    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if scores.shape[0] == 1:
        scores = scores.squeeze()

    fig, axes = plt.subplots(
        nb_masks, nub_prediction, figsize=(15, 15)
    )

    buffer = io.BytesIO()  # Create an in-memory buffer

    for j in range(nb_masks):
        for i in range(nub_prediction):
            if nb_masks == 1 and nub_prediction == 1:
                ax = axes
                mask = masks.cpu().detach()
                score = scores
            elif nb_masks > 1:
                ax = axes[j, i] if nub_prediction > 1 else axes[j]
                mask = (
                    masks[i, j].cpu().detach()
                    if nub_prediction > 1
                    else masks[j].cpu().detach()
                )
                score = (
                    scores[i, j].item()
                    if nub_prediction > 1
                    else scores[j]
                )

            else:
                ax = axes[i]
                mask = masks[i].cpu().detach()
                score = scores[i].item()

            ax.imshow(input_array)
            show_mask(mask, ax)
            ax.title.set_text(
                f"Prediction {i+1}, Mask {j+1}, Score: {score:.3f}"
            )
            ax.axis("off")
            if show_boxs:
                show_box(boxes[i], ax)

    fig.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)
    return buffer



def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate(
            [np.random.random(3), np.array([0.6])], axis=0
        )
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]

    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            w,
            h,
            edgecolor="green",
            facecolor=(0, 0, 0, 0),
            lw=2,
        )
    )


def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    # plt.axis("on")

def output_png(sample, output_):
    
    input_array = sample

   # if len(output_) == 1:
    output__={}
    if len(output_) > 1:
         output__['masks'] = np.array(output_.get('masks'))
    else:
        output__=  output_   
    return show_images(input_array, output__)
    #else:
       # masks = output_.get('masks')
       # scores = output_.get('scores')
      #  return show_masks_on_image(
      #      input_array, masks, scores, boxes=None, show_boxs=False
      #  )

