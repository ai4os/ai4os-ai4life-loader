import ast
from pathlib import Path
from PIL import Image
from urllib.parse import urljoin
import click
import gradio as gr
import requests
import utils
from io import BytesIO
import gradio_image_prompter as gr_ext


gr.close_all()
GRADIO_SERVER = "0.0.0.0"


@click.command()
@click.option(
    "--api_url",
    default="http://0.0.0.0:5000/",
    help="URL of the DEEPaaS API",
)
@click.option("--ui_port", default=80, help="URL of the deployed UI")
def main(api_url, ui_port):
    """
    This module contains several functions that make calls to the deep API.
    Args:
        api_url: URL of the deep api.
        ui_port: port for GUI

    """
    interfaces = []
    sess = requests.Session()
    r = sess.get(urljoin(api_url, "swagger.json"))
    specs = r.json()

    pred_paths = [
        p for p in specs["paths"].keys() if p.endswith("predict/")
    ]

    p = next(
        (p for p in pred_paths if "ai4life" in p), None
    )  # path to the fasterrcnn_pytorch_api
    if p is not None:
        print("Path for ai4life model:", p)
    else:
        print("ai4life model not found in the paths.")

    api_inp = specs["paths"][p]["post"]["parameters"]
    api_out = specs["paths"][p]["post"]["produces"]
    mimes = specs["paths"][p]["post"]["produces"]
    model_name = next(
        (
            (
                inp.get("enum")[0]
                if "enum" in inp
                else ast.literal_eval(inp["default"])[0]
            )
            for inp in api_inp
            if inp.get("name") == "model_name"
            and ("enum" in inp or "default" in inp)
        ),
        None,
    )

    r = sess.get(urljoin(api_url, f"{Path(p).parent}"))
    print(
        f"the path to the get metadata is {api_url}/{Path(p).parent}/"
    )
    _, inp_names, inp_types = utils.api2gr_inputs(api_inp)
    meta = r.json()

    def make_request(params, accept):
        """
         Receives parameters from the GUI and returns the JSON file resulting from calling the deep API.

         Args:
          params:  the parameters entered by the user in the GUI.
          accept: specifies the format of the resulting request.

        Returns:
          rc: the content of the response from the API.
        """
        # convert GUI input to DEEPaaS webargs

        if len(params) == 1:
            params = dict(zip(inp_names, (model_name, params[0])))

            params, files = utils.gr2api_input(params, inp_types)
        else:
            image, point_coords, point_labels, boxes = (
                utils.process_prompts(params[1])
            )

            params = (
                image,
                boxes,
                params[2],
                params[3],
                point_coords,
                point_labels,
            )
            params = dict(zip(inp_names, ((model_name,) + params)))

            params, files = utils.gr2api_input(params, inp_types)

        r = sess.post(
            urljoin(api_url, p),
            headers={"accept": accept},
            params=params,
            files=files,
            verify=False,
        )

        if r.status_code != 200:
            raise Exception(f"HTML {r.status_code} eror: {r}")

        return r.content

    def api_call(*args, mime: str, **kwargs):
        """
        Receives the converted parameters from the GUI as arguments for the deep API,
           calls the API with the specified parameters, and loads the JSON file from the response.

        Args:
             args: The converted parameters from the GUI to deep API arguments.

        Returns:
          rc: A JSON file containing a list of predictions for each input file.
        """

        buffer = make_request(args, mime)
        if mime.startswith("image/"):
            image = Image.open(BytesIO(buffer))
            return image
        else:
            return buffer

    for mime in mimes:
        if mime == "*/*":
            continue
        print(f"Processing MIME: {mime}")
        with gr.Blocks() as interface:
            gr_inp = []
            if len(api_inp) == 3:
                gr_inp.append(
                    gr.File(type="filepath", label="Input an Image")
                )
            else:
                npy_upload = gr.File(
                    type="filepath",
                    label="Upload an NPY image file or png as input. The NPY will be converted to a PNG file,"
                    " allowing you to draw bounding boxes and points on it."
                    "Click to add points (left: foreground, right: "
                    "background) or drag to create boxes. If you are using the "
                    "example file, you still need to draw bounding boxes and points "
                    "shown on the image.",
                    file_types=[".npy"],
                )

                input_image = gr_ext.ImagePrompter(
                    show_label=False,
                    #  container=False,
                    label="Input Image: Click to add points (left: foreground, right: "
                    "background) or drag to create boxes. If you are using the "
                    "example file, you still need to draw bounding boxes and points "
                    "shown on the image.",
                    interactive=True,
                    type="filepath",
                )

                mask_prompts = gr.File(
                    type="filepath",
                    label="Mask prompts (optional): npy or an image file. SAM will take"
                    " this binary input mask as a hint or starting point"
                    " and try to refine the segmentation around the "
                    "provided mask area.",
                )
                embeddings = gr.File(
                    type="filepath",
                    label="Embeddings (optional): The embeddings represent the image features"
                    " that SAM uses for segmentation. It can be generated by"
                    " the image encoder part of SAM. "
                    "Embedding input, with a fixed shape of [1, 256, 64, 64] "
                    "and float32 type."
                    "provided mask area.",
                )

                gr_inp.extend(
                    [
                        npy_upload,
                        input_image,
                        mask_prompts,
                        embeddings,
                    ]
                )

                # Move the event binding here
                npy_upload.upload(
                    fn=utils.load_npy_image,
                    inputs=npy_upload,
                    outputs=input_image,
                )

            if mime == "application/json":
                output = gr.JSON()
            elif mime.startswith("image/"):
                # gr_out = gr.Image(type='filepath')
                output = gr.Image(
                    type="filepath", label="Image with segmentation"
                )
            examples = utils.get_examples(model_name, api_inp)

            project_description = meta.get("model_info", {}).get(
                "description"
            )

            # Define the Gradio interface
            gr.Interface(
                fn=lambda *args, mime=mime: api_call(
                    *args, mime=mime
                ),
                inputs=gr_inp,
                outputs=output,
                title=model_name,
                description=project_description,
                examples=[examples],
                article=utils.generate_footer(meta),
                theme=gr.themes.Default(
                    primary_hue=gr.themes.colors.cyan,
                ),
            )
            interfaces.append(interface)
    if len(interfaces) > 1:
        demo = gr.TabbedInterface(
            interface_list=interfaces,
            tab_names=[mime for mime in mimes if mime != "*/*"],
        )
    demo.launch(
        share=False,
        show_error=True,
        server_name=GRADIO_SERVER,
        server_port=ui_port,
    )


if __name__ == "__main__":

    main()
