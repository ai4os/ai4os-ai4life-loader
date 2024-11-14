"""Module for defining custom API response parsers and content types.
This module is used by the API server to convert the output of the requested
method into the desired format. 

The module shows simple but efficient example functions. However, you may
need to modify them for your needs.
"""
import io
import logging

from fpdf import FPDF
import numpy as np
import xarray as xr
from . import config, utils

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


# EXAMPLE of json_response parser function
# = HAVE TO MODIFY FOR YOUR NEEDS =
def json_response(result, output_ids,sample, **options):
    """Converts the prediction results into json return format.

    Arguments:
        result -- Result value from call, expected either dict or str
          (see https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/stable/user/v2-api.html).
        options -- Not used, added for illustration purpose.

    Raises:
        RuntimeError: Unsupported response type.

    Returns:
        Converted result into json dictionary format.
    """
    logger.debug("Response result type: %d", type(result))
    logger.debug("Response result: %d", result)
    logger.debug("Response options: %d", options)

    try:
        output_={}
        for id in output_ids: 
            #print(f'the id is {id}')
            if id=='embeddings':
                pass
            else:
                output_array = result.members[id].data
                #print(f'the output_array is {output_array}')
                if isinstance(output_array, xr.DataArray):
                    #print('the output is xr array')
                    output_array = output_array.values   # Add directly if not numpy type
                #print(f'the size of the output array is {output_array.shape}')
                output_[id] = np.squeeze(output_array).tolist()
            
        print(f'Final output_: {output_.keys()}')
        return output_

 
    except Exception as err:  # TODO: Fix to specific exception
        logger.warning("Error converting result to json: %s", err)
        raise RuntimeError("Unsupported response type") from err

def png_response(result, output_ids, sample,**options):
    logger.debug("Response result type: %d", type(result))
    logger.debug("Response result: %d",output_ids)
    output_= json_response(result, output_ids, sample, **options)
     
    
    return utils.output_png(sample, output_) 
# EXAMPLE of pdf_response parser function
# = HAVE TO MODIFY FOR YOUR NEEDS =
def pdf_response(result, **options):
    """Converts the prediction results into pdf return format.

    Arguments:
        result -- Result value from call, expected either dict or str
          (see https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/stable/user/v2-api.html).
        options -- Not used, added for illustration purpose.

    Raises:
        RuntimeError: Unsupported response type.

    Returns:
        Converted result into pdf buffer format.
    """
    logger.debug("Response result type: %d", type(result))
    logger.debug("Response result: %d", result)
    logger.debug("Response options: %d", options)
    try:
        # 1. create BytesIO object
        buffer = io.BytesIO()
        buffer.name = "output.pdf"
        # 2. write the output of the method in the buffer
        #    For the proper PDF document, you may use:
        #    * matplotlib for images
        #    * fPDF2 for text documents (pip3 install fPDF2)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        # in this EXAMPLE we also add input parameters
        print_out = {"input": str(options), "predictions": str(result)}
        pdf.multi_cell(w=0, txt=str(print_out).replace(",", ",\n"))
        pdf_output = pdf.output(dest="S")
        buffer.write(pdf_output)
        # 3. rewind buffer to the beginning
        buffer.seek(0)
        return buffer
    except Exception as err:  # TODO: Fix to specific exception
        logger.warning("Error converting result to pdf: %s", err)
        raise RuntimeError("Unsupported response type") from err


content_types = {
    "application/json": json_response,
    "image/png": png_response,
}
