import numpy as np
import requests
from io import BytesIO

# URL of the .npy file
url = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/greedy-whale/1/files/point_labels.npy"

# Fetch the file from the URL
response = requests.get(url)
response.raise_for_status()  # Raise an error if the request failed

# Load the NumPy array from the content
output_array = np.load(BytesIO(response.content))

# Display the loaded array
print(output_array.shape)