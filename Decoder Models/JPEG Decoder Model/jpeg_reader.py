# =======================================================================
#                       Regular JPEG Reader
# =======================================================================
# This module will include the logic for reading JPEG files,
# retrieving the compressed bitstream, as well as the "ground truth"
# uncompressed image matrix. This will be used for training the model
# and evaluating its performance.
# =======================================================================

from ..media_reader import MediaReader
import numpy as np
from PIL import Image

class JPEGReader(MediaReader):
    def __init__(self, jpeg_file_path):
        self.jpeg_file_path = jpeg_file_path

    def get_bitstream(self):
        # Open the JPEG file and read the raw bitstream
        with open(self.jpeg_file_path, 'rb') as f:
            compressed_data = f.read()
        bitstream = np.frombuffer(compressed_data, dtype=np.uint8)
        return bitstream
    
    def get_media_data(self):
        # Use PIL to read the JPEG file and get the uncompressed image matrix
        image = Image.open(self.jpeg_file_path)
        media_data = np.array(image)
        return media_data