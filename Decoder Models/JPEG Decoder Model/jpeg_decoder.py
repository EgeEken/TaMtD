# =======================================================================
#                JPEG Decoder Module
# =======================================================================
# This module will include the logic for training, evaluating and using
# a JPEG decoder model. The model will take in the compressed bitstream
# from a JPEG file and attempt to reconstruct the original uncompressed
# image matrix.
# =======================================================================

from ..media_decoder import MediaDecoder
import numpy as np
import torch

class JPEGDecoder(MediaDecoder):
    def __init__(self):
        pass