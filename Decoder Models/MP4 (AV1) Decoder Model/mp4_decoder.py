# =======================================================================
#                MP4 (AV1) Decoder Module
# =======================================================================
# This module will include the logic for training, evaluating and using
# an MP4 (AV1) decoder model. The model will take in the compressed bit
# stream from an MP4 file and attempt to reconstruct the original uncompressed
# video frames.
# =======================================================================

from ..media_decoder import MediaDecoder
import numpy as np
import torch

class MP4Decoder(MediaDecoder):
    def __init__(self):
        pass