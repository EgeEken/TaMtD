# =======================================================================
#                Abstract Media Decoder Model Module
# =======================================================================
# This module will define the abstract media decoder model, 
# which will be used as a base class for specific decoder models 
# (e.g., JPEGDecoder). This will include the abstract methods that 
# any decoder model must implement, such as the decoding process, 
# as well as any common utilities that all models will share.
# =======================================================================

from abc import ABC, abstractmethod

class MediaDecoder(ABC):
    @abstractmethod
    def decode(self, bitstream):
        pass