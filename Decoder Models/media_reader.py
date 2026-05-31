# =======================================================================
#                Abstract Media Reader Module
# =======================================================================
# This module will include the logic for reading various media files,
# retrieving the compressed bitstream, as well as the "ground truth"
# uncompressed image matrix. Modules inheriting from this class will 
# implement specific logic for different media formats (e.g., JPEG, MP4).
# =======================================================================

from abc import ABC, abstractmethod

class MediaReader(ABC):
    @abstractmethod
    def get_bitstream(self):
        pass
    
    @abstractmethod
    def get_media_data(self):
        pass