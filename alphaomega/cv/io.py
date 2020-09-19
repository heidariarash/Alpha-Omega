import numpy as np
from PIL import Image

class ImageReader:
    """
    Usage: Use this class to read an image. This class is only a wrapper for PILLOW library.
    """
    def __init__(self):
        """
        Usage  : The constructor of ImageReader class.
        Inputs : Nothing.
        Returns: An instantiationn of the ImageReader class.
        """
        self.__image = None

    def apply(self, path):
        """
        Usage: Use this method to read an image.

        Inputs:
            path: The path of the image.

        Returns:
            - A numpy array containing the image.
        """
        self.__image = Image.open(path)
        return np.array(self.__image)


def image_reader_apply(path):
    """
    Usage: Use this functino to read an image. This function is just a wrapper for PILLOW library.

    Inputs:
        path: The path of the image.

    Returns:
        - A numpy array containing the image.
    """
    image = Image.open(path)
    return np.array(image)