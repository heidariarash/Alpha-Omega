import numpy as np
from alphaomega.cv.morphological.dilation import dilation_apply
from alphaomega.cv.morphological.erosion import erosion_apply
from alphaomega.utils.exceptions import WrongAttribute, WrongDimension

class Opening:
    """
    Use this class to perform opening operation on an image.
    """
    def __init__(self):
        self.__kernel_size = 3
        self.__degree      = 1
        self.__border_type = "constant"

    def config(self, **kwargs) -> None:
        """
        Usage: Use this method to configure the parameters of the Opening instantiation.

        Inputs:
            kernel_size: The size of the kernel which performs erosion and dilation operations.
            degree     : How many times should each erosion and dilation operations get performed?
            border_type: This parameter determines how to apply filter to the borders. Options are:
                "constant": default option.
                "reflect"
                "replicate"
                "wrap"
                "reflect_without_border"

        Returns: Nothing
        """
        for key, value in kwargs.items():

            if key == "kernel_size":
                if int(value) %2 == 0 or int(value) < 1:
                    raise WrongAttribute("Kernel size should be a positive odd number.")
                self.__kernel_size = int(value)

            elif key == "degree":
                if int(value) < 1:
                    raise WrongAttribute("Degree should be greater than one.")
                self.__degree = int(value)

            elif key == "border_type":
                if (value not in ["constant", "reflect", "replicate", "wrap", "reflect_without_border"]):
                    raise WrongAttribute('The only options for border are "constant", "reflect", "replicate", "wrap", and "reflect_without_border".')
                else:
                    self.__border_type = value
        
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Usage: Use this method to apply opening operation on your image.

        Inputs:
            image: The opening operation would be applied on this image.

        Returns:
            - the image with opening operation applied on it.
        """
        filtered_image = image.copy()

        for _ in range(self.__degree):
            filtered_image = erosion_apply(filtered_image, self.__kernel_size, self.__border_type)

        for _ in range(self.__degree):
            filtered_image = dilation_apply(filtered_image, self.__kernel_size, self.__border_type)

        return filtered_image