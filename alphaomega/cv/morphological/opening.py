import numpy as np
from alphaomega.cv.morphological.dilation import dilation_apply
from alphaomega.cv.morphological.erosion  import erosion_apply
from alphaomega.utils.exceptions          import WrongAttribute, WrongDimension

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


def opening_apply(image: np.ndarray, kernel_size: int = 3, degree: int = 1, border_type: str = "constant") -> np.ndarray:
    """
    Usage: Use this function to apply opening operation on your image.

    Inputs:
        image: The opening operation would be applied on this image.
        kernel_size: The size of the kernel which performs erosion and dilation operations.
        degree     : How many times should each erosion and dilation operations get performed?
        border_type: This parameter determines how to apply filter to the borders. Options are:
            "constant": default option.
            "reflect"
            "replicate"
            "wrap"
            "reflect_without_border"

    Returns:
        - the image with opening operation applied on it.
    """
    #checking for the correct kernel_size
    if int(kernel_size) < 1 or int(kernel_size) % 2 == 0:
        raise WrongDimension("Kernel size should be a positive odd number.")
    kernel_size = int(kernel_size)

    #checking for the correct degree
    if int(degree) < 1:
        raise WrongAttribute("Degree should be greater than one.")
    degree = int(degree)

    #chekcing for the correct border_type
    if (border_type not in ["constant", "reflect", "replicate", "wrap", "reflect_without_border"]):
        raise WrongAttribute('The only options for border are "constant", "reflect", "replicate", "wrap", and "reflect_without_border".')

    filtered_image = image.copy()

    for _ in range(degree):
        filtered_image = erosion_apply(filtered_image, kernel_size, border_type)

    for _ in range(degree):
        filtered_image = dilation_apply(filtered_image, kernel_size, border_type)

    return filtered_image