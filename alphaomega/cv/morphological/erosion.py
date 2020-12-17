import numpy as np
from alphaomega.utils.exceptions import WrongAttribute, WrongDimension
from alphaomega.cv.border.border_intropolation import border_intropolate_apply
from alphaomega.cv.channel.channel_merge import channel_merger_apply

class Erosion:
    """
    You can use this class to erose an image.
    """
    def __init__(self):
        self.__kernel_size = 3
        self.__border_type = "constant"

    def config(self, **kwargs) -> None:
        """
        Usage: Use this method to configure the parameters of the Erosion instantiation.

        Inputs:
            kernel_size: The size of the kernel.
            border_type: This parameter determines how to apply filter to the borders. Options are:
                "constant": default option.
                "reflect"
                "replicate"
                "wrap"
                "reflect_without_border"

        Returns: Nothing.
        """
        for key, value in kwargs.items():

            if key == "kernel_size":
                if int(value) %2 == 0:
                    raise WrongAttribute("Kernel size should be an odd number.")
                self.__kernel_size = int(value)

            elif key == "border_type":
                if (value not in ["constant", "reflect", "replicate", "wrap", "reflect_without_border"]):
                    raise WrongAttribute('The only options for border are "constant", "reflect", "replicate", "wrap", and "reflect_without_border".')
                else:
                    self.__border_type = value

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Usage: Use this method to apply Dilation to your image.
        
        Inputs:
            image: The erosion will be applied on this image.

        Returns:
            - The erosed image.
        """
        #initializing different parameters
        filtered_image = np.zeros_like(image, dtype=np.int16)
        half_size      = int((self.__kernel_size-1)/2)

        #applying border to image
        image_border = border_intropolate_apply(image, half_size, self.__border_type)
        
        #finding each element of the filtered image.
        for row in range(image.shape[0]):
            for column in range(image.shape[1]):
                filtered_image[row, column] = np.min( image_border[row : row + 2 * half_size + 1 , column :column + 2*half_size + 1], axis = (0,1) )

        return filtered_image


def erosion_apply(image: np.ndarray, kernel_size: int = 3, border_type: str = "constant") -> np.ndarray:
    #checking for the correct border_type
    if (border_type not in ["constant", "reflect", "replicate", "wrap", "reflect_without_border"]):
        raise WrongDimension('The only options for border are "constant", "reflect", "replicate", "wrap", and "reflect_without_border".')

    #checking for the correct kernel_size
    if int(kernel_size) %2 == 0:
        raise WrongDimension("Kernel size should be an odd number.")

    #initializing different parameters
    filtered_image = np.zeros_like(image, dtype=np.int16)
    half_size      = int((kernel_size-1)/2)

    #applying border to image
    image_border = border_intropolate_apply(image, half_size, border_type)
    
    #finding each element of the filtered image.
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            filtered_image[row, column] = np.min( image_border[row : row + 2 * half_size + 1 , column :column + 2*half_size + 1], axis = (0,1) )

    return filtered_image