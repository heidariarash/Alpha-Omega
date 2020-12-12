import numpy as np
from alphaomega.cv.border.border_intropolation import border_intropolate_apply
from alphaomega.utils.exceptions import WrongAttribute

class MedianFilter:
    """
    Usage: Use this filter to blur your images using mean filter.
    """
    def __init__(self):
        self.__kernel_size = 3
        self.__border_type = "constant"

    def config(self, **kwargs) -> None:
        """
        Usage: Use this method to configure the paramteres of MedianFilter instantiation.

        Inputs:
            kernel_size : The size of the MedianFilter to apply.
            border_type : This parameter determines how to apply filter to the borders. Options are:
                "constant": default option.
                "reflect"
                "replicate"
                "wrap"
                "reflect_without_border"

        Returns: Nothing.
        """
        for key, value in kwargs.items():

            if key == "kernel_size":
                if (int(value) <= 1):
                    raise WrongAttribute("Kernel size cannot be less than 2.")
                elif (int(value) %2 == 0):
                    raise WrongAttribute("Please provide an odd number for kernel size.")
                else:
                    self.__kernel_size = int(value)

            elif key == "border_type":
                if (value not in ["constant", "reflect", "replicate", "wrap", "reflect_without_border"]):
                    raise WrongAttribute('The only options for border are "constant", "reflect", "replicate", "wrap", and "reflect_without_border".')
                else:
                    self.__border_type = value
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Usage: Use this method to apply the MedianFilter to your image.
        
        Inputs:
            image: The MedianFilter will be applied on this image.

        Returns:
            - The smoothed image.
        """
        #initializing different parameters
        filtered_image = np.zeros_like(image, dtype=np.int16)
        half_size      = int((self.__kernel_size-1)/2)

        #applying border to image
        image_border = border_intropolate_apply(image, half_size, self.__border_type)
        
        #finding each element of the filtered image.
        if len(image_border.shape) == 2:
            for row in range(image.shape[0]):
                for column in range(image.shape[1]):
                    filtered_image[row, column] = np.median( image_border[row : row + 2 * half_size + 1 , column :column + 2*half_size + 1], axis=(0,1))

        elif len(image_border.shape) == 3:
            for row in range(image.shape[0]):
                for column in range(image.shape[1]):
                    filtered_image[ row, column,:] = np.median( image_border[  row : row + 2 * half_size + 1 , column :column + 2*half_size + 1, :], axis=(0,1))

        return filtered_image


def meidan_filter_apply(image :np.ndarray, kernel_size: int = 3, border_type: str = "constant") -> np.ndarray:
    """
    Usage: Use this function to blur your image using mean filter.

    Inputs:
        image: The mean filter will be applied on this image.
        kernel_size : The size of the MedianFilter to apply.
        border_type : This parameter determines how to apply filter to the borders. Options are:
            "constant": default option.
            "reflect"
            "replicate"
            "wrap"
            "reflect_without_border"

    Returns: 
        - The smoothed image.
    """
    if (int(kernel_size) <= 1):
        raise WrongAttribute("Kernel size cannot be less than 2.")

    if (int(kernel_size) %2 == 0):
        raise WrongAttribute("Please provide an odd number for kernel size.")

    if (border_type not in ["constant", "reflect", "replicate", "wrap", "reflect_without_border"]):
        raise WrongAttribute('The only options for border are "constant", "reflect", "replicate", "wrap", and "reflect_without_border".')

    filtered_image = np.zeros_like(image, dtype=np.int16)
    half_size      = int((kernel_size-1)/2)

    #applying border to image
    image_border = border_intropolate_apply(image, half_size, border_type)
    
    #finding each element of the filtered image.
    if len(image_border.shape) == 2:
        for row in range(image.shape[0]):
            for column in range(image.shape[1]):
                filtered_image[row, column] = np.median( image_border[row : row + 2 * half_size + 1 , column :column + 2*half_size + 1], axis=(0,1))

    elif len(image_border.shape) == 3:
        for row in range(image.shape[0]):
            for column in range(image.shape[1]):
                filtered_image[ row, column,:] = np.median( image_border[  row : row + 2 * half_size + 1 , column :column + 2*half_size + 1, :], axis=(0,1))

    filtered_image = filtered_image.astype(np.int16)
    return filtered_image