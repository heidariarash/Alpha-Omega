import numpy as np
from alphaomega.utils.exceptions import WrongAttribute, WrongDimension
from alphaomega.cv.border.border_intropolation import BorderIntropolation
from alphaomega.cv.channel.channel_merge import channel_merger_apply

class CustomFilter:
    """
    Use this class to apply your custom filter on the specified image.
    """
    def __init__(self):
        self.__filter      = None
        self.__border_type = "constant"
        self.__border_interpolation = BorderIntropolation()

    def config(self, **kwargs) -> None:
        """
        Usage: Use this method to configure the paramteres of CustomFilter instantiation.

        Inputs:
            kernel      : The kernel to apply.
            border_type : This parameter determines how to apply filter to the borders. Options are:
                "constant": default option.
                "reflect"
                "replicate"
                "wrap"
                "reflect_without_border"

        Returns: Nothing.
        """
        for key, value in kwargs.items():

            if key == "kernel":
                if len(value.shape) != 2:
                    raise WrongDimension("kernel should be 2 dimensional.")
                if value.shape[0] % 2 != 1:
                    raise WrongDimension("Dimensions of the kernel should be odd (not even).")
                if value.shape[1] % 2 != 1:
                    raise WrongDimension("Dimensions of the kernel should be odd (not even).")
                self.__filter = value

            elif key == "border_type":
                if (value not in ["constant", "reflect", "replicate", "wrap", "reflect_without_border"]):
                    raise WrongAttribute('The only options for border are "constant", "reflect", "replicate", "wrap", and "reflect_without_border".')
                else:
                    self.__border_type = value

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Usage: Use this method to apply the CustomFilter to your image.
        
        Inputs:
            image: The CustomFilter will be applied on this image.

        Returns:
            - The filtered image.
        """
        #checking for whether the kernel has been specified
        if self.__filter is None:
            raise Exception("Please use config method first and specify your kernel.")

        filtered_image = np.zeros_like(image, dtype=np.int16)
        half_size_x    = int((self.__filter.shape[0]) // 2)
        half_size_y    = int((self.__filter.shape[1]) // 2)

        #applying border to image
        self.__border_interpolation.config(top = half_size_x, bottom = half_size_x, right = half_size_y, left = half_size_y, border_type = self.__border_type)
        bordered_image = self.__border_interpolation.apply(image)
        print(bordered_image.shape)
        print(self.__filter.shape)
        print(half_size_x)
        print(half_size_y)

        #applying filter
        if len(bordered_image.shape) == 2:
            for row in range(image.shape[0]):
                for column in range(image.shape[1]):
                    # print(row, column)
                    filtered_image[row, column] = np.sum( np.multiply(bordered_image[row : row + 2 * half_size_x + 1 , column : column + 2 * half_size_y + 1] , self.__filter))

        elif len(bordered_image.shape) == 3:
            kernel = channel_merger_apply([self.__filter, self.__filter, self.__filter])
            for row in range(image.shape[0]):
                for column in range(image.shape[1]):
                    filtered_image[ row, column,:] = np.sum( bordered_image[  row : row + 2 * half_size_x + 1 , column :column + 2*half_size_y + 1, :] * kernel , axis=(0,1))

        return filtered_image