import numpy as np
from alphaomega.cv.kernel.filter import custom_filter_apply
from alphaomega.utils.exceptions import WrongAttribute, WrongDimension

class Sobel:
    """
    Use this class to apply Sobel filter to your image and find the edges.
    """
    def __init__(self):
        self.__type        = "xy"
        self.__kernel_size = 3
        self.__border_type = "constant"

    def config(self, **kwargs) -> None:
        """
        Usage: Use this method to configure the parameters of the Sobel instantiation.

        Inputs:
            kernel_size: The size of the sobel filter.
            type       : The type of the filter. It could be one of these three options:
                "xy": performs both x and y Sobel filters (default).
                "x" : performs only x Sobel filter.
                "y" : performs only y Sobel filter.
            border_type  : This parameter determines how to apply filter to the borders. Options are:
                "constant": default option.
                "reflect"
                "replicate"
                "wrap"
                "reflect_without_border"

        Returns: Nothing.
        """
        for key, value in kwargs.items():

            if key == "kernel_size":
                if int(value) < 3 or int(value) % 2 == 0:
                    raise WrongDimension("Kernel size should be an odd number greater or equal to three.")
                self.__kernel_size = int(value)

            elif key == "type":
                if value not in ["xy", "x", "y"]:
                    raise WrongAttribute("type could be only 'x', 'y', and 'xy'.")
                self.__type = value

            elif key == "border_type":
                if (value not in ["constant", "reflect", "replicate", "wrap", "reflect_without_border"]):
                    raise WrongAttribute('The only options for border are "constant", "reflect", "replicate", "wrap", and "reflect_without_border".')
                else:
                    self.__border_type = value

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Usage: Use this method to apply sobel filter to your image.

        Inputs:
            image: The sobel filter will be applied on this image.

        Returns:
            - The output of the sobel filter.
        """
        sobel_x = np.zeros((self.__kernel_size, self.__kernel_size))
        p = [(j,i) for j in range(self.__kernel_size) 
            for i in range(self.__kernel_size) 
            if not (i == (self.__kernel_size -1)/2. and j == (self.__kernel_size -1)/2.)]

        for j, i in p:
            j_ = int(j - (self.__kernel_size -1)/2.)
            i_ = int(i - (self.__kernel_size -1)/2.)
            sobel_x[j,i] = (i_ )/float(i_*i_ + j_*j_)

        sobel_y= np.zeros((self.__kernel_size, self.__kernel_size))
        p = [(j,i) for j in range(self.__kernel_size) 
            for i in range(self.__kernel_size) 
            if not (i == (self.__kernel_size -1)/2. and j == (self.__kernel_size -1)/2.)]

        for j, i in p:
            j_ = int(j - (self.__kernel_size -1)/2.)
            i_ = int(i - (self.__kernel_size -1)/2.)
            sobel_y[j,i] = (j_)/float(i_*i_ + j_*j_)

        if self.__type == "x":
            return custom_filter_apply(image, sobel_x, self.__border_type)

        if self.__type == "y":
            return custom_filter_apply(image, sobel_y, self.__border_type)
        
        answer  = custom_filter_apply(image, sobel_x, self.__border_type)
        answer += custom_filter_apply(image, sobel_y, self.__border_type)
        return answer
        