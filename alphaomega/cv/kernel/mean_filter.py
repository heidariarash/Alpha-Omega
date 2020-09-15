import numpy as np

class MeanFilter:
    """
    Usage: Use this filter to blur your images using mean filter.
    """
    def __init__(self):
        """
        Usage  : The constructor of MeanFilter Class.
        Inputs : Nothing.
        Returns: An instantiation of MeanFilter Class.
        """
        self.__kernel_size = 3
        self.__border = "constant"

    def config(self, **kwargs):
        """
        Usage: Use this method to configure the paramteres of MeanFilter instantiation.

        Inputs:
            kernel_size : The size of the MeanFilter to apply.
            border      : This parameter determines how to apply filter to the borders. Options are:
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
                    print("Kernel size cannot be less than 2.")
                elif (int(value) %2 == 0):
                    print("Please provide an odd number for kernel size.")
                else:
                    self.__kernel_size = int(value)
            elif key == "border":
                if (value not in ["constant", "reflect", "replicate", "wrap", "reflect_without_border"]):
                    print('The only options for border are "constant", "reflect", "replicate", "wrap", and "reflect_without_border".')
                else:
                    self.__border = value

    def border_config(self, image, border, num):
        if border == "constant":
            image = np.concatenate((np.zeros((image.shape[0], num)), image), axis = 1)
            image = np.concatenate((image, np.zeros((image.shape[0], num))), axis = 1)
            image = np.concatenate((np.zeros((num, image.shape[1])), image), axis = 0)
            image = np.concatenate((image, np.zeros((num, image.shape[1]))), axis = 0)
            return image

        # if border == "reflect":
        # if border == "wrap":
        # if border == "replicate":
        # if border == "reflect_without_border":
    
    def apply(self, image):
        """
        Usage: Use this method to apply the MeanFilter to your image.
        
        Inputs:
            image: The MeanFilter will be applied on this image.

        Returns:
            - The smoothed image.
        """
        #initializing different parameters
        filtered_image = np.zeros_like(image, dtype=np.float)
        coeficient = 1/(self.__kernel_size * self.__kernel_size)
        half_size = int((self.__kernel_size-1)/2)

        #applying border to image
        image_border = self.border_config(image, self.__border, half_size)
        
        #finding each element of the filtered image.
        for row in range(image.shape[0]):
            for column in range(image.shape[1]):
                filtered_image[row, column] = coeficient * np.sum( image_border[row : row + 2 * half_size + 1 , column :column + 2*half_size + 1])
        
        return filtered_image