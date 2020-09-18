import numpy as np
from alphaomega.cv.border.border_intropolation import border_intropolate_apply
from alphaomega.cv.channel.channel_merge import channel_merger_apply

class GaussianFilter:
    """
    Usage: Use this filter to blur your images using mean filter.
    """
    def __init__(self):
        """
        Usage  : The constructor of GaussianFilter Class.
        Inputs : Nothing.
        Returns: An instantiation of GaussianFilter Class.
        """
        self.__kernel_size = 3
        self.__sigma = 1
        self.__border_type = "constant"

    def config(self, **kwargs):
        """
        Usage: Use this method to configure the paramteres of GaussianFilter instantiation.

        Inputs:
            kernel_size : The size of the GaussianFilter to apply.
            sigma       : The standard deviation of Gaussian distribution.
            border_type : This parameter determines how to apply filter to the borders. Options are:
                "constant": default option.
                "reflect"
                "replicate"
                "wrap"
                "reflect_without_border"

        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "kernel_x":
                if (int(value) <= 1):
                    print("Kernel size cannot be less than 2.")
                elif (int(value) %2 == 0):
                    print("Please provide an odd number for kernel size.")
                else:
                    self.__kernel_size_x = int(value)
            elif key == "sigma":
                if value <= 0:
                    print("Sigma cannot be less than or equal to zero.")
                else:
                    self.__sigma = value
            elif key == "border_type":
                if (value not in ["constant", "reflect", "replicate", "wrap", "reflect_without_border"]):
                    print('The only options for border are "constant", "reflect", "replicate", "wrap", and "reflect_without_border".')
                else:
                    self.__border_type = value
    
    def apply(self, image):
        """
        Usage: Use this method to apply the GaussianFilter to your image.
        
        Inputs:
            image: The GaussianFilter will be applied on this image.

        Returns:
            - The smoothed image.
        """
        #initializing different parameters
        filtered_image = np.zeros_like(image, dtype=np.int16)
        half_size = int((self.__kernel_size-1)/2)

        #applying border to image
        image_border = border_intropolate_apply(image, half_size, self.__border_type)

        y, x = np.ogrid[-half_size:half_size+1, -half_size:half_size+1]
        kernel = np.exp( -(y*y + x*x) / ( 2. * self.__sigma * self.__sigma ) )
        kernel[ kernel < np.finfo(kernel.dtype).eps*kernel.max() ] = 0
        normalizer = kernel.sum()
        if normalizer != 0:
            kernel /= normalizer
        
        #finding each element of the filtered image.
        if len(image_border.shape) == 2:
            for row in range(image.shape[0]):
                for column in range(image.shape[1]):
                    # filtered_image[row, column] = np.sum( image_border[row : row + 2 * half_size + 1 , column :column + 2*half_size + 1] * kernel , axis=(0,1))
                    filtered_image[row, column] = np.sum( np.multiply(image_border[row : row + 2 * half_size + 1 , column :column + 2*half_size + 1] , kernel))

        elif len(image_border.shape) == 3:
            kernel = channel_merger_apply([kernel, kernel, kernel])
            for row in range(image.shape[0]):
                for column in range(image.shape[1]):
                    filtered_image[ row, column,:] = np.sum( image_border[  row : row + 2 * half_size + 1 , column :column + 2*half_size + 1, :] * kernel , axis=(0,1))

        return filtered_image

def mean_filter_apply(image, kernel_size, border_type = "constant"):
    """
    Usage: Use this function to blur your image using mean filter.

    Inputs:
        image: The mean filter will be applied on this image.
        kernel_size : The size of the GaussianFilter to apply.
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
        print("Kernel size cannot be less than 2.")
        return

    if (int(kernel_size) %2 == 0):
        print("Please provide an odd number for kernel size.")
        return

    if (border_type not in ["constant", "reflect", "replicate", "wrap", "reflect_without_border"]):
        print('The only options for border are "constant", "reflect", "replicate", "wrap", and "reflect_without_border".')
        return

    filtered_image = np.zeros_like(image, dtype=np.int16)
    half_size = int((kernel_size-1)/2)

    #applying border to image
    image_border = border_intropolate_apply(image, half_size, border_type)
    
    #finding each element of the filtered image.
    if len(image_border.shape) == 2:
        for row in range(image.shape[0]):
            for column in range(image.shape[1]):
                filtered_image[row, column] = np.mean( image_border[row : row + 2 * half_size + 1 , column :column + 2*half_size + 1], axis=(0,1))

    elif len(image_border.shape) == 3:
        for row in range(image.shape[0]):
            for column in range(image.shape[1]):
                filtered_image[ row, column,:] = np.mean( image_border[  row : row + 2 * half_size + 1 , column :column + 2*half_size + 1, :], axis=(0,1))

    filtered_image = filtered_image.astype(np.int16)
    return filtered_image