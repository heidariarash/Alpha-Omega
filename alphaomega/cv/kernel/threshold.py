import numpy as np
from alphaomega.cv.border.border_intropolation import border_intropolate_apply

class Threshold:
    """
    Usage: Use this class to apply thresholding to a single channel image.
    """
    def __init__(self):
        """
        Usage  : The constructor of the Threshold class.
        Inputs : Nothing.
        Returns: An instantiatino of Thresold clas.
        """
        self.__mode = "binary"
        self.__threshold = 120
        self.__max = 255

    def config(self, **kwargs):
        """
        Usage: Use this method to configure the parameters of the Threshold instantiation.
        
        Inputs:
            threshold: The value of threshold.
            max_value: Maximum value for upward operations.
            mode     : Mode of thresholding. It could be one of these options:
                "binary": default
                "binary_inverse"
                "truncate"
                "to_zero"
                "to_zero_inverse"

        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "threshold":
                self.__threshold = value
            elif key == "max_value":
                if int(value) > 0 and int(value) <= 255:
                    self.__max = value
                else:
                    print("max_value should be an integer between 1 and 255.")
            elif key == "mode":
                if value not in ["binary", "binary_inverse", "truncate", "to_zero", "to_zero_inverse"]:
                    print('mode should be one if this options: "binary", "binary_inverse", "truncate", "to_zero", and "to_zero_inverse"')
                else:
                    self.__mode = value

    def apply(self, image):
        """
        Usage: Use this method to apply thresholding on an image.

        Inputs:
            image: Thresholding will be applied on this image.

        Returns:
            - The thresholded image.
        """
        #checking for the correct dimensions of image
        if (len(image.shape) != 2):
            print("Only single channel images are accepted. Please provide a single channle image.")
            return

        thresholded = np.zeros_like(image)
        if self.__mode == "binary":
            thresholded[image > self.__threshold] = self.__max_value
            return thresholded

        if self.__mode == "binary_inverse":
            thresholded[image <= self.__threshold] = self.__max_value
            return thresholded

        if self.__mode == "truncate":
            thresholded[image > self.__threshold] = self.__threshold
            thresholded[image <= self.__threshold] = image[image <= self.__threshold]
            return thresholded

        if self.__mode == "to_zero":
            thresholded[image > self.__threshold] = image[image > self.__threshold]
            return thresholded
        
        if self.__mode == "to_zero_inverse":
            thresholded[image <= self.__threshold] = image[image <= self.__threshold]
            return thresholded


def threshold_apply(image, threshold, max_value = 255, mode = "binary"):
    """
    Usage: Use this function to apply thresholding to an image.

    Inputs:
        image    : The thresholding will be applied on this image.
        threshold: The value of threshold
        max_value: Maximum value for upward operations.
        mode     : Mode of thresholding. It could be one of these options:
            "binary": default
            "binary_inverse"
            "truncate"
            "to_zero"
            "to_zero_inverse"

    Returns:
        The thresholded image.
    """
    #checking for the correct dimensions of image
    if (len(image.shape) != 2):
        print("Only single channel images are accepted. Please provide a single channle image.")
        return

    #checking for the true value for max_value
    if int(max_value) <= 0 and int(max_value) > 255:
        print("max_value should be an integer between 1 and 255.")
        return

    thresholded = np.zeros_like(image)
    if mode == "binary":
        thresholded[image > threshold] = max_value
        return thresholded

    if mode == "binary_inverse":
        thresholded[image <= threshold] = max_value
        return thresholded

    if mode == "truncate":
        thresholded[image > threshold] = threshold
        thresholded[image <= threshold] = image[image <= threshold]
        return thresholded

    if mode == "to_zero":
        thresholded[image > threshold] = image[image > threshold]
        return thresholded
    
    if mode == "to_zero_inverse":
        thresholded[image <= threshold] = image[image <= threshold]
        return thresholded

    print('mode should be one if this options: "binary", "binary_inverse", "truncate", "to_zero", and "to_zero_inverse"')


class AdaptiveThreshold:
    """
    Usage: Use this class to apply adaptive thresholding to an image.
    """
    def __init__(self):
        """
        Usage  : The constructor of AdaptiveThreshold class.
        Inputs : Nothing.
        Returns: An instantiation of AdaptiveThreshold class.
        """
        self.__max_value = 255
        self.__method = "mean"
        self.__mode = "binary"
        self.__block_size = 3
        self.__constant = 0

    def config(self, **kwargs):
        """
        Usage: Use this method to configure the parameters of AdaptiveThreshold instantiation.

        Inputs:
            max_value: Maximum value for upward operations.
            mode     : Mode of thresholding. It could be one of these options:
                "binary": default
                "binary_inverse"
                "truncate"
                "to_zero"
                "to_zero_inverse"
            method    : The method of calculating threshold. It could one of these two options: "mean" or "gaussian"
            block_size: The block size in which the threshold is calcualted.
            contant   : The constant for calculating the threshold.
        
        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "max_value":
                if int(value) > 0 and int(value) <= 255:
                    self.__max = value
                else:
                    print("max_value should be an integer between 1 and 255.")
            elif key == "mode":
                if value not in ["binary", "binary_inverse", "truncate", "to_zero", "to_zero_inverse"]:
                    print('mode should be one if this options: "binary", "binary_inverse", "truncate", "to_zero", and "to_zero_inverse"')
                else:
                    self.__mode = value
            elif key == "method":
                if value not in ["mean", "gaussian"]:
                    print("method should be one of these two options: 'mean', and 'gaussian'.")
                else:
                    self.__method = value
            elif key == "block_size":
                if value%2 != 1:
                    print("please provide an integer odd number as block_size.")
                else:
                    self.__block_size = value
            elif key == "constant":
                self.__constant = value

    def apply(self, image):
        """
        Usage: Use this method to apply adaptive thresholding to an image.

        Inputs:
            image: Apaptive thresholding will be applied on this image.

        Returns:
            - The thresholded image.
        """
        #checking for the correct shape of the image
        if len(image.shape) != 2:
            print("Only single channel images are acceptable.")
            return

        half_size = int((self.__block_size-1)/2)
        image_border = border_intropolate_apply(image, half_size, "reflect_without_border")

        if self.__method == "mean":
            kernel = np.ones((self.__block_size, self.__block_size)) / (self.__block_size ** 2)
        
        elif self.__method == "gaussian":
            y, x = np.ogrid[-half_size:half_size+1, -half_size:half_size+1]
            kernel = np.exp( -(y*y + x*x) / ( 2 ) )
            kernel[ kernel < np.finfo(kernel.dtype).eps*kernel.max() ] = 0
            normalizer = kernel.sum()
            if normalizer != 0:
                kernel /= normalizer

        thresholded = np.zeros_like(image, dtype=np.int16)
        threshold = np.zeros_like(image)
        for row in range(image.shape[0]):
            for column in range(image.shape[1]):
                threshold[row, column] = (np.sum( np.multiply(image_border[row : row + self.__block_size , column :column + self.__block_size] , kernel)) - self.__constant)

        if self.__mode == "binary":
            thresholded[image > threshold] = self.__max_value
            return thresholded

        if self.__mode == "binary_inverse":
            thresholded[image <= threshold] = self.__max_value
            return thresholded

        if self.__mode == "truncate":
            thresholded[image > threshold] = threshold[image > threshold]
            thresholded[image <= threshold] = image[image <= threshold]
            return thresholded

        if self.__mode == "to_zero":
            thresholded[image > threshold] = image[image > threshold]
            return thresholded
        
        thresholded[image <= threshold] = image[image <= threshold]
        return thresholded


def adaptive_threshold_apply(image, max_value = 255, mode = "binary", method = "mean", block_size = 3, constant = 0):
    """
    Usage: Use this function to apply adaptive thresholding to an image.

    Inputs:
        image    : Thresholding will be applied on this image.
        max_value: Maximum value for upward operations.
        mode     : Mode of thresholding. It could be one of these options:
            "binary": default
            "binary_inverse"
            "truncate"
            "to_zero"
            "to_zero_inverse"
        method    : The method of calculating threshold. It could one of these two options: "mean" or "gaussian"
        block_size: The block size in which the threshold is calcualted.
        contant   : The constant for calculating the threshold.

    Returns:
        - The thresholded image.
    """
    #checking if max_value is inside the range 1 to 255
    if (int(max_value <= 0) or int(max_value) > 255):
        print("max_value should be an integer between 1 and 255.")
        return

    #checking for the true value for block_size
    if (block_size%2 != 1):
        print("please provide an integer odd number as block_size.")
        return

    #checking for the true parameter for mode
    if mode not in ["binary", "binary_inverse", "truncate", "to_zero", "to_zero_inverse"]:
            print('mode should be one if this options: "binary", "binary_inverse", "truncate", "to_zero", and "to_zero_inverse"')
            return

    #checking for the true shape of the image
    if len(image.shape) != 2:
        print("Only single channel images are acceptable.")
        return

    half_size = int((block_size-1)/2)
    image_border = border_intropolate_apply(image, half_size, "reflect_without_border")

    if method == "mean":
        kernel = np.ones((block_size, block_size)) / (block_size ** 2)
    
    elif method == "gaussian":
        y, x = np.ogrid[-half_size:half_size+1, -half_size:half_size+1]
        kernel = np.exp( -(y*y + x*x) / ( 2 ) )
        kernel[ kernel < np.finfo(kernel.dtype).eps*kernel.max() ] = 0
        normalizer = kernel.sum()
        if normalizer != 0:
            kernel /= normalizer

    else:
        print("method should be one of these two options: 'mean' or 'gaussian'.")
        return

    thresholded = np.zeros_like(image, dtype=np.int16)
    threshold = np.zeros_like(image)
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            threshold[row, column] = (np.sum( np.multiply(image_border[row : row + block_size , column :column + block_size] , kernel)) - constant)

    if mode == "binary":
        thresholded[image > threshold] = max_value
        return thresholded

    if mode == "binary_inverse":
        thresholded[image <= threshold] = max_value
        return thresholded

    if mode == "truncate":
        thresholded[image > threshold] = threshold[image > threshold]
        thresholded[image <= threshold] = image[image <= threshold]
        return thresholded

    if mode == "to_zero":
        thresholded[image > threshold] = image[image > threshold]
        return thresholded
    
    thresholded[image <= threshold] = image[image <= threshold]
    return thresholded