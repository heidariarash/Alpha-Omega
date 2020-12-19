import numpy as np
from alphaomega.cv.border.border_intropolation import border_intropolate_apply
from alphaomega.utils.exceptions               import WrongAttribute, WrongDimension

class Threshold:
    """
    Usage: Use this class to apply thresholding to a single channel image.
    """
    def __init__(self):
        self.__mode      = "binary"
        self.__threshold = 120
        self.__max       = 255

    def config(self, **kwargs) -> None:
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
                    self.__max = int(value)
                else:
                    raise WrongAttribute("max_value should be an integer between 1 and 255.")

            elif key == "mode":
                if value not in ["binary", "binary_inverse", "truncate", "to_zero", "to_zero_inverse"]:
                    raise WrongAttribute('mode should be one if this options: "binary", "binary_inverse", "truncate", "to_zero", and "to_zero_inverse"')
                else:
                    self.__mode = value

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Usage: Use this method to apply thresholding on an image.

        Inputs:
            image: Thresholding will be applied on this image.

        Returns:
            - The thresholded image.
        """
        #checking for the correct dimensions of image
        if (len(image.shape) != 2):
            raise WrongDimension("Only single channel images are accepted. Please provide a single channel image (i.e. 2 dimensional).")

        thresholded = np.zeros_like(image)
        if self.__mode == "binary":
            thresholded[image > self.__threshold] = self.__max
            return thresholded

        if self.__mode == "binary_inverse":
            thresholded[image <= self.__threshold] = self.__max
            return thresholded

        if self.__mode == "truncate":
            thresholded[image > self.__threshold]  = self.__threshold
            thresholded[image <= self.__threshold] = image[image <= self.__threshold]
            return thresholded

        if self.__mode == "to_zero":
            thresholded[image > self.__threshold] = image[image > self.__threshold]
            return thresholded
        
        if self.__mode == "to_zero_inverse":
            thresholded[image <= self.__threshold] = image[image <= self.__threshold]
            return thresholded

    def otsu(self, image: np.ndarray) -> int:
        """
        Usage: Use this method to find the best threshold value with the help of otsu algorithm.

        Inputs:
            image: The otsu algorithm will be applied on this image.

        Returns:
            - The threshold value calculated with the help of otsu algorithm.
        """
        #checking for the true shape of the image.
        if (len(image.shape) != 2):
            raise WrongDimension("Only single channel images are accepted. Please provide a single channle image.")

        #calculating maximum and minimum intensities present in the image.
        max_intensity_present = np.max(image, axis= (0,1))
        min_intensity_present = np.min(image, axis= (0,1))
        best_within_class     = np.Inf

        #if there are only one or two intensities present in the image, just return the greater one.
        if max_intensity_present == min_intensity_present or max_intensity_present-1 == min_intensity_present:
            return max_intensity_present

        #if there are only three intensities present, return the middle one.
        if max_intensity_present -2 == min_intensity_present:
            return max_intensity_present - 1

        for thresh in range(min_intensity_present + 1, max_intensity_present):
            background     = len(image[image<=thresh]) / (image.shape[0] * image.shape[1])
            foreground     = 1 - background
            background_var = np.var(image[image<=thresh])
            foreground_var = np.var(image[image>thresh])
            within_class   = foreground * foreground_var + background * background_var

            if within_class < best_within_class:
                best_within_class = within_class
                best_thresh       = thresh

        return best_thresh


def threshold_apply(image: np.ndarray, threshold: int, max_value: int = 255, mode: str = "binary") -> np.ndarray:
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
        raise WrongDimension("Only single channel images are accepted. Please provide a single channle image.")

    #checking for the true value for max_value
    if int(max_value) <= 0 and int(max_value) > 255:
        raise WrongAttribute("max_value should be an integer between 1 and 255.")
    else:
        max_value = int(max_value)

    thresholded = np.zeros_like(image)
    if mode == "binary":
        thresholded[image > threshold] = max_value
        return thresholded

    if mode == "binary_inverse":
        thresholded[image <= threshold] = max_value
        return thresholded

    if mode == "truncate":
        thresholded[image > threshold]  = threshold
        thresholded[image <= threshold] = image[image <= threshold]
        return thresholded

    if mode == "to_zero":
        thresholded[image > threshold] = image[image > threshold]
        return thresholded
    
    if mode == "to_zero_inverse":
        thresholded[image <= threshold] = image[image <= threshold]
        return thresholded

    raise WrongAttribute('mode should be one if this options: "binary", "binary_inverse", "truncate", "to_zero", and "to_zero_inverse"')
    

def threshold_otsu(image: np.ndarray) -> int:
    """
    Usage: Use this method to find the best threshold value with the help of otsu algorithm.

    Inputs:
        image: The otsu algorithm will be applied on this image.

    Returns:
        - The threshold value calculated with the help of otsu algorithm.
    """
    #checking for the true shape of the image.
    if (len(image.shape) != 2):
        raise WrongDimension("Only single channel images are accepted. Please provide a single channle image.")

    #calculating maximum and minimum intensities present in the image.
    max_intensity_present = np.max(image, axis= (0,1))
    min_intensity_present = np.min(image, axis= (0,1))
    best_within_class     = np.Inf

    #if there are only one or two intensities present in the image, just return the greater one.
    if max_intensity_present == min_intensity_present or max_intensity_present-1 == min_intensity_present:
        return max_intensity_present

    #if there are only three intensities present, return the middle one.
    if max_intensity_present -2 == min_intensity_present:
        return max_intensity_present - 1

    for thresh in range(min_intensity_present + 1, max_intensity_present):
        background     = len(image[image<=thresh]) / (image.shape[0] * image.shape[1])
        foreground     = 1 - background
        background_var = np.var(image[image<=thresh])
        foreground_var = np.var(image[image>thresh])
        within_class   = foreground * foreground_var + background * background_var

        if within_class < best_within_class:
            best_within_class = within_class
            best_thresh       = thresh

    return best_thresh



###################################################################################################################################################
###################################################################################################################################################
########################################                         NEW CLASS                      ###################################################
###################################################################################################################################################
###################################################################################################################################################


class AdaptiveThreshold:
    """
    Usage: Use this class to apply adaptive thresholding to an image.
    """
    def __init__(self):
        self.__max_value  = 255
        self.__method     = "mean"
        self.__mode       = "binary"
        self.__block_size = 3
        self.__constant   = 0

    def config(self, **kwargs) -> None:
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
                    self.__max = int(value)
                else:
                    raise WrongAttribute("max_value should be an integer between 1 and 255.")

            elif key == "mode":
                if value not in ["binary", "binary_inverse", "truncate", "to_zero", "to_zero_inverse"]:
                    raise WrongAttribute('mode should be one if this options: "binary", "binary_inverse", "truncate", "to_zero", and "to_zero_inverse"')
                else:
                    self.__mode = value

            elif key == "method":
                if value not in ["mean", "gaussian"]:
                    raise WrongAttribute("method should be one of these two options: 'mean', and 'gaussian'.")
                else:
                    self.__method = value

            elif key == "block_size":
                if value%2 != 1:
                    raise WrongAttribute("please provide an integer odd number as block_size.")
                else:
                    self.__block_size = value

            elif key == "constant":
                self.__constant = int(value)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Usage: Use this method to apply adaptive thresholding to an image.

        Inputs:
            image: Apaptive thresholding will be applied on this image.

        Returns:
            - The thresholded image.
        """
        #checking for the correct shape of the image
        if len(image.shape) != 2:
            raise WrongDimension("Only single channel images are acceptable.")

        half_size    = int((self.__block_size-1)/2)
        image_border = border_intropolate_apply(image, half_size, "reflect_without_border")

        if self.__method == "mean":
            kernel = np.ones((self.__block_size, self.__block_size)) / (self.__block_size ** 2)
        
        elif self.__method == "gaussian":
            y, x   = np.ogrid[-half_size:half_size+1, -half_size:half_size+1]
            kernel = np.exp( -(y*y + x*x) / ( 2 ) )
            kernel[ kernel < np.finfo(kernel.dtype).eps*kernel.max() ] = 0
            normalizer = kernel.sum()
            if normalizer != 0:
                kernel /= normalizer

        thresholded = np.zeros_like(image, dtype=np.int16)
        threshold   = np.zeros_like(image)
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
            thresholded[image > threshold]  = threshold[image > threshold]
            thresholded[image <= threshold] = image[image <= threshold]
            return thresholded

        if self.__mode == "to_zero":
            thresholded[image > threshold] = image[image > threshold]
            return thresholded
        
        thresholded[image <= threshold] = image[image <= threshold]
        return thresholded


def adaptive_threshold_apply(image: np.ndarray, max_value: int = 255, mode: str = "binary", method: str = "mean", block_size: int = 3, constant: int = 0) -> np.ndarray:
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
        raise WrongAttribute("max_value should be an integer between 1 and 255.")
    else:
        max_value = int(max_value)

    #checking for the true value for block_size
    if (block_size%2 != 1):
        raise WrongAttribute("please provide an integer odd number as block_size.")

    #checking for the true parameter for mode
    if mode not in ["binary", "binary_inverse", "truncate", "to_zero", "to_zero_inverse"]:
        raise WrongAttribute('mode should be one if this options: "binary", "binary_inverse", "truncate", "to_zero", and "to_zero_inverse"')

    #checking for the true shape of the image
    if len(image.shape) != 2:
        raise WrongDimension("Only single channel images are acceptable.")

    half_size    = int((block_size-1)/2)
    image_border = border_intropolate_apply(image, half_size, "reflect_without_border")

    if method == "mean":
        kernel = np.ones((block_size, block_size)) / (block_size ** 2)
    
    elif method == "gaussian":
        y, x   = np.ogrid[-half_size:half_size+1, -half_size:half_size+1]
        kernel = np.exp( -(y*y + x*x) / ( 2 ) )
        kernel[ kernel < np.finfo(kernel.dtype).eps*kernel.max() ] = 0
        normalizer = kernel.sum()
        if normalizer != 0:
            kernel /= normalizer

    else:
        raise WrongAttribute("method should be one of these two options: 'mean' or 'gaussian'.")
        return

    thresholded = np.zeros_like(image, dtype=np.int16)
    threshold   = np.zeros_like(image)

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
        thresholded[image > threshold]  = threshold[image > threshold]
        thresholded[image <= threshold] = image[image <= threshold]
        return thresholded

    if mode == "to_zero":
        thresholded[image > threshold] = image[image > threshold]
        return thresholded
    
    thresholded[image <= threshold] = image[image <= threshold]
    return thresholded