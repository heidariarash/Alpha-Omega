import numpy as np

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
            thresholded[image > self.__threshold] = self.__max
            return thresholded

        if self.__mode == "binary_inverse":
            thresholded[image <= self.__threshold] = self.__max
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


