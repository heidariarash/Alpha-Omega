import numpy as np
from alphaomega.cv.channel.channel_split import channel_splitter_apply
from alphaomega.utils.exceptions         import WrongAttribute, WrongDimension

class Histogram:
    """
    Usage: Use this class to compute histogram of an image.
    """
    def __init__(self):
        self.__image     = None
        self.__bins      = 256
        self.__hist      = None
        self.__shape     = 0
        self.__max_value = 255
        self.__min_value = 0
        self.__applied   = False

    def config(self, **kwargs) -> None:
        """
        Usage: Use this method to configure different parameteres of Histogram instantiation.

        Inputs:
            bins     : Number of bins for histogram.
            max_value: The maximum value present in histogram of the image. It usually is 255.
            min_value: The minimum value present in histogram of the imgea. It usually is 0.

        Returns: Nothing.
        """
        for key, value in kwargs.items():

            if key == "bins":
                if int(value) <=0:
                    raise WrongAttribute("bins should be a positive integer.")
                else:
                    self.__bins = int(value)

            elif key == "max_value":
                self.__max_value = value

            elif key == "min_value":
                self.__min_value = value

        if self.__min_value >= self.__max_value:
            print("min_value can not be greater than or equal to max_value. Both reseted to 0 and 255 respectively.")
            self.__max_value = 255
            self.__min_value = 0

    def get(self, attribute: str) -> np.ndarray:
        """
        Usage: Use this method to access different attributes of Histogram instantiation.

        Inputs:
            attirubte: The attribute of desire. It can be one of these options:
                "image"   : The original image of the histogram.
                "histogram: The histogram calculated from the image.

        Returns:
            The value of desired attribute.
        """
        if not self.__applied:
            raise Exception("You should use the apply method first, Then using this method is possible.")

        if attribute == "histogram":
            return self.__hist
        
        if attribute == "image":
            return self.__image

        raise WrongAttribute('Please specify the correct attribute. Options are: "histogram" and "image".')

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Usage: Use this method to calculate the histogram of an image.

        Inputs:
            image: The histogram of this image will be calculated.

        Returns:
            - if the input image is single channel, the output will be a 1D numpy array including the value of each bin.
            - if the input image is multi channel, the output will be a 2D numpy array including the value of each bin for each channel.
        """
        #checking for the shape of the image.
        if len(image.shape) == 2 or ( len(image.shape) == 3 and image.shape[2] == 1):
            self.__shape = 2
            self.__image = image
        
        elif len(image.shape) == 3:
            self.__shape = 3
            self.__image = image

        else:
            raise WrongDimension("image should be 2D(single channel) or 3D(multi channel).")

        bins = np.linspace(self.__max_value, self.__min_value, self.__bins + 1)
        if (self.__shape == 2):
            hist = np.zeros(len(bins) - 1, dtype=np.int32)
            hist[self.__bins - 1] = len(image[image >= bins[1]])
            for index, bin in enumerate(bins[2:]):
                hist[self.__bins - index - 2] = len(image[image >= bin]) - len(image[image>= bins[index + 1]])

            self.__hist = hist
            return hist

        channels = channel_splitter_apply(image)
        hist = np.zeros((len(channels), len(bins) - 1), dtype=np.int32)
        for ch_index, channel in enumerate(channels):
            hist[ch_index, self.__bins - 1] = len(channel[channel >= bins[1]])
            for index, bin in enumerate(bins[2:]):
                hist[ch_index, self.__bins - index - 2] = len(channel[channel >= bin]) - len(channel[channel>= bins[index + 1]])

        self.__hist = hist
        self.__applied = True
        return hist


def histogram_apply(image: np.ndarray, bins: int = 256, max_value: int = 255, min_value: int = 0):
    """
    Usage: Use this function to compute the histogram of an image.

    Inputs:
        image    : The histogram of this image will be calculated.
        bins     : Number of bins for histogram.
        max_value: The maximum value present in histogram of the image. It usually is 256.
        min_value: The minimum value present in histogram of the imgea. It usually is 0.

    Returns:
        - if the input image is single channel, the output will be a 1D numpy array including the value of each bin.
        - if the input image is multi channel, the output will be a 2D numpy array including the value of each bin for each channel.
    """
    #checking fot the correct value of bins.
    if int(bins) <=0:
        raise WrongAttribute("bins should be an positive integer.")
    
    #checking if min_value is actually less than max_value.
    if min_value >= max_value:
        raise WrongAttribute("min_value can not be greater than or equal to max_value.")
    
    #checking for the correct shape of the image.
    if len(image.shape) == 2 or ( len(image.shape) == 3 and image.shape[2] == 1):
        shape = 2
    elif len(image.shape) == 3:
        shape = 3
    else:
        raise WrongDimension("image should be 2D(single channel) or 3D(multi channel).")

    #applying histogramization
    bins = np.linspace(max_value, _min_value, bins + 1)
    if (shape == 2):
        hist = np.zeros(len(bins) - 1, dtype=np.int32)
        hist[bins - 1] = len(image[image >= bins[1]])
        for index, bin in enumerate(bins[2:]):
            hist[bins - index - 2] = len(image[image >= bin]) - len(image[image>= bins[index + 1]])

        return hist

    channels = channel_splitter_apply(image)
    hist = np.zeros((len(channels), len(bins) - 1), dtype=np.int32)
    for ch_index, channel in enumerate(channels):
        hist[ch_index, bins - 1] = len(channel[channel >= bins[1]])
        for index, bin in enumerate(bins[2:]):
            hist[ch_index, bins - index - 2] = len(channel[channel >= bin]) - len(channel[channel>= bins[index + 1]])

    return hist