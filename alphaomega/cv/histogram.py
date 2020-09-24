import numpy as np
from alphaomega.cv.channel.channel_split import channel_splitter_apply

class Histogram:
    """
    Usage: Use this class to compute histogram of an image.
    """
    def __init__(self):
        """
        Usage  : The constructor of Histogram class.
        Inputs : Nothing.
        Returns: An instantiation of Histogram class.
        """
        self.__image = None
        self.__bins = 256
        self.__hist = None
        self.__shape = 0

    def config(self, **kwargs):
        """
        Usage: Use this method to configure different parameteres of Histogram instantiation.

        Inputs:
            bins: Number of bins for histogram.

        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "bins":
                if int(value) <=0 or int(value) > 256:
                    print("bins should be an integer between 1 and 256.")
                else:
                    self.__bins = value

    def apply(self, image):
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
            print("image should be 2D(single channel) or 3D(multi channel).")
            return

        bins = np.linspace(256, 0, self.__bins + 1)
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
        return hist