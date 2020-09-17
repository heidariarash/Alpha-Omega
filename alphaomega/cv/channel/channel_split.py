import numpy as np

class ChannelSpliter:
    """
    Usage: Use this class to split the channels of your image into its constructing channels.
    """
    def __init__(self):
        """
        Usage  : The costructor of ChannelSplit class.
        Inputs : Nothing.
        Returns: An instantiation of ChannelSplit class.
        """
        self.__channels = list()
        self.__image_type = ""
        self.__channels_dimension = 2

    def config(self, **kwargs):
        """
        Usage: Use this method to configure the parameters of the ChannelSplit instantiation.

        Inputs:
            image_type         : the type of the image. This configuration can help you with get method. Use a string as the type, e.g. "RGB" or "BGR"
            channels_dimension: specify where the channels are stacked in the shape of the input image. default is "last". It could be also "first".

        Returns: Nothing.
        """
        for key , value in kwargs.items():
            if key == "image_type":
                self.__image_type = value
            elif key == "channels_dimension":
                if value == "first" or value == 0:
                    self.__channels_dimension = 0
                elif value == "last" or value == 2:
                    self.__channels_dimension = 2
                else:
                    print("Wrong value for channels_dimension. It could be only 'first' or 'last' (0 and 2 are also possible).")

    def get(self, channel):
        """
        Usage: Use this method to obtain the channel of interest.

        Inputs:
            channel: The channel of interest. for example, if you specified image_type as 'RGB', provide "R", "G", or "B" for input. You can also provide a number as the channel number.

        Returns: 
            - The desired channel.
        """
        #checking if the parameter is a channel type. e.g. "R"
        if (channel in self.__image_type):
            channel = self.__image_type.index(channel)

        if (type(channel) == int):
            return self.__channels[channel]

        print("The specified channel is incorrect.")

    def apply(self, image):
        """
        Usage: Use this method to apply the ChannelSplit method to you image. This method also assigns different channels to channels attribute.

        Inputs:
            image: The image to be splitted.

        Returns:
            - A list of numpy arrays each contains a channel.
        """
        #checking if the image has only two dimenstions (if it's grayscale)
        if (len(image.shape) == 2):
            self.__channels.clear()
            self.__channels.append(image)
            return self.__channels

        #checking if the image has three dimensions (e.g. RGB)
        if (len(image.shape) == 3):
            self.__channels.clear()
            for channel in range(image.shape[self.__channels_dimension]):
                if self.__channels_dimension == 2:
                    self.__channels.append(image[:,:, channel])
                elif self.__channels_dimension == 0:
                    self.__channels.append(image[channel,:, :])
            return self.__channels

        #if image has more than three dimesions or is one dimensional
        print("image should be only two or three dimensional.")

def channel_splitter_apply(image, channels_dimension = "last"):
    """
    Usage: Use this function to split the channels of your image into its constructing channels.

    Inputs:
        image              : The image to be splitted.
        channels_dimesnions: The dimension, in which the channels are stacked up in the input image. default is "last". It could be also "first".

    Returns:
        - A list of numpy arrays each contains a channel.
    """
    channels = []

    if channels_dimension == "last" or channels_dimension == 2:
        channels_dimension = 2
    elif channels_dimension == 'first' or channels_dimension == 0:
        channels_dimension = 0
    else:
        print("channels_dimesions should be 'first', or 'last'. (0 or 2 are also possible.)")

    #checking if the image has only two dimenstions (if it's grayscale)
    if (len(image.shape) == 2):
        channels.append(image)
        return channels

    #checking if the image has three dimensions (e.g. RGB)
    if (len(image.shape) == 3):
        for channel in range(image.shape[channels_dimension]):
            if channels_dimension == 2:
                channels.append(image[:,:, channel])
            elif channels_dimension == 0:
                channels.append(image[channel,:, :])
        return channels

    #if image has more than three dimesions or is one dimensional
    print("image should be only two or three dimensional.")