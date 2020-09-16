import numpy as np

class ChannelMergeer:
    """
    Usage: Use this class to merge different channels of an image into an image.
    """
    def __init__(self):
        """
        Usage  : The constructor of the ChannelMerge class.
        Inputs : Nothing.
        Returns: An instantiation of the ChannelMerge class.
        """
        self.__channels_dimension = 2

    def config(self, **kwargs):
        """
        Usage: Use this method to configure the parameters of the ChannelMerger instantiation.

        Inputs:
            channels_dimension: The dimension of the channels. It could be either "first" or "last" (or equally 0 and 2). if it is "first" and you have 3 750x750 channels, the shape of the result is (3,750,750).

        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "channels_dimensions":
                if value == "first" or value == 0:
                    self.__channels_dimension = 0
                elif value == "last" or value == 2:
                    self.__channels_dimension = 2
                else:
                    print("Wrong value for channels_dimension. It could be only 'first' or 'last' (0 and 2 are also possible).")

    def apply(self, *channels):
        """
        Usage: Use this method to apply the merging of the channels and build a new image.

        Inputs:
            channels: The channels of the image. They should be the same size.

        Returns:
            - The constructed image using the channels.
        """
        shape = channels[0].shape

        #checking if the first channel has only 2 dimensions.
        if len(shape) != 2:
            print("Channels should be 2d.")
            return
        
        chs = [np.zeros_like(channels[0])] * len(channels)

        #checking if all the channels have the same shape. If they have, expanding them.
        for index, channel in enumerate(channels):
            if channel.shape != shape:
                print(f"Channels should be the same size. The channel {index+1} has a different shape as the first channel.")
                return
            else:
                chs[index] = np.expand_dims(channel, self.__channels_dimension)
            
        return np.concatenate([*chs], axis = self.__channels_dimension)


def channel_merger(channels_dimensions, *channels):
    """
    Usage: Use this function to merge the channels of an image and construct the image.

    Inputs:
        channels_dimensions: The dimension of the channels. It could be either "first" or "last" (or equally 0 and 2). if it is "first" and you have 3 750x750 channels, the shape of the result is (3,750,750).
        channels           : The channels of the image. They should be the same size.

    Returns:
        - The constructed image using the channels.
    """
    shape = channels[0].shape

    #checking if the first channel has only 2 dimensions.
    if len(shape) != 2:
        print("Channels should be 2d.")
        return
    
    chs = [np.zeros_like(channels[0])] * len(channels)

    #checking if all the channels have the same shape. If they have, expanding them.
    for index, channel in enumerate(channels):
        if channel.shape != shape:
            print(f"Channels should be the same size. The channel {index+1} has a different shape as the first channel.")
            return
        else:
            chs[index] = np.expand_dims(channel, channels_dimension)
        
    return np.concatenate([*chs], axis = channels_dimension)