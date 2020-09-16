import numpy as np

class ChannelMerge:
    """
    Usage: Use this class to merge different channels of an image into an image.
    """
    def __init__(self):
        """
        Usage  : The constructor of the ChannelMerge class.
        Inputs : Nothing.
        Returns: An instantiation of the ChannelMerge class.
        """
        self.__image = None

    def apply(self, *channels):
        """
        Usage: Use this method to apply the merging of the channels and build a new image.

        Inputs:
            channels: specify the channels of the image here. They should be the same size.

        Returns:
            - The constructed image using the channels.
        """
        shape = channels[0].shape

        #checking if the first channel has only 2 dimensions.
        if len(shape) != 2:
            print("Channels should be 2d.")
            return

        #checking if all the channels have the same shape. If they have, expanding them.
        for index, channel in enumerate(channels):
            if channel.shape != shape:
                print(f"Channels should be the same size. The channel {index+1} has a different shape as the first channel.")
                return
            else:
                np.expand_dims(channel, 0)

        self.__image = np.concatenate(*channels, axis=0)
        return self.__image