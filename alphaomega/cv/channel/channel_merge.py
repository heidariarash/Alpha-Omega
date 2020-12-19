import numpy as np
from typing                      import Union
from alphaomega.utils.exceptions import WrongAttribute, WrongDimension

class ChannelMergeer:
    """
    Usage: Use this class to merge different channels of an image into an image.
    """
    def __init__(self):
        self.__channels_dimension = 2

    def config(self, **kwargs) -> None:
        """
        Usage: Use this method to configure the parameters of the ChannelMerger instantiation.

        Inputs:
            channels_dimension: The dimension of the channels. It could be either "first" or "last" (or equally 0 and 2). if it is "first" and you have 3 750x750 channels, the shape of the result is (3,750,750).

        Returns: Nothing.
        """
        for key, value in kwargs.items():

            if key == "channels_dimension":

                if value == "first" or value == 0:
                    self.__channels_dimension = 0

                elif value == "last" or value == 2:
                    self.__channels_dimension = 2

                else:
                    raise WrongAttribute("Wrong value for channels_dimension. It could be only 'first' or 'last' (0 and 2 are also possible).")

    def apply(self, *channels) -> np.ndarray:
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
            raise WrongDimension("Channels should be 2 dimensional.")
        
        chs = [np.zeros_like(channels[0])] * len(channels)

        #checking if all the channels have the same shape. If they have, expanding them.
        for index, channel in enumerate(channels):
            if channel.shape != shape:
                raise WrongDimension(f"Channels should be the same size. The channel {index+1} has a different shape from the first channel.")
            else:
                chs[index] = np.expand_dims(channel, self.__channels_dimension)
            
        return np.concatenate([*chs], axis = self.__channels_dimension)


def channel_merger_apply(channels: list, channels_dimension: Union[str, int] = "last") -> np.ndarray:
    """
    Usage: Use this function to merge the channels of an image and construct the image.

    Inputs:
        channels           : The channels of the image. They should be the same size.
        channels_dimension: The dimension of the channels. It could be either "first" or "last" (or equally 0 and 2). if it is "first" and you have 3 750x750 channels, the shape of the result is (3,750,750).

    Returns:
        - The constructed image using the channels.
    """

    if channels_dimension == "last" or channels_dimension == 2:
        channels_dimension = 2
    elif channels_dimension == 'first' or channels_dimension == 0:
        channels_dimension = 0
    else:
        raise WrongAttribute("channels_dimesions should be 'first', or 'last'. (0 or 2 are also possible.)")

    shape = channels[0].shape

    #checking if the first channel has only 2 dimensions.
    if len(shape) != 2:
        raise WrongDimension("Channels should be 2 dimensional.")
    
    chs = [np.zeros_like(channels[0])] * len(channels)

    #checking if all the channels have the same shape. If they have, expanding them.
    for index, channel in enumerate(channels):
        if channel.shape != shape:
            raise WrongDimension(f"Channels should be the same size. The channel {index+1} has a different shape as the first channel.")
        else:
            chs[index] = np.expand_dims(channel, channels_dimension)
        
    return np.concatenate([*chs], axis = channels_dimension)