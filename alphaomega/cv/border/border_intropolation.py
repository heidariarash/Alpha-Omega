import numpy as np

class BorderIntropolation:
    """
    Usage: Use this class to intropolate the borders of a single channel image.
    """
    def __init__(self):
        """
        Usage: The constructor of the BorderIntropolation class.
        Inputs: Nothing.
        Returns: An instantiation of the BorderIntropolation class.
        """
        self.__top = 1
        self.__bottom = 1
        self.__left = 1
        self.__right = 1
        self.__border_type = "constant"

    def config(self, **kwargs):
        """
        Usage: Use this method to configure the parameteres of the BorderIntropolation instantiation.

        Inputs:
            top        : The number of pixels added to the top of the image.
            bottom     : The number of pixels added to the bottom of the image.
            left       : The number of pixels added to the left of the image.
            rights     : The number of pixels added to the right of the image.
            pixels_add : Instead of specifying the number of pixels added to the top, bottom, left, and right of the image, you can give the same number of pixels to all of them using this parameter.
            border_type: The type of intropolatoin. It could be one of this options:
                "constant": default option.
                "reflect"
                "replicate"
                "wrap"
                "reflect_without_border"
            
        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "top":
                if (int(value) >= 0):
                    self.__top = value
                else:
                    print("The value of top should be an integer greater than -1.")
            elif key == "bottom":
                if (int(value) >= 0):
                    self.__bottom = value
                else:
                    print("The value of bottom should be an integer greater than -1.")
            elif key == "left":
                if (int(value) >= 0):
                    self.__left = value
                else:
                    print("The value of left should be an integer greater than -1.")
            elif key == "right":
                if (int(value) >= 0):
                    self.__right = value
                else:
                    print("The value of right should be an integer greater than -1.")
            elif key == "pixels_add":
                if (int(value) > 0):
                    self.__left = value
                    self.__right = value
                    self.__top = value
                    self.__bottom = value
                else:
                    print("The value of pixels_add should be an integer greater than -1.")
            elif key == "border_type":
                if (value not in ["constant", "reflect", "replicate", "wrap", "reflect_without_border"]):
                    print('The only options for border are "constant", "reflect", "replicate", "wrap", and "reflect_without_border".')
                else:
                    self.__border_type = value

        if (self.__top == 0 and self.__bottom == 0 and self.__right == 0 and self.__left == 0):
            print("Really?? You don't need intropolation? So why are you using this class? Please send me an email and explain: arashheidari94@outlook.com =)")

    def apply(self, image):
        """
        Usage: Use this method to apply BorderIntropolation to an image.

        Inputs:
            image: The intropolation will be applied on this image.

        Returns:
            - The new image with intropolated borders.
        """
        if self.__border_type == "constant":
            image = np.concatenate((np.zeros((image.shape[0], self.__left)), image), axis = 1)
            image = np.concatenate((image, np.zeros((image.shape[0], self.__right))), axis = 1)
            image = np.concatenate((np.zeros((self.__top, image.shape[1])), image), axis = 0)
            image = np.concatenate((image, np.zeros((self.__bottom, image.shape[1]))), axis = 0)
            return image

        if self.__border_type == "replicate":
            image = np.concatenate((np.repeat(np.expand_dims(image[:, 0], 1), self.__left, 1), image), axis = 1)
            image = np.concatenate((image, np.repeat(np.expand_dims(image[:, image.shape[1]-1], 1), self.__right, 1)), axis = 1)
            image = np.concatenate((np.repeat(np.expand_dims(image[0, :], 0), self.__top, 0), image), axis = 0)
            image = np.concatenate((image, np.repeat(np.expand_dims(image[image.shape[0]-1, :], 0), self.__bottom, 0)), axis=0)
            return image

        if self.__border_type == "reflect":
            image = np.concatenate((np.flip(image[:, 0:self.__left], axis= 1), image), axis= 1)
            image = np.concatenate((image, np.flip(image[:, image.shape[1] - self.__right:], axis = 1)), axis= 1)
            image = np.concatenate((np.flip(image[:self.__top,:], axis=0), image), axis= 0)
            image = np.concatenate((image, np.flip(image[image.shape[0] - self.__bottom:,:], axis=0)), axis= 0)
            return image

        if self.__border_type == "reflect_without_border":
            image = np.concatenate((np.flip(image[:, 1:self.__left+1], axis= 1), image), axis= 1)
            image = np.concatenate((image, np.flip(image[:, image.shape[1] - self.__right - 1:-1], axis = 1)), axis= 1)
            image = np.concatenate((np.flip(image[1:self.__top + 1,:], axis=0), image), axis= 0)
            image = np.concatenate((image, np.flip(image[image.shape[0] - self.__bottom - 1:-1,:], axis=0)), axis= 0)
            return image

        #only option is warp
        orig  = image.copy()
        image = np.concatenate((orig[:, image.shape[1] - self.__left:], image), axis= 1)
        image = np.concatenate((image, orig[:, 0:self.__right]), axis= 1)
        orig = image.copy()
        image = np.concatenate((image[image.shape[0] - self.__top:,:], image), axis= 0)
        image = np.concatenate((image, orig[:self.__bottom,:]), axis= 0)
        return image


def border_intropolate_apply(image, pixels_add, border_type = "constant"):
    """
    Usage: Use this function to apply border intropolation to an image.

    Inputs:
        image: The intropolation will be applied on this image.
        pixels_add: The pixels to add to the borders
        border_type: The type of intropolatoin. It could be one of this options:
            "constant": default option.
            "reflect"
            "replicate"
            "wrap"
            "reflect_without_border"

    Returns:
        - The new image with intropolated borders.
    """
    if border_type == "constant":
            image = np.concatenate((np.zeros((image.shape[0], pixels_add)), image), axis = 1)
            image = np.concatenate((image, np.zeros((image.shape[0], pixels_add))), axis = 1)
            image = np.concatenate((np.zeros((pixels_add, image.shape[1])), image), axis = 0)
            image = np.concatenate((image, np.zeros((pixels_add, image.shape[1]))), axis = 0)
            return image

    if border_type == "replicate":
        image = np.concatenate((np.repeat(np.expand_dims(image[:, 0], 1), pixels_add, 1), image), axis = 1)
        image = np.concatenate((image, np.repeat(np.expand_dims(image[:, image.shape[1]-1], 1), pixels_add, 1)), axis = 1)
        image = np.concatenate((np.repeat(np.expand_dims(image[0, :], 0), pixels_add, 0), image), axis = 0)
        image = np.concatenate((image, np.repeat(np.expand_dims(image[image.shape[0]-1, :], 0), pixels_add, 0)), axis=0)
        return image

    if border_type == "reflect":
        image = np.concatenate((np.flip(image[:, 0:pixels_add], axis= 1), image), axis= 1)
        image = np.concatenate((image, np.flip(image[:, image.shape[1] - pixels_add:], axis = 1)), axis= 1)
        image = np.concatenate((np.flip(image[:pixels_add,:], axis=0), image), axis= 0)
        image = np.concatenate((image, np.flip(image[image.shape[0] - pixels_add:,:], axis=0)), axis= 0)
        return image

    if border_type == "reflect_without_border":
        image = np.concatenate((np.flip(image[:, 1:pixels_add+1], axis= 1), image), axis= 1)
        image = np.concatenate((image, np.flip(image[:, image.shape[1] - pixels_add - 1:-1], axis = 1)), axis= 1)
        image = np.concatenate((np.flip(image[1:pixels_add + 1,:], axis=0), image), axis= 0)
        image = np.concatenate((image, np.flip(image[image.shape[0] - pixels_add - 1:-1,:], axis=0)), axis= 0)
        return image

    if border_type == "warp":
        orig  = image.copy()
        image = np.concatenate((orig[:, image.shape[1] - pixels_add:], image), axis= 1)
        image = np.concatenate((image, orig[:, 0:pixels_add]), axis= 1)
        orig = image.copy()
        image = np.concatenate((image[image.shape[0] - pixels_add:,:], image), axis= 0)
        image = np.concatenate((image, orig[:pixels_add,:]), axis= 0)
        return image

    print("wrong argument for border type. It could be one of this options:")
    print("constant")
    print("replicate")
    print("reflect")
    print("reflect_without_border")
    print("warp")