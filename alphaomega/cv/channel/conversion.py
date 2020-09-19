import numpy as np

class Converter:
    """
    Usage: Use this class to convert your image from one color space to another.
    """
    def __init__(self):
        """
        Usage  : The constructor of the Converter class.
        Inputs : Nothing.
        Returns: An instantiation of Converter class.
        """
        self.__dest = None
        self.__src = None

    def config(self, **kwargs):
        """
        Usage: Use this method to configure the parameters of the Converter instantiation.

        Inputs:
            source      : The source color space. It should be one of these options:
                "RGB" or "RGBA"
                "BGR" or "BGRA"
                "HSL"
                "CIE"
                "HSV"
                "LAB"
            destination : The destination color space. It should be one of these options:
                "RGB"
                "BGR"
                "HSL"
                "CIE"
                "HSV"
                "LAB"
                "GRAY"

        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "source":
                if value not in ["RGB", "RGBA", "BGRA", "BGR", "HSL", "CIE", "HSV", "LAB"]:
                    print('The only accpetable values for source are: "RGB", "BGR", "HSL", "CIE", "HSV", and "LAB".')
                else:
                    self.__src = value
            elif key == "destination":
                if value not in ["RGB", "BGR", "RGBA", "BGRA", "HSL", "CIE", "HSV", "LAB", "GRAY"]:
                    print('The only accpetable values for source are: "RGB", "BGR", "HSL", "CIE", "HSV", "LAB", and "GRAY".')
                else:
                    self.__dest = value

        if (self.__src == self.__dest):
            print("Are you sure? Both source and destination color spaces are same...")

    def apply(self, image):
        """
        Usage: Use this method to apply the color space conversion on an image.

        Inputs:
            image: The conversion will be applied on this image.

        Returns:
            - An image with new color space.
        """
        #checking if configuration has been done.
        if (not self.__src) or (not self.__dest):
            print("please use config method first.")
            return

        #checking if both source and destiantion are same.
        if self.__src == self.__dest:
            return image

        #checking if the image has 3 dimensions.
        if len(image.shape) != 3:
            print("This is not an image...")
            return

        #checking if the image has the right number of channels.
        if len(self.__src) != image.shape[2]:
            print("The number of channels in the image is not equal to the expected number of channels (what you have configured using config method).")
            return

        #doing conversion
        if self.__src == "RGB":
            if self.__dest == "BGR":
                return self.__RGB2BGR(image) #done
            elif self.__dest == "RGBA":
                return self.__RGB2RGBA(image) #done
            elif self.__dest == "BGRA":
                return self.__RGB2BGRA(image) #done
            elif self.__dest == "HSL":
                return self.__RGB2HSL(image)
            elif self.__dest == "HSV":
                return self.__RGB2HSV(image)
            elif self.__dest == "CIE":
                return self.__RGB2CIE(image)
            elif self.__dest == "LAB":
                return self.__RGB2LAB(image)
            elif self.__dest == "GRAY":
                return self.__RGB2GRAY(image) #done

        elif self.__src == "BGR":
            if self.__dest == "RGB":
                return self.__RGB2BGR(image) #done
            elif self.__dest == "RGBA":
                return self.__RGB2BGRA(image) #done
            elif self.__dest == "BGRA":
                return self.__RGB2RGBA(image) #done
            elif self.__dest == "HSL":
                return self.__BGR2HSL(image)
            elif self.__dest == "HSV":
                return self.__BGR2HSV(image)
            elif self.__dest == "CIE":
                return self.__BGR2CIE(image)
            elif self.__dest == "LAB":
                return self.__BGR2LAB(image)
            elif self.__dest == "GRAY":
                return self.__BGR2GRAY(image) #done

        elif self.__src == "RGBA":
            if self.__dest == "BGR":
                return self.__RGBA2BGR(image) #done
            elif self.__dest == "RGB":
                return self.__RGBA2RGB(image) #done
            elif self.__dest == "BGRA":
                return self.__RGBA2BGRA(image) #done
            elif self.__dest == "HSL":
                return self.__RGBA2HSL(image)
            elif self.__dest == "HSV":
                return self.__RGBA2HSV(image)
            elif self.__dest == "CIE":
                return self.__RGBA2CIE(image)
            elif self.__dest == "LAB":
                return self.__RGBA2LAB(image)
            elif self.__dest == "GRAY":
                return self.__RGBA2GRAY(image) #done

        elif self.__src == "BGRA":
            if self.__dest == "BGR":
                return self.__RGBA2RGB(image) #done
            elif self.__dest == "RGBA":
                return self.__RGBA2BGRA(image) #done
            elif self.__dest == "RGB":
                return self.__RGBA2BGR(image) #done
            elif self.__dest == "HSL":
                return self.__BGRA2HSL(image)
            elif self.__dest == "HSV":
                return self.__BGRA2HSV(image)
            elif self.__dest == "CIE":
                return self.__BGRA2CIE(image)
            elif self.__dest == "LAB":
                return self.__BGRA2LAB(image)
            elif self.__dest == "GRAY":
                return self.__BGRA2GRAY(image) #done

        elif self.__src == "HSL":
            if self.__dest == "BGR":
                return self.__HSL2BGR(image)
            elif self.__dest == "RGBA":
                return self.__HSL2RGBA(image)
            elif self.__dest == "BGRA":
                return self.__HSL2BGRA(image)
            elif self.__dest == "RGB":
                return self.__HSL2RGB(image)
            elif self.__dest == "HSV":
                return self.__HSL2HSV(image)
            elif self.__dest == "CIE":
                return self.__HSL2CIE(image)
            elif self.__dest == "LAB":
                return self.__HSL2LAB(image)
            elif self.__dest == "GRAY":
                return self.__HSL2GRAY(image)

        elif self.__src == "HSV":
            if self.__dest == "BGR":
                return self.__HSV2BGR(image)
            elif self.__dest == "RGBA":
                return self.__HSV2RGBA(image)
            elif self.__dest == "BGRA":
                return self.__HSV2BGRA(image)
            elif self.__dest == "HSL":
                return self.__HSV2HSL(image)
            elif self.__dest == "RGB":
                return self.__HSV2RGB(image)
            elif self.__dest == "CIE":
                return self.__HSV2CIE(image)
            elif self.__dest == "LAB":
                return self.__HSV2LAB(image)
            elif self.__dest == "GRAY":
                return self.__HSV2GRAY(image)

        elif self.__src == "CIE":
            if self.__dest == "BGR":
                return self.__CIE2BGR(image)
            elif self.__dest == "RGBA":
                return self.__CIE2RGBA(image)
            elif self.__dest == "BGRA":
                return self.__CIE2BGRA(image)
            elif self.__dest == "HSL":
                return self.__CIE2HSL(image)
            elif self.__dest == "HSV":
                return self.__CIE2HSV(image)
            elif self.__dest == "RGB":
                return self.__CIE2RGB(image)
            elif self.__dest == "LAB":
                return self.__CIE2LAB(image)
            elif self.__dest == "GRAY":
                return self.__CIE2GRAY(image)

        elif self.__src == "LAB":
            if self.__dest == "BGR":
                return self.__LAB2BGR(image)
            elif self.__dest == "RGBA":
                return self.__LAB2RGBA(image)
            elif self.__dest == "BGRA":
                return self.__LAB2BGRA(image)
            elif self.__dest == "HSL":
                return self.__LAB2HSL(image)
            elif self.__dest == "HSV":
                return self.__LAB2HSV(image)
            elif self.__dest == "CIE":
                return self.__LAB2CIE(image)
            elif self.__dest == "RGB":
                return self.__LAB2RGB(image)
            elif self.__dest == "GRAY":
                return self.__LAB2GRAY(image)
        
    def __RGB2BGR(self, image):
        converted = np.zeros_like(image)
        converted[:,:,0] = image[:,:,2]
        converted[:,:,2] = image[:,:,0]
        converted[:,:,1] = image[:,:,1]
        return converted

    def __RGB2RGBA(self, image):
        converted = np.concatenate(image, 255 * np.ones((image.shape[0], image.shpae[1]), dtype=np.int16))
        return converted

    def __RGB2BGRA(self, image):
        converted = self.__RGB2BGR(image)
        converted = self.__RGB2RGBA(image)
        return converted

    def __RGB2GRAY(self, image):
        converted = 0.299 * image[:,:,0].astype(np.float) + 0.587 * image[:,:,1].astype(np.float) + 0.114 * image[:,:,2].astype(np.float)
        converted = converted.astype(np.int16)
        return converted

    def __BGR2GRAY(self, image):
        converted = self.__RGB2BGR(image)
        converted = self.__RGB2GRAY(image)
        return converted

    def __RGBA2RGB(self, image):
        converted = image[:,:,:-1]
        return converted

    def __RGBA2BGR(self, image):
        converted = self.__RGBA2RGB(image)
        converted = self.__RGB2BGR(image)
        return converted

    def __RGBA2BGRA(self, image):
        converted = np.zeros_like(image)
        converted[:,:,0] = image[:,:,2]
        converted[:,:,2] = image[:,:,0]
        converted[:,:,1] = image[:,:,1]
        converted[:,:,3] = image[:,:,3]
        return converted

    def __RGBA2GRAY(self, image):
        converted = self.__RGBA2RGB(image)
        converted = self.__RGB2GRAY(image)
        return converted

    def __BGRA2GRAY(self, image):
        converted = self.__RGBA2BGRA(image)
        converted = self.__RGBA2GRAY(image)
        return converted