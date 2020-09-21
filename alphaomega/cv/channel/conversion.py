import numpy as np
from alphaomega.cv.channel.channel_merge import channel_merger_apply

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
                "HSV"
            destination : The destination color space. It should be one of these options:
                "RGB"
                "BGR"
                "HSL"
                "HSV"
                "GRAY"

        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "source":
                if value not in ["RGB", "RGBA", "BGRA", "BGR", "HSL", "HSV"]:
                    print('The only accpetable values for source are: "RGB", "BGR", "HSL", and "HSV".')
                else:
                    self.__src = value
            elif key == "destination":
                if value not in ["RGB", "BGR", "RGBA", "BGRA", "HSL", "HSV", "GRAY"]:
                    print('The only accpetable values for source are: "RGB", "BGR", "HSL", "HSV", and "GRAY".')
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
                return self.__RGB2HSL(image) #done
            elif self.__dest == "HSV":
                return self.__RGB2HSV(image) #done
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
                return self.__BGR2HSL(image) #done
            elif self.__dest == "HSV":
                return self.__BGR2HSV(image) #done
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
                return self.__RGBA2HSL(image) #done
            elif self.__dest == "HSV":
                return self.__RGBA2HSV(image) #done
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
                return self.__BGRA2HSL(image) #done
            elif self.__dest == "HSV":
                return self.__BGRA2HSV(image) #done
            elif self.__dest == "GRAY":
                return self.__BGRA2GRAY(image) #done

        elif self.__src == "HSL": #need revisiting
            if self.__dest == "BGR":
                return self.__HSL2BGR(image) #done
            elif self.__dest == "RGBA":
                return self.__HSL2RGBA(image) #done
            elif self.__dest == "BGRA":
                return self.__HSL2BGRA(image) #done
            elif self.__dest == "RGB":
                return self.__HSL2RGB(image) #done
            elif self.__dest == "HSV":
                return self.__HSL2HSV(image) #done
            elif self.__dest == "GRAY":
                return self.__HSL2GRAY(image) #done

        elif self.__src == "HSV": #need revisiting
            if self.__dest == "BGR":
                return self.__HSV2BGR(image) #done
            elif self.__dest == "RGBA":
                return self.__HSV2RGBA(image) #done
            elif self.__dest == "BGRA":
                return self.__HSV2BGRA(image) #done
            elif self.__dest == "HSL":
                return self.__HSV2HSL(image) #done
            elif self.__dest == "RGB":
                return self.__HSV2RGB(image) #done
            elif self.__dest == "GRAY":
                return self.__HSV2GRAY(image) #done
        
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

    def __RGB2HSL(self, image):
        rprime = image[:,:,0] / 255
        gprime = image[:,:,1] / 255
        bprime = image[:,:,2] / 255
        cmax = np.max(image/255, axis=2)
        cmin = np.min(image/255, axis=2)
        delta = cmax - cmin
        L_channel = ( cmax + cmin ) / 2
        S_channel = np.zeros_like(L_channel)
        S_channel[L_channel < 0.5] = delta[L_channel < 0.5] / (cmax[L_channel<0.5] + cmax[L_channel < 0.5])
        S_channel[L_channel > 0.5] = delta[L_channel > 0.5] / (2 - cmax[L_channel > 0.5] - cmax[L_channel > 0.5])
        H_channel = np.zeros_like(L_channel)
        H_channel[cmax == rprime] = (60 * ((gprime[cmax == rprime] - bprime[cmax == rprime]) / delta[cmax == rprime]) ).astype(np.int16) % 360
        H_channel[cmax == gprime] = (60 * ((bprime[cmax == gprime] - gprime[cmax == gprime]) / delta[cmax == gprime]) + 120).astype(np.int16) % 360
        H_channel[cmax == bprime] = (60 * ((rprime[cmax == bprime] - gprime[cmax == bprime]) / delta[cmax == bprime]) + 240).astype(np.int16) % 360
        converted = channel_merger_apply([H_channel, S_channel, L_channel])
        return converted

    def __BGR2HSL(self, image):
        converted = self.__RGB2BGR(image)
        converted = self.__RGB2HSL(image)
        return converted

    def __RGBA2HSL(self, image):
        converted = self.__RGBA2RGB(image)
        converted = self.__RGB2HSL(image)
        return converted

    def __BGRA2HSL(self, image):
        converted = self.__RGBA2BGR(image)
        converted = self.__RGB2HSL(image)
        return converted

    def __RGB2HSV(self, image):
        rprime = image[:,:,0] / 255
        gprime = image[:,:,1] / 255
        bprime = image[:,:,2] / 255
        cmax = np.max(image/255, axis=2)
        cmin = np.min(image/255, axis=2)
        delta = cmax - cmin
        S_channel = np.zeros_like(delta)
        S_channel[cmax != 0] = delta[cmax != 0] / (cmax[cmax != 0])
        H_channel = np.zeros_like(delta)
        H_channel[cmax == rprime] = (60 * ((gprime[cmax == rprime] - bprime[cmax == rprime]) / delta[cmax == rprime]) ).astype(np.int16) % 360
        H_channel[cmax == gprime] = (60 * ((bprime[cmax == gprime] - gprime[cmax == gprime]) / delta[cmax == gprime]) + 120).astype(np.int16) % 360
        H_channel[cmax == bprime] = (60 * ((rprime[cmax == bprime] - gprime[cmax == bprime]) / delta[cmax == bprime]) + 240).astype(np.int16) % 360
        converted = channel_merger_apply([H_channel, S_channel, cmax])
        return converted

    def __BGR2HSV(self, image):
        converted = self.__RGB2BGR(imgae)
        converted = self.__RGB2HSV(image)
        return converted

    def __RGBA2HSV(self, image):
        converted = self.__RGBA2RGB(image)
        converted = self.__RGB2HSV(image)
        return converted

    def __BGRA2HSV(self, image):
        converted = self.__RGBA2BGR(image)
        converted = self.__RGB2HSV(image)
        return converted

    def __HSV2RGB(self, image):
        h = image[:,:,0]
        s = image[:,:,1]
        v = image[:,:,2]
        C = v * s
        X = C * (np.ones_like(C) - np.abs(((h / 60) % 2) - np.ones_like(C)))
        m = v - C
        rprime = np.zeros_like(C)
        gprime = np.zeros_like(C)
        bprime = np.zeros_like(C)
        rprime, gprime = C, X
        rprime[h>=60], gprime[h>=60] = X[h>=60], C[h>=60]
        gprime[h>=120], bprime[h>=120] = C[h>=120], X[h>=120]
        gprime[h>=180], bprime[h>=180] = X[h>=180], C[h>=180]
        rprime[h>=240], bprime[h>=240] = X[h>=240], C[h>=240]
        rprime[h>=300], bprime[h>=300] = C[h>=300], X[h>=300]

        r = (rprime + m) *255
        g = (gprime + m) *255
        b = (bprime + m) *255
        converted = channel_merger_apply([r,g,b])
        return converted.astype(np.int16)

    def __HSV2BGR(self, image):
        converted = self.__HSV2RGB(image)
        converted = self.__RGB2BGR(image)
        return converted

    def __HSV2BGRA(self, image):
        converted = self.__HSV2RGB(image)
        converted = self.__RGB2BGRA(image)
        return converted

    def __HSV2RGBA(self, image):
        converted = self.__HSV2RGB(image)
        converted = self.__RGB2RGBA(image)
        return converted

    def __HSV2HSL(self, image):
        converted = self.__HSV2RGB(image)
        converted = self.__RGB2HSL(image)
        return converted

    def __HSV2GRAY(self, image):
        converted = self.__HSV2RGB(image)
        converted = self.__RGB2GRAY(image)
        return converted
    
    def __HSL2RGB(self, image):
        h = image[:,:,0]
        s = image[:,:,1]
        l = image[:,:,2]
        C = (np.ones_like(h) - np.abs(2 * l - np.ones_like(h))) * s
        X = C * (np.ones_like(C) - np.abs(((h / 60) % 2) - np.ones_like(C)))
        m = l - C / 2
        rprime = np.zeros_like(C)
        gprime = np.zeros_like(C)
        bprime = np.zeros_like(C)
        rprime, gprime = C, X
        rprime[h>=60], gprime[h>=60] = X[h>=60], C[h>=60]
        gprime[h>=120], bprime[h>=120] = C[h>=120], X[h>=120]
        gprime[h>=180], bprime[h>=180] = X[h>=180], C[h>=180]
        rprime[h>=240], bprime[h>=240] = X[h>=240], C[h>=240]
        rprime[h>=300], bprime[h>=300] = C[h>=300], X[h>=300]

        r = (rprime + m) *255
        g = (gprime + m) *255
        b = (bprime + m) *255
        converted = channel_merger_apply([r,g,b])
        return converted.astype(np.int16)

    def __HSL2BGR(self, image):
        converted = self.__HSL2RGB(image)
        converted = self.__RGB2BGR(image)
        return converted

    def __HSL2BGRA(self, image):
        converted = self.__HSL2RGB(image)
        converted = self.__RGB2BGRA(image)
        return converted

    def __HSL2RGBA(self, image):
        converted = self.__HSL2RGB(image)
        converted = self.__RGB2RGBA(image)
        return converted

    def __HSL2HSV(self, image):
        converted = self.__HSL2RGB(image)
        converted = self.__RGB2HSV(image)
        return converted

    def __HSL2GRAY(self, image):
        converted = self.__HSL2RGB(image)
        converted = self.__RGB2GRAY(image)
        return converted