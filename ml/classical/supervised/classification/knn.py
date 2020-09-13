import numpy as np

class KNN:
    """
    Usage: KNN(k-neareset neighbors) is a machine learning algorithm used for classification.
    """
    def __init__(self):
        """
        Usage  : The constructor of KNN class. 
        Inputs : Nothing
        Returns: An Instantiation of the class.
        """
        self.__k = 1
        self.__features = np.array([])
        self.__labels = np.array([])
    
    def config(self, **kwargs):
        """
        Usage  : Use this method to configure the parameteres of the KNN instantiation.

        Inputs : 
            k  : The number of nearest neighbors.

        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "k":
                self.__k = np.floor(value)
                if np.floor(value) < 1:
                    print("k cannot be less than 1. It reseted to 1.")
                    self.__k = 1

    def train(self, features, labels):
        """
        Usage  : Use this method to train the KNN model.

        Inputs :
            features: The features of the training set.
            labels  : The labels of the training set.

        Returns: Nothing.
        """
