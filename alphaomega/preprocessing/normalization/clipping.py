from typing import Union
import numpy as np
from alphaomega.utils.exceptions import WrongAttribute, WrongDimension

class ClippingNormalizer:
    """
    You can use ClippingNormalizer to normalize your data such as it does not be more or less than some certain percentile.
    """
    def __init__(self) -> None:
         self.__maximum = np.array([])
         self.__minimum = np.array([])
         self.__columns = []
         self.__shape = 0
         self.__max_percentile = 90
         self.__min_percentile = 10
         self.__columns = []

    def config(self, **kwargs) -> None:
        """
        Usage: use this method to configure the parameters of ClippingNormalizer instantiation.

        Inputs:
            max_percentile: maximum percentile to keep. values greater than max_percentile will be clipped to this value.
            min_percentile: minimum percentile to keep. values less than min_percentile will be clipped to this value.
            columns       : a list which determines which featuers should be normalized. If it is empty, it means to normalize all the features.

        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "max_percentile":
                self.__max_percentile = value
                if (value > 100):
                    print("max_percentile should be less 100. It reseted to 90.")
                    self.__max_percentile = 90
            elif key == "min_percentile":
                self.__min_percentile = value
                if (value < 0):
                    print("max_percentile should be greater than 0. It reseted to 10.")
                    self.__min_percentile = 10
            elif key == "columns":
                self.__columns = list(set(value))

        if (self.__max_percentile < self.__min_percentile):
            print("max percentile could not be less than min percentile. Max and Min percentile reseted to 90 and 10 respectively.")
            self.__min_percentile = 10
            self.__max_percentile = 90

    def get(self, attribute: str) -> np.ndarray:
        """
        Usage: Use this method to get the attribute of interest.

        Inputs:
            attribute: The attribute of interest. It could be "maximum" or "minimum".

        Returns: The desired attribute
        """
        if attribute == "maximum":
            return self.__maximum

        if attribute == "minimum":
            return self.__minimum

        raise WrongAttribute("The specified attribute is not valid. Acceptable attributes are 'maximum', and 'minimum'")

    def train(self, train_data: np.ndarray) -> None:
        """
        Usage  : Use this method to train the parameters of MinMaxNormalizer model. The trained parameteres are:
            maximum: a numpy array which contains the mean of the train features for each column.
            minimum : a numpy array which contains the standard deviation of the train features for each column.

        Inputs :
            train_data: The feature matrix used to train the model.

        Returns: Nothing
        """      
        #checking for the correct shape of train_data
        if len(train_data.shape) != 2:
            raise WrongDimension("Only tabular data is acceptable.")
        
        #storing the number of featurs for the apply function
        self.__shape = train_data.shape[1]
        
        #checking for the requested columns to be normalized. If None, all features will normalize.
        if self.__columns:
            data_process = train_data[:,self.__columns].copy()
        else:
            data_process = train_data.copy()
            self.__columns = list(range(train_data.shape[1]))
            
        #calculation minimum and maximum of each feature.
        self.__maximum = np.percentile(data_process, self.__max_percentile, axis = 0)
        self.__minimum = np.percentile(data_process, self.__min_percentile, axis = 0 )
        
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Usage  : Use this method to transform your data to normalized ones.

        Inputs :
            data: Data to be normalized.
            
        Returns: 
            - a numpy array, where:
                1. The columns marked to be normalized in train method are normalized.
                2. The columns not marked to be normalized are untouched.
        """
        #checking for the correct shape of the featuers.
        if len(data.shape) != 2:
            raise WrongDimension("Only tabular data is acceptable.")
        
        #checking if the number of data is exactly the same as the number of train_data.
        if self.__shape != data.shape[1]:
            raise WrongDimension("Number of data features (dimensions) should be the same as the training data features."
        
        data_process = data.copy()
        
        for column in range(data.shape[1]):
            if column not in self.__columns:
                continue
            for row in range(data.shape[0]):
                if data_process[row ,column] > self.__maximum[self.__columns.index(column)]:
                    data_process[row, column] = self.__maximum[self.__columns.index(column)]
                elif data_process[row, column] < self.__minimum[self.__columns.index(column)]:
                    data_process[row, column] = self.__minimum[self.__columns.index(column)]
            
        return data_process


def clipping_normalizer_train(train_data: np.ndarray, min_percentile: int = 10, max_percentile: int = 90) -> tuple[np.ndarray, np.ndarray]:
    """
    Usage: Use this function to obtain the clipping_normalizer parameters.

    Inputs:
        train_data: The feature matrix used to extract the statistics.
        max_percentile: maximum percentile to keep. values greater than max_percentile will be clipped to this value.
        min_percentile: minimum percentile to keep. values less than min_percentile will be clipped to this value.

    Returns:
        - - The parameters of clipping_normalizer. The output of this function is one of the inputs of clipping_normalizer_apply function.
    """
    #checking if the range of min and max percentile is correct.
    if (max_percentile > 100 or min_percentile < 0 or max_percentile <= min_percentile):
        raise WrongAttribute("min_percentile should be less than max_percentile and both need to be between 0 and 100")

    #checking if the shape of features are correct
    if (len(train_data.shape) != 2):
        raise WrongDimension("Only tabular data is acceptable (i.e. 2 dimensional).")

    #calculation minimum and maximum of each feature.
    maximum = np.percentile(train_data, max_percentile, axis = 0)
    minimum = np.percentile(train_data, min_percentile, axis = 0 )

    return maximum, minimum


def clipping_normalizer_apply(data: np.ndarray, normalizer_params: tuple[np.ndarray, np.ndarray], columns: Union[List, None] = None) -> np.ndarray:
    """
    Usage  : Use this function to transform your data to normalized ones such as it does not be more or less than some certain percentile.

    Inputs :
        data             : data to be normalized. This is all your data (including train and test), or you can use this function twice (train and test data seperately).
        normalizer_params: The parameters of clipping_normalizer. You can obtain these parameters by using clipping_normalizer_train function.
        columns          : an array which determines which featuers should be normalized. If it is None, it means to normalize all the data.

    Returns:
        - a numpy array, where:
            1. The columns marked to be normalized in train method are normalized.
            2. The columns not marked to be normalized are untouched.
    """
    maximum, minimum = normalizer_params

    #checking if the shape of data are correct
    if (len(data.shape) != 2):
        raise WrongDimension("Only tabular data is acceptable (i.e. 2 dimensional).")
    
    #checking if number of data in training data and data to be normalized are equal.
    if (len(maximum.shape) != data.shape[1]):
        raise WrongDimension("Number of data features (dimensions) to be normalized should be equal to the number of training data features.")
    
    scaled_data = data.copy()
        
    for column in range(data.shape[1]):
        if column not in columns:
            continue
        for row in range(data.shape[0]):
            if scaled_data[row ,column] > maximum[columns.index(column)]:
                scaled_data[row, column] = maximum[columns.index(column)]
            elif scaled_data[row, column] < minimum[columns.index(column)]:
                scaled_data[row, column] = minimum[columns.index(column)]

    return scaled_data