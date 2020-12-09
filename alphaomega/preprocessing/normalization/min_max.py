from typing import Union
import numpy as np
from alphaomega.utils.exceptions import WrongDimension, WrongAttribute

class MinMaxNormalizer:
    """
    You can use MinMaxNormalizer to normalize your data between 0 and 1.
    """
    def __init__(self) -> None:
         self.__minimum = np.array([])
         self.__maximum = np.array([])
         self.__columns = None
         self.__shape = 0

    def config(self, **kwargs: dict) -> None:
        """
        Usage: use this method to configure the parameters of the MinMaxNormalizer instantiation.

        Inputs:
            columns: a list which determines which featuers should be normalized. If it is None or empty, it means to normalize all the features.

        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "columns":
                self.__columns = value

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
            minimum: a numpy array which contains the maximum of the train features for each column.
            maximum: a numpy array which contains the minimum of the train features for each column.

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
            
        #calculation minimum and maximum of each feature.
        self.__maximum = np.max(data_process,axis = 0)
        self.__minimum = np.min(data_process, axis = 0)
        
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Usage  : Use this method to transform your data to normalized ones.

        Inputs :
            data: data to be normalized.
            
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
            raise WrongDimension("Number of data features (dimensions) should be the same as the training data features.")
        
        data_process = data.copy()
        
        #checking if all columns should be normalized. If self.__columns is None, all columns will be normalized.
        if not self.__columns:
            return (data_process - self.__minimum) / (self.__maximum - self.__minimum)
        
        #if only some columns should get normalized, we do it with the next command.
        data_process[: ,self.__columns] = (data_process[: ,self.__columns] - self.__minimum) / (self.__maximum - self.__minimum)
        return data_process


def min_max_normalizer_train(train_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Usage: Use this functin to obtain the parameters of min_max_normalizer.

    Inputs:
        train_data: The features to get the statistics from. It's equivalent to training data.

    Returns:
        - The parameters of min_max_normalizer. The output of this function is one of the inputs of min_max_normalizer_apply function.
    """
    #checking if the shape of features is correct
    if (len(train_data.shape) != 2):
        raise WrongDimension("Only tabular data is acceptable (i.e. 2 dimensional).")
        return

    #calculation minimum and maximum of each feature.
    maximum = np.max(train_data,axis = 0)
    minimum = np.min(train_data, axis = 0)

    return maximum, minimum


def min_max_normalizer_apply(data: np.ndarray, normalizer_params: tuple[np.ndarray, np.ndarray], columns: Union[List, None] = None) -> np.ndarray:
    """
    Usage  : Use this function to transform your data to normalized ones between 0 and 1.

    Inputs :
        data             : data to be normalized. This is all your data (including train and test), or you can use this function twice. Once with training data as this parameter. Once with test data as this parameter.
        normalizer_params: The parameters of min_max_normalizer. You can obtain these parameters by using min_max_normalizer_train function.
        columns          : an array which determines which featuers should be normalized. If it is None, it means to normalize all the data.
    
    Returns: 
        - a numpy array, where:
            1. The columns marked to be normalized are normalized.
            2. The columns not marked to be normalized are untouched.
    """
    maximum, minimum = normalizer_params

    #checking if the shape of data is correct
    if (len(data.shape) != 2 ):
        raise WrongDimension("Only tabular data is acceptable (i.e. 2 dimensional).")
    
    #checking if number of data in training data and data to be normalized are equal.
    if (len(mean.shape) != data.shape[1]):
        raise WrongDimension("Number of data features (dimensions) to be normalized should be equal to the number of training data features.")
        
    #checking if all columns should be normalized. If columns is None, all columns will be normalized.
    if not columns:
        return (data - minimum) / (maximum - minimum)

    scaled_data = data.copy()
    #if only some columns should get normalized, we do it with the next command.
    scaled_data[: ,columns] = (data[: ,columns] - minimum) / (maximum - minimum)
    return scaled_data