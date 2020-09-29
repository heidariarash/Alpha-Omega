import numpy as np

class ZScoreNormalizer:
    """
    You can use ZScoreNormalizer to normalize your data such as normalize data has a mean of zero and a standard devaiation of one.
    """
    def __init__(self):
         """
        Usage  : The constructor of ZScoreNormalizer class. 
        Inputs : Nothing
        Returns: An Instantiation of the class.
        """
         self.__mean = np.array([])
         self.__std = np.array([])
         self.__columns = None
         self.__shape = 0

    def config(self, **kwargs):
        """
        Usage: use this method to configure the parameters of the MinMaxNormalizer instantiation.

        Inputs:
            columns: a list which determines which featuers should be normalized. If it is None or empty, it means to normalize all the features.

        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "columns":
                self.__columns = value

    def get(self, attribute):
        """
        Usage: Use this method to get the attribute of interest.

        Inputs:
            attribute: The attribute of interest. It could be "std" or "mean".

        Returns: The desired attribute
        """
        if attribute == "mean":
            return self.__mean

        if attribute == "std":
            return self.__std

        print("The specified attribute is not valid. Acceptable attributes are 'mean', and 'std'")
        
    def train(self, train_features):
        """
        Usage  : Use this method to train the parameters of MinMaxNormalizer model. The trained parameteres are:
            mean: a numpy array which contains the mean of the train features for each column.
            std : a numpy array which contains the standard deviation of the train features for each column.

        Inputs :
            train_features: The feature matrix used to train the model.
            columns       : an array which determines which featuers should be normalized. If it is None, it means to normalize all the features.

        Returns: Nothing
        """
        #checking for the correct shape of train_features
        if len(train_features.shape) != 2:
            print("Only tabular data is acceptable.")
            return
        
        #storing the number of featurs for the apply function
        self.__shape = train_features.shape[1]
        
        #checking for the requested columns to be normalized. If None, all features will normalize.
        if self.__columns:
            data_process = train_features[:,self.__columns].copy()
        else:
            data_process = train_features.copy()
            
        #calculation minimum and maximum of each feature.
        self.__mean = np.mean(data_process,axis = 0)
        self.__std = np.std(data_process, axis = 0)
        
    def apply(self, features):
        """
        Usage  : Use this method to transform your features to normalized ones.

        Inputs :
            features: Features to be normalized.
            
        Returns: 
            - a numpy array, where:
                1. The columns marked to be normalized in train method are normalized.
                2. The columns not marked to be normalized are untouched.
        """
        #checking for the correct shape of the featuers.
        if len(features.shape) != 2:
            print("Only tabular data is acceptable.")
            return
        
        #checking if the number of features is exactly the same as the number of train_features.
        if self.__shape != features.shape[1]:
            print("Number of features (dimensions) should be the same as the training data.")
            return
        
        data_process = features.copy()
        
        #checking if all columns should be normalized. If self.__columns is None, all columns will be normalized.
        if not self.__columns:
            return (data_process - self.__mean) / (self.__std)
        
        #if only some columns should get normalized, we do it with the next command.
        data_process[: ,self.__columns] = (data_process[: ,self.__columns] - self.__mean) / (self.__std)
        return data_process

        

def z_score_normalizer_train(train_features):
    """
    Usage: Use this function to obtain the parameters of z_score_normalizer.

    Inputs:
        train_features: The features to get the statistics from. It's equivalent to training data.

    Returns:
        - The z_score_normalizer parameters. This output is one of the inputs of z_score normalizer_apply functino.
    """
    #checking if the shape of features are correct
    if len(train_features.shape) != 2:
        print("Only tabular data is acceptable (e.g. 2 dimensional).")
        return

    #calculation minimum and maximum of each feature.
    mean = np.mean(train_features,axis = 0)
    std = np.std(train_features, axis = 0)

    return mean, std


def z_score_normalizer_apply(features, normalizer_params, columns = None):
    """
    Usage  : Use this function to transform your features to normalized ones such as normalize data has a mean of zero and a standard devaiation of one.

    Inputs :
        features         : Features to be normalized. This is all your features (including train and test), or you can use this function twice. Once with training data as this parameter. Once with test data as this parameter.
        normalizer_params: The paramters of z score normalizer. You can obtain these parameters by using z_score_normalizer_train function.
        columns          : an array which determines which featuers should be normalized. If it is None, it means to normalize all the features.
    
    Returns: 
        - a numpy array, where:
            1. The columns marked to be normalized are normalized.
            2. The columns not marked to be normalized are untouched.
    """
    mean, std = normalizer_params

    #checking if the shape of features are correct
    if len(features.shape) != 2:
        print("Only tabular data is acceptable (e.g. 2 dimensional).")
        return
    
    #checking if number of features in training data and data to be normalized are equal.
    if (len(mean.shape) != features.shape[1]):
        print("Number of features to be normalized should be equal to the number of training features.")
        return
        
    #checking if all columns should be normalized. If columns is None, all columns will be normalized.
    if not columns:
        return (featuers - mean) / (std)

    scaled_featuers = feateurs.copy()
    #if only some columns should get normalized, we do it with the next command.
    scaled_featuers[: ,columns] = (featuers[: ,columns] - mean) / (std)
    return scaled_featuers