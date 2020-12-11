import numpy as np
from alphaomega.utils.exceptions import WrongDimension, WrongAttribute
from typing import Union

class LinearRegression:
    """
    Linear regression is a supervised regression algorithm.
    """
    def __init__(self):
        self.__W             = np.array([])
        self.__bias          = False
        self.__regularizator = None
        self.__penalizer     = 0.1

    def config(self, **kwargs) -> None:
        """
        Usage: use this method to configure the parameters of the LinearRegression instantiation.

        Inputs:
            regularizator: if "no_reg", no regularization is used. Other option is "ridge".
            penalizer    : penalizer is the regularization coefficent. It is ignored in the case of no regularization.
            bias         : If the bias is present in your features, flag this input as True, otherwise False.

        Returns: Nothing
        """
        for key, value in kwargs.items():

            if key == "bias":
                self.__bias = value

            elif key == "regularizator":
                if value not in ["no_reg", "ridge"]:
                    raise WrongAttribute("regularizator can only be 'no_reg' or 'ridge'.")
                self.__regularizator = value

            elif key == "penalizer":
                self.__penalizer = value

    def get(self, attribute: str) -> Union[np.ndarray, float]:
        """
        Usage: Use this method to get the attribute of interest.

        Inputs:
            attribute: The attribute of interest. It could be "weights" or "intercept".

        Returns: The desired attribute
        """
        if attribute == "intercept":
            return self.__W[0][0]

        if attribute == "weights":
            return self.__W[1:].reshape(-1)

        raise WrongAttribute("The specified attribute is not valid. Acceptable attributes are 'wights' and 'intercept'")

    def train(self, train_data: np.ndarray, labels: np.ndarray) -> None:
        """
        Usage  : Use this method to train the LinearRegression model. The trained parameteres are:
            weights  : The weights of the model
            intercept: The bias of the model

        Inputs :
            train_data: The feature matrix used to train the model.
            labels    : The corresponding labels for each data point in data.
            
        Returns: Nothing
        """
        #cheking if the shape of the input is correct.
        if (len(train_data.shape) != 2):
            raise WrongDimension("please provide featuers as a tabular shape: 2 dimensional. rows are samples and colums are features.")
        
        labels.reshape(-1, 1)
        data_process = train_data.copy()
        
        #checking if bias is present in feature maps
        if not self.__bias:
            bias_input   = np.ones((train_data.shape[0], 1))
            data_process = np.concatenate((bias_input, data_process), axis=1)
        
        #calculating the weights matrix
        if self.__regularizator == "no_reg":
            self.__W = np.linalg.inv(data_process.transpose() @ data_process) @ data_process.transpose() @ labels

        elif self.__regularizator == "ridge":
            L        = np.zeros((data_process.shape[1], data_process.shape[1]))
            L[0,0]   = 1
            L        = np.eye(data_process.shape[1]) - L
            self.__W = np.linalg.inv(data_process.transpose() @ data_process + self.__penalizer * L) @ data_process.transpose() @ labels
        
    def apply(self, test_data: np.ndarray, bias: bool = False) -> np.ndarray:
        """
        Usage: Use this method to evalute your LinearRegression Model on test data.

        Inputs:
            test_data: test data to be evaluated
            bias     : If the bias is present in your features, flag this input as True, otherwise False.

        Returns:
            - a numpy array which contains the labels for each data point in test data.

        """
        #cheking if the shape of the input is correct.
        if (len(test_data.shape) != 2):
            raise WrongDimension("please provide featuers as a tabular shape: 2 dimensional. rows are samples and colums are features.")
        
        data_process = test_data.copy()
        #checking if bias is present in feature maps
        if not bias:
            #checking if the features counts is correct
            if test_data.shape[1] != len(self.weights):
                raise WrongDimension("Number of features should be the same as training data.")
            
            bias_input   = np.ones((test_data.shape[0], 1))
            data_process = np.concatenate((bias_input, data_process), axis=1)

        else:
            #checking if the features counts is correct
            if test_data.shape[1] != len(self.__W.reshape(-1)):
                raise WrongDimension("Number of features should be the same as training data.")
            
        predict = data_process @ self.__W
        return predict.reshape(-1)

def linear_regression_train(train_data: np.ndarray, labels: np.ndarray, regularizator: str = "no-reg", penalizer: float = 0.1, bias: bool = False) -> np.ndarray:
    """
    Usage  : Use this function to train a linear regression model. You can use apply_linear_regression_func to predict the labels for test set with the help of the output of this funcion.

    Inputs :
        train_data   : The feature matrix used to train the model.
        labels       : The corresponding labels for each data point in data.
        reqularizator: if "no_reg", no regularization is used. Other option is "ridge".
        penalizer    : penalizer is the regularization coefficent. It is ignored in the case of no regularization.
        bias         : If the bias is present in your features, flag this input as True, otherwise False.

    Returns: 
        - The weights matrix (bias included).
    """
    #cheking if the shape of the input is correct.
    if (len(train_data.shape) != 2):
        raise WrongDimension("please provide featuers as a tabular shape: 2 dimensional. rows are samples and colums are features.")

    labels.reshape(-1, 1)
    data_process = train_data.copy()

    #checking if bias is present in feature maps
    if not bias:
        bias_input   = np.ones((train_data.shape[0], 1))
        data_process = np.concatenate((bias_input, data_process), axis=1)

    #calculating the weights matrix
    if regularizator == "no-reg":
        weights = np.linalg.inv(data_process.transpose() @ data_process) @ data_process.transpose() @ labels

    elif regularizator == "ridge":
        L = np.zeros((data_process.shape[1], data_process.shape[1]))
        L[0,0] = 1
        L = np.eye(data_process.shape[1]) - L
        weights = np.linalg.inv(data_process.transpose() @ data_process + penalizer * L) @ data_process.transpose() @ labels

    else:
        raise WrongAttribute("regularizator can only be 'no_reg or 'ridge'.")

    return weights

def linear_regression_apply(test_data: np.ndarray, weights: np.ndarray, bias: bool = False):
    """
    Usage: Use this function to evalute your linear regression model on test data.

    Inputs:
        test_data: test data to be evaluated.
        weights  : The weights matrix to be applied to test data. Use train_linear_regression_func function to obtain this variable.
        bias     : If the bias is present in your features, flag this input as True, otherwise False.

    Returns:
        - a numpy array which contains the labels for each data point in test data.
    """
    #cheking if the shape of the input is correct.
    if (len(test_data.shape) != 2):
        raise WrongDimension("please provide featuers as a tabular shape: 2 dimensional. rows are samples and colums are features.")

    data_process = test_data.copy()
    #checking if bias is present in feature maps
    if not bias:
        #checking if the features counts is correct
        if test_data.shape[1] != (len(weights.reshape(-1))-1):
            raise WrongDimension("Number of features should be the same as training data.")

        bias_input   = np.ones((test_data.shape[0], 1))
        data_process = np.concatenate((bias_input, data_process), axis=1)
    else:
        #checking if the features counts is correct
        if test_data.shape[1] != len(weights.reshape(-1)):
            raise WrongDimension("Number of features should be the same as training data.")

    predict = data_process @ weights
    return predict.reshape(-1)