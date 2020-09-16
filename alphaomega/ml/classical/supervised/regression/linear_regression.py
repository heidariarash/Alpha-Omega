import numpy as np

class LinearRegression:
    """
    Linear regression is a supervised regression algorithm.
    """
    def __init__(self):
        """
        Usage  : The constructor of LinearRegression class. 
        Inputs : Nothing
        Returns: An Instantiation of the class.
        """
        self.__W = np.array([])
        self.__bias = False
        self.__regularizator = None
        self.__penalizer = 0.1

    def config(self, **kwargs):
        """
        Usage: use this method to configure the parameters of the LinearRegression instantiation.

        Inputs:
            regularizator : if "no_reg", no regularization is used. Other option is "ridge".
            penalizer     : penalizer is the regularization coefficent. It is ignored in the case of no regularization.
            bias          : If the bias is present in your features, flag this input as True, otherwise False.

        Returns: Nothing
        """
        for key, value in kwargs.items():
            if key == "bias":
                self.__bias = value
            elif key == "regularizator":
                self.__regularizator = value
            elif key == "penalizer":
                self.__penalizer = value

    def get(self, attribute):
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

        print("The specified attribute is not valid. Acceptable attributes are 'wights' and 'intercept'")

    def train(self, train_features, labels):
        """
        Usage  : Use this method to train the LinearRegression model. The trained parameteres are:
            weights  : The weights of the model
            intercept: The bias of the model

        Inputs :
            train_features: The feature matrix used to train the model.
            labels        : The corresponding labels for each data point in data.
            
        Returns: Nothing
        """
        #cheking if the shape of the input is correct.
        if (len(train_features.shape) != 2):
            print("please provide featuers as a tabular shape: 2 dimensional. rows are samples and colums are features.")
            return
        
        labels.reshape(-1, 1)
        data_process = train_features.copy()
        
        #checking if bias is present in feature maps
        if not self.__bias:
            bias_input = np.ones((train_features.shape[0], 1))
            data_process = np.concatenate((bias_input, data_process), axis=1)
        
        #calculating the weights matrix
        if self.__regularizator == "no_reg":
            self.__W = np.linalg.inv(data_process.transpose() @ data_process) @ data_process.transpose() @ labels
        elif self.__regularizator == "ridge":
            L = np.zeros((data_process.shape[1], data_process.shape[1]))
            L[0,0] = 1
            L = np.eye(data_process.shape[1]) - L
            self.__W = np.linalg.inv(data_process.transpose() @ data_process + self.__penalizer * L) @ data_process.transpose() @ labels
        
    def apply(self, test_features, bias = False):
        """
        Usage: Use this method to evalute your LinearRegression Model on test data.

        Inputs:
            test_features: test data to be evaluated
            bias         : If the bias is present in your features, flag this input as True, otherwise False.

        Returns:
            - a numpy array which contains the labels for each data point in test data.

        """
        #cheking if the shape of the input is correct.
        if (len(test_features.shape) != 2):
            print("please provide featuers as a tabular shape: 2 dimensional. rows are samples and colums are features.")
            return
        
        data_process = test_features.copy()
        #checking if bias is present in feature maps
        if not bias:
            #checking if the features counts is correct
            if test_features.shape[1] != len(self.weights):
                print("Number of features should be the same as training data.")
                return
            
            bias_input = np.ones((test_features.shape[0], 1))
            data_process = np.concatenate((bias_input, data_process), axis=1)
        else:
            #checking if the features counts is correct
            if test_features.shape[1] != len(self.__W.reshape(-1)):
                print("Number of features should be the same as training data.")
                return
            
        predict = data_process @ self.__W
        return predict.reshape(-1)

def linear_regression_train(train_features, labels, regularizator = None, penalizer = 0.1, bias = False):
    """
    Usage  : Use this function to train a linear regression model. You can use apply_linear_regression_func to predict the labels for test set with the help of the output of this funcion.

    Inputs :
        train_features: The feature matrix used to train the model.
        labels        : The corresponding labels for each data point in data.
        reqularizator : if None, no regularization is used. Other option is "ridge".
        penalizer     : penalizer is the regularization coefficent. It is ignored in the case of no regularization.
        bias          : If the bias is present in your features, flag this input as True, otherwise False.

    Returns: 
        - The weights matrix (bias included).
    """
    #cheking if the shape of the input is correct.
    if (len(train_features.shape) != 2):
        print("please provide featuers as a tabular shape: 2 dimensional. rows are samples and colums are features.")
        return

    labels.reshape(-1, 1)
    data_process = train_features.copy()

    #checking if bias is present in feature maps
    if not bias:
        bias_input = np.ones((train_features.shape[0], 1))
        data_process = np.concatenate((bias_input, data_process), axis=1)

    #calculating the weights matrix
    if not regularizator:
        weights = np.linalg.inv(data_process.transpose() @ data_process) @ data_process.transpose() @ labels
    elif regularizator == "ridge":
        L = np.zeros((data_process.shape[1], data_process.shape[1]))
        L[0,0] = 1
        L = np.eye(data_process.shape[1]) - L
        weights = np.linalg.inv(data_process.transpose() @ data_process + penalizer * L) @ data_process.transpose() @ labels

    return weights

def linear_regression_apply(test_features, weights, bias = False):
    """
    Usage: Use this function to evalute your linear regression model on test data.

    Inputs:
        test_features: test data to be evaluated.
        weights      : The weights matrix to be applied to test data. Use train_linear_regression_func function to obtain this variable.
        bias         : If the bias is present in your features, flag this input as True, otherwise False.

    Returns:
        - a numpy array which contains the labels for each data point in test data.

    """
    #cheking if the shape of the input is correct.
    if (len(test_features.shape) != 2):
        print("please provide featuers as a tabular shape: 2 dimensional. rows are samples and colums are features.")
        return

    data_process = test_features.copy()
    #checking if bias is present in feature maps
    if not bias:
        #checking if the features counts is correct
        if test_features.shape[1] != (len(weights.reshape(-1))-1):
            print("Number of features should be the same as training data.")
            return

        bias_input = np.ones((test_features.shape[0], 1))
        data_process = np.concatenate((bias_input, data_process), axis=1)
    else:
        #checking if the features counts is correct
        if test_features.shape[1] != len(weights.reshape(-1)):
            print("Number of features should be the same as training data.")
            return

    predict = data_process @ weights
    return predict.reshape(-1)