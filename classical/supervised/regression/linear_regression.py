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
        self.weights = np.array([])
        self.intercept = 0

    def train(self, data, label):
        """
        Usage  : Use this method to train the LinearRegression model. The trained parameteres are:
            weights  : The weights of the model 
            intercept: The bias of the model

        Inputs :
            data : The feature matrix used to train the model.
            label: The corresponding labels for each data point in data.

        Returns: Nothing
        """
        data_process = data.copy()
        bias_input = np.ones((data.shape[0], 1))
        data_process = np.concatenate((bias_input, data_process), axis=1)
        self.__W = np.linalg.inv(data_process.transpose() @ data_process) @ data_process.transpose() @ label
        self.weights = self.__W[1:]
        self.intercept = self.__W[0]

    def apply(self, data):
        """
        Usage: Use this method to evalute your LinearRegression Model on test data.

        Inputs:
            data: test data to be evaluated

        Returns:
            - a numpy array which contains the labels for each data point in test data.

        """
        data_process = data.copy()
        bias_input = np.ones((data.shape[0], 1))
        data_process = np.concatenate((bias_input, data_process), axis=1)
        return data_process @ self.__W

        #to be done:
            #adding regularization terms.
            #checking for true data shape.
            #checking if bias is already present.

# lr = LinearRegression()
# x = np.array([[1],
#               [2],
#               [3],
#               [4]])

# y = np.array([[4],
#               [7],
#               [10],
#               [13]])

# lr.train(x, y)
# print(lr.apply(np.array([[5],
#                          [6]])))
