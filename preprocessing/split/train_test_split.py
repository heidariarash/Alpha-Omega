import numpy as np

class TrainTestSplit:
    """
    You can use TrainTestSplit class to split your data to training and testing sets (and if required, calidation set).
    """
    def __init__(self):
        """
        Usage  : Constructor of the TrainTestSplit class.
        Inputs : Nothing.
        Returns: An instantiation of the TrainTestSplit class.
        """
        self.test_idx = np.array([])
        self.validation_idx = np.array([])
        self.train_idx = np.array([])
        self.train_data = np.array([])
        self.test_data = np.array([])
        self.validation_data = np.array([])
        self.__test_rate = 0.3
        self.__validation = False
        self.__validation_rate = 0.2
        self.__random_state = None

    def cofig(self, **kwargs):
        """
        Usage: Use this method to configure the TrainTestSplit instantiation.

        Inputs:
            test_rate      : The percentage of the data that should belong to the test set.
            validation     : If true, it means you also need validation set.
            validation_rate: If validation is true, this parameter shows which percentage of the data belongs to the validation set. It will be ignored if validation is False.
            random_state   : The state of random initializer.

        Returns: Nothing.
        """
        if (kwargs["random_state"] is not None):
            self.__random_state = kwargs["random_state"]
            
        if (kwargs["validation"] is not None):
            self.__validation = kwargs["validation"]

        #checking if the test_rate and validation_rate is in the correct range.
        if (kwargs["test_rate"] is not None):
            self.__test_rate = kwargs["test_rate"]
            if (self.__test_rate < 0 or self.__test_rate > 1.0):
                print("test_rate could not be less than zero or greater than 1. test_rate reseted to 0.3")
                self.__test_rate = 0.3
                return
            

        if (self.__validation and kwargs["validation_rate"] is not None):
            self.__validation_rate = kwargs["validarion_rate"]
            if (self.__validation_rate < 0 or self.__validation_rate > 1.0):
                print("validation_rate could not be less than zero or greater than 1. validarion_rate reseted to 0.2")
                self.__validation_rate = 0.2
                return
            

        if (self.__validation_rate + self.__test_rate > 1.0):
            print("validation_rate + test_rate could not be greater than 1. test and validatoin rate reseted to 0.3 and 0.2 respectively.")
            self.__test_rate = 0.3
            self.__validation_rate = 0.2
            return

    def train(self, features, labels):
        """
        Usage : You can use this method to randomly split the data between train, test(, and validation) sets. Trained parameteres are:
            test_idx       : the index of the data which belongs to the test set.
            validation_idx : the index of the data which belongs to the validation set.
            train_idx      : the index of the data which belongs to the validation set.
            train_data     : The training set
            test_data      : The test set
            validation_data: The validation set.

        Inputs:
            features       : The features of the original dataset.
            labels         : The labels of the original dataset.

        Returns: Nothing.
        """
        