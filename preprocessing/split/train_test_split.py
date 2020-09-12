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
        for key, value in kwargs.items():
            if (key == "random_state"):
                self.__random_state = value
            elif key == "validation":
                self.__validation = value
            elif key == "test_rate":
                self.__test_rate = value
                if (self.__test_rate < 0 or self.__test_rate > 1.0):
                    print("test_rate could not be less than zero or greater than 1. test_rate reseted to 0.3")
                    self.__test_rate = 0.3
            elif key == "validation_rate":
                self.__validation_rate = value
            
        if (self.__validation):
            if (self.__validation_rate < 0 or self.__validation_rate > 1.0):
                print("validation_rate could not be less than zero or greater than 1. validarion_rate reseted to 0.2")
                self.__validation_rate = 0.2

        if (self.__validation_rate + self.__test_rate > 1.0 and self.__validation_rate):
            print("validation_rate + test_rate could not be greater than 1. test and validatoin rate reseted to 0.3 and 0.2 respectively.")
            self.__test_rate = 0.3
            self.__validation_rate = 0.2

    def train(self, features):
        """
        Usage : You can use this method to randomly split the data between train, test(, and validation) sets. Trained parameteres are:
            test_idx       : the index of the data which belongs to the test set.
            validation_idx : the index of the data which belongs to the validation set.
            train_idx      : the index of the data which belongs to the validation set.

        Inputs:
            features       : The features of the original dataset.
            labels         : The labels of the original dataset.

        Returns: Nothing.
        """
        if (self.__random_state):
            np.random.seed(self.__random_state)
            
        self.test_idx = np.random.choice(list(range(features.shape[0])), size=int(self.__test_rate * features.shape[0]), replace=False)
        self.train_idx = np.array(list(set(range(features.shape[0])) - set(self.test_idx)))
        if self.__validation:
            self.validation_idx = np.random.choice(self.train_idx, size=int((self.__validation_rate) / (1-self.__test_rate) * self.train_idx.shape[0]), replace=False)
            self.train_idx = np.array(list(set(self.train_idx) - set(self.validation_idx)))

    def apply(self, featuers, part = "train"):
        """
        Usage: Use this method to extract the desired part of the dataset.

        Inputs:
            features: You extract the desired part of this numpy array.
            part    : This input determines which part to extract. default is "train". Other options are "test" and "validation"

        Returns:
            - a numpy array which is extracted from the features.
        """
        if len(featuers.shape) == 1 or featuers.shape[0] == 1:
            data_process = featuers.reshape(-1, 1).copy()
        else:
            data_process = featuers.copy()
        if part == "train":
            return data_process[self.train_idx]
        if part == "test":
            return data_process[self.test_idx]
        if part == "validation" and self.__validation_rate:
            return data_process[self.validation_idx]

        print("Please specify the part parameter correctly. It could be only 'train', 'test' or if validation is enabled 'validation'.")