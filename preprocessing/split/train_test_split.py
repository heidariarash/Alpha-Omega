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

    def train(self, features, labels, test_rate = 0.3, validation = False, validation_rate = 0.2):
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
            test_rate      : The percentage of the data that should belong to the test set.
            validation     : If true, it means you also need validation set.
            validation_rate: If validation is true, this parameter shows which percentage of the data belongs to the validation set.
            random_state   : The state of random initializer.

        Returns: Nothing.
        """
        #checking if the test_rate and validation_rate is in the correct range.
        if (test_rate < 0 or test_rate > 1.0):
            print("test_rate could not be less than zero or greater than 1.")
            return

        if (validation_rate < 0 or validation_rate > 1.0):
            print("validation_rate could not be less than zero or greater than 1.")
            return

        if (validation_rate + test_rate > 1.0):
            print("validation_rate + test_rate could not be greater than 1.")
            return