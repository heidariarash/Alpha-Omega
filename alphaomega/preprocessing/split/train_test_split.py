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
        self.__test_idx = np.array([])
        self.__validation_idx = np.array([])
        self.__train_idx = np.array([])
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

    def get(self, attribute):
        """
        Usage: Use this method to get the attribute of interest.

        Inputs:
            attribute: The attribute of interest. It could be "train_idx" or "test_idx", or "validation_idx".

        Returns: The desired attribute
        """
        if attribute == "train_idx":
            return self.__train_idx

        if attribute == "test_idx":
            return self.__test_idx

        if attribute == "validation_idx":
            return self.__validation_idx

        print("The specified attribute is not valid. Acceptable attributes are 'maximum', and 'minimum'")

    def train(self, count):
        """
        Usage : You can use this method to randomly split the data between train, test(, and validation) sets. Trained parameteres are:
            test_idx       : the index of the data which belongs to the test set.
            validation_idx : the index of the data which belongs to the validation set.
            train_idx      : the index of the data which belongs to the validation set.

        Inputs:
            count : The number of data points in your dataset.
            labels: The labels of the original dataset.

        Returns: Nothing.
        """
        if (self.__random_state):
            np.random.seed(self.__random_state)
            
        self.__test_idx = np.random.choice(list(range(count)), size=int(self.__test_rate * count), replace=False)
        self.__train_idx = np.array(list(set(range(count)) - set(self.__test_idx)))
        if self.__validation:
            self.__validation_idx = np.random.choice(self.__train_idx, size=int((self.__validation_rate) / (1-self.__test_rate) * self.__train_idx.shape[0]), replace=False)
            self.__train_idx = np.array(list(set(self.__train_idx) - set(self.__validation_idx)))

        np.random.seed(None)

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
            return data_process[self.__train_idx]
        if part == "test":
            return data_process[self.__test_idx]
        if part == "validation" and self.__validation_rate:
            return data_process[self.__validation_idx]

        print("Please specify the part parameter correctly. It could be only 'train', 'test' or if validation is enabled 'validation'.")


def train_test_split(count, test_rate = 0.3, validation = False, validation_rate = 0.2, random_state = None):
    """
    Usage: Use this function to split your data into train and test (and if needed validation) sets.

    Inputs:
        count: The number of data point in your original dataset.
        test_rate      : The percentage of the data that should belong to the test set.
        validation     : If true, it means you also need validation set.
        validation_rate: If validation is true, this parameter shows which percentage of the data belongs to the validation set. It will be ignored if validation is False.
        random_state   : The state of random initializer. 

    Returns:
        - a tupple including:
            1. train data indices
            2. test data indices
            3. validation data indices (if validation is False, it is an empty array).
    """
    if (random_state):
        np.random.seed(random_state)
        
    test_idx = np.random.choice(list(range(count)), size=int(test_rate * count), replace=False)
    train_idx = np.array(list(set(range(count)) - set(test_idx)))
    if validation:
        validation_idx = np.random.choice(train_idx, size=int((validation_rate) / (1-test_rate) * train_idx.shape[0]), replace=False)
        train_idx = np.array(list(set(train_idx) - set(validation_idx)))

    np.random.seed(None)
    return (train_idx, test_idx, validation_idx)