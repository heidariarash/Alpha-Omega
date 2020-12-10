import numpy as np
from collections import Counter
from alphaomega.utils.exceptions import WrongAttribute

class KNN:
    """
    Usage: KNN(k-neareset neighbors) is a machine learning algorithm used for classification. The algorithm written for this class is so:
        - if k is odd and there are only two classes: There will be no conflict, the new data will be labeled as the label of the most k nearest neighbors.
        - if k is even or there are more than two classes: Two different situations could occur:
            1. In the k nearest neighbors of the new data, one label is dominant. In this case the new data labeled as the dominant label.
            2. In the k nearest neighbors of the new data, two or more labels have same quantitiy. In this case, the labels will be assigned randomly between dominant labels.
    """
    def __init__(self):
        self.__k      = 1
        self.__data   = np.array([])
        self.__labels = np.array([])
    
    def config(self, **kwargs: dict) -> None:
        """
        Usage  : Use this method to configure the parameteres of the KNN instantiation.

        Inputs : 
            k  : The number of nearest neighbors.

        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "k":
                if int(value) < 1:
                    raise AttributeError("k cannot be less than 1")
                self.__k = int(value)

    def train(self, train_data: np.ndarray, labels: np.ndarray) -> None:
        """
        Usage  : Use this method to train the KNN model.

        Inputs :
            train_data: The features of the training set.
            labels    : The labels of the training set.

        Returns: Nothing.
        """
        self.__data   = train_data
        self.__labels = labels

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Usage: Use this method to apply the KNN model to the new data for prediction.

        Inputs:
            featurs: data of the test data.

        Returns:
            - A numpy array including the labels for each data point in featurs.
        """
        labels          = np.zeros(data.shape[0], dtype=int)
        distances       = []
        neighbor_labels = []

        for data in range(data.shape[0]):

            for neighbor in range(self.__data.shape[0]):
                distances.append(np.linalg.norm(self.__data[neighbor] - data[data]))
            for neighbor in range(self.__k):
                neighbor_labels.append(self.__labels[np.argmin(distances)])
                distances[np.argmin(distances)] = np.Inf

            most_label   = max(neighbor_labels, key=neighbor_labels.count)
            labels[data] = most_label
            distances.clear()
            neighbor_labels.clear()

        return labels

    def test(self, test_data: np.ndarray, test_labels: np.ndarray) -> float:
        """
        Usage: Use this method to find out the accuracy of KNN model.

        Inputs:
            test_data  : The features of the test set.
            test_labels: The labels of the test set.

        Returns:
             - The accuracy of the model in percentage.
        """
        predicted_labels = self.apply(test_data)
        accuracy         = 100 * (np.mean(predicted_labels == test_labels.reshape(-1)))
        return accuracy
        

def knn_apply(train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Usage: Use this function to apply KNN algorithm to your test set.

    Inputs:
        train_data  : The features of the training set.
        train_labels: The labels of the training set.
        test_data   : The features of the test set.
        k           : The number of nearest neighbors.

    Returns: 
        - A numpy array including the labels for each data point in featurs.
    """
    #checking for the value of k
    if int(k) < 1:
        raise AttributeError("k cannot be less than 1")

    k = int(k)

    labels          = np.zeros(test_data.shape[0], dtype=int)
    distances       = []
    neighbor_labels = []

    for data in range(test_data.shape[0]):

        for neighbor in range(train_data.shape[0]):
            distances.append(np.linalg.norm(train_data[neighbor] - test_data[data]))
        for neighbor in range(k):
            neighbor_labels.append(train_labels[np.argmin(distances)])
            distances[np.argmin(distances)] = np.Inf

        most_label   = max(neighbor_labels, key=neighbor_labels.count)
        labels[data] = most_label
        distances.clear()
        neighbor_labels.clear()

    return labels