import numpy as np
from collections import Counter

class KNN:
    """
    Usage: KNN(k-neareset neighbors) is a machine learning algorithm used for classification. The algorithm written for this class is so:
        - if k is odd and there are only two classes: There will be no conflict, the new data will be labeled as the label of the most k nearest neighbors.
        - if k is even or there are more than two classes: Two different situations could occur:
            1. In the k nearest neighbors of the new data, one label is dominant. In this case the new data labeled as the dominant label.
            2. In the k nearest neighbors of the new data, two or more labels have same quantitiy. In this case, the labels will be assigned randomly between dominant labels.
    """
    def __init__(self):
        """
        Usage  : The constructor of KNN class. 
        Inputs : Nothing
        Returns: An Instantiation of the class.
        """
        self.__k = 1
        self.__features = np.array([])
        self.__labels = np.array([])
    
    def config(self, **kwargs):
        """
        Usage  : Use this method to configure the parameteres of the KNN instantiation.

        Inputs : 
            k  : The number of nearest neighbors.

        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "k":
                self.__k = int(value)
                if int(value) < 1:
                    print("k cannot be less than 1. It reseted to 1.")
                    self.__k = 1

    def train(self, train_features, labels):
        """
        Usage  : Use this method to train the KNN model.

        Inputs :
            train_features: The features of the training set.
            labels        : The labels of the training set.

        Returns: Nothing.
        """
        self.__features = train_features
        self.__labels = labels

    def apply(self, features):
        """
        Usage: Use this method to apply the KNN model to the new data for prediction.

        Inputs:
            featurs: features of the test data.

        Returns:
            - A numpy array including the labels for each data point in featurs.
        """
        labels = np.zeros(features.shape[0], dtype=int)
        distances = []
        neighbor_labels = []
        # neighbor_distances = []
        for data in range(features.shape[0]):
            for neighbor in range(self.__features.shape[0]):
                distances.append(np.linalg.norm(self.__features[neighbor] - features[data]))
            for neighbor in range(self.__k):
                neighbor_labels.append(self.__labels[np.argmin(distances)])
                # neighbor_distances.append(distances[np.argmin(distances)])
                distances[np.argmin(distances)] = np.Inf

            most_label = max(neighbor_labels, key=neighbor_labels.count)
            labels[data] = most_label
            distances.clear()
            # neighbor_distances.clear()
            neighbor_labels.clear()
        return labels

    def test(self, test_features, test_labels):
        """
        Usage: Use this method to find out the accuracy of KNN model.

        Inputs:
            test_features: The features of the test set.
            test_labels  : The labels of the test set.

        Returns:
             - The accuracy of the model in percentage.
        """
        predicted_labels = self.apply(test_features)
        accuracy = 100 * (np.mean(predicted_labels == test_labels.reshape(-1)))
        return accuracy
        

def knn_func(train_features, train_labels, test_features, k = 1):
    """
    Usage: Use this function to apply KNN algorithm to your test set.

    Inputs:
        train_features: The features of the training set.
        train_labels  : The labels of the training set.
        test_features : The features of the test set.
        k             : The number of nearest neighbors.

    Returns: 
        - A numpy array including the labels for each data point in featurs.
    """
    labels = np.zeros(test_features.shape[0], dtype=int)
    distances = []
    neighbor_labels = []
    # neighbor_distances = []
    for data in range(test_features.shape[0]):
        for neighbor in range(train_features.shape[0]):
            distances.append(np.linalg.norm(train_features[neighbor] - test_features[data]))
        for neighbor in range(k):
            neighbor_labels.append(train_labels[np.argmin(distances)])
            # neighbor_distances.append(distances[np.argmin(distances)])
            distances[np.argmin(distances)] = np.Inf

        most_label = max(neighbor_labels, key=neighbor_labels.count)
        labels[data] = most_label
        distances.clear()
        # neighbor_distances.clear()
        neighbor_labels.clear()
    return labels