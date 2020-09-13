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
        """
        labels = np.zeros(features.shape[0], dtype=int)
        distances = []
        neighbor_labels = []
        neighbor_distances = []
        for data in range(features.shape[0]):
            for neighbor in range(self.__features.shape[0]):
                distances.append(np.linalg.norm(self.__features[neighbor] - features[data]))
            for neighbor in range(self.__k):
                neighbor_labels.append(self.__labels[np.argmin(distances)])
                neighbor_distances.append(distances[np.argmin(distances)])
                distances[np.argmin(distances)] = np.Inf

            most_label = max(neighbor_labels, key=neighbor_labels.count)
            labels[data] = most_label
            distances.clear()
            neighbor_distances.clear()
            neighbor_labels.clear()
        return labels


x = np.array([[1,1],
              [1,2],
              [2,2],
              [4,5]])

y = np.array([[0],[0],[1],[1]])
model = KNN()
model.config(k=3)
model.train(x, y)
print(model.apply(np.array([[2,1],[3,3]])))