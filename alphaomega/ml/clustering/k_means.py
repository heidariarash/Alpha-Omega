import numpy as np
from alphaomega.utils.exceptions import WrongAttribute, WrongDimension
from typing import Union

class KMeans:
    """
    KMeans is an unsupervised Machine Learning algorithms used to cluster data.
    """
    def __init__(self):
        self.__cost       = 0
        self.__centroids  = np.array([])
        self.__labels     = []
        self.__k          = 2
        self.__iterations = 10
        self.__max_iter   = 100

    def config(self, **kwargs: dict) -> None:
        """
        Usage: Use this method to configure the parameters of the KMeans instantiation.

        Inputs:
            k        : number of clusters
            iteration: number of iteration in total to find the best clusters.
            max_iter : number of maximum iteration if the centroids are still not to be found.

        Returns: Nothing.
        """
        for key, value in kwargs.items():

            if key == "k":
                if int(value) < 2:
                    raise AttributeError("Value of k (number of clusters) can not be less than two.")
                self.__k = int(k)

            elif key == "iterations":
                if int(value) < 1:
                    raise AttributeError("Iterations could not be less than 1.")
                self.__iterations = int(value)

            elif key == "max_iter":
                if int(value) < 1:
                    raise AttributeError("max_iter could not be less than 1.")
                self.__max_iter = int(value)

    def get(self, attribute: str) -> Union[int, np.ndarray, list[int]]:
        """
        Usage: Use this method to get the attribute of interest.

        Inputs:
            attribute: The attribute of interest. It could be "centroids", "labels", or "cost".

        Returns: The desired attribute
        """
        if attribute == "cost":
            return self.__cost

        if attribute == "labels":
            return self.__labels

        if attribute == "centroids":
            return self.__centroids

        raise AttributeError("The specified attribute is not valid. Acceptable attributes are 'centroids', 'labels', and 'cost'")

    def train(self, data: np.ndarray) -> None:
        """
        Usage: trains the KMeans model. The trained parameteres are:
            labels   : You can check the labels of every data point using this attribute.
            centroids: You can check where the centroids of every cluster is located.
            cost     : The summation over distances of every data point to its own centroid is stored here.
            distances: If you are curious about the distances of each data point to each centroids you can check it out here.
            
        Inputs:
            data: data that traines the model. it should be a numpy ndarray (specifically 2darray).
            
        Returns: Nothing.
        """
        if (len(data.shape) != 2):
            raise WrongDimension("Only 2 dimenaionsial arrays are acceptable.")
        
        best_iter = np.Inf
        for _ in range(self.__iterations):
            cost          = 0
            distance      = {}
            labels        = np.array([0] * data.shape[0])
            change        = True
            iteration     = 0
            new_centroids = np.zeros([self.__k, data.shape[1]])

            #initialising Cetnroids
            idx       = np.random.choice(range(data.shape[0]), self.__k, replace = False)
            centroids = data[idx]

            #training until centroids don't change or max-iter is reached
            while ( iteration < self.__max_iter and change ):
                #calculating distances between points and centroids
                for index, point in enumerate(data):
                    distance[index] = []
                    for center in centroids:
                        distance[index].append(np.linalg.norm(point - center))
                    #choosing the right label for each point
                    labels[index] = np.argmin(distance[index])

                #updating centroids
                for center in range(self.__k):
                    same_label = data[labels == center]
                    new_centroids[center] = np.mean(same_label, axis = 0)

                #checking if the clustering converged and break the loop if true
                if (new_centroids == centroids).all():
                    change = False
                else:
                    centroids = new_centroids.copy()
                iteration +=1

            #calculating cost function (sum over the distance between each point to its corresponiding centroid)
            for index , point in enumerate(data):
                cost += np.linalg.norm(point - centroids[labels[index]])

            #choosing the best clustering based on cost function.
            if cost < best_iter:
                self.__cost      = cost
                self.__labels    = labels
                self.__centroids = centroids
                best_iter        = cost
                              
    def elbow(self, data: np.ndarray, min_k: int = 2, max_k: Union[int, None] = None, max_iter: int = 100) -> dict:
        """
        Usage: Use this method to evalute KMeans model for different ks and choose the best k.
        
        Inputs:
            data    : data for clustering. it should be a numpy ndarray (specifically 2darray).
            min_k   : minimum number of clusters. default is 2.
            max_k   : maximum number of cluster. If you don't specify a maximum number of cluster, number of data points will be assigend automatically.
            max_iter: number of maximum iteration if the centroids are still not to be found.
            
        Returns: a dictionary including cost functions for each k.
        """       
        costs = {}

        #check if there is no input for max_k
        if max_k == None:
            max_k = data.shape[0]

        #check the boundrary of max_k
        if max_k > data.shape[0] or max_k < min_k:
            raise AttributeError("maximum number of clusters can not be more than the number of data points.\nmaximum number of clusters can not be less than minimum number of clusters.")

        #running the model for each K and compute the cost function and store it in an array.
        for k in range(min_k, max_k + 1):
            self.config(k = k, max_iter = max_iter)
            self.train(data)
            costs[k] = self.__cost
            
        return costs
            

def kmeans_apply(data: np.ndarray, k: int = 2, iterations: int = 10, max_iter: int = 100) -> np.ndarray:
    """
    Usage: This function is actually is the same as the class. you can use this API instead of instantiating the KMeans class.

    Inputs:
        data     : data that traines the model. it should be a numpy ndarray (specifically 2darray).
        k        : number of clusters
        iteration: number of iteration in total to find the best clusters.
        max_iter : number of maximum iteration if the centroids are still not to be found.

    Returns: a numpy array including the labels for each data point.
    """
    if (len(data.shape) != 2):
        raise WrongDimension("Only 2 dimensional arrays are acceptable.")

    best_iter = np.Inf
    for _ in range(iterations):
        cost          = 0
        distance      = {}
        labels        = np.array([0] * data.shape[0])
        change        = True
        iteration     = 0
        new_centroids = np.zeros([k, data.shape[1]])
        
        idx       = np.random.choice(range(data.shape[0]), k, replace = False)
        centroids = data[idx]

        while ( iteration < max_iter and change ):
            for index, point in enumerate(data):
                distance[index] = []
                for center in centroids:
                    distance[index].append(np.linalg.norm(point - center))
                labels[index] = np.argmin(distance[index])
                
            for center in range(k):
                same_label = data[labels == center]
                new_centroids[center] = np.mean(same_label, axis = 0)
            if (new_centroids == centroids).all():
                change = False
            else:
                centroids = new_centroids.copy()
            iteration +=1

        for index , point in enumerate(data):
            cost += np.linalg.norm(point - centroids[labels[index]])

        if cost < best_iter:
            labels_best = labels
            best_iter   = cost

    return labels_best


def kmeans_cost(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Usage: Use this function to find out the cost (summation over distances of every data point to its own centroid)

    Inputs:
        data  : data of the dataset.
        labels: labels for each data point in data. This argument is the return parameter of kmeans_apply function.

    Returns:
        - The cost of the clustering.
    """
    #checking for the correct shape of data and labels
    if (len(data.shape) != 2):
        raise WrongDimension("Only 2 dimensional arrays are acceptable.")

    if (data.shape[0] != len(labels.reshape(-1))):
        raise WrongDimension("Number of data points and labels should be equal.")
    
    #finding the number of clusters
    k = len(np.unique(labels))

    #initializing centroids and cost
    centroids = np.zeros([k, data.shape[1]])
    cost      = 0

    #finging the centroids
    for center in range(k):
        same_label        = data[labels == center]
        centroids[center] = np.mean(same_label, axis = 0)

    #calculationg cost
    for index , point in enumerate(data):
        cost += np.linalg.norm(point - centroids[labels[index]])

    return cost