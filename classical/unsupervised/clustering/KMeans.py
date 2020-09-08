import numpy as np

class KMeans:
    """
    KMeans is an unsupervised Machine Learning algorithms used to cluster data.
    """
    def __init__(self):
        """
        Usage  : The constructor of KMeans class. 
        Inputs : Nothing
        Returns: An Instantiation of the class.
        """
        self.cost = 0
        self.centroids = np.array([])
        self.labels = []


    def train(self, data, k = 2, iterations = 10, max_iter = 100):
        """
        Usage: trains the KMeans model. The trained parameteres are:
            labels   : You can check the labels of every data point using this attribute.
            centroids: You can check where the centroids of every cluster is located.
            cost     : The summation over distances of every data point to its own centroid is stored here.
            distances: If you are curious about the distances of each data point to each centroids you can check it out here.
            
        Inputs:
            data     : data that traines the model. it should be a numpy ndarray (specifically 2darray).
            k        : number of clusters
            iteration: number of iteration in total to find the best clusters.
            max_iter : number of maximum iteration if the centroids are still not to be found.
            
        Returns: Nothing.
        """
        if (len(data.shape) != 2):
            print("Only 2darray")
            return
        
        best_iter = np.Inf
        for itera in range(iterations):
            cost = 0
            distance = {}
            labels = np.array([0] * data.shape[0])
            change = True
            iteration = 0
            new_centroids = np.zeros([k, data.shape[1]])

            #initialising Cetnroids
            idx = np.random.choice(range(data.shape[0]), k, replace = False)
            centroids = data[idx]

            #training until centroids don't change or max-iter is reached
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
                self.cost = cost
                self.labels = labels
                self.centroids = centroids
                best_iter = cost
                
                
    def elbow(self, data, min_k = 2, max_k = None, max_iter = 100):
        """
        Usage: Use this method to evalute KMeans model for different ks.
        
        Inputs:
            data    : data for clustering. it should be a numpy ndarray (specifically 2darray).
            min_k   : minimum number of clusters. default is 2.
            max_k   : maximum number of cluster. If you don't specify a maximum number of cluster, number of data points will be assigend automatically.
            max_iter: number of maximum iteration if the centroids are still not to be found.
            
        Returns: a dictionary including cost functions for each k.
        """       
        costs = {}
        if max_k == None:
            max_k = data.shape[0]
        
        if max_k > data.shape[0]:
            print("maximum number of clusters can not be more than the number of data points.")
            return
        
        for k in range(min_k, max_k + 1):
            self.train(data, k, max_iter = max_iter)
            costs[k] = self.cost
            
        return costs
            

def kmeans_func(data, k = 2, iterations = 10, max_iter = 100):
    if (len(data.shape) != 2):
        print("Only 2darray")
        return -1

    best_iter = np.Inf
    for itera in range(iterations):
        cost = 0
        distance = {}
        labels = np.array([0] * data.shape[0])
        change = True
        iteration = 0
        new_centroids = np.zeros([k, data.shape[1]])

        #initialising Cetnroids
        idx = np.random.choice(range(data.shape[0]), k, replace = False)
        centroids = data[idx]

        #training until centroids don't change or max-iter is reached
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
            best_iter = cost
    return labels