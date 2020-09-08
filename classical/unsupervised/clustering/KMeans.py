import numpy as np

class KMeans:
    """
    KMeans is an unsupervised Machine Learning algorithms used to cluster data.
    """
    def __init__(self):
        """
        The constructor of KMeans class. There is no input for constructor.
        """
        self.k = None
        self.data = None
        self.centroids = {}
        self.labels = []
        self.distance = {}


    def train(self, data, k = 2, iter = 10, max_iter = 100):
        """
        Usage:
            trains the KMeans model.
        Inputs:
            data    : data to be trained. it should be a numpy ndarray.
            k       : number of clusters
            iter    : number of iteration in total to find the best clusters.
            max_iter: number of maximum iteration if the centroids are still not to be found.
        """
        self.data = data
        self.k = k
        change = True
        iteration = 0

        #initialising Cetnroids
        idx = np.random.randint(data.shape[0], size= k)
        self.centroids = data[idx]

        #training until centroids don't change or max-iter is reached
        while ( iteration < max_iter and change ):
            for index, point in enumerate(data):
                self.distance[index] = []
                for value in self.centroids:
                    self.distance[index].append(np.linalg.norm(point - value))
                self.labels.append(np.argmin(self.distance[index]))
            

            iteration +=1

        ##to be done:
            #the effect of iter
            #updating centroids
            #save the best cluster of iter clusters
            #write a method to give the elbow data.


############## Testing Codes ####################
kmeans = KMeans()
data = np.array([[2, 3],
                 [2, 4],
                 [3, 5]])
kmeans.train(data, max_iter=1, k=2)
# print(kmeans.distance)
# print(kmeans.centroids)
# print(kmeans.labels)
# print(data)

