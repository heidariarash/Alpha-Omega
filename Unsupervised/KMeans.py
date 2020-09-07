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
        self.centroids = []
        self.labels = None
        self.distance = []


    def train(self, data, k = 2, iter = 10, max_iter = 100):
        """
        Use this method to train the KMeans algorithm.
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
        for _ in range(k):
                self.centroids.append(np.random.randn(*self.data.shape[1:]))

        #training until centroids don't change or max-iter is reached
        while ( iteration < max_iter and change ):
            for index, point in enumerate(data):
                self.distance.append([])
                for center in self.centroids:
                    self.distance[index].append(np.linalg.norm(point - center))
            
            iteration +=1

        ##to be done:
            #the effect of iter
            #updating centroids
            #better initialisation
            #save the best cluster of iter clusters
            #write a method to give the elbow data.
            #updating labels


############## Testing Codes ####################
kmeans = KMeans()
data = np.array([[2, 3],
                 [2, 4],
                 [3, 5]])
kmeans.train(data)
print(kmeans.distance)

