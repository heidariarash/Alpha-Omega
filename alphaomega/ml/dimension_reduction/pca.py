import numpy as np
from alphaomega.preprocessing.normalization.zscore import z_score_normalizer

class PCA:
    """
    Usage: PCA is a dimensionality reductino technique. Use this class to apply PCA to your features.
    """
    def __init__(self):
        """
        Usage  : The constructor of PCA class.
        Inputs : Nothing.
        Returns: An instantiation of PCA class.
        """
        self.__new_features = 2
        self.__transform = None
        self.__explain = None
        self.__shape = 0

    def config(self, **kwargs):
        """
        Usage: Use this method to configure the parameters of PCA instantiation.

        Inputs:
            new_features: The number of new features. You can choose this parameter after training the model. This parameter only affects apply method.

        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key =="new_features":
                if int(value) >= 1:
                    self.__new_features = int(value)
                else:
                    print("The number of new features should be a positive integer.")

    def train(self, train_features):
        """
        Usage: Use this method to train the PCA instantiation.

        Inputs:
            train_features: The train features. Based on these features will The transformation matrix computed.

        Returns: Nothing.
        """
        #checking for the true shape of train_features
        if len(train_features.shape) != 2:
            print("The train_features should be tabular (i.e. have 2 dimensions).")
            return
        
        self.__shape = train_features.shape[1]

        #normalizing the featurs
        features = z_score_normalizer(train_features, train_features)

        #calculating covariance matrix
        cov_matrix = np.cov(features.T)

        #calculating eigen vectors and eigen values
        eig_values, eig_vectors = np.linalg.eig(cov_matrix)

        self.__transform =  eig_vectors
        self.__explain = eig_values

    def apply(self, features):
        """
        Usage: Use this method to apply dimensionality reduction to your features.

        Inputs:
            featuers: The dimensions of these featuers will get reduced.

        Returns:
            - New reduced features.
        """
        #checking for the true shape of features
        if len(features.shape) != 2:
            print("features should be tabular (i.e. have 2 dimensions).")
            return

        if features.shape[1] != self.__shape:
            print("number of features should be equivalent to the number of features of training data.")
            return

        #checking if the model is trained.
        if not self.__transform:
            print("Please train the model first.")
            return

        #nomalizing the features.    
        features_scaled = z_score_normalizer(features, features)
        new_feat = np.zeros((features.shape[0], self.__new_features))

        #applying dimension reductino.
        for i in range(self.__new_features):
            new_feat[:,i] = features_scaled.dot(self.__transform.T[i])

        return new_feat

    def explained_variance(self):
        """
        Usage: Use this method to find out the explained variance for each new feature.

        Inputs: Nothing.

        Returns:
            - A numpy array, each element of that indicates the explained variance by corresponding index feature.
        """
        #checking if the model is trained.
        if not self.__transform:
            print("Please train the model first.")
            return

        explained_variances = []
        for i in range(len(self.__explain)):
            explained_variances.append(self.__explain[i] / np.sum(self.__explain))

        return np.array(explained_variances)