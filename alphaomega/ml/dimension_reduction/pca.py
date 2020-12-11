import numpy as np
from alphaomega.utils.exceptions import WrongDimension, WrongAttribute

class PCA:
    """
    Usage: PCA is a dimensionality reductino technique. Use this class to apply PCA to your features.
    """
    def __init__(self):
        self.__new_data  = 2
        self.__transform = None
        self.__explain   = None
        self.__shape     = 0
        self.__mean      = 0
        self.__std       = 0
        self.__applied   = False

    def config(self, **kwargs) -> None:
        """
        Usage: Use this method to configure the parameters of PCA instantiation.

        Inputs:
            new_features: The number of new features. You can choose this parameter after training the model. This parameter only affects apply method.

        Returns: Nothing.
        """
        for key, value in kwargs.items():
            if key == "new_features":
                if int(value) >= 1:
                    self.__new_data = int(value)
                else:
                    raise WrongAttribute("The number of new features should be a positive integer.")

    def train(self, train_data: np.ndarray) -> None:
        """
        Usage: Use this method to train the PCA instantiation.

        Inputs:
            train_data: The train features. Based on these features will The transformation matrix computed.

        Returns: Nothing.
        """
        #checking for the true shape of train_data
        if len(train_data.shape) != 2:
            raise WrongDimension("The train_data should be tabular (i.e. have 2 dimensions).")
        
        self.__shape = train_data.shape[1]

        #normalizing the featurs
        self.__mean = np.mean(train_data,axis = 0)
        self.__std  = np.std(train_data, axis = 0)
        features    = (train_data - self.__mean) / (self.__std)

        #calculating covariance matrix
        cov_matrix  = np.cov(features.T)

        #calculating eigen vectors and eigen values
        eig_values, eig_vectors = np.linalg.eig(cov_matrix)
        
        self.__transform =  eig_vectors
        self.__explain   = eig_values
        self.__applied   = True

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Usage: Use this method to apply dimensionality reduction to your data.

        Inputs:
            featuers: The dimensions of these featuers will get reduced.

        Returns:
            - New reduced data.
        """
        #checking for the true shape of data
        if len(data.shape) != 2:
            raise WrongDimension("data should be tabular (i.e. have 2 dimensions).")

        if data.shape[1] != self.__shape:
            raise WrongDimension("number of data should be equivalent to the number of data of training data.")

        #checking if the model is trained.
        if not self.__applied:
            raise Exception("Please train the model first.")

        #nomalizing the data.    
        data_scaled = (data - self.__mean) / (self.__std)
        new_feat    = np.zeros((data.shape[0], self.__new_data))

        #applying dimension reductino.
        for i in range(self.__new_data):
            new_feat[:,i] = data_scaled.dot(self.__transform.T[i])

        return new_feat

    def explained_variance(self) -> np.ndarray:
        """
        Usage: Use this method to find out the explained variance for each new feature.

        Inputs: Nothing.

        Returns:
            - A numpy array, each element of that indicates the explained variance by corresponding index feature.
        """
        #checking if the model is trained.
        if not self.__applied:
            raise Exception("Please train the model first.")

        explained_variances = []
        for i in range(len(self.__explain)):
            explained_variances.append(self.__explain[i] / np.sum(self.__explain))

        return np.array(explained_variances)


def pca_train(train_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Usage: Use this function to train a pca model.

    Inputs:
        train_data: The PCA algorithm will get trained based on these features.

    Returns:
        - The PCA parameters. This parameter is one of the inputs of pca_apply function.
    """
    #checking for the correct shapes of train_data
    if len(train_data.shape) != 2:
        raise WrongDimension("train_data should be tabular (i.e. have 2 dimensions).")

    mean         = np.mean(train_data,axis = 0)
    std          = np.std(train_data, axis = 0)
    scaled_train = (train_data - mean) / (std)

    #calculating covariance matrix
    cov_matrix   = np.cov(scaled_train.T)

    #calculating eigen vectors and eigen values
    _ , eig_vectors = np.linalg.eig(cov_matrix)

    return mean, std, eig_vectors


def pca_apply(data: np.ndarray, pca_params: tuple[np.ndarray, np.ndarray, np.ndarray], new_features: int = 2):
    """
    Usage: Use this function to apply PCA to your data.

    Inputs:
        data    : The trained PCA algorihm reduced the dimensions of these featueres.
        pca_params  : The parameters of PCA algorithm. You can obtain this parameters using pca_train function.
        new_feateurs: The number of new data.

    Returns:
        - New reduced data.
    """
    mean, std, eig_vectors = pca_params

    #checking for the correct shape of data:
    if len(data.shape) != 2:
        raise WrongDimension("data should be tabular (i.e. have 2 dimensions).")
    
    if data.shape[1] != eig_vectors.shape[0]:
        raise WrongDimension("number of data should be equivalent to the number of data of training data.")

    #checking for the correct value for new_features
    if int(new_features) < 1:
        raise WrongAttribute("The number of new data should be a positive integer.")

    #normalizing the data.
    scaled_data = (data - mean) / (std)
    new_feat    = np.zeros((data.shape[0], new_features))

    #applying dimension reductino.
    for i in range(new_features):
        new_feat[:,i] = scaled_data.dot(eig_vectors.T[i])

    return new_feat


def pca_explained_variance(train_data: np.ndarray) -> np.ndarray:
    """
    Usage: Use this function to obtain the explained variance of each new feature if you apply PCA algorithm to your featuers.

    Inputs:
        train_data: Explained variance will get calculated based on these featurs.

    Returns:
        - A numpy array, each element of that indicates the explained variance by corresponding index feature.
    """
    #checking for the correct shapes of train_data
    if len(train_data.shape) != 2:
        raise WrongDimension("train_data should be tabular (i.e. have 2 dimensions).")

    mean         = np.mean(train_data,axis = 0)
    std          = np.std(train_data, axis = 0)
    scaled_train = (train_data - mean) / (std)

    #calculating covariance matrix
    cov_matrix   = np.cov(scaled_train.T)

    #calculating eigen vectors and eigen values
    eig_values , _      = np.linalg.eig(cov_matrix)
    explained_variances = []
    
    for i in range(len(eig_values)):
        explained_variances.append(eig_values[i] / np.sum(eig_values))

    return np.array(explained_variances)