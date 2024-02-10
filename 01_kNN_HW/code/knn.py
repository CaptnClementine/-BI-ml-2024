import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        dist_matrix = np.empty((self.train_X.shape[0],X.shape[0]))
        for i, train_element in enumerate(self.train_X):
            for j, test_element in enumerate(X):
                dist_matrix[i][j] = sum(abs(train_element-test_element))
        return dist_matrix.T


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        dist_matrix = np.empty((self.train_X.shape[0],X.shape[0]))
        for i, train_element in enumerate(self.train_X):
                dist_matrix[i] = np.sum(abs(train_element-X), axis=1)

        return dist_matrix.T


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        return np.sum(np.abs(self.train_X[:, np.newaxis] - X), axis=2).T




    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case

        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)


        for i in range(n_test):
            k_nearest_dists = np.sort(distances[i, :])[:self.k]
            mask = np.isin(distances[i, :], k_nearest_dists)
            closest_y = self.train_y[mask]  # self.
            prediction[i] = np.argmax(np.bincount(closest_y))
            # k_nearest_idxs = np.argsort(distances[i])[:self.k]  # Get indices of k nearest neighbors
            # closest_y = self.train_y[k_nearest_idxs]  # Get labels of k nearest neighbors
            # prediction[i] = np.argmax(np.bincount(closest_y))

        return prediction

    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, np.int)

        for i in range(n_test):
            k_nearest_dists = np.sort(distances[i, :])[:self.k]
            mask = np.isin(distances[i, :], k_nearest_dists)
            closest_y = self.train_y[mask]  # self.
            prediction[i] = np.argmax(np.bincount(closest_y))
            # k_nearest_idxs = np.argsort(distances[i])[:self.k]  # Get indices of k nearest neighbors
            # closest_y = self.train_y[k_nearest_idxs]  # Get labels of k nearest neighbors
            # prediction[i] = np.argmax(np.bincount(closest_y))

        return prediction
