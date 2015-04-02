import utils
from sklearn import decomposition
import numpy as np



def generate_pca_features(X, n_components = 16):
    """
    Compresses the data using sklearn PCA implementation.

    :param X: Data (n_samples, n_features)
    :param n_components: Number of dimensions for PCA to keep
    :return: X_prime (the compressed representation), pca
    """

    pca = decomposition.PCA(n_components=n_components)
    pca.fit(X)
    X_prime = pca.transform(X)

    return X_prime, pca

if __name__ == '__main__':
    layer = 'fc6'
    n_components = 128

    X, ids = utils.load_feature_layer(layer)
    X_prime, pca = generate_pca_features(X, n_components)

    print X_prime.shape

