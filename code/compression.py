import utils
from sklearn.decomposition import KernelPCA, IncrementalPCA, MiniBatchSparsePCA
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.lda import LDA

def generate_lda_compression(X, Y, n_components = 16):
    """
    Compresses the data using sklearn PCA implementation.

    :param X: Data (n_samples, n_features)
    :param Y: Labels (n_samples,)
    :param n_components: Number of dimensions for PCA to keep

    :return: X_prime (the compressed representation), pca
    """

    lda = LDA(solver='svd', shrinkage=None, priors=None, n_components=n_components, store_covariance=False, tol=0.0001)
    lda.fit(X, Y)

    return lda.transform(X), lda

def generate_pca_compression(X, n_components = 16, batch_size=100):
    """
    Compresses the data using sklearn PCA implementation.

    :param X: Data (n_samples, n_features)
    :param n_components: Number of dimensions for PCA to keep
    :param batch_size: Batch size for incrimental PCA

    :return: X_prime (the compressed representation), pca
    """

    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    pca.fit(X)

    return pca.transform(X), pca

def generate_kpca_compression(X, n_components = 16):
    """
    Compresses the data using sklearn KernelPCA implementation.

    :param X: Data (n_samples, n_features)
    :param n_components: Number of dimensions for PCA to keep

    :return: X_prime (the compressed representation), pca
    """

    kpca = KernelPCA(n_components=n_components, kernel='rbf', eigen_solver='arpack', fit_inverse_transform=False)
    kpca.fit(X)

    return kpca.transform(X), kpca


def generate_lle_compression(X, n_neighbors = 20, n_components= 16):
    """
    Compresses the data using sklearn Local linear embedding

    :param X: Data (n_samples, n_features)
    :param n_components: Number of dimensions for LLE to keep
    :param n_neighbors: Number of nearest neighbors to consider

    :return: X_prime (the compressed representation), pca
    """

    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors,
                                n_components=n_components,
                                reg=0.001,
                                eigen_solver='auto',
                                tol=1e-06,
                                max_iter=100,
                                method='standard',
                                hessian_tol=0.0001,
                                modified_tol=1e-12,
                                neighbors_algorithm='auto',
                                random_state=None)
    lle.fit(X)

    return lle.transform(X), lle

def generate_isomap_compression(X, n_neighbors = 20, n_components= 16):
    """
    Compresses the data using sklearn Local linear embedding

    :param X: Data (n_samples, n_features)
    :param n_components: Number of dimensions for LLE to keep
    :param n_neighbors: Number of nearest neighbors to consider

    :return: X_prime (the compressed representation), pca
    """

    iso = Isomap(n_neighbors=n_neighbors,
                 n_components=n_components,
                 eigen_solver='auto',
                 tol=0, max_iter=None,
                 path_method='auto',
                 neighbors_algorithm='auto')
    iso.fit(X)

    return iso.transform(X), iso

def generate_spca_compression(X, n_components = 16, batch_size=100):
    """
    Compresses the data using sklearn PCA implementation.

    :param X: Data (n_samples, n_features)
    :param n_components: Number of dimensions for PCA to keep
    :param batch_size: Batch size for incrimental PCA

    :return: X_prime (the compressed representation), pca
    """

    spca = MiniBatchSparsePCA(n_components=n_components,
                              alpha=1,
                              ridge_alpha=0.01,
                              n_iter=100,
                              callback=None,
                              batch_size=batch_size,
                              verbose=False,
                              shuffle=True,
                              n_jobs=1,
                              method='lars',
                              random_state=None)
    spca.fit(X)

    return spca.transform(X), spca


if __name__ == '__main__':
    import time

    layer = 'fc7'
    n_components = 2

    X, ids = utils.load_feature_layer(layer)
    scalar = utils.load_scalar(layer)

    train_class = range(20)
    train_labels = utils.load_train_class_labels()

    keep_idxs = []
    classes = []
    for i, img_file in enumerate(ids):
        klass = train_labels[img_file]
        if klass in train_class:
            keep_idxs.append(i)
            classes.append(klass)

    keep_idxs = np.asarray(keep_idxs, dtype=np.int64)
    ids = np.asarray(ids, dtype=np.object)
    classes = np.asarray(classes, dtype=np.int32)

    X = X[keep_idxs]
    ids = ids[keep_idxs]

    print 'Num Components : ', n_components
    start_time = time.clock()
    # X_prime, compressor = generate_lle_compression(X, n_neighbors=20, n_components=n_components)
    # X_prime, compressor = generate_spca_compression(X, batch_size=100, n_components=n_components)
    # X_prime, compressor = generate_isomap_compression(X, n_neighbors=10, n_components=n_components)
    X_prime, compressor = generate_lda_compression(X, classes,  n_components=n_components)

    print 'Compression time : ', (time.clock() - start_time)
    print 'Compression time per sample : ', (time.clock() - start_time) / X.shape[0]


    import matplotlib as mpl
    import matplotlib.pyplot as plt

    x = X_prime.T[0,:]
    y = X_prime.T[1,:]

    plt.scatter(x,y,c=classes/10.0, cmap=mpl.cm.jet)
    plt.colorbar()

    plt.show()




