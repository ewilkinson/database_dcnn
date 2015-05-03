import utils
from compression import *
import time

if __name__ == '__main__':

    # ------------------------------------------------
    # Params
    # ------------------------------------------------

    compression_type = 'lda'

    # feature_layers = utils.feature_layers
    feature_layers = ['fc7']
    # dimensions = [16, 32,64,128,256]
    dimensions = [32, 64, 256]
    ipca_batch_size = 500


    for layer in feature_layers:
        X, ids = utils.load_feature_layer(layer)
        scalar = utils.load_scalar(layer)
        train_labels = utils.load_train_class_labels()

        classes = []
        for i in range(X.shape[0]):
            classes.append(train_labels[ids[i]])
        classes = np.asarray(classes, dtype=np.int32)


        # zero and normalize the data
        X = scalar.transform(X)

        for n_components in dimensions:

            print 'Num Components : ', n_components
            start_time = time.clock()

            if compression_type == 'pca':
                X_prime, compressor = generate_pca_compression(X, n_components, ipca_batch_size)
            elif compression_type == 'kpca':
                X_prime, compressor = generate_kpca_compression(X, n_components)
            elif compression_type == 'lle':
                X_prime, compressor = generate_lle_compression(X, n_components=n_components, n_neighbors = 20)
            elif compression_type == 'spca':
                X_prime, compressor = generate_spca_compression(X, n_components=n_components, batch_size = ipca_batch_size)
            elif compression_type == 'isomap':
                X_prime, compressor = generate_lle_compression(X, n_components=n_components, n_neighbors = 15)
            elif compression_type == 'lda':
                X_prime, compressor = generate_lda_compression(X, classes,  n_components=n_components)


            print 'Compression time : ', (time.clock() - start_time)
            print 'Compression time per sample : ', (time.clock() - start_time) / X.shape[0]


            print 'X - Prime Shape', X_prime.shape

            utils.dump_compressor(layer,compressor,compression_type,n_components)




