__author__ = 'eric'

import utils
from compression import generate_pca_compression, generate_kpca_compression
import time

if __name__ == '__main__':

    # ------------------------------------------------
    # Params
    # ------------------------------------------------

    compression_type = 'pca'

    feature_layers = utils.feature_layers
    # feature_layers = ['fc7']
    dimensions = [32,64,128,256,512]
    ipca_batch_size = 1000


    for layer in feature_layers:
        X, ids = utils.load_feature_layer(layer)
        scalar = utils.load_scalar(layer)

        # zero and normalize the data
        X = scalar.transform(X)

        for n_components in dimensions:

            print 'Num Components : ', n_components
            start_time = time.clock()

            if compression_type == 'pca':
                X_prime, pca = generate_pca_compression(X, n_components, ipca_batch_size)
            elif compression_type == 'kpca':
                X_prime, pca = generate_kpca_compression(X, n_components)

            print 'Compression time : ', (time.clock() - start_time)
            print 'Compression time per sample : ', (time.clock() - start_time) / X.shape[0]


            print 'X - Prime Shape', X_prime.shape

            utils.dump_compressor(layer,pca,compression_type,n_components)




