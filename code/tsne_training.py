import utils
import numpy as np
import matlab.engine
import array
import os

if __name__ == '__main__':
    import time

#    layers = ['fc7', 'fc6', 'pool5', 'conv4', 'conv3', 'pool2', 'pool1']
    layers = ['fc7']
    #n_components = [64, 128, 256]
    n_components = [128]
    c_type = 'pca'

    #==== dimensions for tsne =========
    dimensions = 4


    #========= t-sne ==============
    for layer in layers:
        #==== load original feature ======
        X0, ids = utils.load_feature_layer(layer)


        #ids = ids[0:500]
        #===== Convert ids (filename) to labels(integer) =========
        labels = []
        print len(ids)
        for i in range(len(ids)):
            labels.append(int(ids[i].split('_')[0].split('n')[1]))

        labels = matlab.double(labels)

        #======== Start MATLAB  ============
        print 'start matlab'
        eng = matlab.engine.start_matlab()


        for n_com in n_components:

            scalar = utils.load_scalar(layer)
            compressor = utils.load_compressor(layer=layer, dimension=n_com, compression=c_type)
            X = scalar.transform(X0)
            comp_X = compressor.transform(X)
            #comp_X = comp_X[0:500,:]



            #===== Convert ndarray to list so that we can pass the variable to matalb function =========
            comp_X = comp_X.tolist()
            comp_X = matlab.double(comp_X)

            print '===============================start tsne in python ==================================='
            print layer, n_com, c_type

            result = eng.tsne_traning_python(comp_X,labels,dimensions, layer, n_com, c_type )

