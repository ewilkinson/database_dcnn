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
    n_components = [4096]
    c_type = 'new'

    #==== dimensions for tsne =========
    dimensions = 128

    trainclass = range(10)
    trainlabels = utils.load_train_class_labels()

    #========= t-sne ==============
    for layer in layers:
        #==== load original feature ======
        X0, ids = utils.load_feature_layer(layer)

        # keep_idxs = []
        # for i, img_file in enumerate(ids):
        #     klass = trainlabels[img_file]
        #     if klass in trainclass:
        #         keep_idxs.append(i)
        #
        # keep_idxs = np.asarray(keep_idxs, dtype=np.int64)
        # ids = np.asarray(ids, dtype=np.object)
        #
        # X0 = X0[keep_idxs]
        # ids = ids[keep_idxs]

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
            X = scalar.transform(X0)

            # for i in range(4096):
            #     g = X[:,i] < 0.0
            #     X[g, i] = 0
            #     X[:,i] = X[:,i] / X[:,i].max()




            #===== Convert ndarray to list so that we can pass the variable to matalb function =========
            # comp_X = comp_X.tolist()
            # comp_X = matlab.double(comp_X)
            print 'Converting to matlab double'
            X = matlab.double(X.tolist())

            print '===============================start tsne in python ==================================='
            print layer, n_com, c_type

            result = eng.tsne_traning_python(X,labels,dimensions, layer, n_com, c_type )

