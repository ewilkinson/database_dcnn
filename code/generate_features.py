import caffe
import numpy as np
import os
from sklearn import preprocessing
import utils

if __name__ == '__main__':
    # ------------------------ Script Parameters ---------------------
    batch_size = 10
    feature_layers = utils.feature_layers
    # feature_layers = ['fc7']
    # ------------------------ Script Parameters ---------------------

    net, params, blobs = utils.load_network()

    image_files = os.listdir(utils.img_dir)

    N = len(image_files)
    print 'Total Files : ', N
    print 'Sample File Name : ', image_files[100]

    for layer in feature_layers:
        print 'Processing Layer : ' + layer

        file = image_files[0]
        f0 = caffe.io.load_image(os.path.join(utils.img_dir, file))
        prediction = net.predict([f0], oversample=False)
        features = net.blobs[layer].data[0]

        X = np.zeros((N, features.size), dtype='float32')
        ids = []

        count = 0
        for files in utils.batch_gen(image_files, batch_size=batch_size):

            if count % 1000 == 0:
                print 'Processing Layer : ' + layer + " Count : ", count

            images = []
            for file in files:
                file_image = caffe.io.load_image(os.path.join(utils.img_dir, file))
                images.append(file_image)


            prediction = net.predict(images, oversample=False)

            # save out all the features
            for i in range(batch_size):
                ids.append(files[i])
                X[count, :] = net.blobs[layer].data[i].ravel()
                count = count + 1

        utils.dump_feature_layer(layer=layer, X=X, ids=ids)

        scaler = preprocessing.StandardScaler().fit(X)
        utils.dump_scalar(layer=layer, scalar=scaler)