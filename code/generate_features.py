import caffe
import numpy as np
import os
import hickle as hkl
from sklearn import preprocessing
import utils

# Simple generator for looping over an array in batches
def batch_gen(data, batch_size):
    for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]

def load_network(use_alexnet=True):

    caffe_root = '/home/eric/caffe/caffe-master/'  # this file is expected to be in {caffe_root}/examples

    if use_alexnet:
        # Set the right path to your model definition file, pretrained model weights,
        # and the image you would like to classify.
        MODEL_FILE = '../caffe/bvlc_reference_caffenet/deploy.prototxt'
        PRETRAINED = '../caffe/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    else:
        # ALEXNET
        MODEL_FILE = '../caffe/bvlc_alexnet/deploy.prototxt'
        PRETRAINED = '../caffe/bvlc_alexnet/bvlc_alexnet.caffemodel'

    net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                           mean=np.load(os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')),
                           channel_swap=(2, 1, 0),
                           raw_scale=255,
                           image_dims=(256, 256))

    blobs = [(k, v.data.shape) for k, v in net.blobs.items()]
    params = [(k, v[0].data.shape) for k, v in net.params.items()]

    print 'Blobs : ', blobs
    print 'Params : ', params

    net.set_phase_test()
    net.set_mode_gpu()

    return net, params, blobs

if __name__ == '__main__':
    # ------------------------ Script Parameters ---------------------
    use_alexnet = True
    batch_size = 10

    # removed
    # feature_layers = ['conv2', 'conv1']

    feature_layers = utils.feature_layers
    # ------------------------ Script Parameters ---------------------

    net, params, blobs = load_network(use_alexnet)

    image_files = os.listdir(utils.img_dir)

    N = len(image_files)
    print 'Total Files : ', N
    print 'Sample File Name : ', image_files[100]

    for layer in feature_layers:
        print 'Processing Layer : ' + layer

        file = image_files[0]
        f0 = caffe.io.load_image(os.path.join(utils.img_dir, file))
        prediction = net.predict([f0])
        features = net.blobs[layer].data[0]

        X = np.zeros((N, features.size), dtype='float32')
        ids = np.zeros((N, ), dtype='int32')

        count = 0
        for files in batch_gen(image_files, batch_size=batch_size):
            years, types, img_ids = [], [], []
            images = []
            for file in files:
                year, type, postfix = file.split('_')
                id, file_type = postfix.split('.')
                years.append(year)
                types.append(type)
                img_ids.append(id)

                file_image = caffe.io.load_image(os.path.join(utils.img_dir, file))
                images.append(file_image)


            prediction = net.predict(images)



            # save out all the features
            for i in range(batch_size):
                ids[count] = img_ids[i]

                features = net.blobs[layer].data[i]
                X[count, :] = features.ravel()
                count = count + 1

        file_name = layer  + '_X_gzip.hkl'
        file_path = os.path.join(utils.feature_dir, layer, file_name)
        print 'Saving : ', file_path

        # Dump data, with compression
        hkl.dump(X, file_path, mode='w', compression='gzip')

        # Compare filesizes
        print 'compressed:   %i bytes' % os.path.getsize(file_path)


        file_name = layer  + '_ids_gzip.hkl'
        file_path = os.path.join(utils.feature_dir, layer, file_name)
        print 'Saving : ', file_path

        # Dump data, with compression
        hkl.dump(ids, file_path, mode='w', compression='gzip')

        # Compare filesizes
        print 'compressed:   %i bytes' % os.path.getsize(file_path)

        scaler = preprocessing.StandardScaler().fit(X)
        file_name = layer  + '_scalar_gzip.hkl'
        file_path = os.path.join(utils.feature_dir, layer, file_name)
        print 'Saving : ', file_path

        # Dump data, with compression
        hkl.dump(scaler, file_path, mode='w', compression='gzip')