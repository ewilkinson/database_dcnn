import caffe
import numpy as np
import os
import time
import hickle as hkl

feature_layers = ['conv3', 'conv4', 'fc6', 'fc7', 'pool1', 'pool2', 'pool5']
img_dir = "../images/imagenet"
feature_dir = "../features"
compression_dir = "../compression"
caffe_root = '/home/eric/caffe/caffe-master/'

# Simple generator for looping over an array in batches
def batch_gen(data, batch_size):
    for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]

def load_network(use_alexnet=True):

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

def load_feature_layer(layer):
    """
    Loads the feature layer specified

    :param layer: A member of utils.feature_layers
    :return: X, ids, scalar
    """
    if not layer in feature_layers:
        raise NotImplementedError('Feature Layer Type Not Found.')


    features_path = os.path.join(feature_dir,layer)
    files = os.listdir(features_path)
    N = len(files)

    if N <= 1:
        raise ValueError('Path provided contained no features : ' + features_path)

    # there is a holder file in each directory which needs to be removed
    files.remove('holder.txt')

    
    start_time = time.clock()
    for file in files:
        sp = file.split('_')
        if 'X' in sp:
            X = hkl.load(os.path.join(features_path, file))
        elif 'ids' in sp:
            ids = hkl.load(os.path.join(features_path, file))
        elif 'scalar' in sp:
            scalar = hkl.load(os.path.join(features_path, file))

    print 'Total Load Time for Layer : ',  layer
    print 'Time (s) : ', time.clock() - start_time

    return X, ids, scalar