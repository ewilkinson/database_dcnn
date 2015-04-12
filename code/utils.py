import caffe
import numpy as np
import os
import time
import hickle as hkl


use_alexnet = True

feature_layers = ['fc7', 'fc6', 'pool5', 'conv4', 'conv3', 'pool2', 'pool1']
img_dir = "../images/imagenet"
feature_dir = "../features"
compression_dir = "../compression"
caffe_root = '/home/eric/caffe/caffe-master/'


# database configuration
user = 'USER_NAME'
password = 'YOUR_PASS'
host = '127.0.0.1'
dbname = 'mydb'

def get_dimension_options(layer, compression):
    if not layer in feature_layers:
        raise NotImplementedError('Feature Layer Type Not Found.')

    compresion_path = os.path.join(compression_dir, compression, layer)
    files = os.listdir(compresion_path)
    N = len(files)

    if N <= 1:
        raise ValueError('Path provided contained no stored algorithms : ' + compresion_path)

    # there is a holder file in each directory which needs to be removed
    files.remove('holder.txt')

    dimensions = []
    for file in files:
        name, dim, postfix = file.split('_')
        dimensions.append(int(dim))

    return dimensions

def load_english_labels():
    """
    Returns a dictionary from class # to the english label.
    :return:
    """
    imagenet_labels_filename = os.path.join('../caffe/synset_words.txt')
    try:
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    except:
        raise ValueError('Could not find synset_works in the correct place.')

    return labels

def load_class_labels():
    fo = open("../caffe/val.txt", "r+")

    # remove the /n
    content = fo.read().splitlines()

    labels = np.zeros((50001, 1), dtype='int32')

    for line in content:
        image, klass = line.split(' ')
        year, type, postfix = image.split('_')
        id, file_type = postfix.split('.')
        labels[id,:] = int(klass)

    fo.close()

    return labels

def load_compressor(layer, dimension, compression):
    """
    Loads the compression algorithm from the file system

    :param layer: Feature layer

    :type dimension: int
    :param dimension: n_components of compressor

    :type compression: str
    :param compression: Compressional algorithm ID

    :return: Compression algorithm
    """
    if not layer in feature_layers:
        raise NotImplementedError('Feature Layer Type Not Found.')

    compression_path = os.path.join(compression_dir, compression, layer)
    file_name = compression + '_' + str(dimension) + '_gzip.hkl'

    file_path = os.path.join(compression_path, file_name)

    return hkl.load(file_path, safe=False)


# Simple generator for looping over an array in batches
def batch_gen(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def load_network():
    if not use_alexnet:
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
    :return: X, imagenet_ids, scalar
    """
    if not layer in feature_layers:
        raise NotImplementedError('Feature Layer Type Not Found.')

    features_path = os.path.join(feature_dir, layer)
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
            imagenet_ids = hkl.load(os.path.join(features_path, file))
        elif 'scalar' in sp:
            scalar = hkl.load(os.path.join(features_path, file), safe=False)

    print 'Total Load Time for Layer : ', layer
    print 'Time (s) : ', time.clock() - start_time

    return X, imagenet_ids, scalar