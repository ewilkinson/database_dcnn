import caffe
import numpy as np
import os
import time
import hickle as hkl


use_alexnet = True

feature_layers = ['fc7', 'fc6', 'pool5', 'conv4', 'conv3', 'pool2', 'pool1']
img_dir = "../images/training_images"
test_dir = "../images/imagenet"
feature_dir = "../features"
compression_dir = "../compression"
distances_dir = "../distances"
caffe_root = '/home/eric/caffe/caffe-master/'


# database configuration
user = 'USER_NAME'
password = 'YOUR_PASS'
host = '127.0.0.1'
dbname = 'mydb'

def get_dimension_options(layer, compression):
    """
    Returns an array of all the possible compression sizes for that layer / compression pair

    :type layer: str
    :param layer: feature layer

    :type compression: str
    :param compression: compression type identifier (pca, kpca, etc.)

    :rtype: array-like
    :return: dimensions
    """
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

    :return: labels
    """
    imagenet_labels_filename = os.path.join('../caffe/synset_words.txt')
    try:
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    except:
        raise ValueError('Could not find synset_works in the correct place.')

    return labels


def load_test_class_labels():
    """
    Return all the class labels for the test set. Note that we are using the validation images as the test set
    since they come with labels

    :return: labels
    """
    fo = open("../caffe/val.txt", "r+")

    # remove the /n
    content = fo.read().splitlines()

    labels = {}

    for line in content:
        image, klass = line.split(' ')
        labels[image] = int(klass)

    fo.close()

    return labels

def load_train_class_labels():
    """
    Return all the class labels for the training set.

    :return: labels
    """
    fo = open("../caffe/train_without_dir.txt", "r+")

    # remove the /n
    content = fo.read().splitlines()

    labels = {}

    for line in content:
        image, klass = line.split(' ')
        labels[image] = int(klass)

    fo.close()

    return labels

def dump_compressor(layer, pca, compression_type, n_components):
    """
    Dumps the compressor

    :type layer: str
    :param layer: feature layer

    :param pca:

    :type compression_type: str
    :param compression_type:

    :type n_components: int
    :param n_components:

    :return:
    """
    dir_path = os.path.join(compression_dir, compression_type, layer)
    file_name = compression_type + '_'  + str(n_components) + '_gzip.hkl'

    file_path = os.path.join(dir_path, file_name)
    print 'Saving to : ', file_path

    hkl.dump(pca, file_path, mode='w', compression='gzip')

def load_compressor(layer, dimension, compression):
    """
    Loads the compression algorithm from the file system

    :type layer: str
    :param layer: feature layer

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


def batch_gen(data, batch_size):
    """
    Simple generator for looping over an array in batches

    :type data: array-like
    :param data:

    :type batch_size: int
    :param batch_size:

    :return: generator
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def load_network():
    """
    Loads the caffe network. The type of network loaded is specified in the utils file.

    :return: caffe network
    """
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


def dump_feature_layer(layer, X, ids):
    """
    Saves out the feature layer using hickle

    :type layer: str
    :param layer: feature layer

    :type X: array_like
    :param X: feature values

    :type ids: array_like
    :param ids: file identifiers

    :return:
    """
    file_name = layer  + '_X_gzip.hkl'
    file_path = os.path.join(feature_dir, layer, file_name)
    print 'Saving : ', file_path

    # Dump data, with compression
    hkl.dump(X, file_path, mode='w', compression='gzip')

    # Compare filesizes
    print 'compressed:   %i bytes' % os.path.getsize(file_path)

    file_name = layer  + '_ids_gzip.hkl'
    file_path = os.path.join(feature_dir, layer, file_name)
    print 'Saving : ', file_path

    # Dump data, with compression
    hkl.dump(ids, file_path, mode='w', compression='gzip')

    # Compare filesizes
    print 'compressed:   %i bytes' % os.path.getsize(file_path)

def load_feature_layer(layer):
    """
    Loads the feature layer specified

    :param layer: A member of utils.feature_layers
    :return: X, imagenet_ids
    """
    if not layer in feature_layers:
        raise NotImplementedError('Feature Layer Type Not Found.')

    features_path = os.path.join(feature_dir, layer)
    files = os.listdir(features_path)
    N = len(files)

    if N <= 1:
        raise ValueError('Path provided contained no features : ' + features_path)

    start_time = time.clock()
    for file in files:
        sp = file.split('_')
        if 'X' in sp:
            X = hkl.load(os.path.join(features_path, file))
        elif 'ids' in sp:
            imagenet_ids = hkl.load(os.path.join(features_path, file))

    print 'Total Load Time for Layer : ', layer
    print 'Time (s) : ', time.clock() - start_time

    return X, imagenet_ids

def dump_scaler(layer, scaler):
    """
    Dumps the scalar

    :type layer: str
    :param layer: Feature layer

    :type: sklearn.preprocessing.StandardScalar()
    :param scaler:

    :return:
    """
    file_name = layer  + '_scalar_gzip.hkl'
    file_path = os.path.join(feature_dir, layer, file_name)
    print 'Saving : ', file_path

    # Dump data, with compression
    hkl.dump(scaler, file_path, mode='w', compression='gzip')

def load_scalar(layer):
    """
    Load the feature mean / variance scalar for the input layer

    :type layer: str
    :param layer: Feature layer

    :return: scalar
    """
    if not layer in feature_layers:
        raise NotImplementedError('Feature Layer Type Not Found.')

    features_path = os.path.join(feature_dir, layer)
    files = os.listdir(features_path)
    N = len(files)

    if N <= 1:
        raise ValueError('Path provided contained no features : ' + features_path)

    for file in files:
        sp = file.split('_')
        if 'scalar' in sp:
            scalar = hkl.load(os.path.join(features_path, file), safe=False)

    return scalar


def generate_test_set(n=1000):
    """
    Store the pointers to the files to be used in the test set.

    :param n: The number of files from the validation set to use as the test set.
    :return:
    """
    import random

    image_files = os.listdir(test_dir)
    N = len(image_files)
    random.shuffle(image_files)
    image_files = image_files[:n]

    hkl.dump(image_files, os.path.join(img_dir, "test_set.hkl"), mode='w')

def load_test_set():
    """
    Returns a list of files to images

    :return:
    """
    return hkl.load("../images/test_set.hkl", safe=False)

def load_distance_matrix(layer):
    """
    Returns the distance matrix as defined by the features of the provided layer
    Note that this must be generated beforehand using generate_dist_func

    :type layer: str
    :param layer: Feature layer

    :return: numpy array
    """
    return hkl.load(os.path.join(distances_dir, 'dist_matrix_' + layer + '.hkl'))

def dump_distance_matrix(layer, dist_matrix):
    """
    Writes the distance matrix to the appropriate location

    :type layer: str
    :param layer: Feature layer

    :type dist_matrix: array-like
    :param dist_matrix:

    :return:
    """
    hkl.dump(dist_matrix, os.path.join(distances_dir, 'dist_matrix_' + layer + '.hkl'))