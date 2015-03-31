import caffe
import numpy as np
import os

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


# ------------------------ Script Parameters ---------------------
use_alexnet = False
batch_size = 10
feature_layers = ['conv3', 'conv4', 'fc6', 'fc7', 'pool1', 'pool2', 'pool5']
img_dir = "../images/imagenet"
feature_dir = "../features"
# ------------------------ Script Parameters ---------------------


net, params, blobs = load_network(use_alexnet)

image_files = os.listdir(img_dir)
print 'Total Files : ', len(image_files)
print 'Sample File Name : ', image_files[100]


for files in batch_gen(os.listdir(img_dir), batch_size=batch_size):
    years, types, img_ids = [], [], []
    images = []
    for file in files:
        year, type, postfix = file.split('_')
        id, file_type = postfix.split('.')
        years.append(year)
        types.append(type)
        img_ids.append(id)

        file_image = caffe.io.load_image(os.path.join(img_dir, file))
        images.append(file_image)


    prediction = net.predict(images)

    # save out all the features
    for i in range(batch_size):
        id = img_ids[i]

        for layer in feature_layers:
            features = net.blobs[layer].data[i]

            file_name = years[i] + '_' + layer + '_' + img_ids[i] + '.npy'
            file_path = os.path.join(feature_dir, layer, file_name)

            print 'Saving : ', file_path
            np.save(file_path, features)

