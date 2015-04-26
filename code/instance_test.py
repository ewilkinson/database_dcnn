import sql
import utils
import os
import numpy as np
import caffe


image_file = 'ubot5_1.JPG'
layer = 'pool5'
dimension = 256
compression = 'pca'
k = 10

compressor = utils.load_compressor(layer=layer,
                                   dimension=dimension,
                                   compression=compression)

scalar = utils.load_scalar(layer=layer)

net, params, blobs = utils.load_network()

input_image = caffe.io.load_image(os.path.join(utils.instances_dir, image_file))

# predict takes any number of images, and formats them for the Caffe net automatically
prediction = net.predict([input_image], oversample=False)
feat = net.blobs[layer].data[0].ravel()
feat = scalar.transform(feat)

comp_feat = compressor.transform(feat).ravel()

results = sql.query_top_k(k=k,
                          features=comp_feat,
                          compression=compression,
                          layer=layer,
                          dimension=dimension)

# compute the expected similarity for random guessing

for x in results:
    print x[0], ' Class : ', x[1], ' Distance : ', x[2]


import matplotlib.pyplot as plt
#
# plt.imshow(input_image)
# plt.plot()
#
image_file = results[8][0]
image = caffe.io.load_image(os.path.join(utils.img_dir, image_file))
plt.imshow(image)
plt.plot()