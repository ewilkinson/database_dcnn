import sql
import utils
import os
import numpy as np
import caffe


# image_file = 'ILSVRC2012_val_00025381.JPEG'
image_file = 'ILSVRC2012_val_00015447.JPEG'
layer = 'fc7'
dimension = 32
compression = 'pca'
k = 5

dist_mat = utils.load_distance_matrix('pool5')

compressor = utils.load_compressor(layer=layer,
                                   dimension=dimension,
                                   compression=compression)

scalar = utils.load_scalar(layer=layer)

net, params, blobs = utils.load_network()

input_image = caffe.io.load_image(os.path.join(utils.test_dir, image_file))

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

test_labels = utils.load_test_class_labels()
labels = utils.load_english_labels()

# compute the expected similarity for random guessing
test_class = test_labels[image_file]
worst_case = np.mean(dist_mat[test_class, :])
best_case = 0.625 * 0 + (1-0.625) * worst_case

class_distance = 0
for x in results:
    class_distance += dist_mat[test_class, x[1]]

avg_dist = class_distance / len(results)

print 'Avg Distace : ', avg_dist
print 'Worst Case : ', worst_case
print 'Best Case : ', best_case
print 'Percent Of Optimal : ', (worst_case - avg_dist) / (worst_case - best_case)

print 'Query class : ', test_class, labels[test_class]
for x in results:
    print x[0], ' Class : ', x[1], labels[x[1]],  ' Distance : ', x[2]


# import matplotlib.pyplot as plt
#
# plt.imshow(input_image)
# plt.plot()
#
# image_file = results[0][0]
# image = caffe.io.load_image(os.path.join(utils.img_dir, image_file))
# plt.imshow(image)
# plt.plot()