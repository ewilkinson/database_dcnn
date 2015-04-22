import sql
import utils
import os, time
import caffe
import numpy as np
import hickle as hkl

# ------------------------------------------------
# Script Params
# ------------------------------------------------

compression_types = ['pca']

distance_matrix_layer = 'pool5'

# feature_layers = utils.feature_layers
feature_layers = ['fc7', 'fc6', 'pool5', 'conv4', 'conv3']
dimensions = [32,64,128,256,512]
# dimensions = [512]

# top k items to be retrieved and measured
k = 5

# number of test files to evaluate. Must keep small otherwise it will take too long
N = 1000

# ------------------------------------------------
# End Params
# ------------------------------------------------

batch_size = 10  # shouldn't change this to anything more than 10 because caffe handles it in an unknown way.

# load the test set
test_files = utils.load_test_set()
test_files = test_files[:N]
net, params, blobs = utils.load_network()

# validation labels are the test set labels
test_labels = utils.load_test_class_labels()
labels = utils.load_english_labels()

dist_mat = utils.load_distance_matrix(distance_matrix_layer)

# initialize results data object
results = {}
for c_type in compression_types:
    results[c_type] = {}
    for layer in feature_layers:
        results[c_type][layer] = {}
        for n_components in dimensions:
            results[c_type][layer][n_components] = {'similarity_dist': 0, 'avg_time': 0}

for c_type in compression_types:
    for layer in feature_layers:
        scalar = utils.load_scalar(layer=layer)

        for n_components in dimensions:
            compressor = utils.load_compressor(layer=layer,
                                               dimension=n_components,
                                               compression=c_type)

            count = 0
            for t_files in utils.batch_gen(test_files, batch_size=batch_size):

                if count % 50 == 0:
                    similarity_dist = results[c_type][layer][n_components]['similarity_dist']
                    avg_time = results[c_type][layer][n_components]['avg_time']
                    print 'Evaluate Script :: C Type : ', c_type, ' // Layer : ', layer, ' // Dim : ', n_components, ' // Count : ', count
                    print 'Evaluate Script :: Similarity Distance : ', similarity_dist / (
                    count + 1e-7), ' // Avg Time : ', avg_time / (count + 1e-7)

                count += 1 * batch_size

                images = []
                for t_file in t_files:
                    image_path = os.path.join(utils.test_dir, t_file)
                    images.append(caffe.io.load_image(image_path))

                # predict takes any number of images, and formats them for the Caffe net automatically
                prediction = net.predict(images, oversample=False)

                for i in range(batch_size):
                    t_file = t_files[i]
                    feat = net.blobs[layer].data[i].ravel()
                    feat = scalar.transform(feat)

                    comp_feat = compressor.transform(feat).ravel()

                    # run the top k query and time it
                    st = time.time()
                    query_results = sql.query_top_k(k=k,
                                                    features=comp_feat,
                                                    compression=c_type,
                                                    layer=layer,
                                                    dimension=n_components)

                    et = time.time()

                    t_class = test_labels[t_file]

                    worst_case = np.mean(dist_mat[t_class, :])
                    best_case = 0

                    class_distance = 0
                    for x in query_results:
                        class_distance += dist_mat[t_class, x[1]]
                    avg_dist = class_distance / len(query_results)

                    results[c_type][layer][n_components]['similarity_dist'] += (worst_case - avg_dist) / (worst_case - best_case)
                    results[c_type][layer][n_components]['avg_time'] += et - st

            results[c_type][layer][n_components]['similarity_dist'] /= len(test_files)
            results[c_type][layer][n_components]['avg_time'] /= len(test_files)

hkl.dump(results, 'results_pca.hkl')