import sql
import utils
import os, time
import caffe
import numpy as np

import lshash
# ------------------------------------------------
# Script Params
# ------------------------------------------------

compression_types = ['lsh']

distance_matrix_layer = 'fc7'

# feature_layers = utils.feature_layers
feature_layers = ['fc7']
# dimensions = [128, 256, 512, 1024, 2048]
dimensions = [256]

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
            results[c_type][layer][n_components] = {'similarity_dist': [], 'avg_time': [], 'success': []}


train_labels = utils.load_train_class_labels()



for c_type in compression_types:
    for layer in feature_layers:
        X, ids = utils.load_feature_layer(layer)

        for n_components in dimensions:
            lsh = lshash.LSHash(dimensions[0], X.shape[1], num_hashtables=1,
                 storage_config=None, matrices_filename=None, overwrite=False)

            hashes = []
            from bitarray import bitarray
            for x in X:
                hashes.append(bitarray(lsh._hash(lsh.uniform_planes[0], x)))

            count = 0
            for t_files in utils.batch_gen(test_files, batch_size=batch_size):

                if count % 50 == 0:
                    mean_dist = np.mean(results[c_type][layer][n_components]['similarity_dist'])
                    mean_time = np.mean(results[c_type][layer][n_components]['avg_time'])
                    success_perc = np.mean(results[c_type][layer][n_components]['success'])
                    print 'Evaluate Script :: C Type : ', c_type, ' // Layer : ', layer, ' // Dim : ', n_components, ' // Count : ', count
                    print 'Evaluate Script :: Similarity Distance : ', mean_dist, ' // Avg Time : ', mean_time, '// Success Rate : ', success_perc

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

                    st = time.clock() # START CLOCK

                    query_hash = bitarray(lsh._hash(lsh.uniform_planes[0], feat))
                    distances = []
                    for hash in hashes:
                        xor_result = hash ^ query_hash
                        distances.append(xor_result.count())

                    distances = np.array(distances)
                    sort_idxs = np.argsort(distances)
                    query_results = sort_idxs[:k]
                    et = time.clock() # END CLOCK

                    t_class = test_labels[t_file]

                    worst_case = np.mean(dist_mat[t_class, :])
                    best_case = 0

                    has_success = 0
                    class_distance = 0

                    for x in query_results:
                        dist = dist_mat[t_class, train_labels[ids[x]]]
                        if dist == 0:
                            has_success = 1
                        class_distance += dist

                    avg_dist = class_distance / len(query_results)

                    results[c_type][layer][n_components]['similarity_dist'].append((worst_case - avg_dist) / (worst_case - best_case))
                    results[c_type][layer][n_components]['avg_time'].append(et - st)
                    results[c_type][layer][n_components]['success'].append(has_success)

    # utils.dump_results(results, c_type, distance_matrix_layer)