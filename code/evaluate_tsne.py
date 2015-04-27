import sql
import utils
import os, time
import caffe
import numpy as np
import hickle as hkl
import matlab.engine

from sklearn.neighbors import KDTree

# ------------------------------------------------
# Script Params
# ------------------------------------------------

compression_types = ['tsne']
tsne_dim = 5

feature_layers = ['fc7'] #, 'conv4', 'conv3']
pca_dimensions = [64, 128, 256]


distance_matrix_layer = 'fc7'

# feature_layers = utils.feature_layers

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

# start matlab engine
eng = matlab.engine.start_matlab()

query_compressor = utils.load_compressor(layer='fc7',
                                               dimension=256,
                                               compression='pca')


# initialize results data object
results = {}
for c_type in compression_types:
    results[c_type] = {}
    for layer in feature_layers:
        results[c_type][layer] = {}
        for n_components in pca_dimensions:
            results[c_type][layer][n_components] = {'similarity_dist': [], 'avg_time': []}

for c_type in compression_types:
    for layer in feature_layers:
        scalar = utils.load_scalar(layer=layer)

        for n_components in pca_dimensions:
            X, ids, classes = utils.load_tsne_features(layer, n_components, tsne_dim)
            tree = KDTree(X, leaf_size=10)

            compressor = utils.load_compressor(layer=layer,
                                               dimension=n_components,
                                               compression='pca')

            count = 0
            for t_files in utils.batch_gen(test_files, batch_size=batch_size):

                if count % 50 == 0:
                    mean_dist = np.mean(results[c_type][layer][n_components]['similarity_dist'])
                    mean_time = np.mean(results[c_type][layer][n_components]['avg_time'])
                    print 'Evaluate Script :: C Type : ', c_type, ' // Layer : ', layer, ' // Dim : ', n_components, ' // Count : ', count
                    print 'Evaluate Script :: Similarity Distance : ', mean_dist, ' // Avg Time : ', mean_time

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

                    # for tsne
                    tsne_feat = comp_feat.tolist()

                    tsne_feat = matlab.double(tsne_feat)
                    tsne_feat = eng.tsne_testing_python(tsne_feat, tsne_dim, layer, n_components, 'pca')

                    tsne_feat = np.array(tsne_feat)
                    tsne_feat= tsne_feat.ravel()

                    st = time.time()

                    # query the KD tree to find the K closest items
                    dist, ind = tree.query(tsne_feat, k=50)
                    ind = np.asarray(ind, dtype=np.int32).ravel()

                    query_feat = query_compressor.transform(feat).ravel()
                    query_results = sql.query_distances_by_file(features=query_feat,
                                                                files=ids[ind].tolist(),
                                                                compression='pca',
                                                                layer='fc7',
                                                                dimension=256)

                    et = time.time()

                    t_class = test_labels[t_file]

                    worst_case = np.mean(dist_mat[t_class, :])
                    best_case = 0

                    class_distance = 0
                    for x in query_results:
                        class_distance += dist_mat[t_class, x[1]]
                    avg_dist = class_distance / len(query_results)

                    results[c_type][layer][n_components]['similarity_dist'].append((worst_case - avg_dist) / (worst_case - best_case))
                    results[c_type][layer][n_components]['avg_time'].append(et - st)

hkl.dump(results, 'tsne_' + str(tsne_dim) + '_results.hkl')
eng.exit()
