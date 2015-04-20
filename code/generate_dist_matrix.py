import utils
import numpy as np
import time

# ------------------------ Script Parameters ---------------------
layer = 'fc7'
# ------------------------ Script Parameters ---------------------

train_labels = utils.load_train_class_labels()
X, ids = utils.load_feature_layer(layer=layer)
scalar = utils.load_scalar(layer=layer)

X = scalar.transform(X)

min_klass_id = min(train_labels.values())
max_klass_id = max(train_labels.values())

klass_seperator = {}
for i in range(min_klass_id, max_klass_id+1):
    klass_seperator[i] = []

for idx, img_file in enumerate(ids):
    klass = train_labels[img_file]
    feats = X[idx,:]
    klass_seperator[klass].append(feats)

for i in range(min_klass_id, max_klass_id+1):
    klass_seperator[i] = np.asarray(klass_seperator[i], dtype=np.float32)

# the covariance matrices are too large to store in RAM
klass_mean = {}
klass_std = {}
for i in range(min_klass_id, max_klass_id+1):
    klass_mean[i] = np.mean(klass_seperator[i], axis=0)

    noise = np.random.randn(klass_seperator[i].shape[0], klass_seperator[i].shape[1])
    klass_std[i] = np.std(klass_seperator[i] + 1e-7*noise, axis=0, dtype=np.float64)


distance_matrix = np.zeros((max_klass_id+1, max_klass_id+1), dtype=np.float32) - 1

print 'Starting to compute distance matrix'
start_time = time.clock()
for i in range(min_klass_id, max_klass_id+1):

    print 'Computing : ', i
    mean_i = klass_mean[i]
    std_i = klass_std[i]

    for j in range(min_klass_id, max_klass_id+1):

        # start_time = time.clock()
        mean_j = klass_mean[j]
        std_j = klass_std[j]

        weights_ij = np.exp(-np.square(mean_i - mean_j) / (2 * np.square(std_i)))

        # we don't care about similarities where the 0 values are the same
        sparse_j = mean_j < 0.01
        sparse_i = mean_i < 0.01

        weights_ij[sparse_i & sparse_j] = 1e-7

        distance_matrix[i,j] = np.sum( weights_ij * np.square(mean_i - mean_j)) / (np.sum(weights_ij))

        # print 'Distance : ', kl_distance_matrix[i,j]
        # print 'Time : ', elapsed_time

print 'Time : ', time.clock() - start_time

utils.dump_distance_matrix(layer, dist_matrix=distance_matrix)
print 'Done'