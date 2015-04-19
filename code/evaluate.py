import sql
import utils
import os, time
import caffe

# ------------------------------------------------
# Script Params
# ------------------------------------------------

compression_types = ['pca']

# feature_layers = utils.feature_layers
feature_layers = ['fc7']
# dimensions = [32,64,128,256,512]
dimensions = [64]

# top k items to be retrieved and measured
# if any one of the k items are valid then retrieval is considered a success
k = 5

# ------------------------------------------------
# End Params
# ------------------------------------------------
batch_size = 10 # shouldn't change this to anything more than 10 because caffe handles it in an unknown way.

# load the test set
test_files = utils.load_test_set()
net, params, blobs = utils.load_network()

# validation labels are the test set labels
test_labels = utils.load_test_class_labels()
labels = utils.load_english_labels()


# initialize results data object
results = {}
for c_type in compression_types:
    results[c_type] = {}
    for layer in feature_layers:
        results[c_type][layer] = {}
        for n_components in dimensions:
            results[c_type][layer][n_components] = {'n_success': 0, 'n_failure': 0, 'avg_time' : 0}


for c_type in compression_types:
    for layer in feature_layers:
        scalar = utils.load_scalar(layer=layer)

        for n_components in dimensions:
            compressor = utils.load_compressor(layer=layer,
                                               dimension=n_components,
                                               compression=c_type)

            count = 0
            for t_files in utils.batch_gen(test_files,batch_size=batch_size):

                if count % 50 == 0:
                    n_success = results[c_type][layer][n_components]['n_success']
                    n_failure = results[c_type][layer][n_components]['n_failure']
                    avg_time = results[c_type][layer][n_components]['avg_time']
                    print 'Evaluate Script :: C Type : ', c_type, ' // Layer : ', layer, ' // Dim : ', n_components, ' // Count : ', count
                    print 'Evaluate Script :: Success : ', n_success, ' // Failure : ', n_failure, ' // Avg Time : ', avg_time / (count + 1e-7)

                count += 1*batch_size

                images = []
                for t_file in t_files:
                    image_path = os.path.join(utils.test_dir, t_file)
                    images.append(caffe.io.load_image(image_path))

                # predict takes any number of images, and formats them for the Caffe net automatically
                prediction = net.predict(images, oversample=False)

                for i in range(batch_size):
                    feat = net.blobs[layer].data[0].ravel()
                    feat = scalar.transform(feat)

                    comp_feat = compressor.transform(feat).ravel()

                    # run the top k query and time it
                    start_time = time.clock()
                    query_results = sql.query_top_k(k=k,
                                                    features=comp_feat,
                                                    compression=c_type,
                                                    layer=layer,
                                                    dimension=n_components)
                    end_time = time.clock()

                    t_class = test_labels[t_file]
                    r_classes = []
                    for x in query_results:
                        r_classes.append(x[1])

                    outcome = 'n_success' if t_class in r_classes else 'n_failure'
                    results[c_type][layer][n_components][outcome] += 1
                    results[c_type][layer][n_components]['avg_time'] += end_time - start_time