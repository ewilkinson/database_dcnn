import psycopg2
import numpy as np
import os
import time
import hickle as hkl
import utils

feature_layers = ['conv3', 'conv4', 'fc6', 'fc7', 'pool1', 'pool2', 'pool5']
dimensions = [32, 64, 128, 256]
img_dir = "../images/imagenet"
feature_dir = "../features"
compression_dir = "../compression"
#caffe_root = '/home/eric/caffe/caffe-master/'
caffe_root = '/home/takeshi//work/CS645/project/Caffe/caffe-master/'




def store_feature(layer, compression):

    conn = psycopg2.connect(dbname="mydb", user="takeshi", password="asdfgh", host="127.0.0.1")
    cur = conn.cursor()
    # cur.execute("DROP TABLE featuretable;")
    # cur.execute("DROP TABLE feature32table;")
    # cur.execute("DROP TABLE feature64table;")
    # cur.execute("DROP TABLE feature128table;")
    # cur.execute("DROP TABLE feature256table;")

    #cur.execute("CREATE TABLE featuretable (id serial PRIMARY KEY, num integer, feature32 float8[], feature64 float8[], feature128 float8[], feature256 float8[]);")

    for condim in dimensions :
        sqlcommand = "CREATE TABLE feature" + str(condim)+ "table (id serial PRIMARY KEY, num integer, feature" + str(condim) + " float8[]);"
        cur.execute(sqlcommand);
    #cur.execute("CREATE TABLE feature32table (id serial PRIMARY KEY, num integer, feature32 float8[]);")
    #cur.execute("CREATE TABLE feature64table (id serial PRIMARY KEY, num integer, feature64 float8[]);")
    #cur.execute("CREATE TABLE feature128table (id serial PRIMARY KEY, num integer, feature128 float8[]);")
    #cur.execute("CREATE TABLE feature256table (id serial PRIMARY KEY, num integer, feature256 float8[]);")

    #cur.execute("CREATE TABLE test2 (id serial PRIMARY KEY, num int, feature32 float8[]);")
#             cur.execute("SELECT data FROM test;")

    #layer = 'fc7'
    X, ids = utils.load_feature_layer(layer)
    X = X[0:1000,:]
    ids = ids[0:1000]



    pca_path = os.path.join(compression_dir+"/" + compression + "/",layer)
    files = os.listdir(pca_path)
    N = len(files)

    if N <= 1:
        raise ValueError('Path provided contained no features : ' + pca_path)

    # there is a holder file in each directory which needs to be removed
    files.remove('holder.txt')


    start_time = time.clock()
    for file in files:
        pca = hkl.load(os.path.join(pca_path, file),safe=False)
        n_components = pca.n_components_
        pca.fit(X)
        X_prime = pca.transform(X)
        rows, cols = X_prime.shape

        sp = file.split('_')
        feature_name = "feature"+sp[1];

        sqlcommand = "INSERT INTO " + feature_name + "table (num," + feature_name + ") VALUES(%s, %s)"
        print sqlcommand

        for i in range (rows) :
            cur.execute(sqlcommand, (i,X_prime[i].ravel().tolist(),))

        #cur.execute(string1, (i,X_prime[i].ravel().tolist(),))

        # if feature_name == "feature32" :
        #     for i in range (rows) :
        #         cur.execute("INSERT INTO feature32table (num, feature32) VALUES (%s, %s)", (i,X_prime[i].ravel().tolist(),))
        #         string1 = "INSERT INTO " + feature_name + "table (num," + feature_name + ") VALUES(%s, %s)"
        #
        #         #string1 = "INSERT INTO feature32table (num, feature32) VALUES (%s, %s)"
        #         cur.execute(string1, (i,X_prime[i].ravel().tolist(),))
        # elif feature_name == "feature64" :
        #     for i in range (rows) :
        #         cur.execute("INSERT INTO feature64table (num, feature64) VALUES (%s, %s)", (i,X_prime[i].ravel().tolist(),))
        # elif feature_name == "feature128" :
        #     for i in range (rows) :
        #         cur.execute("INSERT INTO feature128table (num, feature128) VALUES (%s, %s)", (i,X_prime[i].ravel().tolist(),))
        # elif feature_name == "feature256" :
        #     for i in range (rows) :
        #         cur.execute("INSERT INTO feature256table (num, feature256) VALUES (%s, %s)", (i,X_prime[i].ravel().tolist(),))


        # X_prime.le
        #
        # for range()
        # X_prime = X_prime.ravel().tolist()
        #
        # sp = file.split('_')
        #
        # feature_name = "feature"+sp[1];
        # print feature_name
        # cur.execute("INSERT INTO test (feature32) VALUES (%s)", X_prime)

    sqlcommand = "select "
    for condim in dimensions:
        sqlcommand = sqlcommand + "f" + str(condim) + ".feature"+str(condim)
        if condim != dimensions[len(dimensions)-1]:
            sqlcommand = sqlcommand + ","

    sqlcommand = sqlcommand + " into featuretable" + compression + layer + " from "

    for i in range(len(dimensions)):
        condim = dimensions[i]
        if condim == dimensions[0]:
            sqlcommand = sqlcommand + " feature" + str(condim) + "table as f" +str(32)
        else :
            sqlcommand = sqlcommand + " left outer join feature" + str(condim) + "table as  " + "f" +str(condim) + " on f" + str(dimensions[i-1]) + ".id = " + "f" + str(condim) + ".id "

    sqlcommand = sqlcommand + ";"
    print sqlcommand
    cur.execute(sqlcommand)
     #string2 = "select f32.id, f32.feature32, f64.feature64, f128.feature128, f256.feature256 into featuretable" + layer + " from " +  "feature32table as f32 left outer join feature64table as f64 on f32.id = f64.id left outer join feature128table as f128 on f64.id = f128.id left outer join feature256table as f256 on f128.id = f256.id;"
     #cur.execute(string2);


    # if layer == "conv3" :
    #     cur.execute("select f32.id, f32.feature32, f64.feature64, f128.feature128, f256.feature256 into featuretableconv3 from feature32table as f32 left outer join feature64table as f64 on f32.id = f64.id left outer join feature128table as f128 on f64.id = f128.id left outer join feature256table as f256 on f128.id = f256.id;")
    # elif layer == "conv4" :
    #     cur.execute("select f32.id, f32.feature32, f64.feature64, f128.feature128, f256.feature256 into featuretableconv4 from feature32table as f32 left outer join feature64table as f64 on f32.id = f64.id left outer join feature128table as f128 on f64.id = f128.id left outer join feature256table as f256 on f128.id = f256.id;")
    # elif layer == "fc6" :
    #     cur.execute("select f32.id, f32.feature32, f64.feature64, f128.feature128, f256.feature256 into featuretablefc6 from feature32table as f32 left outer join feature64table as f64 on f32.id = f64.id left outer join feature128table as f128 on f64.id = f128.id left outer join feature256table as f256 on f128.id = f256.id;")
    # elif layer == "fc7" :
    #     cur.execute("select f32.id, f32.feature32, f64.feature64, f128.feature128, f256.feature256 into featuretablefc7 from feature32table as f32 left outer join feature64table as f64 on f32.id = f64.id left outer join feature128table as f128 on f64.id = f128.id left outer join feature256table as f256 on f128.id = f256.id;")
    # elif layer == "pool1" :
    #     cur.execute("select f32.id, f32.feature32, f64.feature64, f128.feature128, f256.feature256 into featuretablepool1 from feature32table as f32 left outer join feature64table as f64 on f32.id = f64.id left outer join feature128table as f128 on f64.id = f128.id left outer join feature256table as f256 on f128.id = f256.id;")
    # elif layer == "pool2" :
    #     cur.execute("select f32.id, f32.feature32, f64.feature64, f128.feature128, f256.feature256 into featuretablepool2 from feature32table as f32 left outer join feature64table as f64 on f32.id = f64.id left outer join feature128table as f128 on f64.id = f128.id left outer join feature256table as f256 on f128.id = f256.id;")
    # elif layer == "pool5" :
    #     cur.execute("select f32.id, f32.feature32, f64.feature64, f128.feature128, f256.feature256 into featuretablepool5 from feature32table as f32 left outer join feature64table as f64 on f32.id = f64.id left outer join feature128table as f128 on f64.id = f128.id left outer join feature256table as f256 on f128.id = f256.id;")

    #cur.execute("DROP TABLE featuretable;")
    for condim in dimensions :
        sqlcommand = "DROP TABLE feature" + str(condim)+ "table;"
        cur.execute(sqlcommand);

    # cur.execute("DROP TABLE feature32table;")
    # cur.execute("DROP TABLE feature64table;")
    # cur.execute("DROP TABLE feature128table;")
    # cur.execute("DROP TABLE feature256table;")

    print 'DONE!'
    conn.commit()
    cur.close()
    conn.close()

if __name__ == '__main__':
    layer = 'fc7'
    compression = 'pca'
    store_feature(layer, compression)

# conn = psycopg2.connect(dbname="mydb", user="takeshi", password="asdfgh", host="127.0.0.1")
#
# cur = conn.cursor()
#
# #cur.execute("CREATE TABLE test (id serial PRIMARY KEY, num integer, data varchar);")
# #cur.execute("INSERT INTO test (num, data) VALUES (%s, %s)",(100,"abc'def"))
#
# cur.execute("DROP TABLE test;")
# cur.execute("CREATE TABLE test (id serial PRIMARY KEY, num integer, data float8[]);")
# #cur.execute("SELECT * FROM test;")
#
#




# #------------------------ Script Parameters ---------------------
#
# feature_layers = ['conv3', 'conv4', 'fc6', 'fc7', 'pool1', 'pool2', 'pool5']
# feature_dir = "../features"
# feature_files = os.listdir(feature_dir)
# num_files_to_use = 30  #max = 50000
# count = 0
# for layer_dir in feature_files:
#     feature_files = np.sort(os.listdir(feature_dir+'/'+layer_dir))
#     if layer_dir == 'fc6':
#         for file in feature_files:
#             years, types, img_ids = [], [], []
#             images = []
#             print file
#             feature = np.load(feature_dir+'/'+layer_dir+'/'+file)
#             feature2 = feature.ravel().tolist()
#             feature2 = feature2[0:100] # need to remove
#             cur.execute("INSERT INTO test (num, data) VALUES (%s, %s)",(count,feature2))
#             cur.execute("SELECT data FROM test;")
#
#             #feature3 =cur.fetchone()
#             print len(feature2)
#             count = count + 1
#             # this is just for testing
#             if count == num_files_to_use: # need to remove
#                 break
 
 # this is for testing

#for i in range(num_files_to_use)  :
    
         
 
#print cur.fetchone()

# conn.commit()
# cur.close()
# conn.close()

