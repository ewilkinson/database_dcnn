import psycopg2
import numpy as np
import os
import time
import hickle as hkl
import utils

dimensions = [32, 64, 128, 256]

def store_feature(layer, compression):

    conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
    cur = conn.cursor()
    # cur.execute("DROP TABLE IF EXISTS featuretable;")

    for condim in dimensions :
        sqlcommand = "CREATE TABLE feature" + str(condim)+ "table (id serial PRIMARY KEY, num integer, feature" + str(condim) + " float8[]);"
        cur.execute(sqlcommand);

    X, ids = utils.load_feature_layer(layer)
    X = X[0:1000,:]
    ids = ids[0:1000]



    pca_path = os.path.join(utils.compression_dir+"/" + compression + "/",layer)
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

    for condim in dimensions :
        sqlcommand = "DROP TABLE feature" + str(condim)+ "table;"
        cur.execute(sqlcommand);

    print 'DONE!'
    conn.commit()
    cur.close()
    conn.close()

if __name__ == '__main__':
    layer = 'fc7'
    compression = 'pca'
    store_feature(layer, compression)
