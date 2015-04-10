import psycopg2
import time
import utils
import numpy as np

def store_feature(layers, compression):
    conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
    cur = conn.cursor()

    start_time = time.clock()
    for layer in layers:
        # CREATE THE TABLE
        table_name = compression + '_' + layer
        cur.execute("DROP TABLE IF EXISTS " + table_name + ";")
        table_command = "CREATE TABLE " + table_name + " (id serial PRIMARY KEY, imagenet_id integer, "
        insert_command = "INSERT INTO " + table_name + " (imagenet_id,"
        values_sql = "VALUES(%s,"

        dimensions = utils.get_dimension_options(layer, compression)
        if len(dimensions) == 0:
            print 'POSSIBLE ERROR: No dimensions loaded for ', layer, ' with ', compression
            continue

        for dim in dimensions:
            table_command += " feature" + str(dim) + " float8[],"
            insert_command += " feature" + str(dim) + ","
            values_sql += "%s,"

        table_command = table_command[:-1] + ");"
        values_sql = values_sql[:-1] + ");"
        insert_command = insert_command[:-1] + ") " + values_sql
        print insert_command

        cur.execute(table_command)

        # INSERT DATA INTO TABLE

        # load the data
        X, imagenet_ids, scalar = utils.load_feature_layer(layer)

        imagenet_ids = np.asarray(imagenet_ids, dtype='int64')
        # debug only
        X = X[0:1000, :]
        imagenet_ids = imagenet_ids[0:1000]

        X = scalar.transform(X)

        transforms = []
        # apply the compression algorithm
        for dim in dimensions:
            compressor = utils.load_compressor(layer,dim,compression)
            transforms.append(compressor.transform(X))

        value = []
        for i in range(X.shape[0]):
            value = [imagenet_ids[i]]
            for X_prime in transforms:
                value.append(X_prime[i,:].tolist())

            cur.execute(insert_command, value)

    conn.commit()
    cur.close()
    conn.close()

    print 'Done Creating Tables'
    print 'Total Time : ', time.clock() - start_time


if __name__ == '__main__':
    layers = ['fc7']
    compression = 'pca'
    store_feature(layers, compression)
