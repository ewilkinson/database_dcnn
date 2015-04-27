import psycopg2
import time
import utils


def store_feature(layers, compression):
    conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
    cur = conn.cursor()

    start_time = time.clock()

    train_labels = utils.load_train_class_labels()
    for layer in layers:
        # CREATE THE TABLE
        table_name = create_table_name(compression, layer)
        cur.execute("DROP TABLE IF EXISTS " + table_name + ";")
        table_command = "CREATE TABLE " + table_name + " (id serial PRIMARY KEY, file text, class integer, "
        insert_command = "INSERT INTO " + table_name + " (file, class,"
        values_sql = "VALUES(%s,%s,"

        dimensions = utils.get_dimension_options(layer, compression)
        if len(dimensions) == 0:
            print 'POSSIBLE ERROR: No dimensions loaded for ', layer, ' with ', compression
            continue

        for dim in dimensions:
            table_command += create_feature_column(dim)
            insert_command += create_feature_name(dim) + ","
            values_sql += "%s,"

        table_command = table_command[:-1] + ");"
        values_sql = values_sql[:-1] + ");"
        insert_command = insert_command[:-1] + ") " + values_sql

        print table_command
        print insert_command

        cur.execute(table_command)

        # INSERT DATA INTO TABLE

        # load the data
        X, imagenet_ids = utils.load_feature_layer(layer)
        scalar = utils.load_scalar(layer)

        X = scalar.transform(X)

        transforms = []
        # apply the compression algorithm
        for dim in dimensions:
            compressor = utils.load_compressor(layer, dim, compression)
            transforms.append(compressor.transform(X))

        value = []
        for i in range(X.shape[0]):
            file_name = imagenet_ids[i]
            value = [file_name, train_labels[file_name]]
            for X_prime in transforms:
                value.append(X_prime[i, :].tolist())

            cur.execute(insert_command, value)

        conn.commit()

    cur.close()
    conn.close()

    print 'Done Creating Tables'
    print 'Total Time : ', time.clock() - start_time

def store_instances(layers, compression):
    conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
    cur = conn.cursor()

    start_time = time.clock()

    for layer in layers:
        # CREATE THE TABLE
        table_name = create_table_name(compression, layer)
        insert_command = "INSERT INTO " + table_name + " (file, class,"
        values_sql = "VALUES(%s,%s,"

        dimensions = utils.get_dimension_options(layer, compression)
        if len(dimensions) == 0:
            print 'POSSIBLE ERROR: No dimensions loaded for ', layer, ' with ', compression
            continue

        for dim in dimensions:
            insert_command += create_feature_name(dim) + ","
            values_sql += "%s,"

        values_sql = values_sql[:-1] + ");"
        insert_command = insert_command[:-1] + ") " + values_sql

        print insert_command

        # INSERT DATA INTO TABLE

        # load the data
        X, ids = utils.load_instance_features(layer)
        scalar = utils.load_scalar(layer)

        X = scalar.transform(X)

        transforms = []
        # apply the compression algorithm
        for dim in dimensions:
            compressor = utils.load_compressor(layer, dim, compression)
            transforms.append(compressor.transform(X))

        value = []
        for i in range(X.shape[0]):
            file_name = ids[i]
            value = [file_name, -1]
            for X_prime in transforms:
                value.append(X_prime[i, :].tolist())

            cur.execute(insert_command, value)

        conn.commit()

    cur.close()
    conn.close()

    print 'Total Time : ', time.clock() - start_time

def drop_instances(layers, compression):
    conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
    cur = conn.cursor()

    start_time = time.clock()

    for layer in layers:
        # CREATE THE TABLE
        table_name = create_table_name(compression, layer)
        delete_command = "DELETE FROM " + table_name + " WHERE class = -1;"
        cur.execute(delete_command)
        conn.commit()

    cur.close()
    conn.close()

    print 'Total Time : ', time.clock() - start_time



def query_top_k(k, features, compression, layer, dimension):
    """
    Returns top k results according to the distance2 function

    Results are of the form [(file, class, distance)] sorted with closest item at 0

    :type k: int
    :param k: top k items returns
    :param features:

    :type compression: str
    :param compression: compression type identifier

    :type layer: str
    :param layer: feature layer

    :type dimension: int
    :param dimension: feature dimensionality

    :return:
    """
    if dimension != features.size:
        raise ValueError('Feature size did not match dimension of query requested.')

    conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
    cur = conn.cursor()

    sql_command = "SELECT file, class, distance2(%s," + create_feature_name(
        dimension) + ") as D FROM " + create_table_name(compression, layer) + " ORDER BY D ASC LIMIT " + str(k)
    cur.execute(sql_command, [features.tolist()])

    results = cur.fetchall()

    cur.close()
    conn.close()

    return results

def retrieve_compression_features(compression, layer, dimension):
    """
    Results are of the form [(file, class, feat)] sorted with closest item at 0

    :type compression: str
    :param compression: compression type identifier

    :type layer: str
    :param layer: feature layer

    :type dimension: int
    :param dimension: feature dimensionality

    :return: results
    """
    conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
    cur = conn.cursor()

    sql_command = "SELECT file, class, " + create_feature_name(dimension) + " FROM " + create_table_name(compression, layer) + ";"
    cur.execute(sql_command)

    results = cur.fetchall()

    cur.close()
    conn.close()

    return results

def query_distances_by_file(features, files, compression, layer, dimension):
    conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
    cur = conn.cursor()

    sql_command = "SELECT file, class, distance2(%s," + create_feature_name(
        dimension) + ") as D  FROM " + create_table_name(compression, layer) + " WHERE file IN ("

    for f in files:
        sql_command += "%s,"

    sql_command = sql_command[:-1] + ") ORDER BY D ASC;"

    values = [features.tolist()]
    values.extend(files)

    cur.execute(sql_command, values)

    results = cur.fetchall()

    cur.close()
    conn.close()

    return results

def create_table_name(compression, layer):
    return compression + '_' + layer


def create_feature_column(dim):
    return " feature" + str(dim) + " float8[],"


def create_feature_name(dim):
    return " feature" + str(dim)


if __name__ == '__main__':
    layers = ['fc7', 'fc6', 'pool5', 'conv4', 'conv3']
    compression = 'pca'
    store_feature(layers, compression)


