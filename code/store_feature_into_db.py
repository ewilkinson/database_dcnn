import psycopg2
import numpy as np
import os


conn = psycopg2.connect(dbname="mydb", user="takeshi", password="asdfgh", host="127.0.0.1")

cur = conn.cursor()

#cur.execute("CREATE TABLE test (id serial PRIMARY KEY, num integer, data varchar);")
#cur.execute("INSERT INTO test (num, data) VALUES (%s, %s)",(100,"abc'def"))

cur.execute("DROP TABLE test;")
cur.execute("CREATE TABLE test (id serial PRIMARY KEY, num integer, data float8[]);")
#cur.execute("SELECT * FROM test;")


#------------------------ Script Parameters ---------------------

feature_layers = ['conv3', 'conv4', 'fc6', 'fc7', 'pool1', 'pool2', 'pool5']
feature_dir = "../features"
feature_files = os.listdir(feature_dir)
num_files_to_use = 30  #max = 50000
count = 0
for layer_dir in feature_files:
    feature_files = np.sort(os.listdir(feature_dir+'/'+layer_dir))
    if layer_dir == 'fc6':
        for file in feature_files:
            years, types, img_ids = [], [], []
            images = []
            print file
            feature = np.load(feature_dir+'/'+layer_dir+'/'+file)
            feature2 = feature.ravel().tolist()
            feature2 = feature2[0:100] # need to remove
            cur.execute("INSERT INTO test (num, data) VALUES (%s, %s)",(count,feature2))
            cur.execute("SELECT data FROM test;")
            
            #feature3 =cur.fetchone()
            print len(feature2)
            count = count + 1
            # this is just for testing
            if count == num_files_to_use: # need to remove
                break
 
 # this is for testing

#for i in range(num_files_to_use)  :
    
         
 
#print cur.fetchone()

conn.commit()
cur.close()
conn.close()