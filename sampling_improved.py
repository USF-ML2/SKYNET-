from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import lit
import numpy as np
import features_m as fb
import csv

all_drivers = []
with open('driver_nums.csv', 'r') as dn:
    reader = csv.reader(dn, delimiter=',')
    for r in reader:
        all_drivers.append(str(r[0]) + ".json")

K = 200


def maketups(x):
    left = (x[0], x[1])
    right = (x[2], x[3], x[4], x[5])
    return left, right


def labelRDDs(driver, path, sc):
    sqlContext = SQLContext(sc)

    target = driver + '.json'
    driver_pool = list(all_drivers)
    driver_pool.remove(target)

    sample_drivers = np.random.choice(driver_pool, K, replace=False)
    sample_drivers_paths = [path + i for i in sample_drivers]
    sampled = sqlContext.read.json(sample_drivers_paths)
    orig = sqlContext.read.json(path + target)
    samples = sampled.sample(False, .0055)
    samples = samples.withColumn('label', lit(0))
    orig = orig.withColumn('label', lit(1))
    rawdata = samples.unionAll(orig)
    rawdata = rawdata.select(rawdata['driver'],
                             rawdata['trip'],
                             rawdata['x'],
                             rawdata['y'],
                             rawdata['step'],
                             rawdata['label'])
    rawRDD = rawdata.rdd
    return rawRDD.map(maketups)
