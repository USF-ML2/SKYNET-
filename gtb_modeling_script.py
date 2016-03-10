__author__ = "mayank"

import sampling_improved as s
import modeling_utils as util
from pyspark import SparkContext
import csv
import random
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
import sys

errors = []


#sc = SparkContext(appName="GBT MODEL",
#                  pyFiles=['/home/hadoop/SKYNET-/features_m.py',
#                           '/home/hadoop/SKYNET-/sampling_improved.py',
#                           '/home/hadoop/SKYNET-/modeling_utils.py'])

sc = SparkContext(appName="GBT MODEL")

AWS_ACCESS_KEY = 'AKIAIXZCIKL5ZHV3TXBQ'
AWS_SECRET_ACCESS_KEY = '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'
#sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY)
#sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)

path = '/Users/mayankkedia/code/kaggle/axa_telematics/jsonsNEW/'
#path = 's3://aml-spark-training/drivers.json/'

#driver_sample = [int(s.all_drivers[i].partition(".")[0]) for i in random.sample(xrange(len(s.all_drivers)), 1)]
driver_sample = [int(sys.argv[1])]

tree_num_range = range(5, 105, 20)

FILENAME = "single_driver_hyperparameter_selection" + "_" + sys.argv[1] + ".csv"

# Write Header for CSV File which records this information
with open(FILENAME, 'a') as fp:
    writer = csv.DictWriter(fp, fieldnames=util.CSV_FIELDNAMES, delimiter=",")
    writer.writeheader()

""" MODELING ITERATIONS BEGIN HERE """

for driver in driver_sample:

    driver_RDD = s.labelRDDs(driver=driver, path=path, sc=sc)
    for version in util.versions:
        total_data = util.create_feature_rdd(driver_RDD, sc, version, s)
        labelIndexer = StringIndexer(inputCol="label",
                                     outputCol="indexedLabel").fit(total_data)
        featureIndexer = VectorIndexer(inputCol="features",
                                       outputCol="indexedFeatures",
                                       maxCategories=4).fit(total_data)

        (trainingData, testData) = total_data.randomSplit([0.7, 0.3])

        """
        splits = total_data.randomSplit([0.2]*5)
        for i, split in enumerate(splits):
            trainingData = None
            testData = splits[i]
            for j in range(5):
                if j != i:
                    if trainingData is None:
                        trainingData = splits[j]
                    else:
                        trainingData = trainingData.unionAll(splits[j])
        """
        for num_tree in tree_num_range:

            # Modeling using GBT Classifier
            gbt = GBTClassifier(labelCol="indexedLabel",
                                featuresCol="indexedFeatures",
                                maxIter=num_tree)

            pipeline = Pipeline(stages=[labelIndexer,
                                        featureIndexer,
                                        gbt])

            model = pipeline.fit(trainingData)

            tr_p = model.transform(trainingData)
            te_p = model.transform(testData)

            accuracy_metrics = util.create_metric_dictionary(test_predictions=te_p,
                                                             training_predictions=tr_p,
                                                             driver=driver,
                                                             version=version,
                                                             num_tree=num_tree,
                                                             cv=i)
            errors.append(accuracy_metrics)

            # Writing current model information to file
            with open(FILENAME, 'a') as fp:
                writer = csv.DictWriter(fp,
                                        fieldnames=util.CSV_FIELDNAMES,
                                        delimiter=",")
                writer.writerow(accuracy_metrics)


