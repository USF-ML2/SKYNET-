__author__ = "mayank"

import features_m as fb
import sampling as s
from pyspark import SparkContext
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel, RandomForest
import csv



#logFile = "/Users/mayankkedia/Downloads/spark-1.6.0-bin-hadoop2.6/README.md"
sc = SparkContext(appName="GBT MODEL")

AWS_ACCESS_KEY="AKIAICY6RQOLZF5NDSCA"
AWS_SECRET_ACCESS_KEY = "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"
sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY)
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)

path = '/Users/mayankkedia/code/kaggle/axa_telematics/driver_conca/'
#path = 's3://aml-spark-training/drivers/'
drivers = ['1', '2', '3', '11', '12', '13', '14', '286', '1060', '1280', '2240']


errors = []

for driver in drivers:
    driver_1_RDD = s.labelRDDs(driver, path, sc)
    feature_RDD = fb.step_level_features(fb.get_polars(fb.vectorRDD(driver_1_RDD)))
    trip_features = fb.trip_level_features(feature_RDD)

    total_data = trip_features.map(fb.create_labelled_vectors)

    (trainingData, testData) = total_data.randomSplit([0.7, 0.3])


    model_gtb = GradientBoostedTrees.trainClassifier(trainingData, categoricalFeaturesInfo={}, numIterations=4)

    predictions_gtb = model_gtb.predict(testData.map(lambda x:x.features))
    labelsAndPredictions_gtb = testData.map(lambda lp: lp.label).zip(predictions_gtb)
    testErr_gtb = labelsAndPredictions_gtb.filter(lambda (v,p): v!=p).count()/float(testData.count())
    errors.append({"driver":driver, "error":testErr_gtb})
    print 'Test Error = {} for Driver {}'.format(testErr_gtb, driver)


with open('test_results.csv', 'wb') as fp:
    writer = csv.DictWriter(fp, fieldnames=["driver", "error"], delimiter=",")
    writer.writeheader()
    for e in errors:
        writer.writerow(e)











