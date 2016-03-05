__author__ = "mayank"

import features_m as fb
import sampling as s
from pyspark import SparkContext
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel

#logFile = "/Users/mayankkedia/Downloads/spark-1.6.0-bin-hadoop2.6/README.md"
sc = SparkContext("local", "GBT MODEL")


path = '/Users/mayankkedia/code/kaggle/axa_telematics/sample_drivers'
driver = '1'
driver_1_RDD = s.labelRDDs('1', path, sc)

print "Length of Driver 1 RDD {}".format(driver_1_RDD.count())


feature_RDD = fb.step_level_features(fb.get_polars(fb.vectorRDD(driver_1_RDD)))
trip_features = fb.trip_level_features(feature_RDD)

total_data = trip_features.map(fb.create_labelled_vectors)
(trainingData, testData) = total_data.randomSplit([0.7, 0.3])
model = GradientBoostedTrees.trainClassifier(trainingData, categoricalFeaturesInfo={}, numIterations=3)

predictions = model.predict(testData.map(lambda x:x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v,p): v!=p).count()/float(testData.count())
print 'Test Error = {}'.format(testErr)






