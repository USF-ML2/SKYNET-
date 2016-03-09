__author__ = "mayank"

import features_m as fb
import sampling_improved as s
from pyspark import SparkContext
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel, RandomForest
import csv
import random
from pyspark.sql import Row, SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


#logFile = "/Users/mayankkedia/Downloads/spark-1.6.0-bin-hadoop2.6/README.md"
sc = SparkContext(appName="GBT MODEL")
#sc = SparkContext(pyFiles = ['/home/hadoop/features_m.py'])
#sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY)
#sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)

path = '/Users/mayankkedia/code/kaggle/axa_telematics/jsonsNEW/'
#path = 's3://aml-spark-training/drivers/'
#drivers = ['1', '2', '3', '11', '12', '13', '14', '286', '1060', '1280']

driver_sample = [int(s.all_drivers[i].partition(".")[0]) for i in random.sample(xrange(len(s.all_drivers)), 100)]
tree_num_range = range(5, 105, 5)

versions = [{"version": 1.0, "smoothed": False, "percentiles": False},
            {"version": 2.0, "smoothed": False, "percentiles": True},
            {"version": 3.0, "smoothed": True, "percentiles": False},
            {"version": 4.0, "smoothed": True, "percentiles": True}]

errors = []

FEATURE_VERSION = 1.0



def filter_false_positives(label, prediction):
    if label == 0.0 and prediction == 1.0:
        return True
    else:
        return False


def filter_false_negatives(label, prediction):
    if label == 1.0 and prediction == 0.0:
        return True
    else:
        return False


def create_rows_for_rdd(x):
    features = list(x[1])
    l = len(features) - 1
    label = float(features.pop(l))
    meta_data = x[0]
    return Row(label=label, features=Vectors.dense(features), meta_data=Vectors.dense(meta_data))


def create_feature_rdd(driver, path, sc, version):
    """

    :param driver:
    :param path:
    :param sc:
    :return:
    """
    driver_RDD = s.labelRDDs(driver, path, sc)
    feature_RDD = fb.step_level_features(fb.get_polars(driver_RDD), version["smoothed"])
    #print "Printing Step Level Feature Return"
    #print feature_RDD.take(1)
    trip_features = fb.trip_level_features(feature_RDD, version["percentiles"])
    #print "Printing Trip Level Feature Return"

    #print trip_features.take(1)
    sqlContext = SQLContext(sc)

    total_data = sqlContext.createDataFrame(trip_features.map(create_rows_for_rdd))
    return total_data


def calculate_accuracy_metrics(predictions, driver, version):

    metrics = {}

    metrics["driver"] = driver

    #predictions.select("prediction", "indexedLabel", "features", "meta_data").show(20)

    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel",
                                                  predictionCol="prediction")
    accuracy = round(evaluator.evaluate(predictions, {evaluator.metricName: "precision"}), 2)

    print "Test Error for driver {} is {} for version {}".format(driver, 1.0 - accuracy, version["version"])

    metrics["error_rate"] = 1.0-accuracy

    metrics["version"] = version["version"]

    gbtModel = model.stages[2]
    print(gbtModel)  # summary only

    return metrics

version = versions[1]
#for version in versions:
for num_tree in tree_num_range:
    for driver in driver_sample:

        # Importing Data

        total_data = create_feature_rdd(driver, path, sc, version)

        labelIndexer = StringIndexer(inputCol="label",
                                     outputCol="indexedLabel").fit(total_data)

        featureIndexer = VectorIndexer(inputCol="features",
                                       outputCol="indexedFeatures",
                                       maxCategories=4).fit(total_data)

        (trainingData, testData) = total_data.randomSplit([0.7, 0.3])

        gbt = GBTClassifier(labelCol="indexedLabel",
                            featuresCol="indexedFeatures",
                            maxIter=num_tree)

        pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])

        model = pipeline.fit(trainingData)

        predictions = model.transform(testData)

        errors.append(calculate_accuracy_metrics(predictions, driver, version))


with open('feature_selection.csv', 'a') as fp:
    writer = csv.DictWriter(fp,
                            fieldnames=["error_rate",
                                        "driver",
                                        "version"],
                            delimiter=",")
    writer.writeheader()
    for e in errors:
        writer.writerow(e)






###################### OLDER CODE FOR MODELING USING RDD's ################################


    """
    model_gtb = GradientBoostedTrees.trainClassifier(trainingData,
                                                     categoricalFeaturesInfo={},
                                                     numIterations=5)

    predictions_gtb = model_gtb.predict(testData.map(lambda x: x.features))
    labelsAndPredictions_gtb = testData.map(lambda lp: lp.label).zip(predictions_gtb)
    testErr_gtb = labelsAndPredictions_gtb.filter(lambda (v,p): v!=p).count()/float(testData.count())

    testErr_gtb = round(testErr_gtb, 2)
    positive_labeled = labelsAndPredictions_gtb.filter(lambda (v, p): v == 1.0)
    negative_labeled = labelsAndPredictions_gtb.filter(lambda (v, p): v == 0.0)
    pos_count = positive_labeled.count()
    neg_count = negative_labeled.count()

    false_positive_rate = labelsAndPredictions_gtb.filter(lambda (v, p):
                                                          filter_false_positives(v, p)).count()/float(pos_count)

    false_negative_rate = labelsAndPredictions_gtb.filter(lambda (v, p):
                                                          filter_false_negatives(v, p)).count()/float(neg_count)

    false_positive_rate = round(false_positive_rate, 2)
    false_negative_rate = round(false_negative_rate, 2)
    errors.append({ "error": testErr_gtb,
                    "false_pos": false_positive_rate,
                    "false_neg": false_negative_rate,
                    "pos_count": pos_count,
                    "neg_count": neg_count,
                    "driver": driver,
                    "features_version": FEATURE_VERSION})
    print 'Num of Positive cases in test {} for Driver {}'.format(pos_count, driver)
    print 'Num of Negative cases in test {} for Driver {}'.format(neg_count, driver)
    print 'Test Error = {} for Driver {}'.format(testErr_gtb, driver)
    print 'False Positive Error = {} for Driver {}'.format(false_positive_rate, driver)
    print 'False Negative Error = {} for Driver {}'.format(false_negative_rate, driver)

    """
    """
with open('limited_test_results.csv', 'a') as fp:
    writer = csv.DictWriter(fp,
                            fieldnames=["error",
                                        "false_pos",
                                        "false_neg",
                                        "pos_count",
                                        "neg_count",
                                        "driver",
                                        "features_version"],
                            delimiter=",")
    writer.writeheader()
    for e in errors:
        writer.writerow(e)

    """











