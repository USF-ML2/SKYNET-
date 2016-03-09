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
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def create_rows_for_rdd(x):
    """

    :param x:
    :return:
    """
    features = list(x[1])
    l = len(features) - 1
    label = float(features.pop(l))
    meta_data = x[0]
    return Row(label=label,
               features=Vectors.dense(features),
               meta_data=Vectors.dense(meta_data))


def create_feature_rdd(driver, path, sc, version):
    """

    :param driver:
    :param path:
    :param sc:
    :return:
    """
    driver_RDD = s.labelRDDs(driver, path, sc)
    feature_RDD = fb.step_level_features(fb.get_polars(driver_RDD), version["smoothed"])
    trip_features = fb.trip_level_features(feature_RDD, version["percentiles"])
    sqlContext = SQLContext(sc)

    total_data = sqlContext.createDataFrame(trip_features.map(create_rows_for_rdd))
    return total_data


def calculate_accuracy_metrics(predictions):

    """
    Calculates accuracy metrics for a Prediction DataFrame

    :param predictions:
    :return:
    """
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel",
                                                  predictionCol="prediction")
    accuracy = round(evaluator.evaluate(predictions, {evaluator.metricName: "precision"}), 2)
    recall = round(evaluator.evaluate(predictions, {evaluator.metricName: "recall"}), 2)

    positive_cases = predictions.filter(predictions["indexedLabel"] == 1.0)
    negative_cases = predictions.filter(predictions["indexedLabel"] == 0.0)
    false_positive_cases = negative_cases.filter(positive_cases["prediction"] == 1.0)
    false_negative_cases = positive_cases.filter(positive_cases["prediction"] == 0.0)

    return [accuracy,
            recall,
            positive_cases.count(),
            negative_cases.count(),
            false_positive_cases.count(),
            false_negative_cases.count()]


def create_metric_dictionary(test_predictions, training_predictions, driver, version, num_tree):
    """
    Creates a dict with important metrics for this round of modeling
    :param test_predictions:
    :param training_predictions:
    :param driver:
    """

    metrics = {}
    metrics["driver"] = driver
    metrics["version"] = version["version"]
    metrics["num_trees"] = num_tree

    pred_stats = calculate_accuracy_metrics(test_predictions)
    for k in TEST_METRIC_FIELDS.keys():
        metrics[k] = pred_stats[TEST_METRIC_FIELDS[k]]

    train_pred_stats = calculate_accuracy_metrics(training_predictions)
    for k in TRAIN_METRIC_FIELD.keys():
        metrics[k] = train_pred_stats[TRAIN_METRIC_FIELD[k]]

    return metrics

########################## ACTUAL SCRIPT STARTS HERE #########################

versions = [{"version": 1.0, "smoothed": False, "percentiles": False},
            {"version": 2.0, "smoothed": False, "percentiles": True},
            {"version": 3.0, "smoothed": True, "percentiles": False},
            {"version": 4.0, "smoothed": True, "percentiles": True}]

errors = []

TEST_METRIC_FIELDS = {"test_precision": 0,
                      "test_recall": 1,
                      "test_positive_count": 2,
                      "test_negative_count": 3,
                      "test_fp_count": 4,
                      "test_fn_count": 5,
                      }
TRAIN_METRIC_FIELD = {"train_precision": 0,
                      "train_recall": 1,
                      "train_positive_count": 2,
                      "train_negative_count": 3,
                      "train_fp_count": 4,
                      "train_fn_count": 5}

METRIC_FIELDS = ["driver", "num_trees", "version"]

CSV_FIELDNAMES = METRIC_FIELDS + TEST_METRIC_FIELDS.keys() + TRAIN_METRIC_FIELD.keys()

FILENAME = "single_driver_hyperparameter_selection.csv"
#FEATURES_FILE = 'features_m.py'

#sc = SparkContext(appName="GBT MODEL",
#                  pyFiles=['/home/hadoop/SKYNET-/features_m.py','/home/hadoop/SKYNET-/sampling_improved.py' ])

sc = SparkContext(appName="GBT MODEL")

AWS_ACCESS_KEY='AKIAIXZCIKL5ZHV3TXBQ'
AWS_SECRET_ACCESS_KEY = '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'
#sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY)
#sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)

path = '/Users/mayankkedia/code/kaggle/axa_telematics/jsonsNEW/'
#path = 's3://aml-spark-training/drivers.json/'

driver_sample = [int(s.all_drivers[i].partition(".")[0]) for i in random.sample(xrange(len(s.all_drivers)), 1)]
tree_num_range = range(5, 10, 5)

# Write Header for CSV File which records this information
with open(FILENAME, 'a') as fp:
    writer = csv.DictWriter(fp, fieldnames=CSV_FIELDNAMES, delimiter=",")
    writer.writeheader()


#version = versions[3]
for version in versions:
    for num_tree in tree_num_range:
        for driver in driver_sample:

            # Importing Data

            total_data = create_feature_rdd(driver, path, sc, version)

            labelIndexer = StringIndexer(inputCol="label",
                                         outputCol="indexedLabel").fit(total_data)

            featureIndexer = VectorIndexer(inputCol="features",
                                           outputCol="indexedFeatures",
                                           maxCategories=4).fit(total_data)


            # Splitting Data into TEST/TRAIN

            (trainingData, testData) = total_data.randomSplit([0.7, 0.3])

            # Modeling using GBT Classifier
            gbt = GBTClassifier(labelCol="indexedLabel",
                                featuresCol="indexedFeatures",
                                maxIter=num_tree)

            pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])

            model = pipeline.fit(trainingData)

            tr_p = model.transform(trainingData)
            te_p = model.transform(testData)

            accuracy_metrics = create_metric_dictionary(test_predictions=te_p,
                                                        training_predictions=tr_p,
                                                        driver=driver,
                                                        version=version,
                                                        num_tree=num_tree)
            errors.append(accuracy_metrics)

            # Writing current model information to file
            with open(FILENAME, 'a') as fp:
                writer = csv.DictWriter(fp, fieldnames=CSV_FIELDNAMES, delimiter=",")
                writer.writerow(accuracy_metrics)






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











