
from pyspark.sql import Row, SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import features_m as fb

versions = [{"version": 1.0, "smoothed": False, "percentiles": False},
            {"version": 2.0, "smoothed": False, "percentiles": True},
            {"version": 3.0, "smoothed": True, "percentiles": False},
            {"version": 4.0, "smoothed": True, "percentiles": True}]


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

    metrics = dict()
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
