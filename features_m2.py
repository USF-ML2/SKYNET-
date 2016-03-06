__author__ = "harry"

import numpy as np
import math
from pyspark.mllib.regression import LabeledPoint

"""
Consists of functions to help us build features on RDDs.

"""


def vectorRDD(RDD):
    """
    :param RDD: RDD, created from labelRDDs in 'sampling.py'
    :return: RDD with driver_id, trip_id, vectorized x and y coordinates,
    step number, and label
    """

    # The RDD created from labelRDDs is first mapped into the key, value pair
    # ((driver_id, trip_id), (x, y, step)). Each element in the value is a
    # list of one element. This is so we can create a list of each x, y, and
    # step coordinate when reducing. We then reduce by key creating a list of
    # x and y coordinates, and a list of trip step number. This will create an
    # RDD of the form ((driver_id, trip_id), ([x coordinates],
    # [y coordinates], [trip steps])). Finally, we map this into the key, value
    # pair ((driver_id, trip_id), ([x cooridinates], [y coordinates],
    # [trip steps], label)). This RDD is the returned.

    vectorRDD = RDD.map(lambda x: (x[0], ([x[1][0]], [x[1][1]],
        [x[1][2]])))\
    .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]))\
    .map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2], 1)) if int(x[0][1])\
        < 201 else (x[0], (x[1][0], x[1][1], x[1][2], 0)))
    return vectorRDD


def get_polars(RDD):
    """
    :param RDD: RDD, created from vectorRDD
    "return: RDD, same as vectorRDD but also with polar coordinates
    """

    # The RDD created from vectorRDD is mapped into the key, value pair
    # ((driver_id, trip_id), ([x coordinates], [y coordinates],
    # [r coordinates], [theta coordinates], [step numbers], label)). Here,
    # the r coordinate is the radial coordinate from the origin, computed using
    # Euclidean distance, i.e. the first inner map statement. The theta
    # coordinate is the angular polar coordinate, computed as the arctan of
    #  y / x, i.e. the second inner map statement.

    polars = RDD.map(lambda x: (x[0], (x[1][0], x[1][1],
        map(lambda x: (x[0] ** 2 + x[1] ** 2) ** 0.5, zip(x[1][0], x[1][1])),
        map(lambda x: math.atan2(x[1], x[0]), zip(x[1][0], x[1][1])),
         x[1][2], x[1][3])))
    return polars


def step_level_features(row):
    """
    :param row: tuple, ((driver_id, trip_id), ([x coordinates], 
    	[y coordinates], [r coordinates], [theta coordinates], [step number],
    	label))
    :return: tuple, ((driver_id, trip_id), ([x coordinates], [y coordinates], 
    	[r coordinates], [theta coordinates], [vel coordinates], 
    	[acc coordinates], [cent_acc coordinates], [ang_vel coordinates],
    	[tan_acc coordinates], [jerk coordinates], [step number], label))
    """

    x = row[1][0]
    y = row[1][1]
    r = row[1][2]
    theta = row[1][3]
    theta_cp = theta[:]
    step = row[1][4]
    label = row[1][5]
    
    x2 = [(x_new - x_old) ** 2 for x_new, x_old in zip(x, [0.0] + x[:-1])]
    y2 = [(y_new - y_old) ** 2 for y_new, y_old in zip(y, [0.0] + y[:-1])]
    vel = [(x_sq + y_sq) ** 0.5 for x_sq, y_sq in zip(x2, y2)]
    acc = [v_new - v_old for v_new, v_old in zip(vel, [0.0] + vel[:-1])]
    
    cent_acc =[0 if rad == 0 else v ** 2 / rad for rad, v in zip(r, vel)]
    ang_vel = [new - old for new, old in zip(theta, [0.0] + theta[:-1])]
    tan_acc = [new - old for new, old in zip(ang_vel, [0.0] + ang_vel[:-1])]
    jerk = [new - old for new, old in zip(acc, [0.0] + acc[:-1])]
    
    return (row[0], (x, y, r, theta, vel, acc, cent_acc, ang_vel, tan_acc, 
    	step, label))


def get_velocity_percentiles(row):
    """
    Generates the percentiles for 5, 10, 15 ... 95 for Velocity
    """
    v = row[1][4]
    return np.percentile(v, range(5, 100, 5))


def get_acceleration_percentiles(row):
    """
    Generates the percentiles for 5, 10, 15 ... 95 for Acceleration
    """
    a = row[1][5]
    return np.percentile(a, range(5, 100, 5))


def trip_features(x):
    """
    Calculates the features of the trip from a row which is of the form
    ((driver_id, trip_id), ([x coordinates], [y coordinates],
    [r coordinates], [theta coordinates], [v coordinates], [step numbers], label))
    This is the form of the rows of the output from step_level_features.

    :@param x:
    """
    min_v = min(x[1][4])
    max_v = max(x[1][4])
    min_a = min(x[1][5])
    max_a = max(x[1][5])
    trip_length = len(x[1][0])
    mean_v = np.mean(x[1][4])
    std_v = np.std(x[1][4])
    mean_a = np.mean(x[1][5])
    std_a = np.std(x[1][5])
    time_stop = sum([elem < 0.5 for elem in x[1][4]])
    label = x[1][7]

    numerical_features = (min_v, max_v,
                   min_a, max_a,
                   trip_length,
                   mean_v, std_v,
                   mean_a, std_a,
                   time_stop,
                   label)

    v_percentiles = get_velocity_percentiles(x)
    a_percentiles = get_acceleration_percentiles(x)
    percentiles = np.append(v_percentiles, a_percentiles)
    second_tuple = np.append(percentiles, numerical_features).tolist()
    return (x[0], second_tuple)


def trip_level_features(RDD):
    """
    :param RDD: RDD, created from step_level_features
    :return: RDD with features, aggregated over the trip
    """

    # The RDD created from step_level_features is mapped to the key, value
    # pair ((driver_id, trip_id), (min(v), max(v), min(a), max(a), trip
    # length, trip distance, mean(v), stddev(v), mean(a), stddev(a),
    # length of time stopped, label). Trip length (s) is the number of
    # coordinates. Trip distance (m) is computed as the sum of the speeds as
    # they are computed per second. Length of time stopped is computed as the
    # number of seconds where the speed is less than 0.5 m/s.

    trip_lv = RDD.map(trip_features)

    return trip_lv

def create_labelled_vectors(x):
     vector = list(x[1])
     l = len(vector) -1
     label = float(vector.pop(l))
     return LabeledPoint(label, vector)