# this is a pyspark file, not python
# functions in this file are meant to read the entire telematic data, and given one driver
# produce a new RDD with 400 trips: 200 from original driver, labeled as '1' and sampled
# from other drivers labeled as '0'

# note: resulting RDD will have 400 trips, the rows of the RDD will be individual observations
# from each of those trips - therefore, there will be more then 400 rows

# note: these functions expect the data to be stored as multiple CSV files, one for each driver,
# stored in the same directory, with the format: 'x, y, driverID, tripID, step'
# and named as '1.csv', '2.csv', etc.

# note: these functions will output in the format: 'x, y, driverID, tripID, step, label'


# import modules
import numpy as np
import os as os
import re as re


# read directory of files and return a list of all driverIDs from csv's insid directory
def get_drivers(dirpath):
    """
    :param dirpath: string, path to directory containing driver csv's
    :return: list, contains all driverIDs as strings
    """
    try:
        allfiles = os.listdir()
        drivers = [re.sub(r'[^0-9]', '', i) for i in allfiles]
        return drivers
    except Exception as e:
        print e

# produces random samples of driverIDs and tripIDs in two separate lists
def random_samples(targ_driv, driv_list, K=200):
    """
    :param targ_driv: str, driverID we want to make false trips for
    :param driv_list: list, list of all drivers, produced by get_drivers()
    :param K: number of trips we want to make for targ_driv
    :return: tuple of lists, first list is random driverIDs, second list is list of tripIDs, both are strings
    """
    try:
        driv_list.remove(targ_driv) #removes the target driver from list of drivers to sample from
        drivers = np.random.choice(driv_list, K, True)
        trips = np.random.choice(np.arange(1,K+1).astype(str), K, True)
        return (drivers, trips)
    except Exception as e:
        print e

# reads directory of files and returns RDD of observations from trips in the sample (driverID, tripID combo)
def sample_data(path, driverIDs, tripIDs):
    """
    :param path: string, path to directory containing driver.csv's
    :param driverIDs: list, list of randomly sampled driverIDs as strings, produced by random_sample()
    :param tripIDs: list, list of randomly sampled tripIDs as strings, produced by random_samples()
        NOTE: the above two zip into a list of (driverID, tripID) tuples, with each tuple being a single item in the
        sample
    :return: RDD, contains only observations from the sample
    """
    combos = zip(driverIDs, tripIDs)
    samplefiles = [path + '/' + i + '.csv' for i in driverIDs]
    samplefiles = ','.join(samplefiles)
    RDD = sc.textFile(samplefiles)
    RDDsamples = RDD.filter(lambda x: (x[2],x[3]) in combos)
    return RDDsamples

# takes RDD of samples and assigns new driverID and tripID to observations in a new RDD
def ID_Data(targ_driver, RDD, K = 200):
    """
    :param targ_driver: string, target driver we used to generate samples
    :param RDD: RDD, trip data RDD produced by sample_data(), format will be original form (x,y,driverID,tripID,step)
    :param K: int, number of trips we sampled
    :return: RDD, in original format, but with driverID and tripID changed to look like new observations of the target
    driver
    """
    newID1 = [targ_driver] * K
    newID2 = np.arange(200, 201+K).astype(str)
    newID = zip(newID1, newID2)
    oldID = RDD.map(lambda x: (x[2],x[3])).collect()
    glossary = sc.parallelize(zip(oldID, newID))
    newRDD = RDD.map(lambda x: ((x[2],x[3]), ([x[0],x[1],x[4]]))).join(glossary)
    newID_RDD = newRDD.map(lambda x: (x[1][0][0], x[1][0][1], x[1][1][0], x[1][1][1], x[1][0][2]))
    return newID_RDD

####
#### guys, should step be int and not float? ditto for label...
####

# takes RDD in original form and converts it into key-value tuple with values being x,y,step,label
def processRDD(RDD, label):
    """
    :param RDD: RDD in original format (x,y,driverID,tripID,step)
    :param label: category of observation, 1 for positive, 0 for negative
    # note, not sure if it needs to be int or float
    :return: RDD, RDD returned in new key/value format: (driverID, tripID), (x, y, step, label)
    # note, x, y, step, and label will be floats
    """
    newRDD = RDD.map(lambda x: ((x[2],x[3]),(float(x[0]),float(x[1]),float(x[4]),label)))
    return newRDD


# takes a driver to target, path to directory of driver.csv's, and returns an RDD labeled with
# (driverID, tripID),(x,y,step,label), where a label 1 is from an actual trip, and label 0 is from
# a trip randomly sampled from other drivers
def labelRDDs(targ_driv, path, K=200):
    """
    :param targ_driv: string, driver we want to create positive and negative labeled data for
    :param path: string, path to directory where driver.csvs are stored
    :param K: int, number of negative (manufactured) trips to sample
    :return: RDD with key, value tuple where key is (driverID, tripID) and value is (x,y,step,label)
    """
    target = sc.textFile(path + '/' + targ_driv + '.csv')
    positives = processRDD(target, 1.0)
    driv_list = get_drivers(path)
    sampdriv, samptrip = random_samples(targ_driv, driv_list, K)
    samples = sample_data(path, sampdriv, samptrip)
    samplesRDD = ID_Data(targ_driv, samples, K)
    negatives = processRDD(samplesRDD, 0.0)
    finalRDD = positives.union(negatives)
    return finalRDD


"""     USE EXAMPLE

path = '/Users/coolguy/Desktop/ML2Project/Data/drivers/1'
driver = '1'
DATA = labelRDDs(driver, path)

"""
