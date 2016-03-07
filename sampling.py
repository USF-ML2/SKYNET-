__author__ = "su-young"

import numpy as np
import os as os
import re as re
import csv

# Might need to hardcode list of drivers


all_drivers = []
with open('driver_nums.csv', 'r') as dn:
    reader = csv.reader(dn, delimiter=',')
    for r in reader:
        all_drivers.append(str(r[0]))


def get_drivers(dirpath):
    """

    Read directory of files and return a list of all driverIDs from csv's insid directory
    :param dirpath: string, path to directory containing driver csv's
    :return: list, contains all driverIDs as strings
    """

    return all_drivers


def random_samples(targ_driv, driv_list, K=200):
    """
    Produces random samples of driverIDs and tripIDs i two separate lists


    :param targ_driv: str, driverID we want to make false trips for
    :param driv_list: list, list of all drivers, produced by get_drivers()
    :param K: number of trips we want to make for targ_driv
    :return: tuple of lists, first list is random driverIDs, second list is list of tripIDs, both are strings
    """
    try:
        #removes the target driver from list of drivers to sample from
        driv_list.remove(targ_driv)
        drivers = np.random.choice(driv_list, K, True)
        trips = np.random.choice(np.arange(1,K+1).astype(str), K, True)

        return (drivers, trips)
    except Exception as e:
        print e


def sample_data(path, driverIDs, tripIDs, sc):
    """
    Reads directory of files and returns RDD of observations from trips in the sample
    (driverID, tripID combo)
    NOTE: this function is VERY SLOW, it is what slows the entire workflow down

    :param path: string, path to directory containing driver.csv's
    :param driverIDs: list, list of randomly sampled driverIDs as strings, produced by random_sample()
    :param tripIDs: list, list of randomly sampled tripIDs as strings, produced by random_samples()
        NOTE: the above two zip into a list of (driverID, tripID) tuples, with each tuple being a single item in the
        sample
    :return: RDD, contains only observations from the sample
    """
    try:
        combos = zip(driverIDs, tripIDs)
        samplefiles = [path + '/' + 'driver_' + i + '.csv' for i in driverIDs]

        #### NOTE: this set() action is a hack for small num. files
        samplefiles = ','.join(set(samplefiles))
        #### NOTE: with large num. files, might need to set num. partitions
        RDD = sc.textFile(samplefiles)
        RDDsplit = RDD.map(lambda x: x.split(','))
        RDDsamples = RDDsplit.filter(lambda x: (x[2],x[3]) in combos)
        RDDsamples.cache()
        return RDDsamples
    except Exception as e:
        print e


def ID_Data(targ_driver, RDD, sc, K = 200):
    """
    Takes RDD of samples and assigns new driverID and tripID to observations in a new RDD


    :@param targ_driver: string, target driver we used to generate samples
    :@param RDD: RDD, trip data RDD produced by sample_data(), format will be original form (x,y,driverID,tripID,step)
    :@param K: int, number of trips we sampled
    :@return: RDD, in original format, but with driverID and tripID changed to look like new observations of the target
    driver
    """
    try:
        newID1 = [targ_driver] * K
        newID2 = np.arange(200, 201+K).astype(str)
        newID = zip(newID1, newID2)
        oldID = RDD.map(lambda x: (x[2],x[3])).distinct().collect()
        glossary = sc.parallelize(zip(oldID, newID))
        newRDD = RDD.map(lambda x: ((x[2],x[3]), ([x[0],x[1],x[4]]))).join(glossary)
        newID_RDD = newRDD.map(lambda x: (x[1][0][0], x[1][0][1], x[1][1][0], x[1][1][1], x[1][0][2]))
        return newID_RDD
    except Exception as e:
        print e


def processRDD(RDD, label):
    """
    Takes RDD in original form and converts it into key-value tuple with values being x,y,step,label


    :param RDD: RDD in original format (x,y,driverID,tripID,step)
    :param label: category of observation, 1 for positive, 0 for negative
    # note, not sure if it needs to be int or float
    :return: RDD, RDD returned in new key/value format: (driverID, tripID), (x, y, step, label)
    # note, x, y, step, and label will be floats
    """
    try:
        newRDD = RDD.map(lambda x: ((x[2], x[3]), (float(x[0]), float(x[1]), float(x[4]), label)))
        return newRDD
    except Exception as e:
        print e


def labelRDDs(targ_driv, path, sc, k=200):
    """
    Takes a driver to target, path to directory of driver.csv's, and returns an RDD labeled with
    (driverID, tripID),(x,y,step,label), where a label 1 is from an actual trip, and label 0 is from
    a trip randomly sampled from other drivers
    :param targ_driv: string, driver we want to create positive and negative labeled data for
    :param path: string, path to directory where driver.csvs are stored
    :param k: int, number of negative (manufactured) trips to sample
    :param sc: Spark Context
    :return: RDD with key, value tuple where key is (driverID, tripID) and value is (x,y,step,label)
    """
    try:
        full_path = path + '/' + 'driver_' + targ_driv + '.csv'
        #print full_path
        target = sc.textFile(path + '/' + 'driver_' + targ_driv + '.csv') #load target driver's data
        target2 = target.map(lambda x: x.split(',')) #convert from string to list of strings
        positives = processRDD(target2, 1.0) #label target driver's RDD
        driv_lis = get_drivers(path) #get python list of all possible drivers to sample from
        #print driv_lis

        #generate random samples of drivers and tripIDs
        sampdriv, samptrip = random_samples(targ_driv, driv_lis, k)
        #generate RDD of random samples
        samples = sample_data(path, sampdriv, samptrip, sc)
        #print "GETS HERE"
        samplesRDD = ID_Data(targ_driv, samples, sc, k) #relabel samples to look like target driver's trips
        #print "GETS HERE TOO"
        negatives = processRDD(samplesRDD, 0.0) #label samples
        finalRDD = positives.union(negatives).cache() #join target driver and samples together
        return finalRDD
    except Exception as e:
        print e
