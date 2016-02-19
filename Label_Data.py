# !/usr/bin/env python

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

__author__ = "Su-Young Hong AKA Da Masta Killa AKA Synth Pop Rocks of Locks AKA Intergalactic Chilympian"
__status__ = "Prototype"

# read directory of files and return a list of all driverIDs from csv's insid directory
def get_drivers(dirpath):
	"""
	:param dirpath: string, path to directory containing driver csv's
	:return: list, contains all driverIDs as strings
	"""
	try:
		allfiles = os.listdir(dirpath)
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
# NOTE: this function is VERY SLOW, it is what slows the entire workflow down
def sample_data(path, driverIDs, tripIDs):
	"""
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
		samplefiles = ','.join(set(samplefiles))  #### NOTE: this set() action is a hack for small num. files
		RDD = sc.textFile(samplefiles)   #### NOTE: with large num. files, might need to set num. partitions
		RDDsplit = RDD.map(lambda x: x.split(','))
		RDDsamples = RDDsplit.filter(lambda x: (x[2],x[3]) in combos)
		RDDsamples.cache()
		return RDDsamples
	except Exception as e:
		print e

# takes RDD of samples and assigns new driverID and tripID to observations in a new RDD
def ID_Data(targ_driver, RDD, K = 200):
	"""
	:param targ_driver: string, target driver we used to generate samples
	:param RDD: RDD, trip data RDD produced by sample_data(), format will be original form (x,y,driverID,tripID,step)
	:param K: int, number of trips we sampled
	:return: RDD, in original format, but with driverID and tripID changed to look like new observations of the target
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


# takes RDD in original form and converts it into key-value tuple with values being x,y,step,label
def processRDD(RDD, label):
	"""
	:param RDD: RDD in original format (x,y,driverID,tripID,step)
	:param label: category of observation, 1 for positive, 0 for negative
	# note, not sure if it needs to be int or float
	:return: RDD, RDD returned in new key/value format: (driverID, tripID), (x, y, step, label)
	# note, x, y, step, and label will be floats
	"""
	try:
		newRDD = RDD.map(lambda x: ((x[2],x[3]),(float(x[0]),float(x[1]),float(x[4]),label)))
		return newRDD
	except Exception as e:
		print e

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
	try:
		target = sc.textFile(path + '/' + 'driver_' + targ_driv + '.csv') #load target driver's data
		target2 = target.map(lambda x: x.split(',')) #convert from string to list of strings
		positives = processRDD(target2, 1.0) #label target driver's RDD
		driv_lis = get_drivers(path) #get python list of all possible drivers to sample from
		sampdriv, samptrip = random_samples(targ_driv, driv_lis, K) #generate random samples of drivers and tripIDs
		samples = sample_data(path, sampdriv, samptrip) #generate RDD of random samples
		samplesRDD = ID_Data(targ_driv, samples, K) #relabel samples to look like target driver's trips
		negatives = processRDD(samplesRDD, 0.0) #label samples
		finalRDD = positives.union(negatives).cache() #join target driver and samples together
		return finalRDD
	except Exception as e:
		print e

def vectorRDD(RDD):
	vectorRDD = newRDD.map(lambda x: (x[0], ([x[1][0]], [x[1][1]], x[1][2], (x[1][3], 1))))\
					  .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))\
					  .map(lambda x: (x[0], (x[1][0], x[1][1], 1)) if int(x[0][1])\
						   < 201 else (x[0], (x[1][0], x[1][1], 0)))
	return vectorRDD

# computes previous points of each x, y coordinate
# computes distance in each x, y direction squared from previous point
# computes euclidean distance from previous point (also speed as in m/s)
# computes previous speed
# computes speed difference (acceleration)
# IF REDUCING A LIST OF TUPLES, MUST USE BOTH TUPLE ELEMENTS!!!
vectorRDD.map(lambda x: (x[0], (zip(x[1][0], [0.0] + x[1][0][:-1]), 
								zip(x[1][1], [0.0] + x[1][1][:-1]))))\
		 .map(lambda x: (x[0], zip(map(lambda x: (x[0] - x[1]) ** 2, x[1][0]), 
								   map(lambda x: (x[0] - x[1]) ** 2, x[1][1]))))\
		 .map(lambda x: (x[0], map(lambda x: (x[0] + x[1]) ** 0.5, x[1])))\
		 .map(lambda x: (x[0], zip(x[1], [0.0] + x[1][:-1])))\
		 .map(lambda x: (x[0], map(lambda x: ([x[0]], [x[0] - x[1]]), x[1])))\
		 .map(lambda x: (x[0], reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), x[1])))\
		 .map(lambda x: (x[0], (min(x[1][0]), max(x[1][0]), min(x[1][1]), max(x[1][1]))))


def get_polars(RDD):
	polars = RDD.map(lambda x: (x[0], (x[1][0], x[1][1], 
								map(lambda x: (x[0] ** 2 + x[1] ** 2) ** 0.5, zip(x[1][0], x[1][1])),
								map(lambda x: math.atan2(x[1], x[0]), zip(x[1][0], x[1][1])))))
	return polars


def step_level_features(polarRDD):
	newRDD = polarRDD.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2], x[1][3], 
										  map(lambda x: (x[0] + x[1]) ** 0.5, 
											  zip(map(lambda x: (x[0] - x[1]) ** 2, 
													  zip(x[1][0], [0.0] + x[1][0][:-1])), 
										  map(lambda x: (x[0] - x[1]) ** 2, 
											  zip(x[1][1], [0.0] + x[1][1][:-1])))))))\
					 .map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2], x[1][3], x[1][4],
											map(lambda x: x[0] - x[1], 
												zip(x[1][4], [0.0] + x[1][4][:-1])))))
	return newRDD


"""
path = '/Users/coolguy/Desktop/ML2Project/Data/processed_data/sample_drivers'
driver = '1'
newRDD = labelRDDs(driver, path)
"""
