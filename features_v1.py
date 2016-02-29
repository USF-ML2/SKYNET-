def vectorRDD(RDD):
	"""
	:param RDD: RDD, created from labelRDDs
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

	vectorRDD = newRDD.map(lambda x: (x[0], ([x[1][0]], [x[1][1]], 
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


def step_level_features(polarRDD):
	"""
	:param RDD: RDD, created from get_polars
	:return: RDD with speed and acceleration at each stage of the trip
	"""

	# First, the RDD created from get_polars is mapped to the key, value pair
	# ((driver_id, trip_id), ([x coordinates], [y coordinates],
	# [r coordinates], [theta coordinates], [v coordinates], [step numbers], 
	# label)). v is the current speed (m/s), which is computed from the 
    # euclidean, distance between the current point and the previous point, 
    # i.e. the two innermost maps square the individual coordinate differences.
    # These are then added and square rooted for speed. The second step maps
    # this key, value pair into the key, value pair ((driver_id, trip_id), 
    # ([x coordinates], [y coordinates], [r coordinates], [theta coordinates], 
    # [v coordinates], [a coordinates], label)). a is the current acceleration,
    # which is computed as the difference between the current speed and the 
    # previous speed

	step_lv = polarRDD.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2], 
		x[1][3], map(lambda x: (x[0] + x[1]) ** 0.5, 
			zip(map(lambda x: (x[0] - x[1]) ** 2, 
				zip(x[1][0], [0.0] + x[1][0][:-1])), 
			map(lambda x: (x[0] - x[1]) ** 2, 
				zip(x[1][1], [0.0] + x[1][1][:-1])))), x[1][4], x[1][5])))\
	.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], 
		map(lambda x: x[0] - x[1], 
			zip(x[1][4], [0.0] + x[1][4][:-1])), x[1][5], x[1][6])))

	return step_lv


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

    trip_lv = RDD.map(lambda x: (x[0], (min(x[1][4]), max(x[1][4]), 
    	min(x[1][5]), max(x[1][5]), len(x[1][0]), sum(x[1][4]), 
    	np.mean(x[1][4]), np.std(x[1][4]), np.mean(x[1][5]), np.std(x[1][5]), 
    	sum([elem < 0.5 for elem in x[1][4]]), x[1][7])))

    return trip_lv

