__author__ = "Harry"

import os
import os.path
import re
import csv
from glob import glob


full_path = "/Users/Harry/Documents/Harry/Education/College/Analytics/Spring_Term_Module1/MSAN630_-_Advanced_Machine_Learning/Assignments/Project/Methods/"


def make_directories(path):
	# Checks if the path to the directory exists
	if not os.path.exists(path):
		# If it does not, create direstory
		os.makedirs(path)
	return None


def file_ids(directory):
	file_ids = []
	for dirpath, dirnames, filenames in os.walk(directory):
		for filename in [f for f in filenames if f.endswith(".csv")]:
			# Appends (file_path, driver_id, trip_id) to file_ids
			file_ids.append((os.path.join(dirpath, filename),
							 re.search(r"[0-9]+$", dirpath).group(),
							 re.sub(r"\.csv", "", filename)))
	return file_ids


def parse_file(file_id, newpath):
	with open(file_id[0], "rb") as f:
		# reads all file content except for header
		content = f.read().splitlines(True)[1:]

	# outputs (x, y, driver_id, trip_id, step) for each line
	content = [re.sub("\n", "", content[i]) + "," + file_id[1] + "," +
			   file_id[2] + "," + str(i) + "\n"
			   for i in range(len(content))]

    # writes to new file
	with open(newpath + "test_drivers/%s/%s.csv" % (file_id[1], file_id[2]), "wb") as f:
		f.writelines(content)

	return None


def concatentate_files(path, i):
	with open(path + "/driver_" + str(i) + ".csv", "a") as f:
		for csvfile in glob(path + str(i) + "/*.csv"):
			for line in open(csvfile, "rb"):
				f.write(line)
	return None


if __name__ == "__main__":

	# File path, driver id, and trip id for each csv file
	# X = file_ids(full_path + "drivers")

	# Set of unique driver ids
	# new_dirs = set([i[1] for i in X])

	# Makes new directories if the directory does not exist
	# for i in new_dirs:
		# make_directories(full_path + "test_drivers/" + i)

	X = file_ids(full_path + "test_drivers")
	new_dirs = set([i[1] for i in X])

	print "File Paths Loaded"

	for i in new_dirs:
		concatentate_files(full_path + "test_drivers/", i)
		print i

	# for entry in X:
	# 	parse_file(entry, full_path)
	# 	print entry[1], entry[2]
