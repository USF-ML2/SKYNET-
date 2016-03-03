# this will be for outputs of step_level_features(). 
# specifically, for the velocity and acceleration fields (index 5 and 6
# respectivly)


# NOTE: the following are meant as functions to feed into the lambda 
# function in a bigger wrapper, i.e. 

# RDD = step_level_features(RDD)
# RDD.map(lambda x: below_function(x[6]))


# this divides up acceleration list into a list of lists, where each inner
# list is sequential accelerations in the same direction.
    # i.e. if you accelerate for 3 steps, they get thrown into same bin
    # if you deccelerate for 10 steps, that gets thrown into same bin
def bin_accel_events(accel_list):
    bins = []
    for i in range(len(accel_list)):
        if i == 0:
            bins.append([accel_list[i]])
        else:
            if cmp(accel_list[i],0) == cmp(accel_list[i-1],0):
                bins[-1] += [accel_list[i]]
            else: 
                bins.append([accel_list[i]])
    return bins



# same as above but with indicies attached so we can look up the speed
# driver ends up at at end of acceleration event
def bin_accel_events_with_index(accel_list):
    bins = []
    temp = []
    for i in range(len(accel_list)):
        if i == 0:
            temp.append(accel_list[i])
        else:
            if cmp(accel_list[i],0) == cmp(accel_list[i-1],0):
                temp.append(accel_list[i])
            else: 
                bins.append((temp, i-1))
                temp = [accel_list[i]]
        if i == len(accel_list) - 1:
            bins.append((temp, i))
    return bins


# takes input of above and returns just events that are positive
def filter_accel(accel_bins_with_index):
    filtered = [i for i in accel_bins_with_index if i[0][0] > 0]
    return filtered

# takes input of above and returns just events that are negative
def filter_deccel(accel_bins_with_index):
    filtered = [i for i in accel_bins_with_index if i[0][0] < 0]
    return filtered

# gets mean of accel/decel EVENTS 
# takes the output of filter_accel, filter_deccel, or bin_accel_events_with_index
# as input
def mean_accel_events(bined_accels_with_index):
    events = [i[0] for i in bined_accels_with_index]
    means = [sum(i)/float(len(i)) for i in events]
    overall_mean = sum(means)/float(len(means))
    return overall_mean


# ideas:

# 1: acceleration rate for speeds of: 15 mph, 30 mph, etc.
    # i.e. in each acceleration event, what speed do they end up at? 
    # what's the rate at which they reached that speed? 
# 2: resting speed at different plateaus: 0-15 mph, 15-30 mph, etc. 
    # i.e. for each bin, what is their average speed? 

