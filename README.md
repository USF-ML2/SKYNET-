# Driver Signatures from Telematic data

Insurance companies now record Telematic data (X,Y coordinates vs Time) for drivers to understand driver behaviour and calculate the premium that the driver must pay for insurance. 

We sourced the data from a Kaggle competition that Axa Telematics had hosted. 

## Motivation 

We were learning Spark and we wanted to take an existing problem and adapt its solution in Spark. We chose to use this competition because the dataset seemed very interesting. This was a dead Kaggle competition and so we actually had access to intelligent solutions, this was important because we wanted to understand best practices and adapt the solution to Spark. 

We used PySpark for the project

## Data 

We had data for about 3500 drivers. For each driver there were 200 files containing telematic data for 200 trips with X,Y coordinates in every line. The dataset was such that a percentage of these 200 trips (< 50%) for each driver actually belonged to someone else. 

So this was an unsupervised learning problem. Since the competition was closed and there was no leaderboard to test our solutions against. We had to construct a supervised learning problem out of this dataset. This was a reasonable approach because most participants had done something like this to test their models while the competition was going on. They reported that the accuracy on the original unsupervised learning problem was very close to a constructed supervised learning problem.

## Constructing a supervised learning problem

This is what we did to construct a supervised learning problem
- Model one driver at a time (since the problem is to find out which trips do not belong to a driver)
- Take 100 trips from a driver and 100 trips randomly from all other drivers to construct a training set of 200 trips
- Train a classification algorithm on this data
- Run the model on the Test set (the remaining 100 trips) and those trips which come up as false negatives are then labelled as trips which do not actually belong to the driver
- Do this for the other 100 trips in the Training set by re-training the model


## Challenges

- 3500 different models (One for each driver)
- Feature generation in spark from the original dataset 
- Spark configurations for handling the volume of data


## Results

- We ended up using Generalized Boosting Trees as our classification algorithm


