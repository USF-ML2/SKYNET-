#!/bin/bash

while read line; do    
    spark-submit --master yarn gtb_modeling_script.py line &
done < sample_drivers.csv
