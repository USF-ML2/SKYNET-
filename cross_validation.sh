#!/bin/bash

while read line; do    
    spark-submit --master yarn --executor-cores 1 --executor-memory 1g gtb_modeling_script.py $line > $line.logs &
done < sample_drivers/sample_driversaa
