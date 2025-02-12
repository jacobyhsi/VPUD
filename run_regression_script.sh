#!/bin/bash

# for loop
for i in {3..6..1}
do
    echo "Running for $i shots"
    python run_regression_fewshot.py --shots $i --num_modified_z 10  --run_name "results_${i}_shot" --num_x_values 10
done

for i in {8..10..2}
do
    echo "Running for $i shots"
    python run_regression_fewshot.py --shots $i --num_modified_z 10  --run_name "results_${i}_shot" --num_x_values 10
done

for i in {15..25..5}
do
    echo "Running for $i shots"
    python run_regression_fewshot.py --shots $i --num_modified_z 10  --run_name "results_${i}_shot" --num_x_values 10
done