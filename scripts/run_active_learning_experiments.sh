#!/bin/bash

for i in {5..20..5}
do
    echo "Running for $i shots"
    echo "Dataset specified"
    python run_regression_fewshot_active_learning.py --shots $i \
                                                     --num_modified_z 10 \
                                                     --run_name "${i}_shot" \
                                                     --num_x_values 10 \
                                                     --save_directory "logistic_regression_3/experiment_2/active_learning/dataset_specified" \
                                                     --specify_dataset_type 1

    echo "Dataset not specified"
    python run_regression_fewshot_active_learning.py --shots $i \
                                                     --num_modified_z 10 \
                                                     --run_name "${i}_shot" \
                                                     --num_x_values 10 \
                                                     --save_directory "logistic_regression_3/experiment_2/active_learning/dataset_not_specified" \
                                                     --specify_dataset_type 0
done

for i in {5..20..5}
do
    echo "Running for $i shots"
    echo "Dataset specified"
    python run_regression_fewshot.py --shots $i \
                                                     --num_modified_z 10 \
                                                     --run_name "${i}_shot" \
                                                     --num_x_values 10 \
                                                     --save_directory "logistic_regression_3/experiment_2/random_z/dataset_specified" \
                                                     --specify_dataset_type 1

    echo "Dataset not specified"
    python run_regression_fewshot.py --shots $i \
                                                     --num_modified_z 10 \
                                                     --run_name "${i}_shot" \
                                                     --num_x_values 10 \
                                                     --save_directory "logistic_regression_3/experiment_2/random_z/dataset_not_specified" \
                                                     --specify_dataset_type 0
done