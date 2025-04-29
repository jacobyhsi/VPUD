#!/bin/bash

python run_toy_classification.py --x_range "{'x1': [-15, 15, 0.2]}" --shots 15 --num_modified_z 15 --num_random_z 5 --save_directory qwen14b/experiment_1 --run_name 15_shot_15_z_5_random_10_perm --num_permutations 10
python run_toy_classification.py --x_range "{'x1': [-15, 15, 0.2]}" --shots 15 --num_modified_z 15 --num_random_z 15 --save_directory qwen14b/experiment_2 --run_name 15_shot_15_z_15_random_10_perm --num_permutations 10 --perturbation_std 0.1
python run_toy_classification.py --x_range "{'x1': [-15, 15, 0.2]}" --shots 15 --num_modified_z 1 --num_random_z 1 --save_directory qwen14b/experiment_3 --run_name 15_shot_1_z_10_perm --num_permutations 10 --perturbation_std 0
python run_toy_classification.py --x_range "{'x1': [-15, 15, 0.2]}" --shots 15 --num_modified_z 15 --num_random_z 15 --save_directory qwen14b/experiment_4 --run_name 15_shot_15_z_15_random_10_perm_random --num_permutations 10 --perturb_about_x 0