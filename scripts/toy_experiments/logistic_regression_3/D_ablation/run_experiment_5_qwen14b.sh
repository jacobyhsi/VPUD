#!/bin/bash

for i in {5..75..10}; do
    python run_toy_classification.py \
        --x_range "{'x1': [-15, 20, 5]}" \
        --shots $i \
        --num_modified_z 5 \
        --num_random_z 5 \
        --save_directory qwen14b/experiment_5/seed_$1 \
        --run_name ${i}_shot_5_z_10_perm \
        --num_permutations 10 \
        --perturbation_std 0.1 \
        --icl_sample_seed $1
done