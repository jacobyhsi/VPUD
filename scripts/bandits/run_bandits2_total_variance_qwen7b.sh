#!/bin/bash

for ((i = $2; i <= $3; i++))
do
    python run_bandit_classification.py --num_random_trials 5 --num_trials 200 --bandit_midpoint $4 --bandit_exploration_rate $1 --save_directory "qwen7b/experiment_2/exploration_mid_${4}_tv_${1}" --bandit_seed $i --numpy_seed $i --run_name "exp_${1}_tv_seed_$i" --uncertainty_type total_variance --model_name Qwen/Qwen2.5-7B --model_port 9000
done