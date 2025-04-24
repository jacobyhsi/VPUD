#!/bin/bash

for ((i = $2; i <= $3; i++))
do
    python run_bandit_classification.py --num_random_trials 5 --num_trials 200 --bandit_exploration_rate $1 --save_directory "qwen7b/exploration_tv_$1" --bandit_seed $i --numpy_seed $i --run_name "exp_${1}_tv_seed_$i" --uncertainty_type total_variance --model_name Qwen/Qwen2.5-7B  --model_name Qwen/Qwen2.5-7B --model_port 9000
done