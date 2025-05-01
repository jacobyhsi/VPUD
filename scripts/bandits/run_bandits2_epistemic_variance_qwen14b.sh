#!/bin/bash

for ((i = $2; i <= $3; i++))
do
    python run_bandit_classification.py --num_random_trials 5 --num_trials 200 --bandit_midpoint $4 --bandit_exploration_rate $1 --save_directory "qwen14b/experiment_2/exploration_mid_${4}_ev_${1}" --bandit_seed $i --numpy_seed $i --run_name "exp_${1}_ev_seed_$i" --model_port 6000
done