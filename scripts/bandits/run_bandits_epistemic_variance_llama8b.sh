#!/bin/bash

for ((i = $2; i <= $3; i++))
do
    python run_bandit_classification.py --num_random_trials 5 --num_trials 200 --bandit_exploration_rate $1 --save_directory "llama8b/exploration_ev_${1}" --bandit_seed $i --numpy_seed $i --run_name "exp_${1}_seed_$i" --model_name meta-llama/Meta-Llama-3-8B --model_port 8000
done