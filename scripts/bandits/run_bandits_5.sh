#!/bin/bash

for i in {0..4}
do
    python run_bandit_classification.py --num_random_trials 5 --num_trials 200 --bandit_exploration_rate 5 --save_directory exploration_5 --bandit_seed $i --numpy_seed $i --run_name "exp_5_seed_$i"
done