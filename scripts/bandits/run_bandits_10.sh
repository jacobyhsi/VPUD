#!/bin/bash

for i in {0..4}
do
    python run_bandit_classification.py --num_random_trials 5 --num_trials 200 --bandit_exploration_rate 10 --save_directory exploration_10 --bandit_seed $i --numpy_seed $i --run_name "exp_10_seed_$i"
done