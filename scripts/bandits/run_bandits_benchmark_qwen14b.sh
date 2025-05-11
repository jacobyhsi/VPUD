#!/bin/bash

for ((i = $1; i <= $2; i++))
do
    python run_bandit_classification_benchmark.py --num_trials 200 --bandit_midpoint $3 --save_directory "qwen14b/experiment_1/mid_${3}" --bandit_seed $i --run_name "seed_$i"
done