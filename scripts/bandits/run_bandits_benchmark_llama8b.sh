#!/bin/bash

for ((i = $1; i <= $2; i++))
do
    python run_bandit_classification_benchmark.py --num_trials 200 --bandit_midpoint $3 --save_directory "llama8b/experiment_1/mid_${3}" --bandit_seed $i --run_name "seed_$i" --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_port 5000
done