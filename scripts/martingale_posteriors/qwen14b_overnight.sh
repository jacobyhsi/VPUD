#!/bin/bash

signal_file="/vol/bitbucket/ij23/projects/vpud/start.txt"

# Wait for script_y to start
while [ ! -f "$signal_file" ]; do
    echo "Waiting for script to start..."
    sleep 10
done

# Read message from signal file
message=$(cat "$signal_file")
echo "Received message: $message"

echo "Waiting for model to start..."

# sleep 3

sleep 3600

# python run_martingale_posterior_classification.py \
#     --dataset_name logistic_regression_3 \
#     --model_name Qwen/Qwen2.5-14B \
#     --model_port 9080 \
#     --model_ip $message \
#     --save_dir qwen14b/permutation_sampling_2 \
#     --D_size 30 \
#     --save_llm_samples True \
#     --permutation_sampling True \
#     --max_num_samples_per_call 2 \
#     --max_llm_sample_size 50 \
#     --max_tokens_per_call 50 \
#     --sample_only True

# python run_martingale_posterior_classification.py \
#     --dataset_name moons \
#     --model_name Qwen/Qwen2.5-14B \
#     --model_port 9080 \
#     --model_ip $message \
#     --save_dir qwen14b/permutation_sampling_2 \
#     --D_size 30 \
#     --save_llm_samples True \
#     --permutation_sampling True \
#     --max_num_samples_per_call 2 \
#     --max_llm_sample_size 50 \
#     --max_tokens_per_call 50 \
#     --sample_only True

python run_martingale_posterior_classification.py \
    --dataset_name linear_noise_2 \
    --model_name Qwen/Qwen2.5-14B \
    --model_port 9080 \
    --model_ip $message \
    --save_dir qwen14b/permutation_sampling_2 \
    --D_size 30 \
    --save_llm_samples True \
    --permutation_sampling True \
    --max_num_samples_per_call 2 \
    --max_llm_sample_size 50 \
    --max_tokens_per_call 50 \
    --sample_only True

# python run_martingale_posterior_classification.py \
#     --dataset_name logistic_regression_3 \
#     --model_name Qwen/Qwen2.5-14B \
#     --model_port 9080 \
#     --model_ip $message \
#     --save_dir qwen14b/one_pass_sampling \
#     --save_llm_samples True \
#     --permutation_sampling False \
#     --max_llm_sample_size 50 \
#     --max_tokens_per_call 750 \
#     --sample_only True

# python run_martingale_posterior_classification.py \
#     --dataset_name moons \
#     --model_name Qwen/Qwen2.5-14B \
#     --model_port 9080 \
#     --model_ip $message \
#     --save_dir qwen14b/one_pass_sampling \
#     --save_llm_samples True \
#     --permutation_sampling False \
#     --max_llm_sample_size 50 \
#     --max_tokens_per_call 750 \
#     --sample_only True

# python run_martingale_posterior_classification.py \
#     --dataset_name linear_noise_2 \
#     --model_name Qwen/Qwen2.5-14B \
#     --model_port 9080 \
#     --model_ip $message \
#     --save_dir qwen14b/one_pass_sampling \
#     --save_llm_samples True \
#     --permutation_sampling False \
#     --max_llm_sample_size 50 \
#     --max_tokens_per_call 750 \
#     --sample_only True