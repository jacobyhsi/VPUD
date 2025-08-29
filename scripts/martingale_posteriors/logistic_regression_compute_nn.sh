#!/bin/bash

# Values for feature_interaction: "linear", "quadratic", "cubic"
# Values for model_name: "qwen7b", "qwen14b", "llama8b"

for model_name in "qwen14b" "qwen7b" "llama8b"; do
    python run_martingale_posterior_classification.py \
        --dataset_name logistic_regression_3     \
        --save_dir ${model_name}/permutation_sampling \
        --reuse_llm_samples True \
        --feature_interaction nn \
        --hidden_dim 20 \
        --num_hidden 2 \
        --save_name nn_logistic
done
