#!/bin/bash

# Values for feature_interaction: "linear", "quadratic", "cubic"
# Values for model_name: "qwen7b", "qwen14b", "llama8b"

for model_name in "qwen7b" "qwen14b"; do
    python run_deep_martingale_posterior_classification.py \
        --dataset_name logistic_regression_3 \
        --save_dir ${model_name}/permutation_sampling_2/deep_martingale \
        --reuse_llm_samples True \
        --empirical_results_dir results/logistic_regression_3/${model_name}/experiment_2
done