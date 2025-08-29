#!/bin/bash

# Values for feature_interaction: "linear", "quadratic", "cubic"
# Values for model_name: "qwen7b", "qwen14b", "llama8b"

for feature_interaction in "linear" "quadratic" "cubic"; do
    for model_name in "qwen14b" "qwen7b" "llama8b"; do
        python run_martingale_posterior_classification.py \
            --dataset_name linear_noise_2 \
            --save_dir ${model_name}/permutation_sampling \
            --reuse_llm_samples True \
            --feature_interaction ${feature_interaction} \
            --save_name ${feature_interaction}_gaussian \
            --likelihood_type gaussian
    done
done