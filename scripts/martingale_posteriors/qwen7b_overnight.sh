
# python run_martingale_posterior_classification.py \
#     --dataset_name logistic_regression_3 \
#     --model_name Qwen/Qwen2.5-7B \
#     --model_port 9090 \
#     --model_ip 155.198.192.50 \
#     --save_dir qwen7b/permutation_sampling \
#     --save_llm_samples True \
#     --permutation_sampling True \
#     --max_num_samples_per_call 2 \
#     --max_llm_sample_size 50 \
#     --max_tokens_per_call 50 \
#     --sample_only True

# python run_martingale_posterior_classification.py \
#     --dataset_name moons \
#     --model_name Qwen/Qwen2.5-7B \
#     --model_port 9090 \
#     --model_ip 155.198.192.50 \
#     --save_dir qwen7b/permutation_sampling_2 \
#     --D_size 30 \
#     --save_llm_samples True \
#     --permutation_sampling True \
#     --max_num_samples_per_call 2 \
#     --max_llm_sample_size 50 \
#     --max_tokens_per_call 50 \
#     --sample_only True

python run_martingale_posterior_classification.py \
    --dataset_name linear_noise_2 \
    --model_name Qwen/Qwen2.5-7B \
    --model_port 9090 \
    --model_ip 155.198.192.50 \
    --D_size 30 \
    --save_dir qwen7b/permutation_sampling_2 \
    --save_llm_samples True \
    --permutation_sampling True \
    --max_num_samples_per_call 2 \
    --max_llm_sample_size 50 \
    --max_tokens_per_call 50 \
    --sample_only True

# python run_martingale_posterior_classification.py \
#     --dataset_name logistic_regression_3 \
#     --model_name Qwen/Qwen2.5-7B \
#     --model_port 9090 \
#     --model_ip 155.198.192.50 \
#     --save_dir qwen7b/one_pass_sampling \
#     --save_llm_samples True \
#     --permutation_sampling False \
#     --max_llm_sample_size 50 \
#     --max_tokens_per_call 750 \
#     --sample_only True

# python run_martingale_posterior_classification.py \
#     --dataset_name moons \
#     --model_name Qwen/Qwen2.5-7B \
#     --model_port 9090 \
#     --model_ip 155.198.192.50 \
#     --save_dir qwen7b/one_pass_sampling \
#     --save_llm_samples True \
#     --permutation_sampling False \
#     --max_llm_sample_size 50 \
#     --max_tokens_per_call 750 \
#     --sample_only True

# python run_martingale_posterior_classification.py \
#     --dataset_name linear_noise_2 \
#     --model_name Qwen/Qwen2.5-7B \
#     --model_port 9090 \
#     --model_ip 155.198.192.50 \
#     --save_dir qwen7b/one_pass_sampling \
#     --save_llm_samples True \
#     --permutation_sampling False \
#     --max_llm_sample_size 50 \
#     --max_tokens_per_call 750 \
#     --sample_only True