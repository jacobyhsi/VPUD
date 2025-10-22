#!/bin/bash

python run_toy_classification.py \
    --x_range "{'x1': [-2, 2.5, 0.5]}" --num_permutations 10 \
    --shots 20 --num_modified_z 15 --num_random_z 15 --perturbation_std 0.1 \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_ip 155.198.192.50 --model_port 6000 \
    --run_name 20_shot_15_z_15_random \
    --save_directory llama8b-instruct/prompt_testing/experiment_1.1/prompt_0

python run_toy_classification.py \
    --x_range "{'x1': [-2, 2.5, 0.5]}" --num_permutations 10 \
    --shots 20 --num_modified_z 15 --num_random_z 15 --perturbation_std 0.1 \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_ip 155.198.192.50 --model_port 6000 \
    --run_name 20_shot_15_z_15_random \
    --save_directory llama8b-instruct/prompt_testing/experiment_1.1/prompt_1 \
    --custom_prompt_text "This is a logistic regression dataset. Continue generating the dataset. ONLY generate lebel for next point in tags <output> </output>.\n{icl}\n {note} "

python run_toy_classification.py \
    --x_range "{'x1': [-2, 2.5, 0.5]}" --num_permutations 10 \
    --shots 20 --num_modified_z 15 --num_random_z 15 --perturbation_std 0.1 \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_ip 155.198.192.50 --model_port 6000 \
    --run_name 20_shot_15_z_15_random \
    --save_directory llama8b-instruct/prompt_testing/experiment_1.1/prompt_2 \
    --custom_prompt_text "This is a logistic regression dataset. Continue generating the dataset. ONLY generate lebel for next point in tags <output> </output>.\n{icl}\n\n Next point:\n {note} "

python run_toy_classification.py \
    --x_range "{'x1': [-2, 2.5, 0.5]}" --num_permutations 10 \
    --shots 20 --num_modified_z 15 --num_random_z 15 --perturbation_std 0.1 \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_ip 155.198.192.50 --model_port 6000 \
    --run_name 20_shot_15_z_15_random \
    --save_directory llama8b-instruct/prompt_testing/experiment_1.1/prompt_3 \
    --custom_prompt_text "<|system|>\n This is a logistic regression dataset. Continue generating the dataset. ONLY generate lebel for next point in tags <output> </output>.\n{icl}\n<|user|>\n Next point: {note} \n<|assistant|>\n"

python run_toy_classification.py \
    --x_range "{'x1': [-2, 2.5, 0.5]}" --num_permutations 10 \
    --shots 20 --num_modified_z 15 --num_random_z 15 --perturbation_std 0.1 \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_ip 155.198.192.50 --model_port 6000 \
    --run_name 20_shot_15_z_15_random \
    --save_directory llama8b-instruct/prompt_testing/experiment_1.1/prompt_4 \
    --custom_prompt_text "Your task is to predict the label of a logistic regression task that performs classification between 0 and 1. Please carefully review the following in-context learning examples and their labels inside <output>LABEL</output> tags:\n\n{icl}\n\n Now, predict the label for this logistic regression example:\n\n {note}\n\nIMPORTANT: Output ONLY the label inside <output></output> tags. Do not add any explanation, text, or formatting. Your response must strictly follow this format:\n\n <output>LABEL</output>"
