import json
import requests
import re
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from openai import OpenAI
from prompt import *

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from torch.nn.functional import softmax
from scipy.optimize import minimize

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument("--seed", default=123)
parser.add_argument("--seed_num", default="5")
parser.add_argument("--data", default="income")
parser.add_argument("--feature", default="Education")
parser.add_argument("--shots", default=3)
parser.add_argument("--sets", default=10)
parser.add_argument("--llm", default="gpt-4o-mini")
args = parser.parse_args()
seed = int(args.seed)
np.random.seed(seed)
shots = int(args.shots)
sets = int(args.sets)
llm = args.llm

pd.set_option('display.max_columns', None)
login(token = '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')
client = OpenAI(project="proj_cv1WcKDUx5NxbJzxzGwjHhJZ")

def get_response(client, llm, system_prompt, user_prompt, label_keys):
    messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
    if llm == "gpt-4o-mini":
        # Simulate the response from `client.some_method_to_get_text`
        completion = client.chat.completions.create(
            model=llm,
            messages=messages,
            logprobs=True,
            top_logprobs=10,
            temperature=1.0
        )
        output = completion.choices[0].message.content
        top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        logprobs = {
            logprob.token: np.round(np.exp(logprob.logprob)*100,2)
            for logprob in top_logprobs
            if logprob.token in label_keys
        }
        # print(f"Filtered Logprobs for label_keys {label_keys}: {logprobs}")
    else:
        model, tokenizer = client['model'], client['tokenizer']
        tokenized_message = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        response_token_ids = model.generate(tokenized_message['input_ids'].cuda(),attention_mask=tokenized_message['attention_mask'].cuda(),  max_new_tokens=4096, pad_token_id = tokenizer.eos_token_id)
        generated_tokens = response_token_ids[:, len(tokenized_message['input_ids'][0]):]
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    output = output.strip()
        
    return output, logprobs

################################################################################################
############################################ LLM ###############################################
################################################################################################

################################################################################################
########################################## Helper ##############################################
################################################################################################

# Function to parse the 'note' string into a dictionary of features
def parse_note_to_features(note, feature=None):
    features = {}
    for feature_str in note.strip('.').split('. '):
        key_value = feature_str.split(' = ')
        if len(key_value) == 2:
            key, value = key_value
            features[key.strip()] = value.strip()
            
    if feature:
        return features.get(feature.strip(), None)
    
    return features

def parse_features_to_note(row):
    feature_strs = []
    for key, value in row.items():
        if pd.notnull(value):
            feature_strs.append(f'{key.strip()} = {str(value).strip()}')
    note = '. '.join(feature_strs)
    if note:
        note += '.'
    return note

def calculate_entropy(probs):
    # Calculate entropy using all probabilities in the dictionary
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 5)

################################################################################################
########################################## Helper ##############################################
################################################################################################

################################################################################################
##################################### Data Preprocessing #######################################
################################################################################################
data_path = f'TabLLM/datasets_serialized/{args.data}'

f = open(f'{data_path}/info.json')
label_map = json.load(f)
label_name = label_map['label']
labels = label_map['map']
label_keys = list(labels)

data = load_from_disk(data_path)
data = data.to_pandas()
if (args.data == "income" or args.data == "calhousing" or args.data == "bank" or args.data == "blood" or args.data == "jungle" or args.data == "creditg"):
    data['label'] = data['label'].apply(lambda x: 0 if x is False else 1)
    
# data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

# Replace The and is with =, remove extra spaces
data['note'] = (data['note']
                  .str.replace(r'\bThe\b', '', regex=True)
                  .str.replace(r'\bis\b', '=', regex=True)
                  .str.replace(r'\s{2,}', ' ', regex=True)
                  .str.lstrip())

D_rows = data.sample(n=shots, random_state=seed)
D = "\n".join(
    [f"- {row['note']} -> {label_name}: {row['label']}" for _, row in D_rows.iterrows()]
)

################################################################################################
##################################### Data Preprocessing #######################################
################################################################################################

# Platt
calibration_data = data.sample(n=int(len(data)*0.1), random_state=42)
# calibration_data = data.sample(n=3, random_state=42)
data = data.drop(calibration_data.index)
calibration_data = calibration_data.reset_index(drop=True)

logit_pairs = []
label_lst = []
for i, row in calibration_data.iterrows():
    print(f"\nProcessing Temperature Scaling Example {i + 1}")
    
    note = row["note"]
    label = row["label"]
    
    prompt = Prompt()
    response_text, label_logits = get_response(client, llm, prompt.get_system_prompt(), prompt.get_user_prompt(D, label_name, label_keys, labels, note), label_keys)
    if label_logits is not None:
        logit_pairs.append(label_logits)  # [logit_for_0, logit_for_1]
        label_lst.append(label)
        
print("Logit Pairs:", logit_pairs)
print("Labels:", label_lst)

label_lst = [int(l) for l in label_lst]
        
def bce_loss_with_temperature(temperature, logit_pairs, labels):
    total_loss = 0.0
    n = len(labels)
    
    for logits, y in zip(logit_pairs, labels):
        z0, z1 = logits.get(0, float('-inf')), logits.get(1, float('-inf'))

        # Apply temperature scaling
        z0_scaled = z0 / temperature
        z1_scaled = z1 / temperature
        
        # Compute probabilities
        max_val = max(z0_scaled, z1_scaled)  # for numerical stability in exp
        exp_z0 = math.exp(z0_scaled - max_val)
        exp_z1 = math.exp(z1_scaled - max_val)
        
        p_y1 = exp_z1 / (exp_z0 + exp_z1)
        
        # BCE loss for this instance
        eps = 1e-12
        loss = -(y * math.log(p_y1 + eps) + (1 - y) * math.log(1 - p_y1 + eps))
        total_loss += loss
    
    return total_loss / n

# Initial guess for T and bounds
initial_T = 1.0
bounds = [(0.05, 20.0)]  # Temperature should be > 0

res = minimize(
    bce_loss_with_temperature, 
    x0=[initial_T], 
    args=(logit_pairs, label_lst), 
    bounds=bounds, 
    method='L-BFGS-B'
)

optimal_temperature = res.x[0]
print(f"Optimal Temperature: {optimal_temperature}")
with open('optimal_temperature.txt', 'w') as file:
    file.write(f"Optimal Temperature: {optimal_temperature}\n")