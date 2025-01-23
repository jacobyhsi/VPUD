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
parser.add_argument("--llm", default="llama70b-nemo")
args = parser.parse_args()
seed = int(args.seed)
np.random.seed(seed)
shots = int(args.shots)
sets = int(args.sets)
pd.set_option('display.max_columns', None)

################################################################################################
############################################ LLM ###############################################
################################################################################################\

# def get_response(prompt, label_keys, seed):
#     # Add label_keys to the payload
#     payload = {
#         'prompt': prompt,
#         'label_keys': label_keys,
#         'seed': seed
#     }
    
#     # Send POST request to the server
#     response = requests.post('http://localhost:5000/predict', json=payload).json()
    
#     # Extract response text and probabilities from the server's response
#     response_text = response.get('response_text', "")
#     probabilities = response.get('probabilities', [])
    
#     return response_text, probabilities

################################################### LLM ###################################################
login(token = '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')

if args.llm == "gemma9b":
    model_id = "google/gemma-2-9b-it"
elif args.llm == "gemma27b":
    model_id = "google/gemma-2-27b-it"
elif args.llm == "llama70b":
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
elif args.llm == "llama70b-nemo":
    model_id = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

# Load the model and tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

# Create the LLM instance
llm = {"tokenizer": tokenizer, "model": model}

# Define get_response function
def get_response(llm, prompt, label_keys, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    tokenizer, model = llm["tokenizer"], llm["model"]
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    # outputs = model.generate(**input_ids, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id,
                            #  output_scores=True, return_dict_in_generate=True, do_sample = True, top_p=0.9, top_k = 50)
    outputs = model.generate(**input_ids, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id,
                             output_scores=True, return_dict_in_generate=True, do_sample = True, temperature = 0.7)
    gen_text = tokenizer.decode(outputs.sequences[0])
    
    # Find the starting point of the prompt in the generated text
    start_pos = gen_text.find(prompt)
    if start_pos == -1:
        return "Prompt not found in the generated text."

    # Extract the response starting from the prompt
    response_text = gen_text[start_pos + len(prompt):].strip()
    
    # Find the end position of <end_of_turn> if it exists -- Gemma
    end_pos = response_text.find("<end_of_turn>")
    if end_pos != -1:
        response_text = response_text[:end_pos].strip()

    # Find the end position of <end_of_turn> if it exists -- Llama
    end_pos = response_text.find("<|eot_id|>")
    if end_pos != -1:
        response_text = response_text[:end_pos].strip()
    
    # Process end-of-turn tags for different models
    for end_tag in ["<end_of_turn>", "<|eot_id|>"]:
        end_pos = response_text.find(end_tag)
        if end_pos != -1:
            response_text = response_text[:end_pos].strip()
            
    print("response_text", response_text)

    # Extract the predicted token within <output> </output> tags
    # match = re.search(r'<output>\s*(.*?)\s*</output>', response_text)
    pattern = r'\b(' + '|'.join(map(str, label_keys)) + r')\b'
    match = re.search(pattern, response_text)
    
    print("match", match)
    
    if not match:
        print("Prediction not found in expected format.")
        exit()

    predicted_token = match.group(1).strip()
    if predicted_token not in label_keys:
        print(f"Predicted token '{predicted_token}' not in label_keys.")
        exit()

    # Now find the position where the predicted token was generated
    # Tokenize the predicted token to get its token ids
    predicted_token_ids = tokenizer(predicted_token, add_special_tokens=False)['input_ids']

    # Get the generated token ids (excluding the input prompt)
    generated_token_ids = outputs.sequences[0][input_ids['input_ids'].shape[1]:].tolist()
    
    # Find the index where the predicted token starts in generated_token_ids
    def find_sublist(sublist, main_list):
        for i in range(len(main_list) - len(sublist) + 1):
            if main_list[i:i+len(sublist)] == sublist:
                return i
        return -1

    start_idx = find_sublist(predicted_token_ids, generated_token_ids)
    if start_idx == -1:
        return "Predicted token ids not found in generated token ids.", None, None

    # At the position where the predicted token starts, get the probability distribution
    # Get the score at that position
    score = outputs.scores[start_idx]

    # # Get the probabilities
    # prob_dist = softmax(score, dim=-1)

    # # Build the probability distribution
    # probability_distribution = {
    #     label: round(prob_dist[0, tokenizer.convert_tokens_to_ids(label)].item(), 5)
    #     for label in label_keys
    # }

    # total_prob = sum(probability_distribution.values())
    # normalized_probability_distribution = {
    #     label: round(prob / total_prob, 5)
    #     for label, prob in probability_distribution.items()
    # }
    
    # Extract logits for each label in label_keys
    label_logits = []
    for label in label_keys:
        label_id = tokenizer.convert_tokens_to_ids(label)
        label_logits.append(score[0, label_id].item())
    
    return response_text, label_logits

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
print("len(data)*0.1", int(len(data)*0.1))
calibration_data = data.sample(n=int(len(data)*0.1), random_state=42)
data = data.drop(calibration_data.index)
calibration_data = calibration_data.reset_index(drop=True)

logit_pairs = []
label_lst = []
for i, row in calibration_data.iterrows():
    print(f"\nProcessing Calibration Example {i + 1}")
    
    note = row["note"]
    label = row["label"]
    
    prompt = (
            f"""Here are some Dataset examples:

{D}

Given the Dataset examples, predict the "{label_name}" of the following. Please output ONLY your predicted {label_name} label key from {label_keys}. DO NOT OUTPUT ANYTHING ELSE!:
            
{note}

"{label_name}" takes the form of the following: {labels}.

Let me repeat again, output your predicted {label_name} label key from {label_keys} ONLY. DON'T OUTPUT ANYTHING ELSE!"""
        )
    
    response_text, label_logits = get_response(llm, prompt, label_keys, seed=i)
    if label_logits is not None:
        logit_pairs.append(label_logits)  # [logit_for_0, logit_for_1]
        label_lst.append(label)
        
print("Logit Pairs:", logit_pairs)
print("Labels:", label_lst)

label_lst = [int(l) for l in label_lst]
        

def bce_loss_with_temperature(temperature, logit_pairs, labels):
    temperature = float(temperature)
    total_loss = 0.0
    n = len(labels)
    
    for (z0, z1), y in zip(logit_pairs, labels):
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