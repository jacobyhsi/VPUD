"""
Running llm on ToyRegression dataset
"""

import json
import requests
import re
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from datasets import load_from_disk
from sklearn.model_selection import train_test_split

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from torch.nn.functional import softmax

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument("--seed", default=123)
parser.add_argument("--seed_num", default="5")
parser.add_argument("--data", default="logistic_regression_1")
parser.add_argument("--feature", default="x1")
parser.add_argument("--shots", default=3)
parser.add_argument("--sets", default=10)
parser.add_argument("--num_modified_z", default=3)
parser.add_argument("--llm", default="llama70b-nemo")
args = parser.parse_args()
seed = int(args.seed)
np.random.seed(seed)
shots = int(args.shots)
sets = int(args.sets)
num_modified_z = int(args.num_modified_z)
pd.set_option('display.max_columns', None)

################################################################################################
############################################ LLM ###############################################
################################################################################################

def get_response(prompt, label_keys, seed):
    # Add label_keys to the payload
    payload = {
        'prompt': prompt,
        'label_keys': label_keys,
        'seed': seed
    }
    
    # Send POST request to the server
    response = requests.post('http://localhost:5000/predict', json=payload).json()
    
    # Extract response text and probabilities from the server's response
    response_text = response.get('response_text', "")
    probabilities = response.get('probabilities', [])
    
    return response_text, probabilities

################################################################################################
############################################ LLM ###############################################
################################################################################################

################################################################################################
##################################### Data Preprocessing #######################################
################################################################################################
data_path = f'ToyRegression/logistic_regression_data/{args.data}.csv'

# f = open(f'{data_path}/info.json')
# label_map = json.load(f)
# label_name = label_map['label']
# labels = label_map['map']
# label_keys = list(labels)

data = pd.read_csv(data_path, index_col=0)

label_name = "y"
label_keys = ["0", "1"]
data = data.rename(columns={label_name: 'label'})
data['label'] = data['label'].astype(int)

feature_columns = [col for col in data.columns if col != 'label']

data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

def parse_features_to_note(row, feature_columns: list[str]):
    note = []
    for feature in feature_columns:
        note.append(f"{feature} = {row[feature]}")
    # join note with ;
    return "; ".join(note)

data["note"] = data.apply(lambda row: parse_features_to_note(row, feature_columns), axis=1)

selected_feature = args.feature

x_row = data.sample(n=1, random_state=seed)
x = x_row['note'].iloc[0]
x_y = x_row['label'].iloc[0]
print("x:", x)

data = data.drop(x_row.index)

D_rows = data.sample(n=shots, random_state=seed)

D = "\n".join(
    [f" {row['note']} -> {label_name}: {row['label']}" for _, row in D_rows.iterrows()]
)

################################################################################################
##################################### Data Preprocessing #######################################
################################################################################################

def calculate_entropy(probs):
    # Calculate entropy using all probabilities in the dictionary
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 5)

data_rows = []

set_dict = {}
################################################################################################
######################################## E[H[p(y|u,z,x)]] ######################################
################################################################################################
# icl_initial_row = data.sample(n=1, random_state=seed)
initial_row = data.sample(n=1)
print("Initial Row:\n", initial_row['note'])
data = data.drop(initial_row.index)

z_lst = []

initial_selected_value = initial_row[selected_feature].values[0]
D_selected_values = D_rows[selected_feature].values

decimal_places = 1
previous_values = [initial_selected_value]
# Take new z values by sampling a normal distribution with mean from z and std from D values

for i in range(num_modified_z):
    for i in range(100):
        new_value = np.random.normal(initial_selected_value, 10*np.std(D_selected_values), 1)[0]
        new_value = round(new_value, decimal_places)
        if new_value not in previous_values:
            previous_values.append(new_value)
            break
        
    modified_row = initial_row.copy()
    modified_row[selected_feature] = new_value
    
    modified_row['note'] = modified_row.apply(lambda row: parse_features_to_note(row, feature_columns), axis=1)
    
    z_lst.append(modified_row)
    
z_data = pd.concat(z_lst, ignore_index=True)

seed_num = int(args.seed_num)

for i, row in z_data.iterrows():
    # print(f"\nProcessing Z Example {i + 1}")
    z = row['note']
    z_y = row['label']
    # print("Row Note:", z)

    # Initialize avg_puz_probs
    avg_puz_probs = {label: 0.0 for label in label_keys}
    
    # Initialize avg_puzx_probs
    avg_puzx_probs = {label: 0.0 for label in label_keys}

    # Initialize avg_pyxu_z_probs with distinct keys for each label
    avg_pyxu_z_probs = {
        f"p(y|x,u={outer_label},z)": {inner_label: 0.0 for inner_label in label_keys}
        for outer_label in label_keys
    }
    # pred_pyxu_z_preds = {f"p(y|x,u={outer_label},z)": 0.0 for outer_label in label_keys}
    
    # Initialize p(y|x,z)
    avg_pyxz_probs = {label: 0.0 for label in label_keys}
    
    # Initialize p(y|x)
    avg_pyx_probs = {label: 0.0 for label in label_keys}
    
    # ----- Processing p(u|z) and p(u|z,x) -----
    for seed in range(seed_num):
        ## p(u|z)
        print(f"\np(u|z) Seed {seed + 1}/{seed_num}")

        prompt_puz = (
            f"""Predict "{label_name}" from the following:

{z}

"{label_name}" takes the form of the following: {label_keys[0]} or {label_keys[1]}.

Please output **ONLY** your predicted {label_name} from {label_keys} and enclose your output in <output> </output> tags. ** DO NOT OUTPUT ANYTHING ELSE! **"""
        )
        
        print("Prompt for p(u|z):")
        print(prompt_puz)
        
        # Get the prediction and probabilities from the model
        # print("Getting response...")
        # print("Prompt:", prompt_puz)
        # print("Label Keys:", label_keys)
        # print("Seed:", seed)
        pred_puz, probs_puz = get_response(prompt_puz, label_keys, seed=seed)
        # print("pred_p(u|z):", pred_puz)
        # print("probs_p(u|z):", probs_puz)
        
        # Accumulate probabilities for puz
        for label, prob in probs_puz.items():
            avg_puz_probs[label] += prob

        # Extract the predicted label using regex
        match = re.search(r'<output>\s*(.*?)\s*</output>', pred_puz, re.DOTALL | re.IGNORECASE)
        if match:
            pred_puz_label = match.group(1).strip()
            # print(f"Extracted prediction: {pred_puz_label}")
            # print(f"True z label: {z_y}")
        else:
            print("Could not find output tags in the response.")
            raise ValueError("Invalid response format.")
        
        ## p(u|z,x)
        print(f"\np(u|z,x) Seed {seed + 1}/{seed_num}")

        prompt_puzx =(
            f"""Here are some examples:
            
- {x}

Given the Dataset examples, predict "{label_name}" from the following:

{z}

"{label_name}" takes the form of the following: {label_keys[0]} or {label_keys[1]}.

Please output **ONLY** your predicted {label_name} from {label_keys} and enclose your output in <output> </output> tags. ** DO NOT OUTPUT ANYTHING ELSE! **"""
        )
        
        
#         f"""Based on the sample provided below, predict the "{label_name}" of the following.
        
# {z}

# "{label_name}" takes the form of the following: {labels}.

# Here are some examples:

# {D}
# {x}

# Please output **ONLY** your predicted {label_name} label key from {label_keys} and enclose your output in <output> </output> tags. ** DO NOT OUTPUT ANYTHING ELSE! **"""
        
        print("Prompt for p(u|z,x):")
        print(prompt_puzx)

        # Get the prediction and probabilities from the model
        pred_puzx, probs_puzx = get_response(prompt_puzx, label_keys, seed=seed)
        # print("pred_p(u|z,x):", pred_puzx)
        # print("probs_p(u|z,x):", probs_puzx)

        # Accumulate probabilities for puz
        for label, prob in probs_puzx.items():
            avg_puzx_probs[label] += prob

        # Extract the predicted label using regex
        match = re.search(r'<output>\s*(.*?)\s*</output>', pred_puzx, re.DOTALL | re.IGNORECASE)
        if match:
            pred_puzx_label = match.group(1).strip()
            # print(f"Extracted prediction: {pred_puzx_label}")
            # print(f"True z label: {z_y}")
        else:
            print("Could not find output tags in the response.")
            raise ValueError("Invalid response format.")
        
        ## p(y|x)
        print(f"\np(y|x) Seed {seed + 1}/{seed_num}")

        prompt_pyx = (
            f"""Predict "{label_name}" from the following:
            
{x}

"{label_name}" takes the form of the following: {label_keys[0]} or {label_keys[1]}.

Please output **ONLY** your predicted {label_name} from {label_keys} and enclose your output in <output> </output> tags. ** DO NOT OUTPUT ANYTHING ELSE! **"""
        )
        
        print("Prompt for p(y|x,D):")
        print(prompt_pyx)

        # Get the prediction and probabilities from the model
        pred_pyx, probs_pyx = get_response(prompt_pyx, label_keys, seed=seed)
        # print("pred_p(y|x):", pred_pyx)
        # print("probs_p(y|x):", probs_pyx)
        
        # Accumulate probabilities for puz
        for label, prob in probs_pyx.items():
            avg_pyx_probs[label] += prob

        # Extract the predicted label using regex
        match = re.search(r'<output>\s*(.*?)\s*</output>', pred_pyx, re.DOTALL | re.IGNORECASE)
        if match:
            pred_pyx_label = match.group(1).strip()
            # print(f"Extracted prediction: {pred_pyx_label}")
            # print(f"True x label: {x_y}")
        else:
            print("Could not find output tags in the response.")
            raise ValueError("Invalid response format.")

    # Calculate the average probabilities for puz and puzx
    avg_puz_probs = {label: prob / seed_num for label, prob in avg_puz_probs.items()}
    # print("\nAveraged puz probabilities:", avg_puz_probs)
    
    avg_puzx_probs = {label: prob / seed_num for label, prob in avg_puzx_probs.items()}
    # print("\nAveraged puzx probabilities:", avg_puzx_probs)
    
    avg_pyx_probs = {label: prob / seed_num for label, prob in avg_pyx_probs.items()}
    # print("\nAveraged puzx probabilities:", avg_pyx_probs)

    # ----- Processing pyxu_z -----
    for outer_label in label_keys:
        print(f"\nProcessing p(y|x,u=_,z) for label '{outer_label}'")

        prompt_pyxuz = (
            f"""Here is an example:

- {z} -> {label_name}: {outer_label}

Given the example, predict "{label_name}" from the following:

{x}            

"{label_name}" takes the form of the following: {label_keys[0]} or {label_keys[1]}.

Please output **ONLY** your predicted {label_name} from {label_keys} and enclose your output in <output> </output> tags. ** DO NOT OUTPUT ANYTHING ELSE! **"""
        )
        
#         prompt_pyxuz = (
#             f"""Based on the sample provided below, predict the "{label_name}". 
# "{label_name}" takes the form of the following: {labels}.

# {x}

# The following is an in-context example that will help you make your prediction:

# {z} -> {label_name}: {outer_label}

# Please output **ONLY** your predicted {label_name} label key from {label_keys} and enclose your output in <output> </output> tags. ** DO NOT OUTPUT ANYTHING ELSE! **"""
#         )

        print("Prompt for p(y|x,u=_,z):")
        print(prompt_pyxuz)

        for seed in range(seed_num):
            print(f"\np(y|x,u={outer_label},z) Seed {seed + 1}/{seed_num}")

            pred_pyxuz, probs_pyxuz = get_response(prompt_pyxuz, label_keys, seed=seed)
            print(f"pred_p(y|x,u={outer_label},z):", pred_pyxuz)
            print(f"probs_p(y|x,u={outer_label},z):", probs_pyxuz)

            # exit()
            
            # Accumulate probabilities for pyxu_z
            for inner_label, prob in probs_pyxuz.items():
                avg_pyxu_z_probs[f"p(y|x,u={outer_label},z)"][inner_label] += prob
                
            match = re.search(r'<output>\s*(.*?)\s*</output>', pred_pyxuz, re.DOTALL | re.IGNORECASE)

            if match:
                pred_pyxuz_label = int(match.group(1).strip())
                # print(f"Extracted prediction: {pred_pyxuz_label}")
                # pred_pyxu_z_preds[f"pred_p(y|x,u={outer_label},z)"] = pred_pyxuz_label
            else:
                print("Could not find output tags in the response.")
                raise ValueError("Invalid response format.")

    # Calculate the average probabilities for pyxu_z
    for key, sub_dict in avg_pyxu_z_probs.items():
        avg_pyxu_z_probs[key] = {label: prob / seed_num for label, prob in sub_dict.items()}

    # print("\nAveraged pyxu_z probabilities:", avg_pyxu_z_probs)
    
    # Calculate the average probabilities for p(y|x,z) using p(y|x,u,z) and p(u|z,x)/could also be p(u|z)???
    # Marginalization
    for label in label_keys:  # Iterate over all possible values of y
        avg_pyxz_probs[label] = sum(
            avg_pyxu_z_probs[f"p(y|x,u={u_label},z)"][label] * avg_puz_probs[u_label]
            for u_label in avg_puz_probs.keys()
        )
    # print("\nAveraged p(y|x,z) probabilities:", avg_pyxz_probs)

    # ----- Optional: Adding Averages to DataFrame -----
    # Add averaged puz probabilities to the DataFrame
    for label, avg_prob in avg_puz_probs.items():
        z_data.at[i, f"p(u={label}|z)"] = avg_prob
        
    # Add averaged puzx probabilities to the DataFrame
    for label, avg_prob in avg_puzx_probs.items():
        z_data.at[i, f"p(u={label}|z,x)"] = avg_prob

    # Add averaged pyxu_z probabilities to the DataFrame
    for key, sub_dict in avg_pyxu_z_probs.items():
        for label, avg_prob in sub_dict.items():
            new_key = re.sub(r'y', f'y={label}', key, count=1)
            z_data.at[i, new_key] = avg_prob
            
    # Add averaged pyxz probabilities to the DataFrame
    for label, avg_prob in avg_pyxz_probs.items():
        z_data.at[i, f"p(y={label}|x,z)"] = avg_prob
        
    # Add averaged pyx probabilities to the DataFrame
    for label, avg_prob in avg_pyx_probs.items():
        z_data.at[i, f"p(y={label}|x)"] = avg_prob
        
    # ----- Compute Entropy for Each avg_pyxu_z_probs -----
    for key, sub_dict in avg_pyxu_z_probs.items():
        entropy = calculate_entropy(sub_dict)
        # H[p(y|x,u0,z)]
        # print(f"H[{key}]")
        z_data.at[i, f"H[{key}]"] = entropy
        
    # ----- Compute Entropy for Each avg_pyx_probs -----
    z_data.at[i, f"H[p(y|x)]"] = calculate_entropy(avg_pyx_probs)
        
    # ----- Compute Entropy for Each avg_pyxz_probs -----
    z_data.at[i, f"H[p(y|x,z)]"] = calculate_entropy(avg_pyxz_probs)
    
    expected_H = 0.0
    for label in label_keys:
        avg_puzx_prob = z_data.at[i, f"p(u={label}|z,x)"]
        avg_pyxuz_entropy = z_data.at[i, f"H[p(y|x,u={label},z)]"]
        expected_H += avg_puzx_prob * avg_pyxuz_entropy
    z_data.at[i, "Va = E[H[p(y|x,u,z)]]"] = round(expected_H, 5)
    
    # Computing Epistemic Uncertainty
    # z_data["Ve = H[p(y|x,z)] - E[H[p(y|x,u,z)]]"] = z_data["H[p(y|x,z)]"] - z_data["Va = E[H[p(y|x,u,z)]]"]
        
    # ----- Store the Predictions -----
    # for outer_label in label_keys:
    #     preds = pred_pyxu_z_preds[f"pred_p(y|x,u={outer_label},z)"]
    #     z_data.at[i, f"pred_p(y|x,u={outer_label},z)"] = preds
            
    # ----- Final Output -----
    print("\nFinal z_data with Averaged Probabilities:")
    print(z_data.head())
    
    total_U = z_data["H[p(y|x)]"][0]
    print("Total Uncertainty =", total_U)
    min_Va = z_data["Va = E[H[p(y|x,u,z)]]"].min()
    print("min Va = E[H[p(y|x,u,z)]] =", min_Va)
    max_Ve = round(total_U - min_Va, 5)
    print("max Ve = H[p(y|x,z)] - E[H[p(y|x,u,z)] =", max_Ve)
    
    z_data.to_csv(f"results_{args.data}.csv", index=False)
