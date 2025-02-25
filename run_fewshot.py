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

# login(token = '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')

# if args.llm == "gemma9b":
#     model_id = "google/gemma-2-9b-it"
# elif args.llm == "gemma27b":
#     model_id = "google/gemma-2-27b-it"
# elif args.llm == "llama70b":
#     model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# elif args.llm == "llama70b-nemo":
#     model_id = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

# # Load the model and tokenizer

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

# # Create the LLM instance
# llm = {"tokenizer": tokenizer, "model": model}

# # Define get_response function
# def get_response(llm, prompt, label_keys, seed):
    
#     print("seed", seed)
    
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
    
#     tokenizer, model = llm["tokenizer"], llm["model"]
#     input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
#     outputs = model.generate(**input_ids, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id,
#                              output_scores=True, return_dict_in_generate=True, do_sample = True, top_p=0.9, top_k = 50)
#     gen_text = tokenizer.decode(outputs.sequences[0])
    
#     # Find the starting point of the prompt in the generated text
#     start_pos = gen_text.find(prompt)
#     if start_pos == -1:
#         return "Prompt not found in the generated text."

#     # Extract the response starting from the prompt
#     response_text = gen_text[start_pos + len(prompt):].strip()

#     # Find the end position of <end_of_turn> if it exists -- Gemma
#     end_pos = response_text.find("<end_of_turn>")
#     if end_pos != -1:
#         response_text = response_text[:end_pos].strip()

#     # Find the end position of <end_of_turn> if it exists -- Llama
#     end_pos = response_text.find("<|eot_id|>")
#     if end_pos != -1:
#         response_text = response_text[:end_pos].strip()
    
#     # Process end-of-turn tags for different models
#     for end_tag in ["<end_of_turn>", "<|eot_id|>"]:
#         end_pos = response_text.find(end_tag)
#         if end_pos != -1:
#             response_text = response_text[:end_pos].strip()

#     # Extract the predicted token within <output> </output> tags
#     match = re.search(r'<output>\s*(.*?)\s*</output>', response_text)
#     if not match:
#         print("Prediction not found in expected format.")
#         exit()

#     predicted_token = match.group(1).strip()
#     if predicted_token not in label_keys:
#         print(f"Predicted token '{predicted_token}' not in label_keys.")
#         exit()

#     # Now find the position where the predicted token was generated
#     # Tokenize the predicted token to get its token ids
#     predicted_token_ids = tokenizer(predicted_token, add_special_tokens=False)['input_ids']

#     # Get the generated token ids (excluding the input prompt)
#     generated_token_ids = outputs.sequences[0][input_ids['input_ids'].shape[1]:].tolist()
    
#     # Find the index where the predicted token starts in generated_token_ids
#     def find_sublist(sublist, main_list):
#         for i in range(len(main_list) - len(sublist) + 1):
#             if main_list[i:i+len(sublist)] == sublist:
#                 return i
#         return -1

#     start_idx = find_sublist(predicted_token_ids, generated_token_ids)
#     if start_idx == -1:
#         return "Predicted token ids not found in generated token ids.", None, None

#     # At the position where the predicted token starts, get the probability distribution
#     # Get the score at that position
#     score = outputs.scores[start_idx]

#     # Get the probabilities
#     prob_dist = softmax(score, dim=-1)

#     # Build the probability distribution
#     probability_distribution = {
#         label: round(prob_dist[0, tokenizer.convert_tokens_to_ids(label)].item(), 5)
#         for label in label_keys
#     }

#     total_prob = sum(probability_distribution.values())
#     normalized_probability_distribution = {
#         label: round(prob / total_prob, 5)
#         for label, prob in probability_distribution.items()
#     }
    
#     return response_text, normalized_probability_distribution

################################################################################################
############################################ LLM ###############################################
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

data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

# Replace The and is with =, remove extra spaces
data['note'] = (data['note']
                  .str.replace(r'\bThe\b', '', regex=True)
                  .str.replace(r'\bis\b', '=', regex=True)
                  .str.replace(r'\s{2,}', ' ', regex=True)
                  .str.lstrip())

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

# Apply the function to the 'note' column and expand the dictionaries into columns
tmp_combined_data_note = data['note']
note2features = data['note'].apply(parse_note_to_features).apply(pd.Series)
feature_columns = note2features.columns
print("Features:", feature_columns)

selected_feature = 'Occupation'
print("Feature to vary:", selected_feature)
note2features = data['note'].apply(parse_note_to_features).apply(pd.Series)[selected_feature]

# exit() # here first to check whats the feature column names

# # Concatenate the features DataFrame with the 'label' column
df_combined = pd.concat([note2features, data['label']], axis=1)
data = pd.concat([tmp_combined_data_note, df_combined], axis=1)

x_row = data.sample(n=1, random_state=seed)
x = x_row['note'].iloc[0]
x_y = x_row['label'].iloc[0]
print("x:", x)

data = data.drop(x_row.index)

num_D = 3
D_rows = data.sample(n=num_D, random_state=seed)

D = "\n".join(
    [f"- {row['note']} -> {label_name}: {row['label']}" for _, row in D_rows.iterrows()]
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

unique_values = data[selected_feature].dropna().unique() # all selected feature values
# num_icl = 5 # debugging
# unique_values = np.random.choice(data[selected_feature].dropna().unique(), size=num_icl, replace=False)

for new_value in unique_values:
    print(f"{selected_feature} changed to: {new_value}")
    modified_row = initial_row.copy()
    modified_row[selected_feature] = new_value
    
    pattern = rf'({selected_feature} = )(.*?)(\.|$)'
    modified_note = re.sub(
        pattern, 
        rf'\1{new_value}\3', 
        modified_row['note'].values[0]
    )
    modified_row['note'] = modified_note
    
    z_lst.append(modified_row)
    

z_data = pd.concat(z_lst, ignore_index=True)
# print("Modified ICL rows:\n", icl_data)

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
            f"""Here are some Dataset examples:

{D}
            
Given the Dataset examples, predict the "{label_name}" of the following:

{z}

"{label_name}" takes the form of the following: {labels}.

Please output **ONLY** your predicted {label_name} label key from {label_keys} and enclose your output in <output> </output> tags. ** DO NOT OUTPUT ANYTHING ELSE! **"""
        )
        
        print("Prompt for p(u|z):")
        print(prompt_puz)

        # Get the prediction and probabilities from the model
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
            f"""Here are some Dataset examples:

{D}
- {x}

Given the Dataset examples, predict the "{label_name}" of the following:

{z}

"{label_name}" takes the form of the following: {labels}.

Please output **ONLY** your predicted {label_name} label key from {label_keys} and enclose your output in <output> </output> tags. ** DO NOT OUTPUT ANYTHING ELSE! **"""
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
            f"""Here are some Dataset examples:

{D}

Given the Dataset examples, predict the "{label_name}" of the following:
            
{x}

"{label_name}" takes the form of the following: {labels}.

Please output **ONLY** your predicted {label_name} label key from {label_keys} and enclose your output in <output> </output> tags. ** DO NOT OUTPUT ANYTHING ELSE! **"""
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
            f"""Here are some Dataset examples:

- {z} -> {label_name}: {outer_label}
{D}

Given the Dataset examples, predict the "{label_name}" of the following:

{x}            

"{label_name}" takes the form of the following: {labels}.

Please output **ONLY** your predicted {label_name} label key from {label_keys} and enclose your output in <output> </output> tags. ** DO NOT OUTPUT ANYTHING ELSE! **"""
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
            # print(f"pred_p(y|x,u={outer_label},z):", pred_pyxuz)
            # print(f"probs_p(y|x,u={outer_label},z):", probs_pyxuz)

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
        z_data.at[i, f"p(u|z)={label}"] = avg_prob
        
    # Add averaged puzx probabilities to the DataFrame
    for label, avg_prob in avg_puzx_probs.items():
        z_data.at[i, f"p(u|z,x)={label}"] = avg_prob

    # Add averaged pyxu_z probabilities to the DataFrame
    for key, sub_dict in avg_pyxu_z_probs.items():
        for label, avg_prob in sub_dict.items():
            z_data.at[i, f"{key}={label}"] = avg_prob
            
    # Add averaged pyxz probabilities to the DataFrame
    for label, avg_prob in avg_pyxz_probs.items():
        z_data.at[i, f"p(y|x,z)={label}"] = avg_prob
        
    # Add averaged pyx probabilities to the DataFrame
    for label, avg_prob in avg_pyx_probs.items():
        z_data.at[i, f"p(y|x)={label}"] = avg_prob
        
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
        avg_puzx_prob = z_data.at[i, f"p(u|z,x)={label}"]
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
