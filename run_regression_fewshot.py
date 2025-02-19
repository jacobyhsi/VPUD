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

from vpud_utils import calculate_entropy
from regression_data_processing import parse_features_to_note, create_x_row_from_x_features

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument("--seed", default=123)
parser.add_argument("--num_x_values", default="1")
parser.add_argument("--x_values", default=None)
parser.add_argument("--seed_num", default="5")
parser.add_argument("--data", default="logistic_regression_3")
parser.add_argument("--feature", default="x1")
parser.add_argument("--shots", default=3)
parser.add_argument("--sets", default=10)
parser.add_argument("--num_modified_z", default=3)
parser.add_argument("--llm", default="llama70b-nemo")
parser.add_argument("--run_name", default="fewshot")
parser.add_argument("--save_directory", default="other")
parser.add_argument("--specify_dataset_type", default=0)
args = parser.parse_args()
seed = int(args.seed)
np.random.seed(seed)
num_x_values = int(args.num_x_values)
x_features = args.x_values
shots = int(args.shots)
sets = int(args.sets)
num_modified_z = int(args.num_modified_z)
run_name = args.run_name
save_directory = args.save_directory
specify_dataset_type = int(args.specify_dataset_type)
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
########################################## Prompts #############################################
################################################################################################

if specify_dataset_type == 0:
    dataset_specification_string = "dataset"
elif specify_dataset_type == 1:
    dataset_specification_string = "LOGISTIC_REGRESSION dataset"

def prompt_start():
    prompt = f"""Here are some samples from a {dataset_specification_string}:"""
    
    return prompt

def prompt_middle(label_name):
    prompt = f"""Given the dataset samples, predict "{label_name}" from the following:"""
    
    return prompt

def prompt_end(label_name, label_keys):
    prompt =  f""""{label_name}" takes the form of the following: {label_keys[0]} or {label_keys[1]}.

Please output **ONLY** your predicted {label_name} label key from {label_keys} and enclose your output in <output> </output> tags. ** DO NOT OUTPUT ANYTHING ELSE! **
"""
    return prompt

def full_prompt(incontext_examples: list[str], example: str, label_name: str = "y", label_keys: list[str] = ["0", "1"]):
    incontext_examples_str = "\n".join(incontext_examples)
    
    prompt = f"""{prompt_start()}

{incontext_examples_str}

{prompt_middle(label_name)}

{example}

{prompt_end(label_name, label_keys)}"""

    return prompt

################################################################################################
########################################## Prompts #############################################
################################################################################################

################################################################################################
##################################### Data Preprocessing #######################################
################################################################################################
data_path = f'ToyRegression/logistic_regression_data/{args.data}.csv'

data = pd.read_csv(data_path, index_col=0)

label_name = "y"
label_keys = ["0", "1"]
data = data.rename(columns={label_name: 'label'})
data['label'] = data['label'].astype(int)

feature_columns = [col for col in data.columns if col != 'label']

data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

data["note"] = data.apply(lambda row: parse_features_to_note(row, feature_columns), axis=1)

print("Features:", feature_columns)

selected_feature = args.feature
print("Feature to vary:", selected_feature)

# exit() # here first to check whats the feature column names

if x_features is None:
    x_row = data.sample(n=num_x_values, random_state=seed)
    data = data.drop(x_row.index)
else:
    x_row = create_x_row_from_x_features(x_features, feature_columns)

data = data.drop(x_row.index)

D_rows = data.sample(n=shots, random_state=seed)

D = "\n".join(
    [f"- {row['note']} -> {label_name}: {row['label']}" for _, row in D_rows.iterrows()]
)

D_rows.to_csv(f"results/{save_directory}/D_{run_name}_{args.data}.csv", index=False)

################################################################################################
##################################### Data Preprocessing #######################################
################################################################################################

################################################################################################
######################################## Z Selection ###########################################
################################################################################################

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
        new_value = np.random.normal(np.mean(D_selected_values), 2*np.std(D_selected_values), 1)[0]
        new_value = round(new_value, decimal_places)
        if new_value not in previous_values:
            previous_values.append(new_value)
            break
        
    modified_row = initial_row.copy()
    modified_row[selected_feature] = new_value
    
    modified_row['note'] = modified_row.apply(lambda row: parse_features_to_note(row, feature_columns), axis=1)
    
    z_lst.append(modified_row)
    
z_data = pd.concat(z_lst, ignore_index=True)
# print("Modified ICL rows:\n", icl_data)

seed_num = int(args.seed_num)

for i in range(num_modified_z):
    # print(f"\nProcessing Z Example {i + 1}")
    
    row = z_data.iloc[i]
    
    z = row['note']

    avg_puz_probs = {label: 0.0 for label in label_keys}

    for seed in range(seed_num):
        # Initialize avg_puz_probs
        
        ## p(u|z)
        print(f"\np(u|z) Seed {seed + 1}/{seed_num}")

        prompt_puz = full_prompt([D], z)
        
        print("Prompt for p(u|z):")
        print(prompt_puz)

        # Get the prediction and probabilities from the model
        pred_puz, probs_puz = get_response(prompt_puz, label_keys, seed=seed)
        # print("pred_p(u|z):", pred_puz)
        # print("probs_p(u|z):", probs_puz)
        # Accumulate probabilities for puz
        for label, prob in probs_puz.items():
            avg_puz_probs[label] += prob
            
    # Calculate the average probabilities for puz and puzx
    avg_puz_probs = {label: prob / seed_num for label, prob in avg_puz_probs.items()}
    # print("\nAveraged puz probabilities:", avg_puz_probs)
    
    # Add averaged puz probabilities to the DataFrame
    for label, avg_prob in avg_puz_probs.items():
        z_data.at[i, f"p(u={label}|z)"] = avg_prob
        
    z_data.at[i, "H[p(u|z)]"] = calculate_entropy(avg_puz_probs)

################################################################################################
######################################## Z Selection ###########################################
################################################################################################

################################################################################################
######################################## E[H[p(y|u,z,x)]] ######################################
################################################################################################

for j in range(num_x_values):
    x = x_row['note'].iloc[j]
    x_y = x_row['label'].iloc[j]
    print("x:", x)
    for i, row in z_data.iterrows():
        # print(f"\nProcessing Z Example {i + 1}")
        z = row['note']
        z_y = row['label']
        # print("Row Note:", z)
        
        # Initialize avg_puzx_probs
        avg_puzx_probs = {label: 0.0 for label in label_keys}

        # Initialize avg_pyxu_z_probs with distinct keys for each label
        avg_pyxu_z_probs = {
            f"p(y|x,u={outer_label},z)": {inner_label: 0.0 for inner_label in label_keys}
            for outer_label in label_keys
        }
       
        # Initialize p(y|x,z)
        avg_pyxz_probs = {label: 0.0 for label in label_keys}
        
        # Initialize p(y|x)
        avg_pyx_probs = {label: 0.0 for label in label_keys}
        
        # ----- Processing p(u|z) and p(u|z,x) -----
        for seed in range(seed_num):
            # ## p(u|z)
            # print(f"\np(u|z) Seed {seed + 1}/{seed_num}")

            # prompt_puz = full_prompt([D], z)
            
            # print("Prompt for p(u|z):")
            # print(prompt_puz)

            # # Get the prediction and probabilities from the model
            # pred_puz, probs_puz = get_response(prompt_puz, label_keys, seed=seed)
            # # print("pred_p(u|z):", pred_puz)
            # # print("probs_p(u|z):", probs_puz)
            
            # # Accumulate probabilities for puz
            # for label, prob in probs_puz.items():
            #     avg_puz_probs[label] += prob
            
            ## p(u|z,x)
            print(f"\np(u|z,x) Seed {seed + 1}/{seed_num}")

            prompt_puzx = full_prompt([D, f"- {x}"], z)
            
            print("Prompt for p(u|z,x):")
            print(prompt_puzx)
            # Get the prediction and probabilities from the model
            pred_puzx, probs_puzx = get_response(prompt_puzx, label_keys, seed=seed)
            # print("pred_p(u|z,x):", pred_puzx)
            # print("probs_p(u|z,x):", probs_puzx)

            # Accumulate probabilities for puz
            for label, prob in probs_puzx.items():
                avg_puzx_probs[label] += prob
            
            ## p(y|x)
            print(f"\np(y|x) Seed {seed + 1}/{seed_num}")

            prompt_pyx = full_prompt([D], x)
            
            print("Prompt for p(y|x,D):")
            print(prompt_pyx)

            # Get the prediction and probabilities from the model
            pred_pyx, probs_pyx = get_response(prompt_pyx, label_keys, seed=seed)
            # print("pred_p(y|x):", pred_pyx)
            # print("probs_p(y|x):", probs_pyx)
            
            # Accumulate probabilities for puz
            for label, prob in probs_pyx.items():
                avg_pyx_probs[label] += prob

        # # Calculate the average probabilities for puz and puzx
        # avg_puz_probs = {label: prob / seed_num for label, prob in avg_puz_probs.items()}
        # # print("\nAveraged puz probabilities:", avg_puz_probs)
        
        avg_puzx_probs = {label: prob / seed_num for label, prob in avg_puzx_probs.items()}
        # print("\nAveraged puzx probabilities:", avg_puzx_probs)
        
        avg_pyx_probs = {label: prob / seed_num for label, prob in avg_pyx_probs.items()}
        # print("\nAveraged puzx probabilities:", avg_pyx_probs)

        # ----- Processing pyxu_z -----
        for outer_label in label_keys:
            print(f"\nProcessing p(y|x,u=_,z) for label '{outer_label}'")

            prompt_pyxuz = full_prompt([f"- {z} -> {label_name}: {outer_label}", D], x)

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

        # Calculate the average probabilities for pyxu_z
        for key, sub_dict in avg_pyxu_z_probs.items():
            avg_pyxu_z_probs[key] = {label: prob / seed_num for label, prob in sub_dict.items()}

        # print("\nAveraged pyxu_z probabilities:", avg_pyxu_z_probs)
        
        # Calculate the average probabilities for p(y|x,z) using p(y|x,u,z) and p(u|z,x)/could also be p(u|z)???
        # Marginalization
        for label in label_keys:  # Iterate over all possible values of y
            avg_pyxz_probs[label] = sum(
                avg_pyxu_z_probs[f"p(y|x,u={u_label},z)"][label] * z_data.at[i, f"p(u={u_label}|z)"]
                for u_label in avg_puz_probs.keys()
            )            
            
            # avg_pyxz_probs[label] = sum(
            #     avg_pyxu_z_probs[f"p(y|x,u={u_label},z)"][label] * avg_puz_probs[u_label]
            #     for u_label in avg_puz_probs.keys()
            # )
        # print("\nAveraged p(y|x,z) probabilities:", avg_pyxz_probs)

        # ----- Optional: Adding Averages to DataFrame -----
        # # Add averaged puz probabilities to the DataFrame
        # for label, avg_prob in avg_puz_probs.items():
        #     z_data.at[i, f"p(u={label}|z)"] = avg_prob
            
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
                
        # ----- Final Output -----
        print(f"\nx value: {x} -> {label_name}: {x_y}")
        print("\nFinal z_data with Averaged Probabilities:")
        print(z_data.head())
        z_data["true_x"] = x_row.iloc[j][selected_feature]
        
        z_data.to_csv(f"results/{save_directory}/results_{run_name}_{args.data}_x{j}.csv", index=False)
        
    total_U = z_data["H[p(y|x)]"][0]
    print("\nTotal Uncertainty =", total_U)
    maximum_entropic_distance = total_U/2
    print("\nMaximum Entropic Distance =", maximum_entropic_distance)
    # Find the valid z values
    valid_Va = []
    for i, row in z_data.iterrows():
        if abs(total_U - row["H[p(y|x,z)]"]) <= maximum_entropic_distance:
            valid_Va.append(row["Va = E[H[p(y|x,u,z)]]"])
    if len(valid_Va) == 0:
        print("No Va values found within threshold. Using the minimum Va value for whole z dataset.")
        min_Va = z_data["Va = E[H[p(y|x,u,z)]]"].min()
        z_data["within_threshold"] = False
    else:
        min_Va = min(valid_Va)
        z_data["within_threshold"] = True
    # min_Va = z_data["Va = E[H[p(y|x,u,z)]]"].min()
    print("min Va = E[H[p(y|x,u,z)]] =", min_Va)
    max_Ve = round(total_U - min_Va, 5)
    print("max Ve = H[p(y|x,z)] - E[H[p(y|x,u,z)] =", max_Ve)
    z_data["min_Va"] = min_Va
    z_data["max_Ve"] = max_Ve
    z_data.to_csv(f"results/{save_directory}/results_{run_name}_{args.data}_x{j}.csv", index=False)
