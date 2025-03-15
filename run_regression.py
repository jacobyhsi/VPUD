import requests
import math
from src.chat import get_first_token_probs
message = \
"""
 x1 = 2.8 <output>1</output>
 x1 = 0.9 <output>0</output>
 x1 = -6.7 <output>0</output>
 x1 = 2.0 <output>1</output>
 x1 = 3.5 <output>1</output>
 x1 = -2.7 <output>0</output>
 x1 = 1.0 <output>0</output>
 x1 = 0.5 <output>0</output>
 x1 = -3.1 <output>0</output>
 x1 = 0.2 <output>0</output>
 x1 = 14.3 <output>0</output>
 x1 = 4.2 <output>1</output>
 x1 = 0.8 <output>0</output>
 x1 = 0.6 <output>0</output>
 x1 = -1.6 <output>0</output>
 x1 = 0.8 <output>0</output>
 x1 = 6.8 <output>"""

import json
import requests
import re
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from datasets import load_from_disk
from sklearn.model_selection import train_test_split

from bayesian_optimisation import new_candidate

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument("--seed", default=123)
parser.add_argument("--num_x_values", default="1")
parser.add_argument("--x_values", default=None)
parser.add_argument("--x_range", default=None)
parser.add_argument("--seed_num", default="5")
parser.add_argument("--data", default="logistic_regression_3")
parser.add_argument("--feature", default="x1")
parser.add_argument("--shots", default=3)
parser.add_argument("--sets", default=10)
parser.add_argument("--num_modified_z", default=3)
parser.add_argument("--num_random_z", default=3)
parser.add_argument("--llm", default="llama70b-nemo")
parser.add_argument("--run_name", default="fewshot")
parser.add_argument("--save_directory", default="other")
parser.add_argument("--port", default=5000)
parser.add_argument("--x_save_value", default=0)
args = parser.parse_args()
seed = int(args.seed)
np.random.seed(seed)
num_x_values = int(args.num_x_values)
x_features = args.x_values
x_range = args.x_range
shots = int(args.shots)
sets = int(args.sets)
num_modified_z = int(args.num_modified_z)
num_random_z = int(args.num_random_z)
run_name = args.run_name
save_directory = args.save_directory
x_save_value = int(args.x_save_value)
port = args.port
pd.set_option('display.max_columns', None)

################################################################################################
########################################## Prompts #############################################
################################################################################################

def short_prompt(incontext_examples: list[str], example: str, *args, **kwargs):
    incontext_examples_str = "\n".join(incontext_examples)
    
    prompt = f"""{incontext_examples_str}\n {example} <output>"""
    
    return prompt

def note_label_prompt(note: str, label: str):
    prompt = f""" {note} <output>{label}</output>"""
    
    return prompt

def note_label_df_to_icl_examples(
        note_label_df: pd.DataFrame,
        seed: int,
        z_note: Optional[str] = None,
        u_label: Optional[str|int] = None,
    ):
    """
    Converts a DataFrame of notes and labels to incontext examples for LLM.
    Shuffles the DataFrame before converting.
    
    If z_note and u_label are provided, the z_note and u_label will be added to data as well.
    """
    
    if z_note is not None and u_label is not None:
        z_note_label_df = pd.DataFrame([{"note": z_note, "label": u_label}])
        note_label_df = pd.concat([note_label_df, z_note_label_df], ignore_index=True)
        
    note_label_df = note_label_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    incontext_examples = []
    
    for _, row in note_label_df.iterrows():
        incontext_examples.append(note_label_prompt(row['note'], row['label']))
    
    return incontext_examples


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

if x_features is not None:
    x_row = create_x_row_from_x_features(x_features, feature_columns)
    num_x_values = len(x_row)
elif x_range is not None:
    x_row = create_x_row_from_x_range(x_range, feature_columns)
    num_x_values = len(x_row)
else:
    x_row = data.sample(n=num_x_values, random_state=seed)
    data = data.drop(x_row.index)
D_rows = data.sample(n=shots, random_state=seed)

D_note_label_df = D_rows[['note', 'label']]

D = "\n".join(f" {row['note']} <output>{row['label']}</output>" for _, row in D_rows.iterrows())

D_rows.to_csv(f"results/{save_directory}/D_{run_name}_{args.data}.csv", index=False)

################################################################################################
##################################### Data Preprocessing #######################################
################################################################################################

################################################################################################
######################################## E[H[p(y|u,z,x)]] ######################################
################################################################################################

initial_row = data.sample(n=1)
print("Initial Row:\n", initial_row['note'])
data = data.drop(initial_row.index)

initial_selected_value = initial_row[selected_feature].values[0]
D_selected_values = D_rows[selected_feature].values

decimal_places = 1
previous_z_values = []

z_negative_Va_negative_kl_list = []
# Take new z values by sampling a normal distribution with mean from z and std from D values

num_random_z = int(num_random_z)

if num_random_z > num_modified_z:
    raise ValueError("Number of initial random z values cannot be greater than number of modified z values.")


seed_num = int(args.seed_num)

for j in range(num_x_values):
    x = x_row['note'].iloc[j]
    x_y = x_row['label'].iloc[j]
    print("x:", x)
    
    # Initialize p(y|x)
    avg_pyx_probs = {label: 0.0 for label in label_keys}
    
    # ----- Processing p(y|x) -----
    successful_seeds = 0
    for seed in range(seed_num):
    
        ## p(y|x)
        print(f"\np(y|x) Seed {seed + 1}/{seed_num}")

        try:
            prompt_pyx = short_prompt(note_label_df_to_icl_examples(D_note_label_df, seed), x)
            
            print("Prompt for p(y|x,D):")
            print(prompt_pyx)

            # Get the prediction and probabilities from the model
            pred_pyx, probs_pyx = get_response(prompt_pyx, label_keys, seed=seed)
            # print("pred_p(y|x):", pred_pyx)
            # print("probs_p(y|x):", probs_pyx)
            
            # Accumulate probabilities for puz
            for label, prob in probs_pyx.items():
                avg_pyx_probs[label] += prob
            
            successful_seeds += 1
        except:
            print(f"Seed {seed + 1} failed.")
            
    avg_pyx_probs = {label: prob / successful_seeds for label, prob in avg_pyx_probs.items()}
    # print("\nAveraged puzx probabilities:", avg_pyx_probs)
        
    for i in range(num_modified_z):
        if i < num_random_z:
            for _ in range(100):
                new_value = np.random.normal(np.mean(D_selected_values), 2*np.std(D_selected_values), 1)[0]
                new_value = round(new_value, decimal_places)
                if new_value not in previous_z_values:
                    previous_z_values.append(new_value)
                    break
                
            
            if i == 0:
                modified_row = initial_row.copy()
                modified_row[selected_feature] = new_value

                # Create a new DataFrame for z_data with note and selected feature
                dict_data = {"note": modified_row.apply(lambda row: parse_features_to_note(row, feature_columns), axis=1)}
                dict_data.update({col: modified_row[col] for col in feature_columns})

                z_data = pd.DataFrame(dict_data).reset_index(drop=True)
            else:
                modified_row = z_data.loc[i-1].copy()
                modified_row[selected_feature] = new_value
                modified_row["note"] = parse_features_to_note(modified_row, feature_columns)
                
                z_data.loc[i] = modified_row
                        
        if i >= num_random_z:
            # Bayesian Optimization for new z values
            new_value = new_candidate(
                z_values=previous_z_values,
                maximisation_quantity=z_negative_Va_negative_kl_list,
                lower_bound=x_row.iloc[j][selected_feature] -  3*np.std(D_selected_values),
                upper_bound=x_row.iloc[j][selected_feature] + 3*np.std(D_selected_values),
            )
            new_value = round(new_value, decimal_places)
            
            print(f"New Z Value: {new_value}")
            previous_z_values.append(new_value)
            
            modified_row = z_data.loc[i-1].copy()
            modified_row[selected_feature] = new_value
            
            modified_row['note'] = parse_features_to_note(modified_row, feature_columns)
                    
            z_data.loc[i] = modified_row
            
        row = z_data.iloc[i]
        
        z = row['note']

        avg_puz_probs = {label: 0.0 for label in label_keys}

        successful_seeds = 0
        for seed in range(seed_num):
            # Initialize avg_puz_probs
            
            ## p(u|z)
            print(f"\np(u|z) Seed {seed + 1}/{seed_num}")
            try:
                prompt_puz = short_prompt(note_label_df_to_icl_examples(D_note_label_df, seed), z)
                
                print("Prompt for p(u|z):")
                print(prompt_puz)

                # Get the prediction and probabilities from the model
                pred_puz, probs_puz = get_response(prompt_puz, label_keys, seed=seed)
                # print("pred_p(u|z):", pred_puz)
                # print("probs_p(u|z):", probs_puz)
                # Accumulate probabilities for puz
                for label, prob in probs_puz.items():
                    avg_puz_probs[label] += prob
                
                successful_seeds += 1
            except:
                print(f"Seed {seed + 1} failed.")
                
        # Calculate the average probabilities for puz and puzx
        avg_puz_probs = {label: prob / successful_seeds for label, prob in avg_puz_probs.items()}
        # print("\nAveraged puz probabilities:", avg_puz_probs)
        
        # Add averaged puz probabilities to the DataFrame
        for label, avg_prob in avg_puz_probs.items():
            z_data.at[i, f"p(u={label}|z)"] = avg_prob
            
        z_data.at[i, "H[p(u|z)]"] = calculate_entropy(avg_puz_probs)
        
        # print(f"\nProcessing Z Example {i + 1}")
        z = row['note']
        # print("Row Note:", z)

        # Initialize avg_pyxu_z_probs with distinct keys for each label
        avg_pyxu_z_probs = {
            f"p(y|x,u={outer_label},z)": {inner_label: 0.0 for inner_label in label_keys}
            for outer_label in label_keys
        }
       
        # Initialize p(y|x,z)
        avg_pyxz_probs = {label: 0.0 for label in label_keys}

        successful_seeds = {}
        # ----- Processing pyxu_z -----
        for outer_label in label_keys:
            print(f"\nProcessing p(y|x,u=_,z) for label '{outer_label}'")

            prompt_pyxuz = short_prompt(note_label_df_to_icl_examples(D_note_label_df, seed, z, outer_label), x)
            print("Prompt for p(y|x,u=_,z):")
            print(prompt_pyxuz)

            successful_seeds[f"p(y|x,u={outer_label},z)"] = 0
            for seed in range(seed_num):
                print(f"\np(y|x,u={outer_label},z) Seed {seed + 1}/{seed_num}")

                try:
                    pred_pyxuz, probs_pyxuz = get_response(prompt_pyxuz, label_keys, seed=seed)
                    # print(f"pred_p(y|x,u={outer_label},z):", pred_pyxuz)
                    # print(f"probs_p(y|x,u={outer_label},z):", probs_pyxuz)

                    # Accumulate probabilities for pyxu_z
                    for inner_label, prob in probs_pyxuz.items():
                        avg_pyxu_z_probs[f"p(y|x,u={outer_label},z)"][inner_label] += prob
                    successful_seeds[f"p(y|x,u={outer_label},z)"] += 1
                except:
                    print(f"Seed {seed + 1} failed.")
        # Calculate the average probabilities for pyxu_z
        for key, sub_dict in avg_pyxu_z_probs.items():
            avg_pyxu_z_probs[key] = {label: prob / successful_seeds[key] for label, prob in sub_dict.items()}

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
            avg_puz_prob = z_data.at[i, f"p(u={label}|z)"]
            avg_pyxuz_entropy = z_data.at[i, f"H[p(y|x,u={label},z)]"]
            expected_H += avg_puz_prob * avg_pyxuz_entropy
        z_data.at[i, "Va = E[H[p(y|x,u,z)]]"] = round(expected_H, 5)
        
        kl_pyx_pyxz = calculate_kl_divergence(avg_pyx_probs, avg_pyxz_probs)
        z_data.at[i, "kl_pyx_pyxz"] = round(kl_pyx_pyxz, 5)
        
        kl_pyxz_pyx = calculate_kl_divergence(avg_pyxz_probs, avg_pyx_probs)
        z_data.at[i, "kl_pyxz_pyx"] = round(kl_pyxz_pyx, 5)
        
        z_negative_Va_negative_kl_list.append(-expected_H - kl_pyx_pyxz)
                
        # ----- Final Output -----
        print(f"\nx value: {x} -> {label_name}: {x_y}")
        print("\nFinal z_data with Averaged Probabilities:")
        print(z_data.head())
        z_data["true_x"] = x_row.iloc[j][selected_feature]
        
        z_data.to_csv(f"results/{save_directory}/results_{run_name}_{args.data}_x{j + x_save_value}.csv", index=False)
        
    total_U = z_data["H[p(y|x)]"][0]
    print("\nTotal Uncertainty =", total_U)
    maximum_entropic_distance = total_U/20
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
        z_data["z_value_for_min_Va"] = False
    else:
        min_Va = min(valid_Va)
        z_data["within_threshold"] = z_data["Va = E[H[p(y|x,u,z)]]"].apply(lambda x: x in valid_Va)
        z_data["z_value_for_min_Va"] = z_data["Va = E[H[p(y|x,u,z)]]"].apply(lambda x: x == min_Va)
    # min_Va = z_data["Va = E[H[p(y|x,u,z)]]"].min()
    print("min Va = E[H[p(y|x,u,z)]] =", min_Va)
    max_Ve = round(total_U - min_Va, 5)
    print("max Ve = H[p(y|x,z)] - E[H[p(y|x,u,z)] =", max_Ve)
    z_data["min_Va"] = min_Va
    z_data["max_Ve"] = max_Ve
    z_data.to_csv(f"results/{save_directory}/results_{run_name}_{args.data}_x{j + x_save_value}.csv", index=False)


if __name__ == "__main__":
    label_keys = ['0', '1']
    seed = 123
    text_output, normalized_probs = get_first_token_probs(message, label_keys, seed)
    print(f"Text Output: {text_output}")
    print(f"Normalized Probabilities: {normalized_probs}")