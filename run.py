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

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument("--seed", default=123)
parser.add_argument("--data", default="income")
parser.add_argument("--feature", default="Education")
parser.add_argument("--shots", default=3)
parser.add_argument("--sets", default=10)
parser.add_argument("--llm", default="llama70b-nemo")
args = parser.parse_args()
seed = int(args.seed)
shots = int(args.shots)
sets = int(args.sets)

pd.set_option('display.max_columns', None)

def get_response(prompt, label_keys):
    # Add label_keys to the payload
    payload = {
        'prompt': prompt,
        'label_keys': label_keys
    }
    
    # Send POST request to the server
    response = requests.post('http://localhost:5000/predict', json=payload).json()
    
    # Extract response text and probabilities from the server's response
    response_text = response.get('response_text', "")
    probabilities = response.get('probabilities', [])
    
    return response_text, probabilities

################################################################################################
##################################### Data Preprocessing #######################################
################################################################################################
data_path = f'TabLLM/datasets_serialized/{args.data}'

f = open(f'{data_path}/info.json')
label_map = json.load(f)

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
def parse_note_to_features(note):
    features = {}
    for feature_str in note.strip('.').split('. '):
        key_value = feature_str.split(' = ')
        if len(key_value) == 2:
            key, value = key_value
            features[key.strip()] = value.strip()
    return features

# Apply the function to the 'note' column and expand the dictionaries into columns
tmp_combined_data_note = data['note']
note2features = data['note'].apply(parse_note_to_features).apply(pd.Series)

# # Concatenate the features DataFrame with the 'label' column
df_combined = pd.concat([note2features, data['label']], axis=1)
data = pd.concat([tmp_combined_data_note, df_combined], axis=1)
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

num_sets = sets
set_dict = {}
for _ in range(num_sets):
    ################################################################################################
    ########################################### P(u|z) #############################################
    ################################################################################################
    # icl_initial_row = data.sample(n=1, random_state=seed)
    icl_initial_row = data.sample(n=1)
    data = data.drop(icl_initial_row.index)
    feature_columns = note2features.columns
    print("Features:", feature_columns)
    
    # exit() here first to check whats the feature column names

    # selected_feature = np.random.choice(feature_columns)
    selected_feature = 'Education'
    print("Feature to vary:", selected_feature)

    modified_rows = []
    num_icl = shots
    unique_values = np.random.choice(data[selected_feature].dropna().unique(), size=num_icl, replace=False)

    for new_value in unique_values:
        print(f"{selected_feature} changed to: {new_value}")
        modified_row = icl_initial_row.copy()
        modified_row[selected_feature] = new_value
        
        pattern = rf'({selected_feature} = )(.*?)(\.|$)'
        modified_note = re.sub(
            pattern, 
            rf'\1{new_value}\3', 
            modified_row['note'].values[0]
        )
        modified_row['note'] = modified_note
        
        modified_rows.append(modified_row)

    icl_data = pd.concat(modified_rows, ignore_index=True)
    # print("Modified ICL rows:\n", icl_data)

    label_name = label_map['label']
    labels = label_map['map']
    label_keys = list(labels)

    for i, row in icl_data.iterrows():
        print(f"ICL Example {i+1}")
        icl_z = row['note']
        icl_y = row['label']
        
        prompt = \
        f"""Based on the sample provided below, predict the "{label_name}".
        "{label_name}" takes the form of the following: {labels}.

        {icl_z}

        Please output **ONLY** your predicted {label_name} label key from {label_keys} and enclose your output in <output> </output> tags."""
        
        # Please output **ONLY** the label key (e.g., 0 or 1) that corresponds to your prediction for "{label_name}".
        # Do not provide any other text or explanation.
        
        print(prompt)
        
        # print("label_keys:", label_keys)
        # print("Prediction:")
        pred, probs = get_response(prompt, label_keys)
        
        print("pred:", pred)
        print("probs:", probs)
        
        # Use regular expression to extract the text between <output> and </output>
        match = re.search(r'<output>\s*(.*?)\s*</output>', pred, re.DOTALL | re.IGNORECASE)

        if match:
            pred_label = match.group(1).strip()
            print(f"Extracted prediction: {pred_label}")
            print(f"True ICL label: {icl_y}")
        else:
            print("Could not find output tags in the response.")
            exit()
        
        icl_data.at[i, 'prediction'] = int(pred_label)
        print(icl_data['prediction'])
        print("=================================================================")
        
    # print(icl_data['note'])
    # print(icl_data['prediction'])
    ################################################################################################
    ########################################### P(u|z) #############################################
    ################################################################################################

    ################################################################################################
    ######################################### P(y|x,u,z) ###########################################
    ################################################################################################
    x_row = data.sample(n=1, random_state=seed)
    # x_row = data.sample(n=1)
    data = data.drop(x_row.index)

    x = x_row['note'].iloc[0]
    y = x_row['label'].iloc[0]

    in_context_examples = ""
    for i, row in icl_data.iterrows():
        note = row['note']
        prediction = int(row['prediction'])  # Convert prediction to an integer if it's a float
        in_context_examples += f"- {note} -> {label_name}: {prediction}\n"
        
    # print(in_context_examples)
    # print(x)
    print("################################ P(y|x,u,z) ################################")

    prompt = \
    f"""Based on the sample provided below, predict the "{label_name}". 
    "{label_name}" takes the form of the following: {labels}.

    {x}

    The following are some in-context examples that will help you make your prediction:

    {in_context_examples}

    Please output **ONLY** your predicted {label_name} label key from {label_keys} and enclose your output in <output> </output> tags."""

    print(prompt)

    print("Prediction:")
    pred, probs = get_response(prompt, label_keys)
    print("pred:", pred)
    print("probs:", probs)

    # Use regular expression to extract the text between <output> and </output>
    match = re.search(r'<output>\s*(.*?)\s*</output>', pred, re.DOTALL | re.IGNORECASE)

    if match:
        pred_label = match.group(1).strip()
        print(f"Extracted prediction: {pred_label}")
        print(f"True label: {y}")
    else:
        print("Could not find output tags in the response.")
        exit()
        
    entropy = calculate_entropy(probs)
    print("entropy:", entropy)
    
    data_rows.append({
        'unique_values': str(unique_values),  # Convert tuple to string for readability
        'prediction_probs': probs,  # Assuming filtered_probs is a dictionary
        'prediction': int(pred_label),
        'true_label': int(y),
        'entropy': entropy
    })

df_results = pd.DataFrame(data_rows).sort_values(by='entropy', ascending=True)
print("Results DataFrame:")
print(df_results)

print("Accuracy:", (df_results['prediction'] == df_results['true_label']).mean())

df_results.to_csv(f"results_{args.data}.csv", index=False)
################################################################################################
######################################### P(y|x,u,z) ###########################################
################################################################################################
