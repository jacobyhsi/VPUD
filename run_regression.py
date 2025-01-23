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
