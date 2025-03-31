import requests
import math
import itertools


#prompt = sys.argv[1]
import numpy as np
with open('prompt_123.txt', 'r') as file:
    prompt = file.read()

API_URL = "http://localhost:8000/v1/completions"  # Correct vLLM endpoint

num_decimals = 1 #assumes only 1 digit before decimal

digit_combos = [str(i).strip("()").replace(", ","") for i in (itertools.product(*[range(10) for _ in range(num_decimals+1)]))]
positive_numbers = [f"{combo[0]}.{combo[1:]}" for combo in digit_combos]
negative_numbers = [f"-{combo[0]}.{combo[1:]}" for combo in digit_combos]
all_numbers = positive_numbers + negative_numbers

stored_probs = {}
for 



def build(number_strings):
    stored_probs = {}
    stored_probs[""] = 1 #base case for recursion
    def return_prob(number_str):
        if number_str in stored_probs:
            return stored_probs[number_str]
        else:
            if number_str
            tot_prompt = prompt + number_str[]
            return 
    if number_str not in stored_probs:


for first_digit in [str(i) for i in range(10)]:
    print("First digit:", first_digit)
    prompt = prompt
    data = {
        "prompt": prompt,
        "max_tokens": 1,  # Only get the next token
        "logprobs": 20    # Get logprobs for top 50 tokens
    }
    response = requests.post(API_URL, json=data).json()
    
    if "choices" in response and len(response["choices"]) > 0:
        top_tokens = response["choices"][0]["logprobs"]["top_logprobs"][0]  # First token's probabilities
        
        if first_digit in top_tokens:
            logprob = top_tokens[first_digit]  # Get log probability
            probability = math.exp(logprob)  # Convert logprob to probability
            one_digit_probs[first_digit] = probability

for first_digit in [str(i) for i in range(10)]:
    print("First digit:", first_digit)
    for second_digit in [str(i) for i in range(10)]:
        prompt_1 = prompt + first_digit + "." 
        data = {
            "prompt": prompt_1,
            "max_tokens": 1,  # Only get the next token
            "logprobs": 20    # Get logprobs for top 50 tokens
        }
        response = requests.post(API_URL, json=data).json()
        
        if "choices" in response and len(response["choices"]) > 0:
            top_tokens = response["choices"][0]["logprobs"]["top_logprobs"][0]  # First token's probabilities
            
            if second_digit in top_tokens:
                logprob = top_tokens[second_digit]  # Get log probability
                probability = math.exp(logprob)  # Convert logprob to probability
                two_digit_probs[first_digit+"."+second_digit] = probability
            else:
                print(first_digit + "." + second_digit + "not there")
        prompt_2 = prompt + "-" + first_digit + "." #negative values
        data = {
            "prompt": prompt_2,
            "max_tokens": 1,  # Only get the next token
            "logprobs": 20    # Get logprobs for top 50 tokens
        }
        response = requests.post(API_URL, json=data).json()
        
        if "choices" in response and len(response["choices"]) > 0:
            top_tokens = response["choices"][0]["logprobs"]["top_logprobs"][0]  # First token's probabilities
            
            if second_digit in top_tokens:
                logprob = top_tokens[second_digit]  # Get log probability
                probability = math.exp(logprob)  # Convert logprob to probability
                two_digit_probs[first_digit+"."+second_digit] = probability
            else:
                print(first_digit + "." + second_digit + "not there")

print("One digit probabilities:", one_digit_probs)
print("Two digit probabilities:", two_digit_probs)

sum_probs = sum(two_digit_probs.values())
normalized_probs = {key: value/sum_probs for key, value in two_digit_probs.items()}
normalized_probs = {float(key): value for key, value in normalized_probs.items()}
import matplotlib.pyplot as plt
plt.plot(normalized_probs.keys(), normalized_probs.values())
plt.savefig("two_digit_probs.png")