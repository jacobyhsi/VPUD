import requests
import math
from collections import Counter
import numpy as np

def merge_logprobs(d1, d2):
    merged = {key: np.logaddexp(d1.get(key, -np.inf), d2.get(key, -np.inf)) for key in set(d1) | set(d2)}
    return merged

#prompt = sys.argv[1]
import numpy as np
import csv
with open('prompt_exp.txt', 'r') as file:
    original_prompt = file.read()

API_URL = "http://localhost:8000/v1/completions"  # Correct vLLM endpoint

branching_number = NUM_LOGPROBS = 15
stop_token = "<"#/output>"
max_queries = 15
include_unfinished = True   #interprets unfinished numbers with no decimal point as integers
threshold_prob = 0#1e-6
threshold_logprob = -np.inf#math.log(threshold_prob)
queries = 1
next_logprob = 0
#intial query
# Request data
data = {
    "prompt": original_prompt,
    "max_tokens": 1,  # Only get the next token
    "logprobs": NUM_LOGPROBS    # Get logprobs for top 50 tokens
}
# Send request
response = requests.post(API_URL, json=data).json()
top_tokens = response["choices"][0]["logprobs"]["top_logprobs"][0]  # First token's probabilities

# Convert logprobs to probabilities
branches = {
    token: llm_logprob
    for token, llm_logprob in top_tokens.items()
    if llm_logprob > threshold_logprob  # Only keep tokens above 0.1 probability
}
next_logprob = max(branches.values())
finished_branches = []
avail_branches = branches
while queries <= max_queries and next_logprob > threshold_logprob and len(avail_branches) > 0:
    print("iteration ", queries)
    sorted_branches = sorted(branches.items(), key=lambda x: x[1], reverse=True)
    for branch, logprob in sorted_branches:
        print(f"Branch: {branch}, Log Probability: {logprob}")
    if queries%10 == 0:
        print(f"Query {queries} of {max_queries}")
    queries += 1
    logprob = next_logprob
    next_digits = max(avail_branches, key=branches.get)
    print("Next digits", next_digits)

    # Request data
    data = {
        "prompt": original_prompt+next_digits,
        "max_tokens": 1,  # Only get the next token
        "logprobs": NUM_LOGPROBS    # Get logprobs for top 50 tokens
    }

    response = requests.post(API_URL, json=data).json()
    # Convert logprobs to probabilities
    top_tokens = response["choices"][0]["logprobs"]["top_logprobs"][0]  # First token's probabilities
    print(top_tokens)
    filtered_tokens = {
        token: llm_logprob + logprob  # Convert log probability to probability and combine with previous probability
        for token, llm_logprob in top_tokens.items()
        if logprob + llm_logprob > threshold_logprob  # Only keep tokens above 0.01 log probability
    }
    print("new branches")
    new_branches = {next_digits+t:filtered_tokens[t] for t in filtered_tokens}
    print(new_branches)
    
    
    
    for b in new_branches:
        if stop_token in b:
            finished_branches.append(b)
    
    del branches[next_digits]
    branches = merge_logprobs(branches, new_branches)
    avail_branches = {key : branches[key] for key in branches if key not in finished_branches}
    next_logprob = max(avail_branches.values())
    print(finished_branches)

print(branches)

#process the finished branches
final_branches = {}
for b in branches:
    #check is a valid number - i.e. either has a decimal point or stop token
    if stop_token in b or "." in b or include_unfinished:
        print("B", b)
        bin_label = b
        if stop_token in bin_label:       #if stop token is there, remove it
            bin_label = bin_label.split(stop_token)[0]
            print("stop token in b", bin_label)
        bin_label = bin_label.lstrip("0")
        if len(bin_label) == 0:
            bin_label = "0"
        if bin_label[0] == ".":
            bin_label = "0" + bin_label
        if bin_label[-1] == ".": #if decimal point is at the end, remove it
            bin_label = bin_label[:-1]
        try:
            assert(abs(float(bin_label)))<np.inf
            assert(float(bin_label) == float(bin_label)) #check if it is a valid number, not Nan
            if bin_label not in final_branches:
                final_branches[bin_label] = branches[b]
            else:
                final_branches[bin_label] = np.logaddexp(final_branches[bin_label], branches[b])
                #if number is already there, add probabilities
        except: #if it is not a valid number
            print("Invalid number", bin_label)


print("BRANCHES", branches)
print("FINAL BRANCHES", final_branches)



# Save branches to a CSV file
with open('branches.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Branch', 'Log Probability'])
    for branch, logprob in final_branches.items():
        writer.writerow([branch, logprob])

# Save finished_branches to a CSV file
with open('finished_branches.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Finished Branch'])
    for finished_branch in finished_branches:
        writer.writerow([finished_branch])
