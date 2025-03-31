import csv
import numpy as np
import math

def load_branches_as_dict(file_path):
    logprob_dict = {}
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            logprob_dict[row[0]] = row[1]  # First element as key, second as value
    return logprob_dict

def load_finished_branches_as_list(file_path):
    finished_branches_list = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            finished_branches_list.append(row[0])
    return finished_branches_list

# Example usage
branches_file = 'branches.csv'
finished_branches_file = 'finished_branches.csv'

logprobs = load_branches_as_dict(branches_file)
logsumexp_value = np.logaddexp.reduce(list(float(v) for v in logprobs.values()))
logprobs = {key: float(value)-logsumexp_value for key, value in logprobs.items()}
finished_branches = load_finished_branches_as_list(finished_branches_file)

probs = {key: math.exp(float(value)) for key, value in logprobs.items()}
print(sum(probs.values()))
print(logprobs)
print(probs)
#need to normalize
max_decimals = max(len(key.split(".")[1]) if "." in key else 0 for key in probs.keys())

def get_density(x):

    height = 0
    x_str = str(x)
    if "." not in x_str:
        bin = x_str
        if bin in probs:
            height += probs[bin]
    else:
        bin = str(int(x))
        if bin in probs:
            height += probs[bin]
        for i in range(max_decimals):
            bin = x_str.split(".")[0]+"."+x_str.split(".")[1][:i+1]
            #print(bin)
            if bin in probs:
                height += probs[bin]/10**-(i+1)
    return height

x = np.linspace(0, 10, 101)[:-1]
y = [get_density(xi) for xi in x]

sorted_branches = sorted(probs.items(), key=lambda item: item[0])
for key, value in sorted_branches:
    print(f"{key}: {value}")

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.savefig('adaptive.pdf', format='pdf')