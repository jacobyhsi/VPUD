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

bins_a = []
bins_b = []
p = []
for branch in probs:
    if "." not in branch:
        precision = 1
    else:
        num_decimals = len(branch.split(".")[1])
        precision = 10**(-num_decimals)
    a, b = float(branch), float(branch)+precision
    bins_a.append(a)
    bins_b.append(b)
    p.append(probs[branch])

bins_a = np.array(bins_a)
bins_b = np.array(bins_b)
p = np.array(p)

print(bins_a)
print(bins_b)
print(p)

midpoints = (bins_a+bins_b)/2
mean = (midpoints*p).sum()

vars = (bins_b-bins_a)**2/12
total_var = (p*vars + p*midpoints**2).sum()-mean**2

print(mean)
print(total_var)

