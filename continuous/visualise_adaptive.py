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

def get_density(x, verbose=False):

    height = 0
    if verbose: print("calculating x=", x)
    x_str = str(x)
    if "." not in x_str:
        bin = x_str
        if bin in probs:
            if verbose:
                print(bin, probs[bin])
            height += probs[bin]
    else:
        while len(x_str.split(".")[1])<max_decimals:
            x_str += "0"
        bin = str(int(x))
        if bin in probs:
            height += probs[bin]
            if verbose:
                print(bin, probs[bin])
        for i in range(max_decimals):
            bin = x_str.split(".")[0]+"."+x_str.split(".")[1][:i+1]
            if verbose: print("bin ", bin, "decimal", i+1)
            if bin in probs:
                height += probs[bin]*10**(i+1)
                if verbose:
                    print(bin, probs[bin], probs[bin]/10**-(i+1))
    return height

x = np.linspace(0, 10, 501)[:-1]
y = [get_density(xi) for xi in x]

sorted_branches = sorted(probs.items(), key=lambda item: item[0])
for key, value in sorted_branches:
    print(f"{key}: {value}")


import matplotlib.pyplot as plt
"""xlines = np.linspace(0, 2, 21)[:-1]
plt.vlines(xlines, 0, 1, color='red', alpha=0.5)
print(xlines)
print(get_density(0.1, verbose=True))
print(get_density(0.2, verbose=True))
print(get_density(0.21, verbose=True))
print(get_density(0.3, verbose=True))"""
plt.plot(x, y)

if True:
    with open("samples_small.csv", newline='') as csvfile:
        reader = csv.reader(csvfile)
        print(next(reader))
        data = [float(row[0]) for row in reader]

    data = np.array([d for d in data if d < 10])
    min_x, max_x = 0,10
    num_bins = 100

    # Plot a histogram of the values
    binwidth = (max_x-min_x)/num_bins
    plt.hist(data, bins=np.linspace(min_x, max_x, num_bins+1)[:-1], weights=(np.zeros_like(data) + 1. / data.size)/binwidth, edgecolor='black')
    plt.title("Histogram of Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.savefig("pdf_and_samples_small.pdf", format='pdf')