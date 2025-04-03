import pandas as pd
import csv

import matplotlib.pyplot as plt

# Load the data from the CSV file
# Load the data from the CSV file without pandas
with open("samples_small.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    print(next(reader))
    data = [float(row[0]) for row in reader]

data = [d for d in data if d < 10]


# Plot a histogram of the values
plt.hist(data, bins=50, edgecolor='black')
plt.title("Histogram of Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.savefig("samples_small.pdf", format='pdf')