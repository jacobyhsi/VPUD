# Given the results csv file, this script will interpret the results

import pandas as pd
import numpy as np
import os
import json
import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument("--filepath", default="results.csv")
args = parser.parse_args()
file_path = args.filepath

# first row is the column names
z_data = pd.read_csv(file_path)

# convert the columns to the correct types
for col in z_data.columns:
    if col != "note":
        z_data[col] = z_data[col].astype(float)
        
        
print("\nFinal z_data with Averaged Probabilities:")
# print every 5 columns
for i in range(0, len(z_data.columns), 5):
    print(z_data.iloc[:, i:i+5].head())
    print("\n")

    
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
else:
    min_Va = min(valid_Va)
# min_Va = z_data["Va = E[H[p(y|x,u,z)]]"].min()
print("min Va = E[H[p(y|x,u,z)]] =", min_Va)
max_Ve = round(total_U - min_Va, 5)
print("max Ve = H[p(y|x,z)] - E[H[p(y|x,u,z)] =", max_Ve)
    
