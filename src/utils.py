import math
import re
import ast
import numpy as np
import pandas as pd

# Helper Functions

def calculate_entropy(probs):
    # Calculate entropy using all probabilities in the dictionary
    probs = np.array(list(probs.values()))
    entropy = -np.sum(probs * np.log2(probs))
    return round(entropy, 5)

def kl_divergence(p, q):
    epsilon = 1e-12  # small constant to avoid log(0)
    kl = 0.0
    for label in p:
        p_val = p[label] + epsilon
        q_val = q[label] + epsilon
        kl += p_val * np.log(p_val / q_val)
    return kl

def extract(text):
    match = re.search(r'(.*?)</output>', text, re.DOTALL | re.IGNORECASE)
    if match:
        output_str = match.group(1).strip()
        output_dict = ast.literal_eval(output_str)
        return output_dict
    else:
        print("Could not find output tags in the response.")
        raise ValueError("Invalid response format.")
    
class TabularUtils:
    @staticmethod
    def pertube_z(data, z_row, z_samples=10):
        # Identify all feature columns to perturb (exclude 'note' and 'label')
        features_to_perturb = [col for col in data.columns if col not in ['note', 'label']]
        
        # Precompute unique values for each feature
        unique_vals = {feature: data[feature].dropna().unique() for feature in features_to_perturb}
        
        perturbed_rows = []
        
        for _ in range(z_samples):
            modified_row = z_row.copy()
            # Get the current note string (assuming z_row has one row)
            new_note = modified_row['note'].iloc[0]
            
            # Perturb all features by randomly assigning new values
            for feature in features_to_perturb:
                original_value = modified_row[feature].iloc[0]
                possible_vals = unique_vals[feature]

                # Exclude the original value if alternative values exist
                alt_vals = [val for val in possible_vals if val != original_value]
                if alt_vals:
                    new_value = np.random.choice(alt_vals)
                else:
                    new_value = original_value  # If no alternative values, keep it the same

                modified_row[feature] = new_value
                
                # Update the note string using regex substitution with a lambda for safety
                pattern = rf'({re.escape(feature)} = )(.*?)(\.|$)'
                new_note = re.sub(pattern, lambda m: f"{m.group(1)}{new_value}{m.group(3)}", new_note)
            
            modified_row['note'] = new_note
            perturbed_rows.append(modified_row)
        
        z_data = pd.concat(perturbed_rows, ignore_index=True)

        for i, row in z_data.iterrows():
            print(f"z_{i}: {row['note']}")

        return z_data
    
class RegressionUtils:
    pass
