import math
import re
import ast
import numpy as np
import pandas as pd
from itertools import product

# Helper Functions

def calculate_entropy(probs):
    # Calculate entropy using all probabilities in the dictionary
    probs = np.array(list(probs.values()))
    entropy = -np.sum(probs * np.log2(probs))
    return round(entropy, 5)

def calculate_kl_divergence(p, q):
    epsilon = 1e-12  # small constant to avoid log(0)
    kl = 0.0
    for label in p:
        p_val = p[label] + epsilon
        q_val = q[label] + epsilon
        kl += p_val * (np.log(p_val) - np.log(q_val))
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
    
    @staticmethod
    def parse_note_to_features(note, feature=None):
        features = {}
        for feature_str in note.strip('.').split('. '):
            key_value = feature_str.split(' = ')
            if len(key_value) == 2:
                key, value = key_value
                features[key.strip()] = value.strip()
        if feature:
            return features.get(feature.strip(), None)
        
        return features

    @staticmethod
    def parse_features_to_note(features, feature_order=None):
        if feature_order is None:
            feature_order = list(features.keys())
            
        note_parts = []
        for key in feature_order:
            if key in features and features[key] is not None:
                note_parts.append(f"{key} = {features[key]}")
        note = ". ".join(note_parts) + "."
        return note
    
class ToyClassificationUtils:
    @staticmethod
    def parse_features_to_note(row: pd.Series, feature_columns: list[str]):
        note_parts = []
        for feature in feature_columns:
            note_parts.append(f"{feature} = {row[feature]}")
        # join note with ;
        return "; ".join(note_parts)
    
    @staticmethod
    def get_feature_columns(data: pd.DataFrame):
        return [col for col in data.columns if col not in ['note', 'label']]
    
    @staticmethod
    def create_x_row_from_x_features(x_features: str, feature_columns: list[str], **kwargs):
        """ 
        Create x_row from given x_features.
        
        x_features is a string with the format "{'feature1': [f1_1, ..., f1_n], 'feature2': [f2_1, ..., f2_n], ...}"
        """
        x_features = ast.literal_eval(x_features)
        x_row = pd.DataFrame(x_features)
        x_row["label"] = 0
        x_row["note"] = x_row.apply(
            lambda row: ToyClassificationUtils.parse_features_to_note(row, feature_columns),
            axis=1,
        )
                
        return x_row

    @staticmethod
    def create_x_row_from_x_range(x_range: str, feature_columns: list[str], decimal_places: int, **kwargs):
        """
        Create x_row grid for a given x_range.
        
        x_range is a string with the format "start, end, step" for each feature.
        
        Example:
        x_range = "{'x1': [0, 10, 0.2],'x2': [1, 5, 1]}"
        """
        
        x_range = ast.literal_eval(x_range)
        x_row = pd.DataFrame()
        
        for feature, (start, end, step) in x_range.items():
            x_range[feature] = np.round(np.arange(float(start), float(end), float(step)), decimal_places)

        values = product(*x_range.values())
        x_row = pd.DataFrame(values, columns=x_range.keys())
        
        x_row["label"] = 0
        x_row["note"] = x_row.apply(
            lambda row: ToyClassificationUtils.parse_features_to_note(row, feature_columns),
            axis=1
        )
                
        return x_row

    @staticmethod
    def create_x_row_from_test_data(
        test_data: pd.DataFrame,
        num_x_samples: int,
        x_sample_seed: int,
        **kwargs,
    ):
        x_row = test_data.sample(n=num_x_samples, random_state=x_sample_seed)
        
        return x_row
    
    @staticmethod
    def create_x_row(method_name: str, **kwargs):
        if method_name == "x_features":
            return ToyClassificationUtils.create_x_row_from_x_features(**kwargs)
        elif method_name == "x_range":
            return ToyClassificationUtils.create_x_row_from_x_range(**kwargs)
        elif method_name == "sample":
            return ToyClassificationUtils.create_x_row_from_test_data(**kwargs)
        else:
            raise ValueError(f"Invalid method_name: {method_name}")
        
    @staticmethod
    def create_icl_data(num_shots: int, data: pd.DataFrame, icl_sample_seed: int):
        pass
    
    