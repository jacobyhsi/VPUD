import math
import re
import ast
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Optional
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

def calculate_kl_divergence_for_z_data(df: pd.DataFrame):
    kl_divergence_pyx_pyxz = []
    kl_divergence_pyxz_pyx = []

    PROB_LABELS = ["0", "1"]
    for index, row in df.iterrows():
        prob_y_x = {}
        prob_y_xz = {}
        for label in PROB_LABELS:
            prob_y_x[label] = row[f"p(y={label}|x,D)"]
            prob_y_xz[label] = row[f"p(y={label}|x,z,D)"]
        kl_divergence_pyx_pyxz.append(calculate_kl_divergence(prob_y_x, prob_y_xz))
        kl_divergence_pyxz_pyx.append(calculate_kl_divergence(prob_y_xz, prob_y_x))
        
    df["kl_pyx_pyxz"] = kl_divergence_pyx_pyxz
    df["kl_pyxz_pyx"] = kl_divergence_pyxz_pyx
    
    return df

def calculate_min_Va_by_KL_threshold(save_data: pd.DataFrame, threshold: float = 0.01, forward_kl = True):
    valid_Va = []
    total_U = save_data["H[p(y|x,D)]"][0]
    for i, row in save_data.iterrows():
        if forward_kl:
            if row["kl_pyx_pyxz"] <= threshold:
                valid_Va.append(row["Va"])
        else:
            if row["kl_pyxz_pyx"] <= threshold:
                valid_Va.append(row["Va"])
    if len(valid_Va) == 0:
        min_Va = np.nan
        save_data["within_threshold"] = False
        save_data["z_value_for_min_Va"] = False
    else:
        min_Va = min(valid_Va)
        save_data["within_threshold"] = save_data["Va"].apply(lambda x: x in valid_Va)
        save_data["z_value_for_min_Va"] = save_data["Va"].apply(lambda x: x == min_Va)
    save_data["min_Va"] = min_Va
    max_Ve = round(total_U - min_Va, 5)
    if min_Va == np.nan:
        save_data["max_Ve"] = np.nan
    else:
        save_data["max_Ve"] = max_Ve
    
    return save_data

def calculate_min_Va_by_KL_rank(save_data: pd.DataFrame, num_valid_Va: int = 5, forward_kl = True, upper_bound_by_total_U = False):
    if forward_kl:
        kl_values = save_data["kl_pyx_pyxz"]
    else:
        kl_values = save_data["kl_pyxz_pyx"]
    # min kl values
    min_kl_values = kl_values.nsmallest(num_valid_Va)
    save_data["within_threshold"] = kl_values.isin(min_kl_values)
    min_Va = save_data[save_data["within_threshold"]]["Va"].min()
    save_data["z_value_for_min_Va"] = save_data["Va"].apply(lambda x: x == min_Va)
    total_U = save_data["H[p(y|x,D)]"][0]
    if upper_bound_by_total_U:
        min_Va = min(min_Va, total_U)
    save_data["min_Va"] = min_Va
    max_Ve = round(total_U - min_Va, 5)
    save_data["max_Ve"] = max_Ve
    
    return save_data

def extract(text):
    match = re.search(r'(.*?)</output>', text, re.DOTALL | re.IGNORECASE)
    if match:
        output_str = match.group(1).strip()
        output_dict = ast.literal_eval(output_str)
        return output_dict
    else:
        print("Could not find output tags in the response.")
        raise ValueError("Invalid response format.")

def calculate_radius(row, center):
        return np.linalg.norm(row - center)
    
class TabularUtils:
    @staticmethod
    # def perturb_z(data, z_row, z_samples=10):
    #     # Identify all feature columns to perturb (exclude 'note' and 'label')
    #     features_to_perturb = [col for col in data.columns if col not in ['note', 'label']]
        
    #     # Precompute unique values for each feature
    #     unique_vals = {feature: data[feature].dropna().unique() for feature in features_to_perturb}
        
    #     perturbed_rows = []
        
    #     for _ in range(z_samples):
    #         modified_row = z_row.copy()
    #         # Get the current note string (assuming z_row has one row)
    #         new_note = modified_row['note'].iloc[0]
            
    #         # Perturb all features by randomly assigning new values
    #         for feature in features_to_perturb:
    #             original_value = modified_row[feature].iloc[0]
    #             possible_vals = unique_vals[feature]

    #             # Exclude the original value if alternative values exist
    #             alt_vals = [val for val in possible_vals if val != original_value]
    #             if alt_vals:
    #                 new_value = np.random.choice(alt_vals)
    #             else:
    #                 new_value = original_value  # If no alternative values, keep it the same

    #             modified_row[feature] = new_value
                
    #             # Update the note string using regex substitution with a lambda for safety
    #             pattern = rf'({re.escape(feature)} = )(.*?)(\.|$)'
    #             new_note = re.sub(pattern, lambda m: f"{m.group(1)}{new_value}{m.group(3)}", new_note)
            
    #         modified_row['note'] = new_note
    #         perturbed_rows.append(modified_row)
        
    #     z_data = pd.concat(perturbed_rows, ignore_index=True)

    #     # for i, row in z_data.iterrows():
    #         # print(f"z_{i}: {row['note']}")

    #     return z_data

    # def perturb_all_z(data, z_row, df_D, radius_list=[1.0,3.0,5.0], samples_per_radius=10):
    #     # Convert relevant columns explicitly to numeric
    #     for df in [data, df_D, z_row]:
    #         for col in df.columns:
    #             if col not in ['note', 'label']:
    #                 df[col] = pd.to_numeric(df[col], errors='coerce')

    #     numeric_cols = [col for col in data.columns if col not in ['note', 'label'] and data[col].notna().all()]
    #     features_to_perturb = numeric_cols

    #     center_point = df_D[features_to_perturb].mean().values

    #     feature_ranges = {
    #         feature: (data[feature].min(), data[feature].max())
    #         for feature in features_to_perturb
    #     }

    #     perturbed_rows = []

    #     for radius in radius_list:
    #         for _ in range(samples_per_radius):
    #             modified_row = z_row.copy()
    #             new_note = modified_row['note'].iloc[0]

    #             w = np.random.normal(0, 1, len(features_to_perturb))
    #             w_normalized = w / np.linalg.norm(w)

    #             perturbation = w_normalized * np.sqrt(radius)
    #             perturbed_point = center_point + perturbation

    #             for idx, feature in enumerate(features_to_perturb):
    #                 original_value = modified_row[feature].iloc[0]
    #                 min_val, max_val = feature_ranges[feature]
    #                 # Round final perturbed value to integer
    #                 new_value = np.clip(perturbed_point[idx], min_val, max_val)
    #                 new_value = int(round(new_value))

    #                 modified_row[feature] = new_value

    #                 pattern = rf'({re.escape(feature)} = )(.*?)(\.|$)'
    #                 new_note = re.sub(pattern, lambda m: f"{m.group(1)}{new_value}{m.group(3)}", new_note)

    #             modified_row['note'] = new_note
    #             perturbed_rows.append(modified_row)

    #     z_data = pd.concat(perturbed_rows, ignore_index=True)

    #     return z_data


    def perturb_z(data, x_row, z_samples, range_fraction=0.1): # perturbing z around the x
        # Identify all feature columns to perturb (exclude 'note' and 'label')
        features_to_perturb = [col for col in data.columns if col not in ['note', 'label']]
        
        # Compute min and max for numerical features
        feature_ranges = {feature: (data[feature].min(), data[feature].max()) 
                        for feature in features_to_perturb if np.issubdtype(data[feature].dtype, np.number)}
        feature_std = {feature: data[feature].std() for feature in features_to_perturb if np.issubdtype(data[feature].dtype, np.number)}
        perturbed_rows = []

        for _ in range(z_samples):
            modified_row = x_row.copy()
            new_note = modified_row['note']

            for feature in features_to_perturb:
                original_value = modified_row[feature]
                x_value = x_row[feature]

                if feature in feature_ranges:
                    # Numerical feature: Perturb within a fraction of its range around x
                    # min_val, max_val = feature_ranges[feature]
                    # delta = range_fraction * (max_val - min_val)
                    
                    # lower_bound = max(min_val, x_value - delta)
                    # upper_bound = min(max_val, x_value + delta)
                    
                    # new_value = np.random.uniform(lower_bound, upper_bound)
                    new_value = np.random.normal(loc=x_value, scale=feature_std[feature] * range_fraction)
                    new_value = round(new_value, 1)  # Round to avoid excessive decimals
                else:
                    # Categorical feature: Sample similar values
                    possible_vals = data[feature].dropna().unique()
                    alt_vals = [val for val in possible_vals if val != x_value]  # Exclude x value
                    
                    if alt_vals:
                        new_value = np.random.choice(alt_vals)
                    else:
                        new_value = original_value  # Keep the same if no alternatives

                modified_row[feature] = new_value
                
                # Update the note string using regex substitution
                note_dict = {f: modified_row[f] for f in features_to_perturb}
                new_note = ", ".join([f"{k} = {v}" for k, v in note_dict.items()])

            modified_row['note'] = new_note
            perturbed_rows.append(modified_row)

        z_data = pd.concat(perturbed_rows, ignore_index=True, axis=1).T
            
        return z_data

    # def perturb_all_x(data, x_row, df_D, radius_list=[1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0], samples_per_radius=10, max_attempts=1000):
    #     # Convert relevant columns explicitly to numeric
    #     for df in [data, df_D, x_row]:
    #         for col in df.columns:
    #             if col not in ['note', 'label']:
    #                 df[col] = pd.to_numeric(df[col], errors='coerce')

    #     numeric_cols = [col for col in data.columns if col not in ['note', 'label'] and data[col].notna().all()]
    #     features_to_perturb = numeric_cols
    #     num_features = len(features_to_perturb)

    #     center_point = df_D[features_to_perturb].mean().values
    #     df_D['radius'] = df_D[features_to_perturb].apply(lambda row: calculate_radius(row.values, center_point), axis=1)

    #     df_D.to_csv(f"df_D_all.csv", index=False)

    #     feature_ranges = {
    #         feature: (data[feature].min(), data[feature].max())
    #         for feature in features_to_perturb
    #     }

    #     perturbed_rows = []

    #     for radius in radius_list:
    #         samples_collected = 0
    #         attempts = 0

    #         while samples_collected < samples_per_radius and attempts < max_attempts:
    #             attempts += 1
    #             w = np.random.normal(0, 1, num_features)
    #             w_normalized = w / np.linalg.norm(w)
    #             perturbation = w_normalized * radius
    #             perturbed_point = center_point + perturbation

    #             # Check if all features are within valid ranges
    #             in_range = True
    #             for idx, feature in enumerate(features_to_perturb):
    #                 min_val, max_val = feature_ranges[feature]
    #                 if not (min_val <= perturbed_point[idx] <= max_val):
    #                     in_range = False
    #                     break

    #             if not in_range:
    #                 continue  # Reject and resample

    #             # If in range, apply the perturbation
    #             modified_row = x_row.copy()
    #             new_note = modified_row['note'].iloc[0]

    #             for idx, feature in enumerate(features_to_perturb):
    #                 new_value = int(round(perturbed_point[idx]))
    #                 modified_row[feature] = new_value

    #                 # Update note using regex substitution
    #                 pattern = rf'({re.escape(feature)} = )(.*?)(\.|$)'
    #                 new_note = re.sub(pattern, lambda m: f"{m.group(1)}{new_value}{m.group(3)}", new_note)

    #             modified_row['note'] = new_note
    #             modified_row['radius'] = radius
    #             perturbed_rows.append(modified_row)
    #             samples_collected += 1

    #         if samples_collected < samples_per_radius:
    #             print(f"[Warning] Only collected {samples_collected}/{samples_per_radius} samples for radius {radius}. May be too large.")

    #     x_data = pd.concat(perturbed_rows, ignore_index=True)
    #     return x_data

    # def perturb_all_x(data, x_row, df_D, radius_list=[1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0], samples_per_radius=10, max_attempts=1000): # 6/4/2025
    #     # Convert relevant columns explicitly to numeric
    #     for df in [data, df_D, x_row]:
    #         for col in df.columns:
    #             if col not in ['note', 'label']:
    #                 df[col] = pd.to_numeric(df[col], errors='coerce')

    #     # Identify numeric features to perturb
    #     numeric_cols = [col for col in data.columns if col not in ['note', 'label'] and data[col].notna().all()]
    #     features_to_perturb = numeric_cols
    #     num_features = len(features_to_perturb)

    #     # Store ICL values for nearest-neighbor computation
    #     df_D_values = df_D[features_to_perturb].values

    #     # Save df_D for logging/debugging
    #     df_D.to_csv("df_D_all.csv", index=False)

    #     # Define valid range for each feature
    #     feature_ranges = {
    #         feature: (data[feature].min(), data[feature].max())
    #         for feature in features_to_perturb
    #     }

    #     perturbed_rows = []

    #     # Use radius_list to scale how far from the ICL centroid we explore
    #     center_point = df_D[features_to_perturb].mean().values

    #     for radius in radius_list:
    #         samples_collected = 0
    #         attempts = 0

    #         while samples_collected < samples_per_radius and attempts < max_attempts:
    #             attempts += 1

    #             # Random direction & perturbation
    #             w = np.random.normal(0, 1, num_features)
    #             w_normalized = w / np.linalg.norm(w)
    #             perturbation = w_normalized * radius
    #             perturbed_point = center_point + perturbation

    #             # Validate perturbed point is within bounds
    #             in_range = all(
    #                 feature_ranges[features_to_perturb[i]][0] <= perturbed_point[i] <= feature_ranges[features_to_perturb[i]][1]
    #                 for i in range(num_features)
    #             )
    #             if not in_range:
    #                 continue  # Resample

    #             # Apply perturbation to copy of x_row
    #             modified_row = x_row.copy()
    #             new_note = modified_row['note'].iloc[0]

    #             for idx, feature in enumerate(features_to_perturb):
    #                 new_value = int(round(perturbed_point[idx]))
    #                 modified_row[feature] = new_value

    #                 # Update note text
    #                 pattern = rf'({re.escape(feature)} = )(.*?)(\.|$)'
    #                 new_note = re.sub(pattern, lambda m: f"{m.group(1)}{new_value}{m.group(3)}", new_note)

    #             modified_row['note'] = new_note

    #             # Compute distance to nearest ICL example
    #             distances = np.linalg.norm(df_D_values - perturbed_point, axis=1)
    #             min_distance = np.min(distances)
    #             modified_row['radius'] = min_distance  # You can rename this field if preferred

    #             perturbed_rows.append(modified_row)
    #             samples_collected += 1

    #         if samples_collected < samples_per_radius:
    #             print(f"[Warning] Only collected {samples_collected}/{samples_per_radius} samples for radius {radius}. May be too large.")

    #     x_data = pd.concat(perturbed_rows, ignore_index=True)
    #     return x_data

    def perturb_all_x(
        data, x_row, df_D,
        radius_list=[1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        samples_per_radius=10,
        max_attempts=1000
    ):
        # Convert relevant columns explicitly to numeric
        for df in [data, df_D, x_row]:
            for col in df.columns:
                if col not in ['note', 'label']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # Identify numeric features to perturb
        numeric_cols = [col for col in data.columns if col not in ['note', 'label'] and data[col].notna().all()]
        features_to_perturb = numeric_cols
        num_features = len(features_to_perturb)

        # Store ICL feature values
        df_D_values = df_D[features_to_perturb].values
        df_D.to_csv("df_D_all.csv", index=False)

        # Define feature value ranges
        feature_ranges = {
            feature: (data[feature].min(), data[feature].max())
            for feature in features_to_perturb
        }

        # Base point to perturb around (x_row)
        x_features = x_row.iloc[0][features_to_perturb].values.astype(float)

        perturbed_rows = []

        for radius in radius_list:
            samples_collected = 0
            attempts = 0

            while samples_collected < samples_per_radius and attempts < max_attempts:
                attempts += 1

                # Gaussian perturbation around x
                noise = np.random.normal(loc=0, scale=1.0, size=num_features)
                noise = noise / np.linalg.norm(noise) * radius
                perturbed_point = x_features + noise

                # Validate ranges
                in_range = all(
                    feature_ranges[features_to_perturb[i]][0] <= perturbed_point[i] <= feature_ranges[features_to_perturb[i]][1]
                    for i in range(num_features)
                )
                if not in_range:
                    continue

                modified_row = x_row.copy()
                new_note = modified_row['note'].iloc[0]

                for i, feature in enumerate(features_to_perturb):
                    new_value = int(round(perturbed_point[i]))
                    modified_row[feature] = new_value

                    pattern = rf'({re.escape(feature)} = )(.*?)(\.|$)'
                    new_note = re.sub(pattern, lambda m: f"{m.group(1)}{new_value}{m.group(3)}", new_note)

                modified_row['note'] = new_note

                # Compute distance to nearest ICL point
                distances = np.linalg.norm(df_D_values - perturbed_point, axis=1)
                min_distance = np.min(distances)

                # ✅ Overwrite radius with nearest ICL distance
                modified_row['radius'] = min_distance

                perturbed_rows.append(modified_row)
                samples_collected += 1

            if samples_collected < samples_per_radius:
                print(f"[Warning] Only collected {samples_collected}/{samples_per_radius} samples for radius {radius}.")

        x_data = pd.concat(perturbed_rows, ignore_index=True)
        return x_data


    def perturb_x(data, x_row, feature_to_perturb): # perturb within min and max range
        # Ensure the feature exists in the dataset
        if feature_to_perturb not in data.columns:
            print(f"Feature '{feature_to_perturb}' not found in the dataset.")
            print(f"Please reselect a feature from the following list: {data.columns}")
            raise ValueError("Invalid feature selection.")

        # Get the min and max of the feature
        min_val = int(data[feature_to_perturb].min())
        max_val = int(data[feature_to_perturb].max())

        # Define step size as 5% of the max value
        step_size = int(0.02 * max_val)

        # Generate perturbations from min to max in steps of `step_size`
        perturbed_values = np.arange(min_val, max_val + step_size, step_size)

        perturbed_rows = []

        for new_value in perturbed_values:
            modified_row = x_row.copy()

            # Update the feature with the new perturbed value
            modified_row[feature_to_perturb] = new_value

            # Update the note using regex substitution
            original_note = modified_row['note'].iloc[0]
            pattern = rf'({re.escape(feature_to_perturb)} = )(.*?)(\.|$)'
            new_note = re.sub(pattern, lambda m: f"{m.group(1)}{new_value}{m.group(3)}", original_note)

            modified_row['note'] = new_note
            perturbed_rows.append(modified_row)

        # Concatenate perturbed rows into a DataFrame
        x_data = pd.concat(perturbed_rows, ignore_index=True)

        return x_data


    # def perturb_x(data, x_row, feature_to_perturb): # original perturb x where we pick 1 feature to perturb
    #     # Get all unique values for the specified feature
        
    #     if feature_to_perturb not in data.columns:
    #         print(f"Feature '{feature_to_perturb}' not found in the dataset.")
    #         print(f"Please reselect a feature from the following list: {data.columns}")
    #         raise ValueError("Invalid feature selection.")

    #     possible_vals = data[feature_to_perturb].dropna().unique()
        
    #     perturbed_rows = []
        
    #     for new_value in possible_vals:
    #         modified_row = x_row.copy()
            
    #         # Update the feature with the new value
    #         modified_row[feature_to_perturb] = new_value
            
    #         # Update the note using regex substitution
    #         original_note = modified_row['note'].iloc[0]
    #         pattern = rf'({re.escape(feature_to_perturb)} = )(.*?)(\.|$)'
    #         new_note = re.sub(pattern, lambda m: f"{m.group(1)}{new_value}{m.group(3)}", original_note)

    #         modified_row['note'] = new_note
    #         perturbed_rows.append(modified_row)
        
    #     x_data = pd.concat(perturbed_rows, ignore_index=True)

    #     # for i, row in x_data.iterrows():
    #     #     print(f"x_{i}: {row['note']}")

    #     return x_data
    
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
        note = ", ".join(note_parts) + "."
        return note
    
    @staticmethod
    def build_note(row):
        return ", ".join([f"{k} = {v}" for k, v in row.items() if k != "label" and k != "note"])
    
class ToyDataUtils:
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
            lambda row: ToyDataUtils.parse_features_to_note(row, feature_columns),
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
            lambda row: ToyDataUtils.parse_features_to_note(row, feature_columns),
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
            return ToyDataUtils.create_x_row_from_x_features(**kwargs)
        elif method_name == "x_range":
            return ToyDataUtils.create_x_row_from_x_range(**kwargs)
        elif method_name == "sample":
            return ToyDataUtils.create_x_row_from_test_data(**kwargs)
        else:
            raise ValueError(f"Invalid method_name: {method_name}")
        
    @staticmethod
    def create_icl_data(num_shots: int, data: pd.DataFrame, icl_sample_seed: int):
        pass

class ToyClassificationUtils(ToyDataUtils):
    pass

class GaussianDistribution:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        
    @property
    def entropy(self):
        return 0.5 * np.log(2 * np.pi * self.std**2) + 0.5
    
    def pdf(self, x: float):
        return stats.norm.pdf(x, loc=self.mean, scale=self.std)
    
    def sample(self, size: Optional[int] = None):
        return np.random.normal(loc=self.mean, scale=self.std, size=size)

class ToyRegressionUtils(ToyDataUtils):
    @staticmethod
    def gaussian_from_samples(data: list[float], num_outlier_pairs_to_remove: int = 0):
        if num_outlier_pairs_to_remove > 0:
            data = sorted(data)[num_outlier_pairs_to_remove:-num_outlier_pairs_to_remove]
        mean = np.mean(data)
        std = np.std(data) * np.sqrt(len(data) / (len(data) - 1))
        return GaussianDistribution(mean, std)
    
    @staticmethod
    def calculate_kl_divergence(p: GaussianDistribution, q: GaussianDistribution):
        kl = np.log(q.std / p.std) + (p.std**2 + (p.mean - q.mean)**2) / (2 * q.std**2) - 0.5
        return kl
    
    