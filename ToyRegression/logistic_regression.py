import numpy as np
import scipy.stats
import scipy.special
import pandas as pd
import os
import json

#####
###
### CREATE DATASET
###
#####


###
# Create features
###

def create_normal_features(num_features: int, feature_dim: int, feature_means: np.ndarray, feature_stds: np.ndarray, round_dp: int=1, seed:int=0):
    if len(feature_means) != feature_dim:
        raise ValueError("feature_means must have length equal to feature_dim")
    if len(feature_stds) != feature_dim:
        raise ValueError("feature_stds must have length equal to feature_dim")
    
    x = scipy.stats.norm.rvs(size=(num_features, feature_dim), loc=feature_means, scale=feature_stds, random_state=seed)
    
    return np.round(x, round_dp)

def create_labels(features: np.ndarray, bias: float, coefficients: np.ndarray, seed:int=0):
    logits = np.matmul(features, coefficients) + bias
    
    probabilities = scipy.special.expit(logits)
        
    y = scipy.stats.bernoulli.rvs(probabilities, size=len(probabilities), random_state=seed)
    
    return y

def create_pandas_dataset(features: np.ndarray, y: np.ndarray):
    data_dict = {f"x{i+1}": features[:,i] for i in range(features.shape[1])}
    data_dict.update({"y": y})
    
    dataset = pd.DataFrame(data_dict).rename_axis("index")
    
    return dataset

def create_save_path(file_name: str, directory: str):
    abs_path = os.path.abspath(os.path.dirname(__file__))
    abs_path_dir = os.path.join(abs_path, directory)
    
    if not os.path.exists(abs_path_dir):
        os.makedirs(abs_path_dir)

    abs_path_file = os.path.join(abs_path_dir, file_name)
    
    return abs_path_file

def create_and_save_dataset(
    dataset_name: str,
    feature_dimensions: int = 1,
    feature_means: list[float] = [0.0],
    feature_stds: list[float] = [1.0],
    bias: float = 0.0,
    coefficients: list[float] = [0.0],
    dataset_size = 100,
    seed: int = 0
    ):
        
        x = create_normal_features(dataset_size, feature_dimensions, np.array(feature_means), np.array(feature_stds), seed=seed)
            
        y = create_labels(x, bias, np.array(coefficients), seed=seed)
        
        dataset = create_pandas_dataset(x, y)
        
        print(dataset.head(20))
        
        dataset.to_csv(create_save_path(f"{dataset_name}.csv", "logistic_regression_data"))
        
        print("Labels distribution")
        print(dataset["y"].value_counts())
        with open(create_save_path(f"{dataset_name}.json", "logistic_regression_info"), "w") as f:
            json.dump(
                {
                    "bias": bias,
                    "coefficients": coefficients,
                    "feature_means": feature_means,
                    "feature_stds": feature_stds,
                    "dataset_size": dataset_size,
                    "column_names": list(dataset.columns),
                    "label_name": "y",
                }, f)
            
    

if __name__ == "__main__":
    #TODO: Add as file args
    
    dataset_name = "logistic_regression_3"
    feature_dim = 1
    feature_means = [1.5]
    feature_stds = [3.]
    bias = - 0.5
    coefficients = [0.25]
    dataset_size = 500
    
    create_and_save_dataset(
        dataset_name,
        feature_dim,
        feature_means,
        feature_stds,
        bias,
        coefficients,
        dataset_size,
        )