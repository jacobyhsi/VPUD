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

if __name__ == "__main__":
    feature_dim = 2
    
    feature_means = np.array([1,4])
    
    feature_stds = np.array([4,3])
    
    bias = 1.0
    
    coefficients = np.array([3.,-3.])
    
    x = create_normal_features(100,feature_dim,feature_means, feature_stds)
        
    y = create_labels(x, bias, coefficients)
    
    dataset = create_pandas_dataset(x, y)
    
    print(dataset.head(20))
    
    dataset.to_csv(create_save_path("logistic_regression_1.csv", "logistic_regression_data"))
    
    with open(create_save_path("logistic_regression_1.json", "logistic_regression_info"), "w") as f:
        json.dump({"bias": bias, "coefficients": list(coefficients)}, f)
    
    