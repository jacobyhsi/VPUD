import numpy as np
import scipy.stats
import scipy.special
import pandas as pd
import os
import json
from sklearn.datasets import make_moons

#####
###
### CREATE DATASET
###
#####


###
# Create features
###

class ToyClassificationData:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        
        self.create_save_path()

    def create_save_path(self):
        abs_path = os.path.abspath(os.getcwd())
        abs_path_dir = os.path.join(abs_path, "datasets_toy_classification", self.dataset_name)
        
        if not os.path.exists(abs_path_dir):
            os.makedirs(abs_path_dir)
        else:
            print(f"Directory {abs_path_dir} already exists")
        
        self.abs_path_dir = abs_path_dir
        
    @staticmethod
    def create_dataset(**kwargs) -> pd.DataFrame:
        raise NotImplementedError("create_dataset method must be implemented")
    
    def save_dataset(
        self,
        dataset_kwargs: dict,
        ):
            
            dataset = self.create_dataset(**dataset_kwargs)
            
            dataset.to_csv(os.path.join(self.abs_path_dir, "data.csv"))
            
            # unique labels
            labels = dataset["label"].unique()
            
            # string to int mapping
            label_map = {str(label): int(label) for label in labels}
            
            print("Labels distribution")
            print(dataset["label"].value_counts())
            with open(os.path.join(self.abs_path_dir, "info.json"), "w") as f:
                json.dump(
                    {
                        **dataset_kwargs,
                        "feature_columns": [column for column in list(dataset.columns) if column != "label"],
                        "label": "y",
                        "map": label_map,
                    }, f)
                
class LogisticRegressionData(ToyClassificationData):
    @staticmethod
    def create_normal_features(num_features: int, feature_dim: int, feature_means: np.ndarray, feature_stds: np.ndarray, round_dp: int=1, seed:int=0):
        if len(feature_means) != feature_dim:
            raise ValueError("feature_means must have length equal to feature_dim")
        if len(feature_stds) != feature_dim:
            raise ValueError("feature_stds must have length equal to feature_dim")
        
        x = scipy.stats.norm.rvs(size=(num_features, feature_dim), loc=feature_means, scale=feature_stds, random_state=seed)
        
        return np.round(x, round_dp)
    
    @staticmethod
    def create_labels(features: np.ndarray, bias: float, coefficients: np.ndarray, seed:int=0):
        logits = np.matmul(features, coefficients) + bias
        
        probabilities = scipy.special.expit(logits)
            
        y = scipy.stats.bernoulli.rvs(probabilities, size=len(probabilities), random_state=seed)
        
        return y

    @staticmethod
    def create_pandas_dataset(features: np.ndarray, y: np.ndarray):
        data_dict = {f"x{i+1}": features[:,i] for i in range(features.shape[1])}
        data_dict.update({"label": y})
        
        dataset = pd.DataFrame(data_dict).rename_axis("index")
        
        return dataset
    
    @staticmethod
    def create_dataset(
        feature_dimensions: int = 1,
        feature_means: list[float] = [0.0],
        feature_stds: list[float] = [1.0],
        bias: float = 0.0,
        coefficients: list[float] = [0.0],
        dataset_size = 100,
        seed: int = 0
        ):
            
            x = LogisticRegressionData.create_normal_features(dataset_size, feature_dimensions, np.array(feature_means), np.array(feature_stds), seed=seed)
                
            y = LogisticRegressionData.create_labels(x, bias, np.array(coefficients), seed=seed)
            
            dataset = LogisticRegressionData.create_pandas_dataset(x, y)
            
            print(dataset.head(20))
            
            print("Labels distribution")
            print(dataset["label"].value_counts())
            
            return dataset
        
class MoonsData(ToyClassificationData):
    @staticmethod
    def create_dataset(
        dataset_size: int = 100,
        noise: float = 0.1,
        round_dp: int = 2,
        seed: int = 0
        ):
            
            x, y = make_moons(n_samples=dataset_size, noise=noise, random_state=seed)
            
            dataset = pd.DataFrame({"x1": np.round(x[:,0], round_dp), "x2": np.round(x[:,1], round_dp), "label": y}).rename_axis("index")
            
            print(dataset.head(20))
            
            print("Labels distribution")
            print(dataset["label"].value_counts())
            
            return dataset
        
if __name__ == "__main__":
    moons_data = MoonsData("moons")
    
    moons_data.save_dataset(
        {
            "dataset_size": 100,
            "noise": 0.1,
            "seed": 0
        }
    )