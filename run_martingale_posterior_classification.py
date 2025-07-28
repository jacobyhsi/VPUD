import os
import re
import argparse
import codecs
import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm
from dataclasses import dataclass

from src.dataset import load_dataset
from src.bayesian_optimisation import new_candidate
from src.utils import ToyClassificationUtils, calculate_entropy, calculate_kl_divergence, calculate_discrete_variance
from src.prompt import ToyClassificationPrompt
from src.chat import chat

pd.set_option('display.max_columns', None)

parser = argparse.ArgumentParser(description='Running Toy Classification')

parser.add_argument("--dataset_name", default="logistic_regression_3")
parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B", type=str)
parser.add_argument("--model_port", default="8000", type=str)
parser.add_argument("--model_ip", default="localhost", type=str)
parser.add_argument("--model_temperature", default=1.0, type=float)
parser.add_argument("--is_local_client", default=1, type=int)

parser.add_argument("--numpy_seed", default=0, type=int)
parser.add_argument("--data_split_seed", default=0, type=int)
parser.add_argument("--icl_sample_seed", default=0, type=int)
parser.add_argument("--use_api_call_seed", default=0, type=int)
parser.add_argument("--fixed_permutation_seed", default=0, type=int)

parser.add_argument("--shots", default=3, type=int)
parser.add_argument("--num_permutations", default=5, type=int)
parser.add_argument("--permute_context", default=1, type=int)
parser.add_argument("--decimal_places", default=1, type=int)

parser.add_argument("--run_name", default="test")
parser.add_argument("--save_directory", default="other")
parser.add_argument("--experiment_type", default="default")
parser.add_argument("--x_save_value", default=0, type=int)
parser.add_argument("--num_api_calls_save_value", default=0, type=int)

parser.add_argument("--custom_prompt_text", default=None, type=str)
parser.add_argument("--verbose_output", default=0, type=int)
args = parser.parse_args()

@dataclass
class ToyClassificationExperimentConfig:
    dataset_name: str
    model_name: str
    model_port: str
    model_ip: str
    model_temperature: float
    is_local_client: int
    numpy_seed: int
    data_split_seed: int
    icl_sample_seed: int
    use_api_call_seed: int
    fixed_permutation_seed: int
    shots: int
    
    permute_context: int
    decimal_places: int
    run_name: int
    experiment_type: str
    save_directory: int
    x_save_value: int
    num_api_calls_save_value: int
    custom_prompt_text: Optional[str]
    verbose_output: int

class ToyClassificationExperiment:
    def __init__(self, config: ToyClassificationExperimentConfig):
        self.config = config
        
        np.random.seed(self.config.numpy_seed)

        self.prompter = ToyClassificationPrompt()

        self.data_preprocessing()
        
        self.use_api_call_seed = self.config.use_api_call_seed == 1
        self.num_api_calls = self.config.num_api_calls_save_value
        
        if self.config.custom_prompt_text is not None:
            self.config.custom_prompt_text = codecs.decode(self.config.custom_prompt_text, 'unicode_escape')

    def data_preprocessing(self):
        self.data_path = f'datasets_toy_classification/{self.config.dataset_name}'

        data, test_data, self.label_keys = load_dataset(
            data_path=self.data_path,
            data_type='toy_classification',
            data_split_seed=self.config.data_split_seed,
        )

        self.feature_columns = ToyClassificationUtils.get_feature_columns(data)

        print("Features:", self.feature_columns)
        
        D_rows = data.sample(n=self.config.shots, random_state=self.config.icl_sample_seed)

        self.D_note_label_df = D_rows[['note', 'label']]

        if not os.path.exists(f"results/{self.config.dataset_name}/{self.config.save_directory}"):
            os.makedirs(f"results/{self.config.dataset_name}/{self.config.save_directory}")
        D_rows.to_csv(f"results/{self.config.dataset_name}/{self.config.save_directory}/D_{self.config.run_name}.csv", index=False)
