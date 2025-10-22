import os
import argparse
import pandas as pd
import numpy as np
import torch
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Callable, Literal

from src.dataset import load_dataset
from src.utils import ToyDataUtils
from src.prompt import ToyClassificationPrompt
from src.chat import chat_response_only

from src.martingale_posterior import (
    parse_response,
    find_maximum_likelihood_estimates,
    find_maximum_likelihood_estimates_probit,
    find_maximum_likelihood_estimates_gaussian,
    find_maximum_likelihood_estimates_nn_classification,
)
parser = argparse.ArgumentParser(description='Running Toy Classification')

parser.add_argument("--dataset_name", default="logistic_regression_3")
parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B", type=str)
parser.add_argument("--model_port", default="8000", type=str)
parser.add_argument("--model_ip", default="localhost", type=str)
parser.add_argument("--model_temperature", default=1.0, type=float)
parser.add_argument("--is_local_client", default=1, type=int)

parser.add_argument("--D_size", default=15, type=int)

parser.add_argument("--permutation_sampling", default=False, type=bool)
parser.add_argument("--max_num_samples_per_call", default=5, type=int)
parser.add_argument("--max_llm_sample_size", default=50, type=int)
parser.add_argument("--max_tokens_per_call", default=500, type=int)

parser.add_argument("--num_posterior_samples", default=50, type=int)

parser.add_argument("--data_split_seed", default=0, type=int)
parser.add_argument("--icl_sample_seed", default=0, type=int)

parser.add_argument("--feature_interaction", default="linear", type=str, choices=["linear", "quadratic", "cubic", "nn"])
parser.add_argument("--likelihood_type", default="logistic", type=str, choices=["logistic", "probit", "gaussian"])

parser.add_argument("--num_epochs", default=10000, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--hidden_dim", default=20, type=int)
parser.add_argument("--num_hidden", default=2, type=int)

parser.add_argument("--save_dir", default="", type=str)
parser.add_argument("--save_name", default="results_martingale_posterior.csv", type=str)
parser.add_argument("--save_llm_samples", default=False, type=bool)

parser.add_argument("--sample_only", default=False, type=bool)
parser.add_argument("--reuse_llm_samples", default=False, type=bool)

LIKELIHOOD_TYPE_TO_MLE_FUNCTION = {
    'logistic': find_maximum_likelihood_estimates,
    'probit': find_maximum_likelihood_estimates_probit,
    'gaussian': find_maximum_likelihood_estimates_gaussian
}

DATASET_NAME_TO_DATASET_TYPE: dict[str, Literal['toy_classification', 'toy_regression']] = {
    'logistic_regression_3': 'toy_classification',
    'moons': 'toy_classification',
    'spiral': 'toy_classification',
    'linear_regression_1': 'toy_regression',
    'linear_noise_1': 'toy_regression',
    'linear_noise_2': 'toy_regression',
}
@dataclass
class MartingalePosteriorExperimentConfig:
    dataset_name: str
    model_name: str
    model_port: str
    model_ip: str
    model_temperature: float
    is_local_client: int
    
    save_dir: str
    save_name: str
    
    D_size: int
    
    permutation_sampling: bool = False
    max_num_samples_per_call: int = 5
    max_llm_sample_size: int = 50
    max_tokens_per_call: int = 500
    
    num_posterior_samples: int = 50
    
    data_split_seed: int = 0
    icl_sample_seed: int = 0

    feature_interaction: Literal['linear', 'quadratic', 'cubic', 'nn'] = 'linear'
    likelihood_type: Literal['logistic', 'probit'] = 'logistic'
    
    num_epochs: int = 10000
    lr: float = 0.0001
    num_hidden: int = 20
    hidden_dim: int = 2
    
        
    save_llm_samples: bool = False
    
    sample_only: bool = False
    
    reuse_llm_samples: bool = False

    
class MartingalePosteriorClassification:
    def __init__(self, config: MartingalePosteriorExperimentConfig) -> None:
        self.config = config
        
        self.dataset_type = DATASET_NAME_TO_DATASET_TYPE[self.config.dataset_name]
        
        self.prompter = ToyClassificationPrompt()
    
        self.maximum_likelihood_estimates_function = LIKELIHOOD_TYPE_TO_MLE_FUNCTION[config.likelihood_type]
        
        self.save_dir = f"results/martingale_posterior/{self.config.dataset_name}/{self.config.save_dir}"
        
        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)
        if self.config.save_llm_samples:
            if not os.path.exists(f"{self.save_dir}/llm_samples"):
                os.makedirs(f"{self.save_dir}/llm_samples")

        self._data_preprocessing()
        
    def _data_preprocessing(self):

        data_path = f'datasets_{self.dataset_type}/{self.config.dataset_name}'

        self.data, self.test_data, label_keys = load_dataset(
            data_path=data_path,
            data_type=self.dataset_type,
            data_split_seed=self.config.data_split_seed,
        )

        self.feature_columns = ToyDataUtils.get_feature_columns(self.data)

        D_rows = self.data.sample(n=self.config.D_size, random_state=self.config.icl_sample_seed)

        self.D_note_label_df = D_rows[['note', 'label']].copy()

        self.D_feature_label_df = D_rows[self.feature_columns + ['label']].copy()
        
        D_rows.to_csv(f"results/martingale_posterior/{self.config.dataset_name}/{self.config.save_dir}/D_{self.config.D_size}.csv", index=False)


    def one_pass_sampling(self, seed: int):
        prompt = self.prompter.get_general_prompt(
            D_df=self.D_note_label_df,
            query_note="",
            permutation_seed=seed,
            custom_prompt_text="{icl}\n"
        )
        
        response = chat_response_only(
            prompt,
            seed=seed,
            max_tokens=self.config.max_tokens_per_call,
            model=self.config.model_name,
            ip=self.config.model_ip,
            port=self.config.model_port,
        )
        
        response_df = parse_response(response, feature_columns=self.feature_columns, dataset_type=self.dataset_type)
        
        return response_df
    
    def permutation_sampling(self, seed: int):
        response_df_list = []
        note_label_df_list = []
        
        total_response_length = 0
        
        for i in range(self.config.max_llm_sample_size):
            prompt = self.prompter.get_general_prompt(
                D_df= pd.concat([self.D_note_label_df, *note_label_df_list], ignore_index=True),
                query_note="",
                permutation_seed=seed*self.config.max_num_samples_per_call + i,
                custom_prompt_text="{icl}\n"
            )
            
            # print(f"Prompt {i+1}: {prompt}")
                                
            response = chat_response_only(
                prompt,
                seed=seed*self.config.max_num_samples_per_call + i,
                max_tokens=self.config.max_tokens_per_call,
                model=self.config.model_name,
                ip=self.config.model_ip,
                port=self.config.model_port,
            )
            
            # print(f"Response {i+1}: {response}")
                    
            response_df = parse_response(response, feature_columns=self.feature_columns, dataset_type=self.dataset_type)
        
            # print(f"Response df {i+1}: {response_df}")
                        
            if response_df is not None and not response_df.empty:
                response_df_shortened = response_df.head(self.config.max_num_samples_per_call).copy()
                if self.dataset_type == 'toy_classification':
                    response_df_shortened['label'] = response_df_shortened['label'].astype(int)
                else:
                    response_df_shortened['label'] = response_df_shortened['label'].astype(float)
                response_df_list.append(response_df_shortened.copy())
        
                for i, row in response_df_shortened.iterrows():
                    note = ToyDataUtils.parse_features_to_note(row, feature_columns=self.feature_columns)
                    response_df_shortened.at[i, 'note'] = note
                note_label_df_list.append(response_df_shortened[['note', 'label']])
                
                total_response_length += len(response_df_shortened)
                if total_response_length >= self.config.max_llm_sample_size:
                    break
            else:
                break
            
        if len(response_df_list) == 0:
            return pd.DataFrame()
        return pd.concat(response_df_list, ignore_index=True)
    
    
    def obtain_sample_from_llm(self, seed: int):
        if self.config.permutation_sampling:
            response_df = self.permutation_sampling(seed)
        else:
            response_df = self.one_pass_sampling(seed)
        
        if self.config.save_llm_samples and response_df is not None:
            response_df.to_csv(f"results/martingale_posterior/{self.config.dataset_name}/{self.config.save_dir}/llm_samples/sample_{seed}.csv", index=False)
        
        return response_df
    
    def reuse_sample_from_llm(self, seed: int):
        if os.path.exists(self.save_dir + f"/llm_samples/sample_{seed}.csv"):
            try:
                response_df = pd.read_csv(self.save_dir + f"/llm_samples/sample_{seed}.csv")
                return response_df
            except Exception as e:
                print(f"Error reading LLM sample file for seed {seed}: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()
                

    def obtain_maximum_likelihood_estimates(self, seed: int):
        if self.config.reuse_llm_samples:
            response_df = self.reuse_sample_from_llm(seed)
        else:
            response_df = self.obtain_sample_from_llm(seed)
        
        combined_df = pd.concat([self.D_feature_label_df, response_df], ignore_index=True)
        
        X = torch.tensor(combined_df[self.feature_columns].values, dtype=torch.float32)
        
        if self.config.feature_interaction == 'quadratic':
            i, j = torch.triu_indices(X.shape[1], X.shape[1])

            cross_features = X[:, i] * X[:, j]            

            X = torch.cat([X, cross_features], dim=1)
        elif self.config.feature_interaction == 'cubic':
            a, b = torch.triu_indices(X.shape[1], X.shape[1])
            cross_features = X[:, a] * X[:, b]            
            
            triples = [(i,j,k) for i in range(X.shape[1]) for j in range(i,X.shape[1]) for k in range(j,X.shape[1])]
            i, j, k = zip(*triples)
            i, j, k = torch.tensor(i), torch.tensor(j), torch.tensor(k)
            cubic_features = X[:, i] * X[:, j] * X[:, k]
            
            X = torch.cat([X, cross_features, cubic_features], dim=1)
            
        y = torch.tensor(combined_df['label'].values, dtype=torch.float32)
        
        mle_estimates = self.maximum_likelihood_estimates_function(X, y)
        
        # print(f"Seed {seed}: b0 = {b0}, b1 = {b1}")

        return mle_estimates
    
    def obtain_maximum_likelihood_estimates_nn_classification(self, seed: int):
        if self.config.reuse_llm_samples:
            response_df = self.reuse_sample_from_llm(seed)
        else:
            response_df = self.obtain_sample_from_llm(seed)
            
        combined_df = pd.concat([self.D_feature_label_df, response_df], ignore_index=True)
        
        X = torch.tensor(combined_df[self.feature_columns].values, dtype=torch.float32)
        
        y = torch.tensor(combined_df['label'].values, dtype=torch.float32)
        
        model = find_maximum_likelihood_estimates_nn_classification(X, y, num_hidden=2, hidden_dim=20, seed=seed)
        
        return model
    
    def run_experiment(self):
        b0_list = []
        b1_list = []
        
        save_path = f"results/martingale_posterior/{self.config.dataset_name}/{self.config.save_dir}/{self.config.save_name}.csv"
        
        for seed in tqdm(
            range(self.config.num_posterior_samples),
            desc="Running Posterior Samples" if not self.config.sample_only else "Sampling Only"
        ):
            if self.config.sample_only:
                self.obtain_sample_from_llm(seed)
                continue
            b0, b1 = self.obtain_maximum_likelihood_estimates(seed)
            b0_list.append(b0)
            b1_list.append(b1)   
            
            results_df = pd.DataFrame({
                'b0': b0_list,
                'b1': b1_list
            })

            results_df = results_df.dropna()
            b0_list = results_df['b0'].tolist()
            b1_list = results_df['b1'].tolist()


            results_df.to_csv(save_path, index=False)
            
    def run_experiment_regression(self):
        b0_list = []
        b1_list = []
        sigma_list = []
        
        save_path = f"results/martingale_posterior/{self.config.dataset_name}/{self.config.save_dir}/{self.config.save_name}.csv"
        
        for seed in tqdm(
            range(self.config.num_posterior_samples),
            desc="Running Posterior Samples" if not self.config.sample_only else "Sampling Only"
        ):
            if self.config.sample_only:
                self.obtain_sample_from_llm(seed)
                continue
            b0, b1, sigma = self.obtain_maximum_likelihood_estimates(seed)
            b0_list.append(b0)
            b1_list.append(b1)
            sigma_list.append(sigma)

            results_df = pd.DataFrame({
                'b0': b0_list,
                'b1': b1_list,
                'sigma': sigma_list
            })

            results_df = results_df.dropna()
            b0_list = results_df['b0'].tolist()
            b1_list = results_df['b1'].tolist()
            sigma_list = results_df['sigma'].tolist()

            results_df.to_csv(save_path, index=False)
            
    def run_experiment_nn_classifcation(self):
        save_dir = f"results/martingale_posterior/{self.config.dataset_name}/{self.config.save_dir}/{self.config.save_name}"
        
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
            
        for seed in tqdm(
            range(self.config.num_posterior_samples),
            desc="Running Posterior Samples" if not self.config.sample_only else "Sampling Only"
        ):
            if self.config.sample_only:
                self.obtain_sample_from_llm(seed)
                continue
            
            model = self.obtain_maximum_likelihood_estimates_nn_classification(seed)
            
            torch.save(model.state_dict(), f"{save_dir}/seed_{seed}.pt")

if __name__ == "__main__":
    args = parser.parse_args()
    
    config = MartingalePosteriorExperimentConfig(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        model_port=args.model_port,
        model_ip=args.model_ip,
        model_temperature=args.model_temperature,
        is_local_client=args.is_local_client,
        
        save_dir=args.save_dir,
        save_name=args.save_name,
        
        D_size=args.D_size,
        
        permutation_sampling=args.permutation_sampling,
        max_num_samples_per_call=args.max_num_samples_per_call,
        max_llm_sample_size=args.max_llm_sample_size,
        max_tokens_per_call=args.max_tokens_per_call,
        
        num_posterior_samples=args.num_posterior_samples,
        
        data_split_seed=args.data_split_seed,
        icl_sample_seed=args.icl_sample_seed,
        
        feature_interaction=args.feature_interaction,
        likelihood_type=args.likelihood_type,
        
        num_epochs=args.num_epochs,
        lr=args.lr,
        num_hidden=args.num_hidden,
        hidden_dim=args.hidden_dim,

        save_llm_samples=args.save_llm_samples,
        
        sample_only=args.sample_only,
        reuse_llm_samples=args.reuse_llm_samples
    )
    
    experiment = MartingalePosteriorClassification(config)
    
    if experiment.dataset_type == 'toy_classification':
        if config.feature_interaction == 'nn':
            experiment.run_experiment_nn_classifcation()
        else:
            experiment.run_experiment()
    else:
        experiment.run_experiment_regression()

