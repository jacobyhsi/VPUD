from datasets import load_from_disk, concatenate_datasets
from datasets import load_dataset as load
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
import numpy as np
from src.utils import TabularUtils, ToyDataUtils
from ucimlrepo import fetch_ucirepo

def load_dataset(
    data_path,
    data_type='tabular',
    data_split_seed=123
    ) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    # Load Dataset
    dataset: Dataset = DATATYPE_TO_DATACLASS[data_type](
        data_path=data_path,
        data_split_seed=data_split_seed,
        )
    data = dataset.get_train_data()
    test = dataset.get_test_data()

    # Load Dataset Configs
    file_config = json.load(open(f'{data_path}/info.json'))
    label_map = file_config['map']
    label_keys = list(label_map)

    return data, test, label_keys

class Dataset():
    def __init__(self, data_path, data_split_seed: int = 123):
        data = self.load_data(data_path)
        # Split data
        self.data, self.test_data = train_test_split(data, test_size=0.8, random_state=data_split_seed)

    def get_train_data(self):
        return self.data

    def get_test_data(self):
        return self.test_data
    
    def load_data(self, data_path):
        raise NotImplementedError

class TabularDataset:
    def __init__(self, dataset_id: int, seed: int = 123):
        self.dataset_id = dataset_id
        self.seed = seed

    def load_data(self, config_path, label):
        # Load dataset using ucimlrepo
        uci_dataset = fetch_ucirepo(id=self.dataset_id)
        df = pd.concat([uci_dataset.data.features, uci_dataset.data.targets], axis=1)

        # Rename label column to "label" if needed
        if label != "label":
            df = df.rename(columns={label: "label"})

        # Encode label as integer if categorical
        if df["label"].dtype == object or pd.api.types.is_categorical_dtype(df["label"]):
            df["label"] = df["label"].astype("category").cat.codes

        # Determine note features
        note_features = [col for col in df.columns if col != "label"]

        # Create 'note' column
        df["note"] = df[note_features].apply(
            lambda row: TabularUtils.parse_features_to_note(row.to_dict(), feature_order=note_features),
            axis=1
        )

        # Load dataset config
        with open(f'{config_path}/info.json', 'r') as f:
            file_config = json.load(f)
        label_map = file_config['map']
        label_keys = list(label_map) 

        # Split the dataset
        value_counts = df["label"].value_counts()
        if (value_counts < 2).any():
            print("[Warning] Some classes have fewer than 2 samples. Disabling stratified split.")
            stratify_param = None
        else:
            stratify_param = df["label"]

        train_df, test_df = train_test_split(df, test_size=0.6, random_state=self.seed, stratify=stratify_param)

        # return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True), label_keys

    def generate_ood(self, ood_id):
        # Load ID
        id_data = fetch_ucirepo(id=self.dataset_id)
        id_df = id_data.data.features.copy()
        id_df_columns = id_df.columns.tolist()

        # classification
        if self.dataset_id == 53: # iris
            if ood_id == 381: # beijing
                ood_features = ['month', 'day', 'hour', 'DEWP', 'TEMP']
            elif ood_id == 296: # diabetes
                ood_features = ['time_in_hospital', 'num_procedures', 'num_medications', 'number_diagnoses']
        elif self.dataset_id == 58: # lenses
            if ood_id == 381:
                ood_features = ['hour', 'DEWP', 'TEMP']
            elif ood_id == 296:
                ood_features = ['time_in_hospital', 'num_procedures', 'num_medications']
        # regression
        elif self.dataset_id == 55: # estate
            if ood_id == 381: # beijing
                ood_features = ['month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES']
            elif ood_id == 296: # diabetes
                ood_features = ['time_in_hospital', 'num_procedures', 'num_medications', 'number_diagnoses', 'number_emergency']
        
        # Load OOD
        ood_data = fetch_ucirepo(id=ood_id)
        ood_df = pd.concat([ood_data.data.features, ood_data.data.targets], axis=1)

        ood_numeric = ood_df[ood_features].copy()

        # Match sample size
        num_samples = len(id_df)
        ood_sampled = ood_numeric.sample(n=num_samples, random_state=self.seed).reset_index(drop=True)
        ood_sampled = ood_sampled.apply(pd.to_numeric, errors='coerce')

        # Normalize to match ID stats
        id_mean = id_df[id_df_columns].mean()
        id_std = id_df[id_df_columns].std()
        ood_normalized = ((ood_sampled - id_mean.values) / id_std.values).round(1)
        ood_normalized.columns = id_df_columns

        # Create notes and label
        ood_normalized['note'] = ood_normalized.apply(
            lambda row: TabularUtils.parse_features_to_note(row.to_dict(), feature_order=id_df_columns),
            axis=1
        )
        ood_normalized['label'] = -1

        # Final OOD dataframe
        ood_final = ood_normalized[id_df_columns + ['label', 'note']]

        ood_train, ood_test = train_test_split(
                ood_final, test_size=0.8, random_state=self.seed
            )

        return ood_train.reset_index(drop=True), ood_test.reset_index(drop=True)
        
    
class ToyClassificationDataset(Dataset):
    def load_data(self, data_path: str):
        data = pd.read_csv(os.path.join(data_path, 'data.csv'), index_col=0)
                    
        data['label'] = data['label'].astype(int)
        
        feature_column = ToyDataUtils.get_feature_columns(data)
        
        data['note'] = data.apply(
            lambda row: ToyDataUtils.parse_features_to_note(row, feature_column),
            axis=1
        )
        
        return data
    
class ToyRegressionDataset(Dataset):
    def load_data(self, data_path: str):
        data = pd.read_csv(os.path.join(data_path, 'data.csv'), index_col=0)
                    
        data['label'] = data['label'].astype(float)
        
        feature_column = ToyDataUtils.get_feature_columns(data)
        
        data['note'] = data.apply(
            lambda row: ToyDataUtils.parse_features_to_note(row, feature_column),
            axis=1
        )
        
        return data   
        
DATATYPE_TO_DATACLASS: dict[str, Dataset] = {
    "tabular": TabularDataset,
    "toy_classification": ToyClassificationDataset,
    "toy_regression": ToyRegressionDataset,
}