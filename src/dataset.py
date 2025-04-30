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

class QADataset:
    """
    A minimal loader for yes/no QA datasets.
    Currently supports:
        • "boolq"  (Boolean Questions, Google Search)
    Extendable by adding new branches in `_load_boolq`-style.
    """

    def __init__(self, name: str = "boolq", seed: int = 123):
        self.name = name.lower()
        self.seed = seed

    # ──────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────
    def load_data(self, test_size: float = 0.2):
        """
        Returns:
            train_df,  test_df,  label_keys
        where
            • label_keys == ["no", "yes"]   (0 / 1)
            • each DataFrame has columns:  note, label, question, passage
        """
        if self.name == "boolq":
            return self._load_boolq(test_size)

        raise ValueError(f"Dataset '{self.name}' not supported yet.")

    # ──────────────────────────────────────────────────────────
    # internal helpers
    # ──────────────────────────────────────────────────────────
    def _load_boolq(self, test_size: float):
        """
        BoolQ has two predefined splits (train / validation).
        We merge them and re-split with `test_size` so callers
        always get a fresh random split that respects class balance.
        """
        ds = load("boolq")                          # HF dataset object
        df_train = pd.DataFrame(ds["train"])
        df_val = pd.DataFrame(ds["validation"])
        df_all = pd.concat([df_train, df_val], ignore_index=True)

        df_all = df_all.sample(n=500, random_state=self.seed).reset_index(drop=True)

        # rename + encode label
        df_all = df_all.rename(columns={"answer": "label"})
        df_all["label"] = df_all["label"].astype(int)    # False→0, True→1

        # build note
        df_all["note"] = (
            "Question: " + df_all["question"].str.lower() + " Context: " + df_all["passage"].str.lower()
        )

        # keep consistent order
        df_all = df_all[["note", "label"]]

        # stratified split ensures 50/50 yes-no distribution in each split
        train_df, test_df = train_test_split(
            df_all,
            test_size=test_size,
            random_state=self.seed,
            stratify=df_all["label"],
        )

        label_keys = ["no", "yes"]
        return (
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
            label_keys,
        )

    def generate_ood(self, test_size: float = 0.2):
        """
        Out-of-Domain data using PubMedQA (pqa_labeled config).

        Returns
        -------
        ood_train, ood_test : pandas.DataFrame
            Columns: note, label, question, passage
            • label is fixed to -1 (unknown) for all OOD rows.
        """
        if self.name != "boolq":
            raise RuntimeError("generate_ood is implemented only when ID == BoolQ")

        # 1) we used 500 BoolQ examples → match that size here
        id_samples = 500

        # 2) load PubMedQA -- only 'train' split exists for pqa_labeled
        pqa = load("pubmed_qa", "pqa_labeled")
        pqa_df = pd.DataFrame(pqa["train"])

        # 3) keep strict yes/no examples
        pqa_df = pqa_df[pqa_df["final_decision"].isin(["yes", "no"])]

        # 4) choose a passage (long answer preferred, else first context sentence)
        def take_passage(row):
            la = row.get("long_answer", "")
            if isinstance(la, str) and la.strip():
                return la.strip()
            ctx = row.get("context", [])
            return ctx[0].strip() if isinstance(ctx, list) and ctx else ""

        pqa_df["passage"] = pqa_df.apply(take_passage, axis=1)

        # 5) sample to match ID size
        pqa_df = pqa_df.sample(
            n=min(id_samples, len(pqa_df)), random_state=self.seed
        ).reset_index(drop=True)

        # 6) construct BoolQ-style columns
        pqa_df["question"] = pqa_df["question"].str.strip()
        pqa_df["note"] = (
            "Question: " + pqa_df["question"].str.lower() + " Context: " + pqa_df["passage"].str.lower()
        )
        pqa_df["label"] = -1  # hide true label

        # 7) final ordering identical to _load_boolq
        ood_df = pqa_df[["note", "label"]]

        # 8) split
        ood_train, ood_test = train_test_split(
            ood_df, test_size=test_size, random_state=self.seed, stratify=None
        )
        return ood_train.reset_index(drop=True), ood_test.reset_index(drop=True)

class TabularDataset:
    def __init__(self, dataset_id: int, seed: int = 123):
        self.dataset_id = dataset_id
        self.seed = seed

    def load_data(self, config_path, label, test_size=0.8):
        # Load dataset using ucimlrepo
        uci_dataset = fetch_ucirepo(id=self.dataset_id)
        df = pd.concat([uci_dataset.data.features, uci_dataset.data.targets], axis=1)

        if self.dataset_id == 257: # user
            df['UNS'] = df['UNS'].replace('very_low', 'Very Low')

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

        train_df, test_df = train_test_split(df, test_size=test_size, random_state=self.seed, stratify=stratify_param)

        # return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True), label_keys

    def generate_ood(self, ood_id, test_size=0.8):
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
        elif self.dataset_id == 763: # mines
            if ood_id == 381:
                ood_features = ['hour', 'DEWP', 'TEMP']
            elif ood_id == 296:
                ood_features = ['time_in_hospital', 'num_procedures', 'num_medications']
        elif self.dataset_id == 257: # breast
            if ood_id == 381:
                ood_features = ['day', 'hour', 'DEWP', 'TEMP', 'PRES']

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
                ood_final, test_size=test_size, random_state=self.seed
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