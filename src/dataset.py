from datasets import load_from_disk
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
from src.utils import TabularUtils, ToyClassificationUtils

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
    # exit() # inspect feature names

    # Load Dataset Configs
    file_config = json.load(open(f'{data_path}/info.json'))
    label_map = file_config['map']
    label_keys = list(label_map)

    return data, test, label_keys

class Dataset():
    def __init__(self, data_path, data_split_seed: int = 123):
        data = self.load_data(data_path)
        # Split data
        self.data, self.test_data = train_test_split(data, test_size=0.2, random_state=data_split_seed)

    def get_train_data(self):
        return self.data

    def get_test_data(self):
        return self.test_data
    
    def load_data(self, data_path):
        raise NotImplementedError

class TabularDataset(Dataset):
    def load_data(self, data_path: str):
        # Load Dataset
        data = load_from_disk(data_path).to_pandas()

        # Preprocess Dataset
        data['label'] = data['label'].apply(lambda x: 0 if x is False else 1)
        data['note'] = (data['note']
                        .str.replace(r'\bThe\b', '', regex=True)
                        .str.replace(r'\bis\b', '=', regex=True)
                        .str.replace(r'\s{2,}', ' ', regex=True)
                        .str.lstrip())

        # Convert Note to Features, then concat to dataset
        note2features = data['note'].apply(TabularUtils.parse_note_to_features).apply(pd.Series)
        print("Features:", ", ".join(note2features.columns))

        if "adult" in data_path.lower():
            salient_features = [
                'Work class', 'Marital status', 'Relation to head of the household', 
                'Race', 'Capital gain last year', 'Work hours per week'
            ] # Based on InterpreTabNet https://arxiv.org/abs/2406.00426
        else:
            salient_features = note2features.columns.tolist()

        data['note'] = note2features[salient_features].apply(
            lambda row: TabularUtils.parse_features_to_note(row.to_dict(), feature_order=salient_features),
            axis=1
        )

        df_filtered = data[['label', 'note']].copy()  # Ensure 'note' and 'label' are included
        
        data = pd.concat([df_filtered, note2features[salient_features]], axis=1)
        
        return data
    
class ToyClassificationDataset(Dataset):
    def load_data(self, data_path: str):
        data = pd.read_csv(os.path.join(data_path, 'data.csv'), index_col=0)
                    
        data['label'] = data['label'].astype(int)
        
        feature_column = ToyClassificationUtils.get_feature_columns(data)
        
        data['note'] = data.apply(
            lambda row: ToyClassificationUtils.parse_features_to_note(row, feature_column),
            axis=1
        )
        
        return data
        
        
DATATYPE_TO_DATACLASS: dict[str, Dataset] = {
    "tabular": TabularDataset,
    "toy_classification": ToyClassificationDataset,
}