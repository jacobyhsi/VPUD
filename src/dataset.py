from datasets import load_from_disk
from sklearn.model_selection import train_test_split
import pandas as pd
import json

def load_dataset(data_path):
    # Load Dataset
    dataset = Dataset(data_path=data_path)
    data = dataset.get_train_data()
    test = dataset.get_test_data()
    # exit() # inspect feature names

    # Load Dataset Configs
    file_config = json.load(open(f'{data_path}/info.json'))
    label_name = file_config['label']
    label_map = file_config['map']
    label_keys = list(label_map)

    return data, test, label_name, label_map, label_keys

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

def parse_features_to_note(features, feature_order=None):
    if feature_order is None:
        feature_order = list(features.keys())
        
    note_parts = []
    for key in feature_order:
        if key in features and features[key] is not None:
            note_parts.append(f"{key} = {features[key]}")
    note = ". ".join(note_parts) + "."
    return note

class Dataset():
    def __init__(self, data_path, seed=123):
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
        note2features = data['note'].apply(parse_note_to_features).apply(pd.Series)
        print("Features:", ", ".join(note2features.columns))

        if "adult" in data_path.lower():
            salient_features = [
                'Work class', 'Marital status', 'Relation to head of the household', 
                'Race', 'Capital gain last year', 'Work hours per week'
            ] # Based on InterpreTabNet https://arxiv.org/abs/2406.00426
        else:
            salient_features = note2features.columns.tolist()

        data['note'] = note2features[salient_features].apply(
            lambda row: parse_features_to_note(row.to_dict(), feature_order=salient_features),
            axis=1
        )

        df_filtered = data[['label', 'note']].copy()  # Ensure 'note' and 'label' are included
        data = pd.concat([df_filtered, note2features[salient_features]], axis=1)

        # Split data
        self.data, self.test_data = train_test_split(data, test_size=0.2, random_state=seed)

    def get_train_data(self):
        print("Train Data Shape:", self.data.shape)
        return self.data

    def get_test_data(self):
        print("Test Data Shape:", self.test_data.shape)
        return self.test_data
