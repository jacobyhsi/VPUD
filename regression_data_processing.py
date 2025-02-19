import pandas as pd
import ast

def parse_features_to_note(row, feature_columns: list[str]):
    note = []
    for feature in feature_columns:
        note.append(f"{feature} = {row[feature]}")
    # join note with ;
    return "; ".join(note)

def create_x_row_from_x_features(x_features: str, feature_columns: list[str]):
    x_features = ast.literal_eval(x_features)
    x_row = pd.DataFrame(x_features)
    x_row["label"] = 0
    x_row["note"] = x_row.apply(lambda row: parse_features_to_note(row, feature_columns), axis=1)
    
    print("X Row:\n", x_row)
    
    return x_row