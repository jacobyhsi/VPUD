
def parse_features_to_note(row, feature_columns: list[str]):
    note = []
    for feature in feature_columns:
        note.append(f"{feature} = {row[feature]}")
    # join note with ;
    return "; ".join(note)