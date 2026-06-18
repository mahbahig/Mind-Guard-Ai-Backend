import pandas as pd
import numpy as np
import os

def load_swell_kw(data_dir: str):
    """
    Loads and structures the pre-extracted SWELL-KW dataset.
    
    Args:
        data_dir: Path to the data directory containing 'final/train.csv' and 'final/test.csv'

    Returns:
        X_train, y_train, X_test, y_test (pandas DataFrames and Series)
    """
    train_path = os.path.join(data_dir, "final", "train.csv")
    test_path = os.path.join(data_dir, "final", "test.csv")
    
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    
    print(f"Loading testing data from {test_path}...")
    test_df = pd.read_csv(test_path)
    
    # Target feature is 'condition', dropping 'datasetId' which is metadata
    # The condition labels are: 'no stress', 'time pressure', 'interruption'
    label_mapping = {
        'no stress': 0,
        'time pressure': 1,
        'interruption': 2
    }
    
    # Map the labels
    train_df['condition'] = train_df['condition'].map(label_mapping)
    test_df['condition'] = test_df['condition'].map(label_mapping)
    
    # Drop any NaNs
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    X_train = train_df.drop(columns=['condition', 'datasetId'])
    y_train = train_df['condition']
    
    X_test = test_df.drop(columns=['condition', 'datasetId'])
    y_test = test_df['condition']
    
    # Optional: we can explicitly select only the paper's highly significant features here
    # key_features = ['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'HR', 'LF_NU', 'HF', 'HF_PCT', 'TP', 'LF_HF', 'sampen', 'higuci']
    # X_train = X_train[key_features]
    # X_test = X_test[key_features]
    
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} testing samples.")
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Test the loader
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    X_train, y_train, X_test, y_test = load_swell_kw(data_dir)
    print("Features:", X_train.columns.tolist())
    print("Class distribution (Train):", y_train.value_counts().to_dict())
