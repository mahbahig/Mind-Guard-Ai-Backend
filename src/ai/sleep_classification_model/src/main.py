"""
Main Pipeline Executor — Sleep Stage Classification
=====================================================
Ties together data ingestion, preprocessing,
feature engineering, and the paper's GSCV-RF model training.

Reference:
  Smarandache et al. (2025) — PPG-Based Sleep Stage Classification
  Using Pulse Wave Feature Fusion and Explainable AI
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from data_ingestion import load_slpdb_data
from preprocessing import preprocess_all_epochs
from feature_engineering import build_feature_vector, get_feature_names
from model_training import build_and_train_rfe_rf, predict, get_selected_feature_names


def run_pipeline():
    # ── 1. Data Ingestion ──
    print("--- 1. Data Ingestion ---")
    records = load_slpdb_data()

    # ── 2. Preprocessing & Epoching ──
    print("\n--- 2. Preprocessing & Epoching ---")
    fs             = 125
    epoch_len_sec  = 30
    num_mock_epochs = 200

    print(f"Generating {num_mock_epochs} mock epochs (FS={fs} Hz) for testing.")
    mock_data = np.random.randn(num_mock_epochs, fs * epoch_len_sec)

    print("Applying baseline wander removal, Savitzky-Golay filtering, Z-score normalization...")
    preprocessed_epochs = preprocess_all_epochs(mock_data, fs)

    # ── 3. Feature Engineering ──
    print("\n--- 3. Feature Engineering ---")
    features_matrix = build_feature_vector(preprocessed_epochs, fs=fs)
    feature_names   = get_feature_names()
    features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Extracted {features_matrix.shape[1]} features per epoch. Array shape: {features_matrix.shape}")

    # ── 4. Subject-Level Z-Score Normalization ──
    print("\n--- 4. Subject-Level Z-Score Normalization ---")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features_matrix)

    # Mock labels (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
    mock_labels = np.random.choice(5, size=num_mock_epochs, p=[0.20, 0.05, 0.50, 0.10, 0.15])
    print(f"Label distribution: { {i: int((mock_labels==i).sum()) for i in range(5)} }")

    # ── 5. RFE + GSCV-RF Training (Paper Setup) ──
    print("\n--- 5. Feature Selection (RFE) + GSCV-RF Training ---")
    best_rf, rfe, best_params = build_and_train_rfe_rf(
        X_scaled, mock_labels, n_features_to_select=10
    )

    sel_names = get_selected_feature_names(rfe, feature_names)
    print(f"Top-10 RFE-selected features: {sel_names}")

    # ── 6. Prediction ──
    print("\n--- 6. Prediction on Training Data (sanity check) ---")
    y_pred, y_proba, X_sel = predict(best_rf, rfe, X_scaled)
    acc = (y_pred == mock_labels).mean()
    print(f"Training accuracy (sanity check): {acc:.4f}")

    print("\nPipeline Execution Complete!")


if __name__ == "__main__":
    run_pipeline()
