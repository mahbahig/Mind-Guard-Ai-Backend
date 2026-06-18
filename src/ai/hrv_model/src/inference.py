import os
import pickle
import numpy as np

def load_model(model_path: str):
    """
    Loads the trained Decision Tree model artifact.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train.py first.")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_stress(model, input_features: dict) -> int:
    """
    Takes the freshly extracted real-time HRV features and returns a stress classification.
    
    Args:
        model: The loaded sklearn model.
        input_features: Dictionary matching the format of SWELL-KW key features:
                        ['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'HR', 'LF_NU', 'HF', 'HF_PCT', 'TP', 'LF_HF', 'sampen', 'higuci']
    
    Returns:
        int: 0 (No Stress), 1 (Moderate Stress/Time Pressure), 2 (Severe Stress/Interruption)
    """
    # The model expects a 2D array ordered exactly like the training features
    key_features_order = ['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'HR', 'LF_NU', 'HF', 'HF_PCT', 'TP', 'LF_HF', 'sampen', 'higuci']
    
    # Extract the values in the exact correct order
    try:
        feature_vector = [input_features[feat] for feat in key_features_order]
    except KeyError as e:
        raise KeyError(f"Missing required real-time HRV feature: {e}")
        
    # Reshape for single sample prediction
    X_inference = np.array(feature_vector).reshape(1, -1)
    
    # Predict
    prediction = model.predict(X_inference)
    return int(prediction[0])
