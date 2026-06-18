import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_swell_kw

def train_model(data_dir: str, model_save_path: str):
    """
    Trains a Decision Tree model on the SWELL-KW dataset.
    """
    print("Initializing Offline Training Pipeline...")
    
    # 1. Load Data
    X_train, y_train, X_test, y_test = load_swell_kw(data_dir)
    
    # 2. Select Features (As per MindGuard Architecture)
    # The reference paper highlights mean RR, median RR, SDNN (SDRR in our CSV), HR, LF_NU, HF, HF_PCT, TP, LF_HF, sampen, higuci
    key_features = ['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'HR', 'LF_NU', 'HF', 'HF_PCT', 'TP', 'LF_HF', 'sampen', 'higuci']
    
    print(f"Filtering down to {len(key_features)} highly significant features...")
    X_train_filtered = X_train[key_features]
    X_test_filtered = X_test[key_features]
    
    # 3. Initialize Model
    # Since fast inference and high interpretability are priorities, we use Decision Tree
    model = DecisionTreeClassifier(random_state=42, max_depth=10)
    
    # 4. Train Model
    print("Training Decision Tree Classifier...")
    model.fit(X_train_filtered, y_train)
    
    # 5. Evaluate Model
    y_pred = model.predict(X_test_filtered)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Evaluation complete.")
    print(f"Accuracy on Test Set: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Stress (0)", "Moderate Stress (1)", "Severe Stress (2)"]))
    
    # 6. Save Model Artifact
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nTraining pipeline successful. Model artifact saved to: {model_save_path}")

if __name__ == "__main__":
    # Define absolute paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    model_save_path = os.path.join(project_root, "models", "stress_model.pkl")
    
    train_model(data_dir, model_save_path)
