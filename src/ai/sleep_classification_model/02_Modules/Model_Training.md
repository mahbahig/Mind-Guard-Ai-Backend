# Feature Selection & Model Training Module

**Follows Implementation Guides:**
- [[PPG-Based Sleep Stage Classification Using Pulse Wave Feature Fusion and Explainable AI]]
- [[Extracted PPG Features for Sleep Staging]]

## 📝 Description
Implements the paper's **GSCV-RF** classification setup:

1. **RFE** (Recursive Feature Elimination) — selects the top 10 most influential features from the full feature vector
2. **GSCV-RF** (Grid Search Cross-Validation + Random Forest) — tunes hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`) over 5-fold CV
3. **SHAP TreeExplainer** — produces exact Shapley values for XAI (replaces black-box permutation importance)

## 💻 Live Implementation
→ `src/model_training.py`

| Function | Purpose |
|---|---|
| `build_and_train_rfe_rf(X, y)` | RFE → GSCV-RF, returns `(best_rf, rfe, best_params)` |
| `predict(rf, rfe, X_test)` | Project via RFE then classify |
| `get_selected_feature_names(rfe, names)` | Extract the RFE-selected feature names |
| `compute_shap_values(rf, X_sel, names)` | SHAP TreeExplainer for XAI |

## ⚙️ Key Hyperparameter Grid (Paper-Aligned)
```python
param_grid = {
    "n_estimators":      [50, 100, 200],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}
```

## 🐛 Known Issues
- RF `predict_proba` may return fewer columns than `N_STAGES` if a class is absent in a LOSO training fold. `evaluate.py` handles this with a class-aligned proba padder.
- `shap.TreeExplainer` is slow on >500 samples — `max_background=300` sub-sampling is applied in `evaluate.py`.
