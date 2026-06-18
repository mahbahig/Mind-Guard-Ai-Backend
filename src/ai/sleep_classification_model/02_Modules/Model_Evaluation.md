# Model Evaluation Module

**Runs:** `python src/evaluate.py`  
**Produces:** 4 plots in `results/plots/` + Markdown metrics table to console

## 📝 Description
Executes Leave-One-Subject-Out (LOSO) cross-validation over all 18 SLPDB subjects, aggregates predictions, computes a full metrics protocol, and produces 4 visualization plots.

**Stage Mapping (AASM → integer):**

| Stage | Index |
|-------|-------|
| Wake  | 0 |
| N1    | 1 |
| N2    | 2 |
| N3    | 3 |
| REM   | 4 |

## 📊 Metrics Protocol

```python
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    balanced_accuracy_score,
)

# Per-class: Precision, Recall, F1-Score, Support
print(classification_report(y_true, y_pred, target_names=STAGE_NAMES))

# Agreement beyond chance
kappa = cohen_kappa_score(y_true, y_pred)

# Average recall per class (immune to class imbalance)
bal_acc = balanced_accuracy_score(y_true, y_pred)
```

## 🖼️ Visualization Suite

### A — Normalized Confusion Matrix
- Seaborn heatmap showing **% correct vs incorrect per stage**
- Reveals N2/N3 confusion (most common failure mode)

### B — Hypnogram Comparison
- Side-by-side Ground Truth vs Predicted step plot over 8-hour timeline
- Reveals whether the model captures **90-minute sleep cycles**
- Stage Y-axis order: Wake → REM → N1 → N2 → N3 (standard hypnogram layout)

### C — Class-Wise ROC & AUC
- One-vs-Rest ROC curve per stage
- AUC values in legend
- Diagonal chance line for reference

### D — SHAP Feature Importance
- `shap.TreeExplainer` on the trained RF
- Aggregated `mean(|SHAP|)` across all classes
- Falls back to Gini Impurity importance if SHAP errors

## 💻 Code (Copy-Pasteable)

```python
# Run the full pipeline from project root:
# cd src && python evaluate.py
# or
import subprocess
subprocess.run(["python", "src/evaluate.py"], cwd="<project_root>")
```

## 🔗 Cross-References
- Data: [[Data_Ingestion]] → `load_slpdb_data()`
- Preprocessing: [[Preprocessing]] → `preprocess_all_epochs()`
- Features: [[Feature_Engineering]] → `build_feature_vector(epochs, fs=125)`
- Model: [[Model_Training]] → `build_and_train_rfe_rf(X, y)`

## 🐛 Known Issues/Bugs
- **LOSO is slow on real SLPDB** (18 folds × full GSCV). Use `n_estimators=50` during experimentation, bump to `200` for final evaluation.
- **SHAP multi-class memory**: For large test arrays, SHAP may consume excessive RAM. The script caps the SHAP sample at 200 epochs.
- **Missing stages in a fold**: If a training fold contains no N1 epochs, `predict_proba` columns will be misaligned. The script pads the probability array to always have 5 columns.
