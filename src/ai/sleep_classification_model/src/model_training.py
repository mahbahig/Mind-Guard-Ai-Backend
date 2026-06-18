"""
Model Training Module — Sleep Stage Classification
====================================================
Implements the classification setup:
  1. Recursive Feature Elimination (RFE) → top-N features
  2. Grid Search Cross-Validation (GSCV) with the best available tree model
  3. SHAP TreeExplainer for explainability

Backend priority:
  XGBoost + CUDA GPU  →  XGBoost CPU  →  sklearn RandomForest CPU.
Progress is printed for every RFE round and every GridSearchCV fold.
"""

import time
import numpy as np
import shap
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def _gpu_available():
    """Return True when a usable CUDA GPU is detected (XGBoost probe takes priority)."""
    if _HAS_XGB:
        try:
            probe = XGBClassifier(
                n_estimators=1, max_depth=1, device="cuda",
                tree_method="hist", verbosity=0,
            )
            probe.fit(np.zeros((2, 1)), np.array([0, 1]))
            return True
        except Exception:
            pass
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    return False


# ─────────────────────────────────────────────────────────────
#  Core: RFE → GSCV (XGBoost GPU / CPU, or RF fallback)
# ─────────────────────────────────────────────────────────────
def build_and_train_rfe_rf(X_train, y_train, n_features_to_select=10, cv_folds=5):
    """
    Trains a tree classifier with RFE feature selection and Grid Search CV.

    Backend is chosen automatically:
      XGBoost + CUDA GPU  →  XGBoost CPU  →  sklearn RandomForest CPU.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_epochs, n_features)
    y_train : np.ndarray, shape (n_epochs,)
    n_features_to_select : int
    cv_folds : int

    Returns
    -------
    best_est    : trained classifier (XGBClassifier or RandomForestClassifier)
    rfe         : fitted RFE selector
    best_params : dict of best hyperparameters
    """
    use_gpu = _HAS_XGB and _gpu_available()

    if _HAS_XGB and use_gpu:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA"
        print(f"[device] {gpu_name} -- training with XGBoost GPU.")
    elif _HAS_XGB:
        print("[device] No CUDA GPU -- training with XGBoost CPU.")
    else:
        print("[device] XGBoost not installed -- training with sklearn RandomForest CPU.")
        print("         pip install xgboost   (recommended for GPU support)")

    # ── Class imbalance handling ──
    classes, counts = np.unique(y_train, return_counts=True)
    print(f"\n  [class distribution]")
    for c, cnt in zip(classes, counts):
        print(f"    class {int(c)}: {cnt:,} ({100 * cnt / len(y_train):.1f}%)")

    spw = 1.0
    if len(classes) == 2:
        neg_count = counts[0]
        pos_count = counts[1]
        spw = float(neg_count) / float(pos_count) if pos_count > 0 else 1.0
        print(f"  [imbalance] scale_pos_weight = {spw:.3f}")

    # ── Build base estimator ──
    if _HAS_XGB:
        base_params = dict(
            n_estimators=100,
            tree_method="hist",
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
            scale_pos_weight=spw,
        )
        if use_gpu:
            base_params.update(device="cuda", n_jobs=1)
        else:
            base_params["n_jobs"] = -1
        base_est = XGBClassifier(**base_params)
        kind = "xgb"
    else:
        base_est = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        kind = "rf"

    # ── Step 1: Recursive Feature Elimination ──
    n_total = X_train.shape[1]
    rfe_step = 5
    n_rounds = max(1, (n_total - n_features_to_select + rfe_step - 1) // rfe_step)
    print(
        f"\n  [Step 1/2] RFE: {n_total} -> {n_features_to_select} features "
        f"(~{n_rounds} rounds, step={rfe_step})"
    )

    t0 = time.time()
    rfe = RFE(
        estimator=base_est,
        n_features_to_select=n_features_to_select,
        step=rfe_step,
        verbose=1,
    )
    rfe.fit(X_train, y_train)
    rfe_elapsed = time.time() - t0

    X_train_sel = rfe.transform(X_train)
    print(
        f"\n  [RFE] Done in {rfe_elapsed:.1f}s -- "
        f"kept {n_features_to_select}/{n_total} features."
    )

    # ── Step 2: Grid Search CV ──
    if kind == "xgb":
        gscv_params = dict(
            tree_method="hist",
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
            scale_pos_weight=spw,
        )
        if use_gpu:
            gscv_params.update(device="cuda", n_jobs=1)
        else:
            gscv_params["n_jobs"] = -1
        gscv_est = XGBClassifier(**gscv_params)
        param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "min_child_weight": [1, 3],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        gscv_n_jobs = 1 if use_gpu else -1
    else:
        gscv_est = RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
        gscv_n_jobs = -1

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    total_fits = n_combos * cv_folds

    print(
        f"\n  [Step 2/2] GridSearchCV: "
        f"{n_combos} param combos × {cv_folds} folds = {total_fits} fits"
    )

    t1 = time.time()
    gscv = GridSearchCV(
        gscv_est,
        param_grid,
        cv=cv_folds,
        scoring="balanced_accuracy",
        n_jobs=gscv_n_jobs,
        verbose=2,
    )
    gscv.fit(X_train_sel, y_train)
    gscv_elapsed = time.time() - t1

    best_params = gscv.best_params_

    print(f"\n  [GSCV] Done in {gscv_elapsed:.1f}s")
    print(f"  [GSCV] Best params: {best_params}")
    print(f"  [GSCV] Best CV balanced accuracy: {gscv.best_score_:.4f}")

    # ── Step 3: Early-stopping refinement (XGBoost only) ──
    best_est = gscv.best_estimator_
    if kind == "xgb":
        print("\n  [Step 3/3] Early-stopping refinement ...")
        t2 = time.time()
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_sel, y_train, test_size=0.15,
            random_state=42, stratify=y_train,
        )
        es_params = {k: v for k, v in best_params.items() if k != "n_estimators"}
        es_params.update(
            n_estimators=2000,
            tree_method="hist",
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
            scale_pos_weight=spw,
            early_stopping_rounds=30,
        )
        if use_gpu:
            es_params.update(device="cuda", n_jobs=1)
        else:
            es_params["n_jobs"] = -1
        es_model = XGBClassifier(**es_params)
        es_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        optimal_n = es_model.best_iteration + 1
        es_elapsed = time.time() - t2
        print(f"  [early stop] Optimal n_estimators: {optimal_n} (in {es_elapsed:.1f}s)")

        final_params = dict(es_params)
        final_params["n_estimators"] = optimal_n
        final_params.pop("early_stopping_rounds", None)
        best_est = XGBClassifier(**final_params)
        best_est.fit(X_train_sel, y_train)
        best_params["n_estimators"] = optimal_n
        print(f"  [early stop] Refitted on full training data with n_estimators={optimal_n}")

    total = time.time() - t0
    print(f"\n  Total training time: {total / 60:.1f} min ({total:.0f}s)")

    return best_est, rfe, best_params


# ─────────────────────────────────────────────────────────────
#  SHAP TreeExplainer (works with both XGBoost and RF)
# ─────────────────────────────────────────────────────────────
def compute_shap_values(trained_model, X_selected, feature_names_selected, max_background=200):
    """
    Applies SHAP TreeExplainer on the trained model.

    Parameters
    ----------
    trained_model          : fitted tree classifier (XGBClassifier or RF)
    X_selected             : np.ndarray (n_samples, n_selected_features)
    feature_names_selected : list[str]
    max_background         : max samples for SHAP background

    Returns
    -------
    shap_values : array or list of arrays
    explainer   : shap.TreeExplainer instance
    """
    bg_size = min(max_background, X_selected.shape[0])
    background = X_selected[:bg_size]

    explainer = shap.TreeExplainer(trained_model, background)
    shap_values = explainer.shap_values(X_selected)

    print(
        f"[SHAP] TreeExplainer computed on {X_selected.shape[0]} samples, "
        f"{X_selected.shape[1]} selected features."
    )
    return shap_values, explainer


# ─────────────────────────────────────────────────────────────
#  Convenience helpers
# ─────────────────────────────────────────────────────────────
def predict(trained_model, rfe, X_test):
    """Project X_test through RFE then return predictions and probabilities."""
    if rfe is not None:
        X_sel = rfe.transform(X_test)
    else:
        X_sel = X_test
        
    y_pred = trained_model.predict(X_sel)
    y_proba = trained_model.predict_proba(X_sel)
    return y_pred, y_proba, X_sel


def get_selected_feature_names(rfe, all_feature_names):
    """Return the feature names that RFE kept."""
    mask = rfe.support_
    return [name for name, keep in zip(all_feature_names, mask) if keep]
