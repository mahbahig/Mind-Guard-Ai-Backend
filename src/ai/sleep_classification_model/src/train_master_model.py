"""
Master Model Training Script
============================
Trains the "master" Random Forest classifier and RFE selector on ALL available data
and saves the resulting models to disk for use in the real-time API simulation.

Loads every matching CSV in the data folder (no subject cap):
  DREAMT ``*_PSG_df_updated.csv`` or legacy ``S00*_whole_df.csv``.

By default, writes ``results/cache/master_training_xy.joblib`` after feature extraction
and reloads it on the next run when the resolved data directory matches, skipping CSV I/O
and extraction. Pass ``--no-feature-cache`` to always load CSVs and recompute features.
"""

import argparse
import os
import sys
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
import json

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_training import build_and_train_rfe_rf
from evaluate import _load_local_csv_data, _extract_features, set_eval_mode
from feature_engineering import get_feature_names as _current_feature_names


def _resolved_data_dir(data_dir):
    """Stable string for cache invalidation."""
    if not data_dir:
        return ""
    return os.path.normcase(os.path.abspath(os.path.expanduser(data_dir)))


def _feature_cache_path(proj_root):
    d = os.path.join(proj_root, "results", "cache")
    return os.path.join(d, "master_training_xy.joblib")


def _try_load_feature_cache(cache_path, expected_dir):
    if not os.path.isfile(cache_path):
        return None
    try:
        blob = joblib.load(cache_path)
    except Exception as e:
        print(f"  [cache] Could not read {cache_path}: {e}")
        return None
    meta = blob.get("meta") or {}
    if meta.get("data_dir") != expected_dir:
        print(
            "  [cache] Stale or different data_dir; recomputing features.\n"
            f"          cache: {meta.get('data_dir')!r}\n"
            f"          run:   {expected_dir!r}"
        )
        return None
    X, y = blob["X"], blob["y"]
    names = blob.get("feature_names")
    expected_n = len(_current_feature_names())
    if X.shape[1] != expected_n:
        print(
            f"  [cache] Feature count changed ({X.shape[1]} cached vs "
            f"{expected_n} current); recomputing features."
        )
        return None
    print(
        f"  [cache] Loaded {X.shape[0]} epochs × {X.shape[1]} features "
        f"({meta.get('n_subjects', '?')} subjects)."
    )
    return X, y, names


def _save_feature_cache(cache_path, data_dir, X, y, feature_names, n_subjects):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    meta = {
        "data_dir": _resolved_data_dir(data_dir),
        "n_subjects": int(n_subjects),
        "x_shape": tuple(X.shape),
        "y_shape": tuple(y.shape),
    }
    joblib.dump(
        {"X": X, "y": y, "feature_names": feature_names, "meta": meta},
        cache_path,
    )
    print(f"  [cache] Saved feature matrix to: {cache_path}")

# ---------------------------------------------------------------------------
# Edit this path to your DREAMT (or any) folder with *_PSG_df_updated.csv.
# If the path exists, training uses it automatically (unless you pass --data-dir).
# Set to "" to ignore and use DREAMT_DATA_DIR env or project data/ instead.
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = r"E:\sleepdata\dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-2.1.0\data_100Hz"


def main():
    parser = argparse.ArgumentParser(
        description="Train master sleep classifier on ALL CSVs in a folder (Wake vs Sleep)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data folder. If omitted: DEFAULT_DATA_DIR (if it exists), "
        "else DREAMT_DATA_DIR env, else project data/.",
    )
    parser.add_argument(
        "--no-feature-cache",
        action="store_true",
        help="Do not read or write results/cache/master_training_xy.joblib; always load "
        "CSVs and extract features (default is cache on).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None and DEFAULT_DATA_DIR:
        data_dir = DEFAULT_DATA_DIR

    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cache_path = _feature_cache_path(proj_root)
    resolved_dir = _resolved_data_dir(data_dir)
    use_cache = not args.no_feature_cache

    # Enforce 2-stage model classification (Wake vs Sleep)
    set_eval_mode("2-stage")
    print("=" * 60)
    print("  Sleep Classification -- Master Model Training")
    print("=" * 60)

    # Pre-flight GPU check so the user knows immediately
    from model_training import _gpu_available, _HAS_XGB
    if _HAS_XGB and _gpu_available():
        import xgboost
        print(f"  [GPU] CUDA GPU detected -- XGBoost {xgboost.__version__} will train on GPU")
    elif _HAS_XGB:
        print("  [CPU] No CUDA GPU found -- XGBoost will train on CPU")
    else:
        print("  [CPU] XGBoost not installed -- will use sklearn RandomForest on CPU")

    X_train_full = y_train_full = feature_names = None

    if use_cache:
        print("\n[cache] Checking feature cache...")
        loaded = _try_load_feature_cache(cache_path, resolved_dir)
        if loaded is not None:
            X_train_full, y_train_full, feature_names = loaded

    if X_train_full is None:
        if use_cache and not os.path.isfile(cache_path):
            print("\n[cache] No cache file yet; after extraction it will be saved for next run.")

        # 1. Load data — subjects 002 to 100
        print("\n[1] Loading subject CSVs (002 to 100)...")
        if data_dir:
            print(f"  Data directory: {data_dir}")
        raw_subjects = _load_local_csv_data(fs=100, data_dir=data_dir, sid_range=(2, 100))

        if not raw_subjects:
            print(
                "[error] No data loaded. Fix DEFAULT_DATA_DIR in train_master_model.py, "
                "or use --data-dir, or set DREAMT_DATA_DIR, or put CSVs in data/."
            )
            return

        print(f"  Subjects loaded: {len(raw_subjects)} (whole folder).")

        # 2. Extract features
        print("\n[2] Feature Extraction ---")
        processed_subjects, feature_names = _extract_features(raw_subjects, fs_target=100)

        # 3. Concatenate all data into one master training set
        print("\n[3] Building master training dataset ---")
        train_Xs = [subj[0] for subj in processed_subjects]
        train_ys = [subj[1] for subj in processed_subjects]

        X_train_full = np.vstack(train_Xs)
        y_train_full = np.concatenate(train_ys)

        print(f"  Master dataset assembled: {X_train_full.shape[0]} total epochs.")

        if use_cache:
            _save_feature_cache(
                cache_path,
                data_dir,
                X_train_full,
                y_train_full,
                feature_names,
                len(processed_subjects),
            )

    # 3b. Diagnostics
    print("\n[3b] Dataset Diagnostics ---")
    classes, counts = np.unique(y_train_full, return_counts=True)
    for c, cnt in zip(classes, counts):
        label = "Wake" if int(c) == 0 else "Sleep"
        print(f"  {label} (class {int(c)}): {cnt:,} epochs ({100 * cnt / len(y_train_full):.1f}%)")

    nan_fracs = np.mean(np.isnan(X_train_full) | np.isinf(X_train_full), axis=0)
    bad_cols = np.where(nan_fracs > 0.5)[0]
    if len(bad_cols) > 0:
        names = feature_names or []
        print(f"  WARNING: {len(bad_cols)} columns have >50% NaN/Inf:")
        for ci in bad_cols:
            name = names[ci] if ci < len(names) else f"col_{ci}"
            print(f"    [{ci}] {name}: {100*nan_fracs[ci]:.1f}% bad")

    const_cols = np.where(np.std(np.nan_to_num(X_train_full), axis=0) < 1e-12)[0]
    if len(const_cols) > 0:
        names = feature_names or []
        print(f"  WARNING: {len(const_cols)} constant columns:")
        for ci in const_cols:
            name = names[ci] if ci < len(names) else f"col_{ci}"
            print(f"    [{ci}] {name}")

    # IBI quality diagnostic — check how many epochs have usable IBI spectral data
    names = feature_names or []
    ibi_tp_idx = None
    for idx, n in enumerate(names):
        if n == "IBI_TP":
            ibi_tp_idx = idx
            break
    if ibi_tp_idx is not None and ibi_tp_idx < X_train_full.shape[1]:
        ibi_col = X_train_full[:, ibi_tp_idx]
        usable = np.sum(np.isfinite(ibi_col) & (ibi_col > 1e-12))
        print(f"  IBI quality: {usable:,}/{len(ibi_col):,} epochs "
              f"({100 * usable / len(ibi_col):.1f}%) have usable IBI spectral data")

    X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=0.0, neginf=0.0)

    # 3c. Mann-Whitney U feature discriminative power analysis
    print("\n[3c] Feature Discriminative Power (Mann-Whitney U) ---")
    wake_mask = y_train_full == 0
    sleep_mask = y_train_full == 1
    mw_pvals = []
    for fi in range(X_train_full.shape[1]):
        w_vals = X_train_full[wake_mask, fi]
        s_vals = X_train_full[sleep_mask, fi]
        if np.std(w_vals) < 1e-12 and np.std(s_vals) < 1e-12:
            mw_pvals.append(1.0)
            continue
        try:
            _, pval = mannwhitneyu(w_vals, s_vals, alternative="two-sided")
            mw_pvals.append(pval)
        except Exception:
            mw_pvals.append(1.0)
    mw_pvals = np.array(mw_pvals)
    ranked = np.argsort(mw_pvals)
    print("  Top-20 most discriminative features (Wake vs Sleep):")
    for rank, fi in enumerate(ranked[:20]):
        fname = names[fi] if fi < len(names) else f"col_{fi}"
        print(f"    {rank+1:2d}. [{fi:3d}] {fname:<30s}  p={mw_pvals[fi]:.2e}")
    n_nonsig = int(np.sum(mw_pvals > 0.05))
    if n_nonsig > 0:
        print(f"  {n_nonsig} features have p > 0.05 (non-discriminative)")

    # 3d. StandardScaler (GPU accelerated if cuml is available, else fallback)
    print("\n[3d] Fitting StandardScaler ---")
    try:
        from cuml.preprocessing import StandardScaler as cuStandardScaler
        scaler = cuStandardScaler()
        print("  Using cuML StandardScaler (GPU)")
    except ImportError:
        scaler = StandardScaler()
    
    X_train_full = scaler.fit_transform(X_train_full)
    # Ensure it's numpy array if cuML returns CuPy
    if hasattr(X_train_full, "get"):
        X_train_full = X_train_full.get()
        
    print(f"  Scaled {X_train_full.shape[1]} features (zero-mean, unit-variance).")

    # 4. Train Model — RFE + GridSearchCV (GPU-accelerated when available)
    # scale_pos_weight inside XGBoost handles class imbalance at the loss level
    print("\n[4] Training Master Model (RFE + GridSearchCV) ---")
    best_rf, rfe, best_params = build_and_train_rfe_rf(
        X_train_full, y_train_full, n_features_to_select=30, cv_folds=5
    )

    # 5. Save the trained artifacts
    print("\n[5] Saving Model Artifacts to Disk ---")
    models_dir = os.path.join(proj_root, "results", "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "master_rf.joblib")
    rfe_path = os.path.join(models_dir, "master_rfe.joblib")
    scaler_path = os.path.join(models_dir, "master_scaler.joblib")

    joblib.dump(best_rf, model_path)
    joblib.dump(rfe, rfe_path)
    joblib.dump(scaler, scaler_path)

    print(f"  [OK] Master model saved to: {model_path}")
    print(f"  [OK] Feature selector saved to: {rfe_path}")
    print(f"  [OK] Scaler saved to: {scaler_path}")

    # 6. Threshold calibration on held-out validation subject (S101)
    print("\n[6] Threshold Calibration (S101 validation) ---")
    wake_threshold = 0.5
    try:
        val_raw = _load_local_csv_data(fs=100, data_dir=data_dir, sid_range=(101, 101))
        if val_raw:
            val_processed, _ = _extract_features(val_raw, fs_target=100)
            X_val = np.vstack([s[0] for s in val_processed])
            y_val = np.concatenate([s[1] for s in val_processed])
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            X_val = scaler.transform(X_val)
            if hasattr(X_val, "get"):
                X_val = X_val.get()

            from model_training import predict as mt_predict
            from sklearn.metrics import balanced_accuracy_score

            _, y_proba, _ = mt_predict(best_rf, rfe, X_val)

            best_thr = 0.5
            best_ba = 0.0
            print("  Sweeping wake thresholds 0.30-0.70 ...")
            for thr in np.arange(0.30, 0.71, 0.01):
                y_pred_thr = np.where(y_proba[:, 0] >= thr, 0, 1)
                ba = balanced_accuracy_score(y_val, y_pred_thr)
                if ba > best_ba:
                    best_ba = ba
                    best_thr = thr

            wake_threshold = round(float(best_thr), 2)
            print(f"  Optimal wake threshold: {wake_threshold}")
            print(f"  Validation balanced accuracy: {best_ba:.4f}")

            y_default = np.where(y_proba[:, 0] >= 0.5, 0, 1)
            ba_default = balanced_accuracy_score(y_val, y_default)
            print(f"  Default (0.50) balanced accuracy: {ba_default:.4f}")
            print(f"  Improvement: +{best_ba - ba_default:.4f}")
        else:
            print("  Could not load S101 for calibration. Using default threshold 0.5.")
    except Exception as e:
        print(f"  Threshold calibration failed ({e}); using default 0.5.")

    params_path = os.path.join(models_dir, "best_params.json")
    params_data = {}
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params_data = json.load(f)
    params_data["wake_threshold"] = wake_threshold
    with open(params_path, "w") as f:
        json.dump(params_data, f, indent=2)
    print(f"  Saved wake_threshold={wake_threshold} to: {params_path}")

    print("\nYou can now run the simulation API using:")
    print("  uvicorn src.api_simulation:app --reload --port 8001")
    print("\nValidate HRV + Sleep models (metrics and plots) from Ai_Models root:")
    print("  python evaluate_all_models.py")

if __name__ == "__main__":
    main()
