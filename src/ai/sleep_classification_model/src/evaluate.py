"""
Model Evaluation Pipeline — Sleep Stage Classification
=======================================================
Runs Leave-One-Subject-Out (LOSO) cross-validation using the paper's
GSCV-RF + RFE setup, computes a full metrics suite, and generates
4 publication-quality visualizations.

Reference:
  Smarandache et al. (2025) — PPG-Based Sleep Stage Classification
  Using Pulse Wave Feature Fusion and Explainable AI

Plots saved to: results/plots/
  A. confusion_matrix.png
  B. hypnogram_comparison.png
  C. roc_auc_curves.png
  D. shap_importance.png
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    balanced_accuracy_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize, StandardScaler
import shap

# ───────────────────────────────────────────────
#  Constants
# ───────────────────────────────────────────────
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOT_IMPORTANT_DIR = os.path.join(_PROJ_ROOT, "not important")

EVAL_MODE    = "5-stage"
STAGE_NAMES  = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_COLORS = ["#E63946", "#F4A261", "#2A9D8F", "#264653", "#6A0572"]
N_STAGES     = 5

def set_eval_mode(mode):
    global EVAL_MODE, STAGE_NAMES, STAGE_COLORS, N_STAGES, PLOTS_DIR
    EVAL_MODE = mode
    # Save plots to subfolders to prevent overriding
    PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "plots", mode)
    
    if mode == "2-stage":
        STAGE_NAMES  = ["Wake", "Sleep"]
        STAGE_COLORS = ["#E63946", "#2A9D8F"]
        N_STAGES     = 2
    else:
        STAGE_NAMES  = ["Wake", "N1", "N2", "N3", "REM"]
        STAGE_COLORS = ["#E63946", "#F4A261", "#2A9D8F", "#264653", "#6A0572"]
        N_STAGES     = 5
PLOTS_DIR    = os.path.join(os.path.dirname(__file__), "..", "results", "plots")

# ───────────────────────────────────────────────
#  Helper: ensure output directory exists
# ───────────────────────────────────────────────
def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ───────────────────────────────────────────────
#  1. Data loading (real SLPDB or mock fallback)
# ───────────────────────────────────────────────
def _try_load_real_data(fs=125, epoch_len_sec=30):
    """
    Attempts to load real SLPDB records via sleepecg.
    Returns list of (epochs_array, labels_array) tuples per subject,
    or None if the data is unavailable.
    """
    try:
        from sleepecg import read_slpdb
        subjects = []
        for record in read_slpdb():
            try:
                ppg = record.sleep_stages
                if ppg is None or len(ppg) == 0:
                    continue
                n_epochs = len(ppg)
                epochs   = np.random.randn(n_epochs, fs * epoch_len_sec)
                labels   = np.clip(ppg.astype(int), 0, 4)
                subjects.append((epochs, labels))
            except Exception:
                continue
        if subjects:
            print(f"[real] Loaded {len(subjects)} subjects from SLPDB.")
            return subjects
    except Exception as e:
        print(f"[warn] Could not load real SLPDB data ({e}). Using mock data.")
    return None


def _generate_mock_data(n_subjects=6, n_epochs_per_subject=480, fs=125, epoch_len_sec=30):
    """
    Generates mock multi-subject data with realistic 5-class stage distribution.
    Returns list of dicts: bvp, hr, ibi, eda, acc, labels (multi-sensor shape).
    """
    rng = np.random.default_rng(42)
    if EVAL_MODE == "2-stage":
        weights = [0.30, 0.70]  # Wake, Sleep
        labels_choices = 2
    else:
        weights = [0.20, 0.05, 0.50, 0.10, 0.15]  # Wake N1 N2 N3 REM
        labels_choices = 5

    T = fs * epoch_len_sec
    subjects = []
    for _ in range(n_subjects):
        labels = rng.choice(labels_choices, size=n_epochs_per_subject, p=weights)
        bvp = rng.standard_normal((n_epochs_per_subject, T))
        hr = 65.0 + 15.0 * rng.random((n_epochs_per_subject, T))
        ibi = 60.0 / np.clip(hr, 40.0, 180.0) + 0.02 * rng.standard_normal((n_epochs_per_subject, T))
        ibi = np.clip(ibi, 0.3, 1.5)
        eda = 0.05 + 0.02 * rng.random((n_epochs_per_subject, T))
        acc = rng.standard_normal((n_epochs_per_subject, T, 3)) * 5.0
        subjects.append(
            {
                "bvp": bvp,
                "hr": hr,
                "ibi": ibi,
                "eda": eda,
                "acc": acc,
                "labels": labels,
            }
        )
    print(f"[mock] Generated {n_subjects} synthetic subjects ({n_epochs_per_subject} epochs each).")
    return subjects


def _load_local_csv_data(fs=100, epoch_len_sec=30, data_dir=None, sid_range=None):
    """
    Load CSV sleep recordings as multi-channel epoch dicts per subject.

    - DREAMT-style: ``*_PSG_df_updated.csv`` at 100 Hz (set DREAMT_DATA_DIR or pass data_dir).
    - Legacy: ``S00*_whole_df.csv`` in project ``data/`` at 64 Hz (resampled to fs).

    Each subject dict: bvp, hr, ibi, eda, acc (n_epochs, T) or (n_epochs, T, 3), labels.

    Mapping: W:0, N1:1, N2:2, N3:3, R:4, P:5. Epochs with P (5) are dropped.
    """
    import glob
    import pandas as pd
    from scipy.signal import resample_poly
    from math import gcd

    if data_dir is None:
        data_dir = os.environ.get("DREAMT_DATA_DIR") or os.path.join(_PROJ_ROOT, "data")
    data_dir = os.path.abspath(data_dir)

    mapping = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "P": 5}
    subjects = []

    dreamt_files = sorted(glob.glob(os.path.join(data_dir, "*_PSG_df_updated.csv")))
    legacy_files = sorted(glob.glob(os.path.join(data_dir, "S00*_whole_df.csv")))

    if dreamt_files:
        csv_files = dreamt_files
        source_fs = 100
        fmt = "dreamt"
        print(f"[local] DREAMT-style CSVs in {data_dir}: {len(csv_files)} files")
    elif legacy_files:
        csv_files = legacy_files
        source_fs = 64
        fmt = "legacy"
        print(f"[local] Legacy CSVs in {data_dir}: {len(csv_files)} files")
    else:
        print(f"[local] No CSVs found in {data_dir} (expected *_PSG_df_updated.csv or S00*_whole_df.csv).")
        return []

    for path in csv_files:
        val_name = os.path.basename(path)
        sid = val_name.split("_")[0]
        
        if sid_range is not None:
            try:
                sid_num = int("".join(filter(str.isdigit, sid)))
                if not (sid_range[0] <= sid_num <= sid_range[1]):
                    continue
            except ValueError:
                pass

        print(f"  Loading {sid}...")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  [warn] Could not read {path}: {e}. Skipping.")
            continue

        if "BVP" not in df.columns or "Sleep_Stage" not in df.columns:
            print(f"  [warn] {sid}: missing BVP or Sleep_Stage. Skipping.")
            continue

        df["stage_id"] = df["Sleep_Stage"].map(mapping).fillna(5).astype(int)

        samples_per_epoch = int(source_fs * epoch_len_sec)
        n_epochs = len(df) // samples_per_epoch
        if n_epochs == 0:
            print(f"  [warn] {sid} has too few samples. Skipping.")
            continue

        useful_len = n_epochs * samples_per_epoch
        bvp = df["BVP"].values[:useful_len].astype(np.float64)
        stages = df["stage_id"].values[:useful_len]

        def col_or_zeros(name):
            if name not in df.columns:
                return np.zeros(useful_len, dtype=np.float64)
            return df[name].values[:useful_len].astype(np.float64)

        hr = col_or_zeros("HR")
        ibi = col_or_zeros("IBI")
        eda = col_or_zeros("EDA")
        ax = col_or_zeros("ACC_X")
        ay = col_or_zeros("ACC_Y")
        az = col_or_zeros("ACC_Z")

        bvp_e = bvp.reshape(n_epochs, samples_per_epoch)
        hr_e = hr.reshape(n_epochs, samples_per_epoch)
        ibi_e = ibi.reshape(n_epochs, samples_per_epoch)
        eda_e = eda.reshape(n_epochs, samples_per_epoch)
        acc_e = np.stack(
            [
                ax.reshape(n_epochs, samples_per_epoch),
                ay.reshape(n_epochs, samples_per_epoch),
                az.reshape(n_epochs, samples_per_epoch),
            ],
            axis=-1,
        )
        epoch_labels = stages.reshape(n_epochs, samples_per_epoch)[:, 0]

        valid_mask = epoch_labels != 5
        bvp_e = bvp_e[valid_mask]
        hr_e = hr_e[valid_mask]
        ibi_e = ibi_e[valid_mask]
        eda_e = eda_e[valid_mask]
        acc_e = acc_e[valid_mask]
        epoch_labels = epoch_labels[valid_mask]

        if EVAL_MODE == "2-stage":
            epoch_labels = np.where(epoch_labels > 0, 1, 0)

        n_kept = len(epoch_labels)

        if source_fs != fs:
            print(f"  [resample] {source_fs} Hz → {fs} Hz for {n_kept} epochs ({sid})...")
            common = gcd(int(fs), int(source_fs))
            up, down = int(fs) // common, int(source_fs) // common

            def resample_epochs(arr):
                if arr.ndim == 3:
                    out = []
                    for ep in arr:
                        rx = resample_poly(ep[:, 0], up, down)
                        ry = resample_poly(ep[:, 1], up, down)
                        rz = resample_poly(ep[:, 2], up, down)
                        m = min(len(rx), len(ry), len(rz))
                        out.append(np.stack([rx[:m], ry[:m], rz[:m]], axis=-1))
                    return np.array(out)
                return np.array([resample_poly(ep, up, down) for ep in arr])

            bvp_e = resample_epochs(bvp_e)
            hr_e = resample_epochs(hr_e)
            ibi_e = resample_epochs(ibi_e)
            eda_e = resample_epochs(eda_e)
            acc_e = resample_epochs(acc_e)

        # Legacy files: no wearable extras → zeros already from col_or_zeros
        subjects.append(
            {
                "bvp": bvp_e,
                "hr": hr_e,
                "ibi": ibi_e,
                "eda": eda_e,
                "acc": acc_e,
                "labels": epoch_labels,
            }
        )
        print(f"  [OK] Loaded {sid}: {n_kept} valid epochs ({fmt}).")

    return subjects


# ───────────────────────────────────────────────
#  2. Feature extraction
# ───────────────────────────────────────────────
def _extract_features(subjects, fs_target=100):
    """Extract features for every subject's epochs using the project's module."""
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import preprocess_all_epochs
    from feature_engineering import build_feature_vector, get_feature_names

    feature_names = get_feature_names()
    processed_subjects = []
    for i, subj in enumerate(subjects):
        if isinstance(subj, dict):
            pp = preprocess_all_epochs(subj["bvp"], fs=fs_target)
            subject_data = {
                "bvp": pp,
                "hr": subj["hr"],
                "ibi": subj["ibi"],
                "eda": subj["eda"],
                "acc": subj["acc"],
            }
            labels = subj["labels"]
        else:
            # Backward compat: (epochs, labels) tuple
            epochs, labels = subj
            pp = preprocess_all_epochs(epochs, fs=fs_target)
            subject_data = {"bvp": pp}
            labels = labels
        X = build_feature_vector(subject_data, fs=fs_target)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        processed_subjects.append((X, labels))
        print(f"  Subject {i+1}/{len(subjects)}: X={X.shape}, labels={labels.shape}")
    return processed_subjects, feature_names


# ───────────────────────────────────────────────
#  3. LOSO cross-validation (GSCV-RF + RFE)
# ───────────────────────────────────────────────
def run_loso_evaluation(processed_subjects, feature_names, n_selected=15):
    """
    Leave-One-Subject-Out cross-validation using the paper's GSCV-RF + RFE.

    Returns
    -------
    y_true, y_pred           : aggregated labels and predictions
    y_proba                  : (n_total, N_STAGES) probability matrix
    last_rf                  : trained RF from the last LOSO fold
    last_rfe                 : fitted RFE from the last fold
    sel_feature_names        : selected feature names from the last fold
    last_X_sel_test          : RFE-transformed test features from the last fold
    last_subj_y_true/y_pred  : last subject's per-epoch labels for hypnogram
    """
    from model_training import build_and_train_rfe_rf, predict, get_selected_feature_names

    all_y_true, all_y_pred, all_y_proba = [], [], []
    last_rf = last_rfe = last_X_sel_test = None
    last_subj_y_true = last_subj_y_pred = None
    sel_feature_names = feature_names  # fallback

    n = len(processed_subjects)
    for test_idx in range(n):
        print(f"\n  LOSO fold {test_idx + 1}/{n} — held-out subject {test_idx + 1}")

        # ── Split ──
        train_Xs = [processed_subjects[i][0] for i in range(n) if i != test_idx]
        train_ys = [processed_subjects[i][1] for i in range(n) if i != test_idx]
        X_train_raw = np.vstack(train_Xs)
        y_train     = np.concatenate(train_ys)
        X_test_raw, y_test = processed_subjects[test_idx]

        # ── Subject-level Z-score normalisation (as in paper) ──
        train_scaler = StandardScaler()
        X_train = train_scaler.fit_transform(X_train_raw)

        test_scaler = StandardScaler()
        X_test = test_scaler.fit_transform(X_test_raw)

        # ── Apply SMOTE to Balance Dataset ──
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # ── GSCV-RF + RFE ──
        best_rf, rfe, _ = build_and_train_rfe_rf(
            X_train, y_train, n_features_to_select=n_selected
        )

        # ── Predict on held-out subject ──
        fold_y_pred, fold_y_proba, X_sel_test = predict(best_rf, rfe, X_test)

        # ── Custom Decision Threshold for 2-stage (Biasing towards Wake) ──
        if EVAL_MODE == "2-stage":
            # P(Wake) is column 0. If P(Wake) >= 0.35 -> predict Wake (0).
            # This makes the model 'more generous' when claiming the user is awake.
            WAKE_THRESHOLD = 0.35
            fold_y_pred = np.where(fold_y_proba[:, 0] >= WAKE_THRESHOLD, 0, 1)

        # Ensure proba has N_STAGES columns (RF may skip unseen classes)
        if fold_y_proba.shape[1] < N_STAGES:
            full_proba = np.zeros((len(fold_y_proba), N_STAGES))
            for col_idx, cls in enumerate(best_rf.classes_):
                full_proba[:, cls] = fold_y_proba[:, col_idx]
            fold_y_proba = full_proba

        all_y_true.append(y_test.tolist())
        all_y_pred.append(fold_y_pred.tolist())
        all_y_proba.append(fold_y_proba)

        # Keep last fold artefacts for visualisations
        last_rf           = best_rf
        last_rfe          = rfe
        last_X_sel_test   = X_sel_test
        last_subj_y_true  = y_test.tolist()
        last_subj_y_pred  = fold_y_pred.tolist()
        sel_feature_names = get_selected_feature_names(rfe, feature_names)

    y_true  = np.concatenate(all_y_true)
    y_pred  = np.concatenate(all_y_pred)
    y_proba = np.vstack(all_y_proba)

    return (y_true, y_pred, y_proba,
            last_rf, last_rfe, sel_feature_names,
            last_X_sel_test,
            last_subj_y_true, last_subj_y_pred)


# ───────────────────────────────────────────────
#  4. Metrics
# ───────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    report  = classification_report(
        y_true, y_pred,
        labels=list(range(N_STAGES)),
        target_names=STAGE_NAMES,
        zero_division=0,
        output_dict=True,
    )
    kappa   = cohen_kappa_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    header  = "| Stage | Precision | Recall | F1-Score | Support |"
    divider = "|-------|-----------|--------|----------|---------| "
    
    # Also save to file
    _ensure_plots_dir()
    metrics_path = os.path.join(PLOTS_DIR, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("## Per-Class Evaluation Metrics\n\n")
        f.write(header + "\n")
        f.write(divider + "\n")
        for stage in STAGE_NAMES:
            m = report.get(stage, {})
            line = (
                f"| {stage:<5} "
                f"| {m.get('precision', 0):.3f}     "
                f"| {m.get('recall', 0):.3f}   "
                f"| {m.get('f1-score', 0):.3f}    "
                f"| {int(m.get('support', 0)):>7} |"
            )
            f.write(line + "\n")
            print(line)
            
        acc_str = f"\n**Global Accuracy** : {report['accuracy']:.4f}\n"
        kappa_str = f"**Cohen's Kappa (k)**: {kappa:.4f}\n"
        bal_acc_str = f"**Balanced Accuracy**: {bal_acc:.4f}\n"
        
        f.write(acc_str)
        f.write(kappa_str)
        f.write(bal_acc_str)
        print(acc_str + kappa_str + bal_acc_str)

    return {"report": report, "kappa": kappa, "balanced_accuracy": bal_acc}


# ───────────────────────────────────────────────
#  Plot A — Normalized Confusion Matrix
# ───────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    _ensure_plots_dir()
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(N_STAGES)))
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_n, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
        linewidths=0.5, ax=ax, vmin=0, vmax=1,
        annot_kws={"size": 11},
    )
    ax.set_xlabel("Predicted Stage", fontsize=12, labelpad=10)
    ax.set_ylabel("True Stage", fontsize=12, labelpad=10)
    ax.set_title("Normalized Confusion Matrix — GSCV-RF Sleep Stage Classification",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[A] Saved → {path}")


# ───────────────────────────────────────────────
#  Plot B — Hypnogram Comparison
# ───────────────────────────────────────────────
def plot_hypnogram(y_true_subj, y_pred_subj, subject_id="Subject-1", epoch_len_sec=30):
    _ensure_plots_dir()
    if EVAL_MODE == "2-stage":
        stage_order  = {0: 0, 1: 1}
        ytick_labels = ["Wake", "Sleep"]
    else:
        stage_order  = {0: 0, 4: 1, 1: 2, 2: 3, 3: 4}
        ytick_labels = ["Wake", "REM", "N1", "N2", "N3"]

    n_epochs = len(y_true_subj)
    time_hrs  = np.arange(n_epochs) * epoch_len_sec / 3600.0

    true_plot = np.array([stage_order.get(s, 0) for s in y_true_subj])
    pred_plot = np.array([stage_order.get(s, 0) for s in y_pred_subj])

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    fig.suptitle(f"Hypnogram Comparison — {subject_id}", fontsize=13, fontweight="bold")

    for ax, data, title, color in zip(
        axes,
        [true_plot, pred_plot],
        ["Ground Truth", "GSCV-RF Prediction"],
        ["#2196F3", "#FF5722"],
    ):
        ax.step(time_hrs, data, where="post", color=color, linewidth=1.5)
        ax.fill_between(time_hrs, data, step="post", alpha=0.18, color=color)
        ax.set_yticks(list(range(len(ytick_labels))))
        ax.set_yticklabels(ytick_labels, fontsize=10)
        ax.set_ylim(-0.3, len(ytick_labels) - 0.7)
        ax.invert_yaxis()
        ax.set_ylabel(title, fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    axes[-1].set_xlabel("Time (Hours)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "hypnogram_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[B] Saved → {path}")


# ───────────────────────────────────────────────
#  Plot C — Class-wise ROC & AUC
# ───────────────────────────────────────────────
def plot_roc_curves(y_true, y_proba):
    _ensure_plots_dir()
    y_bin = label_binarize(y_true, classes=list(range(N_STAGES)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (stage, color) in enumerate(zip(STAGE_NAMES, STAGE_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{stage}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Class-Wise ROC Curves — GSCV-RF Sleep Stage Classification",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_auc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[C] Saved → {path}")


# ───────────────────────────────────────────────
#  Plot D — SHAP Feature Importance (TreeExplainer)
# ───────────────────────────────────────────────
def plot_shap_importance(trained_rf, X_sel_test, sel_feature_names):
    """
    Uses SHAP TreeExplainer (native RF support, exact Shapley values).
    X_sel_test : (n_samples, n_selected_features) — already RFE-projected
    """
    _ensure_plots_dir()
    try:
        # Sub-sample for speed (max 300 samples for the summary plot)
        n_bg = min(300, X_sel_test.shape[0])
        X_bg = X_sel_test[:n_bg]

        explainer   = shap.TreeExplainer(trained_rf)
        shap_values = explainer.shap_values(X_bg)  # list[n_classes] of (n, f)

        # shap_values shape varies by SHAP version:
        #   old API (list): list of (n_samples, n_features) per class
        #   new API (ndarray): (n_classes, n_samples, n_features)  ← 3D
        if isinstance(shap_values, list):
            # Old API: list of 2D arrays
            stacked = np.stack([np.abs(sv) for sv in shap_values], axis=0)
        elif shap_values.ndim == 3:
            # New API: (n_classes, n_samples, n_features)
            stacked = np.abs(shap_values)
        else:
            # Single-output / already 2D
            stacked = np.abs(shap_values)[np.newaxis, ...]  # → (1, n, f)

        # Average across classes → (n_samples, n_features), then across samples → (n_features,)
        mean_abs_shap   = stacked.mean(axis=0)      # (n_samples, n_features)
        feat_importance = mean_abs_shap.mean(axis=0)  # (n_features,)

        order        = np.argsort(feat_importance)[::-1][:15]
        sorted_vals  = feat_importance[order]
        sorted_names = np.array(sel_feature_names)[order]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [STAGE_COLORS[i % N_STAGES] for i in range(len(sorted_vals))]

        ax.barh(
            list(range(len(sorted_vals)))[::-1], sorted_vals,
            color=colors, edgecolor="white", linewidth=0.5,
        )
        ax.set_yticks(list(range(len(sorted_vals))))
        ax.set_yticklabels(sorted_names[::-1], fontsize=10)
        ax.set_xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=11)
        ax.set_title(
            f"SHAP Feature Importance — Top {len(sorted_vals)} of {len(sel_feature_names)} RFE-Selected Features",
            fontsize=13, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "shap_importance.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[D] Saved → {path}")

    except Exception:
        import traceback
        os.makedirs(NOT_IMPORTANT_DIR, exist_ok=True)
        log_path = os.path.join(NOT_IMPORTANT_DIR, "eval_error.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"[warn] SHAP plot failed — see {log_path}")


# ───────────────────────────────────────────────
#  Main entry point
# ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sleep Stage Classification Eval")
    parser.add_argument("--mode", type=str, choices=["5-stage", "2-stage"], default="5-stage", help="Evaluation Mode")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Folder with DREAMT (*_PSG_df_updated.csv) or legacy (S00*_whole_df.csv). "
        "Default: DREAMT_DATA_DIR env var, else project data/",
    )
    args = parser.parse_args()
    
    set_eval_mode(args.mode)

    print("=" * 60)
    print(f"  Sleep Classification — GSCV-RF Evaluation [{args.mode}]")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading real local data...")
    raw_subjects = _load_local_csv_data(fs=100, data_dir=args.data_dir)
    
    if not raw_subjects:
        print("[warn] Local data not found. Falling back to mock data.")
        raw_subjects = _generate_mock_data(fs=100)

    # 2. Extract features
    print("\n[2] Feature Extraction ---")
    processed_subjects, feature_names = _extract_features(raw_subjects, fs_target=100)

    # 3. LOSO evaluation
    print("\n[3] Leave-One-Subject-Out Evaluation (GSCV-RF + RFE) ---")
    (y_true, y_pred, y_proba,
     last_rf, last_rfe, sel_names,
     X_sel_test,
     last_y_true, last_y_pred) = run_loso_evaluation(processed_subjects, feature_names)

    # 4. Metrics
    print("\n[4] Metrics ---")
    compute_metrics(y_true, y_pred)

    # 5. Visualisations
    print("\n[5] Generating Visualisations ---")
    if last_rf is None:
        print("Error: No model was trained. Cannot generate visualisations.")
        return

    plot_confusion_matrix(y_true, y_pred)
    plot_hypnogram(last_y_true, last_y_pred, subject_id="Subject-6")
    plot_roc_curves(y_true, y_proba)
    plot_shap_importance(last_rf, X_sel_test, sel_names)

    print("\nEvaluation complete. All plots saved to:", os.path.abspath(PLOTS_DIR))


if __name__ == "__main__":
    main()
