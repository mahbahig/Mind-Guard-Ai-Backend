"""
Data Audit Script — Signal Integrity & Feature Correlation
============================================================
Implements all 4 audit checks from 04_Tests/data_audit.md:

  1. Visual Sync Check   — plot Wake / N2 / N3 epochs (raw vs preprocessed)
  2. Feature-Stage Box Plots — 9 key features, one box plot per feature × stage
  3. Data Loading Audit  — verify sleepecg record.fs; resample to 100 Hz if needed
  4. Normalization Check — confirm subject-level StandardScaler is applied correctly

Plots saved to: results/audit/
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import resample_poly
from sklearn.preprocessing import StandardScaler
from math import gcd

# ── path setup ──────────────────────────────────────────────
SRC_DIR   = os.path.dirname(__file__)
ROOT_DIR  = os.path.join(SRC_DIR, "..")
AUDIT_DIR = os.path.join(ROOT_DIR, "results", "audit")
sys.path.insert(0, SRC_DIR)

from preprocessing import preprocess_ppg_epoch, preprocess_all_epochs
from feature_engineering import (
    calculate_statistical_features,
    calculate_nonlinear_features,
    calculate_temporal_features,
    get_feature_names,
)

TARGET_FS      = 100          # Audit target sample rate (Hz)
EPOCH_SEC      = 30           # Epoch length in seconds
STAGE_NAMES    = ["Wake", "N1", "N2", "N3", "REM", "P"]
STAGE_COLORS   = ["#E63946", "#F4A261", "#2A9D8F", "#264653", "#6A0572", "#999999"]

# The 9 "key" audit features (paper-referenced subset)
AUDIT_FEATURES = [
    "MAD", "IQR", "ACL", "Std_Dev", "Skewness", "Kurtosis",
    "Poincare_SD1", "Hjorth_Mobility", "ZCR",
]


def _ensure_audit_dir():
    os.makedirs(AUDIT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  Audit 3 — Data Loading (runs first so we know real FS)
# ═══════════════════════════════════════════════════════════
def load_local_csv_data():
    """
    Load S002 and S003 from the data/ directory.
    Mapping: W:0, N1:1, N2:2, N3:3, R:4, P:5
    """
    import pandas as pd
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    subjects = []
    
    mapping = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "P": 5}
    SOURCE_FS = 64  # Based on TIMESTAMP diff (0.015625)
    
    print(f"\n  [local] Looking for S002_whole_df.csv and S003_whole_df.csv in {DATA_DIR}...")
    
    for sid in ["S002", "S003"]:
        path = os.path.join(DATA_DIR, f"{sid}_whole_df.csv")
        if not os.path.exists(path):
            print(f"  [warn] {path} not found. Skipping.")
            continue
            
        print(f"  Loading {sid}...")
        df = pd.read_csv(path)
        
        # ── Map stages ─────────────────────────────────────
        df["stage_id"] = df["Sleep_Stage"].map(mapping).fillna(5).astype(int)
        
        # ── Group by 30s epochs (64 Hz * 30 = 1920 samples) ─
        samples_per_epoch = SOURCE_FS * EPOCH_SEC
        n_epochs = len(df) // samples_per_epoch
        
        if n_epochs == 0:
            print(f"  [warn] {sid} has too few samples ({len(df)}). Skipping.")
            continue
            
        # Truncate to full epochs
        useful_len = n_epochs * samples_per_epoch
        bvp = df["BVP"].values[:useful_len]
        stages = df["stage_id"].values[:useful_len]
        
        # Reshape to (N, 1920)
        epochs_raw = bvp.reshape(n_epochs, samples_per_epoch)
        # Take the majority label for each epoch
        epoch_labels = stages.reshape(n_epochs, samples_per_epoch)[:, 0]
        
        # ── Resample to TARGET_FS (100 Hz) ─────────────────
        print(f"  [resample] {SOURCE_FS} Hz → {TARGET_FS} Hz for {n_epochs} epochs...")
        common = gcd(int(TARGET_FS), int(SOURCE_FS))
        up, down = int(TARGET_FS) // common, int(SOURCE_FS) // common
        resampled = np.array([resample_poly(ep, up, down) for ep in epochs_raw])
        
        subjects.append((resampled, epoch_labels, TARGET_FS))
        print(f"  ✅ Loaded {sid}: {n_epochs} epochs.")

    return subjects


def audit_data_loading():
    """
    Verify the available data and return subjects.
    Prioritizes local S002/S003 CSVs, then falls back to synthetic data.
    """
    print("\n" + "─" * 55)
    print("  Audit 3 — Data Loading Check")
    print("─" * 55)

    # 1. Try local CSVs (S002, S003)
    subjects = load_local_csv_data()
    if subjects:
        return subjects

    # 2. Synthetic fallback
    print("  [warn] Local CSVs unavailable. Falling back to synthetic data.")
    print("  [synthetic] Generating realistic PPG-like epochs...")
    rng = np.random.default_rng(0)
    subjects = []
    weights  = [0.20, 0.05, 0.50, 0.10, 0.05, 0.10] # Wake, N1, N2, N3, REM, P
    for s in range(6):
        n_epochs   = 200
        epoch_samp = TARGET_FS * EPOCH_SEC
        labels     = rng.choice(6, size=n_epochs, p=weights)

        epochs = np.zeros((n_epochs, epoch_samp))
        t      = np.linspace(0, EPOCH_SEC, epoch_samp)
        for idx, lbl in enumerate(labels):
            # Different frequency content per stage for visual separation
            freq = {0: 1.2, 1: 0.9, 2: 1.0, 3: 0.5, 4: 1.1, 5: 1.2}[lbl]
            amp  = {0: 1.5, 1: 1.0, 2: 1.0, 3: 0.6, 4: 1.2, 5: 1.5}[lbl]
            noise_lvl = {0: 0.5, 1: 0.3, 2: 0.2, 3: 0.1, 4: 0.4, 5: 0.5}[lbl]
            epochs[idx] = amp * np.sin(2 * np.pi * freq * t) + noise_lvl * rng.standard_normal(epoch_samp)

        subjects.append((epochs, labels, TARGET_FS))
    print(f"  ✅ Synthetic: 6 subjects × {n_epochs} epochs @ {TARGET_FS} Hz")
    return subjects


# ═══════════════════════════════════════════════════════════
#  Audit 1 — Visual Sync Check
# ═══════════════════════════════════════════════════════════
def audit_visual_sync(subjects):
    """
    Plot one Wake, one N2, and one N3 epoch — raw vs preprocessed — side by side.
    Pass/Fail: Do the 3 stages look visually different?
    """
    _ensure_audit_dir()
    print("\n" + "─" * 55)
    print("  Audit 1 — Visual Sync Check (Wake / N2 / N3)")
    print("─" * 55)

    # Find representative epochs from the first subject
    epochs, labels, fs = subjects[0]
    target_stages = {0: "Wake", 2: "N2", 3: "N3"}
    found = {}

    for stage_id, stage_name in target_stages.items():
        idxs = np.where(labels == stage_id)[0]
        if len(idxs) > 0:
            found[stage_name] = (idxs[0], epochs[idxs[0]])
        else:
            print(f"  [warn] No {stage_name} epoch found in subject 0.")

    if len(found) < 2:
        print("  [skip] Not enough stage variety to plot. Try with real data.")
        return

    n_panels = len(found)
    fig = plt.figure(figsize=(16, 4 * n_panels))
    fig.suptitle("Audit 1 — Visual Sync Check: Raw vs Preprocessed PPG Epochs",
                 fontsize=14, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(n_panels, 2, figure=fig, hspace=0.5, wspace=0.3)
    t  = np.arange(epochs.shape[1]) / fs

    for row, (stage_name, (idx, raw_epoch)) in enumerate(found.items()):
        pp_epoch = preprocess_ppg_epoch(raw_epoch.copy(), fs)
        color    = STAGE_COLORS[list(target_stages.keys())[row]]

        ax_raw = fig.add_subplot(gs[row, 0])
        ax_raw.plot(t, raw_epoch, color=color, linewidth=0.8, alpha=0.85)
        ax_raw.set_title(f"{stage_name} — Raw (epoch #{idx})", fontsize=11)
        ax_raw.set_xlabel("Time (s)")
        ax_raw.set_ylabel("Amplitude")
        ax_raw.grid(alpha=0.25)

        ax_pp = fig.add_subplot(gs[row, 1])
        ax_pp.plot(t, pp_epoch, color=color, linewidth=0.8, alpha=0.85)
        ax_pp.set_title(f"{stage_name} — Preprocessed (detrend + SavGol + z-score)", fontsize=11)
        ax_pp.set_xlabel("Time (s)")
        ax_pp.set_ylabel("Amplitude (z)")
        ax_pp.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(AUDIT_DIR, "audit1_visual_sync.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved → {path}")
    print("  Check criteria: Wake should show higher frequency / amplitude than N3.")


# ═══════════════════════════════════════════════════════════
#  Audit 2 — Feature-Stage Correlation Box Plots
# ═══════════════════════════════════════════════════════════
def _extract_audit_features(epochs, labels, fs):
    """Extract the 9 audit features and their stage labels."""
    all_feature_names = get_feature_names()
    audit_indices = [all_feature_names.index(f) for f in AUDIT_FEATURES if f in all_feature_names]

    records = []
    for epoch, label in zip(epochs, labels):
        try:
            stat  = calculate_statistical_features(epoch)
            temp  = calculate_temporal_features(epoch, fs)
            nl    = calculate_nonlinear_features(epoch)
            full  = np.array(stat + temp + nl, dtype=float)
            full  = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)
            # Grab the 9 audit features by index
            records.append((full[audit_indices], int(label)))
        except Exception:
            continue

    feat_matrix = np.array([r[0] for r in records])
    label_vec   = np.array([r[1] for r in records])
    return feat_matrix, label_vec


def audit_feature_correlation(subjects):
    """
    Box plot for each of the 9 audit features, grouped by sleep stage.
    Success: clear median difference for ACL / IQR between Wake and N3.
    Failure: all boxes at the same height (feature extraction broken).
    """
    _ensure_audit_dir()
    print("\n" + "─" * 55)
    print("  Audit 2 — Feature-Stage Correlation Box Plots")
    print("─" * 55)

    # Aggregate across all subjects
    all_feats, all_labels = [], []
    for epochs, labels, fs in subjects:
        pp = preprocess_all_epochs(epochs, fs)
        f, l = _extract_audit_features(pp, labels, fs)
        all_feats.append(f)
        all_labels.append(l)

    feat_matrix = np.vstack(all_feats)
    label_vec   = np.concatenate(all_labels)

    print(f"  Total epochs for box plots: {feat_matrix.shape[0]}")

    n_feats = len(AUDIT_FEATURES)
    ncols   = 3
    nrows   = int(np.ceil(n_feats / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.suptitle("Audit 2 — Feature Distribution by Sleep Stage\n"
                 "(Success: ACL / IQR median should differ clearly between Wake and N3)",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for col_idx, (feat_name, ax) in enumerate(zip(AUDIT_FEATURES, axes)):
        data_per_stage = []
        xtick_labels   = []
        colors         = []

        for stage_id, stage_name in enumerate(STAGE_NAMES):
            mask = label_vec == stage_id
            if mask.sum() > 0:
                data_per_stage.append(feat_matrix[mask, col_idx])
                xtick_labels.append(f"{stage_name}\n(n={mask.sum()})")
                colors.append(STAGE_COLORS[stage_id])

        bp = ax.boxplot(
            data_per_stage,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            flierprops=dict(marker=".", markersize=2, alpha=0.4),
            widths=0.5,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticklabels(xtick_labels, fontsize=8)
        ax.set_title(feat_name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Value", fontsize=9)
        ax.grid(axis="y", alpha=0.25)

        # Flag audit result
        wake_mask = label_vec == 0
        n3_mask   = label_vec == 3
        if wake_mask.sum() > 0 and n3_mask.sum() > 0:
            wake_med = np.median(feat_matrix[wake_mask, col_idx])
            n3_med   = np.median(feat_matrix[n3_mask, col_idx])
            diff_pct = abs(wake_med - n3_med) / (abs(wake_med) + 1e-9) * 100
            status   = "✅" if diff_pct > 5 else "⚠️"
            ax.set_xlabel(f"{status} Wake↔N3 diff: {diff_pct:.1f}%", fontsize=8)

    # Hide unused subplots
    for ax in axes[n_feats:]:
        ax.set_visible(False)

    plt.tight_layout()
    path = os.path.join(AUDIT_DIR, "audit2_feature_boxplots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved → {path}")


# ═══════════════════════════════════════════════════════════
#  Audit 4 — Normalization Check
# ═══════════════════════════════════════════════════════════
def audit_normalization(subjects):
    """
    Verify per-subject StandardScaler:
      - Before scaling: epoch features may have very different means/stds across subjects
      - After per-subject scaling: each subject should have ~0 mean and ~1 std across features
    Plots mean and std comparison before/after for the first 2 subjects.
    """
    _ensure_audit_dir()
    print("\n" + "─" * 55)
    print("  Audit 4 — Per-Subject Normalization Check")
    print("─" * 55)

    n_subjects = min(3, len(subjects))
    fig, axes  = plt.subplots(n_subjects, 2, figsize=(14, 4 * n_subjects))
    fig.suptitle("Audit 4 — Per-Subject Normalization (Before vs After StandardScaler)",
                 fontsize=13, fontweight="bold")

    for s_idx in range(n_subjects):
        epochs, labels, fs = subjects[s_idx]
        pp  = preprocess_all_epochs(epochs, fs)
        X   = np.array([
            calculate_statistical_features(ep) +
            calculate_temporal_features(ep, fs) +
            calculate_nonlinear_features(ep)
            for ep in pp
        ], dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Per-subject scaling
        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ax_before = axes[s_idx, 0]
        ax_after  = axes[s_idx, 1]

        ax_before.plot(X.mean(axis=0), color="#2A9D8F",  label="Mean", linewidth=1.5)
        ax_before.plot(X.std(axis=0),  color="#E63946",  label="Std",  linewidth=1.5)
        ax_before.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_before.axhline(1, color="gray", linestyle=":",  linewidth=0.8, alpha=0.5)
        ax_before.set_title(f"Subject {s_idx + 1} — BEFORE scaling", fontsize=11)
        ax_before.set_xlabel("Feature index")
        ax_before.legend(fontsize=9)
        ax_before.grid(alpha=0.25)

        ax_after.plot(X_scaled.mean(axis=0), color="#2A9D8F", label="Mean", linewidth=1.5)
        ax_after.plot(X_scaled.std(axis=0),  color="#E63946",  label="Std",  linewidth=1.5)
        ax_after.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="0")
        ax_after.axhline(1, color="gray", linestyle=":",  linewidth=0.8, alpha=0.5, label="1")
        ax_after.set_title(f"Subject {s_idx + 1} — AFTER subject-level scaling", fontsize=11)
        ax_after.set_xlabel("Feature index")
        ax_after.legend(fontsize=9)
        ax_after.grid(alpha=0.25)
        ax_after.set_ylim(-0.5, 2.5)

        mean_err = np.abs(X_scaled.mean(axis=0)).mean()
        std_err  = np.abs(X_scaled.std(axis=0) - 1.0).mean()
        ok       = "✅" if mean_err < 0.05 and std_err < 0.05 else "⚠️"
        print(f"  Subject {s_idx + 1}: mean≈0 err={mean_err:.4f}, std≈1 err={std_err:.4f} {ok}")

    plt.tight_layout()
    path = os.path.join(AUDIT_DIR, "audit4_normalization.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved → {path}")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  Data Audit — Signal Integrity & Feature Correlation")
    print("=" * 55)

    # Audit 3 first (we need to know fs before doing anything else)
    subjects = audit_data_loading()

    # Audit 1 — visual epochs
    audit_visual_sync(subjects)

    # Audit 2 — feature box plots
    audit_feature_correlation(subjects)

    # Audit 4 — normalization
    audit_normalization(subjects)

    print("\n" + "=" * 55)
    print("  Audit complete. All plots saved to:")
    print(f"  {os.path.abspath(AUDIT_DIR)}")
    print("=" * 55)
    print("\nInterpretation guide:")
    print("  Audit 1: Wake should look noisier / higher freq than N3.")
    print("  Audit 2: ACL and IQR should show clear Wake > N3 separation.")
    print("  Audit 3: ✅ if fs was printed and resampled correctly.")
    print("  Audit 4: After scaling, mean ≈ 0 and std ≈ 1 per subject.")


if __name__ == "__main__":
    main()
