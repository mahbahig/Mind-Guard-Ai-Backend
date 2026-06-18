"""Out-of-fold verification plots and tables for GAD-7 training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from scipy.stats import spearmanr

_LOG = logging.getLogger(__name__)


def save_binary_oof_verification(
    out_dir: Path,
    *,
    tag: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Confusion matrix at ``threshold`` on out-of-fold probabilities."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    mask = ~np.isnan(y_score)
    yt, ys = y_true[mask], y_score[mask]
    y_pred = (ys >= threshold).astype(int)

    cm = confusion_matrix(yt, y_pred, labels=[0, 1])
    report = classification_report(yt, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    try:
        auc = float(roc_auc_score(yt, ys)) if len(np.unique(yt)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    summary: dict[str, Any] = {
        "tag": tag,
        "n_samples": int(len(yt)),
        "positive_rate_true": float(yt.mean()) if len(yt) else float("nan"),
        "threshold": float(threshold),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": cm.tolist(),
        "accuracy": float(accuracy_score(yt, y_pred)) if len(yt) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(yt, y_pred)) if len(yt) else float("nan"),
        "roc_auc_oof": auc,
        "classification_report": report,
    }

    if len(np.unique(yt)) > 1:
        fpr, tpr, thr = roc_curve(yt, ys)
        thr_list = []
        for v in thr.tolist():
            if v == np.inf or v == -np.inf or (isinstance(v, float) and not np.isfinite(v)):
                thr_list.append(None)
            else:
                thr_list.append(float(v))
        summary["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr_list}

    with open(out_dir / f"oof_binary_summary_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    df = pd.DataFrame({"y_true": y_true, "y_score": y_score, "y_pred_at_threshold": np.where(np.isnan(y_score), np.nan, (y_score >= threshold).astype(float))})
    df.to_csv(out_dir / f"oof_binary_predictions_{tag}.csv", index=False)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["GAD-7<5", "GAD-7>=5"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"OOF confusion ({tag}) @ prob>={threshold}")
        fig.tight_layout()
        cm_path = out_dir / f"confusion_matrix_oof_{tag}.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        _LOG.info("Wrote %s", cm_path)

        if len(np.unique(yt)) > 1 and not np.isnan(auc):
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.plot(fpr, tpr, label=f"AUC={auc:.3f}")
            ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
            ax2.set_xlabel("FPR")
            ax2.set_ylabel("TPR")
            ax2.set_title(f"OOF ROC ({tag})")
            ax2.legend(loc="lower right")
            fig2.tight_layout()
            roc_path = out_dir / f"roc_oof_{tag}.png"
            fig2.savefig(roc_path, dpi=150)
            plt.close(fig2)
            _LOG.info("Wrote %s", roc_path)
    except ImportError:
        _LOG.warning("matplotlib not installed; skipping confusion matrix / ROC PNG export.")
    except Exception:
        _LOG.exception("Failed to write confusion matrix / ROC PNG for tag=%s", tag)

    _LOG.info(
        "[%s] OOF binary: n=%s auc=%s bal_acc=%s cm=%s",
        tag,
        summary["n_samples"],
        summary["roc_auc_oof"],
        summary["balanced_accuracy"],
        cm.tolist(),
    )
    return summary


def save_regression_oof_verification(
    out_dir: Path,
    *,
    tag: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    mae = float(mean_absolute_error(yt, yp)) if len(yt) else float("nan")
    rmse = float(np.sqrt(mean_squared_error(yt, yp))) if len(yt) else float("nan")
    rho = float(spearmanr(yt, yp).correlation) if len(yt) > 2 and len(np.unique(yt)) > 1 else float("nan")

    summary = {"tag": tag, "n_samples": int(len(yt)), "mae_oof": mae, "rmse_oof": rmse, "spearman_rho_oof": rho}
    with open(out_dir / f"oof_regression_summary_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame({"y_true_gad7": y_true, "y_pred_gad7": y_pred}).to_csv(
        out_dir / f"oof_regression_predictions_{tag}.csv", index=False
    )

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(yt, yp, alpha=0.6, edgecolors="k", linewidths=0.3)
        lims = [0, 21]
        ax.plot(lims, lims, "r--", alpha=0.5, label="identity")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True GAD-7")
        ax.set_ylabel("OOF predicted GAD-7")
        ax.set_title(f"OOF regression ({tag})\nMAE={mae:.2f} RMSE={rmse:.2f} rho={rho:.3f}")
        ax.legend()
        fig.tight_layout()
        sp_path = out_dir / f"scatter_oof_regression_{tag}.png"
        fig.savefig(sp_path, dpi=150)
        plt.close(fig)
        _LOG.info("Wrote %s", sp_path)
    except ImportError:
        _LOG.warning("matplotlib not installed; skipping regression scatter PNG.")
    except Exception:
        _LOG.exception("Failed to write regression scatter PNG for tag=%s", tag)

    _LOG.info("[%s] OOF regression: n=%s mae=%.4f rmse=%.4f rho=%s", tag, summary["n_samples"], mae, rmse, rho)
    return summary


def save_calibration_reliability_plot(
    out_dir: Path,
    *,
    tag: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 8,
) -> None:
    """Reliability diagram: mean observed rate vs mean predicted prob per bin."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        _LOG.warning("matplotlib not installed; skip calibration plot")
        return

    yt = np.asarray(y_true, dtype=float)
    ys = np.asarray(scores, dtype=float)
    m = np.isfinite(ys) & np.isfinite(yt)
    yt, ys = yt[m], ys[m]
    if len(yt) < n_bins + 2:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0, 1, n_bins + 1)
    digit = np.digitize(ys, bins) - 1
    digit = np.clip(digit, 0, n_bins - 1)
    xs, ys_obs, ns = [], [], []
    for b in range(n_bins):
        sel = digit == b
        if not np.any(sel):
            continue
        xs.append(float(np.mean(ys[sel])))
        ys_obs.append(float(np.mean(yt[sel])))
        ns.append(int(np.sum(sel)))
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="ideal")
    ax.scatter(xs, ys_obs, s=[max(8, n) for n in ns], alpha=0.7)
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(f"Calibration reliability ({tag})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    p = out_dir / f"calibration_reliability_{tag}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    _LOG.info("Wrote %s", p)


def save_feature_importance_plot(
    out_dir: Path,
    *,
    tag: str,
    importances: np.ndarray,
    names: list[str],
    top_n: int = 25,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-importances)[:top_n]
    imp = importances[order]
    lbl = [names[i] for i in order]
    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.2)))
    ax.barh(np.arange(len(imp))[::-1], imp[::-1])
    ax.set_yticks(np.arange(len(lbl)))
    ax.set_yticklabels(lbl[::-1], fontsize=7)
    ax.set_xlabel("Importance")
    ax.set_title(f"RF feature importance ({tag})")
    fig.tight_layout()
    p = out_dir / f"feature_importance_{tag}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    _LOG.info("Wrote %s", p)


def save_subject_scatter_plot(
    out_dir: Path,
    *,
    tag: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp, sub = y_true[m], y_pred[m], subjects[m]
    if len(yt) < 3:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    uniq = pd.factorize(sub)[0]
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(yt, yp, c=uniq, cmap="tab20", alpha=0.65, edgecolors="k", linewidths=0.2)
    ax.plot([0, 21], [0, 21], "r--", alpha=0.4)
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 21)
    ax.set_xlabel("True GAD-7")
    ax.set_ylabel("OOF predicted")
    ax.set_title(f"Per-subject colors ({tag})")
    fig.colorbar(sc, ax=ax, label="subject index")
    fig.tight_layout()
    p = out_dir / f"scatter_by_subject_{tag}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    _LOG.info("Wrote %s", p)


def save_learning_curve_plot(
    out_dir: Path,
    *,
    tag: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from sklearn.model_selection import learning_curve
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GroupKFold
    except ImportError:
        return
    n_splits = int(min(4, len(np.unique(groups))))
    if n_splits < 2:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    est = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    gkf = GroupKFold(n_splits=n_splits)
    train_sizes, train_scores, test_scores = learning_curve(
        est,
        X,
        y,
        groups=groups,
        cv=gkf,
        scoring="roc_auc",
        train_sizes=np.linspace(0.25, 1.0, 4),
        n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_sizes, np.mean(train_scores, axis=1), "o-", label="train AUC")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), "o-", label="val AUC")
    ax.fill_between(
        train_sizes,
        np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
        np.mean(test_scores, axis=1) + np.std(test_scores, axis=1),
        alpha=0.2,
    )
    ax.set_xlabel("Training samples")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Learning curve RF ({tag})")
    ax.legend()
    fig.tight_layout()
    p = out_dir / f"learning_curve_{tag}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    _LOG.info("Wrote %s", p)
