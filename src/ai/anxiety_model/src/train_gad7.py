"""
Train GAD-7 models on Baigutanova aggregates (plan 2025 + improvements).

- Feature pipeline: variance + correlation pruning + RFECV (RFE) with RF + GroupKFold (fallback: stability RF top-k).
- Nested-style RF tuning (RandomizedSearchCV + GroupKFold).
- ElasticNetCV / LogisticRegressionCV with integer inner CV (avoids nested ``groups`` issues with outer GroupKFold).
- Binary: ENET + RF ensemble + Platt calibration + Youden threshold on OOF.
- Regression: ENET + RF ensemble (OOF-tuned weight).
- Variant selection: primary = OOF ensemble binary AUC; tiebreaker = elastic-net regression MAE.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_pipeline import (
    fit_platt_calibrator,
    platt_predict,
    prune_correlated_with_target,
    prune_low_variance,
    rfe_groupkfold_select,
    tune_binary_ensemble_weight,
    tune_random_forest_grouped,
    tune_regression_ensemble_weight,
    youden_threshold,
)
from .train_verification import (
    save_binary_oof_verification,
    save_calibration_reliability_plot,
    save_feature_importance_plot,
    save_learning_curve_plot,
    save_regression_oof_verification,
    save_subject_scatter_plot,
)
from .wave_dataset import build_person_level_table, build_person_wave_table

LinearBackend = Literal["sklearn", "torch"]


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=".*use_legacy_attributes.*",
        category=FutureWarning,
        module="sklearn.linear_model._logistic",
    )


def _resolve_data_root(cli_root: str | None) -> Path:
    env = os.environ.get("ANXIETY_DATA_DIR", "").strip()
    if cli_root:
        return Path(cli_root).expanduser().resolve()
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "data" / "raw").resolve()


def _n_splits(groups: np.ndarray) -> int:
    return int(min(5, len(np.unique(groups))))


def _feature_matrix(
    df: pd.DataFrame,
    hrv_only: bool,
    *,
    use_person_binary: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    meta = {"subject", "wave", "gad7", "binary_max_score_ge5", "binary_any_wave_ge5"}
    cols = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    if hrv_only:
        cols = [c for c in cols if not str(c).startswith("sleep_")]
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    y_cont = df["gad7"].astype(float).values
    if use_person_binary and "binary_any_wave_ge5" in df.columns:
        y_bin = df["binary_any_wave_ge5"].astype(int).values
    else:
        y_bin = (df["gad7"].astype(float) >= 5.0).astype(int).values
    groups = df["subject"].astype(str).values
    return X, y_cont, y_bin, groups


def _apply_feature_pipeline(
    X: pd.DataFrame,
    y_cont: np.ndarray,
    y_bin: np.ndarray,
    groups: np.ndarray,
    *,
    top_k: int,
    log: logging.Logger,
) -> tuple[pd.DataFrame, list[str]]:
    X0 = X.replace([np.inf, -np.inf], np.nan)
    med = X0.median(numeric_only=True)
    X0 = X0.fillna(med).fillna(0.0)
    X0 = X0.clip(lower=-1e8, upper=1e8)
    X1 = prune_low_variance(X0)
    X2 = prune_correlated_with_target(X1, y_cont, corr_threshold=0.85)
    X2 = X2.replace([np.inf, -np.inf], np.nan).fillna(X2.median(numeric_only=True)).fillna(0.0).clip(-1e8, 1e8)
    if X2.shape[1] <= top_k:
        return X2, list(X2.columns)
    sel = rfe_groupkfold_select(
        X2,
        y_bin,
        groups,
        max_features=min(top_k, X2.shape[1]),
        min_features=min(15, max(8, X2.shape[1] // 5)),
    )
    return X2[sel], sel


def _cv_regression(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    return_oof: bool = False,
    log: logging.Logger,
) -> dict[str, Any] | tuple[dict[str, Any], np.ndarray]:
    gkf = GroupKFold(n_splits=_n_splits(groups))
    maes, rmses, rhos = [], [], []
    n = len(X)
    oof = np.full(n, np.nan, dtype=float)
    for fold_idx, (tr, va) in enumerate(gkf.split(X, y, groups)):
        log.debug("Regression CV fold %s: train=%s val=%s", fold_idx, len(tr), len(va))
        est = clone(pipe)
        est.fit(X.iloc[tr], y[tr])
        pred_va = est.predict(X.iloc[va])
        oof[va] = pred_va
        maes.append(mean_absolute_error(y[va], pred_va))
        rmses.append(np.sqrt(mean_squared_error(y[va], pred_va)))
        if len(np.unique(y[va])) > 1:
            rho, _ = spearmanr(y[va], pred_va)
            if not np.isnan(rho):
                rhos.append(float(rho))
    metrics = {
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "spearman_mean": float(np.mean(rhos)) if rhos else float("nan"),
        "spearman_std": float(np.std(rhos)) if rhos else float("nan"),
    }
    if return_oof:
        return metrics, oof
    return metrics


def _cv_binary(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    return_oof: bool = False,
    log: logging.Logger,
) -> dict[str, Any] | tuple[dict[str, Any], np.ndarray]:
    gkf = GroupKFold(n_splits=_n_splits(groups))
    aucs = []
    n = len(X)
    oof = np.full(n, np.nan, dtype=float)
    for fold_idx, (tr, va) in enumerate(gkf.split(X, y, groups)):
        ytr, yva = y[tr], y[va]
        if len(np.unique(ytr)) < 2:
            continue
        est = clone(pipe)
        est.fit(X.iloc[tr], ytr)
        proba = est.predict_proba(X.iloc[va])[:, 1]
        oof[va] = proba
        if len(np.unique(yva)) < 2:
            continue
        aucs.append(roc_auc_score(yva, proba))
    metrics = {
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "n_folds_valid": len(aucs),
    }
    if return_oof:
        return metrics, oof
    return metrics


def _make_enet_regression_pipeline(linear_backend: LinearBackend, torch_device: str, epochs: int, log_every: int, groups: np.ndarray) -> Pipeline:
    if linear_backend == "sklearn":
        # Integer cv avoids passing groups into nested CV during outer GroupKFold folds.
        clf = ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8],
            alphas=np.logspace(-3, 0, 10),
            cv=3,
            max_iter=12000,
            random_state=42,
            n_jobs=-1,
        )
    else:
        from .torch_linear import TorchLinearRegressor

        clf = TorchLinearRegressor(
            device=torch_device,
            epochs=epochs,
            alpha=0.12,
            l1_ratio=0.5,
            random_state=42,
            log_every=log_every,
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _make_enet_binary_pipeline(linear_backend: LinearBackend, torch_device: str, epochs: int, log_every: int, groups: np.ndarray) -> Pipeline:
    if linear_backend == "sklearn":
        lr_kw: dict[str, Any] = {
            "Cs": 6,
            "cv": 3,
            "penalty": "elasticnet",
            "solver": "saga",
            "l1_ratios": [0.25, 0.5, 0.75],
            "max_iter": 20000,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        if "use_legacy_attributes" in inspect.signature(LogisticRegressionCV.__init__).parameters:
            lr_kw["use_legacy_attributes"] = False
        clf = LogisticRegressionCV(**lr_kw)
    else:
        from .torch_linear import TorchLogisticClassifier

        clf = TorchLogisticClassifier(
            device=torch_device,
            epochs=epochs,
            alpha=0.12,
            l1_ratio=0.5,
            random_state=42,
            log_every=log_every,
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _train_variant(
    df: pd.DataFrame,
    hrv_only: bool,
    label: str,
    *,
    linear_backend: LinearBackend,
    torch_device: str,
    epochs: int,
    log_every: int,
    verify_dir: Path | None,
    use_person_binary: bool,
    feature_top_k: int,
    log: logging.Logger,
) -> dict[str, Any]:
    X_raw, y_cont, y_bin, groups = _feature_matrix(df, hrv_only=hrv_only, use_person_binary=use_person_binary)
    if len(df) < 10:
        raise ValueError(f"Too few rows for training ({len(df)})")

    X, feat_names = _apply_feature_pipeline(X_raw, y_cont, y_bin, groups, top_k=feature_top_k, log=log)

    n_pos = int(np.sum(y_bin))
    n_neg = int(len(y_bin) - n_pos)
    prevalence = float(n_pos / len(y_bin)) if len(y_bin) else 0.0
    log.info(
        "[%s] binary label: n_pos=%s n_neg=%s prevalence=%.4f (GAD-7>=5)",
        label,
        n_pos,
        n_neg,
        prevalence,
    )

    log.info(
        "[%s] rows=%s features=%s (after pipeline from %s) subjects=%s backend=%s",
        label,
        len(df),
        X.shape[1],
        X_raw.shape[1],
        len(np.unique(groups)),
        linear_backend,
    )

    enet_reg = _make_enet_regression_pipeline(linear_backend, torch_device, epochs, log_every, groups)
    enet_clf = _make_enet_binary_pipeline(linear_backend, torch_device, epochs, log_every, groups)

    rf_params = tune_random_forest_grouped(X, y_bin, groups) if linear_backend == "sklearn" else {}
    rf_params = {**{"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 2}, **rf_params}

    rf_reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestRegressor(
                    n_estimators=int(rf_params.get("n_estimators", 200)),
                    max_depth=rf_params.get("max_depth", 6),
                    min_samples_leaf=int(rf_params.get("min_samples_leaf", 2)),
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=int(rf_params.get("n_estimators", 200)),
                    max_depth=rf_params.get("max_depth", 6),
                    min_samples_leaf=int(rf_params.get("min_samples_leaf", 2)),
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    metrics: dict[str, Any] = {
        "label": label,
        "n_rows": len(df),
        "n_features_raw": int(X_raw.shape[1]),
        "n_features_selected": int(X.shape[1]),
        "linear_backend": linear_backend,
        "binary_label_distribution": {
            "n_positive_gad7_ge_5": n_pos,
            "n_negative_gad7_lt_5": n_neg,
            "prevalence_positive": prevalence,
        },
    }

    reg_cv, oof_reg_en = _cv_regression(enet_reg, X, y_cont, groups, return_oof=True, log=log)  # type: ignore[misc]
    metrics["elasticnet_regression_cv"] = reg_cv
    if linear_backend == "sklearn":
        enet_reg.fit(X, y_cont)
    else:
        enet_reg.fit(X, y_cont)

    bin_cv, oof_bin_en = _cv_binary(enet_clf, X, y_bin, groups, return_oof=True, log=log)  # type: ignore[misc]
    metrics["elasticnet_binary_cv"] = bin_cv
    if linear_backend == "sklearn":
        enet_clf.fit(X, y_bin)
    else:
        enet_clf.fit(X, y_bin)

    rf_reg_cv, oof_reg_rf = _cv_regression(rf_reg, X, y_cont, groups, return_oof=True, log=log)  # type: ignore[misc]
    metrics["rf_regression_cv"] = rf_reg_cv
    rf_bin_cv, oof_bin_rf = _cv_binary(rf_clf, X, y_bin, groups, return_oof=True, log=log)  # type: ignore[misc]
    metrics["rf_binary_cv"] = rf_bin_cv

    rf_reg.fit(X, y_cont)
    rf_clf.fit(X, y_bin)

    w_bin = tune_binary_ensemble_weight(oof_bin_en, oof_bin_rf, y_bin)
    raw_oof = np.where(
        np.isnan(oof_bin_en) | np.isnan(oof_bin_rf),
        np.nan,
        w_bin * oof_bin_en + (1.0 - w_bin) * oof_bin_rf,
    )
    coef, icept = fit_platt_calibrator(raw_oof, y_bin)
    cal_oof = np.array([platt_predict(float(r), coef, icept) if np.isfinite(r) else np.nan for r in raw_oof])
    thr, j = youden_threshold(y_bin, cal_oof)
    w_reg = tune_regression_ensemble_weight(oof_reg_en, oof_reg_rf, y_cont)
    m_auc = np.isfinite(cal_oof)
    metrics["ensemble_binary_oof_auc"] = (
        float(roc_auc_score(y_bin[m_auc], cal_oof[m_auc])) if m_auc.sum() > 2 and len(np.unique(y_bin[m_auc])) > 1 else float("nan")
    )
    metrics["youden_j"] = float(j)
    metrics["platt_coef"] = [float(coef[0])]
    metrics["platt_intercept"] = float(icept)
    metrics["ensemble_binary_weight_enet"] = float(w_bin)
    metrics["ensemble_regression_weight_enet"] = float(w_reg)
    metrics["screening_prob_threshold"] = float(thr)

    binary_screening = {
        "ensemble_weight_enet": float(w_bin),
        "ensemble_regression_weight_enet": float(w_reg),
        "platt_coef": [float(coef[0])],
        "platt_intercept": float(icept),
        "screening_prob_threshold": float(thr),
        "youden_j": float(j),
    }

    oof_reg_comb = np.where(
        np.isnan(oof_reg_en) | np.isnan(oof_reg_rf),
        np.nan,
        w_reg * oof_reg_en + (1.0 - w_reg) * oof_reg_rf,
    )

    if verify_dir is not None:
        tag = f"{label}_{linear_backend}"
        save_regression_oof_verification(verify_dir, tag=tag + "_elasticnet_reg", y_true=y_cont, y_pred=oof_reg_en)
        save_regression_oof_verification(verify_dir, tag=tag + "_rf_reg", y_true=y_cont, y_pred=oof_reg_rf)
        save_regression_oof_verification(verify_dir, tag=tag + "_ensemble_reg", y_true=y_cont, y_pred=oof_reg_comb)

        save_binary_oof_verification(verify_dir, tag=tag + "_elasticnet_bin", y_true=y_bin, y_score=oof_bin_en, threshold=0.5)
        save_binary_oof_verification(verify_dir, tag=tag + "_rf_bin", y_true=y_bin, y_score=oof_bin_rf, threshold=0.5)
        save_binary_oof_verification(verify_dir, tag=tag + "_ensemble_cal_bin", y_true=y_bin, y_score=cal_oof, threshold=thr)

        save_calibration_reliability_plot(verify_dir, tag=tag + "_ensemble_cal", y_true=y_bin, scores=cal_oof)
        save_feature_importance_plot(
            verify_dir,
            tag=tag + "_rf",
            importances=rf_clf.named_steps["clf"].feature_importances_,
            names=list(X.columns),
        )
        save_subject_scatter_plot(verify_dir, tag=tag + "_ensemble_reg", y_true=y_cont, y_pred=oof_reg_comb, subjects=groups)
        save_learning_curve_plot(verify_dir, tag=tag + "_rf_bin", X=X, y=y_bin, groups=groups)

    return {
        "metrics": metrics,
        "pipelines": {
            "elasticnet_regression": enet_reg,
            "elasticnet_binary": enet_clf,
            "rf_regression": rf_reg,
            "rf_binary": rf_clf,
        },
        "feature_names": feat_names,
        "X_shape": X.shape,
        "binary_screening": binary_screening,
    }


def train_all(
    data_root: Path,
    out_dir: Path,
    *,
    linear_backend: LinearBackend = "sklearn",
    torch_device: str = "auto",
    epochs: int = 2500,
    log_every: int = 0,
    verify: bool = True,
    debug: bool = False,
    granularity: Literal["wave", "person"] = "wave",
    person_label: Literal["mean", "max"] = "mean",
    feature_top_k: int = 22,
) -> None:
    log = logging.getLogger("train_gad7")
    _configure_logging(debug)
    warnings.filterwarnings(
        "ignore",
        message=".*'penalty' was deprecated.*",
        category=FutureWarning,
        module=r"sklearn\.linear_model\._logistic",
    )

    if linear_backend == "torch":
        from .torch_linear import _require_torch, resolve_torch_device

        _require_torch()
        dev = resolve_torch_device(torch_device)
        log.info("PyTorch linear backend: requested_device=%s resolved=%s", torch_device, dev)
        if str(dev) == "cpu" and torch_device.lower() == "cuda":
            log.warning("CUDA was requested but is not available; training falls back to CPU (still uses PyTorch).")

    wave_a = build_person_wave_table(data_root, include_sleep=False)
    wave_b = build_person_wave_table(data_root, include_sleep=True)
    if wave_a.empty:
        raise SystemExit("Aggregated wave table is empty. Check data/raw paths.")

    use_pb = granularity == "person"
    if granularity == "person":
        df_a = build_person_level_table(wave_a, label_mode=person_label, binary_mode="any_wave_ge5")
        df_b = build_person_level_table(wave_b, label_mode=person_label, binary_mode="any_wave_ge5")
        log.info("Person-level tables: hrv_only rows=%s hrv_sleep rows=%s", len(df_a), len(df_b))
    else:
        df_a, df_b = wave_a, wave_b
        log.info("Wave-level tables: hrv_only rows=%s hrv_sleep rows=%s", len(df_a), len(df_b))

    verify_dir = (out_dir / "verification") if verify else None

    res_a = _train_variant(
        df_a,
        hrv_only=True,
        label="hrv_only",
        linear_backend=linear_backend,
        torch_device=torch_device,
        epochs=epochs,
        log_every=log_every,
        verify_dir=verify_dir,
        use_person_binary=use_pb,
        feature_top_k=feature_top_k,
        log=log,
    )
    res_b = _train_variant(
        df_b,
        hrv_only=False,
        label="hrv_sleep",
        linear_backend=linear_backend,
        torch_device=torch_device,
        epochs=epochs,
        log_every=log_every,
        verify_dir=verify_dir,
        use_person_binary=use_pb,
        feature_top_k=feature_top_k,
        log=log,
    )

    auc_a = res_a["metrics"].get("ensemble_binary_oof_auc", float("nan"))
    auc_b = res_b["metrics"].get("ensemble_binary_oof_auc", float("nan"))
    mae_a = res_a["metrics"]["elasticnet_regression_cv"]["mae_mean"]
    mae_b = res_b["metrics"]["elasticnet_regression_cv"]["mae_mean"]

    if not np.isfinite(auc_a) or not np.isfinite(auc_b):
        selected = "hrv_sleep" if mae_b < mae_a else "hrv_only"
    elif auc_b > auc_a:
        selected = "hrv_sleep"
    elif auc_b < auc_a:
        selected = "hrv_only"
    else:
        selected = "hrv_sleep" if mae_b < mae_a else "hrv_only"

    bundle = {
        "version": 2,
        "binary_cutoff_gad7": 5.0,
        "selected_variant": selected,
        "hrv_only": res_a,
        "hrv_sleep": res_b,
        "data_root_note": str(data_root),
        "training_options": {
            "linear_backend": linear_backend,
            "torch_device": torch_device,
            "epochs": epochs,
            "verify": verify,
            "granularity": granularity,
            "person_label": person_label,
            "feature_top_k": feature_top_k,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_dir / "gad7_bundle.joblib")

    report = {
        "selected_variant": selected,
        "selection_rule": "max ensemble_binary_oof_auc then min elasticnet_regression_cv_mae",
        "comparison_ensemble_binary_oof_auc": {"hrv_only": auc_a, "hrv_sleep": auc_b},
        "comparison_mae_elasticnet_regression": {"hrv_only": mae_a, "hrv_sleep": mae_b},
        "hrv_only_metrics": res_a["metrics"],
        "hrv_sleep_metrics": res_b["metrics"],
        "linear_backend": linear_backend,
        "torch_device": torch_device,
        "granularity": granularity,
    }
    with open(out_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    sel = res_b if selected == "hrv_sleep" else res_a
    manifest = {
        "version": 2,
        "inference_use_variant": selected,
        "primary_regressor": "elasticnet_regression",
        "primary_classifier": "elasticnet_binary",
        "feature_names_hrv_only": res_a["feature_names"],
        "feature_names_hrv_sleep": res_b["feature_names"],
        "binary_cutoff": 5.0,
        "linear_backend": linear_backend,
        "granularity": granularity,
        "binary_screening_hrv_only": res_a.get("binary_screening", {}),
        "binary_screening_hrv_sleep": res_b.get("binary_screening", {}),
        "screening_prob_threshold": sel.get("binary_screening", {}).get("screening_prob_threshold", 0.5),
    }
    with open(out_dir / "feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info("Selected variant: %s (ensemble OOF AUC / MAE tiebreak)", selected)
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out_dir / 'gad7_bundle.joblib'}")
    print(f"Saved: {out_dir / 'training_metrics.json'}")
    print(f"Saved: {out_dir / 'feature_manifest.json'}")
    if verify_dir:
        print(f"Saved verification under: {verify_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train GAD-7 models (Baigutanova + improvements).")
    p.add_argument("--root", type=str, default=None, help="data/raw folder")
    p.add_argument("--linear-backend", choices=("sklearn", "torch"), default="sklearn")
    p.add_argument("--torch-device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--torch-log-every", type=int, default=0)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--granularity", choices=("wave", "person"), default="wave")
    p.add_argument("--person-label", choices=("mean", "max"), default="mean")
    p.add_argument(
        "--feature-top-k",
        type=int,
        default=22,
        help="Max features after RFECV/RFE+GroupKFold selection (plan: ~15–22)",
    )
    args = p.parse_args()
    root = _resolve_data_root(args.root)
    if not root.is_dir():
        raise SystemExit(f"Data root is not a directory: {root}")
    out = Path(__file__).resolve().parents[1] / "models"
    train_all(
        root,
        out,
        linear_backend=args.linear_backend,  # type: ignore[arg-type]
        torch_device=args.torch_device,
        epochs=args.epochs,
        log_every=args.torch_log_every,
        verify=not args.no_verify,
        debug=args.debug,
        granularity=args.granularity,  # type: ignore[arg-type]
        person_label=args.person_label,  # type: ignore[arg-type]
        feature_top_k=args.feature_top_k,
    )


if __name__ == "__main__":
    main()
